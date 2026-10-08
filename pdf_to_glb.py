#!/usr/bin/env python3
"""Convert static, solid-color U3D CLOD assemblies in PDFs to GLB 2.0."""
import argparse
import hashlib
import json
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from u3d_codec import BitStream

HEADER, CHAIN, GROUP, MODEL = 0x00443355, 0xFFFFFF14, 0xFFFFFF21, 0xFFFFFF22
DECL, BASE, PROGRESSIVE = 0xFFFFFF31, 0xFFFFFF3B, 0xFFFFFF3C
SHADING, SHADER, MATERIAL = 0xFFFFFF45, 0xFFFFFF53, 0xFFFFFF54


class Unsupported(ValueError):
    pass


def blocks(data, offset=0):
    while offset < len(data):
        if offset + 12 > len(data):
            raise ValueError('Truncated U3D block header')
        kind, size, meta = struct.unpack_from('<III', data, offset)
        start = offset + 12
        end = start + ((size + 3) & ~3) + ((meta + 3) & ~3)
        if end > len(data):
            raise ValueError('Truncated U3D block payload')
        yield kind, data[start:start + size]
        offset = end


def string(bs):
    return bs.read_string().decode('utf-8')


def chain_blocks(data):
    bs = BitStream(data)
    string(bs)
    bs.read_u32()  # chain type
    flags = bs.read_u32()
    offset = bs.get_bitcount() // 8
    offset += (16 if flags & 1 else 0) + (24 if flags & 2 else 0)
    offset = (offset + 3) & ~3
    count = struct.unpack_from('<I', data, offset)[0]
    result = list(blocks(data, offset + 4))
    if len(result) != count:
        raise ValueError('Modifier count does not match chain payload')
    return result


def read_source(path):
    path = Path(path)
    if path.suffix.lower() == '.u3d':
        return path.read_bytes()
    if path.suffix.lower() != '.pdf':
        raise Unsupported('Input must be .pdf or .u3d')
    from pypdf import PdfReader, filters
    old_limit = filters.ZLIB_MAX_OUTPUT_LENGTH
    streams = []
    seen = set()
    try:
        filters.ZLIB_MAX_OUTPUT_LENGTH = 512 * 1024 * 1024
        for page in PdfReader(path).pages:
            for ref in page.get('/Annots', []):
                annotation = ref.get_object()
                if '/3DD' not in annotation:
                    continue
                data = annotation['/3DD'].get_object()
                if data.get('/Subtype') != '/U3D':
                    raise Unsupported('PRC and non-U3D 3D streams are unsupported')
                raw = data.get_data()
                digest = hashlib.sha256(raw).digest()
                if digest not in seen:
                    seen.add(digest)
                    streams.append(raw)
    finally:
        filters.ZLIB_MAX_OUTPUT_LENGTH = old_limit
    if len(streams) != 1:
        raise Unsupported(f'Expected one U3D stream; found {len(streams)}')
    return streams[0]


def parse_scene(raw):
    if raw[:4] != b'U3D\x00':
        raise ValueError('Invalid U3D signature')
    scene = dict(nodes={}, shaders={}, materials={}, declarations={}, bases={},
                 shading={}, profile=0, units=1.0)

    def walk(items):
        for kind, data in items:
            bs = BitStream(data)
            if kind == CHAIN:
                walk(chain_blocks(data))
                continue
            if kind == HEADER:
                bs.read_u32()
                scene['profile'] = bs.read_u32()
                bs.read_u32(); bs.read_u64(); bs.read_u32()
                if scene['profile'] & 8:
                    scene['units'] = bs.read_f64()
                continue
            if kind in (PROGRESSIVE, 0xFFFFFF55, 0xFFFFFF5C,
                        0xFFFFFF36, 0xFFFFFF37, 0xFFFFFF5D, 0xFFFFFF5E):
                raise Unsupported(f'Unsupported progressive/texture/line/point block: {kind:#x}')
            if kind not in (GROUP, MODEL, SHADING, SHADER, MATERIAL, DECL, BASE):
                continue
            name = string(bs)
            if kind in (GROUP, MODEL):
                count = bs.read_u32()
                if count > 1:
                    raise Unsupported('Multiple-parent node instances are unsupported')
                parent, matrix = '', np.eye(4)
                if count:
                    parent = string(bs)
                    matrix = np.array([bs.read_f32() for _ in range(16)]).reshape(4, 4, order='F')
                resource = string(bs) if kind == MODEL else ''
                visibility = bs.read_u32() if kind == MODEL else 3
                if name in scene['nodes']:
                    raise ValueError(f'Duplicate node: {name}')
                scene['nodes'][name] = dict(parent=parent, matrix=matrix,
                                           resource=resource, visibility=visibility)
            elif kind == SHADING:
                bs.read_u32()
                flags, count = bs.read_u32(), bs.read_u32()
                shaders = []
                for _ in range(count):
                    shader_count = bs.read_u32()
                    if shader_count != 1:
                        raise Unsupported('Multipass / empty shader lists are unsupported')
                    shaders.append(string(bs))
                if flags & 1:
                    scene['shading'][name] = shaders
            elif kind == SHADER:
                flags = bs.read_u32()
                alpha_ref = bs.read_f32()
                alpha_func, blend, passes, channels, alpha_channels = [bs.read_u32() for _ in range(5)]
                material = string(bs)
                if channels or alpha_channels or flags & 2 or passes != 1 or blend != 0x606:
                    raise Unsupported(f'Unsupported shader effects: {name}')
                scene['shaders'][name] = dict(material=material, flags=flags)
            elif kind == MATERIAL:
                flags = bs.read_u32()
                colors = [[bs.read_f32() for _ in range(3)] for _ in range(4)]
                reflectivity, opacity = bs.read_f32(), bs.read_f32()
                if not flags & 2:
                    raise Unsupported(f'Material without explicit diffuse color: {name}')
                scene['materials'][name] = dict(diffuse=colors[1], emissive=colors[3],
                    ambient=colors[0], specular=colors[2], reflectivity=reflectivity,
                    opacity=opacity if flags & 32 else 1.0, flags=flags)
            else:
                target = scene['declarations'] if kind == DECL else scene['bases']
                if name in target:
                    raise ValueError(f'Duplicate geometry block: {name}')
                target[name] = data

    walk(blocks(raw))
    if not scene['declarations'] or scene['declarations'].keys() != scene['bases'].keys():
        raise Unsupported('CLOD declarations and base meshes must match')
    if not np.isfinite(scene['units']) or scene['units'] <= 0:
        raise ValueError('Invalid U3D unit scale')
    return scene


def decode_geometry(declaration, base, nocomp=False):
    bs = BitStream(declaration, no_compression=nocomp)
    string(bs); bs.read_u32()
    exclude_normals = bs.read_u32() & 1
    for _ in range(6):
        bs.read_u32()
    descriptions = []
    for _ in range(bs.read_u32()):
        flags, layers = bs.read_u32(), bs.read_u32()
        if flags or layers:
            raise Unsupported('Per-corner colors and texture layers are unsupported')
        descriptions.append(bs.read_u32())  # Original Shading ID
    bs = BitStream(base, no_compression=nocomp)
    string(bs); bs.read_u32()
    nf, np_, nn, nd, ns, nt = [bs.read_u32() for _ in range(6)]
    if nd or ns:
        raise Unsupported('Vertex color arrays are unsupported')
    offset = bs.get_bitcount() // 8
    end = offset + (np_ * 3 + nn * 3 + nt * 4) * 4
    if end > len(base):
        raise ValueError('Truncated mesh arrays')
    positions = np.frombuffer(base, '<f4', np_ * 3, offset).reshape(-1, 3).copy()
    normals = np.frombuffer(base, '<f4', nn * 3, offset + np_ * 12).reshape(-1, 3).copy()
    if not np.isfinite(positions).all() or not np.isfinite(normals).all():
        raise ValueError('Non-finite mesh coordinates')
    bs.seek_ro(end * 8)
    corners = np.empty((nf, 3, 2), dtype=np.uint32)
    shading_ids = np.empty(nf, dtype=np.uint32)
    for face in range(nf):
        mid = bs.read_compressed_u32(1)
        if mid >= len(descriptions):
            raise ValueError('Mesh shading index out of range')
        shading_ids[face] = descriptions[mid]
        for corner in range(3):
            pi = bs.read_compressed_u32(0x400 + np_)
            ni = bs.read_compressed_u32(0x400 + nn) if not exclude_normals else 0
            if pi >= np_ or (not exclude_normals and ni >= nn):
                raise ValueError('Mesh vertex/normal index out of range')
            corners[face, corner] = (pi, ni)
    if exclude_normals:
        v = positions[corners[:, :, 0]].reshape(-1, 3)
        n = np.cross(v.reshape(-1, 3, 3)[:, 1] - v.reshape(-1, 3, 3)[:, 0],
                     v.reshape(-1, 3, 3)[:, 2] - v.reshape(-1, 3, 3)[:, 0])
        n = np.repeat(n, 3, axis=0)
        indices = np.arange(nf * 3, dtype=np.uint32).reshape(-1, 3)
    else:
        pairs, inverse = np.unique(corners.reshape(-1, 2), axis=0, return_inverse=True)
        v, n = positions[pairs[:, 0]], normals[pairs[:, 1]]
        indices = inverse.astype(np.uint32).reshape(-1, 3)
    lengths = np.linalg.norm(n, axis=1)
    if np.any(lengths < 1e-12):
        raise ValueError('Zero-length normals')
    n /= lengths[:, None]
    return v, n, indices, shading_ids


def world_matrices(nodes):
    cache = {}
    def resolve(name, visiting):
        if name in cache:
            return cache[name]
        if name in visiting:
            raise ValueError('Cycle in U3D node hierarchy')
        node = nodes[name]
        parent = node['parent']
        if parent and parent not in nodes:
            raise ValueError(f'Missing parent node: {parent}')
        matrix = resolve(parent, visiting | {name}) @ node['matrix'] if parent else node['matrix']
        if not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
            raise ValueError('Invalid affine node transform')
        cache[name] = matrix
        return matrix
    for name in nodes:
        resolve(name, set())
    return cache


def linear_color(rgb):
    rgb = np.clip(np.asarray(rgb, dtype=float), 0, 1)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4).tolist()


def node_trs(matrix):
    """Avoid absolute matrix round-trip errors for CAD transforms with large scales."""
    scale = np.linalg.norm(matrix[:3, :3], axis=0)
    if np.any(scale < 1e-12):
        raise Unsupported('Singular node transforms are unsupported')
    rotation = matrix[:3, :3] / scale
    if np.linalg.det(rotation) < 0:
        scale[0] *= -1
        rotation[:, 0] *= -1
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise Unsupported('Sheared node transforms are unsupported')
    # Original F32 rotations accumulate tiny errors along the CAD hierarchy.
    left, _, right = np.linalg.svd(rotation)
    rotation = left @ right
    trace = np.trace(rotation)
    if trace > 0:
        size = np.sqrt(trace + 1) * 2
        quaternion = [(rotation[2, 1] - rotation[1, 2]) / size,
                      (rotation[0, 2] - rotation[2, 0]) / size,
                      (rotation[1, 0] - rotation[0, 1]) / size, size / 4]
    else:
        i = int(np.argmax(np.diag(rotation)))
        j, k = (i + 1) % 3, (i + 2) % 3
        size = np.sqrt(1 + rotation[i, i] - rotation[j, j] - rotation[k, k]) * 2
        quaternion = [0.0] * 4
        quaternion[i] = size / 4
        quaternion[j] = (rotation[j, i] + rotation[i, j]) / size
        quaternion[k] = (rotation[k, i] + rotation[i, k]) / size
        quaternion[3] = (rotation[k, j] - rotation[j, k]) / size
    return dict(translation=matrix[:3, 3].tolist(), rotation=quaternion, scale=scale.tolist())


class GLB:
    def __init__(self):
        self.binary = bytearray()
        self.doc = dict(asset={'version': '2.0', 'generator': 'pdf-to-glb'},
                        scene=0, scenes=[{'nodes': [0]}], nodes=[{'name': 'Model', 'children': []}],
                        meshes=[], materials=[], accessors=[], bufferViews=[], buffers=[])

    def accessor(self, array, target):
        array = np.ascontiguousarray(array)
        component = 5126 if array.dtype.kind == 'f' else 5125
        array = array.astype('<f4' if component == 5126 else '<u4')
        offset = len(self.binary)
        self.binary.extend(array.tobytes())
        self.doc['bufferViews'].append(dict(buffer=0, byteOffset=offset, byteLength=array.nbytes, target=target))
        accessor = dict(bufferView=len(self.doc['bufferViews']) - 1, componentType=component,
                        count=len(array), type='VEC3' if array.ndim == 2 else 'SCALAR')
        accessor['min'] = np.atleast_1d(array.min(axis=0)).tolist()
        accessor['max'] = np.atleast_1d(array.max(axis=0)).tolist()
        self.doc['accessors'].append(accessor)
        return len(self.doc['accessors']) - 1

    def save(self, path):
        self.doc['buffers'] = [{'byteLength': len(self.binary)}]
        encoded = json.dumps(self.doc, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
        encoded += b' ' * (-len(encoded) % 4)
        self.binary.extend(b'\x00' * (-len(self.binary) % 4))
        with Path(path).open('wb') as output:
            output.write(struct.pack('<4sII', b'glTF', 2, 28 + len(encoded) + len(self.binary)))
            output.write(struct.pack('<I4s', len(encoded), b'JSON')); output.write(encoded)
            output.write(struct.pack('<I4s', len(self.binary), b'BIN\x00')); output.write(self.binary)


LEVELS = {'low': (0.5, 0.001), 'medium': (0.25, 0.005), 'high': (0.1, 0.01)}


def lightweight_options(level='none', ratio=None, error=None, lock_border=False, merge_parts=False):
    if level == 'none':
        if ratio is not None or error is not None or lock_border or merge_parts:
            raise ValueError('Lightweight parameters require --level low, medium, or high')
        return None
    if level not in LEVELS:
        raise ValueError('Unknown lightweight level')
    default_ratio, default_error = LEVELS[level]
    ratio = default_ratio if ratio is None else ratio
    error = default_error if error is None else error
    if not np.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError('Ratio must be finite and greater than 0, up to 1')
    if not np.isfinite(error) or not 0 <= error <= 1:
        raise ValueError('Error must be finite and between 0 and 1')
    return dict(level=level, ratio=ratio, error=error, lock_border=lock_border, merge_parts=merge_parts)


def make_lightweight(source, destination, options):
    node = shutil.which('node')
    if not node:
        raise ValueError('Lightweight conversion requires Node.js 22 or later and npm ci')
    script = Path(__file__).with_name('lightweight.mjs')
    result = subprocess.run([node, str(script), str(source), str(destination), json.dumps(options)],
                            capture_output=True, text=True)
    if result.returncode:
        raise ValueError(f'Lightweight conversion failed (run npm ci): {result.stderr.strip()}')
    return json.loads(result.stdout)


def preserve_local_transforms(original, destination):
    # The optimizer's writer omits near-identity TRS values. Parent scales can
    # amplify those small local offsets, so restore every source TRS exactly.
    data = destination.read_bytes()
    json_size = struct.unpack_from('<I', data, 12)[0]
    glb = GLB()
    glb.doc = json.loads(data[20:20 + json_size])
    binary_size = glb.doc['buffers'][0]['byteLength']
    glb.binary = bytearray(data[28 + json_size:28 + json_size + binary_size])
    originals = {node['name']: node for node in original.doc['nodes']}
    for node in glb.doc['nodes']:
        for field in ('translation', 'rotation', 'scale'):
            node.pop(field, None)
            if field in originals[node['name']]:
                node[field] = originals[node['name']][field]
    glb.save(destination)


def convert(source, destination, cache_dir=None, *, level='none', ratio=None, error=None,
            lock_border=False, merge_parts=False):
    options = lightweight_options(level, ratio, error, lock_border, merge_parts)
    raw = read_source(source)
    digest = hashlib.sha256(raw).hexdigest()
    scene = parse_scene(raw)
    worlds = world_matrices(scene['nodes'])
    glb = GLB()
    material_ids = {}
    for name, mat in scene['materials'].items():
        rgb = linear_color(mat['diffuse'])
        opacity = float(np.clip(mat['opacity'], 0, 1))
        material = dict(name=name, pbrMetallicRoughness=dict(baseColorFactor=rgb + [opacity],
                        metallicFactor=0, roughnessFactor=0.65), doubleSided=True,
                        extras={'u3d': mat, 'sourceColorSpaceAssumption': 'sRGB'})
        if opacity < 1:
            material['alphaMode'] = 'BLEND'
        if mat['flags'] & 8:
            material['emissiveFactor'] = linear_color(mat['emissive'])
        material_ids[name] = len(glb.doc['materials'])
        glb.doc['materials'].append(material)
    if scene['units'] != 1:
        glb.doc['nodes'][0]['scale'] = [scene['units']] * 3
    node_ids = {name: i + 1 for i, name in enumerate(scene['nodes'])}
    for name, node in scene['nodes'].items():
        glb.doc['nodes'].append(dict(name=name, **node_trs(node['matrix']),
                                    extras={'u3dParent': node['parent']}))
    for name, node in scene['nodes'].items():
        parent = node_ids[node['parent']] if node['parent'] else 0
        glb.doc['nodes'][parent].setdefault('children', []).append(node_ids[name])
    geometry = {}
    for i, (name, declaration) in enumerate(scene['declarations'].items()):
        cache = Path(cache_dir) / digest / f'{i}.npz' if cache_dir else None
        if cache and cache.exists():
            with np.load(cache, allow_pickle=False) as saved:
                v, n, faces, shading = [saved[k] for k in ('v', 'n', 'faces', 'shading')]
        else:
            v, n, faces, shading = decode_geometry(declaration, scene['bases'][name], bool(scene['profile'] & 4))
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez(cache, v=v, n=n, faces=faces, shading=shading)
        geometry[name] = (v, n, faces, shading)
        if (i + 1) % 25 == 0 or i + 1 == len(scene['declarations']):
            print(f'Decoded {i + 1}/{len(scene["declarations"])} resources', flush=True)
    mesh_ids, accessors = {}, {}
    bounds = [np.full(3, np.inf), np.full(3, -np.inf)]
    triangles = 0
    for name, node in scene['nodes'].items():
        resource = node['resource']
        if not resource or not node['visibility']:
            continue
        if node['visibility'] != 3:
            raise Unsupported('Only double-sided visible meshes are supported')
        if resource not in geometry:
            raise ValueError(f'Missing geometry: {resource}')
        shader_names = scene['shading'].get(name, scene['shading'].get(resource))
        if not shader_names:
            raise Unsupported(f'No explicit shading for node: {name}')
        materials = []
        for shader_name in shader_names:
            if shader_name not in scene['shaders']:
                raise ValueError(f'Missing shader: {shader_name}')
            material_name = scene['shaders'][shader_name]['material']
            if material_name not in material_ids:
                raise ValueError(f'Missing material: {material_name}')
            materials.append(material_ids[material_name])
        key = (resource, tuple(materials))
        v, n, faces, shading = geometry[resource]
        if key not in mesh_ids:
            if resource not in accessors:
                accessors[resource] = (glb.accessor(v, 34962), glb.accessor(n, 34962))
            pos, norm = accessors[resource]
            primitives = []
            for mid in np.unique(shading):
                if mid >= len(materials):
                    raise Unsupported('Missing shader for Original Shading ID')
                index = glb.accessor(faces[shading == mid].reshape(-1), 34963)
                primitives.append(dict(attributes={'POSITION': pos, 'NORMAL': norm}, indices=index,
                                       material=materials[mid], mode=4))
            mesh_ids[key] = len(glb.doc['meshes'])
            glb.doc['meshes'].append(dict(name=resource, primitives=primitives))
        matrix = worlds[name].copy()
        matrix[:3] *= scene['units']
        transformed = v @ matrix[:3, :3].T + matrix[:3, 3]
        bounds[0] = np.minimum(bounds[0], transformed.min(axis=0))
        bounds[1] = np.maximum(bounds[1], transformed.max(axis=0))
        triangles += len(faces)
        glb.doc['nodes'][node_ids[name]]['mesh'] = mesh_ids[key]
    if not glb.doc['meshes']:
        raise Unsupported('No visible model geometry')
    report = dict(source=str(Path(source).resolve()), u3d_sha256=digest,
                  geometry_resources=len(geometry),
                  visible_instances=sum('mesh' in node for node in glb.doc['nodes']),
                  group_nodes=sum(not node['resource'] for node in scene['nodes'].values()),
                  glb_meshes=len(glb.doc['meshes']), materials=len(material_ids),
                  triangles_in_scene=triangles, units='metres', coordinate_system='Source origin and axes preserved',
                  dimensions_m=(bounds[1] - bounds[0]).tolist(),
                  color_mapping='U3D diffuse sRGB assumed -> glTF linear RGB',
                  material_mapping='Diffuse/emissive/opacity preserved; Phong -> PBR appearance approximate',
                  node_mapping='Source assembly hierarchy and local transforms preserved; unit scale on Model root')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if options:
        with tempfile.TemporaryDirectory(prefix='.pdf-to-glb-', dir=destination.parent) as temporary:
            original = Path(temporary) / 'original.glb'
            candidate = Path(temporary) / 'lightweight.glb'
            glb.save(original)
            stats = make_lightweight(original, candidate, options)
            if not merge_parts:
                preserve_local_transforms(glb, candidate)
            report['lightweight'] = dict(**options, **stats, original_glb_bytes=original.stat().st_size,
                                        original_dimensions_m=report['dimensions_m'],
                                        original_glb_meshes=report['glb_meshes'],
                                        original_visible_instances=report['visible_instances'],
                                        original_materials=report['materials'])
            report['triangles_in_scene'] = stats['output_triangles']
            report['dimensions_m'] = stats['output_dimensions_m']
            report['glb_meshes'] = stats['output_meshes']
            report['visible_instances'] = stats['output_instances']
            report['materials'] = stats['output_materials']
            if merge_parts:
                report['group_nodes'] = 0
                report['node_mapping'] = 'Components merged by material; individual component selection removed'
            candidate.replace(destination)
    else:
        glb.save(destination)
    report['glb_bytes'] = destination.stat().st_size
    destination.with_suffix('.report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--cache-dir', type=Path, help='Optional cache of decoded geometry for repeat conversions')
    parser.add_argument('--level', choices=['none', *LEVELS], default='none', help='Lightweight level; default: none')
    parser.add_argument('--ratio', type=float, help='Override target fraction of triangles to keep (0 < ratio <= 1)')
    parser.add_argument('--error', type=float, help='Override relative mesh error limit (0 <= error <= 1)')
    parser.add_argument('--lock-border', action='store_true', help='Preserve open mesh boundaries during simplification')
    parser.add_argument('--merge-parts', action='store_true', help='Merge components by material; removes individual selection')
    args = parser.parse_args()
    if args.output.suffix.lower() != '.glb':
        parser.error('Output must have .glb extension')
    try:
        report = convert(args.input, args.output, args.cache_dir, level=args.level,
                         ratio=args.ratio, error=args.error, lock_border=args.lock_border, merge_parts=args.merge_parts)
    except (ValueError, IndexError, struct.error) as error:
        parser.exit(1, f'Conversion failed: {error}\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
