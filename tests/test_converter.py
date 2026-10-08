import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pdf_to_glb as c


def u32(*values):
    return struct.pack('<' + 'I' * len(values), *values)


def f32(*values):
    return struct.pack('<' + 'f' * len(values), *values)


def text(value):
    encoded = value.encode()
    return struct.pack('<H', len(encoded)) + encoded


def block(kind, data):
    return u32(kind, len(data), 0) + data + b'\x00' * (-len(data) % 4)


def chain(name, *items):
    header = text(name) + u32(0, 0)
    header += b'\x00' * (-len(header) % 4)
    return block(c.CHAIN, header + u32(len(items)) + b''.join(items))


def fixture_bytes(*, opacity=1.0, progressive=False, shader_count=1, normal=True,
                  bad_index=False, flags=0, texture=False):
    header = block(c.HEADER, struct.pack('<IIIQI', 0, 12, 44, 0, 106) + struct.pack('<d', 0.001))
    declaration = text('mesh') + u32(0, 0 if normal else 1, 1, 3, 3 if normal else 0, 0, 0, 0, 1)
    declaration += u32(flags, 1 if texture else 0)
    if texture:
        declaration += u32(2)
    declaration += u32(0)
    base = text('mesh') + u32(0, 1, 3, 3 if normal else 0, 0, 0, 0)
    base += f32(0, 0, 0, 1000, 0, 0, 0, 1000, 0)
    if normal:
        base += f32(0, 0, 1) * 3
    base += u32(0)
    for index in range(3):
        base += u32(10 if bad_index and index == 2 else index)
        if normal:
            base += u32(index)
    shader = block(c.SHADER, text('shader') + u32(1) + f32(0) + u32(0x617, 0x606, 1, 0, 0) + text('red'))
    material = block(c.MATERIAL, text('red') + u32(63) + f32(.1, .1, .1, 1, 0, 0, .6, .6, .6, 0, 0, 0, .3, opacity))
    nodes = []
    for i, dx in enumerate((0, 2000)):
        matrix = np.eye(4); matrix[0, 3] = dx
        model = block(c.MODEL, text(f'part{i}') + u32(1) + text('') + f32(*matrix.T.reshape(-1)) + text('mesh') + u32(3))
        shade = text(f'part{i}') + u32(1, 15, 1, shader_count) + text('shader') * shader_count
        nodes.append(chain(f'part{i}', model, block(c.SHADING, shade)))
    return header + b''.join(nodes) + chain('mesh', block(c.DECL, declaration)) + shader + material + block(c.BASE, base) + (block(c.PROGRESSIVE, text('mesh')) if progressive else b'')


def read_glb(path):
    data = path.read_bytes()
    magic, version, length = struct.unpack_from('<4sII', data)
    assert (magic, version, length) == (b'glTF', 2, len(data))
    size, kind = struct.unpack_from('<I4s', data, 12)
    assert kind == b'JSON'
    doc = json.loads(data[20:20 + size])
    binary_size, kind = struct.unpack_from('<I4s', data, 20 + size)
    assert kind == b'BIN\x00'
    binary = data[28 + size:]
    assert binary_size == len(binary)
    return doc, binary


def test_end_to_end_material_transform_instancing_and_cache(tmp_path):
    source = tmp_path / 'model.u3d'; source.write_bytes(fixture_bytes(opacity=.35))
    dest = tmp_path / 'nested/model.glb'
    report = c.convert(source, dest, tmp_path / 'cache')
    doc, binary = read_glb(dest)
    assert report['visible_instances'] == 2
    assert report['triangles_in_scene'] == 2
    assert report['dimensions_m'] == pytest.approx([3, 1, 0])
    assert len(doc['meshes']) == 1
    assert doc['nodes'][1]['mesh'] == doc['nodes'][2]['mesh']
    assert doc['nodes'][2]['translation'][0] == pytest.approx(2)
    assert 'translation' not in doc['nodes'][0]
    material = doc['materials'][0]
    assert material['pbrMetallicRoughness']['baseColorFactor'] == pytest.approx([1, 0, 0, .35])
    assert material['alphaMode'] == 'BLEND'
    assert all(view['byteOffset'] % 4 == 0 for view in doc['bufferViews'])
    for view in doc['bufferViews']:
        assert view['byteOffset'] + view['byteLength'] <= len(binary)
    again = tmp_path / 'again.glb'
    c.convert(source, again, tmp_path / 'cache')
    assert dest.read_bytes() == again.read_bytes()


def test_pdf_extraction_and_no_3d(tmp_path):
    from pypdf import PdfWriter
    from pypdf.generic import ArrayObject, DictionaryObject, NameObject, DecodedStreamObject
    writer = PdfWriter(); page = writer.add_blank_page(100, 100)
    stream = DecodedStreamObject(); stream.set_data(fixture_bytes())
    stream[NameObject('/Subtype')] = NameObject('/U3D')
    ref = writer._add_object(stream)
    annotation = DictionaryObject({NameObject('/Subtype'): NameObject('/3D'), NameObject('/3DD'): ref})
    page[NameObject('/Annots')] = ArrayObject([writer._add_object(annotation)])
    pdf = tmp_path / 'source.pdf'; writer.write(pdf)
    assert c.read_source(pdf) == fixture_bytes()
    c.convert(pdf, tmp_path / 'model.glb')
    writer = PdfWriter(); writer.add_blank_page(100, 100); writer.write(pdf)
    with pytest.raises(c.Unsupported, match='found 0'):
        c.read_source(pdf)


@pytest.mark.parametrize('kwargs,message', [
    ({'progressive': True}, 'progressive'),
    ({'shader_count': 2}, 'Multipass'),
    ({'flags': 1}, 'Per-corner'),
    ({'texture': True}, 'texture layers'),
    ({'bad_index': True}, 'index out of range'),
])
def test_unsupported_and_invalid_content(tmp_path, kwargs, message):
    source = tmp_path / 'bad.u3d'; source.write_bytes(fixture_bytes(**kwargs))
    dest = tmp_path / 'bad.glb'
    with pytest.raises(ValueError, match=message):
        c.convert(source, dest)
    assert not dest.exists()


def test_normals_generated_when_source_excludes_them(tmp_path):
    source = tmp_path / 'flat.u3d'; source.write_bytes(fixture_bytes(normal=False))
    scene = c.parse_scene(source.read_bytes())
    v, n, faces, shading = c.decode_geometry(scene['declarations']['mesh'], scene['bases']['mesh'], True)
    assert n == pytest.approx(np.array([[0, 0, 1]] * 3))
    assert faces.tolist() == [[0, 1, 2]]


@pytest.mark.parametrize('data,message', [(b'bad', 'signature'), (b'U3D\0', 'header'),
                                         (u32(1, 50, 0), 'payload')])
def test_corrupt_blocks(data, message):
    with pytest.raises(ValueError, match=message):
        if message == 'signature':
            c.parse_scene(data)
        else:
            list(c.blocks(data))


def test_hierarchy_nested_missing_parent_cycle_and_affine():
    a = np.eye(4); a[0, 3] = 1
    b = np.eye(4); b[1, 3] = 2
    nodes = {'a': dict(parent='', matrix=a), 'b': dict(parent='a', matrix=b)}
    assert c.world_matrices(nodes)['b'][:3, 3] == pytest.approx([1, 2, 0])
    nodes['a']['parent'] = 'b'
    with pytest.raises(ValueError, match='Cycle'):
        c.world_matrices(nodes)
    nodes['a']['parent'] = 'missing'
    with pytest.raises(ValueError, match='Missing parent'):
        c.world_matrices(nodes)
    nodes['a']['parent'] = ''; nodes['a']['matrix'][3, 0] = 1
    with pytest.raises(ValueError, match='affine'):
        c.world_matrices(nodes)


def test_linear_color():
    assert c.linear_color([0, .5, 1]) == pytest.approx([0, .21404114, 1])
    assert c.linear_color([-1, 0.02, 2]) == pytest.approx([0, .02 / 12.92, 1])


def test_invalid_extension(tmp_path):
    with pytest.raises(c.Unsupported, match='Input must'):
        c.read_source(tmp_path / 'model.stl')


@pytest.mark.parametrize('angle', [0, 10, 90, 180, 270])
@pytest.mark.parametrize('mirror', [False, True])
def test_trs_preserves_large_cad_scales_and_rotations(angle, mirror):
    from trimesh.transformations import euler_matrix, quaternion_matrix
    m = np.eye(4)
    original = euler_matrix(*np.deg2rad([angle, angle / 2, angle]))[:3, :3]
    m[:3, :3] = original @ np.diag([-1000 if mirror else 1000, 999.99, 1000.01])
    m[:3, 3] = [1, 2, 3]
    trs = c.node_trs(m)
    q = trs['rotation']
    reconstructed = quaternion_matrix([q[3], *q[:3]])[:3, :3] @ np.diag(trs['scale'])
    assert reconstructed == pytest.approx(m[:3, :3], abs=1e-9)
    assert trs['translation'] == [1, 2, 3]


def test_trs_singular_and_shear():
    m = np.eye(4); m[0, 0] = 0
    with pytest.raises(c.Unsupported, match='Singular'):
        c.node_trs(m)
    m = np.eye(4); m[0, 1] = .1
    with pytest.raises(c.Unsupported, match='Sheared'):
        c.node_trs(m)


@pytest.mark.parametrize('axis', [0, 1, 2])
def test_trs_half_turns(axis):
    m = np.eye(4)
    m[:3, :3] = -np.eye(3); m[axis, axis] = 1
    result = c.node_trs(m)
    expected = [0, 0, 0, 0]; expected[axis] = 1
    assert result['rotation'] == pytest.approx(expected)


def rewrite(raw, kind, transform):
    result = b''
    for bt, data in c.blocks(raw):
        if bt == c.CHAIN:
            inner = c.chain_blocks(data)
            payload = b''.join(block(t, transform(d) if t == kind else d) for t, d in inner)
            offset = len(data) - sum(len(block(t, d)) for t, d in inner)
            data = data[:offset] + payload
        elif bt == kind:
            data = transform(data)
        result += block(bt, data)
    return result


@pytest.mark.parametrize('kind,transform,message', [
    (c.MODEL, lambda d: d[:len(text('part0'))] + u32(2) + d[len(text('part0')) + 4:], 'Multiple-parent'),
    (c.SHADER, lambda d: d[:len(text('shader'))] + u32(3) + d[len(text('shader')) + 4:], 'shader effects'),
    (c.MATERIAL, lambda d: d[:len(text('red'))] + u32(61) + d[len(text('red')) + 4:], 'explicit diffuse'),
    (c.HEADER, lambda d: d[:-8] + struct.pack('<d', 0), 'unit scale'),
    (c.BASE, lambda d: d[:len(text('mesh')) + 16] + u32(1) + d[len(text('mesh')) + 20:], 'Vertex color arrays'),
    (c.BASE, lambda d: d[:len(text('mesh')) + 28] + f32(float('nan')) + d[len(text('mesh')) + 32:], 'Non-finite'),
    (c.BASE, lambda d: d[:len(text('mesh')) + 64] + f32(0, 0, 0) + d[len(text('mesh')) + 76:], 'Zero-length'),
    (c.BASE, lambda d: d[:-28] + u32(1) + d[-24:], 'shading index'),
])
def test_data_failure_branches(tmp_path, kind, transform, message):
    source = tmp_path / 'bad.u3d'; source.write_bytes(rewrite(fixture_bytes(), kind, transform))
    with pytest.raises(ValueError, match=message):
        c.convert(source, tmp_path / 'bad.glb')


@pytest.mark.parametrize('kind,message', [(c.MODEL, 'Duplicate node'), (c.DECL, 'Duplicate geometry')])
def test_duplicate_data(kind, message):
    raw = fixture_bytes()
    data = next(d for bt, d in c.blocks(raw) if bt == c.CHAIN)
    duplicate = next(d for bt, d in c.chain_blocks(data) if bt == c.MODEL)
    if kind == c.DECL:
        data = next(d for bt, d in c.blocks(raw) if bt == c.CHAIN and any(t == c.DECL for t, _ in c.chain_blocks(d)))
        duplicate = next(d for bt, d in c.chain_blocks(data) if bt == c.DECL)
    with pytest.raises(ValueError, match=message):
        c.parse_scene(raw + block(kind, duplicate))


def test_missing_base_and_truncated_arrays():
    raw = fixture_bytes()
    raw = b''.join(block(t, d) for t, d in c.blocks(raw) if t != c.BASE)
    with pytest.raises(c.Unsupported, match='must match'):
        c.parse_scene(raw)
    scene = c.parse_scene(fixture_bytes())
    with pytest.raises(ValueError, match='Truncated mesh arrays'):
        c.decode_geometry(scene['declarations']['mesh'], scene['bases']['mesh'][:50], True)


@pytest.mark.parametrize('change,message', [
    ('resource', 'Missing geometry'), ('shading', 'No explicit shading'),
    ('shader', 'Missing shader'), ('material', 'Missing material'),
    ('visibility', 'double-sided'), ('hidden', 'No visible'),
])
def test_scene_reference_failures(tmp_path, change, message):
    raw = fixture_bytes()
    if change == 'resource':
        raw = rewrite(raw, c.MODEL, lambda d: d.replace(text('mesh'), text('gone')))
    elif change == 'shading':
        raw = rewrite(raw, c.SHADING, lambda d: d[:len(text('part0')) + 4] + u32(0) + d[len(text('part0')) + 8:])
    elif change == 'shader':
        raw = rewrite(raw, c.SHADING, lambda d: d.replace(text('shader'), text('missing')))
    elif change == 'material':
        raw = rewrite(raw, c.SHADER, lambda d: d.replace(text('red'), text('missing')))
    else:
        raw = rewrite(raw, c.MODEL, lambda d: d[:-4] + u32(0 if change == 'hidden' else 1))
    source = tmp_path / 'bad.u3d'; source.write_bytes(raw)
    with pytest.raises(ValueError, match=message):
        c.convert(source, tmp_path / 'bad.glb')


def test_cli_success_and_invalid_output(tmp_path):
    import subprocess
    source = tmp_path / 'ok.u3d'; source.write_bytes(fixture_bytes())
    script = Path(c.__file__)
    command = [sys.executable, str(script), str(source)]
    result = subprocess.run(command + [str(tmp_path / 'ok.glb')], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run(command + [str(tmp_path / 'bad.obj')], capture_output=True, text=True)
    assert result.returncode != 0 and 'Output must' in result.stderr
    source.write_bytes(b'bad')
    result = subprocess.run(command + [str(tmp_path / 'bad.glb')], capture_output=True, text=True)
    assert result.returncode == 1 and 'signature' in result.stderr
