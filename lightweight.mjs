import {Logger, NodeIO, PropertyType} from '@gltf-transform/core';
import {getBounds, join, normals, prune, simplifyPrimitive, weld} from '@gltf-transform/functions';
import {MeshoptSimplifier} from 'meshoptimizer';
import {pathToFileURL} from 'node:url';

function triangleCount(document) {
  return document.getRoot().listNodes().reduce((total, node) => total +
    (node.getMesh()?.listPrimitives().reduce((sum, primitive) =>
      sum + primitive.getIndices().getCount() / 3, 0) ?? 0), 0);
}

export async function makeLightweight(source, destination, options) {
  await MeshoptSimplifier.ready;
  const io = new NodeIO();
  const document = await io.read(source);
  document.setLogger(new Logger(Logger.Verbosity.SILENT));
  const originalTriangles = triangleCount(document);
  await document.transform(weld());
  for (const mesh of document.getRoot().listMeshes()) {
    for (const primitive of mesh.listPrimitives()) {
      // Attribute-aware simplification keeps hard-edge normals in the error metric.
      const simplifier = {
        simplify(indices, positions, stride, target, error, flags) {
          return MeshoptSimplifier.simplifyWithAttributes(
            indices, positions, stride, primitive.getAttribute('NORMAL').getArray(),
            3, [0.01, 0.01, 0.01], null, Math.max(3, target), error, [...flags, 'Permissive']);
        },
      };
      simplifyPrimitive(primitive, {simplifier, ratio: options.ratio,
        error: options.error, lockBorder: options.lock_border});
    }
  }
  // New triangles need normals matching their faces, rather than the original tessellation.
  await document.transform(normals({overwrite: true}));
  for (const mesh of document.getRoot().listMeshes()) {
    for (const primitive of mesh.listPrimitives()) {
      const array = primitive.getAttribute('NORMAL').getArray();
      for (let i = 0; i < array.length; i += 3) {
        // Zero-area triangles have no face normal; provide a valid unit fallback.
        if (Math.hypot(array[i], array[i + 1], array[i + 2]) === 0) array[i + 2] = 1;
      }
    }
  }
  await document.transform(weld());
  if (options.merge_parts) await document.transform(join({keepNamed: false}));
  // Keep occurrence nodes, names, transforms, materials, and shared meshes.
  await document.transform(prune({propertyTypes: [PropertyType.ACCESSOR],
    keepAttributes: true, keepIndices: true}));
  await io.write(destination, document);
  const bounds = getBounds(document.getRoot().getDefaultScene());
  return {original_triangles: originalTriangles, output_triangles: triangleCount(document),
    output_meshes: document.getRoot().listMeshes().length,
    output_materials: document.getRoot().listMaterials().length,
    output_instances: document.getRoot().listNodes().filter(node => node.getMesh()).length,
    output_dimensions_m: bounds.max.map((value, index) => value - bounds.min[index])};
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    const result = await makeLightweight(process.argv[2], process.argv[3], JSON.parse(process.argv[4]));
    console.log(JSON.stringify(result));
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}
