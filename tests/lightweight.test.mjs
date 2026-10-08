import assert from 'node:assert/strict';
import {mkdtemp, rm, stat} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {test} from 'node:test';
import {Document, NodeIO} from '@gltf-transform/core';
import {makeLightweight} from '../lightweight.mjs';

async function grid(path) {
  const document = new Document();
  const buffer = document.createBuffer();
  const mesh = document.createMesh('panel');
  for (const [name, color, offset] of [['red', [1, 0, 0, 1], 0], ['blue', [0, 0, 1, 0.5], 2]]) {
    const positions = [], normals = [], indices = [];
    for (let y = 0; y <= 20; y++) for (let x = 0; x <= 20; x++) {
      positions.push(offset + x / 20, y / 20, 0);
      // Deliberately stale source normals must be replaced with normals of the output faces.
      normals.push(0, 1, 0);
    }
    for (let y = 0; y < 20; y++) for (let x = 0; x < 20; x++) {
      const a = y * 21 + x;
      indices.push(a, a + 1, a + 21, a + 1, a + 22, a + 21);
    }
    const accessor = (array, type) => document.createAccessor().setBuffer(buffer).setType(type).setArray(array);
    const material = document.createMaterial(name).setBaseColorFactor(color).setDoubleSided(true);
    if (color[3] < 1) material.setAlphaMode('BLEND');
    mesh.addPrimitive(document.createPrimitive().setMaterial(material)
      .setAttribute('POSITION', accessor(new Float32Array(positions), 'VEC3'))
      .setAttribute('NORMAL', accessor(new Float32Array(normals), 'VEC3'))
      .setIndices(accessor(new Uint32Array(indices), 'SCALAR')));
  }
  const scene = document.createScene();
  scene.addChild(document.createNode('first').setMesh(mesh));
  scene.addChild(document.createNode('second').setMesh(mesh).setTranslation([10, 2, 3]));
  document.getRoot().setDefaultScene(scene);
  await new NodeIO().write(path, document);
}

test('reduces geometry, regenerates normals, and retains materials and shared occurrences', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'pdf-to-glb-'));
  try {
    const input = join(directory, 'input.glb');
    await grid(input);
    const results = [];
    for (const ratio of [0.5, 0.1]) {
      const output = join(directory, `${ratio}.glb`);
      const result = await makeLightweight(input, output, {ratio, error: 0.01, lock_border: false});
      results.push(result.output_triangles);
      assert.equal(result.original_triangles, 3200);
      assert.ok(result.output_triangles <= 3200 * ratio);
      assert.ok((await stat(output)).size < (await stat(input)).size);
      const document = await new NodeIO().read(output);
      const nodes = document.getRoot().listNodes();
      assert.deepEqual(nodes.map(node => node.getName()), ['first', 'second']);
      assert.deepEqual(nodes[1].getTranslation(), [10, 2, 3]);
      assert.equal(nodes[0].getMesh(), nodes[1].getMesh());
      const primitives = nodes[0].getMesh().listPrimitives();
      assert.deepEqual(primitives.map(p => p.getMaterial().getBaseColorFactor()), [[1, 0, 0, 1], [0, 0, 1, 0.5]]);
      assert.equal(primitives[1].getMaterial().getAlphaMode(), 'BLEND');
      for (const primitive of primitives) {
        const normal = primitive.getAttribute('NORMAL');
        assert.equal(normal.getCount(), primitive.getAttribute('POSITION').getCount());
        assert.ok(Array.from(normal.getArray()).every((v, i) => v === (i % 3 === 2 ? 1 : 0)));
        assert.ok(Array.from(primitive.getIndices().getArray()).every(i => i < normal.getCount()));
      }
    }
    assert.ok(results[1] < results[0]);
    const locked = await makeLightweight(input, join(directory, 'locked.glb'),
      {ratio: 0.01, error: 0, lock_border: true});
    assert.ok(locked.output_triangles > 3200 * 0.01);
    const mergedPath = join(directory, 'merged.glb');
    const merged = await makeLightweight(input, mergedPath,
      {ratio: 0.1, error: 0.01, lock_border: false, merge_parts: true});
    assert.equal(merged.output_triangles, results[1]);
    assert.equal(merged.output_instances, 1);
    assert.deepEqual(merged.output_dimensions_m, [13, 3, 3]);
    const mergedDocument = await new NodeIO().read(mergedPath);
    assert.deepEqual(mergedDocument.getRoot().listMaterials().map(m => m.getBaseColorFactor()),
      [[1, 0, 0, 1], [0, 0, 1, 0.5]]);
    const degenerate = await new NodeIO().read(input);
    const primitive = degenerate.getRoot().listMeshes()[0].listPrimitives()[0];
    primitive.getAttribute('POSITION').setArray(new Float32Array([0, 0, 0, 1, 0, 0, 2, 0, 0]));
    primitive.getAttribute('NORMAL').setArray(new Float32Array([0, 1, 0, 0, 1, 0, 0, 1, 0]));
    primitive.getIndices().setArray(new Uint32Array([0, 1, 2]));
    const degeneratePath = join(directory, 'degenerate.glb');
    await new NodeIO().write(degeneratePath, degenerate);
    const fixedPath = join(directory, 'fixed.glb');
    await makeLightweight(degeneratePath, fixedPath, {ratio: 1, error: 0, lock_border: false});
    const fixed = await new NodeIO().read(fixedPath);
    assert.deepEqual(Array.from(fixed.getRoot().listMeshes()[0].listPrimitives()[0].getAttribute('NORMAL').getArray()),
      [0, 0, 1, 0, 0, 1, 0, 0, 1]);
    await assert.rejects(makeLightweight(join(directory, 'missing.glb'), join(directory, 'bad.glb'),
      {ratio: 0.1, error: 0.01}), /ENOENT/);
  } finally {
    await rm(directory, {recursive: true, force: true});
  }
});
