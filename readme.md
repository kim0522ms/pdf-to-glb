# PDF to GLB

Converts PDFs containing U3D data to GLB. Supports CLOD meshes with solid-color materials.

## Installation

Requires Python 3.11 or later.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Usage

```sh
python pdf_to_glb.py input.pdf output.glb
```

Also accepts `.u3d` files. Preserves geometry, colors, component placement, source
coordinates, and assembly hierarchy. Source groups and components become GLB nodes
with their original names, parent-child relationships, and local transforms.
Unit conversion is applied at the `Model` root.
Colors are assumed to be sRGB. Gloss and reflections are approximated with PBR materials.

## Lightweight conversion

The default `--level none` preserves the original geometry. To enable simplification,
install Node.js 22 or later and the optional Node dependencies:

```sh
npm ci
python pdf_to_glb.py input.pdf output.glb --level medium
```

| Level | Target triangle ratio | Relative error limit |
| --- | --- | --- |
| `none` | Original geometry | No simplification |
| `low` | 0.50 | 0.001 |
| `medium` | 0.25 | 0.005 |
| `high` | 0.10 | 0.010 |

Override the selected level's parameters:

```sh
python pdf_to_glb.py input.pdf output.glb --level medium --ratio 0.3 --error 0.002 --lock-border
```

- `--ratio`: target fraction of triangles to retain, greater than 0 and up to 1.
- `--error`: relative simplification error limit, from 0 to 1, measured per mesh primitive.
  It is not an absolute distance in metres. A smaller value restricts changes.
- `--lock-border`: preserve open mesh boundaries. This can limit simplification.
- `--merge-parts`: merge components by material to reduce separate objects. Removes
  assembly hierarchy, individual component selection, and names. Shared instances are expanded, so the
  file can become larger. Disabled by default.
- `--cache-dir PATH`: cache decoded source geometry for repeated conversions.

Ratio, error, and border options require a lightweight level. Simplification can stop
before reaching the target because of the error limit or mesh topology. At least one
triangle is retained per primitive. Internal parts are included.

Without `--merge-parts`, lightweight output retains material colors, opacity, component names,
assembly hierarchy, placement, and shared mesh instances. It uses standard GLB buffers without Draco or
meshopt compression. Face normals are recalculated after simplification. Simplification
changes geometry and can affect surface appearance.
The adjacent `.report.json` records the options, original and output triangle counts,
file sizes, and dimensions. A failed simplification leaves an existing output intact.

## Tests

```sh
python -m pip install -r requirements-dev.txt
python -m pytest tests -q --cov=pdf_to_glb --cov-branch
npm test
```

The test suite requires `npm ci` for the lightweight conversion tests.

## License

Project code is licensed under MIT. The U3D decoder retains its Apache-2.0 notices.
See `LICENSE`, `THIRD_PARTY_NOTICE.txt`, and `LICENSE-APACHE-2.0.txt`.
