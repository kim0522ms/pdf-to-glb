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

Also accepts `.u3d` files. Preserves geometry, colors, component placement, and source coordinates.
Colors are assumed to be sRGB. Gloss and reflections are approximated with PBR materials.

## Tests

```sh
python -m pip install -r requirements-dev.txt
python -m pytest tests -q --cov=pdf_to_glb --cov-branch
```

## License

Project code is licensed under MIT. The U3D decoder retains its Apache-2.0 notices.
See `LICENSE`, `THIRD_PARTY_NOTICE.txt`, and `LICENSE-APACHE-2.0.txt`.
