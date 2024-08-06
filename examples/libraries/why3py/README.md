# A Python Interface to Why3

## Installation

Just install with pip:

```sh
pip install -e .
```

If there is any problem with the installation, you can try running the build script manually:

```sh
python setup.py build
```

## Commands

Generate obligations:
```sh
pytest -k "prove" -rP --color=yes
```