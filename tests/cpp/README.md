# C++ Backend Unit Tests

## Installation

Build the chosen test extension with `uv` after installing a compatible PyTorch first.

```
uv venv .venv --python 3.12
source .venv/bin/activate

uv pip install --python .venv/bin/python "setuptools>=69" wheel packaging
uv pip install --python .venv/bin/python "torch==2.10.0" \
  --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python numpy ninja

MINKOWSKI_CPU_ONLY=1 MINKOWSKI_BLAS=openblas \
  uv pip install --python .venv/bin/python --no-build-isolation -v \
  ./tests/cpp --config-setting="--test=coordinate_map_key"
```

## Individual Test

```
uv run --no-sync --python .venv/bin/python python -m unittest <test_name>
```

To rebuild the extension without debug flags, add `--config-setting="--nodebug"` to the `uv pip install` command.

e.g.

```
uv run --no-sync --python .venv/bin/python python -m unittest coordinate_map_key_test
```
