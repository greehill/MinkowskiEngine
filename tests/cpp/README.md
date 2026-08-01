# C++ Backend Unit Tests

## Installation

Build the chosen test extension with `uv` after installing a compatible PyTorch first.
The extension is an internal test target, not an installable Python project, so
invoke its setup command from the repository root.

```
uv venv .venv --python 3.12
source .venv/bin/activate

uv pip install --python .venv/bin/python "setuptools>=77" wheel packaging
uv pip install --python .venv/bin/python "torch==2.10.0" \
  --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python numpy ninja

MINKOWSKI_CPU_ONLY=1 MINKOWSKI_BLAS=openblas \
  uv run --no-sync --python .venv/bin/python python tests/cpp/setup.py \
  --test=coordinate_map_key --nodebug build_ext \
  --build-temp build/cpp-tests/coordinate_map_key/temp \
  --build-lib build/cpp-tests/coordinate_map_key/lib

PYTHONPATH="$PWD/build/cpp-tests/coordinate_map_key/lib:$PWD/tests/cpp:$PWD" \
  uv run --no-sync --python .venv/bin/python python -m unittest \
  tests.cpp.coordinate_map_key_test
```

## Individual Test

```
uv run --no-sync --python .venv/bin/python python -m unittest <test_name>
```

Omit `--nodebug` when a debug-symbol build is needed.

e.g.

```
PYTHONPATH="$PWD/build/cpp-tests/coordinate_map_key/lib:$PWD/tests/cpp:$PWD" \
  uv run --no-sync --python .venv/bin/python python -m unittest \
  tests.cpp.coordinate_map_key_test
```
