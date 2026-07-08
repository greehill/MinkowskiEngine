# Quick Start

## Compatibility

- Python `3.10` to `3.14`
- PyTorch `2.5` to `2.10`
- Linux `x86_64` CUDA source builds for official PyTorch channels `cu124`, `cu126`, `cu128`, and `cu130`
- Linux and macOS CPU-only source builds

Notes:

- Python `3.14` is validated with PyTorch `2.9` and `2.10`
- CUDA `13.1` is not supported in this repository yet

## Install with `uv`

Install PyTorch first, then build MinkowskiEngine from source with `uv`.

CPU-only:

```bash
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine

uv venv .venv --python 3.12
source .venv/bin/activate

uv pip install --python .venv/bin/python "setuptools>=69" wheel packaging
uv pip install --python .venv/bin/python "torch==2.10.0" \
  --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python numpy ninja

MINKOWSKI_CPU_ONLY=1 MINKOWSKI_BLAS=openblas \
  uv pip install --python .venv/bin/python --no-build-isolation -v .
```

CUDA on Linux `x86_64`:

```bash
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine

uv venv .venv --python 3.13
source .venv/bin/activate

uv pip install --python .venv/bin/python "setuptools>=69" wheel packaging
uv pip install --python .venv/bin/python "torch==2.10.0" \
  --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/bin/python numpy ninja

export CUDA_HOME=/usr/local/cuda-13.0
MINKOWSKI_FORCE_CUDA=1 MINKOWSKI_BLAS=openblas \
  uv pip install --python .venv/bin/python --no-build-isolation -v .
```

## Build controls

- `MINKOWSKI_CPU_ONLY=1` forces CPU-only builds
- `MINKOWSKI_FORCE_CUDA=1` forces CUDA builds and requires both CUDA-enabled torch and `CUDA_HOME`
- `MINKOWSKI_BLAS` selects the BLAS backend
- `MINKOWSKI_BLAS_INCLUDE_DIRS` and `MINKOWSKI_BLAS_LIBRARY_DIRS` override BLAS discovery
- `TORCH_CUDA_ARCH_LIST`, `CXX`, `MAX_JOBS`, and `USE_NINJA` are still honored

## Running an example

```bash
uv run --no-sync --python .venv/bin/python python -m examples.indoor
```
