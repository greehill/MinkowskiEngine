# Quick Start

## Compatibility

- Python `3.10` to `3.14`
- PyTorch `2.5` to `2.10`
- Linux `x86_64` CUDA 12 source builds, with this release directly validated on `cu128`
- Linux and macOS CPU-only source builds

Notes:

- Python `3.14` is validated with PyTorch `2.9` and `2.10`
- CUDA 13 is not certified by this release

## Install with `uv`

Install PyTorch first, then build MinkowskiEngine from source with `uv`.

CPU-only:

```bash
sudo apt-get update
sudo apt-get install -y build-essential libopenblas-dev pkg-config

git clone --branch v0.6.0+greehill.1 --depth 1 \
  https://github.com/greehill/MinkowskiEngine.git
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
sudo apt-get update
sudo apt-get install -y build-essential libopenblas-dev pkg-config

git clone --branch v0.6.0+greehill.1 --depth 1 \
  https://github.com/greehill/MinkowskiEngine.git
cd MinkowskiEngine

uv venv .venv --python 3.12
source .venv/bin/activate

uv pip install --python .venv/bin/python "setuptools>=69" wheel packaging
uv pip install --python .venv/bin/python "torch==2.10.0" \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python numpy ninja

export CUDA_HOME=/usr/local/cuda-12.8
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
