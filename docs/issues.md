# Common Issues

## `CUDA_HOME` points at the wrong toolkit

MinkowskiEngine must build against the same CUDA major/minor version exposed by the installed PyTorch wheel channel. If torch was installed from `cu130`, then `CUDA_HOME` should point at a CUDA `13.0` toolkit.

Check the current value:

```bash
echo "$CUDA_HOME"
nvcc --version
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
PY
```

If `CUDA_HOME` is wrong, export the correct toolkit path and rebuild:

```bash
export CUDA_HOME=/usr/local/cuda-13.0
MINKOWSKI_FORCE_CUDA=1 MINKOWSKI_BLAS=openblas \
  uv pip install --python .venv/bin/python --no-build-isolation -v .
```

## `cublas_v2.h` or other CUDA headers are missing

This usually means PyTorch is CUDA-enabled but the local toolkit is missing or `CUDA_HOME` is unset.

```bash
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
MINKOWSKI_FORCE_CUDA=1 MINKOWSKI_BLAS=openblas \
  uv pip install --python .venv/bin/python --no-build-isolation -v .
```

## Build runs out of memory

The extension build uses parallel compilation by default. Limit the job count when building on small machines or shared clusters.

```bash
export MAX_JOBS=4
MINKOWSKI_CPU_ONLY=1 MINKOWSKI_BLAS=openblas \
  uv pip install --python .venv/bin/python --no-build-isolation -v .
```

## Undefined symbols after upgrading torch, CUDA, or the compiler

Rebuild in a fresh virtual environment after reinstalling the matching torch wheel.

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install --python .venv/bin/python "setuptools>=69" wheel packaging
uv pip install --python .venv/bin/python "torch==2.10.0" \
  --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python numpy ninja
MINKOWSKI_CPU_ONLY=1 MINKOWSKI_BLAS=openblas \
  uv pip install --python .venv/bin/python --no-build-isolation -v .
```

## CUDA version mismatch: `undefined symbol` or `invalid device function`

Install a torch build whose wheel channel matches the toolkit used for compilation. For example:

```bash
uv pip install --python .venv/bin/python "torch==2.10.0" \
  --index-url https://download.pytorch.org/whl/cu130
export CUDA_HOME=/usr/local/cuda-13.0
```

CUDA `13.1` should not be used with this repository until official PyTorch `cu131` wheels exist.

## GPU out-of-memory during training

Sparse batches can vary in size from step to step, which increases allocator fragmentation. Clearing the PyTorch cache periodically is still the recommended mitigation.

```python
def training(...):
    ...
    sinput = ME.SparseTensor(...)
    loss = criterion(...)
    loss.backward()
    optimizer.step()
    torch.cuda.empty_cache()
```

## Issues not listed

If you still hit an install or runtime issue, open an issue on the [MinkowskiEngine GitHub page](https://github.com/NVIDIA/MinkowskiEngine/issues).
