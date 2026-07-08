#!/usr/bin/env bash
set -euo pipefail

: "${PYTHON_BIN:=.venv/bin/python}"
: "${MINKOWSKI_BLAS:=openblas}"

run_test() {
  local target="$1"
  local module="$2"
  local mode="${3:-cpu}"

  if [[ "$mode" == "gpu" ]]; then
    MINKOWSKI_FORCE_CUDA=1 MINKOWSKI_BLAS="$MINKOWSKI_BLAS" \
      uv pip install --python "$PYTHON_BIN" --no-build-isolation -v ./tests/cpp \
      --config-setting="--test=${target}"
  else
    MINKOWSKI_CPU_ONLY=1 MINKOWSKI_BLAS="$MINKOWSKI_BLAS" \
      uv pip install --python "$PYTHON_BIN" --no-build-isolation -v ./tests/cpp \
      --config-setting="--test=${target}"
  fi
  uv run --no-sync --python "$PYTHON_BIN" python -m unittest "$module"
}

run_test coordinate coordinate_test
run_test coordinate_map_key coordinate_map_key_test
run_test coordinate_map_cpu coordinate_map_cpu_test
run_test coordinate_map_gpu coordinate_map_gpu_test gpu
run_test kernel_region_cpu kernel_region_cpu_test
