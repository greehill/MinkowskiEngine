#!/usr/bin/env bash
set -euo pipefail

: "${PYTHON_BIN:=.venv/bin/python}"
: "${MINKOWSKI_BLAS:=openblas}"

run_test() {
  local target="$1"
  local module="$2"
  local build_root="build/cpp-tests/${target}"

  # CPP_TEST_SOURCE_SETS selects CppExtension or CUDAExtension for each target.
  MINKOWSKI_BLAS="$MINKOWSKI_BLAS" \
    uv run --no-sync --python "$PYTHON_BIN" python tests/cpp/setup.py \
    "--test=${target}" --nodebug build_ext \
    --build-temp "${build_root}/temp" --build-lib "${build_root}/lib"
  PYTHONPATH="${PWD}/${build_root}/lib:${PWD}/tests/cpp:${PWD}" \
    uv run --no-sync --python "$PYTHON_BIN" python -m unittest "tests.cpp.${module}"
}

run_test coordinate coordinate_test
run_test coordinate_map_key coordinate_map_key_test
run_test coordinate_map_cpu coordinate_map_cpu_test
run_test coordinate_map_gpu coordinate_map_gpu_test
run_test kernel_region_cpu kernel_region_cpu_test
