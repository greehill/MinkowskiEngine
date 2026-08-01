from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

try:
    import torch  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "PyTorch is required before building the MinkowskiEngine C++ tests."
    ) from exc

from setuptools import setup

ROOT = Path(__file__).resolve().parents[2]


def _load_build_helpers():
    helper_path = ROOT / "build_helpers.py"
    spec = importlib.util.spec_from_file_location("build_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load build helper module from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _pop_arg(argv: list[str], flag: str, *, expects_value: bool = False):
    if expects_value:
        for index, value in enumerate(argv):
            if value.startswith(f"{flag}="):
                return argv.pop(index).split("=", 1)[1]
        raise RuntimeError(f"{flag} is required")
    if flag in argv:
        argv.remove(flag)
        return True
    return False


if __name__ == "__main__":
    test_target = _pop_arg(sys.argv, "--test", expects_value=True)
    debug = not _pop_arg(sys.argv, "--nodebug")
    build_helpers = _load_build_helpers()
    setup(
        name="MinkowskiEngineTest",
        packages=[],
        ext_modules=build_helpers.build_cpp_test_extension(test_target, debug=debug),
        cmdclass=build_helpers.build_ext_command(),
    )
