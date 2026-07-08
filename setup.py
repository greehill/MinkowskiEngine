from __future__ import annotations

import importlib.util
import sys
try:
    import torch  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "PyTorch is required before building MinkowskiEngine. "
        "Install a matching torch build first, then run `uv pip install --no-build-isolation -v .`."
    ) from exc

from pathlib import Path

from setuptools import setup


def _load_build_helpers():
    helper_path = Path(__file__).resolve().with_name("build_helpers.py")
    spec = importlib.util.spec_from_file_location("build_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load build helper module from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    build_helpers = _load_build_helpers()
    setup(
        ext_modules=build_helpers.build_main_extension(),
        cmdclass=build_helpers.build_ext_command(),
    )
