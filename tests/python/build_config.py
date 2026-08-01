# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from build_helpers import (
    BlasConfig,
    CPP_TEST_SOURCE_SETS,
    ROOT,
    SRC_PATH,
    _configure_macos_platform_environment,
    _macos_llvm_runtime_library_dirs,
    _macos_openmp_flags,
    _normalized_macos_deployment_target,
    build_cpp_test_extension,
    detect_blas_config,
    resolve_cuda_build_enabled,
)
from tests.python.common import DEFAULT_PLY_PATH


class TestBuildConfig(unittest.TestCase):
    def test_resolve_cuda_build_enabled(self):
        self.assertFalse(
            resolve_cuda_build_enabled("darwin", False, False, "12.8", "/usr/local/cuda")
        )
        self.assertFalse(
            resolve_cuda_build_enabled("linux", True, False, "12.8", "/usr/local/cuda")
        )
        self.assertFalse(resolve_cuda_build_enabled("linux", False, False, None, None))
        self.assertTrue(
            resolve_cuda_build_enabled("linux", False, True, "12.8", "/usr/local/cuda")
        )

    def test_conflicting_cuda_build_flags_fail(self):
        with self.assertRaisesRegex(RuntimeError, "mutually exclusive"):
            resolve_cuda_build_enabled("linux", True, True, "12.8", "/usr/local/cuda")

    def test_cpp_test_extension_uses_absolute_source_paths(self):
        blas = BlasConfig("openblas", ("openblas",), (), (), (), ())
        with mock.patch("build_helpers.detect_blas_config", return_value=blas), mock.patch(
            "build_helpers._common_compile_and_link_args",
            return_value=([], [], [], [], []),
        ):
            extension = build_cpp_test_extension("coordinate", debug=False)[0]

        self.assertTrue(extension.sources)
        self.assertTrue(all(Path(source).is_absolute() for source in extension.sources))

    def test_all_cpp_test_sources_exist(self):
        test_root = ROOT / "tests" / "cpp"
        for target, (_, test_files, source_files, _) in CPP_TEST_SOURCE_SETS.items():
            paths = [test_root / filename for filename in test_files]
            paths.extend(SRC_PATH / filename for filename in source_files)
            with self.subTest(target=target):
                missing = [str(path) for path in paths if not path.is_file()]
                self.assertEqual(missing, [])

    def test_blas_fallback_finds_linux_multiarch_library(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = Path(temp_dir)
            library_dir = prefix / "lib" / "x86_64-linux-gnu"
            library_dir.mkdir(parents=True)
            (library_dir / "libopenblas.so").touch()
            include_dir = prefix / "include"
            include_dir.mkdir()
            (include_dir / "cblas.h").touch()

            with (
                mock.patch.dict(
                    "os.environ", {"MINKOWSKI_BLAS": "openblas"}, clear=True
                ),
                mock.patch("build_helpers._run_pkg_config", return_value=None),
                mock.patch(
                    "build_helpers._candidate_prefixes", return_value=[prefix]
                ),
                mock.patch(
                    "build_helpers.sysconfig.get_config_var",
                    return_value="x86_64-linux-gnu",
                ),
            ):
                config = detect_blas_config()

        self.assertEqual(config.library_dirs, (str(library_dir),))
        self.assertEqual(config.include_dirs, (str(include_dir),))

    def test_blas_directory_overrides_require_backend(self):
        with mock.patch.dict(
            "os.environ",
            {"MINKOWSKI_BLAS_LIBRARY_DIRS": "/opt/custom-blas/lib"},
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "require MINKOWSKI_BLAS"):
                detect_blas_config()

    def test_local_point_cloud_fixture_exists(self):
        self.assertTrue(DEFAULT_PLY_PATH.is_file(), DEFAULT_PLY_PATH)

    def test_macos_openmp_flags_for_apple_clang(self):
        compiler = Path("/usr/bin/clang++")
        with mock.patch(
            "build_helpers._compiler_version_output",
            return_value="Apple clang version 17.0.0",
        ):
            self.assertEqual(_macos_openmp_flags(compiler), ["-Xpreprocessor", "-fopenmp"])

    def test_macos_openmp_flags_for_llvm_clang(self):
        compiler = Path("/opt/homebrew/opt/llvm/bin/clang++")
        with mock.patch(
            "build_helpers._compiler_version_output",
            return_value="clang version 22.1.0",
        ):
            self.assertEqual(_macos_openmp_flags(compiler), ["-fopenmp"])

    def test_macos_llvm_runtime_library_dirs_only_when_using_homebrew_llvm(self):
        llvm_prefix = Path("/opt/homebrew/opt/llvm")
        self.assertEqual(
            _macos_llvm_runtime_library_dirs(Path("/usr/bin/clang++"), llvm_prefix),
            [],
        )
        self.assertEqual(
            _macos_llvm_runtime_library_dirs(
                Path("/opt/homebrew/opt/llvm/bin/clang++"),
                llvm_prefix,
            ),
            [
                "/opt/homebrew/opt/llvm/lib",
                "/opt/homebrew/opt/llvm/lib/c++",
            ],
        )

    def test_normalized_macos_deployment_target(self):
        self.assertEqual(_normalized_macos_deployment_target("15"), "15.0")
        self.assertEqual(_normalized_macos_deployment_target("11.2.1"), "11.2")

    def test_configure_macos_platform_environment_defaults_to_native_arch(self):
        with mock.patch("build_helpers.platform.machine", return_value="arm64"):
            with mock.patch.dict("os.environ", {}, clear=True):
                _configure_macos_platform_environment()
                self.assertEqual(os.environ["ARCHFLAGS"], "-arch arm64")
                self.assertEqual(os.environ["MACOSX_DEPLOYMENT_TARGET"], "11.0")
                self.assertEqual(os.environ["_PYTHON_HOST_PLATFORM"], "macosx-11.0-arm64")

    def test_configure_macos_platform_environment_preserves_existing_archflags(self):
        env = {
            "ARCHFLAGS": "-arch arm64 -arch x86_64",
            "MACOSX_DEPLOYMENT_TARGET": "13",
        }
        with mock.patch("build_helpers.platform.machine", return_value="arm64"):
            with mock.patch.dict("os.environ", env, clear=True):
                _configure_macos_platform_environment()
                self.assertEqual(os.environ["ARCHFLAGS"], "-arch arm64 -arch x86_64")
                self.assertEqual(os.environ["MACOSX_DEPLOYMENT_TARGET"], "13.0")
                self.assertEqual(os.environ["_PYTHON_HOST_PLATFORM"], "macosx-13.0-arm64")
