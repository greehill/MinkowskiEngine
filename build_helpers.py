from __future__ import annotations

import os
import platform
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    CppExtension,
    CUDA_HOME,
)


ROOT = Path(__file__).resolve().parent
SRC_PATH = ROOT / "src"
PYBIND_PATH = ROOT / "pybind"
MAX_COMPILATION_THREADS = 12
BLAS_ORDER = ("openblas", "flexiblas", "mkl", "atlas", "blas")
PKG_CONFIG_CANDIDATES = {
    "openblas": ("openblas",),
    "flexiblas": ("flexiblas",),
    "mkl": ("mkl-dynamic-lp64-seq", "mkl-dynamic-lp64-iomp", "mkl-static-lp64-seq"),
    "atlas": ("atlas",),
    "blas": ("blas",),
}
COMMON_PREFIXES = (
    "/usr",
    "/usr/local",
    "/opt/homebrew",
    "/usr/local/opt",
)
MAIN_SOURCE_SETS = {
    "cpu": (
        CppExtension,
        (
            "math_functions_cpu.cpp",
            "coordinate_map_manager.cpp",
            "convolution_cpu.cpp",
            "convolution_transpose_cpu.cpp",
            "local_pooling_cpu.cpp",
            "local_pooling_transpose_cpu.cpp",
            "global_pooling_cpu.cpp",
            "broadcast_cpu.cpp",
            "pruning_cpu.cpp",
            "interpolation_cpu.cpp",
            "quantization.cpp",
            "direct_max_pool.cpp",
        ),
        ("minkowski.cpp",),
        ("-DCPU_ONLY",),
    ),
    "gpu": (
        CUDAExtension,
        (
            "math_functions_cpu.cpp",
            "math_functions_gpu.cu",
            "coordinate_map_manager.cu",
            "coordinate_map_gpu.cu",
            "convolution_kernel.cu",
            "convolution_gpu.cu",
            "convolution_transpose_gpu.cu",
            "pooling_avg_kernel.cu",
            "pooling_max_kernel.cu",
            "local_pooling_gpu.cu",
            "local_pooling_transpose_gpu.cu",
            "global_pooling_gpu.cu",
            "broadcast_kernel.cu",
            "broadcast_gpu.cu",
            "pruning_gpu.cu",
            "interpolation_gpu.cu",
            "spmm.cu",
            "gpu.cu",
            "quantization.cpp",
            "direct_max_pool.cpp",
        ),
        ("minkowski.cu",),
        (),
    ),
}
CPP_TEST_SOURCE_SETS = {
    "convolution_cpu": (
        CppExtension,
        ("convolution_test.cpp",),
        ("math_functions.cpp", "coordinate_map_manager.cpp", "convolution_cpu.cpp"),
        ("-DCPU_ONLY",),
    ),
    "convolution_gpu": (
        CUDAExtension,
        ("convolution_test.cu",),
        (
            "math_functions.cpp",
            "coordinate_map_manager.cu",
            "convolution_gpu.cu",
            "coordinate_map_gpu.cu",
            "convolution_kernel.cu",
        ),
        (),
    ),
    "coordinate_map_manager_cpu": (
        CppExtension,
        ("coordinate_map_manager_cpu_test.cpp",),
        ("coordinate_map_manager.cpp",),
        ("-DCPU_ONLY",),
    ),
    "coordinate_map_manager_gpu": (
        CUDAExtension,
        ("coordinate_map_manager_gpu_test.cu",),
        ("coordinate_map_manager.cu", "coordinate_map_gpu.cu"),
        (),
    ),
    "coordinate_map_key": (CppExtension, ("coordinate_map_key_test.cpp",), (), ()),
    "coordinate_map_cpu": (CppExtension, ("coordinate_map_cpu_test.cpp",), (), ()),
    "coordinate_map_gpu": (
        CUDAExtension,
        ("coordinate_map_gpu_test.cu",),
        ("coordinate_map_gpu.cu",),
        (),
    ),
    "coordinate": (CppExtension, ("coordinate_test.cpp",), (), ()),
    "kernel_region_cpu": (CppExtension, ("kernel_region_cpu_test.cpp",), (), ()),
    "kernel_region_gpu": (
        CUDAExtension,
        ("kernel_region_gpu_test.cu",),
        ("coordinate_map_gpu.cu",),
        (),
    ),
    "type": (CppExtension, ("type_test.cpp",), (), ()),
}


@dataclass(frozen=True)
class BlasConfig:
    name: str
    libraries: tuple[str, ...]
    include_dirs: tuple[str, ...] = ()
    library_dirs: tuple[str, ...] = ()
    extra_link_args: tuple[str, ...] = ()
    compile_defines: tuple[str, ...] = ()


def _parse_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"Invalid boolean value for {name}: {value!r}")


def _split_env_paths(name: str) -> list[str]:
    value = os.getenv(name, "")
    if not value:
        return []
    tokens = []
    current = []
    for chunk in value.replace(os.pathsep, ",").split(","):
        item = chunk.strip()
        if item:
            tokens.append(item)
    return tokens


def _append_unique(target: list[str], values: Iterable[str]) -> None:
    for value in values:
        if value and value not in target:
            target.append(value)


def _relative_source_paths(base: Path, filenames: Iterable[str]) -> list[str]:
    return [str((base / filename).relative_to(ROOT)).replace(os.sep, "/") for filename in filenames]


def _run_pkg_config(package: str) -> BlasConfig | None:
    if shutil.which("pkg-config") is None:
        return None
    try:
        output = subprocess.check_output(
            ["pkg-config", "--cflags", "--libs", package],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    include_dirs: list[str] = []
    library_dirs: list[str] = []
    libraries: list[str] = []
    extra_link_args: list[str] = []
    for token in shlex.split(output):
        if token.startswith("-I"):
            include_dirs.append(token[2:])
        elif token.startswith("-L"):
            library_dirs.append(token[2:])
        elif token.startswith("-l"):
            libraries.append(token[2:])
        else:
            extra_link_args.append(token)

    return BlasConfig(
        name=package,
        libraries=tuple(libraries),
        include_dirs=tuple(include_dirs),
        library_dirs=tuple(library_dirs),
        extra_link_args=tuple(extra_link_args),
    )


def _candidate_prefixes() -> list[Path]:
    prefixes = []
    for env_var in ("MINKOWSKI_BLAS_PREFIX", "CONDA_PREFIX", "VIRTUAL_ENV", "MKLROOT"):
        value = os.getenv(env_var)
        if value:
            prefixes.append(Path(value))
    prefixes.extend(Path(prefix) for prefix in COMMON_PREFIXES)
    resolved: list[Path] = []
    for prefix in prefixes:
        if prefix.exists() and prefix not in resolved:
            resolved.append(prefix)
    return resolved


def _header_candidates(prefix: Path) -> list[Path]:
    return [
        prefix / "include",
        prefix / "include" / "openblas",
        prefix / "opt" / "openblas" / "include",
        prefix / "opt" / "libomp" / "include",
    ]


def _library_candidates(prefix: Path) -> list[Path]:
    return [
        prefix / "lib",
        prefix / "lib64",
        prefix / "opt" / "openblas" / "lib",
        prefix / "opt" / "libomp" / "lib",
        prefix / "lib" / "c++",
        prefix / "lib64" / "c++",
    ]


def _find_library_dir(prefixes: Iterable[Path], lib_names: Iterable[str]) -> Path | None:
    suffixes = (".so", ".dylib", ".a")
    for prefix in prefixes:
        for candidate in _library_candidates(prefix):
            if not candidate.exists():
                continue
            for lib_name in lib_names:
                if any((candidate / f"lib{lib_name}{suffix}").exists() for suffix in suffixes):
                    return candidate
    return None


def _find_include_dir(prefixes: Iterable[Path], headers: Iterable[str]) -> Path | None:
    for prefix in prefixes:
        for candidate in _header_candidates(prefix):
            if not candidate.exists():
                continue
            for header in headers:
                if (candidate / header).exists():
                    return candidate
    return None


def detect_blas_config() -> BlasConfig:
    requested_name = os.getenv("MINKOWSKI_BLAS")
    requested_name = requested_name.lower() if requested_name else None
    if requested_name and requested_name not in BLAS_ORDER:
        valid = ", ".join(BLAS_ORDER)
        raise RuntimeError(f"MINKOWSKI_BLAS must be one of: {valid}")

    env_include_dirs = _split_env_paths("MINKOWSKI_BLAS_INCLUDE_DIRS")
    env_library_dirs = _split_env_paths("MINKOWSKI_BLAS_LIBRARY_DIRS")

    if requested_name and (env_include_dirs or env_library_dirs):
        libraries = ("mkl_rt",) if requested_name == "mkl" else (requested_name,)
        compile_defines = ("-DUSE_MKL",) if requested_name == "mkl" else ()
        return BlasConfig(
            name=requested_name,
            libraries=libraries,
            include_dirs=tuple(env_include_dirs),
            library_dirs=tuple(env_library_dirs),
            compile_defines=compile_defines,
        )

    names_to_try = [requested_name] if requested_name else list(BLAS_ORDER)
    for name in names_to_try:
        for package in PKG_CONFIG_CANDIDATES[name]:
            config = _run_pkg_config(package)
            if config and config.libraries:
                compile_defines = ("-DUSE_MKL",) if name == "mkl" else ()
                return BlasConfig(
                    name=name,
                    libraries=config.libraries,
                    include_dirs=config.include_dirs,
                    library_dirs=config.library_dirs,
                    extra_link_args=config.extra_link_args,
                    compile_defines=compile_defines,
                )

    prefixes = _candidate_prefixes()
    library_name_map = {
        "openblas": ("openblas",),
        "flexiblas": ("flexiblas",),
        "mkl": ("mkl_rt",),
        "atlas": ("atlas",),
        "blas": ("blas",),
    }
    header_map = {
        "openblas": ("cblas.h", "openblas_config.h", "openblas/cblas.h"),
        "flexiblas": ("cblas.h",),
        "mkl": ("mkl.h",),
        "atlas": ("cblas.h",),
        "blas": ("cblas.h",),
    }
    for name in names_to_try:
        lib_dir = _find_library_dir(prefixes, library_name_map[name])
        if lib_dir is None:
            continue
        include_dir = _find_include_dir(prefixes, header_map[name])
        include_dirs = (str(include_dir),) if include_dir else ()
        compile_defines = ("-DUSE_MKL",) if name == "mkl" else ()
        return BlasConfig(
            name=name,
            libraries=library_name_map[name],
            include_dirs=include_dirs,
            library_dirs=(str(lib_dir),),
            compile_defines=compile_defines,
        )

    searched_prefixes = ", ".join(str(prefix) for prefix in prefixes) or "<none>"
    raise RuntimeError(
        "Unable to locate a supported BLAS implementation. Set MINKOWSKI_BLAS and "
        "optionally MINKOWSKI_BLAS_INCLUDE_DIRS/MINKOWSKI_BLAS_LIBRARY_DIRS, or "
        f"install OpenBLAS under one of the searched prefixes: {searched_prefixes}."
    )


def resolve_cuda_build_enabled(
    system_name: str,
    cpu_only: bool,
    force_cuda: bool,
    torch_cuda_version: str | None,
    cuda_home: str | None,
) -> bool:
    if cpu_only:
        return False
    if system_name == "darwin":
        if force_cuda:
            raise RuntimeError("CUDA builds are not supported on macOS in this repository.")
        return False
    if system_name == "win32":
        raise RuntimeError("Windows is currently not supported.")
    if force_cuda:
        if torch_cuda_version is None:
            raise RuntimeError(
                "MINKOWSKI_FORCE_CUDA=1 was set, but the installed torch build does not provide CUDA."
            )
        if not cuda_home:
            raise RuntimeError(
                "MINKOWSKI_FORCE_CUDA=1 was set, but CUDA_HOME was not found. "
                "Install the matching CUDA toolkit and export CUDA_HOME."
            )
        return True
    return torch_cuda_version is not None and bool(cuda_home)


def _macos_sdkroot() -> str | None:
    if os.getenv("SDKROOT"):
        candidate = Path(os.environ["SDKROOT"])
        if candidate.exists():
            return str(candidate)
    try:
        sdkroot = subprocess.check_output(
            ["xcrun", "--show-sdk-path"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return sdkroot or None


def _homebrew_prefixes() -> list[Path]:
    return [Path("/opt/homebrew/opt"), Path("/usr/local/opt")]


def _macos_llvm_prefix() -> Path | None:
    for prefix in _homebrew_prefixes():
        candidate = prefix / "llvm"
        if candidate.exists():
            return candidate
    return None


def _macos_libomp_prefix() -> Path | None:
    for prefix in _homebrew_prefixes():
        candidate = prefix / "libomp"
        if candidate.exists():
            return candidate
    return None


def _active_cxx_compiler() -> Path | None:
    for env_var in ("CXX", "CC"):
        value = os.getenv(env_var)
        if not value:
            continue
        command = shlex.split(value)
        if not command:
            continue
        resolved = shutil.which(command[0]) or command[0]
        return Path(resolved)

    for candidate in ("clang++", "c++"):
        resolved = shutil.which(candidate)
        if resolved:
            return Path(resolved)
    return None


def _compiler_version_output(compiler: Path | None) -> str:
    if compiler is None:
        return ""
    try:
        return subprocess.check_output(
            [str(compiler), "--version"], text=True, stderr=subprocess.STDOUT
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _macos_openmp_flags(compiler: Path | None) -> list[str]:
    if "apple clang" in _compiler_version_output(compiler).lower():
        return ["-Xpreprocessor", "-fopenmp"]
    return ["-fopenmp"]


def _macos_llvm_runtime_library_dirs(compiler: Path | None, llvm_prefix: Path | None) -> list[str]:
    if compiler is None or llvm_prefix is None:
        return []

    compiler_path = compiler.expanduser().resolve(strict=False)
    llvm_root = llvm_prefix.expanduser().resolve(strict=False)
    if compiler_path != llvm_root and llvm_root not in compiler_path.parents:
        return []

    return [str(llvm_prefix / "lib"), str(llvm_prefix / "lib" / "c++")]


def _normalized_macos_deployment_target(value: str) -> str:
    parts = value.strip().split(".")
    if len(parts) == 1:
        return f"{parts[0]}.0"
    return ".".join(parts[:2])


def _configure_macos_platform_environment() -> None:
    machine = platform.machine().lower()
    if machine not in {"arm64", "x86_64"}:
        return

    if "ARCHFLAGS" not in os.environ:
        os.environ["ARCHFLAGS"] = f"-arch {machine}"

    default_target = "11.0" if machine == "arm64" else "10.9"
    deployment_target = _normalized_macos_deployment_target(
        os.getenv("MACOSX_DEPLOYMENT_TARGET", default_target)
    )
    os.environ["MACOSX_DEPLOYMENT_TARGET"] = deployment_target
    os.environ.setdefault("_PYTHON_HOST_PLATFORM", f"macosx-{deployment_target}-{machine}")


def build_environment_summary() -> dict[str, str | bool | None]:
    import torch

    cpu_only = _parse_bool_env("MINKOWSKI_CPU_ONLY", False)
    force_cuda = _parse_bool_env("MINKOWSKI_FORCE_CUDA", False)
    use_cuda = resolve_cuda_build_enabled(
        sys.platform, cpu_only, force_cuda, torch.version.cuda, CUDA_HOME
    )
    return {
        "platform": sys.platform,
        "torch_cuda_version": torch.version.cuda,
        "cuda_home": CUDA_HOME,
        "use_cuda": use_cuda,
        "cpu_only": not use_cuda,
        "blas": detect_blas_config().name,
    }


def _common_compile_and_link_args(use_cuda: bool) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    cxx_flags: list[str] = ["-std=c++17"]
    nvcc_flags: list[str] = ["--std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda"]
    extra_link_args: list[str] = []
    include_dirs: list[str] = []
    library_dirs: list[str] = []

    debug = _parse_bool_env("MINKOWSKI_DEBUG", False)
    if debug:
        _append_unique(cxx_flags, ["-g", "-DDEBUG"])
        _append_unique(nvcc_flags, ["-g", "-DDEBUG", "-Xcompiler=-fno-gnu-unique"])
    else:
        _append_unique(cxx_flags, ["-O3"])
        _append_unique(nvcc_flags, ["-O3", "-Xcompiler=-fno-gnu-unique"])

    if sys.platform == "darwin":
        _configure_macos_platform_environment()
        llvm_prefix = _macos_llvm_prefix()
        libomp_prefix = _macos_libomp_prefix()
        active_cxx = _active_cxx_compiler()
        if libomp_prefix is None:
            raise RuntimeError(
                "libomp was not found. Install it with Homebrew and retry: brew install libomp"
            )

        _append_unique(cxx_flags, ["-stdlib=libc++"])
        sdkroot = _macos_sdkroot()
        if sdkroot:
            os.environ["SDKROOT"] = sdkroot
            _append_unique(cxx_flags, ["-isysroot", sdkroot])
            _append_unique(extra_link_args, ["-isysroot", sdkroot, f"-Wl,-syslibroot,{sdkroot}"])

        _append_unique(include_dirs, [str(libomp_prefix / "include")])
        _append_unique(library_dirs, [str(libomp_prefix / "lib")])
        extra_link_args.append("-lomp")
        _append_unique(library_dirs, _macos_llvm_runtime_library_dirs(active_cxx, llvm_prefix))
        _append_unique(cxx_flags, _macos_openmp_flags(active_cxx))
    else:
        _append_unique(cxx_flags, ["-fopenmp"])

    if use_cuda:
        if "CXX" in os.environ:
            nvcc_flags.append(f"-ccbin={os.environ['CXX']}")
        elif "CC" in os.environ:
            nvcc_flags.append(f"-ccbin={os.environ['CC']}")

    return cxx_flags, nvcc_flags, extra_link_args, include_dirs, library_dirs


def _rpath_args(library_dirs: Iterable[str]) -> list[str]:
    prefix = "-Wl,-rpath,"
    return [f"{prefix}{library_dir}" for library_dir in library_dirs]


def _finalize_parallelism() -> None:
    if "MAX_JOBS" not in os.environ and os.cpu_count() and os.cpu_count() > MAX_COMPILATION_THREADS:
        os.environ["MAX_JOBS"] = str(MAX_COMPILATION_THREADS)


def build_main_extension():
    import torch

    cpu_only = _parse_bool_env("MINKOWSKI_CPU_ONLY", False)
    force_cuda = _parse_bool_env("MINKOWSKI_FORCE_CUDA", False)
    use_cuda = resolve_cuda_build_enabled(
        sys.platform, cpu_only, force_cuda, torch.version.cuda, CUDA_HOME
    )
    blas = detect_blas_config()
    cxx_flags, nvcc_flags, extra_link_args, include_dirs, library_dirs = _common_compile_and_link_args(
        use_cuda
    )
    _append_unique(cxx_flags, list(blas.compile_defines))
    _append_unique(nvcc_flags, list(blas.compile_defines))
    _append_unique(include_dirs, [str(SRC_PATH), str(SRC_PATH / "3rdparty"), *blas.include_dirs])
    _append_unique(library_dirs, list(blas.library_dirs))

    source_set = MAIN_SOURCE_SETS["gpu" if use_cuda else "cpu"]
    extension_cls, source_files, bind_files, define_flags = source_set
    _append_unique(cxx_flags, list(define_flags))
    _append_unique(nvcc_flags, list(define_flags))

    libraries = list(blas.libraries)
    if use_cuda:
        libraries.append("cusparse")
        if CUDA_HOME:
            _append_unique(library_dirs, [str(Path(CUDA_HOME) / "lib64")])

    _append_unique(extra_link_args, list(blas.extra_link_args))
    _append_unique(extra_link_args, _rpath_args(library_dirs))
    _finalize_parallelism()

    sources = _relative_source_paths(SRC_PATH, source_files)
    sources.extend(_relative_source_paths(PYBIND_PATH, bind_files))
    return [
        extension_cls(
            name="MinkowskiEngineBackend._C",
            sources=sources,
            extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            extra_link_args=extra_link_args,
        )
    ]


def build_cpp_test_extension(test_target: str, debug: bool):
    if test_target not in CPP_TEST_SOURCE_SETS:
        valid_targets = ", ".join(sorted(CPP_TEST_SOURCE_SETS))
        raise RuntimeError(f"Unsupported C++ test target {test_target!r}. Valid targets: {valid_targets}")

    import torch

    extension_cls, test_files, source_files, define_flags = CPP_TEST_SOURCE_SETS[test_target]
    wants_cuda = extension_cls is CUDAExtension
    use_cuda = wants_cuda and resolve_cuda_build_enabled(
        sys.platform, False, True, torch.version.cuda, CUDA_HOME
    )
    blas = detect_blas_config()
    cxx_flags, nvcc_flags, extra_link_args, include_dirs, library_dirs = _common_compile_and_link_args(
        use_cuda
    )
    if debug:
        if "-O3" in cxx_flags:
            cxx_flags.remove("-O3")
        if "-O3" in nvcc_flags:
            nvcc_flags.remove("-O3")
        _append_unique(cxx_flags, ["-g", "-DDEBUG"])
        _append_unique(nvcc_flags, ["-g", "-DDEBUG"])

    _append_unique(cxx_flags, list(blas.compile_defines))
    _append_unique(nvcc_flags, list(blas.compile_defines))
    _append_unique(cxx_flags, list(define_flags))
    _append_unique(nvcc_flags, list(define_flags))
    _append_unique(include_dirs, [str(SRC_PATH), str(SRC_PATH / "3rdparty"), *blas.include_dirs])
    _append_unique(library_dirs, list(blas.library_dirs))
    if use_cuda and CUDA_HOME:
        _append_unique(library_dirs, [str(Path(CUDA_HOME) / "lib64")])
    _append_unique(extra_link_args, list(blas.extra_link_args))
    _append_unique(extra_link_args, _rpath_args(library_dirs))
    _finalize_parallelism()

    test_root = ROOT / "tests" / "cpp"
    sources = _relative_source_paths(test_root, test_files)
    sources.extend(_relative_source_paths(SRC_PATH, source_files))
    return [
        extension_cls(
            name="MinkowskiEngineTest._C",
            sources=sources,
            extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=list(blas.libraries),
            extra_link_args=extra_link_args,
        )
    ]


def build_ext_command():
    use_ninja = os.getenv("USE_NINJA", "1") != "0"
    return {"build_ext": BuildExtension.with_options(use_ninja=use_ninja)}
