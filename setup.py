# TODO: fix this vibe-coded mess

import os
import glob
import pybind11
from setuptools import setup, find_packages, Extension

# Only import torch if diff renderer is enabled
ENABLE_DIFF = os.getenv("MGS_ENABLE_DIFF", "0") == "1"
if ENABLE_DIFF:
    import torch
    from torch.utils.cpp_extension import (
        CppExtension,
        CUDAExtension,
        BuildExtension,
        CUDA_HOME,
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def collect_sources(path, exts=(".c", ".cc", ".cpp", ".cu")):
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(path, f"*{ext}")))
    return files


# --------------------------------------------------------------------------- #
# Core extension: pure setuptools, no PyTorch or CUDA
# --------------------------------------------------------------------------- #

def build_core_extension():
    src_root = "csrc"
    include_dir = os.path.join(src_root, "include")
    src_dir = os.path.join(src_root, "src")

    core_sources = collect_sources(src_dir)
    core_sources.append(os.path.join("mgs", "ext.cpp"))

    return Extension(
        name="mgs._C",
        sources=core_sources,
        include_dirs=[
            include_dir,
            os.path.join(src_root, "external"),
			pybind11.get_include()
        ],
        extra_compile_args=["-O3", "-Wno-missing-braces"],
    )


# --------------------------------------------------------------------------- #
# Optional diff renderer: PyTorch + CUDA extension
# --------------------------------------------------------------------------- #

def build_diff_renderer_extension():
    if not ENABLE_DIFF:
        return []

    debug = os.getenv("DEBUG", "0") == "1"
    useCuda = (
        os.getenv("USE_CUDA", "1") == "1"
        and torch.cuda.is_available()
        and CUDA_HOME is not None
    )

    ExtType = CUDAExtension if useCuda else CppExtension

    src_root = "csrc"
    diff_src = os.path.join(src_root, "src", "diff_renderer")
    diff_inc = os.path.join(src_root, "include", "diff_renderer")

    diff_sources = collect_sources(diff_src)
    diff_sources.append(os.path.join("mgs", "diff_renderer", "ext.cpp"))

    cuda_sources = glob.glob(os.path.join(diff_src, "*.cu"))
    if useCuda:
        diff_sources += cuda_sources

    compile_args = {
        "cxx": ["-O3"],
        "nvcc": ["-O3"],
    }

    return ExtType(
        name="mgs.diff_renderer._C",
        sources=diff_sources,
        include_dirs=[
            os.path.join(src_root, "include"),
            diff_inc,
            os.path.join(src_root, "external"),
        ],
        extra_compile_args=compile_args,
    )


# --------------------------------------------------------------------------- #
# Extension list
# --------------------------------------------------------------------------- #

ext_modules = [build_core_extension()]
if ENABLE_DIFF:
    ext_modules.append(build_diff_renderer_extension())


# BuildExtension required only if diff renderer is active
cmdclass = {"build_ext": BuildExtension} if ENABLE_DIFF else {}


# --------------------------------------------------------------------------- #
# setup()
# --------------------------------------------------------------------------- #

setup(
    name="mgs",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[],  # core has no deps
    extras_require={
        "diff": ["torch"],  # optional: used only for diff renderer
    },
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
