import os
import torch
import glob

from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)

# ------------------------------------------- #

LIBRARY_NAME = "mgs"
DIFF_RENDERER_NAME = "diff_renderer"

if torch.__version__ >= "2.6.0":
    limitedAPI = True
else:
    limitedAPI = False

def get_extensions():
    debug = os.getenv("DEBUG", "0") == "1"
    useCuda = os.getenv("USE_CUDA", "1") == "1"
    if debug:
        print("Compiling in debug mode")

    useCuda = useCuda and torch.cuda.is_available() and CUDA_HOME is not None
    extension = CUDAExtension if useCuda else CppExtension

    linkArgs = []
    compileArgs = {
        "cxx": [
            "-O3" if not debug else "-O0",
            "-fdiagnostics-color=always",
            "-DPy_LIMITED_API=0x03090000",
        ],
        "nvcc": [
            "-O3" if not debug else "-O0",
        ],
    }
    if debug:
        compileArgs["cxx"].append("-g")
        compileArgs["nvcc"].append("-g")
        linkArgs.extend(["-O0", "-g"])

    extDir = os.path.join(os.path.dirname(os.path.curdir), LIBRARY_NAME, DIFF_RENDERER_NAME, "src")
    sources = list(glob.glob(os.path.join(extDir, "*.cpp")))

    extDirCuda = os.path.join(extDir, "cuda")
    sourcesCuda = list(glob.glob(os.path.join(extDirCuda, "*.cu")))
    if useCuda:
        sources += sourcesCuda

    extModules = [
        extension(
            f"{LIBRARY_NAME}.{DIFF_RENDERER_NAME}._C",
            sources,
            extra_compile_args=compileArgs,
            extra_link_args=linkArgs,
            py_limited_api=limitedAPI,
        )
    ]
    return extModules

# ------------------------------------------- #

setup(
    name=LIBRARY_NAME,
    version="0.0.1",
    packages=find_packages(),
    ext_modules=get_extensions(),
    install_requires=["torch"],
    description="Differentiable 4D Gaussian Splat Renderer",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}} if limitedAPI else {},
)
