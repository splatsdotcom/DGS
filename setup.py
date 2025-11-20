import os
import glob
import pybind11
from setuptools import find_packages, setup, Extension

# ------------------------------------------- #

LIBRARY_NAME = "mgs"
CSRC_DIR = "csrc"
CSRC_DIR_ABSOLUTE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "csrc")

# ------------------------------------------- #

def get_extension():
	linkArgs = [
		"-O3"
	]
	compileArgs = [
        "-O3",
        "-Wno-missing-braces"
    ]

	srcs = list(
		glob.glob(os.path.join(CSRC_DIR, "src", "*.cpp")) 
	)
	srcs.append(
		os.path.join(CSRC_DIR, "ext.cpp")
	)

	includeDirs = [
		os.path.join(CSRC_DIR_ABSOLUTE, "include"),
		os.path.join(CSRC_DIR_ABSOLUTE, "external"),
        pybind11.get_include()
	]

	return Extension(
		f"{LIBRARY_NAME}._C",
		srcs,
		include_dirs=includeDirs,
		extra_compile_args=compileArgs,
		extra_link_args=linkArgs
	)

# ------------------------------------------- #

setup(
	packages=find_packages(),
	ext_modules=[ get_extension() ],
	cmdclass={},
)