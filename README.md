# DGS
This is the home of the **Dynamic Gaussian Splat** (`.dgs`) file format. This repository contains:
- The reference encoder + decoder for `.dgs` files, implemented in C
- Utility functions for manipulating `.dgs` files, implemented in C
- Python bindings for the aforementioned

This library serves as both the specification and reference implementation for `.dgs`, and is the core of all projects within [Splatkit](https://github.com/splatsdotcom/splatkit).

## Installation + Usage
To use `DGS` in your own project, you can install the package on `pip`:
```bash
pip install dgs-py
```
Then, it can be importated with a simple:
```python
import dgs
```
Here is a full example generating a single `.dgs` file:
```python
import dgs
import numpy as np

# example data:
means = np.array([
	[0.3,  0.0, 0.0], [-0.3, 0.1, 0.1], [0.0, -0.2, 0.0]
], dtype=np.float32)

scales = np.array([
	[0.2, 0.05, 0.05], [0.08, 0.3, 0.08], [0.1, 0.1, 0.25]
], dtype=np.float32)

rotations = np.array([
	[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]
], dtype=np.float32)

opacities = np.array([
	[1.0], [1.0], [1.0]
], dtype=np.float32)

harmonics = (np.array([
	[ [1.0, 0.0, 0.0] ], [ [0.0, 1.0, 0.0] ], [ [0.0, 0.0, 1.0] ]
], dtype=np.float32) - 0.5) / 0.28209479177387814

# encode:
gaussians = dgs.Gaussians(
	means, scales, rotations, opacities, harmonics
)
metadata = dgs.Metadata()

dgs.encode(gaussians, metadata, "example.dgs")
```

## Documentation
Coming soon!

## Building
If you wish to contribute to this project, you will need to build it from source yourself. To build, you will need the tools:
- `setuptools` (`pip install setuptools`)
- `pybind11` (`pip install pybind11`)
Then, to build, you will first need to clone the repository and initialize the submodules:
```bash
git clone git@github.com:splatsdotcom/DGS.git
cd DGS
git submodule update --init --recursive
```
Then, the project can be built simply with:
```bash
pip install -v .
```
And the `dgs` package will become globally available on your system. There is currently no way to build the C library without the python bindings, but this will come in the future.