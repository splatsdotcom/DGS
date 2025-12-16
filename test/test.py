import dgs
import numpy as np

means = np.array([
	[0.3,  0.0, 0.0],
	[-0.3, 0.1, 0.1],
	[0.0, -0.2, 0.0]
], dtype=np.float32)

scales = np.array([
	[0.2, 0.05, 0.05],
	[0.08, 0.3, 0.08],
	[0.1, 0.1, 0.25]
], dtype=np.float32)

rotations = np.array([
	[0.0, 0.0, 0.0, 1.0],
	[0.0, 0.0, 0.0, 1.0],
	[0.0, 0.0, 0.0, 1.0]
], dtype=np.float32)

opacities = np.array([
	[1.0], 
	[1.0], 
	[1.0]
], dtype=np.float32)

harmonics = (np.array([
	[ [1.0, 0.0, 0.0] ],
	[ [0.0, 1.0, 0.0] ],
	[ [0.0, 0.0, 1.0] ]
], dtype=np.float32) - 0.5) / 0.28209479177387814

gaussians = dgs.Gaussians(
	means, scales, rotations, opacities, harmonics
)

metadata = dgs.Metadata()

dgs.encode(gaussians, metadata, "/Users/daniel/Dev/Projects/DGS-JS/test/splats/test.dgs")

gaussiansDecoded, metadataDecoded = dgs.decode("/Users/daniel/Dev/Projects/DGS-JS/test/splats/test.dgs")

dgs.encode(gaussiansDecoded, metadataDecoded, "/Users/daniel/Dev/Projects/DGS-JS/test/splats/test_reencoded.dgs")