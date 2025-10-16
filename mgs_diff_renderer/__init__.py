import torch
from . import _C

# ------------------------------------------- #

class _RenderFunction(torch.autograd.Function):
	
	@staticmethod
	def forward(ctx, 
	            outWidth, outHeight, view, proj, focalX, focalY,
	            means, scales, rotations, opacities, colors, harmonics,
	            debug=False):
		
		img, numRendered, geomBufs, binningBufs, imageBufs = torch.ops.mgs_diff_renderer.forward(
			outWidth, outHeight, view, proj, focalX, focalY,
			means, scales, rotations, opacities, colors, harmonics,
			debug
		)

		print(geomBufs)
		print(binningBufs)
		print(imageBufs)
		print()

		# TODO: wrap into a single params struct
		ctx.numRendered = numRendered
		ctx.width = outWidth
		ctx.height = outHeight
		ctx.focalX = focalX
		ctx.focalY = focalY
		ctx.debug = debug

		ctx.save_for_backward(
			view, proj, 
			means, scales, rotations, opacities, colors, harmonics,
			geomBufs, binningBufs, imageBufs
		)

		return img

	@staticmethod
	def backward(ctx, grad_output):
		view, proj, means, scales, rotations, opacities, colors, harmonics, geomBufs, binningBufs, imageBufs = ctx.saved_tensors

		dMean, dScales, dRotations, dOpacities, dColors, dHarmonics = torch.ops.mgs_diff_renderer.backward(
			ctx.width, ctx.height, view, proj, ctx.focalX, ctx.focalY,
			means, scales, rotations, opacities, colors, harmonics,
			geomBufs, binningBufs, imageBufs,
			ctx.debug
		)

		return (
			None,        # outWidth
			None,        # outHeight
			None,        # view
			None,        # proj
			None,        # focalX
			None,        # focalY
			dMean,       # means
			dScales,     # scales
			dRotations,  # rotations
			dOpacities,  # opacities
			dColors,     # colors
			dHarmonics,  # harmonics
			None         # debug
		)

def render(outWidth: int, outHeight: int, view: torch.Tensor, proj: torch.Tensor, focalX: float, focalY: float,
		   means: torch.Tensor, scales: torch.Tensor, rotations: torch.Tensor, opacities: torch.Tensor, colors: torch.Tensor, harmomics: torch.Tensor,
		   debug: bool = False) -> torch.Tensor:

	return _RenderFunction.apply(
		outWidth, outHeight, view, proj, focalX, focalY,
		means, scales, rotations, opacities, colors, harmomics,
		debug
	)