import torch
from . import _C

# ------------------------------------------- #

# just a wrapper for torch.classes.mgs_diff_renderer.Settings
class Settings:	
	def __init__(self, width: int, height: int, 
	             view: torch.Tensor, proj: torch.Tensor, focalX: float, focalY: float,
				 debug: bool = False):

		self.cSettings = torch.classes.mgs_diff_renderer.Settings(
			width,
			height,
			view,
			proj,
			focalX,
			focalY,
			debug
		)

class RenderFunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx, settings: Settings,
	            means: torch.Tensor, scales: torch.Tensor, rotations: torch.Tensor, opacities: torch.Tensor, colors: torch.Tensor, harmonics: torch.Tensor) -> torch.Tensor:
		
		img, numRendered, geomBufs, binningBufs, imageBufs = torch.ops.mgs_diff_renderer.forward(
			settings.cSettings, means, scales, rotations, opacities, colors, harmonics
		)

		ctx.numRendered = numRendered
		ctx.settings = settings
		ctx.save_for_backward(
			means, scales, rotations, opacities, colors, harmonics,
			geomBufs, binningBufs, imageBufs
		)

		return img

	@staticmethod
	def backward(ctx, grad_output):
		means, scales, rotations, opacities, colors, harmonics, geomBufs, binningBufs, imageBufs = ctx.saved_tensors

		dMean, dScales, dRotations, dOpacities, dColors, dHarmonics = torch.ops.mgs_diff_renderer.backward(
			ctx.settings.cSettings, grad_output,
			means, scales, rotations, opacities, colors, harmonics,
			ctx.numRendered, geomBufs, binningBufs, imageBufs
		)

		return (
			None,       # settings
			dMean,      # means
			dScales,    # scales
			dRotations, # rotations
			dOpacities, # opacities
			dColors,    # colors
			dHarmonics  # harmonics
		)

# ------------------------------------------- #

def render(settings: Settings,
		   means: torch.Tensor, scales: torch.Tensor, rotations: torch.Tensor, opacities: torch.Tensor, colors: torch.Tensor, harmomics: torch.Tensor) -> torch.Tensor:

	return RenderFunction.apply(
		settings,
		means, scales, rotations, opacities, colors, harmomics
	)