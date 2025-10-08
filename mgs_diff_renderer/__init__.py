import torch
from . import _C

# ------------------------------------------- #

class _RenderFunction(torch.autograd.Function):
	
	@staticmethod
	def forward(ctx, 
				outWidth, outHeight, view, proj, focalX, focalY,
				means, scales, rotations, opacities, colors, harmomics):
		
		return torch.ops.mgs_diff_renderer.forward(
			outWidth, outHeight, view, proj, focalX, focalY,
			means, scales, rotations, opacities, colors, harmomics
		)

	@staticmethod
	def backward(ctx, grad_output):
		return torch.zeros_like(grad_output)

def render(outWidth: int, outHeight: int, view: torch.Tensor, proj: torch.Tensor, focalX: float, focalY: float,
		   means: torch.Tensor, scales: torch.Tensor, rotations: torch.Tensor, opacities: torch.Tensor, colors: torch.Tensor, harmomics: torch.Tensor) -> torch.Tensor:

	return _RenderFunction.apply(
		outWidth, outHeight, view, proj, focalX, focalY,
		means, scales, rotations, opacities, colors, harmomics
	)