import torch
from . import _C

# ------------------------------------------- #

class _RenderFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input):
        return torch.ops.mgs_diff_renderer.forward(input)

    @staticmethod
    def backward(ctx, grad_output):
        return torch.zeros_like(grad_output)

def render(x: torch.Tensor) -> torch.Tensor:
    return _RenderFunction.apply(x)