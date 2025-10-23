import torch
import numpy as np
import math
from PIL import Image
import mgs_diff_renderer

def look_at(eye, target, up):
	f = (target - eye)
	f = f / torch.norm(f)
	u = up / torch.norm(up)
	s = torch.cross(f, u)
	s = s / torch.norm(s)
	u = torch.cross(s, f)
	m = torch.eye(4, dtype=torch.float32)
	m[0, :3] = s
	m[1, :3] = u
	m[2, :3] = -f
	m[0, 3] = -torch.dot(s, eye)
	m[1, 3] = -torch.dot(u, eye)
	m[2, 3] = torch.dot(f, eye)
	return m

def perspective(fovy, aspect, znear, zfar):
	tan_half_fovy = math.tan(fovy / 2)
	m = torch.zeros((4, 4), dtype=torch.float32)
	m[0, 0] = 1 / (aspect * tan_half_fovy)
	m[1, 1] = 1 / tan_half_fovy
	m[2, 2] = -(zfar + znear) / (zfar - znear)
	m[2, 3] = -(2 * zfar * znear) / (zfar - znear)
	m[3, 2] = -1.0
	return m

def finite_difference_gradcheck(func, loss_func, inputs, eps=1e-3, atol=1e-3, rtol=1e-2):
	# Compute analytical grads
	for inp in inputs:
		inp.grad = None
	
	out = func(*inputs)
	loss = loss_func(out)
	loss.backward()

	print("Analytical gradients:")
	for i, inp in enumerate(inputs):
		print(f"  Input {i}: grad norm = {inp.grad.norm().item():.6f}")

	# Finite-difference check
	for i, inp in enumerate(inputs):
		grad_fd = torch.zeros_like(inp)
		for idx in np.ndindex(inp.shape):
			# clone all inputs to avoid in-place modification
			perturbed = [x.detach().clone() for x in inputs]
			perturbed[i][idx] += eps
			y1 = loss_func(func(*perturbed)).item()

			perturbed = [x.detach().clone() for x in inputs]
			perturbed[i][idx] -= eps
			y2 = loss_func(func(*perturbed)).item()

			grad_fd[idx] = (y1 - y2) / (2 * eps)

		diff = (grad_fd - inp.grad).abs()
		rel_err = diff / (inp.grad.abs() + 1e-8)
		print(f"Input {i}: max abs diff = {diff.max():.6f}, max rel err = {rel_err.max():.6f}")

		if torch.any(rel_err > rtol) and torch.any(diff > atol):
			print("❌ Grad check failed for input", i)
			print("FD GRADS: ", grad_fd)
			print("ANALYTICAL GRADS: ", inp.grad)
			print("DIFFS: ", diff)
			print("RELATIVE ERROR: ", rel_err)
		else:
			print("✅ Grad check passed for input", i)

def main():
	torch.set_default_device('cuda')

	width, height = 1920, 1080  # smaller for testing
	aspect = width / height

	eye = torch.tensor([0.0, 0.0, 3.0])
	target = torch.tensor([0.0, 0.0, 0.0])
	up = torch.tensor([0.0, 1.0, 0.0])
	view = look_at(eye, target, up)

	fovy = math.radians(60)
	proj = perspective(fovy, aspect, 0.1, 100.0)
	focalX = width / (2 * math.tan(fovy / 2))
	focalY = focalX

	means = torch.tensor([[0.25, 0.0, 0.0],
						  [-0.25, 0.1, 0.1]],
						 dtype=torch.float32, device='cuda', requires_grad=True)
	scales = torch.tensor([[0.2, 0.02, 0.02],
						   [0.3, 0.1, 0.1]], dtype=torch.float32, device='cuda', requires_grad=True)
	rotations = torch.tensor([[0.0, 0.0, 0.0, 1.0],
							  [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device='cuda', requires_grad=True)
	opacities = torch.tensor([[1.0], [1.0]], dtype=torch.float32, device='cuda', requires_grad=True)
	colors = torch.tensor([[0.0, 1.0, 0.0],
						   [0.0, 0.0, 1.0]], dtype=torch.float32, device='cuda', requires_grad=True)
	harmonics = torch.zeros((2, 15, 3), dtype=torch.float32, device='cuda', requires_grad=True)

	y = mgs_diff_renderer.render(width, height, view, proj, focalX, focalY,
								 means, scales, rotations, opacities, colors, harmonics)

	# Define a simple differentiable loss
	target = torch.zeros_like(y)
	target[..., 2] = 1.0  # pure blue target
	loss = torch.nn.functional.mse_loss(y, target)

	# Run backward pass
	loss.backward()

	img = y.detach().cpu().clamp(0, 1) 
	img = (img * 255).byte().numpy() 
	Image.fromarray(img, mode="RGB").save('output.png')

	print("✅ Backward pass completed")
	print("Gradient on means:\n", means.grad)
	print("Gradient on scales:\n", scales.grad)
	print("Gradient on opacities:\n", opacities.grad)
	print("Gradient on rotations:\n", rotations.grad)
	print("Gradient on colors:\n", colors.grad)

	def func(means, scales, rotations, opacities, colors, harmonics):

		rotations_norm = rotations / torch.norm(rotations, dim=-1, keepdim=True).clamp(min=1e-8)
		return mgs_diff_renderer.render(
			1920, 1080, view, proj, focalX, focalY,
			means, scales, rotations_norm, opacities, colors, harmonics
		)

	def loss_func(out):
		target = torch.zeros_like(out)
		target[..., 2] = 1.0  # pure blue target
		return torch.nn.functional.mse_loss(out, target)		

	# finite_difference_gradcheck(func, loss_func, [means, scales, rotations, opacities, colors, harmonics])

if __name__ == "__main__":
	main()
