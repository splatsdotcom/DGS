import torch
import numpy as np
import math
from PIL import Image
import os
import mgs_diff_renderer
import time

# -------------------------------------------------------------
# Utility camera functions
# -------------------------------------------------------------

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

# -------------------------------------------------------------
# Renderer wrapper
# -------------------------------------------------------------

def render_scene(settings, means, scales, rotations, opacities, harmonics):
	rotations = rotations / torch.norm(rotations, dim=-1, keepdim=True).clamp(min=1e-8)
	return mgs_diff_renderer.render(
		settings,
		means, scales, rotations, opacities, harmonics
	)
	
# -------------------------------------------------------------
# Finite-difference gradient check for scales
# -------------------------------------------------------------

def finite_difference_gradcheck(gt_views, width, height, proj, focalX, focalY, features, featIdx, eps=1e-5):
	grad_fd = torch.zeros_like(features[featIdx])
	with torch.no_grad():
		for idx in np.ndindex(features[featIdx].shape):
			perturbed_plus = [x.detach().clone() for x in features]
			perturbed_plus[featIdx][idx] += eps

			perturbed_minus = [x.detach().clone() for x in features]
			perturbed_minus[featIdx][idx] -= eps

			total_loss_plus = 0.0
			total_loss_minus = 0.0
			for (view, gt_img) in gt_views:
				pred_plus = render_scene(width, height, view, proj, focalX, focalY, *perturbed_plus)
				pred_minus = render_scene(width, height, view, proj, focalX, focalY, *perturbed_minus)

				total_loss_plus += torch.nn.functional.mse_loss(pred_plus, gt_img)
				total_loss_minus += torch.nn.functional.mse_loss(pred_minus, gt_img)

			grad_fd[idx] = (total_loss_plus - total_loss_minus) / (2 * eps)
	return grad_fd

# -------------------------------------------------------------
# Scene + training setup
# -------------------------------------------------------------

def main():
	torch.set_default_device('cuda')
	os.makedirs("renders", exist_ok=True)

	torch.manual_seed(0)

	width, height = 1920, 1080
	aspect = width / height
	fovy = math.radians(60)
	proj = perspective(fovy, aspect, 0.1, 100.0)
	focalX = width / (2 * math.tan(fovy / 2))
	focalY = focalX

	# ---------------- Ground Truth Gaussians ----------------
	gt_means = torch.tensor([
		[0.3,  0.0, 0.0],
		[-0.3, 0.1, 0.1],
		[0.0, -0.2, 0.0]
	], dtype=torch.float32, device='cuda')

	gt_scales = torch.tensor([
		[0.2, 0.05, 0.05],
		[0.08, 0.3, 0.08],
		[0.1, 0.1, 0.25]
	], dtype=torch.float32, device='cuda')

	gt_rotations = torch.tensor([
		[0.0, 0.0, 0.0, 1.0],
		[0.0, 0.0, 0.0, 1.0],
		[0.0, 0.0, 0.0, 1.0]
	], dtype=torch.float32, device='cuda')

	gt_opacities = torch.tensor([[1.0], [1.0], [1.0]], dtype=torch.float32, device='cuda')
	gt_harmonics = torch.tensor([
		[ [1.0, 0.0, 0.0] ],
		[ [0.0, 1.0, 0.0] ],
		[ [0.0, 0.0, 1.0] ]
	], dtype=torch.float32, device='cuda')

	# ---------------- Generate Ground Truth Views ----------------
	n_views = 10
	gt_views = []
	for i in range(n_views):
		angle = 2 * math.pi * i / n_views
		eye = torch.tensor([2.0 * math.sin(angle), 0.5, 2.0 * math.cos(angle)], device='cuda')
		target = torch.tensor([0.0, 0.0, 0.0], device='cuda')
		up = torch.tensor([0.0, 1.0, 0.0], device='cuda')
		view = look_at(eye, target, up)

		settings = mgs_diff_renderer.Settings(
			width=width, height=height,
			view=view, proj=proj,
			focalX=focalX, focalY=focalY,
			debug=False
		)

		with torch.no_grad():
			img = render_scene(settings,
							   gt_means, gt_scales, gt_rotations,
							   gt_opacities, gt_harmonics)
		gt_views.append((settings, img))
		img_np = (img.detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
		Image.fromarray(img_np, mode='RGB').save(f"renders/gt_view_{i}.png")

	print("✅ Saved ground truth renders.")

	# ---------------- Perturbed Scene (learnable) ----------------
	means = (gt_means + 0.25 * torch.randn_like(gt_means)).clone().detach().requires_grad_(True)
	scales = (gt_scales + 0.1 * torch.randn_like(gt_scales)).clone().detach().requires_grad_(True)
	rotations = (gt_rotations + 0.5 * torch.randn_like(gt_rotations)).clone().detach().requires_grad_(True)
	opacities = torch.rand_like(gt_opacities).requires_grad_(True)
	harmonics = (torch.rand_like(gt_harmonics) * 0.5 + gt_harmonics * 0.5).detach().clone().requires_grad_(True)

	optimizer = torch.optim.Adam([means, scales, rotations, opacities, harmonics], lr=1e-2)

	# ---------------- Training Loop ----------------
	start = time.time()
	
	for step in range(500):
		optimizer.zero_grad()
		total_loss = 0.0

		# Multi-view consistency loss
		for (settings, gt_img) in gt_views:
			pred_img = render_scene(settings,
									means, scales, rotations, opacities, harmonics)
			loss = torch.nn.functional.mse_loss(pred_img, gt_img)
			total_loss += loss

		total_loss.backward()
		optimizer.step()

		# if step % 10 == 0:
		# 	with torch.no_grad():
		# 		for eps in [1e-3, 1e-4, 1e-5]:
		# 			grad_analytical = scales.grad.detach().clone()
		# 			grad_fd = finite_difference_gradcheck(
		# 				gt_views, width, height, proj, focalX, focalY,
		# 				( means, scales, rotations, opacities, colors, harmonics ), 1, eps
		# 			)

		# 			diff = (grad_fd - grad_analytical).abs()
		# 			rel_err = diff / grad_analytical.abs()

		# 			# print("Analytical grad norm:", grad_analytical.norm().item())
		# 			# print("FD grad norm:", grad_fd.norm().item())
		# 			# print("Max abs diff:", diff.max().item())
		# 			print(f"Max rel err eps={eps}:", rel_err.max().item())

		# 		# Optionally: print full arrays for detailed inspection
		# 		# print("Grad analytical:\n", grad_analytical)
		# 		# print("Grad FD:\n", grad_fd)
		# 		# print("Rel error:\n", rel_err)

		if step % 10 == 0:
			print(f"[Step {step:03d}] Loss = {total_loss.item():.6f}")
			with torch.no_grad():
				settings, _ = gt_views[0]
				img = render_scene(settings,
								   means, scales, rotations, opacities, harmonics)
				img_np = (img.detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
				Image.fromarray(img_np, mode='RGB').save(f"renders/train_step_{step:03d}.png")

	print("✅ Training complete.")
	print(f"Took {time.time() - start}s")

if __name__ == "__main__":
	main()
