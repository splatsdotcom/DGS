import torch
import numpy as np
import math
from PIL import Image
import os
import mgs_diff_renderer as mgsdr
import time

# ------------------------------------------- #

def look_at(eye, target, up):
	f = (target - eye)
	f = f / torch.norm(f)
	u = up / torch.norm(up)
	s = torch.cross(f, u, dim=0)
	s = s / torch.norm(s)
	u = torch.cross(s, f, dim=0)
	m = torch.eye(4, dtype=torch.float32)

	m[0, :3] = s
	m[1, :3] = u
	m[2, :3] = -f
	m[0, 3] = -torch.dot(s, eye)
	m[1, 3] = -torch.dot(u, eye)
	m[2, 3] = torch.dot(f, eye)

	return m

def perspective(fovy, aspect, znear, zfar):
	tanHalfFov = math.tan(fovy / 2)
	m = torch.zeros((4, 4), dtype=torch.float32)
	m[0, 0] = 1 / (aspect * tanHalfFov)
	m[1, 1] = 1 / tanHalfFov
	m[2, 2] = -(zfar + znear) / (zfar - znear)
	m[2, 3] = -(2 * zfar * znear) / (zfar - znear)
	m[3, 2] = -1.0
	return m

# ------------------------------------------- #


def main():

	# torch setup:
	# ---------------
	torch.set_default_device('cuda')
	torch.manual_seed(0)

	# intrinsics:
	# ---------------
	width, height = 1920, 1080
	aspect = width / height
	fovy = math.radians(60)
	proj = perspective(fovy, aspect, 0.1, 100.0)
	focalX = width / (2 * math.tan(fovy / 2))
	focalY = focalX

	# GT data:
	# ---------------
	gtMeans = torch.tensor([
		[0.3,  0.0, 0.0],
		[-0.3, 0.1, 0.1],
		[0.0, -0.2, 0.0]
	], dtype=torch.float32, device='cuda')

	gtScales = torch.tensor([
		[0.2, 0.05, 0.05],
		[0.08, 0.3, 0.08],
		[0.1, 0.1, 0.25]
	], dtype=torch.float32, device='cuda')

	gtRotations = torch.tensor([
		[0.0, 0.0, 0.0, 1.0],
		[0.0, 0.0, 0.0, 1.0],
		[0.0, 0.0, 0.0, 1.0]
	], dtype=torch.float32, device='cuda')

	gtOpacities = torch.tensor([
		[1.0], 
		[1.0], 
		[1.0]
	], dtype=torch.float32, device='cuda')

	gtHarmonics = torch.tensor([
		[ [1.0, 0.0, 0.0] ],
		[ [0.0, 1.0, 0.0] ], # currently harmonics is just treated as RGB color
		[ [0.0, 0.0, 1.0] ]
	], dtype=torch.float32, device='cuda')

	# generate GT views + renderers:
	# ---------------
	numViews = 10
	views = []

	os.makedirs("renders", exist_ok=True)

	for i in range(numViews):
		angle = 2 * math.pi * i / numViews
		eye = torch.tensor([2.0 * math.sin(angle), 0.5, 2.0 * math.cos(angle)], device='cuda')
		target = torch.tensor([0.0, 0.0, 0.0], device='cuda')
		up = torch.tensor([0.0, 1.0, 0.0], device='cuda')
		view = look_at(eye, target, up)

		settings = mgsdr.Settings(
			width=width, height=height,
			view=view, proj=proj,
			focalX=focalX, focalY=focalY,
			debug=False
		)
		renderer = mgsdr.Renderer(settings)

		with torch.no_grad():
			img = renderer(gtMeans, gtScales, gtRotations, gtOpacities, gtHarmonics)

		views.append((renderer, img))

		imgNp = (img.detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
		Image.fromarray(imgNp).save(f"renders/gt_view_{i}.png")

	print("Saved ground truth renders...")

	# generate perturbed scene:
	# ---------------
	means = (gtMeans + 0.25 * torch.randn_like(gtMeans)).clone().detach().requires_grad_(True)
	scales = (gtScales + 0.1 * torch.randn_like(gtScales)).clone().detach().requires_grad_(True)
	rotations = (gtRotations + 0.5 * torch.randn_like(gtRotations)).clone().detach().requires_grad_(True)
	opacities = torch.rand_like(gtOpacities).requires_grad_(True)
	harmonics = (torch.rand_like(gtHarmonics) * 0.5 + gtHarmonics * 0.5).detach().clone().requires_grad_(True)

	optimizer = torch.optim.Adam([means, scales, rotations, opacities, harmonics], lr=1e-2)

	# train:
	# ---------------
	start = time.time()
	
	for step in range(500):
		optimizer.zero_grad()
		totalLoss = 0.0

		for (renderer, gtImg) in views:
			predImg = renderer(means, scales, rotations, opacities, harmonics)
			loss = torch.nn.functional.mse_loss(predImg, gtImg)
			totalLoss += loss

		totalLoss.backward()
		optimizer.step()

		if step % 10 == 0:
			print(f"[Step {step:03d}] Loss = {totalLoss.item():.6f}")

			# save current render (takes too long):
			# with torch.no_grad():
			# 	renderer, _ = views[0]
			# 	img = renderer(means, scales, rotations, opacities, harmonics)
			# 	imgNp = (img.detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
			# 	Image.fromarray(imgNp).save(f"renders/train_step_{step:03d}.png")

	print("Training complete...")
	print(f"Took {time.time() - start}s")

if __name__ == "__main__":
	main()
