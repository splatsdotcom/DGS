import torch
from PIL import Image
import math
import mgs_diff_renderer
import numpy as np
from plyfile import PlyData
import imageio.v2 as imageio
import os

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
	return m.T

def perspective(fovy, aspect, znear, zfar):
	tan_half_fovy = math.tan(fovy / 2)
	m = torch.zeros((4, 4), dtype=torch.float32)
	m[0, 0] = 1 / (aspect * tan_half_fovy)
	m[1, 1] = 1 / tan_half_fovy
	m[2, 2] = -(zfar + znear) / (zfar - znear)
	m[2, 3] = -(2 * zfar * znear) / (zfar - znear)
	m[3, 2] = -1.0
	return m.T

def load_ply_data(path, device='cuda'):
	print(f"📂 Loading PLY from {path} ...")
	plydata = PlyData.read(path)
	vertex = plydata['vertex'].data

	# Convert structured array → torch tensors
	def np_to_torch(name, dim=1):
		arr = np.stack([vertex[n] for n in name], axis=-1) if isinstance(name, (list, tuple)) else vertex[name]
		return torch.tensor(arr, dtype=torch.float32, device=device)

	means = np_to_torch(['x', 'y', 'z'])
	colors = 0.5 + np_to_torch(['f_dc_0', 'f_dc_1', 'f_dc_2']) * 0.28209479177387814
	opacities = torch.sigmoid(np_to_torch('opacity').unsqueeze(1))
	scales = torch.exp(np_to_torch(['scale_0', 'scale_1', 'scale_2']))
	rotations = np_to_torch(['rot_1', 'rot_2', 'rot_3', 'rot_0'])

	numGaussians = means.shape[0]
	print(f"✅ Loaded {numGaussians} Gaussians")

	# Placeholder harmonics
	harmonics = torch.zeros((numGaussians, 15, 3), dtype=torch.float32, device=device)

	return means, scales, rotations, opacities, colors, harmonics


def main():
	torch.set_default_device('cuda')

	width, height = 1920, 1080  # smaller for faster animation
	aspect = width / height

	# --- Camera setup ---
	start_eye = torch.tensor([1.0 + 0.1, 0.0, 5.0])
	end_eye   = torch.tensor([-1.0 + 0.1, 0.0, -5.0])
	target = torch.tensor([0.0, 0.0, 0.0])
	up = torch.tensor([0.0, 1.0, 0.0])

	fovy = math.radians(60)
	proj = perspective(fovy, aspect, 0.1, 1000.0)
	focalY = height / (2 * math.tan(fovy / 2))
	focalX = focalY

	# --- Load data ---
	ply_path = "test.ply"
	means, scales, rotations, opacities, colors, harmonics = load_ply_data(ply_path)

	# --- Animation settings ---
	num_frames = 60
	out_dir = "frames"
	os.makedirs(out_dir, exist_ok=True)
	print(f"🎬 Rendering {num_frames} frames ...")

	frames = []
	for i in range(num_frames):
		t = i / (num_frames - 1)
		eye = (1 - t) * start_eye + t * end_eye
		view = look_at(eye, target, up)

		y = mgs_diff_renderer.render(
			width, height, view, proj, focalX, focalY,
			means, scales, rotations, opacities, colors, harmonics
		)

		img = y.detach().cpu().clamp(0, 1)
		img = (img * 255).byte().numpy()
		frame_path = os.path.join(out_dir, f"frame_{i:04d}.png")
		Image.fromarray(img, mode="RGBA").save(frame_path)
		frames.append(img)
		print(f"Frame {i+1}/{num_frames} saved")

	# --- Write video ---
	video_path = "zoom_in.mp4"
	print(f"🎞️ Writing video to {video_path} ...")
	imageio.mimsave(video_path, frames, fps=15)
	print("✅ Done! Saved zoom_in.mp4")

if __name__ == "__main__":
	main()
