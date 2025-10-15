import torch
from PIL import Image
import mgs_diff_renderer
import math

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

def main():
	torch.set_default_device('cuda')

	width, height = 1920, 1080
	aspect = width / height

	# --- Camera setup ---
	eye = torch.tensor([0.0, 0.0, 3.0])     # camera 3 units away from origin
	target = torch.tensor([0.0, 0.0, 0.0])
	up = torch.tensor([0.0, 1.0, 0.0])
	view = look_at(eye, target, up)

	fovy = math.radians(60)
	proj = perspective(fovy, aspect, 0.1, 100.0)
	focalX = width / (2 * math.tan(fovy / 2))
	focalY = focalX  # same for isotropic pixels

	# --- Single Gaussian setup ---
	numGaussians = 1
	means = torch.tensor([[0.25, 0.0, 0.0], [-0.25, 0.1, 0.1]], dtype=torch.float32, device='cuda')
	scales = torch.tensor([[0.2, 0.02, 0.02], [0.3, 0.1, 0.1]], dtype=torch.float32, device='cuda')  # isotropic radius
	rotations = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device='cuda')  # identity quaternion
	opacities = torch.tensor([[1.0], [1.0]], dtype=torch.float32, device='cuda')
	colors = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32, device='cuda')  # white
	harmonics = torch.zeros((numGaussians, 15, 3), dtype=torch.float32, device='cuda')  # ignore SH

	# --- Render ---
	y = mgs_diff_renderer.render(
		width, height, view, proj, focalX, focalY,
		means, scales, rotations, opacities, colors, harmonics
	)

	print(f"Output: {y.shape}, {y.dtype}, min={y.min().item():.4f}, max={y.max().item():.4f}")

	# --- Save as PNG ---
	img = y.detach().cpu().clamp(0, 1)
	img = (img * 255).byte().numpy()
	img = Image.fromarray(img, mode="RGBA")
	img.save("output.png")
	print("✅ Saved output.png")

if __name__ == "__main__":
	main()
