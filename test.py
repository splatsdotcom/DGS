# python/test_extension.py
import torch
import mgs_diff_renderer
from PIL import Image

def main():
    torch.set_default_device('cuda')

    width = 1920
    height = 1080
    numGaussians = 1

    view = torch.eye(4)
    proj = torch.eye(4)
    focalX = width  / 2
    focalY = height / 2
    means     = torch.zeros((numGaussians, 3))
    scales    = torch.zeros((numGaussians, 3))
    rotations = torch.zeros((numGaussians, 4))
    opacities = torch.zeros((numGaussians, 1))
    colors    = torch.zeros((numGaussians, 3))
    harmonics = torch.zeros((numGaussians, 15, 3))

    y = mgs_diff_renderer.render(
        1920, 1080, view, proj, focalX, focalY,
        means, scales, rotations, opacities, colors, harmonics
    )


    print(y.shape, y.dtype, y.min().item(), y.max().item())

    # Save as PNG
    img = y.detach().cpu().clamp(0, 1)  # keep in [0,1]
    img = (img * 255).byte().numpy()    # to uint8
    img = Image.fromarray(img, mode="RGBA")
    img.save("output.png")
    print("Saved output.png")

    # s = y.sum()
    # s.backward()
    # print("x.grad (should be all zeros):")
    # print(x.grad)

if __name__ == "__main__":
    main()
