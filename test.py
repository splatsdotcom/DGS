# python/test_extension.py
import torch
import mgs_diff_renderer

def main():
    x = torch.randn(16, device='cuda', requires_grad=True)
    y = mgs_diff_renderer.render(x)  # y = x * 2 (from CUDA kernel)

    print(x / y)

    s = y.sum()
    s.backward()
    print("x.grad (should be all zeros):")
    print(x.grad)

if __name__ == "__main__":
    main()
