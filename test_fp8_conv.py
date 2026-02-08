import torch
import time
import os

def test_conv2d_fp8():
    if not hasattr(torch, 'xpu') or not torch.xpu.is_available():
        print("XPU not available, skipping test")
        return

    device = torch.device("xpu")
    dtype = torch.float16

    # Check for FP8 support
    if hasattr(torch, 'float8_e4m3fn'):
        fp8_dtype = torch.float8_e4m3fn
    elif hasattr(torch, 'float8_e5m2'):
        fp8_dtype = torch.float8_e5m2
    else:
        print("FP8 dtype not found in torch")
        return

    print(f"Testing Conv2d on {device} with {dtype} input and {fp8_dtype} weights")

    # Setup
    batch_size = 1
    in_channels = 128
    out_channels = 128
    height = 64
    width = 64
    kernel_size = 3
    padding = 1

    # Input (FP16)
    x = torch.randn(batch_size, in_channels, height, width, device=device, dtype=dtype)

    # Weights (FP8)
    # Typically weights are stored in higher precision and cast on the fly,
    # but for this test let's simulate the scenario where we want to compute in FP8
    # In Forge, weights might be stored in FP8 or cast just before op.
    # The optimization target is to cast input to FP8 and run in FP8.

    # Create standard Conv2d
    conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, device=device, dtype=dtype)

    # Cast weights to FP8 manually to simulate what we want to achieve or test
    # Note: nn.Conv2d doesn't support FP8 parameters natively usually, but we can manually cast tensors
    weight_fp8 = conv.weight.data.to(fp8_dtype)
    bias = conv.bias.data if conv.bias is not None else None

    # Baseline: Run with mixed precision (FP16 input, FP8 weight)
    # This might fail or be slow if backend doesn't support it directly without cast
    print("\n--- Baseline (Mixed Precision: FP16 Input, FP8 Weight) ---")
    try:
        # We need to use functional API because nn.Conv2d expects parameters to match module dtype usually
        # But here we want to test the op itself
        start = time.perf_counter()
        for _ in range(10):
            # If backend supports mixed precision, this runs. If not, it might crash or be slow.
            out = torch.nn.functional.conv2d(x, weight_fp8, bias, padding=padding)
        torch.xpu.synchronize()
        end = time.perf_counter()
        print(f"Time: {(end - start)*100:.2f} ms")
    except Exception as e:
        print(f"Failed: {e}")

    # Optimized: Run with pure FP8 (FP8 Input, FP8 Weight)
    print("\n--- Optimized (Pure FP8: FP8 Input, FP8 Weight) ---")
    try:
        start = time.perf_counter()
        for _ in range(10):
            x_fp8 = x.to(fp8_dtype)
            out_fp8 = torch.nn.functional.conv2d(x_fp8, weight_fp8, bias, padding=padding)
            out = out_fp8.to(dtype)
        torch.xpu.synchronize()
        end = time.perf_counter()
        print(f"Time: {(end - start)*100:.2f} ms")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_conv2d_fp8()
