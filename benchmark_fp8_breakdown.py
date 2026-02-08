import torch
import time
import torch.nn as nn

def run_benchmark():
    if not hasattr(torch, 'xpu') or not torch.xpu.is_available():
        print("XPU not available. Please run on an XPU-enabled system.")
        return

    device = torch.device('xpu')
    dtype = torch.float16

    # Check FP8 support
    if hasattr(torch, 'float8_e4m3fn'):
        fp8_dtype = torch.float8_e4m3fn
        print(f"FP8 Dtype: {fp8_dtype}")
    else:
        print("FP8 not supported on this torch version.")
        return

    # Matrix sizes typical for SDXL Linear layers (e.g., Attention QKV)
    batch_size = 1
    # SDXL uses various sizes, 2048/1280 are common
    m = 1
    n = 2048 # Output features
    k = 2048 # Input features

    print(f"\n--- Matrix Multiplication Benchmark (M={m}, N={n}, K={k}) ---")

    # 1. FP16 Baseline
    x_fp16 = torch.randn(m, k, device=device, dtype=dtype)
    w_fp16 = torch.randn(n, k, device=device, dtype=dtype) # Transposed for linear

    start = time.perf_counter()
    for _ in range(100):
        out = torch.nn.functional.linear(x_fp16, w_fp16)
    torch.xpu.synchronize()
    end = time.perf_counter()
    fp16_time = (end - start) * 10
    print(f"FP16 Linear (Pure): {fp16_time:.4f} ms")

    # 2. FP8 Cast Overhead (Input)
    start = time.perf_counter()
    for _ in range(100):
        x_fp8 = x_fp16.to(fp8_dtype)
    torch.xpu.synchronize()
    end = time.perf_counter()
    input_cast_time = (end - start) * 10
    print(f"Input Cast (FP16->FP8): {input_cast_time:.4f} ms")

    # 3. FP8 Cast Overhead (Weight)
    start = time.perf_counter()
    for _ in range(100):
        w_fp8 = w_fp16.to(fp8_dtype)
    torch.xpu.synchronize()
    end = time.perf_counter()
    weight_cast_time = (end - start) * 10
    print(f"Weight Cast (FP16->FP8): {weight_cast_time:.4f} ms")

    # Prepare FP8 tensors for compute
    x_fp8 = x_fp16.to(fp8_dtype)
    w_fp8 = w_fp16.to(fp8_dtype)

    # 4. FP8 Linear (Pure Compute)
    start = time.perf_counter()
    for _ in range(100):
        out = torch.nn.functional.linear(x_fp8, w_fp8)
    torch.xpu.synchronize()
    end = time.perf_counter()
    fp8_compute_time = (end - start) * 10
    print(f"FP8 Linear (Pure Compute): {fp8_compute_time:.4f} ms")

    # 5. Output Cast Overhead (FP8->FP16)
    out_fp8 = torch.nn.functional.linear(x_fp8, w_fp8)
    start = time.perf_counter()
    for _ in range(100):
        out = out_fp8.to(dtype)
    torch.xpu.synchronize()
    end = time.perf_counter()
    output_cast_time = (end - start) * 10
    print(f"Output Cast (FP8->FP16): {output_cast_time:.4f} ms")

    # 6. Full Forge Operations Loop (Simulated)
    print("\n--- Full Loop Simulation ---")

    # A: Original (Cast Weights Every Time)
    start = time.perf_counter()
    for _ in range(100):
        # Cast input
        x_in = x_fp16.to(fp8_dtype)
        # Cast weight (simulating bad behavior)
        w_in = w_fp16.to(fp8_dtype)
        # Compute
        out = torch.nn.functional.linear(x_in, w_in)
        # Cast output
        res = out.to(dtype)
    torch.xpu.synchronize()
    end = time.perf_counter()
    original_loop_time = (end - start) * 10
    print(f"Original (Cast All): {original_loop_time:.4f} ms")

    # B: Cached Weight (Cast Input Only)
    # Pre-cast weight
    w_cached = w_fp16.to(fp8_dtype)

    start = time.perf_counter()
    for _ in range(100):
        # Cast input
        x_in = x_fp16.to(fp8_dtype)
        # Use cached weight
        w_in = w_cached
        # Compute
        out = torch.nn.functional.linear(x_in, w_in)
        # Cast output
        res = out.to(dtype)
    torch.xpu.synchronize()
    end = time.perf_counter()
    cached_loop_time = (end - start) * 10
    print(f"Optimized (Cached Weight): {cached_loop_time:.4f} ms")

    diff = original_loop_time - cached_loop_time
    print(f"\nPotential Savings: {diff:.4f} ms per Op")

    # 7. Check Mixed Precision Support (Maybe XPU supports FP16 Input + FP8 Weight?)
    print("\n--- Mixed Precision Check (FP16 Input + FP8 Weight) ---")
    try:
        start = time.perf_counter()
        for _ in range(100):
            # This will fail if not supported natively
            out = torch.nn.functional.linear(x_fp16, w_fp8)
        torch.xpu.synchronize()
        end = time.perf_counter()
        mixed_time = (end - start) * 10
        print(f"Mixed Precision (FP16+FP8): {mixed_time:.4f} ms")
        print("Note: If this works and is fast, we can skip input casting!")
    except Exception as e:
        print(f"Mixed Precision Failed: {e}")

if __name__ == "__main__":
    run_benchmark()
