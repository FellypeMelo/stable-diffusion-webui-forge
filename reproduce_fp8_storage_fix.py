import torch
import time
import torch.nn as nn

def benchmark_fp8_storage_fix():
    if not hasattr(torch, 'xpu') or not torch.xpu.is_available():
        print("XPU not available. Please run on an XPU-enabled system.")
        return

    device = torch.device('xpu')
    dtype = torch.float16

    # Check FP8 support
    if hasattr(torch, 'float8_e4m3fn'):
        fp8_dtype = torch.float8_e4m3fn
    else:
        print("FP8 not supported on this torch version.")
        return

    m, n, k = 1, 2048, 2048
    print(f"\n--- Matrix Multiplication Benchmark (M={m}, N={n}, K={k}) ---")

    x_fp16 = torch.randn(m, k, device=device, dtype=dtype)
    w_fp8 = torch.randn(n, k, device=device, dtype=dtype).to(fp8_dtype)

    # Simulate the Optimized Path (FP8 Storage -> FP16 Compute)
    print("\n--- Optimized Path (FP8 Storage -> FP16 Compute) ---")
    start = time.perf_counter()
    for _ in range(100):
        # 1. Dequantize Weight (Fast ~0.02ms)
        w_compute = w_fp8.to(dtype)

        # 2. Compute in FP16 (Fast ~0.35ms)
        out = torch.nn.functional.linear(x_fp16, w_compute)

        # No output cast needed (already FP16)

    torch.xpu.synchronize()
    end = time.perf_counter()
    opt_time = (end - start) * 10
    print(f"Time: {opt_time:.4f} ms")

    print("\nCompare with previous Pure FP8 Result (~1.05 ms).")
    print(f"Speedup: ~{1.05 / opt_time:.2f}x faster than naive FP8")

if __name__ == "__main__":
    benchmark_fp8_storage_fix()
