import torch
import time
import torch.nn as nn

def benchmark_fp8_overhead():
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device('xpu')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"Running on {device}")

    # Setup
    batch_size = 1
    in_features = 4096
    out_features = 4096
    dtype = torch.float16

    # Check for FP8 support
    if hasattr(torch, 'float8_e4m3fn'):
        fp8_dtype = torch.float8_e4m3fn
    elif hasattr(torch, 'float8_e5m2'):
        fp8_dtype = torch.float8_e5m2
    else:
        print("FP8 not supported, simulating with float32 cast")
        fp8_dtype = torch.float32 # Fallback simulation

    # Linear Layer (FP16 weights)
    linear = nn.Linear(in_features, out_features, bias=False, device=device, dtype=dtype)
    x = torch.randn(batch_size, in_features, device=device, dtype=dtype)

    iterations = 100

    # 1. Original: Cast every time
    print("\n--- Original (Cast Every Iteration) ---")
    start_time = time.perf_counter()
    for _ in range(iterations):
        # Simulate Forge behavior:
        # 1. Get weight (already on device in this simplified test)
        weight = linear.weight

        # 2. Cast to FP8
        w_fp8 = weight.to(fp8_dtype)
        x_fp8 = x.to(fp8_dtype)

        # 3. Compute
        out = torch.nn.functional.linear(x_fp8, w_fp8)

        # 4. Cast back (simulated)
        out = out.to(dtype)

    if device.type != 'cpu':
        torch.cuda.synchronize() if device.type == 'cuda' else torch.xpu.synchronize()
    end_time = time.perf_counter()
    original_fps = iterations / (end_time - start_time)
    print(f"FPS: {original_fps:.2f}")

    # 2. Optimized: Cached Cast
    print("\n--- Optimized (Cached Cast) ---")

    # Simulate caching on the module
    cache = {}

    start_time = time.perf_counter()
    for _ in range(iterations):
        # Simulate Forge behavior:
        weight = linear.weight

        # Check cache
        w_id = weight.data_ptr() # Use data pointer as key
        # Also check version for safety in real implementation

        if cache.get('id') == w_id:
            w_fp8 = cache['tensor']
        else:
            w_fp8 = weight.to(fp8_dtype)
            cache = {'id': w_id, 'tensor': w_fp8}

        x_fp8 = x.to(fp8_dtype)

        # Compute
        out = torch.nn.functional.linear(x_fp8, w_fp8)

        # Cast back
        out = out.to(dtype)

    if device.type != 'cpu':
        torch.cuda.synchronize() if device.type == 'cuda' else torch.xpu.synchronize()
    end_time = time.perf_counter()
    optimized_fps = iterations / (end_time - start_time)
    print(f"FPS: {optimized_fps:.2f}")

    print(f"\nSpeedup: {optimized_fps / original_fps:.2f}x")

if __name__ == "__main__":
    try:
        benchmark_fp8_overhead()
    except ImportError:
        print("Torch not found")
