import sys
import time
import dpctl
import numpy as np
import dpctl.memory as dpmem
import argparse
from attention_kernel import AttentionKernel
from stress_VRAM import VRAMStressTest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

OUTPUT_FILE = "final_performance_report.txt"

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.terminal.flush()
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

def get_queue():
    try:
        device = dpctl.SyclDevice("level_zero:gpu")
    except:
        device = dpctl.SyclDevice("gpu")
    return dpctl.SyclQueue(device)

class MetricCalculator:
    @staticmethod
    def compute_bandwidth(b, h, s, d, ms):
        if ms <= 0: return 0.0
        # Total Bytes Read/Written (Approximate Flash Attn IO)
        # Naive: Read Q, K, V (3 * 2B), Write O (1 * 2B) = 4 * 2B results
        # But this is N^2 complexity kernel, memory moved is lower bound.
        # Let's use standard formula for "effective bandwidth" if it were memory bound
        # Elements = B*H*S*D
        # Bytes = 4 tensors * Elements * 2 bytes
        total_bytes = 4 * (b * h * s * d) * 2
        gb = total_bytes / 1e9
        sec = ms / 1000.0
        return gb / sec

    @staticmethod
    def compute_tflops(b, h, s, d, ms):
        if ms <= 0: return 0.0
        # 4 * B * H * S^2 * D
        flops = 4.0 * b * h * s * s * d
        return (flops / 1e12) / (ms / 1000.0)

def run_kernel_benchmark(queue, kernel, b, h, s, d, iter=20):
    try:
        num = b * h * s * d
        size = num * 2
        q_mem = dpmem.MemoryUSMDevice(size, queue=queue)
        k_mem = dpmem.MemoryUSMDevice(size, queue=queue)
        v_mem = dpmem.MemoryUSMDevice(size, queue=queue)
        o_mem = dpmem.MemoryUSMDevice(size, queue=queue)
        
        # Warmup
        for _ in range(3):
            kernel.benchmark(int(q_mem._pointer), int(k_mem._pointer), int(v_mem._pointer), int(o_mem._pointer), b, h, s, d, 1)
        
        # Run
        return kernel.benchmark(int(q_mem._pointer), int(k_mem._pointer), int(v_mem._pointer), int(o_mem._pointer), b, h, s, d, iter)
    except Exception as e:
        # print(f"Bench Error: {e}")
        return 0.0

def run_torch_benchmark(b, h, s, d, iter=20):
    if not HAS_TORCH: return 0.0
    try:
        dt = torch.bfloat16
        dev = torch.device('xpu')
        q = torch.randn((b, h, s, d), dtype=dt, device=dev)
        k = torch.randn((b, h, s, d), dtype=dt, device=dev)
        v = torch.randn((b, h, s, d), dtype=dt, device=dev)
        
        # Warmup
        for _ in range(3):
            torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.xpu.synchronize()
        
        st = time.perf_counter()
        for _ in range(iter):
            torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.xpu.synchronize()
        return ((time.perf_counter() - st) * 1000) / iter
    except:
        return 0.0

def main():
    sys.stdout = Logger(OUTPUT_FILE)
    
    q = get_queue()
    d = q.get_sycl_device()
    kernel = AttentionKernel()
    
    print("="*100)
    print(f"INTEL ARC ATTENTION KERNEL - COMPREHENSIVE PERFORMANCE REPORT")
    print("="*100)
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {d.name}")
    print(f"Driver: {d.driver_version}")
    print(f"Compute Units: {d.max_compute_units}")
    print(f"Global Memory: {d.global_mem_size / (1024**3):.2f} GB")
    print("="*100)
    print("")

    # 1. SD Simulation
    print(f"1. STABLE DIFFUSION SIMULATION (Batch=1, Heads=Vary, Dim=64)")
    print("-" * 100)
    print(f"{'Layer Resolution':<20} | {'Shape (B,H,S,D)':<20} | {'Custom (ms)':<15} | {'Torch (ms)':<15} | {'Speedup':<10} | {'TFLOPS':<10}")
    print("-" * 100)
    
    sd_layers = [
        # ("Stage 1 (64x64)", 1, 8, 4096, 64), # Unsupported
        ("Stage 2 (32x32)", 1, 16, 1024, 64),
        ("Stage 3 (16x16)", 1, 16, 256, 64),
        ("Stage 4 (8x8)",   1, 32, 64, 64)
    ]
    
    total_ops = 0
    total_custom_time = 0
    total_torch_time = 0

    for name, b, h, s, dim in sd_layers:
        tc = run_kernel_benchmark(q, kernel, b, h, s, dim, iter=50)
        tt = run_torch_benchmark(b, h, s, dim, iter=50)
        
        speedup = tt/tc if tc > 0 else 0
        tflops = MetricCalculator.compute_tflops(b, h, s, dim, tc)
        
        print(f"{name:<20} | {f'({b},{h},{s},{dim})':<20} | {tc:<15.4f} | {tt:<15.4f} | {f'{speedup:.2f}x':<10} | {tflops:<10.2f}")
        
        if tc > 0 and tt > 0:
            total_custom_time += tc
            total_torch_time += tt

    print("-" * 100)
    if total_custom_time > 0:
        avg_speedup = total_torch_time / total_custom_time
        print(f"AGGREGATE SPEEDUP (SD LAYERS): {avg_speedup:.2f}x")
    print("")

    # 2. Robust Scaling Sweep
    print(f"2. SCALING SWEEP (Batch Scaling, Fixed S=2048, H=8)")
    print("-" * 100)
    print(f"{'Configuration':<25} | {'Custom (ms)':<15} | {'Torch (ms)':<15} | {'Speedup':<10} | {'GB/s':<10} | {'Status':<15}")
    print("-" * 100)
    
    batches = [1, 2, 4, 8]
    S_fixed = 2048
    H_fixed = 8
    
    for b in batches:
        tc = run_kernel_benchmark(q, kernel, b, H_fixed, S_fixed, 64, iter=10)
        tt = run_torch_benchmark(b, H_fixed, S_fixed, 64, iter=10)
        
        status = "OK"
        if tc == 0: status = "FAIL (Context)"
        
        speedup = tt/tc if tc > 0 else 0
        bw = MetricCalculator.compute_bandwidth(b, H_fixed, S_fixed, 64, tc)
        
        print(f"{f'B={b}, S={S_fixed}, H={H_fixed}':<25} | {tc:<15.4f} | {tt:<15.4f} | {f'{speedup:.2f}x':<10} | {bw:<10.2f} | {status:<15}")
    print("")

    # 3. VRAM Stress
    print(f"3. MEMORY CONTEXT STRESS TEST")
    print("-" * 100)
    print("Determining Maximum supportable Batch Size @ S=2048...")
    tester = VRAMStressTest()
    max_c = tester.find_max_batch('custom', 2048)
    max_t = tester.find_max_batch('torch', 2048)
    
    print(f"\nMAX BATCH SIZE SUMMARY:")
    print(f"Custom Kernel: {max_c}")
    print(f"PyTorch SDPA:  {max_t}")
    print("-" * 100)
    
    print("\nREPORT GENERATED.")

if __name__ == "__main__":
    main()
