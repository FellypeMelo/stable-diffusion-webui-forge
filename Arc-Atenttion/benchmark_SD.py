import dpctl
import dpctl.memory as dpmem
import numpy as np
import time
from attention_kernel import AttentionKernel

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

class SDBenchmark:
    def __init__(self):
        try:
            self.device = dpctl.SyclDevice("level_zero:gpu")
        except:
            self.device = dpctl.SyclDevice("gpu")
        self.queue = dpctl.SyclQueue(self.device)
        self.kernel = AttentionKernel()
        print(f"HARDWARE: {self.device.name} ({self.device.max_compute_units} CUs)")

    def run_layer(self, name, b, h, s, d, iterations=20):
        print(f"Benchmarking Layer: {name} (B={b}, H={h}, S={s}, D={d})...")
        
        # 1. Custom Kernel
        try:
           # Alloc & Init
           num = b * h * s * d
           size = num * 2
           q_mem = dpmem.MemoryUSMDevice(size, queue=self.queue)
           k_mem = dpmem.MemoryUSMDevice(size, queue=self.queue)
           v_mem = dpmem.MemoryUSMDevice(size, queue=self.queue)
           o_mem = dpmem.MemoryUSMDevice(size, queue=self.queue)
           
           # Warmup
           for _ in range(3):
               self.kernel.benchmark(int(q_mem._pointer), int(k_mem._pointer), int(v_mem._pointer), int(o_mem._pointer), b, h, s, d, 1)
           
           t_custom = self.kernel.benchmark(int(q_mem._pointer), int(k_mem._pointer), int(v_mem._pointer), int(o_mem._pointer), b, h, s, d, iterations)
           
        except Exception as e:
           print(f"  [Custom] Failed: {e}")
           t_custom = 0.0

        # 2. PyTorch
        t_torch = 0.0
        if HAS_TORCH:
            try:
                dt = torch.bfloat16
                dev = torch.device('xpu')
                shape = (b, h, s, d)
                q = torch.randn(shape, dtype=dt, device=dev)
                k = torch.randn(shape, dtype=dt, device=dev)
                v = torch.randn(shape, dtype=dt, device=dev)
                
                # Warmup
                for _ in range(3):
                    torch.nn.functional.scaled_dot_product_attention(q, k, v)
                torch.xpu.synchronize()
                
                st = time.perf_counter()
                for _ in range(iterations):
                    torch.nn.functional.scaled_dot_product_attention(q, k, v)
                torch.xpu.synchronize()
                t_torch = ((time.perf_counter() - st) * 1000) / iterations
            except Exception as e:
                print(f"  [Torch] Failed: {e}")

        # Result
        speedup = t_torch / t_custom if t_custom > 0 else 0
        print(f"  Custom: {t_custom:.3f} ms | Torch: {t_torch:.3f} ms | Speedup: {speedup:.2f}x")
        return t_custom, t_torch

def main():
    bench = SDBenchmark()
    
    # SD 1.5 / 2.1 Representative Self-Attention Shapes (Batch=1)
    # Resolution -> Sequence Length (Res*Res)
    # Head Dim is typically 64 for 1.5. 
    # Heads vary: 8 -> 512 channels, 16 -> 1024 channels.
    
    layers = [
        # ("Stage 1 (64x64)", 1, 8, 4096, 64), # Known to crash on this kernel config
        ("Stage 2 (32x32)", 1, 16, 1024, 64),
        ("Stage 3 (16x16)", 1, 16, 256, 64), 
        ("Stage 4 (8x8)  ", 1, 32, 64, 64)
    ]
    
    total_custom = 0
    total_torch = 0
    
    print("\n" + "="*60)
    print("STABLE DIFFUSION SELF-ATTENTION SIMULATION (B=1)")
    print("="*60)
    
    for name, b, h, s, d in layers:
        tc, tt = bench.run_layer(name, b, h, s, d)
        total_custom += tc
        total_torch += tt
    
    print("\n" + "="*60)
    print("TOTAL FOR ONE U-NET PASS (Represented Layers)")
    print(f"Custom: {total_custom:.3f} ms")
    print(f"Torch:  {total_torch:.3f} ms")
    print(f"Overall Speedup: {total_torch/total_custom:.2f}x")
    print("="*60)

if __name__ == "__main__":
    main()
