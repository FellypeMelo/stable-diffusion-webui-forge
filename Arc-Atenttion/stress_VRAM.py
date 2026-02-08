import os
import sys
import argparse
import numpy as np
import dpctl
import dpctl.memory as dpmem
import time
from attention_kernel import AttentionKernel

# Try to import torch for reference
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def get_gpu_device():
    try:
        return dpctl.SyclDevice("level_zero:gpu")
    except:
        return dpctl.SyclDevice("gpu")

class VRAMStressTest:
    def __init__(self):
        self.device = get_gpu_device()
        self.queue = dpctl.SyclQueue(self.device)
        self.kernel = AttentionKernel()
        print(f"[STRESS] Target Device: {self.device.name}")
        print(f"[STRESS] Max Compute Units: {self.device.max_compute_units}")
        print(f"[STRESS] Global Memory: {self.device.global_mem_size / (1024**3):.2f} GB")

    def run_custom(self, b, h, s, d):
        num_elements = b * h * s * d
        size_bytes = num_elements * 2
        
        try:
            # Allocate
            q_mem = dpmem.MemoryUSMDevice(size_bytes, queue=self.queue)
            k_mem = dpmem.MemoryUSMDevice(size_bytes, queue=self.queue)
            v_mem = dpmem.MemoryUSMDevice(size_bytes, queue=self.queue)
            o_mem = dpmem.MemoryUSMDevice(size_bytes, queue=self.queue)
            
            # Init (minimal to avoid host OOM, just zero fill on device if possible or tiny copy)
            # Actually need valid pointers.
            host_data = np.zeros(1, dtype=np.uint16) # Dummy
            # We don't copy full data to save time/host RAM. Just allocate.
            # Wait, kernel needs data? It reads/writes. Uninitialized is fine for stress test (results wrong but we care about crash).
            
            # Execute
            self.kernel.benchmark(
                int(q_mem._pointer), int(k_mem._pointer), int(v_mem._pointer), int(o_mem._pointer),
                b, h, s, d, 2 
            )
            return True
        except Exception as e:
            # print(f"Custom failed at B={b}: {e}")
            return False

    def run_torch(self, b, h, s, d):
        if not HAS_TORCH: return False
        try:
            device = torch.device('xpu')
            dtype = torch.bfloat16
            shape = (b, h, s, d)
            # Allocate
            q = torch.empty(shape, dtype=dtype, device=device)
            k = torch.empty(shape, dtype=dtype, device=device)
            v = torch.empty(shape, dtype=dtype, device=device)
            
            # Execute
            torch.nn.functional.scaled_dot_product_attention(q, k, v)
            torch.xpu.synchronize()
            return True
        except Exception as e:
            # print(f"Torch failed at B={b}: {e}")
            return False

    def find_max_batch(self, mode, s, h=8, d=64):
        low = 1
        high = 512 # Massive upper bound
        max_b = 0
        
        print(f"\n[STRESS] Finding Max Batch for {mode} (S={s})...")
        
        # Exponential search first
        curr = 1
        while True:
            success = False
            if mode == 'custom':
                success = self.run_custom(curr, h, s, d)
            else:
                success = self.run_torch(curr, h, s, d)
            
            if success:
                max_b = curr
                print(f"  B={curr}: OK")
                if curr * 2 > high: break
                curr *= 2
            else:
                print(f"  B={curr}: FAILED")
                break
        
        # Binary search refinement? No, linear back-off or just take the substantial power of 2 is enough for order-of-magnitude.
        # Let's try to refine a bit between max_b and curr
        # If B=64 OK, B=128 Fail -> Try 96.
        if max_b < curr:
            low = max_b
            high = curr
            while low + 1 < high:
                mid = (low + high) // 2
                success = False
                if mode == 'custom':
                    success = self.run_custom(mid, h, s, d)
                else:
                    success = self.run_torch(mid, h, s, d)
                
                if success:
                    max_b = mid
                    low = mid
                    print(f"  B={mid}: OK")
                else:
                    high = mid
                    print(f"  B={mid}: FAILED")

        return max_b

def main():
    test = VRAMStressTest()
    
    # Sequence Length to test
    SEQ_LEN = 2048 # Good balance
    
    print("="*60)
    print(f"VRAM STRESS TEST (S={SEQ_LEN}, H=8, D=64)")
    print("="*60)
    
    # Custom
    max_custom = test.find_max_batch('custom', SEQ_LEN)
    
    # Torch
    max_torch = test.find_max_batch('torch', SEQ_LEN)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Custom Kernel Max Batch: {max_custom}")
    print(f"PyTorch SDPA Max Batch:  {max_torch}")
    
    if max_custom > max_torch:
        print(">> WINNER: CUSTOM KERNEL (More Memory Efficient)")
    elif max_torch > max_custom:
        print(">> WINNER: PYTORCH (More Memory Efficient)")
    else:
        print(">> DRAW (Equal Memory Efficiency)")

if __name__ == "__main__":
    main()
