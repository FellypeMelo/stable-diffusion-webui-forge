import os
import sys
import time
import argparse
import numpy as np
import dpctl
import dpctl.memory as dpmem
import dpctl.tensor as dpt
from attention_kernel import AttentionKernel

# Try to import torch for reference
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

class BenchmarkConfig:
    def __init__(self, b, h, s, d, iterations=20):
        self.B = b
        self.H = h
        self.S = s
        self.D = d
        self.iterations = iterations

    def __str__(self):
        return f"B={self.B}, H={self.H}, S={self.S}, D={self.D}"

class AttentionBenchmark:
    def __init__(self, config: BenchmarkConfig, queue=None):
        self.cfg = config
        self.num_elements = self.cfg.B * self.cfg.H * self.cfg.S * self.cfg.D
        self.size_bytes = self.num_elements * 2  # BF16 = 2 bytes
        
        if queue:
            self.queue = queue
            self.device = queue.get_sycl_device()
        else:
            self.device = self._get_gpu_device()
            self.queue = dpctl.SyclQueue(self.device) if self.device else None
        
        self.kernel = AttentionKernel()

    def _get_gpu_device(self):
        try:
            return dpctl.SyclDevice("level_zero:gpu")
        except:
            try:
                return dpctl.SyclDevice("gpu")
            except:
                return None

    def run_custom(self) -> float:
        if not self.queue: return 0.0
        
        q_mem = k_mem = v_mem = o_mem = None
        try:
            # Allocate USM Device Memory (64-byte aligned)
            # Using uint16 view for BF16 data
            q_mem = dpmem.MemoryUSMDevice(self.size_bytes, queue=self.queue)
            k_mem = dpmem.MemoryUSMDevice(self.size_bytes, queue=self.queue)
            v_mem = dpmem.MemoryUSMDevice(self.size_bytes, queue=self.queue)
            o_mem = dpmem.MemoryUSMDevice(self.size_bytes, queue=self.queue)

            # Random Init (Host -> Device)
            # FORCE view as uint8 to ensure byte-copy semantics if dpctl is picky
            host_data = np.random.randint(0, 65535, self.num_elements, dtype=np.uint16)
            q_mem.copy_from_host(host_data.view(np.uint8))
            k_mem.copy_from_host(host_data.view(np.uint8))
            v_mem.copy_from_host(host_data.view(np.uint8))
            
            # Pointer arithmetic check
            # Verify type of _pointer
            q_ptr = int(q_mem._pointer)
            k_ptr = int(k_mem._pointer)
            v_ptr = int(v_mem._pointer)
            o_ptr = int(o_mem._pointer)

            latency = self.kernel.benchmark(
                q_ptr, k_ptr, v_ptr, o_ptr,
                self.cfg.B, self.cfg.H, self.cfg.S, self.cfg.D,
                self.cfg.iterations
            )
            return latency
            
        except Exception as e:
            print(f"[FATAL] Custom Benchmark failed: {e}")
            return 0.0
        finally:
            # Explicit cleanup
            del q_mem, k_mem, v_mem, o_mem

    def run_torch(self) -> float:
        if not HAS_TORCH or not torch.xpu.is_available():
            return 0.0
            
        try:
            device = torch.device('xpu')
            dtype = torch.bfloat16
            shape = (self.cfg.B, self.cfg.H, self.cfg.S, self.cfg.D)
            
            q = torch.randn(shape, dtype=dtype, device=device)
            k = torch.randn(shape, dtype=dtype, device=device)
            v = torch.randn(shape, dtype=dtype, device=device)
            
            # Warmup
            for _ in range(5):
                torch.nn.functional.scaled_dot_product_attention(q, k, v)
            torch.xpu.synchronize()
            
            # Measure
            start = time.perf_counter()
            for _ in range(self.cfg.iterations):
                torch.nn.functional.scaled_dot_product_attention(q, k, v)
            torch.xpu.synchronize()
            
            return ((time.perf_counter() - start) * 1000) / self.cfg.iterations
            
        except Exception as e:
            print(f"[ERROR] PyTorch failed: {e}")
            return 0.0

class BenchmarkSuite:
    def __init__(self):
        self.results = []
        try:
            self.device = dpctl.SyclDevice("level_zero:gpu")
        except:
            self.device = dpctl.SyclDevice("gpu")
        self.queue = dpctl.SyclQueue(self.device)
        print(f"[SUITE] Using Queue on {self.device.name}")

    def calculate_tflops(self, cfg: BenchmarkConfig, lat_ms: float) -> float:
        if lat_ms <= 0: return 0.0
        # 4 * B * H * S * S * D FLOPs
        flops = 4.0 * cfg.B * cfg.H * cfg.S * cfg.S * cfg.D
        return (flops / 1e12) / (lat_ms / 1000.0)

    def run(self):
        # Suite Configuration
        BATCHES = [1, 2, 4, 8] # Reduced to 8 to be safe for now
        SEQLENS = [1024, 2048, 4096]
        HEADS   = [8] # Keep heads fixed for simplicity or scale if needed
        HEAD_DIM = 64
        
        print(f"{'Config':<30} | {'Custom (ms)':<12} | {'Torch (ms)':<12} | {'Speedup':<8} | {'TFLOPS':<8}")
        print("-" * 80)

        for b in BATCHES:
            for s in SEQLENS:
                for h in HEADS:
                    # Garbage collect before each run
                    import gc
                    gc.collect()
                    
                    cfg = BenchmarkConfig(b, h, s, HEAD_DIM, iterations=10)
                    bench = AttentionBenchmark(cfg, queue=self.queue)
                    
                    try:
                        t_custom = bench.run_custom()
                        t_torch = bench.run_torch()
                        
                        speedup = t_torch / t_custom if t_custom > 0 else 0.0
                        tflops = self.calculate_tflops(cfg, t_custom)
                        
                        print(f"{str(cfg):<30} | {t_custom:<12.3f} | {t_torch:<12.3f} | {speedup:<8.2f} | {tflops:<8.2f}")
                        
                        self.results.append({
                            'config': str(cfg),
                            'custom_ms': t_custom,
                            'torch_ms': t_torch,
                            'speedup': speedup,
                            'tflops': tflops
                        })
                    except Exception as e:
                        print(f"{str(cfg):<30} | ERROR: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', action='store_true', help='Run full benchmark suite')
    args = parser.parse_args()

    # Hardware Info
    try:
        device = dpctl.SyclDevice("level_zero:gpu")
        print("="*60)
        print(f"HARDWARE: {device.name}")
        print(f"  Driver: {device.driver_version}")
        print(f"  CUs:    {device.max_compute_units}")
        print(f"  Mem:    {device.global_mem_size / 1e9:.2f} GB")
        print("="*60)
    except:
        pass

    if args.suite:
        # Run Full Suite
        suite = BenchmarkSuite()
        suite.run()
    else:
        # Default Single Run
        cfg = BenchmarkConfig(1, 8, 1024, 64)
        bench = AttentionBenchmark(cfg)
        t_c = bench.run_custom()
        t_t = bench.run_torch()
        print(f"\nSimple Run (B=1 H=8 S=1024):")
        print(f"Custom: {t_c:.3f} ms")
        print(f"Torch:  {t_t:.3f} ms")
        print(f"Speedup: {t_t/t_c:.2f}x" if t_c > 0 else "Speedup: N/A")

if __name__ == "__main__":
    main()