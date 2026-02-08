import torch
import torch.nn.functional as F
import math
import sys
import os

# Add forge root to path
forge_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(forge_root)

# Mock shared.opts
class MockOpts:
    arc_logging_verbose = True
    arc_enable_native_attention = True

import types
shared = types.SimpleNamespace()
shared.opts = MockOpts()
sys.modules["shared"] = shared

# Import wrapper
from scripts import arc_attention

# --- CRITICAL: Disable Fallback ---
def poisoning_fallback(*args, **kwargs):
    raise RuntimeError("Fallback triggered! Kernel failed to execute or validation conditions not met.")

# Monkeypatch the fallback (which is likely 'backend.attention.attention_basic' or similar alias in the script)
# Inspect arc_attention.py imports to see what it uses as fallback
# It uses: from backend.attention import attention_pytorch as fallback_func
arc_attention.fallback_func = poisoning_fallback
# ----------------------------------

def test_strict():
    print("\n[Strict Test] Initializing...")
    
    B, Heads, Dim = 1, 8, 64
    Seq_Q, Seq_KV = 64, 77
    
    # Try different shapes to ensure no broadcasting issues
    q = torch.randn(B, Seq_Q, Heads * Dim, dtype=torch.bfloat16, device="xpu")
    k = torch.randn(B, Seq_KV, Heads * Dim, dtype=torch.bfloat16, device="xpu")
    v = torch.randn(B, Seq_KV, Heads * Dim, dtype=torch.bfloat16, device="xpu")
    
    print("[Strict Test] Calling wrapper (Fallback Disabled)...")
    try:
        # Wrapper expects [B, S, H*D]
        out = arc_attention.arc_attention_backend_wrapper(q, k, v, Heads)
        print("[Strict Test] Kernel executed without fallback!")
        
        # Verify shape
        if out.shape != (B, Seq_Q, Heads * Dim):
            print(f"[FAIL] Output shape mismatch: {out.shape}")
        else:
            print(f"[PASS] Output shape correct: {out.shape}")
            
    except RuntimeError as e:
        print(f"[FAIL] Fallback triggered: {e}")
    except Exception as e:
        print(f"[FAIL] Kernel crashed: {e}")

if __name__ == "__main__":
    try:
        import intel_extension_for_pytorch as ipex
        print("IPEX imported.")
    except:
        print("IPEX not found (Running on CPU? Strict test requires XPU for kernel).")
        
    test_strict()
