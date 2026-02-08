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

# --- Mock Fallback to detect if it was called ---
def mock_fallback(*args, **kwargs):
    print("[Guard Test] Fallback triggered successfully!")
    return "FALLBACK_CALLED"

import backend.attention
backend.attention.attention_pytorch = mock_fallback
# ----------------------------------

def test_guard():
    print("\n[Guard Test] Initializing...")
    
    B, Heads, Dim = 1, 8, 64
    # Simulate Self-Attention (Large Seq)
    Seq_Q, Seq_KV = 4096, 4096 
    
    print(f"[Guard Test] Testing S_KV={Seq_KV} (Should trigger Fallback > 2048)...")
    
    q = torch.randn(B, Seq_Q, Heads * Dim, dtype=torch.bfloat16, device="xpu")
    k = torch.randn(B, Seq_KV, Heads * Dim, dtype=torch.bfloat16, device="xpu")
    v = torch.randn(B, Seq_KV, Heads * Dim, dtype=torch.bfloat16, device="xpu")
    
    try:
        # Wrapper expects [B, S, H*D]
        out = arc_attention.arc_attention_backend_wrapper(q, k, v, Heads)
        
        if out == "FALLBACK_CALLED":
            print("[PASS] The guard successfully prevented kernel execution.")
        else:
            print("[FAIL] The kernel executed despite S_KV=4096!")
            
    except Exception as e:
        print(f"[FAIL] Exception during test: {e}")

if __name__ == "__main__":
    try:
        import intel_extension_for_pytorch as ipex
        print("IPEX imported.")
    except:
        print("IPEX not found (Running on CPU? Guard should still work if shape is correct).")
        
    test_guard()
