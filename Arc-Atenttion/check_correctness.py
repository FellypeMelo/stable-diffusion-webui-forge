import torch
import torch.nn.functional as F
import math
import sys
import os

# Add forge root to path to find scripts/backend
forge_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(forge_root)

# Mock 'shared.opts' for the wrapper script
class MockOpts:
    arc_logging_verbose = True
    arc_enable_native_attention = True

import types
shared = types.SimpleNamespace()
shared.opts = MockOpts()
sys.modules["shared"] = shared

# Import the wrapper
try:
    from scripts.arc_attention import arc_attention_backend_wrapper
except ImportError:
    # If running from Arc-Atenttion folder
    sys.path.append(os.path.join(forge_root, "scripts"))
    from arc_attention import arc_attention_backend_wrapper

def test_correctness():
    print("\n[Test] Initializing Correctness Check...")
    
    # Config
    B = 2
    Heads = 8
    Dim = 64 # Fixed for kernel
    Seq_Q = 64 # Image-like (8x8)
    Seq_KV = 77 # Text-like
    
    # Inputs (BF16)
    # Forge Layout: [B, Seq, Heads*Dim]
    # Actually backend.attention.attention_pytorch takes [B, Seq, H, D] or [B, H, Seq, D]?
    # Let's look at wrapper expectation: 
    # Wrapper checks: "b, seq_q, total_dim = q.shape" -> So [B, S, H*D]
    
    shape_q = (B, Seq_Q, Heads * Dim)
    shape_kv = (B, Seq_KV, Heads * Dim)
    
    q = torch.randn(shape_q, dtype=torch.bfloat16, device="cpu") # Simulate XPU via CPU fallback if needed? 
    # Wait, kernel requires XPU data ptrs usually? No, the code compiles for Windows/CPU? 
    # The compiled .pyd links against SYCL. It usually needs XPU device memory if using `unnamed_device_selector`?
    # Our kernel uses `gpu_selector_v`. It requires an Arc GPU.
    # We must try to move to 'xpu' if available, else this test will likely fail on `.data_ptr()`.
    
    try:
        device = torch.device("xpu")
        q = q.to(device)
    except:
        print("[WARN] XPU device not found. Using CPU (Kernel might fail/crash if it demands GPU).")
        device = torch.device("cpu")

    k = torch.randn(shape_kv, dtype=torch.bfloat16, device=device)
    v = torch.randn(shape_kv, dtype=torch.bfloat16, device=device)
    
    # 1. Baseline (PyTorch SDPA)
    # Reshape for SDPA: [B, H, S, D]
    q_ref = q.view(B, Seq_Q, Heads, Dim).transpose(1, 2)
    k_ref = k.view(B, Seq_KV, Heads, Dim).transpose(1, 2)
    v_ref = v.view(B, Seq_KV, Heads, Dim).transpose(1, 2)
    
    print("[Test] Running PyTorch Baseline...")
    # Scale is usually applied inside SDPA? Or manually? 
    # attention_basic applies it manually. SDPA does it automatically if not scaled?
    # backend.attention.attention_basic does: einsum(q,k) * scale.
    # checking math...
    out_ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=False)
    
    # 2. Kernel (via Wrapper)
    print("[Test] Running Arc-Attention Kernel...")
    try:
        # Wrapper expects packed inputs [B, S, H*D]
        out_ker = arc_attention_backend_wrapper(q, k, v, Heads)
    except Exception as e:
        print(f"[FAIL] Kernel crashed: {e}")
        return

    # 3. Compare
    # Output of wrapper is [B, S, H*D]
    # Reshape ref to match
    out_ref_flat = out_ref.transpose(1, 2).reshape(B, Seq_Q, Heads * Dim)
    
    # Verify
    diff = (out_ref_flat - out_ker).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
        
    print(f"\n[Results]")
    print(f"Max Diff: {max_diff:.6f}")
    print(f"Mean Diff: {mean_diff:.6f}")
    
    if max_diff < 1e-1: # BF16 precision is low, < 0.1 is usually passable for Attention
        print("[PASS] Kernel matches PyTorch logic!")
    else:
        print("[FAIL] Significant output mismatch. Implementation incorrect.")

if __name__ == "__main__":
    test_correctness()
