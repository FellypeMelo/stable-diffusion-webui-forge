
import sys
import os
from pathlib import Path
import torch
import math
from modules import sd_hijack_optimizations, shared, script_callbacks, devices
from einops import rearrange

# helper
def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d

# Safe Imports for Patching Targets
ldm_cross_attention = None
sgm_cross_attention = None

try:
    import ldm.modules.attention
    import ldm.modules.diffusionmodules.model
    ldm_cross_attention = ldm.modules.attention.CrossAttention
except ImportError:
    pass

try:
    import sgm.modules.attention
    import sgm.modules.diffusionmodules.model
    sgm_cross_attention = sgm.modules.attention.CrossAttention
except ImportError:
    pass

# Add kernel path (Assuming Arc-Atenttion is in root/Arc-Atenttion)
# scripts/arc_attention.py -> parent -> parent -> Arc-Atenttion
forge_root = Path(__file__).parent.parent
kernel_path = forge_root / "Arc-Atenttion"

if str(kernel_path) not in sys.path:
    sys.path.append(str(kernel_path))

# DLL Fix: Add oneAPI bin paths to os.environ['PATH']
def add_oneapi_to_path():
    possible_paths = [
        r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin",
        r"C:\Program Files (x86)\Intel\oneAPI\compiler\2024.0\bin", # Example version
        r"C:\Program Files (x86)\Intel\oneAPI\compiler\2023.2.0\windows\bin",
        # Add common user install locations if needed
    ]
    
    # Try to find compiler root from env if set
    oneapi_root = os.environ.get("ONEAPI_ROOT")
    if oneapi_root:
        possible_paths.append(os.path.join(oneapi_root, "compiler", "latest", "bin"))

    for p in possible_paths:
        if os.path.exists(p) and p not in os.environ['PATH']:
            # print(f"[Arc-Attention] Adding oneAPI bin to PATH: {p}")
            os.environ['PATH'] = p + os.pathsep + os.environ['PATH']
            # Also add to DLL search path for Python 3.8+ safety
            if hasattr(os, 'add_dll_directory'):
                try:
                    os.add_dll_directory(p)
                except Exception:
                    pass

add_oneapi_to_path()

# Helper: Load Kernel
try:
    # V5: Streaming DPAS (XMX Optimized, Zero Memory Overflow)
    import attention_kernel_v5_dpas as attention_kernel
    _kernel_impl = attention_kernel.AttentionKernel()
    print("[Arc-Attn] Native Kernel V5 (DPAS/Streaming) Loaded.")
except ImportError as e:
    _kernel_impl = None

# Global flag to ensure we only log "Active" once per session/generation phase
_HasLoggedActive = False
_HasLoggedFallback = False # One-time fallback log

# ... (Imports)
import backend.attention

# ... (Previous imports and setup)

# Save original function just in case we need it for fallback/undo
_ORIGINAL_ATTENTION_FUNC = None

def arc_attention_backend_wrapper(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False):
    global _HasLoggedActive, _HasLoggedFallback, _ORIGINAL_ATTENTION_FUNC
    
    # Capture Original Dtype for strict return type compliance (Fixes Half/BF16 Mismatch)
    orig_dtype = q.dtype 
    
    # 0. Global Optimization: Prevention of FP16 Black Screens (Arc B580)
    # Force Upcast to BFloat16 to prevnet overflow in valid Self-Attention layers
    
    # 0. Global Optimization: Native FP16 (No Casts)
    # The kernel now supports native sycl::half, matching SD Forge's pipeline.
    # No upcast/downcast required. Epsilon fix in kernel handles stability.
    
    orig_dtype = q.dtype


    # Helper: Universal Fallback with Cast-Back
    fallback_func = _ORIGINAL_ATTENTION_FUNC if _ORIGINAL_ATTENTION_FUNC else backend.attention.attention_pytorch
    
    def run_fallback(reason=None):
        global _HasLoggedFallback
        if reason and not _HasLoggedFallback and reason != "Mask Present":
            print(f"[Arc-Attention] Fallback: {reason}")
            _HasLoggedFallback = True
            
        res = fallback_func(q, k, v, heads, mask, attn_precision, skip_reshape)
        # CRITICAL: Ensure we return the same dtype as input was (before upcast)
        if res.dtype != orig_dtype:
            res = res.to(orig_dtype)
        return res

    # 1. 1st Principle Validation
    if mask is not None:
        return run_fallback("Mask Present (Not supported in Inference-Optimized Kernel)")
    
    # Analyze Shapes
    if q.dim() == 3:
        b, seq_q, total_dim = q.shape
        d = total_dim // heads
    else:
        return run_fallback(f"Input Rank {q.dim()} != 3")

    if d != 64:
        return run_fallback(f"Head Dim {d} != 64 (Kernel is optimized for D=64)")

    # 2. Layout Transformation
    try:
        # Calculate seq_kv early
        if k.dim() == 3:
             seq_kv = k.shape[1]
        else:
             return run_fallback()

        # GUARD: Large Self-Attention (> 256) causes SLM Overflow on this kernel
        # This kernel attempts to load the ENTIRE K/V into Shared Local Memory.
        # Max Safe Size ~ 300 tokens (Cross-Attn is 77). 
        # Self-Attn (1024+) must fallback to PyTorch.
        if seq_kv > 300:
             # print(f"[Arc-Debug] Fallback Triggered: Seq_KV={seq_kv} > 300") 
             return run_fallback() # Implicitly handles casting

        # Transpose Q: [B, S, H*D] -> [B, S, H, D] -> [B, H, S, D]
        q_in = q.view(b, seq_q, heads, d).transpose(1, 2).contiguous()
        
        # Transpose K, V: [B, H, S_KV, D]
        k_in = k.view(b, seq_kv, heads, d).transpose(1, 2).contiguous()
        v_in = v.view(b, seq_kv, heads, d).transpose(1, 2).contiguous()
        
        
        # Ensure Inputs match Kernel Expectation (FP16 or BF16 depending on compile, now FP16)
        if q_in.dtype != torch.float16: q_in = q_in.to(torch.float16)
        if k_in.dtype != torch.float16: k_in = k_in.to(torch.float16)
        if v_in.dtype != torch.float16: v_in = v_in.to(torch.float16)
        
        
        # Allocate Output: [B, H, S, D]
        out_transposed = torch.empty_like(q_in)
        
        # 3. Kernel Execution
        # Explicit Scaling Factor (1/sqrt(d)) to ensure numerical stability before Softmax
        scale = 1.0 / math.sqrt(d)
        
        # Silenced per-step logging for performance
        # if getattr(shared.opts, "arc_logging_verbose", False):
        #      print(f"[Arc-Attn] Executing Kernel. Q={seq_q} K={seq_kv} Scale={scale:.3f}")

        # Validate Inputs (Before Kernel) - "Garbage In" check
        if torch.isnan(q_in).any() or torch.isinf(q_in).any() or \
           torch.isnan(k_in).any() or torch.isinf(k_in).any():
             # print(f"[Arc-Attn] WARN: Input Q/K contains NaN/Inf! Skipping Kernel.")
             return run_fallback()

        # SUPER VERBOSE LOGGING (First 5 steps only for speed)
        global _StepCounter
        try: _StepCounter += 1
        except: _StepCounter = 1
        
        if _StepCounter <= 5:
            print(f"\n[Arc-Debug] Step {_StepCounter}")
            print(f"  Shape: Q={q_in.shape}, K={k_in.shape}")
            print(f"  Stride: Q={q_in.stride()}, K={k_in.stride()}, V={v_in.stride()}")
            print(f"  Ptr: Q={hex(q_in.data_ptr())}, K={hex(k_in.data_ptr())}")
            print(f"  Dtype: Q={q_in.dtype}, K={k_in.dtype}")
            print(f"  Scale: {scale}")
            # print(f"  Sample Q[0,0,0,:4]: {q_in[0,0,0,:4]}") # Heavy CPU copy
        
        _kernel_impl.run(
            q_in.data_ptr(),
            k_in.data_ptr(),
            v_in.data_ptr(),
            out_transposed.data_ptr(),
            seq_q, seq_kv, heads, d, scale, b
        )

        # RUNTIME SAFETY: Check for Black Screen conditions (NaN/Inf)
        # BFloat16 kernel can overflow on some inputs. If so, discard and fallback.
        if torch.isnan(out_transposed).any() or torch.isinf(out_transposed).any():
             print(f"\n[Arc-Attn] CRITICAL: NaN/Inf detected in Kernel Output!")
             print(f"[Arc-Attn] Context: S_Q={seq_q}, S_KV={seq_kv}, Heads={heads}, Dim={d}")
             
             # Diagnostic Stats
             try:
                 print(f"[Arc-Attn] Q Stats: Min={q_in.min().item():.4f}, Max={q_in.max().item():.4f}, Mean={q_in.float().mean().item():.4f}")
                 print(f"[Arc-Attn] K Stats: Min={k_in.min().item():.4f}, Max={k_in.max().item():.4f}, Mean={k_in.float().mean().item():.4f}")
                 print(f"[Arc-Attn] V Stats: Min={v_in.min().item():.4f}, Max={v_in.max().item():.4f}, Mean={v_in.float().mean().item():.4f}")
             except:
                 print("[Arc-Attn] Failed to compute stats.")

             print("[Arc-Attn] Falling back to PyTorch...\n")
             return run_fallback()
        
        # Log Success once
        if not _HasLoggedActive:
             print("\n=======================================================")
             print(f" [Arc-Attention] ACTIVE - Hardware Acceleration Engaged")
             print(f" [Arc-Attention] Mode: Fused Cross-Attention (LSC+DPAS)")
             print(f" [Arc-Attention] Optimization: Selective (Cross-Attn Only)")
             print("=======================================================\n")
             _HasLoggedActive = True

        if not skip_reshape:
            # [B, H, S, D] -> [B, S, H, D] -> [B, S, H*D]
            out = out_transposed.transpose(1, 2).reshape(b, seq_q, total_dim)
            if out.dtype != orig_dtype: out = out.to(orig_dtype)
            return out
        else:
            if out_transposed.dtype != orig_dtype: out_transposed = out_transposed.to(orig_dtype)
            return out_transposed

    except Exception as e:
        print(f"[Arc-Attn] Kernel Execution Failed: {e}. Falling back.")
        return run_fallback()


# ... (Imports)
import backend.attention
# Import known consumers of attention_function to patch them as well
try: import backend.nn.unet; 
except: pass
try: import backend.nn.flux; 
except: pass
try: import backend.nn.chroma; 
except: pass
try: import backend.nn.vae; 
except: pass

# ... (Wrapper definition same as before) ...

def force_apply_arc_attention():
    global _ORIGINAL_ATTENTION_FUNC
    print(f"[Arc-Attention] Force-Applying Arc XMX Optimization (Backend Patch)...")
    
    # Store original if not already stored
    if _ORIGINAL_ATTENTION_FUNC is None:
        _ORIGINAL_ATTENTION_FUNC = backend.attention.attention_function
        
    # Patch main definition
    backend.attention.attention_function = arc_attention_backend_wrapper
    print(f"[Arc-Attention] Patched 'backend.attention.attention_function'")
    
    # Patch consumers
    patch_count = 0
    modules_to_patch = [
        ('backend.nn.unet', backend.nn.unet if 'backend.nn.unet' in sys.modules else None),
        ('backend.nn.flux', backend.nn.flux if 'backend.nn.flux' in sys.modules else None),
        ('backend.nn.chroma', backend.nn.chroma if 'backend.nn.chroma' in sys.modules else None),
        ('backend.nn.vae', backend.nn.vae if 'backend.nn.vae' in sys.modules else None),
    ]
    
    for name, mod in modules_to_patch:
        if mod and hasattr(mod, 'attention_function'):
            mod.attention_function = arc_attention_backend_wrapper
            print(f"[Arc-Attention] Patched '{name}.attention_function'")
            patch_count += 1
            
    print(f"[Arc-Attention] Successfully patched {1 + patch_count} locations.")

def on_app_started(demo, app):
    # Check if the user enabled the setting in arc_settings.py
    enabled = getattr(shared.opts, "arc_enable_native_attention", False)
    
    # User requested to mark as Experimental / Broken
    print(f"\n[Arc-Attention] Status: EXPERIMENTAL / DEV PREVIEW")
    print(f"[Arc-Attention] Note: Native Kernel may fail to load or compile. Using PyTorch fallback by default.")
    print(f"[Arc-Attention] App Started. Enable Native Attention: {enabled}")
    
    if enabled:
        try:
            force_apply_arc_attention()
        except Exception as e:
            print(f"[Arc-Attention] ERROR: Failed to apply optimization: {e}")
            print(f"[Arc-Attention] Reverting to standard attention.")
    else:
        pass

script_callbacks.on_app_started(on_app_started)

# Shim for SdOptimization if missing (AttributeError fix)
if hasattr(sd_hijack_optimizations, 'SdOptimization'):
    BaseOptimization = sd_hijack_optimizations.SdOptimization
else:
    class BaseOptimization:
        name = "Unknown"
        priority = 0
        def is_available(self): return False
        def apply(self): pass
        def undo(self): pass

# Keep the optimization class for manual selection availability if the list works
class SdOptimizationArc(BaseOptimization):
    name = "ArcAttention"
    label = "Intel Arc XMX"
    cmd_opt = "opt_arc_attention"
    priority = 200

    def is_available(self):
        return HAS_KERNEL and hasattr(torch, 'xpu') and torch.xpu.is_available()

    def apply(self):
        force_apply_arc_attention()

    def undo(self):
        pass

def register_optimizer(optimizers):
    if HAS_KERNEL:
        pool = [o.name for o in optimizers]
        if "ArcAttention" not in pool:
            try:
                optimizers.append(SdOptimizationArc())
            except:
                pass

script_callbacks.on_list_optimizers(register_optimizer)
