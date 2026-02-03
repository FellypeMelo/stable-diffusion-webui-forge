from contextlib import AbstractContextManager
import sys
import gc
import torch
import traceback
from backend.xpu.memory import get_memory_stats

class GracefulError(Exception):
    """
    Exception raised when an XPU crash is successfully caught and recovered from.
    This tells the UI to show a friendly error instead of crashing the process.
    """
    pass

class XPUCaptureContext(AbstractContextManager):
    """
    Context manager that captures XPU exceptions, logs them with context,
    and performs emergency cleanup to prevent zombie VRAM usage.
    
    Usage:
        with XPUCaptureContext("Unet Execution"):
            model(x)
    """
    def __init__(self, operation_name: str = "Unknown Operation"):
        self.operation_name = operation_name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            # 1. Check if it's an XPU/Memory related error
            is_oom = "out of memory" in str(exc_val).lower()
            is_device = "xpu" in str(exc_val).lower() or "device" in str(exc_val).lower()
            
            if is_oom or is_device:
                # 2. Capture Context
                vram_state = get_memory_stats()
                
                print(f"\n[Arc-Forge] 🚨 CRITICAL XPU ERROR during '{self.operation_name}'")
                print(f"[Arc-Forge] Exception: {exc_val}")
                print(f"[Arc-Forge] VRAM State at Crash: Used={vram_state.get('active_mb', '?')}MB, Total={vram_state.get('total_mb', '?')}MB")
                
                # 3. Emergency Cleanup
                print("[Arc-Forge] 🚑 Performing Emergency Cleanup...")
                try:
                    if hasattr(torch, 'xpu'):
                         torch.xpu.empty_cache()
                    gc.collect()
                except Exception as cleanup_e:
                    print(f"[Arc-Forge] Cleanup failed: {cleanup_e}")
                
                print("[Arc-Forge] ✅ System recovered. Generation cancelled safely.\n")
                
                # 4. Suppress Crash, Raise User-Friendly Error
                # We return True to suppress the original exception if we handled it
                # But typically we want to propagate a formatted error to the UI
                raise GracefulError(f"Generation interrupted by GPU Error ({self.operation_name}). System recovered.")
            
            # If not an XPU error, let it propagate normally
            return False
