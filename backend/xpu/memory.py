"""
Arc-Forge Memory Management
===========================
XPU memory utilities for Intel Arc GPUs
"""

import torch
from typing import Optional, Tuple


def is_xpu_available() -> bool:
    """Check if XPU is available."""
    try:
        return hasattr(torch, 'xpu') and torch.xpu.is_available()
    except Exception:
        return False


def get_free_memory(
    device: Optional[torch.device] = None,
    torch_free_too: bool = False
) -> int | Tuple[int, int]:
    """
    Get free XPU memory in bytes.
    
    Args:
        device: XPU device (default: current device)
        torch_free_too: If True, returns tuple of (total_free, torch_free)
        
    Returns:
        Free memory in bytes, or tuple if torch_free_too=True
    """
    if not is_xpu_available():
        return (0, 0) if torch_free_too else 0
    
    try:
        if device is None:
            device = torch.device("xpu", torch.xpu.current_device())
        
        stats = torch.xpu.memory_stats(device)
        mem_active = stats.get('active_bytes.all.current', 0)
        mem_reserved = stats.get('reserved_bytes.all.current', 0)
        
        # Free memory within PyTorch's reserved pool
        mem_free_torch = mem_reserved - mem_active
        
        # Total device memory minus reserved
        total_memory = torch.xpu.get_device_properties(device).total_memory
        mem_free_xpu = total_memory - mem_reserved
        
        # Total free = unreserved + free within reserved
        mem_free_total = mem_free_xpu + mem_free_torch
        
        if torch_free_too:
            return (mem_free_total, mem_free_torch)
        return mem_free_total
        
    except Exception:
        return (0, 0) if torch_free_too else 0


def get_memory_stats(device: Optional[torch.device] = None) -> dict:
    """
    Get detailed XPU memory statistics.
    
    Args:
        device: XPU device (default: current device)
        
    Returns:
        Dictionary with memory statistics
    """
    if not is_xpu_available():
        return {"available": False}
    
    try:
        if device is None:
            device = torch.device("xpu", torch.xpu.current_device())
        
        props = torch.xpu.get_device_properties(device)
        stats = torch.xpu.memory_stats(device)
        
        total = props.total_memory
        reserved = stats.get('reserved_bytes.all.current', 0)
        active = stats.get('active_bytes.all.current', 0)
        free = total - reserved
        
        return {
            "available": True,
            "total_mb": round(total / (1024**2), 2),
            "reserved_mb": round(reserved / (1024**2), 2),
            "active_mb": round(active / (1024**2), 2),
            "free_mb": round(free / (1024**2), 2),
            "utilization_percent": round((active / total) * 100, 1) if total > 0 else 0,
        }
        
    except Exception as e:
        return {"available": False, "error": str(e)}


def empty_cache() -> None:
    """
    Clear XPU memory cache.
    
    Call this after operations to free unused memory.
    """
    if is_xpu_available():
        try:
            torch.xpu.empty_cache()
        except Exception:
            pass


def synchronize(device: Optional[torch.device] = None) -> None:
    """
    Synchronize XPU device.
    
    Waits for all operations on the device to complete.
    
    Args:
        device: XPU device (default: current device)
    """
    if is_xpu_available():
        try:
            if device is None:
                torch.xpu.synchronize()
            else:
                torch.xpu.synchronize(device)
        except Exception:
            pass
