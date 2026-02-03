"""
Arc-Forge Device Detection
==========================
Intel Arc GPU detection and device utilities
"""

import torch
from typing import Optional


def is_xpu_available() -> bool:
    """
    Check if Intel XPU device is available.
    
    Returns:
        True if XPU is available (via IPEX or native PyTorch 2.5+)
    """
    try:
        return hasattr(torch, 'xpu') and torch.xpu.is_available()
    except Exception:
        return False


def get_device_name(device_id: int = 0) -> Optional[str]:
    """
    Get the name of the XPU device.
    
    Args:
        device_id: XPU device index (default: 0)
        
    Returns:
        Device name string or None if not available
    """
    if not is_xpu_available():
        return None
    try:
        return torch.xpu.get_device_name(device_id)
    except Exception:
        return None


def get_arc_model(device_id: int = 0) -> Optional[str]:
    """
    Detect Intel Arc GPU model from device name.
    
    Parses the device name to extract the Arc model identifier
    (e.g., "A770", "B580", "A750").
    
    Args:
        device_id: XPU device index (default: 0)
        
    Returns:
        Arc model string (e.g., "B580") or None if not detected
    """
    device_name = get_device_name(device_id)
    if not device_name:
        return None
    
    # Known Arc GPU model patterns
    arc_models = [
        # B-series (Battlemage)
        "B580", "B570",
        # A-series (Alchemist)
        "A770", "A750", "A580", "A380", "A310",
    ]
    
    device_name_upper = device_name.upper()
    for model in arc_models:
        if model in device_name_upper:
            return model
    
    # Check for generic Arc naming
    if "ARC" in device_name_upper:
        return "Arc (Unknown Model)"
    
    return None


def is_arc_gpu(device_id: int = 0) -> bool:
    """
    Check if the XPU device is an Intel Arc GPU.
    
    Args:
        device_id: XPU device index (default: 0)
        
    Returns:
        True if device is an Intel Arc GPU
    """
    return get_arc_model(device_id) is not None


def get_device_info(device_id: int = 0) -> dict:
    """
    Get comprehensive XPU device information.
    
    Args:
        device_id: XPU device index (default: 0)
        
    Returns:
        Dictionary with device properties
    """
    if not is_xpu_available():
        return {"available": False}
    
    try:
        props = torch.xpu.get_device_properties(device_id)
        return {
            "available": True,
            "device_id": device_id,
            "name": props.name,
            "total_memory_gb": round(props.total_memory / (1024**3), 2),
            "total_memory_bytes": props.total_memory,
            "arc_model": get_arc_model(device_id),
            "is_arc": is_arc_gpu(device_id),
            "driver_version": getattr(props, 'driver_version', 'unknown'),
        }
    except Exception as e:
        return {
            "available": True,
            "error": str(e),
        }
