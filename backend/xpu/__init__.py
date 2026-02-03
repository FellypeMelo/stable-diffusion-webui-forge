"""
Arc-Forge XPU Module
====================
Intel Arc GPU Optimizations for Stable Diffusion WebUI Forge

This module provides:
- Intel Arc GPU detection and configuration
- XPU memory management utilities
- Optimized attention mechanisms for Arc's 4GB allocation limit
"""

from .device import (
    is_xpu_available,
    is_arc_gpu,
    get_arc_model,
    get_device_info,
    get_device_name,
)

from .config import (
    ARC_GPU_CONFIGS,
    get_optimal_settings,
)

from .memory import (
    get_free_memory,
    get_memory_stats,
    empty_cache,
)

__version__ = "1.0.0"
__all__ = [
    # Device
    "is_xpu_available",
    "is_arc_gpu", 
    "get_arc_model",
    "get_device_info",
    "get_device_name",
    # Config
    "ARC_GPU_CONFIGS",
    "get_optimal_settings",
    # Memory
    "get_free_memory",
    "get_memory_stats",
    "empty_cache",
]
