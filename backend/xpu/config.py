"""
Arc-Forge GPU Configurations
============================
Optimal settings for Intel Arc GPU models
"""

from typing import Optional


# Arc GPU configurations with optimal defaults
# Format: model -> {vram_gb, vram_mode, vae_tiled, inference_memory_mb}
ARC_GPU_CONFIGS = {
    # =========================================
    # B-series (Battlemage) - 2024+
    # =========================================
    "B580": {
        "vram_gb": 12,
        "vram_mode": "normal",
        "vae_tiled": False,  # 12GB is enough for most operations
        "inference_memory_mb": 1280,
        "description": "Intel Arc B580 - Great for SDXL",
    },
    "B570": {
        "vram_gb": 10,
        "vram_mode": "normal",
        "vae_tiled": True,  # Enable for safety with 10GB
        "inference_memory_mb": 1024,
        "description": "Intel Arc B570",
    },
    
    # =========================================
    # A-series (Alchemist) - 2022-2023
    # =========================================
    "A770": {
        "vram_gb": 16,
        "vram_mode": "normal",
        "vae_tiled": False,  # 16GB handles everything
        "inference_memory_mb": 1536,
        "description": "Intel Arc A770 - Best Arc GPU for SD",
    },
    "A750": {
        "vram_gb": 8,
        "vram_mode": "low",  # 8GB needs careful management
        "vae_tiled": True,
        "inference_memory_mb": 1024,
        "description": "Intel Arc A750",
    },
    "A580": {
        "vram_gb": 8,
        "vram_mode": "low",
        "vae_tiled": True,
        "inference_memory_mb": 1024,
        "description": "Intel Arc A580",
    },
    "A380": {
        "vram_gb": 6,
        "vram_mode": "low",
        "vae_tiled": True,
        "inference_memory_mb": 768,
        "description": "Intel Arc A380 - Entry level, SD 1.5 recommended",
    },
    "A310": {
        "vram_gb": 4,
        "vram_mode": "no_vram",  # Very limited VRAM
        "vae_tiled": True,
        "inference_memory_mb": 512,
        "description": "Intel Arc A310 - Very limited, SD 1.5 only",
    },
}

# Default configuration for unknown Arc GPUs
DEFAULT_ARC_CONFIG = {
    "vram_gb": 8,
    "vram_mode": "low",
    "vae_tiled": True,
    "inference_memory_mb": 1024,
    "description": "Unknown Intel Arc GPU - Using safe defaults",
}


# =========================================================================
# XPU Memory Management Configuration
# =========================================================================
# These constants control memory behavior for Intel Arc GPUs.
# Tuned for stability and performance based on Arc architecture.
#
# KISS Principle: Simple, explicit values with clear rationale.
# =========================================================================

XPU_MEMORY_CONFIG = {
    # ─────────────────────────────────────────────────────────────────────
    # Safety Margin: Keep this percentage of VRAM free as buffer
    # Rationale: XPU driver needs headroom for internal allocations
    # ─────────────────────────────────────────────────────────────────────
    "safety_margin_percent": 0.15,
    
    # ─────────────────────────────────────────────────────────────────────
    # Minimum Free Memory Before Cleanup (MB)
    # Rationale: Avoid aggressive cleanup for small operations
    # ─────────────────────────────────────────────────────────────────────
    "min_free_before_cleanup_mb": 512,
    
    # ─────────────────────────────────────────────────────────────────────
    # Skip Unload Threshold
    # If free memory >= required * this factor, skip unload entirely
    # Rationale: Prevent unnecessary model unload/reload cycles
    # ─────────────────────────────────────────────────────────────────────
    "skip_unload_threshold": 1.3,
    
    # ─────────────────────────────────────────────────────────────────────
    # Maximum Retry Attempts
    # How many times to retry a failed memory operation
    # ─────────────────────────────────────────────────────────────────────
    "max_retry_attempts": 3,
    
    # ─────────────────────────────────────────────────────────────────────
    # Inference Memory Reserve (MB)
    # Memory reserved specifically for inference operations
    # ─────────────────────────────────────────────────────────────────────
    "inference_reserve_mb": 1024,
    
    # ─────────────────────────────────────────────────────────────────────
    # VAE Tile Sizes for Fallback (smallest to largest)
    # Used during progressive retry on VAE memory failures
    # ─────────────────────────────────────────────────────────────────────
    "vae_fallback_tile_sizes": [(32, 32), (48, 48), (64, 64)],
}


def get_optimal_settings(model: Optional[str] = None) -> dict:
    """
    Get optimal settings for an Intel Arc GPU model.
    
    Args:
        model: Arc model string (e.g., "B580", "A770")
               If None, attempts auto-detection
               
    Returns:
        Dictionary with optimal settings for the GPU
    """
    if model is None:
        # Try auto-detection
        from .device import get_arc_model
        model = get_arc_model()
    
    if model and model in ARC_GPU_CONFIGS:
        config = ARC_GPU_CONFIGS[model].copy()
        config["model"] = model
        config["auto_detected"] = True
        return config
    
    # Return safe defaults for unknown models
    config = DEFAULT_ARC_CONFIG.copy()
    config["model"] = model or "Unknown"
    config["auto_detected"] = model is not None
    return config


def should_use_tiled_vae(vram_gb: float) -> bool:
    """
    Determine if tiled VAE should be enabled based on VRAM.
    
    Args:
        vram_gb: Available VRAM in gigabytes
        
    Returns:
        True if tiled VAE should be enabled
    """
    # Enable tiled VAE for GPUs with less than 12GB
    return vram_gb < 12


def get_vram_mode_for_gpu(vram_gb: float) -> str:
    """
    Determine optimal VRAM mode based on available VRAM.
    
    Args:
        vram_gb: Available VRAM in gigabytes
        
    Returns:
        VRAM mode string: "normal", "low", or "no_vram"
    """
    if vram_gb >= 12:
        return "normal"
    elif vram_gb >= 6:
        return "low"
    else:
        return "no_vram"
