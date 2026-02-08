import torch
import math
from modules import shared

# Arc-specific memory limit per allocation.
# Empirically tuned for optimal performance on B580/A770.
# Default limit is 4GB (hard limit for single allocation on Arc)
MAX_ARC_ALLOCATION_LIMIT = 4 * 1024 * 1024 * 1024

# Simple logger
def log_arc(msg):
    # Only log sparingly to avoid spam, or on critical events
    # For now, we print to stdout so the user can verify it's working
    # But we can guard it with a verbose flag if needed
    print(f"[Bolt-Arc] {msg}")

def get_arc_allocation_limit(device_id):
    """
    Returns the maximum safe allocation size for a single tensor on the given device.
    For Arc GPUs, this is typically constrained to avoid driver crashes.
    """
    try:
        if not torch.xpu.is_available():
             return MAX_ARC_ALLOCATION_LIMIT

        # Default heuristic: 1/8 of total VRAM, but capped at 4GB.
        # This is empirically tuned for A770/B580.
        props = torch.xpu.get_device_properties(device_id)
        total_mem = props.total_memory
        # Hard limit of 4GB for single allocation
        limit = min(total_mem // 8, MAX_ARC_ALLOCATION_LIMIT)
        # Ensure minimum usable chunk (256MB)
        return max(limit, 256 * 1024 * 1024)
    except Exception:
        # Fallback if xpu properties fail
        return 2 * 1024 * 1024 * 1024 # Conservative 2GB

def sliced_sdp_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, original_sdp_func=None):
    """
    Computes scaled dot product attention with automatic query chunking
    to respect Arc memory limits.

    Args:
        original_sdp_func: Reference to the unpatched/original scaled_dot_product_attention function.
                           Crucial to avoid infinite recursion if SDPA is monkey-patched.
    """
    # Use provided function or fallback (risky if patched)
    sdp_func = original_sdp_func if original_sdp_func is not None else torch.nn.functional.scaled_dot_product_attention

    # 1. Analyze Shapes
    # query shape: (..., L, E)
    L = query.size(-2)
    S = key.size(-2)

    # Calculate total batch size (product of all dims before L)
    batch_dims = query.shape[:-2]
    total_batch = 1
    for d in batch_dims:
        total_batch *= d

    elem_size = query.element_size()

    # 2. Check Memory Limit
    device_id = query.device.index if query.device.type == 'xpu' else 0
    limit = get_arc_allocation_limit(device_id)

    # Estimate size of the full attention matrix (Batch * L * S * sizeof(dtype))
    required_memory = total_batch * L * S * elem_size

    # If it fits, use the fast path (standard SDPA)
    if required_memory <= limit:
        # Pass explicit args to avoid kwargs issues if function signature differs
        return sdp_func(
            query, key, value,
            attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
        )

    # 3. Chunking Strategy
    # Only log once per generation/batch to avoid spam
    # log_arc(f"Slicing Attention: {required_memory / 1024**2:.1f}MB > {limit / 1024**2:.1f}MB limit. L={L}, S={S}")

    denominator = max(1, total_batch * S * elem_size)
    chunk_size = limit // denominator

    # Ensure chunk_size is at least 1
    chunk_size = max(1, int(chunk_size))

    # Align to 32 if possible (good for GPU SIMD)
    if chunk_size >= 64:
        chunk_size = (chunk_size // 32) * 32

    # 4. Processing Loop (Query Chunking)
    # Optimized: Pre-allocate output tensor to reduce peak memory usage.
    # Instead of holding a list of chunks (memory: Output) and concatenating (memory: Output),
    # resulting in 2x peak memory, we allocate Output once and write into it.
    result = torch.empty_like(query)

    def get_mask_chunk(start, end):
        if attn_mask is None:
            return None
        # Check if mask has L dimension
        if attn_mask.ndim >= 2 and attn_mask.size(-2) == L:
            return attn_mask[..., start:end, :]
        return attn_mask # Broadcast or unrelated shape

    for i in range(0, L, chunk_size):
        end = min(i + chunk_size, L)

        q_chunk = query[..., i:end, :]
        mask_chunk = get_mask_chunk(i, end)

        out_chunk = sdp_func(
            q_chunk, key, value,
            attn_mask=mask_chunk, dropout_p=dropout_p, is_causal=is_causal, scale=scale
        )

        # Write directly to result tensor
        result[..., i:end, :] = out_chunk

    return result
