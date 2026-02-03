"""
Arc-Forge Attention Mechanisms
==============================
XPU-optimized attention with Arc 4GB allocation limit workaround

Intel Arc GPUs cannot allocate a single memory block larger than 4GB.
This module provides chunked attention implementations that split large
operations into smaller blocks to work within this hardware limitation.

Reference: https://github.com/intel/compute-runtime/issues/627
"""

import torch
from typing import Optional

# Arc GPUs cannot allocate >4GB in a single block
ARC_MAX_ALLOCATION_BYTES = 4 * 1024 * 1024 * 1024  # 4GB

# Cache for per-device allocation limits (may vary by GPU)
_device_allocation_limits: dict = {}


def get_allocation_limit(device: torch.device) -> int:
    """
    Get the maximum single allocation size for an XPU device.
    
    Uses a heuristic of min(total_memory / 8, 4GB) which provides
    a good balance between performance and stability.
    
    Args:
        device: XPU device
        
    Returns:
        Maximum allocation size in bytes
    """
    device_id = device.index if device.index is not None else 0
    
    if device_id not in _device_allocation_limits:
        try:
            total_memory = torch.xpu.get_device_properties(device_id).total_memory
            # Use min of 1/8 total memory or 4GB
            limit = min(total_memory // 8, ARC_MAX_ALLOCATION_BYTES)
            _device_allocation_limits[device_id] = limit
        except Exception:
            _device_allocation_limits[device_id] = ARC_MAX_ALLOCATION_BYTES
    
    return _device_allocation_limits[device_id]


def attention_xpu_chunked(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    Chunked scaled dot-product attention for Intel Arc GPUs.
    
    Automatically splits large batch operations into smaller chunks
    to avoid the 4GB allocation limit on Arc GPUs.
    
    Args:
        query: Query tensor [batch, heads, seq_len, dim] or [batch, seq_len, dim]
        key: Key tensor (same shape as query)
        value: Value tensor (same shape as query)
        attn_mask: Optional attention mask
        dropout_p: Dropout probability (default: 0.0)
        is_causal: If True, applies causal masking
        
    Returns:
        Attention output tensor
    """
    # Ensure consistent dtypes
    key = key.to(query.dtype)
    value = value.to(query.dtype)
    if attn_mask is not None and attn_mask.dtype != torch.bool:
        attn_mask = attn_mask.to(query.dtype)
    
    # Get tensor dimensions
    if query.dim() == 4:
        # [batch, heads, seq_len, dim]
        batch_size, num_heads, seq_len, head_dim = query.shape
        source_len = key.size(2)
        total_batch = batch_size * num_heads
        is_4d = True
    else:
        # [batch, seq_len, dim]
        batch_size, seq_len, head_dim = query.shape
        source_len = key.size(1)
        total_batch = batch_size
        is_4d = False
    
    # Calculate if chunking is needed
    allocation_limit = get_allocation_limit(query.device)
    bytes_per_element = query.element_size()
    attention_matrix_size = total_batch * seq_len * source_len * bytes_per_element
    
    # If within limit, use standard SDPA
    if attention_matrix_size <= allocation_limit:
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal
        )
    
    # Calculate optimal chunk size
    batch_limit = max(1, allocation_limit // (seq_len * source_len * bytes_per_element))
    num_chunks = (total_batch + batch_limit - 1) // batch_limit
    
    # Reshape for chunking
    if is_4d:
        query_flat = query.reshape(total_batch, seq_len, head_dim)
        key_flat = key.reshape(total_batch, source_len, head_dim)
        value_flat = value.reshape(total_batch, source_len, head_dim)
        if attn_mask is not None:
            attn_mask_flat = attn_mask.reshape(total_batch, seq_len, source_len)
        else:
            attn_mask_flat = None
    else:
        query_flat = query
        key_flat = key
        value_flat = value
        attn_mask_flat = attn_mask
    
    # Process in chunks
    outputs = []
    for i in range(num_chunks):
        start = i * batch_limit
        end = min((i + 1) * batch_limit, total_batch)
        
        q_chunk = query_flat[start:end].unsqueeze(0) if query_flat[start:end].dim() == 2 else query_flat[start:end]
        k_chunk = key_flat[start:end].unsqueeze(0) if key_flat[start:end].dim() == 2 else key_flat[start:end]
        v_chunk = value_flat[start:end].unsqueeze(0) if value_flat[start:end].dim() == 2 else value_flat[start:end]
        
        mask_chunk = None
        if attn_mask_flat is not None:
            mask_chunk = attn_mask_flat[start:end]
        
        # Reshape for SDPA (expects [batch, heads, seq, dim])
        if q_chunk.dim() == 3:
            q_chunk = q_chunk.unsqueeze(1)
            k_chunk = k_chunk.unsqueeze(1)
            v_chunk = v_chunk.unsqueeze(1)
        
        chunk_output = torch.nn.functional.scaled_dot_product_attention(
            q_chunk, k_chunk, v_chunk,
            attn_mask=mask_chunk,
            dropout_p=dropout_p,
            is_causal=is_causal
        )
        
        # Flatten back
        if chunk_output.dim() == 4 and chunk_output.size(1) == 1:
            chunk_output = chunk_output.squeeze(1)
        
        outputs.append(chunk_output)
    
    # Concatenate results
    result = torch.cat(outputs, dim=0)
    
    # Reshape back to original format
    if is_4d:
        result = result.reshape(batch_size, num_heads, seq_len, head_dim)
    
    return result
