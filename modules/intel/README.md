# Intel Arc Optimization Modules

This directory contains optimization modules specific to Intel Arc (Xe) GPUs.

## arc_attention.py

Implements **Sliced Scaled Dot Product Attention (SDPA)** to work around memory allocation limits on Arc GPUs.

### The Problem
Intel Arc GPUs (and the current driver stack) have a limit on the maximum size of a single memory allocation (typically ~4GB). Standard SDPA implementations in PyTorch can attempt to allocate very large intermediate tensors (e.g., for the attention matrix or gradients) during high-resolution generation, causing `OutOfMemory` errors or driver crashes even if the total VRAM is sufficient.

### The Solution: Query Chunking
We slice the Query (`Q`) tensor along the sequence length dimension (`L`) into smaller chunks. Each chunk is processed sequentially using the standard `torch.nn.functional.scaled_dot_product_attention`. This keeps the intermediate memory footprint of each operation within safe limits.

### Key Features
- **Automatic Slicing**: Dynamically calculates the safe chunk size based on the device's memory properties.
- **Performance**: Uses the optimized `sdpa` kernel (XMX-accelerated) for each chunk.
- **Stability**: Prevents black images and crashes on high-res generation (e.g., Hi-Res Fix, SDXL).

## Usage
These modules are automatically used by `modules/xpu_specific.py` when an XPU device is detected.
