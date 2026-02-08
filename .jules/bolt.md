## 2024-05-22 - VAE Decode Tiled Memory Optimization
**Learning:** The tiled VAE decode implementation accumulates results from 3 different tiling strategies (shifted grids) to reduce artifacts. The original implementation `(A + B + C) / 3` creates multiple temporary full-resolution buffers, spiking memory usage to 3-4x the output image size.
**Action:** Refactored `decode_tiled_` to use in-place accumulation (`output += ...`), reducing peak memory usage to ~2x (1 accumulator + 1 temporary result from `tiled_scale`). This significantly lowers OOM risk during high-res decoding.
