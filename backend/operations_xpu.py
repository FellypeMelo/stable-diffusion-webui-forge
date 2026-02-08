import torch
from backend import operations as ops

class ForgeOperationsXPU(ops.ForgeOperations):
    class Linear(ops.ForgeOperations.Linear):
        def forward(self, x):
            # Check if we are in manual cast mode (standard for Forge operations)
            if self.parameters_manual_cast:
                # Get the "target" dtype which is usually FP8 in this mode
                target_dtype = ops.current_dtype

                # Check for FP8 availability and selection
                is_fp8 = False
                if hasattr(torch, 'float8_e4m3fn') and target_dtype == torch.float8_e4m3fn:
                    is_fp8 = True
                elif hasattr(torch, 'float8_e5m2') and target_dtype == torch.float8_e5m2:
                    is_fp8 = True

                if is_fp8:
                    # OPTIMIZATION: FP8 Weight Storage, FP16 Compute
                    # Reason: On Intel Arc (B580), casting input (FP16->FP8) is very slow (~0.8ms).
                    # Dequantizing weight (FP8->FP16) is very fast (~0.02ms).
                    # Pure FP16 compute is fast (~0.35ms).
                    # Strategy: Keep weights in FP8 to save VRAM, but compute in FP16.

                    # 1. Retrieve the weight (which might be FP8 or FP16)
                    # We pass skip_weight_dtype=True so weights_manual_cast doesn't try to cast it for us.
                    # We pass skip_bias_dtype=True to handle bias separately.
                    weight, bias, signal = ops.weights_manual_cast(self, x, skip_weight_dtype=True, skip_bias_dtype=True)

                    with ops.main_stream_worker(weight, bias, signal):
                        # 2. Prepare Weight for Compute (FP16)
                        # If the weight is ALREADY in FP8 (from loading or caching), we dequantize to FP16.
                        # If it's in FP16/FP32, we use it as is (or cast to x.dtype).

                        # Note: We do NOT cache the dequantized FP16 weight because that would defeat
                        # the purpose of saving VRAM. We want FP8 resident, FP16 temporary.

                        if weight.dtype == target_dtype:
                            # It's FP8, so dequantize to match input (FP16)
                            w_compute = weight.to(dtype=x.dtype)
                        elif weight.dtype != x.dtype:
                            # Ensure it matches input precision (likely FP16)
                            w_compute = weight.to(dtype=x.dtype)
                        else:
                            # Already matches (e.g. both FP16)
                            w_compute = weight

                        # 3. Perform Linear Operation (Matmul) in FP16
                        # This avoids the slow input cast (x -> x_fp8) entirely.
                        out = torch.nn.functional.linear(x, w_compute)

                        # 4. Add Bias (if present)
                        if bias is not None:
                            # Ensure bias matches output dtype
                            if bias.dtype != out.dtype:
                                bias = bias.to(dtype=out.dtype)
                            out = out + bias

                        return out

            # Fallback to standard implementation
            return super().forward(x)

    class Conv2d(ops.ForgeOperations.Conv2d):
        def forward(self, x):
            if self.parameters_manual_cast:
                target_dtype = ops.current_dtype

                is_fp8 = False
                if hasattr(torch, 'float8_e4m3fn') and target_dtype == torch.float8_e4m3fn:
                    is_fp8 = True
                elif hasattr(torch, 'float8_e5m2') and target_dtype == torch.float8_e5m2:
                    is_fp8 = True

                if is_fp8:
                    # OPTIMIZATION: FP8 Weight Storage, FP16 Compute
                    weight, bias, signal = ops.weights_manual_cast(self, x, skip_weight_dtype=True, skip_bias_dtype=True)

                    with ops.main_stream_worker(weight, bias, signal):
                        # Handle Weight Dequantization
                        if weight.dtype == target_dtype:
                            w_compute = weight.to(dtype=x.dtype)
                        elif weight.dtype != x.dtype:
                            w_compute = weight.to(dtype=x.dtype)
                        else:
                            w_compute = weight

                        # Compute in FP16
                        # Pass bias=None to handle it manually later
                        # self._conv_forward handles padding_mode logic
                        out = self._conv_forward(x, w_compute, None)

                        if bias is not None:
                            if bias.dtype != out.dtype:
                                bias = bias.to(dtype=out.dtype)
                            out = out + bias

                        return out

            return super().forward(x)
