import torch
from backend import operations as ops

class ForgeOperationsXPU(ops.ForgeOperations):
    class Linear(ops.ForgeOperations.Linear):
        def forward(self, x):
            # Check if we are in manual cast mode (standard for Forge operations)
            if self.parameters_manual_cast:
                target_dtype = ops.current_dtype

                # Check for FP8 availability and selection
                is_fp8 = False
                if hasattr(torch, 'float8_e4m3fn') and target_dtype == torch.float8_e4m3fn:
                    is_fp8 = True
                elif hasattr(torch, 'float8_e5m2') and target_dtype == torch.float8_e5m2:
                    is_fp8 = True

                if is_fp8:
                    # Retrieve weight/bias on device, but keep original dtype (e.g. FP16)
                    # We skip the automatic cast to target_dtype inside weights_manual_cast
                    # because we want to handle the FP8 cast explicitly just before compute.
                    weight, bias, signal = ops.weights_manual_cast(self, x, skip_weight_dtype=True, skip_bias_dtype=True)

                    with ops.main_stream_worker(weight, bias, signal):
                        # Native FP8 Cast & Compute on XPU
                        # 1. Cast Input to FP8
                        x_fp8 = x.to(target_dtype)

                        # 2. Cast Weight to FP8
                        w_fp8 = weight.to(target_dtype)

                        # 3. Perform Linear Operation (Matmul)
                        # PyTorch XPU backend handles fp8 matmul when inputs are fp8
                        out = torch.nn.functional.linear(x_fp8, w_fp8)

                        # 4. Cast Output back to original input dtype (likely FP16)
                        out = out.to(dtype=x.dtype)

                        # 5. Add Bias (if present)
                        # Bias is typically kept in higher precision (FP16/FP32)
                        if bias is not None:
                            out = out + bias

                        return out

            # Fallback to standard implementation (FP16/FP32)
            return super().forward(x)

    class Conv2d(ops.ForgeOperations.Conv2d):
        def forward(self, x):
            # Check if we are in manual cast mode (standard for Forge operations)
            if self.parameters_manual_cast:
                target_dtype = ops.current_dtype

                # Check for FP8 availability and selection
                is_fp8 = False
                if hasattr(torch, 'float8_e4m3fn') and target_dtype == torch.float8_e4m3fn:
                    is_fp8 = True
                elif hasattr(torch, 'float8_e5m2') and target_dtype == torch.float8_e5m2:
                    is_fp8 = True

                if is_fp8:
                    # Retrieve weight/bias on device, but keep original dtype (e.g. FP16)
                    # We skip the automatic cast to target_dtype inside weights_manual_cast
                    # because we want to handle the FP8 cast explicitly just before compute.
                    weight, bias, signal = ops.weights_manual_cast(self, x, skip_weight_dtype=True, skip_bias_dtype=True)

                    with ops.main_stream_worker(weight, bias, signal):
                        # Native FP8 Cast & Compute on XPU
                        # 1. Cast Input to FP8
                        x_fp8 = x.to(target_dtype)

                        # 2. Cast Weight to FP8
                        w_fp8 = weight.to(target_dtype)

                        # 3. Perform Conv2d Operation
                        # PyTorch XPU backend handles fp8 conv2d when inputs are fp8
                        # We use _conv_forward to handle padding_mode and other details automatically.
                        # We pass None for bias to perform bias addition in high precision later.
                        out = self._conv_forward(x_fp8, w_fp8, None)

                        # 4. Cast Output back to original input dtype (likely FP16)
                        out = out.to(dtype=x.dtype)

                        # 5. Add Bias (if present)
                        # Bias is typically kept in higher precision (FP16/FP32)
                        if bias is not None:
                            out = out + bias

                        return out

            # Fallback to standard implementation (FP16/FP32)
            return super().forward(x)
