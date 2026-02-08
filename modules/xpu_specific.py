from modules import shared
from modules.sd_hijack_utils import CondFunc
try:
    from modules.intel.arc_attention import sliced_sdp_attention
except ImportError:
    sliced_sdp_attention = None

has_ipex = False
try:
    import torch
    import intel_extension_for_pytorch as ipex # noqa: F401
    has_ipex = True
except Exception:
    pass


def check_for_xpu():
    return has_ipex and hasattr(torch, 'xpu') and torch.xpu.is_available()


def get_xpu_device_string():
    if shared.cmd_opts.device_id is not None:
        return f"xpu:{shared.cmd_opts.device_id}"
    return "xpu"


def torch_xpu_gc():
    with torch.xpu.device(get_xpu_device_string()):
        torch.xpu.empty_cache()


has_xpu = check_for_xpu()


# Arc GPU cannot allocate a single block larger than 4GB: https://github.com/intel/compute-runtime/issues/627
# We use a sliced attention implementation (ArcAttention) to work around this.
# ARC_SINGLE_ALLOCATION_LIMIT is now managed inside modules.intel.arc_attention

# Capture original function before patching
orig_sdp_attn_func = torch.nn.functional.scaled_dot_product_attention

def torch_xpu_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, *args, **kwargs
):
    # cast to same dtype first
    key = key.to(query.dtype)
    value = value.to(query.dtype)
    if attn_mask is not None and attn_mask.dtype != torch.bool:
        attn_mask = attn_mask.to(query.dtype)

    # Use robust sliced attention implementation
    if sliced_sdp_attention:
        # Extract scale if present in kwargs
        scale = kwargs.get('scale', None)
        return sliced_sdp_attention(
            query, key, value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            original_sdp_func=orig_sdp_attn_func
        )

    # Fallback to standard SDPA if module import failed (should not happen)
    return orig_sdp_attn_func(
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        *args, **kwargs
    )


def is_xpu_device(device: str | torch.device = None):
    if device is None:
        return False
    if isinstance(device, str):
        return device.startswith("xpu")
    return device.type == "xpu"


if has_xpu:
    try:
        # torch.Generator supports "xpu" device since 2.1
        torch.Generator("xpu")
    except RuntimeError:
        # W/A for https://github.com/intel/intel-extension-for-pytorch/issues/452: torch.Generator API doesn't support XPU device (for torch < 2.1)
        CondFunc('torch.Generator',
            lambda orig_func, device=None: torch.xpu.Generator(device),
            lambda orig_func, device=None: is_xpu_device(device))

    # W/A for some OPs that could not handle different input dtypes
    CondFunc('torch.nn.functional.layer_norm',
        lambda orig_func, input, normalized_shape=None, weight=None, *args, **kwargs:
        orig_func(input.to(weight.data.dtype), normalized_shape, weight, *args, **kwargs),
        lambda orig_func, input, normalized_shape=None, weight=None, *args, **kwargs:
        weight is not None and input.dtype != weight.data.dtype)
    CondFunc('torch.nn.modules.GroupNorm.forward',
        lambda orig_func, self, input: orig_func(self, input.to(self.weight.data.dtype)),
        lambda orig_func, self, input: input.dtype != self.weight.data.dtype)
    CondFunc('torch.nn.modules.linear.Linear.forward',
        lambda orig_func, self, input: orig_func(self, input.to(self.weight.data.dtype)),
        lambda orig_func, self, input: input.dtype != self.weight.data.dtype)
    CondFunc('torch.nn.modules.conv.Conv2d.forward',
        lambda orig_func, self, input: orig_func(self, input.to(self.weight.data.dtype)),
        lambda orig_func, self, input: input.dtype != self.weight.data.dtype)
    CondFunc('torch.bmm',
        lambda orig_func, input, mat2, out=None: orig_func(input.to(mat2.dtype), mat2, out=out),
        lambda orig_func, input, mat2, out=None: input.dtype != mat2.dtype)
    CondFunc('torch.cat',
        lambda orig_func, tensors, dim=0, out=None: orig_func([t.to(tensors[0].dtype) for t in tensors], dim=dim, out=out),
        lambda orig_func, tensors, dim=0, out=None: not all(t.dtype == tensors[0].dtype for t in tensors))
    CondFunc('torch.nn.functional.scaled_dot_product_attention',
        lambda orig_func, *args, **kwargs: torch_xpu_scaled_dot_product_attention(*args, **kwargs),
        lambda orig_func, query, *args, **kwargs: query.is_xpu)
