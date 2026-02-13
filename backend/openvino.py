import os
import torch
import logging
import json
import shutil
import tempfile
from modules import shared, paths
from backend import memory_management

logger = logging.getLogger(__name__)

OPENVINO_AVAILABLE = False
try:
    import openvino
    from optimum.intel import OVStableDiffusionPipeline, OVStableDiffusionXLPipeline
    OPENVINO_AVAILABLE = True
except ImportError:
    pass

def is_enabled():
    if not OPENVINO_AVAILABLE:
        return False
    return getattr(shared.opts, 'openvino_enable', False)

def get_cache_dir(model_hash):
    d = os.path.join(paths.models_path, "OpenVINO", model_hash)
    os.makedirs(d, exist_ok=True)
    return d

class OpenVINOUNetWrapper(torch.nn.Module):
    def __init__(self, ov_unet, config):
        super().__init__()
        self.ov_unet = ov_unet
        self.config = config
        self.dtype = torch.float32
        self.device = torch.device("cpu")
        self.load_device = torch.device("cpu")
        self.offload_device = torch.device("cpu")
        self.initial_device = torch.device("cpu")

        self.is_sdxl = config.get("addition_embed_type", None) == "text_time"

    def forward(self, x, timesteps, context=None, y=None, control=None, transformer_options={}, **kwargs):
        inputs = {
            "sample": x,
            "timestep": timesteps,
            "encoder_hidden_states": context
        }

        if self.is_sdxl and y is not None:
            if y.shape[-1] >= 1280:
                text_embeds = y[..., :1280]
                time_ids = y[..., 1280:]
                inputs["added_cond_kwargs"] = {
                    "text_embeds": text_embeds,
                    "time_ids": time_ids
                }

        output = self.ov_unet(**inputs)

        if isinstance(output, tuple):
            return output[0]
        elif hasattr(output, "sample"):
            return output.sample
        else:
            return output[0]

class OpenVINOVAEWrapper(torch.nn.Module):
    def __init__(self, ov_vae_encoder, ov_vae_decoder, config):
        super().__init__()
        self.encoder = ov_vae_encoder
        self.decoder = ov_vae_decoder
        self.config = config
        self.dtype = torch.float32
        self.device = torch.device("cpu")
        self.load_device = torch.device("cpu")
        self.offload_device = torch.device("cpu")
        self.initial_device = torch.device("cpu")

    def encode(self, x):
        results = self.encoder(sample=x)

        if isinstance(results, tuple):
            moments = results[0]
        elif hasattr(results, "latent_sample"):
            moments = results.latent_sample
        else:
            moments = results

        class FakeDist:
            def sample(self, generator=None):
                if moments.shape[1] > 4:
                    mean, logvar = torch.chunk(moments, 2, dim=1)
                    std = torch.exp(0.5 * logvar)
                    return mean + std * torch.randn_like(mean)
                return moments

            def mode(self):
                if moments.shape[1] > 4:
                    mean, _ = torch.chunk(moments, 2, dim=1)
                    return mean
                return moments

        class FakeOutput:
            def __init__(self):
                self.latent_dist = FakeDist()

        return FakeOutput()

    def decode(self, z):
        results = self.decoder(latent_sample=z)
        if isinstance(results, tuple):
            return results[0]
        elif hasattr(results, "sample"):
            return results.sample
        return results

class OpenVINOTextEncoderWrapper(torch.nn.Module):
    def __init__(self, ov_text_encoder, config):
        super().__init__()
        self.ov_text_encoder = ov_text_encoder
        self.config = config
        self.dtype = torch.float32
        self.device = torch.device("cpu")
        self.load_device = torch.device("cpu")
        self.offload_device = torch.device("cpu")
        self.initial_device = torch.device("cpu")

    def forward(self, input_ids, **kwargs):
        results = self.ov_text_encoder(input_ids=input_ids, **kwargs)
        return results

def get_components(checkpoint_path, model_hash, is_sdxl=False):
    cache_dir = get_cache_dir(model_hash)

    if not os.path.exists(os.path.join(cache_dir, "model_index.json")):
        print(f"[OpenVINO] Converting {os.path.basename(checkpoint_path)} to OpenVINO IR...")
        try:
            from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

            PipelineClass = StableDiffusionXLPipeline if is_sdxl else StableDiffusionPipeline
            OVPipelineClass = OVStableDiffusionXLPipeline if is_sdxl else OVStableDiffusionPipeline

            print("[OpenVINO] Loading PyTorch model for conversion (this may take a while)...")
            pipe = PipelineClass.from_single_file(checkpoint_path, load_safety_checker=False)

            print("[OpenVINO] Exporting to OpenVINO...")
            with tempfile.TemporaryDirectory() as temp_dir:
                pipe.save_pretrained(temp_dir)
                ov_pipe = OVPipelineClass.from_pretrained(temp_dir, export=True)
                ov_pipe.save_pretrained(cache_dir)

            del pipe
            print(f"[OpenVINO] Saved to {cache_dir}")

        except Exception as e:
            print(f"[OpenVINO] Conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    print(f"[OpenVINO] Loading from {cache_dir}...")
    try:
        OVPipelineClass = OVStableDiffusionXLPipeline if is_sdxl else OVStableDiffusionPipeline
        ov_pipe = OVPipelineClass.from_pretrained(cache_dir)

        try:
            ov_pipe.to("GPU")
        except Exception as e:
            print(f"[OpenVINO] GPU not available or failed, using AUTO/CPU: {e}")

        components = {}
        if hasattr(ov_pipe, 'unet'):
            components['unet'] = OpenVINOUNetWrapper(ov_pipe.unet, ov_pipe.unet.config)

        if hasattr(ov_pipe, 'vae_encoder') and hasattr(ov_pipe, 'vae_decoder'):
            components['vae'] = OpenVINOVAEWrapper(ov_pipe.vae_encoder, ov_pipe.vae_decoder, ov_pipe.vae_decoder.config)

        if hasattr(ov_pipe, 'text_encoder'):
            components['text_encoder'] = OpenVINOTextEncoderWrapper(ov_pipe.text_encoder, ov_pipe.text_encoder.config)

        if hasattr(ov_pipe, 'text_encoder_2'):
            components['text_encoder_2'] = OpenVINOTextEncoderWrapper(ov_pipe.text_encoder_2, ov_pipe.text_encoder_2.config)

        if hasattr(ov_pipe, 'tokenizer'):
             components['tokenizer'] = ov_pipe.tokenizer
        if hasattr(ov_pipe, 'tokenizer_2'):
             components['tokenizer_2'] = ov_pipe.tokenizer_2
        if hasattr(ov_pipe, 'scheduler'):
             components['scheduler'] = ov_pipe.scheduler

        return components

    except Exception as e:
        print(f"[OpenVINO] Load failed: {e}")
        return None
