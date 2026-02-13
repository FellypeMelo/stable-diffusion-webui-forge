import torch
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.clip import CLIP
from backend.patcher.vae import VAE
from backend.patcher.unet import UnetPatcher
from backend.text_processing.qwen_engine import QwenTextProcessingEngine
from backend import memory_management

class ZImage(ForgeDiffusionEngine):
    # This will be populated by loader or dummy
    matched_guesses = []

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)

        # Initialize CLIP (Text Encoder wrapper)
        # Z-Image uses Qwen as text encoder.
        clip = CLIP(
            model_dict={'qwen': huggingface_components['text_encoder']},
            tokenizer_dict={'qwen': huggingface_components['tokenizer']}
        )

        vae = VAE(model=huggingface_components['vae'])

        # Initialize Unet
        unet = UnetPatcher.from_model(
            model=huggingface_components['transformer'],
            diffusers_scheduler=huggingface_components['scheduler'],
            config=estimated_config
        )

        # Text Processing
        self.text_processing_engine = QwenTextProcessingEngine(
            text_encoder=clip.cond_stage_model.qwen,
            tokenizer=clip.tokenizer.qwen
        )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        # Turbo Detection
        self.is_turbo = "Turbo" in estimated_config.huggingface_repo

    def set_clip_skip(self, clip_skip):
        # Qwen usually uses last hidden state. Clip skip might not apply directly
        # or implies taking previous layers.
        # For now, we ignore or implement if Qwen supports it.
        pass

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)

        # If Turbo, we might ignore negative prompts (or handled by sampler/user)
        # But get_learned_conditioning just returns embeddings.

        cond = self.text_processing_engine(prompt)

        # Return dict expected by sampler
        # 'crossattn' is the standard key for text embeddings in Forge
        return dict(crossattn=cond)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        token_count = len(self.text_processing_engine.tokenize([prompt])[0])
        return token_count, max(255, token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x):
        sample = self.forge_objects.vae.encode(x.movedim(1, -1) * 0.5 + 0.5)
        if hasattr(self.forge_objects.vae.first_stage_model, 'process_in'):
             sample = self.forge_objects.vae.first_stage_model.process_in(sample)
        return sample.to(x)

    @torch.inference_mode()
    def decode_first_stage(self, x):
        if hasattr(self.forge_objects.vae.first_stage_model, 'process_out'):
            sample = self.forge_objects.vae.first_stage_model.process_out(x)
        else:
            sample = x
        sample = self.forge_objects.vae.decode(sample).movedim(-1, 1) * 2.0 - 1.0
        return sample.to(x)
