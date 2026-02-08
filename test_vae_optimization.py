import torch
import math

# Mocking necessary parts
class MockModel:
    def decode(self, x):
        # x is in [0, 1] (or whatever), returns same shape
        return x

class MockVAE:
    def __init__(self):
        self.first_stage_model = MockModel()
        self.vae_dtype = torch.float32
        self.device = 'cpu'
        self.downscale_ratio = 8
        self.output_device = 'cpu'

    def decode_tiled_original(self, samples, tile_x=64, tile_y=64, overlap=16):
        # Simulating original logic
        decode_fn = lambda a: (self.first_stage_model.decode(a) + 1.0)

        t1 = tiled_scale(samples, decode_fn)
        t2 = tiled_scale(samples, decode_fn)
        t3 = tiled_scale(samples, decode_fn)

        output = torch.clamp(((t1 + t2 + t3) / 3.0) / 2.0, min=0.0, max=1.0)
        return output

    def decode_tiled_optimized(self, samples, tile_x=64, tile_y=64, overlap=16):
        # Simulating optimized logic
        decode_fn = lambda a: (self.first_stage_model.decode(a) + 1.0)

        output = tiled_scale(samples, decode_fn)
        output.add_(tiled_scale(samples, decode_fn))
        output.add_(tiled_scale(samples, decode_fn))

        output.div_(6.0)
        output.clamp_(min=0.0, max=1.0)
        return output

def tiled_scale(samples, function, *args, **kwargs):
    # Returns function(samples) just for simulation
    return function(samples)

def get_tiled_scale_steps(*args):
    return 1

if __name__ == "__main__":
    try:
        vae = MockVAE()
        samples = torch.randn(1, 4, 64, 64)

        # Original
        res1 = vae.decode_tiled_original(samples)

        # Optimized
        res2 = vae.decode_tiled_optimized(samples)

        diff = (res1 - res2).abs().max()
        print(f"Max difference: {diff}")

        assert torch.allclose(res1, res2, atol=1e-6)
        print("Optimization logic is correct!")

    except ImportError:
        print("Torch not installed, skipping test.")
    except Exception as e:
        print(f"Test failed with error: {e}")
