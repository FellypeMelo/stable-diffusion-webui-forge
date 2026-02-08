import torch

def tiled_scale_sim(size):
    return torch.ones(size)

def test_original(size):
    print("Testing original...")
    t1 = tiled_scale_sim(size)
    t2 = tiled_scale_sim(size)
    t3 = tiled_scale_sim(size)

    # Simulating (t1 + t2 + t3) / 3.0 / 2.0
    res = (t1 + t2 + t3) / 3.0 / 2.0
    return res

def test_optimized(size):
    print("Testing optimized...")
    # output = tiled_scale(1)
    output = tiled_scale_sim(size)

    # output += tiled_scale(2)
    output.add_(tiled_scale_sim(size))

    # output += tiled_scale(3)
    output.add_(tiled_scale_sim(size))

    output.div_(3.0)
    output.div_(2.0)
    # output.clamp_...
    return output

if __name__ == "__main__":
    size = (100, 100)
    res1 = test_original(size)
    res2 = test_optimized(size)

    assert torch.allclose(res1, res2)
    print("Results match!")
