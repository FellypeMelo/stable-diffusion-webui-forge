# Arc-Forge

> **Intel Arc-Optimized Stable Diffusion WebUI Forge**

A fork of [Stable Diffusion WebUI Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) with first-class Intel Arc GPU support.

---

## 🚀 One-Step Installation

### Windows

```powershell
git clone https://github.com/FellypeMelo/Arc-Forge
cd Arc-Forge
.\webui-user.bat
```

### Linux

```bash
git clone https://github.com/FellypeMelo/Arc-Forge
cd Arc-Forge
./webui.sh
```

**That's it!** Arc-Forge will automatically:
- ✅ Detect your Intel Arc GPU (B580, A770, A750, etc.)
- ✅ Apply optimal VRAM settings
- ✅ Enable tiled VAE if needed (prevents OOM)

---

## 🎯 Supported GPUs

| GPU | VRAM | Status | Notes |
|-----|------|--------|-------|
| **Arc B580** | 12GB | ✅ Excellent | Best for SDXL |
| **Arc B570** | 10GB | ✅ Great | Good for SDXL |
| **Arc A770** | 16GB | ✅ Excellent | Best overall |
| **Arc A750** | 8GB | ✅ Good | Use with low-vram mode |
| **Arc A580** | 8GB | ✅ Good | Use with low-vram mode |
| **Arc A380** | 6GB | ⚡ Limited | SD 1.5 recommended |
| **Arc A310** | 4GB | ⚡ Very Limited | SD 1.5 only |

> 💡 **NVIDIA GPUs are also supported** - Arc-Forge maintains full compatibility with CUDA.

---

## ⚡ Quick Troubleshooting

### Out of Memory (OOM)?

Add `--always-low-vram` to your launch arguments in `webui-user.bat`:

```batch
set COMMANDLINE_ARGS=--skip-torch-cuda-test --always-low-vram
```

### Still crashing?

Enable tiled VAE in the UI:
1. Open **"Never OOM Integrated"** section
2. Check **"Enabled for VAE (always tiled)"**

---

## 🛠️ Requirements

- **GPU**: Intel Arc A-series or B-series (or NVIDIA)
- **OS**: Windows 10/11 or Linux
- **Python**: 3.10.x
- **Drivers**: Latest Intel GPU drivers

### Install Intel GPU Drivers

- **Windows**: [Intel Arc Graphics Driver](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html)
- **Linux**: Follow [Intel's Linux guide](https://dgpu-docs.intel.com/)

---

## 📊 Performance Tips

1. **Use BF16** - Intel Arc works best with bfloat16
2. **Enable SDP Attention** - `--opt-sdp-attention` (auto-enabled)
3. **Token Merging** - Set ratio to 0.4-0.5 for faster generation
4. **Close other apps** - Free up VRAM for generation

---

## 🔧 Arc-Forge Optimizations

Compared to standard Forge, Arc-Forge includes:

| Feature | Description |
|---------|-------------|
| 🧠 **Auto GPU Detection** | Identifies your Arc model automatically |
| ⚙️ **Smart Defaults** | Optimal VRAM and VAE settings per GPU |
| 🔄 **4GB Chunk Fix** | Workaround for Arc allocation limit |
| 💾 **Better Memory** | Improved XPU memory management |

---

## 📚 Original Forge Documentation

Arc-Forge is based on Stable Diffusion WebUI Forge. For general usage:

- [Flux Tutorial](https://github.com/lllyasviel/stable-diffusion-webui-forge/discussions/981)
- [ControlNet Guide](https://github.com/lllyasviel/stable-diffusion-webui-forge/discussions/932)
- [Extension List](https://github.com/lllyasviel/stable-diffusion-webui-forge/discussions/1754)

---

## 📝 License

Same license as [Stable Diffusion WebUI Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge).

---

## 🙏 Credits

- [lllyasviel](https://github.com/lllyasviel) - Original Forge
- [AUTOMATIC1111](https://github.com/AUTOMATIC1111) - Original WebUI
- Intel Arc Community

---

<p align="center">
  <b>Arc-Forge</b> - Making Stable Diffusion accessible on Intel Arc GPUs
</p>
