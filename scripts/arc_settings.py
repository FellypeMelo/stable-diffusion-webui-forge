import gradio as gr
from modules import shared, script_callbacks
import torch

def on_ui_settings():
    print("[Arc-Forge] Registering UI Settings...")
    section = ('arc_xpu', "Intel Arc")
    
    shared.opts.add_option(
        "arc_precision_mode",
        shared.OptionInfo(
            "Auto (FP16)", 
            "Precision Mode (Requires Restart)", 
            gr.Dropdown, 
            lambda: {"choices": ["Auto (FP16)", "FP8 (Turbo)"]}, 
            section=section
        ).info("Auto uses FP16 (Stable). FP8 is Experimental (B580+).")
    )

    shared.opts.add_option(
        "arc_memory_headroom",
        shared.OptionInfo(
            2048, 
            "Memory Governor Headroom (MB)", 
            gr.Slider, 
            {"minimum": 512, "maximum": 4096, "step": 128}, 
            section=section
        ).info("Amount of VRAM to keep free. Higher = Safer.")
    )

    shared.opts.add_option(
        "arc_logging_verbose",
        shared.OptionInfo(
            False, 
            "Enable Verbose XPU Logging", 
            gr.Checkbox, 
            section=section
        )
    )

script_callbacks.on_ui_settings(on_ui_settings)

# Arc-Forge: Apply Settings to Backend
# This breaks the circular dependency between backend <-> modules
try:
    from backend.xpu import config
    
    # 1. Apply immediately on load (if shared.opts is ready)
    if hasattr(shared.opts, "arc_precision_mode"):
        config.set_precision_mode(shared.opts.arc_precision_mode)
        print(f"[Arc-Forge] Precision Mode Set: {shared.opts.arc_precision_mode}")

    # 2. Register callback to apply when settings change/load
    def on_app_started(demo, app):
        mode = getattr(shared.opts, "arc_precision_mode", "Auto (FP16)")
        config.set_precision_mode(mode)
        print(f"[Arc-Forge] Precision Mode Applied: {mode}")

    script_callbacks.on_app_started(on_app_started)

except Exception as e:
    print(f"[Arc-Forge] Error applying settings: {e}")
