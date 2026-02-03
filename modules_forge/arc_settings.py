import gradio as gr
from modules import shared, script_callbacks
import torch

def on_ui_settings():
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
