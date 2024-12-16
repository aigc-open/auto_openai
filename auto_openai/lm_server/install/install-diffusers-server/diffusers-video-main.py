import time
from PIL import Image
import requests
import io
import os
import json
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Optional, Union
from loguru import logger
import torch
import uuid

if os.environ.get("TOPS_VISIBLE_DEVICES") is not None:
    # 支持GCU算力卡
    try:
        import torch_gcu  # 导入 torch_gcu
        from torch_gcu import transfer_to_gcu  # 导入 transfer_to_gcu
        device = "gcu"
    except Exception as e:
        raise e
elif os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
    device = "cuda"
else:
    device = "cpu"

from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


pipline = None


def load_model(model_path):
    global pipline
    if "CogVideoX-5b" in model_path:
        pipline = CogVideoXPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        )
        pipline.to(device)
        pipline.enable_model_cpu_offload()
        pipline.vae.enable_tiling()


def infer(prompt, num_inference_steps: int = 25, num_frames: int = 49, guidance_scale: float = 6, seed=42, fps: int = 8):
    video = pipline(
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=num_inference_steps,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        generator=torch.Generator(device="cuda").manual_seed(seed),
    ).frames[0]
    path = f"/tmp/{random_uuid()}.mp4"
    export_to_video(video, path, fps=fps)
    return path


def ui():
    CSS = """
    h1 {
        text-align: center;
        display: block;
    }
    """
    with gr.Blocks(css=CSS, theme="soft", fill_height=True) as demo:
        gr.Markdown("# Diffusers Transformer Video")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", lines=2,
                                    placeholder="Enter your prompt here...",
                                    value="A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance.")
                num_inference_steps = gr.Slider(
                    1, 50, value=25, step=1, label="Number of inference steps")
                num_frames = gr.Slider(
                    1, 50, value=49, step=1, label="Number of frames")
                guidance_scale = gr.Slider(
                    1, 50, value=6, step=1, label="Guidance scale")
                seed = gr.Slider(1, 50, value=42, step=1, label="Seed")
                fps = gr.Slider(1, 50, value=8, step=1, label="FPS")
                run = gr.Button("Run")
            with gr.Column():
                video = gr.Video(label="Video")

        run.click(fn=infer, inputs=[prompt, num_inference_steps,
                  num_frames, guidance_scale, seed, fps], outputs=video)
    return demo
# PATH:auto_openai/lm_server/install/install-diffusers-transformer-video-main.py
# Usage: python diffusers-transformer-video-main.py --model_path /root/share_models/webui-models/CogVideo/CogVideoX-5b


def run(port: int = 9000, model_path=""):
    import uvicorn
    load_model(model_path)
    ui().launch(server_name="0.0.0.0", server_port=port, share=False)


if __name__ == "__main__":
    from fire import Fire
    Fire(run)
