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
elif os.environ.get("NVIDIA_VISIBLE_DEVICES") is not None:
    device = "cuda"
else:
    device = "cpu"


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


pipeline = None
data_dir = "/root/share_models/tmp"


def load_model(model_path):
    global pipeline
    
    # 根据模型路径判断使用哪种 pipeline
    if "Kolors" in model_path or "kolors" in model_path:
        from diffusers import KolorsPipeline
        pipeline = KolorsPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            variant="fp16"
        )
        pipeline.to(device)
        logger.info(f"Loaded Kolors pipeline from {model_path}")
        
    elif "stable-diffusion-xl" in model_path.lower() or "sdxl" in model_path.lower():
        from diffusers import StableDiffusionXLPipeline
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            variant="fp16"
        )
        pipeline.to(device)
        logger.info(f"Loaded SDXL pipeline from {model_path}")
        
    elif "stable-diffusion" in model_path.lower() or "sd-" in model_path.lower():
        from diffusers import StableDiffusionPipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16
        )
        pipeline.to(device)
        logger.info(f"Loaded Stable Diffusion pipeline from {model_path}")
        
    else:
        # 通用 pipeline 加载
        from diffusers import DiffusionPipeline
        try:
            pipeline = DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16
            )
            pipeline.to(device)
            logger.info(f"Loaded generic diffusion pipeline from {model_path}")
        except Exception as e:
            raise Exception(f"不支持的模型: {model_path}, 错误: {str(e)}")


def infer(prompt, negative_prompt="",
          height: int = 512,
          width: int = 512,
          num_inference_steps: int = 50,
          guidance_scale: float = 7.5,
          num_images_per_prompt: int = 1,
          seed=42):
    
    if pipeline is None:
        raise Exception("模型未加载，请先加载模型")
    
    seed = int(time.time()) if seed == -1 else seed
    
    # 确保尺寸是8的倍数
    width = width // 8 * 8
    height = height // 8 * 8
    
    # 生成图像
    generator = torch.Generator(device=device).manual_seed(seed)
    
    result = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        generator=generator,
    )
    
    # 保存图像
    save_dir = data_dir
    os.makedirs(save_dir, exist_ok=True)
    
    if num_images_per_prompt == 1:
        # 单张图像，返回路径
        image = result.images[0]
        path = f"{save_dir}/{random_uuid()}.png"
        image.save(path)
        return path
    else:
        # 多张图像，返回路径列表
        paths = []
        for i, image in enumerate(result.images):
            path = f"{save_dir}/{random_uuid()}.png"
            image.save(path)
            paths.append(path)
        return paths


def ui():
    CSS = """
    h1 {
        text-align: center;
        display: block;
    }
    """
    with gr.Blocks(css=CSS, theme="soft", fill_height=True) as demo:
        gr.Markdown("# Diffusers Image Generation")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="Prompt", 
                    lines=3,
                    placeholder="Enter your prompt here...",
                    value="一张瓢虫的照片，微距，变焦，高质量，电影，拿着一个牌子"
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt", 
                    lines=2,
                    placeholder="Enter your negative prompt here...", 
                    value=""
                )
                
                with gr.Row():
                    width = gr.Slider(256, 1024, value=512, step=64, label="Width")
                    height = gr.Slider(256, 1024, value=512, step=64, label="Height")
                
                with gr.Row():
                    num_inference_steps = gr.Slider(
                        10, 100, value=50, step=1, label="Number of inference steps"
                    )
                    guidance_scale = gr.Slider(
                        1.0, 20.0, value=7.5, step=0.5, label="Guidance scale"
                    )
                
                with gr.Row():
                    num_images_per_prompt = gr.Slider(
                        1, 4, value=1, step=1, label="Number of images per prompt"
                    )
                    seed = gr.Number(value=42, label="Seed (-1 for random)")
                
                run = gr.Button("Generate Image", variant="primary")
                
            with gr.Column():
                image_output = gr.Gallery(
                    label="Generated Images", 
                    show_label=True,
                    elem_id="gallery",
                    columns=2,
                    rows=2,
                    object_fit="contain",
                    height="auto"
                )

        def predict(prompt, negative_prompt="",
                    height: int = 512,
                    width: int = 512,
                    num_inference_steps: int = 50,
                    guidance_scale: float = 7.5,
                    num_images_per_prompt: int = 1,
                    seed=42):
            result = infer(prompt, negative_prompt, height, width, 
                           num_inference_steps, guidance_scale, num_images_per_prompt, 
                           seed)
            # 如果返回单个路径，转换为列表
            if isinstance(result, str):
                return [result]
            return result

        run.click(
            fn=predict, 
            inputs=[prompt, negative_prompt, height, width, num_inference_steps, guidance_scale, num_images_per_prompt, seed], 
            outputs=[image_output]
        )
        
    return demo


def run(port: int = 9001, model_path=""):
    if not model_path:
        raise ValueError("请提供模型路径参数 --model_path")
    
    load_model(model_path)
    ui().launch(
        server_name="0.0.0.0", 
        server_port=port, 
        share=False,
        allowed_paths=[data_dir]
    )


# TOPS_VISIBLE_DEVICES=6 python diffusers-image-main.py --model_path /root/share_models/webui-models/diffusers/Kolors-diffusers --port 9002

if __name__ == "__main__":
    from fire import Fire
    Fire(run)
