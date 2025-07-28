import os
import random
import numpy as np
import torch
import gradio as gr
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanPipeline
from diffusers.utils import export_to_video, load_image
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from transformers import CLIPVisionModel
from examples import t2v_examples, i2v_examples
import uuid
from loguru import logger
from fire import Fire

if os.environ.get("TOPS_VISIBLE_DEVICES") is not None:
    # 支持GCU算力卡
    try:
        import torch_gcu  # 导入 torch_gcu
        from torch_gcu import transfer_to_gcu  # 导入 transfer_to_gcu
        device = "gcu"
        torch_dtype = torch.bfloat16
        from gcu_diffusers import modify_apply_rotary_emb
        modify_apply_rotary_emb()
    except Exception as e:
        raise e
elif os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
    device = "cuda"
    torch_dtype = torch.float16
elif os.environ.get("NVIDIA_VISIBLE_DEVICES") is not None:
    device = "cuda"
    torch_dtype = torch.float16
else:
    device = "cpu"
    torch_dtype = torch.float32

# Constants
OUTPUT_DIR = "/root/share_models/tmp"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global variables for models
t2v_model = None
i2v_model = None

def random_uuid() -> str:
    return str(uuid.uuid4().hex)

def load_model(model_path):
    """Load model based on model path"""
    global t2v_model, i2v_model
    
    try:
        logger.info(f"Loading model from: {model_path}")
        
        # 确定是否使用自动设备映射
        use_auto_device_map = device == "cuda"
        if use_auto_device_map:
            # 尝试使用 balanced 策略，如果不行则使用具体的设备映射
            device_map = "balanced"
            # 备选方案：手动设备映射（如果 balanced 不工作）
            # device_map = {
            #     "transformer": 0,
            #     "vae": 0,
            #     "text_encoder": 1,
            #     "scheduler": 0
            # }
        else:
            device_map = device
        logger.info(f"Using device: {device}, device_map: {device_map}")
        
        if "T2V" in model_path:
            # Load T2V model
            logger.info("Loading T2V model")
            vae = AutoencoderKLWan.from_pretrained(
                model_path, 
                subfolder="vae", 
                torch_dtype=torch.float32,
                # device_map=device_map if use_auto_device_map else None
            )
            
            flow_shift = 5.0 if "14B" in model_path else 3.0
            
            if "Diffusers" in model_path:
                t2v_model = WanPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    device_map=device_map if use_auto_device_map else None
                )
            else:
                t2v_model = WanPipeline.from_pretrained(
                    model_path,
                    vae=vae,
                    torch_dtype=torch_dtype,
                    device_map=device_map if use_auto_device_map else None
                )
                scheduler = UniPCMultistepScheduler(
                    prediction_type='flow_prediction', 
                    use_flow_sigmas=True, 
                    num_train_timesteps=1000, 
                    flow_shift=flow_shift
                )
                t2v_model.scheduler = scheduler
            
            if not use_auto_device_map:
                t2v_model.to(device)
        elif "I2V" in model_path:
            # Load I2V model
            logger.info("Loading I2V model")
            image_encoder = CLIPVisionModel.from_pretrained(
                model_path, 
                subfolder="image_encoder", 
                torch_dtype=torch.float32,
                device_map=device_map if use_auto_device_map else None
            )
            vae = AutoencoderKLWan.from_pretrained(
                model_path, 
                subfolder="vae", 
                torch_dtype=torch.float32,
                # device_map=device_map if use_auto_device_map else None
            )
            
            flow_shift = 5.0 if "720P" in model_path else 3.0
            
            if "Diffusers" in model_path:
                i2v_model = WanImageToVideoPipeline.from_pretrained(
                    model_path, 
                    vae=vae, 
                    image_encoder=image_encoder, 
                    torch_dtype=torch_dtype,
                    device_map=device_map if use_auto_device_map else None
                )
            else:
                i2v_model = WanPipeline.from_pretrained(
                    model_path,
                    vae=vae,
                    image_encoder=image_encoder,
                    torch_dtype=torch_dtype,
                    device_map=device_map if use_auto_device_map else None
                )
                scheduler = UniPCMultistepScheduler(
                    prediction_type='flow_prediction', 
                    use_flow_sigmas=True, 
                    num_train_timesteps=1000, 
                    flow_shift=flow_shift
                )
                i2v_model.scheduler = scheduler
            
            if not use_auto_device_map:
                i2v_model.to(device)
            
        else:
            logger.error(f"Unknown model type in path: {model_path}")
            return False
            
        return True
        
    except Exception as e:
        logger.exception(f"Error loading model: {e}")
        return False

def t2v_generation(prompt, size, watermark_wanx, seed=-1, flow_shift=5.0, height=720, width=1280, num_frames=81, num_inference_steps=50, guidance_scale=5.0):
    """Generate video from text synchronously"""
    global t2v_model
    
    if t2v_model is None:
        gr.Warning("Model not loaded yet")
        return None
    
    try:
        seed = seed if seed >= 0 else random.randint(0, 2147483647)
        logger.info(f"T2V generation with seed: {seed}, flow_shift: {flow_shift}")
        
        # Parse resolution from dropdown if not using the sliders
        if size != "custom":
            width, height = map(int, size.split('*'))
        
        # Ensure dimensions are compatible with model
        mod_value = t2v_model.unet.config.sample_size if hasattr(t2v_model, 'unet') else 8
        height = (height // mod_value) * mod_value
        width = (width // mod_value) * mod_value
        
        # Set the generation parameters
        generator = torch.Generator(device=device).manual_seed(seed)
        negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        
        # Update scheduler if needed
        if hasattr(t2v_model, "scheduler") and not isinstance(t2v_model.scheduler, UniPCMultistepScheduler):
            logger.info(f"Updating scheduler with flow_shift={flow_shift}")
            scheduler = UniPCMultistepScheduler(
                prediction_type='flow_prediction', 
                use_flow_sigmas=True, 
                num_train_timesteps=1000, 
                flow_shift=float(flow_shift)
            )
            t2v_model.scheduler = scheduler
        
        # Generate the video
        output = t2v_model(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).frames[0]
        
        # Save the video to a file
        output_path = os.path.join(OUTPUT_DIR, f"{random_uuid()}.mp4")
        export_to_video(output, output_path, fps=16)
        
        return output_path
        
    except Exception as e:
        logger.exception(f"Error in T2V generation: {e}")
        return None

def i2v_generation(prompt, image, watermark_wanx, seed=-1, flow_shift=5.0, height=720, width=1280, num_frames=81, num_inference_steps=50, guidance_scale=5.0):
    """Generate video from image and text synchronously"""
    global i2v_model
    
    if i2v_model is None:
        gr.Warning("Model not loaded yet")
        return None
    
    if image is None:
        gr.Warning("Please upload an image")
        return None
    
    try:
        seed = seed if seed >= 0 else random.randint(0, 2147483647)
        logger.info(f"I2V generation with seed: {seed}, flow_shift: {flow_shift}")
        
        # Update scheduler if needed
        if hasattr(i2v_model, "scheduler") and not isinstance(i2v_model.scheduler, UniPCMultistepScheduler):
            logger.info(f"Updating scheduler with flow_shift={flow_shift}")
            scheduler = UniPCMultistepScheduler(
                prediction_type='flow_prediction', 
                use_flow_sigmas=True, 
                num_train_timesteps=1000, 
                flow_shift=float(flow_shift)
            )
            i2v_model.scheduler = scheduler
        
        # Load and prepare the image
        image = load_image(image)
        max_area = height * width
        aspect_ratio = image.height / image.width
        mod_value = i2v_model.vae_scale_factor_spatial * i2v_model.transformer.config.patch_size[1] if hasattr(i2v_model, 'transformer') else 8
        
        # Calculate dimensions while maintaining aspect ratio
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        image = image.resize((width, height))
        
        # Set the generation parameters
        generator = torch.Generator(device=device).manual_seed(seed)
        negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        
        # Generate the video
        output = i2v_model(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).frames[0]
        
        # Save the video to a file
        output_path = os.path.join(OUTPUT_DIR, f"{random_uuid()}.mp4")
        export_to_video(output, output_path, fps=16)
        
        return output_path
        
    except Exception as e:
        logger.exception(f"Error in I2V generation: {e}")
        return None

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.HTML("""
            <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
                Wan2.1
            </div>
            """)
    
    with gr.Row():
        with gr.Column():
            with gr.Row():
                with gr.Tabs():
                    # Text to Video Tab
                    with gr.TabItem("Text to Video") as t2v_tab:
                        with gr.Row():
                            txt2vid_prompt = gr.Textbox(
                                label="Prompt",
                                placeholder="Describe the video you want to generate",
                                lines=19,
                            )
                        with gr.Row():
                            resolution = gr.Dropdown(
                                label="Resolution",
                                choices=[
                                    "1280*720", "960*960", "720*1280",
                                    "1088*832", "832*1088", "custom"
                                ],
                                value="1280*720",
                            )
                        with gr.Row():
                            flow_shift = gr.Slider(
                                label="Flow Shift",
                                minimum=1.0,
                                maximum=10.0,
                                step=0.5,
                                value=5.0,
                                info="5.0 for high-res (720P), 3.0 for low-res (480P)"
                            )
                        with gr.Row():
                            with gr.Column(scale=1):
                                height = gr.Slider(
                                    label="Height",
                                    minimum=256,
                                    maximum=1280,
                                    step=8,
                                    value=720,
                                    info="Image height (must be divisible by 8)"
                                )
                            with gr.Column(scale=1):
                                width = gr.Slider(
                                    label="Width",
                                    minimum=256,
                                    maximum=1280,
                                    step=8,
                                    value=1280,
                                    info="Image width (must be divisible by 8)"
                                )
                        with gr.Row():
                            with gr.Column(scale=1):
                                num_frames = gr.Slider(
                                    label="Number of Frames",
                                    minimum=16,
                                    maximum=128,
                                    step=1,
                                    value=81,
                                    info="Number of frames to generate"
                                )
                            with gr.Column(scale=1):
                                num_inference_steps = gr.Slider(
                                    label="Inference Steps",
                                    minimum=20,
                                    maximum=100,
                                    step=1,
                                    value=50,
                                    info="Number of denoising steps"
                                )
                        with gr.Row():
                            guidance_scale = gr.Slider(
                                label="Guidance Scale",
                                minimum=1.0,
                                maximum=10.0,
                                step=0.1,
                                value=5.0,
                                info="Higher values enforce prompt adherence"
                            )
                        with gr.Row():
                            seed = gr.Number(label="Seed", value=-1, container=True)
                        with gr.Row():
                            watermark_wanx = gr.Checkbox(label="Watermark",
                                                        value=True,
                                                        container=False)
                        with gr.Row():
                            run_t2v_button = gr.Button("Generate Video")
                    
                    # Image to Video Tab
                    with gr.TabItem("Image to Video") as i2v_tab:
                        with gr.Row():
                            with gr.Column():
                                img2vid_image = gr.Image(
                                    type="filepath",
                                    label="Upload Input Image",
                                    elem_id="image_upload",
                                )
                                img2vid_prompt = gr.Textbox(
                                    label="Prompt",
                                    placeholder="Describe the video you want to generate",
                                    value="",
                                    lines=5,
                                )
                        with gr.Row():
                            flow_shift_i2v = gr.Slider(
                                label="Flow Shift",
                                minimum=1.0,
                                maximum=10.0,
                                step=0.5,
                                value=5.0,
                                info="5.0 for high-res (720P), 3.0 for low-res (480P)"
                            )
                        with gr.Row():
                            with gr.Column(scale=1):
                                height_i2v = gr.Slider(
                                    label="Height",
                                    minimum=256,
                                    maximum=1280,
                                    step=8,
                                    value=720,
                                    info="Image height (will be adjusted to maintain aspect ratio)"
                                )
                            with gr.Column(scale=1):
                                width_i2v = gr.Slider(
                                    label="Width",
                                    minimum=256,
                                    maximum=1280,
                                    step=8,
                                    value=1280,
                                    info="Image width (will be adjusted to maintain aspect ratio)"
                                )
                        with gr.Row():
                            with gr.Column(scale=1):
                                num_frames_i2v = gr.Slider(
                                    label="Number of Frames",
                                    minimum=16,
                                    maximum=128,
                                    step=1,
                                    value=81,
                                    info="Number of frames to generate"
                                )
                            with gr.Column(scale=1):
                                num_inference_steps_i2v = gr.Slider(
                                    label="Inference Steps",
                                    minimum=20,
                                    maximum=100,
                                    step=1,
                                    value=50,
                                    info="Number of denoising steps"
                                )
                        with gr.Row():
                            guidance_scale_i2v = gr.Slider(
                                label="Guidance Scale",
                                minimum=1.0,
                                maximum=10.0,
                                step=0.1,
                                value=5.0,
                                info="Higher values enforce prompt adherence"
                            )
                        with gr.Row():
                            seed = gr.Number(label="Seed", value=-1, container=True)
                        with gr.Row():
                            watermark_wanx = gr.Checkbox(label="Watermark",
                                                        value=True,
                                                        container=False)
                        with gr.Row():
                            run_i2v_button = gr.Button("Generate Video")

                        
        
        with gr.Column():
            with gr.Row():
                result_gallery = gr.Video(label='Generated Video',
                                          interactive=False,
                                          height=500)
            
                

    # Set up the example rows
    with gr.Row(visible=True) as t2v_eg:
        gr.Examples(t2v_examples,
                    inputs=[txt2vid_prompt, result_gallery],
                    outputs=[result_gallery])

    with gr.Row(visible=False) as i2v_eg:
        gr.Examples(i2v_examples,
                    inputs=[img2vid_prompt, img2vid_image, result_gallery],
                    outputs=[result_gallery])

    def switch_i2v_tab():
        return gr.Row(visible=False), gr.Row(visible=True)

    def switch_t2v_tab():
        return gr.Row(visible=True), gr.Row(visible=False)

    i2v_tab.select(switch_i2v_tab, outputs=[t2v_eg, i2v_eg])
    t2v_tab.select(switch_t2v_tab, outputs=[t2v_eg, i2v_eg])

    # Connect buttons to generation functions
    run_t2v_button.click(
        fn=t2v_generation,
        inputs=[
            txt2vid_prompt, resolution, watermark_wanx, seed, flow_shift,
            height, width, num_frames, num_inference_steps, guidance_scale
        ],
        outputs=[result_gallery],
    )

    run_i2v_button.click(
        fn=i2v_generation,
        inputs=[
            img2vid_prompt,
            img2vid_image,
            watermark_wanx,
            seed,
            flow_shift_i2v,
            height_i2v, width_i2v, num_frames_i2v, num_inference_steps_i2v, guidance_scale_i2v
        ],
        outputs=[result_gallery],
    )

def run(port=7860, model_path=None):
    """Run the Gradio app with the specified model path"""
    if model_path:
        success = load_model(model_path)
        if not success:
            logger.error(f"Failed to load model from {model_path}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Launch Gradio app
    demo.queue(max_size=5, default_concurrency_limit=1)
    demo.launch(server_name="0.0.0.0", server_port=port, share=False, allowed_paths=[OUTPUT_DIR])


# python3.10 main.py --model_path=/root/share_models/webui-models/wan/Wan2.1-T2V-1.3B-Diffusers

if __name__ == "__main__":
    Fire(run)