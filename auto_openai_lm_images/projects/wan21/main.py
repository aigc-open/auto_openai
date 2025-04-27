import copy
import os
import random
import threading
import time
import numpy as np
import torch
import gradio as gr
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanPipeline
from diffusers.utils import export_to_video, load_image
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from transformers import CLIPVisionModel
from examples import t2v_examples, i2v_examples
import torch
from fire import Fire
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

# Constants
KEEP_SUCCESS_TASK = 3600 * 10
KEEP_RUNING_TASK = 3600 * 2
LIMIT_RUNING_TASK = 10
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global variables for models
t2v_models = {}
i2v_models = {}
task_status = {}

MODEL_DIR = os.environ.get("MODEL_DIR", "/root/share_models/webui-models/wan")

T2V_MODEL_MAPPINGS = {
    "Wan2.1-T2V-1.3B-Diffusers": os.path.join(MODEL_DIR, "Wan2.1-T2V-1.3B-Diffusers"),
    "Wan2.1-T2V-14B-Diffusers": os.path.join(MODEL_DIR, "Wan2.1-T2V-14B-Diffusers"),
}

I2V_MODEL_MAPPINGS = {
    "Wan2.1-I2V-14B-720P-Diffusers": os.path.join(MODEL_DIR, "Wan2.1-I2V-14B-720P-Diffusers"),
    "Wan2.1-I2V-14B-480P-Diffusers": os.path.join(MODEL_DIR, "Wan2.1-I2V-14B-480P-Diffusers"),
}


# Model IDs mapping
MODEL_MAPPINGS = {
    **T2V_MODEL_MAPPINGS,
    **I2V_MODEL_MAPPINGS
}

def load_model(model_id, model_type="i2v"):
    """Load model based on model ID and type"""
    try:
        if model_type == "i2v":
            if model_id in i2v_models:
                return i2v_models[model_id]
                
            print(f"Loading I2V model: {model_id}")
            diffusers_model_id = MODEL_MAPPINGS[model_id]
            image_encoder = CLIPVisionModel.from_pretrained(
                diffusers_model_id, 
                subfolder="image_encoder", 
                torch_dtype=torch.float32
            )
            vae = AutoencoderKLWan.from_pretrained(
                diffusers_model_id, 
                subfolder="vae", 
                torch_dtype=torch.float32
            )
            
            # Check if it's a high-resolution model (720P) or lower resolution (480P)
            flow_shift = 5.0 if "720P" in model_id else 3.0
            
            if "Diffusers" in model_id:
                # For Diffusers models, use WanImageToVideoPipeline
                pipe = WanImageToVideoPipeline.from_pretrained(
                    diffusers_model_id, 
                    vae=vae, 
                    image_encoder=image_encoder, 
                    torch_dtype=torch.bfloat16
                )
            else:
                # For non-Diffusers models, use WanPipeline with custom scheduler
                pipe = WanPipeline.from_pretrained(
                    diffusers_model_id,
                    vae=vae,
                    image_encoder=image_encoder,
                    torch_dtype=torch.bfloat16
                )
                scheduler = UniPCMultistepScheduler(
                    prediction_type='flow_prediction', 
                    use_flow_sigmas=True, 
                    num_train_timesteps=1000, 
                    flow_shift=flow_shift
                )
                pipe.scheduler = scheduler
                
            pipe.to("cuda")
            i2v_models[model_id] = pipe
            return pipe
        elif model_type == "t2v":
            if model_id in t2v_models:
                return t2v_models[model_id]
                
            print(f"Loading T2V model: {model_id}")
            diffusers_model_id = MODEL_MAPPINGS[model_id]
            vae = AutoencoderKLWan.from_pretrained(
                diffusers_model_id, 
                subfolder="vae", 
                torch_dtype=torch.float32
            )
            
            # Check if high resolution setting should be used
            is_high_res = "14B" in model_id
            flow_shift = 5.0 if is_high_res else 3.0
            
            if "Diffusers" in model_id:
                # For Diffusers models, use WanTextToVideoPipeline
                pipe = WanPipeline.from_pretrained(
                    diffusers_model_id,
                    torch_dtype=torch.bfloat16
                )
            else:
                # For non-Diffusers models, use WanPipeline with custom scheduler
                pipe = WanPipeline.from_pretrained(
                    diffusers_model_id,
                    vae=vae,
                    torch_dtype=torch.bfloat16
                )
                scheduler = UniPCMultistepScheduler(
                    prediction_type='flow_prediction', 
                    use_flow_sigmas=True, 
                    num_train_timesteps=1000, 
                    flow_shift=flow_shift
                )
                pipe.scheduler = scheduler
                
            pipe.to("cuda")
            t2v_models[model_id] = pipe
            return pipe
    except Exception as e:
        print(f"Error loading model {model_id}: {e}")
        return None

def t2v_generation_async(prompt, size, watermark_wanx, model, seed=-1, flow_shift=5.0):
    seed = seed if seed >= 0 else random.randint(0, 2147483647)
    print(f"T2V generation with seed: {seed}, flow_shift: {flow_shift}")
    
    if not allow_task_num():
        gr.Info("Warning: The number of running tasks is too large, please wait for a while.")
        return None, False, gr.Button(visible=True)
    
    # Generate unique task ID
    task_id = f"t2v_{int(time.time())}_{random.randint(1000, 9999)}"
    
    # Parse resolution
    width, height = map(int, size.split('*'))
    
    # Start generation in a thread
    threading.Thread(
        target=run_t2v_generation,
        args=(task_id, prompt, width, height, model, seed, watermark_wanx, flow_shift)
    ).start()
    
    return task_id, False, gr.Button(visible=False)

def run_t2v_generation(task_id, prompt, width, height, model, seed, watermark_wanx, flow_shift=5.0):
    try:
        # Ensure task is in task_status
        if task_id not in task_status:
            task_status[task_id] = {
                "value": 0,
                "status": False,
                "time": time.time(),
                "url": None
            }
        
        # Load the model
        pipe = load_model(model, "t2v")
        if pipe is None:
            task_status[task_id]["status"] = True
            task_status[task_id]["url"] = None
            return
        
        # Update scheduler if it's a non-Diffusers model
        if not "Diffusers" in model and hasattr(pipe, "scheduler"):
            print(f"Updating scheduler with flow_shift={flow_shift}")
            scheduler = UniPCMultistepScheduler(
                prediction_type='flow_prediction', 
                use_flow_sigmas=True, 
                num_train_timesteps=1000, 
                flow_shift=float(flow_shift)
            )
            pipe.scheduler = scheduler
        
        # Ensure dimensions are compatible with model
        mod_value = pipe.unet.config.sample_size if hasattr(pipe, 'unet') else 8
        height = (height // mod_value) * mod_value
        width = (width // mod_value) * mod_value
        
        # Set the generation parameters
        generator = torch.Generator(device="cuda").manual_seed(seed)
        negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        
        # Generate the video
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=81,
            guidance_scale=5.0,
            generator=generator
        ).frames[0]
        
        # Save the video to a file
        output_path = os.path.join(OUTPUT_DIR, f"{task_id}.mp4")
        export_to_video(output, output_path, fps=16)
        
        # Update task status
        task_status[task_id]["status"] = True
        task_status[task_id]["url"] = output_path
        
    except Exception as e:
        print(f"Error in T2V generation: {e}")
        task_status[task_id]["status"] = True
        task_status[task_id]["url"] = None

def i2v_generation_async(prompt, image, watermark_wanx, model, seed=-1, flow_shift=5.0):
    seed = seed if seed >= 0 else random.randint(0, 2147483647)
    print(f"I2V generation with seed: {seed}, flow_shift: {flow_shift}")
    
    if not allow_task_num():
        gr.Info("Warning: The number of running tasks is too large, please wait for a while.")
        return None, False, gr.Button(visible=True)
    
    if image is None:
        gr.Warning("Warning: Please upload an image")
        return "", None, gr.Button(visible=True)
    
    # Generate unique task ID
    task_id = f"i2v_{int(time.time())}_{random.randint(1000, 9999)}"
    
    # Start generation in a thread
    threading.Thread(
        target=run_i2v_generation,
        args=(task_id, prompt, image, model, seed, watermark_wanx, flow_shift)
    ).start()
    
    return task_id, False, gr.Button(visible=False)

def run_i2v_generation(task_id, prompt, image_path, model, seed, watermark_wanx, flow_shift=5.0):
    try:
        # Ensure task is in task_status
        if task_id not in task_status:
            task_status[task_id] = {
                "value": 0,
                "status": False,
                "time": time.time(),
                "url": None
            }
        
        # Load the model
        pipe = load_model(model, "i2v")
        if pipe is None:
            task_status[task_id]["status"] = True
            task_status[task_id]["url"] = None
            return
        
        # Update scheduler if it's a non-Diffusers model
        if not "Diffusers" in model and hasattr(pipe, "scheduler"):
            print(f"Updating scheduler with flow_shift={flow_shift}")
            scheduler = UniPCMultistepScheduler(
                prediction_type='flow_prediction', 
                use_flow_sigmas=True, 
                num_train_timesteps=1000, 
                flow_shift=float(flow_shift)
            )
            pipe.scheduler = scheduler
        
        # Load and prepare the image
        image = load_image(image_path)
        max_area = 720 * 1280
        aspect_ratio = image.height / image.width
        mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1] if hasattr(pipe, 'transformer') else 8
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        image = image.resize((width, height))
        
        # Set the generation parameters
        generator = torch.Generator(device="cuda").manual_seed(seed)
        negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        
        # Generate the video
        output = pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=81,
            guidance_scale=5.0,
            generator=generator
        ).frames[0]
        
        # Save the video to a file
        output_path = os.path.join(OUTPUT_DIR, f"{task_id}.mp4")
        export_to_video(output, output_path, fps=16)
        
        # Update task status
        task_status[task_id]["status"] = True
        task_status[task_id]["url"] = output_path
        
    except Exception as e:
        print(f"Error in I2V generation: {e}")
        task_status[task_id]["status"] = True
        task_status[task_id]["url"] = None

def get_result_with_task_id(task_id):
    if task_id == "": 
        return True, None
    
    if task_id in task_status:
        status = task_status[task_id]["status"]
        video_url = task_status[task_id]["url"]
        return status, video_url
    
    return False, None

def allow_task_num():
    # Count running tasks
    running_tasks = 0
    for task_id in task_status:
        if not task_status[task_id]["status"] and task_status[task_id]["time"] + 1800 > time.time():
            running_tasks += 1
    return running_tasks < LIMIT_RUNING_TASK

def clean_task_status():
    # Clean the task status dictionary
    for task_id in copy.deepcopy(task_status):
        if task_id == "": 
            continue
        
        # Finished task, keep KEEP_SUCCESS_TASK seconds
        if task_status[task_id]["status"]:
            if task_status[task_id]["time"] + KEEP_SUCCESS_TASK < time.time():
                # Remove the video file if it exists
                if task_status[task_id]["url"] and os.path.exists(task_status[task_id]["url"]):
                    try:
                        os.remove(task_status[task_id]["url"])
                    except:
                        pass
                task_status.pop(task_id)
        else:
            # Clean the task over KEEP_RUNING_TASK seconds
            if task_status[task_id]["time"] + KEEP_RUNING_TASK < time.time():
                task_status.pop(task_id)

def cost_time(task_id):
    if task_id in task_status and not task_status[task_id]["status"]:
        et = time.time() - task_status[task_id]["time"]
        return f"{et:.2f}"
    else:
        return gr.Textbox()

def get_process_bar(task_id, status):
    clean_task_status()
    if task_id not in task_status:
        task_status[task_id] = {
            "value": 0 if not task_id == "" else 100,
            "status": status if not task_id == "" else True,
            "time": time.time(),
            "url": None
        }
    
    if not task_status[task_id]["status"]:
        # Only when > 50% do check status
        if task_status[task_id]["value"] >= 10 and task_status[task_id]["value"] % 5 == 0:
            status, video_url = get_result_with_task_id(task_id)
        else:
            status, video_url = False, None
        
        task_status[task_id]["status"] = status
        if status:
            task_status[task_id]["url"] = video_url
    
    if task_status[task_id]["status"]:
        task_status[task_id]["value"] = 100
    else:
        task_status[task_id]["value"] += 1
    
    if task_status[task_id]["value"] >= 100 and not task_status[task_id]["status"]:
        task_status[task_id]["value"] = 95
    
    value = task_status[task_id]["value"]
    return gr.Slider(label=f"({value}%)Generating" if value % 2 == 1 else f"({value}%)Generating.....",
                     value=value)

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.HTML("""
               <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
                   Wan2.1
               </div>
               """)
    task_id = gr.State(value="")
    status = gr.State(value=False)
    task = gr.State(value="t2v")
    with gr.Row():
        with gr.Column():
            with gr.Row():
                with gr.Tabs():
                    # Text to Video Tab
                    with gr.TabItem("Text to Video") as t2v_tab:
                        t2v_model_select = gr.Dropdown(
                            label="Model",
                            value="Wan2.1-T2V-14B-Diffusers",
                            choices=list(T2V_MODEL_MAPPINGS.keys()))
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
                                    "1088*832", "832*1088"
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
                            run_t2v_button = gr.Button("Generate Video")
                    # Image to Video Tab
                    with gr.TabItem("Image to Video") as i2v_tab:
                        i2v_model_select = gr.Dropdown(
                            label="Model",
                            value="Wan2.1-I2V-14B-720P-Diffusers",
                            choices=list(I2V_MODEL_MAPPINGS.keys()))
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
                            run_i2v_button = gr.Button("Generate Video")
        with gr.Column():
            with gr.Row():
                result_gallery = gr.Video(label='Generated Video',
                                          interactive=False,
                                          height=500)
            with gr.Row():
                watermark_wanx = gr.Checkbox(label="Watermark",
                                             value=True,
                                             container=False)
                seed = gr.Number(label="Seed", value=-1, container=True)
                cost_time = gr.Number(label="Cost Time(secs)",
                                      value=cost_time,
                                      interactive=False,
                                      every=2,
                                      inputs=[task_id],
                                      container=True)
                process_bar = gr.Slider(show_label=True,
                                        label="",
                                        value=get_process_bar,
                                        maximum=100,
                                        interactive=True,
                                        every=3,
                                        inputs=[task_id, status],
                                        container=True)

    fake_video = gr.Video(label='Examples', visible=False, interactive=False)
    with gr.Row(visible=True) as t2v_eg:
        gr.Examples(t2v_examples,
                    inputs=[txt2vid_prompt, result_gallery],
                    outputs=[result_gallery])

    with gr.Row(visible=False) as i2v_eg:
        gr.Examples(i2v_examples,
                    inputs=[img2vid_prompt, img2vid_image, result_gallery],
                    outputs=[result_gallery])

    def process_change(task_id, task):
        status = task_status[task_id]["status"] if task_id in task_status else False
        if status:
            video_url = task_status[task_id]["url"] if task_id in task_status else None
            ret_t2v_btn = gr.Button(visible=True) if task == 't2v' else gr.Button()
            ret_i2v_btn = gr.Button(visible=True) if task == 'i2v' else gr.Button()
            return gr.Video(value=video_url), ret_t2v_btn, ret_i2v_btn
        return gr.Video(value=None), gr.Button(), gr.Button()

    process_bar.change(
        process_change,
        inputs=[task_id, task],
        outputs=[result_gallery, run_t2v_button, run_i2v_button])

    def switch_i2v_tab():
        return gr.Row(visible=False), gr.Row(visible=True), "i2v"

    def switch_t2v_tab():
        return gr.Row(visible=True), gr.Row(visible=False), "t2v"

    i2v_tab.select(switch_i2v_tab, outputs=[t2v_eg, i2v_eg, task])
    t2v_tab.select(switch_t2v_tab, outputs=[t2v_eg, i2v_eg, task])

    run_t2v_button.click(
        fn=t2v_generation_async,
        inputs=[
            txt2vid_prompt, resolution, watermark_wanx, t2v_model_select, seed, flow_shift
        ],
        outputs=[task_id, status, run_t2v_button],
    )

    run_i2v_button.click(
        fn=i2v_generation_async,
        inputs=[
            img2vid_prompt,
            img2vid_image,
            watermark_wanx,
            i2v_model_select,
            seed,
            flow_shift_i2v
        ],
        outputs=[task_id, status, run_i2v_button],
    )

def run(port=7860, model_path=None):
    # 更新 MODEL_DIR
    global MODEL_DIR
    if model_path:
        MODEL_DIR = model_path
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 启动 Gradio 应用
    demo.queue(max_size=10, default_concurrency_limit=20)
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)

if __name__ == "__main__":
    Fire(run)