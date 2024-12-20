from auto_openai.utils.init_env import global_config
from loguru import logger
import os
from auto_openai.lm_server.docker_container import Docker


class CMD:

    @classmethod
    def get_environment(cls, device):
        if "NV" in global_config.GPU_TYPE:
            return [f"CUDA_VISIBLE_DEVICES={device}", f"NVIDIA_VISIBLE_DEVICES={device}"]
        else:
            return [f"TOPS_VISIBLE_DEVICES={device}"]

    @classmethod
    def get_image(cls, name):
        if "NV" in global_config.GPU_TYPE:
            image = global_config.IMAGE_BASE_PATH + f"/{name}:gpu"
        else:
            image = global_config.IMAGE_BASE_PATH + f"/{name}:gcu"
        return image

    @classmethod
    def get_vllm(cls, model_name, device, need_gpu_count, port, template, model_max_tokens, device_name, quantization=""):
        if "NV" in global_config.GPU_TYPE:
            block_size = 32
        else:
            block_size = 64
        if quantization:
            quantization = f"--quantization {quantization}"
        else:
            quantization = ""
        cmd = f"""
            python3 -m vllm.entrypoints.openai.api_server 
            --model {model_name} 
            --device={device_name} 
            {quantization}
            --enforce-eager
            --chat-template={template}
            --tensor-parallel-size={need_gpu_count} 
            --max-model-len={model_max_tokens} 
            --dtype=float16 --block-size={block_size} --trust-remote-code 
            --port={port}"""
        cmd = cmd.replace("\n", " ").strip()
        logger.info(f"本次启动模型: \n{cmd}")
        environment = cls.get_environment(device)
        container = Docker().run(image=cls.get_image(name="vllm"), command=cmd,
                                 device_ids=device,
                                 GPU_TYPE=global_config.GPU_TYPE, environment=environment)
        return cmd

    @classmethod
    def get_comfyui(cls, device, port):
        cmd = f"""
            python3 comfyui-main.py --listen 0.0.0.0
            --gpu-only --use-pytorch-cross-attention 
            --port={port}"""
        cmd = cmd.replace("\n", " ").strip()
        logger.info(f"本次启动模型: \n{cmd}")
        environment = cls.get_environment(device)
        environment.extend(['PYTHONPATH="/workspace/ComfyUI:$PYTHONPATH"',
                           f"COMFYUI_MODEL_PATH={global_config.COMFYUI_MODEL_ROOT_PATH}", "PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync"])
        container = Docker().run(image=cls.get_image(name="comfyui"), command=cmd,
                                 device_ids=device,
                                 GPU_TYPE=global_config.GPU_TYPE, environment=environment)

        return cmd

    @classmethod
    def get_maskgct(cls, device, port):
        cmd = f"""
            python3 main.py --port={port}
            --model_root_path={global_config.MASKGCT_MODEL_ROOT_PATH}"""
        cmd = cmd.replace("\n", " ").strip()
        logger.info(f"本次启动模型: \n{cmd}")
        environment = cls.get_environment(device)
        container = Docker().run(image=cls.get_image(name="maskgct"), command=cmd,
                                 device_ids=device,
                                 GPU_TYPE=global_config.GPU_TYPE, environment=environment)
        return cmd

    @classmethod
    def get_funasr(cls, device, port):
        cmd = f"""
            python3 funasr-main.py --port={port}
            --model_root_path={global_config.FUNASR_MODEL_ROOT_PATH}"""
        cmd = cmd.replace("\n", " ").strip()
        logger.info(f"本次启动模型: \n{cmd}")
        environment = cls.get_environment(device)
        container = Docker().run(image=cls.get_image(name="funasr-server"), command=cmd,
                                 device_ids=device,
                                 GPU_TYPE=global_config.GPU_TYPE, environment=environment)
        return cmd

    @classmethod
    def get_embedding(cls, device, port):
        cmd = f"""
            python3 embedding-main.py --port={port}
            --model_root_path={global_config.EMBEDDING_MODEL_ROOT_PATH}"""
        cmd = cmd.replace("\n", " ").strip()
        logger.info(f"本次启动模型: \n{cmd}")
        environment = cls.get_environment(device)
        container = Docker().run(image=cls.get_image(name="embedding-server"), command=cmd,
                                 device_ids=device,
                                 GPU_TYPE=global_config.GPU_TYPE, environment=environment)
        return cmd

    @classmethod
    def get_llm_transformer(cls, model_name, device, port):
        cmd = f"""
            python3 llm-transformer-main.py --port={port}
            --model_path={os.path.join(global_config.LLM_TRANSFORMER_MODEL_ROOT_PATH, model_name)}"""
        cmd = cmd.replace("\n", " ").strip()
        logger.info(f"本次启动模型: \n{cmd}")
        environment = cls.get_environment(device)
        container = Docker().run(image=cls.get_image(name="llm-transformer-server"), command=cmd,
                                 device_ids=device,
                                 GPU_TYPE=global_config.GPU_TYPE, environment=environment)
        return cmd

    @classmethod
    def get_diffusers_video(cls, model_name, device, port):
        cmd = f"""
            python3 diffusers-video-main.py --port={port}
            --model_path={os.path.join(global_config.DIFFUSERS_MODEL_ROOT_PATH, model_name)}"""
        cmd = cmd.replace("\n", " ").strip().strip()
        logger.info(f"本次启动模型: \n{cmd}")
        environment = cls.get_environment(device)
        container = Docker().run(image=cls.get_image(name="diffusers-server"), command=cmd,
                                 device_ids=device,
                                 GPU_TYPE=global_config.GPU_TYPE, environment=environment)
        return cmd

    @classmethod
    def get_rerank(cls, device, port):
        cmd = f"""
            python3 rerank-main.py --port={port}
            --model_root_path={global_config.RERANK_MODEL_ROOT_PATH}"""
        cmd = cmd.replace("\n", " ").strip()
        logger.info(f"本次启动模型: \n{cmd}")
        environment = cls.get_environment(device)
        container = Docker().run(image=cls.get_image(name="rerank-server"), command=cmd,
                                 device_ids=device,
                                 GPU_TYPE=global_config.GPU_TYPE, environment=environment)
        return cmd

    @classmethod
    def get_webui(cls, device, port):
        cmd = f"""
            python3 server.py 
            --skip-version-check --skip-install --skip-prepare-environment 
            --api
            --skip-torch-cuda-test --skip-load-model-at-start --enable-insecure-extension-access 
            --models-dir={global_config.WEBUI_MODEL_ROOT_PATH}
            --enable-insecure-extension-access --listen  
            --port={port}"""

        cmd = cmd.replace("\n", " ").strip()
        logger.info(f"本次启动模型: \n{cmd}")
        environment = cls.get_environment(device)
        container = Docker().run(image=cls.get_image(name="webui"), command=cmd,
                                 device_ids=device,
                                 GPU_TYPE=global_config.GPU_TYPE, environment=environment)
        return cmd

    @classmethod
    def kill(cls):
        Docker().stop()
        Docker().remove()
