from auto_openai.utils.init_env import global_config
from loguru import logger
import os

class CMD:
    @classmethod
    def get_vllm(cls, model_name, device, need_gpu_count, port, template, model_max_tokens, device_name, quantization=""):
        if quantization:
            quantization = f"--quantization {quantization}"
        cmd = f"""
            cd {global_config.VLLM_MODEL_ROOT_PATH} && 
            {global_config.GPU_DEVICE_ENV_NAME}={device} 
            python3 -m vllm.entrypoints.openai.api_server 
            --model {model_name} 
            --device={device_name} 
            {quantization}
            --enforce-eager
            --chat-template={template}
            --tensor-parallel-size={need_gpu_count} 
            --max-model-len={model_max_tokens} 
            --dtype=float16 --block-size=64 --trust-remote-code 
            --port={port} >> /tmp/{port}.log"""
        cmd = cmd.replace("\n", " ")
        logger.info(f"本次启动模型: \n{cmd}")
        return cmd

    @classmethod
    def get_comfyui(cls, device, port):
        cmd = f"""
            {global_config.GPU_DEVICE_ENV_NAME}={device} 
            python3 lm_server/comfyui-server.py --listen 0.0.0.0
            --gpu-only --use-pytorch-cross-attention 
            --port={port} >> /tmp/{port}.log"""
        cmd = cmd.replace("\n", " ")
        logger.info(f"本次启动模型: \n{cmd}")
        return cmd

    @classmethod
    def get_maskgct(cls, device, port):
        cmd = f"""
            cd {global_config.MASKGCT_ROOT_PATH} &&
            {global_config.GPU_DEVICE_ENV_NAME}={device} 
            python3 maskgct-main.py --port={port}
            --model_root_path={global_config.MASKGCT_MODEL_ROOT_PATH} >> /tmp/{port}.log"""
        cmd = cmd.replace("\n", " ")
        logger.info(f"本次启动模型: \n{cmd}")
        return cmd


    @classmethod
    def get_funasr(cls, device, port):
        cmd = f"""
            cd {global_config.FUNASR_ROOT_PATH} &&
            {global_config.GPU_DEVICE_ENV_NAME}={device} 
            python3 funasr-main.py --port={port}
            --model_root_path={global_config.FUNASR_MODEL_ROOT_PATH} >> /tmp/{port}.log"""
        cmd = cmd.replace("\n", " ")
        logger.info(f"本次启动模型: \n{cmd}")
        return cmd

    @classmethod
    def get_embedding(cls, device, port):
        cmd = f"""
            cd {global_config.EMBEDDING_ROOT_PATH} &&
            {global_config.GPU_DEVICE_ENV_NAME}={device} 
            python3 embedding-main.py --port={port}
            --model_root_path={global_config.EMBEDDING_MODEL_ROOT_PATH} >> /tmp/{port}.log"""
        cmd = cmd.replace("\n", " ")
        logger.info(f"本次启动模型: \n{cmd}")
        return cmd


    @classmethod
    def get_llm_transformer(cls, model_name, device, port):
        cmd = f"""
            cd {global_config.LLM_TRANSFORMER_ROOT_PATH} &&
            {global_config.GPU_DEVICE_ENV_NAME}={device} 
            python3 llm-transformer-main.py --port={port}
            --model_path={os.path.join(global_config.LLM_TRANSFORMER_MODEL_ROOT_PATH, model_name)} >> /tmp/{port}.log"""
        cmd = cmd.replace("\n", " ")
        logger.info(f"本次启动模型: \n{cmd}")
        return cmd