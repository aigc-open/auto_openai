from auto_openai.utils.init_env import global_config
from loguru import logger
import os


class CMD:
    @classmethod
    def get_vllm(cls, model_name, device, need_gpu_count, port, template, model_max_tokens, device_name, quantization=""):
        if quantization:
            quantization = f"--quantization {quantization}"
        else:
            quantization = ""
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
    def kill_vllm(cls):
        cmd = "ps -ef|grep vllm.entrypoints.openai.api_server | awk '{print $2}' | xargs kill -9"
        os.system(cmd)
        os.system(cmd)

    @classmethod
    def get_comfyui(cls, device, port):
        cmd = f"""
            {global_config.GPU_DEVICE_ENV_NAME}={device} 
            COMFYUI_MODEL_PATH={global_config.COMFYUI_MODEL_ROOT_PATH}
            python3 -m auto_openai.lm_server.comfyui_modify.server --listen 0.0.0.0
            --gpu-only --use-pytorch-cross-attention 
            --port={port}"""
        cmd = cmd.replace("\n", " ")
        logger.info(f"本次启动模型: \n{cmd}")
        return cmd

    @classmethod
    def kill_comfyui(cls):
        cmd = "ps -ef|grep auto_openai.lm_server.comfyui_modify.server | awk '{print $2}' | xargs kill -9"
        os.system(cmd)
        os.system(cmd)

    @classmethod
    def get_maskgct(cls, device, port):
        cmd = f"""
            cd {global_config.MASKGCT_ROOT_PATH} &&
            {global_config.GPU_DEVICE_ENV_NAME}={device} 
            python3 -m auto_openai.lm_server.maskgct_modify.main --port={port}
            --model_root_path={global_config.MASKGCT_MODEL_ROOT_PATH}"""
        cmd = cmd.replace("\n", " ")
        logger.info(f"本次启动模型: \n{cmd}")
        return cmd

    @classmethod
    def kill_maskgct(cls):
        cmd = "ps -ef|grep auto_openai.lm_server.maskgct_modify.main | awk '{print $2}' | xargs kill -9"
        os.system(cmd)
        os.system(cmd)

    @classmethod
    def get_funasr(cls, device, port):
        cmd = f"""
            cd {global_config.FUNASR_ROOT_PATH} &&
            {global_config.GPU_DEVICE_ENV_NAME}={device} 
            python3 funasr-main.py --port={port}
            --model_root_path={global_config.FUNASR_MODEL_ROOT_PATH}"""
        cmd = cmd.replace("\n", " ")
        logger.info(f"本次启动模型: \n{cmd}")
        return cmd

    @classmethod
    def kill_funasr(cls):
        cmd = "ps -ef|grep funasr-main.py | awk '{print $2}' | xargs kill -9"
        os.system(cmd)
        os.system(cmd)

    @classmethod
    def get_embedding(cls, device, port):
        cmd = f"""
            cd {global_config.EMBEDDING_ROOT_PATH} &&
            {global_config.GPU_DEVICE_ENV_NAME}={device} 
            python3 embedding-main.py --port={port}
            --model_root_path={global_config.EMBEDDING_MODEL_ROOT_PATH}"""
        cmd = cmd.replace("\n", " ")
        logger.info(f"本次启动模型: \n{cmd}")
        return cmd

    @classmethod
    def kill_embedding(cls):
        cmd = "ps -ef|grep embedding-main.py | awk '{print $2}' | xargs kill -9"
        os.system(cmd)
        os.system(cmd)

    @classmethod
    def get_llm_transformer(cls, model_name, device, port):
        cmd = f"""
            cd {global_config.LLM_TRANSFORMER_ROOT_PATH} &&
            {global_config.GPU_DEVICE_ENV_NAME}={device} 
            python3 llm-transformer-main.py --port={port}
            --model_path={os.path.join(global_config.LLM_TRANSFORMER_MODEL_ROOT_PATH, model_name)}"""
        cmd = cmd.replace("\n", " ")
        logger.info(f"本次启动模型: \n{cmd}")
        return cmd

    @classmethod
    def kill_llm_transformer(cls):
        cmd = "ps -ef|grep llm-transformer-main.py | awk '{print $2}' | xargs kill -9"
        os.system(cmd)
        os.system(cmd)

    @classmethod
    def get_rerank(cls, device, port):
        cmd = f"""
            cd {global_config.RERANK_ROOT_PATH} &&
            {global_config.GPU_DEVICE_ENV_NAME}={device} 
            python3 rerank-main.py --port={port}
            --model_root_path={global_config.RERANK_MODEL_ROOT_PATH}"""
        cmd = cmd.replace("\n", " ")
        logger.info(f"本次启动模型: \n{cmd}")
        return cmd

    @classmethod
    def kill_rerank(cls):
        cmd = "ps -ef|grep rerank-main.py | awk '{print $2}' | xargs kill -9"
        os.system(cmd)
        os.system(cmd)

    @classmethod
    def get_webui(cls, device, port):
        cmd = f"""
            {global_config.GPU_DEVICE_ENV_NAME}={device} 
            PYTHONPATH="/workspace/webui-site-packages/:$PYTHONPATH"
            python3 -m auto_openai.lm_server.webui_modify.server  
            --skip-version-check --skip-install --skip-prepare-environment 
            --skip-torch-cuda-test --skip-load-model-at-start --enable-insecure-extension-access 
            --models-dir={global_config.WEBUI_MODEL_ROOT_PATH}
            --enable-insecure-extension-access --listen  
            --port={port}"""

        cmd = cmd.replace("\n", " ")
        logger.info(f"本次启动模型: \n{cmd}")
        return cmd

    @classmethod
    def kill_webui(cls):
        cmd = "ps -ef|grep auto_openai.lm_server.webui_modify.server | awk '{print $2}' | xargs kill -9"
        os.system(cmd)
        os.system(cmd)

    @classmethod
    def kill(cls):
        cls.kill_comfyui()
        cls.kill_vllm()
        cls.kill_maskgct()
        # 自定义的
        cls.kill_funasr()
        cls.kill_embedding()
        cls.kill_llm_transformer()
        cls.kill_webui()
        cls.kill_rerank()