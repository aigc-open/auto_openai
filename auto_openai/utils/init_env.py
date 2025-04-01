import os
from yaml import load
from loguru import logger
import json
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
from pydantic import BaseModel


class GlobalConfig(BaseModel):
    REDIS_CLIENT_CONFIG: dict = {}
    QUEUE_TIMEOUT: int = 600
    INFER_TIMEOUT: int = 600
    MOCK: bool = bool(0)

    #
    COMFYUI_INPUTS_DIR: str = "/tmp"
    VLLM_MODEL_ROOT_PATH: str = "/root/share_models/LLM/"
    COMFYUI_MODEL_ROOT_PATH: str = "/root/share_models/webui-models/"
    MASKGCT_MODEL_ROOT_PATH: str = "/root/share_models/MaskGCT-models/"
    FUNASR_MODEL_ROOT_PATH: str = "/root/share_models/funasr-models/"
    EMBEDDING_MODEL_ROOT_PATH: str = "/root/share_models/Embedding-models/"
    LLM_TRANSFORMER_MODEL_ROOT_PATH: str = "/root/share_models/LLM/"
    WEBUI_MODEL_ROOT_PATH: str = "/root/share_models/webui-models"
    RERANK_MODEL_ROOT_PATH: str = "/root/share_models/Rerank-models/"
    DIFFUSERS_MODEL_ROOT_PATH: str = "/root/share_models/webui-models"

    # image
    IMAGE_BASE_PATH: str = "harbor.uat.enflame.cc/library/enflame.cn"

    # oss
    OSS_CLIENT_CONFIG: dict = {}
    #
    NODE_GPU_TOTAL: str = "0,1"
    USERFULL_TIMES_PER_MODEL: int = 20
    UNUSERFULL_TIMES_PER_MODEL: int = 10
    DEFAULT_MODEL_CONFIG_max_tokens: int = 4096
    #
    AVAILABLE_SERVER_TYPES: str = "ALL"
    GPU_TYPE: str = "EF-S60"
    MODELS: list = []
    AVAILABLE_MODELS: str = "ALL"
    LM_SERVER_BASE_PORT: int = 30000
    LABEL: str = "auto_openai_scheduler"
    CUSTOM_MODLES: list = []

    def get_AVAILABLE_MODELS_LIST(self):
        return self.AVAILABLE_MODELS.split(",")

    def get_SERVER_TYPES_LIST(self):
        return self.AVAILABLE_SERVER_TYPES.split(",")

    def parse_config(self, path=".conf/config.yaml"):
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            return load(f, Loader=Loader)

    def init_config(self, path="./conf/config.yaml"):
        # parse the config.yaml
        CONFIG = self.parse_config(path)
        if CONFIG:
            for k, v in CONFIG.items():
                if hasattr(self, k):
                    logger.info(f"修改配置: {k}={v}, type={type(v)}")
                    setattr(self, k, v)
                else:
                    logger.warning(f"参数{k}={v}设置失败,因为没有这个属性")

    def init_env(self):
        env_ = {}
        for k, v in os.environ.items():
            if hasattr(self, k):
                if type(getattr(self, k)) == bool:
                    if v.lower() == "true":
                        setattr(self, k, True)
                    elif v.lower() == "false":
                        setattr(self, k, False)
                    else:
                        setattr(self, k, False)
                else:
                    setattr(self, k, type(getattr(self, k))(v))
                logger.info(f"修改配置: {k}={v}, type={type(v)}")
            else:
                logger.warning(f"参数{k}={v}设置失败,因为没有这个属性")
        return env_

    def get_MODELS_MAPS(self):
        from auto_openai.utils.depends import get_models_config_list
        data = {}
        for m in get_models_config_list():
            if "api_type" in m:
                data[m["api_type"]] = data.get(m["api_type"], []) + [m]
        return data

    def get_gpu_list(self):
        node_gpu_total: list = list([int(x.strip())
                                     for x in self.NODE_GPU_TOTAL.split(',')])
        return node_gpu_total

    def get_continue_config(self, path="./conf/continueconfig.json"):
        continue_config = {
            "models": [
                {
                    "title": "Qwen2.5-Coder-32B-Instruct-GPTQ-Int4:32k",
                    "provider": "openrouter",
                    "model": "Qwen2.5-Coder-32B-Instruct-GPTQ-Int4:32k",
                    "apiBase": "https://auto-openai.cpolar.cn/openai/v1",
                    "apiKey": "..."
                }
            ],
            "tabAutocompleteModel": {
                "title": "Qwen2.5-Coder-32B-Instruct-GPTQ-Int4:32k",
                "provider": "openrouter",
                "model": "Qwen2.5-Coder-32B-Instruct-GPTQ-Int4:32k",
                "apiBase": "https://auto-openai.cpolar.cn/openai/v1",
                "apiKey": "..."
            },
            "embeddingsProvider": {
                "provider": "transformers.js"
            },
            "contextProviders": [
                {
                    "name": "code",
                    "params": {}
                },
                {
                    "name": "docs",
                    "params": {}
                },
                {
                    "name": "diff",
                    "params": {}
                },
                {
                    "name": "terminal",
                    "params": {}
                },
                {
                    "name": "problems",
                    "params": {}
                },
                {
                    "name": "folder",
                    "params": {}
                },
                {
                    "name": "codebase",
                    "params": {}
                },
                {
                    "name": "commit",
                    "params": {
                        "Depth": 50,
                        "LastXCommitsDepth": 10
                    }
                },
                {
                    "name": "url",
                    "params": {}
                }
            ],
            "slashCommands": [
                {
                    "name": "share",
                    "description": "Export the current chat session to markdown"
                },
                {
                    "name": "cmd",
                    "description": "Generate a shell command"
                },
                {
                    "name": "commit",
                    "description": "Generate a git commit message"
                }
            ],
            "completionOptions": {
                "maxTokens": 20480,
                "temperature": 0.7
            }
        }
        if not os.path.exists(path):
            return continue_config
        with open(path, "r") as f:
            try:
                return json.load(f)
            except Exception as e:
                return continue_config


global_config = GlobalConfig()
AUTO_OPENAI_CONFIG_PATH = os.environ.get(
    "AUTO_OPENAI_CONFIG_PATH", "./conf/config.yaml")
global_config.init_config(path=AUTO_OPENAI_CONFIG_PATH)
global_config.init_env()
