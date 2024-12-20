import os
from yaml import load
from loguru import logger
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
from pydantic import BaseModel


class LMConfig(BaseModel):
    name: str
    server_type: str
    api_type: str
    description: str
    need_gpu_count: int
    gpu_types: dict


class LLMConfig(LMConfig):
    model_max_tokens: int
    template: str
    stop: list


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
    ONLY_SERVER_TYPES: list = []
    GPU_TYPE: str = "EF-S60"
    MODELS: list = []
    AVAILABLE_MODELS: list = ["ALL"]
    LM_SERVER_BASE_PORT: int = 30000
    LABEL:str="auto_openai_scheduler"

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
        data = {}
        for m in self.MODELS:
            if "api_type" in m:
                data[m["api_type"]] = data.get(m["api_type"], []) + [m]
        return data

    def validate_models(self):
        for m in self.MODELS:
            if LMConfig(**m).api_type in ["LLM", "VLLM"]:
                LLMConfig(**m)


global_config = GlobalConfig()
AUTO_OPENAI_CONFIG_PATH = os.environ.get(
    "AUTO_OPENAI_CONFIG_PATH", "./conf/config.yaml")
global_config.init_config(path=AUTO_OPENAI_CONFIG_PATH)
global_config.init_env()
global_config.validate_models()
