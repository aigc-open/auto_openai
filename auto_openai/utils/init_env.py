import os
from yaml import load
from loguru import logger
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
from pydantic import BaseModel


class GlobalConfig(BaseModel):
    REDIS_CLIENT_CONFIG: dict = {}
    QUEUE_TIMEOUT: int = 600
    INFER_TIMEOUT: int = 100
    MOCK: bool = bool(0)

    #
    VLLM_MODEL_ROOT_PATH: str = "/root/.cache"
    COMFYUI_MODEL_ROOT_PATH: str = "/workspace/ComfyUI/models"
    COMFYUI_ROOT_PATH: str = "/workspace/ComfyUI"
    COMFYUI_INPUTS_DIR: str = "/tmp"
    MASKGCT_MODEL_ROOT_PATH: str = "/root/.cache/MaskGCT-models/"
    MASKGCT_ROOT_PATH: str = "/workspace/MaskGCT"
    FUNASR_ROOT_PATH: str = "/workspace/funasr-webui"
    FUNASR_MODEL_ROOT_PATH: str = "/root/.cache/funasr-models/"
    EMBEDDING_ROOT_PATH: str = "/workspace/embedding-webui"
    EMBEDDING_MODEL_ROOT_PATH: str = "/root/.cache/funasr-models/"
    LLM_TRANSFORMER_ROOT_PATH: str = "/workspace/llm-transformer-server"
    LLM_TRANSFORMER_MODEL_ROOT_PATH: str = "/root/share_models/LLM/"
    # oss
    OSS_CLIENT_CONFIG: dict = {}
    #
    NODE_GPU_TOTAL: str = "0,1"
    USERFULL_TIMES_PER_MODEL: int = 20
    UNUSERFULL_TIMES_PER_MODEL: int = 10
    DEFAULT_MODEL_CONFIG_max_tokens: int = 4096
    #
    GPU_DEVICE_ENV_NAME: str = "TOPS_VISIBLE_DEVICES"
    ONLY_SERVER_TYPES: list = []
    GPU_TYPE: str = "EF-S60"
    LLM_MODELS: list = []
    VISION_MODELS: list = []
    SD_MODELS: list = []
    TTS_MODELS: list = []
    ASR_MODELS: list = []
    EMBEDDING_MODELS: list = []

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
            else:
                logger.warning(f"参数{k}={v}设置失败,因为没有这个属性")
        return env_


global_config = GlobalConfig()
AUTO_OPENAI_CONFIG_PATH = os.environ.get(
    "AUTO_OPENAI_CONFIG_PATH", "./conf/config.yaml")
global_config.init_config(path=AUTO_OPENAI_CONFIG_PATH)
global_config.init_env()
