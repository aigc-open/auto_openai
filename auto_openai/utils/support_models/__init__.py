from pydantic import BaseModel
from typing import List, Dict
import os
from auto_openai.utils.init_env import global_config


class GPUConfig(BaseModel):
    need_gpu_count: int


class LMConfig(BaseModel):
    name: str
    server_type: str
    api_type: str
    description: str
    need_gpu_count: int = 1
    gpu_types: Dict[str, GPUConfig] = {}

    def is_available(self) -> bool:
        return False

    def check_resource(self) -> bool:
        if self.gpu_types.get(global_config.GPU_TYPE) and self.gpu_types[global_config.GPU_TYPE].need_gpu_count <= len(global_config.get_gpu_list()):
            if self.server_type in global_config.get_SERVER_TYPES_LIST() or "ALL" in global_config.get_SERVER_TYPES_LIST():
                if self.name in global_config.get_AVAILABLE_MODELS_LIST() or "ALL" in global_config.get_AVAILABLE_MODELS_LIST():
                    return True
        return False


class MultiGPUS(BaseModel):
    model_max_tokens: int
    gpu_types: Dict[str, GPUConfig] = {}


class LLMConfig(LMConfig):
    model_max_tokens: int
    template: str
    stop: List[str]
    server_type: str = "vllm"
    api_type: str = "LLM"

    def extend(self, gpus: List[MultiGPUS]):
        _model_configs_ = []
        if self.gpu_types:
            _model_configs_.append(self)
        for gpu in gpus:
            tmp_config = self.copy()
            tmp_config.model_max_tokens = gpu.model_max_tokens
            tmp_config.gpu_types = gpu.gpu_types

            tmp_config.name = f"{tmp_config.name}:{int(gpu.model_max_tokens/1024)}k"
            _model_configs_.append(tmp_config)
        return _model_configs_

    def is_available(self) -> bool:
        return os.path.exists(os.path.join(global_config.VLLM_MODEL_ROOT_PATH, self.name.split(":")[0])) and self.check_resource()


class CoderLLMConfig(LLMConfig):

    def combine_prompt(self, prompt: str, suffix: str = "") -> str:
        return prompt


class QwenCoderLLMConfig(LLMConfig):
    def combine_prompt(self, prompt: str, suffix: str = "") -> str:
        if suffix:
            return f"""<|fim_prefix|>{prompt}<|fim_suffix|>{suffix}<|fim_middle|>"""
        else:
            return prompt


class DeepseekCoderLLMConfig(LLMConfig):
    def combine_prompt(self, prompt: str, suffix: str = "") -> str:
        if suffix:
            return f"""<｜fim▁begin｜>{prompt}<｜fim▁hole｜>{suffix}<｜fim▁end｜>"""
        else:
            return prompt


class VisionConfig(LLMConfig):
    pass


class SDConfig(LMConfig):
    def is_available(self) -> bool:
        return os.path.exists(os.path.join(global_config.WEBUI_MODEL_ROOT_PATH, "Stable-diffusion", self.name.replace("SD15MultiControlnetGenerateImage/", ""))) and self.check_resource()


class TTSConfig(LMConfig):
    def is_available(self) -> bool:
        if self.name == "maskgct-tts-clone":
            return os.path.exists(os.path.join(global_config.MASKGCT_MODEL_ROOT_PATH, "MaskGCT")) and self.check_resource()
        return False


class ASRConfig(LMConfig):
    def is_available(self) -> bool:
        if self.name == "funasr":
            return os.path.exists(os.path.join(global_config.FUNASR_MODEL_ROOT_PATH, "punc_ct-transformer_zh-cn-common-vocab272727-pytorch")) and self.check_resource()
        return False


class EmbeddingConfig(LMConfig):
    def is_available(self) -> bool:
        return os.path.exists(os.path.join(global_config.EMBEDDING_MODEL_ROOT_PATH, self.name)) and self.check_resource()


class RerankConfig(LMConfig):
    def is_available(self) -> bool:
        return os.path.exists(os.path.join(global_config.RERANK_MODEL_ROOT_PATH, self.name)) and self.check_resource()


class VideoConfig(LMConfig):
    def is_available(self) -> bool:
        return os.path.exists(os.path.join(global_config.WEBUI_MODEL_ROOT_PATH, self.name)) and self.check_resource()


class ModelsConfig:
    def __init__(self):
        self._models = []

    def add(self, model: LMConfig):
        self._models.append(model)

    def extend(self, models: List[LMConfig]):
        self._models.extend(models)

    def list(self) -> List[LMConfig]:
        return self._models

    def get(self, name: str) -> LMConfig:
        for model in self._models:
            if model.name == name:
                return model
        raise ValueError(f"Model {name} not found")


system_models_config = ModelsConfig()
