from pydantic import BaseModel
from typing import List, Dict


class GPUConfig(BaseModel):
    need_gpu_count: int


class LMConfig(BaseModel):
    name: str
    server_type: str
    api_type: str
    description: str
    need_gpu_count: int
    gpu_types: Dict[str, GPUConfig]


class LLMConfig(LMConfig):
    model_max_tokens: int
    template: str
    stop: List[str]
    server_type: str = "vllm"
    api_type: str = "LLM"


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


class VisionConfig(LMConfig):
    model_max_tokens: int
    template: str
    stop: List[str]


class SDConfig(LMConfig):
    pass


class TTSConfig(LMConfig):
    pass


class ASRConfig(LMConfig):
    pass


class EmbeddingConfig(LMConfig):
    pass


class RerankConfig(LMConfig):
    pass


class VideoConfig(LMConfig):
    pass


class ModelsConfig:
    def __init__(self):
        self._models = []

    def add(self, model: LMConfig):
        self._models.append(model)

    def list(self) -> List[LMConfig]:
        return self._models

    def get(self, name: str) -> LMConfig:
        for model in self._models:
            if model.name == name:
                return model
        raise ValueError(f"Model {name} not found")


system_models_config = ModelsConfig()
