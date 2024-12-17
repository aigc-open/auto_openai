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


_models_: List[LLMConfig | VisionConfig | SDConfig | TTSConfig |
               ASRConfig | EmbeddingConfig | RerankConfig | VideoConfig] = list()
#####################################################################################
_models_.append(LLMConfig(
    name="Qwen2.5-72B-Instruct",
    server_type="vllm",
    api_type="LLM",
    model_max_tokens=10240,
    description="Qwen2.5-72B-Instruct",
    need_gpu_count=1,
    template="template_qwen.jinja",
    stop=["<|im_start", "<|", "<|im_end|>", "<|endoftext|>"],
    gpu_types={
        "NV-A100": GPUConfig(need_gpu_count=1),
        "NV-4090": GPUConfig(need_gpu_count=1),
        "EF-S60": GPUConfig(need_gpu_count=4)
    }
))
_models_.append(LLMConfig(
    name="Qwen2.5-32B-Instruct-GPTQ-Int4",
    server_type="vllm",
    api_type="LLM",
    model_max_tokens=32768,
    description="Qwen2.5-32B-Instruct-GPTQ-Int4",
    need_gpu_count=1,
    template="template_qwen.jinja",
    stop=["<|im_start", "<|", "<|im_end|>", "<|endoftext|>"],
    quantization="gptq",
    gpu_types={
        "NV-A100": GPUConfig(need_gpu_count=1),
        "NV-4090": GPUConfig(need_gpu_count=1),
        "EF-S60": GPUConfig(need_gpu_count=1)
    }
))
_models_.append(LLMConfig(
    name="Qwen2.5-7B-Instruct",
    server_type="vllm",
    api_type="LLM",
    model_max_tokens=32768,
    description="Qwen2.5-7B-Instruct",
    need_gpu_count=1,
    template="template_qwen.jinja",
    stop=["<|im_start", "<|", "<|im_end|>", "<|endoftext|>"],
    gpu_types={
        "NV-A100": GPUConfig(need_gpu_count=1),
        "NV-4090": GPUConfig(need_gpu_count=2),
        "EF-S60": GPUConfig(need_gpu_count=1)
    }
))
_models_.append(LLMConfig(
    name="Qwen2.5-Coder-7B-Instruct",
    server_type="vllm",
    api_type="LLM",
    model_max_tokens=32768,
    description="Qwen2.5-Coder-7B-Instruct",
    need_gpu_count=1,
    template="template_qwen.jinja",
    stop=["<|im_start", "<|", "<|im_end|>", "<|endoftext|>"],
    gpu_types={
        "NV-A100": GPUConfig(need_gpu_count=1),
        "NV-4090": GPUConfig(need_gpu_count=2),
        "EF-S60": GPUConfig(need_gpu_count=1)
    }
))
_models_.append(LLMConfig(
    name="codegeex4-all-9b",
    server_type="vllm",
    api_type="LLM",
    model_max_tokens=131072,
    description="codegeex4-all-9b",
    need_gpu_count=1,
    template="template_glm4.jinja",
    stop=["<|user|>", "<|assistant|>"],
    gpu_types={
        "NV-A100": GPUConfig(need_gpu_count=1),
        "NV-4090": GPUConfig(need_gpu_count=2),
        "EF-S60": GPUConfig(need_gpu_count=1)
    }
))
_models_.append(LLMConfig(
    name="glm-4-9b-chat",
    server_type="vllm",
    api_type="LLM",
    model_max_tokens=131072,
    description="glm-4-9b-chat",
    need_gpu_count=1,
    template="template_glm4.jinja",
    stop=["<|user|>", "<|assistant|>"],
    gpu_types={
        "NV-A100": GPUConfig(need_gpu_count=1),
        "NV-4090": GPUConfig(need_gpu_count=2),
        "EF-S60": GPUConfig(need_gpu_count=1)
    }
))
#####################################################################################
_models_.append(VisionConfig(
    name="glm-4v-9b",
    server_type="llm-transformer-server",
    api_type="VLLM",
    model_max_tokens=8192,
    description="glm-4v-9b",
    need_gpu_count=1,
    template="template_glm4.jinja",
    stop=[],
    gpu_types={
        "NV-A100": GPUConfig(need_gpu_count=1),
        "NV-4090": GPUConfig(need_gpu_count=1),
        "EF-S60": GPUConfig(need_gpu_count=1)
    }
))
#####################################################################################
_models_.append(SDConfig(
    name="majicmixRealistic_v7.safetensors/majicmixRealistic_v7.safetensors",
    server_type="comfyui",
    api_type="SolutionBaseGenerateImage",
    description="majicmixRealistic_betterV6",
    need_gpu_count=1,
    gpu_types={
        "NV-A100": GPUConfig(need_gpu_count=1),
        "NV-4090": GPUConfig(need_gpu_count=1),
        "EF-S60": GPUConfig(need_gpu_count=1)
    }
))
_models_.append(SDConfig(
    name="SD15MultiControlnetGenerateImage/majicmixRealistic_v7.safetensors/majicmixRealistic_v7.safetensors",
    server_type="webui",
    api_type="SD15MultiControlnetGenerateImage",
    description="majicmixRealistic_v7",
    need_gpu_count=1,
    gpu_types={
        "NV-A100": GPUConfig(need_gpu_count=1),
        "NV-4090": GPUConfig(need_gpu_count=1),
        "EF-S60": GPUConfig(need_gpu_count=1)
    }
))
#####################################################################################
_models_.append(ASRConfig(
    name="funasr",
    server_type="funasr",
    api_type="ASR",
    description="asr",
    need_gpu_count=1,
    gpu_types={
        "NV-A100": GPUConfig(need_gpu_count=1),
        "NV-4090": GPUConfig(need_gpu_count=1),
        "EF-S60": GPUConfig(need_gpu_count=1),
        "CPU": GPUConfig(need_gpu_count=1)
    }
))
#####################################################################################
_models_.append(ASRConfig(
    name="maskgct-tts-clone",
    server_type="maskgct",
    api_type="TTS",
    description="maskgct-tts-clone",
    need_gpu_count=1,
    gpu_types={
        "NV-A100": GPUConfig(need_gpu_count=1),
        "NV-4090": GPUConfig(need_gpu_count=1),
        "EF-S60": GPUConfig(need_gpu_count=1),
    }
))
#####################################################################################
_models_.append(EmbeddingConfig(
    name="bge-base-zh-v1.5",
    server_type="embedding",
    api_type="Embedding",
    description="bge-base-zh-v1.5",
    need_gpu_count=1,
    gpu_types={
        "NV-A100": GPUConfig(need_gpu_count=1),
        "NV-4090": GPUConfig(need_gpu_count=1),
        "EF-S60": GPUConfig(need_gpu_count=1),
        "CPU": GPUConfig(need_gpu_count=1)
    }
))
_models_.append(EmbeddingConfig(
    name="bge-m3",
    server_type="embedding",
    api_type="Embedding",
    description="bge-m3",
    need_gpu_count=1,
    gpu_types={
        "NV-A100": GPUConfig(need_gpu_count=1),
        "NV-4090": GPUConfig(need_gpu_count=1),
        "EF-S60": GPUConfig(need_gpu_count=1),
        "CPU": GPUConfig(need_gpu_count=1)
    }
))
#####################################################################################
_models_.append(RerankConfig(
    name="bge-reranker-base",
    server_type="rerank",
    api_type="Rerank",
    description="bge-rerank",
    need_gpu_count=1,
    gpu_types={
        "NV-A100": GPUConfig(need_gpu_count=1),
        "NV-4090": GPUConfig(need_gpu_count=1),
        "EF-S60": GPUConfig(need_gpu_count=1),
        "CPU": GPUConfig(need_gpu_count=1)
    }
))

_models_.append(RerankConfig(
    name="bge-reranker-v2-m3",
    server_type="rerank",
    api_type="Rerank",
    description="bge-reranker-v2-m3",
    need_gpu_count=1,
    gpu_types={
        "NV-A100": GPUConfig(need_gpu_count=1),
        "NV-4090": GPUConfig(need_gpu_count=1),
        "EF-S60": GPUConfig(need_gpu_count=1),
        "CPU": GPUConfig(need_gpu_count=1)
    }
))
#####################################################################################
_models_.append(VideoConfig(
    name="CogVideo/CogVideoX-5b",
    server_type="diffusers-video",
    api_type="Video",
    description="CogVideo/CogVideoX-5b",
    need_gpu_count=1,
    gpu_types={
        "NV-A100": GPUConfig(need_gpu_count=1),
        "NV-4090": GPUConfig(need_gpu_count=1),
        "EF-S60": GPUConfig(need_gpu_count=1),
    }
))


if __name__ == "__main__":
    for m in _models_:
        print(m.json())
