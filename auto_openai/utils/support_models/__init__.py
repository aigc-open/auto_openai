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
    model_url: str = ""

    def is_available(self) -> bool:
        return False

    def check_resource(self) -> bool:
        if self.gpu_types.get(global_config.GPU_TYPE) and self.gpu_types[global_config.GPU_TYPE].need_gpu_count <= len(global_config.get_gpu_list()):
            if self.server_type in global_config.get_SERVER_TYPES_LIST() or "ALL" in global_config.get_SERVER_TYPES_LIST():
                if self.name in global_config.get_AVAILABLE_MODELS_LIST() or "ALL" in global_config.get_AVAILABLE_MODELS_LIST():
                    return True
        return False

    def download_shell(self):
        return f"echo {self.name}\n"


class MultiGPUS(BaseModel):
    model_max_tokens: int
    gpu_types: Dict[str, GPUConfig] = {}


class LLMConfig(LMConfig):
    model_max_tokens: int
    template: str = ""
    stop: List[str]
    server_type: str = "vllm"
    api_type: str = "LLM"
    enforce_eager: bool = True
    num_scheduler_steps: int = 1
    reasoning_parser: str = ""

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

    def download_shell(self):
        if not self.model_url:
            return ""
        return f"""
cd $LLM_path && git lfs install && git clone {self.model_url}
"""


class HTTPConfig(LMConfig):
    model_max_tokens: int
    template: str = ""
    stop: List[str]
    server_type: str = "http"
    api_type: str = "LLM"
    api_key: str = ""
    base_url: str = ""
    model_name: str = ""

    def is_available(self) -> bool:
        if self.gpu_types.get(global_config.GPU_TYPE):
            return True
        return False

    def download_shell(self):
        if not self.model_url:
            return ""
        return f"""
cd $LLM_path && git lfs install && git clone {self.model_url}
"""


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
        if self.name == "SolutionBaseGenerateImage/Kolors":
            return os.path.exists(os.path.join(global_config.WEBUI_MODEL_ROOT_PATH, "diffusers", "Kolors")) and self.check_resource()
        elif self.name == "SolutionBaseGenerateImage/majicmixRealistic_v7":
            return os.path.exists(os.path.join(global_config.WEBUI_MODEL_ROOT_PATH, "Stable-diffusion", "majicmixRealistic_v7.safetensors/majicmixRealistic_v7.safetensors")) and self.check_resource()
        return os.path.exists(os.path.join(global_config.WEBUI_MODEL_ROOT_PATH, "Stable-diffusion", self.name.replace("SD15MultiControlnetGenerateImage/", ""))) and self.check_resource()

    def download_shell(self):
        if not self.model_url:
            return ""
        if self.name == "SolutionBaseGenerateImage/Kolors":
            return f"""
mkdir -p $webui_path/diffusers/
cd $webui_path/diffusers && git lfs install && git clone {self.model_url}
mkdir -p $webui_path/LLM/
cd $webui_path/LLM && wget -nc https://www.modelscope.cn/models/Kijai/ChatGLM3-safetensors/resolve/master/chatglm3-fp16.safetensors
mkdir -p $webui_path/VAE/
cd $webui_path/VAE && wget -nc https://www.modelscope.cn/models/AI-ModelScope/sdxl-vae-fp16-fix/resolve/master/sdxl.vae.safetensors
"""
        return f"""
mkdir -p $webui_path/Stable-diffusion
cd $webui_path/Stable-diffusion && git lfs install && git clone {self.model_url}
"""


class TTSConfig(LMConfig):
    def is_available(self) -> bool:
        if self.name == "maskgct-tts-clone":
            return os.path.exists(os.path.join(global_config.MASKGCT_MODEL_ROOT_PATH, "MaskGCT")) and self.check_resource()
        return False

    def download_shell(self):
        if not self.model_url:
            return ""
        if self.name == "maskgct-tts-clone":
            return f"""
cd $MaskGCT_path && git lfs install && git clone https://www.modelscope.cn/AI-ModelScope/MaskGCT.git
cd $MaskGCT_path && git lfs install && git clone https://www.modelscope.cn/AI-ModelScope/w2v-bert-2.0.git
cd $MaskGCT_path && git lfs install && git clone https://www.modelscope.cn/iic/Whisper-large-v3-turbo.git
"""
        else:
            return ""


class ASRConfig(LMConfig):
    def is_available(self) -> bool:
        if self.name == "funasr":
            return os.path.exists(os.path.join(global_config.FUNASR_MODEL_ROOT_PATH, "punc_ct-transformer_zh-cn-common-vocab272727-pytorch")) and self.check_resource()
        return False

    def download_shell(self):
        if self.name == "funasr":
            return f"""
cd $funasr_path && git lfs install && git clone https://www.modelscope.cn/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch.git
cd $funasr_path && git lfs install && git clone https://www.modelscope.cn/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch.git
cd $funasr_path && git lfs install && git clone https://www.modelscope.cn/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch.git
"""
        else:
            return ""


class EmbeddingConfig(LMConfig):
    def is_available(self) -> bool:
        return os.path.exists(os.path.join(global_config.EMBEDDING_MODEL_ROOT_PATH, self.name)) and self.check_resource()

    def download_shell(self):
        if not self.model_url:
            return ""
        return f"""
cd $Embedding_path && git lfs install && git clone {self.model_url}
"""


class RerankConfig(LMConfig):
    def is_available(self) -> bool:
        return os.path.exists(os.path.join(global_config.RERANK_MODEL_ROOT_PATH, self.name)) and self.check_resource()

    def download_shell(self):
        if not self.model_url:
            return ""
        return f"""
cd $rerank_path && git lfs install && git clone {self.model_url}"""


class VideoConfig(LMConfig):
    def is_available(self) -> bool:
        return os.path.exists(os.path.join(global_config.WEBUI_MODEL_ROOT_PATH, self.name)) and self.check_resource()

    def download_shell(self):
        if not self.model_url:
            return ""
        if "CogVideo" in self.name:
            return f"""
mkdir -p $webui_path/CogVideo 
cd $webui_path/CogVideo && git lfs install && git clone {self.model_url}"""
        else:
            return ""


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

    def generate_download_shell(self, models: list):
        shell_ = """
# 存放零时数据的目录,图片等
mkdir -p /root/share_models/tmp

# Embedding
Embedding_path=/root/share_models/Embedding-models
mkdir -p $Embedding_path

# LLM
LLM_path=/root/share_models/LLM
mkdir -p $LLM_path

# maskgct
MaskGCT_path="/root/share_models/MaskGCT-models/"
mkdir -p $MaskGCT_path

# funasr
funasr_path="/root/share_models/funasr-models/"
mkdir -p $funasr_path

# rerank
rerank_path="/root/share_models/Rerank-models/"
mkdir -p $rerank_path

# webui && comfyui
webui_path="/root/share_models/webui-models/"
mkdir -p $webui_path

# 基础绘图模型
# cd $webui_path && git lfs install && git clone https://www.modelscope.cn/chineking/adetailer.git
# cd $webui_path && git lfs install && git clone https://www.modelscope.cn/licyks/controlnet_v1.1_annotator.git
# # for webui controlnet 处理器, 挂载目录: /workspace/stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/downloads
# cd $webui_path && git lfs install && git clone https://www.modelscope.cn/jackle/comfyui_controlnet_aux_ckpts.git
# # for ComfyUI controlnet 处理器，挂载目录: /workspace/ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts
# cd $webui_path && git lfs install && git clone https://www.modelscope.cn/shareAI/lllyasviel-ControlNet-v1-1.git ControlNet 
# # webui/comfyui 公用同一个controlnet
# cd $webui_path && git lfs install && git clone https://www.modelscope.cn/AI-ModelScope/clip-vit-large-patch14.git
# # webui 使用的clip,必须安装

"""
        shell_arr = set()
        for model in self._models:
            if model.name in models:
                shell_arr.add(model.download_shell().strip())

        shell_arr = list(shell_arr)
        shell_arr.sort()

        return shell_ + "\n".join(shell_arr)


system_models_config = ModelsConfig()
