from pydantic import BaseModel
from enum import Enum
import os
from .base import UrlParser
from .base import get_model_names

model_names = get_model_names("base_image_to_image")

# 动态创建枚举类
Models = Enum('Models', {name: name for name in model_names})


class SamplerName(str, Enum):
    euler = "euler"
    euler_cfg_pp = "euler_cfg_pp"
    euler_ancestral = "euler_ancestral"
    euler_ancestral_cfg_pp = "euler_ancestral_cfg_pp"
    heun = "heun"
    heunpp2 = "heunpp2"
    dpm_2 = "dpm_2"
    dpm_2_ancestral = "dpm_2_ancestral"
    lms = "lms"
    dpm_fast = "dpm_fast"
    dpm_adaptive = "dpm_adaptive"
    dpmpp_2s_ancestral = "dpmpp_2s_ancestral"
    dpmpp_sde = "dpmpp_sde"
    dpmpp_sde_gpu = "dpmpp_sde_gpu"
    dpmpp_2m = "dpmpp_2m"
    dpmpp_2m_sde = "dpmpp_2m_sde"
    dpmpp_2m_sde_gpu = "dpmpp_2m_sde_gpu"
    dpmpp_3m_sde = "dpmpp_3m_sde"
    dpmpp_3m_sde_gpu = "dpmpp_3m_sde_gpu"
    ddpm = "ddpm"
    lcm = "lcm"
    ipndm = "ipndm"
    ipndm_v = "ipndm_v"
    deis = "deis"
    ddim = "ddim"
    uni_pc = "uni_pc"
    uni_pc_bh2 = "uni_pc_bh2"


class Scheduler(str, Enum):
    normal = "normal"
    karras = "karras"
    exponential = "exponential"
    sgm_uniform = "sgm_uniform"
    simple = "simple"
    ddim_uniform = "ddim_uniform"
    beta = "beta"


class BaseImageToImage(BaseModel):
    model: Models = Models("sd1.5/majicmixRealistic_betterV6.safetensors")
    seed: int = 0
    steps: int = 20
    batch_size: int = 1
    width: int = 512
    height: int = 512
    sampler_name: SamplerName = SamplerName("euler")  # 使用枚举类型
    cfg: int = 8
    denoise_strength: float = 0.75
    scheduler: Scheduler = Scheduler("normal")  # 使用枚举类型
    prompt: str = "beautiful scenery nature glass bottle landscape, purple galaxy bottle"
    negative_prompt: str = "text, watermark"
    image_url: str

    def format_json(self):
        return self.normal_format_json()

    def normal_format_json(self):
        UrlParser(url=self.image_url).generate_random_local_file_name()
        filename = UrlParser(
            url=self.image_url).generate_random_local_file_name()

        return {
            "177": {
                "inputs": {
                    "ckpt_name": self.model
                },
                "class_type": "CheckpointLoaderSimple",
                "_meta": {
                    "title": "Load Checkpoint"
                }
            },
            "179": {
                "inputs": {
                    "text": self.prompt,
                    "clip": [
                        "177",
                        1
                    ]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {
                    "title": "CLIP Text Encode (Prompt)"
                }
            },
            "180": {
                "inputs": {
                    "text": self.negative_prompt,
                    "clip": [
                        "177",
                        1
                    ]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {
                    "title": "CLIP Text Encode (Prompt)"
                }
            },
            "181": {
                "inputs": {
                    "images": [
                        "186",
                        0
                    ]
                },
                "class_type": "PreviewImage",
                "_meta": {
                    "title": "Preview Image"
                }
            },
            "182": {
                "inputs": {
                    "image": filename,
                    "upload": "image"
                },
                "class_type": "LoadImage",
                "_meta": {
                    "title": "Load Image"
                }
            },
            "185": {
                "inputs": {
                    "seed": self.seed,
                    "steps": self.steps,
                    "cfg": self.cfg,
                    "sampler_name": self.sampler_name,
                    "scheduler": self.scheduler,
                    "denoise": self.denoise_strength,
                    "model": [
                        "177",
                        0
                    ],
                    "positive": [
                        "179",
                        0
                    ],
                    "negative": [
                        "180",
                        0
                    ],
                    "latent_image": [
                        "190",
                        0
                    ]
                },
                "class_type": "KSampler",
                "_meta": {
                    "title": "KSampler"
                }
            },
            "186": {
                "inputs": {
                    "samples": [
                        "185",
                        0
                    ],
                    "vae": [
                        "177",
                        2
                    ]
                },
                "class_type": "VAEDecode",
                "_meta": {
                    "title": "VAE Decode"
                }
            },
            "190": {
                "inputs": {
                    "pixels": [
                        "182",
                        0
                    ],
                    "vae": [
                        "177",
                        2
                    ]
                },
                "class_type": "VAEEncode",
                "_meta": {
                    "title": "VAE Encode"
                }
            }
        }, {filename: self.image_url}
