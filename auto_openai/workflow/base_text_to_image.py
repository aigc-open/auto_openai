from pydantic import BaseModel
from enum import Enum
from auto_openai.utils.openai import gen_random_uuid
from .base import get_model_names

model_names = get_model_names("base_text_to_image")
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


class BaseTextToImage(BaseModel):
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

    def format_json(self):
        if self.model == Models("flux.1-schnell/flux1-schnell.safetensors"):
            return self.flux_format_json()

        return self.normal_format_json()

    def normal_format_json(self):
        return {
            "3": {
                "inputs": {
                    "seed": self.seed,
                    "steps": self.steps,
                    "cfg": self.cfg,
                    "sampler_name": self.sampler_name,
                    "scheduler": self.scheduler,
                    "denoise": self.denoise_strength,
                    "model": [
                        "4",
                        0
                    ],
                    "positive": [
                        "6",
                        0
                    ],
                    "negative": [
                        "7",
                        0
                    ],
                    "latent_image": [
                        "5",
                        0
                    ]
                },
                "class_type": "KSampler",
                "_meta": {
                    "title": "KSampler"
                }
            },
            "4": {
                "inputs": {
                    "ckpt_name": self.model
                },
                "class_type": "CheckpointLoaderSimple",
                "_meta": {
                    "title": "Load Checkpoint"
                }
            },
            "5": {
                "inputs": {
                    "width": self.width,
                    "height": self.height,
                    "batch_size": self.batch_size
                },
                "class_type": "EmptyLatentImage",
                "_meta": {
                    "title": "Empty Latent Image"
                }
            },
            "6": {
                "inputs": {
                    "text": self.prompt,
                    "clip": [
                        "4",
                        1
                    ]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {
                    "title": "CLIP Text Encode (Prompt)"
                }
            },
            "7": {
                "inputs": {
                    "text": self.negative_prompt,
                    "clip": [
                        "4",
                        1
                    ]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {
                    "title": "CLIP Text Encode (Prompt)"
                }
            },
            "8": {
                "inputs": {
                    "samples": [
                        "3",
                        0
                    ],
                    "vae": [
                        "4",
                        2
                    ]
                },
                "class_type": "VAEDecode",
                "_meta": {
                    "title": "VAE Decode"
                }
            },
            "9": {
                "inputs": {
                    "filename_prefix": gen_random_uuid(),
                    "images": [
                        "8",
                        0
                    ]
                },
                "class_type": "SaveImage",
                "_meta": {
                    "title": "Save Image"
                }
            }
        }, {}

    def flux_format_json(self):
        return {
            "5": {
                "inputs": {
                    "width": self.width,
                    "height": self.height,
                    "batch_size": self.batch_size
                },
                "class_type": "EmptyLatentImage",
                "_meta": {
                    "title": "Empty Latent Image"
                }
            },
            "6": {
                "inputs": {
                    "text": self.prompt,
                    "clip": [
                        "11",
                        0
                    ]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {
                    "title": "CLIP Text Encode (Prompt)"
                }
            },
            "8": {
                "inputs": {
                    "samples": [
                        "13",
                        0
                    ],
                    "vae": [
                        "10",
                        0
                    ]
                },
                "class_type": "VAEDecode",
                "_meta": {
                    "title": "VAE Decode"
                }
            },
            "9": {
                "inputs": {
                    "filename_prefix": gen_random_uuid(),
                    "images": [
                        "8",
                        0
                    ]
                },
                "class_type": "SaveImage",
                "_meta": {
                    "title": "Save Image"
                }
            },
            "10": {
                "inputs": {
                    "vae_name": "ae.safetensors"
                },
                "class_type": "VAELoader",
                "_meta": {
                    "title": "Load VAE"
                }
            },
            "11": {
                "inputs": {
                    "clip_name1": "flux_text_encoders/t5xxl_fp16.safetensors",
                    "clip_name2": "flux_text_encoders/clip_l.safetensors",
                    "type": "flux"
                },
                "class_type": "DualCLIPLoader",
                "_meta": {
                    "title": "DualCLIPLoader"
                }
            },
            "12": {
                "inputs": {
                    "unet_name": self.model,
                    "weight_dtype": "default"
                },
                "class_type": "UNETLoader",
                "_meta": {
                    "title": "Load Diffusion Model"
                }
            },
            "13": {
                "inputs": {
                    "noise": [
                        "25",
                        0
                    ],
                    "guider": [
                        "22",
                        0
                    ],
                    "sampler": [
                        "16",
                        0
                    ],
                    "sigmas": [
                        "17",
                        0
                    ],
                    "latent_image": [
                        "5",
                        0
                    ]
                },
                "class_type": "SamplerCustomAdvanced",
                "_meta": {
                    "title": "SamplerCustomAdvanced"
                }
            },
            "16": {
                "inputs": {
                    "sampler_name": self.sampler_name
                },
                "class_type": "KSamplerSelect",
                "_meta": {
                    "title": "KSamplerSelect"
                }
            },
            "17": {
                "inputs": {
                    "scheduler": self.scheduler,
                    "steps": 4,
                    "denoise": 1.0,
                    "model": [
                        "12",
                        0
                    ]
                },
                "class_type": "BasicScheduler",
                "_meta": {
                    "title": "BasicScheduler"
                }
            },
            "22": {
                "inputs": {
                    "model": [
                        "12",
                        0
                    ],
                    "conditioning": [
                        "6",
                        0
                    ]
                },
                "class_type": "BasicGuider",
                "_meta": {
                    "title": "BasicGuider"
                }
            },
            "25": {
                "inputs": {
                    "noise_seed": self.seed,
                },
                "class_type": "RandomNoise",
                "_meta": {
                    "title": "RandomNoise"
                }
            }
        }, {}
