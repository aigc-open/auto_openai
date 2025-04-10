from pydantic import BaseModel
from enum import Enum
import os
from .base import UrlParser, WorkflowFormat
from auto_openai.utils.openai import SolutionBaseGenerateImageRequest
from uuid_extensions import uuid7, uuid7str


def gen_random_uuid() -> str:
    return str(uuid7(as_type="int"))


class SolutionBaseGenerateImage(SolutionBaseGenerateImageRequest, WorkflowFormat):

    def format_json(self):
        if "flux" in self.model:
            return self.flux_format_json()
        elif "majicmixRealistic_v7" in self.model:
            return self.majicmixRealistic_v7_format_json()
        elif "Kolors" in self.model:
            return self.kolors_format_json()
        raise NotImplementedError("Not support model")

    def normal_format_json(self, ckpt_name: str):
        sampler_name = "euler"
        scheduler_name = "normal"
        if self.image_url:
            filename = UrlParser(
                url=self.image_url).generate_random_local_file_name()

            return {
                "177": {
                    "inputs": {
                        "ckpt_name": ckpt_name
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
                        "sampler_name": sampler_name,
                        "scheduler": scheduler_name,
                        "denoise": self.denoising_strength,
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

        else:
            return {
                "3": {
                    "inputs": {
                        "seed": self.seed,
                        "steps": self.steps,
                        "cfg": self.cfg,
                        "sampler_name": sampler_name,
                        "scheduler": scheduler_name,
                        "denoise": self.denoising_strength,
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

    def majicmixRealistic_v7_format_json(self):
        return self.normal_format_json(ckpt_name="majicmixRealistic_v7.safetensors/majicmixRealistic_v7.safetensors")

    def flux_format_json(self):
        sampler_name = "euler"
        scheduler_name = "normal"
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
                    "sampler_name": sampler_name,
                },
                "class_type": "KSamplerSelect",
                "_meta": {
                    "title": "KSamplerSelect"
                }
            },
            "17": {
                "inputs": {
                    "scheduler": scheduler_name,
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

    def kolors_format_json(self):
        return {
            "3": {
                "inputs": {
                    "images": [
                        "10",
                        0
                    ]
                },
                "class_type": "PreviewImage",
                "_meta": {
                    "title": "Preview Image"
                }
            },
            "10": {
                "inputs": {
                    "samples": [
                        "14",
                        0
                    ],
                    "vae": [
                        "11",
                        0
                    ]
                },
                "class_type": "VAEDecode",
                "_meta": {
                    "title": "VAE Decode"
                }
            },
            "11": {
                "inputs": {
                    "vae_name": "sdxl.vae.safetensors"
                },
                "class_type": "VAELoader",
                "_meta": {
                    "title": "Load VAE"
                }
            },
            "12": {
                "inputs": {
                    "prompt": self.prompt,
                    "negative_prompt": self.negative_prompt,
                    "num_images_per_prompt": self.batch_size,
                    "chatglm3_model": [
                        "45",
                        0
                    ]
                },
                "class_type": "KolorsTextEncode",
                "_meta": {
                    "title": "Kolors Text Encode"
                }
            },
            "14": {
                "inputs": {
                    "width": self.width,
                    "height": self.height,
                    "seed": self.seed,
                    "steps": self.steps,
                    "cfg": self.cfg,
                    "scheduler": "EulerDiscreteScheduler",
                    "denoise_strength": 1.0,
                    "kolors_model": [
                        "16",
                        0
                    ],
                    "kolors_embeds": [
                        "12",
                        0
                    ]
                },
                "class_type": "KolorsSampler",
                "_meta": {
                    "title": "Kolors Sampler"
                }
            },
            "16": {
                "inputs": {
                    "model": "Kwai-Kolors/Kolors",
                    "precision": "fp16"
                },
                "class_type": "DownloadAndLoadKolorsModel",
                "_meta": {
                    "title": "(Down)load Kolors Model"
                }
            },
            "45": {
                "inputs": {
                    "chatglm3_checkpoint": "chatglm3-fp16.safetensors"
                },
                "class_type": "LoadChatGLM3",
                "_meta": {
                    "title": "Load ChatGLM3 Model"
                }
            }
        }, {}
