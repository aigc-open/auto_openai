from auto_openai.utils.support_models import *


########################################## LLM ###########################################

system_models_config.add(LLMConfig(name="Qwen2.5-72B-Instruct",
                                   server_type="vllm",
                                   api_type="LLM",
                                   model_max_tokens=10240,
                                   description="Qwen2.5-72B-Instruct",
                                   need_gpu_count=1,
                                   template="template_qwen.jinja",
                                   stop=["<|im_start", "<|",
                                         "<|im_end|>", "<|endoftext|>"],
                                   gpu_types={
                                       "NV-A100": GPUConfig(need_gpu_count=1),
                                       "NV-4090": GPUConfig(need_gpu_count=1),
                                       "EF-S60": GPUConfig(need_gpu_count=4)
                                   }
                                   ))
system_models_config.add(LLMConfig(name="Qwen2.5-32B-Instruct-GPTQ-Int4",
                                   server_type="vllm",
                                   api_type="LLM",
                                   model_max_tokens=32768,
                                   description="Qwen2.5-32B-Instruct-GPTQ-Int4",
                                   need_gpu_count=1,
                                   template="template_qwen.jinja",
                                   stop=["<|im_start", "<|",
                                         "<|im_end|>", "<|endoftext|>"],
                                   quantization="gptq",
                                   gpu_types={
                                       "NV-A100": GPUConfig(need_gpu_count=1),
                                       "NV-4090": GPUConfig(need_gpu_count=1),
                                       "EF-S60": GPUConfig(need_gpu_count=1)
                                   }
                                   ))
system_models_config.add(LLMConfig(name="Qwen2.5-7B-Instruct",
                                   server_type="vllm",
                                   api_type="LLM",
                                   model_max_tokens=32768,
                                   description="Qwen2.5-7B-Instruct",
                                   need_gpu_count=1,
                                   template="template_qwen.jinja",
                                   stop=["<|im_start", "<|",
                                         "<|im_end|>", "<|endoftext|>"],
                                   gpu_types={
                                       "NV-A100": GPUConfig(need_gpu_count=1),
                                       "NV-4090": GPUConfig(need_gpu_count=2),
                                       "EF-S60": GPUConfig(need_gpu_count=1)
                                   }
                                   ))
system_models_config.add(LLMConfig(name="codegeex4-all-9b",
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
system_models_config.add(LLMConfig(name="glm-4-9b-chat",
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

########################################## CoderLLM ###########################################
system_models_config.add(QwenCoderLLMConfig(name="Qwen2.5-Coder-32B-GPTQ-Int4",
                                            server_type="vllm",
                                            api_type="LLM",
                                            model_max_tokens=32768,
                                            description="Qwen2.5-Coder-32B-GPTQ-Int4",
                                            need_gpu_count=1,
                                            template="template_qwen.jinja",
                                            stop=["<|im_start", "<|",
                                                  "<|im_end|>", "<|endoftext|>"],
                                            gpu_types={
                                                "NV-A100": GPUConfig(need_gpu_count=1),
                                                "NV-4090": GPUConfig(need_gpu_count=2),
                                                "NV-A30": GPUConfig(need_gpu_count=2),
                                            }
                                            ))
system_models_config.add(QwenCoderLLMConfig(name="Qwen2.5-Coder-32B-GPTQ-Int4-4k",
                                            server_type="vllm",
                                            api_type="LLM",
                                            model_max_tokens=4096,
                                            description="Qwen2.5-Coder-32B-GPTQ-Int4",
                                            need_gpu_count=1,
                                            template="template_qwen.jinja",
                                            stop=["<|im_start", "<|",
                                                  "<|im_end|>", "<|endoftext|>"],
                                            gpu_types={
                                                "NV-A100": GPUConfig(need_gpu_count=1),
                                                "NV-4090": GPUConfig(need_gpu_count=1),
                                                "NV-A30": GPUConfig(need_gpu_count=1)
                                            }
                                            ))
system_models_config.add(QwenCoderLLMConfig(name="Qwen2.5-Coder-32B-Instruct-GPTQ-Int4",
                                            server_type="vllm",
                                            api_type="LLM",
                                            model_max_tokens=32768,
                                            description="Qwen2.5-Coder-32B-Instruct-GPTQ-Int4",
                                            need_gpu_count=1,
                                            template="template_qwen.jinja",
                                            stop=["<|im_start", "<|",
                                                  "<|im_end|>", "<|endoftext|>"],
                                            gpu_types={
                                                "NV-A100": GPUConfig(need_gpu_count=1),
                                                "NV-4090": GPUConfig(need_gpu_count=2),
                                                "NV-A30": GPUConfig(need_gpu_count=2),
                                            }
                                            ))
system_models_config.add(QwenCoderLLMConfig(name="Qwen2.5-Coder-32B-Instruct-GPTQ-Int4-4k",
                                            server_type="vllm",
                                            api_type="LLM",
                                            model_max_tokens=4096,
                                            description="Qwen2.5-Coder-32B-Instruct-GPTQ-Int4-4k",
                                            need_gpu_count=1,
                                            template="template_qwen.jinja",
                                            stop=["<|im_start", "<|",
                                                  "<|im_end|>", "<|endoftext|>"],
                                            gpu_types={
                                                "NV-A100": GPUConfig(need_gpu_count=1),
                                                "NV-4090": GPUConfig(need_gpu_count=1),
                                                "NV-A30": GPUConfig(need_gpu_count=1),
                                            }
                                            ))
system_models_config.add(QwenCoderLLMConfig(name="Qwen2.5-Coder-7B",
                                            server_type="vllm",
                                            api_type="LLM",
                                            model_max_tokens=32768,
                                            description="Qwen2.5-Coder-7B",
                                            need_gpu_count=1,
                                            template="template_qwen.jinja",
                                            stop=["<|im_start", "<|",
                                                  "<|im_end|>", "<|endoftext|>"],
                                            gpu_types={
                                                "NV-A100": GPUConfig(need_gpu_count=1),
                                                "NV-4090": GPUConfig(need_gpu_count=2),
                                                "EF-S60": GPUConfig(need_gpu_count=1)
                                            }
                                            ))
system_models_config.add(QwenCoderLLMConfig(name="Qwen2.5-Coder-7B-Instruct",
                                            server_type="vllm",
                                            api_type="LLM",
                                            model_max_tokens=32768,
                                            description="Qwen2.5-Coder-7B-Instruct",
                                            need_gpu_count=1,
                                            template="template_qwen.jinja",
                                            stop=["<|im_start", "<|",
                                                  "<|im_end|>", "<|endoftext|>"],
                                            gpu_types={
                                                "NV-A100": GPUConfig(need_gpu_count=1),
                                                "NV-4090": GPUConfig(need_gpu_count=2),
                                                "EF-S60": GPUConfig(need_gpu_count=1)
                                            }
                                            ))
system_models_config.add(QwenCoderLLMConfig(name="Qwen2.5-Coder-14B",
                                            server_type="vllm",
                                            api_type="LLM",
                                            model_max_tokens=32768,
                                            description="Qwen2.5-Coder-14B",
                                            need_gpu_count=1,
                                            template="template_qwen.jinja",
                                            stop=["<|im_start", "<|",
                                                  "<|im_end|>", "<|endoftext|>"],
                                            gpu_types={
                                                "NV-A100": GPUConfig(need_gpu_count=1),
                                                "NV-4090": GPUConfig(need_gpu_count=2),
                                                "EF-S60": GPUConfig(need_gpu_count=1)
                                            }
                                            ))
system_models_config.add(QwenCoderLLMConfig(name="Qwen2.5-Coder-14B-Instruct",
                                            server_type="vllm",
                                            api_type="LLM",
                                            model_max_tokens=32768,
                                            description="Qwen2.5-Coder-14B-Instruct",
                                            need_gpu_count=1,
                                            template="template_qwen.jinja",
                                            stop=["<|im_start", "<|",
                                                  "<|im_end|>", "<|endoftext|>"],
                                            gpu_types={
                                                "NV-A100": GPUConfig(need_gpu_count=1),
                                                "NV-4090": GPUConfig(need_gpu_count=2),
                                                "EF-S60": GPUConfig(need_gpu_count=1)
                                            }
                                            ))
system_models_config.add(QwenCoderLLMConfig(name="deepseek-coder-6.7b-base",
                                            server_type="vllm",
                                            api_type="LLM",
                                            model_max_tokens=32768,
                                            description="deepseek-coder-6.7b-base",
                                            need_gpu_count=1,
                                            template="template_deepseek-coder.jinja",
                                            stop=["User: ", "Assistant: ", "<|im_start",
                                                  "<|", "<|im_end|>", "<|endoftext|"],
                                            gpu_types={
                                                "NV-A100": GPUConfig(need_gpu_count=1),
                                                "NV-4090": GPUConfig(need_gpu_count=2),
                                                "EF-S60": GPUConfig(need_gpu_count=1)
                                            }
                                            ))
system_models_config.add(QwenCoderLLMConfig(name="deepseek-coder-6.7b-instruct",
                                            server_type="vllm",
                                            api_type="LLM",
                                            model_max_tokens=32768,
                                            description="deepseek-coder-6.7b-instruct",
                                            need_gpu_count=1,
                                            template="template_deepseek-coder.jinja",
                                            stop=["User: ", "Assistant: ", "<|im_start",
                                                  "<|", "<|im_end|>", "<|endoftext|"],
                                            gpu_types={
                                                "NV-A100": GPUConfig(need_gpu_count=1),
                                                "NV-4090": GPUConfig(need_gpu_count=1),
                                                "EF-S60": GPUConfig(need_gpu_count=1)
                                            }
                                            ))
system_models_config.add(QwenCoderLLMConfig(name="DeepSeek-Coder-V2-Lite-Instruct",
                                            server_type="vllm",
                                            api_type="LLM",
                                            model_max_tokens=32768,
                                            description="DeepSeek-Coder-V2-Lite-Instruct 16B",
                                            need_gpu_count=1,
                                            template="template_deepseek-coder.jinja",
                                            stop=["User: ", "Assistant: ", "<|im_start",
                                                  "<|", "<|im_end|>", "<|endoftext|"],
                                            gpu_types={
                                                "NV-A100": GPUConfig(need_gpu_count=1),
                                                "NV-4090": GPUConfig(need_gpu_count=2),
                                                "EF-S60": GPUConfig(need_gpu_count=1)
                                            }
                                            ))
system_models_config.add(QwenCoderLLMConfig(name="DeepSeek-Coder-V2-Lite-Base",
                                            server_type="vllm",
                                            api_type="LLM",
                                            model_max_tokens=32768,
                                            description="DeepSeek-Coder-V2-Lite-Base 16B",
                                            need_gpu_count=1,
                                            template="template_deepseek-coder.jinja",
                                            stop=["User: ", "Assistant: ", "<|im_start",
                                                  "<|", "<|im_end|>", "<|endoftext|"],
                                            gpu_types={
                                                "NV-A100": GPUConfig(need_gpu_count=1),
                                                "NV-4090": GPUConfig(need_gpu_count=2),
                                                "EF-S60": GPUConfig(need_gpu_count=1)
                                            }
                                            ))

########################################## VLM ###########################################
system_models_config.add(VisionConfig(name="glm-4v-9b",
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
########################################## SD ###########################################
system_models_config.add(SDConfig(name="majicmixRealistic_v7.safetensors/majicmixRealistic_v7.safetensors",
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
system_models_config.add(SDConfig(name="SD15MultiControlnetGenerateImage/majicmixRealistic_v7.safetensors/majicmixRealistic_v7.safetensors",
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
########################################## ASR ###########################################
system_models_config.add(ASRConfig(name="funasr",
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
########################################## TTS ###########################################
system_models_config.add(TTSConfig(name="maskgct-tts-clone",
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
########################################## Embedding ###########################################
system_models_config.add(EmbeddingConfig(name="bge-base-zh-v1.5",
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
system_models_config.add(EmbeddingConfig(name="bge-m3",
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
########################################## Rerank ###########################################
system_models_config.add(RerankConfig(name="bge-reranker-base",
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
system_models_config.add(RerankConfig(name="bge-reranker-v2-m3",
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
########################################## Video ###########################################
system_models_config.add(VideoConfig(name="CogVideo/CogVideoX-5b",
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
    print("SYSTEM_MODELS:")
    for m in system_models_config.list():
        print(f"  - {m.name}")
