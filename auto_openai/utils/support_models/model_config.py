from auto_openai.utils.support_models import *
from auto_openai.utils.support_devices import all_supported_device
from math import ceil


def get_gpu_types_count(mem: int):
    # 将所有设备向上整除，显存需要171GB,则得到对应的卡数
    valid_counts = [1, 2, 4, 8]
    result = {}

    for device_ in all_supported_device:
        count = ceil(mem / device_.mem)
        valid_count = next((x for x in valid_counts if x >= count), None)
        if valid_count is not None and valid_count <= 8:
            result[device_.name] = GPUConfig(need_gpu_count=valid_count)

    return result


########################################## LLM ###########################################


system_models_config.extend(LLMConfig(name="Qwen2.5-72B-Instruct",
                                      model_url="https://www.modelscope.cn/Qwen/Qwen2.5-72B-Instruct.git",
                                      server_type="vllm",
                                      api_type="LLM",
                                      model_max_tokens=10240,
                                      description="Qwen2.5-72B-Instruct",
                                      need_gpu_count=1,
                                      template="template_qwen.jinja",
                                      stop=["<|im_start", "<|",
                                            "<|im_end|>", "<|endoftext|>"],
                                      gpu_types=get_gpu_types_count(160)
                                      ).extend([
                                          MultiGPUS(model_max_tokens=10*1024,
                                                    gpu_types=get_gpu_types_count(160)),
                                      ]))
system_models_config.extend(LLMConfig(name="Qwen2.5-32B-Instruct-GPTQ-Int4",
                                      model_url="https://www.modelscope.cn/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4.git",
                                      server_type="vllm",
                                      api_type="LLM",
                                      model_max_tokens=4*1024,
                                      description="Qwen2.5-32B-Instruct-GPTQ-Int4",
                                      need_gpu_count=1,
                                      template="template_qwen.jinja",
                                      stop=["<|im_start", "<|",
                                            "<|im_end|>", "<|endoftext|>"],
                                      quantization="gptq",
                                      gpu_types=get_gpu_types_count(24)
                                      ).extend([
                                          MultiGPUS(
                                              model_max_tokens=32*1024, gpu_types=get_gpu_types_count(40)),
                                          MultiGPUS(
                                              model_max_tokens=4*1024, gpu_types=get_gpu_types_count(24))
                                      ]))
system_models_config.extend(LLMConfig(name="Qwen2.5-7B-Instruct",
                                      model_url="https://www.modelscope.cn/Qwen/Qwen2.5-7B-Instruct.git",
                                      server_type="vllm",
                                      api_type="LLM",
                                      model_max_tokens=32768,
                                      description="Qwen2.5-7B-Instruct",
                                      need_gpu_count=1,
                                      template="template_qwen.jinja",
                                      stop=["<|im_start", "<|",
                                            "<|im_end|>", "<|endoftext|>"],
                                      gpu_types=get_gpu_types_count(24)
                                      ).extend([
                                          MultiGPUS(
                                              model_max_tokens=32*1024, gpu_types=get_gpu_types_count(24)),
                                      ]))
system_models_config.extend(LLMConfig(name="glm-4-9b-chat",
                                      model_url="https://www.modelscope.cn/ZhipuAI/glm-4-9b-chat.git",
                                      server_type="vllm-glm",
                                      api_type="LLM",
                                      model_max_tokens=4*1024,
                                      description="glm-4-9b-chat",
                                      need_gpu_count=1,
                                      template="template_glm4.jinja",
                                      stop=["<|user|>", "<|assistant|>"],
                                      gpu_types=get_gpu_types_count(24)
                                      ).extend([
                                          MultiGPUS(
                                              model_max_tokens=32*1024, gpu_types=get_gpu_types_count(48)),
                                          MultiGPUS(
                                              model_max_tokens=10*1024, gpu_types=get_gpu_types_count(24)),
                                      ]))
########################################## CoderLLM ###########################################
system_models_config.extend(QwenCoderLLMConfig(name="Qwen2.5-Coder-32B-GPTQ-Int4",
                                               model_url="https://www.modelscope.cn/jackle/Qwen2.5-Coder-32B-GPTQ-Int4.git",
                                               server_type="vllm",
                                               api_type="LLM",
                                               model_max_tokens=4*1024,
                                               description="Qwen2.5-Coder-32B-GPTQ-Int4",
                                               need_gpu_count=1,
                                               template="template_qwen.jinja",
                                               stop=["<|im_start", "<|",
                                                     "<|im_end|>", "<|endoftext|>"],
                                               gpu_types=get_gpu_types_count(
                                                   24)
                                               ).extend([
                                                   MultiGPUS(model_max_tokens=32*1024,
                                                             gpu_types=get_gpu_types_count(48)),
                                                   MultiGPUS(model_max_tokens=10*1024,
                                                             gpu_types=get_gpu_types_count(24)),
                                               ]))
system_models_config.extend(QwenCoderLLMConfig(name="Qwen2.5-Coder-32B-Instruct-GPTQ-Int4",
                                               model_url="https://www.modelscope.cn/Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-Int4",
                                               server_type="vllm",
                                               api_type="LLM",
                                               model_max_tokens=4*1024,
                                               description="Qwen2.5-Coder-32B-Instruct-GPTQ-Int4",
                                               need_gpu_count=1,
                                               template="template_qwen.jinja",
                                               stop=["<|im_start", "<|",
                                                     "<|im_end|>", "<|endoftext|>"],
                                               gpu_types=get_gpu_types_count(
                                                   24)
                                               ).extend([
                                                   MultiGPUS(model_max_tokens=32*1024,
                                                             gpu_types=get_gpu_types_count(48)),
                                                   MultiGPUS(model_max_tokens=10*1024,
                                                             gpu_types=get_gpu_types_count(24)),
                                               ]))
system_models_config.extend(QwenCoderLLMConfig(name="Qwen2.5-Coder-7B",
                                               model_url="https://www.modelscope.cn/Qwen/Qwen2.5-Coder-7B.git",
                                               server_type="vllm",
                                               api_type="LLM",
                                               model_max_tokens=32768,
                                               description="Qwen2.5-Coder-7B",
                                               need_gpu_count=1,
                                               enforce_eager=True,
                                               num_scheduler_steps=8,
                                               template="template_qwen.jinja",
                                               stop=["<|im_start", "<|",
                                                     "<|im_end|>", "<|endoftext|>"],
                                               gpu_types=get_gpu_types_count(
                                                   40)
                                               ).extend([
                                                   MultiGPUS(model_max_tokens=32*1024,
                                                             gpu_types=get_gpu_types_count(24)),
                                               ]))
system_models_config.extend(QwenCoderLLMConfig(name="Qwen2.5-Coder-7B-Instruct",
                                               model_url="https://www.modelscope.cn/Qwen/Qwen2.5-Coder-7B-Instruct.git",
                                               server_type="vllm",
                                               api_type="LLM",
                                               model_max_tokens=32768,
                                               description="Qwen2.5-Coder-7B-Instruct",
                                               need_gpu_count=1,
                                               template="template_qwen.jinja",
                                               stop=["<|im_start", "<|",
                                                     "<|im_end|>", "<|endoftext|>"],
                                               gpu_types=get_gpu_types_count(
                                                   40)
                                               ).extend([
                                                   MultiGPUS(model_max_tokens=32*1024,
                                                             gpu_types=get_gpu_types_count(24)),
                                               ]))
system_models_config.extend(QwenCoderLLMConfig(name="Qwen2.5-Coder-32B",
                                               model_url="https://www.modelscope.cn/Qwen/Qwen2.5-Coder-32B.git",
                                               server_type="vllm",
                                               api_type="LLM",
                                               model_max_tokens=4*1024,
                                               description="Qwen2.5-Coder-32B",
                                               need_gpu_count=1,
                                               template="template_qwen.jinja",
                                               stop=["<|im_start", "<|",
                                                     "<|im_end|>", "<|endoftext|>"],
                                               gpu_types=get_gpu_types_count(
                                                   80)
                                               ).extend([
                                                   MultiGPUS(model_max_tokens=10*1024,
                                                             gpu_types=get_gpu_types_count(96)),
                                                   MultiGPUS(model_max_tokens=32*1024,
                                                             gpu_types=get_gpu_types_count(120)),
                                               ]))
system_models_config.extend(QwenCoderLLMConfig(name="Qwen2.5-Coder-32B-Instruct",
                                               model_url="https://www.modelscope.cn/Qwen/Qwen2.5-Coder-32B-Instruct.git",
                                               server_type="vllm",
                                               api_type="LLM",
                                               model_max_tokens=4*1024,
                                               description="Qwen2.5-Coder-32B-Instruct",
                                               need_gpu_count=1,
                                               template="template_qwen.jinja",
                                               stop=["<|im_start", "<|",
                                                     "<|im_end|>", "<|endoftext|>"],
                                               gpu_types=get_gpu_types_count(
                                                   80)
                                               ).extend([
                                                   MultiGPUS(model_max_tokens=10*1024,
                                                             gpu_types=get_gpu_types_count(96)),
                                                   MultiGPUS(model_max_tokens=32*1024,
                                                             gpu_types=get_gpu_types_count(120)),
                                               ]))
system_models_config.extend(QwenCoderLLMConfig(name="Qwen2.5-Coder-14B",
                                               model_url="https://www.modelscope.cn/Qwen/Qwen2.5-Coder-14B.git",
                                               server_type="vllm",
                                               api_type="LLM",
                                               model_max_tokens=4*1024,
                                               description="Qwen2.5-Coder-14B",
                                               need_gpu_count=1,
                                               enforce_eager=True,
                                               num_scheduler_steps=8,
                                               template="template_qwen.jinja",
                                               stop=["<|im_start", "<|",
                                                     "<|im_end|>", "<|endoftext|>"],
                                               gpu_types=get_gpu_types_count(
                                                   40)
                                               ).extend([
                                                   MultiGPUS(model_max_tokens=10*1024,
                                                             gpu_types=get_gpu_types_count(40)),
                                                   MultiGPUS(model_max_tokens=20*1024,
                                                             gpu_types=get_gpu_types_count(48)),
                                                   MultiGPUS(model_max_tokens=32*1024,
                                                             gpu_types=get_gpu_types_count(72)),
                                               ]))
system_models_config.extend(QwenCoderLLMConfig(name="Qwen2.5-Coder-14B-Instruct",
                                               model_url="https://www.modelscope.cn/Qwen/Qwen2.5-Coder-14B-Instruct.git",
                                               server_type="vllm",
                                               api_type="LLM",
                                               model_max_tokens=4*1024,
                                               description="Qwen2.5-Coder-14B-Instruct",
                                               need_gpu_count=1,
                                               template="template_qwen.jinja",
                                               stop=["<|im_start", "<|",
                                                     "<|im_end|>", "<|endoftext|>"],
                                               gpu_types=get_gpu_types_count(
                                                   40)
                                               ).extend([
                                                   MultiGPUS(model_max_tokens=10*1024,
                                                             gpu_types=get_gpu_types_count(40)),
                                                   MultiGPUS(model_max_tokens=20*1024,
                                                             gpu_types=get_gpu_types_count(48)),
                                                   MultiGPUS(model_max_tokens=32*1024,
                                                             gpu_types=get_gpu_types_count(72)),
                                               ]))
system_models_config.extend(QwenCoderLLMConfig(name="DeepSeek-R1-Distill-Qwen-32B",
                                               model_url="https://www.modelscope.cn/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B.git",
                                               server_type="vllm",
                                               api_type="LLM",
                                               model_max_tokens=4*1024,
                                               description="Qwen2.5-32B 蒸馏版(DeepSeek-R1)",
                                               need_gpu_count=2,
                                               template="template_qwen.jinja",
                                               stop=["<|im_start", "<|",
                                                     "<|im_end|>", "<|endoftext|>"],
                                               gpu_types=get_gpu_types_count(
                                                   96)
                                               ).extend([
                                                   MultiGPUS(model_max_tokens=10*1024,
                                                             gpu_types=get_gpu_types_count(96)),
                                                   MultiGPUS(model_max_tokens=32*1024,
                                                             gpu_types=get_gpu_types_count(120)),
                                               ]))
system_models_config.extend(QwenCoderLLMConfig(name="DeepSeek-R1-Distill-Qwen-14B",
                                               model_url="https://www.modelscope.cn/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B.git",
                                               server_type="vllm",
                                               api_type="LLM",
                                               model_max_tokens=4*1024,
                                               description="Qwen2.5-14B 蒸馏版(DeepSeek-R1)",
                                               need_gpu_count=1,
                                               template="template_qwen.jinja",
                                               stop=["<|im_start", "<|",
                                                     "<|im_end|>", "<|endoftext|>"],
                                               gpu_types=get_gpu_types_count(
                                                   40)
                                               ).extend([
                                                   MultiGPUS(model_max_tokens=10*1024,
                                                             gpu_types=get_gpu_types_count(40)),
                                                   MultiGPUS(model_max_tokens=20*1024,
                                                             gpu_types=get_gpu_types_count(48)),
                                                   MultiGPUS(model_max_tokens=32*1024,
                                                             gpu_types=get_gpu_types_count(72)),
                                               ]))
system_models_config.extend(QwenCoderLLMConfig(name="DeepSeek-R1-Distill-Qwen-7B",
                                               model_url="https://www.modelscope.cn/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B.git",
                                               server_type="vllm",
                                               api_type="LLM",
                                               model_max_tokens=4*1024,
                                               description="Qwen2.5-7B 蒸馏版(DeepSeek-R1)",
                                               need_gpu_count=1,
                                               template="template_qwen.jinja",
                                               stop=["<|im_start", "<|",
                                                     "<|im_end|>", "<|endoftext|>"],
                                               gpu_types=get_gpu_types_count(
                                                   24)
                                               ).extend([
                                                   MultiGPUS(
                                                        model_max_tokens=32*1024,
                                                        gpu_types=get_gpu_types_count(24)),
                                               ]))

system_models_config.extend(DeepseekCoderLLMConfig(name="deepseek-coder-6.7b-base",
                                                   model_url="https://www.modelscope.cn/deepseek-ai/deepseek-coder-6.7b-base.git",
                                                   server_type="vllm",
                                                   api_type="LLM",
                                                   model_max_tokens=10*1024,
                                                   description="deepseek-coder-6.7b-base",
                                                   need_gpu_count=1,
                                                   enforce_eager=True,
                                                   num_scheduler_steps=8,
                                                   template="template_deepseek-coder.jinja",
                                                   stop=[
                                                       "User: ", "Assistant: "],
                                                   gpu_types=get_gpu_types_count(
                                                       24)
                                                   ).extend([
                                                       MultiGPUS(model_max_tokens=32*1024,
                                                                 gpu_types=get_gpu_types_count(40)),
                                                       MultiGPUS(model_max_tokens=10*1024,
                                                                 gpu_types=get_gpu_types_count(24)),
                                                   ]))
system_models_config.extend(DeepseekCoderLLMConfig(name="deepseek-coder-6.7b-instruct",
                                                   model_url="https://www.modelscope.cn/deepseek-ai/deepseek-coder-6.7b-instruct.git",
                                                   server_type="vllm",
                                                   api_type="LLM",
                                                   model_max_tokens=10*1024,
                                                   description="deepseek-coder-6.7b-instruct",
                                                   need_gpu_count=1,
                                                   template="template_deepseek-coder.jinja",
                                                   stop=[
                                                       "User: ", "Assistant: "],
                                                   gpu_types=get_gpu_types_count(
                                                       24)
                                                   ).extend([
                                                       MultiGPUS(model_max_tokens=32*1024,
                                                                 gpu_types=get_gpu_types_count(40)),
                                                       MultiGPUS(model_max_tokens=10*1024,
                                                                 gpu_types=get_gpu_types_count(24)),
                                                   ]))
system_models_config.extend(DeepseekCoderLLMConfig(name="DeepSeek-Coder-V2-Lite-Instruct",
                                                   model_url="https://www.modelscope.cn/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct.git",
                                                   server_type="vllm",
                                                   api_type="LLM",
                                                   model_max_tokens=4*1024,
                                                   description="DeepSeek-Coder-V2-Lite-Instruct 16B",
                                                   need_gpu_count=1,
                                                   template="template_deepseek-coder.jinja",
                                                   stop=[
                                                       "User: ", "Assistant: "],
                                                   gpu_types=get_gpu_types_count(
                                                       35)
                                                   ).extend([
                                                       MultiGPUS(model_max_tokens=10*1024,
                                                                 gpu_types=get_gpu_types_count(40)),
                                                       MultiGPUS(model_max_tokens=8*1024,
                                                                 gpu_types=get_gpu_types_count(40)),
                                                   ]))
system_models_config.extend(DeepseekCoderLLMConfig(name="DeepSeek-Coder-V2-Lite-Base",
                                                   model_url="https://www.modelscope.cn/deepseek-ai/DeepSeek-Coder-V2-Lite-Base.git",
                                                   server_type="vllm",
                                                   api_type="LLM",
                                                   model_max_tokens=4*1024,
                                                   description="DeepSeek-Coder-V2-Lite-Base 16B",
                                                   need_gpu_count=1,
                                                   enforce_eager=True,
                                                   num_scheduler_steps=8,
                                                   template="template_deepseek-coder.jinja",
                                                   stop=[
                                                       "User: ", "Assistant: "],
                                                   gpu_types=get_gpu_types_count(
                                                       35)
                                                   ).extend([
                                                       MultiGPUS(model_max_tokens=10*1024,
                                                                 gpu_types=get_gpu_types_count(40)),
                                                       MultiGPUS(model_max_tokens=8*1024,
                                                                 gpu_types=get_gpu_types_count(40)),
                                                   ]))
system_models_config.extend(LLMConfig(name="codegeex4-all-9b",
                                      model_url="https://www.modelscope.cn/ZhipuAI/codegeex4-all-9b.git",
                                      server_type="vllm-glm",
                                      api_type="LLM",
                                      model_max_tokens=32*1024,
                                      description="codegeex4-all-9b",
                                      need_gpu_count=1,
                                      template="template_glm4.jinja",
                                      stop=["<|user|>", "<|assistant|>"],
                                      gpu_types=get_gpu_types_count(40)
                                      ).extend([
                                          MultiGPUS(model_max_tokens=10*1024,
                                                    gpu_types=get_gpu_types_count(40)),
                                          MultiGPUS(model_max_tokens=32*1024,
                                                    gpu_types=get_gpu_types_count(40)),
                                      ]))
########################################## VLM ###########################################
system_models_config.add(VisionConfig(name="glm-4v-9b",
                                      model_url="https://www.modelscope.cn/ZhipuAI/glm-4v-9b.git",
                                      #   server_type="llm-transformer-server",
                                      server_type="vllm-glm",
                                      api_type="VLLM",
                                      model_max_tokens=8192,
                                      description="glm-4v-9b",
                                      need_gpu_count=1,
                                      template="template_glm4.jinja",
                                      stop=[],
                                      gpu_types=get_gpu_types_count(40)
                                      ))
system_models_config.extend(LLMConfig(name="Qwen2-VL-7B-Instruct",
                                      model_url="https://www.modelscope.cn/Qwen/Qwen2-VL-7B-Instruct.git",
                                      server_type="vllm-qwen2-vl",
                                      api_type="VLLM",
                                      model_max_tokens=32768,
                                      description="Qwen2-VL-7B-Instruct",
                                      need_gpu_count=1,
                                      template="template_qwen.jinja",
                                      stop=["<|im_start", "<|",
                                            "<|im_end|>", "<|endoftext|>"],
                                      gpu_types=get_gpu_types_count(24)
                                      ).extend([
                                          MultiGPUS(
                                              model_max_tokens=32*1024, gpu_types=get_gpu_types_count(24)),
                                      ]))

########################################## Embedding ###########################################
system_models_config.add(EmbeddingConfig(name="bge-base-zh-v1.5",
                                         model_url="https://www.modelscope.cn/ai-modelscope/bge-base-zh-v1.5.git",
                                         server_type="embedding",
                                         api_type="Embedding",
                                         description="bge-base-zh-v1.5",
                                         need_gpu_count=1,
                                         gpu_types=get_gpu_types_count(10)
                                         ))
system_models_config.add(EmbeddingConfig(name="bge-m3",
                                         model_url="https://www.modelscope.cn/BAAI/bge-m3.git",
                                         server_type="embedding",
                                         api_type="Embedding",
                                         description="bge-m3",
                                         need_gpu_count=1,
                                         gpu_types=get_gpu_types_count(10)
                                         ))
########################################## Rerank ###########################################
system_models_config.add(RerankConfig(name="bge-reranker-base",
                                      model_url="https://www.modelscope.cn/BAAI/bge-reranker-base.git",
                                      server_type="rerank",
                                      api_type="Rerank",
                                      description="bge-rerank",
                                      need_gpu_count=1,
                                      gpu_types=get_gpu_types_count(10)
                                      ))
system_models_config.add(RerankConfig(name="bge-reranker-v2-m3",
                                      model_url="https://www.modelscope.cn/BAAI/bge-reranker-v2-m3.git",
                                      server_type="rerank",
                                      api_type="Rerank",
                                      description="bge-reranker-v2-m3",
                                      need_gpu_count=1,
                                      gpu_types=get_gpu_types_count(10)
                                      ))
########################################## Video ###########################################
system_models_config.add(VideoConfig(name="CogVideo/CogVideoX-5b",
                                     model_url="https://www.modelscope.cn/ZhipuAI/CogVideoX-5b.git",
                                     server_type="diffusers-video",
                                     api_type="Video",
                                     description="CogVideo/CogVideoX-5b",
                                     need_gpu_count=1,
                                     gpu_types=get_gpu_types_count(24)
                                     ))
########################################## SD ###########################################
system_models_config.add(SDConfig(name="majicmixRealistic_v7.safetensors/majicmixRealistic_v7.safetensors",
                                  model_url="https://www.modelscope.cn/GYMaster/majicmixRealistic_v7.safetensors.git",
                                  server_type="comfyui",
                                  api_type="SolutionBaseGenerateImage",
                                  description="majicmixRealistic_betterV6",
                                  need_gpu_count=1,
                                  gpu_types=get_gpu_types_count(16)
                                  ))
system_models_config.add(SDConfig(name="SD15MultiControlnetGenerateImage/majicmixRealistic_v7.safetensors/majicmixRealistic_v7.safetensors",
                                  model_url="https://www.modelscope.cn/GYMaster/majicmixRealistic_v7.safetensors.git",
                                  server_type="webui",
                                  api_type="SD15MultiControlnetGenerateImage",
                                  description="majicmixRealistic_v7",
                                  need_gpu_count=1,
                                  gpu_types=get_gpu_types_count(16)
                                  ))
########################################## ASR ###########################################
system_models_config.add(ASRConfig(name="funasr",
                                   server_type="funasr",
                                   api_type="ASR",
                                   description="asr",
                                   need_gpu_count=1,
                                   gpu_types=get_gpu_types_count(24)
                                   ))
########################################## TTS ###########################################
system_models_config.add(TTSConfig(name="maskgct-tts-clone",
                                   server_type="maskgct",
                                   api_type="TTS",
                                   description="maskgct-tts-clone",
                                   need_gpu_count=1,
                                   gpu_types=get_gpu_types_count(24)
                                   ))

########################################## 定制化模型 ###########################################
for m in global_config.CUSTOM_MODLES:
    system_models_config.add(LLMConfig(**m))
# usage
# python3 -m auto_openai.utils.support_models.model_config

if __name__ == "__main__":
    # print yaml 格式
    print("SYSTEM_MODELS:")
    for m in system_models_config.list():
        print(f"  - {m.name}")
