# AI 大模型调度系统

## 项目概述

本项目是一个基于 vllm 和 ComfyUI 的高效 AI 计算调度系统。旨在优化大型语言模型的推理过程，提供灵活的资源分配和自动扩展能力。通过集成 OpenAI 兼容的 API 接口，用户可以轻松访问和使用系统资源。

## 主要功能

- **基于 vllm 的高效推理**：利用 vllm 的优化技术提升大型语言模型推理的速度和效率。
- **ComfyUI 集成**：集成 ComfyUI，提供直观的用户 API，便于轻松调用。
- **智能计算调度**：根据负载条件自动分配和调整计算资源，确保最佳性能。
- **弹性扩展**：支持内部系统资源的动态扩展和收缩，以适应不断变化的工作负载。
- **OpenAI 兼容的 API**：提供与 OpenAI API 兼容的接口，便于现有应用的快速集成和迁移。

# 安装

```bash
pip install .
```

## 本项目依赖的第三方组件

- vllm/transformers/ComfyUI/embedding/funasr/maskgct/webui(按需安装)
- redis

### 插件安装(按需安装)

```bash
[~]# python3 -m auto_openai.install_plugin --help
COMMANDS
    COMMAND is one of the following:

     comfyui

     embedding

     funasr

     llm_transformer

     maskgct

     tiktoken
```

```bash
python3 -m auto_openai.lm_server.install_plugin tiktoken
python3 -m auto_openai.lm_server.install_plugin comfyui
python3 -m auto_openai.lm_server.install_plugin webui
python3 -m auto_openai.lm_server.install_plugin maskgct
#
python3 -m auto_openai.lm_server.install_plugin embedding
python3 -m auto_openai.lm_server.install_plugin funasr
python3 -m auto_openai.lm_server.install_plugin llm_transformer
python3 -m auto_openai.lm_server.install_plugin rerank
```

## 配置文件

- 这些配置可以通过环境变量进行修改
  `conf/config.yaml`

```yaml
REDIS_CLIENT_CONFIG:
  address: xxxx:6379
  username:
  password:
  cluster: False

OSS_CLIENT_CONFIG:
  endpoint_url: http://xxxxx.com
  aws_access_key_id: xxxx
  aws_secret_access_key: xxxxx
  bucket_name: xxxxxxx
  region_name:

QUEUE_TIMEOUT: 600
INFER_TIMEOUT: 100
USERFULL_TIMES_PER_MODEL: 20
UNUSERFULL_TIMES_PER_MODEL: 10
DEFAULT_MODEL_CONFIG_max_tokens: 4096
NODE_GPU_TOTAL: "0"
MOCK: false

COMFYUI_INPUTS_DIR: "/tmp"
#
COMFYUI_MODEL_ROOT_PATH: /root/share_models/webui-models
#
MASKGCT_MODEL_ROOT_PATH: "/root/share_models/MaskGCT-models/"
#
VLLM_MODEL_ROOT_PATH: /root/share_models/LLM
#
FUNASR_MODEL_ROOT_PATH: "/root/share_models/funasr-models/"
#
EMBEDDING_MODEL_ROOT_PATH: "/root/share_models/Embedding-models/"
#
LLM_TRANSFORMER_MODEL_ROOT_PATH: "/root/share_models/LLM/"
#
WEBUI_MODEL_ROOT_PATH: "/root/share_models/webui-models"
#
RERANK_MODEL_ROOT_PATH: "/root/share_models/Rerank-models/"
#
GPU_DEVICE_ENV_NAME: TOPS_VISIBLE_DEVICES
ONLY_SERVER_TYPES: [] # ["vllm"]
GPU_TYPE: EF-S60
MODELS:
# LLM_MODELS:
  - name: "Qwen2.5-32B-Instruct-GPTQ-Int4"
    server_type: vllm
    api_type: LLM
    model_max_tokens: 32768
    description: "Qwen2.5-32B-Instruct-GPTQ-Int4"
    need_gpu_count: 1
    template: template_qwen.jinja
    stop: ["<|im_start", "<|", "<|im_end|>", "<|endoftext|"]
    quantization: gptq
    gpu_types:
      NV-A100:
        need_gpu_count: 1
      NV-4090:
        need_gpu_count: 1
      EF-S60:
        need_gpu_count: 1
  - name: "Qwen2.5-7B-Instruct"
    server_type: vllm
    api_type: LLM
    model_max_tokens: 32768
    description: "Qwen2.5-7B-Instruct"
    need_gpu_count: 1
    template: template_qwen.jinja
    stop: ["<|im_start", "<|", "<|im_end|>", "<|endoftext|"]
    gpu_types:
      NV-A100:
        need_gpu_count: 1
      NV-4090:
        need_gpu_count: 2
      EF-S60:
        need_gpu_count: 1
  - name: "Qwen2.5-Coder-7B-Instruct"
    server_type: vllm
    api_type: LLM
    model_max_tokens: 32768
    description: "Qwen2.5-Coder-7B-Instruct"
    need_gpu_count: 1
    template: template_qwen.jinja
    stop: ["<|im_start", "<|", "<|im_end|>", "<|endoftext|"]
    gpu_types:
      NV-A100:
        need_gpu_count: 1
      NV-4090:
        need_gpu_count: 2
      EF-S60:
        need_gpu_count: 1
  - name: "codegeex4-all-9b"
    server_type: vllm
    api_type: LLM
    model_max_tokens: 131072
    description: "codegeex4-all-9b"
    need_gpu_count: 1
    template: template_glm4.jinja
    stop: ["<|user|>", "<|assistant|>"]
    gpu_types:
      NV-A100:
        need_gpu_count: 1
      NV-4090:
        need_gpu_count: 2
      EF-S60:
        need_gpu_count: 1
  - name: "glm-4-9b-chat"
    server_type: vllm
    api_type: LLM
    model_max_tokens: 131072
    description: "glm-4-9b-chat"
    need_gpu_count: 1
    template: template_glm4.jinja
    stop: ["<|user|>", "<|assistant|>"]
    gpu_types:
      NV-A100:
        need_gpu_count: 1
      NV-4090:
        need_gpu_count: 2
      EF-S60:
        need_gpu_count: 1

# VISION_MODELS:
  - name: "glm-4v-9b"
    server_type: llm-transformer-server
    api_type: VLLM
    model_max_tokens: 8192
    description: "glm-4v-9b"
    need_gpu_count: 1
    template: template_glm4.jinja
    stop: []
    gpu_types:
      NV-A100:
        need_gpu_count: 1
      NV-4090:
        need_gpu_count: 1
      EF-S60:
        need_gpu_count: 1

# SD_MODELS:
  - name: "sd1.5/majicmixRealistic_betterV6.safetensors"
    server_type: comfyui
    api_type: BaseGenerateImage
    need_gpu_count: 1
    description: "majicmixRealistic_betterV6"
    gpu_types:
      NV-A100:
        need_gpu_count: 1
      NV-4090:
        need_gpu_count: 1
      EF-S60:
        need_gpu_count: 1
  - name: "sdxl/sd_xl_base_1.0.safetensors"
    server_type: comfyui
    api_type: BaseGenerateImage
    need_gpu_count: 1
    description: "sd_xl_base_1"
    gpu_types:
      NV-A100:
        need_gpu_count: 1
      NV-4090:
        need_gpu_count: 1
      EF-S60:
        need_gpu_count: 1
  - name: "flux.1-schnell/flux1-schnell.safetensors"
    server_type: comfyui
    api_type: BaseGenerateImage
    need_gpu_count: 1
    description: "flux1"
    gpu_types:
      NV-A100:
        need_gpu_count: 1
      NV-4090:
        need_gpu_count: 1
      EF-S60:
        need_gpu_count: 1

# TTS_MODELS:
  - name: "maskgct-tts-clone"
    server_type: maskgct
    api_type: TTS
    need_gpu_count: 1
    description: "maskgct-tts-clone"
    gpu_types:
      NV-A100:
        need_gpu_count: 1
      NV-4090:
        need_gpu_count: 1
      EF-S60:
        need_gpu_count: 1

# ASR_MODELS:
  - name: "funasr"
    server_type: funasr
    api_type: ASR
    need_gpu_count: 1
    description: "asr"
    gpu_types:
      NV-A100:
        need_gpu_count: 1
      NV-4090:
        need_gpu_count: 1
      EF-S60:
        need_gpu_count: 1
      CPU:
        need_gpu_count: 1

# EMBEDDING_MODELS:
  - name: "bge-base-zh-v1.5"
    server_type: embedding
    api_type: Embedding
    need_gpu_count: 1
    description: "bge-base-zh-v1.5"
    gpu_types:
      NV-A100:
        need_gpu_count: 1
      NV-4090:
        need_gpu_count: 1
      EF-S60:
        need_gpu_count: 1
      CPU:
        need_gpu_count: 1
  - name: "bge-m3"
    server_type: embedding
    need_gpu_count: 1
    description: "bge-m3"
    gpu_types:
      NV-A100:
        need_gpu_count: 1
      NV-4090:
        need_gpu_count: 1
      EF-S60:
        need_gpu_count: 1
      CPU:
        need_gpu_count: 1


# RERANK_MODELS:
  - name: "bge-reranker-base"
    server_type: rerank
    api_type: Rerank
    need_gpu_count: 1
    description: "bge-reranker-base"
    gpu_types:
      NV-A100:
        need_gpu_count: 1
      NV-4090:
        need_gpu_count: 1
      EF-S60:
        need_gpu_count: 1
      CPU:
        need_gpu_count: 1
  - name: "bge-reranker-v2-m3"
    server_type: rerank
    need_gpu_count: 1
    description: "bge-reranker-v2-m3"
    gpu_types:
      NV-A100:
        need_gpu_count: 1
      NV-4090:
        need_gpu_count: 1
      EF-S60:
        need_gpu_count: 1
      CPU:
        need_gpu_count: 1


```

## [模型下载请参考](./model_download.sh)

## 启动服务

```bash
AUTO_OPENAI_CONFIG_PATH=./conf/config.yaml python3 -m auto_openai.main --port=9000 --workers=2
```

## 分布式部署算力调度器

```bash
AUTO_OPENAI_CONFIG_PATH=./conf/config.yaml python3 -m auto_openai.scheduler
```

## 如何单独启动 webui/comfyui 做测试

```bash
python3 -m auto_openai.lm_server.start --help

NAME
    start.py

SYNOPSIS
    start.py COMMAND | -

COMMANDS
    COMMAND is one of the following:

     get_comfyui

     get_embedding

     get_funasr

     get_llm_transformer

     get_maskgct

     get_vllm

     get_webui
```

# 快速入门

## 示例 OpenAI API 参考

> `base_url = "http://127.0.0.1:9000/openai/v1"`

## 根据提示生成文本

OpenAI 提供了简单的 API，使用大型语言模型根据提示生成文本，类似于使用 ChatGPT。这些模型已经在大量数据上进行训练，以理解多媒体输入和自然语言指令。从这些提示中，模型可以生成几乎任何类型的文本响应，包括代码、数学方程、结构化 JSON 数据或类似人类的散文。

要生成文本，您可以使用 REST API 中的聊天补全端点，如下面的示例所示。您可以使用您选择的 HTTP 客户端使用 REST API，或者使用 OpenAI 为您首选的编程语言提供的官方 SDK。

> `base_url = "http://127.0.0.1:9000/openai/v1"`

```python
from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
    model="Qwen2.5-7B-Instruct",
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {
            "role": "user",
            "content": "Write a haiku about recursion in programming."
        }
    ]
)

print(completion.choices[0].message)
```