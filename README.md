# AI 大模型调度系统

## 项目概述

本项目是一个基于 vllm 和 ComfyUI 的高效 AI 计算调度系统。旨在优化大型语言模型的推理过程，提供灵活的资源分配和自动扩展能力。通过集成 OpenAI 兼容的 API 接口，用户可以轻松访问和使用系统资源。

## 主要功能

- **基于 vllm 的高效推理**：利用 vllm 的优化技术提升大型语言模型推理的速度和效率。
- **ComfyUI 集成**：集成 ComfyUI，提供直观的用户 API，便于轻松调用。
- **transformers 支持**：支持 transformers 库，便于用户使用现有的模型和工具。
- **SD WebUI 支持**: 支持 SD WebUI，提供丰富的图像生成功能。
- **智能计算调度**：根据负载条件自动分配和调整计算资源，确保最佳性能。
- **弹性扩展**：支持内部系统资源的动态扩展和收缩，以适应不断变化的工作负载。
- **OpenAI 兼容的 API**：提供与 OpenAI API 兼容的接口，便于现有应用的快速集成和迁移。
- **多类型 api 支持**: 支持多种类型的 API，包括 LLM, VL, SD, Embedding, Audio, Image, TTS, rerank 等。
- **分布式支持**: 支持分布式计算，提高计算效率。

# 安装

```bash
pip install .
```

# 已支持的模型

## 大语言模型

- [x] [vllm 支持的所有模型](https://docs.vllm.ai/en/latest/models/supported_models.html)

## 多模态

- [x] glm-4v-9b

## 图像生成

- [x] SD1.5 系列所有模型以及 Controlnet
- [x] ComfyUI 基础文生图工作流的所有模型

## Embedding

- [x] bge-base-zh-v1.5
- [x] bge-m3

## Rerank

- [x] bge-reranker-base
- [x] bge-reranker-v2-m3

## TTS

- [x] maskgct-tts-clone

## ASR

- [x] funasr

## 视频生成

- [x] CogVideo/CogVideoX-5b

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
  [`conf/config.yaml`](conf/config.yaml)

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
