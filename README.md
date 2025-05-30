# AI 大模型调度系统

## 项目概述

本项目是一个基于 vllm 和 ComfyUI 的高效 AI 计算调度系统。旨在优化大型语言模型的推理过程，提供灵活的资源分配和自动扩展能力。通过集成 OpenAI 兼容的 API 接口，用户可以轻松访问和使用系统资源。
![](./auto_openai/statics/home.png)
- [快速体验地址](https://auto-openai.cpolar.cn/)

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
- **支持深度思考**: API支持深度思考内容，提供更智能的回答。



# 安装

```bash
pip install .
```

# 已支持的模型

| 模型类别   | 模型名称                                                                           | 状态 |
| ---------- | ---------------------------------------------------------------------------------- | ---- |
| 大语言模型 | [vllm 支持的所有模型](https://docs.vllm.ai/en/latest/models/supported_models.html) | ✅   |
| 多模态     | glm-4v-9b, Qwen2.5-VL系列, Qwen2-VL系列                                           | ✅   |
| 图像生成   | SD1.5 系列所有模型以及 Controlnet                                                  | ✅   |
|            | ComfyUI 基础文生图工作流的所有模型                                                 | ✅   |
| Embedding  | bge-base-zh-v1.5                                                                 | ✅   |
|            | bge-m3                                                                           | ✅   |
| Rerank     | bge-reranker-base                                                                | ✅   |
|            | bge-reranker-v2-m3                                                               | ✅   |
| TTS        | maskgct-tts-clone                                                                | ✅   |
| ASR        | funasr                                                                           | ✅   |
| 视频生成   | CogVideo/CogVideoX-5b                                                             | ✅   |

- 获取支持的模型列表
```bash
python3 -m auto_openai.utils.support_models.model_config
```

## [模型下载请参考](./auto_openai/utils/support_models/model_config.py)

![](./auto_openai/statics/models.png)


## 本项目依赖的第三方组件

- redis

### 内置模型下载

```bash
python3 -m auto_openai.lm_server.install_models tiktoken
```

## 配置文件

- 这些配置可以通过环境变量进行修改
  [`conf/config.yaml`](conf/config.yaml)



## 容器化部署

### 部署网关API

- [容器下载地址](https://hub.docker.com/u/aigc919)
- 大模型镜像请参考: auto_openai_lm_images

```yaml
version: "3"
services:
  redis:
    image: redis:7-alpine
    ports:
      - "16379:6379"
    restart: always
    command:
      - redis-server
  openai-api:
    image: aigc919/auto_openai:0.2
    ports:
      - "19000:9000"
    command:
      - /bin/sh
      - -c
      - |
        python3 -m auto_openai.main --port=9000
    restart: always
    volumes:
      - ./conf:/app/conf
    depends_on:
      - redis
```

### 调度器部署

```yaml
services:
  scheduler-0-of-0:
    command: &id001
    - /bin/sh
    - -c
    - python3 -m auto_openai.scheduler
    environment:
      AVAILABLE_MODELS: ALL
      AVAILABLE_SERVER_TYPES: ALL
      LABEL: lm-server-1736148045-908794112
      LM_SERVER_BASE_PORT: 30184
      NODE_GPU_TOTAL: '0'
    image: aigc919/auto_openai:0.2
    network_mode: host
    privileged: true
    restart: always
    shm_size: 8gb
    volumes: &id002
    - ./conf/:/app/conf
    - /root/share_models/:/root/share_models/
    - /var/run/docker.sock:/var/run/docker.sock
    - /usr/bin/docker:/usr/bin/docker
  scheduler-1-of-1:
    command: *id001
    environment:
      AVAILABLE_MODELS: ALL
      AVAILABLE_SERVER_TYPES: ALL
      LABEL: lm-server-1736148045-1366755522
      LM_SERVER_BASE_PORT: 30192
      NODE_GPU_TOTAL: '1'
    image: aigc919/auto_openai:0.2
    network_mode: host
    privileged: true
    restart: always
    shm_size: 8gb
    volumes: *id002
version: '3'

```

## 本地启动服务

```bash
AUTO_OPENAI_CONFIG_PATH=./conf/config.yaml python3 -m auto_openai.main --port=9000
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
![](./auto_openai/statics/test.png)

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
