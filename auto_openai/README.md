# AI 大模型调度系统

## 项目概述

本项目是一个基于 vllm 和 ComfyUI 的高效 AI 计算调度系统。旨在优化大型语言模型的推理过程，提供灵活的资源分配和自动扩展能力。通过集成 OpenAI 兼容的 API 接口，用户可以轻松访问和使用系统资源。

## 主要功能

- **基于 vllm 的高效推理**：利用 vllm 的优化技术提升大型语言模型推理的速度和效率。
- **ComfyUI 集成**：集成 ComfyUI，提供直观的用户 API，便于轻松调用。
- **智能计算调度**：根据负载条件自动分配和调整计算资源，确保最佳性能。
- **弹性扩展**：支持内部系统资源的动态扩展和收缩，以适应不断变化的工作负载。
- **OpenAI 兼容的 API**：提供与 OpenAI API 兼容的接口，便于现有应用的快速集成和迁移。

# API 功能

> `base_url = "http://127.0.0.1:9000/openai/v1"`

# 示例 OpenAI API 参考

## 根据提示生成文本

OpenAI 提供了简单的 API，使用大型语言模型根据提示生成文本，类似于使用 ChatGPT。这些模型已经在大量数据上进行训练，以理解多媒体输入和自然语言指令。从这些提示中，模型可以生成几乎任何类型的文本响应，包括代码、数学方程、结构化 JSON 数据或类似人类的散文。

## 快速入门

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
