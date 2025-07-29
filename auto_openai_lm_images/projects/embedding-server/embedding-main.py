import gradio as gr
import os
import sys
import json
from langchain_huggingface import HuggingFaceEmbeddings
import torch
if os.environ.get("TOPS_VISIBLE_DEVICES") is not None:
    # 支持GCU算力卡
    try:
        import torch_gcu  # 导入 torch_gcu
        from torch_gcu import transfer_to_gcu  # 导入 transfer_to_gcu
        device = "gcu"
    except Exception as e:
        raise e
elif os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
    device = "cuda"
elif os.environ.get("NVIDIA_VISIBLE_DEVICES") is not None:
    device = "cuda"
else:
    device = "cpu"


print(f"current device: {device}")

embed_model = None
model_name_path = None


def load_model(model_name: str):
    global model_name_path, embed_model
    now_model_name_path = os.path.join(
        os.environ["model_root_path"], model_name)
    if now_model_name_path != model_name_path:
        embed_model = HuggingFaceEmbeddings(
            model_name=now_model_name_path,
            encode_kwargs={'normalize_embeddings': False},
            **{} if device == "cpu" else {"model_kwargs": {'device': device}}
        )
        model_name_path = now_model_name_path


def embed_query(inputs: list, model_name: str):
    load_model(model_name)
    res = embed_model.embed_documents(inputs)
    data = []
    for i, em in enumerate(res):
        data.append({"index": i, "object": 'embedding',
                    "embedding": em})

    return data


def run(port=7861, model_root_path="/root/share_models/Embedding-models/", model_name=""):
    os.environ["model_root_path"] = model_root_path
    if model_name:
        load_model(model_name)
    demo = gr.Interface(
        fn=embed_query,
        inputs=[gr.JSON(label="输入文本", value=["Hello", "World"]),
                gr.Textbox(label="模型名称")],
        outputs=gr.JSON(label="嵌入向量"),
        title="嵌入向量计算",
        description="输入文本，计算其嵌入向量"
    )

    demo.launch(server_name="0.0.0.0", server_port=port)

# bge-base-zh-v1.5
# bge-m3
# ["hello", "world"]
# TOPS_VISIBLE_DEVICES=7 python3 auto_openai/lm_server/install/install-embedding/embedding-main.py --model_name bge-base-zh-v1.5


if __name__ == '__main__':
    from fire import Fire
    Fire(run)
