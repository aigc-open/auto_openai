import gradio as gr
from langchain_huggingface import HuggingFaceEmbeddings
import os
import sys
import json
import torch
if os.environ.get("TOPS_VISIBLE_DEVICES") is not None:
    # 支持GCU算力卡
    try:
        import torch_gcu  # 导入 torch_gcu
        from torch_gcu import transfer_to_gcu  # 导入 transfer_to_gcu
        device = "cuda"
    except Exception as e:
        raise e
elif os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
    device = "cuda"
else:
    device = "cpu"

embed_model = None
model_name_path = None


async def embed_query(inputs:list, model_name: str):
    global model_name_path, embed_model
    now_model_name_path = os.path.join(
        os.environ["model_root_path"], model_name)
    if now_model_name_path != model_name_path:
        embed_model = HuggingFaceEmbeddings(
            model_name=now_model_name_path,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': False}
        )
        model_name_path = now_model_name_path
    res = embed_model.embed_documents(inputs)
    data = []
    for i, em in enumerate(res):
        data.append({"index": i, "object": 'embedding',
                    "embedding": em})

    return data


def run(port=7861, model_root_path="/root/share_models/Embedding-models/"):
    os.environ["model_root_path"] = model_root_path
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


if __name__ == '__main__':
    from fire import Fire
    Fire(run)
