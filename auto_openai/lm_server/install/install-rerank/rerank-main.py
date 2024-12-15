import gradio as gr
from sentence_transformers import CrossEncoder
import os
import sys
import json
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
else:
    device = "cpu"

model = None
tokenizer = None
model_name_path = None


async def infer(inputs: list, query, model_name: str, top_k=3):
    global model_name_path, model, tokenizer
    now_model_name_path = os.path.join(
        os.environ["model_root_path"], model_name)
    if now_model_name_path != model_name_path:
        model = CrossEncoder(now_model_name_path, device=device)
        model_name_path = now_model_name_path
    results = model.rank(query, inputs, return_documents=True, top_k=top_k)
    data = []
    for result in results:
        data.append({
            "index": result["corpus_id"],
            "relevance_score": result["score"],
        })
    return data


def run(port=7861, model_root_path="/root/share_models/Rerank-models/"):
    os.environ["model_root_path"] = model_root_path
    demo = gr.Interface(
        fn=infer,
        inputs=[gr.JSON(label="docs", value=[
            "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.",
                        "what is panda?",
                        "hello world"]),
                gr.Textbox(label="query"),
                gr.Textbox(label="模型名称"),
                gr.Number(label="top_k", value=3)
                ],
        outputs=gr.JSON(label="输出"),
        title="rerank"
    )

    demo.launch(server_name="0.0.0.0", server_port=port)

# bge-reranker-base  bge-reranker-v2-m3
# usage:
# python main.py --port 7861 --model_root_path /root/share_models/Rerank-models/


if __name__ == '__main__':
    from fire import Fire
    Fire(run)
