
# Embedding
Embedding_path=/root/share_models/Embedding-models
mkdir -p $Embedding_path
cd $Embedding_path && git lfs install && git clone https://www.modelscope.cn/ai-modelscope/bge-base-zh-v1.5.git
cd $Embedding_path && git lfs install && git clone https://www.modelscope.cn/BAAI/bge-m3.git

# LLM
LLM_path=/root/share_models/LLM
mkdir -p $LLM_path
cd $LLM_path && git lfs install && git clone https://www.modelscope.cn/Qwen/Qwen2.5-7B-Instruct.git
cd $LLM_path && git lfs install && git clone https://www.modelscope.cn/Qwen/Qwen2.5-Coder-7B.git
cd $LLM_path && git lfs install && git clone https://www.modelscope.cn/Qwen/Qwen2.5-Coder-7B-Instruct.git
cd $LLM_path && git lfs install && git clone https://www.modelscope.cn/Qwen/Qwen2.5-Coder-1.5B.git
cd $LLM_path && git lfs install && git clone https://www.modelscope.cn/Qwen/Qwen2.5-Coder-1.5B-Instruct.git
cd $LLM_path && git lfs install && git clone https://www.modelscope.cn/ZhipuAI/codegeex4-all-9b.git
cd $LLM_path && git lfs install && git clone https://www.modelscope.cn/ZhipuAI/glm-4-9b-chat.git
# VISION
# cd $LLM_path && git lfs install && git clone https://www.modelscope.cn/Qwen/Qwen2-VL-7B-Instruct.git
cd $LLM_path && git lfs install && git clone https://www.modelscope.cn/ZhipuAI/glm-4v-9b.git

# MaskGCT

MaskGCT_path="/root/share_models/MaskGCT-models/"
mkdir -p $MaskGCT_path

cd $MaskGCT_path && git lfs install && git clone https://www.modelscope.cn/AI-ModelScope/MaskGCT.git
cd $MaskGCT_path && git lfs install && git clone https://www.modelscope.cn/AI-ModelScope/w2v-bert-2.0.git
cd $MaskGCT_path && git lfs install && git clone https://www.modelscope.cn/iic/Whisper-large-v3-turbo.git

# funasr

funasr_path="/root/share_models/funasr-models/"
mkdir -p $funasr_path
cd $funasr_path && git lfs install && git clone https://www.modelscope.cn/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch.git
cd $funasr_path && git lfs install && git clone https://www.modelscope.cn/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch.git
cd $funasr_path && git lfs install && git clone https://www.modelscope.cn/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch.git


# rerank
rerank_path="/root/share_models/Rerank-models/"
mkdir -p $rerank_path
cd $rerank_path && git lfs install && git clone https://www.modelscope.cn/BAAI/bge-reranker-base.git
cd $rerank_path && git lfs install && git clone https://www.modelscope.cn/BAAI/bge-reranker-v2-m3.git
cd $rerank_path && git lfs install && git clone https://www.modelscope.cn/mixedbread-ai/mxbai-rerank-xsmall-v1.git


# webui  && comfyui
webui_path="/root/share_models/webui-models/models" 
mkdir -p $webui_path
cd $webui_path && git lfs install && git clone https://www.modelscope.cn/chineking/adetailer.git
cd $webui_path && git lfs install && git clone https://www.modelscope.cn/licyks/controlnet_v1.1_annotator.git  # for webui controlnet 处理器, 挂载目录: /workspace/stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/downloads
cd $webui_path && git lfs install && git clone https://www.modelscope.cn/jackle/comfyui_controlnet_aux_ckpts.git # for webui controlnet 处理器，挂载目录: /workspace/ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts
cd $webui_path && git lfs install && git clone https://www.modelscope.cn/shareAI/lllyasviel-ControlNet-v1-1.git ControlNet # webui/comfyui 公用同一个controlnet
cd $webui_path && git lfs install && git clone https://www.modelscope.cn/AI-ModelScope/clip-vit-large-patch14.git