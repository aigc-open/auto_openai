# 存放零时数据的目录,图片等
mkdir -p /root/share_models/tmp

# Embedding
Embedding_path=/root/share_models/Embedding-models
mkdir -p $Embedding_path

# LLM
LLM_path=/root/share_models/LLM
mkdir -p $LLM_path

# maskgct
MaskGCT_path="/root/share_models/MaskGCT-models/"
mkdir -p $MaskGCT_path

# funasr
funasr_path="/root/share_models/funasr-models/"
mkdir -p $funasr_path

# rerank
rerank_path="/root/share_models/Rerank-models/"
mkdir -p $rerank_path

# webui && comfyui
webui_path="/root/share_models/webui-models/"
mkdir -p $webui_path

# 基础绘图模型
# cd $webui_path && git lfs install && git clone https://www.modelscope.cn/chineking/adetailer.git
# cd $webui_path && git lfs install && git clone https://www.modelscope.cn/licyks/controlnet_v1.1_annotator.git
# # for webui controlnet 处理器, 挂载目录: /workspace/stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/downloads
# cd $webui_path && git lfs install && git clone https://www.modelscope.cn/jackle/comfyui_controlnet_aux_ckpts.git
# # for ComfyUI controlnet 处理器，挂载目录: /workspace/ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts
# cd $webui_path && git lfs install && git clone https://www.modelscope.cn/shareAI/lllyasviel-ControlNet-v1-1.git ControlNet 
# # webui/comfyui 公用同一个controlnet
# cd $webui_path && git lfs install && git clone https://www.modelscope.cn/AI-ModelScope/clip-vit-large-patch14.git
# # webui 使用的clip,必须安装


cd $Embedding_path && git lfs install && git clone https://www.modelscope.cn/BAAI/bge-m3.git
cd $Embedding_path && git lfs install && git clone https://www.modelscope.cn/ai-modelscope/bge-base-zh-v1.5.git
cd $LLM_path && git lfs install && git clone https://www.modelscope.cn/Qwen/Qwen2-VL-7B-Instruct.git
cd $LLM_path && git lfs install && git clone https://www.modelscope.cn/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4.git
cd $LLM_path && git lfs install && git clone https://www.modelscope.cn/Qwen/Qwen2.5-72B-Instruct.git
cd $LLM_path && git lfs install && git clone https://www.modelscope.cn/Qwen/Qwen2.5-7B-Instruct.git
cd $LLM_path && git lfs install && git clone https://www.modelscope.cn/Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-Int4
cd $LLM_path && git lfs install && git clone https://www.modelscope.cn/Qwen/Qwen2.5-Coder-32B-Instruct.git
cd $LLM_path && git lfs install && git clone https://www.modelscope.cn/Qwen/Qwen2.5-Coder-32B.git
cd $LLM_path && git lfs install && git clone https://www.modelscope.cn/Qwen/Qwen2.5-Coder-7B-Instruct.git
cd $LLM_path && git lfs install && git clone https://www.modelscope.cn/Qwen/Qwen2.5-Coder-7B.git
cd $LLM_path && git lfs install && git clone https://www.modelscope.cn/ZhipuAI/codegeex4-all-9b.git
cd $LLM_path && git lfs install && git clone https://www.modelscope.cn/ZhipuAI/glm-4-9b-chat.git
cd $LLM_path && git lfs install && git clone https://www.modelscope.cn/ZhipuAI/glm-4v-9b.git
cd $LLM_path && git lfs install && git clone https://www.modelscope.cn/deepseek-ai/DeepSeek-Coder-V2-Lite-Base.git
cd $LLM_path && git lfs install && git clone https://www.modelscope.cn/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct.git
cd $LLM_path && git lfs install && git clone https://www.modelscope.cn/deepseek-ai/deepseek-coder-6.7b-base.git
cd $LLM_path && git lfs install && git clone https://www.modelscope.cn/deepseek-ai/deepseek-coder-6.7b-instruct.git
cd $LLM_path && git lfs install && git clone https://www.modelscope.cn/jackle/Qwen2.5-Coder-32B-GPTQ-Int4.git
cd $funasr_path && git lfs install && git clone https://www.modelscope.cn/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch.git
cd $funasr_path && git lfs install && git clone https://www.modelscope.cn/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch.git
cd $funasr_path && git lfs install && git clone https://www.modelscope.cn/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch.git
cd $rerank_path && git lfs install && git clone https://www.modelscope.cn/BAAI/bge-reranker-base.git
cd $rerank_path && git lfs install && git clone https://www.modelscope.cn/BAAI/bge-reranker-v2-m3.git
mkdir -p $webui_path/CogVideo 
cd $webui_path/CogVideo && git lfs install && git clone https://www.modelscope.cn/ZhipuAI/CogVideoX-5b.git
mkdir -p $webui_path/Stable-diffusion
cd $webui_path/Stable-diffusion && git lfs install && git clone https://www.modelscope.cn/GYMaster/majicmixRealistic_v7.safetensors.git