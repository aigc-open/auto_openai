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