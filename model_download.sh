
# Embedding
Embedding_path=/root/share_models/Embedding-models

cd $Embedding_path && git lfs install && git clone https://www.modelscope.cn/ai-modelscope/bge-base-zh-v1.5.git
cd $Embedding_path && git lfs install && git clone https://www.modelscope.cn/BAAI/bge-m3.git

# LLM
LLM_path=/root/share_models/LLM

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

cd $MaskGCT_path && git lfs install && git clone https://www.modelscope.cn/AI-ModelScope/MaskGCT.git
cd $MaskGCT_path && git lfs install && git clone https://www.modelscope.cn/AI-ModelScope/w2v-bert-2.0.git
cd $MaskGCT_path && git lfs install && git clone https://www.modelscope.cn/iic/Whisper-large-v3-turbo.git

# funasr

funasr_path="/root/share_models/funasr-models/"
cd $funasr_path && git lfs install && git clone https://www.modelscope.cn/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch.git
cd $funasr_path && git lfs install && git clone https://www.modelscope.cn/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch.git
cd $funasr_path && git lfs install && git clone git clone https://www.modelscope.cn/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch.git