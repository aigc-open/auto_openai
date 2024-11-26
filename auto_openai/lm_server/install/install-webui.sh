
cd /workspace/ && git clone https://gitee.com/webui_1/stable-diffusion-webui.git && pip3.10 install -r /workspace/stable-diffusion-webui/requirements.txt 
git config --global --add safe.directory /workspace/stable-diffusion-webui
root_path=/workspace/stable-diffusion-webui
mkdir -p /workspace/stable-diffusion-webui/extensions
mkdir -p /workspace/stable-diffusion-webui/repositories
# 内置仓库
cd $root_path/repositories && git clone https://gitee.com/webui_1/stable-diffusion-webui-assets.git 
cd $root_path/repositories && git clone https://gitee.com/webui_1/k-diffusion.git && cd k-diffusion && pip install -r requirements.txt
cd $root_path/repositories && git clone https://gitee.com/webui_1/BLIP.git && cd BLIP 
cd $root_path/repositories && git clone https://gitee.com/webui_1/generative-models.git && cd generative-models
cd $root_path/repositories && git clone https://gitee.com/webui_1/stable-diffusion-stability-ai.git && cd stable-diffusion-stability-ai
# 插件
cd $root_path/extensions && git clone https://gitee.com/webui_1/sd-webui-controlnet.git && cd sd-webui-controlnet && pip install -r requirements.txt

# 特定依赖
rm -rf /workspace/webui-site-packages/ && mkdir -p /workspace/webui-site-packages/ && pip install gradio==3.41.0 gradio_client==0.5 pydantic==1.10.17 fastapi==0.94.0 --upgrade -t /workspace/webui-site-packages/