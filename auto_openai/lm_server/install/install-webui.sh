
cd /workspace/ && git clone https://gitee.com/webui_1/stable-diffusion-webui.git
git config --global --add safe.directory /workspace/stable-diffusion-webui
root_path=/workspace/stable-diffusion-webui
mkdir -p /workspace/stable-diffusion-webui/extensions
mkdir -p /workspace/stable-diffusion-webui/repositories
# 内置仓库
cd $root_path/repositories && git clone https://gitee.com/webui_1/stable-diffusion-webui-assets.git 
cd $root_path/repositories && git clone https://gitee.com/webui_1/k-diffusion.git 
cd $root_path/repositories && git clone https://gitee.com/webui_1/BLIP.git && cd BLIP 
cd $root_path/repositories && git clone https://gitee.com/webui_1/generative-models.git && cd generative-models
cd $root_path/repositories && git clone https://gitee.com/webui_1/stable-diffusion-stability-ai.git 
# 插件
cd $root_path/extensions && git clone https://gitee.com/webui_1/sd-webui-controlnet.git

# 依赖
site_packages=/workspace/webui-site-packages/ 
rm -rf $site_packages
mkdir -p $site_packages
pip install -r /workspace/stable-diffusion-webui/requirements_versions.txt -t $site_packages --upgrade
pip install -r $root_path/repositories/k-diffusion/requirements.txt -t $site_packages --upgrade
pip install -r $root_path/repositories/stable-diffusion-stability-ai/requirements.txt -t $site_packages --upgrade
pip install -r $root_path/extensions/sd-webui-controlnet/requirements.txt -t $site_packages --upgrade


rm -rf $site_packages/torch
rm -rf $site_packages/torch-*
rm -rf $site_packages/torchvision*
