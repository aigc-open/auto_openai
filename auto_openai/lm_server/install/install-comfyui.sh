cd /workspace/ && git clone https://gitee.com/ComfyUI_1/ComfyUI.git && pip3.10 install -r /workspace/ComfyUI/requirements.txt
git config --global --add safe.directory /workspace/ComfyUI
root_path=/workspace/ComfyUI
mkdir -p /workspace/ComfyUI/custom_nodes
# 插件
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI_FaceAnalysis.git && cd ComfyUI_FaceAnalysis && pip install -r requirements.txt
# cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI-Manager.git
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI-Impact-Pack && cd ComfyUI-Impact-Pack # && python install.py
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI-Inspire-Pack && cd ComfyUI-Inspire-Pack && pip install -r requirements.txt
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI-Frame-Interpolation && cd ComfyUI-Frame-Interpolation # && python install.py
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI-Video-Matting && cd ComfyUI-Video-Matting && pip install -r requirements.txt
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI_Cutoff
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/PPF_Noise_ComfyUI && cd PPF_Noise_ComfyUI && pip install -r requirements.txt
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/PowerNoiseSuite && cd PowerNoiseSuite && pip install -r requirements.txt
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/comfy-plasma
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI_Comfyroll_CustomNodes
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI-OpenPose-Editor
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/sdxl_prompt_styler
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI-AnimateDiff-Evolved
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/OneButtonPrompt
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/was-node-suite-comfyui && cd was-node-suite-comfyui && pip install -r requirements.txt
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI_essentials
# cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI_UltimateSDUpscale --recursive
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/comfyui_controlnet_aux && cd comfyui_controlnet_aux && pip install -r requirements.txt
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/stability-ComfyUI-nodes && cd stability-ComfyUI-nodes && pip install -r requirements.txt 
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/efficiency-nodes-comfyui && cd efficiency-nodes-comfyui && pip install -r requirements.txt 
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI-VideoHelperSuite && cd ComfyUI-VideoHelperSuite && pip install -r requirements.txt 
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI-Custom-Scripts
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/FreeU_Advanced
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/SD-Advanced-Noise
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI_Custom_Nodes_AlekPet
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyMath && cd ComfyMath && pip install -r requirements.txt 
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/rgthree-comfy && cd rgthree-comfy && pip install -r requirements.txt 
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/comfy-image-saver && cd comfy-image-saver && pip install -r requirements.txt 
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI-Depth-Visualization && cd ComfyUI-Depth-Visualization && pip install -r requirements.txt
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ControlNet-LLLite-ComfyUI
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI-Dream-Interpreter && cd ComfyUI-Dream-Interpreter && pip install -r requirements.txt
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI_IPAdapter_plus
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI-Advanced-ControlNet && cd ComfyUI-Advanced-ControlNet && pip install -r requirements.txt 
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/comfyui-inpaint-nodes 
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI_LayerStyle  && cd ComfyUI_LayerStyle && pip install -r requirements.txt 
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/Derfuu_ComfyUI_ModdedNodes
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI-post-processing-nodes
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/comfy_mtb  && cd comfy_mtb && pip install -r requirements.txt 
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI-KJNodes && cd ComfyUI-KJNodes && pip install -r requirements.txt 
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI-SUPIR && cd ComfyUI-SUPIR && pip install -r requirements.txt
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI-depth-fm && cd ComfyUI-depth-fm && pip install -r requirements.txt
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI-BiRefNet && cd ComfyUI-BiRefNet && pip install -r requirements.txt
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI-Texture-Simple
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI-APISR && cd ComfyUI-APISR && pip install -r requirements.txt
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI-WD14-Tagger.git && cd ComfyUI-WD14-Tagger && pip install -r requirements.txt
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI_FizzNodes.git && cd ComfyUI_FizzNodes && pip install -r requirements.txt
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI-Allor.git && cd ComfyUI-Allor && pip install -r requirements.txt
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI_InstantID.git && cd ComfyUI_InstantID && pip install -r requirements.txt
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/PuLID_ComfyUI.git && cd PuLID_ComfyUI && pip install -r requirements.txt
# cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/StableZero123-comfyui.git && cd StableZero123-comfyui && pip install -r requirements.txt
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/AIGODLIKE-ComfyUI-Translation.git
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI_ADV_CLIP_emb.git
# cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI-Easy-Use.git && cd ComfyUI-Easy-Use && pip install -r requirements.txt
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/cg-use-everywhere.git
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI-Kolors-MZ.git
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/comfyui_segment_anything.git && cd comfyui_segment_anything && pip install -r requirements.txt
# cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/comfyui-ollama.git && cd comfyui-ollama && pip install -r requirements.txt
# cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI-OOTDiffusion.git && cd ComfyUI-OOTDiffusion && pip install -r requirements.txt
cd $root_path/custom_nodes && git clone https://gitee.com/python279/ComfyUI-AdvancedLivePortrait.git && cd ComfyUI-AdvancedLivePortrait && pip install -r requirements.txt
cd $root_path/custom_nodes && git clone https://gitee.com/yubiaohyb/ComfyUI-LivePortraitKJ.git && cd ComfyUI-LivePortraitKJ && pip install -r requirements.txt