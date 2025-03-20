cd /workspace/ && git clone https://gitee.com/ComfyUI_1/ComfyUI.git && pip3.10 install -r /workspace/ComfyUI/requirements.txt
git config --global --add safe.directory /workspace/ComfyUI

root_path=/workspace/ComfyUI
mkdir -p /workspace/ComfyUI/custom_nodes
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
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI_UltimateSDUpscale --recursive
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/comfyui_controlnet_aux && cd comfyui_controlnet_aux && pip install -r requirements.txt
# https://www.modelscope.cn/models/depth-anything/Depth-Anything-V2-Large/files
# https://www.modelscope.cn/models/jackle/comfyui_controlnet_aux_ckpts/files
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/stability-ComfyUI-nodes && cd stability-ComfyUI-nodes && pip install -r requirements.txt 
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
# https://www.modelscope.cn/models/fireicewolf/wd-v1-4-moat-tagger-v2/files
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI_FizzNodes.git && cd ComfyUI_FizzNodes && pip install -r requirements.txt
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI-Allor.git && cd ComfyUI-Allor && pip install -r requirements.txt
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI_InstantID.git && cd ComfyUI_InstantID && pip install -r requirements.txt
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/PuLID_ComfyUI.git && cd PuLID_ComfyUI && pip install -r requirements.txt && pip install facexlib
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/AIGODLIKE-ComfyUI-Translation.git
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI_ADV_CLIP_emb.git
# cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/cg-use-everywhere.git # 在0.3.0以后不能使用
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI-Kolors-MZ.git
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/comfyui_segment_anything.git && cd comfyui_segment_anything && pip install -r requirements.txt
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI-AdvancedLivePortrait.git && cd ComfyUI-AdvancedLivePortrait && pip install -r requirements.txt
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI-LivePortraitKJ.git && cd ComfyUI-LivePortraitKJ && pip install -r requirements.txt && pip install pykalman
# 客户
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI-Florence2.git && cd ComfyUI-Florence2 && pip install -r requirements.txt
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI_MiniCPM-V-2_6-int4.git && cd ComfyUI_MiniCPM-V-2_6-int4 && pip install -r requirements.txt
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI-Advanced-ControlNet.git && cd ComfyUI-Advanced-ControlNet && pip install -r requirements.txt
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI-BiRefNet-Hugo.git && cd ComfyUI-BiRefNet-Hugo && pip install -r requirements.txt
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/Comfyui_mobilesam.git && cd Comfyui_mobilesam && pip install -r requirements.txt
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/was-node-suite-comfyui.git && cd was-node-suite-comfyui && pip install -r requirements.txt
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/efficiency-nodes-comfyui.git && cd efficiency-nodes-comfyui && pip install -r requirements.txt
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/comfyui-tooling-nodes.git && cd comfyui-tooling-nodes && pip install -r requirements.txt
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI_HFDownLoad.git && cd ComfyUI_HFDownLoad && pip install -r requirements.txt
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/masquerade-nodes-comfyui.git 
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/comfyui-mixlab-nodes.git && cd comfyui-mixlab-nodes && pip install -r requirements.txt
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI-Openpose-Editor-Plus.git && cd ComfyUI-Openpose-Editor-Plus && pip install pyOpenSSL watchdog
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/Comfyui_cgem156.git && cd Comfyui_cgem156 && pip install -r requirements.txt
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI-YOLO.git && cd ComfyUI-YOLO && pip install -r requirements.txt
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI-Custom-Scripts.git && cd ComfyUI-Custom-Scripts && pip install -r requirements.txt
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI-Universal-Styler.git && cd ComfyUI-Universal-Styler && pip install -r requirements.txt
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI-Easy-Use.git && cd ComfyUI-Easy-Use && pip install -r requirements.txt
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/comfyui-lama-remover.git && cd comfyui-lama-remover && pip install -r requirements.txt
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/comfyui-gpt-agent.git && cd comfyui-gpt-agent && pip install -r requirements.txt
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/comfyui-mediapipe.git && cd comfyui-mediapipe && pip install -r requirements.txt
cd $root_path/custom_nodes && git clone https://gitee.com/ComfyUI_1/ComfyUI-EmptyHunyuanLatent.git