root_path=/workspace/ComfyUI
modify_path=/modify
cp -f $modify_path/scatter_gather.py $root_path/custom_nodes/comfyui_controlnet_aux/src/custom_mmpkg/custom_mmcv/parallel/scatter_gather.py
cp -f $modify_path/extra_model_paths.yaml $root_path/extra_model_paths.yaml
cp -f $modify_path/comfyui-main.py $root_path/comfyui-main.py