import os
import sys
import shutil
from auto_openai import project_path
from auto_openai.utils.init_env import global_config
ROOT_PATH = global_config.COMFYUI_ROOT_PATH
shutil.copyfile(os.path.join(project_path, "lm_server/comfyui_modify/scatter_gather.py"),
                os.path.join(ROOT_PATH, "custom_nodes/comfyui_controlnet_aux/src/custom_mmpkg/custom_mmcv/parallel/scatter_gather.py"))


model_path_yaml = os.path.join(project_path, "lm_server/comfyui_modify/extra_model_paths.yaml")
with open(model_path_yaml, "r") as f:
    model_path_yaml_content = f.read()
    model_path_yaml_content = model_path_yaml_content.replace("{base_path}", global_config.COMFYUI_MODEL_ROOT_PATH)


with open(os.path.join(ROOT_PATH, "extra_model_paths.yaml"), "w") as f:
    f.write(model_path_yaml_content)