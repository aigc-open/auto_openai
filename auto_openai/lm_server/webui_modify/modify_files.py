import os
import sys
import torch
import shutil
from auto_openai import project_path
from auto_openai.utils.init_env import global_config
current_dir = "webui_modify"

ROOT_PATH = global_config.WEBUI_ROOT_PATH
# cp webui_modify/sd_vae_approx.py /workspace/stable-diffusion-webui/modules
shutil.copyfile(os.path.join(project_path, f"lm_server/{current_dir}/modules/sd_vae_approx.py"),
                os.path.join(ROOT_PATH, "modules/sd_vae_approx.py"))
shutil.copyfile(os.path.join(project_path, f"lm_server/{current_dir}/stable-diffusion-stability-ai/ldm/modules/encoders/modules.py"),
                os.path.join(ROOT_PATH, "repositories/stable-diffusion-stability-ai/ldm/modules/encoders/modules.py"))
shutil.copyfile(os.path.join(project_path, f"lm_server/{current_dir}/modules/devices.py"),
                os.path.join(ROOT_PATH, "modules/devices.py"))
