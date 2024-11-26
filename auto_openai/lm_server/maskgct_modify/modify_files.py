import os
import sys
import torch
import shutil
from auto_openai import project_path
from auto_openai.utils.init_env import global_config
current_dir = "maskgct_modify"

ROOT_PATH = global_config.MASKGCT_ROOT_PATH
shutil.copyfile(os.path.join(project_path, f"lm_server/{current_dir}/app.py"),
                os.path.join(ROOT_PATH, "app.py"))

shutil.copyfile(os.path.join(project_path, f"lm_server/{current_dir}/main.py"),
                os.path.join(ROOT_PATH, "main.py"))

shutil.copyfile(os.path.join(project_path, f"lm_server/{current_dir}/vocos.py"),
                os.path.join(ROOT_PATH, "models/codec/amphion_codec/vocos.py"))
