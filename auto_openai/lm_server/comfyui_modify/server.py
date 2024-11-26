import runpy
import os
import sys
import torch
import shutil
from auto_openai import project_path
from auto_openai.utils.init_env import global_config
ROOT_PATH = global_config.COMFYUI_ROOT_PATH

if os.environ.get("TOPS_VISIBLE_DEVICES") is not None:
    # 支持GCU算力卡
    try:
        import torch_gcu  # 导入 torch_gcu
        from torch_gcu import transfer_to_gcu  # 导入 transfer_to_gcu
        sys.path.append(ROOT_PATH)
        from main import *
    except Exception as e:
        raise e

runpy.run_path(F"{ROOT_PATH}/main.py",
               run_name="__main__")
