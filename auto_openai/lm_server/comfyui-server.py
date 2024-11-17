import runpy
import os
import sys
import torch
COMFYUI_ROOT_PATH = os.environ.get("COMFYUI_ROOT_PATH", "/workspace/ComfyUI")
if os.environ.get("TOPS_VISIBLE_DEVICES") is not None:
    # 支持GCU算力卡
    try:
        import torch_gcu  # 导入 torch_gcu
        from torch_gcu import transfer_to_gcu  # 导入 transfer_to_gcu
        sys.path.append(COMFYUI_ROOT_PATH)
        from main import *
    except Exception as e:
        raise e

runpy.run_path(F"{COMFYUI_ROOT_PATH}/main.py",
               run_name="__main__")
