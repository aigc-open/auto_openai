import runpy
import os
import sys
import torch
if os.environ.get("TOPS_VISIBLE_DEVICES") is not None:
    # 支持GCU算力卡
    try:
        import torch_gcu  # 导入 torch_gcu
        from torch_gcu import transfer_to_gcu  # 导入 transfer_to_gcu
    except Exception as e:
        raise e

runpy.run_path(F"./launch.py",
               run_name="__main__")
