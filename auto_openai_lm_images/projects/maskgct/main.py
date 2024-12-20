import os


# Launch the interface
if os.environ.get("TOPS_VISIBLE_DEVICES") is not None:
    os.environ["ENFLAME_PT_OP_DEBUG_CONFIG"] = "fallback_cpu=convolution,topk_out"
    # 支持GCU算力卡
    try:
        import torch_gcu  # 导入 torch_gcu
        from torch_gcu import transfer_to_gcu  # 导入 transfer_to_gcu
    except Exception as e:
        raise e


def run(port=7861, model_root_path="../MaskGCT-models/"):
    os.environ["MASKGCT_ROOT_PATH"] = model_root_path
    try:
        from app import iface
        from fire import Fire
    except Exception as e:
        raise Exception(f"Please run this script in the main directory. {e}")

    iface.launch(allowed_paths=["./output"],
                 server_name="0.0.0.0", server_port=port)


if __name__ == "__main__":
    from fire import Fire
    Fire(run)
