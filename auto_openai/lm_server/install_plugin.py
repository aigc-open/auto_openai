import os
from pydantic import BaseModel
from enum import Enum
from fire import Fire
from auto_openai import project_path
import pip


class Plugin:

    ############## 第三方项目 ##############
    @classmethod
    def comfyui(cls, skip_requirements: bool = False):
        path = os.path.join(project_path,
                            "lm_server/install/install-comfyui.sh")
        if skip_requirements is False:
            os.system(f"bash {path}")
        os.system("python3 -m auto_openai.lm_server.comfyui_modify.modify_files")
        os.system(
            "rm -rf /workspace/ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts")
        os.system("ln -s /root/share_models/webui-models/comfyui_controlnet_aux_ckpts /workspace/ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts")

    @classmethod
    def tiktoken(cls):
        import tiktoken
        import time
        start_time = time.time()
        tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
        print(f"titoken available, time: {time.time() - start_time}")

    @classmethod
    def maskgct(cls, skip_requirements: bool = False):
        path = os.path.join(
            project_path, "lm_server/install/install-maskgct.sh")
        if skip_requirements is False:
            os.system(f"bash {path}")
        os.system("python3 -m auto_openai.lm_server.maskgct_modify.modify_files")

    @classmethod
    def webui(cls, skip_requirements: bool = False):
        path = os.path.join(project_path, "lm_server/install/install-webui.sh")
        if skip_requirements is False:
            os.system(f"bash {path}")
        os.system("python3 -m auto_openai.lm_server.webui_modify.modify_files")
        os.system(
            "rm -rf /workspace/stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/downloads")
        os.system("ln -s /root/share_models/webui-models/controlnet_v1.1_annotator/ /workspace/stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/downloads")

    ############## 自定义项目 ##############

    @classmethod
    def embedding(cls):
        path = os.path.join(
            project_path, "lm_server/install/install-embedding")
        os.system(f"cp -rf {path} /workspace")
        os.system(f"pip install -r {path}/requirements.txt")

    @classmethod
    def funasr(cls):
        path = os.path.join(project_path, "lm_server/install/install-funasr")
        os.system(f"cp -rf {path} /workspace")
        os.system(f"pip install -r {path}/requirements.txt")

    @classmethod
    def llm_transformer(cls):
        path = os.path.join(
            project_path, "lm_server/install/install-llm-transformer-server")
        os.system(f"cp -rf {path} /workspace")
        os.system(f"pip install -r {path}/requirements.txt")

    @classmethod
    def rerank(cls):
        path = os.path.join(
            project_path, "lm_server/install/install-rerank")
        os.system(f"cp -rf {path} /workspace")
        os.system(f"pip install -r {path}/requirements.txt")


    @classmethod
    def diffusers(cls):
        path = os.path.join(
            project_path, "lm_server/install/install-diffusers-server")
        os.system(f"cp -rf {path} /workspace")
        os.system(f"pip install -r {path}/requirements.txt")


if __name__ == "__main__":
    Fire(Plugin)
