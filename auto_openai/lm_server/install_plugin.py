import os
from pydantic import BaseModel
from enum import Enum
from fire import Fire
from auto_openai import project_path
import pip


class Plugin:

    ############## 第三方项目 ##############
    @classmethod
    def comfyui(cls):
        path = os.path.join(project_path,
                            "lm_server/install/install-comfyui.sh")
        os.system(f"bash {path}")
        os.system("python3 -m auto_openai.lm_server.comfyui_modify.modify_files")

    @classmethod
    def tiktoken(cls):
        import tiktoken
        import time
        start_time = time.time()
        tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
        print(f"titoken available, time: {time.time() - start_time}")

    @classmethod
    def maskgct(cls):
        path = os.path.join(
            project_path, "lm_server/install/install-maskgct.sh")
        os.system(f"bash {path}")
        os.system("python3 -m auto_openai.lm_server.maskgct_modify.modify_files")

    @classmethod
    def webui(cls):
        path = os.path.join(project_path, "lm_server/install/install-webui.sh")
        os.system(f"bash {path}")
        os.system("python3 -m auto_openai.lm_server.webui_modify.modify_files")

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


if __name__ == "__main__":
    Fire(Plugin)
