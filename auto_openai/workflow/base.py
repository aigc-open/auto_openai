import urllib.parse
from auto_openai.utils.openai import gen_random_uuid
from auto_openai.utils.init_env import global_config
import os


class UrlParser:
    def __init__(self, url):
        self.url = url
        self.parts = urllib.parse.urlparse(url)
        self.path = self.parts.path.strip("/")
        self.query = self.parts.query
        self.query_dict = urllib.parse.parse_qs(self.query)
        self.dirs_count = len(self.path.split("/"))
        self.file_name = self.path.split("/")[-1]
        self.file_name = self.file_name.split("?")[0]
        self.file_name = self.file_name.split("#")[0]
        self.file_extension = self.file_name.split(".")[-1]
        self.file_name = self.file_name.split(".")[0]
        self.file_name = self.file_name.replace("-", "_")

    def generate_random_local_file_name(self):
        return os.path.join(global_config.COMFYUI_INPUTS_DIR, f"{gen_random_uuid()}.{self.file_extension}")


def get_model_names(type_):
    model_names = []
    for model_ in global_config.SD_MODELS:
        if model_["name"] == type_:
            for _model_ in model_["model_list"]:
                model_names.append(_model_["name"])
    if not model_names:
        raise Exception(f"Model type {type_} not found")
    return model_names