import json
import time
import urllib.parse
import urllib.request
import uuid
import random
import os
import requests
from PIL import Image
import io
from auto_openai.utils.oss_client import OSSManager
import webuiapi


class WebUIClient:

    def __init__(self, server: str, s3_client: OSSManager = None):
        # server = "http://127.0.0.1:7861"
        if "//" in server:
            base_host = server.split("//")[1]
        else:
            base_host = server
        host = base_host.split(":")[0]
        port = base_host.split(":")[1].split("/")[0]
        self.client = webuiapi.WebUIApi(host=host, port=port)
        self.s3_client = s3_client

    def warmup(self):
        self.client.txt2img(
            prompt="cute squirrel",
            negative_prompt="ugly, out of frame",
            seed=1003,
            steps=3
        )

    def infer(self, webui_data: dict, bucket_name):
        self.client.util_set_model(name=webui_data["model"])
        del webui_data["model"]
        if webui_data.get("images"):
            result = self.client.img2img(**webui_data)
        else:
            result = self.client.txt2img(**webui_data)
        out = []
        batch_size = webui_data.get("batch_size", 1)
        for image in result.images[:batch_size]:
            image_bytes = io.BytesIO()
            image.save(image_bytes, format='PNG')  # 可以根据需要更改格式
            image_bytes.seek(0)  # 重置字节流的位置
            local_path = f"tmp/{str(int(time.time()))+str(random.randint(0, 1000000))}.png"
            self.s3_client.upload_fileobj(image_bytes, bucket_name, local_path)
            url = self.s3_client.get_download_url(
                bucket_name, local_path)
            out.append({"url": url})
        return out
