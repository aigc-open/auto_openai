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
from websockets.sync.client import connect
from auto_openai.utils.oss_client import OSSManager


class WebUIClient:

    def __init__(self, server: str, s3_client: OSSManager = None):
        self.SERVER_ADDRESS = server
        self.CLIENT_ID = str(uuid.uuid4())
        self.session = None
        self.ws = None
        self.headers = {}
        self.connect()
        self.s3_client = s3_client

    def infer(self, json_data, bucket_name):
        outputs = self.wait_for_complete(json_data)
        out = []
        for node_id, data in outputs.items():
            if 'images' in data:
                images = data['images']
                for image in images:
                    # 将image["filename"]内容上传到s3上
                    local_path = f"tmp/{str(int(time.time()))+str(random.randint(0, 1000000))}.png"
                    # image_ = Image.open(io.BytesIO(self.get_artifact(
                    #     image["filename"], image["subfolder"], image["type"])))
                    # image_.save(local_path)
                    self.s3_client.upload_fileobj(io.BytesIO(self.get_artifact(
                        image["filename"], image["subfolder"], image["type"])), bucket_name, local_path)
                    url = self.s3_client.get_download_url(
                        bucket_name, local_path)
                    out.append({"url": url})
            elif 'gifs' in data:
                gifs = data['gifs']
                for gif in gifs:
                    # read mp4 data from comfyui server
                    video = self.get_artifact(
                        gif["filename"], gif["subfolder"], gif["type"])
                    with open(gif["filename"], "wb") as f:
                        f.write(video)
        return out

