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


class ComfyUIClient:

    def __init__(self, server: str, s3_client: OSSManager = None):
        self.SERVER_ADDRESS = server
        self.CLIENT_ID = str(uuid.uuid4())
        self.session = None
        self.ws = None
        self.headers = {}
        self.connect()
        self.s3_client = s3_client

    def connect(self):
        self.session = requests.Session()
        self.session.get(
            f"http://{self.SERVER_ADDRESS}/ws?clientId={self.CLIENT_ID}", headers=self.headers)
        self.ws = connect("ws://{}/ws?clientId={}".format(self.SERVER_ADDRESS, self.CLIENT_ID),
                          additional_headers=self.headers)

    def close(self):
        if self.session is not None:
            self.session.close()
            self.session = None
        if self.ws is not None:
            self.ws.close()
            self.ws = None

    def queue_prompt(self, prompt):
        payload = {"prompt": prompt, "client_id": self.CLIENT_ID}
        data = json.dumps(payload).encode('utf-8')
        response = self.session.post(
            f"http://{self.SERVER_ADDRESS}/prompt", data=data, headers=self.headers)
        return response.json()

    def upload_image(self, image_url: str):
        # get image data from image_url
        image_data = requests.get(image_url).content
        image_name = uuid.uuid4().hex

        # upload image to comfyui server
        resp = self.session.post(f"http://{self.SERVER_ADDRESS}/upload/image",
                                 files={'image': (f"{image_name}.png", image_data)}, data={"subfolder": "temp"}, headers=self.headers)

        # return image path
        resp_json = json.loads(resp.content.decode('utf-8'))
        return resp_json.get('subfolder') + '/' + resp_json.get('name')

    def get_artifact(self, filename, subfolder, folder_type):
        params = {"filename": filename,
                  "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(params)
        response = self.session.get(
            f"http://{self.SERVER_ADDRESS}/view?{url_values}", headers=self.headers)
        return response.content

    def get_history(self, prompt_id):
        response = self.session.get(
            f"http://{self.SERVER_ADDRESS}/history/{prompt_id}", headers=self.headers)
        return response.json()

    def wait_for_complete(self, prompt):
        prompt_id = self.queue_prompt(prompt).get('prompt_id')

        while True:
            out = self.ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing':
                    data = message['data']
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        break  # Execution is done
            else:
                continue  # previews are binary data

        while True:
            history = self.get_history(prompt_id)
            if prompt_id in history and 'outputs' in history[prompt_id]:
                break
            time.sleep(1)

        return history[prompt_id]['outputs']

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


def get_workflow_json(workflow_api_path: str = "", workflow_api_json: dict = {}):
    if workflow_api_path:
        with open(workflow_api_path, "r") as f:
            workflow_api_json = json.loads(f.read())
    return workflow_api_json


def test():
    os.makedirs("./tmp", exist_ok=True)
    prompt = get_workflow_json(workflow_api_path="workflow_api.json")
    prompt["3"]["inputs"]["seed"] = int(time.time()) + random.randint(0, 100)
    cf = ComfyUIClient("58.247.94.54:8083")
    try:
        outputs = cf.wait_for_complete(prompt)
        for node_id, data in outputs.items():
            if 'images' in data:
                images = data['images']
                for image in images:
                    from PIL import Image
                    import io
                    i = Image.open(io.BytesIO(cf.get_artifact(
                        image["filename"], image["subfolder"], image["type"])))
                    i.save(
                        f"./tmp/{str(int(time.time()))+str(random.randint(0, 1000000))}.png")
            elif 'gifs' in data:
                gifs = data['gifs']
                for gif in gifs:
                    # read mp4 data from comfyui server
                    video = cf.get_artifact(
                        gif["filename"], gif["subfolder"], gif["type"])
                    with open(gif["filename"], "wb") as f:
                        f.write(video)
    finally:
        cf.close()


def multi_test():

    import time
    import threading
    import random
    import json
    import os
    start_time = time.time()
    count = 10
    # 多线程并发测试，并统计时间和平均时间，以及每个时间, 调用test()
    for i in range(count):
        t = threading.Thread(target=test)
        t.start()
        # time.sleep(random.uniform(0.1, 1))
    t.join()
    print("done")
    print(f"count: {count}")
    print(f"total time: {time.time() - start_time}")
    print(f"average time: {(time.time() - start_time) / count}")


if __name__ == '__main__':
    multi_test()
