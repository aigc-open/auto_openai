import os
import requests
base_url = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:9000/openai/v1")
api_key = "xxxx"
########################### ###########################

#################################################################
data = {"model": "CogVideo/CogVideoX-5b",
        "prompt": "A man is playing a guitar in a park",
        "width": 480,
        "height": 720,
        "num_frames": 16}
resp = requests.post(base_url + "/video/generations", json=data)
print(resp.json())
