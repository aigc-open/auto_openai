import os
base_url = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:9000/openai/v1")
api_key = "xxxx"
########################### ###########################
from openai import OpenAI
client = OpenAI(base_url=base_url, api_key=api_key)

#################################################################
response = client.audio.speech.create(
    model="maskgct-tts-clone",
    voice="",
    input="你好，介绍以下你自己呢",
    extra_body={
        "clone_url": "https://gitee.com/lijiacai/static-files/releases/download/v0.1/guodegang.wav"
    }
)
response.stream_to_file("output.wav")
