import os
import wget
base_url = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:9000/openai/v1")
api_key = "xxxx"
########################### ###########################
from openai import OpenAI
client = OpenAI(base_url=base_url, api_key=api_key)

#################################################################
# https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav
if not os.path.exists("./guodegang.wav"):
    wget.download("https://gitee.com/lijiacai/static-files/releases/download/v0.1/guodegang.wav", "./guodegang.wav")
with open("./guodegang.wav", "rb") as f:
    response = client.audio.transcriptions.create(
        model="funasr",
        file=f
    )
    print(response)

# Transcription(text='嗯，你好，介绍一下你自己呢。')
