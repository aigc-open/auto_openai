
import requests
import os
base_url = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:9000/openai/v1")

########################### ###########################
for model in [
    "Qwen2.5-32B-Instruct-GPTQ-Int4",
    "Qwen2.5-Coder-14B-Instruct",
    "DeepSeek-Coder-V2-Lite-Instruct",
    "deepseek-coder-6.7b-instruct",
    "Qwen2.5-7B-Instruct"
]:

    data = {"model": model,
            "messages": [
                {"role": "user", "content": "What are some fun things to do in New York?"}],
            "max_tokens": 1024,
            "temperature": 0.0,
            "stream": False}
    resp = requests.post(base_url + "/chat/completions", json=data)
    print(resp.text)
    print("#"*50)
