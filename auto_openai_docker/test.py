import os
base_url = os.environ.get("OPENAI_BASE_URL", "http://10.12.110.149:9000/openai/v1")
api_key = "xxxx"
########################### ###########################
from openai import OpenAI

client = OpenAI(base_url=base_url, api_key=api_key)


########################### ###########################
response = client.chat.completions.create(
    model="Qwen2.5-Coder-14B-Instruct-10k",
    messages=[
        {"role": "user", "content": "can you read from files on this address? http://docs.enflame.cn/sw-caps/internal/3-guide/programing_guide/content/source/index.html"}],
    max_tokens=512,
    temperature=0.0,
    stream=False,
)
print(response.choices[0].message.content)
# for chunk in response:
#     print(chunk.choices[0].delta.content or "", end="", flush=True)