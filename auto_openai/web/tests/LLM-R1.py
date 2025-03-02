
from openai import OpenAI
import os
base_url = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:9000/openai/v1")
api_key = "xxxx"
########################### ###########################
client = OpenAI(base_url=base_url, api_key=api_key)

########################### 深度思考模式 ###########################
model = "DeepSeek-R1-Distill-Qwen-32B:10k"
# model = "Qwen2.5-Coder-7B-Instruct"
# model = "Qwen2.5-Coder-7B-Instruct:SR"
response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": "请你介绍一下人工智能"}],
    max_tokens=1024,
    temperature=0.0,
    stream=True,
)
print("思考中...")
print_content = True
for chunk in response:
    if chunk.choices[0].delta.reasoning_content:
        print(chunk.choices[0].delta.reasoning_content or "", end="", flush=True)
    elif chunk.choices[0].delta.content:
        if print_content:
            print("结果：")
        print(chunk.choices[0].delta.content or "", end="", flush=True)
        print_content = False
print()