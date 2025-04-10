
from openai import OpenAI
import os
base_url = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:9000/openai/v1")
api_key = "xxxx"
########################### ###########################
client = OpenAI(base_url=base_url, api_key=api_key)


########################### 基础模式 ###########################
response = client.chat.completions.create(
    model="Qwen2.5-7B-Instruct",
    messages=[
        {"role": "user", "content": "What are some fun things to do in New York?"}],
    max_tokens=204096,
    temperature=0.0,
    stream=True,
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
print()
########################### 深度思考模式 ###########################
response = client.chat.completions.create(
    model="DeepSeek-R1-Distill-Qwen-32B:32k",
    messages=[
        {"role": "user", "content": "What are some fun things to do in New York?"}],
    max_tokens=204096,
    temperature=0.0,
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.reasoning_content:
        print(chunk.choices[0].delta.reasoning_content or "", end="", flush=True)
    else:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
print()
########################### 续写模式 ###########################
response = client.completions.create(
    model="Qwen2.5-7B-Instruct",
    prompt="def print_hello",
    max_tokens=128,
    temperature=0.0,
    stream=True,
)

for chunk in response:
    print(chunk.choices[0].text or "", end="", flush=True)
print()

########################### tools 调用 ###########################

response = client.chat.completions.create(
    model="Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "How's the weather in Hangzhou?"}],
    max_tokens=204096,
    temperature=0.0,
    stream=False,
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather of an location, the user shoud supply a location first",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        }
                    },
                    "required": ["location"]
                },
            }
        },
    ]
)
print(response)
