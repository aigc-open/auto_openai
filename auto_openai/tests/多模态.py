import os
base_url = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:9000/openai/v1")
api_key = "xxxx"
########################### ###########################
if os.environ.get("OPENAI"):
    from openai import OpenAI
else:
    from together import Together as OpenAI
client = OpenAI(base_url=base_url, api_key=api_key)




stream = client.chat.completions.create(
    model="glm-4v-9b",
    messages=[
        {
            "role": "user",
            "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
                        },
                    },
                {"type": "text", "text": "描述这个图片"},
            ],
        }
    ],
    stream=True,
    temperature=0.0,
    max_tokens=128,
    presence_penalty=1.0,
    frequency_penalty=1.0
)

for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)

# 这张照片展示了海滩上的一个场景。具体而言，一位穿着黑色牛仔裤和格子衬衫的女子坐在沙滩上，她的右手放在一只黄色拉布拉多犬的身上，狗狗戴着蓝色的胸背带，女子面带微笑，看起来很开心。
