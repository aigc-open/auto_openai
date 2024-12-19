import os
import requests
base_url = os.environ.get("OPENAI_BASE_URL", "https://auto-openai.cpolar.cn/openai/v1")
api_key = "xxxx"
########################### ###########################
if os.environ.get("OPENAI"):
    from openai import OpenAI
else:
    from together import Together as OpenAI

client = OpenAI(base_url=base_url, api_key=api_key)


########################### ###########################
# response = client.chat.completions.create(
#     model="Qwen2.5-7B-Instruct",
#     messages=[
#         {"role": "user", "content": "你是谁呢"}],
#     max_tokens=512,
#     temperature=0.0,
#     stream=True,
# )
# for chunk in response:
#     print(chunk.choices[0].delta.content or "", end="", flush=True)

# print("#"*200)
# ########################### ###########################
# response = client.chat.completions.create(
#     model="glm-4v-9b",
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                     {
#                         "type": "image_url",
#                         "image_url": {
#                             "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
#                         },
#                     },
#                 {"type": "text", "text": "描述这个图片"},
#             ],
#         }
#     ],
#     stream=True,
#     temperature=0.0,
#     max_tokens=128,
#     presence_penalty=1.0,
#     frequency_penalty=1.0
# )
# for chunk in response:
#     print(chunk.choices[0].delta.content or "", end="", flush=True)
# print("#"*200)

# ########################### ###########################

def generate_params(data: dict):
    if os.environ.get("OPENAI"):
        return {
            "extra_body": data
        }
    return data


response = client.images.generate(
    model="SD15MultiControlnetGenerateImage/majicmixRealistic_v7.safetensors/majicmixRealistic_v7.safetensors",
    **generate_params({
        "prompt": "a bottle with a beautiful rainbow galaxy inside it on top of a wooden table in the middle of a modern kitchen beside a plate of vegetables and mushrooms and a wine glasse that contains a planet earth with a plate with a half eaten apple pie on it",
        "batch_size": 1,
        "seed": 1234,
        "width": 512,
        "height": 512,
        "steps": 4,
        "denoise_strength": 0.7,
        "image_url": "http://oss-cnsq01.cdsgss.com/maas-2/data/82500db2-ba7f-4afc-85df-2a39f6f1e014/04.png?AWSAccessKeyId=ef55cb62ff7511edb70f46ae5a5d3b50&Signature=TNtoR50PBE5uG3QcvzM3VvuQZGk%3D&Expires=2006601342",
        "controlnets": [
            {
                "image_url": "http://oss-cnsq01.cdsgss.com/maas-2/data/82500db2-ba7f-4afc-85df-2a39f6f1e014/04.png?AWSAccessKeyId=ef55cb62ff7511edb70f46ae5a5d3b50&Signature=TNtoR50PBE5uG3QcvzM3VvuQZGk%3D&Expires=2006601342",
                "module": "canny"
            }
        ]
    })
)
print(response)
print("#"*200)

########################### ###########################

# data = {"model": "CogVideo/CogVideoX-5b",
#         "prompt": "A man is playing a guitar in a park",
#         "width": 480,
#         "height": 720,
#         "num_frames": 16}
# resp = requests.post(base_url + "/video/generations", json=data)
# print(resp.json())