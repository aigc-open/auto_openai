import os
base_url = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:9000/openai/v1")
api_key = "xxxx"
########################### ###########################
if os.environ.get("OPENAI"):
    from openai import OpenAI
    extra_body = True
else:
    from together import Together as OpenAI
    extra_body = False
client = OpenAI(base_url=base_url, api_key=api_key)


########################### ###########################

def generate_params(data: dict):
    if os.environ.get("OPENAI"):
        return {
            "extra_body": data
        }
    return data


models = [
    "sd1.5/majicmixRealistic_betterV6.safetensors",
    # "flux.1-schnell/flux1-schnell.safetensors",
]
for model in models:

    response = client.images.generate(
        prompt="",
        model="base_text_to_image",
        **generate_params({
            "base_text_to_image": {
                "prompt": "a bottle with a beautiful rainbow galaxy inside it on top of a wooden table in the middle of a modern kitchen beside a plate of vegetables and mushrooms and a wine glasse that contains a planet earth with a plate with a half eaten apple pie on it",
                "batch_size": 1,
                "model": model,
                "seed": 1234,
                "width": 512,
                "height": 512,
                "steps": 4
            }
        })
    )
    print(response)


# id=None model=None object=None data=[ImageChoicesData(index=0, b64_json=None, url='http://oss-cnsq01.cdsgss.com/maas-1/tmp/1731035136755155.png?AWSAccessKeyId=ef55cb62ff7511edb70f46ae5a5d3b50&Signature=nUZ49RQKvz0puUPjz4SFqGOz1Vs%3D&Expires=1731121536')]
