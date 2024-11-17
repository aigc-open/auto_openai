from openai import OpenAI
import os
# https://api.openai.com/v1
base_url = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:9000/openai/v1")
client = OpenAI(api_key="xxxxx",
                base_url=base_url)

models = [
    # "sd1.5/majicmixRealistic_betterV6.safetensors",
    "flux.1-schnell/flux1-schnell.safetensors",
]
for model in models:
    response = client.images.generate(
        prompt=None,
        extra_body={
            "model": "base_text_to_image",
            "base_text_to_image": {
                "prompt": "a bottle with a beautiful rainbow galaxy inside it on top of a wooden table in the middle of a modern kitchen beside a plate of vegetables and mushrooms and a wine glasse that contains a planet earth with a plate with a half eaten apple pie on it",
                "batch_size": 1,
                "model": model,
                "seed": 1234,
                "width": 512,
                "height": 512,
                "steps": 4
            }
        }
    )
    print(response)


# ImagesResponse(created=None, data=[Image(b64_json=None, revised_prompt=None, url='http://oss-cnsq01.cdsgss.com/maas-1/tmp/173036573270819.png?AWSAccessKeyId=ef55cb62ff7511edb70f46ae5a5d3b50&Signature=gerae5tNNY3JMamLEyg41K%2BnP4Y%3D&Expires=1730452134')])
