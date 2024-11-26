from gradio_client import Client
import json

client = Client("http://10.9.112.104:7861/")
result = client.predict(
    ["hello", "world"],
    "bge-base-zh-v1.5",
    api_name="/predict"
)
print(result)
print(type(result))
