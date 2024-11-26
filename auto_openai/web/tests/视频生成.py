import os
base_url = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:9000/openai/v1")
api_key = "xxxx"
########################### ###########################
if os.environ.get("OPENAI"):
    from openai import OpenAI
else:
    from together import Together as OpenAI
client = OpenAI(base_url=base_url, api_key=api_key)

#################################################################
