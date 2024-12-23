import os
base_url = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:9000/openai/v1")
api_key = "xxxx"
########################### ###########################
from openai import OpenAI
client = OpenAI(base_url=base_url, api_key=api_key)

#################################################################

response = client.embeddings.create(
  model = "bge-base-zh-v1.5", # bge-m3
  input = [
    "Our solar system orbits the Milky Way galaxy at about 515,000 mph",
    "Jupiter's Great Red Spot is a storm that has been raging for at least 350 years."
  ]
)
print(response)