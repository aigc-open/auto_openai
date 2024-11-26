
from together import Together as OpenAI
import os
base_url = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:9000/openai/v1")
api_key = "xxxx"
########################### ###########################
client = OpenAI(base_url=base_url, api_key=api_key)


########################### ###########################
query = "What animals can I find near Peru?"

documents = [
    "The giant panda (Ailuropoda melanoleuca), also known as the panda bear or simply panda, is a bear species endemic to China.",
    "The llama is a domesticated South American camelid, widely used as a meat and pack animal by Andean cultures since the pre-Columbian era.",
    "The wild Bactrian camel (Camelus ferus) is an endangered species of camel endemic to Northwest China and southwestern Mongolia.",
    "The guanaco is a camelid native to South America, closely related to the llama. Guanacos are one of two wild South American camelids; the other species is the vicuña, which lives at higher elevations.",
]*100

response = client.rerank.create(
    model="bge-reranker-base",
    query=query,
    documents=documents,
    top_n=2
)

for result in response.results:
    print(f"Document Index: {result.index}")
    print(f"Document: {documents[result.index]}")
    print(f"Relevance Score: {result.relevance_score}")
