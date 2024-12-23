curl -X POST "http://127.0.0.1:9000/openai/v1/chat/completions" \
    -H "Authorization: Bearer xxxx" \
    -H "Content-Type: application/json" \
    -d '{
  "model": "Qwen2.5-7B-Instruct",
  "messages": [
    {"role": "user", "content": "What are some fun things to do in New York?"}
  ],
  "max_tokens": 204096,
  "temperature": 0.0,
  "stream": false
}'

curl -X POST "http://127.0.0.1:9000/openai/v1/completions" \
    -H "Authorization: Bearer xxxx" \
    -H "Content-Type: application/json" \
    -d '{
  "model": "Qwen2.5-7B-Instruct",
  "prompt": "def print_hello",
  "max_tokens": 128,
  "temperature": 0.0,
  "stream": false
}'
