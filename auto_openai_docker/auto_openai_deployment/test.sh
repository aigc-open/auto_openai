export OPENAI_BASE_URL=https://auto-openai.cpolar.cn/openai/v1
curl -X POST "$OPENAI_BASE_URL/chat/completions" \
    -H "Authorization: Bearer xxxx" \
    -H "Content-Type: application/json" \
    -d '{
  "model": "Qwen2.5-Coder-7B-Instruct",
  "messages": [
    {"role": "user", "content": "What are some fun things to do in New York?"}
  ],
  "max_tokens": 204096,
  "temperature": 0.0,
  "stream": false
}'

curl -X POST "$OPENAI_BASE_URL/openai/v1/completions" \
    -H "Authorization: Bearer xxxx" \
    -H "Content-Type: application/json" \
    -d '{
  "model": "Qwen2.5-Coder-7B",
  "prompt": "def print_hello",
  "max_tokens": 128,
  "temperature": 0.0,
  "stream": false
}'
