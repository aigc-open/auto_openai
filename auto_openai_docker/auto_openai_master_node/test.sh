export OPENAI_BASE_URL=https://auto-openai.cpolar.cn/openai/v1
export model=Qwen2.5-7B-Instruct
curl -X POST "$OPENAI_BASE_URL/chat/completions" \
    -H "Authorization: Bearer xxxx" \
    -H "Content-Type: application/json" \
    -d '{
  "model": "$model",
  "messages": [
    {"role": "user", "content": "What are some fun things to do in New York?"}
  ],
  "max_tokens": 4096,
  "temperature": 0.0,
  "stream": false
}'