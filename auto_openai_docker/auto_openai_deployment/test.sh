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

curl -X POST "$OPENAI_BASE_URL/completions" \
    -H "Authorization: Bearer xxxx" \
    -H "Content-Type: application/json" \
    -d '{
  "model": "deepseek-coder-6.7b-base",
  "prompt": "<｜fim▁begin｜># 打印冒泡排序 \ndef <｜fim▁hole｜>return sorted_list<｜fim▁end｜>",
  "max_tokens": 1024,
  "temperature": 0.0,
  "stream": false
}'

# Coder 续写模式
curl -X POST "$OPENAI_BASE_URL/completions" \
  -H "Authorization: Bearer xxxx" \
  -H "Content-Type: application/json" \
  -d '{
  "model": "deepseek-coder-6.7b-base",
  "prompt": "# 打印冒泡排序 \ndef",
  "max_tokens": 128,
  "temperature": 0.0,
  "suffix": "return sorted_list",
  "stream": false
}'
