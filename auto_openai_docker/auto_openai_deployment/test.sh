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

# 生图
curl -X POST "$OPENAI_BASE_URL/images/generations" \
    -H "Authorization: Bearer xxxx" \
    -H "Content-Type: application/json" \
    -d '{
  "model": "SD15MultiControlnetGenerateImage/majicmixRealistic_v7.safetensors/majicmixRealistic_v7.safetensors",
  "prompt": "a bottle with a beautiful rainbow galaxy inside it on top of a wooden table in the middle of a modern kitchen beside a plate of vegetables and mushrooms and a wine glass that contains a planet earth with a plate with a half eaten apple pie on it",
  "batch_size": 1,
  "seed": 1234,
  "width": 512,
  "height": 512,
  "steps": 4,
  "denoise_strength": 0.7,
  "image_url": "http://oss-cnsq01.cdsgss.com/maas-2/data/82500db2-ba7f-4afc-85df-2a39f6f1e014/04.png?AWSAccessKeyId=ef55cb62ff7511edb70f46ae5a5d3b50&Signature=TNtoR50PBE5uG3QcvzM3VvuQZGk%3D&Expires=2006601342",
  "controlnets": [
    {
      "image_url": "http://oss-cnsq01.cdsgss.com/maas-2/data/82500db2-ba7f-4afc-85df-2a39f6f1e014/04.png?AWSAccessKeyId=ef55cb62ff7511edb70f46ae5a5d3b50&Signature=TNtoR50PBE5uG3QcvzM3VvuQZGk%3D&Expires=2006601342",
      "module": "canny"
    }
  ]
}'