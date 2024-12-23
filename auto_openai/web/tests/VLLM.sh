curl -X POST "http://127.0.0.1:9000/openai/v1/chat/completions" \
    -H "Authorization: Bearer xxxx" \
    -H "Content-Type: application/json" \
    -d '{
  "model": "glm-4v-9b",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {
            "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
          }
        },
        {
          "type": "text",
          "text": "描述这个图片"
        }
      ]
    }
  ],
  "stream": true,
  "temperature": 0.0,
  "max_tokens": 128,
  "presence_penalty": 1.0,
  "frequency_penalty": 1.0
}'
