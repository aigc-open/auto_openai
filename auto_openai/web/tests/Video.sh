curl -X POST "http://127.0.0.1:9000/openai/v1/video/generations" \
        -H "Authorization: Bearer xxxx" \
        -H "Content-Type: application/json" \
        -d '{
  "model": "CogVideo/CogVideoX-5b",
  "prompt": "A man is playing a guitar in a park",
  "width": 480,
  "height": 720,
  "num_frames": 16
}'
