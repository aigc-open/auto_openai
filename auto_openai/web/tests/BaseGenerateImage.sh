curl -X POST "http://127.0.0.1:9000/openai/v1/images/generations" \
    -H "Authorization: Bearer xxxx" \
    -H "Content-Type: application/json" \
    -d '{
  "model": "majicmixRealistic_v7.safetensors/majicmixRealistic_v7.safetensors",
  "prompt": "a bottle with a beautiful rainbow galaxy inside it on top of a wooden table in the middle of a modern kitchen beside a plate of vegetables and mushrooms and a wine glass that contains a planet earth with a plate with a half eaten apple pie on it",
  "batch_size": 1,
  "seed": 1234,
  "width": 512,
  "height": 512,
  "steps": 4,
  "denoise_strength": 0.7,
  "image_url": "http://oss-cnsq01.cdsgss.com/maas-2/data/82500db2-ba7f-4afc-85df-2a39f6f1e014/04.png?AWSAccessKeyId=ef55cb62ff7511edb70f46ae5a5d3b50&Signature=TNtoR50PBE5uG3QcvzM3VvuQZGk%3D&Expires=2006601342"
}'
