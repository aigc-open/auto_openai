curl -X POST "http://127.0.0.1:9000/openai/v1/audio/speech" \
    -H "Authorization: Bearer xxxx" \
    -H "Content-Type: application/json" \
    -d '{
  "model": "maskgct-tts-clone",
  "voice": "",
  "input": "你好，介绍以下你自己呢",
  "clone_url": "https://gitee.com/lijiacai/static-files/releases/download/v0.1/guodegang.wav"
}' --output output.wav
