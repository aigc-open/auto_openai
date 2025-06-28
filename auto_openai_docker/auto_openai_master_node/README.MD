# 模型 压测(启动多副本预热)

```bash
docker run --rm \
    registry.cn-shanghai.aliyuncs.com/zhph-server/hey:latest \
    -n 4 -c 4 -m POST -H "Content-Type: application/json" \
    -H "Authorization: xxx" \
    -d '{
        "model": "deepseek-coder-6.7b-base",
        "prompt": "# 用python写微信支付",
        "stream": false,
        "max_tokens": 64,
        "temperature": 0.0
    }' https://auto-openai.cpolar.cn/openai/v1/completions
```

```bash
docker run --rm \
    registry.cn-shanghai.aliyuncs.com/zhph-server/hey:latest \
    -n 4 -c 4 -m POST -H "Content-Type: application/json" \
    -H "Authorization: xxx" \
    -d '{
        "model": "Qwen2.5-7B-Instruct",
        "prompt": "# 用python写微信支付",
        "stream": false,
        "max_tokens": 64,
        "temperature": 0.0
    }' https://auto-openai.cpolar.cn/openai/v1/completions
```