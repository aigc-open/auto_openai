# 运行

## 下载 tiktoken 模型

```bash
python3 -m auto_openai.lm_server.install_models tiktoken
```

## 启动

```bash
python3 -m auto_openai.lm_server.start --help

python3 -m auto_openai.lm_server.start get_vllm Qwen2.5-7B-Instruct 5 1 7861 /template/template_qwen.jinja 1024 gcu
python3 -m auto_openai.lm_server.start get_vllm Qwen2.5-7B-Instruct 5 1 7861 /template/template_qwen.jinja 1024 auto
python3 -m auto_openai.lm_server.start get_maskgct 6 7861
python3 -m auto_openai.lm_server.start get_webui 6 7861
python3 -m auto_openai.lm_server.start get_comfyui 6 7861
python3 -m auto_openai.lm_server.start get_funasr 6 7861
python3 -m auto_openai.lm_server.start get_embedding 6 7861
python3 -m auto_openai.lm_server.start get_rerank 6 7861
python3 -m auto_openai.lm_server.start get_llm_transformer glm-4v-9b 6 7861
python3 -m auto_openai.lm_server.start get_diffusers_video CogVideo/CogVideoX-5b 6 7861
python3 -m auto_openai.lm_server.start get_wan21 wan/Wan2.1-T2V-1.3B-Diffusers 6 7861
```
