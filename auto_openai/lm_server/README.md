```bash
python3 -m auto_openai.lm_server.install_plugin tiktoken
python3 -m auto_openai.lm_server.install_plugin comfyui
python3 -m auto_openai.lm_server.install_plugin comfyui skip_requirements
python3 -m auto_openai.lm_server.install_plugin webui
python3 -m auto_openai.lm_server.install_plugin webui skip_requirements
python3 -m auto_openai.lm_server.install_plugin maskgct
python3 -m auto_openai.lm_server.install_plugin maskgct skip_requirements
#
python3 -m auto_openai.lm_server.install_plugin embedding
python3 -m auto_openai.lm_server.install_plugin funasr
python3 -m auto_openai.lm_server.install_plugin llm_transformer
python3 -m auto_openai.lm_server.install_plugin rerank
python3 -m auto_openai.lm_server.install_plugin diffusers
```

# 启动

```bash
python3 -m auto_openai.lm_server.start --help

python3 -m auto_openai.lm_server.start get_maskgct 6 7861
python3 -m auto_openai.lm_server.start get_webui 6 7861
python3 -m auto_openai.lm_server.start get_comfyui 6 7861
python3 -m auto_openai.lm_server.start get_funasr 6 7861
python3 -m auto_openai.lm_server.start get_embedding 6 7861
python3 -m auto_openai.lm_server.start get_rerank 6 7861
python3 -m auto_openai.lm_server.start get_llm_transformer glm-4v-9b 6 7861
python3 -m auto_openai.lm_server.start get_diffusers_video CogVideo/CogVideoX-5b 6 7861
```
