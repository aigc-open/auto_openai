from auto_openai.utils.init_env import global_config
from loguru import logger
import os
import requests
import time
from auto_openai.lm_server.docker_container import Docker


class CMD:
    @classmethod
    def check_status(cls, container, port):
        time.sleep(1)
        start_time = time.time()
        status = True
        status_str = ""
        while True:
            try:
                container.reload()
                if container.status != status_str:
                    logger.info(f"启动状态: {container.status}...")
                    status_str = container.status
                if container.status == "exited":
                    logger.warning(container.logs(tail=10))
                    status = False
                    break
                url = f"http://localhost:{port}/"
                if requests.get(url, timeout=3).status_code < 500:
                    status = True
                    status_str = container.status
                    break
            except Exception as e:
                status = False
                time.sleep(1)

            if time.time() - start_time > 60*20:
                status = False
        logger.info(f"最终启动状态: {container.status}...")
        return status

    @classmethod
    def get_environment(cls, device):
        if "NV" in global_config.GPU_TYPE:
            return [f"CUDA_VISIBLE_DEVICES={device}", f"NVIDIA_VISIBLE_DEVICES={device}"]
        else:
            return [f"TOPS_VISIBLE_DEVICES={device}"]

    @classmethod
    def get_image(cls, name):
        if "NV" in global_config.GPU_TYPE:
            image = global_config.IMAGE_BASE_PATH + f"/{name}:gpu"
        else:
            image = global_config.IMAGE_BASE_PATH + f"/{name}:gcu"
        return image

    @classmethod
    def get_vllm(cls, model_name, device, need_gpu_count, port, template, model_max_tokens, device_name, quantization="", server_type="vllm"):
        """
        INFO 01-06 02:32:41 api_server.py:652] args: Namespace(host=None, port=30104, uvicorn_log_level='info', 
        allow_credentials=False, allowed_origins=['*'], allowed_methods=['*'], allowed_headers=['*'], api_key=None, 
        lora_modules=None, prompt_adapters=None, chat_template='/template/template_deepseek-coder.jinja', chat_template_content_format='auto', 
        response_role='assistant', ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_cert_reqs=0, root_path=None, middleware=[], return_tokens_as_token_ids=False, 
        disable_frontend_multiprocessing=False, enable_auto_tool_choice=False, tool_call_parser=None, tool_parser_plugin='', model='deepseek-coder-6.7b-base', task='auto', 
        tokenizer=None, skip_tokenizer_init=False, revision=None, code_revision=None, tokenizer_revision=None, tokenizer_mode='auto', trust_remote_code=True, allowed_local_media_path=None, 
        download_dir=None, load_format='auto', config_format=<ConfigFormat.AUTO: 'auto'>, dtype='float16', kv_cache_dtype='auto', quantization_param_path=None, max_model_len=32768, 
        guided_decoding_backend='xgrammar', logits_processor_pattern=None, distributed_executor_backend=None, worker_use_ray=False, pipeline_parallel_size=1, tensor_parallel_size=1, 
        max_parallel_loading_workers=None, ray_workers_use_nsight=False, block_size=32, enable_prefix_caching=None, disable_sliding_window=False, use_v2_block_manager=True, num_lookahead_slots=0, 
        seed=0, swap_space=4, cpu_offload_gb=0, gpu_memory_utilization=0.9, num_gpu_blocks_override=None, max_num_batched_tokens=None, max_num_seqs=None, max_logprobs=20, disable_log_stats=False, 
        quantization=None, rope_scaling=None, rope_theta=None, hf_overrides=None, enforce_eager=True, max_seq_len_to_capture=8192, disable_custom_all_reduce=False, 
        tokenizer_pool_size=0, tokenizer_pool_type='ray', tokenizer_pool_extra_config=None, limit_mm_per_prompt=None, mm_processor_kwargs=None, mm_cache_preprocessor=False, 
        enable_lora=False, enable_lora_bias=False, max_loras=1, max_lora_rank=16, lora_extra_vocab_size=256, lora_dtype='auto', long_lora_scaling_factors=None, 
        max_cpu_loras=None, fully_sharded_loras=False, enable_prompt_adapter=False, max_prompt_adapters=1, max_prompt_adapter_token=0, device='auto', 
        num_scheduler_steps=1, multi_step_stream_outputs=True, scheduler_delay_factor=0.0, enable_chunked_prefill=None, speculative_model=None, 
        speculative_model_quantization=None, num_speculative_tokens=None, speculative_disable_mqa_scorer=False, speculative_draft_tensor_parallel_size=None, 
        speculative_max_model_len=None, speculative_disable_by_batch_size=None, ngram_prompt_lookup_max=None, ngram_prompt_lookup_min=None, spec_decoding_acceptance_method='rejection_sampler', 
        typical_acceptance_sampler_posterior_threshold=None, typical_acceptance_sampler_posterior_alpha=None, 
        disable_logprobs_during_spec_decoding=None, model_loader_extra_config=None, ignore_patterns=[], preemption_mode=None, served_model_name=None, 
        qlora_adapter_name_or_path=None, otlp_traces_endpoint=None, collect_detailed_traces=None, disable_async_output_proc=False, scheduling_policy='fcfs', 
        override_neuron_config=None, override_pooler_config=None, compilation_config=None, kv_transfer_config=None, worker_cls='auto', disable_log_requests=False, 
        max_log_len=None, disable_fastapi_docs=False, enable_prompt_tokens_details=False)
        """
        model_path = model_name.split(":")[0]
        if "NV" in global_config.GPU_TYPE:
            block_size = 16
        else:
            block_size = 64
        if quantization:
            quantization = f"--quantization {quantization}"
        else:
            quantization = ""
        if ".jinja" in template:
            template = f"--chat-template {template}"
        else:
            template = ""
        cmd = f"""
            python3 -m vllm.entrypoints.openai.api_server 
            --model {model_path} 
            --device={device_name} 
            {quantization}
            --enforce-eager
            --enable-prefix-caching
            {template}
            --tensor-parallel-size={need_gpu_count} 
            --max-model-len={model_max_tokens} 
            --dtype=float16 --block-size={block_size} --trust-remote-code 
            --served-model-name={model_name}
            --port={port}"""
        cmd = cmd.replace("\n", " ").strip()
        logger.info(f"本次启动模型: \n{cmd}")
        environment = cls.get_environment(device)
        container = Docker().run(image=cls.get_image(name=server_type), command=cmd,
                                 device_ids=device,
                                 GPU_TYPE=global_config.GPU_TYPE, environment=environment)
        return cls.check_status(container=container, port=port)

    @classmethod
    def get_comfyui(cls, device, port):
        cmd = f"""
            python3 comfyui-main.py --listen 0.0.0.0
            --gpu-only --use-pytorch-cross-attention 
            --port={port}"""
        cmd = cmd.replace("\n", " ").strip()
        logger.info(f"本次启动模型: \n{cmd}")
        environment = cls.get_environment(device)
        environment.extend(['PYTHONPATH="/workspace/ComfyUI:$PYTHONPATH"',
                           f"COMFYUI_MODEL_PATH={global_config.COMFYUI_MODEL_ROOT_PATH}", "PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync"])
        container = Docker().run(image=cls.get_image(name="comfyui"), command=cmd,
                                 device_ids=device,
                                 GPU_TYPE=global_config.GPU_TYPE, environment=environment)

        return cls.check_status(container=container, port=port)

    @classmethod
    def get_maskgct(cls, device, port):
        cmd = f"""
            python3 main.py --port={port}
            --model_root_path={global_config.MASKGCT_MODEL_ROOT_PATH}"""
        cmd = cmd.replace("\n", " ").strip()
        logger.info(f"本次启动模型: \n{cmd}")
        environment = cls.get_environment(device)
        container = Docker().run(image=cls.get_image(name="maskgct"), command=cmd,
                                 device_ids=device,
                                 GPU_TYPE=global_config.GPU_TYPE, environment=environment)
        return cls.check_status(container=container, port=port)

    @classmethod
    def get_funasr(cls, device, port):
        cmd = f"""
            python3 funasr-main.py --port={port}
            --model_root_path={global_config.FUNASR_MODEL_ROOT_PATH}"""
        cmd = cmd.replace("\n", " ").strip()
        logger.info(f"本次启动模型: \n{cmd}")
        environment = cls.get_environment(device)
        container = Docker().run(image=cls.get_image(name="funasr-server"), command=cmd,
                                 device_ids=device,
                                 GPU_TYPE=global_config.GPU_TYPE, environment=environment)
        return cls.check_status(container=container, port=port)

    @classmethod
    def get_embedding(cls, device, port):
        cmd = f"""
            python3 embedding-main.py --port={port}
            --model_root_path={global_config.EMBEDDING_MODEL_ROOT_PATH}"""
        cmd = cmd.replace("\n", " ").strip()
        logger.info(f"本次启动模型: \n{cmd}")
        environment = cls.get_environment(device)
        container = Docker().run(image=cls.get_image(name="embedding-server"), command=cmd,
                                 device_ids=device,
                                 GPU_TYPE=global_config.GPU_TYPE, environment=environment)
        return cls.check_status(container=container, port=port)

    @classmethod
    def get_llm_transformer(cls, model_name, device, port):
        cmd = f"""
            python3 llm-transformer-main.py --port={port}
            --model_path={os.path.join(global_config.LLM_TRANSFORMER_MODEL_ROOT_PATH, model_name)}"""
        cmd = cmd.replace("\n", " ").strip()
        logger.info(f"本次启动模型: \n{cmd}")
        environment = cls.get_environment(device)
        container = Docker().run(image=cls.get_image(name="llm-transformer-server"), command=cmd,
                                 device_ids=device,
                                 GPU_TYPE=global_config.GPU_TYPE, environment=environment)
        return cls.check_status(container=container, port=port)

    @classmethod
    def get_diffusers_video(cls, model_name, device, port):
        cmd = f"""
            python3 diffusers-video-main.py --port={port}
            --model_path={os.path.join(global_config.DIFFUSERS_MODEL_ROOT_PATH, model_name)}"""
        cmd = cmd.replace("\n", " ").strip().strip()
        logger.info(f"本次启动模型: \n{cmd}")
        environment = cls.get_environment(device)
        container = Docker().run(image=cls.get_image(name="diffusers-server"), command=cmd,
                                 device_ids=device,
                                 GPU_TYPE=global_config.GPU_TYPE, environment=environment)
        return cls.check_status(container=container, port=port)

    @classmethod
    def get_rerank(cls, device, port):
        cmd = f"""
            python3 rerank-main.py --port={port}
            --model_root_path={global_config.RERANK_MODEL_ROOT_PATH}"""
        cmd = cmd.replace("\n", " ").strip()
        logger.info(f"本次启动模型: \n{cmd}")
        environment = cls.get_environment(device)
        container = Docker().run(image=cls.get_image(name="rerank-server"), command=cmd,
                                 device_ids=device,
                                 GPU_TYPE=global_config.GPU_TYPE, environment=environment)
        return cls.check_status(container=container, port=port)

    @classmethod
    def get_webui(cls, device, port):
        cmd = f"""
            python3 server.py 
            --skip-version-check --skip-install --skip-prepare-environment 
            --api
            --skip-torch-cuda-test --skip-load-model-at-start --enable-insecure-extension-access 
            --models-dir={global_config.WEBUI_MODEL_ROOT_PATH}
            --enable-insecure-extension-access --listen  
            --port={port}"""

        cmd = cmd.replace("\n", " ").strip()
        logger.info(f"本次启动模型: \n{cmd}")
        environment = cls.get_environment(device)
        environment.extend(['PYTHONPATH="/workspace/stable-diffusion-webui:$PYTHONPATH"'])
        container = Docker().run(image=cls.get_image(name="webui"), command=cmd,
                                 device_ids=device,
                                 GPU_TYPE=global_config.GPU_TYPE, environment=environment)
        return cls.check_status(container=container, port=port)

    @classmethod
    def kill(cls):
        Docker().stop()
        Docker().remove()
