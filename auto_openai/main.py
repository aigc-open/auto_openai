import time
import asyncio
import os
import requests
import json
import gradio as gr
from fastapi import HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse, Response
from auto_openai.utils.init_env import global_config
from fastapi.middleware.cors import CORSMiddleware
from enum import Enum
from fastapi import FastAPI, Request, Body, Header, Query, File, UploadFile, Form
from typing import Optional, Union, List
from auto_openai.utils.cut_messages import messages_token_count, string_token_count, cut_string, cut_messages
from auto_openai.utils.depends import get_model_config, get_combine_prompt_function, get_running_models
from auto_openai.utils.public import CustomRequestMiddleware, redis_client, s3_client
from pydantic import BaseModel
from fastapi.responses import FileResponse
from auto_openai.utils.openai import ChatCompletionRequest, CompletionRequest, Scheduler, gen_request_id, \
    ImageGenerateRequest, ImageGenerateResponse, AudioSpeechRequest, EmbeddingsRequest, RerankRequest, VideoGenerationsRequest

from auto_openai.web.pages.routers import UIWeb


app = FastAPI(root_path="/openai", title="Auto Openai")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# app.add_middleware(CustomRequestMiddleware)

########################### openai api ############################


@app.post("/v1/chat/completions")
async def chat_completion(
        request: Request,
        data: ChatCompletionRequest):
    # 处理模型最终需要完成的token数量
    model_config = get_model_config(name=data.model)
    model_max_tokens = model_config.get("model_max_tokens", 2048)

    data.messages = cut_messages(
        messages=data.messages, token_limit=int(model_max_tokens*4/5))
    current_token_count = messages_token_count(
        messages=data.messages, token_limit=int(model_max_tokens*4/5)) + 100  # 上浮100token误差

    max_tokens = min([data.max_tokens, model_max_tokens - current_token_count])
    max_tokens = 1 if max_tokens < 0 else max_tokens
    data.max_tokens = max_tokens
    data.temperature = 0.01 if data.temperature <= 0.01 else data.temperature
    data.top_p = 0.001 if data.top_p <= 0.0 else data.top_p

    scheduler = Scheduler(redis_client=redis_client, http_request=request,
                          queue_timeout=global_config.QUEUE_TIMEOUT, infer_timeout=global_config.INFER_TIMEOUT)

    scheduler.set_model_config(
        model_name=data.model, value=json.dumps(model_config))
    request_id = gen_request_id()
    data.change_messages_image_url_to_base64()
    if data.stream:
        return StreamingResponse(scheduler.ChatCompletionStream(
            request=data,
            request_id=request_id),
            media_type="text/event-stream")
    return await scheduler.ChatCompletion(request=data, request_id=request_id)


@app.post("/{proxy_model}/v1/chat/completions")
async def chat_completion(
        request: Request,
        proxy_model: str,
        data: ChatCompletionRequest):
    data.model = proxy_model
    # 处理模型最终需要完成的token数量
    model_config = get_model_config(name=data.model)
    model_max_tokens = model_config.get("model_max_tokens", 2048)

    data.messages = cut_messages(
        messages=data.messages, token_limit=int(model_max_tokens*4/5))
    current_token_count = messages_token_count(
        messages=data.messages, token_limit=int(model_max_tokens*4/5)) + 100  # 上浮100token误差

    max_tokens = min([data.max_tokens, model_max_tokens - current_token_count])
    max_tokens = 1 if max_tokens < 0 else max_tokens
    data.max_tokens = max_tokens
    data.temperature = 0.01 if data.temperature <= 0.01 else data.temperature
    data.top_p = 0.001 if data.top_p <= 0.0 else data.top_p
    scheduler = Scheduler(redis_client=redis_client, http_request=request,
                          queue_timeout=global_config.QUEUE_TIMEOUT, infer_timeout=global_config.INFER_TIMEOUT)

    scheduler.set_model_config(
        model_name=data.model, value=json.dumps(model_config))
    request_id = gen_request_id()
    if data.stream:
        return StreamingResponse(scheduler.ChatCompletionStream(
            request=data,
            request_id=request_id),
            media_type="text/event-stream")
    return await scheduler.ChatCompletion(request=data, request_id=request_id)


@app.post("/v1/completions")
async def completion(
        request: Request,
        data: CompletionRequest):
    model_config = get_model_config(name=data.model)

    model_max_tokens = model_config.get("model_max_tokens", 2048)
    # 处理模型最终需要完成的token数量
    data.prompt = cut_string(
        str=data.prompt, token_limit=int(model_max_tokens*4/5))

    current_token_count = string_token_count(
        str=data.prompt) + 100  # 上浮100token误差
    # 针对coder 续写模型处理
    if get_combine_prompt_function(data.model) is not None:
        data.prompt = get_combine_prompt_function(data.model)(
            prompt=data.prompt, suffix=data.suffix)
    max_tokens = min([data.max_tokens, model_max_tokens - current_token_count])
    max_tokens = 1 if max_tokens < 0 else max_tokens
    data.max_tokens = max_tokens
    data.temperature = 0.01 if data.temperature <= 0.01 else data.temperature
    data.top_p = 0.001 if data.top_p <= 0.0 else data.top_p
    scheduler = Scheduler(redis_client=redis_client, http_request=request,
                          queue_timeout=global_config.QUEUE_TIMEOUT, infer_timeout=global_config.INFER_TIMEOUT)
    scheduler.set_model_config(
        model_name=data.model, value=json.dumps(model_config))
    request_id = None
    request_id = gen_request_id()
    if data.stream:
        return StreamingResponse(scheduler.CompletionStream(
            request=data,
            request_id=request_id),
            media_type="text/event-stream")
    return await scheduler.Completion(request=data, request_id=request_id)


@app.get("/v1/models")
async def get_model(request: Request):
    # model_list = global_config.get_MODELS_MAPS().get(
    #     "LLM", []) + global_config.get_MODELS_MAPS().get("VLLM", [])
    models_map = global_config.get_MODELS_MAPS()
    model_list = []
    for k, models in models_map.items():
        if isinstance(models, list):
            model_list.extend(models)
        else:
            model_list.append(models)
    out = []
    for model in model_list:
        out.append({
            "id": model["name"],
            "object": "model",
            "created": 1686935002,
            "owned_by": "",
            "info": model
        })
    return {
        "object": "list",
        "data": out
    }


@app.post("/v1/images/generations")
async def image_generations(request: Request):
    scheduler = Scheduler(redis_client=redis_client, http_request=request,
                          queue_timeout=global_config.QUEUE_TIMEOUT, infer_timeout=global_config.INFER_TIMEOUT)
    # 获取body 数据
    data = await request.json()
    model_config = get_model_config(name=data["model"])
    scheduler.set_model_config(
        model_name=data["model"], value=json.dumps(model_config))
    request_id = gen_request_id()
    server_type = model_config["server_type"]
    api_type = model_config["api_type"]
    if server_type == "comfyui":
        if api_type == "SolutionBaseGenerateImage":
            from auto_openai.workflow import SolutionBaseGenerateImage
            req = SolutionBaseGenerateImage(**data)
        req.format()
    elif server_type == "diffusers-image":
        from auto_openai.utils.openai.openai_request import BaseGenerateImageRequest
        req = BaseGenerateImageRequest(**data)
    elif server_type == "webui":
        if api_type == "SD15MultiControlnetGenerateImage":
            from auto_openai.utils.openai.openai_request import SD15MultiControlnetGenerateImageRequest
            req = SD15MultiControlnetGenerateImageRequest(**data)
    return await scheduler.ImageGenerations(request=req, request_id=request_id)


@app.post("/v1/audio/speech")
async def audio_speech(request: Request, data: AudioSpeechRequest):
    scheduler = Scheduler(redis_client=redis_client, http_request=request,
                          queue_timeout=global_config.QUEUE_TIMEOUT, infer_timeout=global_config.INFER_TIMEOUT)
    model_config = get_model_config(name=data.model)
    scheduler.set_model_config(
        model_name=data.model, value=json.dumps(model_config))
    request_id = gen_request_id()
    url = await scheduler.AudioSpeech(request=data, request_id=request_id)

    def _gen():
        with requests.get(url, stream=True) as response:
            if response.status_code == 200:
                for chunk in response.iter_content(chunk_size=1024):
                    yield chunk
    return StreamingResponse(_gen(), media_type='audio/wav')


class AudioTranscriptionsRequest(BaseModel):
    model: str
    url: str


@app.post("/v1/audio/transcriptions")
async def audio_speech(request: Request, model: str = Form(...), file: UploadFile = File(...)):
    # 将文件上传到s3
    filename = s3_client.generate_key(
        "/tmp/", filename=file.filename, is_random=True)
    bucket_name = global_config.OSS_CLIENT_CONFIG["bucket_name"]
    s3_client.upload_fileobj(file.file,
                             bucket_name,
                             filename)
    url = s3_client.get_download_url(bucket_name, filename)
    data = AudioTranscriptionsRequest(model=model, url=url)
    scheduler = Scheduler(redis_client=redis_client, http_request=request,
                          queue_timeout=global_config.QUEUE_TIMEOUT, infer_timeout=global_config.INFER_TIMEOUT)
    model_config = get_model_config(name=model)
    scheduler.set_model_config(
        model_name=model, value=json.dumps(model_config))
    request_id = gen_request_id()
    text = await scheduler.AudioTranscriptions(request=data, request_id=request_id)
    return {"text": text}


@app.post("/v1/embeddings")
async def embeddings(request: Request, data: EmbeddingsRequest):
    scheduler = Scheduler(redis_client=redis_client, http_request=request,
                          queue_timeout=global_config.QUEUE_TIMEOUT, infer_timeout=global_config.INFER_TIMEOUT)
    model_config = get_model_config(name=data.model)
    scheduler.set_model_config(
        model_name=data.model, value=json.dumps(model_config))
    request_id = gen_request_id()
    embeddings = await scheduler.Embeddings(request=data, request_id=request_id)
    return embeddings


@app.post("/v1/rerank")
async def Rerank(request: Request, data: RerankRequest):
    scheduler = Scheduler(redis_client=redis_client, http_request=request,
                          queue_timeout=global_config.QUEUE_TIMEOUT, infer_timeout=global_config.INFER_TIMEOUT)
    model_config = get_model_config(name=data.model)
    scheduler.set_model_config(
        model_name=data.model, value=json.dumps(model_config))
    request_id = gen_request_id()
    out = await scheduler.Rerank(request=data, request_id=request_id)
    return out


@app.post("/v1/video/generations")
async def VideoGenerations(request: Request, data: VideoGenerationsRequest):
    scheduler = Scheduler(redis_client=redis_client, http_request=request,
                          queue_timeout=global_config.QUEUE_TIMEOUT, infer_timeout=60*10)
    model_config = get_model_config(name=data.model)
    scheduler.set_model_config(
        model_name=data.model, value=json.dumps(model_config))
    request_id = gen_request_id()
    out = await scheduler.VideoGenerations(request=data, request_id=request_id)
    return out

########################### other api for frontend ############################


async def get_queue_length(request: Request, request_id: str = gen_request_id()):
    scheduler = Scheduler(redis_client=redis_client, http_request=request)
    if scheduler.get_request_status_ing(request_id=request_id):
        return {
            "length": 0
        }
    params = scheduler.get_request_params(request_id=request_id)
    if params:
        model_name = params.get("model")
        ids = scheduler.get_request_queue_all_ids(model_name=model_name)
        if request_id in ids:
            return {
                "length": ids.index(request_id) + 1
            }
        return {
            "length": 0
        }
    else:
        return {
            "length": -1
        }


@app.get("/v1/running_models")
async def running_models(request: Request):
    return get_running_models()


@app.get("/v1/profiler")
async def get_profiler(request: Request):
    scheduler = Scheduler(redis_client=redis_client, http_request=request,
                          queue_timeout=global_config.QUEUE_TIMEOUT, infer_timeout=global_config.INFER_TIMEOUT)

    return scheduler.get_profiler()


@app.get("/v1/nodes")
async def get_nodes(request: Request):
    scheduler = Scheduler(redis_client=redis_client, http_request=request)
    return scheduler.get_running_node()


@app.get("/v1/get_available_model")
async def get_available_model(request: Request):
    return global_config.get_MODELS_MAPS()


@app.get("/v1/CodeConfig")
async def get_continue_config(request: Request):
    continue_config = global_config.get_continue_config()
    return continue_config
########################### web html ############################
UIWeb.register_ui(app, mount_path="/")


def run(port: int = 9000, workers=1, limit_concurrency=10000):
    import uvicorn
    os.environ["MAINPORT"] = str(port)
    uvicorn.run("auto_openai.main:app", host="0.0.0.0",
                port=port, workers=1, limit_concurrency=10000)


if __name__ == "__main__":
    from fire import Fire
    Fire(run)
