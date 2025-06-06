import requests
import json
import uuid
import time
from typing import Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field
import redis
import time
from enum import Enum
from loguru import logger
from fastapi import Request, HTTPException, File
from fastapi import FastAPI, Request, Body, Header, Query, File, UploadFile, Form
import asyncio
import random
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.completion import Completion
from uuid_extensions import uuid7, uuid7str
from .openai_request import *
##############################


class RedisStreamInfer(BaseModel):
    text: str = ""
    finish: bool = False
    msg: str = ""
    usage: dict = {}


class CompletionClient:
    def __init__(self, api_key="", base_url=""):
        self.base_url = base_url
        self.api_key = api_key
        self.url = self.base_url + "/completions"
        self.chat_flag = False

    @property
    def headers(self):
        return {'Content-Type': 'application/json', "Authorization": f"Bearer {self.api_key}"}

    def create_stream(self, **kwargs):
        if "tools" in kwargs:
            if not kwargs["tools"]:
                del kwargs["tools"]
                if "tool_choice" in kwargs:
                    del kwargs["tool_choice"]
            else:
                if kwargs.get("tool_choice") == "auto":
                    kwargs["tool_choice"] = {
                        "type": "function",
                        "function": {"name": kwargs["tools"][0]["function"]["name"]}
                    }

        else:
            if "tool_choice" in kwargs:
                del kwargs["tool_choice"]
        response = requests.post(
            self.url, headers=self.headers,
            data=json.dumps(kwargs), stream=True)
        if self.chat_flag is True:
            Response = ChatCompletionStreamResponse
        else:
            Response = CompletionStreamResponse
        for chunk in response.iter_content(chunk_size=100*1024):
            if chunk:
                try:
                    chunk = chunk.decode("utf8").strip()
                    for data_ in chunk.split("data: "):
                        if not data_.strip():
                            continue
                        if data_.strip() == "[DONE]":
                            return
                        data_json = json.loads(data_.strip())
                        # chunk["object"] = "text_completion"
                        yield Response(**data_json)
                except Exception as e:
                    logger.exception(str(e))
                    logger.warning(f"流式任务解析失败: {chunk}")
                    error = {
                        "object": "chat.completion.chunk" if self.chat_flag else "text_completion",
                        "message": "Error",
                        "type": "BadRequestError",
                        "param": None,
                        "code": 400
                    }
                    if type(chunk) == dict:
                        try:
                            error_obj = ErrorResponse(**chunk)
                            error_obj.update_max_tokens()
                            yield error_obj
                        except:
                            yield ErrorResponse(**error)
                    else:
                        yield ErrorResponse(**error)
                    # if self.chat_flag is True:
                    #     yield Response(**{'id': 'chat-ecc921188feb4e9993db49938580325c',
                    #                       'object': 'chat.completion.chunk', 'created': 1731385347, 'model': '',
                    #                       'choices': [{'index': 0, 'delta': {'role': 'assistant', "content": error_text}, 'logprobs': None, 'finish_reason': "stop"}]})
                    # else:
                    #     yield Response(**{'id': 'chat-ecc921188feb4e9993db49938580325c',
                    #                       'object': 'text_completion', 'created': 1731385347, 'model': '', 'choices': [{'index': 0, 'text': error_text, 'logprobs': None, 'finish_reason': "stop"}]})
                    return

    def create_(self, **kwargs):
        if "tools" in kwargs:
            if not kwargs["tools"]:
                del kwargs["tools"]
                if "tool_choice" in kwargs:
                    del kwargs["tool_choice"]
            else:
                if kwargs.get("tool_choice") == "auto":
                    kwargs["tool_choice"] = {
                        "type": "function",
                        "function": {"name": kwargs["tools"][0]["function"]["name"]}
                    }
        else:
            if "tool_choice" in kwargs:
                del kwargs["tool_choice"]
        response = requests.post(
            self.url, headers=self.headers, data=json.dumps(kwargs))
        if self.chat_flag is True:
            Response = ChatCompletion
        else:
            Response = Completion
        return Response(**response.json())

    def create(self, **kwargs):
        stream = kwargs.get("stream", False)
        if stream:
            return self.create_stream(**kwargs)
        else:
            return self.create_(**kwargs)


class ChatClient(CompletionClient):
    def __init__(self, api_key="", base_url=""):
        self.base_url = base_url
        self.api_key = api_key
        self.url = self.base_url + "/chat/completions"
        self.chat_flag = True

    @property
    def completions(self):
        return self


class VLLMOpenAI:
    def __init__(self, api_key="", base_url=""):
        self.base_url = base_url
        self.api_key = api_key

    @property
    def chat(self) -> ChatClient:
        return ChatClient(base_url=self.base_url, api_key=self.api_key)

    @property
    def completions(self) -> CompletionClient:
        return CompletionClient(base_url=self.base_url, api_key=self.api_key)


class ImageGenerateRequest(BaseModel):
    model: str = "BaseTextToImage"


class ImageGenerateData(BaseModel):
    index: int = 0
    url: str


class ImageGenerateResponse(BaseModel):

    data: List[ImageGenerateData]


class Scheduler:

    def __init__(self, redis_client: redis.Redis, http_request: Request = None, queue_timeout=10*60, infer_timeout=100):
        self.redis_client = redis_client
        self.http_request = http_request
        self.queue_timeout = queue_timeout
        self.infer_timeout = infer_timeout

    def set_request_queue(self, model_name, request_id):
        """大模型任务队列"""
        self.redis_client.lpush(
            f"lm-request-queue-{model_name}", str(request_id))

    def get_request_queue(self, model_name):
        """大模型任务队列"""
        data = self.redis_client.rpop(name=f"lm-request-queue-{model_name}")
        if data is not None:
            return data.decode()
        return data

    def get_request_queue_length(self, model_name):
        """大模型任务队列长度"""
        return self.redis_client.llen(name=f"lm-request-queue-{model_name}")

    def get_request_queue_all_ids(self, model_name):
        """大模型任务队列"""
        data = self.redis_client.lrange(
            name=f"lm-request-queue-{model_name}", start=0, end=-1)
        out = []
        for id_ in data:
            out.insert(0, str(id_.decode()))
        return out

    def get_request_queue_all_length(self):
        """大模型任务队列长度"""
        keys = self.redis_client.keys(pattern=f"lm-request-queue-*")
        length = 0
        for key in keys:
            length += self.redis_client.llen(name=key)
        return length

    def get_request_queue_names(self):
        """大模型任务队列有请求的模型用于轮询遍历"""
        keys = self.redis_client.keys(pattern=f"lm-request-queue-*")
        out = []
        for key in keys:
            out.append(key.decode().replace("lm-request-queue-", ""))
        return out

    def set_request_params(self, request_id, value, pexpire=2.5*1000):
        """设置请求参数，如果取消将过期，则该请求取消"""
        self.redis_client.set(
            name=f"lm-request-params-{request_id}", value=value, px=int(pexpire))

    def get_request_params(self, request_id):
        """设置请求参数，如果取消将过期，则该请求取消"""
        data = self.redis_client.get(f"lm-request-params-{request_id}")
        if data is not None:
            return json.loads(data.decode())
        return {}

    def set_request_status_ing(self, request_id, pexpire=100*1000):
        self.redis_client.set(
            name=f"lm-request-status-{request_id}", value="ing", px=int(pexpire))

    def get_request_status_ing(self, request_id):
        data = self.redis_client.get(f"lm-request-status-{request_id}")

        if data is not None:
            return data.decode()
        return ""

    def get_request_status_ing_total(self):
        keys = self.redis_client.keys(pattern=f"lm-request-status-*")
        return len(keys)

    def set_result(self, request_id, value: RedisStreamInfer):
        """将推理结果推向数据库"""
        self.redis_client.lpush(
            f"lm-request-result-{request_id}", json.dumps(value.dict(), ensure_ascii=False))

    def get_result(self, request_id) -> RedisStreamInfer:
        """获取推理结果"""
        data = self.redis_client.rpop(name=f"lm-request-result-{request_id}")
        if data is not None:
            return RedisStreamInfer(**json.loads(data.decode()))
        return data

    # 添加一个任务处理完的计数 +1
    def set_request_done(self):
        self.redis_client.incr(f"lm-request-done")

    def get_request_done(self):
        data = self.redis_client.get(f"lm-request-done")
        if data is None:
            return int(0)
        else:
            return int(data.decode())

    def set_running_model(self, model_name, pexpire=10*1000):
        """设置正在运行的模型"""
        self.redis_client.set(
            name=f"lm-running-{model_name}", value=model_name, px=int(pexpire))

    def get_running_model(self):
        """获取正在运行的模型"""
        keys = self.redis_client.keys(pattern=f"lm-running-*")
        out = []
        for key in keys:
            out.append(key.decode().replace("lm-running-", ""))
        return out

    def set_available_model(self, model_name, value: str, pexpire=5*1000):
        """设置可得的模型"""
        self.redis_client.set(
            name=f"available-model-{model_name}", value=value, px=int(pexpire))

    def get_available_model(self):
        """获取可得的模型"""
        keys = self.redis_client.keys(pattern=f"available-model-*")
        out = []

        for key in keys:
            data = self.redis_client.get(key)
            if data is not None:
                out.append(json.loads(data.decode()))
        return out

    def set_running_node(self, node_name, value: str, pexpire=10*1000):
        """设置正在运行的节点"""
        self.redis_client.set(
            name=f"lm-running-node-{node_name}", value=value, px=int(pexpire))

    def get_running_node(self):
        """获取正在运行的节点"""
        keys = self.redis_client.keys(pattern=f"lm-running-node-*")
        out = []

        for key in keys:
            data = self.redis_client.get(key)
            if data is not None:
                out.append(json.loads(data.decode()))
        return out

    def set_model_config(self, model_name, value, pexpire=120*1000):
        """设置模型相关参数"""
        self.redis_client.set(
            name=f"lm-config-{model_name}", value=value)

    def get_model_config(self, model_name):
        """获取模型相关参数"""
        data = self.redis_client.get(f"lm-config-{model_name}")

        if data is not None:
            return json.loads(data.decode())
        return {}

    def set_profiler(self, data: dict):
        """设置模型相关性能参数"""
        self.redis_client.set(
            name=f"lm-profiler", value=json.dumps(data))

    def get_profiler(self):
        """获取模型相关性能参数"""
        data = self.redis_client.get(f"lm-profiler")

        if data is not None:
            return json.loads(data.decode())
        return {}

    async def stream(self, request: ChatCompletionRequest, request_id=gen_request_id()):
        self.set_request_queue(model_name=request.model, request_id=request_id)
        # 排队
        start_time = time.time()
        logger.info(f"排队中 {request_id}")
        while True:
            if int(time.time()) % 1000 == 0:
                logger.info(f"排队中 {request_id}")
            request_data: dict = request.dict()
            # 判断是否存在suffix 这个key
            if "suffix" in request_data:
                del request_data["suffix"]
            self.set_request_params(
                request_id=request_id, value=json.dumps(request_data))
            # await asyncio.sleep(0.1)

            if self.http_request is not None and await self.http_request.is_disconnected():
                # 客户端主动断开连接
                raise HTTPException(status_code=422, detail="客户取消操作")
            if self.get_request_status_ing(request_id=request_id):
                break
            if time.time() - start_time > self.queue_timeout:
                raise HTTPException(status_code=500, detail="队列相当拥挤")
        # 开始推流给用户
        logger.info(f"排队总耗时 {request_id} : {time.time() - start_time}")
        start_time = time.time()
        logger.info(f"推流中 {request_id}")
        while True:
            if int(time.time()) % 1000 == 0:
                logger.info(f"推流中 {request_id}")
            data_ = self.get_result(request_id=request_id)
            if self.http_request is not None and await self.http_request.is_disconnected():
                # 客户端主动断开连接
                logger.warning(f"客户取消操作: {request_id}")
                raise HTTPException(status_code=422, detail="客户取消操作")
            if data_:
                data = data_
                yield data
                if data.finish:
                    break
            else:
                if time.time() - start_time > self.infer_timeout:
                    logger.warning("推理服务器超时未获取到")
                    data = RedisStreamInfer(finish=True, msg="推理服务器超时未获取到")
                    yield data
                    break
                else:
                    pass
                    # await asyncio.sleep(0.5)
        self.set_request_status_ing(
            request_id=request_id, pexpire=0.01*1000)  # 10ms
        logger.info(f"推流总耗时 {request_id} : {time.time() - start_time}")

    async def ChatCompletionStream(self, request: ChatCompletionRequest, request_id=gen_request_id()):
        buffer = ""
        reasoning = True

        async for data_ in self.stream(request=request, request_id=request_id):
            data: RedisStreamInfer = data_
            finish_reason = "stop" if data.finish else None
            current_text = data.text
            tool_calls = []

            if data.finish:
                data_tool_name = self.get_result(
                    request_id=request_id+"_tool_name")
                data_tool_args = self.get_result(
                    request_id=request_id+"_tool_args")
                data_tool_name = data_tool_name.text if data_tool_name else None
                data_tool_args = data_tool_args.text if data_tool_args else None
                tool_calls = [
                    {"function": {"name": data_tool_name, "arguments": data_tool_args}}] if data_tool_name else []

            buffer += current_text
            content = ""
            reasoning_content = ""

            if reasoning:
                if "</think>" in buffer:
                    thinks = buffer.split("</think>")
                    reasoning_content = thinks[0] + "</think>"
                    content = "".join(thinks[1:])
                    reasoning = False
                    buffer = ""
                else:
                    reasoning_content = current_text
            else:
                content = current_text
            if content and reasoning_content:
                out_reasoning_content = json.dumps(
                    ChatCompletionStreamResponse(
                        model=request.model,
                        choices=[{
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": tool_calls,
                                "reasoning_content": reasoning_content.replace("</think>", "").replace("<think>", "")
                            },
                            "finish_reason": finish_reason
                        }],
                        usage=UsageInfo(**data.usage)
                    ).dict(), ensure_ascii=False)
                out_content = json.dumps(
                    ChatCompletionStreamResponse(
                        model=request.model,
                        choices=[{
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": content,
                                "tool_calls": tool_calls,
                                "reasoning_content": ""
                            },
                            "finish_reason": finish_reason
                        }],
                        usage=UsageInfo(**data.usage)
                    ).dict(), ensure_ascii=False)
                yield f"data: {out_reasoning_content}\n\n"
                yield f"data: {out_content}\n\n"
            else:
                chunk = ChatCompletionStreamResponse(
                    model=request.model,
                    choices=[{
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": content,
                            "tool_calls": tool_calls,
                            "reasoning_content": reasoning_content.replace("</think>", "").replace("<think>", "")
                        },
                        "finish_reason": finish_reason
                    }],
                    usage=UsageInfo(**data.usage)
                )
                out = json.dumps(chunk.dict(), ensure_ascii=False)
                yield f"data: {out}\n\n"

    async def ChatCompletion(self, request: ChatCompletionRequest, request_id=gen_request_id()):
        content = ""
        tool_calls = []
        async for data_ in self.stream(request=request, request_id=request_id):
            data: RedisStreamInfer = data_
            content += data.text
            if data.finish:
                usage = UsageInfo(**data.usage)
                data_tool_name = self.get_result(
                    request_id=request_id+"_tool_name")
                data_tool_args = self.get_result(
                    request_id=request_id+"_tool_args")
                data_tool_name = data_tool_name.text if data_tool_name else None
                data_tool_args = data_tool_args.text if data_tool_args else None
                tool_calls = [
                    {"function": {"name": data_tool_name, "arguments": data_tool_args}}] if data_tool_name else []
        content_and_reasoning = content.split("</think>")
        if len(content_and_reasoning) >= 2:
            reasoning_content = content_and_reasoning[0]
            content = "".join(content_and_reasoning[1:])
        else:
            reasoning_content = ""
            content = content_and_reasoning[0]
        finish_reason = "stop"
        response = ChatCompletionResponse(
            model=request.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                    "reasoning_content": reasoning_content.replace("</think>", "").replace("<think>", "")
                },
                "finish_reason": finish_reason
            }],
            usage=usage
        )
        return response.dict()

    async def CompletionStream(self, request: CompletionRequest, request_id=gen_request_id()):
        async for data_ in self.stream(request=request, request_id=request_id):
            data: RedisStreamInfer = data_
            finish_reason = "stop" if data.finish else None
            chunk = CompletionStreamResponse(
                model=request.model,
                choices=[{
                    "index": 0,
                    "text": data.text,
                    "logprobs": None,
                    "finish_reason": finish_reason
                }],
                usage=UsageInfo(**data.usage)
            )
            # data = chunk.json(exclude_unset=True, ensure_ascii=False)
            data = json.dumps(chunk.dict(), ensure_ascii=False)
            yield f"data: {data}\n\n"

    async def Completion(self, request: CompletionRequest, request_id=gen_request_id()):
        content = ""
        async for data_ in self.stream(request=request, request_id=request_id):
            data: RedisStreamInfer = data_
            content += data.text
            if data.finish:
                usage = UsageInfo(**data.usage)
        finish_reason = "stop"
        response = CompletionResponse(
            model=request.model,
            choices=[{
                "index": 0,
                "text": content,
                "finish_reason": finish_reason,
                "logprobs": None
            }],
            usage=usage
        )
        return response.dict()

    async def ImageGenerations(self, request: CompletionRequest, request_id=gen_request_id()):
        # content = json.dumps(
        #     {"created": 0, "data": [{"url": "http://localhost:8000/images/1.png"}]})
        content = ""
        async for data_ in self.stream(request=request, request_id=request_id):
            data: RedisStreamInfer = data_
            content = data.text
            if data.finish:
                pass

        response = ImageGenerateResponse(**json.loads(content))
        return response.dict()

    async def AudioSpeech(self, request: AudioSpeechRequest, request_id=gen_request_id()):
        content = ""
        async for data_ in self.stream(request=request, request_id=request_id):
            data: RedisStreamInfer = data_
            content = data.text
            return content

    async def AudioTranscriptions(self, request: AudioSpeechRequest, request_id=gen_request_id()):
        content = ""
        async for data_ in self.stream(request=request, request_id=request_id):
            data: RedisStreamInfer = data_
            content = data.text
            return content

    async def Embeddings(self, request: CompletionRequest, request_id=gen_request_id()):
        content = ""
        async for data_ in self.stream(request=request, request_id=request_id):
            data: RedisStreamInfer = data_
            content = json.loads(data.text)
            return {
                "model": '',
                "object": 'list',
                "data": content}

    async def Rerank(self, request: CompletionRequest, request_id=gen_request_id()):
        content = ""
        async for data_ in self.stream(request=request, request_id=request_id):
            data: RedisStreamInfer = data_
            content = json.loads(data.text)
            return {
                "results": content
            }

    async def VideoGenerations(self, request: CompletionRequest, request_id=gen_request_id()):
        content = ""
        async for data_ in self.stream(request=request, request_id=request_id):
            data: RedisStreamInfer = data_
            content = data.text
            return {
                "revised_prompt": "",
                "url": content
            }
