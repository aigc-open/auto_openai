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


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


def gen_request_id() -> str:
    return str(uuid7(as_type="int"))
    return str(int(time.time() * 1000)) + str(random.randint(1000, 9999))


def gen_random_uuid() -> str:
    return str(uuid7(as_type="int"))


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class ModelPermission(BaseModel):
    id: str = Field(default_factory=lambda: f"modelperm-{random_uuid()}")
    object: str = "model_permission"
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: Optional[str] = None
    is_blocking: str = False


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "vllm"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: List[ModelPermission] = Field(default_factory=list)


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: int = 0
    tps: float = 0


class ChatCompletionRequest(BaseModel):
    model: str = "chatglm3-6b"
    messages: Union[List[Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, str]]]]]]]] = [
        {"role": "user", "content": "hi"}]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = 512
    stop: Optional[Union[str, List[str]]] = Field(default=[])
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = Field(
        default=1.0, ge=0.0, le=2.0)  # 设置最大值为2
    frequency_penalty: Optional[float] = Field(
        default=1.0, ge=0.0, le=2.0)  # 设置最大值为2
    # logit_bias: Optional[Dict[str, float]] = None
    # user: Optional[str] = None
    # Additional parameters supported by vLLM
    # best_of: Optional[int] = None
    # top_k: Optional[int] = -1
    # ignore_eos: Optional[bool] = False
    # use_beam_search: Optional[bool] = False
    # stop_token_ids: Optional[List[int]] = Field(default_factory=list)
    # skip_special_tokens: Optional[bool] = True
    # spaces_between_special_tokens: Optional[bool] = True
    # add_generation_prompt: Optional[bool] = True
    # echo: Optional[bool] = False
    # repetition_penalty: Optional[float] = 1.0
    # min_p: Optional[float] = 0.0


class CompletionRequest(BaseModel):
    model: str = "chatglm3-6b"
    # a string, array of strings, array of tokens, or array of token arrays
    prompt: Union[str, List[str]] = "hi"
    suffix: Optional[str] = None
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    # logprobs: Optional[int] = None
    echo: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    presence_penalty: Optional[float] = Field(
        default=1.0, ge=0.0, le=2.0)  # 设置最大值为2
    frequency_penalty: Optional[float] = Field(
        default=1.0, ge=0.0, le=2.0)  # 设置最大值为2
    best_of: Optional[int] = None
    # logit_bias: Optional[Dict[str, float]] = None
    # user: Optional[str] = None
    # Additional parameters supported by vLLM
    # top_k: Optional[int] = -1
    # ignore_eos: Optional[bool] = False
    # use_beam_search: Optional[bool] = False
    # stop_token_ids: Optional[List[int]] = Field(default_factory=list)
    # skip_special_tokens: Optional[bool] = True
    # spaces_between_special_tokens: Optional[bool] = True
    # repetition_penalty: Optional[float] = 1.0
    # min_p: Optional[float] = 0.0


class LogProbs(BaseModel):
    text_offset: List[int] = Field(default_factory=list)
    token_logprobs: List[Optional[float]] = Field(default_factory=list)
    tokens: List[str] = Field(default_factory=list)
    top_logprobs: Optional[List[Optional[Dict[int, float]]]] = None


class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{random_uuid()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo


class CompletionResponseStreamChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class CompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{random_uuid()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseStreamChoice]
    usage: Optional[UsageInfo]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(
        default=None, description="data about request and response")


class AudioSpeechRequest(BaseModel):
    model: str
    voice: str
    input: str
    clone_url: str = ""


class AudioTranscriptionsRequest(BaseModel):
    model: str
    file: UploadFile = File(...)


class EmbeddingsRequest(BaseModel):
    model: str
    input: Optional[List[str]] = []


class RerankRequest(BaseModel):
    model: str
    query: str
    documents: List[str]
    top_n: Optional[int] = 2
    return_documents: Optional[bool] = False


class VideoGenerationsRequest(BaseModel):
    model: str
    prompt: str
    seed: int = -1
    width: int = 720
    height: int = 480
    num_frames: int = Field(8, ge=1, le=16)


class SamplerName(str, Enum):
    euler = "euler"
    euler_cfg_pp = "euler_cfg_pp"
    euler_ancestral = "euler_ancestral"
    euler_ancestral_cfg_pp = "euler_ancestral_cfg_pp"
    heun = "heun"
    heunpp2 = "heunpp2"
    dpm_2 = "dpm_2"
    dpm_2_ancestral = "dpm_2_ancestral"
    lms = "lms"
    dpm_fast = "dpm_fast"
    dpm_adaptive = "dpm_adaptive"
    dpmpp_2s_ancestral = "dpmpp_2s_ancestral"
    dpmpp_sde = "dpmpp_sde"
    dpmpp_sde_gpu = "dpmpp_sde_gpu"
    dpmpp_2m = "dpmpp_2m"
    dpmpp_2m_sde = "dpmpp_2m_sde"
    dpmpp_2m_sde_gpu = "dpmpp_2m_sde_gpu"
    dpmpp_3m_sde = "dpmpp_3m_sde"
    dpmpp_3m_sde_gpu = "dpmpp_3m_sde_gpu"
    ddpm = "ddpm"
    lcm = "lcm"
    ipndm = "ipndm"
    ipndm_v = "ipndm_v"
    deis = "deis"
    ddim = "ddim"
    uni_pc = "uni_pc"
    uni_pc_bh2 = "uni_pc_bh2"


class Scheduler(str, Enum):
    normal = "normal"
    karras = "karras"
    exponential = "exponential"
    sgm_uniform = "sgm_uniform"
    simple = "simple"
    ddim_uniform = "ddim_uniform"
    beta = "beta"


class SD15BaseGenerateImageRequest(BaseModel):
    model: str = "sd1.5/majicmixRealistic_betterV6.safetensors"
    seed: int = 0
    steps: int = 20
    batch_size: int = 1
    width: int = 512
    height: int = 512
    sampler_name: SamplerName = SamplerName("euler")  # 使用枚举类型
    cfg: int = 8
    denoise_strength: float = 0.75
    scheduler: Scheduler = Scheduler("normal")  # 使用枚举类型
    prompt: str = "beautiful scenery nature glass bottle landscape, purple galaxy bottle"
    negative_prompt: str = "text, watermark"
    image_url: str = ""


class BaseGenerateImageRequest(BaseModel):
    model: str = "sd1.5/majicmixRealistic_betterV6.safetensors"
    seed: int = 0
    steps: int = 20
    batch_size: int = 1
    width: int = 512
    height: int = 512
    sampler_name: SamplerName = SamplerName("euler")  # 使用枚举类型
    cfg: int = 8
    denoise_strength: float = 0.75
    scheduler: Scheduler = Scheduler("normal")  # 使用枚举类型
    prompt: str = "beautiful scenery nature glass bottle landscape, purple galaxy bottle"
    negative_prompt: str = "text, watermark"
    image_url: str = ""


##############################
