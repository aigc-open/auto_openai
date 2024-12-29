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
import requests
from PIL import Image
from io import BytesIO
import webuiapi
import re


def extract_requested_tokens(text):
    pattern = r"This model's maximum context lengeh is (\d+) tokens. However, you requested (\d+) tokens"
    match = re.search(pattern, text)
    if match:
        try:
            return int(match.group(2)) - int(match.group(1))  # 返回提取到的数字
        except:
            return 0
    return 0


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
    code: Optional[int] = None
    diff_max_tokens: int = 0

    def update_max_tokens(self):
        self.diff_max_tokens = extract_requested_tokens(text=self.message)


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
    model: str = Field(..., description="模型名称")
    messages: Union[List[Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, str]]]]]]]] = [
        {"role": "user", "content": "hi"}]
    temperature: Optional[float] = Field(
        default=0.7, ge=0, le=1, description="控制生成文本的随机性")
    top_p: Optional[float] = Field(
        default=1.0, ge=0, le=1, description="top_p")
    n: Optional[int] = 1
    max_tokens: Optional[int] = Field(default=512, description="生成文本的最大长度")
    stop: Optional[Union[str, List[str]]] = Field(
        default=[], description="停止生成文本的标志")
    stream: Optional[bool] = Field(default=False, description="是否流式输出")
    presence_penalty: Optional[float] = Field(
        default=2.0, ge=0.0, le=2.0)  # 设置最大值为2
    frequency_penalty: Optional[float] = Field(
        default=2.0, ge=0.0, le=2.0)  # 设置最大值为2
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
    model: str = Field(..., description="模型名称")
    # a string, array of strings, array of tokens, or array of token arrays
    prompt: str = Field(..., description="输入提示词")
    suffix: Optional[str] = Field("", description="生成文本的结尾")
    max_tokens: Optional[int] = Field(
        default=16, ge=0, description="生成文本的最大长度")
    temperature: Optional[float] = Field(
        default=1.0, ge=0.0, le=2.0, description="控制生成文本的随机性")  # 设置最大值为2
    top_p: Optional[float] = Field(
        default=1.0, ge=0.0, le=1.0, description="top_p")
    n: Optional[int] = 1
    stream: Optional[bool] = Field(default=False, description="是否流式输出")
    # logprobs: Optional[int] = None
    # echo: Optional[bool] = Field(default=False, description="是否回显输入")
    stop: Optional[Union[str, List[str]]] = Field(
        default_factory=list, description="停止生成文本的标志")
    presence_penalty: Optional[float] = Field(
        default=1.0, ge=0.0, le=2.0)  # 设置最大值为2
    frequency_penalty: Optional[float] = Field(
        default=1.0, ge=0.0, le=2.0)  # 设置最大值为2
    # best_of: Optional[int] = None
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
    finish_reason: Optional[Literal["stop", "lengeh"]] = None


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
    finish_reason: Optional[Literal["stop", "lengeh"]] = None


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
    finish_reason: Optional[Literal["stop", "lengeh"]] = None


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
    finish_reason: Optional[Literal["stop", "lengeh"]] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(
        default=None, description="data about request and response")


class AudioSpeechRequest(BaseModel):
    model: str = Field(..., description="模型名称")
    voice: str = Field(..., description="语音名称")
    input: str = Field(..., description="输入文本")
    clone_url: str = Field(..., description="克隆音色地址")


class AudioTranscriptionsRequest(BaseModel):
    model: str = Field(..., description="模型名称")
    file: UploadFile = File(..., description="音频文件")


class EmbeddingsRequest(BaseModel):
    model: str = Field(..., description="模型名称")
    input: Optional[List[str]] = Field(..., description="输入文本")


class RerankRequest(BaseModel):
    model: str = Field(..., description="模型名称")
    query: str = Field(..., description="查询文本")
    documents: List[str] = Field(..., description="文档列表")
    top_n: Optional[int] = Field(2, ge=1,  description="返回的文档数量")
    return_documents: Optional[bool] = Field(False, description="是否返回文档")


class VideoGenerationsRequest(BaseModel):
    model: str = Field(..., description="模型名称")
    prompt: str = Field(..., description="输入文本")
    seed: int = Field(123456, ge=-1,  description="随机种子")
    width: int = Field(720, ge=64, le=1024, description="视频宽度")
    height: int = Field(480, ge=64, le=1024, description="视频高度")
    num_frames: int = Field(8, ge=1, le=48, description="视频帧数")


class SamplerName(str, Enum):
    DPM_PLUS_PLUS_2M = 'DPM++ 2M'
    DPM_PLUS_PLUS_SDE = 'DPM++ SDE'
    DPM_PLUS_PLUS_2M_SDE = 'DPM++ 2M SDE'
    DPM_PLUS_PLUS_2M_SDE_HEUN = 'DPM++ 2M SDE Heun'
    DPM_PLUS_PLUS_2S_A = 'DPM++ 2S a'
    DPM_PLUS_PLUS_3M_SDE = 'DPM++ 3M SDE'
    EULER_A = 'Euler a'
    EULER = 'Euler'
    LMS = 'LMS'
    HEUN = 'Heun'
    DPM2 = 'DPM2'
    DPM2_A = 'DPM2 a'
    DPM_FAST = 'DPM fast'
    DPM_ADAPTIVE = 'DPM adaptive'
    RESTART = 'Restart'
    DDIM = 'DDIM'
    DDIM_CFG_PLUS_PLUS = 'DDIM CFG++'
    PLMS = 'PLMS'
    UNIPC = 'UniPC'
    LCM = 'LCM'


class Scheduler(str, Enum):
    AUTOMATIC = 'Automatic'
    UNIFORM = 'Uniform'
    KARRAS = 'Karras'
    EXPONENTIAL = 'Exponential'
    POLYEXPONENTIAL = 'Polyexponential'
    SGM_UNIFORM = 'SGM Uniform'
    KL_OPTIMAL = 'KL Optimal'
    ALIGN_YOUR_STEPS = 'Align Your Steps'
    SIMPLE = 'Simple'
    NORMAL = 'Normal'
    DDIM = 'DDIM'
    BETA = 'Beta'


class ControlnetModule(str, Enum):
    NONE = 'none'
    # IP_ADAPTER_AUTO = 'ip-adapter-auto'
    TILE_RESAMPLE = 'tile_resample'
    PIDINET = 'pidinet'
    # ONEFORMER_ADE20K = 'oneformer_ade20k'
    # PIDINET_SCRIBBLE = 'pidinet_scribble'
    REVISION_CLIPVISION = 'revision_clipvision'
    REFERENCE_ONLY = 'reference_only'
    RECOLOR_LUMINANCE = 'recolor_luminance'
    OPENPOSE_FULL = 'openpose_full'
    NORMAL_BAE = 'normal_bae'
    MLSD = 'mlsd'
    LINEART_STANDARD = 'lineart_standard'
    # IP_ADAPTER_CLIP_SD15 = 'ip-adapter_clip_sd15'
    INPAINT_ONLY = 'inpaint_only'
    DEPTH = 'depth'
    CANNY = 'canny'
    # INVERT = 'invert'
    # TILE_COLORFIX_SHARP = 'tile_colorfix+sharp'
    # TILE_COLORFIX = 'tile_colorfix'
    # THRESHOLD = 'threshold'
    # CLIP_VISION = 'clip_vision'
    # PIDINET_SKETCH = 'pidinet_sketch'
    # COLOR = 'color'
    SOFTEDGE_TEED = 'softedge_teed'
    PIDINET_SAFE = 'pidinet_safe'
    HED_SAFE = 'hed_safe'
    HED = 'hed'
    SOFTEDGE_ANYLINE = 'softedge_anyline'
    SHUFFLE = 'shuffle'
    # SEGMENTATION = 'segmentation'
    # ONEFORMER_COCO = 'oneformer_coco'
    # ANIME_FACE_SEGMENT = 'anime_face_segment'
    # SCRIBBLE_XDOG = 'scribble_xdog'
    SCRIBBLE_HED = 'scribble_hed'
    # REVISION_IGNORE_PROMPT = 'revision_ignore_prompt'
    # REFERENCE_ADAIN_ATT = 'reference_adain+attn'
    # REFERENCE_ADAIN = 'reference_adain'
    # RECOLOR_INTENSITY = 'recolor_intensity'
    # OPENPOSE_HAND = 'openpose_hand'
    # OPENPOSE_FACEONLY = 'openpose_faceonly'
    # OPENPOSE_FACE = 'openpose_face'
    OPENPOSE = 'openpose'
    # NORMAL_MAP = 'normal_map'
    # NORMAL_DSINE = 'normal_dsine'
    # MOBILE_SAM = 'mobile_sam'
    # MEDIAPIPE_FACE = 'mediapipe_face'
    LINEART = 'lineart'
    # LINEART_COARSE = 'lineart_coarse'
    # LINEART_ANIME_DENOISE = 'lineart_anime_denoise'
    # LINEART_ANIME = 'lineart_anime'
    # IP_ADAPTER_PULID = 'ip-adapter_pulid'
    IP_ADAPTER_FACE_ID_PLUS = 'ip-adapter_face_id_plus'
    # IP_ADAPTER_FACE_ID = 'ip-adapter_face_id'
    # IP_ADAPTER_CLIP_SDXL_PLUS_VITH = 'ip-adapter_clip_sdxl_plus_vith'
    # IP_ADAPTER_CLIP_SDXL = 'ip-adapter_clip_sdxl'
    INSTANT_ID_FACE_KEYPOINTS = 'instant_id_face_keypoints'
    INSTANT_ID_FACE_EMBEDDING = 'instant_id_face_embedding'
    # INPAINT_ONLY_LAMA = 'inpaint_only+lama'
    INPAINT = 'inpaint'
    # FACEXLIB = 'facexlib'
    # DW_OPENPOSE_FULL = 'dw_openpose_full'
    # DEPTH_ZOE = 'depth_zoe'
    # DEPTH_LERES_PLUS_PLUS = 'depth_leres++'
    # DEPTH_LERES = 'depth_leres'
    # DEPTH_HAND_REFINER = 'depth_hand_refiner'
    # DEPTH_ANYTHING_V2 = 'depth_anything_v2'
    # DEPTH_ANYTHING = 'depth_anything'
    # DENSEPOSE_PARULA = 'densepose_parula'
    # DENSEPOSE = 'densepose'
    # BLUR_GAUSSIAN = 'blur_gaussian'
    # ANIMAL_OPENPOSE = 'animal_openpose'


class ControlnetResizeMode(str, Enum):
    RESIZE_AND_FILL = "Resize and Fill"
    RESIZE_AND_CROP = "Crop and Resize"
    RESIZE = "Just Resize"


class SD15ControlnetUnit(BaseModel):
    image_url: str = Field(default="", title="图片URL")
    mask_url: str = Field(default="", title="遮罩URL")
    module: ControlnetModule = Field(ControlnetModule("none"), title="控制器模块")
    weight: float = Field(default=1.0, ge=0.0, le=1.0, title="权重")
    # resize_mode: ControlnetResizeMode = ControlnetResizeMode.RESIZE_AND_FILL.value
    guidance_start: float = Field(0.0, ge=0.0, le=1.0, description="引导开始")
    guidance_end: float = Field(1.0, ge=0.0, le=1.0, description="引导结束")


class SD15MultiControlnetGenerateImageRequest(BaseModel):
    model: str = Field(
        "sd1.5/majicmixRealistic_betterV6.safetensors", description="模型名称")
    seed: int = Field(12345, description="随机种子")
    steps: int = Field(20, description="生成步骤数")
    batch_size: int = Field(1, ge=1, le=4, description="批量大小")
    width: int = Field(512, ge=32, le=2048, description="生成图像的宽度")
    height: int = Field(512, ge=32, le=2048, description="生成图像的高度")
    sampler_name: SamplerName = Field(
        SamplerName("Euler"), description="采样器名称")  # 使用枚举类型
    cfg: int = Field(7.5, ge=1, le=30, description="CFG Scale")
    denoising_strength: float = Field(0.75, ge=0, le=1, description="去噪强度")
    scheduler: Scheduler = Field(
        Scheduler("Normal"), description="调度器名称")  # 使用枚举类型
    prompt: str = Field(
        "beautiful scenery nature glass bottle landscape, purple galaxy bottle", description="提示词")
    negative_prompt: str = Field("text, watermark", description="负面提示词")
    image_url: str = Field("", description="参考图像URL:图生图")
    # style: list = []
    controlnets: List[SD15ControlnetUnit] = Field(
        [], description="控制图像生成: SD15ControlnetUnit")


class SD15MultiControlnetGenerateImage(SD15MultiControlnetGenerateImageRequest):

    def ImageFromUrl(self, url: str):
        response = requests.get(url)
        image_data = BytesIO(response.content)
        image = Image.open(image_data)
        return image

    def controlnet_model_map(self, module):
        return {
            ControlnetModule.CANNY.value: "control_v11p_sd15_canny",
            ControlnetModule.SHUFFLE.value: "control_v11e_sd15_shuffle",
            ControlnetModule.TILE_RESAMPLE.value: "control_v11f1e_sd15_tile",
            ControlnetModule.DEPTH.value: "control_v11f1p_sd15_depth",
            ControlnetModule.INPAINT.value: "control_v11p_sd15_inpaint",
            ControlnetModule.OPENPOSE.value: "control_v11p_sd15_openpose",
            ControlnetModule.SCRIBBLE_HED.value: "control_v11p_sd15_scribble",
            ControlnetModule.SOFTEDGE_ANYLINE.value: "control_v11p_sd15_softedge",
            ControlnetModule.IP_ADAPTER_FACE_ID_PLUS.value: "ip-adapter-plus-face_sd15"
        }.get(module, None)

    def convert_webui_data(self):
        data = {
            "model": self.model.replace("SD15MultiControlnetGenerateImage/", ""),
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "seed": self.seed,
            "sampler_name": self.sampler_name.value,
            "cfg_scale": self.cfg,
            "denoising_strength": self.denoising_strength,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "width": self.width,
            "height": self.height,
            "scheduler": self.scheduler.value
        }
        if self.image_url:
            data["images"] = [self.ImageFromUrl(self.image_url)]
        controlnets = []
        if self.controlnets:
            for item in self.controlnets:
                image = self.ImageFromUrl(item.image_url)
                if item.mask_url:
                    mask = self.ImageFromUrl(item.mask_url)
                else:
                    mask = None
                controlnets.append(webuiapi.ControlNetUnit(
                    image=image, mask=mask, module=item.module,
                    model=self.controlnet_model_map(item.module),
                    guidance_end=item.guidance_end,
                    guidance_start=item.guidance_start,
                    # resize_mode=item.resize_mode,
                    weight=item.weight
                ))
        data["controlnet_units"] = controlnets
        return data


class SolutionBaseGenerateImageRequest(BaseModel):
    model: str = Field(
        "sd1.5/majicmixRealistic_betterV6.safetensors", description="模型名称")
    seed: int = Field(1234, description="随机种子")
    steps: int = Field(20, description="生成步数")
    batch_size: int = Field(1, ge=1, le=4, description="生成批次")
    width: int = Field(512, ge=32, le=2048, description="生成图片宽度")
    height: int = Field(512, ge=32, le=2048, description="生成图片高度")
    cfg: int = Field(8, ge=1, le=30, description="控制生成细节")
    denoising_strength: float = Field(0.75, ge=0.0, le=1.0, description="去噪强度")
    prompt: str = Field(
        "beautiful scenery nature glass bottle landscape, purple galaxy bottle", description="生成图片的描述")
    negative_prompt: str = Field("text, watermark", description="生成图片的负面描述")
    image_url: str = Field("", description="参考图像URL:图生图")


##############################
