import os
import gradio as gr
from auto_openai.utils.init_env import global_config
from auto_openai.utils.openai import ChatCompletionRequest, CompletionRequest,  \
    AudioSpeechRequest, \
    EmbeddingsRequest, RerankRequest, AudioTranscriptionsRequest, \
    SolutionBaseGenerateImageRequest, VideoGenerationsRequest, \
    SD15MultiControlnetGenerateImageRequest, SD15ControlnetUnit
from nicegui import ui
from .components import DocContentZoneComponent


def index():
    data = global_config.get_MODELS_MAPS()
    with ui.tabs() as tabs:
        for model_type in data:
            ui.tab(model_type, label=model_type)

    with ui.tab_panels(tabs, value='LLM').classes('w-full'):
        with ui.tab_panel('LLM'):
            DocContentZoneComponent.index(
                model_config=data.get("LLM"),
                model_type="LLM",
                model_headers=[
                    "name", "model_max_tokens", "description", "model_url"],
                model_headers_desc=["名称", "最大支持tokens", "描述", "官网"],
                RequestBaseModel=[
                    ChatCompletionRequest, CompletionRequest]
            )
        with ui.tab_panel('VLLM'):
            DocContentZoneComponent.index(
                model_config=data.get("VLLM"),
                model_type="VLLM",
                model_headers=[
                    "name", "model_max_tokens", "description", "model_url"],
                model_headers_desc=["名称", "最大支持tokens", "描述", "官网"],
                RequestBaseModel=[ChatCompletionRequest]
            )
        with ui.tab_panel('SD15MultiControlnetGenerateImage'):
            DocContentZoneComponent.index(
                model_config=data.get(
                    "SD15MultiControlnetGenerateImage"),
                model_type="SD15MultiControlnetGenerateImage",
                RequestBaseModel=[
                    SD15MultiControlnetGenerateImageRequest, SD15ControlnetUnit]
            )
        with ui.tab_panel('SolutionBaseGenerateImage'):
            with gr.Tab("SolutionBaseGenerateImage"):
                DocContentZoneComponent.index(
                    model_config=data.get("SolutionBaseGenerateImage"),
                    model_type="SolutionBaseGenerateImage",
                    RequestBaseModel=[SolutionBaseGenerateImageRequest]
                )
        with ui.tab_panel('Embedding'):
            DocContentZoneComponent.index(
                model_config=data.get("Embedding"),
                model_type="Embedding",
                RequestBaseModel=[EmbeddingsRequest]
            )
        with ui.tab_panel('Rerank'):
            DocContentZoneComponent.index(
                model_config=data.get("Rerank"),
                model_type="Rerank",
                RequestBaseModel=[RerankRequest]
            )
        with ui.tab_panel('TTS'):
            DocContentZoneComponent.index(
                model_config=data.get("TTS"),
                model_type="TTS",
                RequestBaseModel=[AudioSpeechRequest]
            )
        with ui.tab_panel('ASR'):
            DocContentZoneComponent.index(
                model_config=data.get("ASR"),
                model_type="ASR",
                RequestBaseModel=[AudioTranscriptionsRequest]
            )
        with ui.tab_panel('Video'):
            DocContentZoneComponent.index(
                model_config=data.get("Video"),
                model_type="Video",
                RequestBaseModel=[VideoGenerationsRequest]
            )
