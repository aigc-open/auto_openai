
import os
import gradio as gr
from auto_openai.utils.init_env import global_config
from auto_openai import project_path
import pandas as pd
from typing import Dict, Any
from auto_openai.utils.openai import ChatCompletionRequest, CompletionRequest,  AudioSpeechRequest, \
    EmbeddingsRequest, RerankRequest, AudioTranscriptionsRequest, SolutionBaseGenerateImageRequest, VideoGenerationsRequest, SD15MultiControlnetGenerateImageRequest, SD15ControlnetUnit
from auto_openai.utils.public import CustomRequestMiddleware, redis_client, s3_client
from auto_openai.utils.openai import Scheduler
from openai import OpenAI
from fastapi import FastAPI, Request, Body, Header, Query
from nicegui import ui
from pathlib import Path
from urllib.parse import urlparse

web_prefix = ""


# @app.on_connect
# def handle_connect(socket):
#     request = socket.request
#     url_info = urlparse(str(request.url))
#     print(url_info)


def format_constraints(constraints):
    def format_constraint(constraint):
        constraint_strs = []
        if constraint.get("type") == "integer":
            constraint_strs.append("类型: integer")
        elif constraint.get("type") == "boolean":
            constraint_strs.append("类型: boolean")
        elif constraint.get("type") == "string":
            constraint_strs.append("类型: string")
        elif constraint.get("type") == "array":
            constraint_strs.append(
                f"类型: array")
        elif constraint.get("type") == "null":
            constraint_strs.append("类型: null")
        else:
            constraint_strs.append(f"类型: {constraint.get('type')}")

        if "exclusiveMaximum" in constraint and "exclusiveMinimum" in constraint:
            constraint_strs.append(
                f"约束: {constraint['exclusiveMinimum']} < value < {constraint['exclusiveMaximum']}")
        elif "maximum" in constraint and "minimum" in constraint:
            constraint_strs.append(
                f"约束: {constraint['minimum']} ≤ value ≤ {constraint['maximum']}")

        return constraint_strs
    if not constraints:
        return ""
    constraint_strs = []
    if type(constraints) == list:
        for constraint in constraints:
            constraint_strs.extend(format_constraint(constraint))
    elif type(constraints) == dict:
        constraint_strs = format_constraint(constraints)
    return " 或 ".join(constraint_strs)


def generate_api_documentation(schema: Dict[str, Any]):
    # 创建一个空列表来存储每个属性的信息
    data = []
    # 处理属性
    properties = schema.get('properties', {})
    for prop_name, prop_details in properties.items():
        title = prop_details.get('title', prop_name)
        default = prop_details.get('default', '无')
        prop_type = format_constraints(prop_details.get("anyOf"))
        prop_type = prop_type if prop_type else format_constraints(
            prop_details)
        description = prop_details.get('description', '无描述')
        # 处理枚举类型
        if '$ref' in prop_details:
            ref = prop_details['$ref']
            # 动态查找枚举值
            enum_values = schema['$defs'].get(
                ref.split('/')[-1], {}).get('enum', [])
            if enum_values:
                description = f" (可选值: {', '.join(enum_values)})"
                prop_type = "string"  # 假设枚举类型为字符串
        elif 'allOf' in prop_details:
            for item in prop_details['allOf']:
                if '$ref' in item:
                    ref = item['$ref']
                    # 动态查找枚举值
                    enum_values = schema['$defs'].get(
                        ref.split('/')[-1], {}).get('enum', [])
                    if enum_values:
                        description = f" (可选值: {', '.join(enum_values)})"
                        prop_type = "string"  # 假设枚举类型为字符串

        current_data = {
            '名称': prop_name,
            '约束/类型': prop_type,
            '默认值/参考': f'{default}',
            '描述': description
        }
        # 将属性信息添加到列表中
        data.append(current_data)
    df = pd.DataFrame(data)
    return df


class UILayout:
    home_readme = os.path.join(project_path, "README.md")
    demo_path = os.path.join(project_path, "web/tests")

    def _content_page_(self, model_config, model_type,
                       model_headers=["name", "description"],
                       model_headers_desc=["名称", "描述"],
                       RequestBaseModel=[]):
        if not model_config:
            ui.markdown("# 努力开发中...")
            return
        with ui.tabs() as tabs:
            ui.tab("支持的模型列表", label="支持的模型列表")
            ui.tab("文档参数说明", label="文档参数说明")
            ui.tab("python 示例", label="python 示例")
            ui.tab("curl 示例", label="curl 示例")

        model_list = []
        for m in model_config:
            h_ = []
            for i in model_headers:
                h_.append(m[i])
            model_list.append(h_)

        with ui.tab_panels(tabs, value='支持的模型列表').classes('w-full'):
            with ui.tab_panel('支持的模型列表').classes('w-full'):
                ui.table.from_pandas(pd.DataFrame(data=model_list, columns=model_headers_desc), pagination=10).classes(
                    'w-full h-full flex justify-start')

        with ui.tab_panels(tabs, value='文档参数说明').classes('w-full'):
            with ui.tab_panel('文档参数说明').classes('w-full'):
                for r_basemodel in RequestBaseModel:
                    ui.markdown(f"# {r_basemodel.__name__}")
                    data = generate_api_documentation(
                        r_basemodel.model_json_schema())
                    ui.table.from_pandas(pd.DataFrame(data=data)).classes(
                        'w-full h-full flex justify-start')

        with ui.tab_panels(tabs, value='python 示例').classes('w-full'):
            with ui.tab_panel('python 示例').classes('w-full'):
                py_path = os.path.join(self.demo_path, f"{model_type}.py")
                ui.code(self.read_file(py_path), language="python")

        with ui.tab_panels(tabs, value='curl 示例').classes('w-full'):
            with ui.tab_panel('curl 示例').classes('w-full'):
                curl_path = os.path.join(
                    self.demo_path, f"{model_type}.sh")
                ui.code(self.read_file(curl_path), language="shell")

    def read_file(self, file):
        if os.path.exists(file):
            with open(file, "r") as f:
                return f.read()
        else:
            return "# 努力开发中..."

    def header(self):
        with ui.header().classes('bg-blue-500 text-white flex items-center p-4'):
            ui.button('首页', on_click=lambda: ui.navigate.to(
                '/')).classes('mr-2')
            ui.button('模型广场', on_click=lambda: ui.navigate.to(
                f'{web_prefix}/docs-models')).classes('mr-2')
            ui.button('性能查看', on_click=lambda: ui.navigate.to(
                f'{web_prefix}/docs-performance')).classes('mr-2')
            ui.button('系统分布式虚拟节点', on_click=lambda: ui.navigate.to(
                f'{web_prefix}/docs-distributed_nodes'))

    def home_page(self):
        ui.markdown(self.read_file(self.home_readme))

    def model_plaza(self):
        data = global_config.get_MODELS_MAPS()
        with ui.tabs() as tabs:
            for model_type in data:
                ui.tab(model_type, label=model_type)

        with ui.tab_panels(tabs, value='LLM').classes('w-full'):
            with ui.tab_panel('LLM'):
                self._content_page_(
                    model_config=data.get("LLM"),
                    model_type="LLM",
                    model_headers=[
                        "name", "model_max_tokens", "description"],
                    model_headers_desc=["名称", "最大支持tokens", "描述"],
                    RequestBaseModel=[
                        ChatCompletionRequest, CompletionRequest]
                )
            with ui.tab_panel('VLLM'):
                self._content_page_(
                    model_config=data.get("VLLM"),
                    model_type="VLLM",
                    model_headers=[
                        "name", "model_max_tokens", "description"],
                    model_headers_desc=["名称", "最大支持tokens", "描述"],
                    RequestBaseModel=[ChatCompletionRequest]
                )
            with ui.tab_panel('SD15MultiControlnetGenerateImage'):
                self._content_page_(
                    model_config=data.get(
                        "SD15MultiControlnetGenerateImage"),
                    model_type="SD15MultiControlnetGenerateImage",
                    RequestBaseModel=[
                        SD15MultiControlnetGenerateImageRequest, SD15ControlnetUnit]
                )
            with ui.tab_panel('SolutionBaseGenerateImage'):
                with gr.Tab("SolutionBaseGenerateImage"):
                    self._content_page_(
                        model_config=data.get("SolutionBaseGenerateImage"),
                        model_type="SolutionBaseGenerateImage",
                        RequestBaseModel=[SolutionBaseGenerateImageRequest]
                    )
            with ui.tab_panel('Embedding'):
                self._content_page_(
                    model_config=data.get("Embedding"),
                    model_type="Embedding",
                    RequestBaseModel=[EmbeddingsRequest]
                )
            with ui.tab_panel('Rerank'):
                self._content_page_(
                    model_config=data.get("Rerank"),
                    model_type="Rerank",
                    RequestBaseModel=[RerankRequest]
                )
            with ui.tab_panel('TTS'):
                self._content_page_(
                    model_config=data.get("TTS"),
                    model_type="TTS",
                    RequestBaseModel=[AudioSpeechRequest]
                )
            with ui.tab_panel('ASR'):
                self._content_page_(
                    model_config=data.get("ASR"),
                    model_type="ASR",
                    RequestBaseModel=[AudioTranscriptionsRequest]
                )
            with ui.tab_panel('Video'):
                self._content_page_(
                    model_config=data.get("Video"),
                    model_type="Video",
                    RequestBaseModel=[VideoGenerationsRequest]
                )

    def performance_view(self):
        def convert_to_dataframe():
            scheduler = Scheduler(redis_client=redis_client, http_request=None,
                                  queue_timeout=global_config.QUEUE_TIMEOUT, infer_timeout=global_config.INFER_TIMEOUT)
            data = scheduler.get_profiler()
            if not data.get("start_server_time") or not data.get("Profiler"):
                return pd.DataFrame(), pd.DataFrame()
            start_server_time_df = pd.DataFrame.from_dict(
                data["start_server_time"], orient='index') if data.get("start_server_time") else pd.DataFrame()
            tps_spi_df = pd.DataFrame.from_dict(
                data["Profiler"], orient='index') if data.get("Profiler") else pd.DataFrame()

            # 添加模型名称列
            start_server_time_df.reset_index(inplace=True)
            start_server_time_df.columns = [
                'Model Name', 'Max', 'Min', 'New', "Description"]
            tps_spi_df.reset_index(inplace=True)
            tps_spi_df.columns = ['Model Name', 'Max',
                                  'Min', 'New', "Description"]

            return start_server_time_df, tps_spi_df
        start_server_time_df, tps_spi_df = convert_to_dataframe()
        ui.markdown("""
        ## 模型加载时间
        > 注意：模型加载时间是指模型从加载到可以开始处理请求的时间。""")
        ui.table.from_pandas(start_server_time_df).classes('w-full text-left')
        ui.markdown("""
        ## 模型性能
        - 模型性能是指模型在处理请求时的性能指标。
        - 大语言模型: 每秒生成token的数量。
        - 图像生成: 每张图像生成所需时间。
        - 语音识别：每秒处理音频帧的所需时间。
        - 语音合成：每秒处理音频帧的所需时间。
        """)

        ui.table.from_pandas(tps_spi_df).classes('w-full text-left')

    def distributed_nodes(self):
        def convert_to_dataframe():
            scheduler = Scheduler(redis_client=redis_client, http_request=None)
            data = scheduler.get_running_node()
            if data:
                df = pd.DataFrame(data)
                df['device-ids'] = df['device-ids'].apply(
                    lambda x: ','.join(map(str, x)) if isinstance(x, list) else str(x))
                return df
            else:
                return pd.DataFrame([])
        node_df = convert_to_dataframe()
        ui.markdown("""
        ## 虚拟节点
        > 注意：真实的物理节点可能被虚拟成多个虚拟节点，每个虚拟节点可以处理一个请求。
        - 每个虚拟节点独占自己的卡，不会被其他卡获取
        """)
        ui.table.from_pandas(node_df).classes('w-full text-left')


layout = UILayout()


class UIWeb:

    @ui.page('/')
    @staticmethod
    def index():
        layout.header()
        layout.home_page()

    @ui.page(f'{web_prefix}/docs-models')
    @staticmethod
    def models():
        layout.header()
        layout.model_plaza()

    @ui.page(f'{web_prefix}/docs-performance')
    @staticmethod
    def performance():
        layout.header()
        layout.performance_view()

    @ui.page(f'{web_prefix}/docs-distributed_nodes')
    @staticmethod
    def distributed_nodes():
        layout.header()
        layout.distributed_nodes()

    @classmethod
    def register_ui(cls, fastapi_app, mount_path='/'):
        ui.run_with(
            fastapi_app,
            title="AutoOpenai 本地大模型",
            binding_refresh_interval=10,
            # NOTE this can be omitted if you want the paths passed to @ui.page to be at the root
            mount_path=mount_path,
            # NOTE setting a secret is optional but allows for persistent storage per user
            storage_secret='pick your private secret here',
        )
