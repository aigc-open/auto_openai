import os
import gradio as gr
from auto_openai.utils.init_env import global_config
from auto_openai import project_path
import pandas as pd
from typing import Dict, Any
from auto_openai.utils.openai import ChatCompletionRequest, CompletionRequest,  \
    AudioSpeechRequest, \
    EmbeddingsRequest, RerankRequest, AudioTranscriptionsRequest, \
    SolutionBaseGenerateImageRequest, VideoGenerationsRequest, \
    SD15MultiControlnetGenerateImageRequest, SD15ControlnetUnit
from auto_openai.utils.public import CustomRequestMiddleware, redis_client, s3_client
from auto_openai.utils.openai import Scheduler
from auto_openai.utils.depends import get_running_models, get_models_config_list
from auto_openai.utils.public import scheduler
from auto_openai.utils.support_models.model_config import all_supported_device, system_models_config
from openai import AsyncOpenAI
from fastapi import FastAPI, Request, Body, Header, Query
from nicegui import ui
from pathlib import Path
from urllib.parse import urlparse
import plotly.graph_objects as go
from PIL import Image
import asyncio
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


class ExperienceZone:
    base_url = os.environ.get(
        "OPENAI_BASE_URL", "http://127.0.0.1:9000/openai/v1")
    api_key = "xxxx"
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    # async with httpx.AsyncClient(base_url="http://127.0.0.1:8000/") as client:
    #     response = await client.get("/")
    #     response.raise_for_status()
    #     return response.json()

    @classmethod
    def create_llm_chat(self, model_name):
        import asyncio

        # 创建主容器
        with ui.card().classes('w-full max-w-7xl mx-auto p-6 shadow-lg rounded-xl') as card:
            # 顶部标题和输入区域
            with ui.column().classes('w-full gap-4 mb-6'):
                # 标题区域
                with ui.row().classes('w-full items-center mb-2'):
                    ui.icon('chat').classes('text-3xl text-blue-600 mr-2')
                    ui.label('AI 助手').classes(
                        'text-2xl font-bold text-gray-800')

                # 输入区域
                with ui.card().classes('w-full bg-white p-4 rounded-xl shadow-sm'):
                    with ui.row().classes('w-full gap-4 items-end'):
                        with ui.column().classes('flex-grow'):
                            prompt = ui.input(
                                value="请你使用python 写3种排序算法",
                                label='提示词',
                                placeholder='请输入您想问的问题...'
                            ).props('filled outlined').classes('w-full')

                        # 按钮区域
                        with ui.row().classes('gap-2 shrink-0'):
                            send = ui.button('发送', icon='send').classes(
                                'bg-blue-600 text-white px-6 py-2 rounded-lg '
                                'hover:bg-blue-700 transition-colors'
                            ).props('flat')
                            spinner = ui.spinner(
                                size='sm').classes('text-blue-600')
                            spinner.set_visibility(False)

            # 对话内容区域
            with ui.card().classes('w-full bg-gray-50 rounded-xl'):
                ui.icon('textsms').classes('text-3xl text-gray-500 p-4')
                ui.label('对话内容').classes('text-sm text-gray-500 p-4')
                # 添加滚动容器
                with ui.scroll_area().classes('h-[500px] px-4'):
                    chat_messages = ui.markdown(
                        "等待输入...").classes('prose max-w-full')

        async def on_click():
            if not prompt.value:
                return

            try:
                # 禁用输入和按钮
                spinner.set_visibility(True)
                prompt.disabled = True
                send.visible = False

                # full_response = "逻辑待完善中..."
                # chat_messages.set_content(full_response)


                # 调用API
                response = await self.client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt.value}],
                    stream=True
                )

                # 流式处理响应
                full_response = ""
                async for chunk in response:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        chat_messages.set_content(full_response + "\n```")
                        await asyncio.sleep(0.01)
                chat_messages.set_content(full_response)

            except Exception as e:
                chat_messages.set_content(f"❌ 错误: {str(e)}")
                ui.notify(f"发生错误: {str(e)}", type='negative')

            finally:
                # 清理并重置状态
                spinner.set_visibility(False)
                send.set_visibility(True)

        # 绑定事件处理器
        send.on('click', on_click)
        # 添加回车发送功能
        prompt.on('keydown.enter', on_click)
        return card


class UILayout:
    home_readme = os.path.join(project_path, "README.md")
    demo_path = os.path.join(project_path, "web/tests")

    def _stat_card_(self, label: str, value: int):
        with ui.row().classes('gap-4 p-4 mb-2'):
            with ui.card().classes('flex-1 p-4 bg-blue-50 rounded-xl'):
                ui.label(label).classes(
                    'text-sm text-gray-600 mb-1')
                ui.label(str(value)).classes(
                    'text-2xl font-bold text-blue-600')

    def _models_card_(self, data: pd.DataFrame, columns=3):
        with ui.grid(columns=columns).classes('gap-4 p-4'):
            for _, row in data.iterrows():
                with ui.card().classes('p-4 hover:shadow-lg transition-all duration-300 bg-white border rounded-xl h-full'):
                    # 模型名称 - 使用 column 布局来处理长名称
                    with ui.column().classes('gap-2 mb-3 w-full'):
                        with ui.row().classes('items-center gap-2 mb-1'):
                            ui.icon('model_training').classes(
                                'text-2xl text-purple-600 shrink-0')
                            ui.label(row['名称']).classes(
                                'text-lg font-bold text-gray-800 break-all')

                    # 最大支持tokens（如果存在）
                    if '最大支持tokens' in row:
                        with ui.row().classes('items-center gap-2 mb-2'):
                            ui.icon('data_array').classes(
                                'text-blue-500 shrink-0')
                            ui.label(f"最大支持: {row['最大支持tokens']}").classes(
                                'text-sm text-gray-600')

                    # 描述
                    if '描述' in row:
                        with ui.row().classes('items-start gap-2'):
                            ui.icon('description').classes(
                                'text-gray-400 mt-1 shrink-0')
                            ui.label(row['描述']).classes(
                                'text-sm text-gray-600 break-words')
                    if "官网" in row:
                        if row['官网']:
                            with ui.row().classes('items-center gap-2'):
                                ui.link("官网", target=row['官网'], new_tab=True)

    def _content_page_(self, model_config, model_type,
                       model_headers=["name", "description", "model_url"],
                       model_headers_desc=["名称", "描述", "官网"],
                       RequestBaseModel=[]):
        if not model_config:
            with ui.card().classes('w-full p-8 bg-white rounded-xl shadow-lg'):
                ui.markdown("# 努力开发中...").classes(
                    'text-2xl font-bold text-gray-400')
            return

        with ui.card().classes('w-full bg-white'):
            # Tab navigation
            with ui.tabs().classes('w-full bg-gray-50 border-b') as tabs:
                tab_items = [
                    ("支持的模型列表", "view_list"),
                    ("文档参数说明", "description"),
                    ("python 示例", "code"),
                    ("curl 示例", "terminal")
                ]

                for label, icon in tab_items:
                    ui.tab(name=label, icon=icon)
            # Tab content panels
            with ui.tab_panels(tabs, value="支持的模型列表").classes('w-full'):
                # Models list panel
                with ui.tab_panel('支持的模型列表'):
                    model_list = [[m[i] for i in model_headers]
                                  for m in model_config]
                    df = pd.DataFrame(
                        data=model_list, columns=model_headers_desc)

                    # 添加数据统计卡片
                    self._stat_card_(label="模型总数", value=len(df))
                    # 模型卡片网格
                    self._models_card_(data=df)

                # API documentation panel
                with ui.tab_panel('文档参数说明'):
                    for r_basemodel in RequestBaseModel:
                        with ui.card().classes('mb-6'):
                            ui.markdown(f"# {r_basemodel.__name__}").classes(
                                'p-4 bg-gradient-to-r from-gray-50 to-gray-100 font-bold border-b'
                            )
                            data = generate_api_documentation(
                                r_basemodel.model_json_schema())
                            # 优化文档表格样式
                            ui.table.from_pandas(
                                pd.DataFrame(data=data),
                            ).classes('w-full')

                # Python example panel
                with ui.tab_panel('python 示例'):
                    with ui.card().classes('overflow-hidden rounded-xl border'):
                        py_path = os.path.join(
                            self.demo_path, f"{model_type}.py")
                        ui.code(self.read_file(py_path),
                                language="python").classes('w-full')

                # CURL example panel
                with ui.tab_panel('curl 示例'):
                    with ui.card().classes('overflow-hidden rounded-xl border'):
                        curl_path = os.path.join(
                            self.demo_path, f"{model_type}.sh")
                        ui.code(self.read_file(curl_path),
                                language="shell").classes('w-full')

    def read_file(self, file):
        if os.path.exists(file):
            with open(file, "r") as f:
                return f.read()
        else:
            return "努力开发中..."

    def header(self):
        with ui.header().classes('bg-gradient-to-r from-indigo-600 to-purple-600 text-white shadow-lg p-4'):
            with ui.row().classes('w-full max-w-7xl mx-auto flex justify-between items-center'):
                # Logo section
                with ui.row().classes('flex items-center gap-3'):
                    ui.icon('auto_awesome').classes('text-3xl text-yellow-300')
                    ui.label('AI 调度系统').classes(
                        'text-2xl font-bold tracking-wide')

                # Navigation section
                with ui.row().classes('flex items-center gap-2 ml-auto'):
                    nav_items = [
                        ('首页', '/', 'home'),
                        ('设计', f'{web_prefix}/docs-README', 'architecture'),
                        ('模型广场', f'{web_prefix}/docs-models', 'apps'),
                        ("体验区", f'{web_prefix}/experience', 'directions_run'),
                        ('全量模型', f'{web_prefix}/all-models', 'all_inclusive'),
                        ("运行时", f'{web_prefix}/docs-runtime', 'terminal'),
                        ('性能查看', f'{web_prefix}/docs-performance', 'speed'),
                        ('系统分布式虚拟节点',
                         f'{web_prefix}/docs-distributed_nodes', 'hub'),
                        ('Cursor接入', f'{web_prefix}/docs-cursor', 'mouse')
                    ]

                    for label, path, icon in nav_items:
                        # 检查当前页面路径是否匹配
                        is_active = ui.page.path == path

                        # 根据是否激活设置不同的样式
                        btn_classes = (
                            'px-4 py-2 rounded-lg transition-all duration-300 flex items-center gap-2 ' +
                            (
                                'bg-white text-purple-700 shadow-lg font-medium'
                                if is_active else
                                'hover:bg-white/20 text-white'
                            )
                        )

                        with ui.button(on_click=lambda p=path: ui.navigate.to(p)).classes(btn_classes):
                            ui.icon(icon).classes('text-lg')
                            ui.label(label)

    def readme_page(self):
        with ui.column().classes('w-full max-w-7xl mx-auto p-4 gap-8'):
            ui.markdown(self.read_file(self.home_readme))

    def home_page(self):
        # 主要内容区
        # hero section
        with ui.card().classes('w-full p-8 bg-gradient-to-r from-blue-500 to-purple-600 text-white'):
            ui.label('下一代 AI 计算调度系统').classes('text-4xl font-bold mb-4')
            ui.label('基于 vllm 和 ComfyUI 等Backend的高效 AI 计算调度解决方案').classes(
                'text-xl mb-4')

        # 特性展示
        with ui.grid(columns=7).classes('gap-4'):
            for title, desc, icon in [
                ('高效推理', '利用 vllm 优化推理速度', '⚡'),
                ('智能调度', '自动分配计算资源', '🔄'),
                ('弹性扩展', '动态适应工作负载', '📈'),
                ('API 兼容', '支持 OpenAI API', '🔌'),
                ('多模型支持', '支持多种类型的 AI 模型', '🤖'),
                ('分布式计算', '提供分布式计算能力', '🌐'),
                ('异构算力支持', '支持 GPU、CPU、GCU 等多种算力', '🚀')
            ]:
                with ui.card().classes('p-3'):
                    ui.label(icon).classes('text-4xl mb-2')
                    ui.label(title).classes('text-xl font-bold mb-2')
                    ui.label(desc).classes('text-gray-600')

        # 支持的设备展示
        with ui.card().classes('w-full p-8 bg-white shadow-lg rounded-xl'):
            with ui.row().classes('items-center mb-6'):
                ui.icon('devices').classes('text-3xl text-indigo-600 mr-3')
                ui.label('支持的硬件设备').classes('text-2xl font-bold text-gray-800')

            with ui.grid(columns=6).classes('gap-6'):
                for device_ in all_supported_device:
                    device_name = device_.name.\
                        replace("NV-", "NVIDIA ").\
                        replace("EF-", "Enflame ")

                    # 为不同类型设备选择不同的样式
                    if "NVIDIA" in device_name:
                        bg_color = "from-green-50 to-emerald-50 hover:from-green-100 hover:to-emerald-100"
                        icon_color = "text-green-600"
                    elif "Enflame" in device_name:
                        bg_color = "from-orange-50 to-amber-50 hover:from-orange-100 hover:to-amber-100"
                        icon_color = "text-orange-600"
                    else:  # CPU
                        bg_color = "from-blue-50 to-indigo-50 hover:from-blue-100 hover:to-indigo-100"
                        icon_color = "text-blue-600"

                    with ui.card().classes(f'p-6 bg-gradient-to-br {bg_color} rounded-xl transition-all duration-300 transform hover:scale-105 hover:shadow-xl'):
                        with ui.column().classes('items-center text-center gap-3'):
                            # 设备图标
                            if "NVIDIA" in device_name:
                                ui.icon('memory').classes(
                                    f'text-4xl {icon_color}')
                            elif "Enflame" in device_name:
                                ui.icon('developer_board').classes(
                                    f'text-4xl {icon_color}')
                            else:
                                ui.icon('computer').classes(
                                    f'text-4xl {icon_color}')

                            # 设备名称
                            ui.label(device_name).classes(
                                'text-lg font-bold text-gray-800')

                            if device_name != "CPU":
                                # 分隔线
                                ui.element('div').classes(
                                    'w-16 h-0.5 bg-gray-200 my-2')

                                # 设备规格
                                with ui.column().classes('gap-2 text-gray-600'):
                                    with ui.row().classes('items-center justify-center gap-2'):
                                        ui.icon('memory').classes('text-sm')
                                        ui.label(f"{device_.mem}GB").classes(
                                            'text-sm')

                                    with ui.row().classes('items-center justify-center gap-2'):
                                        ui.icon('speed').classes('text-sm')
                                        ui.label(f"{device_.bandwidth}").classes(
                                            'text-sm')

        # 支持的模型展示
        with ui.card().classes('w-full p-6'):
            with ui.row().classes('items-center mb-6'):
                ui.icon('model_training').classes(
                    'text-3xl text-indigo-600 mr-3')
                ui.label('支持的模型类型').classes('text-2xl font-bold text-gray-800')

            # 创建饼图展示模型分布
            fig = go.Figure(data=[go.Pie(
                labels=['大语言模型', '多模态', '图像生成', 'Embedding',
                        'Rerank', 'TTS/ASR', '视频生成'],
                values=[40, 10, 15, 10, 10, 10, 5],
                hole=.3
            )])
            fig.update_layout(
                showlegend=True,
                margin=dict(t=0, b=0, l=0, r=0),
                height=300
            )
            ui.plotly(fig).classes('w-full')

        # 技术架构
        with ui.card().classes('w-full p-6'):
            with ui.row().classes('items-center mb-6'):
                ui.icon('developer_board').classes(
                    'text-3xl text-indigo-600 mr-3')
                ui.label('技术架构').classes('text-2xl font-bold text-gray-800')
            with ui.row().classes('gap-4 justify-center'):
                for tech in ['VLLM', 'ComfyUI', 'Transformers', 'SD WebUI']:
                    with ui.card().classes('p-4 text-center'):
                        ui.label(tech).classes('font-bold')

        # 性能指标
        with ui.card().classes('w-full p-6'):
            with ui.row().classes('items-center mb-6'):
                ui.icon('speed').classes('text-3xl text-indigo-600 mr-3')
                ui.label('性能指标').classes('text-2xl font-bold text-gray-800')
            # 创建性能对比图
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['推理速度', '资源利用率', '并发处理能力'],
                y=[90, 85, 95],
                name='本系统'
            ))
            fig.add_trace(go.Bar(
                x=['推理速度', '资源利用率', '并发处理能力'],
                y=[60, 55, 65],
                name='传统系统'
            ))
            fig.update_layout(
                barmode='group',
                margin=dict(t=0, b=0, l=0, r=0),
                height=300
            )
            ui.plotly(fig).classes('w-full')

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
                        "name", "model_max_tokens", "description", "model_url"],
                    model_headers_desc=["名称", "最大支持tokens", "描述", "官网"],
                    RequestBaseModel=[
                        ChatCompletionRequest, CompletionRequest]
                )
            with ui.tab_panel('VLLM'):
                self._content_page_(
                    model_config=data.get("VLLM"),
                    model_type="VLLM",
                    model_headers=[
                        "name", "model_max_tokens", "description", "model_url"],
                    model_headers_desc=["名称", "最大支持tokens", "描述", "官网"],
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

        # 模型加载时间卡片
        with ui.card().classes('w-full p-6 mb-6 bg-white rounded-xl shadow-lg'):
            with ui.row().classes('items-center mb-4'):
                ui.icon('timer').classes('text-3xl text-blue-600 mr-2')
                ui.markdown('## 模型加载时间').classes(
                    'text-xl font-bold text-gray-800')

            ui.markdown("""
            > 注意：模型加载时间是指模型从加载到可以开始处理请求的时间。
            """).classes('mb-4 text-gray-600 bg-blue-50 p-4 rounded-lg')

            with ui.element('div').classes('w-full overflow-x-auto'):
                ui.table.from_pandas(
                    start_server_time_df
                ).classes('w-full border-collapse min-w-full')

        # 模型性能卡片
        with ui.card().classes('w-full p-6 bg-white rounded-xl shadow-lg'):
            with ui.row().classes('items-center mb-4'):
                ui.icon('speed').classes('text-3xl text-green-600 mr-2')
                ui.markdown('## 模型性能').classes(
                    'text-xl font-bold text-gray-800')

            with ui.element('div').classes('mb-4 bg-green-50 p-4 rounded-lg'):
                ui.markdown("**性能指标说明：**").classes('text-gray-600')
                ui.markdown("- 大语言模型: 每秒生成token的数量").classes('text-gray-600')
                ui.markdown("- 图像生成: 每张图像生成所需时间").classes('text-gray-600')
                ui.markdown("- 语音识别：每秒处理音频帧的所需时间").classes('text-gray-600')
                ui.markdown("- 语音合成：每秒处理音频帧的所需时间").classes('text-gray-600')

            with ui.element('div').classes('w-full overflow-x-auto'):
                ui.table.from_pandas(
                    tps_spi_df
                ).classes('w-full border-collapse min-w-full')

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

        # 虚拟节点卡片
        with ui.card().classes('w-full p-6 bg-white rounded-xl shadow-lg'):
            # 标题部分
            with ui.row().classes('items-center mb-4'):
                ui.icon('hub').classes('text-3xl text-purple-600 mr-2')
                ui.markdown('## 系统虚拟节点').classes(
                    'text-xl font-bold text-gray-800')

            # 说明部分
            with ui.element('div').classes('mb-6 bg-purple-50 p-4 rounded-lg'):
                ui.markdown(
                    """> **注意：** 真实的物理节点可能被虚拟成多个虚拟节点，每个虚拟节点可以处理一个请求。""").classes('text-gray-600')
                ui.markdown('**节点特性：**').classes('text-gray-600')
                ui.markdown('- 每个虚拟节点独占自己的显卡资源').classes('text-gray-600')
                ui.markdown('- 节点间资源隔离，互不影响').classes('text-gray-600')
                ui.markdown('- 支持动态扩缩容').classes('text-gray-600')
                ui.markdown('- 自动负载均衡').classes('text-gray-600')

            # 节点状态统计
            if not node_df.empty:
                with ui.row().classes('gap-4 mb-6'):
                    with ui.card().classes('flex-1 p-4 bg-green-50 rounded-lg'):
                        ui.label('活跃节点数').classes('text-sm text-gray-600')
                        ui.label(str(len(node_df))).classes(
                            'text-2xl font-bold text-green-600')

                    with ui.card().classes('flex-1 p-4 bg-blue-50 rounded-lg'):
                        ui.label('总GPU数').classes('text-sm text-gray-600')
                        total_gpus = node_df['device-ids'].str.count(',') + 1
                        ui.label(str(total_gpus.sum())).classes(
                            'text-2xl font-bold text-blue-600')

            # 节点列表表格
            with ui.element('div').classes('w-full overflow-x-auto'):
                if node_df.empty:
                    with ui.element('div').classes('text-center py-8 bg-gray-50 rounded-lg'):
                        ui.icon('warning').classes(
                            'text-4xl text-yellow-500 mb-2')
                        ui.label('暂无运行中的节点').classes('text-gray-500')
                else:
                    ui.table.from_pandas(
                        node_df
                    ).classes('w-full border-collapse min-w-full')

    def cursor_view(self):
        cursor_settings = Image.open(os.path.join(
            project_path, "statics", 'cursor-settings.png'))
        cursor_code_generate = Image.open(os.path.join(
            project_path, "statics", 'cursor-code-generate.png'))
        cursor_code_chat = Image.open(os.path.join(
            project_path, "statics", 'cursor-code-chat.png'))

        with ui.column().classes('w-full max-w-7xl mx-auto p-8 space-y-12'):
            # 标题部分
            with ui.card().classes('w-full bg-gradient-to-r from-blue-50 to-indigo-50 p-6'):
                ui.markdown('# Cursor 代码编程助手').classes(
                    'text-3xl font-bold text-gray-800 mb-4')
                ui.markdown(
                    "Cursor 是一个基于大语言模型（LLM）的智能代码生成工具，能显著提升您的开发效率。"
                ).classes('text-lg text-gray-600')

            # 下载部分
            with ui.card().classes('w-full p-6 shadow-lg hover:shadow-xl transition-shadow'):
                ui.markdown("### 🚀 快速开始").classes(
                    'text-xl font-semibold text-gray-800 mb-4')
                with ui.row().classes('items-center gap-4'):
                    ui.link(
                        '下载 Cursor',
                        'https://www.cursor.com/'
                    ).classes('px-6 py-2 bg-blue-600 text-white rounded-full hover:bg-blue-700 transition-colors')
                    ui.markdown(
                        "访问 [Cursor 官网](https://www.cursor.com/) 了解更多").classes('text-gray-600')

            # 设置指南
            with ui.card().classes('w-full'):
                ui.markdown("### ⚙️ 代理设置").classes(
                    'text-xl font-semibold text-gray-800')
                ui.image(cursor_settings).classes(
                    'w-full max-w-2xl rounded-lg shadow-md mx-auto')

                # 代码生成
            with ui.card().classes('w-full p-6 space-y-6'):
                ui.markdown("### 💡 代码生成").classes(
                    'text-xl font-semibold text-gray-800')
                ui.image(cursor_code_generate).classes(
                    'w-full rounded-lg shadow-md')

            # 代码聊天
            with ui.card().classes('w-full p-6 space-y-6'):
                ui.markdown("### 💬 代码聊天").classes(
                    'text-xl font-semibold text-gray-800')
                ui.image(cursor_code_chat).classes(
                    'w-full rounded-lg shadow-md')

    def runtime_view(self):
        with ui.column().classes('w-full max-w-7xl mx-auto p-8 space-y-8'):
            # 状态统计卡片组
            with ui.row().classes('gap-6 w-full'):
                # 处理中的任务
                with ui.card().classes('flex-1 p-6 bg-blue-50 rounded-xl hover:shadow-lg transition-shadow'):
                    with ui.column().classes('space-y-2'):
                        ui.label('处理中的任务').classes('text-lg text-gray-600')
                        with ui.row().classes('items-center gap-2'):
                            ui.icon('pending').classes(
                                'text-2xl text-blue-600')
                            ui.label(str(scheduler.get_request_status_ing_total())).classes(
                                'text-3xl font-bold text-blue-600')

                # 排队中的任务
                with ui.card().classes('flex-1 p-6 bg-orange-50 rounded-xl hover:shadow-lg transition-shadow'):
                    with ui.column().classes('space-y-2'):
                        ui.label('排队中的任务').classes('text-lg text-gray-600')
                        with ui.row().classes('items-center gap-2'):
                            ui.icon('queue').classes(
                                'text-2xl text-orange-600')
                            ui.label(str(scheduler.get_request_queue_all_length())).classes(
                                'text-3xl font-bold text-orange-600')

                with ui.card().classes('flex-1 p-6 bg-green-50 rounded-xl hover:shadow-lg transition-shadow'):
                    with ui.column().classes('space-y-2'):
                        ui.label('已完成的任务').classes('text-lg text-gray-600')
                        with ui.row().classes('items-center gap-2'):
                            ui.icon('beenhere').classes(
                                'text-2xl text-green-600')
                            ui.label(str(scheduler.get_request_done())).classes(
                                'text-3xl font-bold text-orange-600')

            # 运行中的模型列表
            with ui.card().classes('w-full p-6'):
                with ui.row().classes('items-center gap-4 mb-6'):
                    ui.icon('model_training').classes(
                        'text-2xl text-purple-600')
                    ui.markdown('## 运行中的模型').classes(
                        'text-2xl font-bold text-gray-800')

                # 获取并展示模型数据
                data = get_running_models().get("results", [])
                if not data or len(data) == 0:
                    with ui.column().classes('w-full items-center py-12 space-y-4'):
                        ui.icon('error_outline').classes(
                            'text-4xl text-gray-400')
                        ui.label('暂无运行中的模型').classes('text-xl text-gray-400')
                else:
                    model_headers = [
                        "name", "description"]
                    model_headers_desc = ["名称", "描述"]
                    model_list = [[m[i] for i in model_headers]
                                  for m in data]
                    df = pd.DataFrame(
                        data=model_list, columns=model_headers_desc)
                    self._models_card_(df)

    def all_models_views(self):
        # 获取并展示模型数据
        data = system_models_config.list()
        if not data or len(data) == 0:
            with ui.column().classes('w-full items-center py-12 space-y-4'):
                ui.icon('error_outline').classes(
                    'text-4xl text-gray-400')
                ui.label('暂无模型').classes('text-xl text-gray-400')
        else:
            # 添加统计卡片
            with ui.expansion('说明', caption='模型数量统计').classes('w-full'):
                with ui.row().classes('gap-4 mb-6'):
                    with ui.card().classes('w-full bg-gradient-to-r from-blue-50 to-indigo-50 p-6'):
                        ui.markdown('- 这里可以查看系统已经支持了的模型列表').classes(
                            'text-lg text-gray-600')
                        ui.markdown("- 如果需要调用模型，请前往模型广场查看是否支持该模型的调用").classes(
                            'text-lg text-gray-600')
                    with ui.card().classes('flex-1 p-4 bg-purple-50 rounded-xl'):
                        ui.label('全部模型数量').classes(
                            'text-sm text-gray-600 mb-1')
                        ui.label(str(len(data))).classes(
                            'text-2xl font-bold text-purple-600')

            # 原有的模型列表展示
            model_headers = [
                "name", "description", "model_url"]
            model_headers_desc = ["名称", "描述", "官网"]
            online_models_config = get_models_config_list()
            online_models_names = [m.get("name") for m in online_models_config]
            online_models = []
            offline_models = []
            for m in data:
                if m.name not in online_models_names:
                    offline_models.append([m.dict()[i] for i in model_headers])
                else:
                    online_models.append([m.dict()[i] for i in model_headers])

            # 创建 tabs 容器
            with ui.tabs().classes('w-full') as tabs:
                ui.tab('在线模型', icon='cloud_done')
                ui.tab('离线模型', icon='cloud_off')
                ui.tab('离线模型下载', icon='cloud_download')

            # Tab 内容面板
            with ui.tab_panels(tabs, value='在线模型').classes('w-full'):
                # 在线模型 tab
                with ui.tab_panel('在线模型'):
                    if len(online_models) > 0:
                        online_df = pd.DataFrame(
                            data=online_models, columns=model_headers_desc)
                        self._models_card_(online_df, columns=3)
                    else:
                        with ui.column().classes('w-full items-center py-12 space-y-4'):
                            ui.icon('cloud_off').classes(
                                'text-4xl text-gray-400')
                            ui.label('暂无在线模型').classes('text-xl text-gray-400')

                # 离线模型 tab
                with ui.tab_panel('离线模型'):
                    if len(offline_models) > 0:
                        offline_df = pd.DataFrame(
                            data=offline_models, columns=model_headers_desc)
                        self._models_card_(
                            offline_df, columns=3)
                    else:
                        with ui.column().classes('w-full items-center py-12 space-y-4'):
                            ui.icon('cloud_done').classes(
                                'text-4xl text-gray-400')
                            ui.label('暂无离线模型').classes('text-xl text-gray-400')
                # 离线模型下载 tab
                with ui.tab_panel('离线模型下载'):
                    if len(offline_models) > 0:
                        offline_df = pd.DataFrame(
                            data=offline_models, columns=model_headers_desc)
                        model_names = offline_df['名称'].tolist()
                        # 命令输出区域（默认隐藏，点击按钮后显示）

                        with ui.card().classes('w-full p-6 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl shadow-sm mb-6'):
                            with ui.column().classes('gap-4'):
                                # 标题和说明
                                with ui.row().classes('items-center gap-2 mb-2'):
                                    ui.icon('download').classes(
                                        'text-2xl text-blue-600')
                                    ui.label('批量下载模型').classes(
                                        'text-xl font-bold text-gray-800')
                                    download_shell_btn = ui.button('生成下载命令', icon='code').classes(
                                        'px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white ' +
                                        'rounded-lg transition-colors duration-200 flex items-center gap-2 ' +
                                        'whitespace-nowrap'  # 防止文字换行
                                    ).props('flat dense')
                                    clear_models_btn = ui.button(
                                        "清空", icon='delete')
                                    load_all_models_btn = ui.button(
                                        "全选", icon='check')
                                ui.label('选择需要下载的模型，系统将自动生成下载命令').classes(
                                    'text-sm text-gray-600 mb-4')

                                # 选择和按钮区域
                                with ui.row().classes('items-end gap-4 w-full'):  # 添加 w-full 确保行占满宽度
                                    with ui.column().classes('w-3/4 flex-grow'):  # 添加 flex-grow 让选择框占据所有可用空间
                                        selected_models = ui.select(
                                            model_names,
                                            multiple=True,
                                            value=[],
                                            label='选择模型'
                                        ).classes('w-full min-w-[500px]').props('use-chips outlined dense fill-width')  # 添加最小宽度和fill-width
                                with ui.row().classes('items-end gap-4 w-full'):
                                    download_shell_component = ui.code(
                                        '#', language="shell").style(
                                            'white-space: pre-wrap !important; '  # 强制换行
                                            'word-wrap: break-word !important; '  # 允许在单词内换行
                                            'max-width: 100% !important; '        # 限制最大宽度
                                            'overflow-x: hidden !important; '     # 隐藏横向滚动条
                                    ).classes('w-full')

                        async def generate_download_shell():
                            download_shell_component.set_content(
                                system_models_config.generate_download_shell(selected_models.value))

                        download_shell_btn.on_click(generate_download_shell)
                        clear_models_btn.on_click(
                            lambda: selected_models.set_value([]))
                        load_all_models_btn.on_click(
                            lambda: selected_models.set_value(model_names))

                    else:
                        with ui.column().classes('w-full items-center py-12 space-y-4'):
                            ui.icon('cloud_done').classes(
                                'text-4xl text-gray-400')
                            ui.label('暂无离线模型').classes('text-xl text-gray-400')

    def experience_view(self):
        online_models_config = get_models_config_list()
        
        running_models = [m.get("name") for m in get_running_models().get("results", [])]
        online_models_map = {}
        for m in online_models_config:
            if m.get("name") not in running_models:
                online_models_map[m.get("name")] = m
            else:
                online_models_map[m.get("name")+" (running)"] = m
        model_names = list(online_models_map.keys())
        with ui.row().classes('items-end gap-4 w-full'):  # 添加 w-full 确保行占满宽度
            with ui.column().classes('w-3/4 flex-grow'):  # 添加 flex-grow 让选择框占据所有可用空间
                selected_models = ui.select(
                    model_names,
                    with_input=True,
                    value=[],
                    label='选择模型'
                ).classes('w-full min-w-[500px]').props('use-chips outlined dense fill-width')
                exp_zone = ui.card().classes('w-full min-w-[500px] p-4')

                def selected_models_on_value_change(e):
                    name_ = selected_models.value.replace(" (running)","")
                    with exp_zone:
                        if selected_models.value in online_models_map:
                            if "LLM" in online_models_map[selected_models.value].get("api_type"):
                                
                                ExperienceZone().create_llm_chat(model_name=name_)
                            else:
                                ui.label('该模型暂不支持体验').classes('text-red-500')
                        else:
                            exp_zone = ui.label('该模型暂不支持体验').classes('text-red-500')

                selected_models.on_value_change(
                    selected_models_on_value_change)


layout = UILayout()


class UIWeb:

    @ui.page('/')
    @staticmethod
    def index():
        layout.header()
        with ui.column().classes('w-full max-w-7xl mx-auto p-4 gap-8'):
            layout.home_page()

    @ui.page(f'{web_prefix}/docs-README')
    @staticmethod
    def readme():
        layout.header()
        with ui.column().classes('w-full max-w-7xl mx-auto p-4 gap-8'):
            layout.readme_page()

    @ui.page(f'{web_prefix}/docs-models')
    @staticmethod
    def models():
        layout.header()
        with ui.column().classes('w-full max-w-7xl mx-auto p-4 gap-8'):
            layout.model_plaza()

    @ui.page(f'{web_prefix}/experience')
    @staticmethod
    def experience():
        layout.header()
        with ui.column().classes('w-full max-w-7xl mx-auto p-4 gap-8'):
            layout.experience_view()

    @ui.page(f'{web_prefix}/all-models')
    @staticmethod
    def all_models():
        layout.header()
        with ui.column().classes('w-full max-w-7xl mx-auto p-4 gap-8'):
            layout.all_models_views()

    @ui.page(f'{web_prefix}/docs-runtime')
    @staticmethod
    def runtime():
        layout.header()
        with ui.column().classes('w-full max-w-7xl mx-auto p-4 gap-8'):
            layout.runtime_view()

    @ui.page(f'{web_prefix}/docs-performance')
    @staticmethod
    def performance():
        layout.header()
        with ui.column().classes('w-full max-w-7xl mx-auto p-4 gap-8'):
            layout.performance_view()

    @ui.page(f'{web_prefix}/docs-distributed_nodes')
    @staticmethod
    def distributed_nodes():
        layout.header()
        with ui.column().classes('w-full max-w-7xl mx-auto p-4 gap-8'):
            layout.distributed_nodes()

    @ui.page(f'{web_prefix}/docs-cursor')
    @staticmethod
    def cursor():
        layout.header()
        with ui.column().classes('w-full max-w-7xl mx-auto p-4 gap-8'):
            layout.cursor_view()

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
