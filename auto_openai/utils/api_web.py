from fastapi import FastAPI, Request, Body, Header, Query
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
import os


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


class APIDocsApp():
    def __init__(self, title="Demo"):
        self.title = title
        self.custom_css = """
        .navbar {
            background: linear-gradient(90deg, #1a1a1a, #4a4a4a);
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .content-container {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 6px 0 rgba(31, 38, 135, 0.37);
        }
        
        .title {
            background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5em;
            text-align: center;
            margin-bottom: 20px;
            font-weight: bold;
        }
        
        .gradio-dropdown {
            background: transparent !important;
            border: 2px solid #ffffff3d !important;
            color: white !important;
            font-size: 1.2em !important;
            transition: all 0.3s ease;
        }
        
        .gradio-dropdown:hover {
            border-color: #fff !important;
            box-shadow: 0 0 10px rgba(255,255,255,0.2);
        }
        """

    @property
    def config(self):
        return {}

    def layout(self):
        with gr.Blocks(css=self.custom_css, theme=gr.themes.Soft()) as demo:
            with gr.Column(elem_classes="navbar", scale=1):
                gr.Markdown(self.title, elem_classes="title")
                compatibility = gr.Dropdown(
                    choices=list(self.config.keys()),
                    value="首页",
                    label="",
                    container=False
                )

            @gr.render(inputs=compatibility)
            def compatibility_change(compatibility):
                self.config.get(compatibility)()

        return demo

    @property
    def app(self):
        return self.layout()

    def run(self, server_name="0.0.0.0", port=50001, auth=None):
        self.app.launch(
            server_name=server_name,
            server_port=port,
            auth=auth
        )

    def read_file(self, file):
        if os.path.exists(file):
            with open(file, "r") as f:
                return f.read()
        else:
            return "# 努力开发中..."


class DemoWebApp(APIDocsApp):
    home_readme = os.path.join(project_path, "README.md")
    demo_path = os.path.join(project_path, "web/tests")

    def get_client(self):
        port = int(os.environ.get("MAINPORT"))
        OPENAI_BASE_URL = f"http://127.0.0.1:{port}/openai/v1"
        return OpenAI(base_url=OPENAI_BASE_URL, api_key="EMPTY")

    def LLM_playgournd(self, model_list):
        self.client = self.get_client()
        model_params = {}
        for m_ in model_list:
            model_params[m_[0]] = m_[1]
        with gr.Accordion(label="⚙️ 点我设置", open=False):
            with gr.Row():
                model = gr.Dropdown(
                    choices=model_params.keys(),
                    label="Model",
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    step=0.1,
                    value=0.0,
                    label="Temperature",
                )

        chatbot = gr.Chatbot(
            elem_id="chatbot", bubble_full_width=False, type="messages")

        chat_input = gr.MultimodalTextbox(
            interactive=True,
            file_count="multiple",
            placeholder="Enter message or upload file...",
            show_label=False,
            file_types=[".txt", ".md", ".MD"]
        )

        def add_message(history, message):
            for x in message["files"]:
                # history.append({"role": "system", "content": {"path": x}})
                with open(x, "r") as f:
                    history.append({"role": "system", "content": f.read()})
            if message["text"] is not None:
                history.append({"role": "user", "content": message["text"]})
            return history, gr.MultimodalTextbox(value=None, interactive=False)

        def bot(history: list, model, temperature):
            messages = [
                {"role": "system", "content": "you are a helpful assistant"}]
            history_ = []
            if len(history) >= 10:
                history_ = history[-10:]
            else:
                history_ = history
            for msg in history_:
                messages.append(
                    {"role": msg["role"], "content": msg["content"]})
            history.append({"role": "assistant", "content": ""})
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=model_params[model],
                temperature=float(temperature),
                stream=True,
                presence_penalty=2.0,
                frequency_penalty=2.0
            )

            for chunk in response:
                chunk_content = chunk.choices[0].delta.content
                chunk_content = chunk_content if chunk_content else ""
                history[-1]["content"] += chunk_content
                yield history

        chat_msg = chat_input.submit(
            add_message, [chatbot, chat_input], [chatbot, chat_input]
        )
        bot_msg = chat_msg.then(
            bot, [chatbot, model, temperature], chatbot, api_name="bot_response")
        bot_msg.then(lambda: gr.MultimodalTextbox(
            interactive=True), None, [chat_input])

    def _content_page_(self, model_config, model_type,
                       model_headers=["name", "description"],
                       model_headers_desc=["名称", "描述"],
                       RequestBaseModel=[]):
        with gr.Column(elem_classes="content-container"):
            if not model_config:
                gr.Markdown("# 努力开发中...")
                return
            model_list = []
            for m in model_config:
                h_ = []
                for i in model_headers:
                    h_.append(m[i])
                model_list.append(h_)

            with gr.Tabs():
                with gr.Tab("支持的模型列表"):
                    gr.DataFrame(
                        model_list,
                        headers=model_headers_desc
                    )
                if RequestBaseModel is not None:
                    with gr.Tab("文档参数说明"):
                        for r_basemodel in RequestBaseModel:
                            gr.Markdown(f"# {r_basemodel.__name__}")
                            data = generate_api_documentation(
                                r_basemodel.model_json_schema())
                            gr.DataFrame(data)
                with gr.Tab("python 示例"):
                    py_path = os.path.join(self.demo_path, f"{model_type}.py")
                    gr.Code(self.read_file(py_path),
                            language="python")
                with gr.Tab("curl 示例"):
                    curl_path = os.path.join(
                        self.demo_path, f"{model_type}.sh")
                    gr.Code(self.read_file(curl_path), language="shell")
                # with gr.Tab("PlayGround"):
                #     if model_type == "LLM":
                #         self.LLM_playgournd(model_list=model_list)
                #     else:
                #         gr.Markdown("# 努力开发中...")

    def Home_page(self):
        with gr.Column(elem_classes="content-container"):
            gr.Markdown(self.read_file(self.home_readme))

    def None_page(self):
        with gr.Column(elem_classes="content-container"):
            gr.Markdown("# 努力开发中...")

    def Models_pages(self):
        data = global_config.get_MODELS_MAPS()
        with gr.Column():
            with gr.Tabs():
                with gr.Tab("LLM"):
                    self._content_page_(
                        model_config=data.get("LLM"),
                        model_type="LLM",
                        model_headers=[
                            "name", "model_max_tokens", "description"],
                        model_headers_desc=["名称", "最大支持tokens", "描述"],
                        RequestBaseModel=[
                            ChatCompletionRequest, CompletionRequest]
                    )
                with gr.Tab("VLLM"):
                    self._content_page_(
                        model_config=data.get("VLLM"),
                        model_type="VLLM",
                        model_headers=[
                            "name", "model_max_tokens", "description"],
                        model_headers_desc=["名称", "最大支持tokens", "描述"],
                        RequestBaseModel=[ChatCompletionRequest]
                    )
                with gr.Tab("SD15MultiControlnetGenerateImage"):
                    self._content_page_(
                        model_config=data.get(
                            "SD15MultiControlnetGenerateImage"),
                        model_type="SD15MultiControlnetGenerateImage",
                        RequestBaseModel=[
                            SD15MultiControlnetGenerateImageRequest, SD15ControlnetUnit]
                    )
                with gr.Tab("SolutionBaseGenerateImage"):
                    self._content_page_(
                        model_config=data.get("SolutionBaseGenerateImage"),
                        model_type="SolutionBaseGenerateImage",
                        RequestBaseModel=[SolutionBaseGenerateImageRequest]
                    )
                with gr.Tab("Embedding"):
                    self._content_page_(
                        model_config=data.get("Embedding"),
                        model_type="Embedding",
                        RequestBaseModel=[EmbeddingsRequest]
                    )
                with gr.Tab("Rerank"):
                    self._content_page_(
                        model_config=data.get("Rerank"),
                        model_type="Rerank",
                        RequestBaseModel=[RerankRequest]
                    )
                with gr.Tab("TTS"):
                    self._content_page_(
                        model_config=data.get("TTS"),
                        model_type="TTS",
                        RequestBaseModel=[AudioSpeechRequest]
                    )
                with gr.Tab("ASR"):
                    self._content_page_(
                        model_config=data.get("ASR"),
                        model_type="ASR",
                        RequestBaseModel=[AudioTranscriptionsRequest]
                    )
                with gr.Tab("视频生成"):
                    self._content_page_(
                        model_config=data.get("Video"),
                        model_type="Video",
                        RequestBaseModel=[VideoGenerationsRequest]
                    )

    def Performance_pages(self):
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
        with gr.Column(elem_classes="content-container"):
            fresh = gr.Button("刷新", elem_id="refresh-button",
                              variant="primary")
            gr.Markdown("""
            ## 模型加载时间
            > 注意：模型加载时间是指模型从加载到可以开始处理请求的时间。""")
            df1 = gr.DataFrame(start_server_time_df, label="模型加载时间")
            gr.Markdown("""
            ## 模型性能
            - 模型性能是指模型在处理请求时的性能指标。
            - 大语言模型: 使用TPS衡量，每秒生成token的数量。
            - 图像生成: 使用SPI衡量，每张图像生成所需时间。
            - 语音识别：使用SPS衡量，每秒处理音频帧的所需时间。
            - 语音合成：使用SPC衡量，每秒处理音频帧的所需时间。
            """)
            df2 = gr.DataFrame(tps_spi_df, label="模型性能")
            fresh.click(fn=convert_to_dataframe, inputs=[], outputs=[df1, df2])

    def vnode_pages(self):
        def convert_to_dataframe():
            scheduler = Scheduler(redis_client=redis_client, http_request=None)
            data = scheduler.get_running_node()
            if data:
                return pd.DataFrame(data)
            else:
                return pd.DataFrame([])
        node_df = convert_to_dataframe()
        with gr.Column(elem_classes="content-container"):
            fresh = gr.Button("刷新", elem_id="refresh-button",
                              variant="primary")
            gr.Markdown("""
            ## 虚拟节点
            > 注意：真实的物理节点可能被虚拟成多个虚拟节点，每个虚拟节点可以处理一个请求。
            - 每个虚拟节点独占自己的卡，不会被其他卡获取""")
            df = gr.DataFrame(node_df, label="节点信息")
            fresh.click(fn=convert_to_dataframe, inputs=[], outputs=[df])

    @property
    def config(self):
        return {
            "首页": self.Home_page,
            "模型广场": self.Models_pages,
            "性能查看": self.Performance_pages,
            "系统分布式虚拟节点": self.vnode_pages
        }


if __name__ == "__main__":
    DemoWebApp(title="Openai-本地大模型API文档").run()
