from fastapi import FastAPI, Request, Body, Header, Query
import os
import gradio as gr
from auto_openai.utils.init_env import global_config
from auto_openai import project_path
import pandas as pd
from typing import Dict, Any
from auto_openai.utils.openai import ChatCompletionRequest, CompletionRequest,  AudioSpeechRequest, \
    EmbeddingsRequest, RerankRequest, AudioTranscriptionsRequest, BaseGenerateImageRequest


def generate_api_documentation(schema: Dict[str, Any]):
    # 创建一个空列表来存储每个属性的信息
    data = []

    # 处理属性
    properties = schema.get('properties', {})
    for prop_name, prop_details in properties.items():
        title = prop_details.get('title', prop_name)
        default = prop_details.get('default', '无')

        # 处理类型
        if 'anyOf' in prop_details:
            types = [option.get('type', '未知类型')
                     for option in prop_details['anyOf']]
            if 'null' in types:
                types.remove('null')
            prop_type = '/'.join(types)
        else:
            prop_type = prop_details.get('type', '未知类型')

        # 处理枚举类型
        enum_str = ""
        if '$ref' in prop_details:
            ref = prop_details['$ref']
            # 动态查找枚举值
            enum_values = schema['$defs'].get(
                ref.split('/')[-1], {}).get('enum', [])
            if enum_values:
                enum_str = f" (可选值: {', '.join(enum_values)})"
                prop_type = "string"  # 假设枚举类型为字符串
        elif 'allOf' in prop_details:
            for item in prop_details['allOf']:
                if '$ref' in item:
                    ref = item['$ref']
                    # 动态查找枚举值
                    enum_values = schema['$defs'].get(
                        ref.split('/')[-1], {}).get('enum', [])
                    if enum_values:
                        enum_str = f" (可选值: {', '.join(enum_values)})"
                        prop_type = "string"  # 假设枚举类型为字符串

        # 将属性信息添加到列表中
        data.append({
            '名称': prop_name,
            '类型': prop_type,
            '默认值/参考': f'{default}',
            '描述': enum_str
        })

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
        with gr.Blocks(css=self.custom_css) as demo:

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

    def _content_page_(self, model_config, model_type,
                       model_headers=["name", "description"],
                       model_headers_desc=["名称", "描述"],
                       RequestBaseModel=[]):
        with gr.Column(elem_classes="content-container"):
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
                        model_config=data["LLM"],
                        model_type="LLM",
                        model_headers=[
                            "name", "model_max_tokens", "description"],
                        model_headers_desc=["名称", "最大支持tokens", "描述"],
                        RequestBaseModel=[
                            ChatCompletionRequest, CompletionRequest]
                    )
                with gr.Tab("VLLM"):
                    self._content_page_(
                        model_config=data["VLLM"],
                        model_type="VLLM",
                        model_headers=[
                            "name", "model_max_tokens", "description"],
                        model_headers_desc=["名称", "最大支持tokens", "描述"],
                        RequestBaseModel=[ChatCompletionRequest]
                    )
                with gr.Tab("BaseGenerateImage"):
                    self._content_page_(
                        model_config=data["BaseGenerateImage"],
                        model_type="BaseGenerateImage",
                        RequestBaseModel=[BaseGenerateImageRequest]
                    )
                with gr.Tab("Embedding"):
                    self._content_page_(
                        model_config=data["Embedding"],
                        model_type="Embedding",
                        RequestBaseModel=[EmbeddingsRequest]
                    )
                with gr.Tab("Rerank"):
                    self._content_page_(
                        model_config=data["Rerank"],
                        model_type="Rerank",
                        RequestBaseModel=[RerankRequest]
                    )
                with gr.Tab("TTS"):
                    self._content_page_(
                        model_config=data["TTS"],
                        model_type="TTS",
                        RequestBaseModel=[AudioSpeechRequest]
                    )
                with gr.Tab("ASR"):
                    self._content_page_(
                        model_config=data["ASR"],
                        model_type="ASR",
                        RequestBaseModel=[AudioTranscriptionsRequest]
                    )

    @property
    def config(self):
        return {
            "首页": self.Home_page,
            "模型广场": self.Models_pages,
        }


if __name__ == "__main__":
    DemoWebApp(title="Openai-本地大模型API文档").run()
