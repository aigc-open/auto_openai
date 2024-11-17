from fastapi import FastAPI, Request, Body, Header, Query
import os
import gradio as gr
from auto_openai.utils.init_env import global_config
from auto_openai import project_path


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
    docs_path = os.path.join(project_path, "docs")
    demo_path = os.path.join(project_path, "tests")

    def _content_page_(self, model_config, model_type,
                       model_headers=["name", "description"],
                       model_headers_desc=["名称", "描述"]):
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
                with gr.Tab("文档说明"):
                    md_path = os.path.join(self.docs_path, f"{model_type}.MD")
                    gr.Markdown(self.read_file(md_path))
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

    def 大语言模型_page(self):
        return self._content_page_(
            model_config=global_config.LLM_MODELS,
            model_type="大语言模型",
            model_headers=["name", "model_max_tokens", "description"],
            model_headers_desc=["名称", "最大支持tokens", "描述"]
        )

    def 向量Embeddings_page(self):
        return self._content_page_(
            model_config=global_config.EMBEDDING_MODELS,
            model_type="向量Embeddings",
            model_headers=["name", "description"],
            model_headers_desc=["名称", "描述"]
        )

    def 语音生成_page(self):
        return self._content_page_(
            model_config=global_config.TTS_MODELS,
            model_type="语音生成",
            model_headers=["name", "description"],
            model_headers_desc=["名称", "描述"]
        )

    def 语音识别_page(self):
        return self._content_page_(
            model_config=global_config.ASR_MODELS,
            model_type="语音识别",
            model_headers=["name", "description"],
            model_headers_desc=["名称", "描述"]
        )

    def 绘图_page(self):
        with gr.Tabs():
            for type_ in global_config.SD_MODELS:
                with gr.Tab(type_["name"]):
                    self._content_page_(
                        model_config=type_["model_list"],
                        model_type=type_["name"],
                        model_headers=["name"],
                        model_headers_desc=["名称"]
                    )
        return

    def 多模态_page(self):
        return self._content_page_(
            model_config=global_config.VISION_MODELS,
            model_type="多模态",
            model_headers=["name", "model_max_tokens", "description"],
            model_headers_desc=["名称", "最大支持tokens", "描述"]
        )

    @property
    def config(self):
        return {
            "首页": self.Home_page,
            "大语言模型": self.大语言模型_page,
            "多模态": self.多模态_page,
            "绘图": self.绘图_page,
            "向量Embeddings": self.向量Embeddings_page,
            "语音生成": self.语音生成_page,
            "语音识别": self.语音识别_page,

        }


if __name__ == "__main__":
    DemoWebApp(title="Openai-本地大模型API文档").run()
