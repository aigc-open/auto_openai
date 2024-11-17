import gradio as gr
from .WebStyle import WebStyle
from .Navigation import Navigation


class BaseWebApp:
    def __init__(self, theme="default"):
        # 使用样式类
        self.web_style = WebStyle(theme)
        self.navigation: Navigation = None
        self.set_navigation()

    def get_style(self):
        # 合并基础样式和导航栏样式
        return self.web_style.get_style() + self.navigation.get_style()

    def create_layout(self):
        with gr.Blocks(css=self.get_style()) as demo:
            with gr.Column(elem_classes=["main-container"]):
                self.navigation.create()
                with gr.Column(elem_classes=["content-container"]):
                    with gr.Column(elem_classes=["content-box"]):
                        # 显示初始页面
                        content = self.navigation.pages["首页"]()
                self.navigation.on_click(content)
        return demo

    def run(self, server_name="0.0.0.0"):
        self.create_layout().launch(server_name=server_name)

    @property
    def app(self):
        return self.create_layout()


class SimpleWebApp(BaseWebApp):

    def set_navigation(self):
        self.navigation = Navigation(
            title="LM-API",
            pages={
                "首页": self.home_page,
            }
        )

    def content_area(self):
        with gr.Column() as column:
            gr.Textbox(
                value="欢迎来到首页",
                label="内容区域",
                lines=4,
                show_label=False,
                elem_classes=["content"]
            )
        return column

    def home_page(self):
        with gr.Column() as column:
            gr.Textbox(
                value="欢迎来到首页",
                label="首页",
                show_label=False
            )
        return column
