import gradio as gr


class Navigation:
    def __init__(self, title, pages):
        """
        初始化导航类
        @param title: 导航栏标题
        @param pages: 页面配置字典，格式为 {"页面名": 页面函数}
        """
        self.title = title
        self.pages = pages
        self.nav_buttons = []

    def get_style(self):
        """获取导航栏样式"""
        return """
            /* 导航栏样式 */
            .nav-row { 
                background: linear-gradient(90deg, #2193b0, #6dd5ed);
                padding: 8px 20px;
                margin: 0 !important;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                display: flex;
                align-items: center;
                justify-content: space-between;
                border-radius: 0 !important;
            }
            .title {
                color: white;
                font-size: 24px;
                font-weight: bold;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
                padding: 3px 0;
                flex-grow: 0;
            }
            .nav-buttons {
                display: flex;
                gap: 6px;
                margin-left: auto;
            }
            .nav-btn {
                background: rgba(255,255,255,0.2) !important;
                border: none !important;
                transition: all 0.3s ease !important;
                font-size: 13px !important;
                min-width: unset !important;
                width: auto !important;
                height: 30px !important;
                padding: 0 12px !important;
                white-space: nowrap !important;
                color: #FFFFFF !important; /* 白色字体 */
                background: linear-gradient(45deg, #ff4081, #00FF7F) !important; /* 渐变色：青草色到春天绿色 */
            }
            .nav-btn:hover {
                transform: translateY(-2px) !important;
                background: rgba(255,255,255,0.3) !important;
            }
            .nav-btn.selected {
                background: rgba(255,255,255,0.4) !important;
                transform: translateY(-1px) !important;
                box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
                border: 1px solid white !important;
            }
        """

    def create(self):
        """
        创建导航栏并绑定事件
        @param output_component: 输出组件，用于显示页面内容
        """
        with gr.Row(elem_classes=["nav-row"]) as nav:
            # 添加标题
            gr.HTML(f'<div class="title">{self.title}</div>')
            # 创建导航按钮
            with gr.Row(elem_classes=["nav-buttons"]):

                for page in self.pages.keys():
                    btn = gr.Button(
                        page,
                        elem_classes=["nav-btn"],
                        variant="primary",
                        size="sm"
                    )

                    self.nav_buttons.append(btn)

    def on_click(self, output_component):
        """
        修改点击事件处理
        """

        # 为每个按钮绑定点击事件
        for btn in self.nav_buttons:
            btn.click(
                fn=self.navigate,
                inputs=btn,
                outputs=output_component,
                api_name=False
            )

    def navigate(self, choice):
        """
        导航处理函数
        @param choice: 选中的导航选项
        """
        if choice in self.pages:
            return self.pages[choice]()
        return "请选择一个导航选项"
