import gradio as gr


class WebStyle:
    def __init__(self, theme="default"):
        self.theme = theme
        self.styles = {
            "default": self._default_style(),
            "dark": self._dark_style(),
            "light": self._light_style()
        }

    def get_style(self):
        """获取当前主题的样式"""
        return self.styles.get(self.theme, self.styles["default"])

    def _default_style(self):
        """默认主题样式 - 基础样式"""
        return """
            /* 重置浏览器默认边距 */
            body {
                margin: 0 !important;
                padding: 0 !important;
                height: 100vh !important;
            }
            /* gradio 容器样式重置 */
            .gradio-container {
                margin: 0 !important;
                padding: 0 !important;
                width: 100% !important;
                max-width: 100% !important;
                min-height: 100vh !important;
            }
            /* 主块级容器 */
            .main-container {
                display: flex !important;
                flex-direction: column !important;
                min-height: 100vh !important;
                background: #f5f7fa !important;
            }
            /* 内容区域样式 */
            .content-container {
                flex: 1 !important;
                padding: 20px !important;
                box-sizing: border-box !important;
            }
            .content-box {
                background: white;
                height: calc(100vh - 120px) !important;
                border-radius: 15px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                padding: 20px;
                overflow-y: auto;
            }
            /* 文本框样式 */
            .content textarea {
                border: none !important;
                background: transparent !important;
                min-height: 200px !important;
            }

            .markdown-content{
                height: calc(100vh) !important;
                overflow-y: auto;
            }
        """

    def _dark_style(self):
        """深色主题样式"""
        return """
            body {
                margin: 0 !important;
                padding: 0 !important;
                height: 100vh !important;
            }
            .gradio-container {
                margin: 0 !important;
                padding: 0 !important;
                width: 100% !important;
                max-width: 100% !important;
                min-height: 100vh !important;
            }
            .main-container {
                display: flex !important;
                flex-direction: column !important;
                min-height: 100vh !important;
                background: #1a1a1a !important;
            }
            /* ... dark theme specific styles ... */
        """

    def _light_style(self):
        """浅色主题样式"""
        return """
            /* ... light theme specific styles ... */
        """
