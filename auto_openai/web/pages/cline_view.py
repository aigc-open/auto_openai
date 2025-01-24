import os
from auto_openai import project_path
from nicegui import ui
from PIL import Image


def index():
    cline_settings = Image.open(os.path.join(
        project_path, "statics", 'cline-settings.png'))
    cline_write = Image.open(os.path.join(
        project_path, "statics", 'cline-write.png'))
    cline_update = Image.open(os.path.join(
        project_path, "statics", 'cline-update.png'))

    with ui.column().classes('w-full max-w-7xl mx-auto p-8 space-y-12'):
        # 标题部分
        with ui.card().classes('w-full bg-gradient-to-r from-blue-50 to-indigo-50 p-6'):
            ui.markdown('# Cline 代码编程助手').classes(
                'text-3xl font-bold text-gray-800 mb-4')
            ui.markdown(
                "Cline 是一个基于大语言模型（LLM）的智能代码生成工具，能显著提升您的开发效率。"
            ).classes('text-lg text-gray-600')
            ui.markdown(
                "代码安全 (开源产品，完全可以本地部署)"
            ).classes('text-lg text-gray-600')
            ui.markdown(
                "Tool Calls 代码生成完全自动化"
            ).classes('text-lg text-gray-600')

        # 下载部分
        with ui.card().classes('w-full p-6 shadow-lg hover:shadow-xl transition-shadow'):
            ui.markdown("### 🚀 快速开始").classes(
                'text-xl font-semibold text-gray-800 mb-4')
            with ui.row().classes('items-center gap-4'):
                ui.link(
                    '下载 cline',
                    'https://github.com/cline/cline'
                ).classes('px-6 py-2 bg-blue-600 text-white rounded-full hover:bg-blue-700 transition-colors')
                ui.markdown(
                    "访问 [cline 插件安装](https://github.com/cline/cline) 了解更多").classes('text-gray-600')

        # 设置指南
        with ui.card().classes('w-full'):
            ui.markdown("### ⚙️ 代理设置").classes(
                'text-xl font-semibold text-gray-800')
            ui.image(cline_settings).classes(
                'w-full max-w-2xl rounded-lg shadow-md mx-auto')

            # 代码生成
        with ui.card().classes('w-full p-6 space-y-6'):
            ui.markdown("### 💡 代码自动生成").classes(
                'text-xl font-semibold text-gray-800')
            ui.image(cline_write).classes(
                'w-full rounded-lg shadow-md')

        # 代码聊天
        with ui.card().classes('w-full p-6 space-y-6'):
            ui.markdown("### 💬 代码自动修改").classes(
                'text-xl font-semibold text-gray-800')
            ui.image(cline_update).classes(
                'w-full rounded-lg shadow-md')
