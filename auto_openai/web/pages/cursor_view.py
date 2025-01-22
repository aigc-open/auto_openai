import os
from auto_openai import project_path
from nicegui import ui
from PIL import Image


def index():
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
