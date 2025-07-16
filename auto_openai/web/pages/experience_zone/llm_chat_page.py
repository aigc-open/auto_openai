import os
from openai import AsyncOpenAI
from nicegui import ui
from . import base_url, api_key
client = AsyncOpenAI(base_url=base_url, api_key=api_key)


# async with httpx.AsyncClient(base_url="http://127.0.0.1:8000/") as client:
#     response = await client.get("/")
#     response.raise_for_status()
#     return response.json()


def index(model_name):
    import asyncio
    
    if "AI-Simulation-Large-Model" in model_name:
        prompt_value = "AI仿真大模型有什么用处呢?"
    elif "Oil-Large-Model" in model_name:
        prompt_value = "石油大模型有什么用处呢?"
    else:
        prompt_value = "请你介绍一下人工智能带来的好处"

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
                with ui.row():
                    temperature = ui.number(label="Temperature(代码生成建议使用0.0)", value=0.0, min=0.0, max=1.0).props('filled outlined').classes('w-[200px]')
                    max_tokens = ui.number(label="Max Tokens", value=512, min=0, max=4096).props('filled outlined').classes('w-[200px]')
                with ui.row().classes('w-full gap-4 items-end'):
                    with ui.column().classes('flex-grow'):
                        prompt = ui.input(
                            value=prompt_value,
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
            response = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt.value}],
                stream=True,
                temperature=temperature.value,
                max_tokens=max_tokens.value
            )

            # 流式处理响应
            full_response = ""
            async for chunk in response:
                if chunk.choices[0].delta.reasoning_content:
                    content = chunk.choices[0].delta.reasoning_content
                    full_response += content
                    chat_messages.set_content(full_response + "\n```")
                    await asyncio.sleep(0.01)
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
