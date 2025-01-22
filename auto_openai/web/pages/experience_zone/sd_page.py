import os
from openai import AsyncOpenAI
from nicegui import ui
from . import base_url, api_key

client = AsyncOpenAI(base_url=base_url, api_key=api_key)


def index(model_name):

    async def generate_image(model_name: str, prompt: str, width: int, height: int, steps: int,
                             seed: int, denoise_strength: float,
                             result_image: ui.image):
        try:
            response = await client.images.generate(
                model=model_name,
                prompt="",
                extra_body={
                    "prompt": prompt,
                    "batch_size": 1,
                    "seed": seed,
                    "width": width,
                    "height": height,
                    "steps": steps,
                    "denoise_strength": denoise_strength
                }
            )
            # 假设响应中包含图片URL或base64数据
            if response.data:
                result_image.source = response.data[0].url  # 根据实际响应格式调整
            else:
                ui.notify('生成失败: 无图片数据', type='negative')
        except Exception as e:
            ui.notify(f'生成失败: {str(e)}', type='negative')

    with ui.card().classes('w-full max-w-7xl mx-auto p-6 shadow-lg rounded-xl') as card:
        ui.markdown('## 图像生成')
        spinner = ui.spinner(size='sm').classes('text-blue-600')
        spinner.set_visibility(False)
        btn = ui.button('点击生成').classes('mt-4')

        # 创建输入控件
        prompt = ui.input(
            '提示词描述',
            value="beautiful scenery nature glass bottle landscape, purple galaxy bottle",
            placeholder='请输入详细的图片描述...').classes('w-full')

        with ui.row().classes('w-full max-w-7xl mx-auto p-6 shadow-lg rounded-xl'):
            width = ui.number('宽度', value=1024, min=64, max=2048, step=64)
            height = ui.number('高度', value=512, min=64, max=2048, step=64)
            steps = ui.number('步数', value=20, min=10, max=50)
            seed = ui.number('随机种子', value=1234, min=0)
            denoise_strength = ui.number(
                '去噪强度', value=0.7, min=0, max=1, step=0.1)

        # 创建图片显示区域
        result_image = ui.image().classes('w-full max-w-7xl')

        # 生成按钮

    async def on_click(e):
        spinner.set_visibility(True)
        btn.set_visibility(False)
        await generate_image(model_name, prompt.value, int(width.value), int(height.value), int(steps.value),
                             int(seed.value), float(denoise_strength.value), result_image)
        spinner.set_visibility(False)
        btn.set_visibility(True)

    btn.on_click(on_click)

    return card
