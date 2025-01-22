import pandas as pd
from auto_openai.utils.depends import get_running_models, get_models_config_list
from auto_openai.utils.public import scheduler
from nicegui import ui
from .components import ModelCardComponent


def index():
    with ui.column().classes('w-full max-w-7xl mx-auto p-8 space-y-8'):
        # 状态统计卡片组
        with ui.row().classes('gap-6 w-full'):
            # 处理中的任务
            with ui.card().classes('flex-1 p-6 bg-blue-50 rounded-xl hover:shadow-lg transition-shadow'):
                with ui.column().classes('space-y-2'):
                    ui.label('处理中的任务').classes('text-lg text-gray-600')
                    with ui.row().classes('items-center gap-2'):
                        ui.icon('pending').classes(
                            'text-2xl text-blue-600')
                        ui.label(str(scheduler.get_request_status_ing_total())).classes(
                            'text-3xl font-bold text-blue-600')

            # 排队中的任务
            with ui.card().classes('flex-1 p-6 bg-orange-50 rounded-xl hover:shadow-lg transition-shadow'):
                with ui.column().classes('space-y-2'):
                    ui.label('排队中的任务').classes('text-lg text-gray-600')
                    with ui.row().classes('items-center gap-2'):
                        ui.icon('queue').classes(
                            'text-2xl text-orange-600')
                        ui.label(str(scheduler.get_request_queue_all_length())).classes(
                            'text-3xl font-bold text-orange-600')

            with ui.card().classes('flex-1 p-6 bg-green-50 rounded-xl hover:shadow-lg transition-shadow'):
                with ui.column().classes('space-y-2'):
                    ui.label('已完成的任务').classes('text-lg text-gray-600')
                    with ui.row().classes('items-center gap-2'):
                        ui.icon('beenhere').classes(
                            'text-2xl text-green-600')
                        ui.label(str(scheduler.get_request_done())).classes(
                            'text-3xl font-bold text-orange-600')

        # 运行中的模型列表
        with ui.card().classes('w-full p-6'):
            with ui.row().classes('items-center gap-4 mb-6'):
                ui.icon('model_training').classes(
                    'text-2xl text-purple-600')
                ui.markdown('## 运行中的模型').classes(
                    'text-2xl font-bold text-gray-800')

            # 获取并展示模型数据
            data = get_running_models().get("results", [])
            if not data or len(data) == 0:
                with ui.column().classes('w-full items-center py-12 space-y-4'):
                    ui.icon('error_outline').classes(
                        'text-4xl text-gray-400')
                    ui.label('暂无运行中的模型').classes('text-xl text-gray-400')
            else:
                model_headers = [
                    "name", "description"]
                model_headers_desc = ["名称", "描述"]
                model_list = [[m[i] for i in model_headers]
                              for m in data]
                df = pd.DataFrame(
                    data=model_list, columns=model_headers_desc)
                ModelCardComponent.index(df)
