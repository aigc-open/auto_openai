import pandas as pd
from nicegui import ui


def index(data: pd.DataFrame, columns=3):
    with ui.grid(columns=columns).classes('gap-4 p-4'):
        for _, row in data.iterrows():
            with ui.card().classes('p-4 hover:shadow-lg transition-all duration-300 bg-white border rounded-xl h-full'):
                # 模型名称 - 使用 column 布局来处理长名称
                with ui.column().classes('gap-2 mb-3 w-full'):
                    with ui.row().classes('items-center gap-2 mb-1'):
                        ui.icon('model_training').classes(
                            'text-2xl text-purple-600 shrink-0')
                        ui.label(row['名称']).classes(
                            'text-lg font-bold text-gray-800 break-all')

                # 最大支持tokens（如果存在）
                if '最大支持tokens' in row:
                    with ui.row().classes('items-center gap-2 mb-2'):
                        ui.icon('data_array').classes(
                            'text-blue-500 shrink-0')
                        ui.label(f"最大支持: {row['最大支持tokens']}").classes(
                            'text-sm text-gray-600')

                # 描述
                if '描述' in row:
                    with ui.row().classes('items-start gap-2'):
                        ui.icon('description').classes(
                            'text-gray-400 mt-1 shrink-0')
                        ui.label(row['描述']).classes(
                            'text-sm text-gray-600 break-words')
                if "官网" in row:
                    if row['官网']:
                        with ui.row().classes('items-center gap-2'):
                            ui.link("官网", target=row['官网'], new_tab=True)
