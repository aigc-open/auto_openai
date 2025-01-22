import pandas as pd
from auto_openai.utils.depends import get_running_models, get_models_config_list
from auto_openai.utils.support_models.model_config import all_supported_device, system_models_config
from nicegui import ui
from .components import ModelCardComponent


def index():
    # 获取并展示模型数据
    data = system_models_config.list()
    if not data or len(data) == 0:
        with ui.column().classes('w-full items-center py-12 space-y-4'):
            ui.icon('error_outline').classes(
                'text-4xl text-gray-400')
            ui.label('暂无模型').classes('text-xl text-gray-400')
    else:
        # 添加统计卡片
        with ui.expansion('说明', caption='模型数量统计').classes('w-full'):
            with ui.row().classes('gap-4 mb-6'):
                with ui.card().classes('w-full bg-gradient-to-r from-blue-50 to-indigo-50 p-6'):
                    ui.markdown('- 这里可以查看系统已经支持了的模型列表').classes(
                        'text-lg text-gray-600')
                    ui.markdown("- 如果需要调用模型，请前往模型广场查看是否支持该模型的调用").classes(
                        'text-lg text-gray-600')
                with ui.card().classes('flex-1 p-4 bg-purple-50 rounded-xl'):
                    ui.label('全部模型数量').classes(
                        'text-sm text-gray-600 mb-1')
                    ui.label(str(len(data))).classes(
                        'text-2xl font-bold text-purple-600')

        # 原有的模型列表展示
        model_headers = [
            "name", "description", "model_url"]
        model_headers_desc = ["名称", "描述", "官网"]
        online_models_config = get_models_config_list()
        online_models_names = [m.get("name") for m in online_models_config]
        online_models = []
        offline_models = []
        for m in data:
            if m.name not in online_models_names:
                offline_models.append([m.dict()[i] for i in model_headers])
            else:
                online_models.append([m.dict()[i] for i in model_headers])

        # 创建 tabs 容器
        with ui.tabs().classes('w-full') as tabs:
            ui.tab('在线模型', icon='cloud_done')
            ui.tab('离线模型', icon='cloud_off')
            ui.tab('离线模型下载', icon='cloud_download')

        # Tab 内容面板
        with ui.tab_panels(tabs, value='在线模型').classes('w-full'):
            # 在线模型 tab
            with ui.tab_panel('在线模型'):
                if len(online_models) > 0:
                    online_df = pd.DataFrame(
                        data=online_models, columns=model_headers_desc)
                    ModelCardComponent.index(online_df, columns=3)
                else:
                    with ui.column().classes('w-full items-center py-12 space-y-4'):
                        ui.icon('cloud_off').classes(
                            'text-4xl text-gray-400')
                        ui.label('暂无在线模型').classes('text-xl text-gray-400')

            # 离线模型 tab
            with ui.tab_panel('离线模型'):
                if len(offline_models) > 0:
                    offline_df = pd.DataFrame(
                        data=offline_models, columns=model_headers_desc)
                    ModelCardComponent.index(
                        offline_df, columns=3)
                else:
                    with ui.column().classes('w-full items-center py-12 space-y-4'):
                        ui.icon('cloud_done').classes(
                            'text-4xl text-gray-400')
                        ui.label('暂无离线模型').classes('text-xl text-gray-400')
            # 离线模型下载 tab
            with ui.tab_panel('离线模型下载'):
                if len(offline_models) > 0:
                    offline_df = pd.DataFrame(
                        data=offline_models, columns=model_headers_desc)
                    model_names = offline_df['名称'].tolist()
                    # 命令输出区域（默认隐藏，点击按钮后显示）

                    with ui.card().classes('w-full p-6 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl shadow-sm mb-6'):
                        with ui.column().classes('gap-4'):
                            # 标题和说明
                            with ui.row().classes('items-center gap-2 mb-2'):
                                ui.icon('download').classes(
                                    'text-2xl text-blue-600')
                                ui.label('批量下载模型').classes(
                                    'text-xl font-bold text-gray-800')
                                download_shell_btn = ui.button('生成下载命令', icon='code').classes(
                                    'px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white ' +
                                    'rounded-lg transition-colors duration-200 flex items-center gap-2 ' +
                                    'whitespace-nowrap'  # 防止文字换行
                                ).props('flat dense')
                                clear_models_btn = ui.button(
                                    "清空", icon='delete')
                                load_all_models_btn = ui.button(
                                    "全选", icon='check')
                            ui.label('选择需要下载的模型，系统将自动生成下载命令').classes(
                                'text-sm text-gray-600 mb-4')

                            # 选择和按钮区域
                            with ui.row().classes('items-end gap-4 w-full'):  # 添加 w-full 确保行占满宽度
                                with ui.column().classes('w-3/4 flex-grow'):  # 添加 flex-grow 让选择框占据所有可用空间
                                    selected_models = ui.select(
                                        model_names,
                                        multiple=True,
                                        value=[],
                                        label='选择模型'
                                    ).classes('w-full min-w-[500px]').props('use-chips outlined dense fill-width')  # 添加最小宽度和fill-width
                            with ui.row().classes('items-end gap-4 w-full'):
                                download_shell_component = ui.code(
                                    '#', language="shell").style(
                                        'white-space: pre-wrap !important; '  # 强制换行
                                        'word-wrap: break-word !important; '  # 允许在单词内换行
                                        'max-width: 100% !important; '        # 限制最大宽度
                                        'overflow-x: hidden !important; '     # 隐藏横向滚动条
                                ).classes('w-full')

                    async def generate_download_shell():
                        download_shell_component.set_content(
                            system_models_config.generate_download_shell(selected_models.value))

                    download_shell_btn.on_click(generate_download_shell)
                    clear_models_btn.on_click(
                        lambda: selected_models.set_value([]))
                    load_all_models_btn.on_click(
                        lambda: selected_models.set_value(model_names))

                else:
                    with ui.column().classes('w-full items-center py-12 space-y-4'):
                        ui.icon('cloud_done').classes(
                            'text-4xl text-gray-400')
                        ui.label('暂无离线模型').classes('text-xl text-gray-400')
