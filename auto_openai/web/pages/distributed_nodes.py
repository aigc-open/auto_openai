import pandas as pd
from auto_openai.utils.public import CustomRequestMiddleware, redis_client, s3_client
from auto_openai.utils.openai import Scheduler
from nicegui import ui


def index():
    def convert_to_dataframe():
        scheduler = Scheduler(redis_client=redis_client, http_request=None)
        data = scheduler.get_running_node()
        if data:
            df = pd.DataFrame(data)
            df['device-ids'] = df['device-ids'].apply(
                lambda x: ','.join(map(str, x)) if isinstance(x, list) else str(x))
            return df
        else:
            return pd.DataFrame([])

    node_df = convert_to_dataframe()

    # 虚拟节点卡片
    with ui.card().classes('w-full p-6 bg-white rounded-xl shadow-lg'):
        # 标题部分
        with ui.row().classes('items-center mb-4'):
            ui.icon('hub').classes('text-3xl text-purple-600 mr-2')
            ui.markdown('## 系统虚拟节点').classes(
                'text-xl font-bold text-gray-800')

        # 说明部分
        with ui.element('div').classes('mb-6 bg-purple-50 p-4 rounded-lg'):
            ui.markdown(
                """> **注意：** 真实的物理节点可能被虚拟成多个虚拟节点，每个虚拟节点可以处理一个请求。""").classes('text-gray-600')
            ui.markdown('**节点特性：**').classes('text-gray-600')
            ui.markdown('- 每个虚拟节点独占自己的显卡资源').classes('text-gray-600')
            ui.markdown('- 节点间资源隔离，互不影响').classes('text-gray-600')
            ui.markdown('- 支持动态扩缩容').classes('text-gray-600')
            ui.markdown('- 自动负载均衡').classes('text-gray-600')

        # 节点状态统计
        if not node_df.empty:
            with ui.row().classes('gap-4 mb-6'):
                with ui.card().classes('flex-1 p-4 bg-green-50 rounded-lg'):
                    ui.label('活跃节点数').classes('text-sm text-gray-600')
                    ui.label(str(len(node_df))).classes(
                        'text-2xl font-bold text-green-600')

                with ui.card().classes('flex-1 p-4 bg-blue-50 rounded-lg'):
                    ui.label('总GPU数').classes('text-sm text-gray-600')
                    total_gpus = node_df['device-ids'].str.count(',') + 1
                    ui.label(str(total_gpus.sum())).classes(
                        'text-2xl font-bold text-blue-600')

        # 节点列表表格
        with ui.element('div').classes('w-full overflow-x-auto'):
            if node_df.empty:
                with ui.element('div').classes('text-center py-8 bg-gray-50 rounded-lg'):
                    ui.icon('warning').classes(
                        'text-4xl text-yellow-500 mb-2')
                    ui.label('暂无运行中的节点').classes('text-gray-500')
            else:
                ui.table.from_pandas(
                    node_df
                ).classes('w-full border-collapse min-w-full')
