from auto_openai.utils.init_env import global_config
import pandas as pd
from auto_openai.utils.public import CustomRequestMiddleware, redis_client, s3_client
from auto_openai.utils.openai import Scheduler
from nicegui import ui


def index():
    def convert_to_dataframe():
        scheduler = Scheduler(redis_client=redis_client, http_request=None,
                              queue_timeout=global_config.QUEUE_TIMEOUT, infer_timeout=global_config.INFER_TIMEOUT)
        data = scheduler.get_profiler()
        if not data.get("start_server_time") or not data.get("Profiler"):
            return pd.DataFrame(), pd.DataFrame()
        start_server_time_df = pd.DataFrame.from_dict(
            data["start_server_time"], orient='index') if data.get("start_server_time") else pd.DataFrame()
        tps_spi_df = pd.DataFrame.from_dict(
            data["Profiler"], orient='index') if data.get("Profiler") else pd.DataFrame()

        # 添加模型名称列
        start_server_time_df.reset_index(inplace=True)
        start_server_time_df.columns = [
            'Model Name', 'Max', 'Min', 'New', "Description"]
        tps_spi_df.reset_index(inplace=True)
        tps_spi_df.columns = ['Model Name', 'Max',
                              'Min', 'New', "Description"]

        return start_server_time_df, tps_spi_df

    start_server_time_df, tps_spi_df = convert_to_dataframe()

    # 模型加载时间卡片
    with ui.card().classes('w-full p-6 mb-6 bg-white rounded-xl shadow-lg'):
        with ui.row().classes('items-center mb-4'):
            ui.icon('timer').classes('text-3xl text-blue-600 mr-2')
            ui.markdown('## 模型加载时间').classes(
                'text-xl font-bold text-gray-800')

        ui.markdown("""
        > 注意：模型加载时间是指模型从加载到可以开始处理请求的时间。
        """).classes('mb-4 text-gray-600 bg-blue-50 p-4 rounded-lg')

        with ui.element('div').classes('w-full overflow-x-auto'):
            ui.table.from_pandas(
                start_server_time_df
            ).classes('w-full border-collapse min-w-full')

    # 模型性能卡片
    with ui.card().classes('w-full p-6 bg-white rounded-xl shadow-lg'):
        with ui.row().classes('items-center mb-4'):
            ui.icon('speed').classes('text-3xl text-green-600 mr-2')
            ui.markdown('## 模型性能').classes(
                'text-xl font-bold text-gray-800')

        with ui.element('div').classes('mb-4 bg-green-50 p-4 rounded-lg'):
            ui.markdown("**性能指标说明：**").classes('text-gray-600')
            ui.markdown("- 大语言模型: 每秒生成token的数量").classes('text-gray-600')
            ui.markdown("- 图像生成: 每张图像生成所需时间").classes('text-gray-600')
            ui.markdown("- 语音识别：每秒处理音频帧的所需时间").classes('text-gray-600')
            ui.markdown("- 语音合成：每秒处理音频帧的所需时间").classes('text-gray-600')

        with ui.element('div').classes('w-full overflow-x-auto'):
            ui.table.from_pandas(
                tps_spi_df
            ).classes('w-full border-collapse min-w-full')
