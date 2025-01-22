import os
import pandas as pd
from nicegui import ui
from auto_openai import project_path
from auto_openai.web.pages import generate_api_documentation
from . import StatCardComponent, ModelCardComponent, read_file
demo_path = os.path.join(project_path, "web/tests")


def index(model_config, model_type,
          model_headers=["name", "description", "model_url"],
          model_headers_desc=["名称", "描述", "官网"],
          RequestBaseModel=[]):
    if not model_config:
        with ui.card().classes('w-full p-8 bg-white rounded-xl shadow-lg'):
            ui.markdown("# 努力开发中...").classes(
                'text-2xl font-bold text-gray-400')
        return

    with ui.card().classes('w-full bg-white'):
        # Tab navigation
        with ui.tabs().classes('w-full bg-gray-50 border-b') as tabs:
            tab_items = [
                ("支持的模型列表", "view_list"),
                ("文档参数说明", "description"),
                ("python 示例", "code"),
                ("curl 示例", "terminal")
            ]

            for label, icon in tab_items:
                ui.tab(name=label, icon=icon)
        # Tab content panels
        with ui.tab_panels(tabs, value="支持的模型列表").classes('w-full'):
            # Models list panel
            with ui.tab_panel('支持的模型列表'):
                model_list = [[m[i] for i in model_headers]
                              for m in model_config]
                df = pd.DataFrame(
                    data=model_list, columns=model_headers_desc)

                # 添加数据统计卡片
                StatCardComponent.index(label="模型总数", value=len(df))
                # 模型卡片网格
                ModelCardComponent.index(data=df)

            # API documentation panel
            with ui.tab_panel('文档参数说明'):
                for r_basemodel in RequestBaseModel:
                    with ui.card().classes('mb-6'):
                        ui.markdown(f"# {r_basemodel.__name__}").classes(
                            'p-4 bg-gradient-to-r from-gray-50 to-gray-100 font-bold border-b'
                        )
                        data = generate_api_documentation(
                            r_basemodel.model_json_schema())
                        # 优化文档表格样式
                        ui.table.from_pandas(
                            pd.DataFrame(data=data),
                        ).classes('w-full')

            # Python example panel
            with ui.tab_panel('python 示例'):
                with ui.card().classes('overflow-hidden rounded-xl border'):
                    py_path = os.path.join(
                        demo_path, f"{model_type}.py")
                    ui.code(read_file(py_path),
                            language="python").classes('w-full')

            # CURL example panel
            with ui.tab_panel('curl 示例'):
                with ui.card().classes('overflow-hidden rounded-xl border'):
                    curl_path = os.path.join(
                        demo_path, f"{model_type}.sh")
                    ui.code(read_file(curl_path),
                            language="shell").classes('w-full')
