from auto_openai.utils.depends import get_running_models, get_models_config_list
from nicegui import ui
from .experience_zone import llm_chat_page, sd_page


def index():
    online_models_config = get_models_config_list()

    running_models = [m.get("name")
                      for m in get_running_models().get("results", [])]
    online_models_map = {}
    for m in online_models_config:
        if m.get("name") not in running_models:
            online_models_map[m.get("name")] = m
        else:
            online_models_map[m.get("name")+" (running)"] = m
    model_names = list(online_models_map.keys())
    with ui.row().classes('items-end gap-4 w-full'):  # 添加 w-full 确保行占满宽度
        with ui.column().classes('w-3/4 flex-grow'):  # 添加 flex-grow 让选择框占据所有可用空间
            selected_models = ui.select(
                model_names,
                with_input=True,
                value=[],
                label='选择模型'
            ).classes('w-full min-w-[500px]').props('use-chips outlined dense fill-width')
            all_models_exp_zone = {}
            for name_ in online_models_map:
                if "LLM" in online_models_map[name_].get("api_type"):
                    zone = llm_chat_page.index(
                        model_name=name_.replace(" (running)", ""))
                    zone.set_visibility(False)
                elif "GenerateImage" in online_models_map[name_].get("api_type"):
                    zone = sd_page.index(
                        model_name=name_.replace(" (running)", ""))
                    zone.set_visibility(False)
                else:
                    with ui.card().classes('w-full min-w-[500px] p-4') as zone:
                        ui.label('该模型暂不支持体验').classes('text-red-500')
                    zone.set_visibility(False)
                all_models_exp_zone.update({name_: zone})

            def selected_models_on_value_change(e):
                name_ = selected_models.value
                for _name, zone_ in all_models_exp_zone.items():
                    if _name == name_:
                        zone_.set_visibility(True)
                    else:
                        zone_.set_visibility(False)

            selected_models.on_value_change(
                selected_models_on_value_change)
