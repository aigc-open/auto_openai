from auto_openai.utils.support_models.model_config import all_supported_device, system_models_config
from nicegui import ui
import plotly.graph_objects as go





def index():
    title = "上海交通大学 软件学院"
    # 主要内容区
    # hero section
    with ui.card().classes('w-full p-8 bg-white text-black'):
        ui.label(title).classes('text-4xl font-bold mb-4')
        ui.label('基于 vllm 和 ComfyUI 等Backend的高效 AI 计算调度解决方案').classes(
            'text-xl mb-4')

    # 特性展示
    with ui.grid(columns=7).classes('gap-4'):
        for title, desc, icon in [
            ('高效推理', '利用 vllm 优化推理速度', '⚡'),
            ('智能调度', '自动分配计算资源', '🔄'),
            ('弹性扩展', '动态适应工作负载', '📈'),
            ('API 兼容', '支持 OpenAI API', '🔌'),
            ('多模型支持', '支持多种类型的 AI 模型', '🤖'),
            ('分布式计算', '提供分布式计算能力', '🌐'),
            ('异构算力支持', '支持 GPU、CPU、GCU 等多种算力', '🚀')
        ]:
            with ui.card().classes('p-3'):
                ui.label(icon).classes('text-4xl mb-2')
                ui.label(title).classes('text-xl font-bold mb-2')
                ui.label(desc).classes('text-gray-600')

    # 支持的设备展示
    with ui.card().classes('w-full p-8 bg-white shadow-lg rounded-xl'):
        with ui.row().classes('items-center mb-6'):
            ui.icon('devices').classes('text-3xl text-indigo-600 mr-3')
            ui.label('支持的硬件设备').classes('text-2xl font-bold text-gray-800')

        with ui.grid(columns=6).classes('gap-6'):
            for device_ in all_supported_device:
                device_name = device_.name.\
                    replace("NV-", "NVIDIA ").\
                    replace("EF-", "Enflame ")

                # 为不同类型设备选择不同的样式
                if "NVIDIA" in device_name:
                    bg_color = "from-green-50 to-emerald-50 hover:from-green-100 hover:to-emerald-100"
                    icon_color = "text-green-600"
                elif "Enflame" in device_name:
                    bg_color = "from-orange-50 to-amber-50 hover:from-orange-100 hover:to-amber-100"
                    icon_color = "text-orange-600"
                else:  # CPU
                    bg_color = "from-blue-50 to-indigo-50 hover:from-blue-100 hover:to-indigo-100"
                    icon_color = "text-blue-600"

                with ui.card().classes(f'p-6 bg-gradient-to-br bg-white rounded-xl transition-all duration-300 transform hover:scale-105 hover:shadow-xl'):
                    with ui.column().classes('items-center text-center gap-3'):
                        # 设备图标
                        if "NVIDIA" in device_name:
                            ui.image('auto_openai/statics/NVIDIA.png').classes('h-12 w-30 mx-auto object-contain')
                        elif "Enflame" in device_name:
                            ui.image('auto_openai/statics/Enflame.png').classes('h-12 w-30 mx-auto object-contain')
                        else:
                            ui.image('auto_openai/statics/CPU.png').classes('h-12 w-30 mx-auto object-contain')

                        # 设备名称
                        if len(device_name) <= 6:
                            device_name = f"通用计算 {device_name}"
                        ui.label(device_name).classes(
                            'text-lg font-bold text-gray-800')

                        # 分隔线（所有设备都显示）
                        ui.element('div').classes(
                            'w-16 h-0.5 bg-gray-200 my-2')

                        # 设备规格
                        if "CPU" not in device_name:
                            with ui.column().classes('gap-2 text-gray-600'):
                                with ui.row().classes('items-center justify-center gap-2'):
                                    ui.icon('memory').classes('text-sm')
                                    ui.label(f"{device_.mem}GB").classes(
                                        'text-sm')

                                with ui.row().classes('items-center justify-center gap-2'):
                                    ui.icon('speed').classes('text-sm')
                                    ui.label(f"{device_.bandwidth}").classes(
                                        'text-sm')
                        else:
                            pass
                            # CPU 显示通用配置信息
                            with ui.column().classes('gap-2 text-gray-600 items-center'):
                                with ui.row().classes('items-center justify-center gap-2'):
                                    ui.icon('computer').classes('text-sm')
                                    ui.label("通用计算").classes('text-sm')

    # 支持的模型展示
    with ui.card().classes('w-full p-6'):
        with ui.row().classes('items-center mb-6'):
            ui.icon('model_training').classes(
                'text-3xl text-indigo-600 mr-3')
            ui.label('支持的模型类型').classes('text-2xl font-bold text-gray-800')

        # 创建饼图展示模型分布
        fig = go.Figure(data=[go.Pie(
            labels=['大语言模型', '多模态', '图像生成', 'Embedding',
                    'Rerank', 'TTS/ASR', '视频生成'],
            values=[40, 10, 15, 10, 10, 10, 5],
            hole=.3
        )])
        fig.update_layout(
            showlegend=True,
            margin=dict(t=0, b=0, l=0, r=0),
            height=300
        )
        ui.plotly(fig).classes('w-full')

    # 技术架构
    with ui.card().classes('w-full p-6'):
        with ui.row().classes('items-center mb-6'):
            ui.icon('developer_board').classes(
                'text-3xl text-indigo-600 mr-3')
            ui.label('技术架构').classes('text-2xl font-bold text-gray-800')
        with ui.row().classes('gap-4 justify-center'):
            for tech in ['VLLM', 'ComfyUI', 'Transformers', 'SD WebUI']:
                with ui.card().classes('p-4 text-center'):
                    ui.label(tech).classes('font-bold')

    # 性能指标
    with ui.card().classes('w-full p-6'):
        with ui.row().classes('items-center mb-6'):
            ui.icon('speed').classes('text-3xl text-indigo-600 mr-3')
            ui.label('性能指标').classes('text-2xl font-bold text-gray-800')
        # 创建性能对比图
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['推理速度', '资源利用率', '并发处理能力'],
            y=[90, 85, 95],
            name='本系统'
        ))
        fig.add_trace(go.Bar(
            x=['推理速度', '资源利用率', '并发处理能力'],
            y=[60, 55, 65],
            name='传统系统'
        ))
        fig.update_layout(
            barmode='group',
            margin=dict(t=0, b=0, l=0, r=0),
            height=300
        )
        ui.plotly(fig).classes('w-full')
