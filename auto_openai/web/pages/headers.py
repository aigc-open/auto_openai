from nicegui import ui
web_prefix = ""


def index():
    with ui.header().classes('bg-gradient-to-r from-indigo-600 to-purple-600 text-white shadow-lg p-4'):
        with ui.row().classes('w-full max-w-7xl mx-auto flex justify-between items-center'):
            # Logo section
            with ui.row().classes('flex items-center gap-3'):
                ui.icon('auto_awesome').classes('text-3xl text-yellow-300')
                ui.label('AI 调度系统').classes(
                    'text-2xl font-bold tracking-wide')

            # Navigation section
            with ui.row().classes('flex items-center gap-2 ml-auto'):
                nav_items = [
                    ('首页', '/', 'home'),
                    ('设计', f'{web_prefix}/docs-README', 'architecture'),
                    ('模型广场', f'{web_prefix}/docs-models', 'apps'),
                    ("体验区", f'{web_prefix}/experience', 'directions_run'),
                    ('全量模型', f'{web_prefix}/all-models', 'all_inclusive'),
                    ("运行时", f'{web_prefix}/docs-runtime', 'terminal'),
                    ('性能查看', f'{web_prefix}/docs-performance', 'speed'),
                    ('系统分布式虚拟节点',
                        f'{web_prefix}/docs-distributed_nodes', 'hub'),
                    ('Cursor接入', f'{web_prefix}/docs-cursor', 'mouse')
                ]

                for label, path, icon in nav_items:
                    # 检查当前页面路径是否匹配
                    is_active = ui.page.path == path

                    # 根据是否激活设置不同的样式
                    btn_classes = (
                        'px-4 py-2 rounded-lg transition-all duration-300 flex items-center gap-2 ' +
                        (
                            'bg-white text-purple-700 shadow-lg font-medium'
                            if is_active else
                            'hover:bg-white/20 text-white'
                        )
                    )

                    with ui.button(on_click=lambda p=path: ui.navigate.to(p)).classes(btn_classes):
                        ui.icon(icon).classes('text-lg')
                        ui.label(label)
