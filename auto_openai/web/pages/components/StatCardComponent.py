from nicegui import ui


def index(label: str, value: int):
    with ui.row().classes('gap-4 p-4 mb-2'):
        with ui.card().classes('flex-1 p-4 bg-blue-50 rounded-xl'):
            ui.label(label).classes(
                'text-sm text-gray-600 mb-1')
            ui.label(str(value)).classes(
                'text-2xl font-bold text-blue-600')
