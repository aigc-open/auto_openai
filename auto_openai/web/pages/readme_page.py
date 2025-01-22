import os
from auto_openai import project_path
from nicegui import ui
from .components import read_file

home_readme = os.path.join(project_path, "README.md")


def index():
    with ui.column().classes('w-full max-w-7xl mx-auto p-4 gap-8'):
        ui.markdown(read_file(home_readme))
