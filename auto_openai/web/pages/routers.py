from nicegui import ui
from . import web_prefix
from . import headers,home_page, readme_page, \
    model_plaza, experience_view, all_models_views, runtime_view,\
    performance_view,distributed_nodes,cursor_view, cline_view


class UIWeb:

    @ui.page('/')
    @staticmethod
    def index():
        headers.index()
        with ui.column().classes('w-full max-w-7xl mx-auto p-4 gap-8'):
            home_page.index()

    @ui.page(f'{web_prefix}/docs-README')
    @staticmethod
    def readme():
        headers.index()
        with ui.column().classes('w-full max-w-7xl mx-auto p-4 gap-8'):
            readme_page.index()

    @ui.page(f'{web_prefix}/docs-models')
    @staticmethod
    def models():
        headers.index()
        with ui.column().classes('w-full max-w-7xl mx-auto p-4 gap-8'):
            model_plaza.index()

    @ui.page(f'{web_prefix}/experience')
    @staticmethod
    def experience():
        headers.index()
        with ui.column().classes('w-full max-w-7xl mx-auto p-4 gap-8'):
            experience_view.index()

    @ui.page(f'{web_prefix}/all-models')
    @staticmethod
    def all_models():
        headers.index()
        with ui.column().classes('w-full max-w-7xl mx-auto p-4 gap-8'):
            all_models_views.index()

    @ui.page(f'{web_prefix}/docs-runtime')
    @staticmethod
    def runtime():
        headers.index()
        with ui.column().classes('w-full max-w-7xl mx-auto p-4 gap-8'):
            runtime_view.index()

    @ui.page(f'{web_prefix}/docs-performance')
    @staticmethod
    def performance():
        headers.index()
        with ui.column().classes('w-full max-w-7xl mx-auto p-4 gap-8'):
            performance_view.index()

    @ui.page(f'{web_prefix}/docs-distributed_nodes')
    @staticmethod
    def distributed_nodes():
        headers.index()
        with ui.column().classes('w-full max-w-7xl mx-auto p-4 gap-8'):
            distributed_nodes.index()

    @ui.page(f'{web_prefix}/solution-cursor')
    @staticmethod
    def cursor():
        headers.index()
        with ui.column().classes('w-full max-w-7xl mx-auto p-4 gap-8'):
            cursor_view.index()

    @ui.page(f'{web_prefix}/solution-cline')
    @staticmethod
    def cline():
        headers.index()
        with ui.column().classes('w-full max-w-7xl mx-auto p-4 gap-8'):
            cline_view.index()

    @classmethod
    def register_ui(cls, fastapi_app, mount_path='/'):
        ui.run_with(
            fastapi_app,
            title="AutoOpenai 本地大模型",
            binding_refresh_interval=10,
            # NOTE this can be omitted if you want the paths passed to @ui.page to be at the root
            mount_path=mount_path,
            # NOTE setting a secret is optional but allows for persistent storage per user
            storage_secret='pick your private secret here',
        )
