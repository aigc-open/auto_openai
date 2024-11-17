from .base_text_to_image import BaseTextToImage
from .base_image_to_image import BaseImageToImage
from pydantic import BaseModel


class Workflow(BaseModel):
    model: str = "base_text_to_image"
    base_text_to_image: BaseTextToImage = None
    base_image_to_image: BaseImageToImage = None
    api_json: dict = {}
    # 素材下载映射关系 {"xxxx.png": "xxxxxxxxx.png"}
    download_json: dict = {}

    def format_json(self):
        if hasattr(self, self.model) and getattr(self, self.model) is not None:
            self.api_json, self.download_json = getattr(
                self, self.model).format_json()
