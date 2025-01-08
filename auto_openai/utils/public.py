from auto_openai.utils.init_env import global_config
from auto_openai.utils.openai import Scheduler
from auto_openai.utils.redis_client import RedisClient
from auto_openai.utils.oss_client import OSSManager
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


class CustomRequest(Request):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.user_config = {}
        self.model_config = {}


class CustomRequestMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request = CustomRequest(request.scope)
        response = await call_next(request)
        return response


redis_client = RedisClient(config=global_config.REDIS_CLIENT_CONFIG)
if global_config.OSS_CLIENT_CONFIG:
    s3_client: OSSManager = OSSManager(endpoint_url=global_config.OSS_CLIENT_CONFIG['endpoint_url'],
                                       aws_access_key_id=global_config.OSS_CLIENT_CONFIG['aws_access_key_id'],
                                       aws_secret_access_key=global_config.OSS_CLIENT_CONFIG[
                                           'aws_secret_access_key'],
                                       region_name=global_config.OSS_CLIENT_CONFIG['region_name'])
else:
    s3_client: OSSManager = None
scheduler = Scheduler(redis_client=redis_client)
