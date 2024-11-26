from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, HTTPBasic
import requests
import yaml
import time
from loguru import logger
from auto_openai.utils.init_env import global_config


def get_model_config(name):
    for i in global_config.MODELS:
        if i['name'] == name:
            return i
    raise HTTPException(status_code=400, detail="model not found")
