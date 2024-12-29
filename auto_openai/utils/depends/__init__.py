from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, HTTPBasic
import requests
import yaml
import time
from loguru import logger
from auto_openai.utils.init_env import global_config
from auto_openai.utils.support_models.model_config import system_models_config


def get_model_config(name):
    for i in global_config.MODELS:
        if i['name'] == name:
            return i
    raise HTTPException(status_code=404, detail="model not found")


def get_combine_prompt_function(name):
    for i in system_models_config.list():
        if i.name == name:
            if hasattr(i, 'combine_prompt'):
                return i.combine_prompt
            return None
    return None
