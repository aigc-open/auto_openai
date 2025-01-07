from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, HTTPBasic
import requests
import yaml
import time
from loguru import logger
from auto_openai.utils.init_env import global_config
from auto_openai.utils.support_models.model_config import system_models_config
from auto_openai.utils.public import scheduler


def get_model_config(name):
    for i in scheduler.get_available_model():
        if i['name'] == name:
            return i
    raise HTTPException(status_code=404, detail="model not found")


def get_models_config_list():
    return scheduler.get_available_model()


def get_combine_prompt_function(name):
    for i in system_models_config.list():
        if i.name == name:
            if hasattr(i, 'combine_prompt'):
                return i.combine_prompt
            return None
    return None
