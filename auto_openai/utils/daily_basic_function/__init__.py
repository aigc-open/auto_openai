import os
import json
import time
import functools
import requests
from loguru import logger
from io import BytesIO
from contextlib import contextmanager
from uuid_extensions import uuid7, uuid7str

def generate_sorted_uuid():
    return str(uuid7(as_type="int"))


def generate_random_dir(cache_data_dir=os.environ.get("CACHE_DATA_DIR", "./tmp/")):
    _save_dir = os.path.join(cache_data_dir, generate_sorted_uuid())
    os.makedirs(_save_dir, exist_ok=True)
    return _save_dir


@contextmanager
def safe_dir(MOVE_SAFE_DIR=bool(os.environ.get("MOVE_SAFE_DIR", True))):
    _save_dir = generate_random_dir()
    try:
        yield _save_dir
    except Exception as e:
        raise e
    finally:
        if MOVE_SAFE_DIR:
            os.system(f"rm -rf {_save_dir}")


def logger_execute_time(doc="执行时间"):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """计算方法执行时间，并输出执行时间"""
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            logger.info(f"{doc}: {execution_time}")
            return result

        return wrapper

    return decorator


def pretty_print_dict(doc="参数如下", **kwargs):
    placeholder = "*" * 100
    s = f"\n{placeholder}\n{doc}:\n"
    for k, v in kwargs.items():
        s += f"\n{k}: {v}"
    s += f"\n{placeholder}"
    logger.info(s)