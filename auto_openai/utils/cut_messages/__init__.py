from typing import Literal
import os
import tiktoken
import time
import json
import math
from loguru import logger
from auto_openai.utils.daily_basic_function import logger_execute_time


@logger_execute_time(doc="tiktoken模型加载: cl100k_base")
def init_tiktoken():
    return tiktoken.get_encoding("cl100k_base")


def messages_token_count(messages, token_limit):
    """Calculate and return the total number of tokens in the provided messages."""
    start_time = time.time()
    tiktoken_encoding = init_tiktoken()
    logger.info(f"加载titoken: {time.time() - start_time}")
    encoding = tiktoken_encoding
    tokens_per_message = 4
    tokens_per_name = 1
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            # 分辨率
            width = 1024
            height = 1024
            block_count = math.ceil(width/512) * math.ceil(height/512)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            num_tokens += len(encoding.encode(item["text"], disallowed_special=()))
                        if item.get("type") == "image url":
                            num_tokens += (85+170*block_count)
            elif isinstance(value, str):
                num_tokens += len(encoding.encode(value, disallowed_special=()))
            else:
                num_tokens += len(encoding.encode(str(value), disallowed_special=()))
            if key == "name":
                num_tokens += tokens_per_name
            if num_tokens > token_limit:
                return num_tokens
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    logger.info(f"messages_token_count time: {time.time() - start_time}")
    return num_tokens


def string_token_count(str):
    """Calculate and return the token count in a given string."""
    tiktoken_encoding = init_tiktoken()
    tokens = tiktoken_encoding.encode(str, disallowed_special=())
    return len(tokens)


def cut_messages(messages, token_limit):
    if len(messages) >= 20:
        messages = messages[-20:]
    message_last = messages[-1]
    if message_last.get("role") == "assistant":
        # 如果最后一个元素是assistant,则不要
        messages.pop()
        message_last = messages[-1]
    while True:
        current_token_count = messages_token_count(messages, token_limit)
        if messages_token_count(messages, token_limit) < token_limit or len(messages) <= 0:
            break
        messages.pop(0)
    if len(messages) == 0:
        content = message_last.get("content", "")
        content = cut_string(content, token_limit=token_limit)
        message_last["content"] = content
        messages.append(message_last)
    logger.info(
        f"cut_messages len,token: {len(messages)},{current_token_count}")
    return messages


def cut_string(str, token_limit):
    while True:
        current_token_count = string_token_count(str)
        if len(str) <= 3 or current_token_count < token_limit:
            break
        str = str[int(len(str)*token_limit/current_token_count):]
    logger.info(f"cut_string len,token: {len(str)},{current_token_count}")
    return str
