import os
from pydantic import BaseModel
from enum import Enum
from fire import Fire
import pip


class Plugin:

    ############## 第三方项目 ##############
    @classmethod
    def tiktoken(cls):
        import tiktoken
        import time
        start_time = time.time()
        tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
        print(f"titoken available, time: {time.time() - start_time}")


if __name__ == "__main__":
    Fire(Plugin)
