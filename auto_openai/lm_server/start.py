from auto_openai.lm_server import CMD
import os
from auto_openai.lm_server.docker_container import Docker

if __name__ == '__main__':
    from fire import Fire
    cmd = Fire(CMD)
    input("按Enter键停止服务")
    CMD.kill()
