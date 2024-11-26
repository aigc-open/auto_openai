from auto_openai.lm_server import CMD
import os

if __name__ == '__main__':
    from fire import Fire
    cmd = Fire(CMD)
    if cmd.strip():
        os.system(cmd)
    else:
        print('请输入命令 --help')
