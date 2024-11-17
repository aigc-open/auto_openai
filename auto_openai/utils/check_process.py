import psutil

def check_process_exists(keyword):
    """
    检查是否存在包含指定关键字的进程。

    :param keyword: 要检查的进程关键字
    :return: 如果进程存在返回 True，否则返回 False
    """
    # 遍历所有正在运行的进程
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # 检查进程的命令行参数是否包含指定的关键字
            if any(keyword in arg for arg in proc.info['cmdline']):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    raise Exception("服务启动异常")


if __name__ == '__main__':
    process_name = 'main.py'
    if check_process_exists(process_name):
        print(f"{process_name} is running.")
    else:
        print(f"{process_name} is not running.")