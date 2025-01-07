import socket
import os


def get_address_hostname():
    ip_address = "0.0.0.0"
    hostname = "host"
    id_ = os.environ.get("LM_SERVER_BASE_PORT", 30000)
    try:
        # 创建一个 socket 对象
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 连接到一个外部地址（不需要实际连接）
        s.connect(("8.8.8.8", 80))  # 使用 Google 的公共 DNS 服务器
        ip_address = s.getsockname()[0]  # 获取本机的 IP 地址
        hostname = socket.gethostname()
    finally:
        s.close()  # 关闭 socket
    return {"ip": ip_address, "hostname": f"{ip_address}@{hostname}-{id_}"}
