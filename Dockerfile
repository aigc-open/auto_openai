FROM python:3.10-buster

# RUN echo "deb https://mirrors.aliyun.com/debian/ buster main contrib non-free" > /etc/apt/sources.list && \
#     echo "deb https://mirrors.aliyun.com/debian/ buster-updates main contrib non-free" >> /etc/apt/sources.list && \
#     echo "deb https://mirrors.aliyun.com/debian/ buster-backports main contrib non-free" >> /etc/apt/sources.list && \
#     echo "deb https://mirrors.aliyun.com/debian-security/ buster/updates main contrib non-free" >> /etc/apt/sources.list

RUN apt-get update && \
    apt-get install -y gcc libc-dev default-mysql-client default-libmysqlclient-dev nginx libsasl2-dev libldap2-dev libssl-dev zip jq && \
    apt-get clean


RUN pip config set global.index-url http://artifact.enflame.cn/artifactory/api/pypi/pypi-remote/simple && pip config set install.trusted-host artifact.enflame.cn
# RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn


WORKDIR /workspace

# 安装本项目
COPY ./dist/auto_openai-0.2-py3-none-any.whl /tmp/auto_openai-0.2-py3-none-any.whl
RUN pip install /tmp/auto_openai-0.2-py3-none-any.whl 

RUN python3 -m auto_openai.lm_server.install_models tiktoken

COPY . /app/

ENTRYPOINT []

WORKDIR /app


CMD "python3 -m auto_openai.main --port=9000 "





