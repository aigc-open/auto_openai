FROM artifact.enflame.cn/enflame_docker_images/ubuntu/qic_ubuntu_2004_gcc9:1.5.5

RUN rm -f /usr/bin/python \
    && rm -f /usr/bin/python3 \
    && rm -f /usr/bin/python-config \
    && rm -f /usr/bin/python3-config \
    && ln -s /usr/local/bin/python3.10 /usr/bin/python \
    && ln -s /usr/local/bin/python3.10 /usr/bin/python3 \
    && ln -s /usr/local/bin/python3.10-config /usr/bin/python-config \
    && ln -s /usr/local/bin/python3.10-config /usr/bin/python3-config \
    && pip3.10 install --upgrade pip

RUN apt update -y && apt install curl git ffmpeg -y && python3.10 -m pip install --upgrade pip==24.0

RUN pip3 config set global.index-url http://artifact.enflame.cn/artifactory/api/pypi/pypi-remote/simple && pip3 config set global.trusted-host artifact.enflame.cn

WORKDIR /workspace

# 安装本项目
COPY ./dist/auto_openai-0.1-py3-none-any.whl /tmp/auto_openai-0.1-py3-none-any.whl
RUN pip3.10 install /tmp/auto_openai-0.1-py3-none-any.whl

# 安装需要的依赖，按需安装
RUN python3 -m auto_openai.lm_server.install_plugin tiktoken
RUN python3 -m auto_openai.lm_server.install_plugin comfyui
RUN python3 -m auto_openai.lm_server.install_plugin webui
RUN python3 -m auto_openai.lm_server.install_plugin maskgct
#
RUN python3 -m auto_openai.lm_server.install_plugin embedding
RUN python3 -m auto_openai.lm_server.install_plugin funasr
RUN python3 -m auto_openai.lm_server.install_plugin llm_transformer
RUN python3 -m auto_openai.lm_server.install_plugin rerank

# 软件栈安装
RUN wget -c -O topsrider.run "http://10.9.115.79:15001/api/public/dl/WpzIWFnM/nfs/tops_daily/TopsRider_i3x_3.2.109_deb_amd64.run" && \
    bash topsrider.run -C vllm --container -y && \
    rm -rf topsrider.run

# 覆盖部分依赖
RUN pip3.10 install /tmp/auto_openai-0.1-py3-none-any.whl
RUN pip3.10 install torch==2.3.0 --force-reinstall

COPY . /app/

ENTRYPOINT []

WORKDIR /app


CMD "python3 -m auto_openai.main --port=9000 --workers=10"





