
import yaml
import json
import time
import random
import os

BASE_PORT = 30000

image = "harbor.uat.enflame.cc/library/enflame.cn/auto_openai:0.2"
generate_dir = "auto_openai_gcu"
image = "registry.cn-shanghai.aliyuncs.com/zhph-server/auto_openai:0.2"
generate_dir = "auto_openai_deployment"


class Gen:
    default = {
        "version": "3",
        "services": {},
    }

    default_container = {
        "image": image,
        "environment": {
            "NODE_GPU_TOTAL": "{NODE_GPU_TOTAL}"
        },
        "shm_size": "8gb",
        "command": [
            "/bin/sh",
            "-c",
            "python3 -m auto_openai.scheduler"
        ],
        "restart": "always",
        "volumes": [
            "./conf/:/app/conf",
            "/root/share_models/:/root/share_models/",
            "/var/run/docker.sock:/var/run/docker.sock",
            "/usr/bin/docker:/usr/bin/docker"
        ],
        "privileged": True,
        "network_mode": "host"
    }

    yaml_filename = "{generate_dir}/scheduler-{gpu}-of-{split_size}{other_name}-docker-compose.yml"

    @classmethod
    def run(cls, gpu: list = [0], split_size=1, AVAILABLE_SERVER_TYPES="ALL", AVAILABLE_MODELS="ALL", GPU_TYPE="", other_name="", CPU_IMAGE_TYPE="NV"):
        global BASE_PORT
        containers = {}
        for idx, data in enumerate([gpu[i:i+split_size] for i in range(0, len(gpu), split_size)]):
            BASE_PORT += 8
            NODE_GPU_TOTAL = ",".join(map(str, data))
            container = dict(cls.default_container)
            environment = dict(container["environment"])
            environment.update(
                {"NODE_GPU_TOTAL": NODE_GPU_TOTAL,
                 "LABEL": f"lm-server-{int(time.time())}-{random.randint(0, int(time.time()))}",
                 "LM_SERVER_BASE_PORT": BASE_PORT,
                 "AVAILABLE_SERVER_TYPES": AVAILABLE_SERVER_TYPES,
                 "AVAILABLE_MODELS": AVAILABLE_MODELS
                 })
            if GPU_TYPE:
                environment.update({"GPU_TYPE": GPU_TYPE})
                if GPU_TYPE == "CPU":
                    environment.update({"CPU_IMAGE_TYPE": CPU_IMAGE_TYPE})
            container.update(
                {"environment": environment, "shm_size": f"8gb"})
            containers[f"scheduler-{NODE_GPU_TOTAL.replace(',','_')}-of-{idx}{other_name}"] = container
        service = dict(cls.default)
        service.update({"services": containers})
        yaml_data = yaml.dump(service)
        os.makedirs(generate_dir, exist_ok=True)
        with open(cls.yaml_filename.format(generate_dir=generate_dir, gpu="-".join(map(str, gpu)), split_size=split_size, other_name=other_name), 'w') as file:
            file.write(yaml_data)
        return service


# size 是指一个实例挂载得卡数
Gen.run(gpu=[0, 1, 2, 3, 4, 5, 6, 7], split_size=1)
Gen.run(gpu=[0, 1, 2, 3, 4, 5, 6, 7], split_size=2)
Gen.run(gpu=[0, 1, 2, 3, 4, 5, 6, 7], split_size=4)
Gen.run(gpu=[0, 1, 2, 3, 4, 5, 6, 7], split_size=8)
Gen.run(gpu=[0, 1, 2, 3], split_size=1)
Gen.run(gpu=[0, 1, 2], split_size=1)
Gen.run(gpu=[0, 1], split_size=1)
Gen.run(gpu=[0, 1, 2, 3], split_size=2)
Gen.run(gpu=[0, 1, 2, 3], split_size=4)
Gen.run(gpu=[4, 4, 4, 4], split_size=1,
        AVAILABLE_SERVER_TYPES="embedding,rerank")
Gen.run(gpu=[3, 3, 3, 3], split_size=1,
        AVAILABLE_SERVER_TYPES="embedding,rerank")
Gen.run(gpu=[100, 100], split_size=1,
        AVAILABLE_SERVER_TYPES="http-llm", GPU_TYPE="CPU")
# 固定的
# node-01
Gen.run(gpu=[0], split_size=1,
        AVAILABLE_MODELS="Qwen2.5-Coder-32B-Instruct-GPTQ-Int4:32k", other_name="-Qwen2.5-Coder-32B-Instruct-GPTQ-Int4")
Gen.run(gpu=[1], split_size=1,
        AVAILABLE_MODELS="DeepSeek-R1-Distill-Qwen-14B:10k", other_name="-DeepSeek-R1-Distill-Qwen-14B")
Gen.run(gpu=[2], split_size=1,
        AVAILABLE_MODELS="Qwen2.5-VL-7B-Instruct:32k", other_name="-Qwen2.5-VL-7B-Instruct")
Gen.run(gpu=[3, 3], split_size=1,
        AVAILABLE_MODELS="SolutionBaseGenerateImage/Kolors", other_name="-Kolors")
# node-02
# 万相视频生成
Gen.run(gpu=[0], split_size=1,
        AVAILABLE_MODELS="wan/Wan2.1-T2V-1.3B-Diffusers", other_name="-Wan2.1-TextToVideo")
# 这张卡被拆分8份，给embeding,rerank使用，小模型
Gen.run(gpu=[1, 1, 1, 1], split_size=1,
        AVAILABLE_MODELS="bge-base-zh-v1.5", other_name="-bge-base-zh-v1.5")
Gen.run(gpu=[1, 1, 1, 1], split_size=1,
        AVAILABLE_MODELS="bge-reranker-v2-m3", other_name="-bge-reranker-v2-m3")
# 这两张卡任意分配
Gen.run(gpu=[2, 3], split_size=1,
        AVAILABLE_MODELS="Qwen2.5-Coder-32B-Instruct-GPTQ-Int4:32k,DeepSeek-R1-Distill-Qwen-14B:20k,Qwen2.5-VL-7B-Instruct:32k,SolutionBaseGenerateImage/Kolors", other_name="-anyone")

Gen.run(gpu=[0], split_size=1)
Gen.run(gpu=[100, 100], split_size=1,
        AVAILABLE_SERVER_TYPES="embedding,rerank", GPU_TYPE="CPU",
        other_name="-embedding-rerank-cpu")
