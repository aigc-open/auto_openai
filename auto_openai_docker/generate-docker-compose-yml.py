
import yaml
import json
import time
import random
import os

BASE_PORT = 30000

image = "registry.cn-shanghai.aliyuncs.com/zhph-server/auto_openai:shdx"

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
        for generate_dir in ["auto_openai_gpu_node", "auto_openai_gcu_node"]:
            os.makedirs(generate_dir, exist_ok=True)
            with open(cls.yaml_filename.format(generate_dir=generate_dir, gpu="-".join(map(str, gpu)), split_size=split_size, other_name=other_name), 'w') as file:
                file.write(yaml_data)
        return service
    
    @classmethod
    def generate_master_node(cls):
        default = {
            "version": "3",
            "services": {
                "openai-api": {
                    "image": image,
                    "ports": ["9000:9000"],
                    "command": [
                        "/bin/sh",
                        "-c",
                        "python3 -m auto_openai.main --port=9000"
                    ],
                    "restart": "always",
                    "volumes": ["./conf:/app/conf"],
                }
            }
        }
        with open("auto_openai_master_node/docker-compose.yml", "w") as file:
            yaml.dump(default, file)

    @classmethod
    def sync_conf(cls):
        import shutil
        # Copy conf directory to gpu and cpu paths
        for generate_dir in ["auto_openai_gpu_node", "auto_openai_master_node"]:
            conf_dest = os.path.join(generate_dir, "conf")
            if os.path.exists("conf"):
                if os.path.exists(conf_dest):
                    shutil.rmtree(conf_dest)
                shutil.copytree("conf", conf_dest)

    @classmethod
    def generate_node_pull_script(cls):
        with open("auto_openai_gpu_node/pull.sh", "w") as file:
            file.write(f"docker pull {image}")
        with open("auto_openai_master_node/pull.sh", "w") as file:
            file.write(f"docker pull {image}")

# size 是指一个实例挂载得卡数
# Gen.run(gpu=[0, 1, 2, 3, 4, 5, 6, 7], split_size=1)
# Gen.run(gpu=[0, 1, 2, 3, 4, 5, 6, 7], split_size=2)
# Gen.run(gpu=[0, 1, 2, 3, 4, 5, 6, 7], split_size=4)
# Gen.run(gpu=[0, 1, 2, 3, 4, 5, 6, 7], split_size=8)
Gen.run(gpu=[0, 1, 2, 3], split_size=1)
Gen.run(gpu=[0, 1], split_size=1)
Gen.run(gpu=[0, 1, 2, 3], split_size=2)
Gen.run(gpu=[0, 1, 2, 3], split_size=4)
Gen.run(gpu=[100, 100, 100, 100], split_size=1,
        AVAILABLE_SERVER_TYPES="embedding,rerank", GPU_TYPE="CPU", other_name="-cpu")
Gen.run(gpu=[101, 101, 101, 101], split_size=1,
        AVAILABLE_SERVER_TYPES="embedding,rerank", GPU_TYPE="CPU", other_name="-cpu")
Gen.generate_master_node()
Gen.generate_node_pull_script()
Gen.sync_conf()
os.system("cd auto_openai_gpu_node && bash echo.sh")
os.system("cd auto_openai_gcu_node && bash echo.sh")