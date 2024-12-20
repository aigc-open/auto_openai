
import yaml
import json

# image = "harbor.uat.enflame.cc/library/enflame.cn/auto_openai:0.2"
image = "registry.cn-shanghai.aliyuncs.com/zhph-server/auto_openai:0.2"


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

    yaml_filename = "scheduler-{gpu}-of-{split_size}-docker-compose.yml"

    @classmethod
    def run(cls, gpu: list = [0], split_size=1):
        containers = {}
        for idx, data in enumerate([gpu[i:i+split_size] for i in range(0, len(gpu), split_size)]):
            NODE_GPU_TOTAL = ",".join(map(str, data))
            container = dict(cls.default_container)
            environment = dict(container["environment"])
            environment.update(
                {"NODE_GPU_TOTAL": NODE_GPU_TOTAL})
            container.update(
                {"environment": environment, "shm_size": f"8gb"})
            containers[f"scheduler-{NODE_GPU_TOTAL.replace(',','_')}"] = container
        service = dict(cls.default)
        service.update({"services": containers})
        yaml_data = yaml.dump(service)
        with open(cls.yaml_filename.format(gpu="-".join(map(str, gpu)), split_size=split_size), 'w') as file:
            file.write(yaml_data)
        return service


# size 是指一个实例挂载得卡数
# Gen.run(gpu=[0, 1, 2, 3, 4, 5, 6, 7], split_size=1)
# Gen.run(gpu=[0, 1, 2, 3, 4, 5, 6, 7], split_size=2)
# Gen.run(gpu=[0, 1, 2, 3, 4, 5, 6, 7], split_size=4)
# Gen.run(gpu=[0, 1, 2, 3, 4, 5, 6, 7], split_size=8)
Gen.run(gpu=[0, 1, 2, 3], split_size=1)
# Gen.run(gpu=[0, 1, 2, 3], split_size=2)
# Gen.run(gpu=[0, 1, 2, 3], split_size=4)
# Gen.run(gpu=[4, 5], split_size=2)
