
import yaml
import json


class Gen:
    default = {
        "version": "3",
        "services": {},
    }

    default_container = {
        "image": "harbor.uat.enflame.cc/library/enflame.cn/auto_openai:0.1",
        "environment": {
            "NODE_GPU_TOTAL": "{NODE_GPU_TOTAL}",
            "GPU_TYPE": "{GPU_TYPE}"
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
            "/root/share_models/:/root/share_models/"
        ],
        "privileged": True,
    }

    yaml_filename = "scheduler-{gpu}-of-{split_size}-{GPU_TYPE}-docker-compose.yml"

    @classmethod
    def run(cls, gpu: list = [0], split_size=1, GPU_TYPE="EF-S60", image: str = ""):
        containers = {}
        for idx, data in enumerate([gpu[i:i+split_size] for i in range(0, len(gpu), split_size)]):
            NODE_GPU_TOTAL = ",".join(map(str, data))
            container = dict(cls.default_container)
            environment = dict(container["environment"])
            environment.update(
                {"NODE_GPU_TOTAL": NODE_GPU_TOTAL, "GPU_TYPE": GPU_TYPE})
            container.update(
                {"environment": environment, "shm_size": f"{8*len(data)}gb"})
            if image:
                container.update({"image": image})
            containers[f"scheduler-{NODE_GPU_TOTAL.replace(',','_')}"] = container
        service = dict(cls.default)
        service.update({"services": containers})
        yaml_data = yaml.dump(service)
        with open(cls.yaml_filename.format(gpu="-".join(map(str, gpu)), split_size=split_size, GPU_TYPE="GCU" if "S60" in GPU_TYPE else "GPU"), 'w') as file:
            file.write(yaml_data)
        return service


# size 是指一个实例挂载得卡数
gcu_image = "harbor.uat.enflame.cc/library/enflame.cn/auto_openai:0.1"
Gen.run(gpu=[0, 1, 2, 3, 4, 5, 6, 7], split_size=1)
Gen.run(gpu=[0, 1, 2, 3, 4, 5, 6, 7], split_size=2)
Gen.run(gpu=[0, 1, 2, 3, 4, 5, 6, 7], split_size=4)
Gen.run(gpu=[0, 1, 2, 3, 4, 5, 6, 7], split_size=8)
Gen.run(gpu=[0, 1, 2, 3], split_size=1)
Gen.run(gpu=[0, 1, 2, 3], split_size=2)
Gen.run(gpu=[0, 1, 2, 3], split_size=4)
Gen.run(gpu=[4, 5], split_size=2)
