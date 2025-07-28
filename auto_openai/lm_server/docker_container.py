import docker
import time
from loguru import logger
from auto_openai.utils.init_env import global_config


class Docker:
    # https://docker-py.readthedocs.io/en/stable/containers.html
    def __init__(self):
        self.client = docker.from_env()
        self.labels = [global_config.LABEL, "auto_openai_all"]

    def run(self, image, command, device_ids: str, GPU_TYPE="S60", network_mode="host", environment=[]):
        data = {
            "image": image,
            "command": f"bash -c '{command}'",
            "network_mode": network_mode,
            "detach": True,
            "labels": self.labels,
            # "restart_policy": {"Name": "on-failure", "MaximumRetryCount": 5},
            "restart_policy": {"Name": "always"},
            # "auto_remove": True,
            "privileged": True,
            "shm_size": "16gb",
            "volumes": ["/root/share_models/:/root/share_models/",
                        "/root/share_models/webui-models/:/workspace/ComfyUI/models",
                        "/root/share_models/webui-models/comfyui_controlnet_aux_ckpts:/workspace/ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts",
                        "/root/share_models/webui-models/controlnet_v1.1_annotator/:/workspace/stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/downloads"
                        ],
            "environment": environment
        }
        if "NV" in GPU_TYPE:
            # data["environment"] = [
            #     f"NVIDIA_VISIBLE_DEVICES={device_ids}", "CUDA_VISIBLE_DEVICES={device_ids}"]
            data["device_requests"] = [
                docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
            ]
        else:
            pass
            # data["environment"] = [f"TOPS_VISIBLE_DEVICES={device_ids}"]
        logger.info(f"容器启动参数: {data}")
        container = self.client.containers.run(**data)
        return container

    def stop(self):
        for i in self.client.containers.list(filters={"label": self.labels[0]}, all=True):
            logger.info(f"停止容器: {i.id}")
            i.stop()
        return True

    def remove(self):
        for i in self.client.containers.list(filters={"label": self.labels[0]}, all=True):
            logger.info(f"删除容器: {i.id}")
            i.remove()
        return True
