import docker


class Docker:
    # https://docker-py.readthedocs.io/en/stable/containers.html
    def __init__(self):
        self.client = docker.from_env()
        self.labels = ["auto_openai_all"]

    def stop(self):
        for i in self.client.containers.list(filters={"label": self.labels[0]}, all=True):
            print(f"停止容器: {i.id}")
            i.stop()
            print(f"删除容器: {i.id}")
            i.remove()
        return True


if __name__ == "__main__":
    Docker().stop()
