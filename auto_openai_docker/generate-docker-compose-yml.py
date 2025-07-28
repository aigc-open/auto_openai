
import yaml
import json
import time
import random
import os
import shutil

BASE_PORT = 30000

image = "registry.cn-shanghai.aliyuncs.com/zhph-server/auto_openai:shdx"

GPU_TYPE_LIST = ["CPU", "NV-A100-80G", "NV-4090", "EF-S60"]

class Gen:
    default = {
        "version": "3",
        "services": {},
    }
    
    # 添加类变量来跟踪生成的目录
    generated_dirs = set()
    
    # 全局控制是否挂载代码
    mount_code = False

    # 基础挂载卷
    base_volumes = [
        "../conf/:/app/conf",
        "/root/share_models/:/root/share_models/",
        "/var/run/docker.sock:/var/run/docker.sock",
        "/usr/bin/docker:/usr/bin/docker"
    ]

    # 代码挂载卷
    code_volume = ["/root/share_models/auto_openai/auto_openai:/app/auto_openai", 
                   "/root/share_models/auto_openai/auto_openai_lm_images:/app/auto_openai_lm_images"]

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
        "volumes": base_volumes,
        "privileged": True,
        "network_mode": "host"
    }

    yaml_filename = "{generate_dir}/scheduler-{gpu}-of-{split_size}{other_name}-docker-compose.yml"

    @classmethod
    def copy_scripts_to_directory(cls, target_dir):
        """复制 echo.sh、start.sh、stop.sh 脚本到目标目录"""
        script_files = ["echo.sh", "start.sh", "stop.sh"]
        source_dir = "scheduler-template"  # 使用 cpu_node 作为脚本模板源
        
        for script_file in script_files:
            source_path = os.path.join(source_dir, script_file)
            target_path = os.path.join(target_dir, script_file)
            
            if os.path.exists(source_path):
                try:
                    shutil.copy2(source_path, target_path)
                    # 确保脚本文件有执行权限
                    os.chmod(target_path, 0o755)
                    print(f"Copied {script_file} to {target_dir}")
                except Exception as e:
                    print(f"Failed to copy {script_file} to {target_dir}: {e}")
            else:
                print(f"Source script {source_path} not found")

    @classmethod
    def generate_embedding_rerank_yaml(cls, target_dir):
        """为目录生成单独的 embedding,rerank 服务 yaml 文件"""
        global BASE_PORT
        containers = {}
        
        for i in range(2):  # 添加两个 embedding,rerank 实例
            BASE_PORT += 8
            cpu_gpu_id = 100 + i
            container = dict(cls.default_container)
            environment = dict(container["environment"])
            environment.update(
                {"NODE_GPU_TOTAL": str(cpu_gpu_id),
                 "LABEL": f"lm-server-{int(time.time())}-{random.randint(0, int(time.time()))}",
                 "LM_SERVER_BASE_PORT": BASE_PORT,
                 "AVAILABLE_SERVER_TYPES": "embedding,rerank",
                 "AVAILABLE_MODELS": "ALL",
                 "GPU_TYPE": "CPU",
                 "CPU_IMAGE_TYPE": "NV"
                 })
            container.update(
                {"environment": environment, "shm_size": f"8gb"})
            containers[f"scheduler-{cpu_gpu_id}-of-{i}-cpu"] = container
        
        service = dict(cls.default)
        service.update({"services": containers})
        yaml_data = yaml.dump(service)
        
        # 生成单独的 embedding,rerank yaml 文件
        embedding_yaml_path = os.path.join(target_dir, "scheduler-embedding-rerank-docker-compose.yml")
        with open(embedding_yaml_path, 'w') as file:
            file.write(yaml_data)
        print(f"Generated embedding-rerank yaml: {embedding_yaml_path}")

    @classmethod
    def run(cls, gpu: list = [0], split_size=1, AVAILABLE_SERVER_TYPES="ALL", AVAILABLE_MODELS="ALL", GPU_TYPE="", other_name="", CPU_IMAGE_TYPE="NV"):
        # GPU_TYPE 必传参数验证
        if not GPU_TYPE or GPU_TYPE.strip() == "":
            raise ValueError("GPU_TYPE is required and cannot be empty")
            
        global BASE_PORT
        containers = {}
        for idx, data in enumerate([gpu[i:i+split_size] for i in range(0, len(gpu), split_size)]):
            BASE_PORT += 8
            NODE_GPU_TOTAL = ",".join(map(str, data))
            container = dict(cls.default_container)
            
            # 根据全局mount_code变量决定是否挂载代码
            if cls.mount_code:
                container["volumes"] = cls.base_volumes + cls.code_volume
            else:
                container["volumes"] = cls.base_volumes.copy()
                
            environment = dict(container["environment"])
            environment.update(
                {"NODE_GPU_TOTAL": NODE_GPU_TOTAL,
                 "LABEL": f"lm-server-{int(time.time())}-{random.randint(0, int(time.time()))}",
                 "LM_SERVER_BASE_PORT": BASE_PORT,
                 "AVAILABLE_SERVER_TYPES": AVAILABLE_SERVER_TYPES,
                 "AVAILABLE_MODELS": AVAILABLE_MODELS,
                 "GPU_TYPE": GPU_TYPE
                 })
            if GPU_TYPE == "CPU":
                environment.update({"CPU_IMAGE_TYPE": CPU_IMAGE_TYPE})
            container.update(
                {"environment": environment, "shm_size": f"8gb"})
            containers[f"scheduler-{NODE_GPU_TOTAL.replace(',','_')}-of-{idx}{other_name}"] = container
        
        service = dict(cls.default)
        service.update({"services": containers})
        yaml_data = yaml.dump(service)
        
        # 根据 GPU_TYPE 和卡数生成目录名
        card_count = len(gpu)
        generate_dirs = [f"auto_openai_{GPU_TYPE}_{card_count}card"]
            
        for generate_dir in generate_dirs:
            os.makedirs(generate_dir, exist_ok=True)
            # 将生成的目录添加到跟踪集合中
            cls.generated_dirs.add(generate_dir)
            # 复制脚本文件到目录
            cls.copy_scripts_to_directory(generate_dir)
            # 生成主服务的 yaml 文件
            with open(cls.yaml_filename.format(generate_dir=generate_dir, gpu="-".join(map(str, gpu)), split_size=split_size, other_name=other_name), 'w') as file:
                file.write(yaml_data)
            # 生成单独的 embedding,rerank yaml 文件
            cls.generate_embedding_rerank_yaml(generate_dir)
        return service
    
    @classmethod
    def generate_master_node(cls):
        # 准备基础配置
        volumes = cls.base_volumes  # 只使用 conf 目录挂载
        
        # 如果需要挂载代码，添加代码卷
        if cls.mount_code:
            volumes.extend(cls.code_volume)
            
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
                    "volumes": volumes,
                }
            }
        }
        os.makedirs("auto_openai_master", exist_ok=True)
        with open("auto_openai_master/docker-compose.yml", "w") as file:
            yaml.dump(default, file)

    @classmethod
    def generate_node_pull_script(cls):
        # 为所有生成的目录创建 pull.sh 脚本
        for generate_dir in cls.generated_dirs:
            os.makedirs(generate_dir, exist_ok=True)
            with open(f"{generate_dir}/pull.sh", "w") as file:
                file.write(f"docker pull {image}")
    
    @classmethod
    def execute_echo_scripts(cls):
        """在所有生成的目录中执行 echo.sh 脚本"""
        for generate_dir in cls.generated_dirs:
            if os.path.exists(f"{generate_dir}/echo.sh"):
                print(f"Executing echo.sh in {generate_dir}")
                os.system(f"cd {generate_dir} && bash echo.sh")
            else:
                print(f"echo.sh not found in {generate_dir}, skipping...")
                
                
    @classmethod
    def clear_generated_dirs(cls, GPU_TYPE):
        import glob
        import os
        for dir_to_remove in glob.glob(f"auto_openai_{GPU_TYPE}_*card"):
            if os.path.exists(dir_to_remove):
                shutil.rmtree(dir_to_remove, ignore_errors=True)

# 设置是否挂载代码
Gen.mount_code = True  # 可以在这里全局控制是否挂载代码

# size 是指一个实例挂载得卡数
for GPU_TYPE in GPU_TYPE_LIST:
    # 清除 
    Gen.clear_generated_dirs(GPU_TYPE)
    if GPU_TYPE == "NV-A100-80G":
        Gen.run(gpu=[0, 1], split_size=1, GPU_TYPE=GPU_TYPE)
    else:
        Gen.run(gpu=[0, 1, 2, 3], split_size=1, GPU_TYPE=GPU_TYPE)
        Gen.run(gpu=[0, 1, 2, 3], split_size=2, GPU_TYPE=GPU_TYPE)
        Gen.run(gpu=[0, 1, 2, 3], split_size=4, GPU_TYPE=GPU_TYPE)
Gen.generate_master_node()
Gen.generate_node_pull_script()

# 自动在所有生成的目录中执行 echo.sh 脚本
Gen.execute_echo_scripts()