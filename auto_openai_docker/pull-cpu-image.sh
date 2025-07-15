image_name=registry.cn-shanghai.aliyuncs.com/zhph-server
device=gpu

# List of services to build
services=(
    # "comfyui"
    "embedding-server"
    # "funasr-server"
    "llm-transformer-server"
    # "maskgct"
    "rerank-server"
    # "vllm"
    # "vllm-glm"
    # "vllm-qwen2-vl"
    # "vllm-qwen25-vl"
    # "webui"
    "diffusers-image-server"
    "diffusers-video-server"
)

# Loop through each service and build the Docker image
for service in "${services[@]}"; do
    docker pull "$image_name/$service:$device"
done

services=(
    "ollama"
)
device=cpu

for service in "${services[@]}"; do
    docker pull "$image_name/$service:$device"
done