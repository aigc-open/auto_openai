image_name=harbor.uat.enflame.cc/library/enflame.cn
# image_name=registry.cn-shanghai.aliyuncs.com/zhph-server
device=gcu

# List of services to build
services=(
    "comfyui"
    "diffusers-server"
    "embedding-server"
    "funasr-server"
    "llm-transformer-server"
    "maskgct"
    "rerank-server"
    "vllm"
    "webui"
)

# Loop through each service and build the Docker image
for service in "${services[@]}"; do
    docker pull "$image_name/$service:$device"
done

docker pull "$image_name/auto_openai:0.2"