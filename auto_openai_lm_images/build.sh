image_name=harbor.uat.enflame.cc/library/enflame.cn
device=gcu
image_name=registry.cn-shanghai.aliyuncs.com/zhph-server
# device=gpu

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
    docker build -t "$image_name/$service:$device" -f "Dockerfile.$service.$device" .
    docker push "$image_name/$service:$device"
done
