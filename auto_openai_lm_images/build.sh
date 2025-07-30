image_name=harbor.uat.enflame.cc/library/enflame.cn
device=gcu
image_name=registry.cn-shanghai.aliyuncs.com/zhph-server
# device=gpu

# List of services to build
services=(
    # "comfyui"
    # "embedding-server"
    # "funasr-server"
    # "llm-transformer-server"
    # "maskgct"
    # "rerank-server"
    # "vllm-glm"
    "vllm-qwen2-vl"
    "vllm-qwen25-vl"
    # "webui"
    # "vllm"
    # "wan21"
    # "diffusers-image-server"
    # "diffusers-video-server"
)

# Loop through each service and build the Docker image
for service in "${services[@]}"; do
    docker build -t "$image_name/$service:$device" -f "Dockerfile.$service.$device" .
    docker push "$image_name/$service:$device"
done


# services=(
#     "ollama"
# )
# device=cpu
# for service in "${services[@]}"; do
#     docker build -t "$image_name/$service:$device" -f "Dockerfile.$service.$device" .
#     docker push "$image_name/$service:$device"
# done