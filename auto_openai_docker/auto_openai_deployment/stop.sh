# down
docker-compose -f scheduler-0-1-2-3-4-5-6-7-of-1-docker-compose.yml down
docker-compose -f scheduler-0-1-2-3-4-5-6-7-of-2-docker-compose.yml down
docker-compose -f scheduler-0-1-2-3-4-5-6-7-of-4-docker-compose.yml down
docker-compose -f scheduler-0-1-2-3-4-5-6-7-of-8-docker-compose.yml down
docker-compose -f scheduler-0-1-2-3-of-1-docker-compose.yml down
docker-compose -f scheduler-0-1-2-3-of-2-docker-compose.yml down
docker-compose -f scheduler-0-1-2-3-of-4-docker-compose.yml down
docker-compose -f scheduler-0-1-2-of-1-docker-compose.yml down
docker-compose -f scheduler-0-1-of-1-docker-compose.yml down
docker-compose -f scheduler-0-of-1-Qwen2.5-Coder-32B-Instruct-GPTQ-Int4-docker-compose.yml down
docker-compose -f scheduler-0-of-1-Wan2.1-TextToVideo-docker-compose.yml down
docker-compose -f scheduler-100-100-of-1-docker-compose.yml down
docker-compose -f scheduler-1-1-1-1-of-1-bge-base-zh-v1.5-docker-compose.yml down
docker-compose -f scheduler-1-1-1-1-of-1-bge-reranker-v2-m3-docker-compose.yml down
docker-compose -f scheduler-1-2-of-1-anyone-docker-compose.yml down
docker-compose -f scheduler-1-of-1-DeepSeek-R1-Distill-Qwen-14B-docker-compose.yml down
docker-compose -f scheduler-2-3-of-1-anyone-docker-compose.yml down
docker-compose -f scheduler-2-of-1-Qwen2.5-VL-7B-Instruct-docker-compose.yml down
docker-compose -f scheduler-3-3-3-3-of-1-docker-compose.yml down
docker-compose -f scheduler-3-3-of-1-Kolors-docker-compose.yml down
docker-compose -f scheduler-4-4-4-4-of-1-docker-compose.yml down
for container in $(docker ps -a -q -f "label=auto_openai_all"); do    
    # 停止容器（如果正在运行）
    echo "正在停止容器... "
    docker stop $container
    
    # 删除容器
    echo "正在删除容器... "
    docker rm $container
    echo "===== 容器处理完成 ====="
done
