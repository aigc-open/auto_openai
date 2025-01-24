# down
docker-compose -f scheduler-0-1-2-3-4-5-6-7-of-1-docker-compose.yml down
docker-compose -f scheduler-0-1-2-3-4-5-6-7-of-2-docker-compose.yml down
docker-compose -f scheduler-0-1-2-3-4-5-6-7-of-4-docker-compose.yml down
docker-compose -f scheduler-0-1-2-3-4-5-6-7-of-8-docker-compose.yml down
docker-compose -f scheduler-0-1-2-3-of-1-docker-compose.yml down
docker-compose -f scheduler-0-1-2-3-of-2-docker-compose.yml down
docker-compose -f scheduler-0-1-2-3-of-4-docker-compose.yml down
docker-compose -f scheduler-0-1-2-of-1-docker-compose.yml down
docker-compose -f scheduler-3-3-3-3-of-1-docker-compose.yml down
docker-compose -f scheduler-4-4-4-4-of-1-docker-compose.yml down
docker-compose -f scheduler-4-5-of-2-docker-compose.yml down

for container in $(docker ps -a -q -f "label=auto_openai_all"); do    
    # 停止容器（如果正在运行）
    echo "正在停止容器... "
    docker stop $container
    
    # 删除容器
    echo "正在删除容器... "
    docker rm $container
    echo "===== 容器处理完成 ====="
done
