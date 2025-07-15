# 获取本目录的scheduler*.yml文件，获取其名字
scheduler_file=$(ls scheduler*.yml | head -n 100)


# 循环启动
echo "# down" > stop.sh
for file in $scheduler_file
do
    echo "docker-compose -f $file down" >> stop.sh
done
echo "# up" > start.sh
first_file=true
for file in $scheduler_file
do
    if [ "$first_file" = true ]; then
        echo "docker-compose -f $file up -d" >> start.sh
        first_file=false
    else
        echo "# docker-compose -f $file up -d" >> start.sh
    fi
done
cat << 'EOF' >> stop.sh
for container in $(docker ps -a -q -f "label=auto_openai_all"); do    
    # 停止容器（如果正在运行）
    echo "正在停止容器... "
    docker stop $container
    
    # 删除容器
    echo "正在删除容器... "
    docker rm $container
    echo "===== 容器处理完成 ====="
done
EOF
echo "Done"