# 获取本目录的scheduler*.yml文件，获取其名字
scheduler_file=$(ls scheduler*.yml | head -n 100)


# 循环启动
echo "# down" > stop.sh
for file in $scheduler_file
do
    echo "docker-compose -f $file down" >> stop.sh
done
echo "# up" > start.sh
for file in $scheduler_file
do
    echo "# docker-compose -f $file up -d" >> start.sh
done
echo "Done