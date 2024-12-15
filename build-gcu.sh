image_name=registry.cn-shanghai.aliyuncs.com/zhph-server/auto_openai:0.1-3.2.109
docker build --network=host -t $image_name -f Dockerfile.gpu .
docker push $image_name