docker pull registry.cn-shanghai.aliyuncs.com/zhph-server/minio-mc:latest

docker run --rm --network auto_openai_network \
  --entrypoint sh registry.cn-shanghai.aliyuncs.com/zhph-server/minio-mc:latest \
  -c "mc alias set myminio http://minio:9001 admin admin123 && mc ls myminio"


docker run --rm --network auto_openai_network \
  --entrypoint sh registry.cn-shanghai.aliyuncs.com/zhph-server/minio-mc:latest \
  -c "mc alias set myminio http://minio:9001 admin admin123 && mc mb myminio/api-platform"


docker run --rm --network auto_openai_network \
  --entrypoint sh registry.cn-shanghai.aliyuncs.com/zhph-server/minio-mc:latest \
  -c "mc alias set myminio http://minio:9001 admin admin123 && mc anonymous set public myminio/api-platform"