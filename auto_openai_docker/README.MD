# 看看哪个盘多
> 如果系统盘够大，则直接使用 `mkdir -p /root/share_models`
```bash
export DISK=/mnt/sdb
mkdir -p $DISK/data
ln -s $DISK/data /
mkdir -p $DISK/share_models
ln -s $DISK/share_models /root/ 
```

# 模型下载
- 参考下载：(`bash auto_openai_docker/model-install.sh`)，可以根据通过平台生成命令


# 中间件部署
- (`auto_openai_docker/auto_openai_middleware`)
- 创建网络 `bash create_network.sh`
- 拉取镜像 `docker-compose pull -f docker-compose.yml`
- 部署 `docker-compose -f docker-compose.yml up -d`
- 创建S3桶 `bash create_bucket.sh`

# 同步配置
- (`auto_openai_docker`)
- 修改中间件配置 `auto_openai_docker/conf/config.yaml` 参考实际部署中间件的地址修改
- 同步配置 `bash sync_conf.sh`

# 主节点部署
- (`auto_openai_docker/auto_openai_master_node/docker-compose.yml`)
- 拉取镜像 `docker-compose pull`
- 部署 `docker-compose up -d`

# gpu节点(模型调度器)
- (`auto_openai_docker/auto_openai_gpu_node`)
- 拉取镜像 `bash pull.sh`
- 根据当前卡数选择不同的调度器 `vim start.sh` 打开不同的配置
- 部署 `bash start.sh`
- 停止 `bash stop.sh`

# gcu节点(模型调度器)
- (`auto_openai_docker/auto_openai_gcu_node`)
- 拉取镜像 `bash pull.sh`
- 根据当前卡数选择不同的调度器 `vim start.sh` 打开不同的配置
- 部署 `bash start.sh`
- 停止 `bash stop.sh`
