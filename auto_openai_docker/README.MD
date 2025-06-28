# 看看哪个盘多
```bash
export DISK=/mnt/sdb
mkdir -p $DISK/data
ln -s $DISK/data /
mkdir -p $DISK/share_models
ln -s $DISK/share_models /root/
```