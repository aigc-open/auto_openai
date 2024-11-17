import logging
import time
from time import strftime, localtime
from typing import *
import json
import redis
from redis.cluster import ClusterNode
from redis.cluster import RedisCluster
from loguru import logger


class RedisClientBase:

    def __init__(self, config: dict):
        """
        Keyword arguments:
        config -- dict: eg., {
            "address": "10.9.115.78:7000,10.9.115.78:7001,10.9.115.78:7002,10.9.115.78:7003,10.9.115.78:7004,10.9.115.78:7005",
            "username": None,
            "password": None,
            "cluster": False
        }
        """

        if config.get("cluster"):
            _startup_nodes = [ClusterNode(*addr.split(":"))
                              for addr in config["address"].split(",")]
            super().__init__(startup_nodes=_startup_nodes,
                             username=config["username"], password=config["password"])
        else:
            host, port = config["address"].split(":")
            super().__init__(host=host, port=port,
                             username=config["username"], password=config["password"])

    def __del__(self: redis.Redis):
        self.close()


class RedisClient:
    def __new__(cls, config: dict) -> redis.Redis:
        if config.get("cluster"):
            cls = type("RedisClient", (RedisClientBase, RedisCluster),
                       dict(vars(RedisClientBase)))
            cls_instance = cls(config)
            return cls_instance
        else:
            cls = type("RedisClient", (RedisClientBase, redis.Redis),
                       dict(vars(RedisClientBase)))
            cls_instance = cls(config)
            return cls_instance
