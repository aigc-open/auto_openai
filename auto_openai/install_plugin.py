import os
from pydantic import BaseModel
from enum import Enum
from fire import Fire


def run(name: str):
    print(name)

if __name__ == "__main__":
    Fire(run)
