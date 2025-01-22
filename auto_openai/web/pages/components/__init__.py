import os

def read_file(file):
    if os.path.exists(file):
        with open(file, "r") as f:
            return f.read()
    else:
        return "努力开发中..."
