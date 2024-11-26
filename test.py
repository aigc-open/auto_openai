from auto_openai import project_path
import os
import sys
for py in os.listdir(os.path.join(project_path, 'web', 'tests')):
    py_ = os.path.join(project_path, 'web', 'tests', py)
    print("="*100)
    os.system(f"python3 {py_}")
