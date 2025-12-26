# utils.py
import os

def ensure_instance_dirs(paths):
    for p in paths:
        if not os.path.exists(p):
            os.makedirs(p, exist_ok=True)