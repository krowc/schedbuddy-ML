"""Decorator to automatically log runtime"""

import time 
from datetime import datetime
import functools

def log_time(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start

        with open("logs.txt", "a") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {func.__name__}: {elapsed_time:.2f}s\n")
        
        print(f"{func.__name__} took {elapsed_time:.2f}s")
        return result
    return wrapper