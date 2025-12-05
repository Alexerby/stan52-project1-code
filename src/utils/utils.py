import time
import functools
from pathlib import Path
import joblib
import sys

def time_execution(func):
    """
    Decorator to measure and print the execution time of a function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        elapsed = end_time - start_time
        
        # Format for readability: Minutes if > 60s, else seconds
        if elapsed > 60:
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            print(f"⏱️  '{func.__name__}' finished in {minutes}m {seconds:.2f}s")
        else:
            print(f"⏱️  '{func.__name__}' finished in {elapsed:.4f}s")
            
        return result
    return wrapper


def load_saved_model(model_name: str, model_dir: Path):
    model_path = model_dir / f"{model_name}.joblib"
    
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        print(f"Have you trained it yet? Run: python3 -m src.training.train_mnist --model {model_name}")
        sys.exit(1)
        
    print(f"Loading model: {model_path}")
    return joblib.load(model_path)
