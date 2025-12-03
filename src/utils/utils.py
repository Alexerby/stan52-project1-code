import time
import functools

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
