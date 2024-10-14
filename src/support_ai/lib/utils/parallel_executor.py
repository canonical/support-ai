from concurrent.futures import ThreadPoolExecutor
import functools
from typing import List, Any, Callable, Tuple


def run_fn_in_parallel(fn_args: List[Tuple[Any]], parallelism: int):
    results = []
    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        futures = [executor.submit(fn, args) for fn, args in fn_args]
        for future in futures:
            results.append(future.result())
    return results

def run_in_parallel(parallelism: int):
    def decorator(fn: Callable):
        @functools.wraps(fn)
        def wrapper(self, args_list: List[Tuple[Any]]):
            results = []
            with ThreadPoolExecutor(max_workers=parallelism) as executor:
                futures = [executor.submit(fn, self, args) for args in args_list]
                for future in futures:
                    results.append(future.result())
            return results
        return wrapper
    return decorator
