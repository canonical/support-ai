"""
This module provides utilities for running functions in parallel using threads.
"""

from concurrent.futures import ThreadPoolExecutor
import functools
from typing import List, Any, Callable, Tuple


def run_fn_in_parallel(fn_args: List[Tuple[Any]], parallelism: int):
    """
    Executes a list of function calls with their respective arguments
    in parallel.

    Args:
        fn_args: A list of tuples, each containing a function and its
                 arguments.
        parallelism: The maximum number of threads to use for parallel
                     execution.

    Returns:
        List[Any]: A list of results from the executed functions, in the order
                   of function calls.
    """
    results = []
    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        futures = [executor.submit(fn, args) for fn, args in fn_args]
        for future in futures:
            results.append(future.result())
    return results


def run_in_parallel(parallelism: int):
    """
    A decorator to enable parallel execution of a method within a class.

    Args:
        parallelism: The maximum number of threads to use for parallel
                     execution.

    Returns:
        Callable: A decorator that wraps a method to run it in parallel for a
                  list of arguments.
    """
    def decorator(fn: Callable):
        @functools.wraps(fn)
        def wrapper(self, args_list: List[Tuple[Any]]):
            results = []
            with ThreadPoolExecutor(max_workers=parallelism) as executor:
                futures = [executor.submit(fn, self, args)
                           for args in args_list]
                for future in futures:
                    results.append(future.result())
            return results
        return wrapper
    return decorator
