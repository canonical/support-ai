"""
This module provides a decorator for caching function results with a time-based
expiration using Python's built-in LRU cache.
"""

from functools import lru_cache, wraps
from datetime import datetime, timedelta


def timed_lru_cache(seconds=60*60, maxsize=32):
    """
    A decorator that applies an LRU cache to a function with a
    time-based expiration.
    """

    def wrapper_cache(func):
        """
        Wraps the target function with LRU caching and expiration logic.

        Args:
            func: The target function to be wrapped with caching.

        Returns:
            function: The wrapped function with caching and expiration
                      behavior.
        """
        func = lru_cache(maxsize=maxsize)(func)
        func.lifetime = timedelta(seconds=seconds)
        func.expiration = datetime.utcnow() + func.lifetime

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            """
            Executes the wrapped function, clearing the cache if expired.

            Args:
                *args: Positional arguments passed to the wrapped function.
                **kwargs: Keyword arguments passed to the wrapped function.

            Returns:
                Any: The result of the wrapped function.
            """
            if datetime.utcnow() >= func.expiration:
                func.cache_clear()
                func.expiration = datetime.utcnow() + func.lifetime
            return func(*args, **kwargs)

        return wrapped_func
    return wrapper_cache
