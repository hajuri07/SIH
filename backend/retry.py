"""
Exponential Backoff Retry Utility for Robust Async I/O (Database & Kafka).
Provides automatic retries with exponential backoff for transient failures.
"""

import asyncio
import functools
import logging
from typing import Callable, Any, Tuple, Type

logger = logging.getLogger("vayupulse.retry")


def async_retry(
    max_retries: int = 3,
    initial_delay: float = 0.1,
    backoff_factor: float = 2.0,
    max_delay: float = 2.0,
    retry_exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    Decorator for retrying asynchronous coroutines with exponential backoff.
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception = None

            for attempt in range(1, max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retry_exceptions as exc:
                    last_exception = exc
                    if attempt == max_retries:
                        logger.error(
                            f"Function '{func.__name__}' failed after {max_retries} attempts: {exc}"
                        )
                        raise exc

                    logger.warning(
                        f"Attempt {attempt}/{max_retries} for '{func.__name__}' failed ({exc}). Retrying in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)

            raise last_exception
        return wrapper
    return decorator
