"""Retry on transient errors, shared by the OpenRouter, HasData and akta.pro clients."""

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from src.config import RETRY_ATTEMPTS, RETRY_BASE_DELAY, RETRY_MAX_DELAY
from src.logger import logger

T = TypeVar("T")


def describe_failed_attempt(attempt: int, error: Exception, retry_in_s: float | None) -> dict:
    """One `failed_attempts` entry: the error plus, when it carries a server
    reply, that reply's status code and body (headers are left out).
    retry_in_s is None when no retry followed."""
    status_code = body = None
    if isinstance(getattr(error, "status_code", None), int):  # OpenRouter SDK errors
        status_code, body = error.status_code, getattr(error, "body", None)
    elif (response := getattr(error, "response", None)) is not None:  # httpx2.HTTPStatusError
        status_code, body = response.status_code, response.text
    return {
        "attempt": attempt,
        "error_type": type(error).__name__,
        "error": str(error),
        "status_code": status_code,
        "body": body,
        "retry_in_s": retry_in_s,
    }


async def retry_transient(
    call: Callable[[], Awaitable[T]],
    retryable: tuple[type[Exception], ...],
    tag: str,
    failed_attempts: list[dict] | None = None,
    attempts: int = RETRY_ATTEMPTS,
    base_delay: float = RETRY_BASE_DELAY,
    max_delay: float = RETRY_MAX_DELAY,
) -> T:
    """Await `call()`, retrying when it raises one of `retryable`: `attempts`
    tries total, full-jitter exponential backoff (`base_delay`, capped at
    `max_delay`). The last attempt's error, or any non-retryable one,
    propagates. Every failed attempt,
    retried or not, is appended to `failed_attempts` when given
    (describe_failed_attempt)."""
    attempt = 1
    while True:
        try:
            return await call()
        except Exception as e:
            if not isinstance(e, retryable) or attempt == attempts:
                if failed_attempts is not None:
                    failed_attempts.append(describe_failed_attempt(attempt, e, None))
                raise
            ceiling = min(max_delay, base_delay * 2 ** (attempt - 1))
            delay = random.uniform(0, ceiling)
            if failed_attempts is not None:
                failed_attempts.append(describe_failed_attempt(attempt, e, round(delay, 1)))
            logger.warning(
                "%s transient error (attempt %d/%d): %s: %s; retrying in %.1fs",
                tag, attempt, attempts, type(e).__name__, e, delay,
            )
            await asyncio.sleep(delay)
            attempt += 1
