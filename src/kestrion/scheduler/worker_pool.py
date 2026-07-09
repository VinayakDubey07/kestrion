"""
Bounded async worker pool.

Controls how many agent runs execute concurrently. Wraps asyncio.Semaphore
with a clean interface and integrates with the shared RateLimiter so that
rate-limit backoffs are respected by all concurrent workers uniformly.

Design: the pool itself is intentionally thin — it only enforces concurrency
bounds. The RateLimiter handles the "are we allowed to call the provider
right now?" question separately. Keeping these as two distinct concerns
makes each one independently testable and composable.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Coroutine, TypeVar

from .rate_limiter import RateLimiter

logger = logging.getLogger("kestrion.scheduler.worker_pool")

T = TypeVar("T")


class WorkerPool:
    """
    Bounded async worker pool with integrated rate limiting.

    Usage::

        pool = WorkerPool(max_workers=5, rate_limiter=limiter)

        result = await pool.submit(my_coroutine())
        # or run many at once:
        results = await pool.map([coro1, coro2, coro3])

    Parameters
    ----------
    max_workers:
        Maximum number of coroutines that can run concurrently.
        Defaults to 5.
    rate_limiter:
        Shared rate limiter. If None, no rate limiting is applied.
    """

    def __init__(
        self,
        max_workers: int = 5,
        rate_limiter: RateLimiter | None = None,
    ):
        self._semaphore = asyncio.Semaphore(max_workers)
        self._rate_limiter = rate_limiter
        self._max_workers = max_workers

    async def submit(
        self,
        coro: Coroutine[Any, Any, T],
        estimated_tokens: int = 1,
    ) -> T:
        """
        Run a coroutine inside the pool.

        Blocks until:
          1. A worker slot is available (respects max_workers), AND
          2. The rate limiter allows the request (respects RPM/TPM limits).

        Parameters
        ----------
        coro:
            The coroutine to run.
        estimated_tokens:
            Hint to the rate limiter about how many tokens this request
            will consume. Used only for TPM bucket accounting; ignored if
            no tokens_per_minute limit is configured.
        """
        # Acquire rate limiter FIRST (may sleep for backoff/bucket refill),
        # then acquire the concurrency slot. This ordering means we don't
        # hold a worker slot while waiting for rate-limit capacity.
        if self._rate_limiter is not None:
            await self._rate_limiter.acquire(tokens=estimated_tokens)

        async with self._semaphore:
            logger.debug("Worker slot acquired (estimated_tokens=%d)", estimated_tokens)
            try:
                result = await coro
                if self._rate_limiter is not None:
                    self._rate_limiter.reset_backoff()
                return result
            except Exception:
                # Don't reset backoff on failure — the caller (Pipeline)
                # decides whether to call on_rate_limited.
                raise

    async def map(
        self,
        coros: list[Coroutine[Any, Any, T]],
        estimated_tokens_each: int = 1,
    ) -> list[T]:
        """
        Run multiple coroutines concurrently, all subject to the pool's
        concurrency and rate-limit constraints.

        Returns results in the same order as the input list.
        Raises the first exception encountered (fail-fast by default).

        Parameters
        ----------
        coros:
            Coroutines to run concurrently.
        estimated_tokens_each:
            Token hint applied uniformly to every coroutine.
        """
        tasks = [
            asyncio.create_task(self.submit(coro, estimated_tokens_each))
            for coro in coros
        ]
        return list(await asyncio.gather(*tasks))

    @property
    def max_workers(self) -> int:
        return self._max_workers
