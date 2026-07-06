"""
Async token-bucket rate limiter with exponential backoff + jitter.

Design principles:
  - Shared across all workers in a Pipeline — when one worker hits a 429,
    ALL workers back off together. This prevents the "thundering herd
    re-hit" where 5 concurrent tasks all get rate-limited, then all retry
    at the same instant and get rate-limited again.
  - Token-bucket (not fixed-window) — smooths out bursts without
    unnecessarily blocking work that would fit within the provider's limits.
  - Jitter on backoff — spreads retries across a range rather than
    synchronizing them, which further reduces re-hit probability when
    multiple clients share a provider quota.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass

logger = logging.getLogger("kestrion.scheduler.rate_limiter")


@dataclass
class RateLimiterConfig:
    """
    Configuration for rate limiting.

    Set whichever limits your provider enforces; unset limits are ignored.
    Common presets:
      - Anthropic claude-3-haiku: 1_000 RPM, 100_000 TPM
      - OpenAI gpt-4o-mini:       500 RPM,  200_000 TPM
      - Ollama (local):            None (unlimited — but keep workers low)
    """
    requests_per_minute: float | None = None   # e.g. 60.0
    tokens_per_minute: float | None = None     # e.g. 100_000.0
    max_backoff_seconds: float = 60.0          # cap on exponential backoff
    base_backoff_seconds: float = 1.0          # starting backoff on first 429


class RateLimiter:
    """
    Async token-bucket rate limiter with shared exponential backoff.

    Usage::

        limiter = RateLimiter(RateLimiterConfig(requests_per_minute=60))

        # Before each LLM call in your worker:
        await limiter.acquire(tokens=500)        # blocks if bucket is empty

        # If the provider returns a 429:
        await limiter.on_rate_limited(retry_after=5.0)
    """

    def __init__(self, config: RateLimiterConfig | None = None):
        self._config = config or RateLimiterConfig()

        # Token-bucket state — one bucket per limit dimension
        self._req_tokens: float = self._config.requests_per_minute or float("inf")
        self._tok_tokens: float = self._config.tokens_per_minute or float("inf")
        self._last_refill: float = time.monotonic()

        # Backoff state — shared across all concurrent callers
        self._backoff_until: float = 0.0          # monotonic time to unblock at
        self._consecutive_429s: int = 0
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def acquire(self, tokens: int = 1) -> None:
        """
        Wait until the rate-limiter allows this request to proceed.

        Parameters
        ----------
        tokens:
            Estimated output tokens for this request (used for TPM bucket).
            If you don't know yet, pass 1 — the bucket is approximate anyway.
        """
        while True:
            async with self._lock:
                now = time.monotonic()

                # --- Global backoff (from 429 responses) ---
                remaining_backoff = self._backoff_until - now
                if remaining_backoff > 0:
                    # Release the lock before sleeping so other waiters can
                    # see the backoff state without deadlocking.
                    pass
                else:
                    self._refill(now)
                    if self._req_tokens >= 1.0 and self._tok_tokens >= tokens:
                        self._req_tokens -= 1.0
                        self._tok_tokens = max(0.0, self._tok_tokens - tokens)
                        return  # acquired — proceed
                    remaining_backoff = 0.0  # bucket empty, not a 429 backoff

            # Sleep outside the lock (either backoff or bucket empty)
            wait = remaining_backoff if remaining_backoff > 0 else self._refill_delay(tokens)
            logger.debug("Rate limiter waiting %.2fs", wait)
            await asyncio.sleep(wait)

    async def on_rate_limited(self, retry_after: float | None = None) -> None:
        """
        Call this when the provider returns a 429 or a rate-limit error.

        Triggers a shared exponential backoff that all concurrent acquire()
        callers will respect — they will sleep until the backoff expires.

        Parameters
        ----------
        retry_after:
            If the provider sent a Retry-After header, pass that value
            (in seconds) here. If provided, it overrides the computed
            exponential backoff — the provider knows best.
        """
        async with self._lock:
            self._consecutive_429s += 1
            if retry_after is not None:
                backoff = retry_after
            else:
                # Exponential backoff with +/-25% jitter
                base = self._config.base_backoff_seconds * (2 ** (self._consecutive_429s - 1))
                capped = min(base, self._config.max_backoff_seconds)
                jitter = capped * 0.25 * (random.random() * 2 - 1)  # +/-25%
                backoff = max(0.1, capped + jitter)

            self._backoff_until = time.monotonic() + backoff
            logger.warning(
                "Rate limited (429 #%d). Backing off for %.1fs.",
                self._consecutive_429s,
                backoff,
            )

    def reset_backoff(self) -> None:
        """Call after a successful request to reset the consecutive-429 counter."""
        self._consecutive_429s = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refill(self, now: float) -> None:
        """Refill both token buckets based on elapsed time."""
        elapsed = now - self._last_refill
        self._last_refill = now

        if self._config.requests_per_minute is not None:
            self._req_tokens = min(
                self._config.requests_per_minute,
                self._req_tokens + elapsed * self._config.requests_per_minute / 60.0,
            )
        if self._config.tokens_per_minute is not None:
            self._tok_tokens = min(
                self._config.tokens_per_minute,
                self._tok_tokens + elapsed * self._config.tokens_per_minute / 60.0,
            )

    def _refill_delay(self, tokens_needed: int) -> float:
        """
        How long (seconds) until the bucket will have enough capacity for
        this request, given the current refill rate.
        """
        delays = []
        if self._config.requests_per_minute is not None and self._req_tokens < 1.0:
            deficit = 1.0 - self._req_tokens
            delays.append(deficit / (self._config.requests_per_minute / 60.0))
        if self._config.tokens_per_minute is not None and self._tok_tokens < tokens_needed:
            deficit = tokens_needed - self._tok_tokens
            delays.append(deficit / (self._config.tokens_per_minute / 60.0))
        return max(delays) if delays else 0.05  # minimum poll interval
