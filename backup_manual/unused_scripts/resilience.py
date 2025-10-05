"""
Resilience Layer

Rate limiting, retry logic, and circuit breakers for external API calls.
"""

import time
import asyncio
import logging
from typing import Callable, Any, Optional
from functools import wraps
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter for API calls.

    Ensures compliance with API rate limits by throttling requests.
    """

    def __init__(self, calls: int, period: int):
        """
        Initialize rate limiter.

        Args:
            calls: Maximum number of calls allowed
            period: Time period in seconds
        """
        self.calls = calls
        self.period = period
        self.timestamps = deque()

    def _clean_old_timestamps(self):
        """Remove timestamps outside the current period."""
        now = time.time()
        cutoff = now - self.period

        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.popleft()

    def acquire(self):
        """
        Acquire permission to make a call.

        Blocks if rate limit is exceeded.
        """
        self._clean_old_timestamps()

        if len(self.timestamps) >= self.calls:
            # Calculate wait time
            oldest = self.timestamps[0]
            wait_time = self.period - (time.time() - oldest)

            if wait_time > 0:
                logger.debug(f"Rate limit reached, waiting {wait_time:.2f}s")
                time.sleep(wait_time)
                self._clean_old_timestamps()

        self.timestamps.append(time.time())

    async def acquire_async(self):
        """Async version of acquire."""
        self._clean_old_timestamps()

        if len(self.timestamps) >= self.calls:
            oldest = self.timestamps[0]
            wait_time = self.period - (time.time() - oldest)

            if wait_time > 0:
                logger.debug(f"Rate limit reached, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                self._clean_old_timestamps()

        self.timestamps.append(time.time())

    def throttle(self, func: Callable) -> Callable:
        """
        Decorator to rate-limit a function.

        Usage:
            rate_limiter = RateLimiter(calls=10, period=60)

            @rate_limiter.throttle
            def api_call():
                ...
        """
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            await self.acquire_async()
            return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            self.acquire()
            return func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper


class CircuitBreaker:
    """
    Circuit breaker pattern for fault tolerance.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failure threshold exceeded, requests fail fast
    - HALF_OPEN: Testing if service recovered
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type to catch
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If circuit is OPEN or function fails
        """
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise Exception(
                    f"Circuit breaker OPEN. "
                    f"Recovery attempt in {self._time_until_recovery():.0f}s"
                )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result

        except self.expected_exception as e:
            self._on_failure()
            raise e

    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """Async version of call."""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise Exception(
                    f"Circuit breaker OPEN. "
                    f"Recovery attempt in {self._time_until_recovery():.0f}s"
                )

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result

        except self.expected_exception as e:
            self._on_failure()
            raise e

    def _on_success(self):
        """Handle successful call."""
        if self.state == "HALF_OPEN":
            logger.info("Circuit breaker recovery successful, closing circuit")
            self.state = "CLOSED"
            self.failure_count = 0

    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            logger.warning(
                f"Circuit breaker opening after {self.failure_count} failures"
            )
            self.state = "OPEN"

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True

        return datetime.now() - self.last_failure_time > timedelta(
            seconds=self.recovery_timeout
        )

    def _time_until_recovery(self) -> float:
        """Calculate seconds until recovery attempt."""
        if self.last_failure_time is None:
            return 0

        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return max(0, self.recovery_timeout - elapsed)


def with_retry(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retry logic with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        backoff_factor: Multiplier for backoff delay
        exceptions: Tuple of exceptions to catch

    Usage:
        @with_retry(max_attempts=3, backoff_factor=2.0)
        async def fetch_data():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        delay = backoff_factor ** attempt
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}), "
                            f"retrying in {delay:.1f}s: {str(e)}"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {str(e)}"
                        )

            raise last_exception

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        delay = backoff_factor ** attempt
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}), "
                            f"retrying in {delay:.1f}s: {str(e)}"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {str(e)}"
                        )

            raise last_exception

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class TimeoutManager:
    """
    Timeout manager for long-running operations.
    """

    @staticmethod
    async def with_timeout(coro, timeout: float):
        """
        Execute coroutine with timeout.

        Args:
            coro: Coroutine to execute
            timeout: Timeout in seconds

        Returns:
            Coroutine result

        Raises:
            asyncio.TimeoutError: If timeout exceeded
        """
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            logger.error(f"Operation timed out after {timeout}s")
            raise


# Pre-configured rate limiters for common APIs
SEC_EDGAR_RATE_LIMITER = RateLimiter(calls=10, period=1)  # 10 req/sec
NEWS_API_RATE_LIMITER = RateLimiter(calls=100, period=86400)  # 100 req/day (free tier)
FINNHUB_RATE_LIMITER = RateLimiter(calls=60, period=60)  # 60 req/min

# Circuit breakers for external services
EDGAR_CIRCUIT_BREAKER = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=60,
    expected_exception=Exception
)

NEWS_API_CIRCUIT_BREAKER = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=60,
    expected_exception=Exception
)

FINNHUB_CIRCUIT_BREAKER = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=60,
    expected_exception=Exception
)
