"""
Rate Limiting Core Algorithms

Implements various rate limiting algorithms:
- Token Bucket Algorithm
- Sliding Window Rate Limiting
- Circuit Breaker Pattern
- Adaptive Rate Limiting
"""

import time
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import threading
import weakref

from .exceptions import RateLimitExceededException, CircuitBreakerOpenException


class RateLimitStrategy(Enum):
    """Rate limiting strategy types"""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window" 
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimitStatus:
    """Rate limit status information"""
    allowed: bool
    remaining: int
    reset_time: float
    limit: int
    algorithm: str
    retry_after: float = 0.0


class RateLimiter(ABC):
    """
    Abstract base class for rate limiters
    
    All rate limiting algorithms must implement this interface
    """
    
    def __init__(self, name: str, limit: int, window_seconds: float):
        self.name = name
        self.limit = limit
        self.window_seconds = window_seconds
        self.last_check = time.time()
        self._stats = {
            "total_requests": 0,
            "allowed_requests": 0,
            "rejected_requests": 0,
            "avg_wait_time": 0.0,
            "peak_usage": 0
        }
        self._lock = threading.RLock()
    
    @abstractmethod
    async def acquire(self, tokens: int = 1) -> RateLimitStatus:
        """
        Acquire permission to make a request
        
        Args:
            tokens: Number of tokens to acquire (default: 1)
            
        Returns:
            RateLimitStatus: Status of the request
        """
        pass
    
    @abstractmethod
    async def wait_for_tokens(self, tokens: int = 1, timeout: float = None) -> bool:
        """
        Wait for tokens to become available
        
        Args:
            tokens: Number of tokens needed
            timeout: Maximum time to wait (None for no timeout)
            
        Returns:
            bool: True if tokens acquired, False if timeout
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current rate limiter statistics"""
        with self._lock:
            return self._stats.copy()
    
    def reset_stats(self):
        """Reset rate limiter statistics"""
        with self._lock:
            self._stats = {
                "total_requests": 0,
                "allowed_requests": 0,
                "rejected_requests": 0,
                "avg_wait_time": 0.0,
                "peak_usage": 0
            }


class TokenBucketRateLimiter(RateLimiter):
    """
    Token Bucket Rate Limiter
    
    Allows burst traffic up to bucket capacity while maintaining average rate.
    Tokens are added at a fixed rate. Each request consumes tokens.
    """
    
    def __init__(self, name: str, limit: int, window_seconds: float, burst_multiplier: float = 2.0):
        super().__init__(name, limit, window_seconds)
        
        # Calculate token generation rate
        self.rate_per_second = limit / window_seconds
        
        # Initialize bucket with full capacity (allows initial burst)
        self.burst_multiplier = burst_multiplier
        self.burst_capacity = int(limit * burst_multiplier)
        self.tokens = float(self.burst_capacity)
        
        # Track last refill time
        self.last_refill = time.time()
    
    async def acquire(self, tokens: int = 1) -> RateLimitStatus:
        """Acquire tokens from bucket"""
        with self._lock:
            self._stats["total_requests"] += 1
            
            # Refill tokens based on time passed
            now = time.time()
            time_passed = now - self.last_refill
            tokens_to_add = time_passed * self.rate_per_second
            
            # Add tokens, but cap at burst capacity
            self.tokens = min(self.burst_capacity, self.tokens + tokens_to_add)
            self.last_refill = now
            
            # Check if enough tokens available
            if self.tokens >= tokens:
                self.tokens -= tokens
                self._stats["allowed_requests"] += 1
                self._stats["peak_usage"] = max(self._stats["peak_usage"], self._stats["allowed_requests"])
                
                # Calculate retry after time
                tokens_needed = tokens
                if tokens_needed > self.tokens:
                    time_for_tokens = (tokens_needed - self.tokens) / self.rate_per_second
                    retry_after = time_for_tokens
                else:
                    retry_after = 0.0
                
                return RateLimitStatus(
                    allowed=True,
                    remaining=int(self.tokens),
                    reset_time=now + self.burst_capacity / self.rate_per_second,
                    limit=self.limit,
                    algorithm="token_bucket",
                    retry_after=retry_after
                )
            else:
                # Not enough tokens
                self._stats["rejected_requests"] += 1
                tokens_needed = tokens - self.tokens
                retry_after = tokens_needed / self.rate_per_second
                
                return RateLimitStatus(
                    allowed=False,
                    remaining=int(self.tokens),
                    reset_time=now + retry_after,
                    limit=self.limit,
                    algorithm="token_bucket",
                    retry_after=retry_after
                )
    
    async def wait_for_tokens(self, tokens: int = 1, timeout: float = None) -> bool:
        """Wait for tokens to become available"""
        with self._lock:
            start_time = time.time()
            
            while True:
                status = await self.acquire(tokens)
                if status.allowed:
                    return True
                
                # Check timeout
                if timeout is not None and (time.time() - start_time) >= timeout:
                    return False
                
                # Wait for tokens to be available
                wait_time = min(status.retry_after, 0.1)  # Wait in small increments
                await asyncio.sleep(wait_time)


class SlidingWindowRateLimiter(RateLimiter):
    """
    Sliding Window Rate Limiter
    
    Uses a sliding time window to track request counts.
    More accurate than fixed window but uses more memory.
    """
    
    def __init__(self, name: str, limit: int, window_seconds: float):
        super().__init__(name, limit, window_seconds)
        self.requests = deque()  # Store timestamps of requests
        self.current_count = 0
    
    async def acquire(self, tokens: int = 1) -> RateLimitStatus:
        """Check if request is allowed in sliding window"""
        with self._lock:
            self._stats["total_requests"] += 1
            now = time.time()
            cutoff = now - self.window_seconds
            
            # Remove old requests from deque
            while self.requests and self.requests[0] < cutoff:
                self.requests.popleft()
                self.current_count -= 1
            
            # Check if we can allow this request
            if self.current_count + tokens <= self.limit:
                # Allow request
                self.requests.extend([now] * tokens)
                self.current_count += tokens
                self._stats["allowed_requests"] += 1
                self._stats["peak_usage"] = max(self._stats["peak_usage"], self.current_count)
                
                # Estimate retry after (time until enough requests expire)
                requests_needed = self.current_count + tokens - self.limit
                if requests_needed > 0 and self.requests:
                    retry_after = max(0.1, self.window_seconds - (now - self.requests[0]))
                else:
                    retry_after = 0.0
                
                return RateLimitStatus(
                    allowed=True,
                    remaining=self.limit - self.current_count,
                    reset_time=now + self.window_seconds,
                    limit=self.limit,
                    algorithm="sliding_window",
                    retry_after=retry_after
                )
            else:
                # Rate limit exceeded
                self._stats["rejected_requests"] += 1
                
                # Calculate retry after based on oldest request
                if self.requests:
                    oldest_request = self.requests[0]
                    retry_after = max(0.1, oldest_request + self.window_seconds - now)
                else:
                    retry_after = self.window_seconds
                
                return RateLimitStatus(
                    allowed=False,
                    remaining=0,
                    reset_time=now + retry_after,
                    limit=self.limit,
                    algorithm="sliding_window",
                    retry_after=retry_after
                )
    
    async def wait_for_tokens(self, tokens: int = 1, timeout: float = None) -> bool:
        """Wait for sliding window to allow request"""
        start_time = time.time()
        
        while True:
            status = await self.acquire(tokens)
            if status.allowed:
                return True
            
            # Check timeout
            if timeout is not None and (time.time() - start_time) >= timeout:
                return False
            
            # Wait for window to reset
            await asyncio.sleep(min(status.retry_after, 0.1))


class CircuitBreakerRateLimiter(RateLimiter):
    """
    Circuit Breaker Rate Limiter
    
    Implements circuit breaker pattern to prevent cascading failures.
    States: CLOSED (normal), OPEN (blocked), HALF_OPEN (testing)
    """
    
    def __init__(self, name: str, limit: int, window_seconds: float, 
                 failure_threshold: int = 5, recovery_timeout: float = 60.0):
        super().__init__(name, limit, window_seconds)
        
        # Circuit breaker configuration
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
        # Statistics tracking
        self.success_count = 0
        self.consecutive_failures = 0
        
        # State transition timestamps
        self.state_change_time = time.time()
    
    async def acquire(self, tokens: int = 1) -> RateLimitStatus:
        """Acquire tokens with circuit breaker protection"""
        now = time.time()
        
        # State management
        if self.state == "OPEN":
            if (now - self.last_failure_time) >= self.recovery_timeout:
                self.state = "HALF_OPEN"
                self.state_change_time = now
                self.consecutive_failures = 0
            else:
                self._stats["rejected_requests"] += 1
                raise CircuitBreakerOpenException(
                    f"Circuit breaker is OPEN. Next retry at {self.last_failure_time + self.recovery_timeout}"
                )
        
        try:
            # Simulate request (in real implementation, this would call underlying rate limiter)
            # For now, we randomly determine success/failure for demonstration
            import random
            if random.random() < 0.9:  # 90% success rate
                await asyncio.sleep(0.001)  # Simulate processing time
                self._record_success(now)
                
                return RateLimitStatus(
                    allowed=True,
                    remaining=max(0, self.limit - 1),
                    reset_time=now + self.window_seconds,
                    limit=self.limit,
                    algorithm="circuit_breaker"
                )
            else:
                raise Exception("Simulated API failure")
                
        except Exception as e:
            self._record_failure(now)
            self._stats["rejected_requests"] += 1
            
            # Don't raise CircuitBreakerOpenException here, let the base rate limiter handle it
            return RateLimitStatus(
                allowed=False,
                remaining=0,
                reset_time=now + self.recovery_timeout,
                limit=self.limit,
                algorithm="circuit_breaker",
                retry_after=self.recovery_timeout
            )
    
    def _record_success(self, now: float):
        """Record successful request"""
        with self._lock:
            self._stats["total_requests"] += 1
            self._stats["allowed_requests"] += 1
            self.success_count += 1
            
            if self.state == "HALF_OPEN":
                # Transition to CLOSED after successful request in HALF_OPEN
                self.state = "CLOSED"
                self.state_change_time = now
                self.failure_count = 0
                self.consecutive_failures = 0
            elif self.state == "CLOSED":
                # Reset failure counter on success
                self.consecutive_failures = 0
    
    def _record_failure(self, now: float):
        """Record failed request"""
        with self._lock:
            self._stats["total_requests"] += 1
            self.failure_count += 1
            self.consecutive_failures += 1
            self.last_failure_time = now
            
            if self.consecutive_failures >= self.failure_threshold:
                self.state = "OPEN"
                self.state_change_time = now
    
    async def wait_for_tokens(self, tokens: int = 1, timeout: float = None) -> bool:
        """Wait for circuit breaker to allow requests"""
        start_time = time.time()
        
        while True:
            try:
                status = await self.acquire(tokens)
                if status.allowed:
                    return True
            except CircuitBreakerOpenException:
                pass  # Continue waiting
            
            # Check timeout
            if timeout is not None and (time.time() - start_time) >= timeout:
                return False
            
            # Wait before retry
            await asyncio.sleep(1.0)  # Wait 1 second before retry
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        with self._lock:
            return {
                "state": self.state,
                "failure_count": self.failure_count,
                "consecutive_failures": self.consecutive_failures,
                "success_count": self.success_count,
                "state_change_time": self.state_change_time,
                "last_failure_time": self.last_failure_time,
                "recovery_timeout": self.recovery_timeout,
                "failure_threshold": self.failure_threshold
            }


class AdaptiveRateLimiter:
    """
    Adaptive Rate Limiter
    
    Dynamically adjusts rate limits based on API response times,
    error rates, and broker-specific feedback.
    """
    
    def __init__(self, base_limiter: RateLimiter):
        self.base_limiter = base_limiter
        self.adaptation_factor = 1.0
        self.min_factor = 0.1
        self.max_factor = 3.0
        self.adaptation_window = 300  # 5 minutes
        self.response_times = deque(maxlen=1000)
        self.error_count = 0
        self.request_count = 0
        self.last_adaptation = time.time()
        
        # Performance tracking
        self._stats = {
            "avg_response_time": 0.0,
            "error_rate": 0.0,
            "total_adaptations": 0,
            "factor_history": []
        }
    
    async def acquire(self, tokens: int = 1) -> RateLimitStatus:
        """Acquire tokens with adaptive rate limiting"""
        start_time = time.time()
        
        # Get base rate limit status
        base_status = await self.base_limiter.acquire(tokens)
        
        # Record performance metrics if request succeeded
        if base_status.allowed:
            response_time = time.time() - start_time
            self.response_times.append(response_time)
        
        # Adapt rate limiting based on performance
        await self._adapt()
        
        return base_status
    
    async def _adapt(self):
        """Adapt rate limiting based on current performance"""
        now = time.time()
        
        # Only adapt every adaptation window
        if now - self.last_adaptation < self.adaptation_window:
            return
        
        self.last_adaptation = now
        self.request_count = len(self.response_times)
        
        if self.request_count < 10:  # Need minimum samples
            return
        
        # Calculate performance metrics
        avg_response_time = sum(self.response_times) / len(self.response_times)
        error_rate = self.error_count / max(1, self.request_count)
        
        # Update stats
        self._stats["avg_response_time"] = avg_response_time
        self._stats["error_rate"] = error_rate
        
        # Adaptation logic
        new_factor = self.adaptation_factor
        
        # Slow down on high error rates
        if error_rate > 0.05:  # 5% error rate
            new_factor *= 0.8
        # Slow down on slow response times
        elif avg_response_time > 1.0:  # > 1 second
            new_factor *= 0.9
        # Speed up on good performance
        elif error_rate < 0.01 and avg_response_time < 0.1:  # < 1% error, < 100ms
            new_factor *= 1.1
        
        # Apply bounds
        new_factor = max(self.min_factor, min(self.max_factor, new_factor))
        
        # Check if adaptation is significant
        if abs(new_factor - self.adaptation_factor) > 0.05:
            self.adaptation_factor = new_factor
            self._stats["total_adaptations"] += 1
            self._stats["factor_history"].append({
                "timestamp": now,
                "factor": new_factor,
                "avg_response_time": avg_response_time,
                "error_rate": error_rate
            })
            
            # Update base limiter
            if hasattr(self.base_limiter, 'rate_per_second'):
                original_rate = self.base_limiter.rate_per_second / (new_factor / self.adaptation_factor)
                self.base_limiter.rate_per_second = original_rate * new_factor
    
    def record_error(self):
        """Record API error for adaptation"""
        self.error_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get adaptive rate limiter statistics"""
        stats = self.base_limiter.get_stats()
        stats.update(self._stats)
        stats["current_factor"] = self.adaptation_factor
        return stats