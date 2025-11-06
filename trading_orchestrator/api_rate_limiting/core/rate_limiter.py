"""
Rate Limiter Manager

Central coordinator for all rate limiting operations across brokers.
Manages multiple rate limiters, combines strategies, and provides unified interface.
"""

import asyncio
import time
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import threading
from contextlib import asynccontextmanager

from .algorithm import (
    RateLimiter, TokenBucketRateLimiter, SlidingWindowRateLimiter,
    CircuitBreakerRateLimiter, AdaptiveRateLimiter, RateLimitStatus
)
from .exceptions import RateLimitExceededException


class RequestType(Enum):
    """Types of API requests"""
    ACCOUNT_INFO = "account_info"
    ORDER_PLACE = "order_place"
    ORDER_CANCEL = "order_cancel"
    ORDER_QUERY = "order_query"
    POSITION_QUERY = "position_query"
    MARKET_DATA = "market_data"
    HISTORICAL_DATA = "historical_data"
    REAL_TIME_DATA = "real_time_data"
    WEBSOCKET = "websocket"
    AUTHENTICATION = "authentication"


@dataclass
class RateLimitRule:
    """Rule defining rate limits for specific request types"""
    request_type: RequestType
    limit: int
    window_seconds: float
    algorithm: str = "token_bucket"
    priority: int = 1
    timeout: float = 30.0
    retry_enabled: bool = True
    circuit_breaker: bool = True


class RateLimiterManager:
    """
    Central rate limiting manager
    
    Coordinates multiple rate limiters across different request types
    and brokers, providing a unified interface for rate limiting.
    """
    
    def __init__(self, broker_name: str, config: Dict[str, Any]):
        self.broker_name = broker_name
        self.config = config
        self._lock = threading.RLock()
        
        # Initialize rate limiters
        self._rate_limiters: Dict[RequestType, RateLimiter] = {}
        self._adaptive_limiters: Dict[RequestType, AdaptiveRateLimiter] = {}
        self._rules: Dict[RequestType, RateLimitRule] = {}
        
        # Statistics tracking
        self._global_stats = {
            "total_requests": 0,
            "allowed_requests": 0,
            "rejected_requests": 0,
            "average_wait_time": 0.0,
            "requests_by_type": defaultdict(int),
            "peak_concurrent_requests": 0
        }
        
        # Current concurrent requests tracking
        self._current_requests = 0
        self._request_history = deque(maxlen=1000)
        
        # Event hooks for monitoring
        self._on_limit_exceeded: List[Callable] = []
        self._on_request_allowed: List[Callable] = []
        
        # Initialize with default rules
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default rate limiting rules"""
        # Default rules for each request type
        default_rules = {
            RequestType.ACCOUNT_INFO: RateLimitRule(
                request_type=RequestType.ACCOUNT_INFO,
                limit=100, window_seconds=60, priority=1
            ),
            RequestType.ORDER_PLACE: RateLimitRule(
                request_type=RequestType.ORDER_PLACE,
                limit=50, window_seconds=60, priority=2
            ),
            RequestType.ORDER_CANCEL: RateLimitRule(
                request_type=RequestType.ORDER_CANCEL,
                limit=100, window_seconds=60, priority=2
            ),
            RequestType.ORDER_QUERY: RateLimitRule(
                request_type=RequestType.ORDER_QUERY,
                limit=200, window_seconds=60, priority=1
            ),
            RequestType.POSITION_QUERY: RateLimitRule(
                request_type=RequestType.POSITION_QUERY,
                limit=100, window_seconds=60, priority=1
            ),
            RequestType.MARKET_DATA: RateLimitRule(
                request_type=RequestType.MARKET_DATA,
                limit=500, window_seconds=60, priority=3
            ),
            RequestType.HISTORICAL_DATA: RateLimitRule(
                request_type=RequestType.HISTORICAL_DATA,
                limit=10, window_seconds=60, priority=4
            ),
            RequestType.REAL_TIME_DATA: RateLimitRule(
                request_type=RequestType.REAL_TIME_DATA,
                limit=1000, window_seconds=60, priority=3
            ),
            RequestType.WEBSOCKET: RateLimitRule(
                request_type=RequestType.WEBSOCKET,
                limit=10, window_seconds=60, priority=5
            ),
            RequestType.AUTHENTICATION: RateLimitRule(
                request_type=RequestType.AUTHENTICATION,
                limit=5, window_seconds=300, priority=1
            )
        }
        
        # Apply broker-specific configurations
        for request_type, rule in default_rules.items():
            self.set_rule(rule)
    
    def set_rule(self, rule: RateLimitRule):
        """Set or update a rate limiting rule"""
        with self._lock:
            self._rules[rule.request_type] = rule
            
            # Create or update rate limiter for this request type
            limiter = self._create_rate_limiter(rule)
            
            # Wrap with adaptive rate limiter if enabled
            if self.config.get("adaptive_enabled", True):
                adaptive_limiter = AdaptiveRateLimiter(limiter)
                self._adaptive_limiters[rule.request_type] = adaptive_limiter
            else:
                self._rate_limiters[rule.request_type] = limiter
    
    def _create_rate_limiter(self, rule: RateLimitRule) -> RateLimiter:
        """Create rate limiter based on rule configuration"""
        if rule.algorithm == "token_bucket":
            limiter = TokenBucketRateLimiter(
                name=f"{self.broker_name}_{rule.request_type.value}",
                limit=rule.limit,
                window_seconds=rule.window_seconds
            )
        elif rule.algorithm == "sliding_window":
            limiter = SlidingWindowRateLimiter(
                name=f"{self.broker_name}_{rule.request_type.value}",
                limit=rule.limit,
                window_seconds=rule.window_seconds
            )
        else:
            # Default to token bucket
            limiter = TokenBucketRateLimiter(
                name=f"{self.broker_name}_{rule.request_type.value}",
                limit=rule.limit,
                window_seconds=rule.window_seconds
            )
        
        # Add circuit breaker if enabled
        if rule.circuit_breaker:
            limiter = CircuitBreakerRateLimiter(
                name=f"{self.broker_name}_{rule.request_type.value}_circuit",
                limit=rule.limit,
                window_seconds=rule.window_seconds,
                failure_threshold=5,
                recovery_timeout=60.0
            )
        
        return limiter
    
    async def acquire(self, request_type: RequestType, tokens: int = 1) -> RateLimitStatus:
        """
        Acquire permission to make a request
        
        Args:
            request_type: Type of request being made
            tokens: Number of tokens to acquire
            
        Returns:
            RateLimitStatus: Status of the request
        """
        with self._lock:
            self._current_requests += 1
            self._global_stats["peak_concurrent_requests"] = max(
                self._global_stats["peak_concurrent_requests"], 
                self._current_requests
            )
            self._request_history.append(time.time())
        
        try:
            self._global_stats["total_requests"] += 1
            self._global_stats["requests_by_type"][request_type.value] += 1
            
            # Get appropriate rate limiter
            if request_type in self._adaptive_limiters:
                limiter = self._adaptive_limiters[request_type]
            elif request_type in self._rate_limiters:
                limiter = self._rate_limiters[request_type]
            else:
                raise ValueError(f"No rate limiter configured for request type: {request_type}")
            
            start_time = time.time()
            
            # Acquire tokens
            status = await limiter.acquire(tokens)
            
            # Record wait time
            wait_time = time.time() - start_time
            self._global_stats["average_wait_time"] = (
                (self._global_stats["average_wait_time"] * 
                 (self._global_stats["total_requests"] - 1) + wait_time) /
                self._global_stats["total_requests"]
            )
            
            if status.allowed:
                self._global_stats["allowed_requests"] += 1
                self._notify_hooks(self._on_request_allowed, request_type, status)
            else:
                self._global_stats["rejected_requests"] += 1
                self._notify_hooks(self._on_limit_exceeded, request_type, status)
            
            return status
            
        finally:
            with self._lock:
                self._current_requests = max(0, self._current_requests - 1)
    
    async def wait_for_tokens(self, request_type: RequestType, tokens: int = 1, timeout: float = None) -> bool:
        """
        Wait for tokens to become available
        
        Args:
            request_type: Type of request
            tokens: Number of tokens needed
            timeout: Maximum time to wait
            
        Returns:
            bool: True if tokens acquired, False if timeout
        """
        if request_type in self._adaptive_limiters:
            limiter = self._adaptive_limiters[request_type]
        elif request_type in self._rate_limiters:
            limiter = self._rate_limiters[request_type]
        else:
            return False
        
        rule = self._rules.get(request_type)
        if rule:
            timeout = timeout or rule.timeout
        
        return await limiter.wait_for_tokens(tokens, timeout)
    
    @asynccontextmanager
    async def rate_limited_request(self, request_type: RequestType, tokens: int = 1):
        """
        Context manager for rate limited requests
        
        Usage:
            async with rate_limiter_manager.rate_limited_request(RequestType.ORDER_PLACE):
                # Make API call here
                result = await self.place_order(...)
        """
        status = await self.acquire(request_type, tokens)
        
        if not status.allowed:
            raise RateLimitExceededException(
                f"Rate limit exceeded for {request_type.value}",
                reset_time=status.reset_time,
                limit=status.limit,
                remaining=status.remaining
            )
        
        try:
            yield status
        except Exception as e:
            # Record error for adaptive rate limiting
            if request_type in self._adaptive_limiters:
                self._adaptive_limiters[request_type].record_error()
            raise
    
    def get_rate_limit_status(self, request_type: RequestType) -> Dict[str, Any]:
        """Get current rate limit status for a request type"""
        with self._lock:
            if request_type in self._adaptive_limiters:
                stats = self._adaptive_limiters[request_type].get_stats()
            elif request_type in self._rate_limiters:
                stats = self._rate_limiters[request_type].get_stats()
            else:
                stats = {}
            
            return {
                "request_type": request_type.value,
                "broker": self.broker_name,
                "limit": self._rules.get(request_type).limit if request_type in self._rules else 0,
                "window_seconds": self._rules.get(request_type).window_seconds if request_type in self._rules else 0,
                **stats
            }
    
    def get_global_status(self) -> Dict[str, Any]:
        """Get global rate limiting status"""
        with self._lock:
            current_load = len([t for t in self._request_history if time.time() - t < 60])
            
            return {
                "broker": self.broker_name,
                "current_requests": self._current_requests,
                "current_load": current_load,
                "peak_concurrent": self._global_stats["peak_concurrent_requests"],
                "total_requests": self._global_stats["total_requests"],
                "allowed_requests": self._global_stats["allowed_requests"],
                "rejected_requests": self._global_stats["rejected_requests"],
                "average_wait_time": self._global_stats["average_wait_time"],
                "success_rate": (
                    self._global_stats["allowed_requests"] / 
                    max(1, self._global_stats["total_requests"])
                ),
                "requests_by_type": dict(self._global_stats["requests_by_type"]),
                "active_limiters": len(self._rate_limiters),
                "adaptive_limiters": len(self._adaptive_limiters),
                "timestamp": time.time()
            }
    
    def add_hook(self, event_type: str, callback: Callable):
        """Add event hook for monitoring"""
        if event_type == "limit_exceeded":
            self._on_limit_exceeded.append(callback)
        elif event_type == "request_allowed":
            self._on_request_allowed.append(callback)
    
    def _notify_hooks(self, hooks: List[Callable], request_type: RequestType, status: RateLimitStatus):
        """Notify all registered hooks"""
        for hook in hooks:
            try:
                hook(request_type, status)
            except Exception as e:
                # Log error but don't fail
                print(f"Hook error: {e}")
    
    def reset_stats(self):
        """Reset all statistics"""
        with self._lock:
            self._global_stats = {
                "total_requests": 0,
                "allowed_requests": 0,
                "rejected_requests": 0,
                "average_wait_time": 0.0,
                "requests_by_type": defaultdict(int),
                "peak_concurrent_requests": 0
            }
            
            for limiter in self._rate_limiters.values():
                limiter.reset_stats()
            
            for limiter in self._adaptive_limiters.values():
                limiter.get_stats()  # Reset through base method
                if hasattr(limiter, 'base_limiter'):
                    limiter.base_limiter.reset_stats()


from collections import deque