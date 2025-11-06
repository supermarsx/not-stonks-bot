"""
Provider-Specific Rate Limiting - Per-provider request throttling

Implements rate limiting for each LLM provider to respect API limits and
optimize request distribution across providers.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import threading

from loguru import logger


class RateLimitStrategy(Enum):
    """Rate limiting strategy"""
    FIXED_WINDOW = "fixed_window"      # Fixed time windows
    SLIDING_WINDOW = "sliding_window"  # Sliding time windows
    TOKEN_BUCKET = "token_bucket"      # Token bucket algorithm
    LEAKY_BUCKET = "leaky_bucket"      # Leaky bucket algorithm


@dataclass
class RateLimitConfig:
    """Rate limiting configuration for a provider"""
    requests_per_minute: int = 60
    requests_per_hour: int = 3600
    requests_per_day: int = 86400
    tokens_per_minute: Optional[int] = None
    tokens_per_hour: Optional[int] = None
    tokens_per_day: Optional[int] = None
    burst_limit: int = 100
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    
    # Request prioritization
    high_priority_limit: int = 10      # Max high-priority requests per minute
    critical_priority_limit: int = 5   # Max critical requests per minute
    
    # Backoff configuration
    initial_backoff: float = 1.0       # Initial backoff seconds
    max_backoff: float = 60.0          # Maximum backoff seconds
    backoff_multiplier: float = 2.0    # Backoff multiplier on failures


@dataclass
class RequestToken:
    """Represents a rate-limited request"""
    request_id: str
    provider_name: str
    timestamp: datetime
    token_count: int = 1
    priority: int = 3  # 1=critical, 2=high, 3=medium, 4=low
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class RateLimitState:
    """Current rate limiting state"""
    request_history: deque = field(default_factory=deque)
    token_history: deque = field(default_factory=deque)
    last_request_time: Optional[datetime] = None
    backoff_until: Optional[datetime] = None
    consecutive_failures: int = 0
    
    # Token bucket state (for token bucket strategy)
    tokens: int = 0
    last_refill: Optional[datetime] = None


class ProviderRateLimiter:
    """
    Rate limiter for a single provider
    
    Handles request throttling, priority queuing, and adaptive backoff
    based on provider API limits and failure patterns.
    """
    
    def __init__(
        self,
        provider_name: str,
        config: RateLimitConfig,
        strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    ):
        """
        Initialize rate limiter for a provider
        
        Args:
            provider_name: Name of the provider
            config: Rate limiting configuration
            strategy: Rate limiting strategy to use
        """
        self.provider_name = provider_name
        self.config = config
        self.strategy = strategy
        self.state = RateLimitState()
        self._lock = threading.Lock()
        
        # Initialize token bucket if using token bucket strategy
        if self.strategy == RateLimitStrategy.TOKEN_BUCKET:
            self.state.tokens = config.burst_limit
            self.state.last_refill = datetime.now()
            
        logger.info(f"Initialized rate limiter for {provider_name}: {config.requests_per_minute} req/min")
        
    async def acquire(self, token_count: int = 1, priority: int = 3) -> bool:
        """
        Acquire permission to make a request
        
        Args:
            token_count: Number of tokens to acquire
            priority: Request priority (1=critical, 2=high, 3=medium, 4=low)
            
        Returns:
            True if request can proceed, False if rate limited
        """
        with self._lock:
            current_time = datetime.now()
            
            # Check backoff
            if self.state.backoff_until and current_time < self.state.backoff_until:
                return False
                
            # Clean old requests
            self._clean_old_requests(current_time)
            
            # Check rate limits based on strategy
            if self.strategy == RateLimitStrategy.FIXED_WINDOW:
                return self._check_fixed_window(current_time, token_count, priority)
            elif self.strategy == RateLimitStrategy.SLIDING_WINDOW:
                return self._check_sliding_window(current_time, token_count, priority)
            elif self.strategy == RateLimitStrategy.TOKEN_BUCKET:
                return self._check_token_bucket(current_time, token_count)
            else:
                return True  # Leaky bucket handled separately
                
    async def release(self, success: bool = True):
        """Release tokens after request completion"""
        with self._lock:
            current_time = datetime.now()
            
            # Update consecutive failures
            if success:
                self.state.consecutive_failures = 0
            else:
                self.state.consecutive_failures += 1
                
            # Apply exponential backoff on consecutive failures
            if self.state.consecutive_failures >= 3:
                backoff_time = min(
                    self.config.initial_backoff * (self.config.backoff_multiplier ** (self.state.consecutive_failures - 3)),
                    self.config.max_backoff
                )
                self.state.backoff_until = current_time + timedelta(seconds=backoff_time)
                logger.warning(f"Provider {self.provider_name} in backoff mode for {backoff_time}s (failures: {self.state.consecutive_failures})")
                
    def _clean_old_requests(self, current_time: datetime):
        """Clean requests outside the time windows"""
        # Clean requests older than 1 day (keeping some margin)
        cutoff_time = current_time - timedelta(days=1)
        
        # Clean request history
        while (self.state.request_history and 
               self.state.request_history[0].timestamp < cutoff_time):
            self.state.request_history.popleft()
            
        # Clean token history
        while (self.state.token_history and
               self.state.token_history[0][0] < cutoff_time):
            self.state.token_history.popleft()
            
    def _check_fixed_window(
        self,
        current_time: datetime,
        token_count: int,
        priority: int
    ) -> bool:
        """Check rate limits using fixed window approach"""
        # Get current minute window start
        minute_start = current_time.replace(second=0, microsecond=0)
        
        # Count requests in current window
        minute_requests = sum(
            1 for req in self.state.request_history
            if req.timestamp >= minute_start
        )
        
        # Check minute limit
        if minute_requests + token_count > self.config.requests_per_minute:
            return False
            
        # Check priority-specific limits
        if priority <= 2:  # High or critical priority
            high_priority_requests = sum(
                1 for req in self.state.request_history
                if (req.timestamp >= minute_start and req.priority <= 2)
            )
            
            priority_limit = (self.config.critical_priority_limit if priority == 1 
                            else self.config.high_priority_limit)
                            
            if high_priority_requests >= priority_limit:
                return False
                
        # Record the request
        request_token = RequestToken(
            request_id=f"{current_time.timestamp()}_{id(self)}",
            provider_name=self.provider_name,
            timestamp=current_time,
            token_count=token_count,
            priority=priority
        )
        
        self.state.request_history.append(request_token)
        self.state.last_request_time = current_time
        
        return True
        
    def _check_sliding_window(
        self,
        current_time: datetime,
        token_count: int,
        priority: int
    ) -> bool:
        """Check rate limits using sliding window approach"""
        # 1-minute sliding window
        minute_ago = current_time - timedelta(minutes=1)
        recent_requests = [
            req for req in self.state.request_history
            if req.timestamp > minute_ago
        ]
        
        # Check minute limit
        recent_token_count = sum(req.token_count for req in recent_requests)
        if recent_token_count + token_count > self.config.requests_per_minute:
            return False
            
        # 1-hour sliding window
        hour_ago = current_time - timedelta(hours=1)
        hourly_requests = [
            req for req in self.state.request_history
            if req.timestamp > hour_ago
        ]
        
        if len(hourly_requests) + token_count > self.config.requests_per_hour:
            return False
            
        # Priority-specific checks for last minute
        priority_requests = [
            req for req in recent_requests
            if req.priority == priority
        ]
        
        if priority == 1 and len(priority_requests) >= self.config.critical_priority_limit:
            return False
        elif priority == 2 and len(priority_requests) >= self.config.high_priority_limit:
            return False
            
        # Record the request
        request_token = RequestToken(
            request_id=f"{current_time.timestamp()}_{id(self)}",
            provider_name=self.provider_name,
            timestamp=current_time,
            token_count=token_count,
            priority=priority
        )
        
        self.state.request_history.append(request_token)
        self.state.last_request_time = current_time
        
        return True
        
    def _check_token_bucket(
        self,
        current_time: datetime,
        token_count: int
    ) -> bool:
        """Check rate limits using token bucket approach"""
        # Refill tokens based on time elapsed
        if self.state.last_refill:
            time_elapsed = (current_time - self.state.last_refill).total_seconds()
            refill_rate = self.config.requests_per_minute / 60.0  # tokens per second
            
            tokens_to_add = int(time_elapsed * refill_rate)
            self.state.tokens = min(
                self.config.burst_limit,
                self.state.tokens + tokens_to_add
            )
            self.state.last_refill = current_time
            
        # Check if enough tokens available
        if self.state.tokens < token_count:
            return False
            
        # Consume tokens
        self.state.tokens -= token_count
        
        # Record the request
        request_token = RequestToken(
            request_id=f"{current_time.timestamp()}_{id(self)}",
            provider_name=self.provider_name,
            timestamp=current_time,
            token_count=token_count
        )
        
        self.state.request_history.append(request_token)
        self.state.last_request_time = current_time
        
        return True
        
    def get_wait_time(self, token_count: int = 1) -> float:
        """
        Calculate wait time until request can proceed
        
        Args:
            token_count: Number of tokens needed
            
        Returns:
            Wait time in seconds (0 if can proceed immediately)
        """
        with self._lock:
            current_time = datetime.now()
            
            if self.strategy == RateLimitStrategy.TOKEN_BUCKET:
                # Calculate time until enough tokens available
                if self.state.tokens >= token_count:
                    return 0
                    
                tokens_needed = token_count - self.state.tokens
                refill_rate = self.config.requests_per_minute / 60.0
                wait_time = tokens_needed / refill_rate
                
                return max(0, wait_time)
                
            elif self.strategy == RateLimitStrategy.SLIDING_WINDOW:
                # Estimate wait time based on recent requests
                minute_ago = current_time - timedelta(minutes=1)
                recent_requests = sum(
                    req.token_count for req in self.state.request_history
                    if req.timestamp > minute_ago
                )
                
                if recent_requests + token_count <= self.config.requests_per_minute:
                    return 0
                    
                # Calculate when request would fit in window
                # Sort by timestamp and find when space becomes available
                sorted_requests = sorted(
                    self.state.request_history,
                    key=lambda r: r.timestamp
                )
                
                token_sum = 0
                for req in sorted_requests:
                    if req.timestamp > minute_ago:
                        token_sum += req.token_count
                        if token_sum + token_count > self.config.requests_per_minute:
                            # Wait until this request falls out of the window
                            wait_until = req.timestamp + timedelta(minutes=1)
                            return max(0, (wait_until - current_time).total_seconds())
                            
                return 0
                
            else:
                return 0
                
    def get_current_usage(self) -> Dict[str, Any]:
        """Get current rate limiting usage statistics"""
        with self._lock:
            current_time = datetime.now()
            
            # Count recent requests
            minute_ago = current_time - timedelta(minutes=1)
            recent_requests = sum(
                req.token_count for req in self.state.request_history
                if req.timestamp > minute_ago
            )
            
            hour_ago = current_time - timedelta(hours=1)
            hourly_requests = sum(
                req.token_count for req in self.state.request_history
                if req.timestamp > hour_ago
            )
            
            # Priority distribution
            priority_counts = {1: 0, 2: 0, 3: 0, 4: 0}
            for req in self.state.request_history:
                if req.timestamp > minute_ago:
                    priority_counts[req.priority] += req.token_count
                    
            # Token bucket state
            token_bucket_state = {}
            if self.strategy == RateLimitStrategy.TOKEN_BUCKET:
                token_bucket_state = {
                    'tokens_available': self.state.tokens,
                    'burst_capacity': self.config.burst_limit
                }
                
            return {
                'provider_name': self.provider_name,
                'requests_last_minute': recent_requests,
                'requests_last_hour': hourly_requests,
                'requests_per_minute_limit': self.config.requests_per_minute,
                'requests_per_hour_limit': self.config.requests_per_hour,
                'usage_percentage': (recent_requests / self.config.requests_per_minute) * 100,
                'priority_distribution': priority_counts,
                'consecutive_failures': self.state.consecutive_failures,
                'in_backoff': self.state.backoff_until is not None,
                'strategy': self.strategy.value,
                'token_bucket': token_bucket_state,
                'last_request_time': self.state.last_request_time.isoformat() if self.state.last_request_time else None
            }
            
    def reset(self):
        """Reset rate limiter state"""
        with self._lock:
            self.state = RateLimitState()
            logger.info(f"Reset rate limiter for {self.provider_name}")


class ProviderRateLimitManager:
    """
    Manager for multiple provider rate limiters
    
    Coordinates rate limiting across all providers and provides
    intelligent request routing to avoid overloading any single provider.
    """
    
    def __init__(self):
        """Initialize rate limit manager"""
        self.limiters: Dict[str, ProviderRateLimiter] = {}
        self.global_lock = threading.Lock()
        
    def add_provider(self, provider_name: str, config: RateLimitConfig, strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW):
        """Add a provider rate limiter"""
        limiter = ProviderRateLimiter(provider_name, config, strategy)
        with self.global_lock:
            self.limiters[provider_name] = limiter
            
        logger.info(f"Added rate limiter for provider: {provider_name}")
        
    def remove_provider(self, provider_name: str):
        """Remove provider rate limiter"""
        with self.global_lock:
            if provider_name in self.limiters:
                del self.limiters[provider_name]
                logger.info(f"Removed rate limiter for provider: {provider_name}")
                
    async def acquire(
        self,
        provider_name: str,
        token_count: int = 1,
        priority: int = 3
    ) -> bool:
        """Acquire rate limit permission for a provider"""
        with self.global_lock:
            limiter = self.limiters.get(provider_name)
            if not limiter:
                return True  # No limiter, allow request
                
        return await limiter.acquire(token_count, priority)
        
    async def release(
        self,
        provider_name: str,
        success: bool = True
    ):
        """Release rate limit tokens"""
        with self.global_lock:
            limiter = self.limiters.get(provider_name)
            if limiter:
                await limiter.release(success)
                
    def get_best_provider(
        self,
        available_providers: List[str],
        token_count: int = 1,
        priority: int = 3
    ) -> Optional[str]:
        """
        Find the best provider based on rate limit availability
        
        Args:
            available_providers: List of available provider names
            token_count: Number of tokens needed
            priority: Request priority
            
        Returns:
            Best provider name or None if all are rate limited
        """
        with self.global_lock:
            best_provider = None
            best_usage = float('inf')
            
            for provider_name in available_providers:
                limiter = self.limiters.get(provider_name)
                if not limiter:
                    # No limiter means provider is available
                    return provider_name
                    
                usage = limiter.get_current_usage()
                usage_percentage = usage['usage_percentage']
                
                # Prefer providers with lower usage
                if usage_percentage < best_usage:
                    best_usage = usage_percentage
                    best_provider = provider_name
                    
        return best_provider
        
    def get_all_usage(self) -> Dict[str, Dict[str, Any]]:
        """Get usage statistics for all providers"""
        with self.global_lock:
            return {
                name: limiter.get_current_usage()
                for name, limiter in self.limiters.items()
            }
            
    def reset_provider(self, provider_name: str):
        """Reset a specific provider's rate limiter"""
        with self.global_lock:
            limiter = self.limiters.get(provider_name)
            if limiter:
                limiter.reset()
                
    def reset_all(self):
        """Reset all provider rate limiters"""
        with self.global_lock:
            for limiter in self.limiters.values():
                limiter.reset()
                
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all rate limiters"""
        usage_stats = self.get_all_usage()
        
        total_providers = len(self.limiters)
        overloaded_providers = sum(
            1 for usage in usage_stats.values()
            if usage['usage_percentage'] > 80
        )
        
        return {
            'total_providers': total_providers,
            'overloaded_providers': overloaded_providers,
            'utilization_percentage': (
                sum(usage['usage_percentage'] for usage in usage_stats.values()) / max(1, total_providers)
            ),
            'provider_details': usage_stats
        }