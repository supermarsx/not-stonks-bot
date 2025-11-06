"""
API Rate Limiting and Compliance System

Comprehensive rate limiting framework for all broker integrations with:
- Token bucket and sliding window rate limiting
- Exponential backoff retry logic
- Request prioritization and batching
- Real-time monitoring and alerting
- Compliance and audit features
"""

from .core.rate_limiter import (
    RateLimiter,
    TokenBucketRateLimiter,
    SlidingWindowRateLimiter,
    CircuitBreakerRateLimiter,
    RateLimitExceededException
)

from .core.request_manager import (
    RequestManager,
    Request,
    RequestPriority,
    RequestStatus,
    RetryPolicy,
    DeadLetterQueue
)

from .core.algorithm import (
    RateLimitingAlgorithm,
    ExponentialBackoff,
    AdaptiveRateLimit
)

from .brokers.rate_limit_configs import (
    RateLimitConfig,
    get_broker_config,
    BinanceRateLimitConfig,
    AlpacaRateLimitConfig,
    IBKRRateLimitConfig,
    DEGIRORateLimitConfig,
    Trading212RateLimitConfig,
    TradeRepublicRateLimitConfig
)

from .monitoring.monitor import (
    RateLimitMonitor,
    UsageAnalytics,
    AlertingSystem,
    HealthChecker
)

from .compliance.compliance_engine import (
    ComplianceEngine,
    APILogging,
    CostOptimization
)

__version__ = "1.0.0"

__all__ = [
    # Core components
    "RateLimiter",
    "TokenBucketRateLimiter", 
    "SlidingWindowRateLimiter",
    "CircuitBreakerRateLimiter",
    "RateLimitExceededException",
    
    # Request management
    "RequestManager",
    "Request",
    "RequestPriority",
    "RequestStatus", 
    "RetryPolicy",
    "DeadLetterQueue",
    
    # Algorithms
    "RateLimitingAlgorithm",
    "ExponentialBackoff",
    "AdaptiveRateLimit",
    
    # Broker configs
    "RateLimitConfig",
    "get_broker_config",
    "BinanceRateLimitConfig",
    "AlpacaRateLimitConfig",
    "IBKRRateLimitConfig", 
    "DEGIRORateLimitConfig",
    "Trading212RateLimitConfig",
    "TradeRepublicRateLimitConfig",
    
    # Monitoring
    "RateLimitMonitor",
    "UsageAnalytics",
    "AlertingSystem",
    "HealthChecker",
    
    # Compliance
    "ComplianceEngine",
    "APILogging",
    "CostOptimization"
]