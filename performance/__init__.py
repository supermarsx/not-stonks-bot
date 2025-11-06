"""
Performance Optimization System

Comprehensive logging, caching, and performance monitoring system
for the trading orchestrator and related components.

Components:
- Redis caching and session management
- Comprehensive logging with structured JSON format
- Performance monitoring and metrics collection
- Application Performance Monitoring (APM)
- Database optimization and connection pooling
- Memory optimization and garbage collection
- Caching strategies and lazy loading
- Performance profiling and analysis
"""

from .redis_manager import RedisManager
from .logging_config import LoggingConfig, get_logger
from .metrics_collector import MetricsCollector
from .apm_system import APMClient
from .cache_strategies import CacheManager, CacheStrategy

__version__ = "1.0.0"
__all__ = [
    "RedisManager",
    "LoggingConfig", 
    "get_logger",
    "MetricsCollector",
    "APMClient",
    "CacheManager",
    "CacheStrategy"
]