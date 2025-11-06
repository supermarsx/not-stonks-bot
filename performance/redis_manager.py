"""
Redis Manager

Comprehensive Redis integration for caching, session storage, 
and performance optimization in the trading system.
"""

import json
import logging
import pickle
import time
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
import redis.asyncio as aioredis
import redis as sync_redis
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, asdict
from enum import Enum


class CacheLevel(Enum):
    """Cache level enumeration"""
    L1_MEMORY = "memory"
    L2_REDIS = "redis"
    L3_DISK = "disk"


class SessionDataType(Enum):
    """Session data type enumeration"""
    USER_DATA = "user"
    TRADE_STATE = "trade"
    MARKET_DATA = "market"
    SYSTEM_STATE = "system"


@dataclass
class CacheStats:
    """Cache statistics"""
    hits: int = 0
    misses: int = 0
    hit_ratio: float = 0.0
    memory_usage: Dict[str, Any] = None
    response_times: List[float] = None
    
    def __post_init__(self):
        if self.memory_usage is None:
            self.memory_usage = {}
        if self.response_times is None:
            self.response_times = []
    
    @property
    def avg_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)


class RedisConnectionManager:
    """Redis connection manager with pool optimization"""
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 6379,
                 db: int = 0,
                 password: Optional[str] = None,
                 max_connections: int = 50,
                 connection_pool_size: int = 20,
                 timeout: float = 5.0,
                 socket_timeout: float = 5.0,
                 socket_connect_timeout: float = 5.0,
                 retry_on_timeout: bool = True,
                 health_check_interval: int = 30):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.max_connections = max_connections
        self.connection_pool_size = connection_pool_size
        self.timeout = timeout
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout
        self.retry_on_timeout = retry_on_timeout
        self.health_check_interval = health_check_interval
        self.health_check_interval_obj = None
        
        self._sync_connection_pool = None
        self._connection_lock = None
        self._initialize_sync_pool()
        
        self.logger = logging.getLogger(__name__)
    
    def _initialize_sync_pool(self):
        """Initialize synchronous connection pool"""
        if self._sync_connection_pool is None:
            self._connection_pool = sync_redis.ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                max_connections=self.max_connections,
                retry_on_timeout=self.retry_on_timeout,
                socket_timeout=self.socket_timeout,
                socket_connect_timeout=self.socket_connect_timeout,
                health_check_interval=self.health_check_interval,
                decode_responses=False  # Keep binary for performance
            )
    
    @contextmanager
    def get_connection(self, timeout: Optional[float] = None):
        """Get a Redis connection from the pool"""
        timeout = timeout or self.timeout
        try:
            conn = sync_redis.Redis(
                connection_pool=self._connection_pool,
                socket_timeout=timeout
            )
            yield conn
        finally:
            pass  # Connection returned to pool automatically
    
    @asynccontextmanager
    async def get_async_connection(self, timeout: Optional[float] = None):
        """Get an async Redis connection"""
        timeout = timeout or self.timeout
        try:
            conn = aioredis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                socket_timeout=timeout,
                socket_connect_timeout=self.socket_connect_timeout,
                retry_on_timeout=self.retry_on_timeout,
                health_check_interval=self.health_check_interval,
                decode_responses=False
            )
            yield conn
        finally:
            await conn.close()
    
    def test_connection(self) -> bool:
        """Test Redis connection"""
        try:
            with self.get_connection() as conn:
                conn.ping()
                return True
        except Exception as e:
            self.logger.error(f"Redis connection test failed: {e}")
            return False
    
    def get_info(self) -> Dict[str, Any]:
        """Get Redis server information"""
        try:
            with self.get_connection() as conn:
                return conn.info()
        except Exception as e:
            self.logger.error(f"Failed to get Redis info: {e}")
            return {}


class RedisManager:
    """Comprehensive Redis manager for trading system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize connection manager
        redis_config = self.config.get('redis', {})
        self.connection_manager = RedisConnectionManager(**redis_config)
        
        # Initialize cache statistics
        self.stats = CacheStats()
        
        # Initialize cache strategies
        self._cache_strategies = {
            CacheLevel.L1_MEMORY: {},
            CacheLevel.L2_REDIS: {},
            CacheLevel.L3_DISK: {}
        }
        
        # Initialize session management
        self._session_prefix = "session:"
        self._session_ttl = self.config.get('session_ttl', 3600)  # 1 hour default
        
        # Performance monitoring
        self._performance_enabled = self.config.get('performance_monitoring', True)
        self._query_count = 0
        self._error_count = 0
        
        self.logger.info("RedisManager initialized")
    
    async def connect(self):
        """Establish Redis connections"""
        try:
            if self.connection_manager.test_connection():
                self.logger.info("Redis connection established successfully")
                return True
            else:
                self.logger.error("Failed to establish Redis connection")
                return False
        except Exception as e:
            self.logger.error(f"Redis connection error: {e}")
            return False
    
    def serialize_value(self, value: Any) -> bytes:
        """Serialize value for Redis storage"""
        try:
            if isinstance(value, (str, int, float, bool)):
                return json.dumps(value, default=str).encode('utf-8')
            else:
                return pickle.dumps(value)
        except Exception as e:
            self.logger.error(f"Serialization error: {e}")
            return json.dumps(str(value)).encode('utf-8')
    
    def deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from Redis storage"""
        try:
            if not data:
                return None
            
            # Try JSON first
            try:
                return json.loads(data.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Try pickle
                return pickle.loads(data)
        except Exception as e:
            self.logger.error(f"Deserialization error: {e}")
            return None
    
    def get(self, key: str, default: Any = None, cache_level: CacheLevel = CacheLevel.L2_REDIS) -> Any:
        """Get value from cache"""
        start_time = time.time()
        
        try:
            if cache_level == CacheLevel.L1_MEMORY:
                value = self._cache_strategies[CacheLevel.L1_MEMORY].get(key)
            elif cache_level == CacheLevel.L2_REDIS:
                with self.connection_manager.get_connection() as conn:
                    data = conn.get(key)
                    value = self.deserialize_value(data) if data else None
            else:
                # L3 disk cache would be implemented here
                value = None
            
            response_time = time.time() - start_time
            self._record_cache_access(value is not None, response_time)
            
            return value if value is not None else default
            
        except Exception as e:
            self.logger.error(f"Cache get error for key {key}: {e}")
            self._error_count += 1
            return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, 
            cache_level: CacheLevel = CacheLevel.L2_REDIS) -> bool:
        """Set value in cache"""
        try:
            if cache_level == CacheLevel.L1_MEMORY:
                self._cache_strategies[CacheLevel.L1_MEMORY][key] = value
                return True
            elif cache_level == CacheLevel.L2_REDIS:
                with self.connection_manager.get_connection() as conn:
                    data = self.serialize_value(value)
                    if ttl:
                        conn.setex(key, ttl, data)
                    else:
                        conn.set(key, data)
                return True
            else:
                # L3 disk cache would be implemented here
                return False
                
        except Exception as e:
            self.logger.error(f"Cache set error for key {key}: {e}")
            self._error_count += 1
            return False
    
    def delete(self, key: str, cache_level: CacheLevel = CacheLevel.L2_REDIS) -> bool:
        """Delete value from cache"""
        try:
            if cache_level == CacheLevel.L1_MEMORY:
                return self._cache_strategies[CacheLevel.L1_MEMORY].pop(key, None) is not None
            elif cache_level == CacheLevel.L2_REDIS:
                with self.connection_manager.get_connection() as conn:
                    return bool(conn.delete(key))
            else:
                return False
        except Exception as e:
            self.logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    def exists(self, key: str, cache_level: CacheLevel = CacheLevel.L2_REDIS) -> bool:
        """Check if key exists in cache"""
        try:
            if cache_level == CacheLevel.L1_MEMORY:
                return key in self._cache_strategies[CacheLevel.L1_MEMORY]
            elif cache_level == CacheLevel.L2_REDIS:
                with self.connection_manager.get_connection() as conn:
                    return bool(conn.exists(key))
            else:
                return False
        except Exception as e:
            self.logger.error(f"Cache exists error for key {key}: {e}")
            return False
    
    def expire(self, key: str, ttl: int, cache_level: CacheLevel = CacheLevel.L2_REDIS) -> bool:
        """Set expiration time for key"""
        try:
            if cache_level == CacheLevel.L1_MEMORY:
                # L1 memory cache doesn't support TTL directly
                return False
            elif cache_level == CacheLevel.L2_REDIS:
                with self.connection_manager.get_connection() as conn:
                    return bool(conn.expire(key, ttl))
            else:
                return False
        except Exception as e:
            self.logger.error(f"Cache expire error for key {key}: {e}")
            return False
    
    def flush_cache_level(self, cache_level: CacheLevel) -> bool:
        """Flush all data from specific cache level"""
        try:
            if cache_level == CacheLevel.L1_MEMORY:
                self._cache_strategies[CacheLevel.L1_MEMORY].clear()
                return True
            elif cache_level == CacheLevel.L2_REDIS:
                with self.connection_manager.get_connection() as conn:
                    return conn.flushdb()
            else:
                return False
        except Exception as e:
            self.logger.error(f"Cache flush error for level {cache_level}: {e}")
            return False
    
    # Session Management
    def create_session(self, session_id: str, data: Dict[str, Any], 
                      session_type: SessionDataType = SessionDataType.USER_DATA) -> bool:
        """Create a new session"""
        try:
            key = f"{self._session_prefix}{session_type.value}:{session_id}"
            session_data = {
                'id': session_id,
                'type': session_type.value,
                'data': data,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'last_accessed': datetime.now().isoformat()
            }
            
            return self.set(key, session_data, ttl=self._session_ttl, cache_level=CacheLevel.L2_REDIS)
        except Exception as e:
            self.logger.error(f"Session creation error: {e}")
            return False
    
    def get_session(self, session_id: str, session_type: SessionDataType = SessionDataType.USER_DATA) -> Optional[Dict[str, Any]]:
        """Get session data"""
        try:
            key = f"{self._session_prefix}{session_type.value}:{session_id}"
            session_data = self.get(key, cache_level=CacheLevel.L2_REDIS)
            
            if session_data:
                # Update last accessed time
                session_data['last_accessed'] = datetime.now().isoformat()
                self.set(key, session_data, ttl=self._session_ttl, cache_level=CacheLevel.L2_REDIS)
            
            return session_data
        except Exception as e:
            self.logger.error(f"Session retrieval error: {e}")
            return None
    
    def update_session(self, session_id: str, data: Dict[str, Any], 
                      session_type: SessionDataType = SessionDataType.USER_DATA) -> bool:
        """Update existing session"""
        try:
            key = f"{self._session_prefix}{session_type.value}:{session_id}"
            session_data = self.get_session(session_id, session_type)
            
            if session_data:
                session_data['data'].update(data)
                session_data['updated_at'] = datetime.now().isoformat()
                return self.set(key, session_data, ttl=self._session_ttl, cache_level=CacheLevel.L2_REDIS)
            
            return False
        except Exception as e:
            self.logger.error(f"Session update error: {e}")
            return False
    
    def delete_session(self, session_id: str, session_type: SessionDataType = SessionDataType.USER_DATA) -> bool:
        """Delete session"""
        try:
            key = f"{self._session_prefix}{session_type.value}:{session_id}"
            return self.delete(key, cache_level=CacheLevel.L2_REDIS)
        except Exception as e:
            self.logger.error(f"Session deletion error: {e}")
            return False
    
    def get_all_sessions(self, session_type: SessionDataType = SessionDataType.USER_DATA) -> List[str]:
        """Get all session IDs of specified type"""
        try:
            pattern = f"{self._session_prefix}{session_type.value}:*"
            with self.connection_manager.get_connection() as conn:
                return [key.decode('utf-8').split(':')[-1] for key in conn.keys(pattern)]
        except Exception as e:
            self.logger.error(f"Get all sessions error: {e}")
            return []
    
    # Trading Data Caching
    async def cache_market_data(self, symbol: str, data: Dict[str, Any], ttl: int = 60) -> bool:
        """Cache market data with symbol-based key"""
        try:
            key = f"market:{symbol}"
            data['cached_at'] = datetime.now().isoformat()
            return self.set(key, data, ttl=ttl, cache_level=CacheLevel.L2_REDIS)
        except Exception as e:
            self.logger.error(f"Market data caching error: {e}")
            return False
    
    async def get_cached_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached market data"""
        try:
            key = f"market:{symbol}"
            return self.get(key, cache_level=CacheLevel.L2_REDIS)
        except Exception as e:
            self.logger.error(f"Get cached market data error: {e}")
            return None
    
    async def cache_trade_execution(self, trade_id: str, data: Dict[str, Any], ttl: int = 3600) -> bool:
        """Cache trade execution data"""
        try:
            key = f"trade:{trade_id}"
            data['cached_at'] = datetime.now().isoformat()
            return self.set(key, data, ttl=ttl, cache_level=CacheLevel.L2_REDIS)
        except Exception as e:
            self.logger.error(f"Trade execution caching error: {e}")
            return False
    
    async def get_cached_trade_execution(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """Get cached trade execution data"""
        try:
            key = f"trade:{trade_id}"
            return self.get(key, cache_level=CacheLevel.L2_REDIS)
        except Exception as e:
            self.logger.error(f"Get cached trade execution error: {e}")
            return None
    
    # Performance Monitoring
    def _record_cache_access(self, hit: bool, response_time: float):
        """Record cache access statistics"""
        if not self._performance_enabled:
            return
        
        self._query_count += 1
        
        if hit:
            self.stats.hits += 1
        else:
            self.stats.misses += 1
        
        self.stats.response_times.append(response_time)
        
        # Keep only last 1000 response times for memory efficiency
        if len(self.stats.response_times) > 1000:
            self.stats.response_times = self.stats.response_times[-1000:]
        
        # Update hit ratio
        total_requests = self.stats.hits + self.stats.misses
        if total_requests > 0:
            self.stats.hit_ratio = self.stats.hits / total_requests
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats_dict = asdict(self.stats)
        stats_dict.update({
            'query_count': self._query_count,
            'error_count': self._error_count,
            'error_rate': self._error_count / max(self._query_count, 1),
            'cache_levels': {
                'memory_keys': len(self._cache_strategies[CacheLevel.L1_MEMORY]),
                'redis_info': self.connection_manager.get_info()
            }
        })
        return stats_dict
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.stats = CacheStats()
        self._query_count = 0
        self._error_count = 0
        self.logger.info("Performance statistics reset")
    
    def benchmark(self, iterations: int = 100) -> Dict[str, float]:
        """Benchmark cache performance"""
        self.logger.info(f"Starting cache benchmark with {iterations} iterations")
        
        benchmark_results = {
            'set_latency': [],
            'get_latency': [],
            'delete_latency': []
        }
        
        # Test SET operations
        for i in range(iterations):
            key = f"benchmark:set:{i}"
            start_time = time.time()
            self.set(key, f"value_{i}", cache_level=CacheLevel.L2_REDIS)
            benchmark_results['set_latency'].append(time.time() - start_time)
        
        # Test GET operations
        for i in range(iterations):
            key = f"benchmark:get:{i}"
            start_time = time.time()
            self.get(key, cache_level=CacheLevel.L2_REDIS)
            benchmark_results['get_latency'].append(time.time() - start_time)
        
        # Test DELETE operations
        for i in range(iterations):
            key = f"benchmark:delete:{i}"
            start_time = time.time()
            self.delete(key, cache_level=CacheLevel.L2_REDIS)
            benchmark_results['delete_latency'].append(time.time() - start_time)
        
        # Calculate averages
        results = {}
        for operation, latencies in benchmark_results.items():
            results[operation] = {
                'avg': sum(latencies) / len(latencies),
                'min': min(latencies),
                'max': max(latencies),
                'total': sum(latencies)
            }
        
        return results
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health_status = {
            'status': 'healthy',
            'checks': {},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Test connection
            connection_ok = self.connection_manager.test_connection()
            health_status['checks']['connection'] = {
                'status': 'ok' if connection_ok else 'error',
                'message': 'Connection successful' if connection_ok else 'Connection failed'
            }
            
            # Get Redis info
            redis_info = self.connection_manager.get_info()
            health_status['checks']['redis_info'] = {
                'status': 'ok',
                'info': redis_info
            }
            
            # Test cache operations
            test_key = f"health_check:{int(time.time())}"
            set_ok = self.set(test_key, "test_value", ttl=60, cache_level=CacheLevel.L2_REDIS)
            get_ok = self.get(test_key, cache_level=CacheLevel.L2_REDIS) == "test_value"
            delete_ok = self.delete(test_key, cache_level=CacheLevel.L2_REDIS)
            
            health_status['checks']['cache_operations'] = {
                'status': 'ok' if all([set_ok, get_ok, delete_ok]) else 'error',
                'set_test': set_ok,
                'get_test': get_ok,
                'delete_test': delete_ok
            }
            
            # Performance stats
            stats = self.get_performance_stats()
            health_status['checks']['performance'] = {
                'status': 'ok',
                'stats': stats
            }
            
        except Exception as e:
            health_status['status'] = 'error'
            health_status['error'] = str(e)
            self.logger.error(f"Health check failed: {e}")
        
        return health_status
    
    def close(self):
        """Close Redis connections and cleanup"""
        try:
            if hasattr(self.connection_manager, '_connection_pool') and self.connection_manager._connection_pool:
                self.connection_manager._connection_pool.disconnect()
            self.logger.info("RedisManager connections closed")
        except Exception as e:
            self.logger.error(f"Error closing RedisManager: {e}")


# Global Redis manager instance
redis_manager = None

def get_redis_manager(config: Optional[Dict[str, Any]] = None) -> RedisManager:
    """Get global Redis manager instance"""
    global redis_manager
    if redis_manager is None:
        redis_manager = RedisManager(config)
    return redis_manager


def initialize_redis(config: Optional[Dict[str, Any]] = None) -> RedisManager:
    """Initialize Redis manager with configuration"""
    global redis_manager
    redis_manager = RedisManager(config)
    return redis_manager