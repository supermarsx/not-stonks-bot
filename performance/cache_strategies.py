"""
Cache Strategies

Multi-level caching implementation with cache warming, invalidation,
and performance metrics for the trading system.
"""

import time
import logging
import hashlib
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, OrderedDict, deque
from contextlib import contextmanager
import functools
import weakref


class CacheLevel(Enum):
    """Cache level enumeration"""
    L1_MEMORY = "memory"
    L2_REDIS = "redis"
    L3_DISK = "disk"
    L4_DATABASE = "database"


class CacheStrategy(Enum):
    """Cache strategy enumeration"""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    TTL = "ttl"
    ADAPTIVE = "adaptive"


class CacheEvent(Enum):
    """Cache event types"""
    HIT = "hit"
    MISS = "miss"
    EVICTION = "eviction"
    EXPIRATION = "expiration"
    INVALIDATION = "invalidation"
    WARMING = "warming"


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl: Optional[int] = None  # Time to live in seconds
    size_bytes: int = 0
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.ttl is None:
            return False
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl
    
    def access(self):
        """Record access to entry"""
        self.last_accessed = datetime.now()
        self.access_count += 1
    
    def get_age_seconds(self) -> float:
        """Get entry age in seconds"""
        return (datetime.now() - self.created_at).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'key': self.key,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'access_count': self.access_count,
            'ttl': self.ttl,
            'size_bytes': self.size_bytes,
            'tags': list(self.tags),
            'metadata': self.metadata,
            'is_expired': self.is_expired(),
            'age_seconds': self.get_age_seconds()
        }


@dataclass
class CacheStatistics:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    invalidations: int = 0
    warmings: int = 0
    total_operations: int = 0
    total_size_bytes: int = 0
    entry_count: int = 0
    
    @property
    def hit_ratio(self) -> float:
        """Calculate hit ratio"""
        total_requests = self.hits + self.misses
        return self.hits / max(total_requests, 1)
    
    @property
    def miss_ratio(self) -> float:
        """Calculate miss ratio"""
        return 1 - self.hit_ratio
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseCache:
    """Base cache implementation"""
    
    def __init__(self, name: str, strategy: CacheStrategy = CacheStrategy.LRU):
        self.name = name
        self.strategy = strategy
        self.stats = CacheStatistics()
        self._lock = threading.RLock()
        self._event_callbacks = []
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        raise NotImplementedError
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, 
            tags: Optional[Set[str]] = None, **kwargs) -> bool:
        """Set value in cache"""
        raise NotImplementedError
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        raise NotImplementedError
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        raise NotImplementedError
    
    def clear(self) -> bool:
        """Clear all cache entries"""
        raise NotImplementedError
    
    def get_stats(self) -> CacheStatistics:
        """Get cache statistics"""
        return self.stats
    
    def add_event_callback(self, callback: Callable[[CacheEvent, str, Any], None]):
        """Add cache event callback"""
        self._event_callbacks.append(callback)
    
    def _emit_event(self, event: CacheEvent, key: str, value: Any = None):
        """Emit cache event"""
        for callback in self._event_callbacks:
            try:
                callback(event, key, value)
            except Exception as e:
                logging.error(f"Error in cache event callback: {e}")


class MemoryCache(BaseCache):
    """In-memory cache implementation"""
    
    def __init__(self, name: str, max_size: int = 1000, 
                 max_memory_mb: int = 100, strategy: CacheStrategy = CacheStrategy.LRU):
        super().__init__(name, strategy)
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self._entries = {}  # key -> CacheEntry
        self._access_order = OrderedDict()  # For LRU
        self._frequency_order = defaultdict(int)  # For LFU
        
        # Calculate approximate object size
        import sys
        self._size_calculator = sys.getsizeof
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache"""
        with self._lock:
            entry = self._entries.get(key)
            
            if entry is None:
                self.stats.misses += 1
                self._emit_event(CacheEvent.MISS, key)
                return None
            
            if entry.is_expired():
                # Remove expired entry
                self._remove_entry(key)
                self.stats.expirations += 1
                self._emit_event(CacheEvent.EXPIRATION, key)
                return None
            
            # Record access
            entry.access()
            self._update_access_tracking(key, entry)
            
            self.stats.hits += 1
            self._emit_event(CacheEvent.HIT, key, entry.value)
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None,
            tags: Optional[Set[str]] = None, **kwargs) -> bool:
        """Set value in memory cache"""
        with self._lock:
            # Calculate entry size
            entry_size = self._calculate_entry_size(value)
            
            # Check if we need to evict entries
            while (len(self._entries) >= self.max_size or 
                   self.stats.total_size_bytes + entry_size > self.max_memory_bytes):
                if not self._evict_entry():
                    # Can't evict, fail to set
                    return False
            
            # Create entry
            now = datetime.now()
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                last_accessed=now,
                ttl=ttl,
                size_bytes=entry_size,
                tags=tags or set(),
                metadata=kwargs
            )
            
            # Update tracking
            self._entries[key] = entry
            self._update_access_tracking(key, entry)
            
            # Update statistics
            self.stats.total_operations += 1
            self.stats.total_size_bytes += entry_size
            self.stats.entry_count += 1
            
            self._emit_event(CacheEvent.WARMING, key, value)
            return True
    
    def delete(self, key: str) -> bool:
        """Delete value from memory cache"""
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry:
                # Update tracking
                self._remove_from_tracking(key)
                
                # Update statistics
                self.stats.total_size_bytes -= entry.size_bytes
                self.stats.entry_count -= 1
                
                self._emit_event(CacheEvent.INVALIDATION, key, entry.value)
                return True
            
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in memory cache"""
        with self._lock:
            entry = self._entries.get(key)
            if entry and not entry.is_expired():
                return True
            
            # Remove expired entry if found
            if entry:
                self._remove_entry(key)
            
            return False
    
    def clear(self) -> bool:
        """Clear all memory cache entries"""
        with self._lock:
            self._entries.clear()
            self._access_order.clear()
            self._frequency_order.clear()
            
            self.stats.total_size_bytes = 0
            self.stats.entry_count = 0
            self.stats.evictions += len(self._entries)
            
            self._emit_event(CacheEvent.EVICTION, "*")
            return True
    
    def get_entries(self) -> List[Dict[str, Any]]:
        """Get all cache entries"""
        with self._lock:
            return [entry.to_dict() for entry in self._entries.values() if not entry.is_expired()]
    
    def _calculate_entry_size(self, value: Any) -> int:
        """Calculate approximate size of cache entry"""
        try:
            # For basic objects, use sys.getsizeof
            if isinstance(value, (str, int, float, bool, type(None))):
                return self._size_calculator(value)
            elif isinstance(value, (list, tuple)):
                size = self._size_calculator(value)
                for item in value:
                    size += self._calculate_entry_size(item)
                return size
            elif isinstance(value, dict):
                size = self._size_calculator(value)
                for k, v in value.items():
                    size += self._calculate_entry_size(k)
                    size += self._calculate_entry_size(v)
                return size
            else:
                # For complex objects, use a heuristic
                return max(self._size_calculator(value), 1024)  # Minimum 1KB
        except Exception:
            return 1024  # Default size
    
    def _update_access_tracking(self, key: str, entry: CacheEntry):
        """Update access tracking for LRU/LFU strategies"""
        if self.strategy == CacheStrategy.LRU:
            # Update LRU order
            if key in self._access_order:
                del self._access_order[key]
            self._access_order[key] = entry
        elif self.strategy == CacheStrategy.LFU:
            # Update frequency count
            self._frequency_order[key] += 1
    
    def _remove_from_tracking(self, key: str):
        """Remove key from access tracking"""
        if key in self._access_order:
            del self._access_order[key]
        if key in self._frequency_order:
            del self._frequency_order[key]
    
    def _evict_entry(self) -> bool:
        """Evict an entry based on strategy"""
        if not self._entries:
            return False
        
        if self.strategy == CacheStrategy.LRU:
            # Evict least recently used
            key = next(iter(self._access_order))
        elif self.strategy == CacheStrategy.LFU:
            # Evict least frequently used
            key = min(self._frequency_order.keys(), 
                     key=lambda k: self._frequency_order[k])
        elif self.strategy == CacheStrategy.FIFO:
            # Evict first inserted (approximation using creation time)
            oldest_entry = min(self._entries.values(), key=lambda e: e.created_at)
            key = oldest_entry.key
        elif self.strategy == CacheStrategy.TTL:
            # Evict oldest expired, or least recently used if none expired
            expired_entries = [e for e in self._entries.values() if e.is_expired()]
            if expired_entries:
                key = min(expired_entries, key=lambda e: e.created_at).key
            else:
                key = next(iter(self._access_order))
        else:  # ADAPTIVE or default
            # Use LRU as fallback
            key = next(iter(self._access_order))
        
        self._remove_entry(key)
        return True
    
    def _remove_entry(self, key: str):
        """Remove entry by key"""
        entry = self._entries.pop(key, None)
        if entry:
            self._remove_from_tracking(key)
            self.stats.total_size_bytes -= entry.size_bytes
            self.stats.entry_count -= 1
            self.stats.evictions += 1
            self._emit_event(CacheEvent.EVICTION, key, entry.value)


class CacheWarmer:
    """Cache warming utility"""
    
    def __init__(self, cache: BaseCache):
        self.cache = cache
        self.logger = logging.getLogger(__name__)
    
    def warm_cache(self, 
                   data_fetcher: Callable[[str], Any],
                   keys: List[str],
                   tags: Optional[Set[str]] = None,
                   ttl: Optional[int] = None,
                   batch_size: int = 100) -> Dict[str, Any]:
        """Warm cache with data"""
        start_time = time.time()
        warmed_count = 0
        failed_keys = []
        
        # Process in batches to avoid overwhelming the data source
        for i in range(0, len(keys), batch_size):
            batch = keys[i:i + batch_size]
            
            for key in batch:
                try:
                    data = data_fetcher(key)
                    if data is not None:
                        success = self.cache.set(key, data, ttl=ttl, tags=tags, warmed=True)
                        if success:
                            warmed_count += 1
                        else:
                            failed_keys.append(key)
                    else:
                        failed_keys.append(key)
                except Exception as e:
                    self.logger.error(f"Failed to warm cache for key {key}: {e}")
                    failed_keys.append(key)
        
        duration = time.time() - start_time
        
        return {
            'total_keys': len(keys),
            'warmed_count': warmed_count,
            'failed_count': len(failed_keys),
            'failed_keys': failed_keys,
            'duration_seconds': duration,
            'rate_per_second': warmed_count / max(duration, 0.1)
        }
    
    def schedule_warming(self,
                        data_fetcher: Callable[[str], Any],
                        keys: List[str],
                        interval_seconds: int = 300,  # 5 minutes
                        **kwargs) -> 'WarmingTask':
        """Schedule periodic cache warming"""
        return WarmingTask(self, data_fetcher, keys, interval_seconds, **kwargs)


class WarmingTask:
    """Scheduled cache warming task"""
    
    def __init__(self, warmer: CacheWarmer, data_fetcher: Callable,
                 keys: List[str], interval_seconds: int, **kwargs):
        self.warmer = warmer
        self.data_fetcher = data_fetcher
        self.keys = keys
        self.interval_seconds = interval_seconds
        self.kwargs = kwargs
        self._running = False
        self._thread = None
    
    def start(self):
        """Start the warming task"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(
            target=self._warming_loop,
            daemon=True
        )
        self._thread.start()
        self.warmer.logger.info(f"Started cache warming task for {len(self.keys)} keys")
    
    def stop(self):
        """Stop the warming task"""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self.warmer.logger.info("Stopped cache warming task")
    
    def _warming_loop(self):
        """Main warming loop"""
        while self._running:
            try:
                result = self.warmer.warm_cache(self.data_fetcher, self.keys, **self.kwargs)
                self.warmer.logger.debug(f"Cache warming completed: {result['warmed_count']} keys warmed")
            except Exception as e:
                self.warmer.logger.error(f"Error in cache warming loop: {e}")
            
            time.sleep(self.interval_seconds)


class CacheInvalidator:
    """Cache invalidation utility"""
    
    def __init__(self, cache: BaseCache):
        self.cache = cache
        self.logger = logging.getLogger(__name__)
    
    def invalidate_by_key(self, key: str) -> bool:
        """Invalidate cache by key"""
        return self.cache.delete(key)
    
    def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate cache entries by tag"""
        invalidated_count = 0
        
        if hasattr(self.cache, 'get_entries'):
            entries = self.cache.get_entries()
            for entry in entries:
                if tag in entry['tags']:
                    if self.cache.delete(entry['key']):
                        invalidated_count += 1
        
        self.logger.info(f"Invalidated {invalidated_count} cache entries with tag: {tag}")
        return invalidated_count
    
    def invalidate_by_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern"""
        import fnmatch
        invalidated_count = 0
        
        if hasattr(self.cache, 'get_entries'):
            entries = self.cache.get_entries()
            for entry in entries:
                if fnmatch.fnmatch(entry['key'], pattern):
                    if self.cache.delete(entry['key']):
                        invalidated_count += 1
        
        self.logger.info(f"Invalidated {invalidated_count} cache entries matching pattern: {pattern}")
        return invalidated_count
    
    def invalidate_expired(self) -> int:
        """Invalidate all expired cache entries"""
        invalidated_count = 0
        
        if hasattr(self.cache, 'get_entries'):
            entries = self.cache.get_entries()
            for entry in entries:
                if entry.get('is_expired', False):
                    if self.cache.delete(entry['key']):
                        invalidated_count += 1
        
        self.logger.info(f"Invalidated {invalidated_count} expired cache entries")
        return invalidated_count


class CacheManager:
    """Multi-level cache manager"""
    
    def __init__(self, 
                 cache_levels: Optional[List[Tuple[CacheLevel, Dict[str, Any]]]] = None,
                 strategy: CacheStrategy = CacheStrategy.ADAPTIVE):
        self.logger = logging.getLogger(__name__)
        self.strategy = strategy
        self._caches = {}  # level -> cache
        self._cache_levels = cache_levels or [
            (CacheLevel.L1_MEMORY, {'max_size': 1000, 'max_memory_mb': 100})
        ]
        
        # Initialize caches
        self._initialize_caches()
        
        # Warming and invalidation
        self._warmers = {}  # level -> CacheWarmer
        self._invalidators = {}  # level -> CacheInvalidator
        
        for level in self._caches.keys():
            self._warmers[level] = CacheWarmer(self._caches[level])
            self._invalidators[level] = CacheInvalidator(self._caches[level])
    
    def _initialize_caches(self):
        """Initialize cache instances"""
        for level, config in self._cache_levels:
            if level == CacheLevel.L1_MEMORY:
                cache = MemoryCache(
                    name=f"memory_cache_{len(self._caches)}",
                    max_size=config.get('max_size', 1000),
                    max_memory_mb=config.get('max_memory_mb', 100),
                    strategy=self.strategy
                )
            else:
                # TODO: Implement other cache levels (Redis, Disk, Database)
                self.logger.warning(f"Cache level {level} not yet implemented, using memory cache")
                cache = MemoryCache(
                    name=f"memory_cache_{level.value}_{len(self._caches)}",
                    max_size=config.get('max_size', 1000),
                    max_memory_mb=config.get('max_memory_mb', 100),
                    strategy=self.strategy
                )
            
            self._caches[level] = cache
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (check all levels)"""
        # Check from highest to lowest level
        for level in [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS, CacheLevel.L3_DISK, CacheLevel.L4_DATABASE]:
            if level in self._caches:
                value = self._caches[level].get(key)
                if value is not None:
                    # Cache hit! Update stats for this level
                    return value
        
        return None
    
    def set(self, key: str, value: Any, 
            levels: Optional[List[CacheLevel]] = None,
            **kwargs) -> bool:
        """Set value in cache (specified levels)"""
        target_levels = levels or list(self._caches.keys())
        success_count = 0
        
        for level in target_levels:
            if level in self._caches:
                if self._caches[level].set(key, value, **kwargs):
                    success_count += 1
        
        return success_count > 0
    
    def delete(self, key: str, 
               levels: Optional[List[CacheLevel]] = None) -> bool:
        """Delete value from cache (specified levels)"""
        target_levels = levels or list(self._caches.keys())
        success_count = 0
        
        for level in target_levels:
            if level in self._caches:
                if self._caches[level].delete(key):
                    success_count += 1
        
        return success_count > 0
    
    def clear(self, level: Optional[CacheLevel] = None) -> bool:
        """Clear cache (specific level or all levels)"""
        if level:
            if level in self._caches:
                return self._caches[level].clear()
            return False
        else:
            # Clear all levels
            success = True
            for cache in self._caches.values():
                if not cache.clear():
                    success = False
            return success
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all cache levels"""
        return {
            level: cache.get_stats().to_dict()
            for level, cache in self._caches.items()
        }
    
    def get_cache(self, level: CacheLevel) -> Optional[BaseCache]:
        """Get specific cache level"""
        return self._caches.get(level)
    
    def get_warmer(self, level: CacheLevel) -> Optional[CacheWarmer]:
        """Get cache warmer for specific level"""
        return self._warmers.get(level)
    
    def get_invalidator(self, level: CacheLevel) -> Optional[CacheInvalidator]:
        """Get cache invalidator for specific level"""
        return self._invalidators.get(level)
    
    def warm_cache(self, 
                   data_fetcher: Callable[[str], Any],
                   keys: List[str],
                   level: Optional[CacheLevel] = None,
                   **kwargs) -> Dict[str, Any]:
        """Warm cache with data"""
        if level:
            warmer = self.get_warmer(level)
            if warmer:
                return warmer.warm_cache(data_fetcher, keys, **kwargs)
            else:
                return {'error': f'No warmer for level {level}'}
        else:
            # Warm all levels
            results = {}
            for level_key, warmer in self._warmers.items():
                try:
                    results[level_key.value] = warmer.warm_cache(data_fetcher, keys, **kwargs)
                except Exception as e:
                    results[level_key.value] = {'error': str(e)}
            return results
    
    def invalidate_cache(self, 
                        invalidation_type: str,
                        pattern: str,
                        level: Optional[CacheLevel] = None) -> Dict[str, Any]:
        """Invalidate cache entries"""
        results = {}
        
        if level:
            invalidator = self.get_invalidator(level)
            if invalidator:
                if invalidation_type == 'tag':
                    count = invalidator.invalidate_by_tag(pattern)
                elif invalidation_type == 'pattern':
                    count = invalidator.invalidate_by_pattern(pattern)
                elif invalidation_type == 'expired':
                    count = invalidator.invalidate_expired()
                else:
                    return {'error': f'Invalid invalidation type: {invalidation_type}'}
                
                results[level.value] = {'invalidated_count': count}
            else:
                results[level.value] = {'error': f'No invalidator for level {level}'}
        else:
            # Invalidate all levels
            for level_key, invalidator in self._invalidators.items():
                try:
                    if invalidation_type == 'tag':
                        count = invalidator.invalidate_by_tag(pattern)
                    elif invalidation_type == 'pattern':
                        count = invalidator.invalidate_by_pattern(pattern)
                    elif invalidation_type == 'expired':
                        count = invalidator.invalidate_expired()
                    else:
                        results[level_key.value] = {'error': f'Invalid invalidation type: {invalidation_type}'}
                        continue
                    
                    results[level_key.value] = {'invalidated_count': count}
                except Exception as e:
                    results[level_key.value] = {'error': str(e)}
        
        return results
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive cache performance report"""
        stats = self.get_stats()
        
        # Calculate aggregate statistics
        total_hits = sum(stat['hits'] for stat in stats.values())
        total_misses = sum(stat['misses'] for stat in stats.values())
        total_operations = sum(stat['total_operations'] for stat in stats.values())
        
        overall_hit_ratio = total_hits / max(total_hits + total_misses, 1)
        
        # Identify best performing level
        best_level = None
        best_hit_ratio = 0
        
        for level, stat in stats.items():
            level_hit_ratio = stat['hit_ratio']
            if level_hit_ratio > best_hit_ratio:
                best_hit_ratio = level_hit_ratio
                best_level = level
        
        # Recommendations
        recommendations = []
        
        if overall_hit_ratio < 0.8:
            recommendations.append("Consider increasing cache sizes or improving cache warming strategy")
        
        if total_misses > total_hits:
            recommendations.append("High miss rate detected - review cache key strategy and data patterns")
        
        # Check for level-specific issues
        for level, stat in stats.items():
            if stat['hit_ratio'] < 0.5:
                recommendations.append(f"Low hit ratio on {level.value} - consider adjusting cache configuration")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_performance': {
                'total_hits': total_hits,
                'total_misses': total_misses,
                'overall_hit_ratio': overall_hit_ratio,
                'total_operations': total_operations
            },
            'level_performance': stats,
            'best_performing_level': best_level.value if best_level else None,
            'best_hit_ratio': best_hit_ratio,
            'recommendations': recommendations
        }


# Decorators for cache usage
def cached(cache_manager: CacheManager, 
           key_prefix: str = "",
           ttl: Optional[int] = None,
           tags: Optional[Set[str]] = None):
    """Decorator for automatic caching of function results"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key_parts = [key_prefix, func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(key_parts)
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            try:
                result = func(*args, **kwargs)
                cache_manager.set(cache_key, result, ttl=ttl, tags=tags)
                return result
            except Exception as e:
                # Don't cache exceptions
                raise
        
        # Add cache management methods
        wrapper.cache_clear = lambda: cache_manager.delete(cache_key)
        wrapper.cache_key = cache_key
        wrapper.cache_manager = cache_manager
        
        return wrapper
    return decorator


@contextmanager
def cache_context(cache_manager: CacheManager, key: str, **kwargs):
    """Context manager for cache-aware operations"""
    # Try to get from cache
    cached_value = cache_manager.get(key)
    
    if cached_value is not None:
        yield cached_value, True  # Second value indicates cache hit
    else:
        # No cache hit, yield None and cache later
        yield None, False


# Global cache manager instance
_cache_manager = None

def get_cache_manager() -> CacheManager:
    """Get global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def initialize_cache_manager(config: Dict[str, Any]) -> CacheManager:
    """Initialize cache manager with configuration"""
    cache_levels = []
    
    # Configure L1 Memory cache
    if 'memory_cache' in config:
        memory_config = config['memory_cache']
        cache_levels.append((
            CacheLevel.L1_MEMORY,
            {
                'max_size': memory_config.get('max_size', 1000),
                'max_memory_mb': memory_config.get('max_memory_mb', 100)
            }
        ))
    
    # Configure other cache levels...
    # (Redis, Disk, Database implementations would go here)
    
    manager = CacheManager(
        cache_levels=cache_levels,
        strategy=CacheStrategy(config.get('strategy', 'adaptive'))
    )
    
    return manager