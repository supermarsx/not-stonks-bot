"""
Memory Optimizer and Garbage Collection

Comprehensive memory management with profiling, object pooling,
and memory leak detection for the trading system.
"""

import gc
import time
import logging
import sys
import tracemalloc
import weakref
import threading
import psutil
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set, Type
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
from contextlib import contextmanager
import functools
import inspect
import types


class MemoryMetric(Enum):
    """Memory metric types"""
    RSS = "rss"  # Resident Set Size
    VMS = "vms"  # Virtual Memory Size
    PERCENT = "percent"  # Memory usage percentage
    AVAILABLE = "available"
    USED = "used"
    TOTAL = "total"
    HEAP = "heap"  # Python heap size
    ALLOCATED = "allocated"  # Current allocations


class OptimizationStrategy(Enum):
    """Memory optimization strategies"""
    OBJECT_POOLING = "object_pooling"
    GARBAGE_COLLECTION = "garbage_collection"
    MEMORY_MAPPING = "memory_mapping"
    COMPRESSION = "compression"
    CACHING = "caching"
    STREAMING = "streaming"


@dataclass
class MemorySnapshot:
    """Memory usage snapshot"""
    timestamp: datetime
    rss_mb: float
    vms_mb: float
    percent: float
    available_mb: float
    used_mb: float
    total_mb: float
    heap_allocated: int
    heap_inuse: int
    gc_collections: Dict[int, int]  # GC generation -> collections count
    top_allocations: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ObjectPoolEntry:
    """Object pool entry"""
    object_type: Type
    creation_time: datetime
    last_used: datetime
    usage_count: int = 0
    total_time_active: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ObjectPool:
    """Reusable object pool to reduce allocation overhead"""
    
    def __init__(self, object_factory: Callable, max_size: int = 100, 
                 max_age_seconds: int = 3600):
        self.object_factory = object_factory
        self.max_size = max_size
        self.max_age_seconds = max_age_seconds
        self._pool = deque()
        self._creation_lock = threading.Lock()
        self._stats = {
            'created': 0,
            'reused': 0,
            'evicted': 0,
            'active_objects': 0
        }
        self.logger = logging.getLogger(__name__)
    
    def get(self) -> Any:
        """Get object from pool"""
        now = datetime.now()
        
        # Try to get existing object
        while self._pool:
            entry, obj = self._pool.popleft()
            
            # Check if object is still valid
            if (now - entry.last_used).total_seconds() < self.max_age_seconds:
                # Reuse the object
                entry.last_used = now
                entry.usage_count += 1
                self._stats['reused'] += 1
                self._stats['active_objects'] += 1
                
                # Reset object state if needed
                if hasattr(obj, 'reset'):
                    obj.reset()
                
                return obj
            else:
                # Object too old, skip it
                self._stats['evicted'] += 1
        
        # Create new object
        with self._creation_lock:
            obj = self.object_factory()
            self._stats['created'] += 1
            self._stats['active_objects'] += 1
        
        return obj
    
    def return_object(self, obj: Any):
        """Return object to pool"""
        if len(self._pool) >= self.max_size:
            # Pool is full, discard oldest object
            try:
                old_entry, old_obj = self._pool.popleft()
                self._stats['evicted'] += 1
            except IndexError:
                pass
        
        # Add object to pool
        entry = ObjectPoolEntry(
            object_type=type(obj),
            creation_time=datetime.now(),
            last_used=datetime.now()
        )
        
        self._pool.append((entry, obj))
        self._stats['active_objects'] -= 1
    
    def clear(self):
        """Clear all objects from pool"""
        while self._pool:
            try:
                entry, obj = self._pool.popleft()
                self._stats['evicted'] += 1
                
                # Cleanup object if needed
                if hasattr(obj, 'cleanup'):
                    obj.cleanup()
            except Exception as e:
                self.logger.error(f"Error clearing object from pool: {e}")
        
        self._stats['active_objects'] = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        return {
            **self._stats,
            'pool_size': len(self._pool),
            'max_size': self.max_size,
            'utilization': len(self._pool) / max(self.max_size, 1)
        }


class MemoryProfiler:
    """Memory profiling and analysis"""
    
    def __init__(self, sample_interval: float = 1.0, max_snapshots: int = 1000):
        self.sample_interval = sample_interval
        self.max_snapshots = max_snapshots
        self.snapshots = deque(maxlen=max_snapshots)
        self.is_profiling = False
        self._profiling_thread = None
        self._stop_profiling = False
        self._thresholds = {
            MemoryMetric.RSS: 1024,  # 1GB in MB
            MemoryMetric.PERCENT: 80.0,
            MemoryMetric.HEAP: 500 * 1024 * 1024  # 500MB in bytes
        }
        self._alert_callbacks = []
        self.logger = logging.getLogger(__name__)
        
        # Start tracemalloc for detailed memory tracking
        if not tracemalloc.is_tracing():
            tracemalloc.start(25)  # Capture 25 stack frames
    
    def start_profiling(self):
        """Start memory profiling"""
        if self.is_profiling:
            return
        
        self.is_profiling = True
        self._stop_profiling = False
        self._profiling_thread = threading.Thread(
            target=self._profiling_loop,
            daemon=True
        )
        self._profiling_thread.start()
        self.logger.info("Started memory profiling")
    
    def stop_profiling(self):
        """Stop memory profiling"""
        self.is_profiling = False
        self._stop_profiling = True
        
        if self._profiling_thread and self._profiling_thread.is_alive():
            self._profiling_thread.join(timeout=5)
        
        self.logger.info("Stopped memory profiling")
    
    def _profiling_loop(self):
        """Main profiling loop"""
        while not self._stop_profiling:
            try:
                snapshot = self._take_memory_snapshot()
                self.snapshots.append(snapshot)
                
                # Check thresholds
                self._check_memory_thresholds(snapshot)
                
                time.sleep(self.sample_interval)
            except Exception as e:
                self.logger.error(f"Error in memory profiling loop: {e}")
                time.sleep(1)
    
    def _take_memory_snapshot(self) -> MemorySnapshot:
        """Take current memory snapshot"""
        # System memory
        memory = psutil.virtual_memory()
        
        # Process memory
        process = psutil.Process()
        process_memory = process.memory_info()
        
        # Tracemalloc snapshot
        current, peak = tracemalloc.get_traced_memory()
        
        # GC statistics
        gc_collections = {
            0: gc.get_count()[0],
            1: gc.get_count()[1],
            2: gc.get_count()[2]
        }
        
        # Top allocations
        top_allocations = []
        try:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            
            for stat in top_stats[:10]:  # Top 10 allocations
                top_allocations.append({
                    'filename': stat.traceback.filename,
                    'lineno': stat.traceback.lineno,
                    'size_mb': stat.size / (1024 * 1024),
                    'count': stat.count
                })
        except Exception as e:
            self.logger.debug(f"Failed to get top allocations: {e}")
        
        return MemorySnapshot(
            timestamp=datetime.now(),
            rss_mb=process_memory.rss / (1024 * 1024),
            vms_mb=process_memory.vms / (1024 * 1024),
            percent=memory.percent,
            available_mb=memory.available / (1024 * 1024),
            used_mb=memory.used / (1024 * 1024),
            total_mb=memory.total / (1024 * 1024),
            heap_allocated=current,
            heap_inuse=current,  # Simplified
            gc_collections=gc_collections,
            top_allocations=top_allocations
        )
    
    def _check_memory_thresholds(self, snapshot: MemorySnapshot):
        """Check memory usage against thresholds"""
        alerts = []
        
        if snapshot.rss_mb > self._thresholds[MemoryMetric.RSS]:
            alerts.append(f"High RSS memory usage: {snapshot.rss_mb:.2f} MB")
        
        if snapshot.percent > self._thresholds[MemoryMetric.PERCENT]:
            alerts.append(f"High memory percentage: {snapshot.percent:.2f}%")
        
        if snapshot.heap_allocated > self._thresholds[MemoryMetric.HEAP]:
            alerts.append(f"High heap allocation: {snapshot.heap_allocated / (1024*1024):.2f} MB")
        
        # Trigger alerts
        for alert in alerts:
            for callback in self._alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Error in alert callback: {e}")
        
        if alerts:
            self.logger.warning(f"Memory threshold alerts: {', '.join(alerts)}")
    
    def add_alert_callback(self, callback: Callable[[str], None]):
        """Add memory threshold alert callback"""
        self._alert_callbacks.append(callback)
    
    def get_current_memory(self) -> Optional[MemorySnapshot]:
        """Get current memory snapshot"""
        if self.snapshots:
            return self.snapshots[-1]
        return None
    
    def get_memory_trend(self, minutes: int = 60) -> Dict[str, Any]:
        """Get memory usage trend"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_snapshots = [
            snapshot for snapshot in self.snapshots 
            if snapshot.timestamp >= cutoff_time
        ]
        
        if not recent_snapshots:
            return {}
        
        rss_values = [s.rss_mb for s in recent_snapshots]
        percent_values = [s.percent for s in recent_snapshots]
        
        return {
            'time_range': f"{minutes} minutes",
            'samples': len(recent_snapshots),
            'rss': {
                'current': rss_values[-1],
                'min': min(rss_values),
                'max': max(rss_values),
                'avg': sum(rss_values) / len(rss_values),
                'trend': 'increasing' if rss_values[-1] > rss_values[0] * 1.1 else 'stable'
            },
            'percent': {
                'current': percent_values[-1],
                'min': min(percent_values),
                'max': max(percent_values),
                'avg': sum(percent_values) / len(percent_values)
            }
        }
    
    def get_memory_leak_analysis(self) -> Dict[str, Any]:
        """Analyze potential memory leaks"""
        if len(self.snapshots) < 10:
            return {'message': 'Insufficient data for leak analysis'}
        
        # Compare memory usage over time
        recent_snapshots = list(self.snapshots)[-10:]
        old_snapshots = list(self.snapshots)[:10]
        
        recent_avg_rss = sum(s.rss_mb for s in recent_snapshots) / len(recent_snapshots)
        old_avg_rss = sum(s.rss_mb for s in old_snapshots) / len(old_snapshots)
        
        memory_growth = recent_avg_rss - old_avg_rss
        growth_rate = (memory_growth / old_avg_rss) * 100 if old_avg_rss > 0 else 0
        
        # Analyze GC effectiveness
        gc_effectiveness = []
        for i in range(1, min(len(self.snapshots), 50)):
            current = self.snapshots[-i]
            previous = self.snapshots[-i-1]
            
            rss_change = current.rss_mb - previous.rss_mb
            gc_runs = (current.gc_collections[0] - previous.gc_collections[0] +
                      current.gc_collections[1] - previous.gc_collections[1] +
                      current.gc_collections[2] - previous.gc_collections[2])
            
            if rss_change < -10:  # Significant memory decrease
                gc_effectiveness.append(gc_runs / max(abs(rss_change), 1))
        
        avg_gc_effectiveness = sum(gc_effectiveness) / len(gc_effectiveness) if gc_effectiveness else 0
        
        # Determine leak risk
        leak_risk = 'low'
        if growth_rate > 50:  # More than 50% growth
            leak_risk = 'high'
        elif growth_rate > 20:  # More than 20% growth
            leak_risk = 'medium'
        
        return {
            'memory_growth_mb': memory_growth,
            'growth_rate_percent': growth_rate,
            'leak_risk': leak_risk,
            'gc_effectiveness': avg_gc_effectiveness,
            'recommendations': self._get_leak_recommendations(growth_rate, leak_risk)
        }
    
    def _get_leak_recommendations(self, growth_rate: float, leak_risk: str) -> List[str]:
        """Get memory leak mitigation recommendations"""
        recommendations = []
        
        if leak_risk == 'high':
            recommendations.extend([
                "Consider immediate memory cleanup",
                "Check for circular references",
                "Review object lifetime management",
                "Enable more aggressive garbage collection"
            ])
        elif leak_risk == 'medium':
            recommendations.extend([
                "Monitor memory usage closely",
                "Consider implementing object pooling",
                "Review cache size limits"
            ])
        
        if growth_rate > 10:
            recommendations.append("Increase garbage collection frequency")
        
        if not recommendations:
            recommendations.append("Memory usage appears stable")
        
        return recommendations


class GarbageCollectionManager:
    """Advanced garbage collection management"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._gc_thresholds = gc.get_threshold()
        self._gc_stats = {
            'collections': {0: 0, 1: 0, 2: 0},
            'collected_objects': {0: 0, 1: 0, 2: 0},
            'uncollectable_objects': 0
        }
        self._monitoring_enabled = False
        self._gc_thread = None
        self._stop_monitoring = False
    
    def set_gc_thresholds(self, generation0: int, generation1: int, generation2: int):
        """Set garbage collection thresholds"""
        gc.set_threshold(generation0, generation1, generation2)
        self._gc_thresholds = (generation0, generation1, generation2)
        self.logger.info(f"Set GC thresholds to ({generation0}, {generation1}, {generation2})")
    
    def get_gc_stats(self) -> Dict[str, Any]:
        """Get garbage collection statistics"""
        counts = gc.get_count()
        stats = gc.get_stats()
        uncollectable = gc.garbage
        
        return {
            'thresholds': self._gc_thresholds,
            'current_counts': {0: counts[0], 1: counts[1], 2: counts[2]},
            'statistics': stats,
            'uncollectable_objects': len(uncollectable),
            'total_uncollectable': self._gc_stats['uncollectable_objects']
        }
    
    def force_collection(self, generation: int = 2) -> Dict[str, int]:
        """Force garbage collection for specific generation"""
        if generation < 0 or generation > 2:
            raise ValueError("Generation must be 0, 1, or 2")
        
        start_time = time.time()
        collected = gc.collect(generation)
        duration = time.time() - start_time
        
        self._gc_stats['collections'][generation] += 1
        self._gc_stats['collected_objects'][generation] += collected
        
        result = {
            'generation': generation,
            'collected_objects': collected,
            'duration_ms': duration * 1000,
            'current_counts': gc.get_count()
        }
        
        self.logger.debug(f"Forced GC generation {generation}: collected {collected} objects in {duration*1000:.2f}ms")
        return result
    
    def aggressive_gc(self) -> Dict[str, Any]:
        """Run aggressive garbage collection across all generations"""
        results = {}
        total_collected = 0
        
        for gen in [2, 1, 0]:  # Start with highest generation
            result = self.force_collection(gen)
            results[f'generation_{gen}'] = result
            total_collected += result['collected_objects']
        
        return {
            'total_collected': total_collected,
            'generation_results': results,
            'gc_stats': self.get_gc_stats()
        }
    
    def start_gc_monitoring(self, interval: int = 60):
        """Start automatic garbage collection monitoring"""
        if self._monitoring_enabled:
            return
        
        self._monitoring_enabled = True
        self._stop_monitoring = False
        self._gc_thread = threading.Thread(
            target=self._gc_monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self._gc_thread.start()
        self.logger.info("Started garbage collection monitoring")
    
    def stop_gc_monitoring(self):
        """Stop garbage collection monitoring"""
        self._monitoring_enabled = False
        self._stop_monitoring = True
        
        if self._gc_thread and self._gc_thread.is_alive():
            self._gc_thread.join(timeout=5)
        
        self.logger.info("Stopped garbage collection monitoring")
    
    def _gc_monitoring_loop(self, interval: int):
        """Garbage collection monitoring loop"""
        while not self._stop_monitoring:
            try:
                # Check memory usage
                process = psutil.Process()
                memory_info = process.memory_info()
                memory_percent = process.memory_percent()
                
                # Force GC if memory usage is high
                if memory_percent > 80:
                    self.logger.warning(f"High memory usage ({memory_percent:.1f}%), running aggressive GC")
                    result = self.aggressive_gc()
                    self.logger.info(f"Aggressive GC completed: collected {result['total_collected']} objects")
                
                # Check for uncollectable objects
                uncollectable = gc.garbage
                if len(uncollectable) > 0:
                    self.logger.warning(f"Found {len(uncollectable)} uncollectable objects")
                    self._gc_stats['uncollectable_objects'] += len(uncollectable)
                
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"Error in GC monitoring loop: {e}")
                time.sleep(10)
    
    def optimize_gc_for_performance(self):
        """Optimize GC settings for performance"""
        # Adjust thresholds based on typical usage patterns
        # These are conservative defaults that can be tuned
        gc.set_threshold(700, 10, 10)  # More frequent collection
        self._gc_thresholds = (700, 10, 10)
        self.logger.info("Optimized GC thresholds for performance")
    
    def optimize_gc_for_memory(self):
        """Optimize GC settings for memory efficiency"""
        # Adjust thresholds for better memory usage
        gc.set_threshold(1000, 100, 100)  # Less frequent collection
        self._gc_thresholds = (1000, 100, 100)
        self.logger.info("Optimized GC thresholds for memory efficiency")
    
    def analyze_gc_performance(self) -> Dict[str, Any]:
        """Analyze garbage collection performance"""
        stats = gc.get_stats()
        counts = gc.get_count()
        
        # Calculate GC efficiency
        total_collections = sum(self._gc_stats['collections'].values())
        total_collected = sum(self._gc_stats['collected_objects'].values())
        
        efficiency_metrics = {}
        for gen in range(3):
            collections = self._gc_stats['collections'][gen]
            collected = self._gc_stats['collected_objects'][gen]
            efficiency_metrics[f'generation_{gen}'] = {
                'collections': collections,
                'objects_collected': collected,
                'avg_objects_per_collection': collected / max(collections, 1)
            }
        
        return {
            'total_collections': total_collections,
            'total_objects_collected': total_collected,
            'efficiency_by_generation': efficiency_metrics,
            'current_counts': counts,
            'uncollectable_objects': len(gc.garbage),
            'recommendations': self._get_gc_recommendations()
        }
    
    def _get_gc_recommendations(self) -> List[str]:
        """Get GC optimization recommendations"""
        recommendations = []
        
        # Check current thresholds
        current_counts = gc.get_count()
        thresholds = self._gc_thresholds
        
        # If collections are happening too frequently
        if current_counts[0] < thresholds[0] * 0.5:
            recommendations.append("Consider reducing generation 0 threshold for more frequent collection")
        elif current_counts[0] > thresholds[0] * 0.9:
            recommendations.append("Generation 0 threshold may be too low")
        
        # Check uncollectable objects
        if len(gc.garbage) > 0:
            recommendations.append("Address uncollectable objects to prevent memory leaks")
        
        # General recommendations
        recommendations.extend([
            "Monitor GC performance regularly",
            "Adjust thresholds based on application patterns",
            "Consider object pooling for frequently allocated objects"
        ])
        
        return recommendations


class MemoryOptimizer:
    """Main memory optimization system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.profiler = MemoryProfiler()
        self.gc_manager = GarbageCollectionManager()
        self._object_pools = {}  # type -> ObjectPool
        self._memory_callbacks = []
        
        # Start monitoring
        self.profiler.start_profiling()
        self.gc_manager.start_gc_monitoring()
        
        # Add memory alert callback
        self.profiler.add_alert_callback(self._handle_memory_alert)
    
    def _handle_memory_alert(self, alert: str):
        """Handle memory threshold alerts"""
        self.logger.warning(f"Memory alert: {alert}")
        
        # Trigger garbage collection
        self.gc_manager.force_collection(2)
        
        # Execute custom callbacks
        for callback in self._memory_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in memory callback: {e}")
    
    def add_memory_callback(self, callback: Callable[[str], None]):
        """Add memory alert callback"""
        self._memory_callbacks.append(callback)
    
    def create_object_pool(self, object_type: Type, max_size: int = 100) -> ObjectPool:
        """Create object pool for specific type"""
        pool = ObjectPool(object_type, max_size)
        self._object_pools[object_type] = pool
        self.logger.info(f"Created object pool for {object_type.__name__}")
        return pool
    
    def get_object_pool(self, object_type: Type) -> Optional[ObjectPool]:
        """Get existing object pool"""
        return self._object_pools.get(object_type)
    
    def optimize_for_high_frequency_trading(self):
        """Optimize memory settings for high-frequency trading"""
        # Optimize GC for performance
        self.gc_manager.optimize_gc_for_performance()
        
        # Enable more aggressive profiling
        self.profiler.sample_interval = 0.5
        
        self.logger.info("Optimized memory settings for high-frequency trading")
    
    def optimize_for_low_latency(self):
        """Optimize memory settings for low-latency operations"""
        # Minimize GC interference
        self.gc_manager.optimize_gc_for_memory()
        
        # Reduce profiling overhead
        self.profiler.sample_interval = 2.0
        
        # Create object pools for common types
        common_types = [dict, list, tuple, str]
        for obj_type in common_types:
            if obj_type not in self._object_pools:
                self.create_object_pool(obj_type, max_size=50)
        
        self.logger.info("Optimized memory settings for low latency")
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive memory optimization report"""
        current_memory = self.profiler.get_current_memory()
        memory_trend = self.profiler.get_memory_trend()
        leak_analysis = self.profiler.get_memory_leak_analysis()
        gc_stats = self.gc_manager.get_gc_stats()
        gc_performance = self.gc_manager.analyze_gc_performance()
        
        pool_stats = {
            obj_type.__name__: pool.get_stats()
            for obj_type, pool in self._object_pools.items()
        }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'current_memory': current_memory.to_dict() if current_memory else None,
            'memory_trend': memory_trend,
            'leak_analysis': leak_analysis,
            'garbage_collection': {
                'stats': gc_stats,
                'performance': gc_performance
            },
            'object_pools': pool_stats,
            'recommendations': self._generate_recommendations(leak_analysis, gc_performance)
        }
    
    def _generate_recommendations(self, leak_analysis: Dict[str, Any], 
                                gc_performance: Dict[str, Any]) -> List[str]:
        """Generate memory optimization recommendations"""
        recommendations = []
        
        # Memory leak recommendations
        if leak_analysis.get('leak_risk') in ['medium', 'high']:
            recommendations.extend([
                "Consider implementing more aggressive memory cleanup",
                "Review object lifetime management",
                "Implement object pooling for frequently allocated objects"
            ])
        
        # GC performance recommendations
        for gen, data in gc_performance.get('efficiency_by_generation', {}).items():
            if data.get('avg_objects_per_collection', 0) < 100:
                recommendations.append(f"Consider adjusting {gen} threshold for better collection efficiency")
        
        # General recommendations
        recommendations.extend([
            "Monitor memory usage trends regularly",
            "Implement memory profiling in production",
            "Consider using memory-mapped files for large datasets",
            "Review cache size limits and expiration policies"
        ])
        
        return list(set(recommendations))  # Remove duplicates
    
    def clear_all_object_pools(self):
        """Clear all object pools"""
        for pool in self._object_pools.values():
            pool.clear()
        
        self.logger.info("Cleared all object pools")
    
    def shutdown(self):
        """Shutdown memory optimizer"""
        self.profiler.stop_profiling()
        self.gc_manager.stop_gc_monitoring()
        self.clear_all_object_pools()
        self.logger.info("Memory optimizer shutdown")


# Decorators for memory optimization
def memory_optimized(func: Callable) -> Callable:
    """Decorator to optimize function memory usage"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Run garbage collection before function
        gc.collect()
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Run garbage collection after function
            gc.collect()
            
            duration = time.time() - start_time
            if duration > 1.0:  # Only log for slow functions
                logging.debug(f"Memory optimized function {func.__name__} took {duration:.3f}s")
    
    return wrapper


def track_memory_usage(func: Callable) -> Callable:
    """Decorator to track memory usage of function"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Start memory tracking
        tracemalloc.start()
        start_snapshot = tracemalloc.take_snapshot()
        start_memory = psutil.Process().memory_info()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # End memory tracking
            end_memory = psutil.Process().memory_info()
            end_snapshot = tracemalloc.take_snapshot()
            
            # Calculate memory differences
            memory_diff = end_memory.rss - start_memory.rss
            
            # Calculate top memory differences
            top_stats = end_snapshot.compare_to(start_snapshot, 'lineno')
            top_diff = sum(stat.size_diff for stat in top_stats[:5])
            
            logging.debug(f"Function {func.__name__} memory usage: {memory_diff / 1024 / 1024:.2f} MB")
            logging.debug(f"Function {func.__name__} allocations: {top_diff / 1024 / 1024:.2f} MB")
            
            tracemalloc.stop()
    
    return wrapper


@contextmanager
def memory_context():
    """Context manager for memory-aware operations"""
    # Force GC before context
    collected = gc.collect()
    
    # Get initial memory
    process = psutil.Process()
    initial_memory = process.memory_info()
    
    try:
        yield initial_memory
    finally:
        # Force GC after context
        new_collected = gc.collect() - collected
        
        final_memory = process.memory_info()
        memory_diff = final_memory.rss - initial_memory.rss
        
        logging.debug(f"Memory context: GC collections={new_collected}, Memory diff={memory_diff / 1024 / 1024:.2f} MB")


# Global memory optimizer instance
_memory_optimizer = None

def get_memory_optimizer() -> MemoryOptimizer:
    """Get global memory optimizer instance"""
    global _memory_optimizer
    if _memory_optimizer is None:
        _memory_optimizer = MemoryOptimizer()
    return _memory_optimizer


def initialize_memory_optimizer() -> MemoryOptimizer:
    """Initialize memory optimizer"""
    return MemoryOptimizer()