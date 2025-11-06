"""
Performance Profiler and Analysis

Comprehensive performance profiling with bottleneck analysis,
performance reports, and recommendations for the trading system.
"""

import time
import cProfile
import pstats
import psutil
import threading
import logging
import json
import functools
import traceback
import inspect
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import io
import sys
import importlib
import gc


class ProfilingType(Enum):
    """Profiling type enumeration"""
    FUNCTION = "function"
    LINE = "line"
    MEMORY = "memory"
    CPU = "cpu"
    I/O = "io"


class MetricType(Enum):
    """Performance metric type enumeration"""
    EXECUTION_TIME = "execution_time"
    CALL_COUNT = "call_count"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    I/O_OPERATIONS = "io_operations"
    THROUGHPUT = "throughput"
    LATENCY = "latency"


class BottleneckType(Enum):
    """Bottleneck type enumeration"""
    CPU_BOUND = "cpu_bound"
    I/O_BOUND = "io_bound"
    MEMORY_BOUND = "memory_bound"
    NETWORK_BOUND = "network_bound"
    LOCK_CONTENTION = "lock_contention"
    SLOW_DATABASE = "slow_database"
    INEFFICIENT_ALGORITHM = "inefficient_algorithm"


@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    metric_type: MetricType
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'value': self.value,
            'unit': self.unit,
            'timestamp': self.timestamp.isoformat(),
            'metric_type': self.metric_type.value,
            'context': self.context
        }


@dataclass
class FunctionProfile:
    """Function profiling result"""
    function_name: str
    module: str
    file_path: str
    line_number: int
    call_count: int
    total_time: float
    cumulative_time: float
    per_call_time: float
    memory_usage_delta: float = 0.0
    cpu_usage_delta: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BottleneckAnalysis:
    """Bottleneck analysis result"""
    bottleneck_type: BottleneckType
    affected_functions: List[str]
    severity: str  # "low", "medium", "high", "critical"
    estimated_impact: float  # Percentage of total performance impact
    description: str
    recommendations: List[str]
    evidence: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PerformanceProfiler:
    """Performance profiling system"""
    
    def __init__(self, sample_rate: float = 1.0, max_samples: int = 10000):
        self.sample_rate = sample_rate
        self.max_samples = max_samples
        self.logger = logging.getLogger(__name__)
        
        # Profiling state
        self._profiling_active = False
        self._profiling_thread = None
        self._stop_profiling = False
        
        # Performance data storage
        self.metrics_history = defaultdict(lambda: deque(maxlen=max_samples))
        self.function_profiles = {}
        self.profiling_stats = None
        
        # Bottleneck detection
        self.bottleneck_threshold_time = 0.1  # 100ms
        self.bottleneck_threshold_memory = 10 * 1024 * 1024  # 10MB
        self.bottleneck_threshold_calls = 1000
        
        # Analysis results
        self.last_analysis = None
        self.bottlenecks_detected = []
        
        # Threading
        self._lock = threading.RLock()
    
    def start_profiling(self):
        """Start performance profiling"""
        if self._profiling_active:
            return
        
        self._profiling_active = True
        self._stop_profiling = False
        
        # Start profiling thread
        self._profiling_thread = threading.Thread(
            target=self._profiling_loop,
            daemon=True
        )
        self._profiling_thread.start()
        
        self.logger.info("Started performance profiling")
    
    def stop_profiling(self):
        """Stop performance profiling"""
        self._profiling_active = False
        self._stop_profiling = True
        
        if self._profiling_thread and self._profiling_thread.is_alive():
            self._profiling_thread.join(timeout=5)
        
        self.logger.info("Stopped performance profiling")
    
    def _profiling_loop(self):
        """Main profiling loop"""
        profiler = cProfile.Profile()
        
        while not self._stop_profiling:
            try:
                # Start profiling period
                profiler.enable()
                
                # Let profiling run for sample interval
                time.sleep(1.0 / self.sample_rate)
                
                # Stop profiling period
                profiler.disable()
                
                # Process profiling results
                self._process_profiling_results(profiler)
                
                # Reset profiler
                profiler = cProfile.Profile()
                
            except Exception as e:
                self.logger.error(f"Error in profiling loop: {e}")
                time.sleep(1)
    
    def _process_profiling_results(self, profiler: cProfile.Profile):
        """Process profiling results"""
        try:
            # Get stats string
            stats_stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stats_stream)
            stats.sort_stats('cumulative')
            
            # Store stats for later analysis
            with self._lock:
                self.profiling_stats = stats
            
            # Extract top functions
            top_functions = stats.get_stats_profile()
            
            with self._lock:
                for func_name, func_stats in top_functions.items():
                    if isinstance(func_stats, tuple):
                        # Handle different pstats versions
                        ncalls, tt, ct, callers = func_stats
                    else:
                        # Use direct attribute access
                        ncalls = func_stats.ncalls
                        tt = func_stats.totaltime
                        ct = func_stats.cumtime
                        callers = func_stats.callers
                    
                    profile = FunctionProfile(
                        function_name=func_name[2],  # function name
                        module=func_name[0],        # module name
                        file_path=func_name[1],     # file path
                        line_number=func_name[3],   # line number
                        call_count=ncalls,
                        total_time=tt,
                        cumulative_time=ct,
                        per_call_time=ct / max(ncalls, 1)
                    )
                    
                    self.function_profiles[func_name] = profile
            
            # Detect bottlenecks
            self._detect_bottlenecks()
            
        except Exception as e:
            self.logger.error(f"Error processing profiling results: {e}")
    
    def _detect_bottlenecks(self):
        """Detect performance bottlenecks"""
        bottlenecks = []
        
        # CPU-bound bottlenecks
        cpu_bottlenecks = self._detect_cpu_bottlenecks()
        bottlenecks.extend(cpu_bottlenecks)
        
        # I/O bottlenecks
        io_bottlenecks = self._detect_io_bottlenecks()
        bottlenecks.extend(io_bottlenecks)
        
        # Memory bottlenecks
        memory_bottlenecks = self._detect_memory_bottlenecks()
        bottlenecks.extend(memory_bottlenecks)
        
        # Database bottlenecks
        db_bottlenecks = self._detect_database_bottlenecks()
        bottlenecks.extend(db_bottlenecks)
        
        # Algorithm inefficiencies
        algo_bottlenecks = self._detect_algorithm_bottlenecks()
        bottlenecks.extend(algo_bottlenecks)
        
        with self._lock:
            self.bottlenecks_detected = bottlenecks
            self.last_analysis = datetime.now()
    
    def _detect_cpu_bottlenecks(self) -> List[BottleneckAnalysis]:
        """Detect CPU-bound bottlenecks"""
        bottlenecks = []
        
        if not self.function_profiles:
            return bottlenecks
        
        # Find functions with high execution time
        total_time = sum(profile.total_time for profile in self.function_profiles.values())
        
        for func_name, profile in self.function_profiles.items():
            if profile.total_time > self.bottleneck_threshold_time:
                impact_percent = (profile.total_time / max(total_time, 0.001)) * 100
                
                # Determine severity
                if impact_percent > 50:
                    severity = "critical"
                elif impact_percent > 20:
                    severity = "high"
                elif impact_percent > 10:
                    severity = "medium"
                else:
                    severity = "low"
                
                recommendations = self._get_cpu_optimization_recommendations(profile)
                
                bottleneck = BottleneckAnalysis(
                    bottleneck_type=BottleneckType.CPU_BOUND,
                    affected_functions=[profile.function_name],
                    severity=severity,
                    estimated_impact=impact_percent,
                    description=f"Function '{profile.function_name}' consumes {impact_percent:.1f}% of total execution time",
                    recommendations=recommendations,
                    evidence={
                        'execution_time': profile.total_time,
                        'call_count': profile.call_count,
                        'per_call_time': profile.per_call_time
                    }
                )
                bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    def _detect_io_bottlenecks(self) -> List[BottleneckAnalysis]:
        """Detect I/O-bound bottlenecks"""
        bottlenecks = []
        
        # Look for I/O related functions
        io_functions = [
            'read', 'write', 'open', 'close', 'socket', 'connect',
            'send', 'recv', 'request', 'get', 'post', 'fetch'
        ]
        
        for func_name, profile in self.function_profiles.items():
            if any(io_func in profile.function_name.lower() for io_func in io_functions):
                if profile.total_time > self.bottleneck_threshold_time:
                    bottleneck = BottleneckAnalysis(
                        bottleneck_type=BottleneckType.I/O_BOUND,
                        affected_functions=[profile.function_name],
                        severity="medium" if profile.total_time > 0.5 else "low",
                        estimated_impact=0,  # Would need more sophisticated analysis
                        description=f"I/O operation '{profile.function_name}' is slow",
                        recommendations=[
                            "Consider implementing I/O batching",
                            "Use asynchronous I/O operations",
                            "Implement connection pooling",
                            "Add caching for frequently accessed data"
                        ],
                        evidence={
                            'execution_time': profile.total_time,
                            'call_count': profile.call_count
                        }
                    )
                    bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    def _detect_memory_bottlenecks(self) -> List[BottleneckAnalysis]:
        """Detect memory-related bottlenecks"""
        bottlenecks = []
        
        # Check for high memory allocation functions
        process = psutil.Process()
        process_memory = process.memory_info()
        
        # Look for functions with high memory usage patterns
        for func_name, profile in self.function_profiles.items():
            # Simple heuristic: functions with many calls might cause memory pressure
            if profile.call_count > self.bottleneck_threshold_calls:
                bottleneck = BottleneckAnalysis(
                    bottleneck_type=BottleneckType.MEMORY_BOUND,
                    affected_functions=[profile.function_name],
                    severity="medium" if profile.call_count > 10000 else "low",
                    estimated_impact=0,
                    description=f"High call frequency in '{profile.function_name}' may cause memory pressure",
                    recommendations=[
                        "Implement object pooling",
                        "Optimize memory allocation patterns",
                        "Add garbage collection hints",
                        "Consider lazy loading for large objects"
                    ],
                    evidence={
                        'call_count': profile.call_count,
                        'execution_time': profile.total_time
                    }
                )
                bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    def _detect_database_bottlenecks(self) -> List[BottleneckAnalysis]:
        """Detect database-related bottlenecks"""
        bottlenecks = []
        
        # Look for database-related functions
        db_functions = [
            'execute', 'query', 'fetch', 'commit', 'rollback',
            'select', 'insert', 'update', 'delete'
        ]
        
        for func_name, profile in self.function_profiles.items():
            if any(db_func in profile.function_name.lower() for db_func in db_functions):
                if profile.total_time > self.bottleneck_threshold_time:
                    bottleneck = BottleneckAnalysis(
                        bottleneck_type=BottleneckType.SLOW_DATABASE,
                        affected_functions=[profile.function_name],
                        severity="high" if profile.total_time > 1.0 else "medium",
                        estimated_impact=0,
                        description=f"Database operation '{profile.function_name}' is slow",
                        recommendations=[
                            "Add database indexes",
                            "Optimize SQL queries",
                            "Implement query result caching",
                            "Use connection pooling",
                            "Consider database query optimization"
                        ],
                        evidence={
                            'execution_time': profile.total_time,
                            'call_count': profile.call_count
                        }
                    )
                    bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    def _detect_algorithm_bottlenecks(self) -> List[BottleneckAnalysis]:
        """Detect algorithmic inefficiencies"""
        bottlenecks = []
        
        # Look for inefficient algorithms
        inefficient_patterns = [
            ('nested_loops', r'for.*for'),
            ('recursion_depth', r'def \w+\(.*\):.*return \w+\('),  # Simple recursive functions
            ('sort_inefficient', r'\.sort\(\)'),
            ('search_inefficient', r'in \w+'),  # Linear search in lists
        ]
        
        for func_name, profile in self.function_profiles.items():
            # Check function source code for inefficient patterns
            try:
                source_file = profile.file_path
                with open(source_file, 'r') as f:
                    lines = f.readlines()
                    if profile.line_number <= len(lines):
                        func_code = ''.join(lines[profile.line_number-1:profile.line_number+10])
                        
                        for pattern_name, pattern in inefficient_patterns:
                            import re
                            if re.search(pattern, func_code, re.IGNORECASE):
                                bottleneck = BottleneckAnalysis(
                                    bottleneck_type=BottleneckType.INEFFICIENT_ALGORITHM,
                                    affected_functions=[profile.function_name],
                                    severity="medium",
                                    estimated_impact=0,
                                    description=f"Potential algorithmic inefficiency: {pattern_name}",
                                    recommendations=self._get_algorithm_optimization_recommendations(pattern_name),
                                    evidence={
                                        'function_code': func_code.strip(),
                                        'execution_time': profile.total_time
                                    }
                                )
                                bottlenecks.append(bottleneck)
                                break
            except Exception:
                # Skip if can't read source file
                continue
        
        return bottlenecks
    
    def _get_cpu_optimization_recommendations(self, profile: FunctionProfile) -> List[str]:
        """Get CPU optimization recommendations"""
        recommendations = []
        
        if profile.per_call_time > 0.1:
            recommendations.extend([
                "Optimize algorithm complexity (O(n²) to O(n log n))",
                "Consider memoization for repeated calculations",
                "Use more efficient data structures"
            ])
        
        if profile.call_count > 1000:
            recommendations.extend([
                "Reduce function call overhead",
                "Consider inline functions or caching",
                "Batch operations where possible"
            ])
        
        if not recommendations:
            recommendations.append("Profile-specific optimization needed")
        
        return recommendations
    
    def _get_algorithm_optimization_recommendations(self, pattern_name: str) -> List[str]:
        """Get algorithm optimization recommendations"""
        recommendations_map = {
            'nested_loops': [
                "Optimize nested loops with early termination",
                "Use more efficient algorithms (e.g., divide and conquer)",
                "Consider parallel processing for independent iterations"
            ],
            'recursion_depth': [
                "Convert to iterative implementation",
                "Use tail recursion optimization",
                "Implement memoization for recursive calls"
            ],
            'sort_inefficient': [
                "Use built-in sort with custom comparator",
                "Consider Timsort for partially sorted data",
                "Parallel sort for large datasets"
            ],
            'search_inefficient': [
                "Use binary search for sorted data",
                "Implement hash table for O(1) lookups",
                "Consider using sets for membership testing"
            ]
        }
        
        return recommendations_map.get(pattern_name, ["Review algorithm implementation"])
    
    def profile_function(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Profile a specific function"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info()
        
        try:
            # Run function with profiling
            profiler = cProfile.Profile()
            profiler.enable()
            
            result = func(*args, **kwargs)
            
            profiler.disable()
            
            # Get profiling stats
            stats_stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stats_stream)
            stats.sort_stats('cumulative')
            
            # Calculate metrics
            end_time = time.time()
            end_memory = psutil.Process().memory_info()
            
            execution_time = end_time - start_time
            memory_usage = end_memory.rss - start_memory.rss
            
            # Extract top functions
            top_functions = []
            for func_info, func_stats in stats.get_stats_profile().items():
                if isinstance(func_stats, tuple):
                    ncalls, tt, ct, callers = func_stats
                else:
                    ncalls = func_stats.ncalls
                    tt = func_stats.totaltime
                    ct = func_stats.cumtime
                
                top_functions.append({
                    'function': func_info[2],
                    'module': func_info[0],
                    'call_count': ncalls,
                    'total_time': tt,
                    'cumulative_time': ct
                })
            
            return {
                'function': func.__name__,
                'execution_time': execution_time,
                'memory_usage_delta': memory_usage,
                'result': result,
                'top_functions': top_functions[:10]  # Top 10 functions
            }
            
        except Exception as e:
            self.logger.error(f"Function profiling failed: {e}")
            return {
                'function': func.__name__,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        with self._lock:
            # Get current profiling stats
            profile_stats = self.profiling_stats
            
            # Calculate summary metrics
            total_functions = len(self.function_profiles)
            total_calls = sum(profile.call_count for profile in self.function_profiles.values())
            total_time = sum(profile.total_time for profile in self.function_profiles.values())
            
            # Find top bottlenecks
            top_bottlenecks = sorted(
                self.bottlenecks_detected,
                key=lambda x: x.estimated_impact,
                reverse=True
            )[:10]
            
            # System performance
            process = psutil.Process()
            system_metrics = {
                'cpu_percent': process.cpu_percent(),
                'memory_mb': process.memory_info().rss / (1024 * 1024),
                'threads': process.num_threads(),
                'open_files': len(process.open_files())
            }
            
            # Memory analysis
            memory_info = psutil.virtual_memory()
            memory_metrics = {
                'total_mb': memory_info.total / (1024 * 1024),
                'used_mb': memory_info.used / (1024 * 1024),
                'available_mb': memory_info.available / (1024 * 1024),
                'usage_percent': memory_info.percent
            }
            
            # Function analysis
            function_analysis = []
            for func_name, profile in self.function_profiles.items():
                function_analysis.append({
                    'function': profile.function_name,
                    'module': profile.module,
                    'call_count': profile.call_count,
                    'total_time': profile.total_time,
                    'cumulative_time': profile.cumulative_time,
                    'per_call_time': profile.per_call_time,
                    'complexity_estimate': self._estimate_complexity(profile)
                })
            
            # Sort by cumulative time
            function_analysis.sort(key=lambda x: x['cumulative_time'], reverse=True)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_functions': total_functions,
                    'total_calls': total_calls,
                    'total_time': total_time,
                    'average_time_per_call': total_time / max(total_calls, 1),
                    'bottlenecks_detected': len(self.bottlenecks_detected)
                },
                'system_metrics': system_metrics,
                'memory_metrics': memory_metrics,
                'function_analysis': function_analysis[:20],  # Top 20 functions
                'bottlenecks': [b.to_dict() for b in top_bottlenecks],
                'recommendations': self._generate_recommendations(),
                'profiling_active': self._profiling_active,
                'last_analysis': self.last_analysis.isoformat() if self.last_analysis else None
            }
    
    def _estimate_complexity(self, profile: FunctionProfile) -> str:
        """Estimate algorithmic complexity from profiling data"""
        if profile.call_count == 0:
            return "unknown"
        
        avg_calls_per_second = profile.call_count / max(profile.total_time, 0.001)
        
        if profile.per_call_time < 0.001 and avg_calls_per_second > 1000:
            return "O(1) or O(log n)"
        elif profile.per_call_time < 0.01:
            return "O(n) - linear"
        elif profile.per_call_time < 0.1:
            return "O(n log n) - linearithmic"
        elif profile.call_count > 1000:
            return "O(n²) - quadratic or worse"
        else:
            return "O(n²) - quadratic"
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        if not self.bottlenecks_detected:
            return ["Performance appears optimal - no bottlenecks detected"]
        
        # Collect recommendations from detected bottlenecks
        for bottleneck in self.bottlenecks_detected:
            recommendations.extend(bottleneck.recommendations)
        
        # Add general recommendations based on system metrics
        process = psutil.Process()
        cpu_percent = process.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        if cpu_percent > 80:
            recommendations.append("High CPU usage detected - consider optimizing CPU-intensive operations")
        
        if memory_percent > 80:
            recommendations.append("High memory usage detected - implement memory optimization strategies")
        
        # Remove duplicates and limit
        recommendations = list(set(recommendations))[:10]
        
        return recommendations if recommendations else ["System performance appears optimal"]
    
    def export_profile_data(self, format_type: str = "json") -> str:
        """Export profiling data in specified format"""
        report = self.get_performance_report()
        
        if format_type.lower() == "json":
            return json.dumps(report, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def clear_profile_data(self):
        """Clear all profile data"""
        with self._lock:
            self.function_profiles.clear()
            self.bottlenecks_detected.clear()
            self.profiling_stats = None
            self.last_analysis = None
        
        self.logger.info("Cleared all profile data")


# Decorators for performance profiling
def profile_function(profiler: PerformanceProfiler = None):
    """Decorator for function performance profiling"""
    if profiler is None:
        profiler = get_profiler()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Profile the function
            profile_result = profiler.profile_function(func, *args, **kwargs)
            
            # Log performance metrics
            execution_time = profile_result.get('execution_time', 0)
            memory_usage = profile_result.get('memory_usage_delta', 0)
            
            profiler.logger.debug(
                f"Profiled {func.__name__}: {execution_time:.3f}s, "
                f"{memory_usage / 1024:.1f}KB"
            )
            
            return profile_result
        
        return wrapper
    return decorator


def profile_performance(threshold_ms: float = 100):
    """Decorator to profile and alert on slow functions"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                execution_time = (time.time() - start_time) * 1000
                end_memory = psutil.Process().memory_info()
                memory_usage = end_memory.rss - start_memory.rss
                
                if execution_time > threshold_ms:
                    logging.warning(
                        f"Slow function detected: {func.__name__} took "
                        f"{execution_time:.1f}ms ({memory_usage / 1024:.1f}KB)"
                    )
        
        return wrapper
    return decorator


@contextmanager
def performance_context(profiler: PerformanceProfiler, context_name: str = "performance_context"):
    """Context manager for performance monitoring"""
    start_time = time.time()
    start_memory = psutil.Process().memory_info()
    
    try:
        yield {
            'start_time': start_time,
            'start_memory': start_memory,
            'context_name': context_name
        }
    finally:
        end_time = time.time()
        end_memory = psutil.Process().memory_info()
        
        execution_time = end_time - start_time
        memory_usage = end_memory.rss - start_memory.rss
        
        logging.debug(
            f"Performance context '{context_name}': "
            f"{execution_time:.3f}s ({memory_usage / 1024:.1f}KB)"
        )


# Global performance profiler instance
_profiler = None

def get_profiler() -> PerformanceProfiler:
    """Get global performance profiler instance"""
    global _profiler
    if _profiler is None:
        _profiler = PerformanceProfiler()
    return _profiler


def initialize_profiler(config: Dict[str, Any]) -> PerformanceProfiler:
    """Initialize performance profiler with configuration"""
    profiler = PerformanceProfiler(
        sample_rate=config.get('sample_rate', 1.0),
        max_samples=config.get('max_samples', 10000)
    )
    
    # Configure thresholds
    profiler.bottleneck_threshold_time = config.get('bottleneck_threshold_time', 0.1)
    profiler.bottleneck_threshold_memory = config.get('bottleneck_threshold_memory', 10 * 1024 * 1024)
    profiler.bottleneck_threshold_calls = config.get('bottleneck_threshold_calls', 1000)
    
    return profiler