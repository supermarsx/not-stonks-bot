"""
Application Performance Monitoring (APM)

Comprehensive APM system with request/response monitoring, 
database query performance tracking, and transaction analysis.
"""

import time
import asyncio
import logging
import json
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
from contextlib import asynccontextmanager, contextmanager
import functools
import threading
import uuid


class SpanType(Enum):
    """Span type enumeration"""
    HTTP_REQUEST = "http_request"
    DATABASE_QUERY = "database_query"
    CACHE_OPERATION = "cache_operation"
    EXTERNAL_API = "external_api"
    MESSAGE_QUEUE = "message_queue"
    CUSTOM = "custom"


class TransactionStatus(Enum):
    """Transaction status enumeration"""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class Span:
    """Individual span in a transaction"""
    span_id: str
    operation_name: str
    span_type: SpanType
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    parent_span_id: Optional[str] = None
    error: Optional[Exception] = None
    status: TransactionStatus = TransactionStatus.SUCCESS
    
    def finish(self, status: TransactionStatus = TransactionStatus.SUCCESS):
        """Finish the span"""
        self.end_time = datetime.now()
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.status = status
    
    def add_tag(self, key: str, value: Any):
        """Add tag to span"""
        self.tags[key] = value
    
    def add_log(self, message: str, level: str = "info", **kwargs):
        """Add log entry to span"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'level': level
        }
        log_entry.update(kwargs)
        self.logs.append(log_entry)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary"""
        return {
            'span_id': self.span_id,
            'operation_name': self.operation_name,
            'span_type': self.span_type.value,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_ms': self.duration_ms,
            'tags': self.tags,
            'logs': self.logs,
            'parent_span_id': self.parent_span_id,
            'status': self.status.value,
            'error_message': str(self.error) if self.error else None
        }


@dataclass
class Transaction:
    """Complete transaction trace"""
    trace_id: str
    transaction_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    spans: List[Span] = field(default_factory=list)
    root_span: Optional[Span] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    status: TransactionStatus = TransactionStatus.SUCCESS
    error: Optional[Exception] = None
    
    def add_span(self, span: Span):
        """Add span to transaction"""
        self.spans.append(span)
        if not self.root_span:
            self.root_span = span
    
    def finish(self, status: TransactionStatus = TransactionStatus.SUCCESS):
        """Finish the transaction"""
        self.end_time = datetime.now()
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.status = status
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert transaction to dictionary"""
        return {
            'trace_id': self.trace_id,
            'transaction_name': self.transaction_name,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_ms': self.duration_ms,
            'spans_count': len(self.spans),
            'tags': self.tags,
            'status': self.status.value,
            'error_message': str(self.error) if self.error else None,
            'spans': [span.to_dict() for span in self.spans]
        }


@dataclass
class PerformanceThreshold:
    """Performance threshold configuration"""
    name: str
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    enabled: bool = True
    
    def check(self, value: float) -> str:
        """Check if value exceeds thresholds"""
        if not self.enabled:
            return "ok"
        
        if value >= self.critical_threshold:
            return "critical"
        elif value >= self.warning_threshold:
            return "warning"
        else:
            return "ok"


class DatabaseMonitor:
    """Database query performance monitor"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.query_stats = defaultdict(lambda: {
            'count': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'errors': 0
        })
        self.query_history = deque(maxlen=1000)
        self.slow_query_threshold = 1000.0  # 1 second in ms
        self.thresholds = [
            PerformanceThreshold("Fast Query", "database.query_time", 100, 500),
            PerformanceThreshold("Slow Query", "database.query_time", 1000, 5000)
        ]
    
    def record_query(self, 
                    query: str, 
                    duration_ms: float, 
                    success: bool = True,
                    connection_info: Dict[str, Any] = None):
        """Record database query performance"""
        self.query_stats[query]['count'] += 1
        self.query_stats[query]['total_time'] += duration_ms
        self.query_stats[query]['avg_time'] = (
            self.query_stats[query]['total_time'] / self.query_stats[query]['count']
        )
        self.query_stats[query]['min_time'] = min(self.query_stats[query]['min_time'], duration_ms)
        self.query_stats[query]['max_time'] = max(self.query_stats[query]['max_time'], duration_ms)
        
        if not success:
            self.query_stats[query]['errors'] += 1
        
        # Record in history
        self.query_history.append({
            'timestamp': datetime.now(),
            'query': query,
            'duration_ms': duration_ms,
            'success': success,
            'connection_info': connection_info or {}
        })
    
    def get_query_statistics(self) -> Dict[str, Any]:
        """Get database query statistics"""
        return {
            'total_queries': sum(stats['count'] for stats in self.query_stats.values()),
            'avg_duration_ms': self._calculate_global_avg(),
            'slow_queries_count': len([
                q for q in self.query_history 
                if q['duration_ms'] > self.slow_query_threshold
            ]),
            'error_rate': self._calculate_error_rate(),
            'query_details': dict(self.query_stats)
        }
    
    def get_slow_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get slowest queries"""
        return sorted(
            self.query_history, 
            key=lambda x: x['duration_ms'], 
            reverse=True
        )[:limit]
    
    def _calculate_global_avg(self) -> float:
        """Calculate global average query time"""
        total_time = sum(stats['total_time'] for stats in self.query_stats.values())
        total_count = sum(stats['count'] for stats in self.query_stats.values())
        return total_time / max(total_count, 1)
    
    def _calculate_error_rate(self) -> float:
        """Calculate global error rate"""
        total_queries = sum(stats['count'] for stats in self.query_stats.values())
        total_errors = sum(stats['errors'] for stats in self.query_stats.values())
        return total_errors / max(total_queries, 1)


class CacheMonitor:
    """Cache operation performance monitor"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cache_stats = defaultdict(lambda: {
            'hits': 0,
            'misses': 0,
            'hit_ratio': 0.0,
            'total_operations': 0
        })
        self.operation_history = deque(maxlen=1000)
        self.thresholds = [
            PerformanceThreshold("Low Cache Hit Rate", "cache.hit_ratio", 0.7, 0.5),
            PerformanceThreshold("High Cache Operations", "cache.operations_per_minute", 1000, 5000)
        ]
    
    def record_cache_operation(self, 
                              cache_type: str,
                              operation: str,
                              hit: bool,
                              duration_ms: float = 0):
        """Record cache operation"""
        stats = self.cache_stats[f"{cache_type}:{operation}"]
        stats['total_operations'] += 1
        
        if hit:
            stats['hits'] += 1
        else:
            stats['misses'] += 1
        
        # Calculate hit ratio
        total = stats['hits'] + stats['misses']
        stats['hit_ratio'] = stats['hits'] / max(total, 1)
        
        # Record in history
        self.operation_history.append({
            'timestamp': datetime.now(),
            'cache_type': cache_type,
            'operation': operation,
            'hit': hit,
            'duration_ms': duration_ms
        })
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        global_hits = sum(stats['hits'] for stats in self.cache_stats.values())
        global_misses = sum(stats['misses'] for stats in self.cache_stats.values())
        global_operations = global_hits + global_misses
        
        return {
            'global_hit_ratio': global_hits / max(global_operations, 1),
            'total_operations': global_operations,
            'cache_types': {
                cache_key: {
                    'hit_ratio': stats['hit_ratio'],
                    'operations': stats['total_operations']
                }
                for cache_key, stats in self.cache_stats.items()
            }
        }


class ExternalServiceMonitor:
    """External service call monitor"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.service_stats = defaultdict(lambda: {
            'calls': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'errors': 0,
            'timeouts': 0,
            'success_rate': 1.0
        })
        self.call_history = deque(maxlen=1000)
        self.thresholds = [
            PerformanceThreshold("Slow External Call", "external.call_time", 2000, 10000),
            PerformanceThreshold("Low Success Rate", "external.success_rate", 0.95, 0.8)
        ]
    
    def record_external_call(self,
                           service_name: str,
                           endpoint: str,
                           duration_ms: float,
                           status_code: int,
                           success: bool = True,
                           timeout: bool = False):
        """Record external service call"""
        stats = self.service_stats[f"{service_name}:{endpoint}"]
        stats['calls'] += 1
        stats['total_time'] += duration_ms
        stats['avg_time'] = stats['total_time'] / stats['calls']
        
        if not success:
            stats['errors'] += 1
        
        if timeout:
            stats['timeouts'] += 1
        
        # Calculate success rate
        stats['success_rate'] = (stats['calls'] - stats['errors']) / stats['calls']
        
        # Record in history
        self.call_history.append({
            'timestamp': datetime.now(),
            'service': service_name,
            'endpoint': endpoint,
            'duration_ms': duration_ms,
            'status_code': status_code,
            'success': success,
            'timeout': timeout
        })
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """Get external service statistics"""
        return {
            'total_calls': sum(stats['calls'] for stats in self.service_stats.values()),
            'overall_success_rate': self._calculate_overall_success_rate(),
            'services': {
                service_key: {
                    'calls': stats['calls'],
                    'avg_duration_ms': stats['avg_time'],
                    'success_rate': stats['success_rate'],
                    'errors': stats['errors'],
                    'timeouts': stats['timeouts']
                }
                for service_key, stats in self.service_stats.items()
            }
        }
    
    def _calculate_overall_success_rate(self) -> float:
        """Calculate overall success rate across all services"""
        total_calls = sum(stats['calls'] for stats in self.service_stats.values())
        total_errors = sum(stats['errors'] for stats in self.service_stats.values())
        return (total_calls - total_errors) / max(total_calls, 1)
    
    def get_failed_calls(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent failed calls"""
        return sorted(
            [call for call in self.call_history if not call['success']],
            key=lambda x: x['timestamp'],
            reverse=True
        )[:limit]


class APMClient:
    """Application Performance Monitoring Client"""
    
    def __init__(self, 
                 service_name: str = "trading_system",
                 sampling_rate: float = 1.0,
                 max_transactions: int = 1000,
                 auto_instrument: bool = True):
        self.service_name = service_name
        self.sampling_rate = sampling_rate
        self.max_transactions = max_transactions
        self.auto_instrument = auto_instrument
        
        # Transaction storage
        self.transactions = deque(maxlen=max_transactions)
        self.active_transactions = {}
        
        # Performance monitors
        self.db_monitor = DatabaseMonitor()
        self.cache_monitor = CacheMonitor()
        self.external_monitor = ExternalServiceMonitor()
        
        self.logger = logging.getLogger(__name__)
        
        # Performance thresholds
        self.thresholds = {
            'transaction_duration': [
                PerformanceThreshold("Slow Transaction", "transaction.duration", 2000, 10000),
                PerformanceThreshold("Very Slow Transaction", "transaction.duration", 5000, 30000)
            ],
            'span_count': [
                PerformanceThreshold("Too Many Spans", "transaction.spans", 50, 100)
            ]
        }
    
    def start_transaction(self, 
                         transaction_name: str, 
                         tags: Dict[str, Any] = None,
                         sample: bool = None) -> str:
        """Start a new transaction"""
        # Determine if we should sample this transaction
        if sample is None:
            sample = self.sampling_rate >= 1.0 or hash(transaction_name) % int(1/self.sampling_rate) == 0
        
        if not sample:
            return ""
        
        trace_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        transaction = Transaction(
            trace_id=trace_id,
            transaction_name=transaction_name,
            start_time=start_time,
            tags=tags or {}
        )
        
        # Create root span
        root_span = Span(
            span_id=str(uuid.uuid4()),
            operation_name=transaction_name,
            span_type=SpanType.CUSTOM,
            start_time=start_time,
            parent_span_id=None
        )
        
        transaction.add_span(root_span)
        self.active_transactions[trace_id] = transaction
        
        self.logger.debug(f"Started transaction: {transaction_name} (trace_id: {trace_id})")
        return trace_id
    
    def finish_transaction(self, 
                          trace_id: str, 
                          status: TransactionStatus = TransactionStatus.SUCCESS,
                          error: Exception = None):
        """Finish a transaction"""
        if trace_id not in self.active_transactions:
            self.logger.warning(f"Trying to finish non-existent transaction: {trace_id}")
            return
        
        transaction = self.active_transactions[trace_id]
        transaction.finish(status)
        
        if error:
            transaction.error = error
            root_span = transaction.root_span
            if root_span:
                root_span.error = error
                root_span.add_log(f"Transaction failed: {error}", "error")
        
        # Add to storage
        self.transactions.append(transaction)
        del self.active_transactions[trace_id]
        
        self.logger.debug(f"Finished transaction: {transaction.transaction_name} (trace_id: {trace_id})")
    
    def start_span(self, 
                  trace_id: str,
                  operation_name: str,
                  span_type: SpanType = SpanType.CUSTOM,
                  tags: Dict[str, Any] = None) -> str:
        """Start a new span"""
        if trace_id not in self.active_transactions:
            return ""
        
        transaction = self.active_transactions[trace_id]
        span_id = str(uuid.uuid4())
        
        # Find parent span (last span in transaction)
        parent_span = None
        if transaction.spans:
            parent_span = transaction.spans[-1]
        
        span = Span(
            span_id=span_id,
            operation_name=operation_name,
            span_type=span_type,
            start_time=datetime.now(),
            parent_span_id=parent_span.span_id if parent_span else None,
            tags=tags or {}
        )
        
        transaction.add_span(span)
        return span_id
    
    def finish_span(self, 
                   trace_id: str,
                   span_id: str,
                   status: TransactionStatus = TransactionStatus.SUCCESS,
                   error: Exception = None):
        """Finish a span"""
        if trace_id not in self.active_transactions:
            return
        
        transaction = self.active_transactions[trace_id]
        
        # Find the span
        span = None
        for s in transaction.spans:
            if s.span_id == span_id:
                span = s
                break
        
        if not span:
            self.logger.warning(f"Trying to finish non-existent span: {span_id}")
            return
        
        span.finish(status)
        if error:
            span.error = error
            span.add_log(f"Span failed: {error}", "error")
        
        self.logger.debug(f"Finished span: {operation_name} (span_id: {span_id})")
    
    def record_database_query(self,
                             query: str,
                             duration_ms: float,
                             success: bool = True,
                             connection_info: Dict[str, Any] = None,
                             trace_id: str = None):
        """Record database query performance"""
        self.db_monitor.record_query(query, duration_ms, success, connection_info)
        
        # Add span to active transaction if trace_id provided
        if trace_id and trace_id in self.active_transactions:
            self.start_span(trace_id, "database_query", SpanType.DATABASE_QUERY)
            # Would need to track span_id to finish it, simplified here
            # In practice, you'd track the span_id returned by start_span
    
    def record_cache_operation(self,
                              cache_type: str,
                              operation: str,
                              hit: bool,
                              duration_ms: float = 0,
                              trace_id: str = None):
        """Record cache operation performance"""
        self.cache_monitor.record_cache_operation(cache_type, operation, hit, duration_ms)
        
        # Add span to active transaction if trace_id provided
        if trace_id and trace_id in self.active_transactions:
            self.start_span(trace_id, f"cache_{operation}", SpanType.CACHE_OPERATION)
    
    def record_external_call(self,
                           service_name: str,
                           endpoint: str,
                           duration_ms: float,
                           status_code: int,
                           success: bool = True,
                           timeout: bool = False,
                           trace_id: str = None):
        """Record external service call"""
        self.external_monitor.record_external_call(
            service_name, endpoint, duration_ms, status_code, success, timeout
        )
        
        # Add span to active transaction if trace_id provided
        if trace_id and trace_id in self.active_transactions:
            self.start_span(trace_id, f"{service_name}_{endpoint}", SpanType.EXTERNAL_API)
    
    def get_transaction_statistics(self) -> Dict[str, Any]:
        """Get transaction statistics"""
        if not self.transactions:
            return {
                'total_transactions': 0,
                'avg_duration_ms': 0,
                'success_rate': 0,
                'error_rate': 0
            }
        
        durations = [t.duration_ms for t in self.transactions if t.duration_ms]
        total_transactions = len(self.transactions)
        successful_transactions = sum(1 for t in self.transactions if t.status == TransactionStatus.SUCCESS)
        error_transactions = sum(1 for t in self.transactions if t.status == TransactionStatus.ERROR)
        
        return {
            'total_transactions': total_transactions,
            'avg_duration_ms': sum(durations) / max(len(durations), 1),
            'median_duration_ms': sorted(durations)[len(durations)//2] if durations else 0,
            'p95_duration_ms': sorted(durations)[int(len(durations) * 0.95)] if durations else 0,
            'p99_duration_ms': sorted(durations)[int(len(durations) * 0.99)] if durations else 0,
            'success_rate': successful_transactions / max(total_transactions, 1),
            'error_rate': error_transactions / max(total_transactions, 1),
            'active_transactions': len(self.active_transactions)
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        transaction_stats = self.get_transaction_statistics()
        db_stats = self.db_monitor.get_query_statistics()
        cache_stats = self.cache_monitor.get_cache_statistics()
        external_stats = self.external_monitor.get_service_statistics()
        
        # Check thresholds
        threshold_violations = []
        for transaction in self.transactions[-100:]:  # Last 100 transactions
            duration = transaction.duration_ms or 0
            for threshold in self.thresholds.get('transaction_duration', []):
                status = threshold.check(duration)
                if status in ['warning', 'critical']:
                    threshold_violations.append({
                        'type': 'transaction_duration',
                        'threshold': threshold.name,
                        'value': duration,
                        'status': status,
                        'trace_id': transaction.trace_id
                    })
            
            span_count = len(transaction.spans)
            for threshold in self.thresholds.get('span_count', []):
                status = threshold.check(span_count)
                if status in ['warning', 'critical']:
                    threshold_violations.append({
                        'type': 'span_count',
                        'threshold': threshold.name,
                        'value': span_count,
                        'status': status,
                        'trace_id': transaction.trace_id
                    })
        
        return {
            'timestamp': datetime.now().isoformat(),
            'service_name': self.service_name,
            'transaction_statistics': transaction_stats,
            'database_statistics': db_stats,
            'cache_statistics': cache_stats,
            'external_service_statistics': external_stats,
            'threshold_violations': threshold_violations,
            'total_transactions': len(self.transactions),
            'active_transactions': len(self.active_transactions)
        }
    
    def export_traces(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Export recent transaction traces"""
        return [transaction.to_dict() for transaction in list(self.transactions)[-limit:]]
    
    def clear_data(self):
        """Clear all collected data"""
        self.transactions.clear()
        self.active_transactions.clear()
        # Reset monitors
        self.db_monitor.query_stats.clear()
        self.db_monitor.query_history.clear()
        self.cache_monitor.cache_stats.clear()
        self.cache_monitor.operation_history.clear()
        self.external_monitor.service_stats.clear()
        self.external_monitor.call_history.clear()
        
        self.logger.info("Cleared all APM data")


# Context managers and decorators for easy instrumentation
@contextmanager
def apm_transaction(apm_client: APMClient, transaction_name: str, tags: Dict[str, Any] = None):
    """Context manager for automatic transaction tracking"""
    trace_id = apm_client.start_transaction(transaction_name, tags)
    try:
        yield trace_id
    except Exception as e:
        apm_client.finish_transaction(trace_id, TransactionStatus.ERROR, e)
        raise
    else:
        apm_client.finish_transaction(trace_id, TransactionStatus.SUCCESS)


def apm_transaction_decorator(apm_client: APMClient, transaction_name: str = None):
    """Decorator for automatic transaction tracking"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = transaction_name or f"{func.__name__}"
            with apm_transaction(apm_client, name) as trace_id:
                kwargs['trace_id'] = trace_id
                return func(*args, **kwargs)
        return wrapper
    return decorator


@contextmanager
def apm_span(apm_client: APMClient, trace_id: str, operation_name: str, span_type: SpanType = SpanType.CUSTOM):
    """Context manager for automatic span tracking"""
    span_id = apm_client.start_span(trace_id, operation_name, span_type)
    try:
        yield span_id
    except Exception as e:
        apm_client.finish_span(trace_id, span_id, TransactionStatus.ERROR, e)
        raise
    else:
        apm_client.finish_span(trace_id, span_id, TransactionStatus.SUCCESS)


# Global APM client instance
_apm_client = None

def get_apm_client(service_name: str = "trading_system") -> APMClient:
    """Get global APM client instance"""
    global _apm_client
    if _apm_client is None:
        _apm_client = APMClient(service_name=service_name)
    return _apm_client


def initialize_apm(config: Dict[str, Any]) -> APMClient:
    """Initialize APM with configuration"""
    client = APMClient(
        service_name=config.get('service_name', 'trading_system'),
        sampling_rate=config.get('sampling_rate', 1.0),
        max_transactions=config.get('max_transactions', 1000),
        auto_instrument=config.get('auto_instrument', True)
    )
    return client