"""
Rate Limiting Monitoring and Alerting System

Provides real-time monitoring, alerting, analytics, and health checks
for API rate limiting compliance across all broker integrations.
"""

import asyncio
import time
import json
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import threading
from pathlib import Path

from ..core.rate_limiter import RateLimiterManager, RateLimitStatus
from ..core.request_manager import RequestManager
from ..brokers.rate_limit_configs import RateLimitConfig


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MetricType(Enum):
    """Types of metrics"""
    REQUEST_COUNT = "request_count"
    RATE_LIMIT_HITS = "rate_limit_hits"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    CONCURRENT_REQUESTS = "concurrent_requests"
    QUEUE_DEPTH = "queue_depth"
    THROUGHPUT = "throughput"
    COST = "cost"


@dataclass
class Alert:
    """Alert definition"""
    id: str
    severity: AlertSeverity
    title: str
    message: str
    broker: str
    metric_type: MetricType
    threshold_value: float
    current_value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "broker": self.broker,
            "metric_type": self.metric_type.value,
            "threshold_value": self.threshold_value,
            "current_value": self.current_value,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
            "resolved": self.resolved,
            "duration_seconds": (datetime.utcnow() - self.timestamp).total_seconds()
        }


@dataclass
class Metric:
    """Metric data point"""
    timestamp: datetime
    value: float
    broker: str
    metric_type: MetricType
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UsageAnalytics:
    """Usage analytics summary"""
    broker: str
    period_start: datetime
    period_end: datetime
    
    # Request statistics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rate_limited_requests: int = 0
    
    # Performance metrics
    average_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    throughput_requests_per_second: float = 0.0
    
    # Rate limiting metrics
    rate_limit_hit_rate: float = 0.0
    peak_concurrent_requests: int = 0
    average_queue_depth: float = 0.0
    
    # Cost metrics
    estimated_cost: float = 0.0
    cost_per_request: float = 0.0
    
    # Request distribution
    requests_by_type: Dict[str, int] = field(default_factory=dict)
    requests_by_hour: Dict[int, int] = field(default_factory=dict)
    
    # Alerts summary
    total_alerts: int = 0
    critical_alerts: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "broker": self.broker,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "rate_limited_requests": self.rate_limited_requests,
            "success_rate": self.successful_requests / max(1, self.total_requests),
            "error_rate": self.failed_requests / max(1, self.total_requests),
            "rate_limit_hit_rate": self.rate_limited_requests / max(1, self.total_requests),
            "average_response_time": self.average_response_time,
            "p95_response_time": self.p95_response_time,
            "p99_response_time": self.p99_response_time,
            "throughput_requests_per_second": self.throughput_requests_per_second,
            "peak_concurrent_requests": self.peak_concurrent_requests,
            "average_queue_depth": self.average_queue_depth,
            "estimated_cost": self.estimated_cost,
            "cost_per_request": self.cost_per_request,
            "requests_by_type": self.requests_by_type,
            "requests_by_hour": self.requests_by_hour,
            "total_alerts": self.total_alerts,
            "critical_alerts": self.critical_alerts
        }


class AlertRule:
    """Rule for generating alerts"""
    
    def __init__(
        self,
        id: str,
        metric_type: MetricType,
        broker: str,
        threshold: float,
        severity: AlertSeverity,
        duration_seconds: float = 60.0,
        comparison: str = "greater_than"  # greater_than, less_than, equals
    ):
        self.id = id
        self.metric_type = metric_type
        self.broker = broker
        self.threshold = threshold
        self.severity = severity
        self.duration_seconds = duration_seconds
        self.comparison = comparison
        
        # State tracking
        self.violation_start: Optional[datetime] = None
        self.last_value: Optional[float] = None
        self.evaluation_count = 0
    
    def evaluate(self, metric: Metric) -> Optional[Alert]:
        """Evaluate metric against rule"""
        self.last_value = metric.value
        self.evaluation_count += 1
        
        # Check if threshold is violated
        is_violated = False
        
        if self.comparison == "greater_than":
            is_violated = metric.value > self.threshold
        elif self.comparison == "less_than":
            is_violated = metric.value < self.threshold
        elif self.comparison == "equals":
            is_violated = abs(metric.value - self.threshold) < 0.001
        
        now = datetime.utcnow()
        
        if is_violated:
            # Start violation tracking if not already started
            if self.violation_start is None:
                self.violation_start = now
            elif (now - self.violation_start).total_seconds() >= self.duration_seconds:
                # Duration threshold exceeded, create alert
                return Alert(
                    id=f"{self.id}_{int(now.timestamp())}",
                    severity=self.severity,
                    title=f"{self.metric_type.value.replace('_', ' ').title()} Alert",
                    message=f"{self.metric_type.value.replace('_', ' ')} for {metric.broker} is {metric.value:.2f}, "
                           f"which is {self.comparison.replace('_', ' ')} threshold {self.threshold:.2f}",
                    broker=metric.broker,
                    metric_type=self.metric_type,
                    threshold_value=self.threshold,
                    current_value=metric.value
                )
        else:
            # Reset violation tracking
            self.violation_start = None
        
        return None


class RateLimitMonitor:
    """
    Central monitoring system for API rate limiting
    
    Collects metrics, generates alerts, and provides analytics
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        collection_interval: float = 10.0,
        retention_days: int = 30
    ):
        self.storage_path = Path(storage_path) if storage_path else Path("./rate_limit_monitoring")
        self.collection_interval = collection_interval
        self.retention_days = retention_days
        
        # Storage setup
        self.storage_path.mkdir(exist_ok=True)
        
        # Configuration
        self._rate_limiters: Dict[str, RateLimiterManager] = {}
        self._request_managers: Dict[str, RequestManager] = {}
        self._configs: Dict[str, RateLimitConfig] = {}
        
        # Metrics storage
        self._metrics: deque = deque(maxlen=100000)  # Store last 100k metrics
        self._metrics_by_broker: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        
        # Alerting
        self._alert_rules: List[AlertRule] = []
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: deque = deque(maxlen=5000)
        self._alert_callbacks: List[Callable[[Alert], None]] = []
        
        # Control
        self._is_running = False
        self._collection_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Setup default alert rules
        self._setup_default_alert_rules()
    
    def _setup_default_alert_rules(self):
        """Setup default alert rules for all brokers"""
        # Rate limit threshold alerts
        self._alert_rules = [
            AlertRule(
                id="rate_limit_warning",
                metric_type=MetricType.RATE_LIMIT_HITS,
                broker="*",  # All brokers
                threshold=0.8,  # 80% of limit
                severity=AlertSeverity.WARNING,
                duration_seconds=120.0
            ),
            AlertRule(
                id="rate_limit_critical",
                metric_type=MetricType.RATE_LIMIT_HITS,
                broker="*",
                threshold=0.95,  # 95% of limit
                severity=AlertSeverity.CRITICAL,
                duration_seconds=60.0
            ),
            AlertRule(
                id="error_rate_warning",
                metric_type=MetricType.ERROR_RATE,
                broker="*",
                threshold=0.05,  # 5% error rate
                severity=AlertSeverity.WARNING,
                duration_seconds=300.0
            ),
            AlertRule(
                id="error_rate_critical",
                metric_type=MetricType.ERROR_RATE,
                broker="*",
                threshold=0.10,  # 10% error rate
                severity=AlertSeverity.CRITICAL,
                duration_seconds=180.0
            ),
            AlertRule(
                id="high_response_time",
                metric_type=MetricType.RESPONSE_TIME,
                broker="*",
                threshold=5.0,  # 5 seconds
                severity=AlertSeverity.WARNING,
                duration_seconds=300.0
            ),
            AlertRule(
                id="queue_depth_warning",
                metric_type=MetricType.QUEUE_DEPTH,
                broker="*",
                threshold=100,  # 100 requests in queue
                severity=AlertSeverity.WARNING,
                duration_seconds=300.0
            ),
            AlertRule(
                id="cost_warning",
                metric_type=MetricType.COST,
                broker="*",
                threshold=100.0,  # $100 per day
                severity=AlertSeverity.WARNING,
                duration_seconds=86400.0  # 24 hours
            )
        ]
    
    def add_rate_limiter(self, broker_name: str, rate_limiter: RateLimiterManager):
        """Add rate limiter for monitoring"""
        self._rate_limiters[broker_name] = rate_limiter
        
        # Add broker-specific alert rules
        self._add_broker_alert_rules(broker_name)
    
    def add_request_manager(self, broker_name: str, request_manager: RequestManager):
        """Add request manager for monitoring"""
        self._request_managers[broker_name] = request_manager
    
    def add_config(self, broker_name: str, config: RateLimitConfig):
        """Add broker configuration for monitoring"""
        self._configs[broker_name] = config
    
    def _add_broker_alert_rules(self, broker_name: str):
        """Add broker-specific alert rules"""
        config = self._configs.get(broker_name)
        if not config:
            return
        
        # Add custom alert rules based on broker configuration
        if hasattr(config, 'warning_threshold'):
            self._alert_rules.append(AlertRule(
                id=f"{broker_name}_rate_limit_warning",
                metric_type=MetricType.RATE_LIMIT_HITS,
                broker=broker_name,
                threshold=config.warning_threshold,
                severity=AlertSeverity.WARNING,
                duration_seconds=120.0
            ))
        
        if hasattr(config, 'critical_threshold'):
            self._alert_rules.append(AlertRule(
                id=f"{broker_name}_rate_limit_critical",
                metric_type=MetricType.RATE_LIMIT_HITS,
                broker=broker_name,
                threshold=config.critical_threshold,
                severity=AlertSeverity.CRITICAL,
                duration_seconds=60.0
            ))
    
    def add_alert_rule(self, rule: AlertRule):
        """Add custom alert rule"""
        self._alert_rules.append(rule)
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback for alert notifications"""
        self._alert_callbacks.append(callback)
    
    async def start_monitoring(self):
        """Start monitoring collection"""
        if self._is_running:
            return
        
        self._is_running = True
        
        # Start collection task
        self._collection_task = asyncio.create_task(self._collection_loop())
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        print(f"Rate limit monitoring started (interval: {self.collection_interval}s)")
    
    async def stop_monitoring(self):
        """Stop monitoring collection"""
        if not self._is_running:
            return
        
        self._is_running = False
        
        # Cancel tasks
        for task in [self._collection_task, self._cleanup_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        print("Rate limit monitoring stopped")
    
    async def _collection_loop(self):
        """Main collection loop"""
        while self._is_running:
            try:
                await self._collect_metrics()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Collection loop error: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _cleanup_loop(self):
        """Cleanup loop for old data"""
        while self._is_running:
            try:
                await self._cleanup_old_data()
                await asyncio.sleep(3600)  # Run every hour
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Cleanup loop error: {e}")
                await asyncio.sleep(3600)
    
    async def _collect_metrics(self):
        """Collect metrics from all monitored components"""
        for broker_name in set(list(self._rate_limiters.keys()) + list(self._request_managers.keys())):
            try:
                # Collect rate limiter metrics
                if broker_name in self._rate_limiters:
                    await self._collect_rate_limiter_metrics(broker_name)
                
                # Collect request manager metrics
                if broker_name in self._request_managers:
                    await self._collect_request_manager_metrics(broker_name)
                
                # Evaluate alert rules
                await self._evaluate_alerts(broker_name)
                
            except Exception as e:
                print(f"Error collecting metrics for {broker_name}: {e}")
    
    async def _collect_rate_limiter_metrics(self, broker_name: str):
        """Collect metrics from rate limiter"""
        rate_limiter = self._rate_limiters[broker_name]
        
        try:
            # Get global status
            status = rate_limiter.get_global_status()
            
            # Create metrics
            metrics = [
                Metric(
                    timestamp=datetime.utcnow(),
                    value=status.get("total_requests", 0),
                    broker=broker_name,
                    metric_type=MetricType.REQUEST_COUNT
                ),
                Metric(
                    timestamp=datetime.utcnow(),
                    value=status.get("rejected_requests", 0),
                    broker=broker_name,
                    metric_type=MetricType.RATE_LIMIT_HITS
                ),
                Metric(
                    timestamp=datetime.utcnow(),
                    value=status.get("average_wait_time", 0.0),
                    broker=broker_name,
                    metric_type=MetricType.RESPONSE_TIME
                ),
                Metric(
                    timestamp=datetime.utcnow(),
                    value=status.get("current_requests", 0),
                    broker=broker_name,
                    metric_type=MetricType.CONCURRENT_REQUESTS
                )
            ]
            
            # Calculate rate limit hit rate
            total = status.get("total_requests", 0)
            rejected = status.get("rejected_requests", 0)
            hit_rate = rejected / max(1, total)
            
            metrics.append(Metric(
                timestamp=datetime.utcnow(),
                value=hit_rate,
                broker=broker_name,
                metric_type=MetricType.RATE_LIMIT_HITS
            ))
            
            # Calculate error rate
            allowed = status.get("allowed_requests", 0)
            error_rate = (total - allowed) / max(1, total)
            
            metrics.append(Metric(
                timestamp=datetime.utcnow(),
                value=error_rate,
                broker=broker_name,
                metric_type=MetricType.ERROR_RATE
            ))
            
            # Store metrics
            for metric in metrics:
                self._metrics.append(metric)
                self._metrics_by_broker[broker_name].append(metric)
        
        except Exception as e:
            print(f"Error collecting rate limiter metrics for {broker_name}: {e}")
    
    async def _collect_request_manager_metrics(self, broker_name: str):
        """Collect metrics from request manager"""
        request_manager = self._request_managers[broker_name]
        
        try:
            # Get stats
            stats = request_manager.get_stats()
            
            # Create metrics
            metrics = [
                Metric(
                    timestamp=datetime.utcnow(),
                    value=stats.get("pending_requests", 0),
                    broker=broker_name,
                    metric_type=MetricType.QUEUE_DEPTH
                ),
                Metric(
                    timestamp=datetime.utcnow(),
                    value=stats.get("processing_requests", 0),
                    broker=broker_name,
                    metric_type=MetricType.CONCURRENT_REQUESTS
                ),
                Metric(
                    timestamp=datetime.utcnow(),
                    value=stats.get("average_processing_time", 0.0),
                    broker=broker_name,
                    metric_type=MetricType.RESPONSE_TIME
                )
            ]
            
            # Calculate throughput
            completed = stats.get("completed_requests", 0)
            time_window = 60.0  # requests per minute
            throughput = completed / time_window
            
            metrics.append(Metric(
                timestamp=datetime.utcnow(),
                value=throughput,
                broker=broker_name,
                metric_type=MetricType.THROUGHPUT
            ))
            
            # Calculate cost
            config = self._configs.get(broker_name)
            if config and config.market_data_fee_per_request > 0:
                total_requests = stats.get("total_requests", 0)
                cost = total_requests * config.market_data_fee_per_request
                
                metrics.append(Metric(
                    timestamp=datetime.utcnow(),
                    value=cost,
                    broker=broker_name,
                    metric_type=MetricType.COST
                ))
            
            # Store metrics
            for metric in metrics:
                self._metrics.append(metric)
                self._metrics_by_broker[broker_name].append(metric)
        
        except Exception as e:
            print(f"Error collecting request manager metrics for {broker_name}: {e}")
    
    async def _evaluate_alerts(self, broker_name: str):
        """Evaluate alert rules for broker"""
        broker_metrics = self._metrics_by_broker[broker_name]
        
        # Get recent metrics for evaluation
        recent_metrics = [m for m in broker_metrics if 
                         (datetime.utcnow() - m.timestamp).total_seconds() <= 300]
        
        for rule in self._alert_rules:
            # Check if rule applies to this broker
            if rule.broker != "*" and rule.broker != broker_name:
                continue
            
            # Find latest metric of this type
            relevant_metrics = [m for m in recent_metrics if m.metric_type == rule.metric_type]
            if not relevant_metrics:
                continue
            
            latest_metric = max(relevant_metrics, key=lambda m: m.timestamp)
            
            # Evaluate rule
            alert = rule.evaluate(latest_metric)
            
            if alert:
                await self._handle_alert(alert)
    
    async def _handle_alert(self, alert: Alert):
        """Handle alert creation"""
        # Check if alert already exists
        if alert.id in self._active_alerts:
            return
        
        # Add to active alerts
        self._active_alerts[alert.id] = alert
        self._alert_history.append(alert)
        
        # Notify callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                print(f"Alert callback error: {e}")
        
        print(f"ALERT [{alert.severity.value.upper()}]: {alert.title} - {alert.message}")
    
    async def _cleanup_old_data(self):
        """Cleanup old metrics and alerts"""
        cutoff_time = datetime.utcnow() - timedelta(days=self.retention_days)
        
        # Clean up old metrics
        while self._metrics and self._metrics[0].timestamp < cutoff_time:
            self._metrics.popleft()
        
        # Clean up old alerts
        while self._alert_history and self._alert_history[0].timestamp < cutoff_time:
            self._alert_history.popleft()
        
        # Clean up broker-specific metrics
        for broker_metrics in self._metrics_by_broker.values():
            while broker_metrics and broker_metrics[0].timestamp < cutoff_time:
                broker_metrics.popleft()
    
    def get_active_alerts(self, broker: Optional[str] = None, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active alerts"""
        alerts = list(self._active_alerts.values())
        
        if broker:
            alerts = [a for a in alerts if a.broker == broker]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return alerts
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge alert"""
        if alert_id in self._active_alerts:
            self._active_alerts[alert_id].acknowledged = True
            return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve alert"""
        if alert_id in self._active_alerts:
            alert = self._active_alerts[alert_id]
            alert.resolved = True
            del self._active_alerts[alert_id]
            return True
        return False
    
    def get_analytics(
        self, 
        broker: str, 
        period_hours: int = 24
    ) -> UsageAnalytics:
        """Get usage analytics for broker"""
        cutoff_time = datetime.utcnow() - timedelta(hours=period_hours)
        
        # Filter metrics for broker and period
        broker_metrics = [m for m in self._metrics_by_broker[broker] 
                         if m.timestamp >= cutoff_time]
        
        # Calculate analytics
        analytics = UsageAnalytics(
            broker=broker,
            period_start=cutoff_time,
            period_end=datetime.utcnow()
        )
        
        # Count requests
        request_metrics = [m for m in broker_metrics if m.metric_type == MetricType.REQUEST_COUNT]
        if request_metrics:
            analytics.total_requests = int(max(m.value for m in request_metrics))
        
        # Count rate limit hits
        rate_limit_metrics = [m for m in broker_metrics if m.metric_type == MetricType.RATE_LIMIT_HITS]
        if rate_limit_metrics:
            analytics.rate_limited_requests = int(max(m.value for m in rate_limit_metrics))
        
        # Calculate response times
        response_metrics = [m for m in broker_metrics if m.metric_type == MetricType.RESPONSE_TIME]
        if response_metrics:
            values = [m.value for m in response_metrics]
            analytics.average_response_time = sum(values) / len(values)
            values.sort()
            analytics.p95_response_time = values[int(len(values) * 0.95)]
            analytics.p99_response_time = values[int(len(values) * 0.99)]
        
        # Calculate queue depth
        queue_metrics = [m for m in broker_metrics if m.metric_type == MetricType.QUEUE_DEPTH]
        if queue_metrics:
            values = [m.value for m in queue_metrics]
            analytics.average_queue_depth = sum(values) / len(values)
        
        # Calculate cost
        cost_metrics = [m for m in broker_metrics if m.metric_type == MetricType.COST]
        if cost_metrics:
            analytics.estimated_cost = max(m.value for m in cost_metrics)
            analytics.cost_per_request = analytics.estimated_cost / max(1, analytics.total_requests)
        
        return analytics
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        return {
            "is_monitoring": self._is_running,
            "monitored_brokers": len(self._rate_limiters),
            "total_metrics": len(self._metrics),
            "active_alerts": len(self._active_alerts),
            "critical_alerts": len([a for a in self._active_alerts.values() if a.severity == AlertSeverity.CRITICAL]),
            "collection_interval": self.collection_interval,
            "retention_days": self.retention_days,
            "last_collection": max([m.timestamp for m in self._metrics], default=None)
        }
    
    def export_metrics(self, file_path: str, broker: Optional[str] = None, hours: int = 24):
        """Export metrics to file"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        if broker:
            metrics = [m for m in self._metrics_by_broker[broker] if m.timestamp >= cutoff_time]
        else:
            metrics = [m for m in self._metrics if m.timestamp >= cutoff_time]
        
        export_data = {
            "export_time": datetime.utcnow().isoformat(),
            "broker": broker,
            "period_hours": hours,
            "metrics": [self._serialize_metric(m) for m in metrics]
        }
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def _serialize_metric(self, metric: Metric) -> Dict[str, Any]:
        """Serialize metric for export"""
        return {
            "timestamp": metric.timestamp.isoformat(),
            "value": metric.value,
            "broker": metric.broker,
            "metric_type": metric.metric_type.value,
            "metadata": metric.metadata
        }


class HealthChecker:
    """
    Health checking system for rate limiting components
    """
    
    def __init__(self, monitor: RateLimitMonitor):
        self.monitor = monitor
        self._health_checks: List[Callable] = []
    
    def add_health_check(self, check: Callable):
        """Add custom health check"""
        self._health_checks.append(check)
    
    async def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health_status = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "healthy",
            "checks": {},
            "issues": [],
            "recommendations": []
        }
        
        # Basic monitoring health
        monitor_health = self.monitor.get_health_status()
        health_status["monitor"] = monitor_health
        
        if not monitor_health["is_monitoring"]:
            health_status["overall_status"] = "degraded"
            health_status["issues"].append("Monitoring is not active")
        
        # Check active alerts
        active_alerts = self.monitor.get_active_alerts()
        critical_alerts = [a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]
        
        if critical_alerts:
            health_status["overall_status"] = "critical"
            health_status["issues"].append(f"{len(critical_alerts)} critical alerts active")
        
        # Check broker health
        for broker_name in self.monitor._rate_limiters.keys():
            broker_health = await self._check_broker_health(broker_name)
            health_status["checks"][f"broker_{broker_name}"] = broker_health
            
            if broker_health["status"] != "healthy":
                health_status["issues"].append(f"{broker_name}: {broker_health['issues']}")
                if broker_health["status"] == "critical":
                    health_status["overall_status"] = "critical"
        
        # Run custom checks
        for check in self._health_checks:
            try:
                check_result = await check()
                health_status["checks"][check.__name__] = check_result
                
                if check_result.get("status") != "healthy":
                    health_status["issues"].append(f"{check.__name__}: {check_result.get('issue', 'Unknown')}")
            except Exception as e:
                health_status["checks"][check.__name__] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return health_status
    
    async def _check_broker_health(self, broker_name: str) -> Dict[str, Any]:
        """Check health of specific broker"""
        health = {
            "status": "healthy",
            "issues": [],
            "metrics": {}
        }
        
        # Get recent analytics
        try:
            analytics = self.monitor.get_analytics(broker_name, period_hours=1)
            health["metrics"] = {
                "requests_last_hour": analytics.total_requests,
                "success_rate": analytics.successful_requests / max(1, analytics.total_requests),
                "rate_limit_hit_rate": analytics.rate_limited_requests / max(1, analytics.total_requests),
                "average_response_time": analytics.average_response_time
            }
            
            # Check thresholds
            if health["metrics"]["success_rate"] < 0.95:
                health["status"] = "degraded"
                health["issues"].append(f"Low success rate: {health['metrics']['success_rate']:.2%}")
            
            if health["metrics"]["rate_limit_hit_rate"] > 0.05:
                health["status"] = "degraded"
                health["issues"].append(f"High rate limit hit rate: {health['metrics']['rate_limit_hit_rate']:.2%}")
            
            if health["metrics"]["average_response_time"] > 5.0:
                health["status"] = "degraded"
                health["issues"].append(f"High response time: {health['metrics']['average_response_time']:.2f}s")
            
            if len(health["issues"]) > 2:
                health["status"] = "critical"
        
        except Exception as e:
            health["status"] = "error"
            health["issues"].append(f"Error getting analytics: {str(e)}")
        
        return health