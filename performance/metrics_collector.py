"""
Performance Metrics Collector

Comprehensive performance monitoring and metrics collection system
with real-time tracking, alerts, and threshold management.
"""

import time
import asyncio
import logging
import threading
import json
import psutil
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import socket
import platform
import gc


class MetricType(Enum):
    """Metric type enumeration"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


class MetricCategory(Enum):
    """Metric category enumeration"""
    SYSTEM = "system"
    APPLICATION = "application"
    PERFORMANCE = "performance"
    BUSINESS = "business"
    ERROR = "error"
    SECURITY = "security"


class AlertSeverity(Enum):
    """Alert severity enumeration"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Metric:
    """Individual metric definition"""
    name: str
    type: MetricType
    value: Union[int, float, str]
    timestamp: datetime
    category: MetricCategory
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary"""
        return {
            'name': self.name,
            'type': self.type.value,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'category': self.category.value,
            'tags': self.tags,
            'metadata': self.metadata
        }


@dataclass
class AlertRule:
    """Alert rule definition"""
    name: str
    metric_name: str
    condition: str  # "gt", "lt", "eq", "ne", "gte", "lte"
    threshold: Union[int, float]
    severity: AlertSeverity
    enabled: bool = True
    duration_seconds: int = 60  # How long condition must persist
    cooldown_seconds: int = 300  # Min time between same alerts
    notification_callback: Optional[Callable] = None
    description: str = ""
    
    def check(self, current_value: Union[int, float]) -> bool:
        """Check if metric value triggers alert"""
        if not self.enabled:
            return False
        
        if self.condition == "gt":
            return current_value > self.threshold
        elif self.condition == "lt":
            return current_value < self.threshold
        elif self.condition == "eq":
            return current_value == self.threshold
        elif self.condition == "ne":
            return current_value != self.threshold
        elif self.condition == "gte":
            return current_value >= self.threshold
        elif self.condition == "lte":
            return current_value <= self.threshold
        else:
            return False


@dataclass
class Alert:
    """Alert instance"""
    rule: AlertRule
    metric_value: Union[int, float]
    triggered_at: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    triggered_count: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            'name': self.rule.name,
            'metric_name': self.rule.metric_name,
            'condition': self.rule.condition,
            'threshold': self.rule.threshold,
            'current_value': self.metric_value,
            'severity': self.rule.severity.value,
            'triggered_at': self.triggered_at.isoformat(),
            'resolved': self.resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'triggered_count': self.triggered_count,
            'description': self.rule.description
        }


class SystemMonitor:
    """System resource monitoring"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._process = psutil.Process()
        self._network_io = psutil.net_io_counters()
        self._disk_io = psutil.disk_io_counters()
    
    def get_system_metrics(self) -> Dict[str, Metric]:
        """Collect system-level metrics"""
        metrics = {}
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
            cpu_freq = psutil.cpu_freq()
            
            metrics.update([
                Metric("system.cpu.usage_percent", MetricType.GAUGE, cpu_percent, 
                       datetime.now(), MetricCategory.SYSTEM),
                Metric("system.cpu.cores", MetricType.GAUGE, len(cpu_per_core),
                       datetime.now(), MetricCategory.SYSTEM),
                Metric("system.cpu.frequency", MetricType.GAUGE, 
                       cpu_freq.current if cpu_freq else 0, 
                       datetime.now(), MetricCategory.SYSTEM),
                Metric("system.cpu.per_core_usage", MetricType.HISTOGRAM, 
                       cpu_per_core, datetime.now(), MetricCategory.SYSTEM)
            ])
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            metrics.update([
                Metric("system.memory.usage_percent", MetricType.GAUGE, memory.percent,
                       datetime.now(), MetricCategory.SYSTEM),
                Metric("system.memory.available", MetricType.GAUGE, memory.available,
                       datetime.now(), MetricCategory.SYSTEM),
                Metric("system.memory.used", MetricType.GAUGE, memory.used,
                       datetime.now(), MetricCategory.SYSTEM),
                Metric("system.memory.total", MetricType.GAUGE, memory.total,
                       datetime.now(), MetricCategory.SYSTEM),
                Metric("system.swap.usage_percent", MetricType.GAUGE, swap.percent,
                       datetime.now(), MetricCategory.SYSTEM),
                Metric("system.swap.used", MetricType.GAUGE, swap.used,
                       datetime.now(), MetricCategory.SYSTEM)
            ])
            
            # Disk metrics
            disk_usage = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            if disk_io:
                metrics.update([
                    Metric("system.disk.usage_percent", MetricType.GAUGE,
                           (disk_usage.used / disk_usage.total) * 100,
                           datetime.now(), MetricCategory.SYSTEM),
                    Metric("system.disk.used", MetricType.GAUGE, disk_usage.used,
                           datetime.now(), MetricCategory.SYSTEM),
                    Metric("system.disk.total", MetricType.GAUGE, disk_usage.total,
                           datetime.now(), MetricCategory.SYSTEM),
                    Metric("system.disk.io.read_bytes", MetricType.COUNTER,
                           disk_io.read_bytes, datetime.now(), MetricCategory.SYSTEM),
                    Metric("system.disk.io.write_bytes", MetricType.COUNTER,
                           disk_io.write_bytes, datetime.now(), MetricCategory.SYSTEM)
                ])
            
            # Network metrics
            network_io = psutil.net_io_counters()
            if network_io:
                metrics.update([
                    Metric("system.network.bytes_sent", MetricType.COUNTER,
                           network_io.bytes_sent, datetime.now(), MetricCategory.SYSTEM),
                    Metric("system.network.bytes_recv", MetricType.COUNTER,
                           network_io.bytes_recv, datetime.now(), MetricCategory.SYSTEM),
                    Metric("system.network.packets_sent", MetricType.COUNTER,
                           network_io.packets_sent, datetime.now(), MetricCategory.SYSTEM),
                    Metric("system.network.packets_recv", MetricType.COUNTER,
                           network_io.packets_recv, datetime.now(), MetricCategory.SYSTEM)
                ])
            
            # Process metrics
            process_memory = self._process.memory_info()
            process_cpu_percent = self._process.cpu_percent()
            thread_count = self._process.num_threads()
            file_count = len(self._process.open_files())
            
            metrics.update([
                Metric("process.memory.rss", MetricType.GAUGE, process_memory.rss,
                       datetime.now(), MetricCategory.PERFORMANCE),
                Metric("process.memory.vms", MetricType.GAUGE, process_memory.vms,
                       datetime.now(), MetricCategory.PERFORMANCE),
                Metric("process.cpu.usage_percent", MetricType.GAUGE, process_cpu_percent,
                       datetime.now(), MetricCategory.PERFORMANCE),
                Metric("process.threads", MetricType.GAUGE, thread_count,
                       datetime.now(), MetricCategory.PERFORMANCE),
                Metric("process.open_files", MetricType.GAUGE, file_count,
                       datetime.now(), MetricCategory.PERFORMANCE)
            ])
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
        
        return metrics


class ApplicationMonitor:
    """Application-level metrics monitoring"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._request_counts = defaultdict(int)
        self._error_counts = defaultdict(int)
        self._response_times = defaultdict(deque)
        self._active_connections = 0
        self._session_counts = defaultdict(int)
    
    def record_request(self, endpoint: str, duration: float, status_code: int):
        """Record HTTP request metrics"""
        timestamp = datetime.now()
        
        # Request count
        metric = Metric(
            "app.requests.total",
            MetricType.COUNTER,
            1,
            timestamp,
            MetricCategory.APPLICATION,
            tags={'endpoint': endpoint, 'status_code': str(status_code)}
        )
        self._request_counts[f"{endpoint}_{status_code}"] += 1
        
        # Response time
        if status_code >= 200 and status_code < 300:
            self._response_times[endpoint].append(duration)
            # Keep only last 1000 response times
            if len(self._response_times[endpoint]) > 1000:
                self._response_times[endpoint].popleft()
    
    def record_error(self, error_type: str, endpoint: str = None):
        """Record error metrics"""
        self._error_counts[f"{error_type}_{endpoint or 'unknown'}"] += 1
    
    def increment_connection(self):
        """Record active connection"""
        self._active_connections += 1
    
    def decrement_connection(self):
        """Record closed connection"""
        self._active_connections = max(0, self._active_connections - 1)
    
    def create_session(self, session_type: str):
        """Record session creation"""
        self._session_counts[session_type] += 1
    
    def get_application_metrics(self) -> Dict[str, Metric]:
        """Collect application-level metrics"""
        metrics = {}
        timestamp = datetime.now()
        
        # Request metrics
        total_requests = sum(self._request_counts.values())
        metrics["app.requests.total"] = Metric(
            "app.requests.total",
            MetricType.COUNTER,
            total_requests,
            timestamp,
            MetricCategory.APPLICATION
        )
        
        # Error metrics
        total_errors = sum(self._error_counts.values())
        error_rate = total_errors / max(total_requests, 1)
        metrics["app.errors.total"] = Metric(
            "app.errors.total",
            MetricType.COUNTER,
            total_errors,
            timestamp,
            MetricCategory.ERROR
        )
        metrics["app.errors.rate"] = Metric(
            "app.errors.rate",
            MetricType.GAUGE,
            error_rate,
            timestamp,
            MetricCategory.ERROR
        )
        
        # Response time metrics
        for endpoint, times in self._response_times.items():
            if times:
                avg_response_time = statistics.mean(times)
                median_response_time = statistics.median(times)
                p95_response_time = statistics.quantiles(times, n=20)[18]  # 95th percentile
                
                metrics[f"app.response_time.{endpoint}.avg"] = Metric(
                    f"app.response_time.{endpoint}.avg",
                    MetricType.GAUGE,
                    avg_response_time,
                    timestamp,
                    MetricCategory.PERFORMANCE
                )
                
                metrics[f"app.response_time.{endpoint}.median"] = Metric(
                    f"app.response_time.{endpoint}.median",
                    MetricType.GAUGE,
                    median_response_time,
                    timestamp,
                    MetricCategory.PERFORMANCE
                )
                
                metrics[f"app.response_time.{endpoint}.p95"] = Metric(
                    f"app.response_time.{endpoint}.p95",
                    MetricType.GAUGE,
                    p95_response_time,
                    timestamp,
                    MetricCategory.PERFORMANCE
                )
        
        # Connection metrics
        metrics["app.connections.active"] = Metric(
            "app.connections.active",
            MetricType.GAUGE,
            self._active_connections,
            timestamp,
            MetricCategory.APPLICATION
        )
        
        # Session metrics
        for session_type, count in self._session_counts.items():
            metrics[f"app.sessions.{session_type}.total"] = Metric(
                f"app.sessions.{session_type}.total",
                MetricType.COUNTER,
                count,
                timestamp,
                MetricCategory.APPLICATION
            )
        
        return metrics


class CustomMetricsCollector:
    """Custom business metrics collector"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._custom_metrics = defaultdict(lambda: deque(maxlen=1000))
        self._counters = defaultdict(int)
    
    def record_counter(self, name: str, value: int = 1, tags: Dict[str, str] = None):
        """Record a counter metric"""
        self._counters[name] += value
        self._custom_metrics[f"{name}_counter"].append(
            (datetime.now(), self._counters[name])
        )
        self.logger.debug(f"Recorded counter: {name} = {self._counters[name]}")
    
    def record_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a gauge metric"""
        self._custom_metrics[name].append((datetime.now(), value))
        self.logger.debug(f"Recorded gauge: {name} = {value}")
    
    def record_timing(self, name: str, duration: float, tags: Dict[str, str] = None):
        """Record a timing metric"""
        self._custom_metrics[name].append((datetime.now(), duration))
        self.logger.debug(f"Recorded timing: {name} = {duration}s")
    
    def get_custom_metrics(self) -> Dict[str, Metric]:
        """Collect custom metrics"""
        metrics = {}
        timestamp = datetime.now()
        
        for name, values in self._custom_metrics.items():
            if not values:
                continue
            
            latest_value = values[-1][1]
            
            # Determine metric type based on name
            if name.endswith('_counter'):
                metric_type = MetricType.COUNTER
                metric_name = name.replace('_counter', '')
            elif 'response_time' in name or 'duration' in name:
                metric_type = MetricType.TIMER
                metric_name = name
                # Also compute statistics
                timings = [v[1] for v in values if isinstance(v[1], (int, float))]
                if timings:
                    metrics[f"{name}.avg"] = Metric(
                        f"{name}.avg",
                        MetricType.GAUGE,
                        statistics.mean(timings),
                        timestamp,
                        MetricCategory.PERFORMANCE
                    )
                    metrics[f"{name}.median"] = Metric(
                        f"{name}.median",
                        MetricType.GAUGE,
                        statistics.median(timings),
                        timestamp,
                        MetricCategory.PERFORMANCE
                    )
            else:
                metric_type = MetricType.GAUGE
                metric_name = name
            
            metrics[metric_name] = Metric(
                metric_name,
                metric_type,
                latest_value,
                timestamp,
                MetricCategory.BUSINESS
            )
        
        return metrics


class AlertManager:
    """Alert management system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.alert_rules = []
        self.active_alerts = {}
        self.alert_history = []
    
    def add_rule(self, rule: AlertRule):
        """Add alert rule"""
        self.alert_rules.append(rule)
        self.logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """Remove alert rule"""
        self.alert_rules = [r for r in self.alert_rules if r.name != rule_name]
        self.logger.info(f"Removed alert rule: {rule_name}")
    
    def check_alerts(self, metrics: Dict[str, Metric]) -> List[Alert]:
        """Check current metrics against alert rules"""
        triggered_alerts = []
        timestamp = datetime.now()
        
        for rule in self.alert_rules:
            # Find matching metric
            matching_metrics = [
                metric for name, metric in metrics.items() 
                if rule.metric_name in name
            ]
            
            if not matching_metrics:
                continue
            
            # Check latest metric value
            latest_metric = matching_metrics[-1]
            current_value = latest_metric.value
            
            if not isinstance(current_value, (int, float)):
                continue
            
            # Check if alert condition is met
            alert_triggered = rule.check(current_value)
            
            # Check if rule already has an active alert
            alert_key = f"{rule.name}_{rule.metric_name}"
            existing_alert = self.active_alerts.get(alert_key)
            
            if alert_triggered:
                if existing_alert:
                    # Update existing alert
                    existing_alert.triggered_count += 1
                    existing_alert.metric_value = current_value
                else:
                    # Create new alert
                    alert = Alert(
                        rule=rule,
                        metric_value=current_value,
                        triggered_at=timestamp
                    )
                    
                    self.active_alerts[alert_key] = alert
                    triggered_alerts.append(alert)
                    
                    # Call notification callback
                    if rule.notification_callback:
                        try:
                            rule.notification_callback(alert)
                        except Exception as e:
                            self.logger.error(f"Alert notification callback failed: {e}")
                
                self.logger.warning(f"Alert triggered: {rule.name} - {current_value}")
            
            else:
                # Alert condition no longer met, resolve active alert
                if existing_alert and not existing_alert.resolved:
                    existing_alert.resolved = True
                    existing_alert.resolved_at = timestamp
                    self.alert_history.append(existing_alert)
                    del self.active_alerts[alert_key]
                    self.logger.info(f"Alert resolved: {rule.name}")
        
        return triggered_alerts
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts"""
        return [alert.to_dict() for alert in self.active_alerts.values()]
    
    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get alert history"""
        return [alert.to_dict() for alert in self.alert_history[-limit:]]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        active_count = len(self.active_alerts)
        total_rules = len(self.alert_rules)
        
        # Count by severity
        severity_counts = defaultdict(int)
        for alert in self.alert_history:
            severity_counts[alert.rule.severity.value] += 1
        
        return {
            'active_alerts': active_count,
            'total_rules': total_rules,
            'rules_enabled': sum(1 for rule in self.alert_rules if rule.enabled),
            'alert_history_count': len(self.alert_history),
            'severity_distribution': dict(severity_counts)
        }


class MetricsCollector:
    """Comprehensive performance metrics collector"""
    
    def __init__(self, collection_interval: int = 10):
        self.collection_interval = collection_interval
        self.is_collecting = False
        self.collection_task = None
        
        # Initialize monitors
        self.system_monitor = SystemMonitor()
        self.app_monitor = ApplicationMonitor()
        self.custom_collector = CustomMetricsCollector()
        self.alert_manager = AlertManager()
        
        # Metrics storage
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        self.latest_metrics = {}
        
        self.logger = logging.getLogger(__name__)
        
        # Setup default alert rules
        self._setup_default_alerts()
    
    def _setup_default_alerts(self):
        """Setup default alert rules"""
        # CPU usage alerts
        self.alert_manager.add_rule(AlertRule(
            name="High CPU Usage",
            metric_name="system.cpu.usage_percent",
            condition="gt",
            threshold=80.0,
            severity=AlertSeverity.WARNING,
            duration_seconds=60,
            description="CPU usage is above 80%"
        ))
        
        self.alert_manager.add_rule(AlertRule(
            name="Critical CPU Usage",
            metric_name="system.cpu.usage_percent",
            condition="gt",
            threshold=95.0,
            severity=AlertSeverity.CRITICAL,
            duration_seconds=30,
            description="CPU usage is above 95%"
        ))
        
        # Memory usage alerts
        self.alert_manager.add_rule(AlertRule(
            name="High Memory Usage",
            metric_name="system.memory.usage_percent",
            condition="gt",
            threshold=85.0,
            severity=AlertSeverity.WARNING,
            duration_seconds=60,
            description="Memory usage is above 85%"
        ))
        
        # Error rate alerts
        self.alert_manager.add_rule(AlertRule(
            name="High Error Rate",
            metric_name="app.errors.rate",
            condition="gt",
            threshold=0.05,
            severity=AlertSeverity.ERROR,
            duration_seconds=120,
            description="Error rate is above 5%"
        ))
        
        # Response time alerts
        self.alert_manager.add_rule(AlertRule(
            name="Slow Response Time",
            metric_name="app.response_time.avg",
            condition="gt",
            threshold=5.0,
            severity=AlertSeverity.WARNING,
            duration_seconds=60,
            description="Average response time is above 5 seconds"
        ))
    
    async def start_collection(self):
        """Start metrics collection"""
        if self.is_collecting:
            return
        
        self.is_collecting = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        self.logger.info("Started metrics collection")
    
    async def stop_collection(self):
        """Stop metrics collection"""
        self.is_collecting = False
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Stopped metrics collection")
    
    async def _collection_loop(self):
        """Main collection loop"""
        while self.is_collecting:
            try:
                await self.collect_metrics()
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                self.logger.error(f"Error in collection loop: {e}")
                await asyncio.sleep(1)
    
    async def collect_metrics(self):
        """Collect all metrics"""
        try:
            # Collect system metrics
            system_metrics = self.system_monitor.get_system_metrics()
            
            # Collect application metrics
            app_metrics = self.app_monitor.get_application_metrics()
            
            # Collect custom metrics
            custom_metrics = self.custom_collector.get_custom_metrics()
            
            # Combine all metrics
            all_metrics = {**system_metrics, **app_metrics, **custom_metrics}
            
            # Store metrics
            timestamp = datetime.now()
            for name, metric in all_metrics.items():
                self.latest_metrics[name] = metric
                self.metrics_history[name].append(metric)
            
            # Check alerts
            alerts = self.alert_manager.check_alerts(all_metrics)
            
            if alerts:
                self.logger.warning(f"Generated {len(alerts)} alerts")
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
    
    def get_latest_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get latest metrics"""
        return {name: metric.to_dict() for name, metric in self.latest_metrics.items()}
    
    def get_metrics_history(self, metric_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get metrics history for specific metric"""
        if metric_name not in self.metrics_history:
            return []
        
        return [metric.to_dict() for metric in list(self.metrics_history[metric_name])[-limit:]]
    
    def get_all_metrics_history(self, limit: int = 100) -> Dict[str, List[Dict[str, Any]]]:
        """Get metrics history for all metrics"""
        return {
            name: [metric.to_dict() for metric in list(metrics)[-limit:]]
            for name, metrics in self.metrics_history.items()
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        latest = self.latest_metrics
        alerts = self.alert_manager.get_active_alerts()
        
        # Extract key system metrics
        cpu_usage = None
        memory_usage = None
        disk_usage = None
        
        for name, metric in latest.items():
            if name == "system.cpu.usage_percent":
                cpu_usage = metric.value
            elif name == "system.memory.usage_percent":
                memory_usage = metric.value
            elif name == "system.disk.usage_percent":
                disk_usage = metric.value
        
        # Calculate uptime (approximate)
        uptime = time.time() - getattr(self, '_start_time', time.time())
        
        return {
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': uptime,
            'system': {
                'cpu_usage_percent': cpu_usage,
                'memory_usage_percent': memory_usage,
                'disk_usage_percent': disk_usage
            },
            'application': {
                'active_alerts': len(alerts),
                'total_metrics': len(self.latest_metrics),
                'collecting_active': self.is_collecting
            },
            'alerts': {
                'active': len(alerts),
                'active_alerts': alerts
            }
        }
    
    def export_metrics(self, format_type: str = "json") -> str:
        """Export metrics in specified format"""
        if format_type.lower() == "json":
            return json.dumps({
                'timestamp': datetime.now().isoformat(),
                'latest_metrics': self.get_latest_metrics(),
                'alert_summary': self.alert_manager.get_alert_statistics()
            }, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def add_custom_alert_rule(self, 
                             name: str,
                             metric_name: str,
                             condition: str,
                             threshold: Union[int, float],
                             severity: AlertSeverity,
                             **kwargs):
        """Add custom alert rule"""
        rule = AlertRule(
            name=name,
            metric_name=metric_name,
            condition=condition,
            threshold=threshold,
            severity=severity,
            **kwargs
        )
        self.alert_manager.add_rule(rule)
    
    def record_custom_metric(self, 
                            name: str, 
                            value: Union[int, float], 
                            metric_type: str = "gauge",
                            category: str = "business"):
        """Record custom metric"""
        metric_category = getattr(MetricCategory, category.upper(), MetricCategory.BUSINESS)
        
        metric = Metric(
            name=name,
            type=MetricType(metric_type),
            value=value,
            timestamp=datetime.now(),
            category=metric_category
        )
        
        self.latest_metrics[name] = metric
        self.metrics_history[name].append(metric)
    
    def shutdown(self):
        """Shutdown metrics collector"""
        asyncio.create_task(self.stop_collection())
        self.logger.info("Metrics collector shutdown")


# Global metrics collector instance
_metrics_collector = None

def get_metrics_collector(collection_interval: int = 10) -> MetricsCollector:
    """Get global metrics collector instance"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector(collection_interval)
    return _metrics_collector


def initialize_metrics_collection(config: Dict[str, Any]) -> MetricsCollector:
    """Initialize metrics collection with configuration"""
    collector = MetricsCollector(
        collection_interval=config.get('collection_interval', 10)
    )
    return collector