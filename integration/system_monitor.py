"""
System Monitor
Component health checks, monitoring, and alert escalation system.
"""

import asyncio
import logging
import json
import time
import psutil
import smtplib
import ssl
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiohttp
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ComponentStatus(Enum):
    """Component status enumeration."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"
    OFFLINE = "offline"


class MetricType(Enum):
    """Metric type enumeration."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    CUSTOM = "custom"


@dataclass
class HealthMetric:
    """Health metric data."""
    name: str
    value: float
    threshold: float
    unit: str
    timestamp: datetime
    status: ComponentStatus


@dataclass
class ComponentHealth:
    """Component health information."""
    name: str
    status: ComponentStatus
    last_check: datetime
    metrics: List[HealthMetric]
    uptime: float
    error_count: int
    warning_count: int


@dataclass
class Alert:
    """Alert information."""
    id: str
    component: str
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    acknowledged: bool = False


class SystemMonitor:
    """Comprehensive system monitoring and alerting."""
    
    def __init__(self, config_file: str = "config/monitoring.json"):
        self.config_file = config_file
        self.config = self._load_config()
        self.components: Dict[str, ComponentHealth] = {}
        self.alerts: List[Alert] = []
        self.monitoring_active = False
        self.check_interval = self.config.get("check_interval", 60)
        self.alert_thresholds = self.config.get("alert_thresholds", {})
        self.notification_settings = self.config.get("notifications", {})
        
        # Monitoring metrics history
        self.metrics_history: Dict[str, List[HealthMetric]] = {}
        
        # Alert escalation rules
        self.escalation_rules = self.config.get("escalation_rules", {})
        
        # Active monitoring tasks
        self.monitoring_tasks: List[asyncio.Task] = []
    
    def _load_config(self) -> Dict[str, Any]:
        """Load monitoring configuration."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading monitoring config: {str(e)}")
        
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default monitoring configuration."""
        return {
            "check_interval": 60,
            "alert_thresholds": {
                "cpu": {"warning": 70, "critical": 90},
                "memory": {"warning": 80, "critical": 95},
                "disk": {"warning": 85, "critical": 95},
                "response_time": {"warning": 2.0, "critical": 5.0},
                "error_rate": {"warning": 5.0, "critical": 10.0}
            },
            "escalation_rules": {
                "critical": {"immediate": True, "retry_interval": 300},
                "high": {"immediate": False, "retry_interval": 1800},
                "medium": {"immediate": False, "retry_interval": 3600}
            },
            "notifications": {
                "email": {
                    "enabled": False,
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "username": "",
                    "password": "",
                    "from_email": "",
                    "to_emails": []
                },
                "webhook": {
                    "enabled": False,
                    "url": "",
                    "headers": {}
                }
            },
            "components": [
                {
                    "name": "system",
                    "type": "system",
                    "enabled": True,
                    "endpoints": []
                },
                {
                    "name": "api",
                    "type": "api",
                    "enabled": True,
                    "endpoints": [
                        {"url": "http://localhost:8000/health", "method": "GET"},
                        {"url": "http://localhost:8000/api/status", "method": "GET"}
                    ]
                }
            ]
        }
    
    async def start_monitoring(self):
        """Start system monitoring."""
        logger.info("Starting system monitoring...")
        
        self.monitoring_active = True
        
        # Initialize components
        await self._initialize_components()
        
        # Start monitoring tasks
        for component_config in self.config.get("components", []):
            if component_config.get("enabled", True):
                task = asyncio.create_task(
                    self._monitor_component(component_config)
                )
                self.monitoring_tasks.append(task)
        
        # Start alert processing
        alert_task = asyncio.create_task(self._process_alerts())
        self.monitoring_tasks.append(alert_task)
        
        logger.info(f"System monitoring started with {len(self.monitoring_tasks)} tasks")
    
    async def stop_monitoring(self):
        """Stop system monitoring."""
        logger.info("Stopping system monitoring...")
        
        self.monitoring_active = False
        
        # Cancel all monitoring tasks
        for task in self.monitoring_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.monitoring_tasks:
            await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        
        self.monitoring_tasks.clear()
        logger.info("System monitoring stopped")
    
    async def _initialize_components(self):
        """Initialize monitoring components."""
        for component_config in self.config.get("components", []):
            component_name = component_config["name"]
            self.components[component_name] = ComponentHealth(
                name=component_name,
                status=ComponentStatus.UNKNOWN,
                last_check=datetime.now(),
                metrics=[],
                uptime=0,
                error_count=0,
                warning_count=0
            )
            
            # Initialize metrics history
            self.metrics_history[component_name] = []
        
        logger.info(f"Initialized {len(self.components)} monitoring components")
    
    async def _monitor_component(self, component_config: Dict[str, Any]):
        """Monitor a specific component."""
        component_name = component_config["name"]
        component_type = component_config["type"]
        
        while self.monitoring_active:
            try:
                start_time = time.time()
                
                # Perform health check based on component type
                if component_type == "system":
                    health_info = await self._check_system_health(component_name)
                elif component_type == "api":
                    health_info = await self._check_api_health(component_name, component_config)
                elif component_type == "database":
                    health_info = await self._check_database_health(component_name, component_config)
                else:
                    health_info = await self._check_custom_component(component_name, component_config)
                
                # Update component health
                self.components[component_name] = health_info
                
                # Store metrics in history
                if component_name in self.metrics_history:
                    self.metrics_history[component_name].extend(health_info.metrics)
                    
                    # Keep only recent metrics (last 1000 per component)
                    if len(self.metrics_history[component_name]) > 1000:
                        self.metrics_history[component_name] = self.metrics_history[component_name][-1000:]
                
                # Check for alerts
                await self._check_alerts(health_info)
                
                # Calculate check duration
                check_duration = time.time() - start_time
                logger.debug(f"Component {component_name} check completed in {check_duration:.2f}s")
                
            except Exception as e:
                logger.error(f"Error monitoring component {component_name}: {str(e)}")
                
                # Update component status to critical on errors
                if component_name in self.components:
                    self.components[component_name].status = ComponentStatus.CRITICAL
                    self.components[component_name].error_count += 1
                
                # Create error alert
                await self._create_alert(
                    component=component_name,
                    severity=AlertSeverity.HIGH,
                    title="Component Monitoring Error",
                    message=f"Failed to check component health: {str(e)}"
                )
            
            # Wait for next check
            await asyncio.sleep(self.check_interval)
    
    async def _check_system_health(self, component_name: str) -> ComponentHealth:
        """Check system-level health metrics."""
        metrics = []
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_metric = HealthMetric(
                name="cpu_usage",
                value=cpu_percent,
                threshold=self.alert_thresholds.get("cpu", {}).get("critical", 90),
                unit="%",
                timestamp=datetime.now(),
                status=self._get_metric_status(cpu_percent, "cpu")
            )
            metrics.append(cpu_metric)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_metric = HealthMetric(
                name="memory_usage",
                value=memory.percent,
                threshold=self.alert_thresholds.get("memory", {}).get("critical", 95),
                unit="%",
                timestamp=datetime.now(),
                status=self._get_metric_status(memory.percent, "memory")
            )
            metrics.append(memory_metric)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_metric = HealthMetric(
                name="disk_usage",
                value=disk_percent,
                threshold=self.alert_thresholds.get("disk", {}).get("critical", 95),
                unit="%",
                timestamp=datetime.now(),
                status=self._get_metric_status(disk_percent, "disk")
            )
            metrics.append(disk_metric)
            
            # System uptime
            uptime = time.time() - psutil.boot_time()
            uptime_metric = HealthMetric(
                name="system_uptime",
                value=uptime,
                threshold=0,
                unit="seconds",
                timestamp=datetime.now(),
                status=ComponentStatus.HEALTHY
            )
            metrics.append(uptime_metric)
            
            # Determine overall status
            overall_status = ComponentStatus.HEALTHY
            warning_count = 0
            critical_count = 0
            
            for metric in metrics:
                if metric.status == ComponentStatus.CRITICAL:
                    critical_count += 1
                    overall_status = ComponentStatus.CRITICAL
                elif metric.status == ComponentStatus.WARNING:
                    warning_count += 1
                    if overall_status == ComponentStatus.HEALTHY:
                        overall_status = ComponentStatus.WARNING
            
            return ComponentHealth(
                name=component_name,
                status=overall_status,
                last_check=datetime.now(),
                metrics=metrics,
                uptime=uptime,
                error_count=critical_count,
                warning_count=warning_count
            )
            
        except Exception as e:
            logger.error(f"Error checking system health: {str(e)}")
            raise
    
    async def _check_api_health(self, component_name: str, 
                              component_config: Dict[str, Any]) -> ComponentHealth:
        """Check API health."""
        metrics = []
        endpoints = component_config.get("endpoints", [])
        
        if not endpoints:
            # If no endpoints configured, check default health endpoint
            endpoints = [{"url": "http://localhost:8000/health", "method": "GET"}]
        
        total_response_time = 0
        successful_requests = 0
        total_requests = len(endpoints)
        
        async with aiohttp.ClientSession() as session:
            for endpoint in endpoints:
                try:
                    start_time = time.time()
                    
                    url = endpoint["url"]
                    method = endpoint.get("method", "GET")
                    
                    if method.upper() == "GET":
                        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                            response_time = time.time() - start_time
                            
                            if response.status == 200:
                                successful_requests += 1
                            
                            # Response time metric
                            response_time_metric = HealthMetric(
                                name=f"response_time_{url}",
                                value=response_time,
                                threshold=self.alert_thresholds.get("response_time", {}).get("critical", 5.0),
                                unit="seconds",
                                timestamp=datetime.now(),
                                status=self._get_metric_status(response_time, "response_time")
                            )
                            metrics.append(response_time_metric)
                    
                except asyncio.TimeoutError:
                    response_time = 10.0  # Timeout value
                    response_time_metric = HealthMetric(
                        name=f"response_time_{url}",
                        value=response_time,
                        threshold=self.alert_thresholds.get("response_time", {}).get("critical", 5.0),
                        unit="seconds",
                        timestamp=datetime.now(),
                        status=ComponentStatus.CRITICAL
                    )
                    metrics.append(response_time_metric)
                    
                except Exception as e:
                    logger.warning(f"API endpoint check failed for {url}: {str(e)}")
                    error_metric = HealthMetric(
                        name=f"error_{url}",
                        value=1.0,
                        threshold=0.0,
                        unit="error",
                        timestamp=datetime.now(),
                        status=ComponentStatus.CRITICAL
                    )
                    metrics.append(error_metric)
                
                total_response_time += response_time if 'response_time' in locals() else 10.0
        
        # Calculate success rate
        success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0
        error_rate = 100 - success_rate
        
        # Error rate metric
        error_rate_metric = HealthMetric(
            name="error_rate",
            value=error_rate,
            threshold=self.alert_thresholds.get("error_rate", {}).get("critical", 10.0),
            unit="%",
            timestamp=datetime.now(),
            status=self._get_metric_status(error_rate, "error_rate")
        )
        metrics.append(error_rate_metric)
        
        # Determine overall status
        overall_status = ComponentStatus.HEALTHY
        error_count = 0
        warning_count = 0
        
        for metric in metrics:
            if metric.status == ComponentStatus.CRITICAL:
                error_count += 1
                overall_status = ComponentStatus.CRITICAL
            elif metric.status == ComponentStatus.WARNING:
                warning_count += 1
                if overall_status == ComponentStatus.HEALTHY:
                    overall_status = ComponentStatus.WARNING
        
        return ComponentHealth(
            name=component_name,
            status=overall_status,
            last_check=datetime.now(),
            metrics=metrics,
            uptime=0,  # Would be calculated based on service uptime
            error_count=error_count,
            warning_count=warning_count
        )
    
    async def _check_database_health(self, component_name: str, 
                                   component_config: Dict[str, Any]) -> ComponentHealth:
        """Check database health."""
        # This would implement actual database health checks
        # For now, return a mock healthy status
        
        metrics = [
            HealthMetric(
                name="connection_count",
                value=10,
                threshold=100,
                unit="connections",
                timestamp=datetime.now(),
                status=ComponentStatus.HEALTHY
            ),
            HealthMetric(
                name="query_response_time",
                value=0.05,
                threshold=1.0,
                unit="seconds",
                timestamp=datetime.now(),
                status=ComponentStatus.HEALTHY
            )
        ]
        
        return ComponentHealth(
            name=component_name,
            status=ComponentStatus.HEALTHY,
            last_check=datetime.now(),
            metrics=metrics,
            uptime=3600,  # Mock uptime
            error_count=0,
            warning_count=0
        )
    
    async def _check_custom_component(self, component_name: str, 
                                    component_config: Dict[str, Any]) -> ComponentHealth:
        """Check custom component health."""
        # This would implement custom health checks based on component configuration
        # For now, return a mock healthy status
        
        return ComponentHealth(
            name=component_name,
            status=ComponentStatus.HEALTHY,
            last_check=datetime.now(),
            metrics=[],
            uptime=0,
            error_count=0,
            warning_count=0
        )
    
    def _get_metric_status(self, value: float, metric_type: str) -> ComponentStatus:
        """Determine metric status based on value and thresholds."""
        thresholds = self.alert_thresholds.get(metric_type, {})
        warning_threshold = thresholds.get("warning", float('inf'))
        critical_threshold = thresholds.get("critical", float('inf'))
        
        if value >= critical_threshold:
            return ComponentStatus.CRITICAL
        elif value >= warning_threshold:
            return ComponentStatus.WARNING
        else:
            return ComponentStatus.HEALTHY
    
    async def _check_alerts(self, component_health: ComponentHealth):
        """Check if alerts should be triggered based on component health."""
        component_name = component_health.name
        
        # Check for new critical/warning statuses
        for metric in component_health.metrics:
            if metric.status in [ComponentStatus.CRITICAL, ComponentStatus.WARNING]:
                # Check if an alert already exists for this condition
                existing_alert = self._find_existing_alert(component_name, metric.name)
                
                if not existing_alert or existing_alert.resolved:
                    # Create new alert
                    severity = AlertSeverity.CRITICAL if metric.status == ComponentStatus.CRITICAL else AlertSeverity.HIGH
                    title = f"{component_name}: {metric.name} threshold exceeded"
                    message = f"{metric.name} is {metric.value}{metric.unit}, threshold is {metric.threshold}{metric.unit}"
                    
                    await self._create_alert(component_name, severity, title, message)
            
            elif metric.status == ComponentStatus.HEALTHY and metric.value != 0:
                # Check if we should resolve any existing alerts
                await self._resolve_alerts(component_name, metric.name)
        
        # Check overall component status
        if component_health.status == ComponentStatus.CRITICAL:
            existing_alert = self._find_existing_alert(component_name, "overall_status")
            if not existing_alert or existing_alert.resolved:
                await self._create_alert(
                    component_name,
                    AlertSeverity.CRITICAL,
                    f"{component_name} is in critical state",
                    f"Component {component_name} health check failed"
                )
        elif component_health.status == ComponentStatus.HEALTHY:
            await self._resolve_alerts(component_name, "overall_status")
    
    def _find_existing_alert(self, component: str, metric_name: str) -> Optional[Alert]:
        """Find existing unresolved alert for component and metric."""
        for alert in self.alerts:
            if (not alert.resolved and 
                alert.component == component and 
                metric_name in alert.title):
                return alert
        return None
    
    async def _create_alert(self, component: str, severity: AlertSeverity, 
                          title: str, message: str):
        """Create new alert."""
        alert_id = f"{component}_{int(time.time())}"
        
        alert = Alert(
            id=alert_id,
            component=component,
            severity=severity,
            title=title,
            message=message,
            timestamp=datetime.now()
        )
        
        self.alerts.append(alert)
        
        # Log alert
        logger.warning(f"ALERT [{severity.value.upper()}] {component}: {title}")
        
        # Send notifications if configured
        await self._send_notifications(alert)
        
        return alert
    
    async def _resolve_alerts(self, component: str, metric_name: str):
        """Resolve alerts for component and metric."""
        for alert in self.alerts:
            if (not alert.resolved and 
                alert.component == component and 
                metric_name in alert.title):
                alert.resolved = True
                alert.resolved_at = datetime.now()
                logger.info(f"Resolved alert: {alert.title}")
    
    async def _send_notifications(self, alert: Alert):
        """Send alert notifications."""
        # Email notification
        if self.notification_settings.get("email", {}).get("enabled", False):
            await self._send_email_notification(alert)
        
        # Webhook notification
        if self.notification_settings.get("webhook", {}).get("enabled", False):
            await self._send_webhook_notification(alert)
    
    async def _send_email_notification(self, alert: Alert):
        """Send email notification for alert."""
        try:
            email_config = self.notification_settings["email"]
            
            msg = MIMEMultipart()
            msg['From'] = email_config["from_email"]
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            body = f"""
Alert Details:
Component: {alert.component}
Severity: {alert.severity.value.upper()}
Title: {alert.title}
Message: {alert.message}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(email_config["smtp_server"], email_config["smtp_port"])
            server.starttls()
            server.login(email_config["username"], email_config["password"])
            
            for to_email in email_config["to_emails"]:
                msg['To'] = to_email
                text = msg.as_string()
                server.sendmail(email_config["from_email"], to_email, text)
                del msg['To']
            
            server.quit()
            logger.info(f"Email notification sent for alert: {alert.id}")
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {str(e)}")
    
    async def _send_webhook_notification(self, alert: Alert):
        """Send webhook notification for alert."""
        try:
            webhook_config = self.notification_settings["webhook"]
            webhook_url = webhook_config["url"]
            headers = webhook_config.get("headers", {})
            
            payload = {
                "alert_id": alert.id,
                "component": alert.component,
                "severity": alert.severity.value,
                "title": alert.title,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat()
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logger.info(f"Webhook notification sent for alert: {alert.id}")
                    else:
                        logger.warning(f"Webhook notification failed: {response.status}")
                        
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {str(e)}")
    
    async def _process_alerts(self):
        """Process alert escalation and cleanup."""
        while self.monitoring_active:
            try:
                current_time = datetime.now()
                
                # Process alert escalation
                for alert in self.alerts:
                    if not alert.resolved and not alert.acknowledged:
                        escalation_rule = self.escalation_rules.get(alert.severity.value, {})
                        
                        # Check if alert should be escalated
                        time_since_creation = (current_time - alert.timestamp).total_seconds()
                        retry_interval = escalation_rule.get("retry_interval", 3600)
                        
                        if time_since_creation > retry_interval:
                            logger.warning(f"Escalating alert: {alert.title}")
                            # Re-send notifications (escalation)
                            await self._send_notifications(alert)
                
                # Clean up old resolved alerts (older than 7 days)
                cutoff_time = current_time - timedelta(days=7)
                self.alerts = [
                    alert for alert in self.alerts
                    if not alert.resolved or alert.resolved_at > cutoff_time
                ]
                
                # Wait before processing again
                await asyncio.sleep(300)  # Process every 5 minutes
                
            except Exception as e:
                logger.error(f"Error processing alerts: {str(e)}")
                await asyncio.sleep(300)
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                logger.info(f"Alert acknowledged: {alert_id}")
                return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Manually resolve an alert."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                logger.info(f"Alert manually resolved: {alert_id}")
                return True
        return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        active_alerts = [a for a in self.alerts if not a.resolved]
        critical_alerts = [a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]
        high_alerts = [a for a in active_alerts if a.severity == AlertSeverity.HIGH]
        
        # Calculate overall system health
        total_components = len(self.components)
        healthy_components = len([c for c in self.components.values() if c.status == ComponentStatus.HEALTHY])
        
        if total_components == 0:
            system_health = "unknown"
        elif healthy_components == total_components:
            system_health = "healthy"
        elif healthy_components >= total_components * 0.8:
            system_health = "warning"
        else:
            system_health = "critical"
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system_health": system_health,
            "monitoring_active": self.monitoring_active,
            "components": {
                name: {
                    "status": component.status.value,
                    "last_check": component.last_check.isoformat(),
                    "error_count": component.error_count,
                    "warning_count": component.warning_count,
                    "metrics_count": len(component.metrics)
                }
                for name, component in self.components.items()
            },
            "alerts": {
                "total": len(self.alerts),
                "active": len(active_alerts),
                "critical": len(critical_alerts),
                "high": len(high_alerts),
                "resolved": len([a for a in self.alerts if a.resolved])
            },
            "recent_alerts": [
                {
                    "id": alert.id,
                    "component": alert.component,
                    "severity": alert.severity.value,
                    "title": alert.title,
                    "timestamp": alert.timestamp.isoformat(),
                    "resolved": alert.resolved
                }
                for alert in sorted(self.alerts, key=lambda a: a.timestamp, reverse=True)[:10]
            ]
        }
    
    def get_component_metrics(self, component_name: str, 
                            hours: int = 24) -> List[Dict[str, Any]]:
        """Get historical metrics for a component."""
        if component_name not in self.metrics_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_metrics = [
            metric for metric in self.metrics_history[component_name]
            if metric.timestamp > cutoff_time
        ]
        
        return [asdict(metric) for metric in recent_metrics]


# Global system monitor instance
system_monitor = SystemMonitor()