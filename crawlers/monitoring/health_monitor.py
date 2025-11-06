"""
Crawler Health Monitoring and Alert System
Monitors crawler health, performance, and sends alerts
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

from ..base.base_crawler import CrawlerStatus
from ..scheduling.crawler_manager import CrawlerManager


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert delivery channels"""
    EMAIL = "email"
    WEBHOOK = "webhook"
    LOG = "log"
    SLACK = "slack"
    SMS = "sms"


@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    condition: str  # Expression to evaluate
    threshold: float
    level: AlertLevel
    channels: List[AlertChannel]
    enabled: bool = True
    cooldown_minutes: int = 60  # Minimum time between similar alerts
    description: str = ""


@dataclass
class Alert:
    """Alert instance"""
    id: str
    level: AlertLevel
    title: str
    message: str
    crawler_name: Optional[str]
    timestamp: datetime
    rule_name: str
    data: Dict[str, Any]
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class HealthMetric:
    """Health metric data point"""
    crawler_name: str
    metric_name: str
    value: float
    timestamp: datetime
    unit: str = ""
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None


class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Alert rules
        self.rules: Dict[str, AlertRule] = {}
        
        # Active alerts
        self.active_alerts: Dict[str, Alert] = {}
        
        # Alert history
        self.alert_history: List[Alert] = []
        
        # Alert tracking for cooldown
        self.last_alert_times: Dict[str, datetime] = {}
        
        # Notification channels
        self.email_config = config.get('email', {})
        self.webhook_config = config.get('webhook', {})
        self.slack_config = config.get('slack', {})
    
    def add_rule(self, rule: AlertRule):
        """Add alert rule"""
        self.rules[rule.name] = rule
        self.logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """Remove alert rule"""
        if rule_name in self.rules:
            del self.rules[rule_name]
            self.logger.info(f"Removed alert rule: {rule_name}")
    
    async def check_and_send_alert(self, crawler_name: str, metric_name: str, 
                                  value: float, timestamp: datetime,
                                  additional_data: Dict[str, Any] = None) -> bool:
        """Check conditions and send alert if needed"""
        alerts_sent = False
        
        for rule_name, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            try:
                # Check cooldown
                cooldown_key = f"{rule_name}_{crawler_name}_{metric_name}"
                if cooldown_key in self.last_alert_times:
                    time_since_last = (datetime.now() - self.last_alert_times[cooldown_key]).total_seconds()
                    if time_since_last < (rule.cooldown_minutes * 60):
                        continue  # Still in cooldown period
                
                # Evaluate condition
                if self._evaluate_condition(rule.condition, value, rule.threshold):
                    alert = await self._create_alert(
                        rule=rule,
                        crawler_name=crawler_name,
                        metric_name=metric_name,
                        value=value,
                        timestamp=timestamp,
                        additional_data=additional_data or {}
                    )
                    
                    if await self._send_alert(alert):
                        self.last_alert_times[cooldown_key] = datetime.now()
                        alerts_sent = True
                
            except Exception as e:
                self.logger.error(f"Error checking rule {rule_name}: {e}")
        
        return alerts_sent
    
    def _evaluate_condition(self, condition: str, value: float, threshold: float) -> bool:
        """Evaluate alert condition"""
        try:
            if condition == ">":
                return value > threshold
            elif condition == "<":
                return value < threshold
            elif condition == ">=":
                return value >= threshold
            elif condition == "<=":
                return value <= threshold
            elif condition == "==":
                return abs(value - threshold) < 0.001  # Float comparison
            elif condition == "!=":
                return abs(value - threshold) >= 0.001
            else:
                self.logger.warning(f"Unknown condition: {condition}")
                return False
        except Exception as e:
            self.logger.error(f"Error evaluating condition: {e}")
            return False
    
    async def _create_alert(self, rule: AlertRule, crawler_name: str, 
                           metric_name: str, value: float, timestamp: datetime,
                           additional_data: Dict[str, Any]) -> Alert:
        """Create alert instance"""
        alert_id = f"{rule.name}_{crawler_name}_{metric_name}_{int(timestamp.timestamp())}"
        
        title = f"{rule.level.value.upper()}: {rule.name}"
        message = f"Crawler '{crawler_name}' {rule.description or rule.name}: {value}"
        
        alert = Alert(
            id=alert_id,
            level=rule.level,
            title=title,
            message=message,
            crawler_name=crawler_name,
            timestamp=timestamp,
            rule_name=rule.name,
            data={
                'metric_name': metric_name,
                'value': value,
                'threshold': rule.threshold,
                'condition': rule.condition,
                'additional_data': additional_data
            }
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Keep only last 1000 alerts
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
        
        return alert
    
    async def _send_alert(self, alert: Alert) -> bool:
        """Send alert through configured channels"""
        sent = False
        
        for rule in self.rules.values():
            if rule.name == alert.rule_name:
                for channel in rule.channels:
                    try:
                        if channel == AlertChannel.EMAIL:
                            sent = sent or await self._send_email_alert(alert)
                        elif channel == AlertChannel.WEBHOOK:
                            sent = sent or await self._send_webhook_alert(alert)
                        elif channel == AlertChannel.SLACK:
                            sent = sent or await self._send_slack_alert(alert)
                        elif channel == AlertChannel.LOG:
                            sent = sent or await self._send_log_alert(alert)
                    except Exception as e:
                        self.logger.error(f"Error sending alert via {channel.value}: {e}")
        
        return sent
    
    async def _send_email_alert(self, alert: Alert) -> bool:
        """Send email alert"""
        try:
            if not self.email_config:
                return False
            
            msg = MIMEMultipart()
            msg['From'] = self.email_config['from_email']
            msg['To'] = self.email_config.get('to_emails', [])
            msg['Subject'] = f"[{alert.level.value.upper()}] {alert.title}"
            
            body = f"""
            Alert Details:
            Level: {alert.level.value.upper()}
            Title: {alert.title}
            Message: {alert.message}
            Crawler: {alert.crawler_name}
            Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
            
            Additional Data:
            {json.dumps(alert.data, indent=2, default=str)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config.get('smtp_port', 587))
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            
            text = msg.as_string()
            server.sendmail(self.email_config['from_email'], self.email_config['to_emails'], text)
            server.quit()
            
            self.logger.info(f"Email alert sent for {alert.id}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error sending email alert: {e}")
            return False
    
    async def _send_webhook_alert(self, alert: Alert) -> bool:
        """Send webhook alert"""
        try:
            if not self.webhook_config:
                return False
            
            import aiohttp
            
            payload = {
                'alert_id': alert.id,
                'level': alert.level.value,
                'title': alert.title,
                'message': alert.message,
                'crawler_name': alert.crawler_name,
                'timestamp': alert.timestamp.isoformat(),
                'data': alert.data
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_config['url'],
                    json=payload,
                    headers=self.webhook_config.get('headers', {})
                ) as response:
                    if response.status == 200:
                        self.logger.info(f"Webhook alert sent for {alert.id}")
                        return True
                    else:
                        self.logger.error(f"Webhook alert failed: {response.status}")
                        return False
        
        except Exception as e:
            self.logger.error(f"Error sending webhook alert: {e}")
            return False
    
    async def _send_slack_alert(self, alert: Alert) -> bool:
        """Send Slack alert"""
        try:
            if not self.slack_config:
                return False
            
            import aiohttp
            
            # Color based on alert level
            color_map = {
                AlertLevel.INFO: '#36a64f',
                AlertLevel.WARNING: '#ff9500',
                AlertLevel.ERROR: '#ff0000',
                AlertLevel.CRITICAL: '#800080'
            }
            
            payload = {
                'text': alert.title,
                'attachments': [{
                    'color': color_map.get(alert.level, '#36a64f'),
                    'fields': [
                        {'title': 'Message', 'value': alert.message, 'short': False},
                        {'title': 'Crawler', 'value': alert.crawler_name, 'short': True},
                        {'title': 'Time', 'value': alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'), 'short': True}
                    ]
                }]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.slack_config['webhook_url'],
                    json=payload
                ) as response:
                    if response.status == 200:
                        self.logger.info(f"Slack alert sent for {alert.id}")
                        return True
                    else:
                        self.logger.error(f"Slack alert failed: {response.status}")
                        return False
        
        except Exception as e:
            self.logger.error(f"Error sending Slack alert: {e}")
            return False
    
    async def _send_log_alert(self, alert: Alert) -> bool:
        """Send log alert"""
        try:
            log_level_map = {
                AlertLevel.INFO: logging.INFO,
                AlertLevel.WARNING: logging.WARNING,
                AlertLevel.ERROR: logging.ERROR,
                AlertLevel.CRITICAL: logging.CRITICAL
            }
            
            log_message = f"ALERT [{alert.level.value.upper()}] {alert.title}: {alert.message} (Crawler: {alert.crawler_name})"
            
            self.logger.log(log_level_map.get(alert.level, logging.INFO), log_message)
            return True
        
        except Exception as e:
            self.logger.error(f"Error sending log alert: {e}")
            return False
    
    def resolve_alert(self, alert_id: str):
        """Mark alert as resolved"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].resolved = True
            self.active_alerts[alert_id].resolved_at = datetime.now()
            self.logger.info(f"Alert resolved: {alert_id}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get list of active alerts"""
        return [alert for alert in self.active_alerts.values() if not alert.resolved]
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.timestamp >= cutoff_time]


class CrawlerMonitor:
    """Monitors crawler health and performance"""
    
    def __init__(self, manager: CrawlerManager, alert_manager: AlertManager):
        self.manager = manager
        self.alert_manager = alert_manager
        self.logger = logging.getLogger(__name__)
        
        # Health metrics storage
        self.metrics: List[HealthMetric] = []
        
        # Monitoring configuration
        self.monitor_config = {
            'check_interval': 60,  # seconds
            'metrics_retention_days': 30,
            'enable_performance_monitoring': True,
            'enable_error_monitoring': True,
            'enable_latency_monitoring': True
        }
        
        # Default alert rules
        self._setup_default_alert_rules()
        
        # Background monitoring task
        self._monitor_task: Optional[asyncio.Task] = None
    
    def _setup_default_alert_rules(self):
        """Setup default alert rules"""
        rules = [
            AlertRule(
                name="high_error_rate",
                condition=">",
                threshold=0.2,  # 20% error rate
                level=AlertLevel.ERROR,
                channels=[AlertChannel.LOG, AlertChannel.EMAIL],
                description="High error rate detected"
            ),
            AlertRule(
                name="slow_execution",
                condition=">",
                threshold=300,  # 5 minutes
                level=AlertLevel.WARNING,
                channels=[AlertChannel.LOG],
                description="Slow execution time detected"
            ),
            AlertRule(
                name="crawler_down",
                condition="==",
                threshold=0,  # crawler_status == 0 (down)
                level=AlertLevel.CRITICAL,
                channels=[AlertChannel.LOG, AlertChannel.EMAIL, AlertChannel.SLACK],
                description="Crawler is down or not responding"
            ),
            AlertRule(
                name="no_data_received",
                condition="<",
                threshold=1,  # No data points
                level=AlertLevel.WARNING,
                channels=[AlertChannel.LOG],
                description="No data received from crawler"
            ),
            AlertRule(
                name="data_quality_low",
                condition="<",
                threshold=0.8,  # 80% success rate
                level=AlertLevel.WARNING,
                channels=[AlertChannel.LOG],
                description="Low data quality detected"
            )
        ]
        
        for rule in rules:
            self.alert_manager.add_rule(rule)
    
    async def start_monitoring(self):
        """Start continuous monitoring"""
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Started crawler monitoring")
    
    async def stop_monitoring(self):
        """Stop monitoring"""
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Stopped crawler monitoring")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while True:
            try:
                await self.perform_health_checks()
                await asyncio.sleep(self.monitor_config['check_interval'])
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitor_config['check_interval'])
    
    async def perform_health_checks(self):
        """Perform comprehensive health checks"""
        try:
            # Get crawler status
            crawler_status = self.manager.get_status()
            
            for crawler_name in crawler_status['registered_crawlers']:
                await self._check_crawler_health(crawler_name, crawler_status)
            
            # Cleanup old metrics
            await self._cleanup_old_metrics()
        
        except Exception as e:
            self.logger.error(f"Error performing health checks: {e}")
    
    async def _check_crawler_health(self, crawler_name: str, status_data: Dict[str, Any]):
        """Check health of individual crawler"""
        try:
            # Get crawler-specific status
            crawler_status = status_data['crawler_status'].get(crawler_name, {})
            
            # Check if crawler is running
            is_running = crawler_name in status_data['running_crawlers']
            
            # Record uptime metric
            await self._record_metric(
                crawler_name=crawler_name,
                metric_name="is_running",
                value=1.0 if is_running else 0.0,
                unit="boolean"
            )
            
            # Check error rate
            run_count = crawler_status.get('run_count', 0)
            error_count = crawler_status.get('error_count', 0)
            
            error_rate = error_count / run_count if run_count > 0 else 1.0
            
            await self._record_metric(
                crawler_name=crawler_name,
                metric_name="error_rate",
                value=error_rate,
                unit="percentage"
            )
            
            # Check execution time
            last_execution_time = crawler_status.get('last_execution_time', 0)
            if last_execution_time:
                await self._record_metric(
                    crawler_name=crawler_name,
                    metric_name="execution_time",
                    value=last_execution_time,
                    unit="seconds"
                )
            
            # Check data freshness
            last_run = crawler_status.get('last_run')
            if last_run:
                last_run_time = datetime.fromisoformat(last_run)
                time_since_last_run = (datetime.now() - last_run_time).total_seconds()
                
                await self._record_metric(
                    crawler_name=crawler_name,
                    metric_name="data_freshness",
                    value=time_since_last_run,
                    unit="seconds"
                )
            
            # Check consecutive errors
            consecutive_errors = crawler_status.get('consecutive_errors', 0)
            await self._record_metric(
                crawler_name=crawler_name,
                metric_name="consecutive_errors",
                value=float(consecutive_errors),
                unit="count"
            )
            
            # Send alerts based on metrics
            await self._check_alerts(crawler_name, {
                'error_rate': error_rate,
                'execution_time': last_execution_time,
                'is_running': 1.0 if is_running else 0.0,
                'consecutive_errors': consecutive_errors
            })
        
        except Exception as e:
            self.logger.error(f"Error checking health for {crawler_name}: {e}")
    
    async def _record_metric(self, crawler_name: str, metric_name: str, 
                           value: float, unit: str = "", 
                           threshold_warning: Optional[float] = None,
                           threshold_critical: Optional[float] = None):
        """Record health metric"""
        metric = HealthMetric(
            crawler_name=crawler_name,
            metric_name=metric_name,
            value=value,
            timestamp=datetime.now(),
            unit=unit,
            threshold_warning=threshold_warning,
            threshold_critical=threshold_critical
        )
        
        self.metrics.append(metric)
        
        # Keep only recent metrics
        cutoff_time = datetime.now() - timedelta(days=self.monitor_config['metrics_retention_days'])
        self.metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]
    
    async def _check_alerts(self, crawler_name: str, metrics: Dict[str, float]):
        """Check metrics against alert rules"""
        timestamp = datetime.now()
        
        # Check error rate
        await self.alert_manager.check_and_send_alert(
            crawler_name=crawler_name,
            metric_name="error_rate",
            value=metrics.get('error_rate', 0),
            timestamp=timestamp,
            additional_data={'current_metrics': metrics}
        )
        
        # Check execution time
        execution_time = metrics.get('execution_time', 0)
        if execution_time > 0:
            await self.alert_manager.check_and_send_alert(
                crawler_name=crawler_name,
                metric_name="execution_time",
                value=execution_time,
                timestamp=timestamp,
                additional_data={'current_metrics': metrics}
            )
        
        # Check if crawler is down
        is_running = metrics.get('is_running', 0)
        await self.alert_manager.check_and_send_alert(
            crawler_name=crawler_name,
            metric_name="is_running",
            value=is_running,
            timestamp=timestamp,
            additional_data={'current_metrics': metrics}
        )
        
        # Check consecutive errors
        consecutive_errors = metrics.get('consecutive_errors', 0)
        if consecutive_errors > 3:  # Alert for repeated failures
            await self.alert_manager.check_and_send_alert(
                crawler_name=crawler_name,
                metric_name="consecutive_errors",
                value=float(consecutive_errors),
                timestamp=timestamp,
                additional_data={'current_metrics': metrics}
            )
    
    async def _cleanup_old_metrics(self):
        """Cleanup old metrics"""
        cutoff_time = datetime.now() - timedelta(days=self.monitor_config['metrics_retention_days'])
        self.metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]
    
    def get_health_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get health summary for specified time period"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]
            
            # Group by crawler
            crawler_metrics = {}
            for metric in recent_metrics:
                if metric.crawler_name not in crawler_metrics:
                    crawler_metrics[metric.crawler_name] = {}
                
                if metric.metric_name not in crawler_metrics[metric.crawler_name]:
                    crawler_metrics[metric.crawler_name][metric.metric_name] = []
                
                crawler_metrics[metric.crawler_name][metric.metric_name].append(metric)
            
            # Calculate summary statistics
            summary = {
                'timestamp': datetime.now().isoformat(),
                'period_hours': hours,
                'total_crawlers': len(crawler_metrics),
                'crawler_health': {}
            }
            
            for crawler_name, metrics in crawler_metrics.items():
                crawler_summary = {
                    'metrics_count': sum(len(metric_list) for metric_list in metrics.values()),
                    'last_metric_time': None,
                    'avg_metrics': {}
                }
                
                # Calculate averages and find latest timestamp
                all_timestamps = []
                for metric_name, metric_list in metrics.items():
                    if metric_list:
                        values = [m.value for m in metric_list]
                        crawler_summary['avg_metrics'][metric_name] = {
                            'average': sum(values) / len(values),
                            'min': min(values),
                            'max': max(values),
                            'count': len(values)
                        }
                        all_timestamps.extend([m.timestamp for m in metric_list])
                
                if all_timestamps:
                    crawler_summary['last_metric_time'] = max(all_timestamps).isoformat()
                
                summary['crawler_health'][crawler_name] = crawler_summary
            
            return summary
        
        except Exception as e:
            self.logger.error(f"Error generating health summary: {e}")
            return {'error': str(e)}
    
    def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Get detailed performance report"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]
            
            # Performance analysis
            performance_data = {
                'period_hours': hours,
                'timestamp': datetime.now().isoformat(),
                'execution_times': {},
                'error_rates': {},
                'availability': {}
            }
            
            # Group metrics by crawler and type
            for metric in recent_metrics:
                crawler_name = metric.crawler_name
                
                if metric.metric_name == "execution_time":
                    if crawler_name not in performance_data['execution_times']:
                        performance_data['execution_times'][crawler_name] = []
                    performance_data['execution_times'][crawler_name].append(metric.value)
                
                elif metric.metric_name == "error_rate":
                    if crawler_name not in performance_data['error_rates']:
                        performance_data['error_rates'][crawler_name] = []
                    performance_data['error_rates'][crawler_name].append(metric.value)
                
                elif metric.metric_name == "is_running":
                    if crawler_name not in performance_data['availability']:
                        performance_data['availability'][crawler_name] = []
                    performance_data['availability'][crawler_name].append(metric.value)
            
            # Calculate statistics
            for crawler_name in performance_data['execution_times']:
                times = performance_data['execution_times'][crawler_name]
                performance_data['execution_times'][crawler_name] = {
                    'average': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                    'total_runs': len(times)
                }
            
            for crawler_name in performance_data['error_rates']:
                rates = performance_data['error_rates'][crawler_name]
                performance_data['error_rates'][crawler_name] = {
                    'average': sum(rates) / len(rates),
                    'max': max(rates),
                    'min': min(rates),
                    'total_checks': len(rates)
                }
            
            for crawler_name in performance_data['availability']:
                availability_data = performance_data['availability'][crawler_name]
                uptime_percentage = (sum(availability_data) / len(availability_data)) * 100
                performance_data['availability'][crawler_name] = {
                    'uptime_percentage': uptime_percentage,
                    'total_checks': len(availability_data),
                    'up_checks': sum(availability_data),
                    'down_checks': len(availability_data) - sum(availability_data)
                }
            
            return performance_data
        
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
            return {'error': str(e)}