"""
@file monitoring.py
@brief Exit Strategy Monitoring and Alerting Framework

@details
This module provides monitoring and alerting capabilities for exit strategies,
enabling real-time tracking of strategy performance and automatic notifications
when certain conditions are met.

Key Features:
- Real-time strategy monitoring
- Performance alert generation
- Multi-channel notifications
- Alert filtering and prioritization
- Historical alert tracking

@author Trading Orchestrator System
@version 1.0
@date 2025-11-06

@note
Monitoring should be comprehensive to catch performance issues early and
ensure strategies are functioning as expected.

@see base_exit_strategy.py for exit strategy framework
"""

from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
from abc import ABC, abstractmethod

from loguru import logger

from .base_exit_strategy import (
    BaseExitStrategy,
    ExitReason,
    ExitType,
    ExitSignal,
    ExitMetrics,
    ExitStatus
)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert notification channels"""
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    SLACK = "slack"
    DISCORD = "discord"
    LOG = "log"
    DATABASE = "database"


@dataclass
class Alert:
    """Alert notification"""
    alert_id: str
    strategy_id: str
    alert_type: str
    severity: AlertSeverity
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    channels: List[AlertChannel] = field(default_factory=lambda: [AlertChannel.LOG])


class AlertRule:
    """Rule for triggering alerts"""
    
    def __init__(
        self,
        rule_id: str,
        name: str,
        condition_func: Callable,
        severity: AlertSeverity = AlertSeverity.WARNING,
        channels: List[AlertChannel] = None,
        cooldown_period: timedelta = timedelta(minutes=10),
        metadata: Dict[str, Any] = None
    ):
        self.rule_id = rule_id
        self.name = name
        self.condition_func = condition_func
        self.severity = severity
        self.channels = channels or [AlertChannel.LOG]
        self.cooldown_period = cooldown_period
        self.metadata = metadata or {}
        self.last_triggered = None
        self.trigger_count = 0


class AlertChannelHandler(ABC):
    """Abstract base class for alert channel handlers"""
    
    @abstractmethod
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert through this channel"""
        pass


class LogAlertHandler(AlertChannelHandler):
    """Alert handler that logs alerts"""
    
    async def send_alert(self, alert: Alert) -> bool:
        """Log alert message"""
        try:
            log_level = {
                AlertSeverity.INFO: "info",
                AlertSeverity.WARNING: "warning",
                AlertSeverity.ERROR: "error",
                AlertSeverity.CRITICAL: "critical"
            }.get(alert.severity, "info")
            
            logger.log(log_level, f"ALERT [{alert.severity.value}] {alert.message}")
            logger.info(f"Alert metadata: {alert.metadata}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error logging alert: {e}")
            return False


class EmailAlertHandler(AlertChannelHandler):
    """Alert handler for email notifications"""
    
    def __init__(self, smtp_config: Dict[str, Any]):
        self.smtp_config = smtp_config
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send email alert"""
        try:
            # Simplified email implementation
            # In real implementation, would use SMTP library
            logger.info(f"EMAIL ALERT: {alert.message}")
            logger.info(f"Would send email to: {self.smtp_config.get('recipients', [])}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
            return False


class WebhookAlertHandler(AlertChannelHandler):
    """Alert handler for webhook notifications"""
    
    def __init__(self, webhook_url: str, headers: Dict[str, str] = None):
        self.webhook_url = webhook_url
        self.headers = headers or {'Content-Type': 'application/json'}
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send webhook alert"""
        try:
            # Prepare alert payload
            payload = {
                'alert_id': alert.alert_id,
                'strategy_id': alert.strategy_id,
                'alert_type': alert.alert_type,
                'severity': alert.severity.value,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'metadata': alert.metadata
            }
            
            # In real implementation, would make HTTP request
            logger.info(f"WEBHOOK ALERT: {json.dumps(payload)}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending webhook alert: {e}")
            return False


class SlackAlertHandler(AlertChannelHandler):
    """Alert handler for Slack notifications"""
    
    def __init__(self, webhook_url: str, channel: str = None):
        self.webhook_url = webhook_url
        self.channel = channel
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send Slack alert"""
        try:
            # Prepare Slack message
            color = {
                AlertSeverity.INFO: "#36a64f",
                AlertSeverity.WARNING: "#ffb84d",
                AlertSeverity.ERROR: "#ff6b6b",
                AlertSeverity.CRITICAL: "#ff0000"
            }.get(alert.severity, "#36a64f")
            
            slack_payload = {
                "attachments": [
                    {
                        "color": color,
                        "title": f"Exit Strategy Alert: {alert.alert_type}",
                        "text": alert.message,
                        "fields": [
                            {"title": "Strategy", "value": alert.strategy_id, "short": True},
                            {"title": "Severity", "value": alert.severity.value, "short": True},
                            {"title": "Time", "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S"), "short": True}
                        ],
                        "footer": "Trading Orchestrator",
                        "ts": int(alert.timestamp.timestamp())
                    }
                ]
            }
            
            # In real implementation, would post to Slack
            logger.info(f"SLACK ALERT: {json.dumps(slack_payload)}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")
            return False


class ExitStrategyMonitor:
    """
    @class ExitStrategyMonitor
    @brief Real-time monitoring for exit strategies
    
    @details
    Monitors exit strategies in real-time, evaluates alert conditions,
    and sends notifications when specified criteria are met.
    """
    
    def __init__(self):
        self.strategies: Dict[str, BaseExitStrategy] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        self.alerts: List[Alert] = []
        self.alert_handlers: Dict[AlertChannel, AlertChannelHandler] = {}
        
        # Initialize default handlers
        self._initialize_default_handlers()
        
        # Initialize default alert rules
        self._initialize_default_alert_rules()
        
        self.is_monitoring = False
        self.monitor_task = None
        
        logger.info("Exit strategy monitor initialized")
    
    def _initialize_default_handlers(self):
        """Initialize default alert handlers"""
        self.alert_handlers[AlertChannel.LOG] = LogAlertHandler()
        # Add other handlers as needed
    
    def _initialize_default_alert_rules(self):
        """Initialize default alert rules"""
        # Strategy stopped unexpectedly
        self.add_alert_rule(AlertRule(
            rule_id="strategy_stopped",
            name="Strategy Stopped Unexpectedly",
            condition_func=self._check_strategy_stopped,
            severity=AlertSeverity.ERROR,
            channels=[AlertChannel.LOG, AlertChannel.SLACK]
        ))
        
        # High number of failed exits
        self.add_alert_rule(AlertRule(
            rule_id="high_failure_rate",
            name="High Exit Failure Rate",
            condition_func=self._check_high_failure_rate,
            severity=AlertSeverity.WARNING,
            cooldown_period=timedelta(minutes=30)
        ))
        
        # Strategy performance degradation
        self.add_alert_rule(AlertRule(
            rule_id="performance_degradation",
            name="Strategy Performance Degradation",
            condition_func=self._check_performance_degradation,
            severity=AlertSeverity.WARNING,
            cooldown_period=timedelta(hours=1)
        ))
        
        # Unusual exit frequency
        self.add_alert_rule(AlertRule(
            rule_id="unusual_exit_frequency",
            name="Unusual Exit Frequency",
            condition_func=self._check_unusual_exit_frequency,
            severity=AlertSeverity.INFO
        ))
    
    async def start_monitoring(self):
        """Start monitoring strategies"""
        if self.is_monitoring:
            logger.warning("Monitor is already running")
            return
        
        self.is_monitoring = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Exit strategy monitoring started")
    
    async def stop_monitoring(self):
        """Stop monitoring strategies"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Exit strategy monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        try:
            while self.is_monitoring:
                await self._check_all_strategies()
                await asyncio.sleep(30)  # Check every 30 seconds
                
        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
    
    async def _check_all_strategies(self):
        """Check all registered strategies"""
        for strategy_id, strategy in self.strategies.items():
            try:
                await self._check_strategy(strategy)
            except Exception as e:
                logger.error(f"Error checking strategy {strategy_id}: {e}")
    
    async def _check_strategy(self, strategy: BaseExitStrategy):
        """Check individual strategy for alerts"""
        try:
            strategy_id = strategy.config.strategy_id
            
            # Get current strategy status
            status = await strategy.get_status()
            
            # Check each alert rule
            for rule in self.alert_rules.values():
                await self._evaluate_alert_rule(rule, strategy, status)
                
        except Exception as e:
            logger.error(f"Error checking strategy {strategy}: {e}")
    
    async def _evaluate_alert_rule(
        self,
        rule: AlertRule,
        strategy: BaseExitStrategy,
        status: Dict[str, Any]
    ):
        """Evaluate alert rule for strategy"""
        try:
            # Check cooldown period
            if rule.last_triggered:
                time_since_last = datetime.utcnow() - rule.last_triggered
                if time_since_last < rule.cooldown_period:
                    return
            
            # Evaluate condition
            should_alert = await rule.condition_func(strategy, status)
            
            if should_alert:
                # Create alert
                alert = await self._create_alert(rule, strategy, status)
                
                # Send alert
                await self._send_alert(alert)
                
                # Update rule
                rule.last_triggered = datetime.utcnow()
                rule.trigger_count += 1
                
        except Exception as e:
            logger.error(f"Error evaluating alert rule {rule.rule_id}: {e}")
    
    async def _create_alert(
        self,
        rule: AlertRule,
        strategy: BaseExitStrategy,
        status: Dict[str, Any]
    ) -> Alert:
        """Create alert from rule evaluation"""
        alert_id = f"alert_{datetime.utcnow().timestamp()}"
        
        alert = Alert(
            alert_id=alert_id,
            strategy_id=strategy.config.strategy_id,
            alert_type=rule.name,
            severity=rule.severity,
            message=rule.name,
            metadata={
                'rule_id': rule.rule_id,
                'strategy_status': status,
                'rule_metadata': rule.metadata
            },
            channels=rule.channels.copy()
        )
        
        return alert
    
    async def _send_alert(self, alert: Alert):
        """Send alert through configured channels"""
        try:
            success_count = 0
            
            for channel in alert.channels:
                handler = self.alert_handlers.get(channel)
                if handler:
                    success = await handler.send_alert(alert)
                    if success:
                        success_count += 1
                else:
                    logger.warning(f"No handler found for channel {channel}")
            
            # Store alert
            self.alerts.append(alert)
            
            # Keep only recent alerts (last 1000)
            if len(self.alerts) > 1000:
                self.alerts = self.alerts[-500:]
            
            logger.info(f"Alert sent through {success_count}/{len(alert.channels)} channels")
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
    
    def register_strategy(self, strategy: BaseExitStrategy):
        """Register strategy for monitoring"""
        strategy_id = strategy.config.strategy_id
        self.strategies[strategy_id] = strategy
        logger.info(f"Registered strategy for monitoring: {strategy_id}")
    
    def unregister_strategy(self, strategy_id: str):
        """Unregister strategy from monitoring"""
        if strategy_id in self.strategies:
            del self.strategies[strategy_id]
            logger.info(f"Unregistered strategy from monitoring: {strategy_id}")
    
    def add_alert_rule(self, rule: AlertRule):
        """Add alert rule"""
        self.alert_rules[rule.rule_id] = rule
        logger.info(f"Added alert rule: {rule.rule_id}")
    
    def remove_alert_rule(self, rule_id: str):
        """Remove alert rule"""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            logger.info(f"Removed alert rule: {rule_id}")
    
    def add_alert_handler(self, channel: AlertChannel, handler: AlertChannelHandler):
        """Add alert handler for channel"""
        self.alert_handlers[channel] = handler
        logger.info(f"Added alert handler for channel: {channel.value}")
    
    # Alert condition functions
    async def _check_strategy_stopped(
        self,
        strategy: BaseExitStrategy,
        status: Dict[str, Any]
    ) -> bool:
        """Check if strategy stopped unexpectedly"""
        try:
            strategy_status = status.get('status', '')
            expected_statuses = ['active', 'running']
            
            return strategy_status not in expected_statuses
            
        except Exception as e:
            logger.error(f"Error checking strategy stopped: {e}")
            return False
    
    async def _check_high_failure_rate(
        self,
        strategy: BaseExitStrategy,
        status: Dict[str, Any]
    ) -> bool:
        """Check for high exit failure rate"""
        try:
            metrics = status.get('metrics', {})
            total_exits = metrics.get('total_exits', 0)
            failed_exits = metrics.get('failed_exits', 0)
            
            if total_exits < 10:  # Need minimum sample size
                return False
            
            failure_rate = failed_exits / total_exits
            return failure_rate > 0.2  # Alert if more than 20% failures
            
        except Exception as e:
            logger.error(f"Error checking failure rate: {e}")
            return False
    
    async def _check_performance_degradation(
        self,
        strategy: BaseExitStrategy,
        status: Dict[str, Any]
    ) -> bool:
        """Check for strategy performance degradation"""
        try:
            metrics = status.get('metrics', {})
            win_rate = metrics.get('win_rate', 0.0)
            
            # Alert if win rate drops significantly
            return win_rate < 0.4  # Less than 40% win rate
            
        except Exception as e:
            logger.error(f"Error checking performance degradation: {e}")
            return False
    
    async def _check_unusual_exit_frequency(
        self,
        strategy: BaseExitStrategy,
        status: Dict[str, Any]
    ) -> bool:
        """Check for unusual exit frequency"""
        try:
            metrics = status.get('metrics', {})
            total_exits = metrics.get('total_exits', 0)
            runtime = status.get('runtime_seconds', 0)
            
            if runtime < 3600:  # Less than 1 hour running
                return False
            
            # Calculate exits per hour
            exits_per_hour = total_exits / (runtime / 3600)
            
            # Alert if very high frequency (more than 10 exits per hour)
            return exits_per_hour > 10
            
        except Exception as e:
            logger.error(f"Error checking exit frequency: {e}")
            return False
    
    # Alert management functions
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system"):
        """Acknowledge an alert"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_at = datetime.utcnow()
                alert.acknowledged_by = acknowledged_by
                logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
                break
    
    async def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.resolved_at = datetime.utcnow()
                logger.info(f"Alert {alert_id} resolved")
                break
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts"""
        return [alert for alert in self.alerts if not alert.resolved]
    
    def get_strategy_alerts(
        self,
        strategy_id: str,
        severity: Optional[AlertSeverity] = None,
        resolved: Optional[bool] = None,
        limit: int = 100
    ) -> List[Alert]:
        """Get alerts for specific strategy"""
        alerts = [alert for alert in self.alerts if alert.strategy_id == strategy_id]
        
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        if resolved is not None:
            alerts = [alert for alert in alerts if alert.resolved == resolved]
        
        return alerts[-limit:]  # Return most recent
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get monitoring summary"""
        total_strategies = len(self.strategies)
        active_alerts = len(self.get_active_alerts())
        total_alerts = len(self.alerts)
        
        # Count alerts by severity
        severity_counts = {}
        for severity in AlertSeverity:
            severity_counts[severity.value] = len([
                alert for alert in self.alerts 
                if alert.severity == severity
            ])
        
        # Count strategies by status
        strategy_statuses = {}
        for strategy_id, strategy in self.strategies.items():
            status = strategy.status.value if hasattr(strategy, 'status') else 'unknown'
            strategy_statuses[status] = strategy_statuses.get(status, 0) + 1
        
        return {
            'total_strategies': total_strategies,
            'active_alerts': active_alerts,
            'total_alerts': total_alerts,
            'severity_distribution': severity_counts,
            'strategy_statuses': strategy_statuses,
            'monitoring_active': self.is_monitoring
        }


class ExitStrategyAlertManager:
    """
    @class ExitStrategyAlertManager
    @brief Centralized alert management for exit strategies
    
    @details
    Provides centralized management of alerts, rules, and notifications
    across multiple exit strategies.
    """
    
    def __init__(self):
        self.monitors: Dict[str, ExitStrategyMonitor] = {}
        self.global_alert_rules: Dict[str, AlertRule] = {}
        
        logger.info("Alert manager initialized")
    
    def create_monitor(self, monitor_id: str) -> ExitStrategyMonitor:
        """Create new monitoring instance"""
        monitor = ExitStrategyMonitor()
        self.monitors[monitor_id] = monitor
        logger.info(f"Created monitor: {monitor_id}")
        return monitor
    
    def get_monitor(self, monitor_id: str) -> Optional[ExitStrategyMonitor]:
        """Get monitor by ID"""
        return self.monitors.get(monitor_id)
    
    def add_global_alert_rule(self, rule: AlertRule):
        """Add global alert rule to all monitors"""
        for monitor in self.monitors.values():
            monitor.add_alert_rule(rule)
        
        self.global_alert_rules[rule.rule_id] = rule
        logger.info(f"Added global alert rule: {rule.rule_id}")
    
    def get_all_active_alerts(self) -> List[Alert]:
        """Get all active alerts from all monitors"""
        all_alerts = []
        for monitor in self.monitors.values():
            all_alerts.extend(monitor.get_active_alerts())
        return sorted(all_alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_global_monitoring_summary(self) -> Dict[str, Any]:
        """Get global monitoring summary"""
        summary = {
            'total_monitors': len(self.monitors),
            'total_strategies': sum(len(monitor.strategies) for monitor in self.monitors.values()),
            'total_active_alerts': len(self.get_all_active_alerts()),
            'monitor_summaries': {}
        }
        
        for monitor_id, monitor in self.monitors.items():
            summary['monitor_summaries'][monitor_id] = monitor.get_monitoring_summary()
        
        return summary


# Global alert manager instance
alert_manager = ExitStrategyAlertManager()

# Convenience functions

def create_basic_monitor() -> ExitStrategyMonitor:
    """Create basic monitoring setup"""
    return alert_manager.create_monitor("basic_monitor")


def create_email_alert_handler(
    smtp_server: str,
    smtp_port: int,
    username: str,
    password: str,
    recipients: List[str]
) -> EmailAlertHandler:
    """Create email alert handler"""
    smtp_config = {
        'server': smtp_server,
        'port': smtp_port,
        'username': username,
        'password': password,
        'recipients': recipients
    }
    return EmailAlertHandler(smtp_config)


def create_slack_alert_handler(
    webhook_url: str,
    channel: str = None
) -> SlackAlertHandler:
    """Create Slack alert handler"""
    return SlackAlertHandler(webhook_url, channel)


def create_webhook_alert_handler(
    webhook_url: str,
    headers: Dict[str, str] = None
) -> WebhookAlertHandler:
    """Create webhook alert handler"""
    return WebhookAlertHandler(webhook_url, headers)


def add_performance_alert_rule(
    monitor: ExitStrategyMonitor,
    strategy_id: str,
    min_win_rate: float = 0.4,
    max_drawdown: float = 0.1
):
    """Add performance-based alert rule"""
    
    async def performance_condition(strategy: BaseExitStrategy, status: Dict[str, Any]) -> bool:
        metrics = status.get('metrics', {})
        win_rate = metrics.get('win_rate', 1.0)
        max_dd = metrics.get('max_drawdown', 0.0)
        
        return win_rate < min_win_rate or abs(max_dd) > max_drawdown
    
    rule = AlertRule(
        rule_id=f"performance_{strategy_id}",
        name=f"Performance Alert for {strategy_id}",
        condition_func=performance_condition,
        severity=AlertSeverity.WARNING,
        channels=[AlertChannel.LOG, AlertChannel.SLACK],
        cooldown_period=timedelta(hours=2)
    )
    
    monitor.add_alert_rule(rule)


def add_exit_frequency_alert_rule(
    monitor: ExitStrategyMonitor,
    strategy_id: str,
    max_exits_per_hour: float = 10.0
):
    """Add exit frequency alert rule"""
    
    async def frequency_condition(strategy: BaseExitStrategy, status: Dict[str, Any]) -> bool:
        metrics = status.get('metrics', {})
        total_exits = metrics.get('total_exits', 0)
        runtime = status.get('runtime_seconds', 0)
        
        if runtime < 3600:  # Less than 1 hour
            return False
        
        exits_per_hour = total_exits / (runtime / 3600)
        return exits_per_hour > max_exits_per_hour
    
    rule = AlertRule(
        rule_id=f"frequency_{strategy_id}",
        name=f"High Exit Frequency for {strategy_id}",
        condition_func=frequency_condition,
        severity=AlertSeverity.INFO,
        channels=[AlertChannel.LOG],
        cooldown_period=timedelta(minutes=30)
    )
    
    monitor.add_alert_rule(rule)
