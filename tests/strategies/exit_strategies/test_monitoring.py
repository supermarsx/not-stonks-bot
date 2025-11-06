"""
@file test_monitoring.py
@brief Comprehensive unit tests for exit strategy monitoring module

@details
This module provides comprehensive unit tests for the ExitStrategyMonitor
and related monitoring functionality. Tests cover alert management, channel handlers,
monitoring loops, and alert rule evaluation.

@author Trading Orchestrator System
@version 1.0
@date 2025-11-06

@see monitoring.py for implementation details
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from decimal import Decimal

from trading_orchestrator.strategies.exit_strategies.monitoring import (
    ExitStrategyMonitor,
    ExitStrategyAlertManager,
    AlertManager,
    AlertSeverity,
    AlertChannel,
    Alert,
    AlertRule,
    LogAlertHandler,
    EmailAlertHandler,
    WebhookAlertHandler,
    SlackAlertHandler,
    create_basic_monitor,
    create_email_alert_handler,
    create_slack_alert_handler,
    create_webhook_alert_handler,
    add_performance_alert_rule,
    add_exit_frequency_alert_rule
)
from trading_orchestrator.strategies.exit_strategies.base_exit_strategy import (
    BaseExitStrategy,
    ExitConfiguration,
    ExitType,
    ExitStatus,
    ExitMetrics
)


class TestAlertDataClasses:
    """Test alert-related data classes"""
    
    def test_alert_creation(self):
        """Test Alert dataclass creation"""
        alert = Alert(
            alert_id="alert_001",
            strategy_id="test_strategy",
            alert_type="Performance Degradation",
            severity=AlertSeverity.WARNING,
            message="Strategy performance below threshold",
            channels=[AlertChannel.LOG, AlertChannel.SLACK]
        )
        
        assert alert.alert_id == "alert_001"
        assert alert.strategy_id == "test_strategy"
        assert alert.severity == AlertSeverity.WARNING
        assert AlertChannel.LOG in alert.channels
        assert AlertChannel.SLACK in alert.channels
        assert isinstance(alert.timestamp, datetime)
        assert alert.acknowledged == False
        assert alert.resolved == False
    
    def test_alert_rule_creation(self):
        """Test AlertRule class creation"""
        async def test_condition(strategy, status):
            return status.get("win_rate", 1.0) < 0.5
        
        rule = AlertRule(
            rule_id="performance_rule",
            name="Low Win Rate Alert",
            condition_func=test_condition,
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.LOG, AlertChannel.SLACK],
            cooldown_period=timedelta(minutes=30)
        )
        
        assert rule.rule_id == "performance_rule"
        assert rule.name == "Low Win Rate Alert"
        assert rule.severity == AlertSeverity.WARNING
        assert rule.cooldown_period == timedelta(minutes=30)
        assert rule.last_triggered is None
        assert rule.trigger_count == 0
    
    def test_alert_enums(self):
        """Test alert enums"""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.ERROR.value == "error"
        assert AlertSeverity.CRITICAL.value == "critical"
        
        assert AlertChannel.EMAIL.value == "email"
        assert AlertChannel.SMS.value == "sms"
        assert AlertChannel.WEBHOOK.value == "webhook"
        assert AlertChannel.SLACK.value == "slack"
        assert AlertChannel.LOG.value == "log"


class TestAlertChannelHandlers:
    """Test alert channel handlers"""
    
    @pytest.mark.asyncio
    async def test_log_alert_handler(self):
        """Test LogAlertHandler"""
        handler = LogAlertHandler()
        
        alert = Alert(
            alert_id="test_alert",
            strategy_id="test_strategy",
            alert_type="Test Alert",
            severity=AlertSeverity.WARNING,
            message="This is a test alert",
            channels=[AlertChannel.LOG]
        )
        
        result = await handler.send_alert(alert)
        assert result == True
    
    @pytest.mark.asyncio
    async def test_email_alert_handler(self):
        """Test EmailAlertHandler"""
        smtp_config = {
            'server': 'smtp.test.com',
            'port': 587,
            'username': 'test@example.com',
            'password': 'password123',
            'recipients': ['admin@example.com', 'trader@example.com']
        }
        
        handler = EmailAlertHandler(smtp_config)
        
        alert = Alert(
            alert_id="email_test_alert",
            strategy_id="test_strategy",
            alert_type="Email Test",
            severity=AlertSeverity.ERROR,
            message="Email alert test message",
            channels=[AlertChannel.EMAIL]
        )
        
        result = await handler.send_alert(alert)
        assert result == True
    
    @pytest.mark.asyncio
    async def test_webhook_alert_handler(self):
        """Test WebhookAlertHandler"""
        webhook_url = "https://hooks.slack.com/services/test"
        headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer token123'}
        
        handler = WebhookAlertHandler(webhook_url, headers)
        
        alert = Alert(
            alert_id="webhook_test",
            strategy_id="webhook_strategy",
            alert_type="Webhook Test",
            severity=AlertSeverity.INFO,
            message="Webhook alert test",
            channels=[AlertChannel.WEBHOOK]
        )
        
        result = await handler.send_alert(alert)
        assert result == True
        
        # Verify payload structure
        # In real implementation, would make actual HTTP request
        assert alert.metadata is not None
    
    @pytest.mark.asyncio
    async def test_slack_alert_handler(self):
        """Test SlackAlertHandler"""
        webhook_url = "https://hooks.slack.com/services/test"
        channel = "#trading-alerts"
        
        handler = SlackAlertHandler(webhook_url, channel)
        
        alert = Alert(
            alert_id="slack_test",
            strategy_id="slack_strategy",
            alert_type="Slack Test",
            severity=AlertSeverity.CRITICAL,
            message="Critical alert message",
            channels=[AlertChannel.SLACK]
        )
        
        result = await handler.send_alert(alert)
        assert result == True
        
        # Verify alert metadata is preserved
        assert alert.metadata is not None


class TestExitStrategyMonitor:
    """Test ExitStrategyMonitor class"""
    
    @pytest.fixture
    def monitor(self):
        """Create monitor instance"""
        return ExitStrategyMonitor()
    
    @pytest.fixture
    def mock_strategy(self):
        """Create mock strategy"""
        strategy = Mock(spec=BaseExitStrategy)
        strategy.config = ExitConfiguration(
            strategy_id="test_strategy",
            strategy_type=ExitType.TRAILING_STOP,
            symbol="AAPL",
            parameters={}
        )
        strategy.status = ExitStatus.ACTIVE
        return strategy
    
    def test_monitor_initialization(self, monitor):
        """Test monitor initialization"""
        assert monitor.is_monitoring == False
        assert monitor.monitor_task is None
        assert len(monitor.strategies) == 0
        assert len(monitor.alert_rules) > 0  # Default rules initialized
        assert len(monitor.alert_handlers) > 0  # Default handlers initialized
        assert len(monitor.alerts) == 0
    
    def test_register_strategy(self, monitor, mock_strategy):
        """Test strategy registration"""
        monitor.register_strategy(mock_strategy)
        
        assert "test_strategy" in monitor.strategies
        assert monitor.strategies["test_strategy"] == mock_strategy
    
    def test_unregister_strategy(self, monitor, mock_strategy):
        """Test strategy unregistration"""
        monitor.register_strategy(mock_strategy)
        assert "test_strategy" in monitor.strategies
        
        monitor.unregister_strategy("test_strategy")
        assert "test_strategy" not in monitor.strategies
    
    def test_add_alert_rule(self, monitor):
        """Test alert rule addition"""
        async def test_condition(strategy, status):
            return True
        
        rule = AlertRule(
            rule_id="test_rule",
            name="Test Rule",
            condition_func=test_condition
        )
        
        monitor.add_alert_rule(rule)
        
        assert "test_rule" in monitor.alert_rules
        assert monitor.alert_rules["test_rule"] == rule
    
    def test_remove_alert_rule(self, monitor):
        """Test alert rule removal"""
        async def test_condition(strategy, status):
            return True
        
        rule = AlertRule(
            rule_id="test_rule",
            name="Test Rule",
            condition_func=test_condition
        )
        
        monitor.add_alert_rule(rule)
        assert "test_rule" in monitor.alert_rules
        
        monitor.remove_alert_rule("test_rule")
        assert "test_rule" not in monitor.alert_rules
    
    def test_add_alert_handler(self, monitor):
        """Test alert handler addition"""
        handler = LogAlertHandler()
        
        monitor.add_alert_handler(AlertChannel.LOG, handler)
        assert AlertChannel.LOG in monitor.alert_handlers
        assert monitor.alert_handlers[AlertChannel.LOG] == handler
    
    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, monitor):
        """Test starting and stopping monitoring"""
        assert monitor.is_monitoring == False
        
        # Start monitoring
        await monitor.start_monitoring()
        assert monitor.is_monitoring == True
        assert monitor.monitor_task is not None
        
        # Stop monitoring
        await monitor.stop_monitoring()
        assert monitor.is_monitoring == False
        assert monitor.monitor_task is None
    
    @pytest.mark.asyncio
    async def test_monitoring_loop_exception_handling(self, monitor):
        """Test monitoring loop exception handling"""
        # Start monitoring
        await monitor.start_monitoring()
        
        # Let it run briefly
        await asyncio.sleep(0.1)
        
        # Stop monitoring
        await monitor.stop_monitoring()
        
        # Should not crash even with no strategies
        assert monitor.is_monitoring == False
    
    @pytest.mark.asyncio
    async def test_check_all_strategies(self, monitor, mock_strategy):
        """Test checking all strategies"""
        monitor.register_strategy(mock_strategy)
        
        # Mock the get_status method
        mock_strategy.get_status = AsyncMock(return_value={
            'status': 'active',
            'metrics': {'total_exits': 10, 'failed_exits': 1}
        })
        
        # Mock the _check_strategy method to avoid actual rule evaluation
        with patch.object(monitor, '_check_strategy', new_callable=AsyncMock):
            await monitor._check_all_strategies()
    
    @pytest.mark.asyncio
    async def test_check_strategy(self, monitor, mock_strategy):
        """Test individual strategy checking"""
        monitor.register_strategy(mock_strategy)
        
        # Mock strategy status
        mock_strategy.get_status = AsyncMock(return_value={
            'status': 'active',
            'metrics': {'total_exits': 10, 'failed_exits': 1}
        })
        
        # Mock rule evaluation
        with patch.object(monitor, '_evaluate_alert_rule', new_callable=AsyncMock):
            await monitor._check_strategy(mock_strategy)
    
    @pytest.mark.asyncio
    async def test_evaluate_alert_rule_with_cooldown(self, monitor, mock_strategy):
        """Test alert rule evaluation with cooldown"""
        # Create a rule with short cooldown
        async def test_condition(strategy, status):
            return True
        
        rule = AlertRule(
            rule_id="cooldown_rule",
            name="Cooldown Test Rule",
            condition_func=test_condition,
            cooldown_period=timedelta(seconds=0.1)  # Very short cooldown
        )
        
        monitor.add_alert_rule(rule)
        
        # Mock strategy status
        status = {'status': 'active', 'metrics': {}}
        
        # First evaluation should trigger
        with patch.object(monitor, '_create_alert', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = Alert(
                alert_id="test_alert",
                strategy_id="test_strategy",
                alert_type="Test",
                severity=AlertSeverity.INFO,
                message="Test"
            )
            
            await monitor._evaluate_alert_rule(rule, mock_strategy, status)
            
            # Should have been called once
            mock_create.assert_called_once()
        
        # Second evaluation should be within cooldown
        await monitor._evaluate_alert_rule(rule, mock_strategy, status)
        
        # create_alert should still only be called once (cooldown prevents second)
        assert rule.last_triggered is not None
        assert rule.trigger_count == 1
    
    @pytest.mark.asyncio
    async def test_create_alert(self, monitor, mock_strategy):
        """Test alert creation"""
        async def test_condition(strategy, status):
            return True
        
        rule = AlertRule(
            rule_id="test_rule",
            name="Test Rule",
            condition_func=test_condition,
            severity=AlertSeverity.WARNING
        )
        
        status = {'status': 'active', 'metrics': {}}
        
        alert = await monitor._create_alert(rule, mock_strategy, status)
        
        assert alert.strategy_id == "test_strategy"
        assert alert.alert_type == "Test Rule"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.metadata is not None
        assert 'rule_id' in alert.metadata
        assert 'strategy_status' in alert.metadata
        assert alert.metadata['rule_id'] == "test_rule"
    
    @pytest.mark.asyncio
    async def test_send_alert(self, monitor):
        """Test alert sending"""
        alert = Alert(
            alert_id="test_alert",
            strategy_id="test_strategy",
            alert_type="Test Alert",
            severity=AlertSeverity.INFO,
            message="Test message",
            channels=[AlertChannel.LOG]
        )
        
        initial_alert_count = len(monitor.alerts)
        
        await monitor._send_alert(alert)
        
        assert len(monitor.alerts) == initial_alert_count + 1
        assert monitor.alerts[-1] == alert
    
    @pytest.mark.asyncio
    async def test_alert_condition_functions(self, monitor, mock_strategy):
        """Test alert condition functions"""
        # Test strategy stopped condition
        status_stopped = {'status': 'inactive', 'metrics': {}}
        result = await monitor._check_strategy_stopped(mock_strategy, status_stopped)
        assert result == True
        
        status_active = {'status': 'active', 'metrics': {}}
        result = await monitor._check_strategy_stopped(mock_strategy, status_active)
        assert result == False
        
        # Test high failure rate
        status_high_failures = {
            'status': 'active',
            'metrics': {'total_exits': 10, 'failed_exits': 3}  # 30% failure rate
        }
        result = await monitor._check_high_failure_rate(mock_strategy, status_high_failures)
        assert result == True  # > 20% threshold
        
        status_low_failures = {
            'status': 'active',
            'metrics': {'total_exits': 10, 'failed_exits': 1}  # 10% failure rate
        }
        result = await monitor._check_high_failure_rate(mock_strategy, status_low_failures)
        assert result == False  # < 20% threshold
        
        # Test performance degradation
        status_poor_performance = {
            'status': 'active',
            'metrics': {'win_rate': 0.3}  # 30% win rate
        }
        result = await monitor._check_performance_degradation(mock_strategy, status_poor_performance)
        assert result == True  # < 40% threshold
        
        status_good_performance = {
            'status': 'active',
            'metrics': {'win_rate': 0.6}  # 60% win rate
        }
        result = await monitor._check_performance_degradation(mock_strategy, status_good_performance)
        assert result == False  # > 40% threshold
        
        # Test unusual exit frequency
        status_high_frequency = {
            'status': 'active',
            'metrics': {'total_exits': 50},
            'runtime_seconds': 3600  # 1 hour
        }
        result = await monitor._check_unusual_exit_frequency(mock_strategy, status_high_frequency)
        assert result == True  # > 10 exits per hour
        
        status_normal_frequency = {
            'status': 'active',
            'metrics': {'total_exits': 5},
            'runtime_seconds': 3600  # 1 hour
        }
        result = await monitor._check_unusual_exit_frequency(mock_strategy, status_normal_frequency)
        assert result == False  # < 10 exits per hour
    
    @pytest.mark.asyncio
    async def test_acknowledge_alert(self, monitor):
        """Test alert acknowledgment"""
        alert = Alert(
            alert_id="test_alert",
            strategy_id="test_strategy",
            alert_type="Test Alert",
            severity=AlertSeverity.INFO,
            message="Test message"
        )
        
        monitor.alerts.append(alert)
        
        await monitor.acknowledge_alert("test_alert", "test_user")
        
        assert alert.acknowledged == True
        assert alert.acknowledged_by == "test_user"
        assert alert.acknowledged_at is not None
    
    @pytest.mark.asyncio
    async def test_resolve_alert(self, monitor):
        """Test alert resolution"""
        alert = Alert(
            alert_id="test_alert",
            strategy_id="test_strategy",
            alert_type="Test Alert",
            severity=AlertSeverity.INFO,
            message="Test message"
        )
        
        monitor.alerts.append(alert)
        
        await monitor.resolve_alert("test_alert")
        
        assert alert.resolved == True
        assert alert.resolved_at is not None
    
    def test_get_active_alerts(self, monitor):
        """Test getting active alerts"""
        # Add resolved and unresolved alerts
        alert1 = Alert(
            alert_id="alert1",
            strategy_id="test_strategy",
            alert_type="Alert 1",
            severity=AlertSeverity.INFO,
            message="Alert 1"
        )
        
        alert2 = Alert(
            alert_id="alert2",
            strategy_id="test_strategy",
            alert_type="Alert 2",
            severity=AlertSeverity.WARNING,
            message="Alert 2"
        )
        alert2.resolved = True
        
        monitor.alerts.extend([alert1, alert2])
        
        active_alerts = monitor.get_active_alerts()
        
        assert len(active_alerts) == 1
        assert active_alerts[0].alert_id == "alert1"
        assert active_alerts[0].resolved == False
    
    def test_get_strategy_alerts(self, monitor):
        """Test getting strategy-specific alerts"""
        # Add alerts for different strategies
        alert1 = Alert(
            alert_id="alert1",
            strategy_id="strategy_1",
            alert_type="Alert 1",
            severity=AlertSeverity.INFO,
            message="Alert 1"
        )
        
        alert2 = Alert(
            alert_id="alert2",
            strategy_id="strategy_2",
            alert_type="Alert 2",
            severity=AlertSeverity.WARNING,
            message="Alert 2"
        )
        
        alert3 = Alert(
            alert_id="alert3",
            strategy_id="strategy_1",
            alert_type="Alert 3",
            severity=AlertSeverity.ERROR,
            message="Alert 3"
        )
        
        monitor.alerts.extend([alert1, alert2, alert3])
        
        # Get alerts for strategy_1
        strategy1_alerts = monitor.get_strategy_alerts("strategy_1")
        
        assert len(strategy1_alerts) == 2
        assert all(alert.strategy_id == "strategy_1" for alert in strategy1_alerts)
        
        # Get ERROR severity alerts
        error_alerts = monitor.get_strategy_alerts("strategy_1", severity=AlertSeverity.ERROR)
        
        assert len(error_alerts) == 1
        assert error_alerts[0].alert_id == "alert3"
    
    def test_get_monitoring_summary(self, monitor):
        """Test monitoring summary"""
        # Add some test data
        strategy1 = Mock()
        strategy1.config.strategy_id = "strategy_1"
        strategy1.status = ExitStatus.ACTIVE
        
        strategy2 = Mock()
        strategy2.config.strategy_id = "strategy_2"
        strategy2.status = ExitStatus.STOPPED
        
        monitor.register_strategy(strategy1)
        monitor.register_strategy(strategy2)
        
        # Add some alerts
        alert1 = Alert(
            alert_id="alert1",
            strategy_id="strategy_1",
            alert_type="Test Alert",
            severity=AlertSeverity.INFO,
            message="Info alert"
        )
        
        alert2 = Alert(
            alert_id="alert2",
            strategy_id="strategy_2",
            alert_type="Warning Alert",
            severity=AlertSeverity.WARNING,
            message="Warning alert"
        )
        
        monitor.alerts.extend([alert1, alert2])
        
        summary = monitor.get_monitoring_summary()
        
        assert summary['total_strategies'] == 2
        assert summary['active_alerts'] == 2
        assert summary['total_alerts'] == 2
        assert 'severity_distribution' in summary
        assert 'strategy_statuses' in summary
        assert summary['monitoring_active'] == False
        
        assert summary['severity_distribution']['info'] == 1
        assert summary['severity_distribution']['warning'] == 1


class TestExitStrategyAlertManager:
    """Test ExitStrategyAlertManager class"""
    
    @pytest.fixture
    def alert_manager(self):
        """Create alert manager instance"""
        return ExitStrategyAlertManager()
    
    def test_alert_manager_initialization(self, alert_manager):
        """Test alert manager initialization"""
        assert len(alert_manager.monitors) == 0
        assert len(alert_manager.global_alert_rules) == 0
    
    def test_create_monitor(self, alert_manager):
        """Test monitor creation"""
        monitor = alert_manager.create_monitor("test_monitor")
        
        assert "test_monitor" in alert_manager.monitors
        assert alert_manager.monitors["test_monitor"] == monitor
        assert isinstance(monitor, ExitStrategyMonitor)
    
    def test_get_monitor(self, alert_manager):
        """Test getting monitor"""
        monitor = alert_manager.create_monitor("test_monitor")
        
        retrieved = alert_manager.get_monitor("test_monitor")
        assert retrieved == monitor
        
        # Test non-existent monitor
        non_existent = alert_manager.get_monitor("non_existent")
        assert non_existent is None
    
    def test_add_global_alert_rule(self, alert_manager):
        """Test adding global alert rule"""
        async def test_condition(strategy, status):
            return status.get("win_rate", 1.0) < 0.5
        
        rule = AlertRule(
            rule_id="global_rule",
            name="Global Performance Rule",
            condition_func=test_condition
        )
        
        # Create two monitors
        monitor1 = alert_manager.create_monitor("monitor1")
        monitor2 = alert_manager.create_monitor("monitor2")
        
        alert_manager.add_global_alert_rule(rule)
        
        # Rule should be added to all monitors
        assert "global_rule" in monitor1.alert_rules
        assert "global_rule" in monitor2.alert_rules
        assert "global_rule" in alert_manager.global_alert_rules
    
    def test_get_all_active_alerts(self, alert_manager):
        """Test getting all active alerts"""
        # Create monitors and add alerts
        monitor1 = alert_manager.create_monitor("monitor1")
        monitor2 = alert_manager.create_monitor("monitor2")
        
        alert1 = Alert(
            alert_id="alert1",
            strategy_id="strategy1",
            alert_type="Alert 1",
            severity=AlertSeverity.INFO,
            message="Alert 1"
        )
        
        alert2 = Alert(
            alert_id="alert2",
            strategy_id="strategy2",
            alert_type="Alert 2",
            severity=AlertSeverity.WARNING,
            message="Alert 2"
        )
        alert2.resolved = True  # Resolved alert
        
        monitor1.alerts.append(alert1)
        monitor2.alerts.append(alert2)
        
        all_alerts = alert_manager.get_all_active_alerts()
        
        # Should only include active (unresolved) alerts
        assert len(all_alerts) == 1
        assert all_alerts[0].alert_id == "alert1"
    
    def test_get_global_monitoring_summary(self, alert_manager):
        """Test global monitoring summary"""
        # Create monitors with strategies
        monitor1 = alert_manager.create_monitor("monitor1")
        monitor2 = alert_manager.create_monitor("monitor2")
        
        strategy1 = Mock()
        strategy1.config.strategy_id = "strategy1"
        monitor1.register_strategy(strategy1)
        
        strategy2 = Mock()
        strategy2.config.strategy_id = "strategy2"
        strategy2.status = ExitStatus.ACTIVE
        monitor2.register_strategy(strategy2)
        
        summary = alert_manager.get_global_monitoring_summary()
        
        assert summary['total_monitors'] == 2
        assert summary['total_strategies'] == 2
        assert 'monitor_summaries' in summary
        assert 'monitor1' in summary['monitor_summaries']
        assert 'monitor2' in summary['monitor_summaries']


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    def test_create_basic_monitor(self):
        """Test create_basic_monitor function"""
        monitor = create_basic_monitor()
        assert isinstance(monitor, ExitStrategyMonitor)
        assert monitor is not None
    
    def test_create_email_alert_handler(self):
        """Test create_email_alert_handler function"""
        handler = create_email_alert_handler(
            smtp_server="smtp.test.com",
            smtp_port=587,
            username="test@example.com",
            password="password123",
            recipients=["admin@example.com"]
        )
        
        assert isinstance(handler, EmailAlertHandler)
        assert handler.smtp_config['server'] == "smtp.test.com"
        assert handler.smtp_config['port'] == 587
        assert "admin@example.com" in handler.smtp_config['recipients']
    
    def test_create_slack_alert_handler(self):
        """Test create_slack_alert_handler function"""
        handler = create_slack_alert_handler(
            webhook_url="https://hooks.slack.com/services/test",
            channel="#alerts"
        )
        
        assert isinstance(handler, SlackAlertHandler)
        assert handler.webhook_url == "https://hooks.slack.com/services/test"
        assert handler.channel == "#alerts"
    
    def test_create_webhook_alert_handler(self):
        """Test create_webhook_alert_handler function"""
        headers = {'Content-Type': 'application/json'}
        handler = create_webhook_alert_handler(
            webhook_url="https://webhook.site/test",
            headers=headers
        )
        
        assert isinstance(handler, WebhookAlertHandler)
        assert handler.webhook_url == "https://webhook.site/test"
        assert handler.headers == headers
    
    def test_add_performance_alert_rule(self):
        """Test add_performance_alert_rule function"""
        monitor = ExitStrategyMonitor()
        
        add_performance_alert_rule(
            monitor=monitor,
            strategy_id="test_strategy",
            min_win_rate=0.3,
            max_drawdown=0.15
        )
        
        rule_id = "performance_test_strategy"
        assert rule_id in monitor.alert_rules
        
        rule = monitor.alert_rules[rule_id]
        assert rule.name == "Performance Alert for test_strategy"
        assert rule.severity == AlertSeverity.WARNING
        assert AlertChannel.SLACK in rule.channels
    
    def test_add_exit_frequency_alert_rule(self):
        """Test add_exit_frequency_alert_rule function"""
        monitor = ExitStrategyMonitor()
        
        add_exit_frequency_alert_rule(
            monitor=monitor,
            strategy_id="test_strategy",
            max_exits_per_hour=15.0
        )
        
        rule_id = "frequency_test_strategy"
        assert rule_id in monitor.alert_rules
        
        rule = monitor.alert_rules[rule_id]
        assert rule.name == "High Exit Frequency for test_strategy"
        assert rule.severity == AlertSeverity.INFO


class TestMonitoringEdgeCases:
    """Test monitoring edge cases and error conditions"""
    
    @pytest.mark.asyncio
    async def test_monitor_with_failing_strategy(self):
        """Test monitoring with failing strategy"""
        monitor = ExitStrategyMonitor()
        
        # Create strategy that fails on status check
        failing_strategy = Mock(spec=BaseExitStrategy)
        failing_strategy.config.strategy_id = "failing_strategy"
        failing_strategy.get_status = AsyncMock(side_effect=Exception("Strategy failed"))
        
        monitor.register_strategy(failing_strategy)
        
        # Should handle gracefully
        await monitor._check_strategy(failing_strategy)
    
    @pytest.mark.asyncio
    async def test_alert_with_missing_channels(self):
        """Test alert with missing channel handlers"""
        monitor = ExitStrategyMonitor()
        
        alert = Alert(
            alert_id="test_alert",
            strategy_id="test_strategy",
            alert_type="Test Alert",
            severity=AlertSeverity.WARNING,
            message="Test message",
            channels=[AlertChannel.SMS]  # No handler for SMS
        )
        
        await monitor._send_alert(alert)
        
        # Should still store alert even if no handlers available
        assert len(monitor.alerts) == 1
        assert monitor.alerts[0].alert_id == "test_alert"
    
    def test_monitor_with_many_alerts(self):
        """Test monitor with many alerts (alert limiting)"""
        monitor = ExitStrategyMonitor()
        
        # Add many alerts to test limiting
        for i in range(1500):
            alert = Alert(
                alert_id=f"alert_{i}",
                strategy_id="test_strategy",
                alert_type=f"Alert {i}",
                severity=AlertSeverity.INFO,
                message=f"Alert message {i}"
            )
            monitor.alerts.append(alert)
        
        # Should limit to last 500 alerts
        assert len(monitor.alerts) <= 500
        # Should keep most recent ones
        assert monitor.alerts[-1].alert_id == "alert_1499"
    
    @pytest.mark.asyncio
    async def test_concurrent_alert_processing(self):
        """Test concurrent alert processing"""
        monitor = ExitStrategyMonitor()
        
        # Simulate rapid alert generation
        tasks = []
        for i in range(10):
            alert = Alert(
                alert_id=f"concurrent_alert_{i}",
                strategy_id="test_strategy",
                alert_type="Concurrent Alert",
                severity=AlertSeverity.INFO,
                message=f"Concurrent message {i}"
            )
            task = monitor._send_alert(alert)
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        assert len(monitor.alerts) == 10


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])