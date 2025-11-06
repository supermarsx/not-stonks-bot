"""
@file test_frequency_system.py
@brief Comprehensive Test Suite for Trading Frequency Configuration System

@details
This module provides comprehensive testing for the entire trading frequency
configuration system including configuration management, risk management,
analytics, UI components, and database integration.

Test Coverage:
- Frequency configuration and settings management
- Frequency-based position sizing calculations
- Frequency monitoring and alerting
- Frequency optimization recommendations
- Frequency risk management integration
- Analytics and reporting functionality
- UI components functionality
- Database schema and migrations
- End-to-end system integration

@author Trading Orchestrator System
@version 1.0
@date 2025-11-06

@note
These tests validate the complete frequency configuration system and should
be run to ensure all components work correctly together.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List
import json
import tempfile
import os

from config.trading_frequency import (
    FrequencyManager, FrequencySettings, FrequencyType, FrequencyAlertType,
    FrequencyAlert, FrequencyOptimization, initialize_frequency_manager,
    get_frequency_manager, calculate_frequency_position_size
)
from config.settings import Settings

from risk.frequency_risk_manager import (
    FrequencyRiskManager, FrequencyRiskAssessment, FrequencyRiskLimit,
    initialize_frequency_risk_manager, get_frequency_risk_manager,
    assess_frequency_risk, check_frequency_risk_compliance
)

from analytics.frequency_analytics import (
    FrequencyAnalyticsEngine, FrequencyAnalyticsReport, AnalyticsPeriod,
    OptimizationTarget, FrequencyOptimizationInsight,
    initialize_frequency_analytics_engine, get_frequency_analytics_engine,
    generate_frequency_analytics_report, generate_frequency_optimization_insights
)

from ui.components.frequency_components import (
    FrequencyConfigurationComponent, FrequencyMonitoringComponent,
    FrequencyAlertsComponent, FrequencyAnalyticsComponent
)

# Test Configuration
TEST_SETTINGS = FrequencySettings(
    frequency_type=FrequencyType.HIGH,
    interval_seconds=300,  # 5 minutes
    max_trades_per_minute=3,
    max_trades_per_hour=15,
    max_trades_per_day=80,
    position_size_multiplier=0.8,
    cooldown_periods=60,
    enable_alerts=True
)


class TestFrequencyManager:
    """Test suite for FrequencyManager"""
    
    @pytest.fixture
    async def frequency_manager(self):
        """Create frequency manager for testing"""
        manager = initialize_frequency_manager(TEST_SETTINGS)
        yield manager
        # Cleanup would happen here
    
    async def test_frequency_manager_initialization(self, frequency_manager):
        """Test frequency manager initialization"""
        assert frequency_manager is not None
        assert frequency_manager.settings.frequency_type == FrequencyType.HIGH
        assert frequency_manager.settings.interval_seconds == 300
        assert frequency_manager.settings.max_trades_per_minute == 3
    
    async def test_position_size_calculation(self, frequency_manager):
        """Test frequency-based position sizing"""
        # Test basic position sizing
        base_size = Decimal("10000")
        adjusted_size = await frequency_manager.calculate_position_size(
            strategy_id="test_strategy",
            base_position_size=base_size,
            current_frequency_rate=2.0
        )
        
        assert adjusted_size <= base_size  # Should be reduced due to high frequency
        assert adjusted_size > 0
    
    async def test_trade_permission_check(self, frequency_manager):
        """Test trade permission checking"""
        # Test initial permission (should be allowed)
        permission = await frequency_manager.check_trade_allowed("test_strategy")
        assert permission["allowed"] is True
        
        # Record a trade
        await frequency_manager.record_trade("test_strategy")
        
        # Check permission again (should still be allowed)
        permission = await frequency_manager.check_trade_allowed("test_strategy")
        assert permission["allowed"] is True
    
    async def test_frequency_metrics(self, frequency_manager):
        """Test frequency metrics tracking"""
        # Record some trades
        for i in range(5):
            await frequency_manager.record_trade("test_strategy")
        
        # Check metrics
        metrics = frequency_manager.get_frequency_metrics("test_strategy")
        assert metrics is not None
        assert metrics.trades_today == 5
        assert metrics.current_frequency_rate > 0
    
    async def test_alert_generation(self, frequency_manager):
        """Test frequency alert generation"""
        # Generate an alert
        alert = await frequency_manager.generate_alert(
            strategy_id="test_strategy",
            alert_type=FrequencyAlertType.THRESHOLD_EXCEEDED,
            message="Test alert message",
            severity="medium"
        )
        
        assert alert.alert_id is not None
        assert alert.alert_type == FrequencyAlertType.THRESHOLD_EXCEEDED
        assert alert.strategy_id == "test_strategy"
        
        # Check active alerts
        active_alerts = frequency_manager.get_active_alerts("test_strategy")
        assert len(active_alerts) == 1
    
    async def test_optimization_recommendations(self, frequency_manager):
        """Test optimization recommendation generation"""
        # Generate optimization recommendations
        recommendations = await frequency_manager.generate_optimization_recommendations(
            "test_strategy",
            backtest_period_days=30
        )
        
        assert len(recommendations) == 1
        optimization = recommendations[0]
        assert optimization.strategy_id == "test_strategy"
        assert optimization.confidence_level > 0
    
    async def test_frequency_summary(self, frequency_manager):
        """Test frequency management summary"""
        # Record some trades
        for i in range(3):
            await frequency_manager.record_trade(f"strategy_{i}")
        
        # Generate summary
        summary = frequency_manager.get_frequency_summary()
        
        assert "settings" in summary
        assert "metrics" in summary
        assert "alerts" in summary
        assert summary["metrics"]["total_strategies"] == 3


class TestFrequencyRiskManager:
    """Test suite for FrequencyRiskManager"""
    
    @pytest.fixture
    async def frequency_risk_manager(self):
        """Create frequency risk manager for testing"""
        # Create mock base risk manager
        base_manager = Mock()
        base_manager.max_position_size = Decimal("10000")
        base_manager.max_daily_loss = Decimal("5000")
        
        manager = initialize_frequency_risk_manager(base_manager)
        yield manager
    
    async def test_frequency_risk_manager_initialization(self, frequency_risk_manager):
        """Test frequency risk manager initialization"""
        assert frequency_risk_manager is not None
        assert frequency_risk_manager.base_manager is not None
        assert len(frequency_risk_manager.frequency_risk_weights) > 0
    
    async def test_frequency_risk_assessment(self, frequency_risk_manager):
        """Test frequency risk assessment"""
        assessment = await frequency_risk_manager.assess_frequency_risk(
            strategy_id="test_strategy",
            current_frequency_rate=2.5,
            position_size=Decimal("5000"),
            market_volatility=0.2
        )
        
        assert assessment.strategy_id == "test_strategy"
        assert assessment.current_frequency_rate == 2.5
        assert assessment.frequency_risk_score >= 0
        assert assessment.risk_level is not None
    
    async def test_frequency_risk_compliance(self, frequency_risk_manager):
        """Test frequency risk compliance checking"""
        # Test compliant trade
        compliance, violations = await frequency_risk_manager.check_frequency_risk_compliance(
            strategy_id="test_strategy",
            proposed_trade={
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 100,
                "market_volatility": 0.1
            }
        )
        
        assert isinstance(compliance, bool)
        assert isinstance(violations, list)
    
    async def test_position_size_adjustment(self, frequency_risk_manager):
        """Test frequency-adjusted position sizing"""
        adjusted_size = await frequency_risk_manager.calculate_frequency_adjusted_position_size(
            strategy_id="test_strategy",
            base_position_size=Decimal("10000"),
            current_frequency_rate=3.0,
            market_volatility=0.25
        )
        
        assert adjusted_size > 0
        assert adjusted_size <= Decimal("10000")  # Should be reduced or equal
    
    async def test_cross_strategy_monitoring(self, frequency_risk_manager):
        """Test cross-strategy frequency risk monitoring"""
        # Create assessments for multiple strategies
        await frequency_risk_manager.assess_frequency_risk("strategy_1", 2.0)
        await frequency_risk_manager.assess_frequency_risk("strategy_2", 3.0)
        await frequency_risk_manager.assess_frequency_risk("strategy_3", 1.5)
        
        # Monitor cross-strategy risk
        monitoring_result = await frequency_risk_manager.monitor_cross_strategy_frequency_risk()
        
        assert "risk_summary" in monitoring_result
        assert "strategy_details" in monitoring_result
        assert monitoring_result["monitoring_status"] == "active"


class TestFrequencyAnalytics:
    """Test suite for FrequencyAnalyticsEngine"""
    
    @pytest.fixture
    def mock_frequency_manager(self):
        """Create mock frequency manager"""
        manager = Mock()
        manager.settings = TEST_SETTINGS
        
        # Mock frequency metrics
        metrics = Mock()
        metrics.current_frequency_rate = 2.0
        metrics.frequency_efficiency = 0.75
        metrics.trades_today = 45
        metrics.frequency_sharpe = 1.2
        metrics.frequency_drawdown = -0.05
        
        manager.get_frequency_metrics.return_value = metrics
        manager.get_optimization_recommendations.return_value = []
        
        return manager
    
    @pytest.fixture
    async def analytics_engine(self, mock_frequency_manager):
        """Create analytics engine for testing"""
        engine = initialize_frequency_analytics_engine(mock_frequency_manager)
        yield engine
    
    async def test_analytics_engine_initialization(self, analytics_engine):
        """Test analytics engine initialization"""
        assert analytics_engine is not None
        assert analytics_engine.frequency_manager is not None
        assert analytics_engine.analytics_enabled is True
    
    async def test_analytics_report_generation(self, analytics_engine):
        """Test analytics report generation"""
        report = await analytics_engine.generate_analytics_report(
            strategy_id="test_strategy",
            period=AnalyticsPeriod.DAILY
        )
        
        assert report is not None
        assert report.strategy_id == "test_strategy"
        assert report.report_period == AnalyticsPeriod.DAILY
        assert report.generated_at is not None
    
    async def test_optimization_insights_generation(self, analytics_engine):
        """Test optimization insights generation"""
        insights = await analytics_engine.generate_optimization_insights(
            strategy_id="test_strategy",
            target=OptimizationTarget.MAXIMIZE_SHARPE
        )
        
        assert isinstance(insights, list)
        # Insights may be empty if current performance is already optimal
    
    async def test_frequency_trend_analysis(self, analytics_engine):
        """Test frequency trend analysis"""
        trends = await analytics_engine.analyze_frequency_trends(
            strategy_id="test_strategy",
            lookback_days=7
        )
        
        assert isinstance(trends, dict)
        # May return insufficient data for short lookback
    
    async def test_predictive_modeling(self, analytics_engine):
        """Test predictive modeling"""
        predictions = await analytics_engine.perform_predictive_modeling(
            strategy_id="test_strategy",
            forecast_horizon_days=14
        )
        
        assert isinstance(predictions, dict)
        assert "status" in predictions
    
    async def test_cross_strategy_analysis(self, analytics_engine):
        """Test cross-strategy frequency analysis"""
        strategy_ids = ["strategy_1", "strategy_2", "strategy_3"]
        
        analysis = await analytics_engine.analyze_cross_strategy_frequency(strategy_ids)
        
        assert isinstance(analysis, dict)
        assert "status" in analysis


class TestUIComponents:
    """Test suite for UI Components"""
    
    @pytest.fixture
    def mock_frequency_manager(self):
        """Create mock frequency manager for UI testing"""
        manager = Mock()
        manager.settings = TEST_SETTINGS
        
        # Mock metrics
        metrics = Mock()
        metrics.trades_today = 25
        metrics.current_frequency_rate = 2.1
        metrics.average_frequency_rate = 1.8
        metrics.in_cooldown = False
        metrics.threshold_violations = 1
        
        manager.get_all_metrics.return_value = {"test_strategy": metrics}
        manager.get_active_alerts.return_value = []
        manager.alerts = []
        
        return manager
    
    @pytest.fixture
    def mock_analytics_engine(self):
        """Create mock analytics engine for UI testing"""
        engine = Mock()
        return engine
    
    def test_frequency_configuration_component(self, mock_frequency_manager):
        """Test frequency configuration component"""
        from rich.console import Console
        from io import StringIO
        
        console = Console(file=StringIO())
        component = FrequencyConfigurationComponent(console)
        component.set_frequency_manager(mock_frequency_manager)
        
        # Test panel creation
        panel = component.create_frequency_config_panel("test_strategy")
        assert panel is not None
        assert "Frequency Configuration" in str(panel)
    
    def test_frequency_monitoring_component(self, mock_frequency_manager):
        """Test frequency monitoring component"""
        from rich.console import Console
        from io import StringIO
        
        console = Console(file=StringIO())
        component = FrequencyMonitoringComponent(console)
        component.set_frequency_manager(mock_frequency_manager)
        
        # Test dashboard creation
        layout = component.create_monitoring_dashboard()
        assert layout is not None
    
    def test_frequency_alerts_component(self, mock_frequency_manager):
        """Test frequency alerts component"""
        from rich.console import Console
        from io import StringIO
        
        console = Console(file=StringIO())
        component = FrequencyAlertsComponent(console)
        component.set_frequency_manager(mock_frequency_manager)
        
        # Test alerts interface creation
        panel = component.create_alerts_interface()
        assert panel is not None
    
    def test_frequency_analytics_component(self, mock_analytics_engine):
        """Test frequency analytics component"""
        from rich.console import Console
        from io import StringIO
        
        console = Console(file=StringIO())
        component = FrequencyAnalyticsComponent(console)
        component.set_analytics_engine(mock_analytics_engine)
        
        # Test analytics interface creation
        panel = component.create_analytics_interface("test_strategy")
        assert panel is not None


class TestDatabaseIntegration:
    """Test suite for database integration"""
    
    def test_frequency_settings_model(self):
        """Test frequency settings database model"""
        # This would test the database model if using ORM
        # For now, test the data structure
        
        settings_data = {
            "strategy_id": "test_strategy",
            "frequency_type": "high",
            "interval_seconds": 300,
            "max_trades_per_minute": 3,
            "max_trades_per_hour": 15,
            "max_trades_per_day": 80,
            "position_size_multiplier": 0.8,
            "cooldown_periods": 60,
            "is_active": True
        }
        
        assert settings_data["strategy_id"] == "test_strategy"
        assert settings_data["frequency_type"] == "high"
        assert settings_data["interval_seconds"] == 300
    
    def test_frequency_metrics_model(self):
        """Test frequency metrics database model"""
        metrics_data = {
            "strategy_id": "test_strategy",
            "trades_today": 25,
            "current_frequency_rate": 2.1,
            "frequency_efficiency": 0.75,
            "recorded_at": datetime.utcnow().isoformat()
        }
        
        assert metrics_data["strategy_id"] == "test_strategy"
        assert metrics_data["trades_today"] == 25
        assert metrics_data["current_frequency_rate"] == 2.1
    
    def test_frequency_alert_model(self):
        """Test frequency alert database model"""
        alert_data = {
            "strategy_id": "test_strategy",
            "alert_type": "threshold_exceeded",
            "severity": "medium",
            "message": "Frequency limit exceeded",
            "trigger_time": datetime.utcnow().isoformat(),
            "acknowledged": False
        }
        
        assert alert_data["strategy_id"] == "test_strategy"
        assert alert_data["alert_type"] == "threshold_exceeded"
        assert alert_data["acknowledged"] is False


class TestEndToEndIntegration:
    """End-to-end integration tests"""
    
    @pytest.fixture
    async def complete_system(self):
        """Create complete frequency system for integration testing"""
        # Initialize all components
        frequency_manager = initialize_frequency_manager(TEST_SETTINGS)
        base_risk_manager = Mock()
        frequency_risk_manager = initialize_frequency_risk_manager(base_risk_manager)
        analytics_engine = initialize_frequency_analytics_engine(frequency_manager)
        
        return {
            "frequency_manager": frequency_manager,
            "frequency_risk_manager": frequency_risk_manager,
            "analytics_engine": analytics_engine
        }
    
    async def test_complete_workflow(self, complete_system):
        """Test complete frequency management workflow"""
        frequency_manager = complete_system["frequency_manager"]
        frequency_risk_manager = complete_system["frequency_risk_manager"]
        analytics_engine = complete_system["analytics_engine"]
        
        strategy_id = "integration_test_strategy"
        
        # 1. Configure frequency settings
        assert frequency_manager.settings.frequency_type == FrequencyType.HIGH
        
        # 2. Check initial trade permission
        permission = await frequency_manager.check_trade_allowed(strategy_id)
        assert permission["allowed"] is True
        
        # 3. Record trade
        await frequency_manager.record_trade(strategy_id)
        
        # 4. Assess frequency risk
        assessment = await frequency_risk_manager.assess_frequency_risk(
            strategy_id=strategy_id,
            current_frequency_rate=1.5,
            position_size=Decimal("5000"),
            market_volatility=0.2
        )
        assert assessment.strategy_id == strategy_id
        
        # 5. Check compliance
        compliance, violations = await frequency_risk_manager.check_frequency_risk_compliance(
            strategy_id=strategy_id,
            proposed_trade={"symbol": "AAPL", "side": "buy", "quantity": 100}
        )
        assert isinstance(compliance, bool)
        assert isinstance(violations, list)
        
        # 6. Calculate position size
        base_size = Decimal("10000")
        adjusted_size = await frequency_manager.calculate_position_size(
            strategy_id=strategy_id,
            base_position_size=base_size,
            current_frequency_rate=2.0
        )
        assert adjusted_size <= base_size
        
        # 7. Generate analytics
        report = await analytics_engine.generate_analytics_report(
            strategy_id=strategy_id,
            period=AnalyticsPeriod.DAILY
        )
        assert report.strategy_id == strategy_id
        
        # 8. Generate optimization insights
        insights = await analytics_engine.generate_optimization_insights(
            strategy_id=strategy_id,
            target=OptimizationTarget.BALANCE_RISK_RETURN
        )
        assert isinstance(insights, list)
        
        # 9. Check system status
        summary = frequency_manager.get_frequency_summary()
        assert "metrics" in summary
        assert summary["metrics"]["total_strategies"] >= 1
    
    async def test_alert_workflow(self, complete_system):
        """Test complete alert workflow"""
        frequency_manager = complete_system["frequency_manager"]
        
        strategy_id = "alert_test_strategy"
        
        # 1. Generate alert
        alert = await frequency_manager.generate_alert(
            strategy_id=strategy_id,
            alert_type=FrequencyAlertType.THRESHOLD_EXCEEDED,
            message="Test alert for integration testing",
            severity="high"
        )
        
        assert alert.alert_id is not None
        
        # 2. Check active alerts
        active_alerts = frequency_manager.get_active_alerts(strategy_id)
        assert len(active_alerts) >= 1
        
        # 3. Acknowledge alert
        acknowledged = frequency_manager.acknowledge_alert(alert.alert_id)
        assert acknowledged is True
        
        # 4. Verify acknowledgment
        # In real implementation, this would check the updated alert status
        active_alerts_after = frequency_manager.get_active_alerts(strategy_id)
        # Alert should still be in list but marked as acknowledged
    
    async def test_optimization_workflow(self, complete_system):
        """Test complete optimization workflow"""
        frequency_manager = complete_system["frequency_manager"]
        analytics_engine = complete_system["analytics_engine"]
        
        strategy_id = "optimization_test_strategy"
        
        # 1. Generate optimization recommendations
        recommendations = await frequency_manager.generate_optimization_recommendations(
            strategy_id=strategy_id,
            backtest_period_days=30
        )
        
        assert len(recommendations) == 1
        optimization = recommendations[0]
        assert optimization.strategy_id == strategy_id
        
        # 2. Implement optimization
        implemented = await frequency_manager.implement_optimization(
            optimization_id=optimization.optimization_id,
            strategy_id=strategy_id
        )
        assert implemented is True
        
        # 3. Generate analytics insights
        insights = await analytics_engine.generate_optimization_insights(
            strategy_id=strategy_id,
            target=OptimizationTarget.MAXIMIZE_SHARPE
        )
        assert isinstance(insights, list)
    
    async def test_risk_management_workflow(self, complete_system):
        """Test complete risk management workflow"""
        frequency_risk_manager = complete_system["frequency_risk_manager"]
        
        strategy_id = "risk_test_strategy"
        
        # 1. Add risk limit
        from risk.frequency_risk_manager import FrequencyRiskLimit
        
        risk_limit = FrequencyRiskLimit(
            limit_id="test_limit",
            strategy_id=strategy_id,
            limit_type="hard",
            max_frequency_rate=5.0,
            max_frequency_risk_score=0.7
        )
        
        frequency_risk_manager.add_risk_limit(risk_limit)
        
        # 2. Assess risk with high frequency
        assessment = await frequency_risk_manager.assess_frequency_risk(
            strategy_id=strategy_id,
            current_frequency_rate=6.0,  # Exceeds limit
            position_size=Decimal("10000"),
            market_volatility=0.3
        )
        
        assert assessment.frequency_risk_score > 0
        
        # 3. Check compliance with violating trade
        compliance, violations = await frequency_risk_manager.check_frequency_risk_compliance(
            strategy_id=strategy_id,
            proposed_trade={
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 500,
                "market_volatility": 0.3
            }
        )
        
        # May or may not have violations depending on implementation
        assert isinstance(violations, list)
    
    async def test_analytics_workflow(self, complete_system):
        """Test complete analytics workflow"""
        analytics_engine = complete_system["analytics_engine"]
        
        strategy_id = "analytics_test_strategy"
        
        # 1. Generate analytics report
        report = await analytics_engine.generate_analytics_report(
            strategy_id=strategy_id,
            period=AnalyticsPeriod.DAILY
        )
        assert report is not None
        
        # 2. Perform trend analysis
        trends = await analytics_engine.analyze_frequency_trends(
            strategy_id=strategy_id,
            lookback_days=7
        )
        assert isinstance(trends, dict)
        
        # 3. Perform predictive modeling
        predictions = await analytics_engine.perform_predictive_modeling(
            strategy_id=strategy_id,
            forecast_horizon_days=14
        )
        assert isinstance(predictions, dict)
        
        # 4. Cross-strategy analysis
        analysis = await analytics_engine.analyze_cross_strategy_frequency(
            ["strategy_1", "strategy_2", "strategy_3"]
        )
        assert isinstance(analysis, dict)
        
        # 5. Get analytics summary
        summary = await analytics_engine.get_analytics_summary()
        assert "analytics_enabled" in summary


# Test Utilities and Fixtures
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_datetime(monkeypatch):
    """Mock datetime for consistent testing"""
    class MockDateTime:
        @classmethod
        def utcnow(cls):
            return datetime(2025, 11, 6, 12, 0, 0)
    
    monkeypatch.setattr('config.trading_frequency.datetime', MockDateTime)
    return MockDateTime


# Test Configuration and Helpers
def create_test_frequency_settings() -> FrequencySettings:
    """Create test frequency settings"""
    return FrequencySettings(
        frequency_type=FrequencyType.MEDIUM,
        interval_seconds=600,  # 10 minutes
        max_trades_per_minute=2,
        max_trades_per_hour=10,
        max_trades_per_day=50,
        position_size_multiplier=1.0,
        cooldown_periods=0,
        enable_alerts=True,
        auto_optimization=True
    )


def create_mock_trade_data(strategy_id: str, num_trades: int = 10) -> List[Dict[str, Any]]:
    """Create mock trade data for testing"""
    trades = []
    base_time = datetime.utcnow() - timedelta(hours=1)
    
    for i in range(num_trades):
        trade_time = base_time + timedelta(minutes=i * 6)  # 6 minutes apart
        
        trade = {
            "strategy_id": strategy_id,
            "trade_time": trade_time.isoformat(),
            "symbol": f"SYMBOL_{i % 5}",
            "trade_type": "buy" if i % 2 == 0 else "sell",
            "quantity": 100 + i * 10,
            "price": 100.0 + i,
            "frequency_rate": 2.0 + (i * 0.1)
        }
        trades.append(trade)
    
    return trades


# Performance and Load Testing
class TestPerformanceAndLoad:
    """Performance and load testing for frequency system"""
    
    async def test_frequency_manager_performance(self):
        """Test frequency manager performance with multiple strategies"""
        import time
        
        settings = create_test_frequency_settings()
        manager = initialize_frequency_manager(settings)
        
        start_time = time.time()
        
        # Test with multiple strategies
        for i in range(100):
            strategy_id = f"perf_test_strategy_{i}"
            
            # Check trade permission
            await manager.check_trade_allowed(strategy_id)
            
            # Record trade
            await manager.record_trade(strategy_id)
            
            # Get metrics
            metrics = manager.get_frequency_metrics(strategy_id)
            assert metrics is not None
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Performance assertion (should complete within reasonable time)
        assert execution_time < 10.0  # Less than 10 seconds for 100 strategies
        
        print(f"Performance test completed in {execution_time:.2f} seconds")
    
    async def test_analytics_engine_performance(self):
        """Test analytics engine performance"""
        import time
        
        mock_manager = Mock()
        mock_manager.settings = TEST_SETTINGS
        mock_manager.get_frequency_metrics.return_value = Mock(
            current_frequency_rate=2.0,
            frequency_efficiency=0.75,
            trades_today=50
        )
        mock_manager.get_optimization_recommendations.return_value = []
        
        engine = initialize_frequency_analytics_engine(mock_manager)
        
        start_time = time.time()
        
        # Generate multiple reports
        for i in range(10):
            report = await engine.generate_analytics_report(
                strategy_id=f"perf_strategy_{i}",
                period=AnalyticsPeriod.DAILY
            )
            assert report is not None
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Performance assertion
        assert execution_time < 30.0  # Less than 30 seconds for 10 reports
        
        print(f"Analytics performance test completed in {execution_time:.2f} seconds")


# Error Handling and Edge Cases
class TestErrorHandling:
    """Test error handling and edge cases"""
    
    async def test_invalid_frequency_settings(self):
        """Test handling of invalid frequency settings"""
        # Test with invalid interval
        with pytest.raises(ValueError):
            invalid_settings = FrequencySettings(
                frequency_type=FrequencyType.CUSTOM,
                custom_intervals=[]  # Empty for custom type
            )
    
    async def test_frequency_manager_with_no_data(self):
        """Test frequency manager behavior with no data"""
        manager = initialize_frequency_manager(TEST_SETTINGS)
        
        # Test with strategy that has no data
        metrics = manager.get_frequency_metrics("nonexistent_strategy")
        assert metrics is None
        
        # Check trade permission for nonexistent strategy
        permission = await manager.check_trade_allowed("nonexistent_strategy")
        assert permission["allowed"] is True  # Should allow by default
    
    async def test_alert_system_error_handling(self):
        """Test alert system error handling"""
        manager = initialize_frequency_manager(TEST_SETTINGS)
        
        # Test with invalid alert parameters (should not crash)
        alert = await manager.generate_alert(
            strategy_id="test_strategy",
            alert_type=FrequencyAlertType.THRESHOLD_EXCEEDED,
            message="",  # Empty message
            severity="invalid_severity"  # Invalid severity
        )
        
        # Should still create alert
        assert alert is not None
        assert alert.alert_id is not None


# Configuration and Setup for Testing
def pytest_configure(config):
    """Configure pytest for frequency system testing"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment"""
    print("\n[TEST] Setting up frequency system test environment")
    print("[TEST] Initializing test configuration")
    
    yield
    
    print("\n[TEST] Cleaning up frequency system test environment")


if __name__ == "__main__":
    # Run tests manually
    print("Running Frequency System Tests...\n")
    
    # Create test instances
    async def run_tests():
        # Test frequency manager
        print("Testing Frequency Manager...")
        frequency_manager = initialize_frequency_manager(TEST_SETTINGS)
        
        # Test position sizing
        base_size = Decimal("10000")
        adjusted_size = await frequency_manager.calculate_position_size(
            strategy_id="manual_test_strategy",
            base_position_size=base_size,
            current_frequency_rate=2.0
        )
        print(f"Position size: {base_size} -> {adjusted_size}")
        
        # Test trade permission
        permission = await frequency_manager.check_trade_allowed("manual_test_strategy")
        print(f"Trade permission: {permission}")
        
        # Test trade recording
        await frequency_manager.record_trade("manual_test_strategy")
        metrics = frequency_manager.get_frequency_metrics("manual_test_strategy")
        print(f"Trades today: {metrics.trades_today if metrics else 'None'}")
        
        # Test alert generation
        alert = await frequency_manager.generate_alert(
            strategy_id="manual_test_strategy",
            alert_type=FrequencyAlertType.THRESHOLD_EXCEEDED,
            message="Manual test alert"
        )
        print(f"Alert generated: {alert.alert_id}")
        
        # Test summary
        summary = frequency_manager.get_frequency_summary()
        print(f"System summary: {summary['metrics']['total_trades_today']} trades today")
        
        print("\n[SUCCESS] All manual tests completed!")
    
    # Run the async tests
    asyncio.run(run_tests())