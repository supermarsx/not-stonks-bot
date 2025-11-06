"""
Integration Tests for Complete Trading System
Tests the integration between all major components
"""

import asyncio
import pytest
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.application import app_config, ApplicationConfig
from config.settings import settings
from ui.terminal import TerminalUI
from ui.components.dashboard import DashboardManager
from ai.orchestrator import AITradingOrchestrator, TradingMode, StrategyType
from risk.manager import RiskManager
from oms.manager import OrderManager


class TestSystemIntegration:
    """Integration tests for the complete trading system"""
    
    @pytest.fixture
    async def test_app_config(self):
        """Create test application configuration"""
        config = ApplicationConfig()
        yield config
        # Cleanup is handled by the config's shutdown method
    
    @pytest.fixture
    async def initialized_system(self):
        """Initialize complete system for testing"""
        config = ApplicationConfig()
        
        # Initialize system
        success = await config.initialize()
        assert success, "System initialization failed"
        
        yield config
        
        # Cleanup
        await config.shutdown()
    
    async def test_app_config_initialization(self, test_app_config):
        """Test application configuration initialization"""
        # Test directory creation
        await test_app_config._create_directories()
        
        # Test database initialization
        await test_app_config._initialize_database()
        
        # Test AI components
        await test_app_config._initialize_ai_components()
        
        # Test risk management
        await test_app_config._initialize_risk_management()
        
        # Test order management
        await test_app_config._initialize_order_management()
        
        assert test_app_config.state.initialized
        assert test_app_config.state.ai_orchestrator is not None
        assert test_app_config.state.risk_manager is not None
        assert test_app_config.state.order_manager is not None
    
    async def test_complete_system_startup(self, initialized_system):
        """Test complete system startup flow"""
        config = initialized_system
        
        # Check all components are initialized
        assert config.state.initialized
        assert config.state.health_checks['database']['healthy']
        assert config.state.health_checks['ai']['healthy']
        assert config.state.health_checks['risk']['healthy']
        assert config.state.health_checks['oms']['healthy']
        
        # Check health status
        health_status = await config.get_health_status()
        assert health_status['status'] in ['healthy', 'degraded']  # degraded is OK if no brokers
        
        # Check system overview
        overview = await config.get_system_overview()
        assert 'system_health' in overview
        assert 'performance' in overview
        assert 'components' in overview
    
    async def test_ui_integration(self, initialized_system):
        """Test UI integration with real data"""
        config = initialized_system
        ui = TerminalUI()
        
        # Create dashboard manager
        dashboard = DashboardManager(config, ui)
        
        # Setup data feeds
        await dashboard.setup_real_data_feeds()
        
        # Test data retrieval
        account_data = dashboard.get_account_data()
        assert isinstance(account_data, dict)
        assert 'balance' in account_data
        assert 'equity' in account_data
        
        positions_data = dashboard.get_positions_data()
        assert isinstance(positions_data, list)
        
        risk_data = dashboard.get_risk_data()
        assert isinstance(risk_data, dict)
        assert 'daily_pnl' in risk_data
        
        system_data = dashboard.get_system_data()
        assert isinstance(system_data, dict)
        assert 'brokers' in system_data
    
    async def test_ai_orchestrator_integration(self, initialized_system):
        """Test AI orchestrator integration"""
        config = initialized_system
        orchestrator = config.state.ai_orchestrator
        
        assert orchestrator is not None
        assert orchestrator.trading_mode == TradingMode.PAPER  # Should be paper in test/dev
        
        # Test market analysis
        analysis = await orchestrator.analyze_market(
            symbols=['AAPL', 'MSFT'],
            analysis_type='quick'
        )
        
        assert 'analysis' in analysis
        assert 'session_id' in analysis
        assert 'model_used' in analysis
        
        # Test trading opportunity evaluation
        evaluation = await orchestrator.evaluate_trading_opportunity(
            symbol='AAPL',
            opportunity_type='breakout',
            context={'price': 150.0, 'volume': 1000000}
        )
        
        assert 'evaluation' in evaluation
        assert 'timestamp' in evaluation
        
        # Test performance stats
        stats = orchestrator.get_performance_stats()
        assert 'decisions_made' in stats
        assert 'trades_executed' in stats
        assert 'analysis_count' in stats
    
    async def test_risk_management_integration(self, initialized_system):
        """Test risk management integration"""
        config = initialized_system
        risk_manager = config.state.risk_manager
        
        assert risk_manager is not None
        
        # Test trade risk check
        from decimal import Decimal
        risk_check = await risk_manager.check_trade_risk(
            symbol='AAPL',
            side='buy',
            quantity=Decimal('10'),
            price=Decimal('150.00'),
            account_value=Decimal('100000')
        )
        
        assert 'approved' in risk_check
        assert isinstance(risk_check['approved'], bool)
        
        # Test position risk calculation
        positions = [
            {
                'symbol': 'AAPL',
                'quantity': 10,
                'market_value': 1500.0,
                'unrealized_pnl': 50.0
            }
        ]
        
        position_risks = await risk_manager.calculate_position_risks(positions)
        assert len(position_risks) == 1
        assert position_risks[0].symbol == 'AAPL'
        
        # Test risk summary
        risk_summary = await risk_manager.get_risk_summary()
        assert 'risk_metrics' in risk_summary
        assert 'compliance_status' in risk_summary
    
    async def test_order_management_integration(self, initialized_system):
        """Test order management integration"""
        config = initialized_system
        order_manager = config.state.order_manager
        
        assert order_manager is not None
        
        # Mock broker for testing
        class MockBroker:
            async def submit_order(self, order):
                return {'success': True, 'broker_order_id': 'MOCK_123'}
        
        order_manager.register_broker('mock', MockBroker())
        
        # Test order submission
        from decimal import Decimal
        result = await order_manager.submit_order(
            symbol='AAPL',
            side='buy',
            quantity=Decimal('10'),
            order_type='limit',
            price=Decimal('150.00'),
            broker_name='mock'
        )
        
        assert 'success' in result
        assert isinstance(result['success'], bool)
        
        if result['success']:
            # Test order status retrieval
            order_id = result['order_id']
            status = await order_manager.get_order_status(order_id)
            assert status is not None
            assert status['symbol'] == 'AAPL'
            
            # Test performance metrics
            metrics = await order_manager.get_performance_metrics()
            assert 'orders' in metrics
            assert 'volume' in metrics
        
        # Test order cancellation
        if result['success']:
            cancel_result = await order_manager.cancel_order(order_id)
            assert 'success' in cancel_result
    
    async def test_dashboard_data_integration(self, initialized_system):
        """Test dashboard data integration"""
        config = initialized_system
        ui = TerminalUI()
        dashboard = DashboardManager(config, ui)
        
        await dashboard.setup_real_data_feeds()
        
        # Test data refresh
        await dashboard.refresh_all_data()
        
        # Verify all data types are populated
        account_data = dashboard.get_account_data()
        assert account_data['balance'] > 0
        
        positions_data = dashboard.get_positions_data()
        assert isinstance(positions_data, list)
        
        orders_data = dashboard.get_orders_data()
        assert isinstance(orders_data, list)
        
        risk_data = dashboard.get_risk_data()
        assert risk_data['daily_pnl'] is not None
        
        system_data = dashboard.get_system_data()
        assert 'brokers' in system_data
    
    async def test_end_to_end_workflow(self, initialized_system):
        """Test complete end-to-end trading workflow"""
        config = initialized_system
        
        # 1. System is running
        assert config.state.initialized
        
        # 2. AI analyzes market
        ai_orchestrator = config.state.ai_orchestrator
        analysis = await ai_orchestrator.analyze_market(['AAPL'])
        assert analysis['analysis'] is not None
        
        # 3. AI evaluates opportunity
        evaluation = await ai_orchestrator.evaluate_trading_opportunity(
            'AAPL', 'momentum', {'price': 150.0}
        )
        assert evaluation['evaluation'] is not None
        
        # 4. Risk check passes
        risk_manager = config.state.risk_manager
        from decimal import Decimal
        risk_check = await risk_manager.check_trade_risk(
            'AAPL', 'buy', Decimal('10'), Decimal('150.00'), Decimal('100000')
        )
        assert risk_check['approved']
        
        # 5. Order gets submitted
        order_manager = config.state.order_manager
        
        # Mock broker
        class MockBroker:
            async def submit_order(self, order):
                return {'success': True, 'broker_order_id': f'MOCK_{order.order_id}'}
        
        order_manager.register_broker('mock', MockBroker())
        
        order_result = await order_manager.submit_order(
            'AAPL', 'buy', Decimal('10'), 'limit', Decimal('150.00'), 'mock'
        )
        assert order_result['success']
        
        # 6. Dashboard reflects changes
        ui = TerminalUI()
        dashboard = DashboardManager(config, ui)
        await dashboard.setup_real_data_feeds()
        
        # Verify order appears in dashboard
        orders_data = dashboard.get_orders_data()
        assert any(order['symbol'] == 'AAPL' for order in orders_data)
    
    async def test_health_monitoring(self, initialized_system):
        """Test system health monitoring"""
        config = initialized_system
        
        # Get initial health status
        health = await config.get_health_status()
        assert health['status'] in ['healthy', 'degraded']
        assert 'components' in health
        assert 'uptime_seconds' in health
        
        # Verify component health checks
        components = health['components']
        
        expected_components = ['database', 'ai', 'risk', 'oms']
        for component in expected_components:
            assert component in components
            assert 'healthy' in components[component]
            assert 'timestamp' in components[component]
        
        # Verify system overview includes health
        overview = await config.get_system_overview()
        assert 'system_health' in overview
    
    async def test_error_handling(self, initialized_system):
        """Test system error handling"""
        config = initialized_system
        
        # Test invalid order submission
        order_manager = config.state.order_manager
        
        from decimal import Decimal
        
        # Should handle invalid symbols gracefully
        result = await order_manager.submit_order(
            symbol='INVALID_SYMBOL_123',
            side='buy',
            quantity=Decimal('10'),
            broker_name='nonexistent'
        )
        
        assert 'success' in result
        assert result['success'] is False  # Should fail gracefully
        
        # Test risk manager with invalid data
        risk_manager = config.state.risk_manager
        
        # Should handle zero account value gracefully
        risk_check = await risk_manager.check_trade_risk(
            'AAPL', 'buy', Decimal('10'), Decimal('150.00'), Decimal('0')
        )
        
        assert 'approved' in risk_check
        # Should still provide a response even with edge case
    
    async def test_performance_metrics(self, initialized_system):
        """Test system performance metrics collection"""
        config = initialized_system
        
        # AI orchestrator metrics
        ai_stats = config.state.ai_orchestrator.get_performance_stats()
        assert isinstance(ai_stats, dict)
        assert all(key in ai_stats for key in ['decisions_made', 'trades_executed', 'analysis_count'])
        
        # Order manager metrics
        order_metrics = await config.state.order_manager.get_performance_metrics()
        assert isinstance(order_metrics, dict)
        assert 'orders' in order_metrics
        assert 'volume' in order_metrics
        
        # Risk manager summary
        risk_summary = await config.state.risk_manager.get_risk_summary()
        assert isinstance(risk_summary, dict)
        assert 'risk_metrics' in risk_summary
        assert 'compliance_status' in risk_summary
        
        # System overview
        overview = await config.get_system_overview()
        assert 'performance' in overview
        assert isinstance(overview['performance'], dict)


class TestSystemStartupShutdown:
    """Test system startup and shutdown procedures"""
    
    async def test_clean_startup(self):
        """Test clean system startup"""
        config = ApplicationConfig()
        
        try:
            # Initialize system
            success = await config.initialize()
            assert success
            
            # Verify all components are healthy
            health = await config.get_health_status()
            assert health['status'] in ['healthy', 'degraded']
            
        finally:
            # Ensure cleanup
            await config.shutdown()
    
    async def test_graceful_shutdown(self):
        """Test graceful shutdown procedure"""
        config = ApplicationConfig()
        
        try:
            # Start system
            await config.initialize()
            assert config.state.initialized
            
            # Simulate shutdown
            await config.shutdown()
            assert not config.state.initialized
            
        except Exception as e:
            # Cleanup even on failure
            await config.shutdown()
            raise e
    
    async def test_multiple_startup_attempts(self):
        """Test handling multiple startup attempts"""
        config = ApplicationConfig()
        
        try:
            # First startup
            success1 = await config.initialize()
            assert success1
            
            # Second startup attempt should be handled gracefully
            # (actual behavior depends on implementation)
            # For now, just verify system is still running
            assert config.state.initialized
            
        finally:
            await config.shutdown()


async def run_integration_tests():
    """Run all integration tests"""
    print("🧪 Running Integration Tests...")
    
    test_classes = [
        TestSystemIntegration,
        TestSystemStartupShutdown
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n📋 Running {test_class.__name__}...")
        test_instance = test_class()
        
        # Get all test methods
        test_methods = [
            method for method in dir(test_instance)
            if method.startswith('test_') and callable(getattr(test_instance, method))
        ]
        
        for test_method_name in test_methods:
            total_tests += 1
            test_method = getattr(test_instance, test_method_name)
            
            try:
                print(f"  🧪 {test_method_name}... ", end="")
                
                # Create fresh test fixtures for each test
                if 'initialized_system' in test_method_name:
                    async with TestSystemIntegration().initialized_system() as initialized_system:
                        await test_method(initialized_system)
                elif 'test_app_config' in test_method_name:
                    test_instance = test_class()
                    await test_method(test_instance.test_app_config())
                else:
                    test_instance = test_class()
                    await test_method()
                
                print("✅ PASS")
                passed_tests += 1
                
            except Exception as e:
                print(f"❌ FAIL: {str(e)[:100]}")
                failed_tests.append(f"{test_class.__name__}.{test_method_name}: {str(e)}")
    
    # Print summary
    print(f"\n📊 Test Results:")
    print(f"  Total Tests: {total_tests}")
    print(f"  Passed: {passed_tests}")
    print(f"  Failed: {len(failed_tests)}")
    
    if failed_tests:
        print(f"\n❌ Failed Tests:")
        for failure in failed_tests:
            print(f"  - {failure}")
    
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    print(f"\n🎯 Success Rate: {success_rate:.1f}%")
    
    return len(failed_tests) == 0


if __name__ == "__main__":
    # Run integration tests
    success = asyncio.run(run_integration_tests())
    sys.exit(0 if success else 1)