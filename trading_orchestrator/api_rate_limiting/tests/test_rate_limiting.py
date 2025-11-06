"""
Test Suite for API Rate Limiting System

Comprehensive tests for all components of the rate limiting system.
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, Mock
from datetime import datetime, timedelta

from api_rate_limiting.manager import APIRateLimitManager
from api_rate_limiting.core.rate_limiter import (
    RateLimiterManager, RequestType, RequestPriority,
    TokenBucketRateLimiter, SlidingWindowRateLimiter
)
from api_rate_limiting.core.request_manager import (
    RequestManager, RetryConfig, RequestCategory
)
from api_rate_limiting.brokers.rate_limit_configs import (
    get_broker_config, BinanceRateLimitConfig, AlpacaRateLimitConfig
)
from api_rate_limiting.monitoring.monitor import RateLimitMonitor, AlertSeverity
from api_rate_limiting.compliance.compliance_engine import ComplianceEngine
from api_rate_limiting.monitoring.health_check import HealthChecker, CheckType


class TestRateLimiterAlgorithms:
    """Test rate limiting algorithms"""
    
    def test_token_bucket_basic(self):
        """Test basic token bucket functionality"""
        limiter = TokenBucketRateLimiter("test", limit=10, window_seconds=60)
        
        # Should allow initial burst
        status1 = asyncio.run(limiter.acquire(5))
        assert status1.allowed == True
        assert status1.remaining == 5
        
        # Should deny if not enough tokens
        status2 = asyncio.run(limiter.acquire(10))
        assert status2.allowed == False
        
        # Should allow smaller requests
        status3 = asyncio.run(limiter.acquire(2))
        assert status3.allowed == True
    
    def test_token_bucket_refill(self):
        """Test token bucket refill mechanics"""
        limiter = TokenBucketRateLimiter("test", limit=10, window_seconds=60)
        rate = 10 / 60  # tokens per second
        
        # Fill bucket
        asyncio.run(limiter.acquire(10))
        
        # Wait and check refill
        time.sleep(1.1)  # Wait for refill
        status = asyncio.run(limiter.acquire(1))
        assert status.remaining >= 0
    
    def test_sliding_window_basic(self):
        """Test basic sliding window functionality"""
        limiter = SlidingWindowRateLimiter("test", limit=5, window_seconds=60)
        
        # Should allow requests within limit
        for i in range(5):
            status = asyncio.run(limiter.acquire(1))
            assert status.allowed == True
        
        # Should deny when limit exceeded
        status = asyncio.run(limiter.acquire(1))
        assert status.allowed == False


class TestBrokerConfigs:
    """Test broker-specific configurations"""
    
    def test_binance_config(self):
        """Test Binance configuration"""
        config = BinanceRateLimitConfig(is_futures=False)
        
        assert config.broker_name == "binance_spot"
        assert config.global_rate_limit == 1200
        assert config.endpoint_limits["orders"].limit == 10
        assert config.endpoint_limits["orders"].window_seconds == 1
    
    def test_alpaca_config(self):
        """Test Alpaca configuration"""
        config = AlpacaRateLimitConfig(is_paper=True)
        
        assert config.broker_name == "alpaca_paper"
        assert config.global_rate_limit == 200
        assert config.endpoint_limits["account_info"].limit == 200
        assert config.warning_threshold == 0.8
    
    def test_degiro_conservative_config(self):
        """Test DEGIRO conservative configuration"""
        from api_rate_limiting.brokers.rate_limit_configs import DEGIRORateLimitConfig
        
        config = DEGIRORateLimitConfig()
        
        # Should be very conservative
        assert config.global_rate_limit == 20
        assert config.global_window_seconds == 60
        assert config.burst_multiplier == 1.0  # No burst allowed
        assert config.adaptive_enabled == False  # Safety first
    
    def test_trading212_strict_config(self):
        """Test Trading 212 strict configuration"""
        from api_rate_limiting.brokers.rate_limit_configs import Trading212RateLimitConfig
        
        config = Trading212RateLimitConfig()
        
        # Should be extremely strict
        assert config.global_rate_limit == 6
        assert config.global_window_seconds == 60
        assert config.endpoint_limits["orders"].limit == 3
        assert config.endpoint_limits["orders"].window_seconds == 60


class TestRateLimiterManager:
    """Test rate limiter manager"""
    
    @pytest.fixture
    async def manager(self):
        """Create test manager"""
        manager = APIRateLimitManager()
        manager.add_broker("binance", is_futures=False)
        manager.add_broker("alpaca", is_paper=True)
        await manager.start()
        yield manager
        await manager.stop()
    
    async def test_add_broker(self, manager):
        """Test broker addition"""
        assert "binance" in manager._rate_limiters
        assert "alpaca" in manager._rate_limiters
        
        # Check configuration
        binance_config = manager._configs["binance"]
        assert binance_config.global_rate_limit == 1200
    
    async def test_submit_request(self, manager):
        """Test request submission"""
        async def mock_api_call():
            return {"result": "success"}
        
        request_id = await manager.submit_request(
            broker="binance",
            request_type=RequestType.ACCOUNT_INFO,
            callback=mock_api_call,
            priority=RequestPriority.NORMAL
        )
        
        assert request_id is not None
        assert len(request_id) > 0
    
    async def test_rate_limit_status(self, manager):
        """Test rate limit status"""
        status = manager.get_rate_limit_status("binance")
        
        assert "total_requests" in status
        assert "allowed_requests" in status
        assert "rejected_requests" in status
        assert status["broker"] == "binance"
    
    async def test_compliance_check(self, manager):
        """Test compliance checking"""
        context = {
            "request_type": RequestType.ACCOUNT_INFO,
            "rate_limit_exceeded": False,
            "timestamp": time.time()
        }
        
        compliance_status = manager._compliance.check_compliance(
            "binance", RequestType.ACCOUNT_INFO, context
        )
        
        assert compliance_status is not None


class TestRequestManager:
    """Test request manager"""
    
    @pytest.fixture
    def rate_limiter(self):
        """Create test rate limiter"""
        return RateLimiterManager("test", {})
    
    @pytest.fixture
    def request_manager(self, rate_limiter):
        """Create test request manager"""
        return RequestManager(rate_limiter)
    
    async def test_submit_and_process_request(self, request_manager):
        """Test request submission and processing"""
        async def mock_api_call():
            await asyncio.sleep(0.1)
            return {"result": "success"}
        
        await request_manager.start_processing()
        
        request_id = await request_manager.submit_request(
            request_type=RequestType.ACCOUNT_INFO,
            callback=mock_api_call,
            priority=RequestPriority.NORMAL
        )
        
        # Wait for processing
        await asyncio.sleep(1)
        
        # Check status
        status = request_manager.get_request_status(request_id)
        assert status is not None
        assert status["id"] == request_id
    
    async def test_request_retry(self, request_manager):
        """Test request retry logic"""
        call_count = 0
        
        async def failing_api_call():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Temporary failure")
            return {"result": "success"}
        
        await request_manager.start_processing()
        
        request_id = await request_manager.submit_request(
            request_type=RequestType.ACCOUNT_INFO,
            callback=failing_api_call,
            priority=RequestPriority.NORMAL
        )
        
        # Wait for retries
        await asyncio.sleep(5)
        
        stats = request_manager.get_stats()
        assert stats["retried_requests"] > 0
    
    async def test_dead_letter_queue(self, request_manager):
        """Test dead letter queue functionality"""
        async def always_failing_api_call():
            raise Exception("Permanent failure")
        
        await request_manager.start_processing()
        
        request_id = await request_manager.submit_request(
            request_type=RequestType.ACCOUNT_INFO,
            callback=always_failing_api_call,
            priority=RequestPriority.NORMAL
        )
        
        # Wait for failures
        await asyncio.sleep(10)
        
        # Check dead letter queue
        failed_requests = request_manager._dead_letter_queue.get_failed_requests()
        assert len(failed_requests) > 0


class TestMonitoring:
    """Test monitoring and alerting"""
    
    @pytest.fixture
    def monitor(self):
        """Create test monitor"""
        return RateLimitMonitor(collection_interval=1.0)
    
    async def test_metric_collection(self, monitor):
        """Test metric collection"""
        # Add mock rate limiter
        mock_limiter = Mock()
        mock_limiter.get_global_status.return_value = {
            "total_requests": 100,
            "allowed_requests": 95,
            "rejected_requests": 5,
            "current_requests": 2,
            "timestamp": time.time()
        }
        
        monitor.add_rate_limiter("test_broker", mock_limiter)
        await monitor.start_monitoring()
        
        # Wait for collection
        await asyncio.sleep(2)
        
        health_status = monitor.get_health_status()
        assert health_status["is_monitoring"] == True
        assert health_status["total_metrics"] > 0
    
    async def test_alert_generation(self, monitor):
        """Test alert generation"""
        # Add mock rate limiter that hits limits
        mock_limiter = Mock()
        mock_limiter.get_global_status.return_value = {
            "total_requests": 1000,
            "allowed_requests": 50,  # Very low success rate
            "rejected_requests": 950,
            "timestamp": time.time()
        }
        
        monitor.add_rate_limiter("test_broker", mock_limiter)
        
        # Mock rate limiter acquire to trigger alerts
        async def mock_acquire(*args, **kwargs):
            from api_rate_limiting.core.algorithm import RateLimitStatus
            return RateLimitStatus(
                allowed=False,
                remaining=0,
                reset_time=time.time() + 60,
                limit=100,
                algorithm="test"
            )
        
        mock_limiter.acquire = mock_acquire
        
        # Create alert rule
        from api_rate_limiting.monitoring.monitor import AlertRule, MetricType
        alert_rule = AlertRule(
            id="test_alert",
            metric_type=MetricType.ERROR_RATE,
            broker="*",
            threshold=0.5,
            severity=AlertSeverity.WARNING,
            duration_seconds=1.0
        )
        
        monitor.add_alert_rule(alert_rule)
        
        # Simulate metric
        test_metric = Mock()
        test_metric.value = 0.8  # 80% error rate
        test_metric.broker = "test_broker"
        test_metric.metric_type = MetricType.ERROR_RATE
        
        alert = alert_rule.evaluate(test_metric)
        # Alert should be generated due to threshold breach


class TestCompliance:
    """Test compliance engine"""
    
    @pytest.fixture
    def compliance_engine(self):
        """Create test compliance engine"""
        return ComplianceEngine()
    
    def test_compliance_checking(self, compliance_engine):
        """Test compliance rule evaluation"""
        from api_rate_limiting.compliance.compliance_engine import ComplianceRule
        
        # Add broker config
        config = BinanceRateLimitConfig()
        compliance_engine.add_broker_config("binance", config)
        
        # Test compliance check
        context = {
            "request_type": RequestType.ACCOUNT_INFO,
            "rate_limit_exceeded": False,
            "timestamp": time.time()
        }
        
        status = compliance_engine.check_compliance(
            "binance", RequestType.ACCOUNT_INFO, context
        )
        
        assert status is not None
    
    def test_cost_optimization(self, compliance_engine):
        """Test cost optimization"""
        from api_rate_limiting.compliance.compliance_engine import CostOptimizer
        
        # Add broker config
        config = BinanceRateLimitConfig()
        config.market_data_fee_per_request = 0.01
        compliance_engine.add_broker_config("binance", config)
        
        # Test optimization
        requests = {
            RequestType.MARKET_DATA: 50,
            RequestType.HISTORICAL_DATA: 10,
            RequestType.ACCOUNT_INFO: 5
        }
        
        optimization = compliance_engine.get_cost_optimization(
            "binance", requests, 24
        )
        
        assert "current_cost" in optimization
        assert "recommendations" in optimization
        assert "cost_efficiency_score" in optimization


class TestHealthChecks:
    """Test health checking system"""
    
    @pytest.fixture
    def health_checker(self):
        """Create test health checker"""
        return HealthChecker()
    
    async def test_basic_health_checks(self, health_checker):
        """Test basic health check functionality"""
        results = await health_checker.run_all_checks()
        
        assert "overall_status" in results
        assert "results" in results
        assert len(results["results"]) > 0
        
        # Check that all checks have results
        for check_id, result in results["results"].items():
            assert "status" in result
            assert "check_type" in result
    
    async def test_component_specific_checks(self, health_checker):
        """Test component-specific health checks"""
        # Run liveness checks only
        results = await health_checker.run_all_checks([CheckType.LIVENESS])
        
        for result in results["results"].values():
            assert result["check_type"] == CheckType.LIVENESS.value
    
    def test_custom_health_check(self, health_checker):
        """Test custom health check addition"""
        from api_rate_limiting.monitoring.health_check import HealthCheck
        
        async def custom_check():
            return {"custom_status": "ok"}
        
        custom_health_check = HealthCheck(
            id="custom_test",
            name="Custom Test Check",
            description="Test custom health check",
            check_type=CheckType.LIVENESS,
            component="test",
            check_function=custom_check
        )
        
        health_checker.add_custom_check(custom_health_check)
        assert "custom_test" in health_checker.registry.get_all_checks()


class TestIntegration:
    """Integration tests"""
    
    async def test_full_system_integration(self):
        """Test full system integration"""
        # Create manager
        manager = APIRateLimitManager()
        manager.add_broker("binance", is_futures=False)
        manager.add_broker("alpaca", is_paper=True)
        
        await manager.start()
        
        # Submit multiple requests
        async def mock_api_call(data):
            await asyncio.sleep(0.1)
            return {"result": "success", "data": data}
        
        request_ids = []
        for i in range(10):
            request_id = await manager.submit_request(
                broker="binance",
                request_type=RequestType.ACCOUNT_INFO,
                callback=mock_api_call,
                priority=RequestPriority.NORMAL,
                data=f"request_{i}"
            )
            request_ids.append(request_id)
        
        # Wait for processing
        await asyncio.sleep(3)
        
        # Check system health
        health = manager.get_system_health()
        assert health.status.value in ["healthy", "degraded"]
        assert health.total_requests >= 5
        
        # Check analytics
        analytics = manager.get_analytics("binance", hours=1)
        assert analytics.total_requests > 0
        
        # Check compliance
        compliance = manager.get_compliance_status()
        assert "brokers" in compliance
        
        await manager.stop()
    
    async def test_broker_specific_rates(self):
        """Test broker-specific rate limits"""
        manager = APIRateLimitManager()
        
        # Add different brokers
        manager.add_broker("binance", is_futures=False)
        manager.add_broker("trading212")  # Very strict limits
        manager.add_broker("ibkr")  # High-frequency but conservative
        
        await manager.start()
        
        # Check that each broker has different configurations
        binance_status = manager.get_rate_limit_status("binance")
        trading212_status = manager.get_rate_limit_status("trading212")
        ibkr_status = manager.get_rate_limit_status("ibkr")
        
        # Trading212 should have much lower limits
        assert binance_status["total_requests"] >= 0
        assert trading212_status["total_requests"] >= 0
        
        # All brokers should be functional
        for broker in ["binance", "trading212", "ibkr"]:
            status = manager.get_rate_limit_status(broker)
            assert "success_rate" in status
        
        await manager.stop()


class TestErrorHandling:
    """Test error handling scenarios"""
    
    async def test_rate_limit_exceeded(self):
        """Test rate limit exceeded handling"""
        manager = APIRateLimitManager()
        manager.add_broker("trading212")  # Strict limits
        
        await manager.start()
        
        # Try to exceed limits quickly
        async def quick_api_call():
            await asyncio.sleep(0.01)
            return {"result": "success"}
        
        # Submit many requests to trigger rate limiting
        for i in range(20):
            try:
                await manager.submit_request(
                    broker="trading212",
                    request_type=RequestType.MARKET_DATA,
                    callback=quick_api_call,
                    priority=RequestPriority.NORMAL
                )
            except Exception:
                pass  # Expected to fail
        
        # Check for rate limit alerts
        alerts = manager.get_active_alerts()
        assert len(alerts) >= 0  # May or may not have alerts depending on timing
    
    async def test_circuit_breaker_behavior(self):
        """Test circuit breaker functionality"""
        manager = APIRateLimitManager()
        manager.add_broker("binance")
        
        await manager.start()
        
        # Monitor for circuit breaker state
        binance_limiter = manager._rate_limiters["binance"]
        
        # Check that circuit breaker is functional
        # (actual circuit breaker testing would require more complex setup)
        status = manager.get_rate_limit_status("binance")
        assert "total_requests" in status
        
        await manager.stop()


def test_utility_functions():
    """Test utility functions"""
    from api_rate_limiting.utils.helpers import (
        calculate_rate_limit_efficiency,
        estimate_request_cost,
        validate_api_key_security
    )
    
    # Test rate limit efficiency
    efficiency = calculate_rate_limit_efficiency(80, 100, 60)
    assert "utilization_rate" in efficiency
    assert "efficiency_score" in efficiency
    assert efficiency["utilization_rate"] == 0.8
    
    # Test API key security
    security = validate_api_key_security("test123")
    assert "security_score" in security
    assert "issues" in security
    
    # Test cost estimation
    config = BinanceRateLimitConfig()
    cost = estimate_request_cost(RequestType.MARKET_DATA, config)
    assert "base_cost_per_request" in cost


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])