"""
Example Usage of API Rate Limiting System

Demonstrates how to use the comprehensive API rate limiting and compliance system
across multiple broker integrations.
"""

import asyncio
import time
from typing import Dict, List, Any

# Import the rate limiting components
from api_rate_limiting.manager import APIRateLimitManager
from api_rate_limiting.core.rate_limiter import RequestType, RequestPriority
from api_rate_limiting.core.request_manager import RetryConfig
from api_rate_limiting.brokers.rate_limit_configs import (
    BinanceRateLimitConfig, AlpacaRateLimitConfig
)


async def example_basic_usage():
    """
    Basic usage example - setting up rate limiting for multiple brokers
    """
    print("=== Basic Rate Limiting Setup ===")
    
    # Create rate limit manager with configuration
    config = {
        "monitoring_interval": 10.0,
        "monitoring_retention_days": 30,
        "audit_retention_days": 90
    }
    
    manager = APIRateLimitManager(config)
    
    # Add brokers with different configurations
    manager.add_broker("binance", is_futures=False, is_paper=True)
    manager.add_broker("alpaca", is_paper=True)
    manager.add_broker("trading212")  # Uses default conservative config
    
    # Start the system
    await manager.start()
    
    # Get system health
    health = manager.get_system_health()
    print(f"System Health: {health.status.value}")
    print(f"Active Brokers: {health.active_brokers}")
    print(f"Compliance Score: {health.compliance_score:.1f}")
    
    # Stop the system
    await manager.stop()
    
    print("Basic setup complete!\n")


async def example_request_submission():
    """
    Example of submitting requests with rate limiting
    """
    print("=== Request Submission Example ===")
    
    manager = APIRateLimitManager()
    manager.add_broker("binance", is_futures=False)
    manager.add_broker("alpaca", is_paper=True)
    
    await manager.start()
    
    # Define mock API functions
    async def mock_account_info():
        """Mock account info API call"""
        await asyncio.sleep(0.1)  # Simulate API call
        return {"balance": 1000.0, "currency": "USD"}
    
    async def mock_market_data(symbol: str):
        """Mock market data API call"""
        await asyncio.sleep(0.05)  # Simulate API call
        return {"symbol": symbol, "price": 150.0, "volume": 1000}
    
    async def mock_place_order(symbol: str, side: str, quantity: float):
        """Mock order placement"""
        await asyncio.sleep(0.2)  # Simulate API call
        return {"order_id": "12345", "symbol": symbol, "side": side, "quantity": quantity}
    
    # Submit various requests
    print("Submitting requests...")
    
    # High priority order (should be processed first)
    order_request_id = await manager.submit_request(
        broker="binance",
        request_type=RequestType.ORDER_PLACE,
        callback=mock_place_order,
        priority=RequestPriority.CRITICAL,
        symbol="BTCUSDT",
        side="buy",
        quantity=0.1
    )
    print(f"Submitted order request: {order_request_id}")
    
    # Market data request (lower priority)
    market_data_id = await manager.submit_request(
        broker="binance",
        request_type=RequestType.MARKET_DATA,
        callback=mock_market_data,
        priority=RequestPriority.NORMAL,
        symbol="ETHUSDT"
    )
    print(f"Submitted market data request: {market_data_id}")
    
    # Account info request
    account_id = await manager.submit_request(
        broker="alpaca",
        request_type=RequestType.ACCOUNT_INFO,
        callback=mock_account_info,
        priority=RequestPriority.HIGH
    )
    print(f"Submitted account request: {account_id}")
    
    # Wait for requests to complete
    await asyncio.sleep(5)
    
    # Check request status
    print("\nRequest Status:")
    for broker in ["binance", "alpaca"]:
        status = manager.get_rate_limit_status(broker)
        print(f"{broker}: {status['total_requests']} total, {status['allowed_requests']} allowed")
    
    await manager.stop()
    print("Request submission example complete!\n")


async def example_monitoring_and_alerts():
    """
    Example of monitoring and alert system
    """
    print("=== Monitoring and Alerts Example ===")
    
    manager = APIRateLimitManager({"monitoring_interval": 5.0})
    manager.add_broker("trading212")  # Very strict limits for demo
    
    await manager.start()
    
    # Simulate some API calls that might trigger alerts
    async def api_call_that_might_fail():
        """API call that might hit rate limits"""
        # Simulate random failures
        import random
        if random.random() < 0.1:  # 10% failure rate
            raise Exception("Simulated API failure")
        return {"success": True}
    
    print("Simulating API calls...")
    
    # Submit multiple requests to trigger rate limiting
    for i in range(20):
        try:
            await manager.submit_request(
                broker="trading212",
                request_type=RequestType.MARKET_DATA,
                callback=api_call_that_might_fail,
                priority=RequestPriority.NORMAL,
                symbol=f"TEST{i}"
            )
        except Exception as e:
            print(f"Request {i} failed: {e}")
    
    # Wait for processing and monitoring
    await asyncio.sleep(10)
    
    # Check for alerts
    alerts = manager.get_active_alerts()
    print(f"Active Alerts: {len(alerts)}")
    
    for alert in alerts:
        print(f"- {alert.severity.value}: {alert.title}")
    
    # Get analytics
    analytics = manager.get_analytics("trading212", hours=1)
    print(f"Analytics - Total: {analytics.total_requests}, Success: {analytics.successful_requests}")
    
    # Get compliance status
    compliance = manager.get_compliance_status()
    print(f"Compliance Status: {compliance}")
    
    await manager.stop()
    print("Monitoring example complete!\n")


async def example_cost_optimization():
    """
    Example of cost optimization features
    """
    print("=== Cost Optimization Example ===")
    
    # Create manager with cost-sensitive broker
    manager = APIRateLimitManager()
    
    # Add broker with market data costs
    manager.add_broker("binance")
    
    await manager.start()
    
    # Simulate different types of requests
    request_patterns = {
        RequestType.MARKET_DATA: 50,     # Expensive market data
        RequestType.HISTORICAL_DATA: 10,  # Very expensive historical data
        RequestType.ACCOUNT_INFO: 5,      # Low cost account info
        RequestType.ORDER_PLACE: 2        # Medium cost orders
    }
    
    # Get cost optimization recommendations
    recommendations = manager.get_cost_optimization(
        broker="binance",
        requests=request_patterns,
        time_window_hours=1
    )
    
    print("Cost Optimization Analysis:")
    print(f"Current Cost: ${recommendations.get('current_cost', 0):.2f}")
    print(f"Estimated Savings: ${recommendations.get('estimated_savings', 0):.2f}")
    print(f"Cost Efficiency Score: {recommendations.get('cost_efficiency_score', 0):.1f}/100")
    
    print("\nRecommendations:")
    for rec in recommendations.get('recommendations', []):
        print(f"- {rec}")
    
    print("\nOptimization Opportunities:")
    for opt_name, opt_data in recommendations.get('optimizations', {}).items():
        print(f"- {opt_name}: {opt_data}")
    
    await manager.stop()
    print("Cost optimization example complete!\n")


async def example_compliance_and_audit():
    """
    Example of compliance monitoring and audit logging
    """
    print("=== Compliance and Audit Example ===")
    
    manager = APIRateLimitManager()
    manager.add_broker("ibkr")  # Conservative broker
    
    await manager.start()
    
    # Simulate various API operations
    async def api_operation(operation_type: str, broker: str):
        """Simulate different API operations"""
        await asyncio.sleep(0.1)
        return {"operation": operation_type, "result": "success"}
    
    # Submit requests that will be logged
    print("Performing various API operations...")
    
    operations = [
        ("account_info", "ibkr"),
        ("order_place", "ibkr"),
        ("market_data", "ibkr"),
        ("order_cancel", "ibkr")
    ]
    
    for op_type, broker in operations:
        await manager.submit_request(
            broker=broker,
            request_type=RequestType.ACCOUNT_INFO if op_type == "account_info" else
                        RequestType.ORDER_PLACE if op_type == "order_place" else
                        RequestType.MARKET_DATA if op_type == "market_data" else
                        RequestType.ORDER_CANCEL,
            callback=api_operation,
            priority=RequestPriority.NORMAL,
            operation=op_type,
            broker=broker
        )
    
    await asyncio.sleep(2)  # Let operations complete
    
    # Generate audit report
    audit_report = manager._compliance.get_audit_report(hours=1)
    print(f"\nAudit Report:")
    print(f"Total Events: {audit_report.get('total_events', 0)}")
    print(f"Compliance Violations: {audit_report.get('compliance_violations', 0)}")
    print(f"Rate Limit Exceeded: {audit_report.get('rate_limit_exceeded', 0)}")
    
    # Export compliance data
    manager.export_data("compliance_report.json", hours=1)
    print("Exported compliance data to compliance_report.json")
    
    await manager.stop()
    print("Compliance example complete!\n")


async def example_error_handling_and_retry():
    """
    Example of error handling and retry logic
    """
    print("=== Error Handling and Retry Example ===")
    
    manager = APIRateLimitManager()
    manager.add_broker("alpaca")
    
    await manager.start()
    
    # Simulate unreliable API
    request_count = 0
    
    async def unreliable_api_call():
        """API call that fails intermittently"""
        nonlocal request_count
        request_count += 1
        
        import random
        
        # Fail first few times, then succeed
        if request_count <= 3:
            raise Exception("Simulated API failure")
        
        await asyncio.sleep(0.1)  # Simulate processing time
        return {"success": True, "attempts": request_count}
    
    # Custom retry configuration
    retry_config = RetryConfig(
        max_retries=5,
        initial_delay=0.5,
        max_delay=10.0,
        exponential_base=2.0
    )
    
    print("Submitting request with custom retry config...")
    
    request_id = await manager.submit_request(
        broker="alpaca",
        request_type=RequestType.ACCOUNT_INFO,
        callback=unreliable_api_call,
        retry_config=retry_config
    )
    
    print(f"Request submitted: {request_id}")
    
    # Monitor request status
    for i in range(10):  # Check for 10 seconds
        await asyncio.sleep(1)
        
        status = manager.get_request_status("alpaca", request_id)
        if status:
            print(f"Check {i+1}: {status['status']}, attempts: {status.get('attempts', 0)}")
            
            if status['status'] in ['COMPLETED', 'FAILED']:
                break
    
    # Check final analytics
    analytics = manager.get_analytics("alpaca", hours=1)
    print(f"\nFinal Analytics:")
    print(f"Total Requests: {analytics.total_requests}")
    print(f"Completed: {analytics.successful_requests}")
    print(f"Retried Requests: {analytics.total_requests - analytics.successful_requests}")
    
    await manager.stop()
    print("Error handling example complete!\n")


async def example_health_checks():
    """
    Example of health checking system
    """
    print("=== Health Checks Example ===")
    
    from api_rate_limiting.monitoring.health_check import HealthChecker
    
    manager = APIRateLimitManager()
    manager.add_broker("binance")
    manager.add_broker("alpaca")
    
    await manager.start()
    
    # Create health checker
    health_checker = HealthChecker(manager)
    
    # Run comprehensive health checks
    print("Running health checks...")
    
    health_results = await health_checker.run_all_checks()
    
    print(f"Overall Health Status: {health_results['overall_status']}")
    print(f"Checks Run: {health_results['checks_run']}")
    
    print("\nIndividual Check Results:")
    for check_id, result in health_results['results'].items():
        status = result['status']
        print(f"- {check_id}: {status}")
        if status in ['degraded', 'critical']:
            print(f"  Message: {result.get('message', 'No details')}")
    
    # Get summary report
    summary = health_checker.get_summary_report(hours=1)
    print(f"\nHealth Summary:")
    print(f"Status Summary: {summary.get('status_summary', {})}")
    print(f"Component Summary: {summary.get('component_summary', {})}")
    
    # Export health data
    health_checker.registry.get_history()  # This would export to file in real usage
    print("Health data exported")
    
    await manager.stop()
    print("Health checks example complete!\n")


async def main():
    """
    Run all examples
    """
    print("API Rate Limiting System - Usage Examples")
    print("=" * 50)
    
    try:
        await example_basic_usage()
        await example_request_submission()
        await example_monitoring_and_alerts()
        await example_cost_optimization()
        await example_compliance_and_audit()
        await example_error_handling_and_retry()
        await example_health_checks()
        
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())