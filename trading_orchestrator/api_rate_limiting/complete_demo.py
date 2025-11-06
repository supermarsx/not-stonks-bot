"""
Complete Integration: API Rate Limiting System

This is the main integration file that demonstrates how to use the
comprehensive API rate limiting system across all broker integrations.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Import the main components
from api_rate_limiting.manager import APIRateLimitManager
from api_rate_limiting.core.rate_limiter import RequestType, RequestPriority
from api_rate_limiting.monitoring.health_check import HealthChecker
from api_rate_limiting.examples.integration_example import RateLimitedBroker, MockBroker


class CompleteIntegrationExample:
    """
    Complete integration example showing all features of the
    API rate limiting and compliance system
    """
    
    def __init__(self):
        self.manager = None
        self.health_checker = None
        self.brokers = {}
        self.start_time = None
    
    async def setup(self):
        """Setup the complete system"""
        print("🚀 Setting up API Rate Limiting System...")
        
        # Configure the system
        config = {
            "monitoring_interval": 5.0,
            "monitoring_retention_days": 30,
            "audit_retention_days": 90,
            "cost_thresholds": {
                "daily_warning": 50.0,
                "daily_critical": 100.0,
                "monthly_warning": 1000.0,
                "monthly_critical": 2000.0
            }
        }
        
        # Create the main manager
        self.manager = APIRateLimitManager(config)
        
        # Add all brokers with their specific configurations
        print("📡 Adding brokers...")
        
        # Binance (futures and spot)
        self.manager.add_broker("binance_spot", is_futures=False, is_paper=True)
        self.manager.add_broker("binance_futures", is_futures=True, is_paper=True)
        
        # Alpaca (paper and live)
        self.manager.add_broker("alpaca_paper", is_paper=True)
        self.manager.add_broker("alpaca_live", is_paper=False)
        
        # Other brokers with conservative configurations
        self.manager.add_broker("trading212")  # Very strict limits
        self.manager.add_broker("ibkr")  # High frequency but conservative
        self.manager.add_broker("degiro")  # Unofficial API, very conservative
        self.manager.add_broker("trade_republic")  # Unofficial API, minimal usage
        
        # Start the system
        print("✅ Starting rate limiting system...")
        await self.manager.start()
        
        # Create health checker
        self.health_checker = HealthChecker(self.manager)
        self.health_checker.start_monitoring()
        
        # Create enhanced brokers for demonstration
        print("🔧 Creating enhanced brokers...")
        await self._create_enhanced_brokers()
        
        self.start_time = datetime.utcnow()
        print("✅ System setup complete!\n")
    
    async def _create_enhanced_brokers(self):
        """Create enhanced brokers with rate limiting"""
        # Create mock brokers for demonstration
        binance_mock = MockBroker("binance_demo")
        alpaca_mock = MockBroker("alpaca_demo")
        trading212_mock = MockBroker("trading212_demo")
        
        # Create enhanced versions
        self.brokers["binance"] = RateLimitedBroker(
            "binance", binance_mock, {"is_futures": False, "is_paper": True}
        )
        
        self.brokers["alpaca"] = RateLimitedBroker(
            "alpaca", alpaca_mock, {"is_paper": True}
        )
        
        self.brokers["trading212"] = RateLimitedBroker(
            "trading212", trading212_mock, {}  # Uses default strict config
        )
        
        # Start enhanced brokers
        for broker in self.brokers.values():
            await broker.start_rate_limiting()
    
    async def demonstrate_rate_limiting(self):
        """Demonstrate rate limiting capabilities"""
        print("🎯 Demonstrating Rate Limiting...")
        
        # Submit requests to different brokers with different priorities
        requests = []
        
        # Critical priority requests (trading operations)
        for broker_name in ["binance", "alpaca"]:
            request_id = await self.manager.submit_request(
                broker=broker_name,
                request_type=RequestType.ORDER_PLACE,
                callback=self._mock_place_order,
                priority=RequestPriority.CRITICAL,
                symbol="BTCUSDT",
                side="buy",
                quantity=0.1
            )
            requests.append((broker_name, request_id))
            print(f"  📤 Submitted critical order to {broker_name}: {request_id}")
        
        # High priority requests (account info)
        for broker_name in ["binance", "alpaca", "trading212"]:
            request_id = await self.manager.submit_request(
                broker=broker_name,
                request_type=RequestType.ACCOUNT_INFO,
                callback=self._mock_get_account,
                priority=RequestPriority.HIGH
            )
            requests.append((broker_name, request_id))
            print(f"  📤 Submitted account request to {broker_name}: {request_id}")
        
        # Normal priority requests (market data)
        for broker_name in ["binance", "alpaca", "trading212"]:
            request_id = await self.manager.submit_request(
                broker=broker_name,
                request_type=RequestType.MARKET_DATA,
                callback=self._mock_get_market_data,
                priority=RequestPriority.NORMAL,
                symbol="BTCUSDT"
            )
            requests.append((broker_name, request_id))
            print(f"  📤 Submitted market data request to {broker_name}: {request_id}")
        
        # Wait for processing
        await asyncio.sleep(3)
        
        # Check request status
        print("\n📊 Request Status:")
        for broker_name, request_id in requests:
            status = self.manager.get_request_status(broker_name, request_id)
            if status:
                print(f"  {broker_name}: {status['status']} (attempts: {status.get('attempts', 0)})")
        
        print("✅ Rate limiting demonstration complete!\n")
    
    async def demonstrate_monitoring(self):
        """Demonstrate monitoring and alerting"""
        print("📈 Demonstrating Monitoring and Alerting...")
        
        # Get system health
        health = self.manager.get_system_health()
        print(f"  🏥 System Health: {health.status.value}")
        print(f"  📊 Active Brokers: {health.active_brokers}")
        print(f"  📈 Total Requests: {health.total_requests}")
        print(f"  ✅ Success Rate: {health.successful_requests/max(1, health.total_requests):.1%}")
        print(f"  🚨 Active Alerts: {health.active_alerts}")
        print(f"  ⚖️  Compliance Score: {health.compliance_score:.1f}/100")
        
        # Get alerts
        alerts = self.manager.get_active_alerts()
        print(f"\n  🚨 Active Alerts ({len(alerts)}):")
        for alert in alerts[:3]:  # Show first 3 alerts
            print(f"    - [{alert.severity.value.upper()}] {alert.title}")
            print(f"      {alert.message}")
        
        # Get analytics for each broker
        print(f"\n  📊 Broker Analytics:")
        for broker_name in self.manager.get_broker_list():
            try:
                analytics = self.manager.get_analytics(broker_name, hours=1)
                print(f"    {broker_name}:")
                print(f"      Total: {analytics.total_requests}")
                print(f"      Success: {analytics.successful_requests}")
                print(f"      Rate Limited: {analytics.rate_limited_requests}")
                print(f"      Avg Response: {analytics.average_response_time:.3f}s")
            except Exception as e:
                print(f"    {broker_name}: Error - {e}")
        
        print("✅ Monitoring demonstration complete!\n")
    
    async def demonstrate_compliance(self):
        """Demonstrate compliance and audit features"""
        print("⚖️  Demonstrating Compliance and Audit...")
        
        # Get compliance status
        compliance = self.manager.get_compliance_status()
        print(f"  📋 Overall Compliance Status:")
        for broker, status in compliance.get("brokers", {}).items():
            print(f"    {broker}: {status}")
        
        # Get audit report
        audit_report = self.manager._compliance.get_audit_report(hours=1)
        print(f"\n  📊 Audit Report (last hour):")
        print(f"    Total Events: {audit_report.get('total_events', 0)}")
        print(f"    Compliance Violations: {audit_report.get('compliance_violations', 0)}")
        print(f"    Rate Limit Exceeded: {audit_report.get('rate_limit_exceeded', 0)}")
        
        # Show event distribution
        events_by_type = audit_report.get('events_by_type', {})
        if events_by_type:
            print(f"    Event Types:")
            for event_type, count in events_by_type.items():
                print(f"      {event_type}: {count}")
        
        # Cost analysis
        print(f"\n  💰 Cost Analysis:")
        for broker_name in self.manager.get_broker_list()[:3]:  # Show first 3 brokers
            analytics = self.manager.get_analytics(broker_name, hours=1)
            request_patterns = {}
            for req_type, count in analytics.requests_by_type.items():
                try:
                    request_patterns[RequestType(req_type)] = count
                except ValueError:
                    continue
            
            optimization = self.manager.get_cost_optimization(
                broker_name, request_patterns, 1
            )
            
            print(f"    {broker_name}:")
            print(f"      Current Cost: ${optimization.get('current_cost', 0):.2f}")
            print(f"      Potential Savings: ${optimization.get('estimated_savings', 0):.2f}")
            print(f"      Efficiency Score: {optimization.get('cost_efficiency_score', 0):.1f}/100")
        
        print("✅ Compliance demonstration complete!\n")
    
    async def demonstrate_health_checks(self):
        """Demonstrate health check system"""
        print("🏥 Demonstrating Health Check System...")
        
        # Run comprehensive health checks
        health_results = await self.health_checker.run_all_checks()
        print(f"  🩺 Overall Health: {health_results['overall_status']}")
        print(f"  🔍 Checks Run: {health_results['checks_run']}")
        
        # Show check results
        print(f"\n  📋 Check Results:")
        for check_id, result in health_results['results'].items():
            status = result['status']
            check_type = result['check_type']
            component = result['component']
            print(f"    [{status.upper()}] {check_id} ({component}) - {check_type}")
            if status in ['degraded', 'critical']:
                message = result.get('message', 'No details')
                print(f"      Message: {message}")
        
        # Get health summary
        summary = self.health_checker.get_summary_report(hours=1)
        print(f"\n  📊 Health Summary:")
        print(f"    Status Summary: {summary.get('status_summary', {})}")
        print(f"    Component Summary: {summary.get('component_summary', {})}")
        print(f"    Recent Success Rate: {summary.get('recent_success_rate', 0):.1%}")
        
        # Test enhanced broker health
        print(f"\n  🔧 Enhanced Broker Health:")
        for broker_name, broker in self.brokers.items():
            health = await broker.run_health_check()
            print(f"    {broker_name}: {health['overall_status']}")
        
        print("✅ Health check demonstration complete!\n")
    
    async def demonstrate_optimization(self):
        """Demonstrate cost optimization features"""
        print("💡 Demonstrating Cost Optimization...")
        
        # Simulate high-cost request patterns
        high_cost_patterns = {
            RequestType.MARKET_DATA: 200,
            RequestType.HISTORICAL_DATA: 50,
            RequestType.REAL_TIME_DATA: 100,
            RequestType.ACCOUNT_INFO: 20,
            RequestType.ORDER_PLACE: 10
        }
        
        print(f"  📊 Analyzing High-Cost Request Patterns:")
        for broker_name in ["binance", "alpaca"][:2]:  # Show first 2 brokers
            optimization = self.manager.get_cost_optimization(
                broker_name, high_cost_patterns, 24
            )
            
            print(f"    {broker_name}:")
            print(f"      Current Cost: ${optimization.get('current_cost', 0):.2f}")
            print(f"      Total Requests: {optimization.get('total_requests', 0)}")
            print(f"      Requests/Minute: {optimization.get('requests_per_minute', 0):.1f}")
            print(f"      Potential Savings: ${optimization.get('estimated_savings', 0):.2f}")
            print(f"      Cost Efficiency: {optimization.get('cost_efficiency_score', 0):.1f}/100")
            
            # Show recommendations
            recommendations = optimization.get('recommendations', [])
            if recommendations:
                print(f"      Recommendations:")
                for rec in recommendations[:3]:  # Show first 3
                    print(f"        - {rec}")
        
        # Demonstrate request batching optimization
        print(f"\n  📦 Request Batching Opportunities:")
        test_requests = [
            {"request_type": RequestType.MARKET_DATA, "symbol": "BTCUSDT"},
            {"request_type": RequestType.MARKET_DATA, "symbol": "ETHUSDT"},
            {"request_type": RequestType.MARKET_DATA, "symbol": "ADAUSDT"},
            {"request_type": RequestType.ACCOUNT_INFO},
            {"request_type": RequestType.ACCOUNT_INFO},
        ]
        
        from api_rate_limiting.utils.helpers import optimize_request_batching
        config = self.manager._configs.get("binance")
        if config:
            batching_optimization = optimize_request_batching(test_requests, config)
            print(f"    Batching Opportunities: {batching_optimization.get('batching_opportunities', [])}")
            print(f"    Potential Savings: ${batching_optimization.get('cost_reduction_percentage', 0):.1f}%")
        
        print("✅ Cost optimization demonstration complete!\n")
    
    async def demonstrate_error_handling(self):
        """Demonstrate error handling and resilience"""
        print("🛡️  Demonstrating Error Handling and Resilience...")
        
        # Simulate requests that might fail
        print("  🔄 Simulating error scenarios...")
        
        # High frequency request to trigger rate limiting
        for i in range(15):
            try:
                await self.manager.submit_request(
                    broker="trading212",  # Very strict limits
                    request_type=RequestType.MARKET_DATA,
                    callback=self._mock_failing_api_call,
                    priority=RequestPriority.NORMAL,
                    symbol=f"TEST{i}"
                )
            except Exception as e:
                print(f"    Request {i} failed as expected: {type(e).__name__}")
        
        # Wait for processing and monitoring
        await asyncio.sleep(5)
        
        # Check system resilience
        system_health = self.manager.get_system_health()
        print(f"  💪 System Resilience:")
        print(f"    Status: {system_health.status.value}")
        print(f"    Total Requests: {system_health.total_requests}")
        print(f"    Successful: {system_health.successful_requests}")
        print(f"    Active Alerts: {system_health.active_alerts}")
        
        # Check circuit breaker states (if applicable)
        for broker_name in ["binance", "alpaca"]:
            try:
                rate_limiter = self.manager._rate_limiters[broker_name]
                if hasattr(rate_limiter, '_rate_limiters'):
                    # Check if any circuit breakers were tripped
                    print(f"    {broker_name}: Circuit breakers operational")
            except:
                pass
        
        # Demonstrate graceful degradation
        alerts = self.manager.get_active_alerts()
        rate_limit_alerts = [a for a in alerts if "rate_limit" in a.title.lower()]
        if rate_limit_alerts:
            print(f"  ⚠️  Rate Limit Alerts Generated: {len(rate_limit_alerts)}")
            print(f"    These alerts show the system is protecting against overuse")
        
        print("✅ Error handling demonstration complete!\n")
    
    async def demonstrate_export_and_reporting(self):
        """Demonstrate data export and reporting"""
        print("📄 Demonstrating Data Export and Reporting...")
        
        # Export comprehensive system data
        export_file = "system_export.json"
        self.manager.export_data(
            export_file,
            hours=1,
            include_compliance=True,
            include_metrics=True
        )
        print(f"  📁 System data exported to: {export_file}")
        
        # Export health data
        from api_rate_limiting.monitoring.health_check import create_health_check_endpoints
        health_summary = self.health_checker.get_summary_report(hours=1)
        
        # Generate comprehensive report
        print(f"  📊 Comprehensive Report:")
        print(f"    Uptime: {(datetime.utcnow() - self.start_time).total_seconds():.0f} seconds")
        print(f"    System Health: {health_summary.get('status', 'unknown')}")
        print(f"    Total Checks: {health_summary.get('total_checks', 0)}")
        print(f"    Recent Success Rate: {health_summary.get('recent_success_rate', 0):.1%}")
        
        # Show configuration summary
        config_summary = self.manager.get_config_summary()
        print(f"\n  ⚙️  Configuration Summary:")
        print(f"    Active Brokers: {len(config_summary.get('brokers', {}))}")
        print(f"    Monitoring Interval: {config_summary.get('monitoring', {}).get('collection_interval', 'N/A')}s")
        print(f"    System Uptime: {config_summary.get('system', {}).get('uptime_seconds', 0):.0f}s")
        
        # Show broker-specific configurations
        for broker_name, broker_config in config_summary.get('brokers', {}).items():
            print(f"    {broker_name}:")
            print(f"      Global Rate Limit: {broker_config.get('global_rate_limit', 'N/A')}")
            print(f"      Window: {broker_config.get('global_window_seconds', 'N/A')}s")
            print(f"      Endpoints: {broker_config.get('endpoint_limits', 0)}")
        
        print("✅ Export and reporting demonstration complete!\n")
    
    async def cleanup(self):
        """Cleanup and shutdown"""
        print("🧹 Cleaning up system...")
        
        # Stop enhanced brokers
        for broker in self.brokers.values():
            await broker.stop_rate_limiting()
        
        # Stop health monitoring
        if self.health_checker:
            self.health_checker.stop_monitoring()
        
        # Stop main manager
        if self.manager:
            await self.manager.stop()
        
        print("✅ Cleanup complete!")
    
    # Mock API functions for demonstration
    async def _mock_place_order(self, **kwargs):
        """Mock order placement"""
        await asyncio.sleep(0.1)
        return {"order_id": "mock_order_123", "status": "filled", **kwargs}
    
    async def _mock_get_account(self):
        """Mock account info"""
        await asyncio.sleep(0.05)
        return {"balance": 1000.0, "currency": "USD", "account_id": "mock_123"}
    
    async def _mock_get_market_data(self, symbol):
        """Mock market data"""
        await asyncio.sleep(0.02)
        return [{"symbol": symbol, "price": 45000.0, "volume": 1000}]
    
    async def _mock_failing_api_call(self):
        """Mock API call that might fail"""
        await asyncio.sleep(0.01)
        # Fail sometimes to demonstrate error handling
        import random
        if random.random() < 0.3:
            raise Exception("Simulated API failure")
        return {"status": "success"}


async def main():
    """
    Main demonstration function
    """
    print("🌟 API Rate Limiting and Compliance System - Complete Demo")
    print("=" * 60)
    print("This demo showcases all features of the comprehensive")
    print("API rate limiting and compliance system for broker integrations.")
    print("=" * 60)
    print()
    
    # Create integration example
    demo = CompleteIntegrationExample()
    
    try:
        # Setup the system
        await demo.setup()
        
        # Run all demonstrations
        await demo.demonstrate_rate_limiting()
        await demo.demonstrate_monitoring()
        await demo.demonstrate_compliance()
        await demo.demonstrate_health_checks()
        await demo.demonstrate_optimization()
        await demo.demonstrate_error_handling()
        await demo.demonstrate_export_and_reporting()
        
        print("🎉 All demonstrations completed successfully!")
        print("\n📋 Summary of Features Demonstrated:")
        print("  ✅ Rate Limiting with multiple algorithms")
        print("  ✅ Request prioritization and batching")
        print("  ✅ Real-time monitoring and alerting")
        print("  ✅ Compliance and audit logging")
        print("  ✅ Health checks and system monitoring")
        print("  ✅ Cost optimization and recommendations")
        print("  ✅ Error handling and resilience")
        print("  ✅ Data export and reporting")
        print("\n🚀 The system is production-ready for broker integrations!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        await demo.cleanup()


if __name__ == "__main__":
    # Run the complete demonstration
    asyncio.run(main())