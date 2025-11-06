"""
Integration Example: Enhanced Broker with Rate Limiting

This example shows how to integrate the API rate limiting system
with existing broker implementations.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime

# Import rate limiting components
from api_rate_limiting.manager import APIRateLimitManager
from api_rate_limiting.core.rate_limiter import RequestType, RequestPriority
from api_rate_limiting.core.request_manager import RetryConfig
from api_rate_limiting.monitoring.health_check import HealthChecker

# Import base broker interface (if available)
# from brokers.base import BaseBroker, BrokerConfig


class RateLimitedBroker:
    """
    Enhanced broker wrapper with comprehensive rate limiting
    
    This wrapper can be used with any broker implementation to add
    rate limiting, request management, monitoring, and compliance.
    """
    
    def __init__(self, broker_name: str, original_broker, config: Dict[str, Any] = None):
        self.broker_name = broker_name
        self.original_broker = original_broker
        self.config = config or {}
        
        # Create rate limiting manager
        self.rate_limit_manager = APIRateLimitManager(self.config)
        
        # Add this broker to rate limiting
        is_futures = self.config.get("is_futures", False)
        is_paper = self.config.get("is_paper", True)
        self.rate_limit_manager.add_broker(broker_name, is_futures=is_futures, is_paper=is_paper)
        
        # Create health checker
        self.health_checker = HealthChecker(self.rate_limit_manager)
        
        # Start rate limiting system
        self._rate_limiting_started = False
    
    async def start_rate_limiting(self):
        """Start the rate limiting system"""
        if not self._rate_limiting_started:
            await self.rate_limit_manager.start()
            self.health_checker.start_monitoring()
            self._rate_limiting_started = True
    
    async def stop_rate_limiting(self):
        """Stop the rate limiting system"""
        if self._rate_limiting_started:
            await self.rate_limit_manager.stop()
            self.health_checker.stop_monitoring()
            self._rate_limiting_started = False
    
    async def get_account(self) -> Dict[str, Any]:
        """Get account information with rate limiting"""
        async with self.rate_limit_manager.transaction(
            self.broker_name, RequestType.ACCOUNT_INFO, RequestPriority.HIGH
        ) as txn:
            # Original broker call
            account_info = await self.original_broker.get_account()
            txn.result = account_info
            return account_info
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get positions with rate limiting"""
        async with self.rate_limit_manager.transaction(
            self.broker_name, RequestType.POSITION_QUERY, RequestPriority.HIGH
        ) as txn:
            positions = await self.original_broker.get_positions()
            txn.result = positions
            return positions
    
    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Place order with rate limiting"""
        async with self.rate_limit_manager.transaction(
            self.broker_name, RequestType.ORDER_PLACE, RequestPriority.CRITICAL
        ) as txn:
            order_result = await self.original_broker.place_order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                limit_price=limit_price,
                stop_price=stop_price,
                **kwargs
            )
            txn.result = order_result
            return order_result
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order with rate limiting"""
        async with self.rate_limit_manager.transaction(
            self.broker_name, RequestType.ORDER_CANCEL, RequestPriority.HIGH
        ) as txn:
            result = await self.original_broker.cancel_order(order_id)
            txn.result = result
            return result
    
    async def get_orders(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get orders with rate limiting"""
        async with self.rate_limit_manager.transaction(
            self.broker_name, RequestType.ORDER_QUERY, RequestPriority.NORMAL
        ) as txn:
            orders = await self.original_broker.get_orders(status)
            txn.result = orders
            return orders
    
    async def get_market_data(
        self,
        symbol: str,
        timeframe: str = "1d",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get market data with rate limiting"""
        async with self.rate_limit_manager.transaction(
            self.broker_name, RequestType.HISTORICAL_DATA, RequestPriority.NORMAL
        ) as txn:
            market_data = await self.original_broker.get_market_data(
                symbol=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                limit=limit
            )
            txn.result = market_data
            return market_data
    
    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time quote with rate limiting"""
        async with self.rate_limit_manager.transaction(
            self.broker_name, RequestType.MARKET_DATA, RequestPriority.NORMAL
        ) as txn:
            quote = await self.original_broker.get_quote(symbol)
            txn.result = quote
            return quote
    
    # Enhanced methods with rate limiting
    
    async def batch_get_account_info(self) -> Dict[str, Any]:
        """Get comprehensive account information"""
        async def get_comprehensive_account():
            # Combine multiple API calls
            account = await self.original_broker.get_account()
            positions = await self.original_broker.get_positions()
            
            return {
                "account": account,
                "positions": positions,
                "timestamp": datetime.utcnow()
            }
        
        # Submit batch request
        request_id = await self.rate_limit_manager.submit_request(
            broker=self.broker_name,
            request_type=RequestType.ACCOUNT_INFO,
            callback=get_comprehensive_account,
            priority=RequestPriority.HIGH
        )
        
        # Wait for completion (in production, you'd implement async response handling)
        await asyncio.sleep(2)
        
        return self.rate_limit_manager._request_managers[self.broker_name].get_request_status(request_id)
    
    async def get_multiple_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get quotes for multiple symbols efficiently"""
        async def get_batch_quotes():
            quotes = {}
            for symbol in symbols:
                try:
                    quote = await self.original_broker.get_quote(symbol)
                    quotes[symbol] = quote
                except Exception as e:
                    quotes[symbol] = {"error": str(e)}
            return quotes
        
        # Submit batch request with batching
        request_id = await self.rate_limit_manager.submit_request(
            broker=self.broker_name,
            request_type=RequestType.MARKET_DATA,
            callback=get_batch_quotes,
            priority=RequestPriority.NORMAL,
            symbols=symbols
        )
        
        # Wait for completion
        await asyncio.sleep(1)
        
        return self.rate_limit_manager._request_managers[self.broker_name].get_request_status(request_id)
    
    # Monitoring and health methods
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status"""
        return self.rate_limit_manager.get_rate_limit_status(self.broker_name)
    
    def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific request"""
        return self.rate_limit_manager.get_request_status(self.broker_name, request_id)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get enhanced system health including original broker"""
        rate_limit_health = self.rate_limit_manager.get_system_health()
        
        # Add original broker health (if supported)
        original_health = {}
        if hasattr(self.original_broker, 'health_check'):
            try:
                original_health = asyncio.run(self.original_broker.health_check())
            except Exception as e:
                original_health = {"error": str(e)}
        
        return {
            "broker_name": self.broker_name,
            "rate_limiting_system": rate_limit_health.to_dict(),
            "original_broker": original_health,
            "integration_status": "active" if self._rate_limiting_started else "inactive"
        }
    
    async def run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check"""
        # Run rate limiting health checks
        rate_limit_results = await self.health_checker.run_all_checks()
        
        # Run original broker health check
        original_healthy = True
        original_error = None
        if hasattr(self.original_broker, 'health_check'):
            try:
                original_health = await self.original_broker.health_check()
                original_healthy = original_health.get("is_connected", False)
            except Exception as e:
                original_healthy = False
                original_error = str(e)
        
        # Determine overall health
        if not self._rate_limiting_started:
            overall_status = "inactive"
        elif rate_limit_results["overall_status"] == "critical" or not original_healthy:
            overall_status = "critical"
        elif rate_limit_results["overall_status"] == "degraded":
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        return {
            "overall_status": overall_status,
            "rate_limiting": rate_limit_results,
            "original_broker": {
                "healthy": original_healthy,
                "error": original_error
            },
            "integration": {
                "rate_limiting_active": self._rate_limiting_started,
                "broker_name": self.broker_name
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_cost_optimization(self, hours: int = 24) -> Dict[str, Any]:
        """Get cost optimization recommendations"""
        # Get recent request patterns
        analytics = self.rate_limit_manager.get_analytics(self.broker_name, hours)
        
        request_patterns = {}
        for req_type, count in analytics.requests_by_type.items():
            try:
                request_patterns[RequestType(req_type)] = count
            except ValueError:
                continue
        
        return self.rate_limit_manager.get_cost_optimization(
            self.broker_name, request_patterns, hours
        )
    
    def get_compliance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Get compliance report"""
        compliance = self.rate_limit_manager.get_compliance_status()
        audit_report = self.rate_limit_manager._compliance.get_audit_report(hours)
        
        return {
            "broker": self.broker_name,
            "compliance_status": compliance,
            "audit_report": audit_report,
            "recommendations": self.get_cost_optimization(hours)
        }
    
    # Context manager for lifecycle management
    async def __aenter__(self):
        await self.start_rate_limiting()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop_rate_limiting()


class MockBroker:
    """
    Mock broker for demonstration purposes
    """
    
    def __init__(self, name: str):
        self.name = name
        self._connected = True
        self._call_count = 0
    
    async def connect(self) -> bool:
        await asyncio.sleep(0.1)
        self._connected = True
        return True
    
    async def disconnect(self) -> bool:
        self._connected = False
        return True
    
    async def is_connection_alive(self) -> bool:
        return self._connected
    
    async def get_account(self) -> Dict[str, Any]:
        self._call_count += 1
        await asyncio.sleep(0.1)
        return {
            "account_id": f"{self.name}_123",
            "balance": 1000.0,
            "currency": "USD",
            "equity": 1050.0
        }
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        self._call_count += 1
        await asyncio.sleep(0.15)
        return [
            {"symbol": "BTCUSDT", "quantity": 0.1, "unrealized_pnl": 25.0},
            {"symbol": "ETHUSDT", "quantity": 1.0, "unrealized_pnl": -10.0}
        ]
    
    async def place_order(self, **kwargs) -> Dict[str, Any]:
        self._call_count += 1
        await asyncio.sleep(0.2)
        return {
            "order_id": f"{self.name}_order_{self._call_count}",
            "status": "filled",
            "filled_price": kwargs.get("limit_price", 45000.0),
            **kwargs
        }
    
    async def cancel_order(self, order_id: str) -> bool:
        self._call_count += 1
        await asyncio.sleep(0.1)
        return True
    
    async def get_orders(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        self._call_count += 1
        await asyncio.sleep(0.1)
        return [
            {"order_id": f"{self.name}_order_1", "status": "filled"},
            {"order_id": f"{self.name}_order_2", "status": "pending"}
        ]
    
    async def get_market_data(self, **kwargs) -> List[Dict[str, Any]]:
        self._call_count += 1
        await asyncio.sleep(0.05)
        return [
            {
                "timestamp": datetime.utcnow(),
                "open": 45000.0,
                "high": 45500.0,
                "low": 44800.0,
                "close": 45200.0,
                "volume": 1000.0
            }
        ]
    
    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        self._call_count += 1
        await asyncio.sleep(0.02)
        return {
            "symbol": symbol,
            "bid": 45100.0,
            "ask": 45150.0,
            "last": 45125.0,
            "timestamp": datetime.utcnow()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        return {
            "broker_name": self.name,
            "is_connected": self._connected,
            "call_count": self._call_count,
            "timestamp": datetime.utcnow().isoformat()
        }


async def example_enhanced_broker_usage():
    """
    Example of using the enhanced broker with rate limiting
    """
    print("=== Enhanced Broker with Rate Limiting ===")
    
    # Create original broker
    original_broker = MockBroker("binance_demo")
    
    # Create enhanced broker with rate limiting
    config = {
        "monitoring_interval": 5.0,
        "monitoring_retention_days": 7,
        "is_futures": False,
        "is_paper": True
    }
    
    enhanced_broker = RateLimitedBroker("enhanced_binance", original_broker, config)
    
    # Use as context manager
    async with enhanced_broker:
        print("Enhanced broker started")
        
        # Get comprehensive health check
        health = await enhanced_broker.run_health_check()
        print(f"System Health: {health['overall_status']}")
        
        # Perform various operations
        print("\nPerforming operations...")
        
        # Account info (high priority)
        print("Getting account info...")
        account = await enhanced_broker.get_account()
        print(f"Account balance: {account['balance']}")
        
        # Market data (normal priority)
        print("Getting market data...")
        market_data = await enhanced_broker.get_market_data(symbol="BTCUSDT")
        print(f"Market data entries: {len(market_data)}")
        
        # Place order (critical priority)
        print("Placing order...")
        order = await enhanced_broker.place_order(
            symbol="BTCUSDT",
            side="buy",
            order_type="limit",
            quantity=0.01,
            limit_price=45000.0
        )
        print(f"Order placed: {order['order_id']}")
        
        # Batch operations
        print("Getting batch quotes...")
        quotes = await enhanced_broker.get_multiple_quotes(["BTCUSDT", "ETHUSDT", "ADAUSDT"])
        print(f"Quotes received for {len(quotes)} symbols")
        
        # Get system status
        print("\nSystem Status:")
        status = enhanced_broker.get_system_health()
        print(f"Rate limiting active: {status['integration']['rate_limiting_active']}")
        print(f"Broker calls made: {status['original_broker']['call_count']}")
        
        # Get cost optimization
        print("\nCost Optimization:")
        optimization = enhanced_broker.get_cost_optimization(hours=1)
        print(f"Current cost estimate: ${optimization.get('current_cost', 0):.2f}")
        if optimization.get('recommendations'):
            print("Recommendations:")
            for rec in optimization['recommendations'][:2]:  # Show first 2
                print(f"  - {rec}")
        
        # Get compliance report
        print("\nCompliance Report:")
        compliance = enhanced_broker.get_compliance_report(hours=1)
        print(f"Compliance status: {compliance['compliance_status']}")
        print(f"Total audit events: {compliance['audit_report'].get('total_events', 0)}")
        
        print("\nEnhanced broker operations completed successfully!")


async def example_multi_broker_management():
    """
    Example of managing multiple enhanced brokers
    """
    print("\n=== Multi-Broker Management ===")
    
    # Create multiple brokers
    brokers_config = [
        ("binance", {"is_futures": False, "is_paper": True}),
        ("alpaca", {"is_paper": True}),
        ("trading212", {})  # Uses default conservative config
    ]
    
    enhanced_brokers = {}
    
    for broker_name, config in brokers_config:
        original = MockBroker(broker_name)
        enhanced_brokers[broker_name] = RateLimitedBroker(broker_name, original, config)
    
    # Start all brokers
    print("Starting all enhanced brokers...")
    for broker in enhanced_brokers.values():
        await broker.start_rate_limiting()
    
    print("All brokers started successfully!")
    
    # Monitor all brokers
    print("\nMonitoring all brokers...")
    health_summary = {}
    
    for broker_name, broker in enhanced_brokers.items():
        health = await broker.run_health_check()
        health_summary[broker_name] = health['overall_status']
        print(f"{broker_name}: {health['overall_status']}")
    
    # Check system-wide metrics
    print("\nSystem-wide Metrics:")
    total_requests = 0
    for broker in enhanced_brokers.values():
        status = broker.get_rate_limit_status()
        total_requests += status.get('total_requests', 0)
    
    print(f"Total requests across all brokers: {total_requests}")
    
    # Stop all brokers
    print("\nStopping all brokers...")
    for broker in enhanced_brokers.values():
        await broker.stop_rate_limiting()
    
    print("All brokers stopped successfully!")
    
    return health_summary


if __name__ == "__main__":
    async def main():
        print("API Rate Limiting Integration Examples")
        print("=" * 50)
        
        try:
            await example_enhanced_broker_usage()
            await example_multi_broker_management()
            
            print("\nAll integration examples completed successfully!")
            
        except Exception as e:
            print(f"Example failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Run the examples
    asyncio.run(main())