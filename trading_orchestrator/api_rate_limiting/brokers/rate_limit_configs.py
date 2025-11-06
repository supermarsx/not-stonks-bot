"""
Broker-Specific Rate Limit Configurations

Defines precise rate limiting configurations for each supported broker
based on their official API documentation and recommended usage patterns.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from ..core.rate_limiter import RateLimitRule, RequestType, RequestPriority


class BrokerType(Enum):
    """Supported broker types"""
    BINANCE = "binance"
    ALPACA = "alpaca"
    IBKR = "ibkr"
    DEGIRO = "degiro"
    TRADING212 = "trading212"
    TRADE_REPUBLIC = "trade_republic"


@dataclass
class RateLimitConfig:
    """Base rate limit configuration"""
    broker_name: str
    broker_type: BrokerType
    
    # Global limits
    global_rate_limit: int
    global_window_seconds: float
    
    # Per-endpoint limits
    endpoint_limits: Dict[str, RateLimitRule]
    
    # Special configurations
    burst_multiplier: float = 2.0
    adaptive_enabled: bool = True
    circuit_breaker_enabled: bool = True
    
    # Warnings and safe usage patterns
    warning_threshold: float = 0.8
    critical_threshold: float = 0.95
    
    # Authentication and key rotation
    key_rotation_enabled: bool = False
    authentication_limits: Optional[RateLimitRule] = None
    
    # Market data specific
    market_data_fee_per_request: float = 0.0
    max_market_data_cost_per_hour: float = 0.0
    
    # Order-related limits
    order_creation_limits: RateLimitRule = None
    order_cancellation_limits: RateLimitRule = None
    
    def get_rule_for_request(self, request_type: RequestType) -> Optional[RateLimitRule]:
        """Get rate limit rule for specific request type"""
        # Direct mapping
        if request_type in [rt for rt in self.endpoint_limits.values() if rt.request_type == request_type]:
            for rule in self.endpoint_limits.values():
                if rule.request_type == request_type:
                    return rule
        
        # Fallback rules based on request type
        fallback_mapping = {
            RequestType.ACCOUNT_INFO: "account_info",
            RequestType.ORDER_PLACE: "orders",
            RequestType.ORDER_CANCEL: "orders",
            RequestType.ORDER_QUERY: "orders",
            RequestType.POSITION_QUERY: "positions",
            RequestType.MARKET_DATA: "market_data",
            RequestType.HISTORICAL_DATA: "market_data"
        }
        
        endpoint_name = fallback_mapping.get(request_type)
        if endpoint_name and endpoint_name in self.endpoint_limits:
            return self.endpoint_limits[endpoint_name]
        
        return None
    
    def get_safe_usage_pattern(self) -> Dict[str, Any]:
        """Get recommended safe usage patterns"""
        return {
            "max_concurrent_requests": min(5, self.global_rate_limit // 10),
            "recommended_batch_size": min(10, self.global_rate_limit // 20),
            "monitoring_interval_seconds": 60,
            "rate_limit_warning_at": int(self.global_rate_limit * self.warning_threshold),
            "critical_limit_at": int(self.global_rate_limit * self.critical_threshold)
        }


class BinanceRateLimitConfig(RateLimitConfig):
    """
    Binance API Rate Limits
    
    - Weight-based system (1200 weight/minute)
    - Order rate limits (10 orders/second, 100k orders/day)
    - Historical data limits vary by endpoint
    - Futures and spot have different limits
    
    Note: Based on https://binance-docs.github.io/apidocs/spot/en/#limits
    """
    
    def __init__(self, is_futures: bool = False):
        if is_futures:
            # Binance Futures API limits
            broker_name = "binance_futures"
            broker_type = BrokerType.BINANCE
            global_rate_limit = 1200  # weight/minute
        else:
            # Binance Spot API limits
            broker_name = "binance_spot"
            broker_type = BrokerType.BINANCE
            global_rate_limit = 1200  # weight/minute
        
        # Define endpoint-specific limits
        endpoint_limits = {
            "account_info": RateLimitRule(
                request_type=RequestType.ACCOUNT_INFO,
                limit=1200, window_seconds=60, priority=2
            ),
            "orders": RateLimitRule(
                request_type=RequestType.ORDER_PLACE,
                limit=10, window_seconds=1, priority=1, timeout=10.0
            ),
            "order_query": RateLimitRule(
                request_type=RequestType.ORDER_QUERY,
                limit=600, window_seconds=60, priority=2
            ),
            "positions": RateLimitRule(
                request_type=RequestType.POSITION_QUERY,
                limit=600, window_seconds=60, priority=2
            ),
            "market_data": RateLimitRule(
                request_type=RequestType.MARKET_DATA,
                limit=1200, window_seconds=60, priority=3
            ),
            "klines": RateLimitRule(
                request_type=RequestType.HISTORICAL_DATA,
                limit=200, window_seconds=60, priority=4
            ),
            "depth": RateLimitRule(
                request_type=RequestType.MARKET_DATA,
                limit=1200, window_seconds=60, priority=3
            ),
            "trades": RateLimitRule(
                request_type=RequestType.MARKET_DATA,
                limit=1200, window_seconds=60, priority=3
            ),
            "ticker": RateLimitRule(
                request_type=RequestType.MARKET_DATA,
                limit=1200, window_seconds=60, priority=3
            ),
            "websocket": RateLimitRule(
                request_type=RequestType.WEBSOCKET,
                limit=5, window_seconds=60, priority=5
            ),
            "authentication": RateLimitRule(
                request_type=RequestType.AUTHENTICATION,
                limit=5, window_seconds=300, priority=1
            )
        }
        
        super().__init__(
            broker_name=broker_name,
            broker_type=broker_type,
            global_rate_limit=global_rate_limit,
            global_window_seconds=60.0,
            endpoint_limits=endpoint_limits,
            burst_multiplier=1.5,  # Conservative for Binance
            adaptive_enabled=True,
            circuit_breaker_enabled=True,
            warning_threshold=0.7,  # Conservative warning
            critical_threshold=0.9,
            market_data_fee_per_request=0.0,
            max_market_data_cost_per_hour=0.0
        )
        
        # Special order limits
        self.order_creation_limits = RateLimitRule(
            request_type=RequestType.ORDER_PLACE,
            limit=10, window_seconds=1, priority=1, timeout=5.0
        )
        
        self.order_cancellation_limits = RateLimitRule(
            request_type=RequestType.ORDER_CANCEL,
            limit=100, window_seconds=1, priority=1, timeout=5.0
        )


class AlpacaRateLimitConfig(RateLimitConfig):
    """
    Alpaca API Rate Limits
    
    - 200 requests per minute
    - Market data: 200 requests/minute
    - Trading: 200 requests/minute
    - Paper trading has same limits
    
    Note: Based on https://alpaca.markets/docs/api-documentation/
    """
    
    def __init__(self, is_paper: bool = True):
        broker_name = "alpaca_paper" if is_paper else "alpaca_live"
        broker_type = BrokerType.ALPACA
        
        # Alpaca has consistent limits across endpoints
        endpoint_limits = {
            "account_info": RateLimitRule(
                request_type=RequestType.ACCOUNT_INFO,
                limit=200, window_seconds=60, priority=2
            ),
            "orders": RateLimitRule(
                request_type=RequestType.ORDER_PLACE,
                limit=50, window_seconds=60, priority=1, timeout=10.0
            ),
            "order_query": RateLimitRule(
                request_type=RequestType.ORDER_QUERY,
                limit=200, window_seconds=60, priority=2
            ),
            "positions": RateLimitRule(
                request_type=RequestType.POSITION_QUERY,
                limit=200, window_seconds=60, priority=2
            ),
            "market_data": RateLimitRule(
                request_type=RequestType.MARKET_DATA,
                limit=200, window_seconds=60, priority=3
            ),
            "latest_quote": RateLimitRule(
                request_type=RequestType.MARKET_DATA,
                limit=200, window_seconds=60, priority=3
            ),
            "latest_trade": RateLimitRule(
                request_type=RequestType.MARKET_DATA,
                limit=200, window_seconds=60, priority=3
            ),
            "bars": RateLimitRule(
                request_type=RequestType.HISTORICAL_DATA,
                limit=100, window_seconds=60, priority=4
            ),
            "websocket": RateLimitRule(
                request_type=RequestType.WEBSOCKET,
                limit=10, window_seconds=60, priority=5
            ),
            "authentication": RateLimitRule(
                request_type=RequestType.AUTHENTICATION,
                limit=10, window_seconds=300, priority=1
            )
        }
        
        super().__init__(
            broker_name=broker_name,
            broker_type=broker_type,
            global_rate_limit=200,
            global_window_seconds=60.0,
            endpoint_limits=endpoint_limits,
            burst_multiplier=2.0,
            adaptive_enabled=True,
            circuit_breaker_enabled=True,
            warning_threshold=0.8,
            critical_threshold=0.95,
            authentication_limits=RateLimitRule(
                request_type=RequestType.AUTHENTICATION,
                limit=10, window_seconds=300, priority=1
            )
        )


class IBKRRateLimitConfig(RateLimitConfig):
    """
    Interactive Brokers Rate Limits
    
    - TWS API: 50 requests/second
    - Data requests: 60 requests/second  
    - Order submissions: limited to avoid position limits
    - WebSocket connections: limited
    
    Note: Based on Interactive Brokers API documentation
    """
    
    def __init__(self):
        broker_name = "ibkr"
        broker_type = BrokerType.IBKR
        
        endpoint_limits = {
            "account_info": RateLimitRule(
                request_type=RequestType.ACCOUNT_INFO,
                limit=50, window_seconds=1, priority=2
            ),
            "orders": RateLimitRule(
                request_type=RequestType.ORDER_PLACE,
                limit=20, window_seconds=1, priority=1, timeout=15.0
            ),
            "order_query": RateLimitRule(
                request_type=RequestType.ORDER_QUERY,
                limit=50, window_seconds=1, priority=2
            ),
            "positions": RateLimitRule(
                request_type=RequestType.POSITION_QUERY,
                limit=30, window_seconds=1, priority=2
            ),
            "market_data": RateLimitRule(
                request_type=RequestType.MARKET_DATA,
                limit=60, window_seconds=1, priority=3
            ),
            "historical_data": RateLimitRule(
                request_type=RequestType.HISTORICAL_DATA,
                limit=30, window_seconds=1, priority=4
            ),
            "websocket": RateLimitRule(
                request_type=RequestType.WEBSOCKET,
                limit=3, window_seconds=60, priority=5
            ),
            "authentication": RateLimitRule(
                request_type=RequestType.AUTHENTICATION,
                limit=5, window_seconds=300, priority=1
            )
        }
        
        super().__init__(
            broker_name=broker_name,
            broker_type=broker_type,
            global_rate_limit=50,
            global_window_seconds=1.0,  # Very short window
            endpoint_limits=endpoint_limits,
            burst_multiplier=1.2,  # Conservative burst limit
            adaptive_enabled=True,
            circuit_breaker_enabled=True,
            warning_threshold=0.7,  # Very conservative
            critical_threshold=0.85,
            authentication_limits=RateLimitRule(
                request_type=RequestType.AUTHENTICATION,
                limit=5, window_seconds=300, priority=1
            )
        )


class DEGIRORateLimitConfig(RateLimitConfig):
    """
    DEGIRO Rate Limits (Unofficial API)
    
    - Conservative limits to avoid detection
    - Very restrictive to prevent account restrictions
    - No official rate limits published
    
    Note: Based on community research and testing
    """
    
    def __init__(self):
        broker_name = "degiro"
        broker_type = BrokerType.DEGIRO
        
        # Very conservative limits
        endpoint_limits = {
            "account_info": RateLimitRule(
                request_type=RequestType.ACCOUNT_INFO,
                limit=10, window_seconds=60, priority=2
            ),
            "orders": RateLimitRule(
                request_type=RequestType.ORDER_PLACE,
                limit=5, window_seconds=60, priority=1, timeout=30.0
            ),
            "order_query": RateLimitRule(
                request_type=RequestType.ORDER_QUERY,
                limit=20, window_seconds=60, priority=2
            ),
            "positions": RateLimitRule(
                request_type=RequestType.POSITION_QUERY,
                limit=15, window_seconds=60, priority=2
            ),
            "market_data": RateLimitRule(
                request_type=RequestType.MARKET_DATA,
                limit=30, window_seconds=60, priority=3
            ),
            "historical_data": RateLimitRule(
                request_type=RequestType.HISTORICAL_DATA,
                limit=5, window_seconds=300, priority=4
            ),
            "websocket": RateLimitRule(
                request_type=RequestType.WEBSOCKET,
                limit=2, window_seconds=300, priority=5
            ),
            "authentication": RateLimitRule(
                request_type=RequestType.AUTHENTICATION,
                limit=3, window_seconds=300, priority=1
            )
        }
        
        super().__init__(
            broker_name=broker_name,
            broker_type=broker_type,
            global_rate_limit=20,  # Very conservative global limit
            global_window_seconds=60.0,
            endpoint_limits=endpoint_limits,
            burst_multiplier=1.0,  # No burst allowed
            adaptive_enabled=False,  # Disable adaptation for safety
            circuit_breaker_enabled=True,
            warning_threshold=0.6,  # Very conservative
            critical_threshold=0.8,
            authentication_limits=RateLimitRule(
                request_type=RequestType.AUTHENTICATION,
                limit=3, window_seconds=300, priority=1
            )
        )


class Trading212RateLimitConfig(RateLimitConfig):
    """
    Trading 212 Rate Limits
    
    - Very strict: 1-6 requests per minute
    - Order limits even more restrictive
    - Account restrictions if limits exceeded
    
    Note: Based on https://helpcentre.trading212.com/hc/en-us/articles/360007273797-Trading212-API
    """
    
    def __init__(self):
        broker_name = "trading212"
        broker_type = BrokerType.TRADING212
        
        # Extremely restrictive limits
        endpoint_limits = {
            "account_info": RateLimitRule(
                request_type=RequestType.ACCOUNT_INFO,
                limit=6, window_seconds=60, priority=2, timeout=30.0
            ),
            "orders": RateLimitRule(
                request_type=RequestType.ORDER_PLACE,
                limit=3, window_seconds=60, priority=1, timeout=60.0
            ),
            "order_query": RateLimitRule(
                request_type=RequestType.ORDER_QUERY,
                limit=6, window_seconds=60, priority=2, timeout=30.0
            ),
            "positions": RateLimitRule(
                request_type=RequestType.POSITION_QUERY,
                limit=6, window_seconds=60, priority=2, timeout=30.0
            ),
            "market_data": RateLimitRule(
                request_type=RequestType.MARKET_DATA,
                limit=6, window_seconds=60, priority=3, timeout=30.0
            ),
            "historical_data": RateLimitRule(
                request_type=RequestType.HISTORICAL_DATA,
                limit=2, window_seconds=300, priority=4, timeout=60.0
            ),
            "websocket": RateLimitRule(
                request_type=RequestType.WEBSOCKET,
                limit=1, window_seconds=300, priority=5, timeout=120.0
            ),
            "authentication": RateLimitRule(
                request_type=RequestType.AUTHENTICATION,
                limit=2, window_seconds=300, priority=1, timeout=120.0
            )
        }
        
        super().__init__(
            broker_name=broker_name,
            broker_type=broker_type,
            global_rate_limit=6,  # Very strict global limit
            global_window_seconds=60.0,
            endpoint_limits=endpoint_limits,
            burst_multiplier=1.0,  # No burst
            adaptive_enabled=False,  # Disable adaptation
            circuit_breaker_enabled=True,
            warning_threshold=0.5,  # Very conservative
            critical_threshold=0.7,
            authentication_limits=RateLimitRule(
                request_type=RequestType.AUTHENTICATION,
                limit=2, window_seconds=300, priority=1, timeout=120.0
            )
        )


class TradeRepublicRateLimitConfig(RateLimitConfig):
    """
    Trade Republic Rate Limits (Unofficial API)
    
    - Minimal usage patterns required
    - Conservative limits to avoid detection
    - No public rate limit documentation
    
    Note: Based on community research
    """
    
    def __init__(self):
        broker_name = "trade_republic"
        broker_type = BrokerType.TRADE_REPUBLIC
        
        # Minimal usage patterns
        endpoint_limits = {
            "account_info": RateLimitRule(
                request_type=RequestType.ACCOUNT_INFO,
                limit=5, window_seconds=60, priority=2
            ),
            "orders": RateLimitRule(
                request_type=RequestType.ORDER_PLACE,
                limit=2, window_seconds=60, priority=1, timeout=45.0
            ),
            "order_query": RateLimitRule(
                request_type=RequestType.ORDER_QUERY,
                limit=10, window_seconds=60, priority=2
            ),
            "positions": RateLimitRule(
                request_type=RequestType.POSITION_QUERY,
                limit=5, window_seconds=60, priority=2
            ),
            "market_data": RateLimitRule(
                request_type=RequestType.MARKET_DATA,
                limit=15, window_seconds=60, priority=3
            ),
            "historical_data": RateLimitRule(
                request_type=RequestType.HISTORICAL_DATA,
                limit=3, window_seconds=300, priority=4
            ),
            "websocket": RateLimitRule(
                request_type=RequestType.WEBSOCKET,
                limit=1, window_seconds=300, priority=5
            ),
            "authentication": RateLimitRule(
                request_type=RequestType.AUTHENTICATION,
                limit=2, window_seconds=300, priority=1
            )
        }
        
        super().__init__(
            broker_name=broker_name,
            broker_type=broker_type,
            global_rate_limit=10,  # Very low global limit
            global_window_seconds=60.0,
            endpoint_limits=endpoint_limits,
            burst_multiplier=1.0,
            adaptive_enabled=False,
            circuit_breaker_enabled=True,
            warning_threshold=0.6,
            critical_threshold=0.8,
            authentication_limits=RateLimitRule(
                request_type=RequestType.AUTHENTICATION,
                limit=2, window_seconds=300, priority=1
            )
        )


def get_broker_config(
    broker_name: str, 
    is_futures: bool = False, 
    is_paper: bool = True
) -> RateLimitConfig:
    """
    Get rate limit configuration for specific broker
    
    Args:
        broker_name: Name of the broker
        is_futures: For Binance, whether this is futures API
        is_paper: For Alpaca, whether this is paper trading
        
    Returns:
        RateLimitConfig: Configuration for the broker
    """
    broker_name_lower = broker_name.lower()
    
    if broker_name_lower == "binance":
        return BinanceRateLimitConfig(is_futures=is_futures)
    elif broker_name_lower == "alpaca":
        return AlpacaRateLimitConfig(is_paper=is_paper)
    elif broker_name_lower == "ibkr":
        return IBKRRateLimitConfig()
    elif broker_name_lower == "degiro":
        return DEGIRORateLimitConfig()
    elif broker_name_lower == "trading212":
        return Trading212RateLimitConfig()
    elif broker_name_lower == "trade_republic":
        return TradeRepublicRateLimitConfig()
    else:
        raise ValueError(f"Unknown broker: {broker_name}")


def get_all_broker_configs() -> Dict[str, RateLimitConfig]:
    """Get configurations for all supported brokers"""
    configs = {}
    
    # Binance configurations
    configs["binance_spot"] = BinanceRateLimitConfig(is_futures=False)
    configs["binance_futures"] = BinanceRateLimitConfig(is_futures=True)
    
    # Alpaca configurations
    configs["alpaca_paper"] = AlpacaRateLimitConfig(is_paper=True)
    configs["alpaca_live"] = AlpacaRateLimitConfig(is_paper=False)
    
    # Other brokers
    configs["ibkr"] = IBKRRateLimitConfig()
    configs["degiro"] = DEGIRORateLimitConfig()
    configs["trading212"] = Trading212RateLimitConfig()
    configs["trade_republic"] = TradeRepublicRateLimitConfig()
    
    return configs


def validate_rate_limit_config(config: RateLimitConfig) -> List[str]:
    """Validate rate limit configuration for issues"""
    warnings = []
    
    # Check for reasonable limits
    if config.global_rate_limit < 1:
        warnings.append("Global rate limit is too low (< 1)")
    
    if config.global_window_seconds < 1:
        warnings.append("Global window is too short (< 1 second)")
    
    # Check endpoint limits
    for name, rule in config.endpoint_limits.items():
        if rule.limit < 1:
            warnings.append(f"Endpoint '{name}' limit is too low (< 1)")
        
        if rule.window_seconds < 1:
            warnings.append(f"Endpoint '{name}' window is too short (< 1 second)")
        
        if rule.timeout > 60:
            warnings.append(f"Endpoint '{name}' timeout is very long (> 60 seconds)")
    
    # Check for consistency
    total_endpoint_requests = sum(rule.limit for rule in config.endpoint_limits.values())
    if total_endpoint_requests > config.global_rate_limit * 10:
        warnings.append("Sum of endpoint limits may be too high compared to global limit")
    
    return warnings


# Example usage and testing
if __name__ == "__main__":
    # Test configurations
    configs = get_all_broker_configs()
    
    for name, config in configs.items():
        print(f"\n{name}:")
        print(f"  Global: {config.global_rate_limit} requests per {config.global_window_seconds}s")
        print(f"  Endpoints: {len(config.endpoint_limits)}")
        print(f"  Safe usage: {config.get_safe_usage_pattern()}")
        
        # Validate configuration
        warnings = validate_rate_limit_config(config)
        if warnings:
            print(f"  Warnings: {warnings}")
        else:
            print("  ✓ Configuration looks good")