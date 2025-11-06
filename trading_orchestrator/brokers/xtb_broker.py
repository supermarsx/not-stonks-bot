"""
XTB API Integration (DISCONTINUED March 14, 2025)

⚠️  DEPRECATED - API DISABLED ⚠️
XTB officially discontinued their xAPI service on March 14, 2025.
All API access endpoints have been disabled.

This implementation is provided for:
1. Historical reference only
2. Migration planning to alternative brokers
3. Legacy integration documentation

Migration Required:
- XTB xAPI (xStation5 API) is no longer available
- All connection endpoints are disabled
- Use web platform or mobile app for trading
- Migrate to alternative broker APIs (Alpaca, IBKR, Oanda, etc.)

Historical Features (no longer available):
- JSON over socket and WebSocket connections
- Real-time market data streaming
- Order management (tradeTransaction)
- Account information (getCurrentUserData)
- Historical data (candles, ticks)

Reference: research/xtb/xtb_api_analysis.md
"""

import asyncio
import warnings
from typing import List, Dict, Optional, Any, AsyncIterator
from datetime import datetime, timedelta
from decimal import Decimal
import json

from brokers.base import (
    BaseBroker, BrokerConfig, AccountInfo, PositionInfo, 
    OrderInfo, MarketDataPoint
)
from loguru import logger


# Deprecation warning decorator
def deprecation_warning(func):
    """Decorator to warn about XTB API discontinuation"""
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"XTB API was discontinued on March 14, 2025. "
            f"{func.__name__} will not work. Migrate to alternative broker.",
            DeprecationWarning,
            stacklevel=2
        )
        logger.error(f"⚠️  XTB API DISCONTINUED - {func.__name__} unavailable")
        return func(*args, **kwargs)
    return wrapper


class XTBBroker(BaseBroker):
    """
    XTB API Implementation (DISCONTINUED)
    
    ⚠️  API DISABLED AS OF MARCH 14, 2025 ⚠️
    
    Historical xAPI provided:
    - JSON protocol over socket/WebSocket
    - Real-time market data streams
    - Order management system
    - Account and portfolio data
    
    This integration will not function - provided for reference only.
    Migrate to alternative brokers immediately.
    """
    
    def __init__(self, config: BrokerConfig):
        super().__init__(config)
        
        # Issue critical deprecation warnings
        logger.critical("⚠️  XTB API DISCONTINUATION WARNING ⚠️")
        logger.critical("⚠️  XTB disabled API service on March 14, 2025")
        logger.critical("⚠️  This integration is non-functional")
        logger.critical("⚠️  IMMEDIATE MIGRATION REQUIRED")
        
        # Extract configuration (historical)
        config_dict = config.config or {}
        self.user_id = config_dict.get('user_id', config.api_key)
        self.password = config_dict.get('password', config.api_secret)
        
        # Historical connection info (now disabled)
        self.main_socket_host = config_dict.get('main_host', 'xapi.xtb.com')
        self.main_socket_port = config_dict.get('main_port', 5112)
        self.stream_host = config_dict.get('stream_host', 'xapi.xtb.com')
        self.stream_port = config_dict.get('stream_port', 5113)
        
        # Legacy rate limits
        self.min_command_interval = 0.2  # 200ms minimum between commands
        self.max_payload_size = 1024  # 1KB per command
        
        logger.warning(f"XTB broker initialized (DISCONTINUED, migration_required=True)")
    
    @deprecation_warning
    async def connect(self) -> bool:
        """Connect to XTB API (will fail - API discontinued)"""
        
        logger.critical("❌ CONNECTION IMPOSSIBLE - XTB API DISABLED")
        logger.critical("📍 Migration required to alternative broker")
        
        # Historical connection code (now non-functional)
        try:
            logger.info("Historical: Would connect to XTB xAPI...")
            logger.info(f"Historical: Main socket: {self.main_socket_host}:{self.main_socket_port}")
            logger.info(f"Historical: Stream socket: {self.stream_host}:{self.stream_port}")
            
            # This would have been the connection logic:
            # self.main_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # self.stream_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            
            raise ConnectionError(
                "XTB API service disabled on March 14, 2025. "
                "Migrate to alternative broker immediately."
            )
            
        except Exception as e:
            logger.error(f"XTB connection failed (expected): {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from XTB (cleanup only)"""
        try:
            # Historical cleanup
            self.is_connected = False
            logger.info("Historical: XTB disconnection completed")
            return True
            
        except Exception as e:
            logger.error(f"Disconnect failed: {e}")
            return False
    
    async def is_connection_alive(self) -> bool:
        """Check if XTB connection is alive (always false)"""
        logger.warning("XTB API discontinued - connection impossible")
        return False
    
    @deprecation_warning
    async def get_account(self) -> AccountInfo:
        """Get XTB account information (non-functional)"""
        
        logger.critical("❌ Cannot retrieve account - API disabled")
        
        # Historical implementation would have been:
        # response = await self._send_command("getCurrentUserData")
        # return AccountInfo(...)
        
        raise ConnectionError(
            "XTB API discontinued. Cannot retrieve account information. "
            "Use XTB web platform or migrate to alternative broker."
        )
    
    async def get_positions(self) -> List[PositionInfo]:
        """Get XTB positions (non-functional)"""
        logger.critical("❌ Cannot retrieve positions - API disabled")
        return []
    
    async def get_position(self, symbol: str) -> Optional[PositionInfo]:
        """Get XTB position (non-functional)"""
        logger.critical("❌ Cannot retrieve position - API disabled")
        return None
    
    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "day",
        extended_hours: bool = False,
        **kwargs
    ) -> OrderInfo:
        """⚠️  RISK: Order placement via discontinued API ⚠️"""
        
        logger.critical("❌ ORDER PLACEMENT IMPOSSIBLE - API DISABLED")
        logger.critical("📍 Use XTB web platform for trading")
        logger.critical("📍 Migrate to alternative broker for API trading")
        
        # Historical order placement would have been:
        # order_data = {
        #     "cmd": cmd_code,  # BUY=0, SELL=1, BUY_LIMIT=2, etc.
        #     "type": type_code,  # OPEN=0, PENDING=1, CLOSE=2, etc.
        #     "price": price,
        #     "volume": volume,
        #     "sl": stop_loss,
        #     "tp": take_profit
        # }
        # response = await self._send_command("tradeTransaction", order_data)
        
        raise ConnectionError(
            "XTB API discontinued. Cannot place orders. "
            "Use XTB web platform or migrate to alternative broker API."
        )
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel XTB order (non-functional)"""
        logger.critical("❌ Cannot cancel orders - API disabled")
        return False
    
    async def get_orders(self, status: Optional[str] = None) -> List[OrderInfo]:
        """Get XTB orders (non-functional)"""
        logger.critical("❌ Cannot retrieve orders - API disabled")
        return []
    
    async def get_order(self, order_id: str) -> Optional[OrderInfo]:
        """Get XTB order (non-functional)"""
        logger.critical("❌ Cannot retrieve order - API disabled")
        return None
    
    async def get_market_data(
        self,
        symbol: str,
        timeframe: str = "1d",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 100
    ) -> List[MarketDataPoint]:
        """Get XTB market data (non-functional)"""
        logger.critical("❌ Cannot retrieve market data - API disabled")
        return []
    
    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get XTB quote (non-functional)"""
        logger.critical("❌ Cannot retrieve quote - API disabled")
        return {}
    
    async def stream_quotes(self, symbols: List[str]) -> AsyncIterator[Dict[str, Any]]:
        """XTB quote streaming (discontinued)"""
        logger.critical("❌ Streaming discontinued - API disabled")
        raise NotImplementedError("XTB streaming discontinued March 14, 2025")
    
    async def stream_trades(self, symbols: List[str]) -> AsyncIterator[Dict[str, Any]]:
        """XTB trade streaming (discontinued)"""
        logger.critical("❌ Trading streaming discontinued - API disabled")
        raise NotImplementedError("XTB trading streaming discontinued March 14, 2025")
    
    def get_migration_recommendations(self) -> Dict[str, Any]:
        """Get recommendations for XTB API migration"""
        
        return {
            "status": "DISCONTINUED",
            "discontinuation_date": "2025-03-14",
            "current_action": "MIGRATION_REQUIRED",
            "immediate_steps": [
                "Stop using XTB API immediately",
                "Use XTB web platform for manual trading",
                "Select alternative broker API",
                "Migrate existing integrations"
            ],
            "alternative_brokers": {
                "Alpaca": {
                    "strengths": "US equities, developer-friendly, paper trading",
                    "api_type": "REST + WebSocket",
                    "rate_limits": "~200 RPM",
                    "streaming": True
                },
                "Interactive Brokers": {
                    "strengths": "Broad markets, professional tools",
                    "api_type": "TWS API",
                    "rate_limits": "~3000 RPM", 
                    "streaming": True
                },
                "Oanda": {
                    "strengths": "FX focus, research quality",
                    "api_type": "REST + WebSocket",
                    "rate_limits": "~7200 RPM",
                    "streaming": True
                },
                "Tradier": {
                    "strengths": "Stocks/options, transparent fees",
                    "api_type": "REST",
                    "rate_limits": "~120 RPM",
                    "streaming": True
                }
            },
            "migration_checklist": [
                "Select target broker API",
                "Set up sandbox/demo environment",
                "Migrate authentication logic",
                "Implement streaming equivalent",
                "Map order workflows",
                "Adapt rate limiting",
                "End-to-end testing",
                "Production deployment"
            ],
            "legacy_features": {
                "protocol": "JSON over socket/WebSocket (historical)",
                "real_time_streaming": "Disabled",
                "order_types": "BUY, SELL, BUY_LIMIT, SELL_LIMIT, BUY_STOP, SELL_STOP (historical)",
                "market_data": "Ticks, candles, symbols (historical)",
                "rate_limits": "200ms command cadence, 1KB payload (historical)"
            }
        }
    
    def __repr__(self):
        return f"<XTBBroker(broker='XTB', connected=False, status=DISCONTINUED)>"


# Migration helper functions
def create_migration_plan(current_xtb_workflows: Dict[str, Any]) -> Dict[str, Any]:
    """Create migration plan from XTB workflows"""
    
    return {
        "source": "XTB xAPI (DISCONTINUED)",
        "target_required": True,
        "migration_timeline": "IMMEDIATE",
        "current_workflows": current_xtb_workflows,
        "feature_mapping": {
            "market_data_streaming": {
                "xtb": "WebSocket tick and candle streams",
                "alternatives": "WebSocket streams from Alpaca, IBKR, Oanda"
            },
            "order_management": {
                "xtb": "tradeTransaction with rich order types",
                "alternatives": "REST order APIs with market/limit/stop orders"
            },
            "account_data": {
                "xtb": "getCurrentUserData, getMarginLevel",
                "alternatives": "Account endpoints in REST APIs"
            },
            "rate_limiting": {
                "xtb": "200ms command cadence, payload limits",
                "alternatives": "Per-endpoint rate limits with headers"
            }
        },
        "priority_migration_targets": [
            "Market data streaming replacement",
            "Order execution workflow migration", 
            "Account and portfolio data mapping",
            "Rate limit adaptation"
        ]
    }
