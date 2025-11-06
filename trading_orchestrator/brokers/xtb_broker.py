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