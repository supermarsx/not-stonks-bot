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