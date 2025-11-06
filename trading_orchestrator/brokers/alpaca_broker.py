"""
Alpaca Markets API Integration

Implements broker interface using Alpaca's official REST + WebSocket APIs
Supports US equities, ETFs, and cryptocurrencies with commission-free trading

Key Features:
- REST API for account, orders, positions
- WebSocket streams for real-time updates
- Paper trading environment with realistic simulation
- Fractional share trading (market orders only)
- Multiple time-in-force options
- Real-time market data streaming

Configuration Requirements (in BrokerConfig.config dict):
- base_url: API base URL (auto-set based on is_paper flag)
- feed: Market data feed ('iex' or 'sip', default: 'iex')

Note: Based on research/alpaca/alpaca_api_analysis.md
"""

from brokers.base import (
    BaseBroker, BrokerConfig, AccountInfo, PositionInfo, 
    OrderInfo, MarketDataPoint
)
from typing import List, Dict, Optional, Any, AsyncIterator
from datetime import datetime, timedelta
from decimal import Decimal
import asyncio

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, LimitOrderRequest, StopOrderRequest, 
    StopLimitOrderRequest, GetOrdersRequest
)
from alpaca.trading.enums import OrderSide as AlpacaOrderSide, TimeInForce as AlpacaTimeInForce, OrderType as AlpacaOrderType
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from loguru import logger


class AlpacaBroker(BaseBroker):
    """
    Alpaca Markets API implementation
    
    Provides access to US equities, ETFs, and crypto trading.
    Supports paper trading with realistic order simulation.
    """
    
    def __init__(self, config: BrokerConfig):
        super().__init__(config)
        
        # Extract configuration
        config_dict = config.config or {}
        self.feed = config_dict.get('feed', 'iex')  # iex or sip
        
        # Initialize clients
        self.trading_client: Optional[TradingClient] = None
        self.data_client: Optional[StockHistoricalDataClient] = None
        
        # Determine base URL based on paper/live
        if config.is_paper:
            self.base_url = "https://paper-api.alpaca.markets"
        else:
            self.base_url = "https://api.alpaca.markets"
            
        logger.info(f"Alpaca broker initialized (paper_trading={config.is_paper})")