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
        
    async def connect(self) -> bool:
        """Connect to Alpaca API"""
        async with self._connection_lock:
            try:
                logger.info(f"Connecting to Alpaca ({self.base_url})...")
                
                # Create trading client
                self.trading_client = TradingClient(
                    api_key=self.config.api_key,
                    secret_key=self.config.api_secret,
                    paper=self.config.is_paper
                )
                
                # Create data client
                self.data_client = StockHistoricalDataClient(
                    api_key=self.config.api_key,
                    secret_key=self.config.api_secret
                )
                
                # Test connection by fetching account
                account = self.trading_client.get_account()
                
                self.is_connected = True
                logger.success(f"Connected to Alpaca (Account: {account.account_number})")
                return True
                
            except Exception as e:
                logger.error(f"Connection failed: {e}")
                self.is_connected = False
                return False
                
    async def disconnect(self) -> bool:
        """Disconnect from Alpaca API"""
        async with self._connection_lock:
            try:
                # Alpaca SDK doesn't require explicit disconnect for REST
                self.trading_client = None
                self.data_client = None
                self.is_connected = False
                logger.info("Disconnected from Alpaca")
                return True
                
            except Exception as e:
                logger.error(f"Disconnection error: {e}")
                return False
                
    async def is_connection_alive(self) -> bool:
        """Check if connection is still alive"""
        if not self.trading_client:
            return False
        try:
            # Quick ping by checking account
            self.trading_client.get_account()
            return True
        except:
            return False
            
    async def get_account(self) -> AccountInfo:
        """Get account information"""
        if not self.trading_client:
            raise ConnectionError("Not connected to Alpaca")
            
        try:
            account = self.trading_client.get_account()
            
            return AccountInfo(
                account_id=account.account_number,
                broker_name=self.broker_name,
                currency=account.currency,
                balance=float(account.equity),
                available_balance=float(account.cash),
                equity=float(account.equity),
                buying_power=float(account.buying_power),
                margin_used=float(account.initial_margin) if account.initial_margin else 0.0,
                margin_available=float(account.buying_power),
                is_pattern_day_trader=account.pattern_day_trader,
                metadata={
                    'portfolio_value': float(account.portfolio_value),
                    'long_market_value': float(account.long_market_value) if account.long_market_value else 0.0,
                    'short_market_value': float(account.short_market_value) if account.short_market_value else 0.0,
                    'daytrading_buying_power': float(account.daytrading_buying_power) if account.daytrading_buying_power else 0.0,
                    'regt_buying_power': float(account.regt_buying_power) if account.regt_buying_power else 0.0,
                    'daytrade_count': account.daytrade_count,
                    'account_blocked': account.account_blocked,
                    'trading_blocked': account.trading_blocked,
                    'status': account.status
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to get account: {e}")
            raise
            
    async def get_positions(self) -> List[PositionInfo]:
        """Get all open positions"""
        if not self.trading_client:
            raise ConnectionError("Not connected to Alpaca")
            
        try:
            positions = self.trading_client.get_all_positions()
            
            position_list = []
            for pos in positions:
                quantity = float(pos.qty)
                
                position_list.append(PositionInfo(
                    symbol=pos.symbol,
                    broker_name=self.broker_name,
                    side='long' if quantity > 0 else 'short',
                    quantity=abs(quantity),
                    avg_entry_price=float(pos.avg_entry_price),
                    current_price=float(pos.current_price),
                    market_value=float(pos.market_value),
                    unrealized_pnl=float(pos.unrealized_pl),
                    unrealized_pnl_percent=float(pos.unrealized_plpc) * 100,
                    cost_basis=float(pos.cost_basis),
                    metadata={
                        'asset_id': pos.asset_id,
                        'exchange': pos.exchange,
                        'asset_class': pos.asset_class,
                        'qty_available': float(pos.qty_available) if pos.qty_available else abs(quantity)
                    }
                ))
                
            return position_list
            
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            raise