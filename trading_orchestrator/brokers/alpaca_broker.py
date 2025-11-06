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
            
    async def get_position(self, symbol: str) -> Optional[PositionInfo]:
        """Get position for specific symbol"""
        if not self.trading_client:
            raise ConnectionError("Not connected to Alpaca")
            
        try:
            pos = self.trading_client.get_open_position(symbol)
            quantity = float(pos.qty)
            
            return PositionInfo(
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
            )
            
        except Exception as e:
            # Position doesn't exist
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
        """Place a new order"""
        if not self.trading_client:
            raise ConnectionError("Not connected to Alpaca")
            
        try:
            # Map side
            alpaca_side = AlpacaOrderSide.BUY if side.lower() == 'buy' else AlpacaOrderSide.SELL
            
            # Map time in force
            tif_map = {
                'day': AlpacaTimeInForce.DAY,
                'gtc': AlpacaTimeInForce.GTC,
                'ioc': AlpacaTimeInForce.IOC,
                'fok': AlpacaTimeInForce.FOK,
                'opg': AlpacaTimeInForce.OPG,  # Market on open
                'cls': AlpacaTimeInForce.CLS   # Market on close
            }
            alpaca_tif = tif_map.get(time_in_force.lower(), AlpacaTimeInForce.DAY)
            
            # Create order request based on type
            order_type_lower = order_type.lower()
            
            if order_type_lower == 'market':
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=alpaca_side,
                    time_in_force=alpaca_tif,
                    extended_hours=extended_hours
                )
            elif order_type_lower == 'limit':
                if not limit_price:
                    raise ValueError("Limit price required for limit orders")
                order_request = LimitOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=alpaca_side,
                    time_in_force=alpaca_tif,
                    limit_price=limit_price,
                    extended_hours=extended_hours
                )
            elif order_type_lower == 'stop':
                if not stop_price:
                    raise ValueError("Stop price required for stop orders")
                order_request = StopOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=alpaca_side,
                    time_in_force=alpaca_tif,
                    stop_price=stop_price,
                    extended_hours=extended_hours
                )
            elif order_type_lower in ['stop_limit', 'stoplimit']:
                if not limit_price or not stop_price:
                    raise ValueError("Both limit and stop price required for stop-limit orders")
                order_request = StopLimitOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=alpaca_side,
                    time_in_force=alpaca_tif,
                    limit_price=limit_price,
                    stop_price=stop_price,
                    extended_hours=extended_hours
                )
            else:
                raise ValueError(f"Unsupported order type: {order_type}")
                
            # Submit order
            order = self.trading_client.submit_order(order_request)
            
            logger.info(f"Order placed: {order.id} - {side} {quantity} {symbol}")
            
            # Map order status
            status_map = {
                'new': 'open',
                'accepted': 'pending',
                'pending_new': 'pending',
                'filled': 'filled',
                'partially_filled': 'partially_filled',
                'canceled': 'cancelled',
                'pending_cancel': 'pending_cancel',
                'rejected': 'rejected',
                'expired': 'expired',
                'done_for_day': 'done_for_day',
                'stopped': 'stopped',
                'suspended': 'suspended'
            }
            
            return OrderInfo(
                order_id=str(order.id),
                broker_name=self.broker_name,
                symbol=order.symbol,
                order_type=order_type,
                side=side.lower(),
                quantity=float(order.qty),
                filled_quantity=float(order.filled_qty) if order.filled_qty else 0.0,
                status=status_map.get(order.status, 'open'),
                limit_price=float(order.limit_price) if order.limit_price else None,
                stop_price=float(order.stop_price) if order.stop_price else None,
                avg_fill_price=float(order.filled_avg_price) if order.filled_avg_price else None,
                time_in_force=time_in_force,
                extended_hours=extended_hours,
                submitted_at=order.submitted_at,
                filled_at=order.filled_at,
                metadata={
                    'asset_id': order.asset_id,
                    'asset_class': order.asset_class,
                    'order_class': order.order_class,
                    'client_order_id': order.client_order_id
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            raise
            
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order"""
        if not self.trading_client:
            raise ConnectionError("Not connected to Alpaca")
            
        try:
            self.trading_client.cancel_order_by_id(order_id)
            logger.info(f"Order cancellation requested: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
            
    async def get_orders(self, status: Optional[str] = None) -> List[OrderInfo]:
        """Get orders with optional status filter"""
        if not self.trading_client:
            raise ConnectionError("Not connected to Alpaca")
            
        try:
            # Map status filter
            filter_map = {
                'open': 'open',
                'closed': 'closed',
                'all': 'all'
            }
            
            # Create request
            if status:
                filter_status = filter_map.get(status.lower(), 'open')
                request = GetOrdersRequest(status=filter_status)
            else:
                request = GetOrdersRequest(status='all')
                
            orders = self.trading_client.get_orders(request)
            
            # Convert to OrderInfo objects
            order_list = []
            status_map = {
                'new': 'open',
                'accepted': 'pending',
                'pending_new': 'pending',
                'filled': 'filled',
                'partially_filled': 'partially_filled',
                'canceled': 'cancelled',
                'pending_cancel': 'pending_cancel',
                'rejected': 'rejected',
                'expired': 'expired',
                'done_for_day': 'done_for_day',
                'stopped': 'stopped',
                'suspended': 'suspended'
            }
            
            for order in orders:
                order_list.append(OrderInfo(
                    order_id=str(order.id),
                    broker_name=self.broker_name,
                    symbol=order.symbol,
                    order_type=order.type.lower(),
                    side='buy' if order.side == AlpacaOrderSide.BUY else 'sell',
                    quantity=float(order.qty),
                    filled_quantity=float(order.filled_qty) if order.filled_qty else 0.0,
                    status=status_map.get(order.status, 'open'),
                    limit_price=float(order.limit_price) if order.limit_price else None,
                    stop_price=float(order.stop_price) if order.stop_price else None,
                    avg_fill_price=float(order.filled_avg_price) if order.filled_avg_price else None,
                    time_in_force=order.time_in_force.lower(),
                    extended_hours=order.extended_hours if order.extended_hours else False,
                    submitted_at=order.submitted_at,
                    filled_at=order.filled_at,
                    metadata={
                        'asset_id': order.asset_id,
                        'asset_class': order.asset_class,
                        'order_class': order.order_class,
                        'client_order_id': order.client_order_id
                    }
                ))
                
            return order_list
            
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return []
            
    async def get_order(self, order_id: str) -> Optional[OrderInfo]:
        """Get specific order details"""
        if not self.trading_client:
            raise ConnectionError("Not connected to Alpaca")
            
        try:
            order = self.trading_client.get_order_by_id(order_id)
            
            status_map = {
                'new': 'open',
                'accepted': 'pending',
                'pending_new': 'pending',
                'filled': 'filled',
                'partially_filled': 'partially_filled',
                'canceled': 'cancelled',
                'pending_cancel': 'pending_cancel',
                'rejected': 'rejected',
                'expired': 'expired',
                'done_for_day': 'done_for_day',
                'stopped': 'stopped',
                'suspended': 'suspended'
            }
            
            return OrderInfo(
                order_id=str(order.id),
                broker_name=self.broker_name,
                symbol=order.symbol,
                order_type=order.type.lower(),
                side='buy' if order.side == AlpacaOrderSide.BUY else 'sell',
                quantity=float(order.qty),
                filled_quantity=float(order.filled_qty) if order.filled_qty else 0.0,
                status=status_map.get(order.status, 'open'),
                limit_price=float(order.limit_price) if order.limit_price else None,
                stop_price=float(order.stop_price) if order.stop_price else None,
                avg_fill_price=float(order.filled_avg_price) if order.filled_avg_price else None,
                time_in_force=order.time_in_force.lower(),
                extended_hours=order.extended_hours if order.extended_hours else False,
                submitted_at=order.submitted_at,
                filled_at=order.filled_at,
                metadata={
                    'asset_id': order.asset_id,
                    'asset_class': order.asset_class,
                    'order_class': order.order_class,
                    'client_order_id': order.client_order_id
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            return None
            
    async def get_market_data(
        self,
        symbol: str,
        timeframe: str = "1d",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 100
    ) -> List[MarketDataPoint]:
        """Get historical market data (OHLCV)"""
        if not self.data_client:
            raise ConnectionError("Not connected to Alpaca")
            
        try:
            # Map timeframe to Alpaca TimeFrame
            timeframe_map = {
                '1m': TimeFrame(1, TimeFrameUnit.Minute),
                '5m': TimeFrame(5, TimeFrameUnit.Minute),
                '15m': TimeFrame(15, TimeFrameUnit.Minute),
                '30m': TimeFrame(30, TimeFrameUnit.Minute),
                '1h': TimeFrame(1, TimeFrameUnit.Hour),
                '1d': TimeFrame(1, TimeFrameUnit.Day),
                '1w': TimeFrame(1, TimeFrameUnit.Week),
                '1M': TimeFrame(1, TimeFrameUnit.Month)
            }
            alpaca_timeframe = timeframe_map.get(timeframe, TimeFrame(1, TimeFrameUnit.Day))
            
            # Set default start/end if not provided
            if not end:
                end = datetime.utcnow()
            if not start:
                start = end - timedelta(days=30)
                
            # Create request
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=alpaca_timeframe,
                start=start,
                end=end,
                limit=limit
            )
            
            # Fetch data
            bars = self.data_client.get_stock_bars(request)
            
            # Convert to MarketDataPoint objects
            market_data = []
            if symbol in bars:
                for bar in bars[symbol]:
                    market_data.append(MarketDataPoint(
                        symbol=symbol,
                        broker_name=self.broker_name,
                        timestamp=bar.timestamp,
                        open=float(bar.open),
                        high=float(bar.high),
                        low=float(bar.low),
                        close=float(bar.close),
                        volume=float(bar.volume),
                        timeframe=timeframe,
                        metadata={
                            'trade_count': bar.trade_count,
                            'vwap': float(bar.vwap) if bar.vwap else None
                        }
                    ))
                    
            return market_data
            
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            return []
            
    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time quote for symbol"""
        if not self.data_client:
            raise ConnectionError("Not connected to Alpaca")
            
        try:
            # Create request
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            
            # Fetch quote
            quotes = self.data_client.get_stock_latest_quote(request)
            
            if symbol in quotes:
                quote = quotes[symbol]
                return {
                    'symbol': symbol,
                    'bid': float(quote.bid_price),
                    'ask': float(quote.ask_price),
                    'bid_size': float(quote.bid_size),
                    'ask_size': float(quote.ask_size),
                    'timestamp': quote.timestamp.isoformat(),
                    'conditions': quote.conditions,
                    'tape': quote.tape
                }
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            return {}
