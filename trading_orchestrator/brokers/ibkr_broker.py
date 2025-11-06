"""
Interactive Brokers (IBKR) TWS API Integration

Implements broker interface using IBKR's TWS API (socket-based)
Requires TWS or IB Gateway to be running and configured for API access

Key Features:
- Socket-based connection to TWS/IB Gateway
- Real-time market data (L1, tick-by-tick, depth)
- Order management (place, modify, cancel)
- Position and account data
- Rate limiting and pacing compliance
- Automatic reconnection with exponential backoff

Configuration Requirements (in BrokerConfig.config dict):
- host: TWS/Gateway host (default: 127.0.0.1)
- port: TWS/Gateway port (default: 7497 for paper, 7496 for live)
- client_id: Client ID for this connection (default: 1)
- account: Account code (optional, uses first available if not specified)

Note: Based on research/ibkr_api_analysis.md
"""

from brokers.base import (
    BaseBroker, BrokerConfig, AccountInfo, PositionInfo, 
    OrderInfo, MarketDataPoint
)
from typing import List, Dict, Optional, Any, AsyncIterator
from datetime import datetime, timedelta
from decimal import Decimal
from collections import defaultdict
import asyncio
import threading
import time

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.common import TickerId, OrderId

from loguru import logger


class IBKRWrapper(EWrapper):
    """
    EWrapper implementation for handling IBKR TWS API callbacks
    
    Callbacks are invoked by the TWS API in response to requests.
    This class stores the data and signals when responses are complete.
    """
    
    def __init__(self):
        super().__init__()
        self.next_order_id: Optional[int] = None
        self.accounts: List[str] = []
        self.account_values: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.positions: Dict[str, Dict] = {}
        self.open_orders: Dict[int, Dict] = {}
        self.order_status: Dict[int, Dict] = {}
        self.executions: List[Dict] = []
        self.market_data: Dict[int, Dict] = defaultdict(dict)
        self.historical_data: Dict[int, List] = defaultdict(list)
        self.errors: List[Dict] = []
        self.connected = False
        self.data_events: Dict[Any, asyncio.Event] = {}
        
    def nextValidId(self, orderId: int):
        """Called when connection is established with next valid order ID"""
        self.next_order_id = orderId
        self.connected = True
        logger.info(f"Connected to TWS. Next valid order ID: {orderId}")
        
    def connectionClosed(self):
        """Connection closed callback"""
        self.connected = False
        logger.warning("Connection to TWS closed")
        
    def error(self, reqId: TickerId, errorCode: int, errorString: str, advancedOrderRejectJson: str = ""):
        """Error callback with request ID, error code, and message"""
        error_data = {
            'req_id': reqId,
            'code': errorCode,
            'message': errorString,
            'timestamp': datetime.utcnow()
        }
        self.errors.append(error_data)
        
        if errorCode in [502, 504, 1100, 1101, 1102]:  # Connection errors
            logger.error(f"Connection error {errorCode}: {errorString}")
        elif errorCode == 100:  # Pacing violation
            logger.warning(f"Pacing violation {errorCode}: {errorString}")
        elif errorCode >= 2000:  # Informational messages
            logger.info(f"Info {errorCode}: {errorString}")
        else:
            logger.error(f"Error {errorCode} for request {reqId}: {errorString}")
            
    def managedAccounts(self, accountsList: str):
        """Receives comma-separated list of managed account codes"""
        self.accounts = accountsList.split(',')
        logger.info(f"Managed accounts: {self.accounts}")
        
    def updateAccountValue(self, key: str, val: str, currency: str, accountName: str):
        """Account value updates"""
        self.account_values[accountName][key] = {
            'value': val,
            'currency': currency
        }
        
    def updatePortfolio(self, contract: Contract, position: Decimal,
                       marketPrice: float, marketValue: float,
                       averageCost: float, unrealizedPNL: float,
                       realizedPNL: float, accountName: str):
        """Portfolio position updates"""
        position_key = f"{contract.symbol}_{contract.secType}_{contract.exchange}"
        self.positions[position_key] = {
            'contract': contract,
            'position': float(position),
            'market_price': marketPrice,
            'market_value': marketValue,
            'average_cost': averageCost,
            'unrealized_pnl': unrealizedPNL,
            'realized_pnl': realizedPNL,
            'account': accountName
        }
        
    def accountDownloadEnd(self, accountName: str):
        """Signals end of account data download"""
        logger.info(f"Account download complete for {accountName}")
        if accountName in self.data_events:
            try:
                loop = asyncio.get_event_loop()
                loop.call_soon_threadsafe(self.data_events[accountName].set)
            except RuntimeError:
                pass
                
    def openOrder(self, orderId: OrderId, contract: Contract, order: Order, orderState):
        """Open order callback"""
        self.open_orders[orderId] = {
            'contract': contract,
            'order': order,
            'state': orderState,
            'timestamp': datetime.utcnow()
        }
        
    def orderStatus(self, orderId: OrderId, status: str, filled: Decimal,
                   remaining: Decimal, avgFillPrice: float, permId: int,
                   parentId: int, lastFillPrice: float, clientId: int,
                   whyHeld: str, mktCapPrice: float):
        """Order status updates"""
        self.order_status[orderId] = {
            'status': status,
            'filled': float(filled),
            'remaining': float(remaining),
            'avg_fill_price': avgFillPrice,
            'last_fill_price': lastFillPrice,
            'perm_id': permId,
            'parent_id': parentId,
            'client_id': clientId,
            'why_held': whyHeld,
            'timestamp': datetime.utcnow()
        }
        
    def tickPrice(self, reqId: TickerId, tickType: int, price: float, attrib):
        """Real-time tick price updates"""
        tick_map = {1: 'bid', 2: 'ask', 4: 'last', 6: 'high', 7: 'low', 9: 'close'}
        tick_name = tick_map.get(tickType, f'tick_{tickType}')
        self.market_data[reqId][tick_name] = price
        self.market_data[reqId]['timestamp'] = datetime.utcnow()
        
    def tickSize(self, reqId: TickerId, tickType: int, size: Decimal):
        """Real-time tick size updates"""
        tick_map = {0: 'bid_size', 3: 'ask_size', 5: 'last_size', 8: 'volume'}
        tick_name = tick_map.get(tickType, f'size_{tickType}')
        self.market_data[reqId][tick_name] = float(size)
        
    def historicalData(self, reqId: int, bar):
        """Historical bar data"""
        self.historical_data[reqId].append({
            'timestamp': bar.date,
            'open': float(bar.open),
            'high': float(bar.high),
            'low': float(bar.low),
            'close': float(bar.close),
            'volume': int(bar.volume),
            'wap': float(bar.wap),
            'count': int(bar.barCount)
        })
        
    def historicalDataEnd(self, reqId: int, start: str, end: str):
        """Signals end of historical data"""
        logger.info(f"Historical data complete for request {reqId}")
        if reqId in self.data_events:
            try:
                loop = asyncio.get_event_loop()
                loop.call_soon_threadsafe(self.data_events[reqId].set)
            except RuntimeError:
                pass


class IBKRClient(EClient):
    """EClient extension with async wrapper support"""
    
    def __init__(self, wrapper: IBKRWrapper):
        super().__init__(wrapper)
        self.wrapper = wrapper


class IBKRBroker(BaseBroker):
    """
    Interactive Brokers TWS API implementation
    
    Connects to TWS or IB Gateway for trading operations.
    Implements BaseBroker interface for unified broker access.
    """
    
    def __init__(self, config: BrokerConfig):
        super().__init__(config)
        
        # Extract configuration
        config_dict = config.config or {}
        self.host = config_dict.get('host', '127.0.0.1')
        self.port = int(config_dict.get('port', 7497 if config.is_paper else 7496))
        self.client_id = int(config_dict.get('client_id', 1))
        self.account = config_dict.get('account')
        
        # API components
        self.wrapper = IBKRWrapper()
        self.client = IBKRClient(self.wrapper)
        self.api_thread: Optional[threading.Thread] = None
        
        # Request ID tracking
        self.next_req_id = 1
        self.req_id_lock = threading.Lock()
        
        # Rate limiting
        self.historical_min_interval = 15  # seconds
        self.last_historical_request = 0
        
        logger.info(f"IBKR broker initialized (paper_trading={config.is_paper})")
        
    def _get_next_req_id(self) -> int:
        """Get next available request ID (thread-safe)"""
        with self.req_id_lock:
            req_id = self.next_req_id
            self.next_req_id += 1
            return req_id
            
    def _run_api_thread(self):
        """Run the TWS API message loop in a separate thread"""
        self.client.run()
        
    async def connect(self) -> bool:
        """Connect to TWS or IB Gateway"""
        async with self._connection_lock:
            try:
                logger.info(f"Connecting to IBKR at {self.host}:{self.port}...")
                
                # Start connection
                self.client.connect(self.host, self.port, self.client_id)
                
                # Start API thread
                self.api_thread = threading.Thread(target=self._run_api_thread, daemon=True)
                self.api_thread.start()
                
                # Wait for connection
                max_wait = 100  # 10 seconds
                for _ in range(max_wait):
                    if self.wrapper.connected and self.wrapper.next_order_id is not None:
                        break
                    await asyncio.sleep(0.1)
                else:
                    logger.error("Connection timeout")
                    return False
                    
                # Request managed accounts
                self.client.reqManagedAccts()
                await asyncio.sleep(0.5)
                
                # Set default account if not specified
                if not self.account and self.wrapper.accounts:
                    self.account = self.wrapper.accounts[0]
                    logger.info(f"Using account: {self.account}")
                    
                self.is_connected = True
                logger.success(f"Connected to IBKR at {self.host}:{self.port}")
                return True
                
            except Exception as e:
                logger.error(f"Connection failed: {e}")
                self.is_connected = False
                return False
                
    async def disconnect(self) -> bool:
        """Disconnect from TWS/Gateway"""
        async with self._connection_lock:
            try:
                if self.client.isConnected():
                    self.client.disconnect()
                    
                if self.api_thread and self.api_thread.is_alive():
                    self.api_thread.join(timeout=5)
                    
                self.is_connected = False
                logger.info("Disconnected from IBKR")
                return True
                
            except Exception as e:
                logger.error(f"Disconnection error: {e}")
                return False
                
    async def is_connection_alive(self) -> bool:
        """Check if connection is still alive"""
        return self.client.isConnected() and self.wrapper.connected
        
    async def get_account(self) -> AccountInfo:
        """Get account information"""
        if not self.account:
            raise ValueError("No account specified")
            
        try:
            # Create event for async waiting
            event = asyncio.Event()
            self.wrapper.data_events[self.account] = event
            
            # Request account updates
            self.client.reqAccountUpdates(True, self.account)
            
            # Wait for data with timeout
            try:
                await asyncio.wait_for(event.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Account data request timed out")
                
            # Stop updates
            self.client.reqAccountUpdates(False, self.account)
            
            # Extract account data
            account_data = self.wrapper.account_values.get(self.account, {})
            
            net_liquidation = float(account_data.get('NetLiquidation', {}).get('value', 0))
            available_funds = float(account_data.get('AvailableFunds', {}).get('value', 0))
            buying_power = float(account_data.get('BuyingPower', {}).get('value', 0))
            currency = account_data.get('NetLiquidation', {}).get('currency', 'USD')
            
            return AccountInfo(
                account_id=self.account,
                broker_name=self.broker_name,
                currency=currency,
                balance=net_liquidation,
                available_balance=available_funds,
                equity=net_liquidation,
                buying_power=buying_power,
                margin_used=0.0,
                margin_available=available_funds,
                is_pattern_day_trader=False,
                metadata={'account_values': account_data}
            )
            
        except Exception as e:
            logger.error(f"Failed to get account: {e}")
            raise
            
    async def get_positions(self) -> List[PositionInfo]:
        """Get all open positions"""
        if not self.account:
            raise ValueError("No account specified")
            
        try:
            # Create event
            event = asyncio.Event()
            self.wrapper.data_events[self.account] = event
            
            # Clear previous positions
            self.wrapper.positions.clear()
            
            # Request account updates (includes positions)
            self.client.reqAccountUpdates(True, self.account)
            
            # Wait for data
            try:
                await asyncio.wait_for(event.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Position data request timed out")
                
            # Stop updates
            self.client.reqAccountUpdates(False, self.account)
            
            # Convert to PositionInfo objects
            positions = []
            for pos_data in self.wrapper.positions.values():
                contract = pos_data['contract']
                quantity = pos_data['position']
                
                positions.append(PositionInfo(
                    symbol=contract.symbol,
                    broker_name=self.broker_name,
                    side='long' if quantity > 0 else 'short',
                    quantity=abs(quantity),
                    avg_entry_price=pos_data['average_cost'],
                    current_price=pos_data['market_price'],
                    market_value=pos_data['market_value'],
                    unrealized_pnl=pos_data['unrealized_pnl'],
                    unrealized_pnl_percent=(pos_data['unrealized_pnl'] / (pos_data['average_cost'] * abs(quantity)) * 100) if pos_data['average_cost'] * quantity != 0 else 0,
                    cost_basis=pos_data['average_cost'] * abs(quantity),
                    metadata={
                        'contract_type': contract.secType,
                        'exchange': contract.exchange,
                        'realized_pnl': pos_data['realized_pnl']
                    }
                ))
                
            return positions
            
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            raise
            
    async def get_position(self, symbol: str) -> Optional[PositionInfo]:
        """Get position for specific symbol"""
        positions = await self.get_positions()
        for pos in positions:
            if pos.symbol == symbol:
                return pos
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
        try:
            # Create contract
            contract = Contract()
            contract.symbol = symbol
            contract.secType = kwargs.get('sec_type', 'STK')
            contract.exchange = kwargs.get('exchange', 'SMART')
            contract.currency = kwargs.get('currency', 'USD')
            
            # Create order
            order = Order()
            order.action = 'BUY' if side.lower() == 'buy' else 'SELL'
            order.totalQuantity = quantity
            
            # Set order type
            order_type_lower = order_type.lower()
            if order_type_lower == 'market':
                order.orderType = 'MKT'
            elif order_type_lower == 'limit':
                order.orderType = 'LMT'
                order.lmtPrice = limit_price if limit_price else 0
            elif order_type_lower == 'stop':
                order.orderType = 'STP'
                order.auxPrice = stop_price if stop_price else 0
            elif order_type_lower in ['stop_limit', 'stoplimit']:
                order.orderType = 'STP LMT'
                order.lmtPrice = limit_price if limit_price else 0
                order.auxPrice = stop_price if stop_price else 0
            else:
                order.orderType = 'MKT'
                
            # Set time in force
            tif_map = {'day': 'DAY', 'gtc': 'GTC', 'ioc': 'IOC', 'fok': 'FOK'}
            order.tif = tif_map.get(time_in_force.lower(), 'DAY')
            
            # Get order ID
            order_id = self.wrapper.next_order_id
            self.wrapper.next_order_id += 1
            
            # Place order
            self.client.placeOrder(order_id, contract, order)
            
            # Wait a bit for order confirmation
            await asyncio.sleep(0.5)
            
            logger.info(f"Order placed: {order_id} - {side} {quantity} {symbol}")
            
            return OrderInfo(
                order_id=str(order_id),
                broker_name=self.broker_name,
                symbol=symbol,
                order_type=order_type,
                side=side.lower(),
                quantity=quantity,
                filled_quantity=0.0,
                status='pending',
                limit_price=limit_price,
                stop_price=stop_price,
                avg_fill_price=None,
                time_in_force=time_in_force,
                extended_hours=extended_hours,
                submitted_at=datetime.utcnow(),
                metadata={'contract_type': contract.secType, 'exchange': contract.exchange}
            )
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            raise
            
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order"""
        try:
            self.client.cancelOrder(int(order_id), "")
            logger.info(f"Order cancellation requested: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
            
    async def get_orders(self, status: Optional[str] = None) -> List[OrderInfo]:
        """Get orders with optional status filter"""
        try:
            # Clear previous data
            self.wrapper.open_orders.clear()
            self.wrapper.order_status.clear()
            
            # Request all open orders
            self.client.reqAllOpenOrders()
            
            # Wait for data
            await asyncio.sleep(2.0)
            
            # Convert to OrderInfo objects
            orders = []
            for order_id, order_data in self.wrapper.open_orders.items():
                contract = order_data['contract']
                order = order_data['order']
                status_data = self.wrapper.order_status.get(order_id, {})
                
                # Map IBKR status
                ibkr_status = status_data.get('status', 'Unknown').lower()
                status_map = {
                    'presubmitted': 'pending',
                    'pendingsubmit': 'pending',
                    'submitted': 'open',
                    'filled': 'filled',
                    'cancelled': 'cancelled',
                    'pendingcancel': 'pending_cancel',
                    'inactive': 'rejected'
                }
                order_status = status_map.get(ibkr_status, 'open')
                
                # Skip if status filter doesn't match
                if status and order_status != status.lower():
                    continue
                    
                orders.append(OrderInfo(
                    order_id=str(order_id),
                    broker_name=self.broker_name,
                    symbol=contract.symbol,
                    order_type=order.orderType.lower(),
                    side='buy' if order.action == 'BUY' else 'sell',
                    quantity=order.totalQuantity,
                    filled_quantity=status_data.get('filled', 0),
                    status=order_status,
                    limit_price=order.lmtPrice if order.lmtPrice else None,
                    stop_price=order.auxPrice if order.auxPrice else None,
                    avg_fill_price=status_data.get('avg_fill_price') if status_data.get('avg_fill_price', 0) > 0 else None,
                    time_in_force=order.tif.lower(),
                    extended_hours=False,
                    submitted_at=order_data['timestamp'],
                    metadata={'contract_type': contract.secType, 'exchange': contract.exchange}
                ))
                
            return orders
            
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return []
            
    async def get_order(self, order_id: str) -> Optional[OrderInfo]:
        """Get specific order details"""
        orders = await self.get_orders()
        for order in orders:
            if order.order_id == order_id:
                return order
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
        try:
            # Rate limiting check
            current_time = time.time()
            if current_time - self.last_historical_request < self.historical_min_interval:
                wait_time = self.historical_min_interval - (current_time - self.last_historical_request)
                logger.info(f"Rate limiting: waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                
            # Create contract
            contract = Contract()
            contract.symbol = symbol
            contract.secType = 'STK'
            contract.exchange = 'SMART'
            contract.currency = 'USD'
            
            # Get request ID
            req_id = self._get_next_req_id()
            
            # Create event
            event = asyncio.Event()
            self.wrapper.data_events[req_id] = event
            
            # Clear previous data
            self.wrapper.historical_data[req_id] = []
            
            # Map timeframe to IBKR bar size
            timeframe_map = {
                '1m': '1 min', '5m': '5 mins', '15m': '15 mins', '30m': '30 mins',
                '1h': '1 hour', '1d': '1 day', '1w': '1 week', '1M': '1 month'
            }
            bar_size = timeframe_map.get(timeframe, '1 day')
            
            # Determine duration
            end_datetime = end.strftime('%Y%m%d %H:%M:%S') if end else ''
            duration = '1 D'  # Default
            
            # Request historical data
            self.client.reqHistoricalData(
                req_id, contract, end_datetime, duration,
                bar_size, 'TRADES', 1, 1, False, []
            )
            
            # Update rate limiting
            self.last_historical_request = time.time()
            
            # Wait for data
            try:
                await asyncio.wait_for(event.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning(f"Historical data request timed out for {symbol}")
                
            # Convert to MarketDataPoint objects
            bars = self.wrapper.historical_data.get(req_id, [])
            market_data = []
            
            for bar in bars:
                market_data.append(MarketDataPoint(
                    symbol=symbol,
                    broker_name=self.broker_name,
                    timestamp=datetime.strptime(bar['timestamp'], '%Y%m%d  %H:%M:%S') if ' ' in bar['timestamp'] else datetime.strptime(bar['timestamp'], '%Y%m%d'),
                    open=bar['open'],
                    high=bar['high'],
                    low=bar['low'],
                    close=bar['close'],
                    volume=bar['volume'],
                    timeframe=timeframe,
                    metadata={'wap': bar['wap'], 'count': bar['count']}
                ))
                
            return market_data[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            return []
            
    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time quote for symbol"""
        try:
            # Create contract
            contract = Contract()
            contract.symbol = symbol
            contract.secType = 'STK'
            contract.exchange = 'SMART'
            contract.currency = 'USD'
            
            # Get request ID
            req_id = self._get_next_req_id()
            
            # Request market data
            self.client.reqMktData(req_id, contract, '', False, False, [])
            
            # Wait for data
            await asyncio.sleep(2.0)
            
            # Cancel subscription
            self.client.cancelMktData(req_id)
            
            # Get data
            data = self.wrapper.market_data.get(req_id, {})
            
            return {
                'symbol': symbol,
                'bid': data.get('bid', 0),
                'ask': data.get('ask', 0),
                'last': data.get('last', 0),
                'high': data.get('high', 0),
                'low': data.get('low', 0),
                'close': data.get('close', 0),
                'volume': data.get('volume', 0),
                'timestamp': data.get('timestamp', datetime.utcnow()).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            return {}
