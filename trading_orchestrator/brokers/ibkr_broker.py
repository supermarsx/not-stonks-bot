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