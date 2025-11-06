from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import asyncio


@dataclass
class BrokerConfig:
    """Broker configuration"""
    broker_name: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    is_paper: bool = True
    config: Dict[str, Any] = None


@dataclass
class AccountInfo:
    """Unified account information"""
    account_id: str
    broker_name: str
    currency: str
    balance: float
    available_balance: float
    equity: float
    buying_power: float
    margin_used: float = 0.0
    margin_available: float = 0.0
    is_pattern_day_trader: bool = False
    metadata: Dict[str, Any] = None


@dataclass
class PositionInfo:
    """Unified position information"""
    symbol: str
    broker_name: str
    side: str  # long or short
    quantity: float
    avg_entry_price: float
    current_price: Optional[float] = None
    market_value: Optional[float] = None
    unrealized_pnl: float = 0.0
    unrealized_pnl_percent: float = 0.0
    cost_basis: float = 0.0
    metadata: Dict[str, Any] = None


@dataclass
class OrderInfo:
    """Unified order information"""
    order_id: str
    broker_name: str
    symbol: str
    order_type: str
    side: str  # buy or sell
    quantity: float
    filled_quantity: float = 0.0
    status: str = "pending"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    avg_fill_price: Optional[float] = None
    time_in_force: str = "day"
    extended_hours: bool = False
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None


@dataclass
class MarketDataPoint:
    """Unified market data"""
    symbol: str
    broker_name: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str  # 1m, 5m, 1h, 1d
    metadata: Dict[str, Any] = None


class BaseBroker(ABC):
    """
    Abstract base class for all broker integrations.
    
    All methods must be implemented by concrete broker classes.
    Provides unified interface for trading operations across different brokers.
    """
    
    def __init__(self, config: BrokerConfig):
        """
        Initialize broker with configuration
        
        Args:
            config: BrokerConfig object with credentials and settings
        """
        self.config = config
        self.broker_name = config.broker_name
        self.is_connected = False
        self._connection_lock = asyncio.Lock()
    
    # Connection Management
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to broker API
        
        Returns:
            bool: True if connection successful
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Close connection to broker API
        
        Returns:
            bool: True if disconnection successful
        """
        pass
    
    @abstractmethod
    async def is_connection_alive(self) -> bool:
        """
        Check if connection is still alive
        
        Returns:
            bool: True if connection is active
        """
        pass
    
    # Account Operations
    
    @abstractmethod
    async def get_account(self) -> AccountInfo:
        """
        Get account information
        
        Returns:
            AccountInfo: Account balance, equity, and other details
        """
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[PositionInfo]:
        """
        Get all open positions
        
        Returns:
            List[PositionInfo]: List of current positions
        """
        pass
    
    @abstractmethod
    async def get_position(self, symbol: str) -> Optional[PositionInfo]:
        """
        Get position for specific symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            PositionInfo or None: Position details if exists
        """
        pass
    
    # Order Management
    
    @abstractmethod
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
        """
        Place a new order
        
        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            order_type: "market", "limit", "stop", etc.
            quantity: Order quantity
            limit_price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            time_in_force: Order duration ("day", "gtc", etc.)
            extended_hours: Allow extended hours trading
            **kwargs: Additional broker-specific parameters
            
        Returns:
            OrderInfo: Placed order details
        """
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            bool: True if cancellation successful
        """
        pass
    
    @abstractmethod
    async def get_orders(self, status: Optional[str] = None) -> List[OrderInfo]:
        """
        Get orders with optional status filter
        
        Args:
            status: Filter by status (None for all orders)
            
        Returns:
            List[OrderInfo]: List of orders
        """
        pass
    
    @abstractmethod
    async def get_order(self, order_id: str) -> Optional[OrderInfo]:
        """
        Get specific order details
        
        Args:
            order_id: Order ID
            
        Returns:
            OrderInfo or None: Order details if exists
        """
        pass
    
    # Market Data
    
    @abstractmethod
    async def get_market_data(
        self,
        symbol: str,
        timeframe: str = "1d",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 100
    ) -> List[MarketDataPoint]:
        """
        Get historical market data (OHLCV)
        
        Args:
            symbol: Trading symbol
            timeframe: Candle timeframe (1m, 5m, 1h, 1d, etc.)
            start: Start datetime (optional)
            end: End datetime (optional)
            limit: Maximum number of candles
            
        Returns:
            List[MarketDataPoint]: Historical OHLCV data
        """
        pass
    
    @abstractmethod
    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get real-time quote for symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict: Quote data with bid, ask, last, volume, etc.
        """
        pass
    
    # Streaming Data (optional, can return None if not supported)
    
    async def stream_quotes(self, symbols: List[str]) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream real-time quotes (if supported)
        
        Args:
            symbols: List of symbols to stream
            
        Yields:
            Dict: Real-time quote updates
        """
        raise NotImplementedError(f"{self.broker_name} does not support quote streaming")
    
    async def stream_trades(self, symbols: List[str]) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream real-time trades (if supported)
        
        Args:
            symbols: List of symbols to stream
            
        Yields:
            Dict: Real-time trade updates
        """
        raise NotImplementedError(f"{self.broker_name} does not support trade streaming")
    
    # Utility Methods
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform broker health check
        
        Returns:
            Dict: Health status information
        """
        return {
            "broker_name": self.broker_name,
            "is_connected": await self.is_connection_alive(),
            "is_paper": self.config.is_paper,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def __repr__(self):
        return f"<{self.__class__.__name__}(broker='{self.broker_name}', connected={self.is_connected})>"