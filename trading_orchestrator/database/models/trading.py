"""
Trading Data Models - Positions, Orders, Trades
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, JSON, Float, Text, Enum as SQLEnum
from sqlalchemy.sql import func
from config.database import Base
from enum import Enum
from datetime import datetime


class PositionSide(str, Enum):
    LONG = "long"
    SHORT = "short"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    OCO = "oco"  # One-Cancels-Other
    BRACKET = "bracket"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    PENDING = "pending"
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(str, Enum):
    DAY = "day"
    GTC = "gtc"  # Good Till Cancel
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill
    GTD = "gtd"  # Good Till Date


class Position(Base):
    """Open positions across all brokers"""
    __tablename__ = "positions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    broker_name = Column(String(50), nullable=False, index=True)
    account_id = Column(String(100), nullable=False, index=True)
    
    # Position identification
    symbol = Column(String(50), nullable=False, index=True)
    exchange = Column(String(50), nullable=True)
    asset_class = Column(String(50), nullable=False)
    
    # Position details
    side = Column(SQLEnum(PositionSide), nullable=False)
    quantity = Column(Float, nullable=False)
    available_quantity = Column(Float, nullable=False)
    
    # Price information
    avg_entry_price = Column(Float, nullable=False)
    current_price = Column(Float, nullable=True)
    market_value = Column(Float, nullable=True)
    
    # P&L
    unrealized_pnl = Column(Float, default=0.0, nullable=False)
    unrealized_pnl_percent = Column(Float, default=0.0, nullable=False)
    realized_pnl = Column(Float, default=0.0, nullable=False)
    
    # Cost basis
    cost_basis = Column(Float, nullable=False)
    
    # Additional metadata
    config_metadata = Column(JSON, default=dict)
    
    # Timestamps
    opened_at = Column(DateTime(timezone=True), nullable=False)
    last_updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    def __repr__(self):
        return f"<Position(id={self.id}, symbol='{self.symbol}', side='{self.side}', qty={self.quantity})>"


class Order(Base):
    """Orders submitted to brokers"""
    __tablename__ = "orders"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    broker_name = Column(String(50), nullable=False, index=True)
    account_id = Column(String(100), nullable=False, index=True)
    
    # Order identification
    broker_order_id = Column(String(100), unique=True, nullable=False, index=True)
    client_order_id = Column(String(100), unique=True, nullable=True, index=True)
    
    # Order details
    symbol = Column(String(50), nullable=False, index=True)
    exchange = Column(String(50), nullable=True)
    asset_class = Column(String(50), nullable=False)
    
    # Order parameters
    order_type = Column(SQLEnum(OrderType), nullable=False)
    side = Column(SQLEnum(OrderSide), nullable=False)
    quantity = Column(Float, nullable=False)
    filled_quantity = Column(Float, default=0.0, nullable=False)
    remaining_quantity = Column(Float, nullable=True)
    
    # Pricing
    limit_price = Column(Float, nullable=True)
    stop_price = Column(Float, nullable=True)
    avg_fill_price = Column(Float, nullable=True)
    
    # Status
    status = Column(SQLEnum(OrderStatus), default=OrderStatus.PENDING, nullable=False, index=True)
    time_in_force = Column(SQLEnum(TimeInForce), default=TimeInForce.DAY, nullable=False)
    
    # Advanced features
    extended_hours = Column(Boolean, default=False, nullable=False)
    trailing_percent = Column(Float, nullable=True)
    trailing_amount = Column(Float, nullable=True)
    
    # Strategy/Tag
    strategy_id = Column(Integer, nullable=True, index=True)
    tags = Column(JSON, default=list)
    
    # Additional metadata
    config_metadata = Column(JSON, default=dict)
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    submitted_at = Column(DateTime(timezone=True), nullable=False)
    filled_at = Column(DateTime(timezone=True), nullable=True)
    cancelled_at = Column(DateTime(timezone=True), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<Order(id={self.id}, symbol='{self.symbol}', type='{self.order_type}', status='{self.status}')>"


class Trade(Base):
    """Executed trades"""
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(Integer, nullable=False, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    broker_name = Column(String(50), nullable=False, index=True)
    account_id = Column(String(100), nullable=False, index=True)
    
    # Trade identification
    broker_trade_id = Column(String(100), unique=True, nullable=False, index=True)
    broker_order_id = Column(String(100), nullable=False, index=True)
    
    # Trade details
    symbol = Column(String(50), nullable=False, index=True)
    exchange = Column(String(50), nullable=True)
    side = Column(SQLEnum(OrderSide), nullable=False)
    
    # Execution
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    commission = Column(Float, default=0.0, nullable=False)
    
    # P&L (for closing trades)
    realized_pnl = Column(Float, nullable=True)
    
    # Additional metadata
    config_metadata = Column(JSON, default=dict)
    
    # Timestamp
    executed_at = Column(DateTime(timezone=True), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    def __repr__(self):
        return f"<Trade(id={self.id}, symbol='{self.symbol}', side='{self.side}', qty={self.quantity}, price={self.price})>"


class MarketData(Base):
    """Historical market data storage"""
    __tablename__ = "market_data"
    
    id = Column(Integer, primary_key=True, index=True)
    broker_name = Column(String(50), nullable=False, index=True)
    
    # Identification
    symbol = Column(String(50), nullable=False, index=True)
    exchange = Column(String(50), nullable=True)
    timeframe = Column(String(20), nullable=False, index=True)  # 1m, 5m, 1h, 1d, etc.
    
    # OHLCV data
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    
    # Additional data
    vwap = Column(Float, nullable=True)
    trades_count = Column(Integer, nullable=True)
    
    # Metadata
    config_metadata = Column(JSON, default=dict)
    
    # Index for efficient queries
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    def __repr__(self):
        return f"<MarketData(symbol='{self.symbol}', timeframe='{self.timeframe}', timestamp='{self.timestamp}')>"