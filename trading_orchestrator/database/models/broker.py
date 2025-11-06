"""
Broker Connection and Status Models
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, JSON, Float, Text, Enum as SQLEnum
from sqlalchemy.sql import func
from config.database import Base
from enum import Enum
from datetime import datetime


class ConnectionStatus(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    RECONNECTING = "reconnecting"


class BrokerConnection(Base):
    """Broker connection status and configuration"""
    __tablename__ = "broker_connections"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    broker_name = Column(String(50), nullable=False, index=True)
    
    # Connection details
    status = Column(SQLEnum(ConnectionStatus), default=ConnectionStatus.DISCONNECTED, nullable=False, index=True)
    connection_type = Column(String(20), nullable=False)  # rest, websocket, socket
    
    # Configuration (encrypted sensitive data)
    credentials = Column(Text, nullable=False)  # Encrypted JSON
    config = Column(JSON, default=dict)
    is_paper = Column(Boolean, default=True, nullable=False)
    
    # Health metrics
    last_heartbeat_at = Column(DateTime(timezone=True), nullable=True)
    connection_uptime_seconds = Column(Integer, default=0)
    reconnect_count = Column(Integer, default=0)
    error_count = Column(Integer, default=0)
    last_error = Column(Text, nullable=True)
    
    # Rate limiting
    requests_sent = Column(Integer, default=0)
    requests_remaining = Column(Integer, nullable=True)
    rate_limit_reset_at = Column(DateTime(timezone=True), nullable=True)
    
    # Timestamps
    connected_at = Column(DateTime(timezone=True), nullable=True)
    disconnected_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<BrokerConnection(id={self.id}, broker='{self.broker_name}', status='{self.status}')>"


class BrokerAccount(Base):
    """Broker account information"""
    __tablename__ = "broker_accounts"
    
    id = Column(Integer, primary_key=True, index=True)
    connection_id = Column(Integer, nullable=False, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    broker_name = Column(String(50), nullable=False, index=True)
    
    # Account details
    account_id = Column(String(100), nullable=False, index=True)
    account_type = Column(String(50), nullable=True)  # cash, margin, etc.
    currency = Column(String(10), default="USD", nullable=False)
    
    # Balance information
    balance = Column(Float, default=0.0, nullable=False)
    available_balance = Column(Float, default=0.0, nullable=False)
    equity = Column(Float, default=0.0, nullable=False)
    margin_used = Column(Float, default=0.0, nullable=False)
    margin_available = Column(Float, default=0.0, nullable=False)
    
    # Buying power
    buying_power = Column(Float, default=0.0, nullable=False)
    day_trading_buying_power = Column(Float, nullable=True)
    
    # Account status
    is_active = Column(Boolean, default=True, nullable=False)
    is_pattern_day_trader = Column(Boolean, default=False, nullable=False)
    can_trade = Column(Boolean, default=True, nullable=False)
    
    # Additional metadata
    account_config_metadata = Column(JSON, default=dict)
    
    # Timestamps
    last_synced_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<BrokerAccount(id={self.id}, broker='{self.broker_name}', account_id='{self.account_id}')>"


class BrokerSymbol(Base):
    """Supported symbols for each broker"""
    __tablename__ = "broker_symbols"
    
    id = Column(Integer, primary_key=True, index=True)
    connection_id = Column(Integer, nullable=False, index=True)
    broker_name = Column(String(50), nullable=False, index=True)
    
    # Symbol information
    symbol = Column(String(50), nullable=False, index=True)
    display_symbol = Column(String(50), nullable=True)
    security_type = Column(String(50), nullable=True)  # equity, option, crypto, etc.
    asset_class = Column(String(50), nullable=False, index=True)  # stock, crypto, forex, etc.
    
    # Exchange and market data
    exchange = Column(String(50), nullable=True)
    market = Column(String(50), nullable=True)
    
    # Trading configuration
    is_tradable = Column(Boolean, default=True, nullable=False)
    is_marginable = Column(Boolean, default=False, nullable=False)
    is_shortable = Column(Boolean, default=False, nullable=False)
    is_fractional = Column(Boolean, default=False, nullable=False)
    min_order_size = Column(Float, nullable=True)
    max_order_size = Column(Float, nullable=True)
    
    # Price data
    tick_size = Column(Float, nullable=True)
    lot_size = Column(Integer, nullable=True)
    price_increment = Column(Float, nullable=True)
    size_increment = Column(Float, nullable=True)
    
    # Additional metadata
    name = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    symbol_config_metadata = Column(JSON, default=dict)
    config_metadata = Column(JSON, default=dict)
    
    # Timestamps
    last_updated_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<BrokerSymbol(id={self.id}, broker='{self.broker_name}', symbol='{self.symbol}', exchange='{self.exchange}')>"


# BrokerSymbol class was merged above to avoid duplicate table definitions