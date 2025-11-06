"""
Market Circuit Breakers Module

This module implements advanced market circuit breakers for extreme market conditions,
including market volatility circuit breakers, price gap protection, and liquidity crisis detection.

Features:
- Market volatility circuit breakers
- Price gap protection mechanisms
- Liquidity crisis detection and response
- Multi-level circuit breaker triggers
- Automatic market halt procedures
- Recovery mechanisms
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Callable
from enum import Enum
from dataclasses import dataclass, field
from decimal import Decimal
import numpy as np
from sqlalchemy import (
    Column, Integer, String, DateTime, Boolean, Float, Text, 
    Index, ForeignKey
)
from sqlalchemy.orm import declarative_base, relationship, Session
from sqlalchemy.sql import func

from trading_orchestrator.database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

Base = declarative_base()


class CircuitBreakerLevel(Enum):
    """Circuit breaker activation levels."""
    LEVEL_1 = "level_1"  # 7% decline - 15-minute halt
    LEVEL_2 = "level_2"  # 13% decline - 15-minute halt
    LEVEL_3 = "level_3"  # 20% decline - Trading halted for day
    CUSTOM = "custom"     # Custom threshold


class MarketCondition(Enum):
    """Current market condition states."""
    NORMAL = "normal"
    VOLATILE = "volatile"
    EXTREME_VOLATILITY = "extreme_volatility"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    WIDE_GAPS = "wide_gaps"
    HALTED = "halted"


class PriceGapType(Enum):
    """Types of price gaps."""
    UP_GAP = "up_gap"
    DOWN_GAP = "down_gap"
    WIDE_GAP = "wide_gap"
    GAP_UP_LIMIT = "gap_up_limit"
    GAP_DOWN_LIMIT = "gap_down_limit"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker parameters."""
    level_1_threshold: Decimal = Decimal("0.07")  # 7% decline
    level_2_threshold: Decimal = Decimal("0.13")  # 13% decline
    level_3_threshold: Decimal = Decimal("0.20")  # 20% decline
    custom_thresholds: Dict[str, Decimal] = field(default_factory=dict)
    
    # Market volatility parameters
    volatility_threshold: Decimal = Decimal("0.025")  # 2.5% intraday volatility
    volatility_period_minutes: int = 30  # Volatility calculation period
    
    # Price gap parameters
    gap_threshold_percent: Decimal = Decimal("0.02")  # 2% price gap
    wide_gap_threshold_percent: Decimal = Decimal("0.05")  # 5% wide gap
    
    # Liquidity crisis parameters
    min_bid_ask_spread: Decimal = Decimal("0.01")  # 1% minimum spread
    min_volume_threshold: int = 1000  # Minimum volume threshold
    volume_decline_threshold: Decimal = Decimal("0.5")  # 50% volume decline
    
    # Recovery parameters
    recovery_period_minutes: int = 60  # Time before circuit breaker reset
    automatic_recovery: bool = True
    
    # Market hours
    market_open_time: str = "09:30"
    market_close_time: str = "16:00"
    extended_hours_trading: bool = False


@dataclass
class MarketData:
    """Market data for circuit breaker evaluation."""
    symbol: str
    timestamp: datetime
    current_price: Decimal
    previous_close: Decimal
    open_price: Optional[Decimal] = None
    high_price: Optional[Decimal] = None
    low_price: Optional[Decimal] = None
    volume: Optional[int] = None
    bid_price: Optional[Decimal] = None
    ask_price: Optional[Decimal] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None


class MarketCircuitBreaker(Base):
    """Database model for market circuit breakers."""
    __tablename__ = "market_circuit_breakers"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    breaker_type = Column(String(50), nullable=False)
    level = Column(String(20), nullable=False)
    threshold_value = Column(Float, nullable=False)
    current_value = Column(Float, nullable=False)
    triggered_at = Column(DateTime, nullable=True)
    triggered = Column(Boolean, default=False, nullable=False)
    reset_at = Column(DateTime, nullable=True)
    market_condition = Column(String(50), nullable=True)
    details = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())


class CircuitBreakerEvent(Base):
    """Database model for circuit breaker events."""
    __tablename__ = "circuit_breaker_events"
    
    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String(50), nullable=False)
    level = Column(String(20), nullable=True)
    symbol = Column(String(20), nullable=True)
    trigger_value = Column(Float, nullable=False)
    current_value = Column(Float, nullable=False)
    action_taken = Column(String(100), nullable=False)
    triggered_by = Column(String(100), nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    duration_minutes = Column(Integer, nullable=True)
    impact_description = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


class MarketCircuitBreakerManager:
    """Manager for market circuit breakers."""
    
    def __init__(self, db_manager: DatabaseManager, config: CircuitBreakerConfig):
        self.db_manager = db_manager
        self.config = config
        self._active_breakers: Dict[str, CircuitBreakerLevel] = {}
        self._monitoring_tasks: Dict[str, asyncio.Task] = {}
        self._market_condition = MarketCondition.NORMAL
        self._price_history: Dict[str, List[Decimal]] = {}
        self._volume_history: Dict[str, List[int]] = {}
        
    async def initialize(self) -> None:
        """Initialize the circuit breaker manager."""
        logger.info("Initializing market circuit breaker manager")
        try:
            # Create database tables
            await self._create_tables()
            
            # Load existing circuit breakers
            await self._load_active_breakers()
            
            # Start monitoring tasks
            await self._start_monitoring()
            
            logger.info("Market circuit breaker manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize circuit breaker manager: {e}")
            raise
    
    async def _create_tables(self) -> None:
        """Create database tables for circuit breakers."""
        async with self.db_manager.get_session() as session:
            Base.metadata.create_all(bind=session.get_bind())
    
    async def _load_active_breakers(self) -> None:
        """Load active circuit breakers from database."""
        async with self.db_manager.get_session() as session:
            active_breakers = await session.execute(
                CircuitBreaker.__table__.select().where(
                    CircuitBreaker.triggered == True
                )
            )
            
            for breaker in active_breakers:
                self._active_breakers[breaker.symbol] = CircuitBreakerLevel(breaker.level)
    
    async def _start_monitoring(self) -> None:
        """Start market monitoring tasks."""
        # This would integrate with real market data feeds
        # For now, we'll create placeholder monitoring
        logger.info("Started market monitoring tasks")
    
    async def evaluate_market_volatility(self, market_data: MarketData) -> CircuitBreakerLevel:
        """Evaluate market volatility and return appropriate circuit breaker level."""
        try:
            # Calculate price changes
            if market_data.previous_close > 0:
                price_change_percent = float(
                    (market_data.current_price - market_data.previous_close) / 
                    market_data.previous_close * 100
                )
            else:
                price_change_percent = 0.0
            
            # Store price history
            if market_data.symbol not in self._price_history:
                self._price_history[market_data.symbol] = []
            self._price_history[market_data.symbol].append(market_data.current_price)
            
            # Keep only last 30 minutes of data
            cutoff_time = market_data.timestamp - timedelta(minutes=self.config.volatility_period_minutes)
            # In a real implementation, you'd track timestamps too
            
            # Determine circuit breaker level
            if abs(price_change_percent) >= float(self.config.level_3_threshold * 100):
                return CircuitBreakerLevel.LEVEL_3
            elif abs(price_change_percent) >= float(self.config.level_2_threshold * 100):
                return CircuitBreakerLevel.LEVEL_2
            elif abs(price_change_percent) >= float(self.config.level_1_threshold * 100):
                return CircuitBreakerLevel.LEVEL_1
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error evaluating market volatility for {market_data.symbol}: {e}")
            return None
    
    async def detect_price_gaps(self, market_data: MarketData) -> Optional[PriceGapType]:
        """Detect price gaps and classify their type."""
        try:
            if not market_data.open_price or not market_data.previous_close:
                return None
            
            gap_percent = float(
                (market_data.open_price - market_data.previous_close) / 
                market_data.previous_close * 100
            )
            
            gap_type = None
            if gap_percent >= float(self.config.wide_gap_threshold_percent * 100):
                gap_type = PriceGapType.WIDE_GAP
            elif gap_percent >= float(self.config.gap_threshold_percent * 100):
                gap_type = PriceGapType.UP_GAP
            elif gap_percent <= -float(self.config.wide_gap_threshold_percent * 100):
                gap_type = PriceGapType.WIDE_GAP
            elif gap_percent <= -float(self.config.gap_threshold_percent * 100):
                gap_type = PriceGapType.DOWN_GAP
            
            return gap_type
            
        except Exception as e:
            logger.error(f"Error detecting price gaps for {market_data.symbol}: {e}")
            return None
    
    async def detect_liquidity_crisis(self, market_data: MarketData) -> bool:
        """Detect potential liquidity crisis conditions."""
        try:
            if not all([market_data.bid_price, market_data.ask_price, market_data.bid_size, market_data.ask_size]):
                return False
            
            # Calculate bid-ask spread
            spread = market_data.ask_price - market_data.bid_price
            spread_percent = float(spread / market_data.current_price * 100)
            
            # Check if spread is too wide
            if spread_percent >= float(self.config.min_bid_ask_spread * 100):
                logger.warning(f"Wide spread detected for {market_data.symbol}: {spread_percent:.2f}%")
                return True
            
            # Store volume history
            if market_data.symbol not in self._volume_history:
                self._volume_history[market_data.symbol] = []
            if market_data.volume:
                self._volume_history[market_data.symbol].append(market_data.volume)
                
                # Check volume decline
                if len(self._volume_history[market_data.symbol]) >= 2:
                    recent_volume = self._volume_history[market_data.symbol][-1]
                    previous_volume = self._volume_history[market_data.symbol][-2]
                    
                    if previous_volume > 0:
                        volume_decline = (previous_volume - recent_volume) / previous_volume
                        if volume_decline >= float(self.config.volume_decline_threshold):
                            logger.warning(f"Volume decline detected for {market_data.symbol}: {volume_decline:.2f}%")
                            return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting liquidity crisis for {market_data.symbol}: {e}")
            return False
    
    async def activate_circuit_breaker(self, symbol: str, level: CircuitBreakerLevel, 
                                     trigger_value: float, market_data: MarketData) -> None:
        """Activate a circuit breaker for a symbol."""
        try:
            if symbol in self._active_breakers:
                logger.warning(f"Circuit breaker already active for {symbol}: {self._active_breakers[symbol]}")
                return
            
            self._active_breakers[symbol] = level
            
            # Save to database
            async with self.db_manager.get_session() as session:
                breaker = MarketCircuitBreaker(
                    symbol=symbol,
                    breaker_type="market_volatility",
                    level=level.value,
                    threshold_value=trigger_value,
                    current_value=float(market_data.current_price),
                    triggered_at=datetime.utcnow(),
                    triggered=True,
                    market_condition=level.value
                )
                session.add(breaker)
                
                # Log circuit breaker event
                event = CircuitBreakerEvent(
                    event_type="circuit_breaker_activated",
                    level=level.value,
                    symbol=symbol,
                    trigger_value=trigger_value,
                    current_value=float(market_data.current_price),
                    action_taken="market_halt",
                    triggered_by="volatility_detector"
                )
                session.add(event)
                await session.commit()
            
            # Execute circuit breaker actions
            await self._execute_circuit_breaker_actions(symbol, level, market_data)
            
            logger.warning(f"Circuit breaker activated for {symbol}: Level {level.value}")
            
        except Exception as e:
            logger.error(f"Error activating circuit breaker for {symbol}: {e}")
    
    async def _execute_circuit_breaker_actions(self, symbol: str, level: CircuitBreakerLevel,
                                             market_data: MarketData) -> None:
        """Execute actions based on circuit breaker level."""
        try:
            if level == CircuitBreakerLevel.LEVEL_1:
                # Level 1: 15-minute trading halt
                await self._initiate_trading_halt(symbol, 15)
                self._market_condition = MarketCondition.VOLATILE
                
            elif level == CircuitBreakerLevel.LEVEL_2:
                # Level 2: 15-minute trading halt
                await self._initiate_trading_halt(symbol, 15)
                self._market_condition = MarketCondition.EXTREME_VOLATILITY
                
            elif level == CircuitBreakerLevel.LEVEL_3:
                # Level 3: Trading halted for the day
                await self._initiate_trading_halt(symbol, 1440)  # Full day
                self._market_condition = MarketCondition.HALTED
                
        except Exception as e:
            logger.error(f"Error executing circuit breaker actions for {symbol}: {e}")
    
    async def _initiate_trading_halt(self, symbol: str, duration_minutes: int) -> None:
        """Initiate trading halt for a symbol."""
        try:
            # This would integrate with order management system to halt trading
            logger.critical(f"Trading halt initiated for {symbol} for {duration_minutes} minutes")
            
            # Schedule automatic recovery if enabled
            if self.config.automatic_recovery and duration_minutes < 1440:
                recovery_task = asyncio.create_task(
                    self._schedule_recovery(symbol, duration_minutes)
                )
                
        except Exception as e:
            logger.error(f"Error initiating trading halt for {symbol}: {e}")
    
    async def _schedule_recovery(self, symbol: str, duration_minutes: int) -> None:
        """Schedule automatic circuit breaker recovery."""
        try:
            await asyncio.sleep(duration_minutes * 60)
            await self.reset_circuit_breaker(symbol)
            
        except Exception as e:
            logger.error(f"Error scheduling recovery for {symbol}: {e}")
    
    async def reset_circuit_breaker(self, symbol: str) -> None:
        """Reset circuit breaker for a symbol."""
        try:
            if symbol not in self._active_breakers:
                logger.warning(f"No active circuit breaker to reset for {symbol}")
                return
            
            level = self._active_breakers[symbol]
            del self._active_breakers[symbol]
            
            # Update database
            async with self.db_manager.get_session() as session:
                # Update breaker status
                await session.execute(
                    MarketCircuitBreaker.__table__.update()
                    .where(MarketCircuitBreaker.symbol == symbol)
                    .values(triggered=False, reset_at=datetime.utcnow())
                )
                
                # Log reset event
                event = CircuitBreakerEvent(
                    event_type="circuit_breaker_reset",
                    level=level.value,
                    symbol=symbol,
                    trigger_value=0.0,
                    current_value=0.0,
                    action_taken="automatic_reset",
                    resolved_at=datetime.utcnow()
                )
                session.add(event)
                await session.commit()
            
            logger.info(f"Circuit breaker reset for {symbol}")
            
        except Exception as e:
            logger.error(f"Error resetting circuit breaker for {symbol}: {e}")
    
    async def get_market_condition(self) -> MarketCondition:
        """Get current market condition."""
        return self._market_condition
    
    async def is_trading_halted(self, symbol: str) -> bool:
        """Check if trading is halted for a symbol."""
        return symbol in self._active_breakers
    
    async def get_active_breakers(self) -> Dict[str, CircuitBreakerLevel]:
        """Get all active circuit breakers."""
        return self._active_breakers.copy()
    
    async def process_market_data(self, market_data: MarketData) -> None:
        """Process new market data and evaluate circuit breaker conditions."""
        try:
            # Skip if trading is already halted
            if await self.is_trading_halted(market_data.symbol):
                return
            
            # Evaluate market volatility
            volatility_level = await self.evaluate_market_volatility(market_data)
            if volatility_level:
                await self.activate_circuit_breaker(
                    market_data.symbol, 
                    volatility_level, 
                    float(market_data.current_price),
                    market_data
                )
            
            # Detect price gaps
            gap_type = await self.detect_price_gaps(market_data)
            if gap_type == PriceGapType.WIDE_GAP:
                # Wide gaps trigger circuit breaker
                await self.activate_circuit_breaker(
                    market_data.symbol,
                    CircuitBreakerLevel.CUSTOM,
                    float(market_data.current_price),
                    market_data
                )
            
            # Detect liquidity crisis
            liquidity_crisis = await self.detect_liquidity_crisis(market_data)
            if liquidity_crisis:
                self._market_condition = MarketCondition.LIQUIDITY_CRISIS
                logger.warning(f"Liquidity crisis detected for {market_data.symbol}")
            
        except Exception as e:
            logger.error(f"Error processing market data for {market_data.symbol}: {e}")
    
    async def manual_trigger(self, symbol: str, level: CircuitBreakerLevel, 
                           reason: str) -> None:
        """Manually trigger a circuit breaker."""
        try:
            logger.critical(f"Manual circuit breaker trigger for {symbol}: {reason}")
            
            # Create market data placeholder for manual trigger
            market_data = MarketData(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                current_price=Decimal("0"),
                previous_close=Decimal("0")
            )
            
            await self.activate_circuit_breaker(symbol, level, 0.0, market_data)
            
        except Exception as e:
            logger.error(f"Error manually triggering circuit breaker for {symbol}: {e}")
    
    async def get_circuit_breaker_status(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current circuit breaker status for a symbol."""
        try:
            if symbol not in self._active_breakers:
                return None
            
            level = self._active_breakers[symbol]
            
            return {
                "symbol": symbol,
                "level": level.value,
                "active": True,
                "triggered_at": datetime.utcnow().isoformat(),
                "market_condition": self._market_condition.value
            }
            
        except Exception as e:
            logger.error(f"Error getting circuit breaker status for {symbol}: {e}")
            return None
    
    async def get_market_statistics(self) -> Dict[str, Any]:
        """Get market statistics for circuit breakers."""
        try:
            active_count = len(self._active_breakers)
            condition = self._market_condition.value
            
            return {
                "active_breakers": active_count,
                "market_condition": condition,
                "total_symbols_monitored": len(self._price_history),
                "last_update": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting market statistics: {e}")
            return {}


# Factory function for creating circuit breaker manager
async def create_circuit_breaker_manager(
    db_manager: DatabaseManager,
    config: Optional[CircuitBreakerConfig] = None
) -> MarketCircuitBreakerManager:
    """Create a configured circuit breaker manager."""
    if config is None:
        config = CircuitBreakerConfig()
    
    manager = MarketCircuitBreakerManager(db_manager, config)
    await manager.initialize()
    
    return manager


# Circuit breaker level configurations for different asset classes
STOCK_CIRCUIT_BREAKER_CONFIG = CircuitBreakerConfig(
    level_1_threshold=Decimal("0.07"),  # 7%
    level_2_threshold=Decimal("0.13"),  # 13%
    level_3_threshold=Decimal("0.20"),  # 20%
    gap_threshold_percent=Decimal("0.02"),  # 2%
    wide_gap_threshold_percent=Decimal("0.05")  # 5%
)

FUTURES_CIRCUIT_BREAKER_CONFIG = CircuitBreakerConfig(
    level_1_threshold=Decimal("0.05"),  # 5%
    level_2_threshold=Decimal("0.10"),  # 10%
    level_3_threshold=Decimal("0.15"),  # 15%
    gap_threshold_percent=Decimal("0.015"),  # 1.5%
    wide_gap_threshold_percent=Decimal("0.04")  # 4%
)

CRYPTO_CIRCUIT_BREAKER_CONFIG = CircuitBreakerConfig(
    level_1_threshold=Decimal("0.10"),  # 10%
    level_2_threshold=Decimal("0.20"),  # 20%
    level_3_threshold=Decimal("0.30"),  # 30%
    gap_threshold_percent=Decimal("0.03"),  # 3%
    wide_gap_threshold_percent=Decimal("0.08")  # 8%
)