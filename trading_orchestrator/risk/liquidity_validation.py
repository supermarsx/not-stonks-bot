"""
Liquidity Validation Module

This module implements comprehensive liquidity checks and validation for trading operations,
including market liquidity monitoring, order book depth analysis, and slippage protection.

Features:
- Market liquidity monitoring
- Order book depth analysis
- Slippage protection
- Liquidity scoring system
- Market impact estimation
- Time-weighted average price (TWAP) execution
- Volume-weighted average price (VWAP) execution
- Liquidity risk assessment
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
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


class LiquidityLevel(Enum):
    """Liquidity quality levels."""
    EXCELLENT = "excellent"    # Deep order book, tight spreads
    GOOD = "good"             # Adequate liquidity, moderate spreads
    FAIR = "fair"             # Limited liquidity, wider spreads
    POOR = "poor"             # Thin order book, very wide spreads
    ILLIQUID = "illiquid"     # No meaningful liquidity


class ExecutionStrategy(Enum):
    """Order execution strategies."""
    IMMEDIATE = "immediate"           # Market order, immediate execution
    LIMIT = "limit"                  # Limit order at specified price
    TWAP = "twap"                    # Time-weighted average price
    VWAP = "vwap"                    # Volume-weighted average price
    ICEBERG = "iceberg"              # Iceberg order
    DARK_POOL = "dark_pool"          # Dark pool execution


class SlippageRisk(Enum):
    """Slippage risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class OrderBookLevel:
    """Individual order book level."""
    price: Decimal
    size: int
    order_count: int
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class OrderBook:
    """Complete order book structure."""
    symbol: str
    timestamp: datetime
    bid_levels: List[OrderBookLevel] = field(default_factory=list)
    ask_levels: List[OrderBookLevel] = field(default_factory=list)
    spread: Decimal = Decimal("0")
    mid_price: Decimal = Decimal("0")
    total_bid_size: int = 0
    total_ask_size: int = 0
    book_depth_10: int = 0  # Sum of sizes at top 10 levels
    spread_percent: Decimal = Decimal("0")


@dataclass
class LiquidityMetrics:
    """Liquidity quality metrics."""
    symbol: str
    timestamp: datetime
    spread_bps: Decimal            # Spread in basis points
    mid_price: Decimal            # Mid price
    top_of_book_liquidity: int    # Size at top of book
    book_depth_1_percent: int     # Depth within 1% of mid
    book_depth_5_percent: int     # Depth within 5% of mid
    trade_velocity: Decimal       # Recent trade velocity
    volatility_5min: Decimal      # 5-minute volatility
    liquidity_score: Decimal      # Overall liquidity score (0-100)
    liquidity_level: LiquidityLevel


@dataclass
class MarketImpactEstimate:
    """Market impact estimation."""
    symbol: str
    order_size: Decimal
    execution_strategy: ExecutionStrategy
    estimated_slippage_bps: Decimal
    estimated_market_impact_bps: Decimal
    total_cost_bps: Decimal
    confidence_level: Decimal  # 0-1
    execution_time_minutes: int
    recommendation: str


@dataclass
class LiquidityLimit:
    """Liquidity-based trading limits."""
    symbol: str
    max_order_size: Decimal
    min_spread_bps: Decimal
    min_liquidity_score: Decimal
    max_slippage_risk: SlippageRisk
    required_book_depth: int
    volatility_threshold: Decimal
    enabled: bool = True


class OrderBookModel(Base):
    """Database model for order book snapshots."""
    __tablename__ = "order_book_snapshots"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    bid_price = Column(Float, nullable=False)
    bid_size = Column(Integer, nullable=False)
    ask_price = Column(Float, nullable=False)
    ask_size = Column(Integer, nullable=False)
    spread = Column(Float, nullable=False)
    mid_price = Column(Float, nullable=False)
    total_bid_size = Column(Integer, nullable=False)
    total_ask_size = Column(Integer, nullable=False)
    book_depth = Column(Integer, nullable=False)
    recorded_at = Column(DateTime, server_default=func.now())


class LiquidityMetricModel(Base):
    """Database model for liquidity metrics."""
    __tablename__ = "liquidity_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    spread_bps = Column(Float, nullable=False)
    top_of_book_liquidity = Column(Integer, nullable=False)
    book_depth_1_percent = Column(Integer, nullable=False)
    book_depth_5_percent = Column(Integer, nullable=False)
    trade_velocity = Column(Float, nullable=False)
    volatility_5min = Column(Float, nullable=False)
    liquidity_score = Column(Float, nullable=False)
    liquidity_level = Column(String(20), nullable=False)
    recorded_at = Column(DateTime, server_default=func.now())


class MarketImpactModel(Base):
    """Database model for market impact estimates."""
    __tablename__ = "market_impact_estimates"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    order_size = Column(Float, nullable=False)
    execution_strategy = Column(String(30), nullable=False)
    estimated_slippage_bps = Column(Float, nullable=False)
    estimated_market_impact_bps = Column(Float, nullable=False)
    total_cost_bps = Column(Float, nullable=False)
    confidence_level = Column(Float, nullable=False)
    execution_time_minutes = Column(Integer, nullable=False)
    created_at = Column(DateTime, server_default=func.now())


class LiquidityValidationManager:
    """Manager for liquidity validation and analysis."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self._order_books: Dict[str, OrderBook] = {}
        self._liquidity_limits: Dict[str, LiquidityLimit] = {}
        self._market_data_subscribers: Dict[str, Callable] = {}
        self._liquidity_monitor_task: Optional[asyncio.Task] = None
        
    async def initialize(self) -> None:
        """Initialize the liquidity validation manager."""
        logger.info("Initializing liquidity validation manager")
        try:
            # Create database tables
            await self._create_tables()
            
            # Load liquidity limits
            await self._load_liquidity_limits()
            
            # Start monitoring tasks
            await self._start_monitoring_tasks()
            
            logger.info("Liquidity validation manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize liquidity validation manager: {e}")
            raise
    
    async def _create_tables(self) -> None:
        """Create database tables for liquidity data."""
        async with self.db_manager.get_session() as session:
            Base.metadata.create_all(bind=session.get_bind())
    
    async def _load_liquidity_limits(self) -> None:
        """Load liquidity-based trading limits."""
        await self._initialize_default_limits()
    
    async def _initialize_default_limits(self) -> None:
        """Initialize default liquidity limits."""
        limits = [
            LiquidityLimit(
                symbol="AAPL",
                max_order_size=Decimal("100000"),  # $100k max order
                min_spread_bps=Decimal("5"),       # 5 bps minimum spread
                min_liquidity_score=Decimal("70"), # 70% minimum liquidity score
                max_slippage_risk=SlippageRisk.MEDIUM,
                required_book_depth=1000,
                volatility_threshold=Decimal("0.02")  # 2% volatility threshold
            ),
            LiquidityLimit(
                symbol="TSLA",
                max_order_size=Decimal("50000"),   # $50k max order (higher volatility)
                min_spread_bps=Decimal("10"),      # 10 bps minimum spread
                min_liquidity_score=Decimal("60"), # 60% minimum liquidity score
                max_slippage_risk=SlippageRisk.HIGH,
                required_book_depth=500,
                volatility_threshold=Decimal("0.05")  # 5% volatility threshold
            ),
            LiquidityLimit(
                symbol="SPY",
                max_order_size=Decimal("500000"),  # $500k max order (ETF)
                min_spread_bps=Decimal("2"),       # 2 bps minimum spread
                min_liquidity_score=Decimal("90"), # 90% minimum liquidity score
                max_slippage_risk=SlippageRisk.LOW,
                required_book_depth=5000,
                volatility_threshold=Decimal("0.015")  # 1.5% volatility threshold
            )
        ]
        
        for limit in limits:
            self._liquidity_limits[limit.symbol] = limit
        
        logger.info("Loaded default liquidity limits")
    
    async def _start_monitoring_tasks(self) -> None:
        """Start background monitoring tasks."""
        # Start liquidity monitoring
        self._liquidity_monitor_task = asyncio.create_task(self._liquidity_monitor())
        
        logger.info("Started liquidity monitoring tasks")
    
    async def _liquidity_monitor(self) -> None:
        """Monitor liquidity for all tracked symbols."""
        while True:
            try:
                await asyncio.sleep(5)  # Update every 5 seconds
                
                # This would integrate with real market data feeds
                # For now, we'll simulate order book updates
                for symbol in self._liquidity_limits.keys():
                    await self._simulate_order_book_update(symbol)
                    
            except Exception as e:
                logger.error(f"Error in liquidity monitor: {e}")
    
    async def _simulate_order_book_update(self, symbol: str) -> None:
        """Simulate order book update for testing."""
        try:
            import random
            from decimal import Decimal
            
            # Generate mock order book data
            base_price = Decimal(str(random.uniform(100, 500)))
            
            bid_levels = []
            ask_levels = []
            
            # Generate bid levels (below mid price)
            for i in range(10):
                price = base_price - Decimal(str(random.uniform(0.01, 0.05)))
                size = random.randint(100, 1000)
                bid_levels.append(OrderBookLevel(
                    price=price, size=size, order_count=random.randint(1, 5)
                ))
            
            # Generate ask levels (above mid price)
            for i in range(10):
                price = base_price + Decimal(str(random.uniform(0.01, 0.05)))
                size = random.randint(100, 1000)
                ask_levels.append(OrderBookLevel(
                    price=price, size=size, order_count=random.randint(1, 5)
                ))
            
            # Create order book
            spread = ask_levels[0].price - bid_levels[0].price
            mid_price = (ask_levels[0].price + bid_levels[0].price) / 2
            spread_percent = (spread / mid_price) * 100
            
            order_book = OrderBook(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                bid_levels=sorted(bid_levels, key=lambda x: x.price, reverse=True),
                ask_levels=sorted(ask_levels, key=lambda x: x.price),
                spread=spread,
                mid_price=mid_price,
                total_bid_size=sum(level.size for level in bid_levels),
                total_ask_size=sum(level.size for level in ask_levels),
                book_depth_10=sum(level.size for level in bid_levels[:5] + ask_levels[:5]),
                spread_percent=Decimal(str(spread_percent))
            )
            
            self._order_books[symbol] = order_book
            
            # Save to database
            await self._save_order_book_snapshot(order_book)
            
            # Calculate and save liquidity metrics
            metrics = await self._calculate_liquidity_metrics(order_book)
            await self._save_liquidity_metrics(metrics)
            
        except Exception as e:
            logger.error(f"Error simulating order book update for {symbol}: {e}")
    
    async def _save_order_book_snapshot(self, order_book: OrderBook) -> None:
        """Save order book snapshot to database."""
        try:
            async with self.db_manager.get_session() as session:
                snapshot = OrderBookModel(
                    symbol=order_book.symbol,
                    timestamp=order_book.timestamp,
                    bid_price=float(order_book.bid_levels[0].price) if order_book.bid_levels else 0,
                    bid_size=order_book.bid_levels[0].size if order_book.bid_levels else 0,
                    ask_price=float(order_book.ask_levels[0].price) if order_book.ask_levels else 0,
                    ask_size=order_book.ask_levels[0].size if order_book.ask_levels else 0,
                    spread=float(order_book.spread),
                    mid_price=float(order_book.mid_price),
                    total_bid_size=order_book.total_bid_size,
                    total_ask_size=order_book.total_ask_size,
                    book_depth=order_book.book_depth_10
                )
                session.add(snapshot)
                await session.commit()
        except Exception as e:
            logger.error(f"Error saving order book snapshot: {e}")
    
    async def _calculate_liquidity_metrics(self, order_book: OrderBook) -> LiquidityMetrics:
        """Calculate comprehensive liquidity metrics."""
        try:
            # Calculate spread in basis points
            spread_bps = (order_book.spread / order_book.mid_price) * 10000
            
            # Calculate top of book liquidity (sum of first levels)
            top_bid_size = order_book.bid_levels[0].size if order_book.bid_levels else 0
            top_ask_size = order_book.ask_levels[0].size if order_book.ask_levels else 0
            top_of_book_liquidity = top_bid_size + top_ask_size
            
            # Calculate depth within 1% and 5% of mid price
            price_range_1pct = order_book.mid_price * Decimal("0.01")
            price_range_5pct = order_book.mid_price * Decimal("0.05")
            
            book_depth_1_percent = 0
            book_depth_5_percent = 0
            
            for level in order_book.bid_levels:
                if order_book.mid_price - level.price <= price_range_1pct:
                    book_depth_1_percent += level.size
                if order_book.mid_price - level.price <= price_range_5pct:
                    book_depth_5_percent += level.size
            
            for level in order_book.ask_levels:
                if level.price - order_book.mid_price <= price_range_1pct:
                    book_depth_1_percent += level.size
                if level.price - order_book.mid_price <= price_range_5pct:
                    book_depth_5_percent += level.size
            
            # Calculate trade velocity (simulated)
            trade_velocity = Decimal(str(np.random.uniform(0.1, 2.0)))
            
            # Calculate 5-minute volatility (simulated)
            volatility_5min = Decimal(str(np.random.uniform(0.005, 0.02)))
            
            # Calculate overall liquidity score
            liquidity_score = await self._calculate_liquidity_score(
                spread_bps, top_of_book_liquidity, book_depth_1_percent, 
                book_depth_5_percent, trade_velocity
            )
            
            # Determine liquidity level
            liquidity_level = self._determine_liquidity_level(liquidity_score, spread_bps)
            
            return LiquidityMetrics(
                symbol=order_book.symbol,
                timestamp=order_book.timestamp,
                spread_bps=spread_bps,
                mid_price=order_book.mid_price,
                top_of_book_liquidity=top_of_book_liquidity,
                book_depth_1_percent=book_depth_1_percent,
                book_depth_5_percent=book_depth_5_percent,
                trade_velocity=trade_velocity,
                volatility_5min=volatility_5min,
                liquidity_score=liquidity_score,
                liquidity_level=liquidity_level
            )
            
        except Exception as e:
            logger.error(f"Error calculating liquidity metrics for {order_book.symbol}: {e}")
            raise
    
    async def _calculate_liquidity_score(self, spread_bps: Decimal, top_liquidity: int,
                                       depth_1pct: int, depth_5pct: int, 
                                       velocity: Decimal) -> Decimal:
        """Calculate overall liquidity score (0-100)."""
        try:
            # Spread component (lower spread = higher score)
            spread_score = max(0, 100 - float(spread_bps * 10))  # 10 bps = 10 point deduction
            
            # Top of book liquidity component
            liquidity_score = min(100, top_liquidity / 10)  # 1000 shares = 100 points
            
            # Depth component
            depth_score = min(100, (depth_1pct + depth_5pct) / 20)  # 2000 shares = 100 points
            
            # Velocity component
            velocity_score = min(100, float(velocity * 50))  # 2.0 = 100 points
            
            # Weighted average
            total_score = (spread_score * 0.3 + liquidity_score * 0.25 + 
                          depth_score * 0.25 + velocity_score * 0.2)
            
            return Decimal(str(max(0, min(100, total_score))))
            
        except Exception as e:
            logger.error(f"Error calculating liquidity score: {e}")
            return Decimal("50")  # Default neutral score
    
    def _determine_liquidity_level(self, score: Decimal, spread_bps: Decimal) -> LiquidityLevel:
        """Determine liquidity level based on score and spread."""
        try:
            if float(score) >= 80 and float(spread_bps) <= 5:
                return LiquidityLevel.EXCELLENT
            elif float(score) >= 60 and float(spread_bps) <= 10:
                return LiquidityLevel.GOOD
            elif float(score) >= 40 and float(spread_bps) <= 20:
                return LiquidityLevel.FAIR
            elif float(score) >= 20 and float(spread_bps) <= 50:
                return LiquidityLevel.POOR
            else:
                return LiquidityLevel.ILLIQUID
                
        except Exception as e:
            logger.error(f"Error determining liquidity level: {e}")
            return LiquidityLevel.FAIR
    
    async def _save_liquidity_metrics(self, metrics: LiquidityMetrics) -> None:
        """Save liquidity metrics to database."""
        try:
            async with self.db_manager.get_session() as session:
                metric_record = LiquidityMetricModel(
                    symbol=metrics.symbol,
                    timestamp=metrics.timestamp,
                    spread_bps=float(metrics.spread_bps),
                    top_of_book_liquidity=metrics.top_of_book_liquidity,
                    book_depth_1_percent=metrics.book_depth_1_percent,
                    book_depth_5_percent=metrics.book_depth_5_percent,
                    trade_velocity=float(metrics.trade_velocity),
                    volatility_5min=float(metrics.volatility_5min),
                    liquidity_score=float(metrics.liquidity_score),
                    liquidity_level=metrics.liquidity_level.value
                )
                session.add(metric_record)
                await session.commit()
        except Exception as e:
            logger.error(f"Error saving liquidity metrics: {e}")
    
    async def validate_liquidity(self, symbol: str, order_size: Decimal, 
                               execution_strategy: ExecutionStrategy) -> Dict[str, Any]:
        """Validate if order size is appropriate for current liquidity."""
        try:
            order_book = self._order_books.get(symbol)
            limit = self._liquidity_limits.get(symbol)
            
            if not order_book or not limit:
                return {
                    "valid": False,
                    "reason": "No order book or liquidity limit data available",
                    "risk_level": "unknown"
                }
            
            validation_result = {
                "valid": True,
                "warnings": [],
                "recommendations": [],
                "risk_level": "low"
            }
            
            # Check maximum order size
            if order_size > limit.max_order_size:
                validation_result["valid"] = False
                validation_result["warnings"].append(
                    f"Order size ${order_size} exceeds maximum ${limit.max_order_size}"
                )
                validation_result["risk_level"] = "high"
            
            # Check minimum spread
            spread_bps = (order_book.spread / order_book.mid_price) * 10000
            if spread_bps > limit.min_spread_bps:
                validation_result["warnings"].append(
                    f"Spread {spread_bps:.1f} bps exceeds minimum {limit.min_spread_bps} bps"
                )
                validation_result["risk_level"] = "medium"
            
            # Get latest liquidity metrics
            latest_metrics = await self._get_latest_liquidity_metrics(symbol)
            if latest_metrics:
                # Check liquidity score
                if latest_metrics.liquidity_score < limit.min_liquidity_score:
                    validation_result["warnings"].append(
                        f"Liquidity score {latest_metrics.liquidity_score:.1f} below minimum {limit.min_liquidity_score}"
                    )
                    validation_result["risk_level"] = "high"
                
                # Check volatility
                if latest_metrics.volatility_5min > limit.volatility_threshold:
                    validation_result["warnings"].append(
                        f"Volatility {latest_metrics.volatility_5min:.2%} exceeds threshold {limit.volatility_threshold:.2%}"
                    )
                    validation_result["risk_level"] = "high"
                
                # Check book depth
                if latest_metrics.top_of_book_liquidity < limit.required_book_depth:
                    validation_result["warnings"].append(
                        f"Top of book liquidity {latest_metrics.top_of_book_liquidity} below required {limit.required_book_depth}"
                    )
                    validation_result["risk_level"] = "high"
            
            # Provide recommendations based on validation results
            if validation_result["risk_level"] == "high":
                validation_result["recommendations"].append("Consider reducing order size")
                validation_result["recommendations"].append("Consider using TWAP execution strategy")
            elif validation_result["risk_level"] == "medium":
                validation_result["recommendations"].append("Consider using limit orders")
                validation_result["recommendations"].append("Monitor market conditions closely")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating liquidity for {symbol}: {e}")
            return {
                "valid": False,
                "reason": f"Validation error: {str(e)}",
                "risk_level": "high"
            }
    
    async def estimate_market_impact(self, symbol: str, order_size: Decimal,
                                   execution_strategy: ExecutionStrategy) -> MarketImpactEstimate:
        """Estimate market impact for an order."""
        try:
            # Get current market data
            order_book = self._order_books.get(symbol)
            if not order_book:
                raise ValueError(f"No order book data available for {symbol}")
            
            # Get latest liquidity metrics
            latest_metrics = await self._get_latest_liquidity_metrics(symbol)
            if not latest_metrics:
                raise ValueError(f"No liquidity metrics available for {symbol}")
            
            # Calculate estimated costs based on execution strategy
            if execution_strategy == ExecutionStrategy.IMMEDIATE:
                # Market order - higher slippage
                estimated_slippage = float(latest_metrics.spread_bps) * 1.5
                market_impact = self._estimate_market_impact_market(order_size, latest_metrics)
                execution_time = 1
                
            elif execution_strategy == ExecutionStrategy.TWAP:
                # TWAP - lower impact, longer execution time
                estimated_slippage = float(latest_metrics.spread_bps) * 0.3
                market_impact = self._estimate_market_impact_twap(order_size, latest_metrics)
                execution_time = 30  # 30 minutes
                
            elif execution_strategy == ExecutionStrategy.VWAP:
                # VWAP - moderate impact and execution time
                estimated_slippage = float(latest_metrics.spread_bps) * 0.5
                market_impact = self._estimate_market_impact_vwap(order_size, latest_metrics)
                execution_time = 15  # 15 minutes
                
            else:
                # Limit order
                estimated_slippage = float(latest_metrics.spread_bps) * 0.1
                market_impact = Decimal("0")
                execution_time = 5
            
            total_cost_bps = estimated_slippage + float(market_impact)
            
            # Calculate confidence level based on liquidity
            confidence = min(0.95, float(latest_metrics.liquidity_score) / 100)
            
            # Generate recommendation
            recommendation = self._generate_impact_recommendation(
                total_cost_bps, latest_metrics.liquidity_score, execution_strategy
            )
            
            estimate = MarketImpactEstimate(
                symbol=symbol,
                order_size=order_size,
                execution_strategy=execution_strategy,
                estimated_slippage_bps=Decimal(str(estimated_slippage)),
                estimated_market_impact_bps=market_impact,
                total_cost_bps=Decimal(str(total_cost_bps)),
                confidence_level=Decimal(str(confidence)),
                execution_time_minutes=execution_time,
                recommendation=recommendation
            )
            
            # Save estimate to database
            await self._save_market_impact_estimate(estimate)
            
            return estimate
            
        except Exception as e:
            logger.error(f"Error estimating market impact for {symbol}: {e}")
            raise
    
    def _estimate_market_impact_market(self, order_size: Decimal, 
                                     metrics: LiquidityMetrics) -> Decimal:
        """Estimate market impact for market order."""
        # Simplified market impact model
        # Market impact increases with order size and decreases with liquidity
        
        base_impact = Decimal("2")  # 2 bps base impact
        size_factor = order_size / Decimal("10000")  # Impact per $10k
        liquidity_factor = Decimal("100") / metrics.liquidity_score  # Higher score = lower impact
        
        impact = base_impact * size_factor * liquidity_factor
        return min(impact, Decimal("50"))  # Cap at 50 bps
    
    def _estimate_market_impact_twap(self, order_size: Decimal,
                                   metrics: LiquidityMetrics) -> Decimal:
        """Estimate market impact for TWAP execution."""
        # TWAP spreads impact over time
        base_impact = Decimal("0.5")  # Lower base impact
        size_factor = order_size / Decimal("10000")
        liquidity_factor = Decimal("100") / metrics.liquidity_score
        
        impact = base_impact * size_factor * liquidity_factor
        return min(impact, Decimal("10"))  # Cap at 10 bps
    
    def _estimate_market_impact_vwap(self, order_size: Decimal,
                                   metrics: LiquidityMetrics) -> Decimal:
        """Estimate market impact for VWAP execution."""
        base_impact = Decimal("1")  # Moderate base impact
        size_factor = order_size / Decimal("10000")
        liquidity_factor = Decimal("100") / metrics.liquidity_score
        
        impact = base_impact * size_factor * liquidity_factor
        return min(impact, Decimal("25"))  # Cap at 25 bps
    
    def _generate_impact_recommendation(self, total_cost_bps: float, 
                                      liquidity_score: Decimal,
                                      strategy: ExecutionStrategy) -> str:
        """Generate recommendation based on market impact estimate."""
        if total_cost_bps > 20:
            return "Consider reducing order size or using passive execution"
        elif total_cost_bps > 10:
            return "Monitor execution closely, consider TWAP for large orders"
        elif total_cost_bps > 5:
            return "Standard execution acceptable, monitor slippage"
        else:
            return "Low impact execution, proceed with confidence"
    
    async def _save_market_impact_estimate(self, estimate: MarketImpactEstimate) -> None:
        """Save market impact estimate to database."""
        try:
            async with self.db_manager.get_session() as session:
                impact_record = MarketImpactModel(
                    symbol=estimate.symbol,
                    order_size=float(estimate.order_size),
                    execution_strategy=estimate.execution_strategy.value,
                    estimated_slippage_bps=float(estimate.estimated_slippage_bps),
                    estimated_market_impact_bps=float(estimate.estimated_market_impact_bps),
                    total_cost_bps=float(estimate.total_cost_bps),
                    confidence_level=float(estimate.confidence_level),
                    execution_time_minutes=estimate.execution_time_minutes
                )
                session.add(impact_record)
                await session.commit()
        except Exception as e:
            logger.error(f"Error saving market impact estimate: {e}")
    
    async def _get_latest_liquidity_metrics(self, symbol: str) -> Optional[LiquidityMetrics]:
        """Get the latest liquidity metrics for a symbol."""
        try:
            async with self.db_manager.get_session() as session:
                result = await session.execute(
                    LiquidityMetricModel.__table__.select()
                    .where(LiquidityMetricModel.symbol == symbol)
                    .order_by(LiquidityMetricModel.timestamp.desc())
                    .limit(1)
                )
                
                record = result.fetchone()
                if not record:
                    return None
                
                return LiquidityMetrics(
                    symbol=record.symbol,
                    timestamp=record.timestamp,
                    spread_bps=Decimal(str(record.spread_bps)),
                    mid_price=Decimal("0"),  # Not stored in simplified model
                    top_of_book_liquidity=record.top_of_book_liquidity,
                    book_depth_1_percent=record.book_depth_1_percent,
                    book_depth_5_percent=record.book_depth_5_percent,
                    trade_velocity=Decimal(str(record.trade_velocity)),
                    volatility_5min=Decimal(str(record.volatility_5min)),
                    liquidity_score=Decimal(str(record.liquidity_score)),
                    liquidity_level=LiquidityLevel(record.liquidity_level)
                )
                
        except Exception as e:
            logger.error(f"Error getting latest liquidity metrics for {symbol}: {e}")
            return None
    
    async def get_liquidity_summary(self, symbol: str) -> Dict[str, Any]:
        """Get current liquidity summary for a symbol."""
        try:
            order_book = self._order_books.get(symbol)
            if not order_book:
                return {"error": "No order book data available"}
            
            latest_metrics = await self._get_latest_liquidity_metrics(symbol)
            
            return {
                "symbol": symbol,
                "timestamp": order_book.timestamp.isoformat(),
                "mid_price": float(order_book.mid_price),
                "spread": float(order_book.spread),
                "spread_percent": float(order_book.spread_percent),
                "total_bid_size": order_book.total_bid_size,
                "total_ask_size": order_book.total_ask_size,
                "top_bid": {
                    "price": float(order_book.bid_levels[0].price) if order_book.bid_levels else 0,
                    "size": order_book.bid_levels[0].size if order_book.bid_levels else 0
                },
                "top_ask": {
                    "price": float(order_book.ask_levels[0].price) if order_book.ask_levels else 0,
                    "size": order_book.ask_levels[0].size if order_book.ask_levels else 0
                },
                "liquidity_metrics": {
                    "liquidity_score": float(latest_metrics.liquidity_score) if latest_metrics else 0,
                    "liquidity_level": latest_metrics.liquidity_level.value if latest_metrics else "unknown",
                    "spread_bps": float(latest_metrics.spread_bps) if latest_metrics else 0,
                    "top_of_book_liquidity": latest_metrics.top_of_book_liquidity if latest_metrics else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting liquidity summary for {symbol}: {e}")
            return {"error": str(e)}
    
    async def get_liquidity_status(self) -> Dict[str, Any]:
        """Get overall liquidity validation system status."""
        try:
            tracked_symbols = len(self._order_books)
            active_limits = len([l for l in self._liquidity_limits.values() if l.enabled])
            
            # Calculate average liquidity score across all symbols
            total_score = Decimal("0")
            valid_symbols = 0
            
            for symbol in self._order_books.keys():
                metrics = await self._get_latest_liquidity_metrics(symbol)
                if metrics:
                    total_score += metrics.liquidity_score
                    valid_symbols += 1
            
            avg_liquidity_score = float(total_score / valid_symbols) if valid_symbols > 0 else 0
            
            return {
                "tracked_symbols": tracked_symbols,
                "active_limits": active_limits,
                "average_liquidity_score": avg_liquidity_score,
                "order_books": len(self._order_books),
                "monitoring_active": self._liquidity_monitor_task is not None,
                "last_update": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting liquidity status: {e}")
            return {}


# Factory function for creating liquidity validation manager
async def create_liquidity_manager(db_manager: DatabaseManager) -> LiquidityValidationManager:
    """Create a configured liquidity validation manager."""
    manager = LiquidityValidationManager(db_manager)
    await manager.initialize()
    
    return manager


# Predefined liquidity limits for common instruments
HIGH_LIQUIDITY_LIMITS = {
    "SPY": LiquidityLimit(
        symbol="SPY",
        max_order_size=Decimal("1000000"),
        min_spread_bps=Decimal("2"),
        min_liquidity_score=Decimal("85"),
        max_slippage_risk=SlippageRisk.LOW,
        required_book_depth=10000,
        volatility_threshold=Decimal("0.01")
    ),
    "QQQ": LiquidityLimit(
        symbol="QQQ",
        max_order_size=Decimal("500000"),
        min_spread_bps=Decimal("3"),
        min_liquidity_score=Decimal("80"),
        max_slippage_risk=SlippageRisk.LOW,
        required_book_depth=5000,
        volatility_threshold=Decimal("0.015")
    )
}

MEDIUM_LIQUIDITY_LIMITS = {
    "AAPL": LiquidityLimit(
        symbol="AAPL",
        max_order_size=Decimal("250000"),
        min_spread_bps=Decimal("5"),
        min_liquidity_score=Decimal("70"),
        max_slippage_risk=SlippageRisk.MEDIUM,
        required_book_depth=2000,
        volatility_threshold=Decimal("0.02")
    ),
    "MSFT": LiquidityLimit(
        symbol="MSFT",
        max_order_size=Decimal("200000"),
        min_spread_bps=Decimal("4"),
        min_liquidity_score=Decimal("75"),
        max_slippage_risk=SlippageRisk.MEDIUM,
        required_book_depth=1500,
        volatility_threshold=Decimal("0.02")
    )
}