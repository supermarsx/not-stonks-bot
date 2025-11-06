"""
@file base.py
@brief Trading Strategies Framework - Base Classes and Interfaces

@details
This module provides the foundational classes and interfaces for implementing
trading strategies in the orchestrator system. It defines abstract base classes,
data structures, and common utilities that all trading strategies must implement.

Key Features:
- Abstract base classes for strategy implementation
- Trading signal types and strategy status enums
- Signal generation and execution framework
- Strategy performance tracking
- Position and order management integration
- Strategy lifecycle management

@author Trading Orchestrator System
@version 1.0
@date 2025-11-06

@warning
Trading strategies can execute real trades. Always test strategies thoroughly
in paper trading mode before live deployment.

@note
This module defines the core framework that all trading strategies must follow:
- BaseStrategy: Abstract base for all strategies
- SignalType: Buy/sell/hold signal enums
- StrategyType: Strategy category definitions
- StrategyStatus: Execution state tracking

@see strategies.trend_following for trend following implementation
@see strategies.mean_reversion for mean reversion strategy
@see strategies.pairs_trading for pairs trading strategy
@see strategies.arbitrage for arbitrage strategy
"""

from typing import Dict, Any, List, Optional, Union, Protocol, Generic, TypeVar
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import uuid
from abc import ABC, abstractmethod

from loguru import logger

from trading.models import Order, OrderSide, OrderType, TimeInForce
from trading.database import get_db_session


class StrategyType(Enum):
    """
    @enum StrategyType
    @brief Trading strategy categories
    
    @details
    Enumerates the different types of trading strategies available in the system.
    Each strategy type represents a distinct trading approach with specific
    entry/exit criteria and risk characteristics.
    
    @par Strategy Categories:
    - TREND_FOLLOWING: Identifies and trades market trends
    - MEAN_REVERSION: Exploits price deviations from historical averages
    - PAIRS_TRADING: Statistical arbitrage between correlated instruments
    - ARBITRAGE: Risk-free profit opportunities across markets
    - MOMENTUM: Trades based on price momentum indicators
    - BREAKOUT: Identifies breakouts from consolidation ranges
    - SCALPING: Short-term trades for small profits
    - SWING_TRADING: Medium-term trades capturing price swings
    
    @warning
    Each strategy type has different risk profiles and performance
    characteristics. Choose appropriate strategies for your risk tolerance.
    """
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    PAIRS_TRADING = "pairs_trading"
    ARBITRAGE = "arbitrage"
    MOMENTUM = "momentum"
    BREAKOUT = "breakout"
    SCALPING = "scalping"
    SWING_TRADING = "swing_trading"


class StrategyStatus(Enum):
    """
    @enum StrategyStatus
    @brief Strategy execution states
    
    @details
    Tracks the current execution state of a trading strategy throughout
    its lifecycle from initialization to completion or error conditions.
    
    @par Status Flow:
    1. INITIALIZING: Strategy setup and configuration
    2. RUNNING: Active strategy execution
    3. PAUSED: Temporary pause (manual or automatic)
    4. STOPPED: Normal completion or manual stop
    5. ERROR: Unexpected error condition
    
    @note
    Status transitions are managed by the strategy execution framework
    and should not be set manually during normal operation.
    """
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class SignalType(Enum):
    """
    @enum SignalType
    @brief Trading signal action types
    
    @details
    Defines the different types of trading signals that strategies can generate.
    Each signal type represents a specific trading action or recommendation.
    
    @par Signal Actions:
    - BUY: Initiate long position or increase existing position
    - SELL: Close long position or initiate short position
    - HOLD: Maintain current position, no action required
    - CLOSE: Exit position completely
    - REDUCE: Decrease position size while maintaining direction
    - INCREASE: Add to existing position in same direction
    
    @note
    Signal interpretation may vary by strategy type. Always consult
    strategy-specific documentation for precise signal meanings.
    
    @warning
    All signals should be validated by the risk management system
    before execution to ensure compliance with risk limits.
    """
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"
    REDUCE = "reduce"
    INCREASE = "increase"


class RiskLevel(Enum):
    """
    @enum RiskLevel
    @brief Risk assessment levels for strategies
    
    @details
    Categorizes trading strategies and trades by their inherent risk level
    for portfolio management and risk control purposes.
    
    @par Risk Categories:
    - LOW: Conservative strategies with minimal risk exposure
    - MEDIUM: Moderate risk strategies with balanced risk/reward
    - HIGH: Aggressive strategies with higher risk potential
    - VERY_HIGH: Highest risk strategies requiring special monitoring
    
    @note
    Risk levels should be assigned based on historical volatility,
    drawdown potential, and position sizing characteristics.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class TradingSignal:
    """
    @class TradingSignal
    @brief Trading signal generated by strategy
    
    @details
    Represents a trading signal generated by a strategy, containing all
    necessary information for trade execution including price, quantity,
    risk management parameters, and confidence metrics.
    
    @par Signal Components:
    - signal_id: Unique identifier for tracking
    - strategy_id: Strategy that generated the signal
    - symbol: Target trading instrument
    - signal_type: Action to take (BUY/SELL/HOLD/etc)
    - confidence: Confidence level (0.0-1.0)
    - strength: Signal strength (0.0-1.0)
    - price: Target price for execution
    - quantity: Position size to trade
    - stop_loss: Risk management stop loss level
    - take_profit: Target profit level
    - time_horizon: Expected signal duration
    - metadata: Strategy-specific data
    - created_at: Signal generation timestamp
    - expires_at: Optional expiration time
    
    @warning
    All signals should be validated by risk management before execution
    to ensure compliance with portfolio limits and risk parameters.
    
    @par Example:
    @code
    signal = TradingSignal(
        signal_id="sig_123",
        strategy_id="trend_001",
        symbol="AAPL",
        signal_type=SignalType.BUY,
        confidence=0.85,
        strength=0.72,
        price=Decimal('150.00'),
        quantity=Decimal('100'),
        stop_loss=Decimal('145.00'),
        take_profit=Decimal('160.00'),
        time_horizon=timedelta(hours=4)
    )
    @endcode
    
    @note
    Signal validation and execution are handled by the strategy framework
    and risk management system.
    """
    signal_id: str
    strategy_id: str
    symbol: str
    signal_type: SignalType
    confidence: float  # 0.0 to 1.0
    strength: float    # 0.0 to 1.0
    price: Decimal
    quantity: Decimal
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    time_horizon: Optional[timedelta] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None


@dataclass
class StrategyMetrics:
    """
    @class StrategyMetrics
    @brief Performance metrics for trading strategy
    
    @details
    Tracks comprehensive performance metrics for a trading strategy
    including profitability, risk measures, and execution statistics.
    
    @par Core Metrics:
    - total_trades: Total number of trades executed
    - winning_trades: Number of profitable trades
    - losing_trades: Number of losing trades
    - win_rate: Percentage of winning trades
    - total_return: Cumulative return percentage
    - sharpe_ratio: Risk-adjusted return measure
    - max_drawdown: Maximum peak-to-trough decline
    - profit_factor: Gross profit / Gross loss ratio
    - average_trade: Mean profit/loss per trade
    - largest_win: Biggest single winning trade
    - largest_loss: Biggest single losing trade
    
    @par Streak Tracking:
    - consecutive_wins: Current winning streak count
    - consecutive_losses: Current losing streak count
    
    @par Example:
    @code
    metrics = StrategyMetrics(
        strategy_id="momentum_001",
        total_trades=150,
        winning_trades=90,
        losing_trades=60,
        win_rate=0.60,
        total_return=12.5,
        sharpe_ratio=1.8,
        max_drawdown=-8.2
    )
    @endcode
    
    @note
    Metrics are updated automatically by the strategy framework
    during execution and can be used for performance analysis.
    """
    strategy_id: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    average_trade: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class StrategyConfig:
    """
    @class StrategyConfig
    @brief Configuration parameters for trading strategy
    
    @details
    Contains all configuration parameters needed to initialize and
    configure a trading strategy including risk limits, trading parameters,
    and operational settings.
    
    @par Required Fields:
    - strategy_id: Unique identifier
    - strategy_type: Category of strategy
    - name: Human-readable name
    - description: Detailed description
    - parameters: Strategy-specific parameters
    - risk_level: Risk category
    
    @par Risk Limits:
    - max_position_size: Maximum position size in currency
    - max_daily_loss: Maximum allowed daily loss
    - max_trades_per_day: Trading frequency limit
    
    @par Operational Settings:
    - symbols: List of symbols to trade
    - enabled: Strategy enabled status
    - created_at: Configuration creation time
    
    @par Example:
    @code
    config = StrategyConfig(
        strategy_id="trend_001",
        strategy_type=StrategyType.TREND_FOLLOWING,
        name="Momentum Trend Strategy",
        description="Long-only trend following strategy",
        parameters={
            "fast_period": 12,
            "slow_period": 26,
            "signal_threshold": 0.02,
            "trailing_stop": 0.05
        },
        risk_level=RiskLevel.MEDIUM,
        max_position_size=Decimal('100000'),
        max_daily_loss=Decimal('5000'),
        max_trades_per_day=20,
        symbols=['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    )
    @endcode
    
    @warning
    Risk parameters should be carefully set based on portfolio size
    and risk tolerance. Incorrect settings can lead to significant losses.
    
    @note
    Configuration changes require strategy restart to take effect.
    """
    strategy_id: str
    strategy_type: StrategyType
    name: str
    description: str
    parameters: Dict[str, Any]
    risk_level: RiskLevel
    max_position_size: Decimal = Decimal('100000')
    max_daily_loss: Decimal = Decimal('10000')
    max_trades_per_day: int = 50
    symbols: List[str] = field(default_factory=list)
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)


class StrategyContext(Protocol):
    """Context interface for strategy execution"""
    
    async def get_market_data(self, symbol: str, timeframe: str) -> List[Dict[str, Any]]:
        """Get market data for symbol"""
        ...
    
    async def get_current_price(self, symbol: str) -> Decimal:
        """Get current price for symbol"""
        ...
    
    async def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        order_type: str,
        price: Optional[Decimal] = None
    ) -> Dict[str, Any]:
        """Submit order"""
        ...
    
    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get current positions"""
        ...
    
    async def get_portfolio_value(self) -> Decimal:
        """Get current portfolio value"""
        ...
    
    def log_message(self, level: str, message: str, **kwargs):
        """Log a message"""
        ...


class BaseStrategy(ABC):
    """
    @class BaseStrategy
    @brief Abstract Base Class for All Trading Strategies
    
    @details
    Provides the foundational framework for implementing trading strategies
    in the orchestrator system. This abstract class defines the interface
    and common functionality that all concrete strategy implementations must provide.
    
    @par Core Functionality:
    - <b>Signal Generation</b>: Abstract method for strategy-specific signal creation
    - <b>Signal Validation</b>: Abstract method for signal quality assessment
    - <b>Risk Management</b>: Integration with portfolio and position limits
    - <b>Performance Tracking</b>: Real-time metrics and performance monitoring
    - <b>Execution Context</b>: Integration with order management system
    - <b>Lifecycle Management</b>: Start, pause, resume, and stop operations
    
    @par Strategy Lifecycle:
    1. Initialization: Load configuration and set up context
    2. Validation: Verify strategy parameters and constraints
    3. Execution: Generate and process trading signals
    4. Monitoring: Track performance and risk metrics
    5. Shutdown: Clean up resources and finalize metrics
    
    @par Signal Processing Pipeline:
    1. Generate signals using strategy-specific logic
    2. Validate signals against quality thresholds
    3. Check risk compliance and limits
    4. Submit orders through execution context
    5. Track execution results and update metrics
    
    @par Performance Metrics:
    - Real-time P&L calculation
    - Win rate and profit factor tracking
    - Drawdown monitoring and alerting
    - Trade frequency and execution quality
    
    @warning
    All concrete strategy implementations must properly handle:
    - Risk limits and position sizing
    - Market data validation and error handling
    - Signal expiration and cleanup
    - Resource management and cleanup
    
    @par Usage Example:
    @code
    # Create strategy configuration
    config = StrategyConfig(
        strategy_id="momentum_001",
        strategy_type=StrategyType.MOMENTUM,
        name="Momentum Strategy",
        description="Trend following momentum strategy",
        parameters={"period": 20, "threshold": 0.02},
        risk_level=RiskLevel.MEDIUM
    )
    
    # Initialize strategy
    strategy = MomentumStrategy(config)
    
    # Set execution context
    strategy.set_context(context)
    
    # Start strategy execution
    await strategy.run()
    
    # Monitor strategy status
    status = await strategy.get_status()
    print(f"Status: {status['status']}")
    print(f"Trades: {status['metrics']['total_trades']}")
    
    # Pause strategy
    await strategy.pause()
    
    # Resume strategy
    await strategy.resume()
    
    # Stop strategy
    await strategy.stop()
    @endcode
    
    @note
    This class provides the core framework. Concrete implementations
    must override the abstract methods to provide strategy-specific logic.
    
    @see TradingSignal for signal structure
    @see StrategyConfig for configuration details
    @see StrategyContext for execution context interface
    @see StrategyMetrics for performance tracking
    """
    
    def __init__(self, config: StrategyConfig):
        """
        @brief Initialize base strategy with configuration
        
        @param config StrategyConfig containing all strategy parameters
        
        @details
        Initializes the strategy with provided configuration and sets up
        internal state including metrics tracking, signal management,
        and performance monitoring.
        
        @par Initialization Process:
        1. Store strategy configuration
        2. Set initial status to INITIALIZING
        3. Initialize metrics tracking
        4. Set up signal and trade tracking
        5. Record start time for runtime calculations
        
        @throws ValueError if configuration is invalid
        
        @par Example:
        @code
        config = StrategyConfig(...)
        strategy = ConcreteStrategy(config)
        # Strategy is now initialized but not running
        @endcode
        
        @note
        Strategy must be started explicitly using the run() method
        or through the strategy registry.
        """
        self.config = config
        self.status = StrategyStatus.INITIALIZING
        self.context: Optional[StrategyContext] = None
        
        # Metrics and tracking
        self.metrics = StrategyMetrics(strategy_id=config.strategy_id)
        self.active_signals: Dict[str, TradingSignal] = {}
        self.trade_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.start_time = datetime.utcnow()
        self.last_signal_time = None
        self.total_signals_generated = 0
        self.total_orders_submitted = 0
        
        logger.info(f"Strategy initialized: {config.name} ({config.strategy_type.value})")
    
    @abstractmethod
    async def generate_signals(self) -> List[TradingSignal]:
        """
        Generate trading signals based on strategy logic
        
        Returns:
            List of trading signals
        """
        pass
    
    @abstractmethod
    async def validate_signal(self, signal: TradingSignal) -> bool:
        """
        Validate a trading signal
        
        Args:
            signal: Signal to validate
            
        Returns:
            True if signal is valid
        """
        pass
    
    def set_context(self, context: StrategyContext):
        """Set strategy execution context"""
        self.context = context
        logger.info(f"Context set for strategy: {self.config.name}")
    
    async def run(self):
        """Run the strategy"""
        try:
            self.status = StrategyStatus.RUNNING
            logger.info(f"Strategy started: {self.config.name}")
            
            while self.status == StrategyStatus.RUNNING:
                try:
                    # Generate signals
                    signals = await self.generate_signals()
                    
                    for signal in signals:
                        # Validate signal
                        if await self.validate_signal(signal):
                            await self.process_signal(signal)
                        else:
                            logger.warning(f"Signal rejected: {signal.signal_id}")
                    
                    # Update metrics
                    await self.update_metrics()
                    
                    # Sleep before next iteration
                    await asyncio.sleep(1)  # Adjust based on strategy requirements
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in strategy loop: {e}")
                    self.status = StrategyStatus.ERROR
                    break
            
            logger.info(f"Strategy stopped: {self.config.name}")
            
        except Exception as e:
            logger.error(f"Error running strategy: {e}")
            self.status = StrategyStatus.ERROR
    
    async def process_signal(self, signal: TradingSignal):
        """
        Process a validated trading signal
        
        Args:
            signal: Signal to process
        """
        try:
            # Store signal
            self.active_signals[signal.signal_id] = signal
            self.total_signals_generated += 1
            self.last_signal_time = datetime.utcnow()
            
            # Submit order if context available
            if self.context:
                order_result = await self.context.submit_order(
                    symbol=signal.symbol,
                    side=signal.signal_type.value,
                    quantity=signal.quantity,
                    order_type=OrderType.MARKET.value
                )
                
                if order_result.get('success'):
                    self.total_orders_submitted += 1
                    logger.info(f"Order submitted: {signal.signal_id}")
                else:
                    logger.warning(f"Order failed: {order_result.get('reason')}")
            
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
    
    async def update_metrics(self):
        """Update strategy metrics"""
        try:
            if not self.context:
                return
            
            # Get portfolio performance
            portfolio_value = await self.context.get_portfolio_value()
            
            # Calculate basic metrics
            if self.trade_history:
                winning_trades = len([t for t in self.trade_history if t.get('pnl', 0) > 0])
                losing_trades = len([t for t in self.trade_history if t.get('pnl', 0) < 0])
                
                self.metrics.total_trades = len(self.trade_history)
                self.metrics.winning_trades = winning_trades
                self.metrics.losing_trades = losing_trades
                self.metrics.win_rate = winning_trades / max(len(self.trade_history), 1)
                
                total_pnl = sum(t.get('pnl', 0) for t in self.trade_history)
                self.metrics.total_return = total_pnl
                
                if self.trade_history:
                    self.metrics.average_trade = total_pnl / len(self.trade_history)
            
            self.metrics.last_updated = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    async def pause(self):
        """Pause strategy execution"""
        if self.status == StrategyStatus.RUNNING:
            self.status = StrategyStatus.PAUSED
            logger.info(f"Strategy paused: {self.config.name}")
    
    async def resume(self):
        """Resume strategy execution"""
        if self.status == StrategyStatus.PAUSED:
            self.status = StrategyStatus.RUNNING
            logger.info(f"Strategy resumed: {self.config.name}")
    
    async def stop(self):
        """Stop strategy execution"""
        self.status = StrategyStatus.STOPPED
        logger.info(f"Strategy stopped: {self.config.name}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get strategy status"""
        runtime = datetime.utcnow() - self.start_time
        
        return {
            'strategy_id': self.config.strategy_id,
            'name': self.config.name,
            'type': self.config.strategy_type.value,
            'status': self.status.value,
            'risk_level': self.config.risk_level.value,
            'runtime_seconds': runtime.total_seconds(),
            'total_signals': self.total_signals_generated,
            'total_orders': self.total_orders_submitted,
            'active_signals': len(self.active_signals),
            'last_signal': self.last_signal_time.isoformat() if self.last_signal_time else None,
            'metrics': {
                'total_trades': self.metrics.total_trades,
                'win_rate': self.metrics.win_rate,
                'total_return': self.metrics.total_return,
                'sharpe_ratio': self.metrics.sharpe_ratio,
                'max_drawdown': self.metrics.max_drawdown
            }
        }
    
    async def cleanup_expired_signals(self):
        """Clean up expired signals"""
        current_time = datetime.utcnow()
        expired_signals = [
            signal_id for signal_id, signal in self.active_signals.items()
            if signal.expires_at and signal.expires_at <= current_time
        ]
        
        for signal_id in expired_signals:
            del self.active_signals[signal_id]
        
        if expired_signals:
            logger.debug(f"Cleaned up {len(expired_signals)} expired signals")


class StrategyRegistry:
    """Registry for managing trading strategies"""
    
    def __init__(self):
        self.strategies: Dict[str, BaseStrategy] = {}
        self.configs: Dict[str, StrategyConfig] = {}
    
    def register_strategy(self, strategy: BaseStrategy):
        """Register a strategy"""
        self.strategies[strategy.config.strategy_id] = strategy
        self.configs[strategy.config.strategy_id] = strategy.config
        logger.info(f"Strategy registered: {strategy.config.name}")
    
    def get_strategy(self, strategy_id: str) -> Optional[BaseStrategy]:
        """Get strategy by ID"""
        return self.strategies.get(strategy_id)
    
    def get_all_strategies(self) -> List[BaseStrategy]:
        """Get all registered strategies"""
        return list(self.strategies.values())
    
    def get_strategies_by_type(self, strategy_type: StrategyType) -> List[BaseStrategy]:
        """Get strategies by type"""
        return [
            strategy for strategy in self.strategies.values()
            if strategy.config.strategy_type == strategy_type
        ]
    
    def get_active_strategies(self) -> List[BaseStrategy]:
        """Get currently running strategies"""
        return [
            strategy for strategy in self.strategies.values()
            if strategy.status == StrategyStatus.RUNNING
        ]
    
    async def start_all_strategies(self):
        """Start all enabled strategies"""
        tasks = []
        
        for strategy in self.strategies.values():
            if strategy.config.enabled and strategy.status in [StrategyStatus.INITIALIZING, StrategyStatus.PAUSED]:
                task = asyncio.create_task(strategy.run())
                tasks.append(task)
        
        if tasks:
            logger.info(f"Starting {len(tasks)} strategies")
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def stop_all_strategies(self):
        """Stop all strategies"""
        for strategy in self.strategies.values():
            if strategy.status in [StrategyStatus.RUNNING, StrategyStatus.PAUSED]:
                await strategy.stop()
        
        logger.info("All strategies stopped")
    
    async def get_registry_status(self) -> Dict[str, Any]:
        """Get registry status"""
        strategies_by_status = {}
        strategies_by_type = {}
        
        for strategy in self.strategies.values():
            # By status
            status = strategy.status.value
            if status not in strategies_by_status:
                strategies_by_status[status] = 0
            strategies_by_status[status] += 1
            
            # By type
            strategy_type = strategy.config.strategy_type.value
            if strategy_type not in strategies_by_type:
                strategies_by_type[strategy_type] = 0
            strategies_by_type[strategy_type] += 1
        
        return {
            'total_strategies': len(self.strategies),
            'enabled_strategies': len([s for s in self.strategies.values() if s.config.enabled]),
            'strategies_by_status': strategies_by_status,
            'strategies_by_type': strategies_by_type,
            'active_strategies': len([s for s in self.strategies.values() if s.status == StrategyStatus.RUNNING])
        }


# Global strategy registry
strategy_registry = StrategyRegistry()


# Example base implementation
class BaseTimeSeriesStrategy(BaseStrategy):
    """Base class for time series-based strategies"""
    
    def __init__(self, config: StrategyConfig, timeframe: str = "1m"):
        super().__init__(config)
        self.timeframe = timeframe
        self.data_cache: Dict[str, List[Dict[str, Any]]] = {}
    
    async def get_indicator_data(self, symbol: str, indicator: str) -> Optional[List[float]]:
        """
        Get indicator data for symbol
        
        Args:
            symbol: Trading symbol
            indicator: Indicator name
            
        Returns:
            List of indicator values
        """
        # This would typically fetch from a market data service
        # or calculate indicators from cached price data
        return None
    
    async def get_latest_price(self, symbol: str) -> Optional[Decimal]:
        """Get latest price for symbol"""
        if self.context:
            return await self.context.get_current_price(symbol)
        return None


# Utility functions for strategy development
async def validate_strategy_config(config: StrategyConfig) -> List[str]:
    """
    Validate strategy configuration
    
    Args:
        config: Strategy configuration to validate
        
    Returns:
        List of validation errors
    """
    errors = []
    
    # Check required fields
    if not config.strategy_id:
        errors.append("strategy_id is required")
    
    if not config.name:
        errors.append("name is required")
    
    if not config.description:
        errors.append("description is required")
    
    # Check parameters
    if not isinstance(config.parameters, dict):
        errors.append("parameters must be a dictionary")
    
    # Check risk parameters
    try:
        max_pos = Decimal(str(config.max_position_size))
        if max_pos <= 0:
            errors.append("max_position_size must be positive")
    except (ValueError, TypeError):
        errors.append("max_position_size must be a valid number")
    
    try:
        max_loss = Decimal(str(config.max_daily_loss))
        if max_loss <= 0:
            errors.append("max_daily_loss must be positive")
    except (ValueError, TypeError):
        errors.append("max_daily_loss must be a valid number")
    
    if config.max_trades_per_day <= 0:
        errors.append("max_trades_per_day must be positive")
    
    return errors


# Example usage and testing
if __name__ == "__main__":
    async def test_base_strategy():
        # Create a test strategy configuration
        config = StrategyConfig(
            strategy_id="test_strategy",
            strategy_type=StrategyType.MOMENTUM,
            name="Test Momentum Strategy",
            description="A simple momentum-based strategy for testing",
            parameters={
                "fast_period": 10,
                "slow_period": 30,
                "signal_threshold": 0.02
            },
            risk_level=RiskLevel.MEDIUM,
            max_position_size=Decimal('50000'),
            max_daily_loss=Decimal('5000'),
            symbols=['AAPL', 'GOOGL', 'MSFT']
        )
        
        # Validate configuration
        errors = await validate_strategy_config(config)
        if errors:
            print("Configuration errors:", errors)
        else:
            print("Configuration valid")
        
        # Create strategy (this would be a concrete implementation)
        # strategy = MomentumStrategy(config)
        # await strategy.run()
    
    asyncio.run(test_base_strategy())