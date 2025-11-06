"""
@file base_exit_strategy.py
@brief Exit Strategies Framework - Base Classes and Interfaces

@details
This module provides the foundational classes and interfaces for implementing
exit strategies in the trading orchestrator system. It defines abstract base classes,
data structures, and common utilities that all exit strategies must implement.

Key Features:
- Abstract base classes for exit strategy implementation
- Exit signal types and exit status enums
- Exit condition evaluation and signal generation framework
- Exit performance tracking and metrics
- Position and order management integration
- Exit strategy lifecycle management

@author Trading Orchestrator System
@version 1.0
@date 2025-11-06

@warning
Exit strategies control when positions are closed and can trigger real trades.
Always test exit strategies thoroughly in paper trading mode before live deployment.

@note
This module defines the core framework that all exit strategies must follow:
- BaseExitStrategy: Abstract base for all exit strategies
- ExitReason: Exit trigger reason enums
- ExitType: Strategy type definitions
- ExitStatus: Execution state tracking

@see strategies.exit_strategies.trailing_stop for trailing stop implementation
@see strategies.exit_strategies.fixed_target for fixed target strategy
@see strategies.exit_strategies.stop_loss for stop loss strategy
"""

from typing import Dict, Any, List, Optional, Union, Protocol, Generic, TypeVar, Callable
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import uuid
from abc import ABC, abstractmethod
import statistics

from loguru import logger

T = TypeVar('T')

class ExitReason(Enum):
    """
    @enum ExitReason
    @brief Reasons for exiting a position
    
    @details
    Enumerates the different reasons why an exit strategy might trigger
    and close a position. Each reason represents a specific market condition
    or strategic objective.
    
    @par Exit Categories:
    - PROFIT_TARGET: Position closed at profit target
    - STOP_LOSS: Position closed due to stop loss trigger
    - TRAILING_STOP: Position closed by trailing stop mechanism
    - VOLATILITY_STOP: Position closed due to high volatility
    - TIME_EXIT: Position closed based on time constraints
    - MARKET_CONDITION: Position closed due to market conditions
    - AI_SIGNAL: Position closed by AI decision
    - MANUAL: Position closed manually
    
    @warning
    Exit reason tracking is important for strategy performance analysis
    and risk management review.
    
    @note
    Multiple exit conditions may trigger simultaneously. The first applicable
    reason in the priority order will be used.
    """
    PROFIT_TARGET = "profit_target"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    VOLATILITY_STOP = "volatility_stop"
    TIME_EXIT = "time_exit"
    MARKET_CONDITION = "market_condition"
    AI_SIGNAL = "ai_signal"
    MANUAL = "manual"
    SYSTEM_ERROR = "system_error"
    RISK_LIMIT = "risk_limit"


class ExitType(Enum):
    """
    @enum ExitType
    @brief Exit strategy categories
    
    @details
    Defines the different types of exit strategies available in the system.
    Each strategy type represents a distinct approach to exiting positions.
    
    @par Strategy Categories:
    - TRAILING_STOP: Dynamic stop loss that follows favorable price movements
    - FIXED_TARGET: Fixed profit and loss targets
    - STOP_LOSS: Traditional stop loss at fixed percentage
    - VOLATILITY_STOP: Stop loss based on volatility measures
    - TIME_BASED: Exit based on time elapsed
    - CONDITIONAL: Exit based on market conditions
    - AI_DRIVEN: AI-powered exit decision making
    
    @warning
    Each exit strategy has different characteristics and risk profiles.
    Choose appropriate strategies for your trading style and risk tolerance.
    
    @note
    Multiple exit strategies can be combined for comprehensive position management.
    """
    TRAILING_STOP = "trailing_stop"
    FIXED_TARGET = "fixed_target"
    STOP_LOSS = "stop_loss"
    VOLATILITY_STOP = "volatility_stop"
    TIME_BASED = "time_based"
    CONDITIONAL = "conditional"
    AI_DRIVEN = "ai_driven"


class ExitStatus(Enum):
    """
    @enum ExitStatus
    @brief Exit strategy execution states
    
    @details
    Tracks the current execution state of an exit strategy throughout
    its lifecycle from initialization to completion or error conditions.
    
    @par Status Flow:
    1. INITIALIZING: Exit strategy setup and configuration
    2. ACTIVE: Monitoring positions for exit conditions
    3. TRIGGERED: Exit condition met, executing exit
    4. COMPLETED: Exit strategy finished execution
    5. PAUSED: Temporary pause (manual or automatic)
    6. STOPPED: Normal completion or manual stop
    7. ERROR: Unexpected error condition
    
    @note
    Status transitions are managed by the exit strategy execution framework
    and should not be set manually during normal operation.
    """
    INITIALIZING = "initializing"
    ACTIVE = "active"
    TRIGGERED = "triggered"
    COMPLETED = "completed"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class ExitCondition:
    """
    @class ExitCondition
    @brief Condition that triggers an exit signal
    
    @details
    Represents a specific condition that, when met, will trigger
    an exit signal for a position. Conditions can be based on price,
    time, volatility, technical indicators, or other market factors.
    
    @par Condition Components:
    - condition_id: Unique identifier for tracking
    - name: Human-readable condition name
    - condition_type: Type of condition (price, time, volatility, etc.)
    - threshold_value: Value that triggers the condition
    - comparison_operator: How to compare current value to threshold
    - parameters: Additional condition-specific parameters
    - is_active: Whether condition is currently active
    - priority: Priority for condition evaluation (lower = higher priority)
    - created_at: Condition creation timestamp
    - last_evaluated: Last time condition was checked
    
    @par Example:
    @code
    price_condition = ExitCondition(
        condition_id="price_001",
        name="Price below stop loss",
        condition_type="price",
        threshold_value=Decimal('145.00'),
        comparison_operator="<=",
        priority=1
    )
    @endcode
    
    @warning
    Always validate condition parameters before activation to prevent
    unexpected behavior or trading errors.
    
    @note
    Conditions are evaluated in priority order. First condition met
    will trigger the exit signal.
    """
    condition_id: str
    name: str
    condition_type: str
    threshold_value: Union[Decimal, float, int, datetime]
    comparison_operator: str  # ">", "<", ">=", "<=", "==", "!="
    parameters: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    priority: int = 1
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_evaluated: Optional[datetime] = None
    evaluation_count: int = 0


@dataclass
class ExitSignal:
    """
    @class ExitSignal
    @brief Signal generated by exit strategy
    
    @details
    Represents a signal generated by an exit strategy when exit conditions
    are met. Contains all necessary information for position closure including
    exit price, quantity, timing, and reason.
    
    @par Signal Components:
    - signal_id: Unique identifier for tracking
    - strategy_id: Exit strategy that generated the signal
    - position_id: Position to be exited
    - symbol: Trading instrument
    - exit_reason: Reason for exit
    - exit_price: Target exit price
    - exit_quantity: Quantity to exit
    - confidence: Confidence level in the exit (0.0-1.0)
    - urgency: Exit urgency (0.0-1.0)
    - estimated_execution_time: Expected time to execute exit
    - market_impact: Expected market impact of exit
    - metadata: Strategy-specific exit data
    - created_at: Signal generation timestamp
    - expires_at: Optional signal expiration time
    
    @par Example:
    @code
    signal = ExitSignal(
        signal_id="exit_123",
        strategy_id="trailing_001",
        position_id="pos_456",
        symbol="AAPL",
        exit_reason=ExitReason.TRAILING_STOP,
        exit_price=Decimal('150.00'),
        exit_quantity=Decimal('100'),
        confidence=0.92,
        urgency=0.85
    )
    @endcode
    
    @warning
    Exit signals should be validated by risk management before execution
    to ensure compliance with portfolio limits and risk parameters.
    
    @note
    Signal validation and execution are handled by the exit strategy framework
    and risk management system.
    """
    signal_id: str
    strategy_id: str
    position_id: str
    symbol: str
    exit_reason: ExitReason
    exit_price: Decimal
    exit_quantity: Decimal
    confidence: float  # 0.0 to 1.0
    urgency: float    # 0.0 to 1.0
    estimated_execution_time: Optional[timedelta] = None
    market_impact: Optional[Decimal] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None


@dataclass
class ExitMetrics:
    """
    @class ExitMetrics
    @brief Performance metrics for exit strategy
    
    @details
    Tracks comprehensive performance metrics for an exit strategy
    including exit timing, profitability, risk control, and execution
    statistics.
    
    @par Core Metrics:
    - total_exits: Total number of exits executed
    - successful_exits: Number of exits executed successfully
    - failed_exits: Number of exits that failed
    - profit_exits: Number of profitable exits
    - loss_exits: Number of losing exits
    - success_rate: Percentage of successful exits
    - win_rate: Percentage of profitable exits
    - total_profit: Total profit generated by exits
    - total_loss: Total loss from exits
    - net_profit: Net profit (total_profit - total_loss)
    - avg_exit_time: Average time from position open to exit
    - median_exit_time: Median exit time
    - avg_profit_per_exit: Average profit per exit
    - avg_loss_per_exit: Average loss per exit
    - profit_factor: Total profit / Total loss ratio
    - largest_profit: Biggest single profitable exit
    - largest_loss: Biggest single losing exit
    - avg_confidence: Average confidence level of exit signals
    - avg_urgency: Average urgency level of exit signals
    
    @par Streak Tracking:
    - consecutive_wins: Current profitable exit streak
    - consecutive_losses: Current losing exit streak
    - longest_win_streak: Longest profitable streak
    - longest_loss_streak: Longest losing streak
    
    @par Performance Metrics:
    - sharpe_ratio: Risk-adjusted return measure
    - max_drawdown: Maximum peak-to-trough decline
    - calmar_ratio: Annual return / Max drawdown
    - sortino_ratio: Risk-adjusted return using downside deviation
    
    @par Example:
    @code
    metrics = ExitMetrics(
        strategy_id="trailing_001",
        total_exits=150,
        successful_exits=145,
        profit_exits=90,
        success_rate=0.967,
        win_rate=0.621,
        total_profit=15000.0,
        total_loss=-8500.0,
        net_profit=6500.0
    )
    @endcode
    
    @note
    Metrics are updated automatically by the exit strategy framework
    during execution and can be used for performance analysis and optimization.
    """
    strategy_id: str
    total_exits: int = 0
    successful_exits: int = 0
    failed_exits: int = 0
    profit_exits: int = 0
    loss_exits: int = 0
    success_rate: float = 0.0
    win_rate: float = 0.0
    total_profit: float = 0.0
    total_loss: float = 0.0
    net_profit: float = 0.0
    avg_exit_time: float = 0.0
    median_exit_time: float = 0.0
    avg_profit_per_exit: float = 0.0
    avg_loss_per_exit: float = 0.0
    profit_factor: float = 0.0
    largest_profit: float = 0.0
    largest_loss: float = 0.0
    avg_confidence: float = 0.0
    avg_urgency: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    longest_win_streak: int = 0
    longest_loss_streak: int = 0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ExitConfiguration:
    """
    @class ExitConfiguration
    @brief Configuration parameters for exit strategy
    
    @details
    Contains all configuration parameters needed to initialize and
    configure an exit strategy including exit rules, risk limits,
    and operational settings.
    
    @par Required Fields:
    - strategy_id: Unique identifier
    - strategy_type: Category of exit strategy
    - name: Human-readable name
    - description: Detailed description
    - parameters: Strategy-specific parameters
    - conditions: List of exit conditions
    - is_active: Strategy enabled status
    
    @par Risk Limits:
    - max_exit_size: Maximum position size to exit
    - max_exit_time: Maximum time to hold position
    - min_confidence: Minimum confidence required for exit
    - max_loss_threshold: Maximum allowed loss per exit
    
    @par Operational Settings:
    - monitoring_interval: How often to check exit conditions
    - execution_timeout: Maximum time to execute exit
    - slippage_tolerance: Acceptable slippage for execution
    - retry_attempts: Number of retry attempts for failed exits
    
    @par Example:
    @code
    config = ExitConfiguration(
        strategy_id="trailing_001",
        strategy_type=ExitType.TRAILING_STOP,
        name="5% Trailing Stop",
        description="Dynamic 5% trailing stop loss",
        parameters={
            "trailing_distance": 0.05,
            "min_stop_distance": 0.02,
            "update_frequency": 60
        },
        conditions=[
            ExitCondition(...),
            ExitCondition(...)
        ],
        max_exit_size=Decimal('100000'),
        min_confidence=0.80,
        monitoring_interval=30,
        execution_timeout=timedelta(seconds=60)
    )
    @endcode
    
    @warning
    Risk parameters should be carefully set based on portfolio size
    and risk tolerance. Incorrect settings can lead to significant losses.
    
    @note
    Configuration changes require exit strategy restart to take effect.
    """
    strategy_id: str
    strategy_type: ExitType
    name: str
    description: str
    parameters: Dict[str, Any]
    conditions: List[ExitCondition] = field(default_factory=list)
    is_active: bool = True
    max_exit_size: Decimal = Decimal('100000')
    max_exit_time: Optional[timedelta] = None
    min_confidence: float = 0.50
    max_loss_threshold: Decimal = Decimal('10000')
    monitoring_interval: int = 60  # seconds
    execution_timeout: timedelta = field(default_factory=lambda: timedelta(seconds=60))
    slippage_tolerance: Decimal = Decimal('0.001')  # 0.1%
    retry_attempts: int = 3
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ExitRule:
    """
    @class ExitRule
    @brief Rule defining when to trigger exit
    
    @details
    Defines a specific rule that, when satisfied, will trigger
    an exit signal. Rules combine conditions with logical operators
    and can be nested for complex exit logic.
    
    @par Rule Components:
    - rule_id: Unique identifier for tracking
    - name: Human-readable rule name
    - conditions: List of conditions to evaluate
    - logical_operator: How to combine conditions (AND/OR)
    - threshold: Additional threshold for rule activation
    - priority: Rule priority (lower = higher priority)
    - is_active: Whether rule is currently active
    - evaluation_count: Number of times rule has been evaluated
    - success_count: Number of times rule triggered successfully
    
    @par Example:
    @code
    rule = ExitRule(
        rule_id="rule_001",
        name="Price drop with high volume",
        conditions=[
            ExitCondition(...),  # Price drop condition
            ExitCondition(...)   # High volume condition
        ],
        logical_operator="AND",
        threshold=Decimal('0.03'),
        priority=1
    )
    @endcode
    
    @warning
    Rules should be thoroughly tested to ensure they behave as expected
    under various market conditions.
    
    @note
    Rules are evaluated in priority order. First rule satisfied triggers exit.
    """
    rule_id: str
    name: str
    conditions: List[ExitCondition]
    logical_operator: str  # "AND", "OR"
    threshold: Optional[Union[Decimal, float, int]] = None
    priority: int = 1
    is_active: bool = True
    evaluation_count: int = 0
    success_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)


class ExitContext(Protocol):
    """Context interface for exit strategy execution"""
    
    async def get_current_price(self, symbol: str) -> Decimal:
        """Get current price for symbol"""
        ...
    
    async def get_position(self, position_id: str) -> Optional[Dict[str, Any]]:
        """Get position information"""
        ...
    
    async def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str, 
        limit: int
    ) -> List[Dict[str, Any]]:
        """Get historical market data"""
        ...
    
    async def calculate_volatility(self, symbol: str, period: int) -> Decimal:
        """Calculate volatility for symbol"""
        ...
    
    async def submit_exit_order(
        self,
        position_id: str,
        symbol: str,
        quantity: Decimal,
        exit_price: Decimal,
        order_type: str = "market"
    ) -> Dict[str, Any]:
        """Submit exit order"""
        ...
    
    async def get_portfolio_value(self) -> Decimal:
        """Get current portfolio value"""
        ...
    
    async def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics"""
        ...


class BaseExitStrategy(ABC):
    """
    @class BaseExitStrategy
    @brief Abstract base class for all exit strategies
    
    @details
    Provides the foundational interface and common functionality for all
    exit strategies in the trading orchestrator. Defines abstract methods
    that all exit strategies must implement and provides common functionality
    for signal processing, metrics tracking, and position monitoring.
    
    @par Key Features:
    - Abstract interface for exit strategy implementation
    - Common exit condition evaluation framework
    - Signal generation and processing
    - Performance metrics tracking
    - Position monitoring and exit execution
    - Strategy lifecycle management
    
    @par Exit Process:
    1. Monitor positions for exit conditions
    2. Evaluate exit conditions using custom logic
    3. Generate exit signals when conditions are met
    4. Process and validate exit signals
    5. Execute position closure
    6. Update performance metrics
    
    @par Example:
    @code
    class CustomExitStrategy(BaseExitStrategy):
        def __init__(self, config):
            super().__init__(config)
            
        async def evaluate_exit_conditions(self, position):
            # Custom exit logic here
            return False  # or True to trigger exit
            
        async def generate_exit_signal(self, position, exit_reason):
            # Custom signal generation here
            return ExitSignal(...)
    
    # Create and use strategy
    config = ExitConfiguration(...)
    strategy = CustomExitStrategy(config)
    await strategy.start()
    @endcode
    
    @warning
    Exit strategies execute real trades. Always test thoroughly in
    paper trading mode before live deployment.
    
    @note
    Exit strategies work in conjunction with the risk management system
    to ensure portfolio-level risk controls are maintained.
    
    @see ExitType for available strategy types
    @see ExitReason for exit trigger reasons
    @see ExitSignal for signal structure
    """
    
    def __init__(self, config: ExitConfiguration):
        """
        @brief Initialize base exit strategy with configuration
        
        @param config ExitConfiguration containing all strategy parameters
        
        @details
        Initializes the exit strategy with provided configuration and sets up
        internal state including metrics tracking, condition management,
        and position monitoring.
        
        @par Initialization Process:
        1. Store strategy configuration
        2. Set initial status to INITIALIZING
        3. Initialize metrics tracking
        4. Set up condition and rule management
        5. Initialize signal tracking
        6. Record start time for runtime calculations
        
        @throws ValueError if configuration is invalid
        
        @par Example:
        @code
        config = ExitConfiguration(...)
        strategy = BaseExitStrategy(config)
        # Strategy is now initialized but not running
        @endcode
        
        @note
        Strategy must be started explicitly using the start() method.
        """
        self.config = config
        self.status = ExitStatus.INITIALIZING
        self.context: Optional[ExitContext] = None
        
        # Metrics and tracking
        self.metrics = ExitMetrics(strategy_id=config.strategy_id)
        self.exit_signals: Dict[str, ExitSignal] = {}
        self.exit_history: List[Dict[str, Any]] = []
        
        # Position monitoring
        self.monitored_positions: Dict[str, Dict[str, Any]] = {}
        self.conditions = config.conditions
        self.rules: List[ExitRule] = []
        
        # Performance tracking
        self.start_time = datetime.utcnow()
        self.last_evaluation_time = None
        self.total_evaluations = 0
        self.total_signals_generated = 0
        self.total_exits_executed = 0
        
        # Exit tracking
        self.active_exits: Dict[str, Dict[str, Any]] = {}
        self.pending_exits: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Exit strategy initialized: {config.name} ({config.strategy_type.value})")
    
    @abstractmethod
    async def evaluate_exit_conditions(self, position: Dict[str, Any]) -> bool:
        """
        Evaluate whether exit conditions are met for a position
        
        Args:
            position: Position information dictionary
            
        Returns:
            True if exit conditions are met, False otherwise
        """
        pass
    
    @abstractmethod
    async def generate_exit_signal(
        self, 
        position: Dict[str, Any], 
        exit_reason: ExitReason
    ) -> Optional[ExitSignal]:
        """
        Generate exit signal based on position and exit reason
        
        Args:
            position: Position information dictionary
            exit_reason: Reason for exiting
            
        Returns:
            ExitSignal if exit should be triggered, None otherwise
        """
        pass
    
    def set_context(self, context: ExitContext):
        """Set exit strategy execution context"""
        self.context = context
        logger.info(f"Context set for exit strategy: {self.config.name}")
    
    async def start(self):
        """Start the exit strategy monitoring"""
        try:
            self.status = ExitStatus.ACTIVE
            logger.info(f"Exit strategy started: {self.config.name}")
            
            # Start monitoring loop
            while self.status == ExitStatus.ACTIVE:
                try:
                    # Monitor all positions
                    await self.monitor_positions()
                    
                    # Update metrics
                    await self.update_metrics()
                    
                    # Sleep before next evaluation
                    await asyncio.sleep(self.config.monitoring_interval)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in exit strategy monitoring: {e}")
                    self.status = ExitStatus.ERROR
                    break
            
            logger.info(f"Exit strategy stopped: {self.config.name}")
            
        except Exception as e:
            logger.error(f"Error starting exit strategy: {e}")
            self.status = ExitStatus.ERROR
    
    async def monitor_positions(self):
        """Monitor all positions for exit conditions"""
        try:
            if not self.context:
                logger.warning("No context available for position monitoring")
                return
            
            # Get current positions to monitor
            positions = await self.context.get_positions()
            
            for position in positions:
                position_id = position.get('position_id')
                if not position_id:
                    continue
                
                # Add to monitored positions if not already monitored
                if position_id not in self.monitored_positions:
                    self.monitored_positions[position_id] = position.copy()
                    logger.info(f"Monitoring position: {position_id}")
                
                # Evaluate exit conditions
                await self.evaluate_position(position)
            
            # Clean up stale monitored positions
            current_position_ids = {p.get('position_id') for p in positions}
            stale_ids = [
                pid for pid in self.monitored_positions.keys()
                if pid not in current_position_ids
            ]
            for stale_id in stale_ids:
                del self.monitored_positions[stale_id]
                logger.info(f"Stopped monitoring position: {stale_id}")
                
        except Exception as e:
            logger.error(f"Error monitoring positions: {e}")
    
    async def evaluate_position(self, position: Dict[str, Any]):
        """Evaluate exit conditions for a specific position"""
        try:
            self.total_evaluations += 1
            self.last_evaluation_time = datetime.utcnow()
            
            # Check if position should be evaluated
            if not await self.should_evaluate_position(position):
                return
            
            # Evaluate exit conditions
            exit_triggered = await self.evaluate_exit_conditions(position)
            
            if exit_triggered:
                # Determine exit reason
                exit_reason = await self.determine_exit_reason(position)
                
                # Generate exit signal
                exit_signal = await self.generate_exit_signal(position, exit_reason)
                
                if exit_signal:
                    await self.process_exit_signal(exit_signal)
            else:
                # Update position monitoring data
                await self.update_position_monitoring(position)
                
        except Exception as e:
            logger.error(f"Error evaluating position {position.get('position_id', 'unknown')}: {e}")
    
    async def should_evaluate_position(self, position: Dict[str, Any]) -> bool:
        """Check if position should be evaluated for exit"""
        try:
            # Skip if position is already being exited
            position_id = position.get('position_id')
            if position_id in self.active_exits or position_id in self.pending_exits:
                return False
            
            # Skip if position is too new (activation delay)
            created_at = position.get('created_at')
            if created_at:
                age = datetime.utcnow() - created_at
                activation_delay = timedelta(seconds=self.config.parameters.get('activation_delay', 0))
                if age < activation_delay:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking position evaluation eligibility: {e}")
            return False
    
    async def determine_exit_reason(self, position: Dict[str, Any]) -> ExitReason:
        """Determine the reason for exit based on conditions"""
        try:
            # Default implementation - can be overridden by specific strategies
            # Check each condition to determine exit reason
            
            for condition in self.conditions:
                if not condition.is_active:
                    continue
                
                condition_result = await self.evaluate_single_condition(condition, position)
                if condition_result:
                    # Map condition type to exit reason
                    if condition.condition_type == "price_profit":
                        return ExitReason.PROFIT_TARGET
                    elif condition.condition_type == "price_loss":
                        return ExitReason.STOP_LOSS
                    elif condition.condition_type == "time":
                        return ExitReason.TIME_EXIT
                    elif condition.condition_type == "volatility":
                        return ExitReason.VOLATILITY_STOP
                    elif condition.condition_type == "technical":
                        return ExitReason.MARKET_CONDITION
            
            # Default to generic exit
            return ExitReason.MANUAL
            
        except Exception as e:
            logger.error(f"Error determining exit reason: {e}")
            return ExitReason.SYSTEM_ERROR
    
    async def evaluate_single_condition(self, condition: ExitCondition, position: Dict[str, Any]) -> bool:
        """Evaluate a single exit condition"""
        try:
            condition.last_evaluated = datetime.utcnow()
            condition.evaluation_count += 1
            
            if not condition.is_active:
                return False
            
            # Get current value for comparison
            current_value = await self.get_condition_current_value(condition, position)
            
            if current_value is None:
                return False
            
            # Compare with threshold
            threshold = condition.threshold_value
            operator = condition.comparison_operator
            
            result = self._compare_values(current_value, threshold, operator)
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating condition {condition.condition_id}: {e}")
            return False
    
    async def get_condition_current_value(self, condition: ExitCondition, position: Dict[str, Any]) -> Optional[Union[Decimal, float, int]]:
        """Get current value for condition evaluation"""
        try:
            symbol = position.get('symbol')
            if not symbol or not self.context:
                return None
            
            condition_type = condition.condition_type
            
            if condition_type == "price":
                return await self.context.get_current_price(symbol)
            elif condition_type == "volatility":
                period = condition.parameters.get('period', 20)
                return await self.context.calculate_volatility(symbol, period)
            elif condition_type == "time":
                # Calculate time since position opened
                created_at = position.get('created_at')
                if created_at:
                    return (datetime.utcnow() - created_at).total_seconds()
            elif condition_type == "profit":
                # Calculate current profit percentage
                entry_price = position.get('entry_price')
                current_price = await self.context.get_current_price(symbol)
                if entry_price and current_price:
                    return float((current_price - entry_price) / entry_price)
            elif condition_type == "technical":
                # Get technical indicator value
                indicator = condition.parameters.get('indicator', 'rsi')
                period = condition.parameters.get('period', 14)
                data = await self.context.get_historical_data(symbol, '1h', period + 1)
                if data and len(data) >= period:
                    return self._calculate_technical_indicator(data, indicator, period)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting condition current value: {e}")
            return None
    
    def _compare_values(self, current: Union[Decimal, float, int], threshold: Union[Decimal, float, int], operator: str) -> bool:
        """Compare current value with threshold using operator"""
        try:
            if isinstance(current, Decimal):
                current = float(current)
            if isinstance(threshold, Decimal):
                threshold = float(threshold)
            
            if operator == ">":
                return current > threshold
            elif operator == "<":
                return current < threshold
            elif operator == ">=":
                return current >= threshold
            elif operator == "<=":
                return current <= threshold
            elif operator == "==":
                return current == threshold
            elif operator == "!=":
                return current != threshold
            else:
                logger.warning(f"Unknown comparison operator: {operator}")
                return False
                
        except Exception as e:
            logger.error(f"Error comparing values: {e}")
            return False
    
    def _calculate_technical_indicator(self, data: List[Dict[str, Any]], indicator: str, period: int) -> Optional[float]:
        """Calculate technical indicator value"""
        try:
            if not data:
                return None
            
            prices = [Decimal(str(item.get('close', 0))) for item in data]
            
            if indicator.lower() == "rsi":
                return self._calculate_rsi(prices, period)
            elif indicator.lower() == "sma":
                return self._calculate_sma(prices, period)
            elif indicator.lower() == "ema":
                return self._calculate_ema(prices, period)
            else:
                logger.warning(f"Unknown technical indicator: {indicator}")
                return None
                
        except Exception as e:
            logger.error(f"Error calculating technical indicator: {e}")
            return None
    
    def _calculate_rsi(self, prices: List[Decimal], period: int) -> float:
        """Calculate RSI indicator"""
        try:
            if len(prices) < period + 1:
                return 50.0  # Neutral RSI
            
            deltas = []
            for i in range(1, len(prices)):
                delta = float(prices[i] - prices[i-1])
                deltas.append(delta)
            
            if len(deltas) < period:
                return 50.0
            
            gains = [max(0, delta) for delta in deltas]
            losses = [abs(min(0, delta)) for delta in deltas]
            
            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return 50.0
    
    def _calculate_sma(self, prices: List[Decimal], period: int) -> float:
        """Calculate Simple Moving Average"""
        try:
            if len(prices) < period:
                return 0.0
            
            recent_prices = prices[-period:]
            return float(sum(recent_prices) / len(recent_prices))
            
        except Exception as e:
            logger.error(f"Error calculating SMA: {e}")
            return 0.0
    
    def _calculate_ema(self, prices: List[Decimal], period: int) -> float:
        """Calculate Exponential Moving Average"""
        try:
            if len(prices) == 0:
                return 0.0
            if len(prices) < period:
                return float(sum(prices) / len(prices))
            
            multiplier = 2 / (period + 1)
            ema = float(prices[0])
            
            for price in prices[1:]:
                ema = float(price) * multiplier + ema * (1 - multiplier)
            
            return ema
            
        except Exception as e:
            logger.error(f"Error calculating EMA: {e}")
            return 0.0
    
    async def update_position_monitoring(self, position: Dict[str, Any]):
        """Update position monitoring data"""
        try:
            position_id = position.get('position_id')
            if position_id:
                self.monitored_positions[position_id] = position.copy()
                
        except Exception as e:
            logger.error(f"Error updating position monitoring: {e}")
    
    async def process_exit_signal(self, signal: ExitSignal):
        """Process an exit signal"""
        try:
            # Store signal
            self.exit_signals[signal.signal_id] = signal
            self.total_signals_generated += 1
            
            # Validate signal
            if not await self.validate_exit_signal(signal):
                logger.warning(f"Exit signal rejected: {signal.signal_id}")
                return
            
            # Check confidence threshold
            if signal.confidence < self.config.min_confidence:
                logger.info(f"Exit signal below confidence threshold: {signal.signal_id}")
                return
            
            # Add to pending exits
            self.pending_exits[signal.position_id] = {
                'signal': signal,
                'created_at': datetime.utcnow(),
                'status': 'pending'
            }
            
            # Execute exit
            await self.execute_exit(signal)
            
        except Exception as e:
            logger.error(f"Error processing exit signal: {e}")
    
    async def validate_exit_signal(self, signal: ExitSignal) -> bool:
        """Validate an exit signal before execution"""
        try:
            # Check if signal has required fields
            if not all([signal.signal_id, signal.position_id, signal.symbol, signal.exit_price]):
                return False
            
            # Check signal expiration
            if signal.expires_at and datetime.utcnow() > signal.expires_at:
                return False
            
            # Check position size
            position_info = await self.context.get_position(signal.position_id)
            if position_info:
                position_quantity = Decimal(str(position_info.get('quantity', 0)))
                if position_quantity < signal.exit_quantity:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating exit signal: {e}")
            return False
    
    async def execute_exit(self, signal: ExitSignal):
        """Execute the exit order"""
        try:
            self.status = ExitStatus.TRIGGERED
            position_id = signal.position_id
            
            logger.info(f"Executing exit: {signal.signal_id} for position {position_id}")
            
            # Submit exit order
            if self.context:
                order_result = await self.context.submit_exit_order(
                    position_id=position_id,
                    symbol=signal.symbol,
                    quantity=signal.exit_quantity,
                    exit_price=signal.exit_price,
                    order_type="market"
                )
                
                if order_result.get('success'):
                    # Move from pending to active
                    if position_id in self.pending_exits:
                        self.active_exits[position_id] = self.pending_exits.pop(position_id)
                    
                    self.total_exits_executed += 1
                    logger.info(f"Exit order submitted: {signal.signal_id}")
                    
                    # Start monitoring exit execution
                    asyncio.create_task(self.monitor_exit_execution(signal))
                    
                else:
                    # Failed order
                    logger.error(f"Exit order failed: {order_result.get('reason')}")
                    if position_id in self.pending_exits:
                        del self.pending_exits[position_id]
                    
                    self.metrics.failed_exits += 1
            
            self.status = ExitStatus.ACTIVE
            
        except Exception as e:
            logger.error(f"Error executing exit: {e}")
            self.status = ExitStatus.ACTIVE
            self.metrics.failed_exits += 1
    
    async def monitor_exit_execution(self, signal: ExitSignal):
        """Monitor the execution of an exit order"""
        try:
            max_attempts = self.config.retry_attempts
            attempt = 0
            
            while attempt < max_attempts:
                if not self.context:
                    break
                
                # Check if exit was executed
                position_info = await self.context.get_position(signal.position_id)
                if not position_info:
                    # Position no longer exists - exit completed
                    await self.complete_exit(signal, True)
                    return
                
                attempt += 1
                await asyncio.sleep(5)  # Wait 5 seconds between checks
            
            # Check execution timeout
            if attempt >= max_attempts:
                await self.complete_exit(signal, False)
                
        except Exception as e:
            logger.error(f"Error monitoring exit execution: {e}")
            await self.complete_exit(signal, False)
    
    async def complete_exit(self, signal: ExitSignal, success: bool):
        """Complete an exit execution"""
        try:
            # Update exit history
            exit_record = {
                'signal_id': signal.signal_id,
                'position_id': signal.position_id,
                'symbol': signal.symbol,
                'exit_reason': signal.exit_reason.value,
                'exit_price': signal.exit_price,
                'exit_quantity': signal.exit_quantity,
                'confidence': signal.confidence,
                'urgency': signal.urgency,
                'success': success,
                'executed_at': datetime.utcnow()
            }
            
            self.exit_history.append(exit_record)
            
            # Update metrics
            if success:
                self.metrics.successful_exits += 1
            else:
                self.metrics.failed_exits += 1
            
            # Remove from active exits
            position_id = signal.position_id
            if position_id in self.active_exits:
                del self.active_exits[position_id]
            
            # Update status
            self.status = ExitStatus.COMPLETED
            self.status = ExitStatus.ACTIVE
            
        except Exception as e:
            logger.error(f"Error completing exit: {e}")
    
    async def update_metrics(self):
        """Update exit strategy metrics"""
        try:
            # Basic counts
            self.metrics.total_exits = len(self.exit_history)
            
            if self.exit_history:
                # Success metrics
                successful_exits = [e for e in self.exit_history if e.get('success', False)]
                self.metrics.successful_exits = len(successful_exits)
                self.metrics.failed_exits = self.metrics.total_exits - self.metrics.successful_exits
                
                if self.metrics.total_exits > 0:
                    self.metrics.success_rate = self.metrics.successful_exits / self.metrics.total_exits
                
                # Profit metrics
                if successful_exits:
                    # This would need real P&L calculation
                    # For now, use confidence as proxy for profitability
                    self.metrics.avg_confidence = statistics.mean([e.get('confidence', 0) for e in self.exit_history])
                    self.metrics.avg_urgency = statistics.mean([e.get('urgency', 0) for e in self.exit_history])
            
            # Update timestamp
            self.metrics.last_updated = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    async def pause(self):
        """Pause exit strategy execution"""
        if self.status == ExitStatus.ACTIVE:
            self.status = ExitStatus.PAUSED
            logger.info(f"Exit strategy paused: {self.config.name}")
    
    async def resume(self):
        """Resume exit strategy execution"""
        if self.status == ExitStatus.PAUSED:
            self.status = ExitStatus.ACTIVE
            logger.info(f"Exit strategy resumed: {self.config.name}")
    
    async def stop(self):
        """Stop exit strategy execution"""
        self.status = ExitStatus.STOPPED
        logger.info(f"Exit strategy stopped: {self.config.name}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get exit strategy status"""
        runtime = datetime.utcnow() - self.start_time
        
        return {
            'strategy_id': self.config.strategy_id,
            'name': self.config.name,
            'type': self.config.strategy_type.value,
            'status': self.status.value,
            'runtime_seconds': runtime.total_seconds(),
            'total_evaluations': self.total_evaluations,
            'total_signals_generated': self.total_signals_generated,
            'total_exits_executed': self.total_exits_executed,
            'monitored_positions': len(self.monitored_positions),
            'active_exits': len(self.active_exits),
            'pending_exits': len(self.pending_exits),
            'metrics': {
                'total_exits': self.metrics.total_exits,
                'successful_exits': self.metrics.successful_exits,
                'success_rate': self.metrics.success_rate,
                'win_rate': self.metrics.win_rate
            },
            'last_updated': self.metrics.last_updated.isoformat()
        }


class ExitStrategyRegistry:
    """Registry for managing exit strategies"""
    
    def __init__(self):
        self._strategies: Dict[str, BaseExitStrategy] = {}
        self._strategy_types: Dict[str, type] = {}
        self._is_registered = False
    
    def register_strategy_type(self, strategy_type: ExitType, strategy_class: type):
        """Register an exit strategy class"""
        self._strategy_types[strategy_type.value] = strategy_class
        logger.info(f"Registered exit strategy type: {strategy_type.value}")
    
    def create_strategy(self, strategy_type: ExitType, config: ExitConfiguration) -> BaseExitStrategy:
        """Create an exit strategy instance"""
        if strategy_type.value not in self._strategy_types:
            raise ValueError(f"Unknown exit strategy type: {strategy_type.value}")
        
        strategy_class = self._strategy_types[strategy_type.value]
        return strategy_class(config)
    
    def register_instance(self, strategy: BaseExitStrategy):
        """Register a strategy instance"""
        self._strategies[strategy.config.strategy_id] = strategy
        logger.info(f"Registered exit strategy instance: {strategy.config.strategy_id}")
    
    def get_strategy(self, strategy_id: str) -> Optional[BaseExitStrategy]:
        """Get a registered strategy instance"""
        return self._strategies.get(strategy_id)
    
    def unregister_strategy(self, strategy_id: str):
        """Unregister a strategy instance"""
        if strategy_id in self._strategies:
            del self._strategies[strategy_id]
            logger.info(f"Unregistered exit strategy: {strategy_id}")
    
    def get_all_strategies(self) -> Dict[str, BaseExitStrategy]:
        """Get all registered strategies"""
        return self._strategies.copy()
    
    def get_available_types(self) -> List[ExitType]:
        """Get all available strategy types"""
        return [ExitType(t) for t in self._strategy_types.keys()]
    
    def clear(self):
        """Clear all registered strategies"""
        self._strategies.clear()
        logger.info("Cleared all exit strategy registrations")


# Global registry instance
exit_strategy_registry = ExitStrategyRegistry()
