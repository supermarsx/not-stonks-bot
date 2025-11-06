"""
@file trailing_stop.py
@brief Trailing Stop Exit Strategy Implementation

@details
This module implements various trailing stop exit strategies that dynamically
adjust the stop loss level based on favorable price movements. Trailing stops
help lock in profits while allowing positions to continue benefiting from
favorable trends.

Key Features:
- ATR-based trailing stops for volatility-adjusted positioning
- Fixed percentage trailing stops for simple implementation
- Dynamic trailing distance adjustment
- Multiple trailing stop modes (profit-based, time-based, volatility-based)
- Integration with position management and risk controls

@author Trading Orchestrator System
@version 1.0
@date 2025-11-06

@warning
Trailing stops can trigger frequent exits in volatile markets. Consider
the transaction costs and market impact when configuring trailing parameters.

@note
Trailing stops work best in trending markets and may generate more trades
than fixed stop losses. Backtesting is recommended before live deployment.

@see base_exit_strategy.py for base framework
@see volatility_stop.py for volatility-based stops
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
import asyncio

from loguru import logger

from .base_exit_strategy import (
    BaseExitStrategy,
    ExitSignal,
    ExitReason,
    ExitType,
    ExitCondition,
    ExitConfiguration,
    ExitMetrics,
    ExitStatus
)


@dataclass
class TrailingStopConfig:
    """
    @class TrailingStopConfig
    @brief Configuration for trailing stop strategies
    
    @details
    Contains all configuration parameters specific to trailing stop
    exit strategies including trailing distances, update frequencies,
    and volatility adjustments.
    
    @par Trailing Parameters:
    - initial_stop: Initial stop loss level
    - trailing_distance: Distance to maintain from current price
    - min_trailing_distance: Minimum allowed trailing distance
    - max_trailing_distance: Maximum allowed trailing distance
    - trailing_mode: How trailing distance is calculated
    
    @par Update Parameters:
    - update_frequency: How often to update trailing stop (seconds)
    - price_change_threshold: Minimum price change to trigger update
    - volatility_adjustment: Whether to adjust for volatility
    
    @par Advanced Parameters:
    - profit_threshold: Minimum profit before trailing starts
    - activation_delay: Delay before trailing becomes active
    - reset_on_loss: Whether to reset trail on adverse moves
    - multiple_trails: Support for multiple trailing levels
    
    @par Example:
    @code
    config = TrailingStopConfig(
        initial_stop=Decimal('0.95'),  # 5% stop
        trailing_distance=Decimal('0.03'),  # 3% trail
        min_trailing_distance=Decimal('0.01'),  # 1% min
        trailing_mode="percentage",
        update_frequency=60,
        profit_threshold=Decimal('0.02')  # Start trailing after 2% profit
    )
    @endcode
    
    @note
    Conservative settings (wider stops) reduce risk of premature exits
    but may reduce profit protection. Aggressive settings provide tighter
    profit protection but risk whipsaw exits.
    """
    initial_stop: Decimal  # Initial stop loss level
    trailing_distance: Decimal  # Distance to maintain from price
    min_trailing_distance: Decimal = Decimal('0.005')  # 0.5%
    max_trailing_distance: Decimal = Decimal('0.10')   # 10%
    trailing_mode: str = "percentage"  # "percentage", "atr", "fixed"
    update_frequency: int = 60  # seconds
    price_change_threshold: Decimal = Decimal('0.001')  # 0.1%
    volatility_adjustment: bool = True
    profit_threshold: Decimal = Decimal('0.0')  # Start trailing after profit
    activation_delay: int = 0  # seconds
    reset_on_loss: bool = False
    multiple_trails: bool = False
    atr_period: int = 14  # ATR calculation period
    atr_multiplier: Decimal = Decimal('2.0')  # ATR multiplier for stops
    trailing_sensitivity: Decimal = Decimal('1.0')  # Trail speed multiplier


class TrailingStopStrategy(BaseExitStrategy):
    """
    @class TrailingStopStrategy
    @brief Base class for trailing stop exit strategies
    
    @details
    Provides common functionality for all trailing stop strategies including
    trail calculation, position tracking, and signal generation. This class
    serves as the foundation for specific trailing stop implementations.
    
    @par Key Features:
    - Dynamic stop loss adjustment based on price movements
    - Support for multiple trailing modes (percentage, ATR, fixed)
    - Configurable update frequencies and thresholds
    - Position-based trail management
    - Integration with risk management systems
    
    @par Trailing Logic:
    1. Calculate current trailing stop level based on configuration
    2. Compare current stop to position entry price and current price
    3. Adjust stop level only when conditions are met
    4. Generate exit signal when price falls below trail level
    5. Update position trail tracking
    
    @warning
    Trailing stops can be triggered by temporary price movements (whipsaws)
    in volatile markets. Consider using wider trailing distances for volatile
    assets or shorter update frequencies for more responsive stops.
    
    @note
    This class provides the common framework. Specific implementations should
    override the _calculate_trailing_stop method for their specific logic.
    
    @see ATRTrailingStop for volatility-adjusted trailing stops
    @see FixedTrailingStop for percentage-based trailing stops
    """
    
    def __init__(self, config: ExitConfiguration):
        super().__init__(config)
        
        # Trailing stop specific configuration
        self.trailing_config = TrailingStopConfig(**config.parameters)
        
        # Position tracking for trailing stops
        self.position_trails: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.last_trail_updates: Dict[str, datetime] = {}
        self.trail_adjustments: Dict[str, List[Dict[str, Any]]] = {}
        
        # Market data caching
        self.price_history: Dict[str, List[Dict[str, Any]]] = {}
        self.volatility_cache: Dict[str, Decimal] = {}
        
        logger.info(f"Trailing stop strategy initialized: {config.name}")
    
    async def evaluate_exit_conditions(self, position: Dict[str, Any]) -> bool:
        """
        Evaluate trailing stop exit conditions
        
        @param position Position information dictionary
        
        @details
        Checks if the current price has fallen below the trailing stop level
        or if other exit conditions have been met. Considers profit thresholds,
        activation delays, and trail reset conditions.
        
        @returns True if exit should be triggered, False otherwise
        """
        try:
            position_id = position.get('position_id')
            if not position_id:
                return False
            
            # Check if position is eligible for trailing stop evaluation
            if not await self._is_position_eligible(position):
                return False
            
            # Get or initialize trailing stop for this position
            trail_data = await self._get_or_initialize_trail(position)
            
            # Update trailing stop if conditions are met
            await self._update_trailing_stop(position, trail_data)
            
            # Check if exit should be triggered
            exit_triggered = await self._check_exit_trigger(position, trail_data)
            
            if exit_triggered:
                logger.info(f"Trailing stop triggered for position {position_id}")
            
            return exit_triggered
            
        except Exception as e:
            logger.error(f"Error evaluating trailing stop conditions: {e}")
            return False
    
    async def generate_exit_signal(
        self, 
        position: Dict[str, Any], 
        exit_reason: ExitReason
    ) -> Optional[ExitSignal]:
        """
        Generate trailing stop exit signal
        
        @param position Position information dictionary
        @param exit_reason Reason for exit
        
        @details
        Creates an exit signal with trailing stop-specific parameters including
        current trail level, confidence based on trail strength, and urgency
        based on how close price is to the trail level.
        
        @returns ExitSignal if exit should be triggered, None otherwise
        """
        try:
            position_id = position.get('position_id')
            symbol = position.get('symbol')
            quantity = Decimal(str(position.get('quantity', 0)))
            
            if not all([position_id, symbol, quantity]):
                return None
            
            # Get current trailing stop data
            trail_data = self.position_trails.get(position_id)
            if not trail_data:
                return None
            
            # Calculate exit parameters
            current_price = await self.context.get_current_price(symbol)
            trail_level = trail_data.get('current_trail', Decimal('0'))
            
            # Determine confidence based on various factors
            confidence = await self._calculate_exit_confidence(position, trail_data)
            
            # Determine urgency based on how close price is to trail
            urgency = self._calculate_exit_urgency(current_price, trail_level)
            
            # Calculate market impact estimate
            market_impact = await self._estimate_market_impact(symbol, quantity)
            
            # Create exit signal
            exit_signal = ExitSignal(
                signal_id=f"trail_{position_id}_{datetime.utcnow().timestamp()}",
                strategy_id=self.config.strategy_id,
                position_id=position_id,
                symbol=symbol,
                exit_reason=ExitReason.TRAILING_STOP,
                exit_price=current_price,
                exit_quantity=quantity,
                confidence=confidence,
                urgency=urgency,
                estimated_execution_time=timedelta(seconds=30),
                market_impact=market_impact,
                metadata={
                    'trail_level': float(trail_level),
                    'entry_price': float(position.get('entry_price', 0)),
                    'current_price': float(current_price),
                    'trail_type': self.trailing_config.trailing_mode,
                    'trail_adjustments': len(trail_data.get('adjustments', []))
                }
            )
            
            logger.info(f"Generated trailing stop exit signal: {exit_signal.signal_id}")
            return exit_signal
            
        except Exception as e:
            logger.error(f"Error generating trailing stop exit signal: {e}")
            return None
    
    async def _is_position_eligible(self, position: Dict[str, Any]) -> bool:
        """Check if position is eligible for trailing stop evaluation"""
        try:
            # Check if position has required data
            entry_price = position.get('entry_price')
            symbol = position.get('symbol')
            if not entry_price or not symbol:
                return False
            
            # Check activation delay
            if self.trailing_config.activation_delay > 0:
                created_at = position.get('created_at')
                if created_at:
                    age = datetime.utcnow() - created_at
                    if age.total_seconds() < self.trailing_config.activation_delay:
                        return False
            
            # Check profit threshold for activation
            if self.trailing_config.profit_threshold > 0:
                current_price = await self.context.get_current_price(symbol)
                profit_pct = (current_price - entry_price) / entry_price
                if profit_pct < self.trailing_config.profit_threshold:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking position eligibility: {e}")
            return False
    
    async def _get_or_initialize_trail(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """Get or initialize trailing stop data for position"""
        try:
            position_id = position.get('position_id')
            if not position_id:
                return {}
            
            if position_id in self.position_trails:
                return self.position_trails[position_id]
            
            # Initialize new trail data
            trail_data = {
                'position_id': position_id,
                'initial_trail': self.trailing_config.initial_stop,
                'current_trail': self.trailing_config.initial_stop,
                'highest_price': Decimal('0'),
                'lowest_price': Decimal('0'),
                'is_long': position.get('side', 'long') == 'long',
                'adjustments': [],
                'activated_at': datetime.utcnow(),
                'last_update': datetime.utcnow()
            }
            
            # Set initial highest/lowest prices based on entry
            entry_price = Decimal(str(position.get('entry_price', 0)))
            if trail_data['is_long']:
                trail_data['highest_price'] = entry_price
            else:
                trail_data['lowest_price'] = entry_price
            
            self.position_trails[position_id] = trail_data
            logger.info(f"Initialized trailing stop for position {position_id}")
            
            return trail_data
            
        except Exception as e:
            logger.error(f"Error initializing trail for position: {e}")
            return {}
    
    async def _update_trailing_stop(self, position: Dict[str, Any], trail_data: Dict[str, Any]):
        """Update trailing stop level based on current market conditions"""
        try:
            if not trail_data:
                return
            
            symbol = position.get('symbol')
            position_id = position.get('position_id')
            
            # Check if update is needed
            last_update = trail_data.get('last_update')
            if last_update:
                time_since_update = datetime.utcnow() - last_update
                if time_since_update.total_seconds() < self.trailing_config.update_frequency:
                    return
            
            # Get current market data
            current_price = await self.context.get_current_price(symbol)
            if not current_price:
                return
            
            # Calculate new trailing stop level
            new_trail_level = await self._calculate_trailing_stop(
                current_price, trail_data, position
            )
            
            # Validate new trail level
            validated_trail = await self._validate_trail_level(
                new_trail_level, trail_data, position
            )
            
            # Update trail if it has changed
            if validated_trail != trail_data.get('current_trail'):
                old_trail = trail_data.get('current_trail')
                trail_data['current_trail'] = validated_trail
                trail_data['last_update'] = datetime.utcnow()
                
                # Record adjustment
                adjustment = {
                    'timestamp': datetime.utcnow(),
                    'old_trail': old_trail,
                    'new_trail': validated_trail,
                    'current_price': current_price,
                    'reason': 'price_movement'
                }
                trail_data.setdefault('adjustments', []).append(adjustment)
                
                # Track adjustments
                if position_id not in self.trail_adjustments:
                    self.trail_adjustments[position_id] = []
                self.trail_adjustments[position_id].append(adjustment)
                
                logger.info(f"Updated trailing stop for {position_id}: {old_trail} -> {validated_trail}")
            
            # Update highest/lowest prices
            if trail_data.get('is_long', True):
                if current_price > trail_data.get('highest_price', Decimal('0')):
                    trail_data['highest_price'] = current_price
            else:
                if current_price < trail_data.get('lowest_price', Decimal('0')):
                    trail_data['lowest_price'] = current_price
            
        except Exception as e:
            logger.error(f"Error updating trailing stop: {e}")
    
    async def _calculate_trailing_stop(
        self, 
        current_price: Decimal, 
        trail_data: Dict[str, Any], 
        position: Dict[str, Any]
    ) -> Decimal:
        """
        Calculate new trailing stop level
        
        @param current_price Current market price
        @param trail_data Current trailing stop data
        @param position Position information
        
        @details
        Calculates the new trailing stop level based on the configured
        trailing mode. This method should be overridden by specific
        trailing stop implementations.
        
        @returns New trailing stop level
        """
        # Default implementation - should be overridden
        is_long = trail_data.get('is_long', True)
        trailing_distance = self.trailing_config.trailing_distance
        
        if is_long:
            # For long positions, trail is below current price
            return current_price * (Decimal('1') - trailing_distance)
        else:
            # For short positions, trail is above current price
            return current_price * (Decimal('1') + trailing_distance)
    
    async def _validate_trail_level(
        self, 
        new_trail: Decimal, 
        trail_data: Dict[str, Any], 
        position: Dict[str, Any]
    ) -> Decimal:
        """Validate and adjust trailing stop level"""
        try:
            # Ensure trail is within configured bounds
            min_trail = self.trailing_config.min_trailing_distance
            max_trail = self.trailing_config.max_trailing_distance
            
            # Adjust for minimum distance
            if new_trail < min_trail:
                new_trail = min_trail
            elif new_trail > max_trail:
                new_trail = max_trail
            
            # Ensure trail moves in favorable direction only
            current_trail = trail_data.get('current_trail', Decimal('0'))
            is_long = trail_data.get('is_long', True)
            
            if is_long:
                # For long positions, trail should only move up
                if new_trail < current_trail:
                    new_trail = current_trail
            else:
                # For short positions, trail should only move down
                if new_trail > current_trail:
                    new_trail = current_trail
            
            return new_trail
            
        except Exception as e:
            logger.error(f"Error validating trail level: {e}")
            return trail_data.get('current_trail', self.trailing_config.initial_stop)
    
    async def _check_exit_trigger(self, position: Dict[str, Any], trail_data: Dict[str, Any]) -> bool:
        """Check if exit should be triggered"""
        try:
            symbol = position.get('symbol')
            position_id = position.get('position_id')
            
            current_price = await self.context.get_current_price(symbol)
            trail_level = trail_data.get('current_trail', Decimal('0'))
            
            if not current_price or not trail_level:
                return False
            
            is_long = trail_data.get('is_long', True)
            
            # Check if price has breached the trail level
            if is_long:
                exit_triggered = current_price <= trail_level
            else:
                exit_triggered = current_price >= trail_level
            
            # Additional checks for reset conditions
            if self.trailing_config.reset_on_loss and not exit_triggered:
                entry_price = Decimal(str(position.get('entry_price', 0)))
                
                if is_long and current_price < entry_price:
                    # Reset trail for long position in loss
                    exit_triggered = True
                elif not is_long and current_price > entry_price:
                    # Reset trail for short position in loss
                    exit_triggered = True
            
            return exit_triggered
            
        except Exception as e:
            logger.error(f"Error checking exit trigger: {e}")
            return False
    
    async def _calculate_exit_confidence(self, position: Dict[str, Any], trail_data: Dict[str, Any]) -> float:
        """Calculate confidence level for exit signal"""
        try:
            # Base confidence from signal type
            base_confidence = 0.8
            
            # Adjust based on number of trail adjustments (more adjustments = higher confidence)
            num_adjustments = len(trail_data.get('adjustments', []))
            if num_adjustments > 5:
                base_confidence += 0.1
            elif num_adjustments > 10:
                base_confidence += 0.15
            
            # Adjust based on time in position
            activated_at = trail_data.get('activated_at')
            if activated_at:
                time_in_position = (datetime.utcnow() - activated_at).total_seconds()
                if time_in_position > 3600:  # More than 1 hour
                    base_confidence += 0.05
            
            # Adjust based on volatility
            symbol = position.get('symbol')
            if symbol in self.volatility_cache:
                volatility = float(self.volatility_cache[symbol])
                if volatility > 0.05:  # High volatility
                    base_confidence -= 0.1
            
            return min(0.95, max(0.5, base_confidence))
            
        except Exception as e:
            logger.error(f"Error calculating exit confidence: {e}")
            return 0.8
    
    def _calculate_exit_urgency(self, current_price: Decimal, trail_level: Decimal) -> float:
        """Calculate urgency level for exit signal"""
        try:
            if not current_price or not trail_level:
                return 0.5
            
            # Calculate distance between current price and trail
            price_diff = abs(current_price - trail_level)
            relative_distance = price_diff / current_price
            
            # Convert to urgency (closer = more urgent)
            if relative_distance < 0.001:  # Less than 0.1% away
                return 0.9
            elif relative_distance < 0.005:  # Less than 0.5% away
                return 0.7
            elif relative_distance < 0.01:  # Less than 1% away
                return 0.5
            else:
                return 0.3
                
        except Exception as e:
            logger.error(f"Error calculating exit urgency: {e}")
            return 0.5
    
    async def _estimate_market_impact(self, symbol: str, quantity: Decimal) -> Optional[Decimal]:
        """Estimate market impact of exit order"""
        try:
            # Simple market impact estimation based on order size
            # This would be enhanced with real market microstructure data
            current_price = await self.context.get_current_price(symbol)
            if not current_price:
                return None
            
            order_value = quantity * current_price
            
            # Estimate impact as percentage of price (simplified)
            # Larger orders have more impact
            impact_percentage = min(0.001, quantity / Decimal('1000'))  # Max 0.1%
            
            return current_price * impact_percentage
            
        except Exception as e:
            logger.error(f"Error estimating market impact: {e}")
            return None


class ATRTrailingStop(TrailingStopStrategy):
    """
    @class ATRTrailingStop
    @brief ATR-based trailing stop strategy
    
    @details
    Uses Average True Range (ATR) to dynamically adjust the trailing stop
    distance based on market volatility. This provides more adaptive
    stop placement that responds to changing market conditions.
    
    @par ATR Calculation:
    - Uses configurable ATR period (default: 14)
    - Multiplies ATR by configurable multiplier (default: 2.0)
    - Adjusts trailing distance based on current volatility
    
    @par Benefits:
    - Adapts to changing market volatility
    - Provides better risk-adjusted stops
    - Reduces false signals in volatile markets
    - Automatically widens stops during high volatility periods
    
    @par Configuration:
    @code
    config = ATRTrailingStop.create_config(
        strategy_id="atr_trail_001",
        symbol="AAPL",
        atr_period=14,
        atr_multiplier=Decimal('2.5'),
        initial_stop=Decimal('0.05'),
        min_atr_distance=Decimal('0.01')
    )
    @endcode
    
    @warning
    ATR-based stops can be wider during high volatility periods, which may
    reduce their effectiveness in fast-moving markets. Consider using
    multiple trailing levels for better protection.
    
    @note
    ATR calculation requires sufficient historical data. Ensure adequate
    data is available for accurate volatility measurement.
    """
    
    def __init__(self, config: ExitConfiguration):
        # Set ATR-specific defaults
        if 'atr_period' not in config.parameters:
            config.parameters['atr_period'] = 14
        if 'atr_multiplier' not in config.parameters:
            config.parameters['atr_multiplier'] = Decimal('2.0')
        
        super().__init__(config)
        
        # ATR-specific configuration
        self.atr_period = config.parameters.get('atr_period', 14)
        self.atr_multiplier = config.parameters.get('atr_multiplier', Decimal('2.0'))
        self.min_atr_distance = config.parameters.get('min_atr_distance', Decimal('0.01'))
        
        logger.info(f"ATR trailing stop initialized with period={self.atr_period}, multiplier={self.atr_multiplier}")
    
    async def _calculate_trailing_stop(
        self, 
        current_price: Decimal, 
        trail_data: Dict[str, Any], 
        position: Dict[str, Any]
    ) -> Decimal:
        """Calculate ATR-based trailing stop level"""
        try:
            symbol = position.get('symbol')
            is_long = trail_data.get('is_long', True)
            
            # Get ATR value
            atr_value = await self._get_atr_value(symbol)
            
            # Convert ATR to percentage of price
            atr_percentage = (atr_value / current_price) if current_price > 0 else Decimal('0')
            
            # Calculate trailing distance with ATR adjustment
            base_distance = self.trailing_config.trailing_distance
            atr_adjusted_distance = (atr_percentage * self.atr_multiplier) + base_distance
            
            # Apply sensitivity adjustment
            sensitivity = self.trailing_config.trailing_sensitivity
            adjusted_distance = atr_adjusted_distance * sensitivity
            
            # Apply bounds
            adjusted_distance = max(
                self.trailing_config.min_trailing_distance,
                min(self.trailing_config.max_trailing_distance, adjusted_distance)
            )
            
            # Calculate absolute trail level
            if is_long:
                trail_level = current_price * (Decimal('1') - adjusted_distance)
            else:
                trail_level = current_price * (Decimal('1') + adjusted_distance)
            
            # Ensure minimum ATR distance
            min_distance = self.min_atr_distance * current_price
            if is_long:
                min_trail = current_price - min_distance
                trail_level = max(trail_level, min_trail)
            else:
                max_trail = current_price + min_distance
                trail_level = min(trail_level, max_trail)
            
            return trail_level
            
        except Exception as e:
            logger.error(f"Error calculating ATR trailing stop: {e}")
            # Fall back to simple percentage-based trailing stop
            return await super()._calculate_trailing_stop(current_price, trail_data, position)
    
    async def _get_atr_value(self, symbol: str) -> Decimal:
        """Get ATR value for symbol"""
        try:
            # Check cache first
            if symbol in self.volatility_cache:
                cache_time = self.price_history.get(f"{symbol}_atr_time", datetime.min)
                if (datetime.utcnow() - cache_time).total_seconds() < 300:  # 5 minute cache
                    return self.volatility_cache[symbol]
            
            # Get historical data for ATR calculation
            if not self.context:
                return Decimal('0')
            
            historical_data = await self.context.get_historical_data(
                symbol, '1h', self.atr_period + 1
            )
            
            if not historical_data or len(historical_data) < self.atr_period + 1:
                return Decimal('0')
            
            # Calculate ATR
            atr_values = []
            for i in range(1, len(historical_data)):
                current_high = Decimal(str(historical_data[i].get('high', 0)))
                current_low = Decimal(str(historical_data[i].get('low', 0)))
                prev_close = Decimal(str(historical_data[i-1].get('close', 0)))
                
                # Calculate True Range components
                tr1 = current_high - current_low
                tr2 = abs(current_high - prev_close)
                tr3 = abs(current_low - prev_close)
                
                true_range = max(tr1, tr2, tr3)
                atr_values.append(true_range)
            
            # Calculate ATR as average of True Ranges
            if atr_values:
                atr = sum(atr_values) / len(atr_values)
                self.volatility_cache[symbol] = atr
                self.price_history[f"{symbol}_atr_time"] = datetime.utcnow()
                return atr
            
            return Decimal('0')
            
        except Exception as e:
            logger.error(f"Error calculating ATR for {symbol}: {e}")
            return Decimal('0')
    
    @classmethod
    def create_config(
        cls,
        strategy_id: str,
        symbol: str,
        initial_stop: Decimal = Decimal('0.05'),
        atr_period: int = 14,
        atr_multiplier: Decimal = Decimal('2.0'),
        min_atr_distance: Decimal = Decimal('0.01'),
        **kwargs
    ) -> ExitConfiguration:
        """Create configuration for ATR trailing stop strategy"""
        parameters = {
            'initial_stop': initial_stop,
            'trailing_mode': 'atr',
            'atr_period': atr_period,
            'atr_multiplier': atr_multiplier,
            'min_atr_distance': min_atr_distance,
            'volatility_adjustment': True,
            **kwargs
        }
        
        return ExitConfiguration(
            strategy_id=strategy_id,
            strategy_type=ExitType.TRAILING_STOP,
            name=f"ATR Trailing Stop ({symbol})",
            description=f"ATR-based trailing stop for {symbol}",
            parameters=parameters,
            symbols=[symbol]
        )


class FixedTrailingStop(TrailingStopStrategy):
    """
    @class FixedTrailingStop
    @brief Fixed percentage trailing stop strategy
    
    @details
    Uses a fixed percentage distance for the trailing stop, regardless of
    market volatility or price movements. This provides simple and predictable
    trailing stop behavior.
    
    @par Fixed Percentage Logic:
    - Maintains constant percentage distance from current price
    - Stops only move in favorable direction
    - Simple calculation and implementation
    - No volatility adjustment
    
    @par Benefits:
    - Simple and predictable behavior
    - Easy to understand and configure
    - Suitable for consistent risk management
    - Lower computational requirements
    
    @par Configuration:
    @code
    config = FixedTrailingStop.create_config(
        strategy_id="fixed_trail_001",
        symbol="AAPL",
        trailing_percentage=Decimal('0.03'),  # 3% trail
        initial_stop=Decimal('0.05')  # 5% initial stop
    )
    @endcode
    
    @warning
    Fixed percentage stops don't adapt to changing volatility. In highly
    volatile markets, consider using ATR-based stops for better risk management.
    
    @note
    This is the most straightforward trailing stop implementation and is
    suitable for most trading applications where volatility is relatively stable.
    """
    
    def __init__(self, config: ExitConfiguration):
        super().__init__(config)
        
        # Fixed percentage-specific configuration
        self.trailing_percentage = config.parameters.get('trailing_percentage', Decimal('0.03'))
        
        logger.info(f"Fixed trailing stop initialized with percentage={self.trailing_percentage}")
    
    async def _calculate_trailing_stop(
        self, 
        current_price: Decimal, 
        trail_data: Dict[str, Any], 
        position: Dict[str, Any]
    ) -> Decimal:
        """Calculate fixed percentage trailing stop level"""
        try:
            is_long = trail_data.get('is_long', True)
            
            # Calculate trail based on fixed percentage
            if is_long:
                # For long positions, trail is below current price
                trail_level = current_price * (Decimal('1') - self.trailing_percentage)
            else:
                # For short positions, trail is above current price
                trail_level = current_price * (Decimal('1') + self.trailing_percentage)
            
            return trail_level
            
        except Exception as e:
            logger.error(f"Error calculating fixed trailing stop: {e}")
            return await super()._calculate_trailing_stop(current_price, trail_data, position)
    
    @classmethod
    def create_config(
        cls,
        strategy_id: str,
        symbol: str,
        trailing_percentage: Decimal = Decimal('0.03'),
        initial_stop: Decimal = Decimal('0.05'),
        **kwargs
    ) -> ExitConfiguration:
        """Create configuration for fixed trailing stop strategy"""
        parameters = {
            'initial_stop': initial_stop,
            'trailing_percentage': trailing_percentage,
            'trailing_mode': 'percentage',
            'volatility_adjustment': False,
            **kwargs
        }
        
        return ExitConfiguration(
            strategy_id=strategy_id,
            strategy_type=ExitType.TRAILING_STOP,
            name=f"Fixed Trailing Stop ({symbol})",
            description=f"Fixed percentage trailing stop for {symbol}",
            parameters=parameters,
            symbols=[symbol]
        )


# Factory functions for easy strategy creation

def create_trailing_stop_strategy(
    strategy_id: str,
    symbol: str,
    trailing_distance: Decimal = Decimal('0.03'),
    initial_stop: Decimal = Decimal('0.05'),
    trailing_mode: str = "percentage",
    **kwargs
) -> TrailingStopStrategy:
    """
    Create a trailing stop exit strategy
    
    @param strategy_id Unique identifier for the strategy
    @param symbol Trading symbol to monitor
    @param trailing_distance Distance to maintain from price
    @param initial_stop Initial stop loss level
    @param trailing_mode Type of trailing ("percentage", "atr", "fixed")
    @param kwargs Additional configuration parameters
    
    @returns Configured trailing stop strategy instance
    """
    # Create base configuration
    config = ExitConfiguration(
        strategy_id=strategy_id,
        strategy_type=ExitType.TRAILING_STOP,
        name=f"Trailing Stop ({symbol})",
        description=f"Trailing stop strategy for {symbol}",
        parameters={
            'initial_stop': initial_stop,
            'trailing_distance': trailing_distance,
            'trailing_mode': trailing_mode,
            **kwargs
        },
        symbols=[symbol]
    )
    
    # Create appropriate strategy class
    if trailing_mode.lower() == 'atr':
        return ATRTrailingStop(config)
    elif trailing_mode.lower() == 'fixed':
        return FixedTrailingStop(config)
    else:
        return TrailingStopStrategy(config)


def create_atr_trailing_stop(
    strategy_id: str,
    symbol: str,
    atr_period: int = 14,
    atr_multiplier: Decimal = Decimal('2.0'),
    initial_stop: Decimal = Decimal('0.05'),
    **kwargs
) -> ATRTrailingStop:
    """Create ATR-based trailing stop strategy"""
    config = ATRTrailingStop.create_config(
        strategy_id=strategy_id,
        symbol=symbol,
        initial_stop=initial_stop,
        atr_period=atr_period,
        atr_multiplier=atr_multiplier,
        **kwargs
    )
    return ATRTrailingStop(config)


def create_fixed_trailing_stop(
    strategy_id: str,
    symbol: str,
    trailing_percentage: Decimal = Decimal('0.03'),
    initial_stop: Decimal = Decimal('0.05'),
    **kwargs
) -> FixedTrailingStop:
    """Create fixed percentage trailing stop strategy"""
    config = FixedTrailingStop.create_config(
        strategy_id=strategy_id,
        symbol=symbol,
        trailing_percentage=trailing_percentage,
        initial_stop=initial_stop,
        **kwargs
    )
    return FixedTrailingStop(config)
