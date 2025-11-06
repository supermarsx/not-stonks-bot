"""
@file stop_loss.py
@brief Stop Loss Exit Strategy Implementation

@details
This module implements various stop loss exit strategies that limit losses
when trades move against the expected direction. Stop losses are fundamental
risk management tools that help protect capital and limit drawdowns.

Key Features:
- Percentage-based stop losses
- Volatility-adjusted stop losses
- Multiple stop loss activation modes
- Stop loss modification capabilities
- Integration with trailing stops and profit targets

@author Trading Orchestrator System
@version 1.0
@date 2025-11-06

@warning
Stop losses can be triggered by temporary price movements (whipsaws) in volatile
markets. Consider using wider stops or volatility-adjusted stops for highly
volatile assets.

@note
Stop losses should be set based on position size, account risk tolerance, and
historical volatility. Overly tight stops may increase transaction costs.

@see base_exit_strategy.py for base framework
@see volatility_stop.py for volatility-adjusted stops
@see trailing_stop.py for dynamic trailing stops
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
class StopLossConfig:
    """
    @class StopLossConfig
    @brief Configuration for stop loss exit strategies
    
    @details
    Contains all configuration parameters for stop loss exit strategies
    including stop levels, activation modes, volatility adjustments, and
    management rules.
    
    @par Stop Parameters:
    - stop_percentage: Percentage below entry price for stop loss
    - stop_price: Fixed price level for stop loss
    - stop_mode: How stop level is calculated
    
    @par Activation Parameters:
    - activation_delay: Delay before stop becomes active
    - activation_threshold: Minimum movement before activation
    - stop_adjustment: Whether stop can be moved
    
    @par Advanced Parameters:
    - volatility_adjustment: Whether to adjust stop for volatility
    - stop_increment: Minimum price increment for stop moves
    - max_stop_distance: Maximum allowed distance from entry
    - stop_reset_on_profit: Whether to reset stop on profit moves
    
    @par Example:
    @code
    config = StopLossConfig(
        stop_percentage=Decimal('0.03'),  # 3% stop
        stop_mode="percentage",
        activation_delay=300,  # 5 minutes
        stop_adjustment=True,
        volatility_adjustment=True,
        stop_reset_on_profit=True
    )
    @endcode
    
    @note
    More conservative settings (wider stops) provide more stability but
    reduce capital efficiency. Consider the asset's typical daily range
    when setting stop levels.
    """
    stop_percentage: Optional[Decimal] = None  # Percentage-based stop
    stop_price: Optional[Decimal] = None      # Fixed price stop
    stop_mode: str = "percentage"  # "percentage", "fixed", "volatility"
    activation_delay: int = 0  # seconds
    activation_threshold: Decimal = Decimal('0')  # Minimum movement before activation
    stop_adjustment: bool = False
    stop_increment: Decimal = Decimal('0.01')
    max_stop_distance: Optional[Decimal] = None
    stop_reset_on_profit: bool = False
    volatility_adjustment: bool = False
    atr_period: int = 14
    atr_multiplier: Decimal = Decimal('2.0')


class StopLossStrategy(BaseExitStrategy):
    """
    @class StopLossStrategy
    @brief Base class for stop loss exit strategies
    
    @details
    Provides common functionality for all stop loss exit strategies
    including stop calculation, position tracking, and activation
    management. This class serves as the foundation for specific
    stop loss implementations.
    
    @par Key Features:
    - Dynamic stop level calculation
    - Stop activation and management
    - Stop adjustment capabilities
    - Volatility-based adjustments
    - Integration with position management
    
    @par Stop Loss Logic:
    1. Calculate stop level based on configuration
    2. Monitor position for stop activation
    3. Allow stop adjustments if configured
    4. Trigger exit when stop level is breached
    5. Manage stop-related position tracking
    
    @warning
    Stop losses provide protection but may be triggered by normal market
    volatility. Consider using volatility-adjusted stops for highly
    volatile assets or combining with other exit strategies.
    
    @note
    This class provides the common framework. Specific implementations should
    override stop calculation methods for their specific logic.
    
    @see PercentageStopLoss for percentage-based stops
    @see VolatilityStopLoss for volatility-adjusted stops
    """
    
    def __init__(self, config: ExitConfiguration):
        super().__init__(config)
        
        # Stop loss specific configuration
        self.stop_config = StopLossConfig(**config.parameters)
        
        # Stop tracking
        self.position_stops: Dict[str, Dict[str, Any]] = {}
        self.stop_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Performance tracking
        self.stop_activations: Dict[str, List[Dict[str, Any]]] = {}
        self.stop_adjustments: Dict[str, List[Dict[str, Any]]] = {}
        
        # Market data cache for volatility calculations
        self.price_history: Dict[str, List[Dict[str, Any]]] = {}
        self.volatility_cache: Dict[str, Decimal] = {}
        
        logger.info(f"Stop loss strategy initialized: {config.name}")
    
    async def evaluate_exit_conditions(self, position: Dict[str, Any]) -> bool:
        """
        Evaluate stop loss exit conditions
        
        @param position Position information dictionary
        
        @details
        Checks if the current price has reached or breached the stop loss
        level or if other stop-related exit conditions have been met.
        Handles stop activation delays, adjustment logic, and reset conditions.
        
        @returns True if exit should be triggered, False otherwise
        """
        try:
            position_id = position.get('position_id')
            if not position_id:
                return False
            
            # Check if position is eligible for stop evaluation
            if not await self._is_position_eligible(position):
                return False
            
            # Get or initialize stop loss for this position
            stop_data = await self._get_or_initialize_stop(position)
            
            # Update stop level if configured to allow adjustments
            if self.stop_config.stop_adjustment:
                await self._update_stop_level(position, stop_data)
            
            # Check if stop has been triggered
            stop_triggered = await self._check_stop_triggered(position, stop_data)
            
            if stop_triggered:
                logger.info(f"Stop loss triggered for position {position_id}")
            
            return stop_triggered
            
        except Exception as e:
            logger.error(f"Error evaluating stop loss conditions: {e}")
            return False
    
    async def generate_exit_signal(
        self, 
        position: Dict[str, Any], 
        exit_reason: ExitReason
    ) -> Optional[ExitSignal]:
        """
        Generate stop loss exit signal
        
        @param position Position information dictionary
        @param exit_reason Reason for exit
        
        @details
        Creates an exit signal with stop loss-specific parameters including
        current stop level, confidence based on stop breach clarity, and
        urgency based on how severely the stop has been breached.
        
        @returns ExitSignal if exit should be triggered, None otherwise
        """
        try:
            position_id = position.get('position_id')
            symbol = position.get('symbol')
            quantity = Decimal(str(position.get('quantity', 0)))
            
            if not all([position_id, symbol, quantity]):
                return None
            
            # Get current stop data
            stop_data = self.position_stops.get(position_id)
            if not stop_data:
                return None
            
            # Calculate exit parameters
            current_price = await self.context.get_current_price(symbol)
            stop_price = stop_data.get('current_stop', Decimal('0'))
            
            if not current_price or not stop_price:
                return None
            
            # Determine confidence and urgency
            confidence = await self._calculate_stop_confidence(position, stop_data, current_price)
            urgency = await self._calculate_stop_urgency(current_price, stop_price)
            market_impact = await self._estimate_market_impact(symbol, quantity)
            
            # Create exit signal
            exit_signal = ExitSignal(
                signal_id=f"stop_{position_id}_{datetime.utcnow().timestamp()}",
                strategy_id=self.config.strategy_id,
                position_id=position_id,
                symbol=symbol,
                exit_reason=ExitReason.STOP_LOSS,
                exit_price=current_price,
                exit_quantity=quantity,
                confidence=confidence,
                urgency=urgency,
                estimated_execution_time=timedelta(seconds=30),
                market_impact=market_impact,
                metadata={
                    'stop_price': float(stop_price),
                    'stop_level': float(stop_data.get('initial_stop', 0)),
                    'stop_mode': self.stop_config.stop_mode,
                    'stop_adjustments': len(stop_data.get('adjustments', [])),
                    'entry_price': float(position.get('entry_price', 0)),
                    'current_price': float(current_price),
                    'breach_severity': self._calculate_breach_severity(current_price, stop_price)
                }
            )
            
            logger.info(f"Generated stop loss exit signal: {exit_signal.signal_id}")
            return exit_signal
            
        except Exception as e:
            logger.error(f"Error generating stop loss exit signal: {e}")
            return None
    
    async def _is_position_eligible(self, position: Dict[str, Any]) -> bool:
        """Check if position is eligible for stop evaluation"""
        try:
            # Check if position has required data
            entry_price = position.get('entry_price')
            if not entry_price:
                return False
            
            # Check activation delay
            if self.stop_config.activation_delay > 0:
                created_at = position.get('created_at')
                if created_at:
                    age = datetime.utcnow() - created_at
                    if age.total_seconds() < self.stop_config.activation_delay:
                        return False
            
            # Check activation threshold
            if self.stop_config.activation_threshold > 0:
                symbol = position.get('symbol')
                current_price = await self.context.get_current_price(symbol)
                if current_price:
                    price_change = abs((current_price - entry_price) / entry_price)
                    if price_change < self.stop_config.activation_threshold:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking position eligibility: {e}")
            return False
    
    async def _get_or_initialize_stop(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """Get or initialize stop loss data for position"""
        try:
            position_id = position.get('position_id')
            if not position_id:
                return {}
            
            if position_id in self.position_stops:
                return self.position_stops[position_id]
            
            # Initialize new stop data
            stop_data = {
                'position_id': position_id,
                'initial_stop': await self._calculate_initial_stop(position),
                'current_stop': Decimal('0'),
                'is_long': position.get('side', 'long') == 'long',
                'activated_at': datetime.utcnow(),
                'activated': False,
                'adjustments': [],
                'last_update': datetime.utcnow()
            }
            
            # Set initial stop price
            stop_data['current_stop'] = stop_data['initial_stop']
            
            self.position_stops[position_id] = stop_data
            logger.info(f"Initialized stop loss for position {position_id}")
            
            return stop_data
            
        except Exception as e:
            logger.error(f"Error initializing stop for position: {e}")
            return {}
    
    async def _calculate_initial_stop(self, position: Dict[str, Any]) -> Decimal:
        """Calculate initial stop level for position"""
        try:
            entry_price = Decimal(str(position.get('entry_price', 0)))
            if not entry_price:
                return Decimal('0')
            
            # Calculate stop based on configuration
            if self.stop_config.stop_price:
                # Fixed price stop
                return self.stop_config.stop_price
            elif self.stop_config.stop_percentage:
                # Percentage-based stop
                percentage = self.stop_config.stop_percentage
                if self.stop_config.volatility_adjustment:
                    # Apply volatility adjustment
                    volatility = await self._get_volatility(position.get('symbol', ''))
                    if volatility:
                        adjusted_percentage = percentage * (Decimal('1') + volatility * self.stop_config.atr_multiplier)
                        percentage = adjusted_percentage
                
                # Calculate absolute stop price
                is_long = position.get('side', 'long') == 'long'
                if is_long:
                    return entry_price * (Decimal('1') - percentage)
                else:
                    return entry_price * (Decimal('1') + percentage)
            else:
                # Default: 3% stop
                is_long = position.get('side', 'long') == 'long'
                if is_long:
                    return entry_price * Decimal('0.97')
                else:
                    return entry_price * Decimal('1.03')
                    
        except Exception as e:
            logger.error(f"Error calculating initial stop: {e}")
            return Decimal('0')
    
    async def _get_volatility(self, symbol: str) -> Optional[Decimal]:
        """Get volatility for symbol"""
        try:
            if not symbol:
                return None
            
            if symbol in self.volatility_cache:
                return self.volatility_cache[symbol]
            
            if not self.context:
                return None
            
            # Calculate ATR-based volatility
            historical_data = await self.context.get_historical_data(
                symbol, '1h', self.stop_config.atr_period + 1
            )
            
            if not historical_data or len(historical_data) < 2:
                return None
            
            # Calculate ATR
            atr_values = []
            for i in range(1, len(historical_data)):
                current_high = Decimal(str(historical_data[i].get('high', 0)))
                current_low = Decimal(str(historical_data[i].get('low', 0)))
                prev_close = Decimal(str(historical_data[i-1].get('close', 0)))
                
                # Calculate True Range
                tr1 = current_high - current_low
                tr2 = abs(current_high - prev_close)
                tr3 = abs(current_low - prev_close)
                
                true_range = max(tr1, tr2, tr3)
                atr_values.append(true_range)
            
            # Calculate ATR
            if atr_values:
                atr = sum(atr_values) / len(atr_values)
                current_price = Decimal(str(historical_data[-1].get('close', 0)))
                volatility = atr / current_price if current_price > 0 else Decimal('0')
                
                self.volatility_cache[symbol] = volatility
                return volatility
            
            return None
            
        except Exception as e:
            logger.error(f"Error calculating volatility for {symbol}: {e}")
            return None
    
    async def _update_stop_level(self, position: Dict[str, Any], stop_data: Dict[str, Any]):
        """Update stop level if configured to allow adjustments"""
        try:
            symbol = position.get('symbol')
            current_price = await self.context.get_current_price(symbol)
            if not current_price:
                return
            
            entry_price = Decimal(str(position.get('entry_price', 0)))
            current_stop = stop_data.get('current_stop', Decimal('0'))
            is_long = stop_data.get('is_long', True)
            
            # Check if stop should be adjusted based on profit
            if self.stop_config.stop_reset_on_profit and current_stop > 0:
                # Check if position is in profit
                if is_long and current_price > entry_price:
                    # Calculate new stop based on current price
                    new_stop = await self._calculate_adjusted_stop(current_price, is_long)
                    if new_stop > current_stop:
                        # Move stop up for long position in profit
                        old_stop = current_stop
                        stop_data['current_stop'] = new_stop
                        stop_data['adjustments'].append({
                            'timestamp': datetime.utcnow(),
                            'old_stop': old_stop,
                            'new_stop': new_stop,
                            'reason': 'profit_move'
                        })
                        logger.info(f"Adjusted stop for {position.get('position_id')}: {old_stop} -> {new_stop}")
                
                elif not is_long and current_price < entry_price:
                    # Calculate new stop based on current price
                    new_stop = await self._calculate_adjusted_stop(current_price, is_long)
                    if new_stop < current_stop:
                        # Move stop down for short position in profit
                        old_stop = current_stop
                        stop_data['current_stop'] = new_stop
                        stop_data['adjustments'].append({
                            'timestamp': datetime.utcnow(),
                            'old_stop': old_stop,
                            'new_stop': new_stop,
                            'reason': 'profit_move'
                        })
                        logger.info(f"Adjusted stop for {position.get('position_id')}: {old_stop} -> {new_stop}")
            
            # Update timestamp
            stop_data['last_update'] = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error updating stop level: {e}")
    
    async def _calculate_adjusted_stop(self, current_price: Decimal, is_long: bool) -> Decimal:
        """Calculate adjusted stop level based on current price"""
        try:
            # Calculate stop based on configuration
            if self.stop_config.stop_percentage:
                percentage = self.stop_config.stop_percentage
                if is_long:
                    return current_price * (Decimal('1') - percentage)
                else:
                    return current_price * (Decimal('1') + percentage)
            elif self.stop_config.stop_price:
                # Fixed price - don't adjust
                return self.stop_config.stop_price
            else:
                # Default adjustment
                if is_long:
                    return current_price * Decimal('0.97')
                else:
                    return current_price * Decimal('1.03')
                    
        except Exception as e:
            logger.error(f"Error calculating adjusted stop: {e}")
            return Decimal('0')
    
    async def _check_stop_triggered(self, position: Dict[str, Any], stop_data: Dict[str, Any]) -> bool:
        """Check if stop loss has been triggered"""
        try:
            symbol = position.get('symbol')
            current_price = await self.context.get_current_price(symbol)
            stop_price = stop_data.get('current_stop', Decimal('0'))
            is_long = stop_data.get('is_long', True)
            
            if not current_price or not stop_price:
                return False
            
            # Mark stop as activated when price first breaches
            if not stop_data.get('activated', False):
                if is_long:
                    stop_triggered = current_price <= stop_price
                else:
                    stop_triggered = current_price >= stop_price
                
                if stop_triggered:
                    stop_data['activated'] = True
                    stop_data['triggered_at'] = datetime.utcnow()
                    
                    # Record stop activation
                    position_id = position.get('position_id')
                    if position_id not in self.stop_activations:
                        self.stop_activations[position_id] = []
                    
                    self.stop_activations[position_id].append({
                        'triggered_at': datetime.utcnow(),
                        'stop_price': stop_price,
                        'current_price': current_price,
                        'breach_amount': abs(current_price - stop_price)
                    })
            
            return stop_data.get('activated', False)
            
        except Exception as e:
            logger.error(f"Error checking stop trigger: {e}")
            return False
    
    async def _calculate_stop_confidence(
        self, 
        position: Dict[str, Any], 
        stop_data: Dict[str, Any], 
        current_price: Decimal
    ) -> float:
        """Calculate confidence level for stop exit signal"""
        try:
            # Base confidence from stop type
            base_confidence = 0.85
            
            # Adjust based on breach severity
            stop_price = stop_data.get('current_stop', Decimal('0'))
            if stop_price > 0:
                breach_percentage = abs(current_price - stop_price) / current_price
                if breach_percentage > 0.01:  # More than 1% breach
                    base_confidence += 0.1
                elif breach_percentage > 0.005:  # More than 0.5% breach
                    base_confidence += 0.05
            
            # Adjust based on stop adjustment history
            num_adjustments = len(stop_data.get('adjustments', []))
            if num_adjustments > 0:
                base_confidence -= 0.05  # Slightly lower confidence for adjusted stops
            
            return min(0.95, max(0.7, base_confidence))
            
        except Exception as e:
            logger.error(f"Error calculating stop confidence: {e}")
            return 0.85
    
    async def _calculate_stop_urgency(self, current_price: Decimal, stop_price: Decimal) -> float:
        """Calculate urgency level for stop exit signal"""
        try:
            if not current_price or not stop_price:
                return 0.8
            
            # Calculate how far price is beyond stop
            breach_amount = abs(current_price - stop_price)
            relative_breach = breach_amount / current_price
            
            # Higher urgency for larger breaches
            if relative_breach > 0.02:  # More than 2% beyond stop
                return 0.95
            elif relative_breach > 0.01:  # More than 1% beyond stop
                return 0.85
            else:
                return 0.75
                
        except Exception as e:
            logger.error(f"Error calculating stop urgency: {e}")
            return 0.8
    
    def _calculate_breach_severity(self, current_price: Decimal, stop_price: Decimal) -> str:
        """Calculate breach severity classification"""
        try:
            if not current_price or not stop_price or stop_price == 0:
                return "unknown"
            
            breach_percentage = abs(current_price - stop_price) / current_price
            
            if breach_percentage > 0.05:
                return "severe"
            elif breach_percentage > 0.02:
                return "significant"
            elif breach_percentage > 0.01:
                return "moderate"
            else:
                return "minor"
                
        except Exception as e:
            logger.error(f"Error calculating breach severity: {e}")
            return "unknown"
    
    async def _estimate_market_impact(self, symbol: str, quantity: Decimal) -> Optional[Decimal]:
        """Estimate market impact of stop loss order"""
        try:
            # Stop loss orders often have higher market impact due to urgency
            if not self.context:
                return None
            
            current_price = await self.context.get_current_price(symbol)
            if not current_price:
                return None
            
            order_value = quantity * current_price
            
            # Higher impact for stop loss orders due to urgency
            impact_percentage = min(0.003, quantity / Decimal('500'))  # Max 0.3%
            
            return current_price * impact_percentage
            
        except Exception as e:
            logger.error(f"Error estimating market impact: {e}")
            return None


class PercentageStopLoss(StopLossStrategy):
    """
    @class PercentageStopLoss
    @brief Percentage-based stop loss strategy
    
    @details
    Uses a fixed percentage distance from the entry price as the stop loss
    level. This provides simple and consistent risk management across all
    positions regardless of absolute price levels.
    
    @par Percentage Logic:
    - Calculates stop as percentage below entry price (long) or above (short)
    - Simple and predictable risk management
    - Consistent percentage risk across positions
    - Easy to understand and configure
    
    @par Benefits:
    - Simple percentage-based risk management
    - Consistent risk across different price levels
    - Easy to backtest and optimize
    - Suitable for most trading applications
    
    @par Configuration:
    @code
    config = PercentageStopLoss.create_config(
        strategy_id="pct_stop_001",
        symbol="AAPL",
        stop_percentage=Decimal('0.03'),  # 3% stop
        activation_delay=300,  # 5 minutes
        stop_adjustment=True
    )
    @endcode
    
    @warning
    Percentage stops don't account for asset-specific volatility. Highly
    volatile assets may need wider stops, while low volatility assets
    can use tighter stops.
    
    @note
    This is the most common and straightforward stop loss implementation
    and is suitable for most trading strategies.
    """
    
    def __init__(self, config: ExitConfiguration):
        # Ensure we have percentage configuration
        if 'stop_percentage' not in config.parameters:
            config.parameters['stop_percentage'] = Decimal('0.03')
        
        super().__init__(config)
        
        # Percentage-specific configuration
        self.stop_percentage = config.parameters.get('stop_percentage', Decimal('0.03'))
        
        logger.info(f"Percentage stop loss initialized with percentage={self.stop_percentage}")
    
    async def _calculate_initial_stop(self, position: Dict[str, Any]) -> Decimal:
        """Calculate percentage-based initial stop level"""
        try:
            entry_price = Decimal(str(position.get('entry_price', 0)))
            if not entry_price:
                return Decimal('0')
            
            is_long = position.get('side', 'long') == 'long'
            percentage = self.stop_percentage
            
            if is_long:
                # For long positions, stop is below entry
                return entry_price * (Decimal('1') - percentage)
            else:
                # For short positions, stop is above entry
                return entry_price * (Decimal('1') + percentage)
                
        except Exception as e:
            logger.error(f"Error calculating percentage stop: {e}")
            return Decimal('0')
    
    @classmethod
    def create_config(
        cls,
        strategy_id: str,
        symbol: str,
        stop_percentage: Decimal = Decimal('0.03'),
        activation_delay: int = 0,
        stop_adjustment: bool = False,
        **kwargs
    ) -> ExitConfiguration:
        """Create configuration for percentage stop loss strategy"""
        parameters = {
            'stop_percentage': stop_percentage,
            'stop_mode': 'percentage',
            'activation_delay': activation_delay,
            'stop_adjustment': stop_adjustment,
            **kwargs
        }
        
        return ExitConfiguration(
            strategy_id=strategy_id,
            strategy_type=ExitType.STOP_LOSS,
            name=f"Percentage Stop Loss ({symbol})",
            description=f"Percentage-based stop loss for {symbol}",
            parameters=parameters,
            symbols=[symbol]
        )


class VolatilityStopLoss(StopLossStrategy):
    """
    @class VolatilityStopLoss
    @brief Volatility-adjusted stop loss strategy
    
    @details
    Uses volatility measures (typically ATR) to calculate dynamic stop
    loss levels that adapt to changing market conditions. This provides
    more appropriate risk management for assets with varying volatility.
    
    @par Volatility Logic:
    - Uses ATR to measure current market volatility
    - Multiplies ATR by configurable multiplier
    - Calculates stop distance based on volatility
    - Adapts to changing market conditions
    
    @par Benefits:
    - Adapts to changing market volatility
    - Provides more appropriate risk levels
    - Reduces false stop triggers in volatile markets
    - Better risk-adjusted position sizing
    
    @par Configuration:
    @code
    config = VolatilityStopLoss.create_config(
        strategy_id="vol_stop_001",
        symbol="AAPL",
        atr_period=14,
        atr_multiplier=Decimal('2.0'),  # 2x ATR
        min_stop_percentage=Decimal('0.02'),  # Minimum 2%
        max_stop_percentage=Decimal('0.10')   # Maximum 10%
    )
    @endcode
    
    @warning
    Volatility-based stops can become very wide during high volatility
    periods, potentially reducing their effectiveness as risk management tools.
    Consider combining with position sizing adjustments.
    
    @note
    ATR calculation requires sufficient historical data. Ensure adequate
    data is available for accurate volatility measurement.
    """
    
    def __init__(self, config: ExitConfiguration):
        super().__init__(config)
        
        # Volatility-specific configuration
        self.atr_period = config.parameters.get('atr_period', 14)
        self.atr_multiplier = config.parameters.get('atr_multiplier', Decimal('2.0'))
        self.min_stop_percentage = config.parameters.get('min_stop_percentage', Decimal('0.02'))
        self.max_stop_percentage = config.parameters.get('max_stop_percentage', Decimal('0.10'))
        
        logger.info(f"Volatility stop loss initialized with ATR period={self.atr_period}, multiplier={self.atr_multiplier}")
    
    async def _calculate_initial_stop(self, position: Dict[str, Any]) -> Decimal:
        """Calculate volatility-based initial stop level"""
        try:
            entry_price = Decimal(str(position.get('entry_price', 0)))
            if not entry_price:
                return Decimal('0')
            
            symbol = position.get('symbol', '')
            is_long = position.get('side', 'long') == 'long'
            
            # Get ATR value
            atr_value = await self._get_atr_value(symbol)
            
            if atr_value > 0:
                # Calculate volatility-adjusted stop distance
                stop_distance = atr_value * self.atr_multiplier
                stop_percentage = stop_distance / entry_price
                
                # Apply bounds
                stop_percentage = max(self.min_stop_percentage, min(self.max_stop_percentage, stop_percentage))
            else:
                # Fallback to default percentage if ATR unavailable
                stop_percentage = self.min_stop_percentage
            
            # Calculate absolute stop price
            if is_long:
                return entry_price * (Decimal('1') - stop_percentage)
            else:
                return entry_price * (Decimal('1') + stop_percentage)
                
        except Exception as e:
            logger.error(f"Error calculating volatility stop: {e}")
            # Fallback to percentage stop
            return await super()._calculate_initial_stop(position)
    
    async def _get_atr_value(self, symbol: str) -> Decimal:
        """Get ATR value for symbol"""
        try:
            if not symbol or not self.context:
                return Decimal('0')
            
            # Get historical data for ATR calculation
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
        atr_period: int = 14,
        atr_multiplier: Decimal = Decimal('2.0'),
        min_stop_percentage: Decimal = Decimal('0.02'),
        max_stop_percentage: Decimal = Decimal('0.10'),
        **kwargs
    ) -> ExitConfiguration:
        """Create configuration for volatility stop loss strategy"""
        parameters = {
            'stop_mode': 'volatility',
            'atr_period': atr_period,
            'atr_multiplier': atr_multiplier,
            'min_stop_percentage': min_stop_percentage,
            'max_stop_percentage': max_stop_percentage,
            'volatility_adjustment': True,
            **kwargs
        }
        
        return ExitConfiguration(
            strategy_id=strategy_id,
            strategy_type=ExitType.STOP_LOSS,
            name=f"Volatility Stop Loss ({symbol})",
            description=f"ATR-based volatility stop loss for {symbol}",
            parameters=parameters,
            symbols=[symbol]
        )


# Factory functions

def create_stop_loss_strategy(
    strategy_id: str,
    symbol: str,
    stop_percentage: Optional[Decimal] = None,
    stop_price: Optional[Decimal] = None,
    stop_mode: str = "percentage",
    **kwargs
) -> StopLossStrategy:
    """
    Create a stop loss exit strategy
    
    @param strategy_id Unique identifier for the strategy
    @param symbol Trading symbol to monitor
    @param stop_percentage Percentage-based stop (e.g., 0.03 for 3%)
    @param stop_price Fixed price for stop loss
    @param stop_mode Type of stop ("percentage", "volatility", "fixed")
    @param kwargs Additional configuration parameters
    
    @returns Configured stop loss strategy instance
    """
    parameters = {
        'stop_percentage': stop_percentage,
        'stop_price': stop_price,
        'stop_mode': stop_mode,
        **kwargs
    }
    
    config = ExitConfiguration(
        strategy_id=strategy_id,
        strategy_type=ExitType.STOP_LOSS,
        name=f"Stop Loss ({symbol})",
        description=f"Stop loss strategy for {symbol}",
        parameters=parameters,
        symbols=[symbol]
    )
    
    # Choose appropriate strategy class
    if stop_mode.lower() == 'volatility':
        return VolatilityStopLoss(config)
    else:
        return StopLossStrategy(config)


def create_percentage_stop_loss(
    strategy_id: str,
    symbol: str,
    stop_percentage: Decimal = Decimal('0.03'),
    activation_delay: int = 0,
    **kwargs
) -> PercentageStopLoss:
    """Create percentage-based stop loss strategy"""
    config = PercentageStopLoss.create_config(
        strategy_id=strategy_id,
        symbol=symbol,
        stop_percentage=stop_percentage,
        activation_delay=activation_delay,
        **kwargs
    )
    return PercentageStopLoss(config)


def create_volatility_stop_loss(
    strategy_id: str,
    symbol: str,
    atr_period: int = 14,
    atr_multiplier: Decimal = Decimal('2.0'),
    **kwargs
) -> VolatilityStopLoss:
    """Create volatility-adjusted stop loss strategy"""
    config = VolatilityStopLoss.create_config(
        strategy_id=strategy_id,
        symbol=symbol,
        atr_period=atr_period,
        atr_multiplier=atr_multiplier,
        **kwargs
    )
    return VolatilityStopLoss(config)
