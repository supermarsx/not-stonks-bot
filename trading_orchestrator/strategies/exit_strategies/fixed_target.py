"""
@file fixed_target.py
@brief Fixed Target Exit Strategy Implementation

@details
This module implements fixed target exit strategies that close positions when
predetermined profit or loss levels are reached. Fixed target strategies provide
clear, predetermined exit points that are easy to understand and manage.

Key Features:
- Configurable profit and loss targets
- Partial exit capabilities at different levels
- Time-based target adjustment
- Multiple target levels support
- Integration with risk management and position sizing

@author Trading Orchestrator System
@version 1.0
@date 2025-11-06

@warning
Fixed targets don't adapt to changing market conditions. Consider market
volatility and trends when setting target levels to avoid premature exits
or missed opportunities.

@note
Fixed targets are best used in conjunction with other exit strategies
for comprehensive position management.

@see base_exit_strategy.py for base framework
@see trailing_stop.py for dynamic trailing strategies
"""

from typing import Dict, Any, List, Optional, Tuple
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
class FixedTargetConfig:
    """
    @class FixedTargetConfig
    @brief Configuration for fixed target exit strategies
    
    @details
    Contains all configuration parameters for fixed target exit strategies
    including profit and loss targets, partial exit settings, and timing
    parameters.
    
    @par Target Parameters:
    - profit_target: Target profit percentage to trigger exit
    - loss_target: Maximum loss percentage to trigger exit
    - partial_exits: Whether to use partial exits at multiple levels
    - partial_sizes: Sizes for partial exits as percentages of total position
    
    @par Target Levels:
    - profit_levels: List of profit percentages for multiple profit targets
    - loss_levels: List of loss percentages for multiple loss targets
    - level_priorities: Priority order for level evaluation
    
    @par Timing Parameters:
    - target_timeout: Maximum time to wait for target achievement
    - early_exit_enabled: Whether to allow early exits before full target
    - time_decay_factor: How targets decay over time
    
    @par Example:
    @code
    config = FixedTargetConfig(
        profit_target=Decimal('0.10'),  # 10% profit target
        loss_target=Decimal('0.05'),    # 5% loss target
        partial_exits=True,
        profit_levels=[Decimal('0.03'), Decimal('0.07'), Decimal('0.10')],
        partial_sizes=[Decimal('0.25'), Decimal('0.25'), Decimal('0.50')],
        target_timeout=timedelta(days=30)
    )
    @endcode
    
    @note
    Targets should be set based on historical volatility, market conditions,
    and risk tolerance. Wider targets reduce the frequency of exits.
    """
    profit_target: Decimal  # Target profit percentage
    loss_target: Decimal    # Maximum loss percentage
    partial_exits: bool = False
    partial_sizes: List[Decimal] = field(default_factory=list)  # Must sum to 1.0
    profit_levels: List[Decimal] = field(default_factory=list)
    loss_levels: List[Decimal] = field(default_factory=list)
    level_priorities: List[int] = field(default_factory=list)
    target_timeout: Optional[timedelta] = None
    early_exit_enabled: bool = True
    time_decay_factor: Decimal = Decimal('1.0')  # 1.0 = no decay
    activation_delay: int = 0  # seconds
    min_profit_hold_time: int = 0  # seconds
    max_hold_time: Optional[timedelta] = None


class FixedTargetStrategy(BaseExitStrategy):
    """
    @class FixedTargetStrategy
    @brief Base class for fixed target exit strategies
    
    @details
    Provides common functionality for all fixed target exit strategies
    including target calculation, level monitoring, and partial exit
    management. Handles both simple single-target strategies and
    complex multi-level target strategies.
    
    @par Key Features:
    - Single profit/loss target monitoring
    - Multiple target level support
    - Partial exit capabilities
    - Time-based target management
    - Target timeout and decay mechanisms
    
    @par Target Evaluation Logic:
    1. Check if position has reached any target level
    2. Determine appropriate exit quantity based on level
    3. Calculate exit confidence and urgency
    4. Generate exit signal with target-specific parameters
    
    @warning
    Fixed targets may be reached by temporary price movements (false breakouts).
    Consider adding confirmation mechanisms for critical targets.
    
    @note
    This class provides the common framework. Specific implementations should
    override target calculation methods for their specific logic.
    
    @see ExitReason.PROFIT_TARGET for profit-based exits
    @see ExitReason.STOP_LOSS for loss-based exits
    """
    
    def __init__(self, config: ExitConfiguration):
        super().__init__(config)
        
        # Fixed target specific configuration
        self.target_config = FixedTargetConfig(**config.parameters)
        
        # Target tracking
        self.position_targets: Dict[str, Dict[str, Any]] = {}
        self.target_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Performance tracking
        self.target_achievements: Dict[str, List[Dict[str, Any]]] = {}
        self.partial_exits: Dict[str, List[Dict[str, Any]]] = {}
        
        logger.info(f"Fixed target strategy initialized: {config.name}")
    
    async def evaluate_exit_conditions(self, position: Dict[str, Any]) -> bool:
        """
        Evaluate fixed target exit conditions
        
        @param position Position information dictionary
        
        @details
        Checks if the current position has reached any of the configured
        profit or loss targets. Handles both simple single-target and
        complex multi-level target strategies.
        
        @returns True if exit should be triggered, False otherwise
        """
        try:
            position_id = position.get('position_id')
            if not position_id:
                return False
            
            # Check if position is eligible for target evaluation
            if not await self._is_position_eligible(position):
                return False
            
            # Get or initialize target tracking
            target_data = await self._get_or_initialize_targets(position)
            
            # Update target levels if using time decay
            await self._update_target_levels(position, target_data)
            
            # Check if any target has been reached
            target_reached = await self._check_target_reached(position, target_data)
            
            if target_reached:
                logger.info(f"Fixed target reached for position {position_id}")
            
            return target_reached
            
        except Exception as e:
            logger.error(f"Error evaluating fixed target conditions: {e}")
            return False
    
    async def generate_exit_signal(
        self, 
        position: Dict[str, Any], 
        exit_reason: ExitReason
    ) -> Optional[ExitSignal]:
        """
        Generate fixed target exit signal
        
        @param position Position information dictionary
        @param exit_reason Reason for exit
        
        @details
        Creates an exit signal with fixed target-specific parameters including
        target level reached, exit quantity based on partial exit configuration,
        and confidence based on target achievement strength.
        
        @returns ExitSignal if exit should be triggered, None otherwise
        """
        try:
            position_id = position.get('position_id')
            symbol = position.get('symbol')
            quantity = Decimal(str(position.get('quantity', 0)))
            
            if not all([position_id, symbol, quantity]):
                return None
            
            # Get current target data
            target_data = self.position_targets.get(position_id)
            if not target_data:
                return None
            
            # Calculate exit parameters
            current_price = await self.context.get_current_price(symbol)
            entry_price = Decimal(str(position.get('entry_price', 0)))
            
            if not current_price or not entry_price:
                return None
            
            # Determine exit quantity
            exit_quantity = await self._calculate_exit_quantity(position, target_data, exit_reason)
            
            if exit_quantity <= 0:
                return None
            
            # Calculate target-specific parameters
            confidence = await self._calculate_target_confidence(position, target_data, exit_reason)
            urgency = await self._calculate_target_urgency(position, target_data, exit_reason)
            market_impact = await self._estimate_market_impact(symbol, exit_quantity)
            
            # Determine target level details
            target_level = await self._get_reached_target_level(position, target_data, exit_reason)
            
            # Create exit signal
            exit_signal = ExitSignal(
                signal_id=f"target_{position_id}_{datetime.utcnow().timestamp()}",
                strategy_id=self.config.strategy_id,
                position_id=position_id,
                symbol=symbol,
                exit_reason=exit_reason,
                exit_price=current_price,
                exit_quantity=exit_quantity,
                confidence=confidence,
                urgency=urgency,
                estimated_execution_time=timedelta(seconds=30),
                market_impact=market_impact,
                metadata={
                    'target_level': float(target_level) if target_level else None,
                    'entry_price': float(entry_price),
                    'current_price': float(current_price),
                    'target_type': 'profit' if exit_reason == ExitReason.PROFIT_TARGET else 'loss',
                    'is_partial_exit': exit_quantity < quantity,
                    'remaining_quantity': float(quantity - exit_quantity),
                    'target_config': {
                        'profit_target': float(self.target_config.profit_target),
                        'loss_target': float(self.target_config.loss_target)
                    }
                }
            )
            
            logger.info(f"Generated fixed target exit signal: {exit_signal.signal_id}")
            return exit_signal
            
        except Exception as e:
            logger.error(f"Error generating fixed target exit signal: {e}")
            return None
    
    async def _is_position_eligible(self, position: Dict[str, Any]) -> bool:
        """Check if position is eligible for target evaluation"""
        try:
            # Check if position has required data
            entry_price = position.get('entry_price')
            if not entry_price:
                return False
            
            # Check activation delay
            if self.target_config.activation_delay > 0:
                created_at = position.get('created_at')
                if created_at:
                    age = datetime.utcnow() - created_at
                    if age.total_seconds() < self.target_config.activation_delay:
                        return False
            
            # Check minimum profit hold time
            if self.target_config.min_profit_hold_time > 0 and self._is_position_profitable(position):
                created_at = position.get('created_at')
                if created_at:
                    hold_time = (datetime.utcnow() - created_at).total_seconds()
                    if hold_time < self.target_config.min_profit_hold_time:
                        return False
            
            # Check maximum hold time
            if self.target_config.max_hold_time:
                created_at = position.get('created_at')
                if created_at:
                    hold_time = datetime.utcnow() - created_at
                    if hold_time > self.target_config.max_hold_time:
                        return True  # Trigger timeout exit
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking position eligibility: {e}")
            return False
    
    async def _get_or_initialize_targets(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """Get or initialize target tracking for position"""
        try:
            position_id = position.get('position_id')
            if not position_id:
                return {}
            
            if position_id in self.position_targets:
                return self.position_targets[position_id]
            
            # Initialize target data
            target_data = {
                'position_id': position_id,
                'profit_targets': [],
                'loss_targets': [],
                'current_targets': {},
                'achieved_targets': [],
                'partial_exits': [],
                'initialized_at': datetime.utcnow(),
                'last_update': datetime.utcnow()
            }
            
            # Set up profit targets
            if self.target_config.profit_levels:
                # Use multiple profit levels
                target_data['profit_targets'] = self.target_config.profit_levels.copy()
            elif self.target_config.profit_target > 0:
                # Use single profit target
                target_data['profit_targets'] = [self.target_config.profit_target]
            
            # Set up loss targets
            if self.target_config.loss_levels:
                # Use multiple loss levels
                target_data['loss_targets'] = self.target_config.loss_levels.copy()
            elif self.target_config.loss_target > 0:
                # Use single loss target
                target_data['loss_targets'] = [self.target_config.loss_target]
            
            # Set current target levels
            await self._initialize_current_targets(target_data, position)
            
            self.position_targets[position_id] = target_data
            logger.info(f"Initialized targets for position {position_id}")
            
            return target_data
            
        except Exception as e:
            logger.error(f"Error initializing targets for position: {e}")
            return {}
    
    async def _initialize_current_targets(self, target_data: Dict[str, Any], position: Dict[str, Any]):
        """Initialize current target levels based on configuration"""
        try:
            entry_price = Decimal(str(position.get('entry_price', 0)))
            if not entry_price:
                return
            
            symbol = position.get('symbol')
            current_price = await self.context.get_current_price(symbol)
            
            # Calculate absolute target prices
            for target_pct in target_data.get('profit_targets', []):
                target_price = entry_price * (Decimal('1') + target_pct)
                target_data['current_targets'][f"profit_{target_pct}"] = {
                    'percentage': target_pct,
                    'price': target_price,
                    'is_long': position.get('side', 'long') == 'long',
                    'priority': target_data.get('profit_targets', []).index(target_pct)
                }
            
            for target_pct in target_data.get('loss_targets', []):
                target_price = entry_price * (Decimal('1') - target_pct)
                target_data['current_targets'][f"loss_{target_pct}"] = {
                    'percentage': target_pct,
                    'price': target_price,
                    'is_long': position.get('side', 'long') == 'long',
                    'priority': target_data.get('loss_targets', []).index(target_pct)
                }
                
        except Exception as e:
            logger.error(f"Error initializing current targets: {e}")
    
    async def _update_target_levels(self, position: Dict[str, Any], target_data: Dict[str, Any]):
        """Update target levels based on time decay"""
        try:
            if self.target_config.time_decay_factor >= Decimal('1.0'):
                return  # No decay to apply
            
            entry_price = Decimal(str(position.get('entry_price', 0)))
            if not entry_price:
                return
            
            # Calculate time decay
            created_at = target_data.get('initialized_at')
            if not created_at:
                return
            
            elapsed_time = (datetime.utcnow() - created_at).total_seconds()
            decay_factor = self.target_config.time_decay_factor ** (elapsed_time / 86400)  # Daily decay
            
            # Update target levels with decay
            for target_key, target_info in target_data.get('current_targets', {}).items():
                if not target_info.get('achieved', False):
                    original_price = entry_price * (Decimal('1') + target_info['percentage'])
                    if target_info['percentage'] > 0:
                        # Profit target - decay moves target further away
                        new_price = original_price * (Decimal('1') + (target_info['percentage'] * (Decimal('1') - decay_factor)))
                    else:
                        # Loss target - decay moves target closer
                        new_price = original_price * (Decimal('1') + (target_info['percentage'] * decay_factor))
                    
                    target_info['price'] = new_price
            
            target_data['last_update'] = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error updating target levels: {e}")
    
    async def _check_target_reached(self, position: Dict[str, Any], target_data: Dict[str, Any]) -> bool:
        """Check if any target has been reached"""
        try:
            symbol = position.get('symbol')
            current_price = await self.context.get_current_price(symbol)
            is_long = position.get('side', 'long') == 'long'
            
            if not current_price:
                return False
            
            # Check each target
            for target_key, target_info in target_data.get('current_targets', {}).items():
                if target_info.get('achieved', False):
                    continue
                
                target_price = target_info.get('price', Decimal('0'))
                if not target_price:
                    continue
                
                # Determine if target is reached
                target_reached = False
                if is_long:
                    target_reached = current_price >= target_price
                else:
                    target_reached = current_price <= target_price
                
                if target_reached:
                    # Mark target as achieved
                    target_info['achieved'] = True
                    target_info['achieved_at'] = datetime.utcnow()
                    target_info['achieved_price'] = current_price
                    
                    # Record achievement
                    if position.get('position_id') not in self.target_achievements:
                        self.target_achievements[position.get('position_id')] = []
                    
                    self.target_achievements[position.get('position_id')].append({
                        'target_key': target_key,
                        'target_percentage': target_info['percentage'],
                        'achieved_at': datetime.utcnow(),
                        'achieved_price': current_price
                    })
                    
                    logger.info(f"Target {target_key} reached for position {position.get('position_id')}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking target reach: {e}")
            return False
    
    async def _calculate_exit_quantity(
        self, 
        position: Dict[str, Any], 
        target_data: Dict[str, Any], 
        exit_reason: ExitReason
    ) -> Decimal:
        """Calculate exit quantity based on partial exit configuration"""
        try:
            position_id = position.get('position_id')
            total_quantity = Decimal(str(position.get('quantity', 0)))
            
            if not self.target_config.partial_exits:
                # Full exit
                return total_quantity
            
            # Determine which target was reached
            achieved_target = None
            for target_key, target_info in target_data.get('current_targets', {}).items():
                if target_info.get('achieved', False):
                    achieved_target = target_info
                    break
            
            if not achieved_target:
                return total_quantity
            
            # Calculate exit percentage based on target level
            exit_percentage = await self._get_exit_percentage_for_target(achieved_target)
            
            # Calculate exit quantity
            exit_quantity = total_quantity * exit_percentage
            
            # Track partial exit
            if position_id not in self.partial_exits:
                self.partial_exits[position_id] = []
            
            self.partial_exits[position_id].append({
                'exit_percentage': exit_percentage,
                'exit_quantity': exit_quantity,
                'exit_time': datetime.utcnow(),
                'target_type': exit_reason.value
            })
            
            return exit_quantity
            
        except Exception as e:
            logger.error(f"Error calculating exit quantity: {e}")
            return Decimal('0')
    
    async def _get_exit_percentage_for_target(self, target_info: Dict[str, Any]) -> Decimal:
        """Get exit percentage for a specific target level"""
        try:
            target_percentage = target_info.get('percentage', Decimal('0'))
            
            # Find matching partial size for this target
            if self.target_config.partial_sizes and len(self.target_config.partial_sizes) > 0:
                # Use configured partial sizes
                if target_percentage in self.target_config.profit_levels:
                    profit_index = self.target_config.profit_levels.index(target_percentage)
                    if profit_index < len(self.target_config.partial_sizes):
                        return self.target_config.partial_sizes[profit_index]
                elif target_percentage in self.target_config.loss_levels:
                    loss_index = self.target_config.loss_levels.index(target_percentage)
                    if loss_index < len(self.target_config.partial_sizes):
                        return self.target_config.partial_sizes[loss_index]
            
            # Default: exit full position for profit targets, 100% for loss targets
            if target_percentage > 0:
                return Decimal('1.0')  # Full exit on profit target
            else:
                return Decimal('1.0')  # Full exit on loss target
                
        except Exception as e:
            logger.error(f"Error getting exit percentage: {e}")
            return Decimal('1.0')
    
    async def _get_reached_target_level(
        self, 
        position: Dict[str, Any], 
        target_data: Dict[str, Any], 
        exit_reason: ExitReason
    ) -> Optional[Decimal]:
        """Get the specific target level that was reached"""
        try:
            for target_key, target_info in target_data.get('current_targets', {}).items():
                if target_info.get('achieved', False):
                    return target_info.get('percentage')
            return None
            
        except Exception as e:
            logger.error(f"Error getting reached target level: {e}")
            return None
    
    def _is_position_profitable(self, position: Dict[str, Any]) -> bool:
        """Check if position is currently profitable"""
        try:
            entry_price = Decimal(str(position.get('entry_price', 0)))
            current_price = Decimal(str(position.get('current_price', 0)))
            is_long = position.get('side', 'long') == 'long'
            
            if not entry_price or not current_price:
                return False
            
            if is_long:
                return current_price > entry_price
            else:
                return current_price < entry_price
                
        except Exception as e:
            logger.error(f"Error checking position profitability: {e}")
            return False
    
    async def _calculate_target_confidence(
        self, 
        position: Dict[str, Any], 
        target_data: Dict[str, Any], 
        exit_reason: ExitReason
    ) -> float:
        """Calculate confidence level for target exit signal"""
        try:
            # Base confidence based on exit reason
            if exit_reason == ExitReason.PROFIT_TARGET:
                base_confidence = 0.85
            elif exit_reason == ExitReason.STOP_LOSS:
                base_confidence = 0.75
            else:
                base_confidence = 0.80
            
            # Adjust based on how much target was exceeded
            achieved_target = await self._get_reached_target_level(position, target_data, exit_reason)
            if achieved_target:
                # Higher excess margin = higher confidence
                if achieved_target > 0:  # Profit target
                    base_confidence += 0.1
                else:  # Loss target
                    base_confidence += 0.05  # Lower confidence for losses
            
            # Adjust based on time held
            created_at = target_data.get('initialized_at')
            if created_at:
                hold_time = (datetime.utcnow() - created_at).total_seconds()
                if hold_time > 3600:  # More than 1 hour
                    base_confidence += 0.05
            
            return min(0.95, max(0.6, base_confidence))
            
        except Exception as e:
            logger.error(f"Error calculating target confidence: {e}")
            return 0.8
    
    async def _calculate_target_urgency(
        self, 
        position: Dict[str, Any], 
        target_data: Dict[str, Any], 
        exit_reason: ExitReason
    ) -> float:
        """Calculate urgency level for target exit signal"""
        try:
            # Higher urgency for loss targets, moderate for profit targets
            if exit_reason == ExitReason.STOP_LOSS:
                return 0.9
            elif exit_reason == ExitReason.PROFIT_TARGET:
                return 0.7
            else:
                return 0.8
            
        except Exception as e:
            logger.error(f"Error calculating target urgency: {e}")
            return 0.7
    
    async def _estimate_market_impact(self, symbol: str, quantity: Decimal) -> Optional[Decimal]:
        """Estimate market impact of exit order"""
        try:
            # Simple market impact estimation
            if not self.context:
                return None
            
            current_price = await self.context.get_current_price(symbol)
            if not current_price:
                return None
            
            order_value = quantity * current_price
            
            # Estimate impact as percentage of price
            impact_percentage = min(0.002, quantity / Decimal('1000'))  # Max 0.2%
            
            return current_price * impact_percentage
            
        except Exception as e:
            logger.error(f"Error estimating market impact: {e}")
            return None


# Factory functions

def create_fixed_target_strategy(
    strategy_id: str,
    symbol: str,
    profit_target: Decimal = Decimal('0.10'),
    loss_target: Decimal = Decimal('0.05'),
    partial_exits: bool = False,
    partial_sizes: Optional[List[Decimal]] = None,
    **kwargs
) -> FixedTargetStrategy:
    """
    Create a fixed target exit strategy
    
    @param strategy_id Unique identifier for the strategy
    @param symbol Trading symbol to monitor
    @param profit_target Target profit percentage (e.g., 0.10 for 10%)
    @param loss_target Maximum loss percentage (e.g., 0.05 for 5%)
    @param partial_exits Whether to use partial exits
    @param partial_sizes Sizes for partial exits as percentages
    @param kwargs Additional configuration parameters
    
    @returns Configured fixed target strategy instance
    """
    parameters = {
        'profit_target': profit_target,
        'loss_target': loss_target,
        'partial_exits': partial_exits,
        'partial_sizes': partial_sizes or [],
        **kwargs
    }
    
    config = ExitConfiguration(
        strategy_id=strategy_id,
        strategy_type=ExitType.FIXED_TARGET,
        name=f"Fixed Target ({symbol})",
        description=f"Fixed target strategy for {symbol}",
        parameters=parameters,
        symbols=[symbol]
    )
    
    return FixedTargetStrategy(config)


def create_multi_level_target_strategy(
    strategy_id: str,
    symbol: str,
    profit_levels: List[Decimal],
    loss_levels: List[Decimal],
    partial_sizes: Optional[List[Decimal]] = None,
    **kwargs
) -> FixedTargetStrategy:
    """
    Create a multi-level target exit strategy
    
    @param strategy_id Unique identifier for the strategy
    @param symbol Trading symbol to monitor
    @param profit_levels List of profit percentage targets
    @param loss_levels List of loss percentage targets
    @param partial_sizes Sizes for partial exits at each level
    @param kwargs Additional configuration parameters
    
    @returns Configured multi-level target strategy instance
    """
    parameters = {
        'profit_levels': profit_levels,
        'loss_levels': loss_levels,
        'partial_exits': True,
        'partial_sizes': partial_sizes or [Decimal('1.0')],
        **kwargs
    }
    
    config = ExitConfiguration(
        strategy_id=strategy_id,
        strategy_type=ExitType.FIXED_TARGET,
        name=f"Multi-Level Target ({symbol})",
        description=f"Multi-level target strategy for {symbol}",
        parameters=parameters,
        symbols=[symbol]
    )
    
    return FixedTargetStrategy(config)
