"""
@file time_based_exit.py
@brief Time-Based Exit Strategy Implementation

@details
This module implements time-based exit strategies that close positions based
on time constraints rather than price movements. Time-based exits are useful
for managing positions during specific market periods or to limit exposure time.

Key Features:
- Maximum hold time limits
- Time-decay target adjustments
- Session-based exits
- Time-weighted exit decisions
- Integration with market hours and events

@author Trading Orchestrator System
@version 1.0
@date 2025-11-06

@note
Time-based exits are particularly useful for avoiding overnight risk, managing
day-trading positions, and ensuring positions don't remain open too long.

@see base_exit_strategy.py for base framework
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
class TimeBasedExitConfig:
    """Configuration for time-based exit strategies"""
    max_hold_time: timedelta
    min_hold_time: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    session_end_exit: bool = False  # Exit before market close
    profit_acceleration: bool = False  # Exit faster if in profit
    time_decay_factor: Decimal = Decimal('1.0')  # Time decay for targets
    market_hours_only: bool = False  # Only active during market hours
    weekend_exit: bool = True  # Exit before weekend
    earnings_exit: bool = False  # Exit before earnings announcements
    time_scaling: bool = False  # Exit probability increases with time


class TimeBasedExitStrategy(BaseExitStrategy):
    """
    @class TimeBasedExitStrategy
    @brief Time-based exit strategy
    
    @details
    Manages position exits based on time constraints rather than price movements.
    Useful for limiting exposure time, avoiding overnight risk, and managing
    day-trading positions.
    """
    
    def __init__(self, config: ExitConfiguration):
        super().__init__(config)
        self.time_config = TimeBasedExitConfig(**config.parameters)
        
        # Time tracking
        self.position_timers: Dict[str, Dict[str, Any]] = {}
        self.session_data: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Time-based exit strategy initialized: {config.name}")
    
    async def evaluate_exit_conditions(self, position: Dict[str, Any]) -> bool:
        """Evaluate time-based exit conditions"""
        try:
            position_id = position.get('position_id')
            if not position_id:
                return False
            
            # Check if position is eligible for time evaluation
            if not await self._is_position_eligible(position):
                return False
            
            # Get or initialize timer for position
            timer_data = await self._get_or_initialize_timer(position)
            
            # Check various time-based conditions
            return (
                await self._check_max_hold_time(position, timer_data) or
                await self._check_session_end(position) or
                await self._check_weekend_exit(position) or
                await self._check_market_hours_exit(position) or
                await self._check_time_scaling_exit(position, timer_data)
            )
            
        except Exception as e:
            logger.error(f"Error evaluating time-based conditions: {e}")
            return False
    
    async def generate_exit_signal(
        self, 
        position: Dict[str, Any], 
        exit_reason: ExitReason
    ) -> Optional[ExitSignal]:
        """Generate time-based exit signal"""
        try:
            position_id = position.get('position_id')
            symbol = position.get('symbol')
            quantity = Decimal(str(position.get('quantity', 0)))
            
            if not all([position_id, symbol, quantity]):
                return None
            
            current_price = await self.context.get_current_price(symbol)
            if not current_price:
                return None
            
            # Calculate time-based parameters
            confidence = await self._calculate_time_confidence(position)
            urgency = await self._calculate_time_urgency(position)
            
            # Determine exit reason based on trigger
            trigger_reason = await self._determine_time_exit_reason(position)
            
            exit_signal = ExitSignal(
                signal_id=f"time_{position_id}_{datetime.utcnow().timestamp()}",
                strategy_id=self.config.strategy_id,
                position_id=position_id,
                symbol=symbol,
                exit_reason=ExitReason.TIME_EXIT,
                exit_price=current_price,
                exit_quantity=quantity,
                confidence=confidence,
                urgency=urgency,
                metadata={
                    'exit_trigger': trigger_reason,
                    'hold_duration': self._calculate_hold_duration(position),
                    'time_remaining': self._calculate_time_remaining(position)
                }
            )
            
            return exit_signal
            
        except Exception as e:
            logger.error(f"Error generating time-based exit signal: {e}")
            return None
    
    async def _is_position_eligible(self, position: Dict[str, Any]) -> bool:
        """Check if position is eligible for time evaluation"""
        try:
            # Check minimum hold time
            created_at = position.get('created_at')
            if created_at:
                age = datetime.utcnow() - created_at
                if age < self.time_config.min_hold_time:
                    return False
            
            # Check market hours restriction
            if self.time_config.market_hours_only and not self._is_market_hours():
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking time eligibility: {e}")
            return False
    
    async def _get_or_initialize_timer(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """Get or initialize timer for position"""
        try:
            position_id = position.get('position_id')
            if not position_id:
                return {}
            
            if position_id in self.position_timers:
                return self.position_timers[position_id]
            
            # Initialize timer data
            timer_data = {
                'position_id': position_id,
                'start_time': datetime.utcnow(),
                'max_exit_time': datetime.utcnow() + self.time_config.max_hold_time,
                'exit_probability': 0.1,  # Start with low probability
                'decay_rate': self.time_config.time_decay_factor
            }
            
            self.position_timers[position_id] = timer_data
            logger.info(f"Initialized timer for position {position_id}")
            
            return timer_data
            
        except Exception as e:
            logger.error(f"Error initializing timer: {e}")
            return {}
    
    async def _check_max_hold_time(self, position: Dict[str, Any], timer_data: Dict[str, Any]) -> bool:
        """Check if maximum hold time has been exceeded"""
        try:
            current_time = datetime.utcnow()
            max_exit_time = timer_data.get('max_exit_time')
            
            if max_exit_time and current_time >= max_exit_time:
                logger.info(f"Max hold time exceeded for position {position.get('position_id')}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking max hold time: {e}")
            return False
    
    async def _check_session_end(self, position: Dict[str, Any]) -> bool:
        """Check if should exit before market close"""
        try:
            if not self.time_config.session_end_exit:
                return False
            
            # Simplified market hours check
            now = datetime.utcnow()
            current_time = now.time()
            
            # Assuming US market hours (9:30 AM - 4:00 PM EST)
            market_close = datetime.combine(now.date(), datetime.strptime("16:00", "%H:%M").time())
            
            # Exit 30 minutes before close
            exit_time = market_close - timedelta(minutes=30)
            
            if now >= exit_time and now <= market_close:
                logger.info(f"Session end exit triggered for position {position.get('position_id')}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking session end: {e}")
            return False
    
    async def _check_weekend_exit(self, position: Dict[str, Any]) -> bool:
        """Check if should exit before weekend"""
        try:
            if not self.time_config.weekend_exit:
                return False
            
            now = datetime.utcnow()
            days_ahead = 5 - now.weekday()  # Days until Friday
            
            if days_ahead <= 1:  # Thursday or later
                friday_close = now + timedelta(days=days_ahead)
                exit_time = friday_close - timedelta(hours=2)  # Exit 2 hours before Friday close
                
                if now >= exit_time:
                    logger.info(f"Weekend exit triggered for position {position.get('position_id')}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking weekend exit: {e}")
            return False
    
    async def _check_market_hours_exit(self, position: Dict[str, Any]) -> bool:
        """Check if should exit outside market hours"""
        try:
            if not self.time_config.market_hours_only:
                return False
            
            if not self._is_market_hours():
                logger.info(f"Outside market hours exit triggered for position {position.get('position_id')}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking market hours: {e}")
            return False
    
    async def _check_time_scaling_exit(self, position: Dict[str, Any], timer_data: Dict[str, Any]) -> bool:
        """Check if time-scaled exit probability triggers"""
        try:
            if not self.time_config.time_scaling:
                return False
            
            start_time = timer_data.get('start_time', datetime.utcnow())
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            total_time = self.time_config.max_hold_time.total_seconds()
            
            if total_time <= 0:
                return False
            
            # Calculate exit probability based on time elapsed
            elapsed_ratio = elapsed / total_time
            
            # Exponential decay of probability
            exit_probability = 1 - (1 - 0.1) ** (elapsed_ratio * 10)  # Exponentially increasing
            
            # Add profit acceleration if enabled
            if self.time_config.profit_acceleration and self._is_position_profitable(position):
                profit_multiplier = 1.5  # Exit faster if in profit
                exit_probability *= profit_multiplier
            
            exit_probability = min(1.0, exit_probability)
            
            # Random check based on probability
            import random
            exit_triggered = random.random() < exit_probability
            
            if exit_triggered:
                logger.info(f"Time-scaling exit triggered for position {position.get('position_id')} "
                           f"(probability: {exit_probability:.3f})")
            
            return exit_triggered
            
        except Exception as e:
            logger.error(f"Error checking time scaling: {e}")
            return False
    
    def _is_market_hours(self) -> bool:
        """Check if current time is within market hours"""
        try:
            now = datetime.utcnow()
            current_time = now.time()
            
            # Simplified check - assuming 24/7 market
            # In real implementation, would check specific market hours
            return True
            
        except Exception as e:
            logger.error(f"Error checking market hours: {e}")
            return False
    
    def _is_position_profitable(self, position: Dict[str, Any]) -> bool:
        """Check if position is currently profitable"""
        try:
            entry_price = Decimal(str(position.get('entry_price', 0)))
            current_price = Decimal(str(position.get('current_price', 0)))
            
            if not entry_price or not current_price:
                return False
            
            return current_price > entry_price
            
        except Exception as e:
            logger.error(f"Error checking position profitability: {e}")
            return False
    
    async def _calculate_time_confidence(self, position: Dict[str, Any]) -> float:
        """Calculate confidence for time-based exit"""
        try:
            # Base confidence for time-based exits
            confidence = 0.75
            
            # Higher confidence as time limit approaches
            timer_data = self.position_timers.get(position.get('position_id', ''))
            if timer_data:
                time_remaining = timer_data.get('max_exit_time', datetime.utcnow()) - datetime.utcnow()
                total_time = self.time_config.max_hold_time
                
                if total_time.total_seconds() > 0:
                    time_ratio = time_remaining.total_seconds() / total_time.total_seconds()
                    confidence += (1.0 - time_ratio) * 0.2  # More confident as time runs out
            
            return min(0.95, confidence)
            
        except Exception as e:
            logger.error(f"Error calculating time confidence: {e}")
            return 0.75
    
    async def _calculate_time_urgency(self, position: Dict[str, Any]) -> float:
        """Calculate urgency for time-based exit"""
        try:
            # Time-based exits often have high urgency
            return 0.80
            
        except Exception as e:
            logger.error(f"Error calculating time urgency: {e}")
            return 0.80
    
    async def _determine_time_exit_reason(self, position: Dict[str, Any]) -> str:
        """Determine the specific time-based exit reason"""
        try:
            # Check which condition was triggered
            # This would be enhanced to track specific triggers
            return "time_limit"
            
        except Exception as e:
            logger.error(f"Error determining time exit reason: {e}")
            return "time_based"
    
    def _calculate_hold_duration(self, position: Dict[str, Any]) -> float:
        """Calculate current hold duration in hours"""
        try:
            created_at = position.get('created_at')
            if created_at:
                duration = datetime.utcnow() - created_at
                return duration.total_seconds() / 3600
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating hold duration: {e}")
            return 0.0
    
    def _calculate_time_remaining(self, position: Dict[str, Any]) -> float:
        """Calculate time remaining until exit in hours"""
        try:
            timer_data = self.position_timers.get(position.get('position_id', ''))
            if timer_data:
                remaining = timer_data.get('max_exit_time', datetime.utcnow()) - datetime.utcnow()
                return max(0, remaining.total_seconds() / 3600)
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating time remaining: {e}")
            return 0.0


# Factory functions

def create_time_based_exit_strategy(
    strategy_id: str,
    symbol: str,
    max_hold_hours: int = 24,
    min_hold_minutes: int = 5,
    session_end_exit: bool = True,
    weekend_exit: bool = True,
    **kwargs
) -> TimeBasedExitStrategy:
    """
    Create a time-based exit strategy
    
    @param strategy_id Unique identifier for the strategy
    @param symbol Trading symbol to monitor
    @param max_hold_hours Maximum hours to hold position
    @param min_hold_minutes Minimum minutes to hold position
    @param session_end_exit Exit before market close
    @param weekend_exit Exit before weekend
    @param kwargs Additional configuration parameters
    
    @returns Configured time-based exit strategy instance
    """
    parameters = {
        'max_hold_time': timedelta(hours=max_hold_hours),
        'min_hold_time': timedelta(minutes=min_hold_minutes),
        'session_end_exit': session_end_exit,
        'weekend_exit': weekend_exit,
        'time_scaling': True,
        'profit_acceleration': True,
        **kwargs
    }
    
    config = ExitConfiguration(
        strategy_id=strategy_id,
        strategy_type=ExitType.TIME_BASED,
        name=f"Time-Based Exit ({symbol})",
        description=f"Time-based exit strategy for {symbol}",
        parameters=parameters,
        symbols=[symbol]
    )
    
    return TimeBasedExitStrategy(config)


def create_day_trading_exit_strategy(
    strategy_id: str,
    symbol: str,
    max_hold_minutes: int = 480,  # 8 hours
    **kwargs
) -> TimeBasedExitStrategy:
    """Create day trading focused time-based exit strategy"""
    return create_time_based_exit_strategy(
        strategy_id=strategy_id,
        symbol=symbol,
        max_hold_hours=max_hold_minutes // 60,
        session_end_exit=True,
        weekend_exit=True,
        market_hours_only=True,
        **kwargs
    )


def create_overnight_avoidance_strategy(
    strategy_id: str,
    symbol: str,
    **kwargs
) -> TimeBasedExitStrategy:
    """Create strategy to avoid overnight positions"""
    parameters = {
        'session_end_exit': True,
        'weekend_exit': True,
        'market_hours_only': True,
        'min_hold_time': timedelta(minutes=15),
        **kwargs
    }
    
    config = ExitConfiguration(
        strategy_id=strategy_id,
        strategy_type=ExitType.TIME_BASED,
        name=f"Overnight Avoidance ({symbol})",
        description=f"Strategy to avoid overnight positions in {symbol}",
        parameters=parameters,
        symbols=[symbol]
    )
    
    return TimeBasedExitStrategy(config)
