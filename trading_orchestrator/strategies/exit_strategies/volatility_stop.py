"""
@file volatility_stop.py
@brief Volatility Stop Exit Strategy Implementation

@details
This module implements volatility-based exit strategies that adjust exit
parameters based on market volatility measures. Volatility stops help
adapt to changing market conditions and provide better risk management
in volatile environments.

Key Features:
- ATR-based volatility calculations
- Dynamic stop adjustment based on volatility
- Volatility breakout detection
- Multiple volatility timeframe support
- Integration with other exit strategies

@author Trading Orchestrator System
@version 1.0
@date 2025-11-06

@warning
Volatility can change rapidly, especially during market stress periods.
Always monitor volatility indicators and adjust parameters accordingly.

@note
Volatility stops work best when combined with position sizing based on volatility.

@see base_exit_strategy.py for base framework
@see stop_loss.py for basic stop loss strategies
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
import asyncio
import numpy as np

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
class VolatilityStopConfig:
    """
    @class VolatilityStopConfig
    @brief Configuration for volatility stop strategies
    
    @details
    Contains all configuration parameters for volatility-based exit
    strategies including ATR calculations, volatility thresholds,
    and adjustment mechanisms.
    """
    atr_period: int = 14
    atr_multiplier: Decimal = Decimal('2.0')
    volatility_threshold: Decimal = Decimal('0.03')  # 3% daily volatility
    min_atr_multiplier: Decimal = Decimal('1.5')
    max_atr_multiplier: Decimal = Decimal('4.0')
    lookback_periods: List[int] = field(default_factory=lambda: [14, 21, 50])
    volatility_breakout_threshold: Decimal = Decimal('2.0')  # 2x normal volatility
    max_hold_periods: int = 10  # Maximum periods to hold if volatility too high


class VolatilityStopStrategy(BaseExitStrategy):
    """
    @class VolatilityStopStrategy
    @brief Base volatility-based exit strategy
    
    @details
    Provides volatility-based exit logic using various volatility measures
    including ATR, realized volatility, and volatility percentiles.
    """
    
    def __init__(self, config: ExitConfiguration):
        super().__init__(config)
        self.vol_config = VolatilityStopConfig(**config.parameters)
        
        # Volatility tracking
        self.volatility_history: Dict[str, List[Decimal]] = {}
        self.atr_history: Dict[str, List[Decimal]] = {}
        self.volatility_regimes: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Volatility stop strategy initialized: {config.name}")
    
    async def evaluate_exit_conditions(self, position: Dict[str, Any]) -> bool:
        """Evaluate volatility-based exit conditions"""
        try:
            position_id = position.get('position_id')
            if not position_id:
                return False
            
            symbol = position.get('symbol')
            if not symbol:
                return False
            
            # Get current volatility metrics
            volatility_metrics = await self._calculate_volatility_metrics(symbol)
            
            # Check for volatility breakout
            if await self._is_volatility_breakout(symbol, volatility_metrics):
                logger.info(f"Volatility breakout detected for {symbol}")
                return True
            
            # Check for excessive volatility hold time
            if await self._should_exit_due_to_volatility_duration(position):
                logger.info(f"Exiting due to high volatility duration for {position_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating volatility conditions: {e}")
            return False
    
    async def generate_exit_signal(
        self, 
        position: Dict[str, Any], 
        exit_reason: ExitReason
    ) -> Optional[ExitSignal]:
        """Generate volatility-based exit signal"""
        try:
            position_id = position.get('position_id')
            symbol = position.get('symbol')
            quantity = Decimal(str(position.get('quantity', 0)))
            
            if not all([position_id, symbol, quantity]):
                return None
            
            current_price = await self.context.get_current_price(symbol)
            if not current_price:
                return None
            
            # Calculate confidence and urgency
            confidence = await self._calculate_volatility_confidence(position)
            urgency = 0.85  # High urgency for volatility exits
            
            exit_signal = ExitSignal(
                signal_id=f"vol_{position_id}_{datetime.utcnow().timestamp()}",
                strategy_id=self.config.strategy_id,
                position_id=position_id,
                symbol=symbol,
                exit_reason=ExitReason.VOLATILITY_STOP,
                exit_price=current_price,
                exit_quantity=quantity,
                confidence=confidence,
                urgency=urgency,
                metadata={
                    'exit_type': 'volatility_breakout',
                    'volatility_metrics': await self._get_volatility_summary(symbol)
                }
            )
            
            return exit_signal
            
        except Exception as e:
            logger.error(f"Error generating volatility exit signal: {e}")
            return None
    
    async def _calculate_volatility_metrics(self, symbol: str) -> Dict[str, Any]:
        """Calculate comprehensive volatility metrics"""
        try:
            if not self.context:
                return {}
            
            metrics = {}
            
            # Calculate ATR for multiple periods
            for period in self.vol_config.lookback_periods:
                atr_value = await self._calculate_atr(symbol, period)
                metrics[f'atr_{period}'] = atr_value
                
                # Store in history
                if symbol not in self.atr_history:
                    self.atr_history[symbol] = []
                self.atr_history[symbol].append(atr_value)
                
                # Keep only recent values
                if len(self.atr_history[symbol]) > period * 2:
                    self.atr_history[symbol] = self.atr_history[symbol][-period*2:]
            
            # Calculate realized volatility
            realized_vol = await self._calculate_realized_volatility(symbol)
            metrics['realized_volatility'] = realized_vol
            
            # Store in history
            if symbol not in self.volatility_history:
                self.volatility_history[symbol] = []
            self.volatility_history[symbol].append(realized_vol)
            
            # Keep only recent values
            if len(self.volatility_history[symbol]) > 100:
                self.volatility_history[symbol] = self.volatility_history[symbol][-100:]
            
            # Calculate volatility percentiles
            vol_percentiles = self._calculate_volatility_percentiles(symbol)
            metrics.update(vol_percentiles)
            
            # Determine volatility regime
            regime = await self._determine_volatility_regime(symbol, metrics)
            metrics['regime'] = regime
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating volatility metrics: {e}")
            return {}
    
    async def _calculate_atr(self, symbol: str, period: int) -> Decimal:
        """Calculate ATR for given period"""
        try:
            data = await self.context.get_historical_data(symbol, '1h', period + 1)
            if len(data) < period + 1:
                return Decimal('0')
            
            true_ranges = []
            for i in range(1, len(data)):
                high = Decimal(str(data[i].get('high', 0)))
                low = Decimal(str(data[i].get('low', 0)))
                prev_close = Decimal(str(data[i-1].get('close', 0)))
                
                tr1 = high - low
                tr2 = abs(high - prev_close)
                tr3 = abs(low - prev_close)
                
                true_range = max(tr1, tr2, tr3)
                true_ranges.append(true_range)
            
            atr = sum(true_ranges) / len(true_ranges)
            return atr
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return Decimal('0')
    
    async def _calculate_realized_volatility(self, symbol: str) -> Decimal:
        """Calculate realized volatility"""
        try:
            data = await self.context.get_historical_data(symbol, '1d', 30)
            if len(data) < 2:
                return Decimal('0')
            
            returns = []
            for i in range(1, len(data)):
                prev_close = Decimal(str(data[i-1].get('close', 0)))
                current_close = Decimal(str(data[i].get('close', 0)))
                
                if prev_close > 0:
                    daily_return = (current_close - prev_close) / prev_close
                    returns.append(float(daily_return))
            
            if len(returns) < 2:
                return Decimal('0')
            
            # Calculate annualized volatility
            variance = np.var(returns)
            volatility = np.sqrt(variance * 252)  # Annualized
            
            return Decimal(str(volatility))
            
        except Exception as e:
            logger.error(f"Error calculating realized volatility: {e}")
            return Decimal('0')
    
    def _calculate_volatility_percentiles(self, symbol: str) -> Dict[str, Decimal]:
        """Calculate volatility percentiles"""
        try:
            if symbol not in self.volatility_history or not self.volatility_history[symbol]:
                return {}
            
            vol_history = self.volatility_history[symbol]
            vol_array = np.array([float(v) for v in vol_history])
            
            percentiles = {}
            for p in [25, 50, 75, 90, 95]:
                percentiles[f'vol_percentile_{p}'] = Decimal(str(np.percentile(vol_array, p)))
            
            return percentiles
            
        except Exception as e:
            logger.error(f"Error calculating volatility percentiles: {e}")
            return {}
    
    async def _determine_volatility_regime(self, symbol: str, metrics: Dict[str, Any]) -> str:
        """Determine current volatility regime"""
        try:
            if 'vol_percentile_75' in metrics and 'realized_volatility' in metrics:
                current_vol = metrics['realized_volatility']
                percentile_75 = metrics['vol_percentile_75']
                
                if current_vol > percentile_75 * Decimal('2.0'):
                    return 'extreme_high'
                elif current_vol > percentile_75 * Decimal('1.5'):
                    return 'high'
                elif current_vol < percentile_75 * Decimal('0.5'):
                    return 'low'
                else:
                    return 'normal'
            
            return 'unknown'
            
        except Exception as e:
            logger.error(f"Error determining volatility regime: {e}")
            return 'unknown'
    
    async def _is_volatility_breakout(self, symbol: str, metrics: Dict[str, Any]) -> bool:
        """Check if volatility breakout condition is met"""
        try:
            # Check if current volatility exceeds threshold
            if 'realized_volatility' in metrics:
                current_vol = metrics['realized_volatility']
                threshold = self.vol_config.volatility_breakout_threshold * self.vol_config.volatility_threshold
                
                if current_vol > threshold:
                    return True
            
            # Check ATR-based breakout
            for period in self.vol_config.lookback_periods:
                atr_key = f'atr_{period}'
                if atr_key in metrics:
                    atr_current = metrics[atr_key]
                    
                    # Compare to historical average
                    if symbol in self.atr_history and len(self.atr_history[symbol]) >= period:
                        historical_atr = sum(self.atr_history[symbol][-period:]) / period
                        
                        if atr_current > historical_atr * self.vol_config.volatility_breakout_threshold:
                            return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking volatility breakout: {e}")
            return False
    
    async def _should_exit_due_to_volatility_duration(self, position: Dict[str, Any]) -> bool:
        """Check if position should exit due to sustained high volatility"""
        try:
            position_id = position.get('position_id')
            symbol = position.get('symbol')
            
            if position_id not in self.volatility_regimes:
                return False
            
            regime_history = self.volatility_regimes[position_id].get('history', [])
            
            # Count consecutive high volatility periods
            high_vol_count = 0
            for regime in reversed(regime_history[-10:]):  # Check last 10 periods
                if regime in ['high', 'extreme_high']:
                    high_vol_count += 1
                else:
                    break
            
            return high_vol_count >= self.vol_config.max_hold_periods
            
        except Exception as e:
            logger.error(f"Error checking volatility duration: {e}")
            return False
    
    async def _calculate_volatility_confidence(self, position: Dict[str, Any]) -> float:
        """Calculate confidence for volatility exit"""
        try:
            # Base confidence for volatility exits
            confidence = 0.80
            
            # Adjust based on volatility regime
            symbol = position.get('symbol')
            metrics = await self._calculate_volatility_metrics(symbol)
            
            if metrics.get('regime') == 'extreme_high':
                confidence += 0.15
            elif metrics.get('regime') == 'high':
                confidence += 0.10
            
            return min(0.95, confidence)
            
        except Exception as e:
            logger.error(f"Error calculating volatility confidence: {e}")
            return 0.80
    
    async def _get_volatility_summary(self, symbol: str) -> Dict[str, Any]:
        """Get volatility summary for signal metadata"""
        try:
            metrics = await self._calculate_volatility_metrics(symbol)
            
            return {
                'current_regime': metrics.get('regime', 'unknown'),
                'realized_volatility': float(metrics.get('realized_volatility', 0)),
                'atr_14': float(metrics.get('atr_14', 0)),
                'vol_percentile_75': float(metrics.get('vol_percentile_75', 0))
            }
            
        except Exception as e:
            logger.error(f"Error getting volatility summary: {e}")
            return {}


class ATRVolatilityStop(VolatilityStopStrategy):
    """ATR-based volatility stop strategy"""
    
    def __init__(self, config: ExitConfiguration):
        super().__init__(config)
        self.atr_config = config.parameters
        
        logger.info(f"ATR Volatility stop initialized")
    
    async def evaluate_exit_conditions(self, position: Dict[str, Any]) -> bool:
        """Evaluate ATR-based exit conditions"""
        try:
            symbol = position.get('symbol')
            if not symbol:
                return False
            
            # Get current ATR
            atr_value = await self._calculate_atr(symbol, self.vol_config.atr_period)
            
            # Calculate dynamic stop distance
            current_price = await self.context.get_current_price(symbol)
            if not current_price or atr_value == 0:
                return False
            
            # Calculate stop level
            is_long = position.get('side', 'long') == 'long'
            stop_distance = atr_value * self.vol_config.atr_multiplier
            
            if is_long:
                stop_price = current_price - stop_distance
            else:
                stop_price = current_price + stop_distance
            
            # Check if stop is triggered
            entry_price = Decimal(str(position.get('entry_price', 0)))
            if is_long and current_price <= stop_price:
                return True
            elif not is_long and current_price >= stop_price:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating ATR volatility conditions: {e}")
            return False
    
    @classmethod
    def create_config(
        cls,
        strategy_id: str,
        symbol: str,
        atr_period: int = 14,
        atr_multiplier: Decimal = Decimal('2.0'),
        **kwargs
    ) -> ExitConfiguration:
        """Create ATR volatility stop configuration"""
        parameters = {
            'atr_period': atr_period,
            'atr_multiplier': atr_multiplier,
            'volatility_threshold': Decimal('0.03'),
            **kwargs
        }
        
        return ExitConfiguration(
            strategy_id=strategy_id,
            strategy_type=ExitType.VOLATILITY_STOP,
            name=f"ATR Volatility Stop ({symbol})",
            description=f"ATR-based volatility stop for {symbol}",
            parameters=parameters,
            symbols=[symbol]
        )


# Factory functions

def create_volatility_stop_strategy(
    strategy_id: str,
    symbol: str,
    volatility_threshold: Decimal = Decimal('0.03'),
    atr_period: int = 14,
    **kwargs
) -> VolatilityStopStrategy:
    """Create volatility stop strategy"""
    config = ExitConfiguration(
        strategy_id=strategy_id,
        strategy_type=ExitType.VOLATILITY_STOP,
        name=f"Volatility Stop ({symbol})",
        description=f"Volatility-based stop for {symbol}",
        parameters={
            'volatility_threshold': volatility_threshold,
            'atr_period': atr_period,
            **kwargs
        },
        symbols=[symbol]
    )
    
    return VolatilityStopStrategy(config)


def create_atr_volatility_stop(
    strategy_id: str,
    symbol: str,
    atr_period: int = 14,
    atr_multiplier: Decimal = Decimal('2.0'),
    **kwargs
) -> ATRVolatilityStop:
    """Create ATR volatility stop strategy"""
    config = ATRVolatilityStop.create_config(
        strategy_id=strategy_id,
        symbol=symbol,
        atr_period=atr_period,
        atr_multiplier=atr_multiplier,
        **kwargs
    )
    return ATRVolatilityStop(config)
