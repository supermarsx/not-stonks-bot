"""
@file momentum.py
@brief Momentum Strategies Implementation

@details
This module implements 15+ momentum-based trading strategies that identify
and trade market trends and price momentum. Each strategy focuses on different
aspects of momentum including moving averages, oscillators, and trend strength.

Strategy Categories:
- Simple Moving Average Strategies (3): SMA crossover, SMA slope, SMA breakout
- Exponential Moving Average Strategies (3): EMA crossover, EMA ribbon, EMA slope
- MACD Strategies (2): Classic MACD, MACD divergence
- RSI Momentum Strategies (3): RSI momentum, RSI divergence, RSI overbought/oversold
- Price Momentum Strategies (2): Price momentum, rate of change (ROC)
- Volume-Weighted Strategies (2): Volume momentum, volume rate of change

Key Features:
- Signal generation with confidence levels
- Risk management and position sizing
- Backtesting and optimization support
- Real-time execution framework

@author Trading Orchestrator System
@version 2.0
@date 2025-11-06

@warning
Momentum strategies can experience significant drawdowns during market
reversals. Always use appropriate stop losses and position sizing.

@see library.StrategyLibrary for strategy management
@see base.BaseStrategy for base implementation
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass

from loguru import logger

from .base import (
    BaseStrategy,
    StrategyConfig,
    StrategyType,
    TradingSignal,
    SignalType,
    RiskLevel,
    BaseTimeSeriesStrategy,
    StrategyMetadata,
    strategy_registry
)
from .library import StrategyCategory, strategy_library
from trading.models import OrderSide
import math


# ============================================================================
# SIMPLE MOVING AVERAGE STRATEGIES
# ============================================================================

class SMACrossoverStrategy(BaseTimeSeriesStrategy):
    """Simple Moving Average Crossover Strategy
    
    Uses fast and slow simple moving averages to identify trend changes.
    Buy when fast MA crosses above slow MA, sell when it crosses below.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['fast_period', 'slow_period']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1h")
        
        self.fast_period = int(config.parameters.get('fast_period', 10))
        self.slow_period = int(config.parameters.get('slow_period', 30))
        self.min_cross_strength = float(config.parameters.get('min_cross_strength', 0.01))
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.7))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        for symbol in self.config.symbols:
            data = await self._get_historical_data(symbol)
            if len(data) < self.slow_period + 2:
                continue
            
            prices = [Decimal(str(item['close'])) for item in data]
            current_price = prices[-1]
            
            # Calculate moving averages
            fast_ma = self._calculate_sma(prices, self.fast_period)
            slow_ma = self._calculate_sma(prices, self.slow_period)
            
            if len(fast_ma) < 2 or len(slow_ma) < 2:
                continue
            
            # Check for crossover
            current_fast = fast_ma[-1]
            current_slow = slow_ma[-1]
            prev_fast = fast_ma[-2]
            prev_slow = slow_ma[-2]
            
            # Bullish crossover
            if prev_fast <= prev_slow and current_fast > current_slow:
                ma_diff = (current_fast - current_slow) / current_slow
                if ma_diff >= self.min_cross_strength:
                    strength = min(1.0, ma_diff * 10)
                    signal = await self._create_signal(
                        symbol, SignalType.BUY, strength, current_price
                    )
                    if signal:
                        signals.append(signal)
            
            # Bearish crossover
            elif prev_fast >= prev_slow and current_fast < current_slow:
                ma_diff = (current_slow - current_fast) / current_slow
                if ma_diff >= self.min_cross_strength:
                    strength = min(1.0, ma_diff * 10)
                    signal = await self._create_signal(
                        symbol, SignalType.SELL, strength, current_price
                    )
                    if signal:
                        signals.append(signal)
        
        return signals
    
    def _calculate_sma(self, prices: List[Decimal], period: int) -> List[Decimal]:
        if len(prices) < period:
            return []
        
        sma = []
        for i in range(period - 1, len(prices)):
            avg = sum(prices[i - period + 1:i + 1]) / period
            sma.append(avg)
        return sma
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, price: Decimal) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        return TradingSignal(
            signal_id=f"SMA_CROSS_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            stop_loss=price * Decimal('0.95') if signal_type == SignalType.BUY else price * Decimal('1.05'),
            take_profit=price * Decimal('1.10') if signal_type == SignalType.BUY else price * Decimal('0.90'),
            time_horizon=timedelta(hours=24),
            metadata={
                'strategy_type': 'sma_crossover',
                'fast_period': self.fast_period,
                'slow_period': self.slow_period
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


class SMASlopeStrategy(BaseTimeSeriesStrategy):
    """Simple Moving Average Slope Strategy
    
    Analyzes the slope of moving averages to identify trend strength.
    Trades in the direction of MA slope with strength-based position sizing.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['ma_period', 'min_slope']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1h")
        
        self.ma_period = int(config.parameters.get('ma_period', 20))
        self.min_slope = float(config.parameters.get('min_slope', 0.001))
        self.slope_threshold = float(config.parameters.get('slope_threshold', 0.5))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        for symbol in self.config.symbols:
            data = await self._get_historical_data(symbol)
            if len(data) < self.ma_period + 2:
                continue
            
            prices = [Decimal(str(item['close'])) for item in data]
            current_price = prices[-1]
            
            # Calculate moving average
            ma_values = self._calculate_sma(prices, self.ma_period)
            if len(ma_values) < 2:
                continue
            
            # Calculate slope
            current_ma = ma_values[-1]
            prev_ma = ma_values[-2]
            slope = (current_ma - prev_ma) / prev_ma
            
            # Generate signals based on slope
            if abs(slope) >= self.min_slope:
                if slope > 0:  # Upward slope - bullish
                    strength = min(1.0, abs(slope) * 100)
                    signal = await self._create_signal(
                        symbol, SignalType.BUY, strength, current_price
                    )
                    if signal:
                        signals.append(signal)
                else:  # Downward slope - bearish
                    strength = min(1.0, abs(slope) * 100)
                    signal = await self._create_signal(
                        symbol, SignalType.SELL, strength, current_price
                    )
                    if signal:
                        signals.append(signal)
        
        return signals
    
    def _calculate_sma(self, prices: List[Decimal], period: int) -> List[Decimal]:
        if len(prices) < period:
            return []
        
        sma = []
        for i in range(period - 1, len(prices)):
            avg = sum(prices[i - period + 1:i + 1]) / period
            sma.append(avg)
        return sma
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, price: Decimal) -> Optional[TradingSignal]:
        if strength < self.slope_threshold:
            return None
        
        return TradingSignal(
            signal_id=f"SMA_SLOPE_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            metadata={
                'strategy_type': 'sma_slope',
                'ma_period': self.ma_period
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.slope_threshold


class SMABreakoutStrategy(BaseTimeSeriesStrategy):
    """Simple Moving Average Breakout Strategy
    
    Identifies price breakouts above or below moving averages with volume confirmation.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['ma_period', 'breakout_threshold']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1h")
        
        self.ma_period = int(config.parameters.get('ma_period', 20))
        self.breakout_threshold = float(config.parameters.get('breakout_threshold', 0.02))
        self.volume_threshold = float(config.parameters.get('volume_threshold', 1.5))
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.6))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        for symbol in self.config.symbols:
            data = await self._get_historical_data(symbol)
            if len(data) < self.ma_period + 20:
                continue
            
            prices = [Decimal(str(item['close'])) for item in data]
            volumes = [float(item.get('volume', 0)) for item in data]
            current_price = prices[-1]
            
            # Calculate moving average
            ma_values = self._calculate_sma(prices, self.ma_period)
            if len(ma_values) < 2:
                continue
            
            current_ma = ma_values[-1]
            
            # Calculate breakout
            breakout_pct = (current_price - current_ma) / current_ma
            
            # Volume confirmation
            avg_volume = sum(volumes[-20:]) / 20
            current_volume = volumes[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Generate signals
            if abs(breakout_pct) >= self.breakout_threshold and volume_ratio >= self.volume_threshold:
                if breakout_pct > 0:  # Price above MA
                    strength = min(1.0, abs(breakout_pct) * 20 + volume_ratio * 0.2)
                    signal = await self._create_signal(
                        symbol, SignalType.BUY, strength, current_price
                    )
                    if signal:
                        signals.append(signal)
                else:  # Price below MA
                    strength = min(1.0, abs(breakout_pct) * 20 + volume_ratio * 0.2)
                    signal = await self._create_signal(
                        symbol, SignalType.SELL, strength, current_price
                    )
                    if signal:
                        signals.append(signal)
        
        return signals
    
    def _calculate_sma(self, prices: List[Decimal], period: int) -> List[Decimal]:
        if len(prices) < period:
            return []
        
        sma = []
        for i in range(period - 1, len(prices)):
            avg = sum(prices[i - period + 1:i + 1]) / period
            sma.append(avg)
        return sma
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, price: Decimal) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        return TradingSignal(
            signal_id=f"SMA_BREAKOUT_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            metadata={
                'strategy_type': 'sma_breakout',
                'ma_period': self.ma_period
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


# ============================================================================
# EXPONENTIAL MOVING AVERAGE STRATEGIES
# ============================================================================

class EMACrossoverStrategy(BaseTimeSeriesStrategy):
    """Exponential Moving Average Crossover Strategy
    
    Uses fast and slow EMA crossovers for more responsive trend signals.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['fast_period', 'slow_period']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1h")
        
        self.fast_period = int(config.parameters.get('fast_period', 12))
        self.slow_period = int(config.parameters.get('slow_period', 26))
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.7))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        for symbol in self.config.symbols:
            data = await self._get_historical_data(symbol)
            if len(data) < self.slow_period + 2:
                continue
            
            prices = [Decimal(str(item['close'])) for item in data]
            current_price = prices[-1]
            
            # Calculate EMAs
            fast_ema = self._calculate_ema(prices, self.fast_period)
            slow_ema = self._calculate_ema(prices, self.slow_period)
            
            if len(fast_ema) < 2 or len(slow_ema) < 2:
                continue
            
            # Check for crossover
            current_fast = fast_ema[-1]
            current_slow = slow_ema[-1]
            prev_fast = fast_ema[-2]
            prev_slow = slow_ema[-2]
            
            # Bullish crossover
            if prev_fast <= prev_slow and current_fast > current_slow:
                strength = min(1.0, (current_fast - current_slow) / current_slow * 20)
                signal = await self._create_signal(
                    symbol, SignalType.BUY, strength, current_price
                )
                if signal:
                    signals.append(signal)
            
            # Bearish crossover
            elif prev_fast >= prev_slow and current_fast < current_slow:
                strength = min(1.0, (current_slow - current_fast) / current_slow * 20)
                signal = await self._create_signal(
                    symbol, SignalType.SELL, strength, current_price
                )
                if signal:
                    signals.append(signal)
        
        return signals
    
    def _calculate_ema(self, prices: List[Decimal], period: int) -> List[Decimal]:
        if len(prices) < period:
            return []
        
        multiplier = 2.0 / (period + 1)
        ema = []
        ema.append(prices[period - 1])  # Start with SMA
        
        for i in range(period, len(prices)):
            ema_value = (prices[i] - ema[-1]) * multiplier + ema[-1]
            ema.append(ema_value)
        
        return ema
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, price: Decimal) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        return TradingSignal(
            signal_id=f"EMA_CROSS_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            metadata={
                'strategy_type': 'ema_crossover',
                'fast_period': self.fast_period,
                'slow_period': self.slow_period
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


class EMARibbonStrategy(BaseTimeSeriesStrategy):
    """Exponential Moving Average Ribbon Strategy
    
    Uses multiple EMAs of different periods to create a trend ribbon.
    Trades based on ribbon slope and price position relative to ribbon.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['ema_periods']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1h")
        
        self.ema_periods = config.parameters.get('ema_periods', [10, 20, 30, 50, 100])
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.6))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        for symbol in self.config.symbols:
            data = await self._get_historical_data(symbol)
            if len(data) < max(self.ema_periods) + 2:
                continue
            
            prices = [Decimal(str(item['close'])) for item in data]
            current_price = prices[-1]
            
            # Calculate EMA ribbon
            ema_values = {}
            for period in self.ema_periods:
                ema_values[period] = self._calculate_ema(prices, period)
            
            # Calculate ribbon slope and position
            ribbon_slope = self._calculate_ribbon_slope(ema_values)
            ribbon_position = self._calculate_ribbon_position(current_price, ema_values)
            
            # Generate signals
            if ribbon_slope is not None and ribbon_position is not None:
                if ribbon_slope > 0.001 and ribbon_position > 0.6:  # Strong uptrend
                    strength = min(1.0, ribbon_slope * 100 + ribbon_position * 0.3)
                    signal = await self._create_signal(
                        symbol, SignalType.BUY, strength, current_price
                    )
                    if signal:
                        signals.append(signal)
                elif ribbon_slope < -0.001 and ribbon_position < 0.4:  # Strong downtrend
                    strength = min(1.0, abs(ribbon_slope) * 100 + (1 - ribbon_position) * 0.3)
                    signal = await self._create_signal(
                        symbol, SignalType.SELL, strength, current_price
                    )
                    if signal:
                        signals.append(signal)
        
        return signals
    
    def _calculate_ema(self, prices: List[Decimal], period: int) -> List[Decimal]:
        if len(prices) < period:
            return []
        
        multiplier = 2.0 / (period + 1)
        ema = []
        ema.append(prices[period - 1])
        
        for i in range(period, len(prices)):
            ema_value = (prices[i] - ema[-1]) * multiplier + ema[-1]
            ema.append(ema_value)
        
        return ema
    
    def _calculate_ribbon_slope(self, ema_values: Dict[int, List[Decimal]]) -> Optional[float]:
        """Calculate overall ribbon slope"""
        if not ema_values or not all(ema_values.values()):
            return None
        
        latest_values = [ema[-1] for ema in ema_values.values() if len(ema) >= 2]
        if len(latest_values) < 2:
            return None
        
        # Simple slope calculation
        avg_current = sum(latest_values) / len(latest_values)
        prev_values = [ema[-2] for ema in ema_values.values() if len(ema) >= 2]
        avg_prev = sum(prev_values) / len(prev_values)
        
        return float((avg_current - avg_prev) / avg_prev)
    
    def _calculate_ribbon_position(self, price: Decimal, ema_values: Dict[int, List[Decimal]]) -> Optional[float]:
        """Calculate price position relative to EMA ribbon"""
        if not ema_values or not all(ema_values.values()):
            return None
        
        current_emas = [ema[-1] for ema in ema_values.values() if len(ema) >= 1]
        if not current_emas:
            return None
        
        min_ema = min(current_emas)
        max_ema = max(current_emas)
        
        if max_ema == min_ema:
            return 0.5
        
        position = float((price - min_ema) / (max_ema - min_ema))
        return position
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, price: Decimal) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        return TradingSignal(
            signal_id=f"EMA_RIBBON_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            metadata={
                'strategy_type': 'ema_ribbon',
                'ema_periods': self.ema_periods
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


class EMASlopeStrategy(BaseTimeSeriesStrategy):
    """Exponential Moving Average Slope Strategy
    
    Analyzes EMA slope for trend direction and strength.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['ema_period', 'min_slope']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1h")
        
        self.ema_period = int(config.parameters.get('ema_period', 21))
        self.min_slope = float(config.parameters.get('min_slope', 0.001))
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.6))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        for symbol in self.config.symbols:
            data = await self._get_historical_data(symbol)
            if len(data) < self.ema_period + 2:
                continue
            
            prices = [Decimal(str(item['close'])) for item in data]
            current_price = prices[-1]
            
            # Calculate EMA
            ema_values = self._calculate_ema(prices, self.ema_period)
            if len(ema_values) < 2:
                continue
            
            # Calculate slope
            current_ema = ema_values[-1]
            prev_ema = ema_values[-2]
            slope = (current_ema - prev_ema) / prev_ema
            
            # Generate signals
            if abs(slope) >= self.min_slope:
                if slope > 0:
                    strength = min(1.0, abs(slope) * 100)
                    signal = await self._create_signal(
                        symbol, SignalType.BUY, strength, current_price
                    )
                    if signal:
                        signals.append(signal)
                else:
                    strength = min(1.0, abs(slope) * 100)
                    signal = await self._create_signal(
                        symbol, SignalType.SELL, strength, current_price
                    )
                    if signal:
                        signals.append(signal)
        
        return signals
    
    def _calculate_ema(self, prices: List[Decimal], period: int) -> List[Decimal]:
        if len(prices) < period:
            return []
        
        multiplier = 2.0 / (period + 1)
        ema = []
        ema.append(prices[period - 1])
        
        for i in range(period, len(prices)):
            ema_value = (prices[i] - ema[-1]) * multiplier + ema[-1]
            ema.append(ema_value)
        
        return ema
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, price: Decimal) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        return TradingSignal(
            signal_id=f"EMA_SLOPE_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            metadata={
                'strategy_type': 'ema_slope',
                'ema_period': self.ema_period
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


# ============================================================================
# MACD STRATEGIES
# ============================================================================

class MACDClassicStrategy(BaseTimeSeriesStrategy):
    """Classic MACD (Moving Average Convergence Divergence) Strategy
    
    Uses MACD line and signal line crossovers with histogram analysis.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['fast_period', 'slow_period', 'signal_period']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1h")
        
        self.fast_period = int(config.parameters.get('fast_period', 12))
        self.slow_period = int(config.parameters.get('slow_period', 26))
        self.signal_period = int(config.parameters.get('signal_period', 9))
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.6))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        for symbol in self.config.symbols:
            data = await self._get_historical_data(symbol)
            if len(data) < self.slow_period + self.signal_period:
                continue
            
            prices = [Decimal(str(item['close'])) for item in data]
            current_price = prices[-1]
            
            # Calculate MACD
            macd_line, signal_line, histogram = self._calculate_macd(prices)
            
            if len(macd_line) < 2 or len(signal_line) < 2:
                continue
            
            # Check for crossover
            current_macd = macd_line[-1]
            current_signal = signal_line[-1]
            prev_macd = macd_line[-2]
            prev_signal = signal_line[-2]
            
            # Bullish crossover
            if prev_macd <= prev_signal and current_macd > current_signal:
                strength = min(1.0, abs(histogram[-1]) * 10)
                signal = await self._create_signal(
                    symbol, SignalType.BUY, strength, current_price
                )
                if signal:
                    signals.append(signal)
            
            # Bearish crossover
            elif prev_macd >= prev_signal and current_macd < current_signal:
                strength = min(1.0, abs(histogram[-1]) * 10)
                signal = await self._create_signal(
                    symbol, SignalType.SELL, strength, current_price
                )
                if signal:
                    signals.append(signal)
        
        return signals
    
    def _calculate_macd(self, prices: List[Decimal]) -> Tuple[List[float], List[float], List[float]]:
        """Calculate MACD components"""
        # Calculate EMAs
        fast_ema = self._calculate_ema(prices, self.fast_period)
        slow_ema = self._calculate_ema(prices, self.slow_period)
        
        # Calculate MACD line
        macd_line = []
        for i in range(min(len(fast_ema), len(slow_ema))):
            macd_value = fast_ema[i] - slow_ema[i]
            macd_line.append(float(macd_value))
        
        # Calculate signal line
        signal_line = []
        if macd_line:
            signal_line = self._calculate_simple_sma([Decimal(x) for x in macd_line], self.signal_period)
            signal_line = [float(x) for x in signal_line]
        
        # Calculate histogram
        histogram = []
        for i in range(min(len(macd_line), len(signal_line))):
            hist_value = macd_line[i] - signal_line[i]
            histogram.append(hist_value)
        
        return macd_line, signal_line, histogram
    
    def _calculate_ema(self, prices: List[Decimal], period: int) -> List[Decimal]:
        if len(prices) < period:
            return []
        
        multiplier = 2.0 / (period + 1)
        ema = []
        ema.append(prices[period - 1])
        
        for i in range(period, len(prices)):
            ema_value = (prices[i] - ema[-1]) * multiplier + ema[-1]
            ema.append(ema_value)
        
        return ema
    
    def _calculate_simple_sma(self, prices: List[Decimal], period: int) -> List[Decimal]:
        if len(prices) < period:
            return []
        
        sma = []
        for i in range(period - 1, len(prices)):
            avg = sum(prices[i - period + 1:i + 1]) / period
            sma.append(avg)
        return sma
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, price: Decimal) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        return TradingSignal(
            signal_id=f"MACD_CLASSIC_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            metadata={
                'strategy_type': 'macd_classic',
                'fast_period': self.fast_period,
                'slow_period': self.slow_period,
                'signal_period': self.signal_period
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


class MACDDivergenceStrategy(BaseTimeSeriesStrategy):
    """MACD Divergence Strategy
    
    Identifies divergences between price and MACD for reversal signals.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['fast_period', 'slow_period', 'signal_period', 'lookback_period']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1h")
        
        self.fast_period = int(config.parameters.get('fast_period', 12))
        self.slow_period = int(config.parameters.get('slow_period', 26))
        self.signal_period = int(config.parameters.get('signal_period', 9))
        self.lookback_period = int(config.parameters.get('lookback_period', 10))
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.7))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        for symbol in self.config.symbols:
            data = await self._get_historical_data(symbol)
            if len(data) < self.lookback_period + self.slow_period:
                continue
            
            prices = [Decimal(str(item['close'])) for item in data]
            current_price = prices[-1]
            
            # Calculate MACD
            macd_line, signal_line, histogram = self._calculate_macd(prices)
            
            if len(macd_line) < self.lookback_period or len(prices) < self.lookback_period:
                continue
            
            # Find price and MACD peaks/troughs
            price_peaks = self._find_peaks([float(p) for p in prices], self.lookback_period)
            price_troughs = self._find_troughs([float(p) for p in prices], self.lookback_period)
            
            macd_peaks = self._find_peaks(macd_line, self.lookback_period)
            macd_troughs = self._find_troughs(macd_line, self.lookback_period)
            
            # Check for divergences
            if price_peaks and macd_peaks:
                bullish_divergence = self._check_bearish_divergence(
                    price_peaks, macd_peaks, prices, macd_line
                )
                if bullish_divergence:
                    strength = 0.8
                    signal = await self._create_signal(
                        symbol, SignalType.BUY, strength, current_price
                    )
                    if signal:
                        signals.append(signal)
            
            if price_troughs and macd_troughs:
                bearish_divergence = self._check_bullish_divergence(
                    price_troughs, macd_troughs, prices, macd_line
                )
                if bearish_divergence:
                    strength = 0.8
                    signal = await self._create_signal(
                        symbol, SignalType.SELL, strength, current_price
                    )
                    if signal:
                        signals.append(signal)
        
        return signals
    
    def _calculate_macd(self, prices: List[Decimal]) -> Tuple[List[float], List[float], List[float]]:
        fast_ema = self._calculate_ema(prices, self.fast_period)
        slow_ema = self._calculate_ema(prices, self.slow_period)
        
        macd_line = []
        for i in range(min(len(fast_ema), len(slow_ema))):
            macd_value = fast_ema[i] - slow_ema[i]
            macd_line.append(float(macd_value))
        
        signal_line = []
        if macd_line:
            signal_line = self._calculate_simple_sma([Decimal(x) for x in macd_line], self.signal_period)
            signal_line = [float(x) for x in signal_line]
        
        histogram = []
        for i in range(min(len(macd_line), len(signal_line))):
            hist_value = macd_line[i] - signal_line[i]
            histogram.append(hist_value)
        
        return macd_line, signal_line, histogram
    
    def _calculate_ema(self, prices: List[Decimal], period: int) -> List[Decimal]:
        if len(prices) < period:
            return []
        
        multiplier = 2.0 / (period + 1)
        ema = []
        ema.append(prices[period - 1])
        
        for i in range(period, len(prices)):
            ema_value = (prices[i] - ema[-1]) * multiplier + ema[-1]
            ema.append(ema_value)
        
        return ema
    
    def _calculate_simple_sma(self, prices: List[Decimal], period: int) -> List[Decimal]:
        if len(prices) < period:
            return []
        
        sma = []
        for i in range(period - 1, len(prices)):
            avg = sum(prices[i - period + 1:i + 1]) / period
            sma.append(avg)
        return sma
    
    def _find_peaks(self, data: List[float], lookback: int) -> List[int]:
        """Find peak indices in data"""
        peaks = []
        if len(data) < lookback * 2 + 1:
            return peaks
        
        for i in range(lookback, len(data) - lookback):
            is_peak = True
            for j in range(i - lookback, i + lookback + 1):
                if j != i and data[j] >= data[i]:
                    is_peak = False
                    break
            if is_peak:
                peaks.append(i)
        
        return peaks
    
    def _find_troughs(self, data: List[float], lookback: int) -> List[int]:
        """Find trough indices in data"""
        troughs = []
        if len(data) < lookback * 2 + 1:
            return troughs
        
        for i in range(lookback, len(data) - lookback):
            is_trough = True
            for j in range(i - lookback, i + lookback + 1):
                if j != i and data[j] <= data[i]:
                    is_trough = False
                    break
            if is_trough:
                troughs.append(i)
        
        return troughs
    
    def _check_bearish_divergence(self, price_peaks: List[int], macd_peaks: List[int], 
                                 prices: List[Decimal], macd_line: List[float]) -> bool:
        """Check for bearish divergence (higher price peaks, lower MACD peaks)"""
        if len(price_peaks) < 2 or len(macd_peaks) < 2:
            return False
        
        # Check last two price peaks
        price1, price2 = float(prices[price_peaks[-2]]), float(prices[price_peaks[-1]])
        macd1, macd2 = macd_peaks[-2], macd_peaks[-1]
        macd_val1, macd_val2 = macd_line[macd1], macd_line[macd2]
        
        # Bearish divergence: higher price, lower MACD
        return price2 > price1 and macd_val2 < macd_val1
    
    def _check_bullish_divergence(self, price_troughs: List[int], macd_troughs: List[int], 
                                 prices: List[Decimal], macd_line: List[float]) -> bool:
        """Check for bullish divergence (lower price troughs, higher MACD troughs)"""
        if len(price_troughs) < 2 or len(macd_troughs) < 2:
            return False
        
        # Check last two price troughs
        price1, price2 = float(prices[price_troughs[-2]]), float(prices[price_troughs[-1]])
        macd1, macd2 = macd_troughs[-2], macd_troughs[-1]
        macd_val1, macd_val2 = macd_line[macd1], macd_line[macd2]
        
        # Bullish divergence: lower price, higher MACD
        return price2 < price1 and macd_val2 > macd_val1
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, price: Decimal) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        return TradingSignal(
            signal_id=f"MACD_DIV_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            metadata={
                'strategy_type': 'macd_divergence',
                'lookback_period': self.lookback_period
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


# ============================================================================
# RSI MOMENTUM STRATEGIES
# ============================================================================

class RSIMomentumStrategy(BaseTimeSeriesStrategy):
    """RSI Momentum Strategy
    
    Uses RSI (Relative Strength Index) momentum for trend identification.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['rsi_period', 'momentum_threshold']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1h")
        
        self.rsi_period = int(config.parameters.get('rsi_period', 14))
        self.momentum_threshold = float(config.parameters.get('momentum_threshold', 0.5))
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.6))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        for symbol in self.config.symbols:
            data = await self._get_historical_data(symbol)
            if len(data) < self.rsi_period + 2:
                continue
            
            prices = [Decimal(str(item['close'])) for item in data]
            current_price = prices[-1]
            
            # Calculate RSI
            rsi_values = self._calculate_rsi(prices, self.rsi_period)
            
            if len(rsi_values) < 2:
                continue
            
            # Calculate RSI momentum
            current_rsi = rsi_values[-1]
            prev_rsi = rsi_values[-2]
            rsi_momentum = current_rsi - prev_rsi
            
            # Generate signals based on RSI momentum
            if abs(rsi_momentum) >= self.momentum_threshold:
                if rsi_momentum > 0:  # Rising RSI - bullish
                    strength = min(1.0, rsi_momentum / 2.0)  # Normalize
                    signal = await self._create_signal(
                        symbol, SignalType.BUY, strength, current_price
                    )
                    if signal:
                        signals.append(signal)
                else:  # Falling RSI - bearish
                    strength = min(1.0, abs(rsi_momentum) / 2.0)
                    signal = await self._create_signal(
                        symbol, SignalType.SELL, strength, current_price
                    )
                    if signal:
                        signals.append(signal)
        
        return signals
    
    def _calculate_rsi(self, prices: List[Decimal], period: int) -> List[float]:
        """Calculate RSI values"""
        if len(prices) < period + 1:
            return []
        
        deltas = []
        for i in range(1, len(prices)):
            delta = float(prices[i] - prices[i-1])
            deltas.append(delta)
        
        if len(deltas) < period:
            return []
        
        # Calculate initial average gain and loss
        gains = [max(0, delta) for delta in deltas]
        losses = [max(0, -delta) for delta in deltas]
        
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        rsi_values = []
        
        # Calculate RSI for each subsequent period
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            rsi_values.append(rsi)
        
        return rsi_values
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, price: Decimal) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        return TradingSignal(
            signal_id=f"RSI_MOM_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            metadata={
                'strategy_type': 'rsi_momentum',
                'rsi_period': self.rsi_period
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


class RSIDivergenceStrategy(BaseTimeSeriesStrategy):
    """RSI Divergence Strategy
    
    Identifies divergences between price and RSI for reversal signals.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['rsi_period', 'lookback_period']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1h")
        
        self.rsi_period = int(config.parameters.get('rsi_period', 14))
        self.lookback_period = int(config.parameters.get('lookback_period', 10))
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.8))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        for symbol in self.config.symbols:
            data = await self._get_historical_data(symbol)
            if len(data) < self.lookback_period + self.rsi_period:
                continue
            
            prices = [Decimal(str(item['close'])) for item in data]
            current_price = prices[-1]
            
            # Calculate RSI
            rsi_values = self._calculate_rsi(prices, self.rsi_period)
            
            if len(rsi_values) < self.lookback_period or len(prices) < self.lookback_period:
                continue
            
            # Find price and RSI peaks/troughs
            price_peaks = self._find_peaks([float(p) for p in prices], self.lookback_period)
            price_troughs = self._find_troughs([float(p) for p in prices], self.lookback_period)
            
            rsi_peaks = self._find_peaks(rsi_values, self.lookback_period)
            rsi_troughs = self._find_troughs(rsi_values, self.lookback_period)
            
            # Check for divergences
            if price_peaks and rsi_peaks:
                bearish_divergence = self._check_bearish_divergence(
                    price_peaks, rsi_peaks, prices, rsi_values
                )
                if bearish_divergence:
                    strength = 0.8
                    signal = await self._create_signal(
                        symbol, SignalType.SELL, strength, current_price
                    )
                    if signal:
                        signals.append(signal)
            
            if price_troughs and rsi_troughs:
                bullish_divergence = self._check_bullish_divergence(
                    price_troughs, rsi_troughs, prices, rsi_values
                )
                if bullish_divergence:
                    strength = 0.8
                    signal = await self._create_signal(
                        symbol, SignalType.BUY, strength, current_price
                    )
                    if signal:
                        signals.append(signal)
        
        return signals
    
    def _calculate_rsi(self, prices: List[Decimal], period: int) -> List[float]:
        if len(prices) < period + 1:
            return []
        
        deltas = []
        for i in range(1, len(prices)):
            delta = float(prices[i] - prices[i-1])
            deltas.append(delta)
        
        if len(deltas) < period:
            return []
        
        gains = [max(0, delta) for delta in deltas]
        losses = [max(0, -delta) for delta in deltas]
        
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        rsi_values = []
        
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            rsi_values.append(rsi)
        
        return rsi_values
    
    def _find_peaks(self, data: List[float], lookback: int) -> List[int]:
        peaks = []
        if len(data) < lookback * 2 + 1:
            return peaks
        
        for i in range(lookback, len(data) - lookback):
            is_peak = True
            for j in range(i - lookback, i + lookback + 1):
                if j != i and data[j] >= data[i]:
                    is_peak = False
                    break
            if is_peak:
                peaks.append(i)
        
        return peaks
    
    def _find_troughs(self, data: List[float], lookback: int) -> List[int]:
        troughs = []
        if len(data) < lookback * 2 + 1:
            return troughs
        
        for i in range(lookback, len(data) - lookback):
            is_trough = True
            for j in range(i - lookback, i + lookback + 1):
                if j != i and data[j] <= data[i]:
                    is_trough = False
                    break
            if is_trough:
                troughs.append(i)
        
        return troughs
    
    def _check_bearish_divergence(self, price_peaks: List[int], rsi_peaks: List[int], 
                                 prices: List[Decimal], rsi_values: List[float]) -> bool:
        """Check for bearish divergence (higher price, lower RSI)"""
        if len(price_peaks) < 2 or len(rsi_peaks) < 2:
            return False
        
        price1, price2 = float(prices[price_peaks[-2]]), float(prices[price_peaks[-1]])
        rsi1, rsi2 = rsi_values[rsi_peaks[-2]], rsi_values[rsi_peaks[-1]]
        
        return price2 > price1 and rsi2 < rsi1
    
    def _check_bullish_divergence(self, price_troughs: List[int], rsi_troughs: List[int], 
                                 prices: List[Decimal], rsi_values: List[float]) -> bool:
        """Check for bullish divergence (lower price, higher RSI)"""
        if len(price_troughs) < 2 or len(rsi_troughs) < 2:
            return False
        
        price1, price2 = float(prices[price_troughs[-2]]), float(prices[price_troughs[-1]])
        rsi1, rsi2 = rsi_values[rsi_troughs[-2]], rsi_values[rsi_troughs[-1]]
        
        return price2 < price1 and rsi2 > rsi1
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, price: Decimal) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        return TradingSignal(
            signal_id=f"RSI_DIV_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            metadata={
                'strategy_type': 'rsi_divergence',
                'lookback_period': self.lookback_period
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


class RSIOversoldOverboughtStrategy(BaseTimeSeriesStrategy):
    """RSI Overbought/Oversold Strategy
    
    Trades RSI extreme levels (overbought > 70, oversold < 30).
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['rsi_period']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1h")
        
        self.rsi_period = int(config.parameters.get('rsi_period', 14))
        self.overbought_threshold = float(config.parameters.get('overbought_threshold', 70.0))
        self.oversold_threshold = float(config.parameters.get('oversold_threshold', 30.0))
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.7))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        for symbol in self.config.symbols:
            data = await self._get_historical_data(symbol)
            if len(data) < self.rsi_period + 2:
                continue
            
            prices = [Decimal(str(item['close'])) for item in data]
            current_price = prices[-1]
            
            # Calculate RSI
            rsi_values = self._calculate_rsi(prices, self.rsi_period)
            
            if len(rsi_values) < 2:
                continue
            
            current_rsi = rsi_values[-1]
            prev_rsi = rsi_values[-2] if len(rsi_values) > 1 else current_rsi
            
            # Check for overbought condition (RSI crosses above threshold)
            if prev_rsi < self.overbought_threshold and current_rsi > self.overbought_threshold:
                # Generate sell signal (reversal expected)
                excess = current_rsi - self.overbought_threshold
                strength = min(1.0, excess / 30.0)  # Normalize by max possible excess
                signal = await self._create_signal(
                    symbol, SignalType.SELL, strength, current_price
                )
                if signal:
                    signals.append(signal)
            
            # Check for oversold condition (RSI crosses below threshold)
            elif prev_rsi > self.oversold_threshold and current_rsi < self.oversold_threshold:
                # Generate buy signal (reversal expected)
                deficit = self.oversold_threshold - current_rsi
                strength = min(1.0, deficit / 30.0)
                signal = await self._create_signal(
                    symbol, SignalType.BUY, strength, current_price
                )
                if signal:
                    signals.append(signal)
        
        return signals
    
    def _calculate_rsi(self, prices: List[Decimal], period: int) -> List[float]:
        if len(prices) < period + 1:
            return []
        
        deltas = []
        for i in range(1, len(prices)):
            delta = float(prices[i] - prices[i-1])
            deltas.append(delta)
        
        if len(deltas) < period:
            return []
        
        gains = [max(0, delta) for delta in deltas]
        losses = [max(0, -delta) for delta in deltas]
        
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        rsi_values = []
        
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            rsi_values.append(rsi)
        
        return rsi_values
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, price: Decimal) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        return TradingSignal(
            signal_id=f"RSI_EXTREME_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            metadata={
                'strategy_type': 'rsi_extreme',
                'rsi_period': self.rsi_period,
                'overbought_threshold': self.overbought_threshold,
                'oversold_threshold': self.oversold_threshold
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


# ============================================================================
# PRICE MOMENTUM STRATEGIES
# ============================================================================

class PriceMomentumStrategy(BaseTimeSeriesStrategy):
    """Price Momentum Strategy
    
    Analyzes price momentum using rate of change and acceleration.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['momentum_period', 'acceleration_period']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1h")
        
        self.momentum_period = int(config.parameters.get('momentum_period', 10))
        self.acceleration_period = int(config.parameters.get('acceleration_period', 5))
        self.momentum_threshold = float(config.parameters.get('momentum_threshold', 0.02))
        self.acceleration_threshold = float(config.parameters.get('acceleration_threshold', 0.005))
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.6))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        for symbol in self.config.symbols:
            data = await self._get_historical_data(symbol)
            if len(data) < self.momentum_period + self.acceleration_period:
                continue
            
            prices = [Decimal(str(item['close'])) for item in data]
            current_price = prices[-1]
            
            # Calculate momentum
            momentum = self._calculate_momentum(prices, self.momentum_period)
            
            if momentum == 0:
                continue
            
            # Calculate acceleration
            acceleration = self._calculate_acceleration(prices, self.acceleration_period)
            
            # Generate signals based on momentum and acceleration
            if abs(momentum) >= self.momentum_threshold:
                if momentum > 0 and acceleration > self.acceleration_threshold:
                    # Strong upward momentum with acceleration
                    strength = min(1.0, (momentum * 10 + acceleration * 100))
                    signal = await self._create_signal(
                        symbol, SignalType.BUY, strength, current_price
                    )
                    if signal:
                        signals.append(signal)
                elif momentum < 0 and acceleration < -self.acceleration_threshold:
                    # Strong downward momentum with acceleration
                    strength = min(1.0, (abs(momentum) * 10 + abs(acceleration) * 100))
                    signal = await self._create_signal(
                        symbol, SignalType.SELL, strength, current_price
                    )
                    if signal:
                        signals.append(signal)
        
        return signals
    
    def _calculate_momentum(self, prices: List[Decimal], period: int) -> float:
        """Calculate price momentum as percentage change"""
        if len(prices) < period + 1:
            return 0.0
        
        current_price = float(prices[-1])
        period_ago_price = float(prices[-period - 1])
        
        if period_ago_price == 0:
            return 0.0
        
        momentum = (current_price - period_ago_price) / period_ago_price
        return momentum
    
    def _calculate_acceleration(self, prices: List[Decimal], period: int) -> float:
        """Calculate momentum acceleration"""
        if len(prices) < period * 2 + 1:
            return 0.0
        
        current_momentum = self._calculate_momentum(prices, period)
        prev_momentum = self._calculate_momentum(prices[:-1], period)
        
        acceleration = current_momentum - prev_momentum
        return acceleration
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, price: Decimal) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        return TradingSignal(
            signal_id=f"PRICE_MOM_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            metadata={
                'strategy_type': 'price_momentum',
                'momentum_period': self.momentum_period,
                'acceleration_period': self.acceleration_period
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


class ROCStrategy(BaseTimeSeriesStrategy):
    """Rate of Change (ROC) Strategy
    
    Uses Rate of Change indicator to identify price momentum.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['roc_period', 'roc_threshold']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1h")
        
        self.roc_period = int(config.parameters.get('roc_period', 12))
        self.roc_threshold = float(config.parameters.get('roc_threshold', 0.03))
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.6))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        for symbol in self.config.symbols:
            data = await self._get_historical_data(symbol)
            if len(data) < self.roc_period + 2:
                continue
            
            prices = [Decimal(str(item['close'])) for item in data]
            current_price = prices[-1]
            
            # Calculate ROC
            roc_values = self._calculate_roc(prices, self.roc_period)
            
            if len(roc_values) < 2:
                continue
            
            current_roc = roc_values[-1]
            prev_roc = roc_values[-2]
            
            # Check for ROC crossover
            if abs(current_roc) >= self.roc_threshold:
                if current_roc > 0 and prev_roc <= self.roc_threshold:
                    # Positive ROC breakout
                    strength = min(1.0, current_roc / 0.1)  # Normalize
                    signal = await self._create_signal(
                        symbol, SignalType.BUY, strength, current_price
                    )
                    if signal:
                        signals.append(signal)
                elif current_roc < 0 and prev_roc >= -self.roc_threshold:
                    # Negative ROC breakout
                    strength = min(1.0, abs(current_roc) / 0.1)
                    signal = await self._create_signal(
                        symbol, SignalType.SELL, strength, current_price
                    )
                    if signal:
                        signals.append(signal)
        
        return signals
    
    def _calculate_roc(self, prices: List[Decimal], period: int) -> List[float]:
        """Calculate Rate of Change"""
        if len(prices) < period + 1:
            return []
        
        roc_values = []
        for i in range(period, len(prices)):
            current_price = float(prices[i])
            period_ago_price = float(prices[i - period])
            
            if period_ago_price == 0:
                roc = 0.0
            else:
                roc = (current_price - period_ago_price) / period_ago_price
            
            roc_values.append(roc)
        
        return roc_values
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, price: Decimal) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        return TradingSignal(
            signal_id=f"ROC_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            metadata={
                'strategy_type': 'roc',
                'roc_period': self.roc_period
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


# ============================================================================
# VOLUME-WEIGHTED STRATEGIES
# ============================================================================

class VolumeMomentumStrategy(BaseTimeSeriesStrategy):
    """Volume Momentum Strategy
    
    Analyzes volume momentum to confirm price movements.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['volume_period', 'volume_threshold']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__( config, timeframe="1h")
        
        self.volume_period = int(config.parameters.get('volume_period', 20))
        self.volume_threshold = float(config.parameters.get('volume_threshold', 1.5))
        self.price_threshold = float(config.parameters.get('price_threshold', 0.01))
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.6))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        for symbol in self.config.symbols:
            data = await self._get_historical_data(symbol)
            if len(data) < self.volume_period + 2:
                continue
            
            prices = [Decimal(str(item['close'])) for item in data]
            volumes = [float(item.get('volume', 0)) for item in data]
            current_price = prices[-1]
            
            # Calculate price momentum
            price_change = self._calculate_price_change(prices, 1)
            
            # Calculate volume momentum
            volume_momentum = self._calculate_volume_momentum(volumes, self.volume_period)
            
            # Generate signals
            if abs(price_change) >= self.price_threshold:
                if price_change > 0 and volume_momentum >= self.volume_threshold:
                    # Price up with volume confirmation
                    strength = min(1.0, price_change * 100 + volume_momentum * 0.3)
                    signal = await self._create_signal(
                        symbol, SignalType.BUY, strength, current_price
                    )
                    if signal:
                        signals.append(signal)
                elif price_change < 0 and volume_momentum >= self.volume_threshold:
                    # Price down with volume confirmation
                    strength = min(1.0, abs(price_change) * 100 + volume_momentum * 0.3)
                    signal = await self._create_signal(
                        symbol, SignalType.SELL, strength, current_price
                    )
                    if signal:
                        signals.append(signal)
        
        return signals
    
    def _calculate_price_change(self, prices: List[Decimal], period: int) -> float:
        """Calculate price change percentage"""
        if len(prices) < period + 1:
            return 0.0
        
        current_price = float(prices[-1])
        prev_price = float(prices[-period - 1])
        
        if prev_price == 0:
            return 0.0
        
        change = (current_price - prev_price) / prev_price
        return change
    
    def _calculate_volume_momentum(self, volumes: List[float], period: int) -> float:
        """Calculate volume momentum relative to average"""
        if len(volumes) < period + 1:
            return 1.0
        
        current_volume = volumes[-1]
        avg_volume = sum(volumes[-period-1:-1]) / period
        
        if avg_volume == 0:
            return 1.0
        
        momentum = current_volume / avg_volume
        return momentum
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, price: Decimal) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        return TradingSignal(
            signal_id=f"VOL_MOM_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            metadata={
                'strategy_type': 'volume_momentum',
                'volume_period': self.volume_period
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


class VolumeROCStrategy(BaseTimeSeriesStrategy):
    """Volume Rate of Change Strategy
    
    Uses volume rate of change to identify unusual volume activity.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['volume_roc_period', 'volume_roc_threshold']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1h")
        
        self.volume_roc_period = int(config.parameters.get('volume_roc_period', 10))
        self.volume_roc_threshold = float(config.parameters.get('volume_roc_threshold', 2.0))
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.7))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        for symbol in self.config.symbols:
            data = await self._get_historical_data(symbol)
            if len(data) < self.volume_roc_period + 2:
                continue
            
            volumes = [float(item.get('volume', 0)) for item in data]
            current_price = Decimal(str(data[-1]['close']))
            
            # Calculate volume ROC
            volume_roc_values = self._calculate_volume_roc(volumes, self.volume_roc_period)
            
            if len(volume_roc_values) < 2:
                continue
            
            current_volume_roc = volume_roc_values[-1]
            
            # Generate signals based on volume ROC
            if abs(current_volume_roc) >= self.volume_roc_threshold:
                # High volume activity - could indicate significant price movement
                if current_volume_roc > 0:
                    strength = min(1.0, current_volume_roc / 5.0)
                    signal = await self._create_signal(
                        symbol, SignalType.BUY, strength, current_price
                    )
                    if signal:
                        signals.append(signal)
                else:
                    strength = min(1.0, abs(current_volume_roc) / 5.0)
                    signal = await self._create_signal(
                        symbol, SignalType.SELL, strength, current_price
                    )
                    if signal:
                        signals.append(signal)
        
        return signals
    
    def _calculate_volume_roc(self, volumes: List[float], period: int) -> List[float]:
        """Calculate Volume Rate of Change"""
        if len(volumes) < period + 1:
            return []
        
        roc_values = []
        for i in range(period, len(volumes)):
            current_volume = volumes[i]
            period_ago_volume = volumes[i - period]
            
            if period_ago_volume == 0:
                roc = 0.0
            else:
                roc = (current_volume - period_ago_volume) / period_ago_volume
            
            roc_values.append(roc)
        
        return roc_values
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, price: Decimal) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        return TradingSignal(
            signal_id=f"VOL_ROC_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            metadata={
                'strategy_type': 'volume_roc',
                'volume_roc_period': self.volume_roc_period
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


# ============================================================================
# STRATEGY REGISTRATION
# ============================================================================

def register_momentum_strategies():
    """Register all momentum strategies with the strategy library"""
    
    # Register Simple Moving Average strategies
    strategy_library.register_strategy(
        SMACrossoverStrategy,
        StrategyMetadata(
            strategy_id="sma_crossover",
            name="SMA Crossover Strategy",
            category=StrategyCategory.MOMENTUM,
            description="Simple Moving Average crossover strategy for trend identification",
            long_description="This strategy uses fast and slow simple moving averages to identify trend changes. Buy when fast MA crosses above slow MA, sell when it crosses below.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["moving-average", "crossover", "trend", "simple", "beginner"],
            parameters_schema={
                "required": ["fast_period", "slow_period"],
                "properties": {
                    "fast_period": {"type": "integer", "min": 5, "max": 50},
                    "slow_period": {"type": "integer", "min": 20, "max": 200},
                    "min_cross_strength": {"type": "float", "min": 0.001, "max": 0.1},
                    "signal_threshold": {"type": "float", "min": 0.3, "max": 1.0}
                }
            },
            example_config={
                "fast_period": 10,
                "slow_period": 30,
                "min_cross_strength": 0.01,
                "signal_threshold": 0.7
            },
            risk_warning="Can experience significant drawdowns during market reversals."
        )
    )
    
    strategy_library.register_strategy(
        SMASlopeStrategy,
        StrategyMetadata(
            strategy_id="sma_slope",
            name="SMA Slope Strategy",
            category=StrategyCategory.MOMENTUM,
            description="Simple Moving Average slope analysis for trend strength",
            long_description="Analyzes the slope of moving averages to identify trend strength and direction.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["moving-average", "slope", "trend-strength", "simple"],
            parameters_schema={
                "required": ["ma_period", "min_slope"],
                "properties": {
                    "ma_period": {"type": "integer", "min": 10, "max": 100},
                    "min_slope": {"type": "float", "min": 0.0001, "max": 0.01},
                    "slope_threshold": {"type": "float", "min": 0.3, "max": 1.0}
                }
            }
        )
    )
    
    strategy_library.register_strategy(
        SMABreakoutStrategy,
        StrategyMetadata(
            strategy_id="sma_breakout",
            name="SMA Breakout Strategy",
            category=StrategyCategory.MOMENTUM,
            description="SMA breakout with volume confirmation",
            long_description="Identifies price breakouts above or below moving averages with volume confirmation.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["moving-average", "breakout", "volume", "confirmation"],
            parameters_schema={
                "required": ["ma_period", "breakout_threshold"],
                "properties": {
                    "ma_period": {"type": "integer", "min": 10, "max": 100},
                    "breakout_threshold": {"type": "float", "min": 0.005, "max": 0.05},
                    "volume_threshold": {"type": "float", "min": 1.0, "max": 3.0},
                    "signal_threshold": {"type": "float", "min": 0.3, "max": 1.0}
                }
            }
        )
    )
    
    # Register Exponential Moving Average strategies
    strategy_library.register_strategy(
        EMACrossoverStrategy,
        StrategyMetadata(
            strategy_id="ema_crossover",
            name="EMA Crossover Strategy",
            category=StrategyCategory.MOMENTUM,
            description="Exponential Moving Average crossover for responsive signals",
            long_description="Uses fast and slow EMA crossovers for more responsive trend signals than SMA.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["moving-average", "exponential", "crossover", "responsive"],
            parameters_schema={
                "required": ["fast_period", "slow_period"],
                "properties": {
                    "fast_period": {"type": "integer", "min": 5, "max": 30},
                    "slow_period": {"type": "integer", "min": 20, "max": 100},
                    "signal_threshold": {"type": "float", "min": 0.3, "max": 1.0}
                }
            }
        )
    )
    
    strategy_library.register_strategy(
        EMARibbonStrategy,
        StrategyMetadata(
            strategy_id="ema_ribbon",
            name="EMA Ribbon Strategy",
            category=StrategyCategory.MOMENTUM,
            description="Multiple EMA ribbon for comprehensive trend analysis",
            long_description="Uses multiple EMAs of different periods to create a trend ribbon for comprehensive trend analysis.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["moving-average", "ribbon", "multi-timeframe", "comprehensive"],
            parameters_schema={
                "required": ["ema_periods"],
                "properties": {
                    "ema_periods": {"type": "array", "items": {"type": "integer"}},
                    "signal_threshold": {"type": "float", "min": 0.3, "max": 1.0}
                }
            }
        )
    )
    
    strategy_library.register_strategy(
        EMASlopeStrategy,
        StrategyMetadata(
            strategy_id="ema_slope",
            name="EMA Slope Strategy",
            category=StrategyCategory.MOMENTUM,
            description="Exponential Moving Average slope analysis",
            long_description="Analyzes EMA slope for trend direction and strength.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["moving-average", "exponential", "slope", "responsive"],
            parameters_schema={
                "required": ["ema_period", "min_slope"],
                "properties": {
                    "ema_period": {"type": "integer", "min": 5, "max": 50},
                    "min_slope": {"type": "float", "min": 0.0001, "max": 0.01},
                    "signal_threshold": {"type": "float", "min": 0.3, "max": 1.0}
                }
            }
        )
    )
    
    # Register MACD strategies
    strategy_library.register_strategy(
        MACDClassicStrategy,
        StrategyMetadata(
            strategy_id="macd_classic",
            name="Classic MACD Strategy",
            category=StrategyCategory.MOMENTUM,
            description="Classic MACD with signal line crossovers",
            long_description="Uses MACD line and signal line crossovers with histogram analysis for trend identification.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["macd", "signal-line", "histogram", "classic"],
            parameters_schema={
                "required": ["fast_period", "slow_period", "signal_period"],
                "properties": {
                    "fast_period": {"type": "integer", "min": 5, "max": 20},
                    "slow_period": {"type": "integer", "min": 20, "max": 50},
                    "signal_period": {"type": "integer", "min": 5, "max": 15},
                    "signal_threshold": {"type": "float", "min": 0.3, "max": 1.0}
                }
            }
        )
    )
    
    strategy_library.register_strategy(
        MACDDivergenceStrategy,
        StrategyMetadata(
            strategy_id="macd_divergence",
            name="MACD Divergence Strategy",
            category=StrategyCategory.MOMENTUM,
            description="MACD divergence for reversal signals",
            long_description="Identifies divergences between price and MACD for reversal signals.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["macd", "divergence", "reversal", "advanced"],
            parameters_schema={
                "required": ["fast_period", "slow_period", "signal_period", "lookback_period"],
                "properties": {
                    "fast_period": {"type": "integer", "min": 5, "max": 20},
                    "slow_period": {"type": "integer", "min": 20, "max": 50},
                    "signal_period": {"type": "integer", "min": 5, "max": 15},
                    "lookback_period": {"type": "integer", "min": 5, "max": 20},
                    "signal_threshold": {"type": "float", "min": 0.5, "max": 1.0}
                }
            }
        )
    )
    
    # Register RSI strategies
    strategy_library.register_strategy(
        RSIMomentumStrategy,
        StrategyMetadata(
            strategy_id="rsi_momentum",
            name="RSI Momentum Strategy",
            category=StrategyCategory.MOMENTUM,
            description="RSI momentum for trend identification",
            long_description="Uses RSI momentum for trend identification.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["rsi", "momentum", "trend", "oscillator"],
            parameters_schema={
                "required": ["rsi_period", "momentum_threshold"],
                "properties": {
                    "rsi_period": {"type": "integer", "min": 5, "max": 30},
                    "momentum_threshold": {"type": "float", "min": 0.1, "max": 5.0},
                    "signal_threshold": {"type": "float", "min": 0.3, "max": 1.0}
                }
            }
        )
    )
    
    strategy_library.register_strategy(
        RSIDivergenceStrategy,
        StrategyMetadata(
            strategy_id="rsi_divergence",
            name="RSI Divergence Strategy",
            category=StrategyCategory.MOMENTUM,
            description="RSI divergence for reversal signals",
            long_description="Identifies divergences between price and RSI for reversal signals.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["rsi", "divergence", "reversal", "advanced"],
            parameters_schema={
                "required": ["rsi_period", "lookback_period"],
                "properties": {
                    "rsi_period": {"type": "integer", "min": 5, "max": 30},
                    "lookback_period": {"type": "integer", "min": 5, "max": 20},
                    "signal_threshold": {"type": "float", "min": 0.5, "max": 1.0}
                }
            }
        )
    )
    
    strategy_library.register_strategy(
        RSIOversoldOverboughtStrategy,
        StrategyMetadata(
            strategy_id="rsi_extreme",
            name="RSI Overbought/Oversold Strategy",
            category=StrategyCategory.MOMENTUM,
            description="RSI extreme levels for mean reversion signals",
            long_description="Trades RSI extreme levels (overbought > 70, oversold < 30) for mean reversion.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["rsi", "extreme", "overbought", "oversold", "mean-reversion"],
            parameters_schema={
                "required": ["rsi_period"],
                "properties": {
                    "rsi_period": {"type": "integer", "min": 5, "max": 30},
                    "overbought_threshold": {"type": "float", "min": 60.0, "max": 90.0},
                    "oversold_threshold": {"type": "float", "min": 10.0, "max": 40.0},
                    "signal_threshold": {"type": "float", "min": 0.5, "max": 1.0}
                }
            }
        )
    )
    
    # Register Price Momentum strategies
    strategy_library.register_strategy(
        PriceMomentumStrategy,
        StrategyMetadata(
            strategy_id="price_momentum",
            name="Price Momentum Strategy",
            category=StrategyCategory.MOMENTUM,
            description="Price momentum with acceleration analysis",
            long_description="Analyzes price momentum using rate of change and acceleration.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["momentum", "acceleration", "rate-of-change", "advanced"],
            parameters_schema={
                "required": ["momentum_period", "acceleration_period"],
                "properties": {
                    "momentum_period": {"type": "integer", "min": 5, "max": 50},
                    "acceleration_period": {"type": "integer", "min": 2, "max": 20},
                    "momentum_threshold": {"type": "float", "min": 0.001, "max": 0.1},
                    "acceleration_threshold": {"type": "float", "min": 0.001, "max": 0.02},
                    "signal_threshold": {"type": "float", "min": 0.3, "max": 1.0}
                }
            }
        )
    )
    
    strategy_library.register_strategy(
        ROCStrategy,
        StrategyMetadata(
            strategy_id="roc",
            name="Rate of Change Strategy",
            category=StrategyCategory.MOMENTUM,
            description="ROC indicator for momentum identification",
            long_description="Uses Rate of Change indicator to identify price momentum.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["roc", "rate-of-change", "momentum", "simple"],
            parameters_schema={
                "required": ["roc_period", "roc_threshold"],
                "properties": {
                    "roc_period": {"type": "integer", "min": 5, "max": 30},
                    "roc_threshold": {"type": "float", "min": 0.01, "max": 0.1},
                    "signal_threshold": {"type": "float", "min": 0.3, "max": 1.0}
                }
            }
        )
    )
    
    # Register Volume-weighted strategies
    strategy_library.register_strategy(
        VolumeMomentumStrategy,
        StrategyMetadata(
            strategy_id="volume_momentum",
            name="Volume Momentum Strategy",
            category=StrategyCategory.MOMENTUM,
            description="Volume momentum to confirm price movements",
            long_description="Analyzes volume momentum to confirm price movements.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["volume", "momentum", "confirmation", "advanced"],
            parameters_schema={
                "required": ["volume_period", "volume_threshold"],
                "properties": {
                    "volume_period": {"type": "integer", "min": 10, "max": 50},
                    "volume_threshold": {"type": "float", "min": 1.0, "max": 5.0},
                    "price_threshold": {"type": "float", "min": 0.001, "max": 0.05},
                    "signal_threshold": {"type": "float", "min": 0.3, "max": 1.0}
                }
            }
        )
    )
    
    strategy_library.register_strategy(
        VolumeROCStrategy,
        StrategyMetadata(
            strategy_id="volume_roc",
            name="Volume Rate of Change Strategy",
            category=StrategyCategory.MOMENTUM,
            description="Volume ROC for unusual activity detection",
            long_description="Uses volume rate of change to identify unusual volume activity.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["volume", "roc", "unusual-activity", "advanced"],
            parameters_schema={
                "required": ["volume_roc_period", "volume_roc_threshold"],
                "properties": {
                    "volume_roc_period": {"type": "integer", "min": 5, "max": 30},
                    "volume_roc_threshold": {"type": "float", "min": 1.0, "max": 10.0},
                    "signal_threshold": {"type": "float", "min": 0.5, "max": 1.0}
                }
            }
        )
    )
    
    logger.info(f"Registered {len(strategy_library.categories[StrategyCategory.MOMENTUM])} momentum strategies")


# Auto-register when module is imported
register_momentum_strategies()


if __name__ == "__main__":
    async def test_momentum_strategies():
        # Test strategy registration
        momentum_strategies = strategy_library.get_strategies_by_category(StrategyCategory.MOMENTUM)
        print(f"Registered momentum strategies: {len(momentum_strategies)}")
        
        for strategy in momentum_strategies:
            print(f"- {strategy.name} ({strategy.strategy_id})")
        
        # Test strategy creation
        try:
            from .base import StrategyConfig
            
            # Create SMA crossover strategy
            config = StrategyConfig(
                strategy_id="test_sma",
                strategy_type=StrategyType.MOMENTUM,
                name="Test SMA Strategy",
                description="Test SMA crossover strategy",
                parameters={
                    "fast_period": 10,
                    "slow_period": 30,
                    "signal_threshold": 0.7
                },
                risk_level=RiskLevel.MEDIUM,
                symbols=["AAPL", "GOOGL"]
            )
            
            strategy = strategy_library.create_strategy_instance("sma_crossover", config)
            if strategy:
                print(f"\nSuccessfully created strategy: {strategy.config.name}")
            else:
                print("\nFailed to create strategy")
                
        except Exception as e:
            print(f"Error during testing: {e}")
    
    import asyncio
    asyncio.run(test_momentum_strategies())