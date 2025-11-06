"""
@file mean_reversion.py
@brief Mean Reversion Strategies Implementation

@details
This module implements 15+ mean reversion trading strategies that identify
and trade price deviations from historical averages, bands, and statistical measures.

Strategy Categories:
- Bollinger Band Strategies (3): Classic Bollinger, Bollinger squeeze, Bollinger breakout
- RSI Mean Reversion Strategies (2): RSI overbought/oversold, RSI extreme mean reversion
- Statistical Mean Reversion (3): Z-score reversion, percentile reversion, price deviation
- Pairs Trading Strategies (3): Statistical arbitrage, correlation-based, cointegration
- Support/Resistance Strategies (3): Pivot points, Fibonacci retracement, channel trading
- Value-Based Strategies (2): Moving average reversion, fundamental value reversion
- Volume Profile Strategies (2): Volume-weighted average price (VWAP), volume profile

Key Features:
- Statistical analysis and signal generation
- Risk management and position sizing
- Backtesting and optimization support
- Real-time execution framework

@author Trading Orchestrator System
@version 2.0
@date 2025-11-06

@warning
Mean reversion strategies can fail during strong trending markets.
Always use appropriate stop losses and risk management.

@see library.StrategyLibrary for strategy management
@see base.BaseStrategy for base implementation
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass
import math
import numpy as np
from scipy import stats

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


# ============================================================================
# BOLLINGER BAND STRATEGIES
# ============================================================================

class BollingerBandsStrategy(BaseTimeSeriesStrategy):
    """Classic Bollinger Bands Mean Reversion Strategy
    
    Uses Bollinger Bands to identify overbought and oversold conditions.
    Buy when price touches lower band, sell when price touches upper band.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['period', 'std_dev']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1h")
        
        self.period = int(config.parameters.get('period', 20))
        self.std_dev = float(config.parameters.get('std_dev', 2.0))
        self.band_tolerance = float(config.parameters.get('band_tolerance', 0.01))
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.6))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        for symbol in self.config.symbols:
            data = await self._get_historical_data(symbol)
            if len(data) < self.period + 2:
                continue
            
            prices = [Decimal(str(item['close'])) for item in data]
            current_price = prices[-1]
            
            # Calculate Bollinger Bands
            upper_band, middle_band, lower_band = self._calculate_bollinger_bands(prices)
            
            if not upper_band or not lower_band:
                continue
            
            current_upper = upper_band[-1]
            current_middle = middle_band[-1]
            current_lower = lower_band[-1]
            
            # Check for band touches
            # Buy signal: price touches or goes below lower band
            if current_price <= current_lower * (1 + self.band_tolerance):
                deviation = float((current_middle - current_price) / current_middle)
                strength = min(1.0, deviation * 5)  # Normalize deviation
                signal = await self._create_signal(
                    symbol, SignalType.BUY, strength, current_price, current_middle, current_lower
                )
                if signal:
                    signals.append(signal)
            
            # Sell signal: price touches or goes above upper band
            elif current_price >= current_upper * (1 - self.band_tolerance):
                deviation = float((current_price - current_middle) / current_middle)
                strength = min(1.0, deviation * 5)
                signal = await self._create_signal(
                    symbol, SignalType.SELL, strength, current_price, current_middle, current_upper
                )
                if signal:
                    signals.append(signal)
        
        return signals
    
    def _calculate_bollinger_bands(self, prices: List[Decimal]) -> Tuple[List[Decimal], List[Decimal], List[Decimal]]:
        """Calculate Bollinger Bands"""
        if len(prices) < self.period:
            return [], [], []
        
        upper_bands = []
        middle_bands = []
        lower_bands = []
        
        for i in range(self.period - 1, len(prices)):
            price_window = prices[i - self.period + 1:i + 1]
            
            # Calculate middle band (simple moving average)
            middle = sum(price_window) / self.period
            middle_bands.append(middle)
            
            # Calculate standard deviation
            variance = sum((p - middle) ** 2 for p in price_window) / self.period
            std = variance ** Decimal('0.5')
            
            # Calculate upper and lower bands
            upper = middle + (self.std_dev * std)
            lower = middle - (self.std_dev * std)
            
            upper_bands.append(upper)
            lower_bands.append(lower)
        
        return upper_bands, middle_bands, lower_bands
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, 
                           price: Decimal, target_price: Decimal, stop_level: Decimal) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        # Calculate stop loss and take profit
        if signal_type == SignalType.BUY:
            stop_loss = price * Decimal('0.95')  # 5% stop loss
            take_profit = target_price  # Middle band target
        else:  # SELL
            stop_loss = price * Decimal('1.05')  # 5% stop loss
            take_profit = target_price  # Middle band target
        
        return TradingSignal(
            signal_id=f"BBANDS_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            stop_loss=stop_loss,
            take_profit=take_profit,
            time_horizon=timedelta(hours=12),
            metadata={
                'strategy_type': 'bollinger_bands',
                'period': self.period,
                'std_dev': self.std_dev,
                'target_price': target_price
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


class BollingerSqueezeStrategy(BaseTimeSeriesStrategy):
    """Bollinger Bands Squeeze Strategy
    
    Identifies periods of low volatility (squeeze) for breakout signals.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['period', 'std_dev', 'squeeze_threshold']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1h")
        
        self.period = int(config.parameters.get('period', 20))
        self.std_dev = float(config.parameters.get('std_dev', 2.0))
        self.squeeze_threshold = float(config.parameters.get('squeeze_threshold', 0.1))
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.7))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        for symbol in self.config.symbols:
            data = await self._get_historical_data(symbol)
            if len(data) < self.period * 2:
                continue
            
            prices = [Decimal(str(item['close'])) for item in data]
            current_price = prices[-1]
            
            # Calculate Bollinger Bands
            upper_bands, middle_bands, lower_bands = self._calculate_bollinger_bands(prices)
            
            if len(upper_bands) < 2:
                continue
            
            # Calculate band width (squeeze indicator)
            current_upper = upper_bands[-1]
            current_lower = lower_bands[-1]
            band_width = (current_upper - current_lower) / middle_bands[-1]
            
            # Calculate average band width over lookback period
            recent_widths = [
                (upper_bands[i] - lower_bands[i]) / middle_bands[i]
                for i in range(-min(len(upper_bands), 20), 0)
            ]
            avg_band_width = sum(recent_widths) / len(recent_widths)
            
            # Check for squeeze
            if band_width < avg_band_width * (1 - self.squeeze_threshold):
                # Squeeze detected - look for breakout
                if len(prices) >= 2:
                    price_change = (prices[-1] - prices[-2]) / prices[-2]
                    
                    if price_change > 0.02:  # Strong upward breakout
                        strength = min(1.0, (1 - band_width / avg_band_width) * 2)
                        signal = await self._create_signal(
                            symbol, SignalType.BUY, strength, current_price
                        )
                        if signal:
                            signals.append(signal)
                    elif price_change < -0.02:  # Strong downward breakout
                        strength = min(1.0, (1 - band_width / avg_band_width) * 2)
                        signal = await self._create_signal(
                            symbol, SignalType.SELL, strength, current_price
                        )
                        if signal:
                            signals.append(signal)
        
        return signals
    
    def _calculate_bollinger_bands(self, prices: List[Decimal]) -> Tuple[List[Decimal], List[Decimal], List[Decimal]]:
        """Calculate Bollinger Bands"""
        if len(prices) < self.period:
            return [], [], []
        
        upper_bands = []
        middle_bands = []
        lower_bands = []
        
        for i in range(self.period - 1, len(prices)):
            price_window = prices[i - self.period + 1:i + 1]
            
            middle = sum(price_window) / self.period
            middle_bands.append(middle)
            
            variance = sum((p - middle) ** 2 for p in price_window) / self.period
            std = variance ** Decimal('0.5')
            
            upper = middle + (self.std_dev * std)
            lower = middle - (self.std_dev * std)
            
            upper_bands.append(upper)
            lower_bands.append(lower)
        
        return upper_bands, middle_bands, lower_bands
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, price: Decimal) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        return TradingSignal(
            signal_id=f"BB_SQUEEZE_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            metadata={
                'strategy_type': 'bollinger_squeeze',
                'period': self.period,
                'squeeze_threshold': self.squeeze_threshold
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


class BollingerBreakoutStrategy(BaseTimeSeriesStrategy):
    """Bollinger Bands Breakout Strategy
    
    Trades breakouts from Bollinger Bands with momentum confirmation.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['period', 'std_dev', 'breakout_threshold']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1h")
        
        self.period = int(config.parameters.get('period', 20))
        self.std_dev = float(config.parameters.get('std_dev', 2.0))
        self.breakout_threshold = float(config.parameters.get('breakout_threshold', 1.2))
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.6))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        for symbol in self.config.symbols:
            data = await self._get_historical_data(symbol)
            if len(data) < self.period + 2:
                continue
            
            prices = [Decimal(str(item['close'])) for item in data]
            volumes = [float(item.get('volume', 0)) for item in data]
            current_price = prices[-1]
            
            # Calculate Bollinger Bands
            upper_bands, middle_bands, lower_bands = self._calculate_bollinger_bands(prices)
            
            if len(upper_bands) < 2:
                continue
            
            current_upper = upper_bands[-1]
            current_middle = middle_bands[-1]
            current_lower = lower_bands[-1]
            prev_upper = upper_bands[-2]
            prev_lower = lower_bands[-2]
            
            # Volume confirmation
            avg_volume = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else volumes[-1] if volumes else 1
            current_volume = volumes[-1] if volumes else 1
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Check for breakout
            # Upward breakout
            if prev_upper >= prices[-2] and current_price > current_upper * self.breakout_threshold:
                if volume_ratio > 1.2:  # Volume confirmation
                    strength = min(1.0, (current_price - current_upper) / current_upper * 10)
                    signal = await self._create_signal(
                        symbol, SignalType.BUY, strength, current_price
                    )
                    if signal:
                        signals.append(signal)
            
            # Downward breakout
            elif prev_lower <= prices[-2] and current_price < current_lower * self.breakout_threshold:
                if volume_ratio > 1.2:  # Volume confirmation
                    strength = min(1.0, (current_lower - current_price) / current_price * 10)
                    signal = await self._create_signal(
                        symbol, SignalType.SELL, strength, current_price
                    )
                    if signal:
                        signals.append(signal)
        
        return signals
    
    def _calculate_bollinger_bands(self, prices: List[Decimal]) -> Tuple[List[Decimal], List[Decimal], List[Decimal]]:
        """Calculate Bollinger Bands"""
        if len(prices) < self.period:
            return [], [], []
        
        upper_bands = []
        middle_bands = []
        lower_bands = []
        
        for i in range(self.period - 1, len(prices)):
            price_window = prices[i - self.period + 1:i + 1]
            
            middle = sum(price_window) / self.period
            middle_bands.append(middle)
            
            variance = sum((p - middle) ** 2 for p in price_window) / self.period
            std = variance ** Decimal('0.5')
            
            upper = middle + (self.std_dev * std)
            lower = middle - (self.std_dev * std)
            
            upper_bands.append(upper)
            lower_bands.append(lower)
        
        return upper_bands, middle_bands, lower_bands
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, price: Decimal) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        return TradingSignal(
            signal_id=f"BB_BREAKOUT_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            metadata={
                'strategy_type': 'bollinger_breakout',
                'period': self.period,
                'breakout_threshold': self.breakout_threshold
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


# ============================================================================
# RSI MEAN REVERSION STRATEGIES
# ============================================================================

class RSIOverboughtOversoldStrategy(BaseTimeSeriesStrategy):
    """RSI Overbought/Oversold Mean Reversion Strategy
    
    Classic RSI mean reversion using overbought (>70) and oversold (<30) levels.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['rsi_period', 'overbought_threshold', 'oversold_threshold']
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
            prev_rsi = rsi_values[-2]
            
            # Generate signals
            # Buy signal: RSI crosses from oversold back above threshold
            if prev_rsi <= self.oversold_threshold and current_rsi > self.oversold_threshold:
                strength = (current_rsi - self.oversold_threshold) / 20  # Normalize
                signal = await self._create_signal(
                    symbol, SignalType.BUY, strength, current_price
                )
                if signal:
                    signals.append(signal)
            
            # Sell signal: RSI crosses from overbought back below threshold
            elif prev_rsi >= self.overbought_threshold and current_rsi < self.overbought_threshold:
                strength = (self.overbought_threshold - current_rsi) / 20
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
            signal_id=f"RSI_MEAN_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            metadata={
                'strategy_type': 'rsi_mean_reversion',
                'rsi_period': self.rsi_period,
                'overbought_threshold': self.overbought_threshold,
                'oversold_threshold': self.oversold_threshold
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


class RSIExtremeMeanReversionStrategy(BaseTimeSeriesStrategy):
    """RSI Extreme Mean Reversion Strategy
    
    Uses extreme RSI levels (below 20 or above 80) for strong mean reversion signals.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['rsi_period', 'extreme_high', 'extreme_low']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1h")
        
        self.rsi_period = int(config.parameters.get('rsi_period', 14))
        self.extreme_high = float(config.parameters.get('extreme_high', 80.0))
        self.extreme_low = float(config.parameters.get('extreme_low', 20.0))
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.8))
    
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
            prev_rsi = rsi_values[-2]
            
            # Generate signals for extreme levels
            # Buy signal: RSI extremely oversold (below extreme_low)
            if current_rsi <= self.extreme_low:
                strength = (self.extreme_low - current_rsi) / 20  # Stronger signal for more extreme levels
                signal = await self._create_signal(
                    symbol, SignalType.BUY, strength, current_price
                )
                if signal:
                    signals.append(signal)
            
            # Sell signal: RSI extremely overbought (above extreme_high)
            elif current_rsi >= self.extreme_high:
                strength = (current_rsi - self.extreme_high) / 20
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
                'strategy_type': 'rsi_extreme_mean_reversion',
                'rsi_period': self.rsi_period,
                'extreme_high': self.extreme_high,
                'extreme_low': self.extreme_low
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


# ============================================================================
# STATISTICAL MEAN REVERSION STRATEGIES
# ============================================================================

class ZScoreReversionStrategy(BaseTimeSeriesStrategy):
    """Z-Score Mean Reversion Strategy
    
    Uses statistical z-scores to identify price deviations from mean.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['period', 'z_score_threshold']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1h")
        
        self.period = int(config.parameters.get('period', 20))
        self.z_score_threshold = float(config.parameters.get('z_score_threshold', 2.0))
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.7))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        for symbol in self.config.symbols:
            data = await self._get_historical_data(symbol)
            if len(data) < self.period + 2:
                continue
            
            prices = [Decimal(str(item['close'])) for item in data]
            current_price = prices[-1]
            
            # Calculate z-score
            z_score = self._calculate_z_score(prices)
            
            if z_score is None:
                continue
            
            # Generate signals
            if abs(z_score) >= self.z_score_threshold:
                if z_score < -self.z_score_threshold:  # Price significantly below mean
                    strength = min(1.0, abs(z_score) / 4.0)  # Normalize
                    signal = await self._create_signal(
                        symbol, SignalType.BUY, strength, current_price, abs(z_score)
                    )
                    if signal:
                        signals.append(signal)
                elif z_score > self.z_score_threshold:  # Price significantly above mean
                    strength = min(1.0, abs(z_score) / 4.0)
                    signal = await self._create_signal(
                        symbol, SignalType.SELL, strength, current_price, abs(z_score)
                    )
                    if signal:
                        signals.append(signal)
        
        return signals
    
    def _calculate_z_score(self, prices: List[Decimal]) -> Optional[float]:
        """Calculate z-score of current price relative to historical mean"""
        if len(prices) < self.period:
            return None
        
        # Get recent prices for calculation
        recent_prices = [float(p) for p in prices[-self.period:]]
        current_price = float(prices[-1])
        
        # Calculate mean and standard deviation
        mean_price = sum(recent_prices) / len(recent_prices)
        if mean_price == 0:
            return None
        
        variance = sum((p - mean_price) ** 2 for p in recent_prices) / len(recent_prices)
        std_dev = variance ** 0.5
        
        if std_dev == 0:
            return None
        
        # Calculate z-score
        z_score = (current_price - mean_price) / std_dev
        return z_score
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, price: Decimal, z_score: float) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        return TradingSignal(
            signal_id=f"ZSCORE_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            metadata={
                'strategy_type': 'z_score_reversion',
                'period': self.period,
                'z_score': z_score
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


class PercentileReversionStrategy(BaseTimeSeriesStrategy):
    """Percentile Mean Reversion Strategy
    
    Uses price percentiles to identify extreme conditions for mean reversion.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['period', 'high_percentile', 'low_percentile']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1h")
        
        self.period = int(config.parameters.get('period', 50))
        self.high_percentile = float(config.parameters.get('high_percentile', 90.0))
        self.low_percentile = float(config.parameters.get('low_percentile', 10.0))
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.7))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        for symbol in self.config.symbols:
            data = await self._get_historical_data(symbol)
            if len(data) < self.period + 2:
                continue
            
            prices = [Decimal(str(item['close'])) for item in data]
            current_price = prices[-1]
            
            # Calculate percentiles
            high_threshold, low_threshold = self._calculate_percentile_thresholds(prices)
            
            if high_threshold is None or low_threshold is None:
                continue
            
            # Generate signals
            if current_price <= low_threshold:
                percentile_position = self._calculate_percentile_position(prices, current_price)
                strength = max(0, (self.low_percentile - percentile_position) / self.low_percentile)
                signal = await self._create_signal(
                    symbol, SignalType.BUY, strength, current_price
                )
                if signal:
                    signals.append(signal)
            
            elif current_price >= high_threshold:
                percentile_position = self._calculate_percentile_position(prices, current_price)
                strength = max(0, (percentile_position - self.high_percentile) / (100 - self.high_percentile))
                signal = await self._create_signal(
                    symbol, SignalType.SELL, strength, current_price
                )
                if signal:
                    signals.append(signal)
        
        return signals
    
    def _calculate_percentile_thresholds(self, prices: List[Decimal]) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """Calculate high and low percentile thresholds"""
        if len(prices) < self.period:
            return None, None
        
        recent_prices = [float(p) for p in prices[-self.period:]]
        recent_prices.sort()
        
        if not recent_prices:
            return None, None
        
        try:
            high_threshold = np.percentile(recent_prices, self.high_percentile)
            low_threshold = np.percentile(recent_prices, self.low_percentile)
            
            return Decimal(str(high_threshold)), Decimal(str(low_threshold))
        except:
            return None, None
    
    def _calculate_percentile_position(self, prices: List[Decimal], current_price: Decimal) -> float:
        """Calculate current price percentile position"""
        recent_prices = [float(p) for p in prices[-self.period:]]
        current_price_f = float(current_price)
        
        # Count prices below current price
        below_count = sum(1 for p in recent_prices if p < current_price_f)
        
        # Calculate percentile
        percentile = (below_count / len(recent_prices)) * 100
        return percentile
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, price: Decimal) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        return TradingSignal(
            signal_id=f"PERCENTILE_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            metadata={
                'strategy_type': 'percentile_reversion',
                'period': self.period,
                'high_percentile': self.high_percentile,
                'low_percentile': self.low_percentile
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


class PriceDeviationReversionStrategy(BaseTimeSeriesStrategy):
    """Price Deviation Mean Reversion Strategy
    
    Uses price deviation from moving average for mean reversion signals.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['ma_period', 'deviation_threshold']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1h")
        
        self.ma_period = int(config.parameters.get('ma_period', 20))
        self.deviation_threshold = float(config.parameters.get('deviation_threshold', 0.05))
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.6))
    
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
            
            if len(ma_values) < 1:
                continue
            
            current_ma = ma_values[-1]
            
            # Calculate price deviation from MA
            deviation_pct = float((current_price - current_ma) / current_ma)
            
            # Generate signals
            if abs(deviation_pct) >= self.deviation_threshold:
                if deviation_pct < -self.deviation_threshold:  # Price below MA
                    strength = min(1.0, abs(deviation_pct) * 10)
                    signal = await self._create_signal(
                        symbol, SignalType.BUY, strength, current_price
                    )
                    if signal:
                        signals.append(signal)
                elif deviation_pct > self.deviation_threshold:  # Price above MA
                    strength = min(1.0, abs(deviation_pct) * 10)
                    signal = await self._create_signal(
                        symbol, SignalType.SELL, strength, current_price
                    )
                    if signal:
                        signals.append(signal)
        
        return signals
    
    def _calculate_sma(self, prices: List[Decimal], period: int) -> List[Decimal]:
        """Calculate Simple Moving Average"""
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
            signal_id=f"DEV_REVERSION_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            metadata={
                'strategy_type': 'price_deviation_reversion',
                'ma_period': self.ma_period,
                'deviation_threshold': self.deviation_threshold
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


# ============================================================================
# PAIRS TRADING STRATEGIES
# ============================================================================

class StatisticalArbitrageStrategy(BaseTimeSeriesStrategy):
    """Statistical Arbitrage Strategy
    
    Implements pairs trading using statistical relationships between assets.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['asset1', 'asset2', 'lookback_period', 'entry_threshold']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1h")
        
        self.asset1 = config.parameters.get('asset1')
        self.asset2 = config.parameters.get('asset2')
        self.lookback_period = int(config.parameters.get('lookback_period', 100))
        self.entry_threshold = float(config.parameters.get('entry_threshold', 2.0))
        self.exit_threshold = float(config.parameters.get('exit_threshold', 0.5))
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.7))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        # For pairs trading, we need two symbols
        if len(self.config.symbols) < 2:
            return signals
        
        symbol1, symbol2 = self.config.symbols[0], self.config.symbols[1]
        
        try:
            # Get data for both assets
            data1 = await self._get_historical_data(symbol1)
            data2 = await self._get_historical_data(symbol2)
            
            if len(data1) < self.lookback_period + 2 or len(data2) < self.lookback_period + 2:
                return signals
            
            prices1 = [Decimal(str(item['close'])) for item in data1]
            prices2 = [Decimal(str(item['close'])) for item in data2]
            
            current_price1 = prices1[-1]
            current_price2 = prices2[-1]
            
            # Calculate spread
            spread = self._calculate_spread(prices1, prices2, self.lookback_period)
            
            if spread is None:
                return signals
            
            current_spread = spread[-1]
            
            # Calculate z-score
            mean_spread = sum(spread) / len(spread)
            variance = sum((s - mean_spread) ** 2 for s in spread) / len(spread)
            std_spread = variance ** 0.5
            
            if std_spread == 0:
                return signals
            
            z_score = (current_spread - mean_spread) / std_spread
            
            # Generate trading signals
            if abs(z_score) >= self.entry_threshold:
                if z_score > self.entry_threshold:  # Spread too wide - short spread
                    signal1 = await self._create_signal(
                        symbol1, SignalType.SELL, min(1.0, z_score / 3.0), current_price1
                    )
                    signal2 = await self._create_signal(
                        symbol2, SignalType.BUY, min(1.0, z_score / 3.0), current_price2
                    )
                    if signal1 and signal2:
                        signals.extend([signal1, signal2])
                
                elif z_score < -self.entry_threshold:  # Spread too narrow - long spread
                    signal1 = await self._create_signal(
                        symbol1, SignalType.BUY, min(1.0, abs(z_score) / 3.0), current_price1
                    )
                    signal2 = await self._create_signal(
                        symbol2, SignalType.SELL, min(1.0, abs(z_score) / 3.0), current_price2
                    )
                    if signal1 and signal2:
                        signals.extend([signal1, signal2])
        
        except Exception as e:
            logger.error(f"Error in statistical arbitrage strategy: {e}")
        
        return signals
    
    def _calculate_spread(self, prices1: List[Decimal], prices2: List[Decimal], period: int) -> Optional[List[float]]:
        """Calculate price spread using linear regression"""
        if len(prices1) < period or len(prices2) < period:
            return None
        
        # Use recent period for calculation
        recent_prices1 = [float(p) for p in prices1[-period:]]
        recent_prices2 = [float(p) for p in prices2[-period:]]
        
        try:
            # Calculate linear regression to find best fit
            slope, intercept = np.polyfit(recent_prices2, recent_prices1, 1)
            
            # Calculate spread as price1 - (slope * price2 + intercept)
            spread = [
                recent_prices1[i] - (slope * recent_prices2[i] + intercept)
                for i in range(len(recent_prices1))
            ]
            
            return spread
        except:
            return None
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, price: Decimal) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        return TradingSignal(
            signal_id=f"STAT_ARB_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            metadata={
                'strategy_type': 'statistical_arbitrage',
                'pairs_trade': True,
                'asset1': self.asset1,
                'asset2': self.asset2
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


class CorrelationBasedPairsStrategy(BaseTimeSeriesStrategy):
    """Correlation-Based Pairs Trading Strategy
    
    Uses correlation analysis to identify trading opportunities between correlated assets.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['asset1', 'asset2', 'min_correlation', 'deviation_threshold']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1h")
        
        self.asset1 = config.parameters.get('asset1')
        self.asset2 = config.parameters.get('asset2')
        self.min_correlation = float(config.parameters.get('min_correlation', 0.7))
        self.deviation_threshold = float(config.parameters.get('deviation_threshold', 0.02))
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.6))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        if len(self.config.symbols) < 2:
            return signals
        
        symbol1, symbol2 = self.config.symbols[0], self.config.symbols[1]
        
        try:
            # Get data for both assets
            data1 = await self._get_historical_data(symbol1)
            data2 = await self._get_historical_data(symbol2)
            
            if len(data1) < 50 or len(data2) < 50:  # Need sufficient data for correlation
                return signals
            
            prices1 = [float(item['close']) for item in data1]
            prices2 = [float(item['close']) for item in data2]
            
            # Calculate returns
            returns1 = [(prices1[i] - prices1[i-1]) / prices1[i-1] for i in range(1, len(prices1))]
            returns2 = [(prices2[i] - prices2[i-1]) / prices2[i-1] for i in range(1, len(prices2))]
            
            # Calculate correlation
            correlation = self._calculate_correlation(returns1[-50:], returns2[-50:])  # Use last 50 periods
            
            if correlation is None or abs(correlation) < self.min_correlation:
                return signals
            
            current_price1 = Decimal(str(prices1[-1]))
            current_price2 = Decimal(str(prices2[-1]))
            
            # Calculate price ratio
            ratio = current_price1 / current_price2
            
            # Calculate historical ratio statistics
            historical_ratios = [prices1[i] / prices2[i] for i in range(len(prices1))]
            mean_ratio = sum(historical_ratios[-50:]) / 50  # Last 50 periods
            std_ratio = (sum((r - mean_ratio) ** 2 for r in historical_ratios[-50:]) / 50) ** 0.5
            
            # Calculate z-score of current ratio
            if std_ratio > 0:
                z_score = float((ratio - mean_ratio) / std_ratio)
                
                # Generate signals based on deviation
                if abs(z_score) >= 2.0:  # Significant deviation
                    if z_score > 2.0:  # Asset1 relatively expensive
                        strength = min(1.0, abs(z_score) / 4.0)
                        signal1 = await self._create_signal(
                            symbol1, SignalType.SELL, strength, current_price1
                        )
                        signal2 = await self._create_signal(
                            symbol2, SignalType.BUY, strength, current_price2
                        )
                        if signal1 and signal2:
                            signals.extend([signal1, signal2])
                    
                    elif z_score < -2.0:  # Asset1 relatively cheap
                        strength = min(1.0, abs(z_score) / 4.0)
                        signal1 = await self._create_signal(
                            symbol1, SignalType.BUY, strength, current_price1
                        )
                        signal2 = await self._create_signal(
                            symbol2, SignalType.SELL, strength, current_price2
                        )
                        if signal1 and signal2:
                            signals.extend([signal1, signal2])
        
        except Exception as e:
            logger.error(f"Error in correlation pairs strategy: {e}")
        
        return signals
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> Optional[float]:
        """Calculate correlation coefficient"""
        if len(x) != len(y) or len(x) < 2:
            return None
        
        try:
            correlation, _ = stats.pearsonr(x, y)
            return correlation
        except:
            return None
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, price: Decimal) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        return TradingSignal(
            signal_id=f"CORR_PAIRS_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            metadata={
                'strategy_type': 'correlation_pairs',
                'pairs_trade': True,
                'asset1': self.asset1,
                'asset2': self.asset2
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


class CointegrationPairsStrategy(BaseTimeSeriesStrategy):
    """Cointegration Pairs Trading Strategy
    
    Uses cointegration analysis to identify long-term equilibrium relationships.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['asset1', 'asset2', 'lookback_period']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1h")
        
        self.asset1 = config.parameters.get('asset1')
        self.asset2 = config.parameters.get('asset2')
        self.lookback_period = int(config.parameters.get('lookback_period', 200))
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.7))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        if len(self.config.symbols) < 2:
            return signals
        
        symbol1, symbol2 = self.config.symbols[0], self.config.symbols[1]
        
        try:
            # Get data for both assets
            data1 = await self._get_historical_data(symbol1)
            data2 = await self._get_historical_data(symbol2)
            
            if len(data1) < self.lookback_period or len(data2) < self.lookback_period:
                return signals
            
            prices1 = [float(item['close']) for item in data1]
            prices2 = [float(item['close']) for item in data2]
            
            # Test for cointegration
            cointegration_result = self._test_cointegration(prices1, prices2)
            
            if not cointegration_result['is_cointegrated']:
                return signals
            
            # Calculate hedge ratio
            hedge_ratio = cointegration_result['hedge_ratio']
            
            # Calculate spread
            spread = [
                prices1[i] - hedge_ratio * prices2[i]
                for i in range(len(prices1))
            ]
            
            current_price1 = Decimal(str(prices1[-1]))
            current_price2 = Decimal(str(prices2[-1]))
            current_spread = spread[-1]
            
            # Calculate spread statistics
            spread_mean = sum(spread) / len(spread)
            spread_std = (sum((s - spread_mean) ** 2 for s in spread) / len(spread)) ** 0.5
            
            # Calculate z-score
            if spread_std > 0:
                z_score = (current_spread - spread_mean) / spread_std
                
                # Generate signals
                if abs(z_score) >= 2.0:
                    if z_score > 2.0:  # Spread too high - short spread
                        strength = min(1.0, abs(z_score) / 3.0)
                        signal1 = await self._create_signal(
                            symbol1, SignalType.SELL, strength, current_price1
                        )
                        signal2 = await self._create_signal(
                            symbol2, SignalType.BUY, strength, current_price2
                        )
                        if signal1 and signal2:
                            signals.extend([signal1, signal2])
                    
                    elif z_score < -2.0:  # Spread too low - long spread
                        strength = min(1.0, abs(z_score) / 3.0)
                        signal1 = await self._create_signal(
                            symbol1, SignalType.BUY, strength, current_price1
                        )
                        signal2 = await self._create_signal(
                            symbol2, SignalType.SELL, strength, current_price2
                        )
                        if signal1 and signal2:
                            signals.extend([signal1, signal2])
        
        except Exception as e:
            logger.error(f"Error in cointegration pairs strategy: {e}")
        
        return signals
    
    def _test_cointegration(self, prices1: List[float], prices2: List[float]) -> Dict[str, Any]:
        """Test for cointegration using Engle-Granger test"""
        try:
            # Calculate hedge ratio using OLS
            slope, intercept = np.polyfit(prices2, prices1, 1)
            
            # Calculate residuals
            residuals = [prices1[i] - (slope * prices2[i] + intercept) for i in range(len(prices1))]
            
            # Perform ADF test on residuals (simplified version)
            # In a real implementation, you would use statsmodels for proper ADF test
            # For now, we'll use a simple check based on residual variance
            residual_std = (sum(r ** 2 for r in residuals) / len(residuals)) ** 0.5
            
            # Simple cointegration test: if residuals are relatively stable
            is_cointegrated = residual_std < np.std(prices1) * 0.1
            
            return {
                'is_cointegrated': is_cointegrated,
                'hedge_ratio': slope,
                'residuals': residuals
            }
        except:
            return {'is_cointegrated': False, 'hedge_ratio': 1.0, 'residuals': []}
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, price: Decimal) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        return TradingSignal(
            signal_id=f"COINT_PAIRS_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            metadata={
                'strategy_type': 'cointegration_pairs',
                'pairs_trade': True,
                'asset1': self.asset1,
                'asset2': self.asset2
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


# ============================================================================
# SUPPORT/RESISTANCE STRATEGIES
# ============================================================================

class PivotPointsStrategy(BaseTimeSeriesStrategy):
    """Pivot Points Strategy
    
    Uses classical pivot points for support/resistance identification.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['pivot_type']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1h")
        
        self.pivot_type = config.parameters.get('pivot_type', 'classic')  # classic, fibonacci, woodie
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.6))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        for symbol in self.config.symbols:
            data = await self._get_historical_data(symbol)
            if len(data) < 2:
                continue
            
            current_price = Decimal(str(data[-1]['close']))
            
            # Calculate pivot points
            pivot_data = self._calculate_pivot_points(data[-2:])  # Use previous day
            
            if not pivot_data:
                continue
            
            # Check for bounce off support/resistance
            if self._is_near_support(current_price, pivot_data):
                # Support bounce - buy signal
                strength = 0.7
                signal = await self._create_signal(
                    symbol, SignalType.BUY, strength, current_price, pivot_data
                )
                if signal:
                    signals.append(signal)
            
            elif self._is_near_resistance(current_price, pivot_data):
                # Resistance bounce - sell signal
                strength = 0.7
                signal = await self._create_signal(
                    symbol, SignalType.SELL, strength, current_price, pivot_data
                )
                if signal:
                    signals.append(signal)
        
        return signals
    
    def _calculate_pivot_points(self, data: List[Dict[str, Any]]) -> Dict[str, Decimal]:
        """Calculate pivot points"""
        if len(data) < 1:
            return {}
        
        # Use daily data for pivot calculation
        daily_data = data[0]  # Previous day's data
        
        high = Decimal(str(daily_data.get('high', daily_data.get('close', 0))))
        low = Decimal(str(daily_data.get('low', daily_data.get('close', 0))))
        close = Decimal(str(daily_data.get('close', 0)))
        
        if self.pivot_type == 'classic':
            # Classic pivot points
            pivot = (high + low + close) / Decimal('3')
            r1 = (pivot * Decimal('2')) - low
            s1 = (pivot * Decimal('2')) - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
            r3 = high + (high - low * Decimal('2'))
            s3 = low - (high - low * Decimal('2'))
        
        elif self.pivot_type == 'fibonacci':
            # Fibonacci pivot points
            pivot = (high + low + close) / Decimal('3')
            r1 = pivot + (high - low) * Decimal('0.382')
            s1 = pivot - (high - low) * Decimal('0.382')
            r2 = pivot + (high - low) * Decimal('0.618')
            s2 = pivot - (high - low) * Decimal('0.618')
            r3 = pivot + (high - low)
            s3 = pivot - (high - low)
        
        else:  # woodie
            # Woodie's pivot points
            pivot = (high + low + Decimal('2') * close) / Decimal('4')
            r1 = (Decimal('2') * pivot) - low
            s1 = (Decimal('2') * pivot) - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
        
        return {
            'pivot': pivot,
            'r1': r1, 'r2': r2, 'r3': r3 if self.pivot_type != 'woodie' else r2,
            's1': s1, 's2': s2, 's3': s3 if self.pivot_type != 'woodie' else s2
        }
    
    def _is_near_support(self, price: Decimal, pivot_data: Dict[str, Decimal]) -> bool:
        """Check if price is near support level"""
        tolerance = Decimal('0.005')  # 0.5% tolerance
        
        if 's3' in pivot_data and abs(price - pivot_data['s3']) <= pivot_data['s3'] * tolerance:
            return True
        if 's2' in pivot_data and abs(price - pivot_data['s2']) <= pivot_data['s2'] * tolerance:
            return True
        if 's1' in pivot_data and abs(price - pivot_data['s1']) <= pivot_data['s1'] * tolerance:
            return True
        
        return False
    
    def _is_near_resistance(self, price: Decimal, pivot_data: Dict[str, Decimal]) -> bool:
        """Check if price is near resistance level"""
        tolerance = Decimal('0.005')  # 0.5% tolerance
        
        if 'r3' in pivot_data and abs(price - pivot_data['r3']) <= pivot_data['r3'] * tolerance:
            return True
        if 'r2' in pivot_data and abs(price - pivot_data['r2']) <= pivot_data['r2'] * tolerance:
            return True
        if 'r1' in pivot_data and abs(price - pivot_data['r1']) <= pivot_data['r1'] * tolerance:
            return True
        
        return False
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, 
                           price: Decimal, pivot_data: Dict[str, Decimal]) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        # Determine target levels
        if signal_type == SignalType.BUY:
            target = pivot_data.get('pivot', price * Decimal('1.02'))
            stop_loss = price * Decimal('0.98')
        else:  # SELL
            target = pivot_data.get('pivot', price * Decimal('0.98'))
            stop_loss = price * Decimal('1.02')
        
        return TradingSignal(
            signal_id=f"PIVOT_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            stop_loss=stop_loss,
            take_profit=target,
            metadata={
                'strategy_type': 'pivot_points',
                'pivot_type': self.pivot_type,
                'pivot_data': {k: float(v) for k, v in pivot_data.items()}
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


class FibonacciRetracementStrategy(BaseTimeSeriesStrategy):
    """Fibonacci Retracement Strategy
    
    Uses Fibonacci retracement levels for support/resistance identification.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['lookback_period']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1h")
        
        self.lookback_period = int(config.parameters.get('lookback_period', 100))
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.6))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        for symbol in self.config.symbols:
            data = await self._get_historical_data(symbol)
            if len(data) < self.lookback_period:
                continue
            
            prices = [Decimal(str(item['close'])) for item in data]
            current_price = prices[-1]
            
            # Find recent swing high and low
            swing_high, swing_low = self._find_swing_points(prices)
            
            if not swing_high or not swing_low:
                continue
            
            # Calculate Fibonacci retracement levels
            fib_levels = self._calculate_fibonacci_levels(swing_high, swing_low)
            
            # Check for bounce at Fibonacci levels
            nearest_level = self._find_nearest_fib_level(current_price, fib_levels)
            
            if nearest_level:
                level_distance = abs(current_price - nearest_level['price']) / current_price
                
                if level_distance < 0.01:  # Within 1% of level
                    if nearest_level['type'] == 'support':
                        strength = min(1.0, 0.8 - level_distance * 10)
                        signal = await self._create_signal(
                            symbol, SignalType.BUY, strength, current_price
                        )
                        if signal:
                            signals.append(signal)
                    elif nearest_level['type'] == 'resistance':
                        strength = min(1.0, 0.8 - level_distance * 10)
                        signal = await self._create_signal(
                            symbol, SignalType.SELL, strength, current_price
                        )
                        if signal:
                            signals.append(signal)
        
        return signals
    
    def _find_swing_points(self, prices: List[Decimal]) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """Find recent swing high and low"""
        if len(prices) < self.lookback_period:
            return None, None
        
        # Find swing high (local maximum)
        swing_high = None
        swing_high_idx = None
        
        for i in range(20, len(prices) - 20):
            if all(prices[i] >= prices[j] for j in range(i-20, i+21) if j != i):
                swing_high = prices[i]
                swing_high_idx = i
                break
        
        # Find swing low (local minimum)
        swing_low = None
        swing_low_idx = None
        
        for i in range(20, len(prices) - 20):
            if all(prices[i] <= prices[j] for j in range(i-20, i+21) if j != i):
                swing_low = prices[i]
                swing_low_idx = i
                break
        
        # Determine if it's an uptrend or downtrend
        if swing_high_idx and swing_low_idx:
            if swing_high_idx > swing_low_idx:
                # Uptrend: high is recent, low is earlier
                return swing_high, swing_low
            else:
                # Downtrend: low is recent, high is earlier
                return swing_low, swing_high
        
        return swing_high, swing_low
    
    def _calculate_fibonacci_levels(self, high: Decimal, low: Decimal) -> List[Dict[str, Any]]:
        """Calculate Fibonacci retracement levels"""
        diff = high - low
        
        # Standard Fibonacci retracement levels
        fib_percentages = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        
        levels = []
        
        for pct in fib_percentages:
            level_price = low + (diff * pct)
            levels.append({
                'price': level_price,
                'percentage': pct,
                'type': 'support' if pct < 0.5 else 'resistance'
            })
        
        return levels
    
    def _find_nearest_fib_level(self, current_price: Decimal, fib_levels: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find nearest Fibonacci level to current price"""
        min_distance = Decimal('999999')
        nearest_level = None
        
        for level in fib_levels:
            distance = abs(current_price - level['price'])
            if distance < min_distance:
                min_distance = distance
                nearest_level = level
        
        return nearest_level
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, price: Decimal) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        return TradingSignal(
            signal_id=f"FIB_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            metadata={
                'strategy_type': 'fibonacci_retracement',
                'lookback_period': self.lookback_period
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


class ChannelTradingStrategy(BaseTimeSeriesStrategy):
    """Channel Trading Strategy
    
    Identifies price channels and trades bounces off channel boundaries.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['channel_period', 'channel_tolerance']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1h")
        
        self.channel_period = int(config.parameters.get('channel_period', 50))
        self.channel_tolerance = float(config.parameters.get('channel_tolerance', 0.02))
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.6))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        for symbol in self.config.symbols:
            data = await self._get_historical_data(symbol)
            if len(data) < self.channel_period:
                continue
            
            prices = [Decimal(str(item['close'])) for item in data]
            current_price = prices[-1]
            
            # Calculate channel
            channel_data = self._calculate_channel(prices)
            
            if not channel_data:
                continue
            
            upper_channel = channel_data['upper'][-1]
            lower_channel = channel_data['lower'][-1]
            middle_channel = channel_data['middle'][-1]
            
            # Check for channel bounce
            upper_distance = (current_price - upper_channel) / upper_channel
            lower_distance = (current_price - lower_channel) / lower_channel
            
            # Near upper channel - potential reversal down
            if abs(upper_distance) <= self.channel_tolerance:
                strength = 0.7 - abs(upper_distance) * 10
                signal = await self._create_signal(
                    symbol, SignalType.SELL, strength, current_price
                )
                if signal:
                    signals.append(signal)
            
            # Near lower channel - potential reversal up
            elif abs(lower_distance) <= self.channel_tolerance:
                strength = 0.7 - abs(lower_distance) * 10
                signal = await self._create_signal(
                    symbol, SignalType.BUY, strength, current_price
                )
                if signal:
                    signals.append(signal)
        
        return signals
    
    def _calculate_channel(self, prices: List[Decimal]) -> Optional[Dict[str, List[Decimal]]]:
        """Calculate price channel using rolling min/max"""
        if len(prices) < self.channel_period:
            return None
        
        upper_channel = []
        lower_channel = []
        middle_channel = []
        
        for i in range(self.channel_period - 1, len(prices)):
            window = prices[i - self.channel_period + 1:i + 1]
            
            upper = max(window)
            lower = min(window)
            middle = (upper + lower) / Decimal('2')
            
            upper_channel.append(upper)
            lower_channel.append(lower)
            middle_channel.append(middle)
        
        return {
            'upper': upper_channel,
            'lower': lower_channel,
            'middle': middle_channel
        }
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, price: Decimal) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        return TradingSignal(
            signal_id=f"CHANNEL_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            metadata={
                'strategy_type': 'channel_trading',
                'channel_period': self.channel_period,
                'channel_tolerance': self.channel_tolerance
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


# ============================================================================
# VALUE-BASED STRATEGIES
# ============================================================================

class MovingAverageReversionStrategy(BaseTimeSeriesStrategy):
    """Moving Average Reversion Strategy
    
    Trades price reversion to moving average with volatility bands.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['ma_period', 'std_multiplier']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1h")
        
        self.ma_period = int(config.parameters.get('ma_period', 20))
        self.std_multiplier = float(config.parameters.get('std_multiplier', 2.0))
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.6))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        for symbol in self.config.symbols:
            data = await self._get_historical_data(symbol)
            if len(data) < self.ma_period + 2:
                continue
            
            prices = [Decimal(str(item['close'])) for item in data]
            current_price = prices[-1]
            
            # Calculate moving average and standard deviation bands
            ma, upper_band, lower_band = self._calculate_moving_average_bands(prices)
            
            if not ma or not upper_band or not lower_band:
                continue
            
            current_ma = ma[-1]
            current_upper = upper_band[-1]
            current_lower = lower_band[-1]
            
            # Generate mean reversion signals
            if current_price <= current_lower:
                # Price below lower band - buy signal
                deviation = (current_ma - current_price) / current_ma
                strength = min(1.0, abs(deviation) * 20)
                signal = await self._create_signal(
                    symbol, SignalType.BUY, strength, current_price, current_ma
                )
                if signal:
                    signals.append(signal)
            
            elif current_price >= current_upper:
                # Price above upper band - sell signal
                deviation = (current_price - current_ma) / current_ma
                strength = min(1.0, abs(deviation) * 20)
                signal = await self._create_signal(
                    symbol, SignalType.SELL, strength, current_price, current_ma
                )
                if signal:
                    signals.append(signal)
        
        return signals
    
    def _calculate_moving_average_bands(self, prices: List[Decimal]) -> Tuple[List[Decimal], List[Decimal], List[Decimal]]:
        """Calculate moving average with standard deviation bands"""
        if len(prices) < self.ma_period:
            return [], [], []
        
        ma_values = []
        upper_bands = []
        lower_bands = []
        
        for i in range(self.ma_period - 1, len(prices)):
            price_window = prices[i - self.ma_period + 1:i + 1]
            
            # Calculate moving average
            ma = sum(price_window) / self.ma_period
            ma_values.append(ma)
            
            # Calculate standard deviation
            variance = sum((p - ma) ** 2 for p in price_window) / self.ma_period
            std = variance ** Decimal('0.5')
            
            # Calculate bands
            upper = ma + (self.std_multiplier * std)
            lower = ma - (self.std_multiplier * std)
            
            upper_bands.append(upper)
            lower_bands.append(lower)
        
        return ma_values, upper_bands, lower_bands
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, 
                           price: Decimal, target_price: Decimal) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        # Calculate stop loss and take profit
        if signal_type == SignalType.BUY:
            stop_loss = price * Decimal('0.95')  # 5% stop loss
            take_profit = target_price  # MA target
        else:  # SELL
            stop_loss = price * Decimal('1.05')  # 5% stop loss
            take_profit = target_price  # MA target
        
        return TradingSignal(
            signal_id=f"MA_REVERSION_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                'strategy_type': 'moving_average_reversion',
                'ma_period': self.ma_period,
                'std_multiplier': self.std_multiplier
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


class FundamentalValueReversionStrategy(BaseTimeSeriesStrategy):
    """Fundamental Value Reversion Strategy
    
    Uses fundamental metrics for value-based mean reversion signals.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['pe_threshold', 'pb_threshold']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1h")
        
        self.pe_threshold = float(config.parameters.get('pe_threshold', 15.0))
        self.pb_threshold = float(config.parameters.get('pb_threshold', 1.5))
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.7))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        for symbol in self.config.symbols:
            # In a real implementation, fetch fundamental data
            # For now, simulate fundamental ratios
            pe_ratio = await self._get_pe_ratio(symbol)
            pb_ratio = await self._get_pb_ratio(symbol)
            
            if pe_ratio is None or pb_ratio is None:
                continue
            
            # Get current price
            data = await self._get_historical_data(symbol)
            if not data:
                continue
            
            current_price = Decimal(str(data[-1]['close']))
            
            # Generate value-based signals
            if pe_ratio < self.pe_threshold and pb_ratio < self.pb_threshold:
                # Undervalued - buy signal
                value_score = (self.pe_threshold - pe_ratio) / self.pe_threshold + \
                             (self.pb_threshold - pb_ratio) / self.pb_threshold
                strength = min(1.0, value_score / 2.0)
                
                signal = await self._create_signal(
                    symbol, SignalType.BUY, strength, current_price, pe_ratio, pb_ratio
                )
                if signal:
                    signals.append(signal)
            
            elif pe_ratio > self.pe_threshold * 2 and pb_ratio > self.pb_threshold * 2:
                # Overvalued - sell signal
                overvalue_score = (pe_ratio - self.pe_threshold * 2) / (self.pe_threshold * 2) + \
                                 (pb_ratio - self.pb_threshold * 2) / (self.pb_threshold * 2)
                strength = min(1.0, overvalue_score / 2.0)
                
                signal = await self._create_signal(
                    symbol, SignalType.SELL, strength, current_price, pe_ratio, pb_ratio
                )
                if signal:
                    signals.append(signal)
        
        return signals
    
    async def _get_pe_ratio(self, symbol: str) -> Optional[float]:
        """Get P/E ratio (simulated)"""
        # In real implementation, fetch from fundamental data provider
        import random
        return random.uniform(8.0, 35.0)
    
    async def _get_pb_ratio(self, symbol: str) -> Optional[float]:
        """Get P/B ratio (simulated)"""
        # In real implementation, fetch from fundamental data provider
        import random
        return random.uniform(0.5, 4.0)
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, 
                           price: Decimal, pe_ratio: float, pb_ratio: float) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        return TradingSignal(
            signal_id=f"FUNDAMENTAL_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            metadata={
                'strategy_type': 'fundamental_value_reversion',
                'pe_ratio': pe_ratio,
                'pb_ratio': pb_ratio,
                'pe_threshold': self.pe_threshold,
                'pb_threshold': self.pb_threshold
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


# ============================================================================
# VOLUME PROFILE STRATEGIES
# ============================================================================

class VWAPReversionStrategy(BaseTimeSeriesStrategy):
    """VWAP (Volume-Weighted Average Price) Reversion Strategy
    
    Uses VWAP for mean reversion signals.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['reversion_threshold']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1h")
        
        self.reversion_threshold = float(config.parameters.get('reversion_threshold', 0.02))
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.6))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        for symbol in self.config.symbols:
            data = await self._get_historical_data(symbol)
            if len(data) < 20:
                continue
            
            current_price = Decimal(str(data[-1]['close']))
            
            # Calculate VWAP
            vwap = self._calculate_vwap(data)
            
            if vwap is None:
                continue
            
            # Calculate deviation from VWAP
            deviation_pct = float((current_price - vwap) / vwap)
            
            # Generate mean reversion signals
            if abs(deviation_pct) >= self.reversion_threshold:
                if deviation_pct < -self.reversion_threshold:
                    # Price significantly below VWAP - buy signal
                    strength = min(1.0, abs(deviation_pct) * 20)
                    signal = await self._create_signal(
                        symbol, SignalType.BUY, strength, current_price, vwap
                    )
                    if signal:
                        signals.append(signal)
                
                elif deviation_pct > self.reversion_threshold:
                    # Price significantly above VWAP - sell signal
                    strength = min(1.0, abs(deviation_pct) * 20)
                    signal = await self._create_signal(
                        symbol, SignalType.SELL, strength, current_price, vwap
                    )
                    if signal:
                        signals.append(signal)
        
        return signals
    
    def _calculate_vwap(self, data: List[Dict[str, Any]]) -> Optional[Decimal]:
        """Calculate Volume-Weighted Average Price"""
        if len(data) < 2:
            return None
        
        total_volume_price = Decimal('0')
        total_volume = Decimal('0')
        
        for item in data:
            price = Decimal(str(item['close']))
            volume = Decimal(str(item.get('volume', 0)))
            
            if volume > 0:
                total_volume_price += price * volume
                total_volume += volume
        
        if total_volume == 0:
            return None
        
        vwap = total_volume_price / total_volume
        return vwap
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, 
                           price: Decimal, vwap: Decimal) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        return TradingSignal(
            signal_id=f"VWAP_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            stop_loss=vwap if signal_type == SignalType.BUY else price * Decimal('0.98'),
            take_profit=vwap if signal_type == SignalType.SELL else price * Decimal('1.02'),
            metadata={
                'strategy_type': 'vwap_reversion',
                'vwap': float(vwap),
                'reversion_threshold': self.reversion_threshold
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


class VolumeProfileReversionStrategy(BaseTimeSeriesStrategy):
    """Volume Profile Reversion Strategy
    
    Uses volume profile to identify high-volume price areas for mean reversion.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['price_bins', 'volume_threshold']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1h")
        
        self.price_bins = int(config.parameters.get('price_bins', 20))
        self.volume_threshold = float(config.parameters.get('volume_threshold', 1.5))
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.7))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        for symbol in self.config.symbols:
            data = await self._get_historical_data(symbol)
            if len(data) < 100:
                continue
            
            current_price = Decimal(str(data[-1]['close']))
            
            # Calculate volume profile
            volume_profile = self._calculate_volume_profile(data)
            
            if not volume_profile:
                continue
            
            # Find high-volume areas
            high_volume_areas = self._find_high_volume_areas(volume_profile)
            
            # Check if current price is near high-volume area
            nearby_volume_area = self._find_nearby_volume_area(current_price, high_volume_areas)
            
            if nearby_volume_area:
                # Price near high-volume area - potential support/resistance
                area_distance = abs(current_price - nearby_volume_area['price']) / current_price
                
                if area_distance < 0.01:  # Within 1% of volume area
                    if nearby_volume_area['type'] == 'support':
                        strength = min(1.0, nearby_volume_area['volume_ratio'] * 0.5)
                        signal = await self._create_signal(
                            symbol, SignalType.BUY, strength, current_price
                        )
                        if signal:
                            signals.append(signal)
                    elif nearby_volume_area['type'] == 'resistance':
                        strength = min(1.0, nearby_volume_area['volume_ratio'] * 0.5)
                        signal = await self._create_signal(
                            symbol, SignalType.SELL, strength, current_price
                        )
                        if signal:
                            signals.append(signal)
        
        return signals
    
    def _calculate_volume_profile(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate volume profile by price"""
        if len(data) < 10:
            return {}
        
        # Get price range
        prices = [Decimal(str(item['close'])) for item in data]
        min_price = min(prices)
        max_price = max(prices)
        price_range = max_price - min_price
        
        if price_range == 0:
            return {}
        
        # Create price bins
        bin_size = price_range / self.price_bins
        volume_bins = [0.0] * self.price_bins
        
        # Distribute volume into bins
        for item in data:
            price = Decimal(str(item['close']))
            volume = float(item.get('volume', 0))
            
            bin_index = int((price - min_price) / bin_size)
            if 0 <= bin_index < self.price_bins:
                volume_bins[bin_index] += volume
        
        return {
            'min_price': min_price,
            'max_price': max_price,
            'bin_size': bin_size,
            'volume_bins': volume_bins
        }
    
    def _find_high_volume_areas(self, volume_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find areas with above-average volume"""
        if not volume_profile:
            return []
        
        volume_bins = volume_profile['volume_bins']
        avg_volume = sum(volume_bins) / len(volume_bins)
        
        high_volume_areas = []
        
        for i, volume in enumerate(volume_bins):
            if volume > avg_volume * self.volume_threshold:
                price_level = volume_profile['min_price'] + (i * volume_profile['bin_size'])
                high_volume_areas.append({
                    'price': price_level,
                    'volume': volume,
                    'volume_ratio': volume / avg_volume,
                    'type': 'support' if i < len(volume_bins) / 2 else 'resistance'
                })
        
        return high_volume_areas
    
    def _find_nearby_volume_area(self, current_price: Decimal, high_volume_areas: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find high-volume area near current price"""
        min_distance = Decimal('999999')
        nearby_area = None
        
        for area in high_volume_areas:
            distance = abs(current_price - area['price'])
            if distance < min_distance:
                min_distance = distance
                nearby_area = area
        
        return nearby_area
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, price: Decimal) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        return TradingSignal(
            signal_id=f"VOL_PROFILE_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            metadata={
                'strategy_type': 'volume_profile_reversion',
                'price_bins': self.price_bins,
                'volume_threshold': self.volume_threshold
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


# ============================================================================
# STRATEGY REGISTRATION
# ============================================================================

def register_mean_reversion_strategies():
    """Register all mean reversion strategies with the strategy library"""
    
    # Register Bollinger Band strategies
    strategy_library.register_strategy(
        BollingerBandsStrategy,
        StrategyMetadata(
            strategy_id="bollinger_bands",
            name="Bollinger Bands Strategy",
            category=StrategyCategory.MEAN_REVERSION,
            description="Classic Bollinger Bands mean reversion strategy",
            long_description="Uses Bollinger Bands to identify overbought and oversold conditions. Buy when price touches lower band, sell when price touches upper band.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["bollinger", "bands", "mean-reversion", "overbought", "oversold"],
            parameters_schema={
                "required": ["period", "std_dev"],
                "properties": {
                    "period": {"type": "integer", "min": 10, "max": 50},
                    "std_dev": {"type": "float", "min": 1.5, "max": 3.0},
                    "band_tolerance": {"type": "float", "min": 0.001, "max": 0.02},
                    "signal_threshold": {"type": "float", "min": 0.3, "max": 1.0}
                }
            },
            example_config={
                "period": 20,
                "std_dev": 2.0,
                "band_tolerance": 0.01,
                "signal_threshold": 0.6
            },
            risk_warning="Can fail during strong trending markets. Use proper position sizing."
        )
    )
    
    strategy_library.register_strategy(
        BollingerSqueezeStrategy,
        StrategyMetadata(
            strategy_id="bollinger_squeeze",
            name="Bollinger Squeeze Strategy",
            category=StrategyCategory.MEAN_REVERSION,
            description="Bollinger Bands squeeze for volatility breakout signals",
            long_description="Identifies periods of low volatility (squeeze) for breakout signals.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["bollinger", "squeeze", "volatility", "breakout"],
            parameters_schema={
                "required": ["period", "std_dev", "squeeze_threshold"],
                "properties": {
                    "period": {"type": "integer", "min": 10, "max": 50},
                    "std_dev": {"type": "float", "min": 1.5, "max": 3.0},
                    "squeeze_threshold": {"type": "float", "min": 0.05, "max": 0.3},
                    "signal_threshold": {"type": "float", "min": 0.5, "max": 1.0}
                }
            }
        )
    )
    
    strategy_library.register_strategy(
        BollingerBreakoutStrategy,
        StrategyMetadata(
            strategy_id="bollinger_breakout",
            name="Bollinger Breakout Strategy",
            category=StrategyCategory.MEAN_REVERSION,
            description="Bollinger Bands breakout with volume confirmation",
            long_description="Trades breakouts from Bollinger Bands with momentum confirmation.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["bollinger", "breakout", "volume", "confirmation"],
            parameters_schema={
                "required": ["period", "std_dev", "breakout_threshold"],
                "properties": {
                    "period": {"type": "integer", "min": 10, "max": 50},
                    "std_dev": {"type": "float", "min": 1.5, "max": 3.0},
                    "breakout_threshold": {"type": "float", "min": 1.1, "max": 1.5},
                    "signal_threshold": {"type": "float", "min": 0.3, "max": 1.0}
                }
            }
        )
    )
    
    # Register RSI mean reversion strategies
    strategy_library.register_strategy(
        RSIOverboughtOversoldStrategy,
        StrategyMetadata(
            strategy_id="rsi_mean_reversion",
            name="RSI Mean Reversion Strategy",
            category=StrategyCategory.MEAN_REVERSION,
            description="Classic RSI overbought/oversold mean reversion",
            long_description="Classic RSI mean reversion using overbought (>70) and oversold (<30) levels.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["rsi", "overbought", "oversold", "oscillator", "mean-reversion"],
            parameters_schema={
                "required": ["rsi_period", "overbought_threshold", "oversold_threshold"],
                "properties": {
                    "rsi_period": {"type": "integer", "min": 5, "max": 30},
                    "overbought_threshold": {"type": "float", "min": 60.0, "max": 80.0},
                    "oversold_threshold": {"type": "float", "min": 20.0, "max": 40.0},
                    "signal_threshold": {"type": "float", "min": 0.5, "max": 1.0}
                }
            }
        )
    )
    
    strategy_library.register_strategy(
        RSIExtremeMeanReversionStrategy,
        StrategyMetadata(
            strategy_id="rsi_extreme_reversion",
            name="RSI Extreme Mean Reversion Strategy",
            category=StrategyCategory.MEAN_REVERSION,
            description="RSI extreme levels for strong mean reversion signals",
            long_description="Uses extreme RSI levels (below 20 or above 80) for strong mean reversion signals.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["rsi", "extreme", "mean-reversion", "strong-signals"],
            parameters_schema={
                "required": ["rsi_period", "extreme_high", "extreme_low"],
                "properties": {
                    "rsi_period": {"type": "integer", "min": 5, "max": 30},
                    "extreme_high": {"type": "float", "min": 75.0, "max": 90.0},
                    "extreme_low": {"type": "float", "min": 10.0, "max": 25.0},
                    "signal_threshold": {"type": "float", "min": 0.6, "max": 1.0}
                }
            }
        )
    )
    
    # Register Statistical mean reversion strategies
    strategy_library.register_strategy(
        ZScoreReversionStrategy,
        StrategyMetadata(
            strategy_id="zscore_reversion",
            name="Z-Score Mean Reversion Strategy",
            category=StrategyCategory.MEAN_REVERSION,
            description="Statistical z-score mean reversion",
            long_description="Uses statistical z-scores to identify price deviations from mean.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["z-score", "statistical", "deviation", "mean-reversion"],
            parameters_schema={
                "required": ["period", "z_score_threshold"],
                "properties": {
                    "period": {"type": "integer", "min": 10, "max": 100},
                    "z_score_threshold": {"type": "float", "min": 1.5, "max": 4.0},
                    "signal_threshold": {"type": "float", "min": 0.5, "max": 1.0}
                }
            }
        )
    )
    
    strategy_library.register_strategy(
        PercentileReversionStrategy,
        StrategyMetadata(
            strategy_id="percentile_reversion",
            name="Percentile Mean Reversion Strategy",
            category=StrategyCategory.MEAN_REVERSION,
            description="Price percentile mean reversion",
            long_description="Uses price percentiles to identify extreme conditions for mean reversion.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["percentile", "extreme-conditions", "mean-reversion"],
            parameters_schema={
                "required": ["period", "high_percentile", "low_percentile"],
                "properties": {
                    "period": {"type": "integer", "min": 20, "max": 200},
                    "high_percentile": {"type": "float", "min": 75.0, "max": 95.0},
                    "low_percentile": {"type": "float", "min": 5.0, "max": 25.0},
                    "signal_threshold": {"type": "float", "min": 0.5, "max": 1.0}
                }
            }
        )
    )
    
    strategy_library.register_strategy(
        PriceDeviationReversionStrategy,
        StrategyMetadata(
            strategy_id="price_deviation_reversion",
            name="Price Deviation Mean Reversion Strategy",
            category=StrategyCategory.MEAN_REVERSION,
            description="Price deviation from moving average reversion",
            long_description="Uses price deviation from moving average for mean reversion signals.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["price-deviation", "moving-average", "mean-reversion"],
            parameters_schema={
                "required": ["ma_period", "deviation_threshold"],
                "properties": {
                    "ma_period": {"type": "integer", "min": 10, "max": 100},
                    "deviation_threshold": {"type": "float", "min": 0.02, "max": 0.1},
                    "signal_threshold": {"type": "float", "min": 0.3, "max": 1.0}
                }
            }
        )
    )
    
    # Register Pairs trading strategies
    strategy_library.register_strategy(
        StatisticalArbitrageStrategy,
        StrategyMetadata(
            strategy_id="statistical_arbitrage",
            name="Statistical Arbitrage Strategy",
            category=StrategyCategory.MEAN_REVERSION,
            description="Statistical arbitrage pairs trading",
            long_description="Implements pairs trading using statistical relationships between assets.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["pairs-trading", "statistical", "arbitrage", "advanced"],
            parameters_schema={
                "required": ["asset1", "asset2", "lookback_period", "entry_threshold"],
                "properties": {
                    "asset1": {"type": "string"},
                    "asset2": {"type": "string"},
                    "lookback_period": {"type": "integer", "min": 50, "max": 500},
                    "entry_threshold": {"type": "float", "min": 1.5, "max": 3.0},
                    "exit_threshold": {"type": "float", "min": 0.2, "max": 1.0},
                    "signal_threshold": {"type": "float", "min": 0.5, "max": 1.0}
                }
            }
        )
    )
    
    strategy_library.register_strategy(
        CorrelationBasedPairsStrategy,
        StrategyMetadata(
            strategy_id="correlation_pairs",
            name="Correlation-Based Pairs Strategy",
            category=StrategyCategory.MEAN_REVERSION,
            description="Correlation analysis for pairs trading",
            long_description="Uses correlation analysis to identify trading opportunities between correlated assets.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["correlation", "pairs-trading", "regression"],
            parameters_schema={
                "required": ["asset1", "asset2", "min_correlation", "deviation_threshold"],
                "properties": {
                    "asset1": {"type": "string"},
                    "asset2": {"type": "string"},
                    "min_correlation": {"type": "float", "min": 0.5, "max": 0.95},
                    "deviation_threshold": {"type": "float", "min": 0.01, "max": 0.05},
                    "signal_threshold": {"type": "float", "min": 0.3, "max": 1.0}
                }
            }
        )
    )
    
    strategy_library.register_strategy(
        CointegrationPairsStrategy,
        StrategyMetadata(
            strategy_id="cointegration_pairs",
            name="Cointegration Pairs Strategy",
            category=StrategyCategory.MEAN_REVERSION,
            description="Cointegration analysis for pairs trading",
            long_description="Uses cointegration analysis to identify long-term equilibrium relationships.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["cointegration", "pairs-trading", "equilibrium", "advanced"],
            parameters_schema={
                "required": ["asset1", "asset2", "lookback_period"],
                "properties": {
                    "asset1": {"type": "string"},
                    "asset2": {"type": "string"},
                    "lookback_period": {"type": "integer", "min": 100, "max": 1000},
                    "signal_threshold": {"type": "float", "min": 0.5, "max": 1.0}
                }
            }
        )
    )
    
    # Register Support/Resistance strategies
    strategy_library.register_strategy(
        PivotPointsStrategy,
        StrategyMetadata(
            strategy_id="pivot_points",
            name="Pivot Points Strategy",
            category=StrategyCategory.MEAN_REVERSION,
            description="Classical pivot points for support/resistance",
            long_description="Uses classical pivot points for support/resistance identification.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["pivot-points", "support", "resistance", "classical"],
            parameters_schema={
                "required": ["pivot_type"],
                "properties": {
                    "pivot_type": {"type": "string", "enum": ["classic", "fibonacci", "woodie"]},
                    "signal_threshold": {"type": "float", "min": 0.3, "max": 1.0}
                }
            }
        )
    )
    
    strategy_library.register_strategy(
        FibonacciRetracementStrategy,
        StrategyMetadata(
            strategy_id="fibonacci_retracement",
            name="Fibonacci Retracement Strategy",
            category=StrategyCategory.MEAN_REVERSION,
            description="Fibonacci retracement levels for support/resistance",
            long_description="Uses Fibonacci retracement levels for support/resistance identification.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["fibonacci", "retracement", "support", "resistance"],
            parameters_schema={
                "required": ["lookback_period"],
                "properties": {
                    "lookback_period": {"type": "integer", "min": 50, "max": 500},
                    "signal_threshold": {"type": "float", "min": 0.3, "max": 1.0}
                }
            }
        )
    )
    
    strategy_library.register_strategy(
        ChannelTradingStrategy,
        StrategyMetadata(
            strategy_id="channel_trading",
            name="Channel Trading Strategy",
            category=StrategyCategory.MEAN_REVERSION,
            description="Price channel trading for bounces",
            long_description="Identifies price channels and trades bounces off channel boundaries.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["channels", "bounces", "support", "resistance"],
            parameters_schema={
                "required": ["channel_period", "channel_tolerance"],
                "properties": {
                    "channel_period": {"type": "integer", "min": 20, "max": 200},
                    "channel_tolerance": {"type": "float", "min": 0.01, "max": 0.05},
                    "signal_threshold": {"type": "float", "min": 0.3, "max": 1.0}
                }
            }
        )
    )
    
    # Register Value-based strategies
    strategy_library.register_strategy(
        MovingAverageReversionStrategy,
        StrategyMetadata(
            strategy_id="moving_average_reversion",
            name="Moving Average Reversion Strategy",
            category=StrategyCategory.MEAN_REVERSION,
            description="Moving average reversion with volatility bands",
            long_description="Trades price reversion to moving average with volatility bands.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["moving-average", "bands", "mean-reversion", "volatility"],
            parameters_schema={
                "required": ["ma_period", "std_multiplier"],
                "properties": {
                    "ma_period": {"type": "integer", "min": 10, "max": 100},
                    "std_multiplier": {"type": "float", "min": 1.0, "max": 3.0},
                    "signal_threshold": {"type": "float", "min": 0.3, "max": 1.0}
                }
            }
        )
    )
    
    strategy_library.register_strategy(
        FundamentalValueReversionStrategy,
        StrategyMetadata(
            strategy_id="fundamental_value_reversion",
            name="Fundamental Value Reversion Strategy",
            category=StrategyCategory.MEAN_REVERSION,
            description="Fundamental metrics for value-based mean reversion",
            long_description="Uses fundamental metrics for value-based mean reversion signals.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["fundamental", "value", "pe-ratio", "pb-ratio"],
            parameters_schema={
                "required": ["pe_threshold", "pb_threshold"],
                "properties": {
                    "pe_threshold": {"type": "float", "min": 10.0, "max": 25.0},
                    "pb_threshold": {"type": "float", "min": 1.0, "max": 3.0},
                    "signal_threshold": {"type": "float", "min": 0.5, "max": 1.0}
                }
            }
        )
    )
    
    # Register Volume Profile strategies
    strategy_library.register_strategy(
        VWAPReversionStrategy,
        StrategyMetadata(
            strategy_id="vwap_reversion",
            name="VWAP Reversion Strategy",
            category=StrategyCategory.MEAN_REVERSION,
            description="Volume-Weighted Average Price reversion",
            long_description="Uses VWAP for mean reversion signals.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["vwap", "volume-weighted", "mean-reversion"],
            parameters_schema={
                "required": ["reversion_threshold"],
                "properties": {
                    "reversion_threshold": {"type": "float", "min": 0.01, "max": 0.05},
                    "signal_threshold": {"type": "float", "min": 0.3, "max": 1.0}
                }
            }
        )
    )
    
    strategy_library.register_strategy(
        VolumeProfileReversionStrategy,
        StrategyMetadata(
            strategy_id="volume_profile_reversion",
            name="Volume Profile Reversion Strategy",
            category=StrategyCategory.MEAN_REVERSION,
            description="Volume profile for high-volume area reversion",
            long_description="Uses volume profile to identify high-volume price areas for mean reversion.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["volume-profile", "high-volume", "support", "resistance"],
            parameters_schema={
                "required": ["price_bins", "volume_threshold"],
                "properties": {
                    "price_bins": {"type": "integer", "min": 10, "max": 50},
                    "volume_threshold": {"type": "float", "min": 1.2, "max": 3.0},
                    "signal_threshold": {"type": "float", "min": 0.5, "max": 1.0}
                }
            }
        )
    )
    
    logger.info(f"Registered {len(strategy_library.categories[StrategyCategory.MEAN_REVERSION])} mean reversion strategies")


# Auto-register when module is imported
register_mean_reversion_strategies()


if __name__ == "__main__":
    async def test_mean_reversion_strategies():
        # Test strategy registration
        mean_reversion_strategies = strategy_library.get_strategies_by_category(StrategyCategory.MEAN_REVERSION)
        print(f"Registered mean reversion strategies: {len(mean_reversion_strategies)}")
        
        for strategy in mean_reversion_strategies:
            print(f"- {strategy.name} ({strategy.strategy_id})")
        
        # Test strategy creation
        try:
            from .base import StrategyConfig
            
            # Create Bollinger Bands strategy
            config = StrategyConfig(
                strategy_id="test_bb",
                strategy_type=StrategyType.MEAN_REVERSION,
                name="Test Bollinger Bands Strategy",
                description="Test Bollinger Bands mean reversion strategy",
                parameters={
                    "period": 20,
                    "std_dev": 2.0,
                    "signal_threshold": 0.6
                },
                risk_level=RiskLevel.MEDIUM,
                symbols=["AAPL", "GOOGL"]
            )
            
            strategy = strategy_library.create_strategy_instance("bollinger_bands", config)
            if strategy:
                print(f"\nSuccessfully created strategy: {strategy.config.name}")
            else:
                print("\nFailed to create strategy")
                
        except Exception as e:
            print(f"Error during testing: {e}")
    
    import asyncio
    asyncio.run(test_mean_reversion_strategies())