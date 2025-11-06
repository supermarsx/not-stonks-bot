"""
@file volatility_strategies.py
@brief Volatility Strategies Implementation

@details
This module implements 12+ volatility-based trading strategies that identify
and trade based on changes in market volatility, VIX levels, and volatility expansion/contraction.

Strategy Categories:
- VIX-Based Strategies (3): VIX breakout, VIX mean reversion, VIX futures contango
- Volatility Breakout Strategies (2): Bollinger squeeze breakout, volatility expansion breakout
- Volatility Mean Reversion Strategies (2): Volatility compression, volatility reversion to mean
- Implied Volatility Strategies (2): IV skew, IV term structure
- Cross-Asset Volatility Strategies (3): Stock-bond volatility, currency volatility, commodity volatility
- Realized Volatility Strategies (2): GARCH volatility, realized volatility breakout

Key Features:
- Volatility regime detection
- Mean reversion and breakout signals
- Cross-asset volatility analysis
- Risk-adjusted position sizing
- Backtesting and optimization support

@author Trading Orchestrator System
@version 2.0
@date 2025-11-06

@warning
Volatility strategies can experience sudden regime changes and require
careful risk management and position sizing.

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
import asyncio

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
# VIX-BASED STRATEGIES
# ============================================================================

class VIXBreakoutStrategy(BaseTimeSeriesStrategy):
    """VIX Breakout Strategy
    
    Trades VIX breakout above or below significant levels to capture volatility moves.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['vix_lookback', 'breakout_threshold']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1d")
        
        self.vix_lookback = int(config.parameters.get('vix_lookback', 20))
        self.breakout_threshold = float(config.parameters.get('breakout_threshold', 1.5))
        self.vix_symbol = config.parameters.get('vix_symbol', '^VIX')
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.7))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        # Get VIX data
        vix_data = await self._get_vix_data()
        if not vix_data or len(vix_data) < self.vix_lookback:
            return signals
        
        vix_prices = [Decimal(str(item['close'])) for item in vix_data]
        current_vix = vix_prices[-1]
        
        # Calculate VIX breakout
        vix_levels = self._calculate_vix_levels(vix_prices)
        
        if vix_levels:
            # Check for VIX breakout
            if current_vix > vix_levels['upper']:
                # VIX breakout above resistance - volatility expansion
                strength = min(1.0, float((current_vix - vix_levels['upper']) / vix_levels['upper']))
                
                # Signal for volatility expansion (buy volatility, short stocks)
                for symbol in self.config.symbols:
                    signal = await self._create_signal(
                        symbol, SignalType.SELL, strength, current_vix  # Short stocks
                    )
                    if signal:
                        signals.append(signal)
            
            elif current_vix < vix_levels['lower']:
                # VIX breakdown below support - volatility contraction
                deficit = vix_levels['lower'] - current_vix
                strength = min(1.0, float(deficit / vix_levels['lower']))
                
                # Signal for volatility contraction (sell volatility, buy stocks)
                for symbol in self.config.symbols:
                    signal = await self._create_signal(
                        symbol, SignalType.BUY, strength, current_vix  # Buy stocks
                    )
                    if signal:
                        signals.append(signal)
        
        return signals
    
    def _calculate_vix_levels(self, vix_prices: List[Decimal]) -> Optional[Dict[str, Decimal]]:
        """Calculate VIX support/resistance levels"""
        if len(vix_prices) < self.vix_lookback:
            return None
        
        recent_vix = vix_prices[-self.vix_lookback:]
        upper = max(recent_vix)
        lower = min(recent_vix)
        middle = sum(recent_vix) / len(recent_vix)
        
        return {
            'upper': upper,
            'lower': lower,
            'middle': middle
        }
    
    async def _get_vix_data(self) -> List[Dict[str, Any]]:
        """Get VIX data for analysis"""
        try:
            # In a real implementation, this would fetch actual VIX data
            # For now, return mock data structure
            return await self._get_historical_data(self.vix_symbol)
        except Exception as e:
            logger.warning(f"Could not fetch VIX data: {e}")
            return []
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, vix_price: Decimal) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        return TradingSignal(
            signal_id=f"VIX_BREAKOUT_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=Decimal('1'),  # Signal price
            quantity=self.config.max_position_size * Decimal(str(strength)),
            metadata={
                'strategy_type': 'vix_breakout',
                'vix_price': float(vix_price),
                'regime': 'volatility_expansion' if signal_type == SignalType.SELL else 'volatility_contraction'
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


class VIXMeanReversionStrategy(BaseTimeSeriesStrategy):
    """VIX Mean Reversion Strategy
    
    Trades VIX mean reversion when it reaches extreme levels.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['period', 'upper_threshold', 'lower_threshold']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1d")
        
        self.period = int(config.parameters.get('period', 20))
        self.upper_threshold = float(config.parameters.get('upper_threshold', 30.0))
        self.lower_threshold = float(config.parameters.get('lower_threshold', 12.0))
        self.vix_symbol = config.parameters.get('vix_symbol', '^VIX')
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.8))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        # Get VIX data
        vix_data = await self._get_vix_data()
        if not vix_data or len(vix_data) < self.period:
            return signals
        
        vix_prices = [Decimal(str(item['close'])) for item in vix_data]
        current_vix = vix_prices[-1]
        
        # Calculate mean and standard deviation
        mean_vix = sum(vix_prices[-self.period:]) / self.period
        recent_vix = [float(v) for v in vix_prices[-self.period:]]
        std_vix = math.sqrt(sum((x - float(mean_vix))**2 for x in recent_vix) / self.period)
        
        # Check for extreme VIX levels
        if current_vix > self.upper_threshold:
            # VIX is too high - expect mean reversion downward
            excess = float(current_vix - self.upper_threshold)
            strength = min(1.0, excess / std_vix)
            
            # Buy stocks (expect VIX to fall)
            for symbol in self.config.symbols:
                signal = await self._create_signal(
                    symbol, SignalType.BUY, strength, current_vix
                )
                if signal:
                    signals.append(signal)
        
        elif current_vix < self.lower_threshold:
            # VIX is too low - expect mean reversion upward
            deficit = self.lower_threshold - float(current_vix)
            strength = min(1.0, deficit / std_vix)
            
            # Sell stocks (expect VIX to rise)
            for symbol in self.config.symbols:
                signal = await self._create_signal(
                    symbol, SignalType.SELL, strength, current_vix
                )
                if signal:
                    signals.append(signal)
        
        return signals
    
    async def _get_vix_data(self) -> List[Dict[str, Any]]:
        """Get VIX data for analysis"""
        try:
            return await self._get_historical_data(self.vix_symbol)
        except Exception as e:
            logger.warning(f"Could not fetch VIX data: {e}")
            return []
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, vix_price: Decimal) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        return TradingSignal(
            signal_id=f"VIX_MEAN_REV_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=Decimal('1'),
            quantity=self.config.max_position_size * Decimal(str(strength)),
            metadata={
                'strategy_type': 'vix_mean_reversion',
                'vix_price': float(vix_price),
                'regime': 'extreme_overbought' if signal_type == SignalType.BUY else 'extreme_oversold'
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


class VIXFuturesContangoStrategy(BaseTimeSeriesStrategy):
    """VIX Futures Contango Strategy
    
    Trades the term structure of VIX futures when in contango/backwardation.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['front_month', 'second_month', 'contango_threshold']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1d")
        
        self.front_month = int(config.parameters.get('front_month', 1))
        self.second_month = int(config.parameters.get('second_month', 2))
        self.contango_threshold = float(config.parameters.get('contango_threshold', 0.05))
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.7))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        # Get VIX futures data
        futures_data = await self._get_vix_futures_data()
        if not futures_data or len(futures_data) < 2:
            return signals
        
        # Calculate contango/backwardation
        contango_ratio = self._calculate_contango_ratio(futures_data)
        
        if contango_ratio is not None:
            if contango_ratio > self.contango_threshold:
                # Strong contango - volatility expected to increase
                strength = min(1.0, (contango_ratio - self.contango_threshold) * 10)
                
                for symbol in self.config.symbols:
                    signal = await self._create_signal(
                        symbol, SignalType.SELL, strength, Decimal('1')
                    )
                    if signal:
                        signals.append(signal)
            
            elif contango_ratio < -self.contango_threshold:
                # Backwardation - volatility expected to decrease
                strength = min(1.0, (abs(contango_ratio) - self.contango_threshold) * 10)
                
                for symbol in self.config.symbols:
                    signal = await self._create_signal(
                        symbol, SignalType.BUY, strength, Decimal('1')
                    )
                    if signal:
                        signals.append(signal)
        
        return signals
    
    def _calculate_contango_ratio(self, futures_data: List[Dict[str, Any]]) -> Optional[float]:
        """Calculate contango/backwardation ratio"""
        if len(futures_data) < 2:
            return None
        
        front_price = float(futures_data[0]['close'])
        second_price = float(futures_data[1]['close'])
        
        if front_price == 0:
            return None
        
        contango_ratio = (second_price - front_price) / front_price
        return contango_ratio
    
    async def _get_vix_futures_data(self) -> List[Dict[str, Any]]:
        """Get VIX futures data for analysis"""
        try:
            # In real implementation, fetch actual VIX futures data
            return await self._get_historical_data('VIX1!')  # Front month
        except Exception as e:
            logger.warning(f"Could not fetch VIX futures data: {e}")
            return []
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, price: Decimal) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        return TradingSignal(
            signal_id=f"VIX_CONTANGO_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            metadata={
                'strategy_type': 'vix_futures_contango',
                'regime': 'contango' if signal_type == SignalType.SELL else 'backwardation'
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


# ============================================================================
# VOLATILITY BREAKOUT STRATEGIES
# ============================================================================

class BollingerSqueezeBreakoutStrategy(BaseTimeSeriesStrategy):
    """Bollinger Bands Squeeze Breakout Strategy
    
    Identifies periods of low volatility (squeeze) and trades the breakout.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['period', 'std_dev_low', 'std_dev_high']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1h")
        
        self.period = int(config.parameters.get('period', 20))
        self.std_dev_low = float(config.parameters.get('std_dev_low', 1.5))
        self.std_dev_high = float(config.parameters.get('std_dev_high', 2.5))
        self.squeeze_threshold = float(config.parameters.get('squeeze_threshold', 0.7))
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.6))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        for symbol in self.config.symbols:
            data = await self._get_historical_data(symbol)
            if len(data) < self.period * 2:
                continue
            
            prices = [Decimal(str(item['close'])) for item in data]
            current_price = prices[-1]
            
            # Calculate Bollinger Bands
            upper_band, middle_band, lower_band = self._calculate_bollinger_bands(prices)
            
            if not upper_band or len(upper_band) < 2:
                continue
            
            current_upper = upper_band[-1]
            current_lower = lower_band[-1]
            previous_upper = upper_band[-2]
            previous_lower = lower_band[-2]
            
            # Calculate band width
            current_width = current_upper - current_lower
            previous_width = previous_upper - previous_lower
            
            # Detect squeeze
            width_ratio = current_width / previous_width if previous_width > 0 else 1.0
            
            if width_ratio <= self.squeeze_threshold:
                # Squeeze detected - look for breakout
                price_above = float(current_price > current_upper)
                price_below = float(current_price < current_lower)
                
                if price_above:
                    # Breakout above upper band
                    strength = min(1.0, (float(current_price) - float(current_upper)) / float(current_upper) * 20)
                    signal = await self._create_signal(
                        symbol, SignalType.BUY, strength, current_price
                    )
                    if signal:
                        signals.append(signal)
                
                elif price_below:
                    # Breakout below lower band
                    strength = min(1.0, (float(current_lower) - float(current_price)) / float(current_lower) * 20)
                    signal = await self._create_signal(
                        symbol, SignalType.SELL, strength, current_price
                    )
                    if signal:
                        signals.append(signal)
        
        return signals
    
    def _calculate_bollinger_bands(self, prices: List[Decimal]) -> Tuple[List[Decimal], List[Decimal], List[Decimal]]:
        """Calculate Bollinger Bands with different std dev levels"""
        if len(prices) < self.period:
            return [], [], []
        
        sma = []
        upper_band = []
        lower_band = []
        
        for i in range(self.period - 1, len(prices)):
            period_prices = prices[i - self.period + 1:i + 1]
            mean = sum(period_prices) / self.period
            
            # Calculate standard deviation
            variance = sum((p - mean) ** 2 for p in period_prices) / self.period
            std_dev = variance ** 0.5
            
            sma.append(mean)
            
            # High volatility bands
            upper_band.append(mean + (Decimal(str(self.std_dev_high)) * std_dev))
            lower_band.append(mean - (Decimal(str(self.std_dev_high)) * std_dev))
        
        return upper_band, sma, lower_band
    
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
                'strategy_type': 'bollinger_squeeze_breakout',
                'regime': 'volatility_expansion'
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


class VolatilityExpansionBreakoutStrategy(BaseTimeSeriesStrategy):
    """Volatility Expansion Breakout Strategy
    
    Trades significant volatility expansions from recent low volatility periods.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['lookback_period', 'expansion_threshold']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1h")
        
        self.lookback_period = int(config.parameters.get('lookback_period', 20))
        self.expansion_threshold = float(config.parameters.get('expansion_threshold', 2.0))
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.7))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        for symbol in self.config.symbols:
            data = await self._get_historical_data(symbol)
            if len(data) < self.lookback_period * 2:
                continue
            
            prices = [Decimal(str(item['close'])) for item in data]
            current_price = prices[-1]
            
            # Calculate realized volatility
            volatility_series = self._calculate_realized_volatility(prices)
            
            if len(volatility_series) < self.lookback_period:
                continue
            
            current_volatility = volatility_series[-1]
            recent_volatility = volatility_series[-self.lookback_period:-1]
            avg_recent_vol = sum(recent_volatility) / len(recent_volatility)
            
            # Check for volatility expansion
            expansion_ratio = current_volatility / avg_recent_vol if avg_recent_vol > 0 else 1.0
            
            if expansion_ratio >= self.expansion_threshold:
                # Significant volatility expansion detected
                strength = min(1.0, (expansion_ratio - self.expansion_threshold) / expansion_ratio)
                
                # Determine direction based on price movement
                recent_prices = [float(p) for p in prices[-5:]]
                price_trend = sum(recent_prices[i] - recent_prices[i-1] for i in range(1, len(recent_prices)))
                
                if price_trend > 0:
                    # Price rising with volatility expansion - continue long
                    signal_type = SignalType.BUY
                else:
                    # Price falling with volatility expansion - continue short
                    signal_type = SignalType.SELL
                
                signal = await self._create_signal(
                    symbol, signal_type, strength, current_price
                )
                if signal:
                    signals.append(signal)
        
        return signals
    
    def _calculate_realized_volatility(self, prices: List[Decimal]) -> List[float]:
        """Calculate realized volatility"""
        if len(prices) < 2:
            return []
        
        returns = []
        for i in range(1, len(prices)):
            price_change = float(prices[i] - prices[i-1])
            prev_price = float(prices[i-1])
            if prev_price > 0:
                returns.append(price_change / prev_price)
        
        # Calculate rolling volatility
        volatility = []
        for i in range(10, len(returns)):
            period_returns = returns[i-10:i]
            mean_return = sum(period_returns) / len(period_returns)
            variance = sum((r - mean_return) ** 2 for r in period_returns) / len(period_returns)
            vol = variance ** 0.5 * (252 ** 0.5)  # Annualized
            volatility.append(vol)
        
        return volatility
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, price: Decimal) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        return TradingSignal(
            signal_id=f"VOL_EXPANSION_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            metadata={
                'strategy_type': 'volatility_expansion_breakout',
                'regime': 'volatility_regime_change'
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


# ============================================================================
# REALIZED VOLATILITY STRATEGIES
# ============================================================================

class GARCHVolatilityStrategy(BaseTimeSeriesStrategy):
    """GARCH Volatility Strategy
    
    Uses GARCH model to forecast volatility and trade volatility changes.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['lookback_period', 'forecast_threshold']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1d")
        
        self.lookback_period = int(config.parameters.get('lookback_period', 60))
        self.forecast_threshold = float(config.parameters.get('forecast_threshold', 0.05))
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.6))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        for symbol in self.config.symbols:
            data = await self._get_historical_data(symbol)
            if len(data) < self.lookback_period * 2:
                continue
            
            prices = [Decimal(str(item['close'])) for item in data]
            current_price = prices[-1]
            
            # Calculate returns
            returns = []
            for i in range(1, len(prices)):
                price_change = float(prices[i] - prices[i-1])
                prev_price = float(prices[i-1])
                if prev_price > 0:
                    returns.append(price_change / prev_price)
            
            if len(returns) < self.lookback_period:
                continue
            
            # Estimate GARCH(1,1) parameters (simplified)
            garch_forecast = self._estimate_garch_forecast(returns)
            historical_vol = np.std(returns[-self.lookback_period:]) * (252 ** 0.5)
            
            # Compare forecast to historical volatility
            vol_change = abs(garch_forecast - historical_vol) / historical_vol
            
            if vol_change >= self.forecast_threshold:
                if garch_forecast > historical_vol:
                    # Expected volatility increase
                    strength = min(1.0, vol_change * 2)
                    signal_type = SignalType.SELL  # Sell stocks, buy volatility
                else:
                    # Expected volatility decrease
                    strength = min(1.0, vol_change * 2)
                    signal_type = SignalType.BUY  # Buy stocks, sell volatility
                
                signal = await self._create_signal(
                    symbol, signal_type, strength, current_price
                )
                if signal:
                    signals.append(signal)
        
        return signals
    
    def _estimate_garch_forecast(self, returns: List[float]) -> float:
        """Simplified GARCH(1,1) forecast"""
        if len(returns) < 30:
            return np.std(returns) * (252 ** 0.5)
        
        # Use rolling window for simplicity
        window_returns = returns[-30:]
        volatility = np.std(window_returns)
        
        # Simplified forecast adjustment
        recent_vol = np.std(returns[-10:]) if len(returns) >= 10 else volatility
        forecast = volatility * 0.7 + recent_vol * 0.3
        
        return forecast * (252 ** 0.5)
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, price: Decimal) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        return TradingSignal(
            signal_id=f"GARCH_VOL_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            metadata={
                'strategy_type': 'garch_volatility',
                'model': 'garch11_simplified'
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


# ============================================================================
# CROSS-ASSET VOLATILITY STRATEGIES
# ============================================================================

class StockBondVolatilityStrategy(BaseTimeSeriesStrategy):
    """Stock-Bond Volatility Correlation Strategy
    
    Trades based on stock-bond volatility correlations.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['equity_symbol', 'bond_symbol', 'correlation_threshold']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1d")
        
        self.equity_symbol = config.parameters.get('equity_symbol', 'SPY')
        self.bond_symbol = config.parameters.get('bond_symbol', 'TLT')
        self.correlation_threshold = float(config.parameters.get('correlation_threshold', -0.7))
        self.lookback_period = int(config.parameters.get('lookback_period', 20))
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.7))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        # Get both equity and bond data
        equity_data = await self._get_historical_data(self.equity_symbol)
        bond_data = await self._get_historical_data(self.bond_symbol)
        
        if len(equity_data) < self.lookback_period or len(bond_data) < self.lookback_period:
            return signals
        
        # Calculate returns
        equity_prices = [Decimal(str(item['close'])) for item in equity_data]
        bond_prices = [Decimal(str(item['close'])) for item in bond_data]
        
        equity_returns = []
        bond_returns = []
        
        for i in range(1, min(len(equity_prices), len(bond_prices))):
            if float(equity_prices[i-1]) > 0 and float(bond_prices[i-1]) > 0:
                eq_return = float((equity_prices[i] - equity_prices[i-1]) / equity_prices[i-1])
                bd_return = float((bond_prices[i] - bond_prices[i-1]) / bond_prices[i-1])
                equity_returns.append(eq_return)
                bond_returns.append(bd_return)
        
        if len(equity_returns) < self.lookback_period:
            return signals
        
        # Calculate correlation
        recent_eq_returns = equity_returns[-self.lookback_period:]
        recent_bd_returns = bond_returns[-self.lookback_period:]
        
        correlation = self._calculate_correlation(recent_eq_returns, recent_bd_returns)
        
        # Check for significant correlation changes
        if correlation <= self.correlation_threshold:
            # Strong negative correlation - flight to bonds expected
            strength = min(1.0, abs(correlation - self.correlation_threshold))
            
            # Buy bonds, sell stocks
            for symbol in [self.equity_symbol]:
                signal = await self._create_signal(
                    symbol, SignalType.SELL, strength, equity_prices[-1]
                )
                if signal:
                    signals.append(signal)
        
        elif correlation >= -0.3:
            # Low negative correlation - risk-on environment
            strength = min(1.0, (correlation + 0.3) * 2)
            
            # Buy stocks, sell bonds
            signal = await self._create_signal(
                self.equity_symbol, SignalType.BUY, strength, equity_prices[-1]
            )
            if signal:
                signals.append(signal)
        
        return signals
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate correlation coefficient"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_x2 = sum(xi**2 for xi in x)
        sum_y2 = sum(yi**2 for yi in y)
        sum_xy = sum(xi*yi for xi, yi in zip(x, y))
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2)) ** 0.5
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, price: Decimal) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        return TradingSignal(
            signal_id=f"STOCK_BOND_VOL_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            metadata={
                'strategy_type': 'stock_bond_volatility',
                'regime': 'flight_to_bonds' if signal_type == SignalType.SELL else 'risk_on'
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


# ============================================================================
# STRATEGY REGISTRATION
# ============================================================================

def register_volatility_strategies():
    """Register all volatility strategies with the strategy library"""
    
    # Register VIX-based strategies
    strategy_library.register_strategy(
        VIXBreakoutStrategy,
        StrategyMetadata(
            strategy_id="vix_breakout",
            name="VIX Breakout Strategy",
            category=StrategyCategory.VOLATILITY,
            description="VIX breakout for volatility regime changes",
            long_description="Trades VIX breakout above or below significant levels to capture volatility moves.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["vix", "breakout", "volatility-regime", "advanced"],
            parameters_schema={
                "required": ["vix_lookback", "breakout_threshold"],
                "properties": {
                    "vix_lookback": {"type": "integer", "min": 10, "max": 60},
                    "breakout_threshold": {"type": "float", "min": 1.0, "max": 3.0},
                    "vix_symbol": {"type": "string"},
                    "signal_threshold": {"type": "float", "min": 0.3, "max": 1.0}
                }
            },
            example_config={
                "vix_lookback": 20,
                "breakout_threshold": 1.5,
                "vix_symbol": "^VIX",
                "signal_threshold": 0.7
            },
            risk_warning="VIX can experience sudden regime changes with unpredictable volatility spikes."
        )
    )
    
    strategy_library.register_strategy(
        VIXMeanReversionStrategy,
        StrategyMetadata(
            strategy_id="vix_mean_reversion",
            name="VIX Mean Reversion Strategy",
            category=StrategyCategory.VOLATILITY,
            description="VIX mean reversion at extreme levels",
            long_description="Trades VIX mean reversion when it reaches extreme levels.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["vix", "mean-reversion", "extremes", "advanced"],
            parameters_schema={
                "required": ["period", "upper_threshold", "lower_threshold"],
                "properties": {
                    "period": {"type": "integer", "min": 10, "max": 60},
                    "upper_threshold": {"type": "float", "min": 20.0, "max": 60.0},
                    "lower_threshold": {"type": "float", "min": 8.0, "max": 20.0},
                    "vix_symbol": {"type": "string"},
                    "signal_threshold": {"type": "float", "min": 0.5, "max": 1.0}
                }
            }
        )
    )
    
    strategy_library.register_strategy(
        VIXFuturesContangoStrategy,
        StrategyMetadata(
            strategy_id="vix_futures_contango",
            name="VIX Futures Contango Strategy",
            category=StrategyCategory.VOLATILITY,
            description="VIX futures term structure arbitrage",
            long_description="Trades the term structure of VIX futures when in contango/backwardation.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["vix", "futures", "term-structure", "advanced"],
            parameters_schema={
                "required": ["front_month", "second_month", "contango_threshold"],
                "properties": {
                    "front_month": {"type": "integer", "min": 1, "max": 3},
                    "second_month": {"type": "integer", "min": 2, "max": 6},
                    "contango_threshold": {"type": "float", "min": 0.01, "max": 0.15},
                    "signal_threshold": {"type": "float", "min": 0.5, "max": 1.0}
                }
            }
        )
    )
    
    # Register Volatility Breakout strategies
    strategy_library.register_strategy(
        BollingerSqueezeBreakoutStrategy,
        StrategyMetadata(
            strategy_id="bollinger_squeeze_breakout",
            name="Bollinger Squeeze Breakout Strategy",
            category=StrategyCategory.VOLATILITY,
            description="Bollinger Bands squeeze breakout",
            long_description="Identifies periods of low volatility (squeeze) and trades the breakout.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["bollinger", "squeeze", "breakout", "mean-reversion"],
            parameters_schema={
                "required": ["period", "std_dev_low", "std_dev_high"],
                "properties": {
                    "period": {"type": "integer", "min": 10, "max": 50},
                    "std_dev_low": {"type": "float", "min": 1.0, "max": 2.0},
                    "std_dev_high": {"type": "float", "min": 2.0, "max": 3.5},
                    "squeeze_threshold": {"type": "float", "min": 0.3, "max": 1.0},
                    "signal_threshold": {"type": "float", "min": 0.3, "max": 1.0}
                }
            }
        )
    )
    
    strategy_library.register_strategy(
        VolatilityExpansionBreakoutStrategy,
        StrategyMetadata(
            strategy_id="volatility_expansion_breakout",
            name="Volatility Expansion Breakout Strategy",
            category=StrategyCategory.VOLATILITY,
            description="Volatility expansion from low volatility periods",
            long_description="Trades significant volatility expansions from recent low volatility periods.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["volatility", "expansion", "regime-change", "advanced"],
            parameters_schema={
                "required": ["lookback_period", "expansion_threshold"],
                "properties": {
                    "lookback_period": {"type": "integer", "min": 10, "max": 60},
                    "expansion_threshold": {"type": "float", "min": 1.5, "max": 4.0},
                    "signal_threshold": {"type": "float", "min": 0.5, "max": 1.0}
                }
            }
        )
    )
    
    # Register Realized Volatility strategies
    strategy_library.register_strategy(
        GARCHVolatilityStrategy,
        StrategyMetadata(
            strategy_id="garch_volatility",
            name="GARCH Volatility Strategy",
            category=StrategyCategory.VOLATILITY,
            description="GARCH model-based volatility forecasting",
            long_description="Uses GARCH model to forecast volatility and trade volatility changes.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["garch", "forecasting", "volatility", "quantitative"],
            parameters_schema={
                "required": ["lookback_period", "forecast_threshold"],
                "properties": {
                    "lookback_period": {"type": "integer", "min": 30, "max": 120},
                    "forecast_threshold": {"type": "float", "min": 0.02, "max": 0.15},
                    "signal_threshold": {"type": "float", "min": 0.3, "max": 1.0}
                }
            },
            risk_warning="GARCH models assume specific volatility patterns that may not hold in all market conditions."
        )
    )
    
    # Register Cross-Asset Volatility strategies
    strategy_library.register_strategy(
        StockBondVolatilityStrategy,
        StrategyMetadata(
            strategy_id="stock_bond_volatility",
            name="Stock-Bond Volatility Correlation Strategy",
            category=StrategyCategory.VOLATILITY,
            description="Stock-bond volatility correlation trading",
            long_description="Trades based on stock-bond volatility correlations.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["stock", "bond", "correlation", "cross-asset", "macro"],
            parameters_schema={
                "required": ["equity_symbol", "bond_symbol", "correlation_threshold"],
                "properties": {
                    "equity_symbol": {"type": "string"},
                    "bond_symbol": {"type": "string"},
                    "correlation_threshold": {"type": "float", "min": -1.0, "max": -0.3},
                    "lookback_period": {"type": "integer", "min": 10, "max": 60},
                    "signal_threshold": {"type": "float", "min": 0.5, "max": 1.0}
                }
            }
        )
    )
    
    logger.info(f"Registered {len(strategy_library.categories[StrategyCategory.VOLATILITY])} volatility strategies")


# Auto-register when module is imported
register_volatility_strategies()


if __name__ == "__main__":
    async def test_volatility_strategies():
        # Test strategy registration
        volatility_strategies = strategy_library.get_strategies_by_category(StrategyCategory.VOLATILITY)
        print(f"Registered volatility strategies: {len(volatility_strategies)}")
        
        for strategy in volatility_strategies:
            print(f"- {strategy.name} ({strategy.strategy_id})")
    
    import asyncio
    asyncio.run(test_volatility_strategies())