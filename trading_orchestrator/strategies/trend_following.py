"""
@file trend_following.py
@brief Trend Following Strategy Implementation

@details
This module implements a comprehensive trend following strategy using moving
average crossovers with momentum and volume confirmation. The strategy identifies
and trades market trends while providing robust risk management and signal filtering.

Strategy Logic:
1. Moving Average Crossover: Fast MA crossing above/below slow MA
2. Momentum Confirmation: Price momentum validates trend direction
3. Volume Confirmation: Volume supports trend sustainability
4. Risk Management: Stop loss and take profit levels
5. Signal Filtering: Minimum signal strength and quality thresholds

@author Trading Orchestrator System
@version 1.0
@date 2025-11-06

@warning
Trend following strategies can experience significant drawdowns during
market reversals. Always use appropriate position sizing and stop losses.

@note
This strategy is most effective in trending markets and may underperform
during periods of low volatility or market consolidation.

@see base.BaseStrategy for base class implementation
@see strategies.mean_reversion for complementary strategy approach
@see strategies.momentum for momentum-based variant

@par Strategy Parameters:
- fast_period: Fast moving average period (10-20 recommended)
- slow_period: Slow moving average period (25-50 recommended)
- momentum_period: Momentum calculation period (10-20 recommended)
- volume_threshold: Volume confirmation threshold (1.1-1.5 recommended)
- signal_threshold: Minimum signal strength (0.5-0.8 recommended)
- stop_loss: Stop loss percentage (0.015-0.03 recommended)
- take_profit: Take profit percentage (0.04-0.08 recommended)

@par Performance Characteristics:
- Best in: Strong trending markets
- Risk Level: Medium to High
- Expected Win Rate: 40-60%
- Average Hold Time: 1-5 trading days
- Maximum Drawdown: 10-20%
"""

from typing import Dict, Any, List, Optional
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
    BaseTimeSeriesStrategy
)
from trading.models import OrderSide


class TrendFollowingStrategy(BaseTimeSeriesStrategy):
    """
    Trend Following Strategy
    
    Uses moving average crossovers and momentum confirmation:
    1. Fast and slow moving averages
    2. Price momentum confirmation
    3. Volume confirmation
    4. Risk-adjusted position sizing
    
    Parameters:
    - fast_period: Fast moving average period (default: 10)
    - slow_period: Slow moving average period (default: 30)
    - momentum_period: Momentum calculation period (default: 14)
    - volume_threshold: Volume confirmation threshold (default: 1.2)
    - signal_threshold: Minimum signal strength (default: 0.6)
    - stop_loss: Stop loss percentage (default: 0.02)
    - take_profit: Take profit percentage (default: 0.06)
    """
    
    def __init__(self, config: StrategyConfig):
        # Validate required parameters
        required_params = ['fast_period', 'slow_period', 'momentum_period']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1h")
        
        # Extract parameters with defaults
        self.fast_period = int(config.parameters.get('fast_period', 10))
        self.slow_period = int(config.parameters.get('slow_period', 30))
        self.momentum_period = int(config.parameters.get('momentum_period', 14))
        self.volume_threshold = float(config.parameters.get('volume_threshold', 1.2))
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.6))
        self.stop_loss_pct = float(config.parameters.get('stop_loss', 0.02))
        self.take_profit_pct = float(config.parameters.get('take_profit', 0.06))
        
        # Cache for calculations
        self.last_signals: Dict[str, TradingSignal] = {}
        
        logger.info(f"Trend Following Strategy initialized with periods: fast={self.fast_period}, slow={self.slow_period}")
    
    async def generate_signals(self) -> List[TradingSignal]:
        """Generate trend following signals"""
        signals = []
        
        try:
            # Process each configured symbol
            for symbol in self.config.symbols:
                signal = await self._analyze_symbol(symbol)
                if signal:
                    signals.append(signal)
            
            logger.debug(f"Generated {len(signals)} trend following signals")
            
        except Exception as e:
            logger.error(f"Error generating trend following signals: {e}")
        
        return signals
    
    async def _analyze_symbol(self, symbol: str) -> Optional[TradingSignal]:
        """Analyze symbol for trend following signals"""
        try:
            # Get historical data
            data = await self._get_historical_data(symbol)
            if len(data) < self.slow_period + self.momentum_period:
                logger.debug(f"Insufficient data for {symbol}")
                return None
            
            # Calculate indicators
            prices = [Decimal(str(item['close'])) for item in data]
            volumes = [float(item.get('volume', 0)) for item in data]
            
            # Moving averages
            fast_ma = self._calculate_sma(prices, self.fast_period)
            slow_ma = self._calculate_sma(prices, self.slow_period)
            
            if not fast_ma or not slow_ma:
                return None
            
            # Current values
            current_price = prices[-1]
            current_fast_ma = fast_ma[-1]
            current_slow_ma = slow_ma[-1]
            
            # Momentum
            momentum = self._calculate_momentum(prices, self.momentum_period)
            
            # Volume confirmation
            avg_volume = sum(volumes[-20:]) / min(20, len(volumes))  # 20-period average
            recent_volume = volumes[-1]
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Trend analysis
            trend_strength, signal_type = self._analyze_trend(
                current_fast_ma, current_slow_ma, momentum, volume_ratio
            )
            
            # Generate signal if strong enough
            if trend_strength >= self.signal_threshold and signal_type in [SignalType.BUY, SignalType.SELL]:
                signal = await self._create_signal(
                    symbol, signal_type, trend_strength, current_price
                )
                
                if signal:
                    logger.info(f"Trend signal for {symbol}: {signal_type.value} (strength: {trend_strength:.2f})")
                    return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def _calculate_sma(self, prices: List[Decimal], period: int) -> List[Decimal]:
        """Calculate Simple Moving Average"""
        if len(prices) < period:
            return []
        
        sma = []
        for i in range(period - 1, len(prices)):
            avg = sum(prices[i - period + 1:i + 1]) / period
            sma.append(avg)
        
        return sma
    
    def _calculate_momentum(self, prices: List[Decimal], period: int) -> float:
        """Calculate price momentum as percentage change"""
        if len(prices) < period + 1:
            return 0.0
        
        current_price = float(prices[-1])
        previous_price = float(prices[-period - 1])
        
        if previous_price == 0:
            return 0.0
        
        momentum = (current_price - previous_price) / previous_price
        return momentum
    
    def _analyze_trend(
        self,
        fast_ma: Decimal,
        slow_ma: Decimal,
        momentum: float,
        volume_ratio: float
    ) -> tuple[float, SignalType]:
        """
        Analyze trend strength and direction
        
        Returns:
            Tuple of (strength, signal_type)
        """
        try:
            # MA crossover signals
            ma_diff_pct = float((fast_ma - slow_ma) / slow_ma)
            
            # Strong trend indicators
            strong_uptrend = (
                ma_diff_pct > 0.01 and      # Fast MA above slow MA by >1%
                momentum > 0.02 and         # Positive momentum >2%
                volume_ratio > self.volume_threshold
            )
            
            strong_downtrend = (
                ma_diff_pct < -0.01 and     # Fast MA below slow MA by >1%
                momentum < -0.02 and        # Negative momentum <-2%
                volume_ratio > self.volume_threshold
            )
            
            # Moderate trend signals
            moderate_uptrend = (
                ma_diff_pct > 0.005 and     # Fast MA above slow MA by >0.5%
                momentum > 0.01 and         # Positive momentum >1%
                volume_ratio > 1.0
            )
            
            moderate_downtrend = (
                ma_diff_pct < -0.005 and    # Fast MA below slow MA by >0.5%
                momentum < -0.01 and        # Negative momentum <-1%
                volume_ratio > 1.0
            )
            
            # Determine signal and strength
            if strong_uptrend:
                strength = min(1.0, abs(momentum) * 10 + volume_ratio * 0.2)
                return strength, SignalType.BUY
            
            elif strong_downtrend:
                strength = min(1.0, abs(momentum) * 10 + volume_ratio * 0.2)
                return strength, SignalType.SELL
            
            elif moderate_uptrend:
                strength = min(0.8, abs(momentum) * 5 + volume_ratio * 0.1)
                return strength, SignalType.BUY
            
            elif moderate_downtrend:
                strength = min(0.8, abs(momentum) * 5 + volume_ratio * 0.1)
                return strength, SignalType.SELL
            
            else:
                return 0.0, SignalType.HOLD
                
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            return 0.0, SignalType.HOLD
    
    async def _create_signal(
        self,
        symbol: str,
        signal_type: SignalType,
        strength: float,
        price: Decimal
    ) -> Optional[TradingSignal]:
        """Create a trading signal"""
        try:
            # Calculate position size based on strength and risk parameters
            position_size = self._calculate_position_size(strength)
            
            # Calculate stop loss and take profit
            stop_loss = None
            take_profit = None
            
            if signal_type == SignalType.BUY:
                stop_loss = price * (1 - self.stop_loss_pct)
                take_profit = price * (1 + self.take_profit_pct)
            elif signal_type == SignalType.SELL:
                stop_loss = price * (1 + self.stop_loss_pct)
                take_profit = price * (1 - self.take_profit_pct)
            
            signal = TradingSignal(
                signal_id=f"TF_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{int(strength*100)}",
                strategy_id=self.config.strategy_id,
                symbol=symbol,
                signal_type=signal_type,
                confidence=strength,
                strength=strength,
                price=price,
                quantity=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                time_horizon=timedelta(hours=4),  # Short to medium term
                metadata={
                    'strategy_type': 'trend_following',
                    'fast_period': self.fast_period,
                    'slow_period': self.slow_period,
                    'momentum': strength,
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error creating signal: {e}")
            return None
    
    def _calculate_position_size(self, strength: float) -> Decimal:
        """Calculate position size based on signal strength and risk parameters"""
        try:
            # Base position size (can be adjusted based on account size)
            base_size = self.config.max_position_size * Decimal('0.1')  # 10% of max position
            
            # Scale by signal strength
            adjusted_size = base_size * Decimal(strength)
            
            # Ensure within risk limits
            max_size = self.config.max_position_size
            final_size = min(adjusted_size, max_size)
            
            return final_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return self.config.max_position_size * Decimal('0.05')  # Conservative default
    
    async def _get_historical_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Get historical market data for symbol"""
        try:
            if self.context:
                # Request data from context
                data = await self.context.get_market_data(
                    symbol=symbol,
                    timeframe=self.timeframe
                )
                return data[-100:]  # Last 100 periods
            else:
                # Return mock data for testing
                return self._generate_mock_data(symbol)
                
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return []
    
    def _generate_mock_data(self, symbol: str, periods: int = 100) -> List[Dict[str, Any]]:
        """Generate mock data for testing"""
        import random
        
        base_price = 100.0
        data = []
        
        for i in range(periods):
            # Simulate price movement with trend
            trend_factor = 0.001 * (i / periods)  # Slight upward trend
            price_change = random.gauss(0, 0.02) + trend_factor
            base_price *= (1 + price_change)
            
            volume = random.randint(100000, 1000000)
            
            data.append({
                'timestamp': datetime.utcnow() - timedelta(minutes=periods-i),
                'open': base_price * random.uniform(0.99, 1.01),
                'high': base_price * random.uniform(1.01, 1.03),
                'low': base_price * random.uniform(0.97, 0.99),
                'close': base_price,
                'volume': volume
            })
        
        return data
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        """Validate trend following signal"""
        try:
            # Check basic validation
            if not signal.symbol or signal.quantity <= 0:
                return False
            
            # Check if signal is not too old (expired)
            if signal.expires_at and datetime.utcnow() > signal.expires_at:
                return False
            
            # Check signal strength threshold
            if signal.strength < self.signal_threshold:
                return False
            
            # Check if we already have a signal for this symbol recently
            recent_signals = [
                s for s in self.last_signals.values()
                if s.symbol == signal.symbol and 
                (datetime.utcnow() - s.created_at).total_seconds() < 3600  # 1 hour
            ]
            
            if recent_signals and signal.signal_type == recent_signals[-1].signal_type:
                logger.debug(f"Recent {signal.signal_type.value} signal exists for {signal.symbol}")
                return False
            
            # Additional risk checks could be added here
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating signal: {e}")
            return False
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get detailed strategy information"""
        return {
            'strategy_name': 'Trend Following Strategy',
            'description': 'Moving average crossover with momentum confirmation',
            'parameters': {
                'fast_period': self.fast_period,
                'slow_period': self.slow_period,
                'momentum_period': self.momentum_period,
                'volume_threshold': self.volume_threshold,
                'signal_threshold': self.signal_threshold,
                'stop_loss_pct': self.stop_loss_pct,
                'take_profit_pct': self.take_profit_pct
            },
            'indicators_used': ['Simple Moving Average', 'Momentum', 'Volume'],
            'timeframe': self.timeframe,
            'risk_level': self.config.risk_level.value,
            'position_sizing': 'Strength-based with risk limits'
        }


# Factory function to create trend following strategy
def create_trend_following_strategy(
    strategy_id: str,
    symbols: List[str],
    fast_period: int = 10,
    slow_period: int = 30,
    **kwargs
) -> TrendFollowingStrategy:
    """Factory function to create trend following strategy"""
    
    config = StrategyConfig(
        strategy_id=strategy_id,
        strategy_type=StrategyType.TREND_FOLLOWING,
        name="Trend Following Strategy",
        description="Moving average crossover strategy with momentum and volume confirmation",
        parameters={
            'fast_period': fast_period,
            'slow_period': slow_period,
            'momentum_period': kwargs.get('momentum_period', 14),
            'volume_threshold': kwargs.get('volume_threshold', 1.2),
            'signal_threshold': kwargs.get('signal_threshold', 0.6),
            'stop_loss': kwargs.get('stop_loss', 0.02),
            'take_profit': kwargs.get('take_profit', 0.06)
        },
        risk_level=RiskLevel.MEDIUM,
        symbols=symbols,
        max_position_size=Decimal(kwargs.get('max_position_size', '50000')),
        max_daily_loss=Decimal(kwargs.get('max_daily_loss', '5000'))
    )
    
    return TrendFollowingStrategy(config)


# Example usage and testing
if __name__ == "__main__":
    async def test_trend_following_strategy():
        # Create strategy
        strategy = create_trend_following_strategy(
            strategy_id="tf_001",
            symbols=['AAPL', 'GOOGL'],
            fast_period=10,
            slow_period=30
        )
        
        # Mock context for testing
        class MockContext:
            async def get_market_data(self, symbol, timeframe):
                return strategy._generate_mock_data(symbol)
        
        strategy.set_context(MockContext())
        
        # Generate signals
        signals = await strategy.generate_signals()
        
        print(f"Generated {len(signals)} signals:")
        for signal in signals:
            print(f"  {signal.signal_type.value} {signal.symbol} @ {signal.price} (strength: {signal.strength:.2f})")
        
        # Get strategy info
        info = strategy.get_strategy_info()
        print("\nStrategy Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    # Run test
    import asyncio
    asyncio.run(test_trend_following_strategy())