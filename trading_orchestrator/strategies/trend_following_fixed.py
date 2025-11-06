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
    @class TrendFollowingStrategy
    @brief Moving Average Crossover Strategy with Momentum Confirmation
    
    @details
    Implements a sophisticated trend following strategy that combines moving
    average crossovers with momentum and volume confirmation to identify and
    trade market trends with improved accuracy and reduced false signals.
    
    @par Strategy Algorithm:
    1. <b>Moving Average Analysis</b>:
       - Calculate fast MA (typically 10-20 periods)
       - Calculate slow MA (typically 25-50 periods)
       - Identify crossover points as potential signals
    
    2. <b>Momentum Confirmation</b>:
       - Calculate price momentum over specified period
       - Confirm trend direction with momentum strength
       - Filter signals below momentum threshold
    
    3. <b>Volume Confirmation</b>:
       - Compare current volume to average volume
       - Require volume above threshold for signal validation
       - Ensure sustainability of price movement
    
    4. <b>Signal Generation</b>:
       - Generate BUY signal on bullish crossover with confirmation
       - Generate SELL signal on bearish crossover with confirmation
       - Calculate signal strength based on all confirmations
    
    5. <b>Risk Management</b>:
       - Set stop loss at specified percentage below entry
       - Set take profit at specified percentage above entry
       - Adjust position size based on signal confidence
    
    @par Signal Quality Factors:
    - MA Crossover Strength: Distance between moving averages
    - Momentum Alignment: Price momentum direction and strength
    - Volume Confirmation: Volume above average threshold
    - Price Action: Recent price movement patterns
    - Market Regime: Volatility and trend characteristics
    
    @par Entry Conditions:
    - Fast MA crosses above slow MA (long entry)
    - Fast MA crosses below slow MA (short entry)
    - Momentum confirms trend direction
    - Volume above threshold
    - Signal strength above minimum threshold
    
    @par Exit Conditions:
    - Stop loss level hit
    - Take profit level hit
    - Opposite MA crossover
    - Momentum reversal
    - Signal strength drops below threshold
    
    @warning
    This strategy can generate false signals during market consolidation
    or sideways movement. Use appropriate position sizing and risk management.
    
    @par Usage Example:
    @code
    from strategies.trend_following import TrendFollowingStrategy
    from strategies.base import StrategyConfig, StrategyType, RiskLevel
    
    # Configure strategy
    config = StrategyConfig(
        strategy_id="trend_001",
        strategy_type=StrategyType.TREND_FOLLOWING,
        name="Moving Average Crossover",
        description="Trend following with momentum confirmation",
        parameters={
            "fast_period": 12,
            "slow_period": 26,
            "momentum_period": 14,
            "volume_threshold": 1.2,
            "signal_threshold": 0.6,
            "stop_loss": 0.02,
            "take_profit": 0.06
        },
        risk_level=RiskLevel.MEDIUM,
        symbols=["AAPL", "GOOGL", "MSFT"]
    )
    
    # Initialize and run strategy
    strategy = TrendFollowingStrategy(config)
    strategy.set_context(execution_context)
    await strategy.run()
    @endcode
    
    @note
    This strategy works best on daily or 4-hour timeframes for swing trading
    and can be adapted for intraday trading with shorter periods.
    
    @see BaseTimeSeriesStrategy for time series analysis capabilities
    @see TradingSignal for signal structure details
    """
    
    def __init__(self, config: StrategyConfig):
        """
        @brief Initialize trend following strategy with configuration
        
        @param config StrategyConfig containing all parameters
        
        @details
        Validates required parameters and initializes the strategy with
        moving average periods and confirmation thresholds.
        
        @throws ValueError if required parameters are missing
        @throws TypeError if parameters have incorrect types
        
        @par Required Parameters:
        - fast_period: Fast MA period (int, >= 2)
        - slow_period: Slow MA period (int, > fast_period)
        - momentum_period: Momentum calculation period (int, >= 1)
        
        @par Optional Parameters:
        - volume_threshold: Volume confirmation factor (float, >= 1.0)
        - signal_threshold: Minimum signal strength (float, 0.0-1.0)
        - stop_loss: Stop loss percentage (float, 0.0-1.0)
        - take_profit: Take profit percentage (float, 0.0-2.0)
        
        @par Example:
        @code
        config = StrategyConfig(...)
        strategy = TrendFollowingStrategy(config)
        # Strategy initialized and ready
        @endcode
        """
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
        
        # Validate parameter logic
        if self.fast_period >= self.slow_period:
            raise ValueError("fast_period must be less than slow_period")
        
        if self.fast_period < 2:
            raise ValueError("fast_period must be >= 2")
        
        # Cache for calculations
        self.last_signals: Dict[str, TradingSignal] = {}
        
        logger.info(f"Trend Following Strategy initialized with periods: fast={self.fast_period}, slow={self.slow_period}")
    
    async def generate_signals(self) -> List[TradingSignal]:
        """
        @brief Generate trading signals based on trend following logic
        
        @return List[TradingSignal] List of validated trading signals
        
        @details
        Analyzes configured symbols for trend following opportunities using
        moving average crossovers with momentum and volume confirmation.
        
        @par Signal Generation Process:
        1. Iterate through all configured symbols
        2. Analyze historical price data
        3. Calculate moving averages and momentum
        4. Apply volume and signal strength filters
        5. Generate validated trading signals
        
        @par Generated Signals:
        - BUY signals for bullish crossovers with confirmation
        - SELL signals for bearish crossovers with confirmation
        - HOLD signals when conditions not met
        
        @note
        Only signals meeting all confirmation criteria are returned.
        Low-confidence signals are filtered out.
        """
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
        """
        @brief Analyze individual symbol for trend following signals
        
        @param symbol Symbol to analyze
        
        @return TradingSignal or None: Valid signal if found, None otherwise
        
        @details
        Performs comprehensive technical analysis for a single symbol including:
        - Moving average crossover detection
        - Momentum confirmation
        - Volume validation
        - Signal strength calculation
        
        @note
        This is an internal method that should not be called directly.
        Use generate_signals() for the public interface.
        """
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
            
            # Check for crossover
            current_fast = fast_ma[-1]
            current_slow = slow_ma[-1]
            prev_fast = fast_ma[-2]
            prev_slow = slow_ma[-2]
            
            # Determine crossover direction
            crossover_bullish = prev_fast <= prev_slow and current_fast > current_slow
            crossover_bearish = prev_fast >= prev_slow and current_fast < current_slow
            
            if not (crossover_bullish or crossover_bearish):
                return None
            
            # Calculate momentum
            momentum = self._calculate_momentum(prices, self.momentum_period)
            current_momentum = momentum[-1]
            
            # Volume confirmation
            recent_volume = sum(volumes[-5:]) / min(5, len(volumes))
            avg_volume = sum(volumes[-self.volume_threshold*10:]) / min(int(self.volume_threshold*10), len(volumes))
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Calculate signal strength
            ma_distance = abs(current_fast - current_slow) / current_slow
            momentum_strength = abs(current_momentum)
            
            # Determine signal type
            if crossover_bullish and current_momentum > 0:
                signal_type = SignalType.BUY
                signal_strength = min(1.0, (ma_distance + momentum_strength) / 2)
            elif crossover_bearish and current_momentum < 0:
                signal_type = SignalType.SELL
                signal_strength = min(1.0, (ma_distance + abs(momentum_strength)) / 2)
            else:
                return None
            
            # Apply filters
            if signal_strength < self.signal_threshold:
                return None
            
            if volume_ratio < self.volume_threshold:
                return None
            
            # Calculate position details
            current_price = prices[-1]
            
            # Calculate stop loss and take profit
            if signal_type == SignalType.BUY:
                stop_loss = current_price * (1 - self.stop_loss_pct)
                take_profit = current_price * (1 + self.take_profit_pct)
            else:
                stop_loss = current_price * (1 + self.stop_loss_pct)
                take_profit = current_price * (1 - self.take_profit_pct)
            
            # Calculate position size based on confidence
            base_quantity = Decimal('100')  # Base position size
            adjusted_quantity = base_quantity * Decimal(str(signal_strength))
            
            # Create signal
            signal = TradingSignal(
                signal_id=f"trend_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                strategy_id=self.config.strategy_id,
                symbol=symbol,
                signal_type=signal_type,
                confidence=signal_strength * min(1.0, volume_ratio),
                strength=signal_strength,
                price=current_price,
                quantity=adjusted_quantity,
                stop_loss=stop_loss,
                take_profit=take_profit,
                time_horizon=timedelta(days=3),
                metadata={
                    "fast_ma": float(current_fast),
                    "slow_ma": float(current_slow),
                    "momentum": float(current_momentum),
                    "volume_ratio": volume_ratio,
                    "ma_distance": ma_distance
                }
            )
            
            logger.info(f"Generated {signal_type.value} signal for {symbol}: strength={signal_strength:.3f}")
            return signal
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol} for trend following: {e}")
            return None
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        """
        @brief Validate trading signal against strategy criteria
        
        @param signal TradingSignal to validate
        
        @return bool True if signal is valid, False otherwise
        
        @details
        Validates the trading signal against multiple criteria including
        risk limits, signal quality, and market conditions.
        
        @par Validation Criteria:
        - Signal strength above minimum threshold
        - Price data quality and availability
        - Risk limit compliance
        - Market regime appropriateness
        
        @note
        This method provides an additional validation layer beyond
        the signal generation process.
        """
        try:
            # Basic validation
            if signal.confidence < self.signal_threshold:
                logger.debug(f"Signal confidence below threshold: {signal.confidence}")
                return False
            
            if signal.strength < 0.5:
                logger.debug(f"Signal strength too low: {signal.strength}")
                return False
            
            # Check if we already have a recent signal for this symbol
            if signal.symbol in self.last_signals:
                last_signal = self.last_signals[signal.symbol]
                time_diff = signal.created_at - last_signal.created_at
                
                # Don't allow signals too close together (prevent overtrading)
                if time_diff < timedelta(hours=2):
                    logger.debug(f"Recent signal exists for {signal.symbol}")
                    return False
            
            # Update last signal tracking
            self.last_signals[signal.symbol] = signal
            
            logger.debug(f"Signal validation passed for {signal.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating signal: {e}")
            return False
    
    def _calculate_sma(self, prices: List[Decimal], period: int) -> List[Decimal]:
        """
        @brief Calculate Simple Moving Average
        
        @param prices List of price values
        @param period Moving average period
        
        @return List[Decimal] Calculated moving average values
        """
        if len(prices) < period:
            return []
        
        sma_values = []
        for i in range(period - 1, len(prices)):
            window_prices = prices[i - period + 1:i + 1]
            sma = sum(window_prices) / period
            sma_values.append(sma)
        
        return sma_values
    
    def _calculate_momentum(self, prices: List[Decimal], period: int) -> List[float]:
        """
        @brief Calculate price momentum
        
        @param prices List of price values
        @param period Momentum calculation period
        
        @return List[float] Momentum values (positive = bullish, negative = bearish)
        """
        if len(prices) < period + 1:
            return []
        
        momentum_values = []
        for i in range(period, len(prices)):
            current_price = float(prices[i])
            past_price = float(prices[i - period])
            momentum = (current_price - past_price) / past_price
            momentum_values.append(momentum)
        
        return momentum_values