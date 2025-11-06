"""
Pairs Trading Strategy
Implements statistical arbitrage by trading correlated asset pairs
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
import statistics
import math

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


class PairsTradingStrategy(BaseTimeSeriesStrategy):
    """
    Pairs Trading Strategy
    
    Uses statistical arbitrage principles:
    1. Cointegration testing for long-term relationships
    2. Z-score calculation for mean reversion signals
    3. Hedge ratio calculation using linear regression
    4. Entry/exit based on statistical thresholds
    
    Parameters:
    - lookback_period: Historical data period for analysis (default: 252 trading days)
    - entry_threshold: Z-score threshold for entry signals (default: 2.0)
    - exit_threshold: Z-score threshold for exit signals (default: 0.5)
    - stop_loss_threshold: Z-score threshold for stop loss (default: 3.0)
    - min_correlation: Minimum correlation for pair selection (default: 0.7)
    - min_cointegration: Minimum cointegration test statistic (default: 0.05)
    - rebalance_frequency: Hours between hedge ratio recalculation (default: 24)
    """
    
    def __init__(self, config: StrategyConfig):
        # Validate required parameters
        required_params = ['pair_symbols', 'entry_threshold', 'exit_threshold']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1d")  # Daily data for pairs trading
        
        # Extract parameters
        self.pair_symbols = config.parameters.get('pair_symbols', [])
        self.entry_threshold = float(config.parameters.get('entry_threshold', 2.0))
        self.exit_threshold = float(config.parameters.get('exit_threshold', 0.5))
        self.stop_loss_threshold = float(config.parameters.get('stop_loss_threshold', 3.0))
        self.lookback_period = int(config.parameters.get('lookback_period', 252))
        self.min_correlation = float(config.parameters.get('min_correlation', 0.7))
        self.min_cointegration = float(config.parameters.get('min_cointegration', 0.05))
        self.rebalance_frequency = int(config.parameters.get('rebalance_frequency', 24))  # hours
        
        # Pair analysis results
        self.pair_data: Dict[str, Dict[str, Any]] = {}
        self.active_pairs: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.last_hedge_calculation = {}
        
        logger.info(f"Pairs Trading Strategy initialized for {len(self.pair_symbols)} pairs")
    
    async def generate_signals(self) -> List[TradingSignal]:
        """Generate pairs trading signals"""
        signals = []
        
        try:
            # Analyze pairs and update hedge ratios
            await self._update_pair_relationships()
            
            # Generate signals for each pair
            for pair_id, pair_info in self.pair_data.items():
                signal = await self._analyze_pair(pair_id, pair_info)
                if signal:
                    signals.append(signal)
            
            logger.debug(f"Generated {len(signals)} pairs trading signals")
            
        except Exception as e:
            logger.error(f"Error generating pairs trading signals: {e}")
        
        return signals
    
    async def _update_pair_relationships(self):
        """Update hedge ratios and statistical relationships for all pairs"""
        try:
            for pair_id in self.pair_symbols:
                await self._calculate_pair_statistics(pair_id)
                
        except Exception as e:
            logger.error(f"Error updating pair relationships: {e}")
    
    async def _calculate_pair_statistics(self, pair_id: str):
        """Calculate correlation, cointegration, and hedge ratio for a pair"""
        try:
            # Parse pair symbols (format: "SYMBOL1:SYMBOL2")
            if ':' not in pair_id:
                logger.warning(f"Invalid pair format: {pair_id}")
                return
            
            symbol1, symbol2 = pair_id.split(':', 1)
            
            # Get historical data for both symbols
            data1 = await self._get_historical_data(symbol1)
            data2 = await self._get_historical_data(symbol2)
            
            if len(data1) < self.lookback_period or len(data2) < self.lookback_period:
                logger.debug(f"Insufficient data for pair {pair_id}")
                return
            
            # Extract closing prices
            prices1 = [Decimal(str(item['close'])) for item in data1[-self.lookback_period:]]
            prices2 = [Decimal(str(item['close'])) for item in data2[-self.lookback_period:]]
            
            # Ensure same length
            min_length = min(len(prices1), len(prices2))
            prices1 = prices1[-min_length:]
            prices2 = prices2[-min_length:]
            
            # Calculate correlation
            correlation = self._calculate_correlation(prices1, prices2)
            
            # Calculate cointegration
            cointegration_pvalue = self._calculate_cointegration(prices1, prices2)
            
            # Skip pairs that don't meet criteria
            if correlation < self.min_correlation or cointegration_pvalue > self.min_cointegration:
                logger.debug(f"Pair {pair_id} doesn't meet criteria: corr={correlation:.3f}, cointegration={cointegration_pvalue:.3f}")
                return
            
            # Calculate hedge ratio using linear regression
            hedge_ratio = self._calculate_hedge_ratio(prices1, prices2)
            
            # Calculate spread and z-score
            spread = self._calculate_spread(prices1, prices2, hedge_ratio)
            z_score = self._calculate_zscore(spread)
            
            # Store pair statistics
            self.pair_data[pair_id] = {
                'symbol1': symbol1,
                'symbol2': symbol2,
                'correlation': correlation,
                'cointegration_pvalue': cointegration_pvalue,
                'hedge_ratio': hedge_ratio,
                'spread': spread,
                'z_score': z_score,
                'spread_mean': statistics.mean(spread),
                'spread_std': statistics.stdev(spread) if len(spread) > 1 else Decimal('0'),
                'last_updated': datetime.utcnow(),
                'price1': prices1[-1],
                'price2': prices2[-1]
            }
            
            logger.debug(f"Updated pair {pair_id}: correlation={correlation:.3f}, hedge_ratio={hedge_ratio:.4f}")
            
        except Exception as e:
            logger.error(f"Error calculating pair statistics for {pair_id}: {e}")
    
    def _calculate_correlation(self, prices1: List[Decimal], prices2: List[Decimal]) -> float:
        """Calculate Pearson correlation coefficient"""
        try:
            if len(prices1) != len(prices2) or len(prices1) < 2:
                return 0.0
            
            # Convert to floats
            x = [float(p) for p in prices1]
            y = [float(p) for p in prices2]
            
            # Calculate means
            mean_x = statistics.mean(x)
            mean_y = statistics.mean(y)
            
            # Calculate correlation
            numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))
            
            sum_sq_x = sum((x[i] - mean_x) ** 2 for i in range(len(x)))
            sum_sq_y = sum((y[i] - mean_y) ** 2 for i in range(len(y)))
            
            denominator = math.sqrt(sum_sq_x * sum_sq_y)
            
            if denominator == 0:
                return 0.0
            
            return numerator / denominator
            
        except Exception as e:
            logger.error(f"Error calculating correlation: {e}")
            return 0.0
    
    def _calculate_cointegration(self, prices1: List[Decimal], prices2: List[Decimal]) -> float:
        """Simplified cointegration test using Engle-Granger approach"""
        try:
            if len(prices1) != len(prices2) or len(prices1) < 10:
                return 1.0  # High p-value (not cointegrated)
            
            # Convert to floats
            x = [float(p) for p in prices1]
            y = [float(p) for p in prices2]
            
            # Step 1: Regress y on x to get hedge ratio
            mean_x = statistics.mean(x)
            mean_y = statistics.mean(y)
            
            covariance = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))) / len(x)
            variance_x = sum((x[i] - mean_x) ** 2 for i in range(len(x))) / len(x)
            
            if variance_x == 0:
                return 1.0
            
            hedge_ratio = covariance / variance_x
            
            # Step 2: Calculate residuals (spread)
            residuals = [y[i] - hedge_ratio * x[i] for i in range(len(y))]
            
            # Step 3: Test residuals for stationarity (simplified ADF test)
            # For simplicity, we'll use a basic test based on variance change
            first_half = residuals[:len(residuals)//2]
            second_half = residuals[len(residuals)//2:]
            
            if len(first_half) < 2 or len(second_half) < 2:
                return 1.0
            
            var_first = statistics.variance(first_half)
            var_second = statistics.variance(second_half)
            
            # If variances are similar, assume stationarity (cointegrated)
            var_ratio = min(var_first, var_second) / max(var_first, var_second)
            
            if var_ratio > 0.5:  # Variances similar enough
                return 0.05  # Low p-value (cointegrated)
            else:
                return 0.5   # High p-value (not cointegrated)
                
        except Exception as e:
            logger.error(f"Error calculating cointegration: {e}")
            return 1.0
    
    def _calculate_hedge_ratio(self, prices1: List[Decimal], prices2: List[Decimal]) -> Decimal:
        """Calculate hedge ratio using linear regression"""
        try:
            if len(prices1) != len(prices2) or len(prices1) < 2:
                return Decimal('1.0')
            
            # Convert to floats
            x = [float(p) for p in prices1]
            y = [float(p) for p in prices2]
            
            # Linear regression: y = alpha + beta * x
            # Beta (hedge ratio) = Cov(x,y) / Var(x)
            mean_x = statistics.mean(x)
            mean_y = statistics.mean(y)
            
            covariance = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))) / len(x)
            variance_x = sum((x[i] - mean_x) ** 2 for i in range(len(x))) / len(x)
            
            if variance_x == 0:
                return Decimal('1.0')
            
            hedge_ratio = covariance / variance_x
            return Decimal(str(hedge_ratio))
            
        except Exception as e:
            logger.error(f"Error calculating hedge ratio: {e}")
            return Decimal('1.0')
    
    def _calculate_spread(
        self, 
        prices1: List[Decimal], 
        prices2: List[Decimal], 
        hedge_ratio: Decimal
    ) -> List[Decimal]:
        """Calculate the spread between the pair"""
        try:
            spread = []
            for i in range(len(prices1)):
                spread_value = prices2[i] - hedge_ratio * prices1[i]
                spread.append(spread_value)
            return spread
            
        except Exception as e:
            logger.error(f"Error calculating spread: {e}")
            return []
    
    def _calculate_zscore(self, spread: List[Decimal]) -> float:
        """Calculate current z-score of the spread"""
        try:
            if len(spread) < 2:
                return 0.0
            
            current_spread = float(spread[-1])
            mean_spread = statistics.mean(spread)
            std_spread = statistics.stdev(spread)
            
            if std_spread == 0:
                return 0.0
            
            z_score = (current_spread - mean_spread) / std_spread
            return z_score
            
        except Exception as e:
            logger.error(f"Error calculating z-score: {e}")
            return 0.0
    
    async def _analyze_pair(self, pair_id: str, pair_info: Dict[str, Any]) -> Optional[TradingSignal]:
        """Analyze pair for trading signals"""
        try:
            # Update current prices
            current_price1 = await self._get_current_price(pair_info['symbol1'])
            current_price2 = await self._get_current_price(pair_info['symbol2'])
            
            if not current_price1 or not current_price2:
                return None
            
            # Calculate current spread and z-score
            current_spread = current_price2 - pair_info['hedge_ratio'] * current_price1
            spread_mean = pair_info['spread_mean']
            spread_std = pair_info['spread_std']
            
            if spread_std == 0:
                return None
            
            current_zscore = float((current_spread - spread_mean) / spread_std)
            
            # Update pair info with current data
            pair_info['current_spread'] = current_spread
            pair_info['current_zscore'] = current_zscore
            pair_info['price1'] = current_price1
            pair_info['price2'] = current_price2
            
            # Check for entry signals
            if current_zscore > self.entry_threshold:
                # Spread is too high: Sell symbol2, Buy symbol1
                return await self._create_pair_signal(
                    pair_id, pair_info, SignalType.SELL, SignalType.BUY, abs(current_zscore)
                )
            
            elif current_zscore < -self.entry_threshold:
                # Spread is too low: Buy symbol2, Sell symbol1
                return await self._create_pair_signal(
                    pair_id, pair_info, SignalType.BUY, SignalType.SELL, abs(current_zscore)
                )
            
            # Check for exit signals if we have an active position
            elif pair_id in self.active_pairs:
                active_pair = self.active_pairs[pair_id]
                if abs(current_zscore) < self.exit_threshold:
                    # Close positions
                    return await self._create_pair_signal(
                        pair_id, pair_info, SignalType.CLOSE, SignalType.CLOSE, abs(current_zscore)
                    )
            
            # Check for stop loss
            elif pair_id in self.active_pairs:
                active_pair = self.active_pairs[pair_id]
                if abs(current_zscore) > self.stop_loss_threshold:
                    # Forced exit
                    return await self._create_pair_signal(
                        pair_id, pair_info, SignalType.CLOSE, SignalType.CLOSE, abs(current_zscore)
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing pair {pair_id}: {e}")
            return None
    
    async def _create_pair_signal(
        self,
        pair_id: str,
        pair_info: Dict[str, Any],
        signal1_type: SignalType,
        signal2_type: SignalType,
        strength: float
    ) -> Optional[TradingSignal]:
        """Create trading signals for both assets in the pair"""
        try:
            symbol1 = pair_info['symbol1']
            symbol2 = pair_info['symbol2']
            price1 = pair_info['price1']
            price2 = pair_info['price2']
            hedge_ratio = pair_info['hedge_ratio']
            
            # Calculate position sizes
            # We want to be dollar neutral, so adjust sizes based on hedge ratio
            base_position = self._calculate_pair_position_size(strength)
            
            # Calculate quantity for symbol1
            quantity1 = base_position / price1
            
            # Calculate quantity for symbol2 to maintain hedge
            quantity2 = (base_position * hedge_ratio) / price2
            
            # Create the primary signal (arbitrary which symbol we consider primary)
            primary_signal = TradingSignal(
                signal_id=f"PAIRS_{pair_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{int(strength*100)}",
                strategy_id=self.config.strategy_id,
                symbol=f"{symbol1}:{symbol2}",  # Composite symbol for pairs
                signal_type=signal1_type,
                confidence=min(strength, 1.0),
                strength=min(strength, 1.0),
                price=price1,
                quantity=quantity1,
                metadata={
                    'strategy_type': 'pairs_trading',
                    'pair_id': pair_id,
                    'symbol1': symbol1,
                    'symbol2': symbol2,
                    'symbol1_signal': signal1_type.value,
                    'symbol2_signal': signal2_type.value,
                    'hedge_ratio': float(hedge_ratio),
                    'quantity1': float(quantity1),
                    'quantity2': float(quantity2),
                    'zscore': pair_info.get('current_zscore', 0.0),
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
            
            # Store active pair information
            if signal1_type in [SignalType.BUY, SignalType.SELL]:
                self.active_pairs[pair_id] = {
                    'symbol1': symbol1,
                    'symbol2': symbol2,
                    'hedge_ratio': hedge_ratio,
                    'entry_price1': price1,
                    'entry_price2': price2,
                    'quantity1': quantity1,
                    'quantity2': quantity2,
                    'signal_type': signal1_type,
                    'entry_time': datetime.utcnow(),
                    'entry_zscore': pair_info.get('current_zscore', 0.0)
                }
            elif signal1_type == SignalType.CLOSE and pair_id in self.active_pairs:
                # Calculate P&L
                active_pair = self.active_pairs[pair_id]
                pnl = self._calculate_pair_pnl(active_pair, pair_info)
                logger.info(f"Pair {pair_id} closed with P&L: {pnl}")
                del self.active_pairs[pair_id]
            
            return primary_signal
            
        except Exception as e:
            logger.error(f"Error creating pair signal: {e}")
            return None
    
    def _calculate_pair_pnl(
        self, 
        active_pair: Dict[str, Any], 
        current_pair: Dict[str, Any]
    ) -> float:
        """Calculate P&L for closed pair position"""
        try:
            # Simplified P&L calculation
            symbol1_pnl = 0.0
            symbol2_pnl = 0.0
            
            if active_pair['signal_type'] == SignalType.BUY:
                # We bought symbol1, sold symbol2
                symbol1_pnl = float(active_pair['quantity1']) * (float(current_pair['price1']) - float(active_pair['entry_price1']))
                symbol2_pnl = float(active_pair['quantity2']) * (float(active_pair['entry_price2']) - float(current_pair['price2']))
            else:
                # We sold symbol1, bought symbol2
                symbol1_pnl = float(active_pair['quantity1']) * (float(active_pair['entry_price1']) - float(current_pair['price1']))
                symbol2_pnl = float(active_pair['quantity2']) * (float(current_pair['price2']) - float(active_pair['entry_price2']))
            
            return symbol1_pnl + symbol2_pnl
            
        except Exception as e:
            logger.error(f"Error calculating pair P&L: {e}")
            return 0.0
    
    def _calculate_pair_position_size(self, strength: float) -> Decimal:
        """Calculate position size for pairs trading"""
        try:
            # Pairs trading typically uses moderate position sizes
            # since we're hedged and risk is lower
            base_size = self.config.max_position_size * Decimal('0.15')  # 15% of max position
            
            # Scale by signal strength
            adjusted_size = base_size * Decimal(strength)
            
            # Ensure within risk limits
            max_size = self.config.max_position_size
            final_size = min(adjusted_size, max_size)
            
            return final_size
            
        except Exception as e:
            logger.error(f"Error calculating pair position size: {e}")
            return self.config.max_position_size * Decimal('0.1')
    
    async def _get_current_price(self, symbol: str) -> Optional[Decimal]:
        """Get current price for a symbol"""
        try:
            if self.context:
                return await self.context.get_current_price(symbol)
            else:
                # Mock price for testing
                return Decimal('100.0')  # Simplified
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    async def _get_historical_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Get historical market data for symbol"""
        try:
            if self.context:
                data = await self.context.get_market_data(symbol=symbol, timeframe=self.timeframe)
                return data[-self.lookback_period:]
            else:
                return self._generate_mock_data(symbol)
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return []
    
    def _generate_mock_data(self, symbol: str, periods: int = 300) -> List[Dict[str, Any]]:
        """Generate mock data with correlated movements for pairs trading"""
        import random
        
        base_price = 100.0
        data = []
        
        # Generate base correlated movement
        for i in range(periods):
            # Create correlated price movements
            common_factor = random.gauss(0, 0.01)  # Common market factor
            
            # Add symbol-specific noise
            symbol_noise = random.gauss(0, 0.005)  # 0.5% idiosyncratic noise
            
            price_change = common_factor + symbol_noise
            base_price *= (1 + price_change)
            
            volume = random.randint(100000, 1000000)
            
            data.append({
                'timestamp': datetime.utcnow() - timedelta(days=periods-i),
                'open': base_price * random.uniform(0.998, 1.002),
                'high': base_price * random.uniform(1.001, 1.01),
                'low': base_price * random.uniform(0.99, 0.999),
                'close': base_price,
                'volume': volume
            })
        
        return data
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        """Validate pairs trading signal"""
        try:
            # Basic validation
            if not signal.symbol or ':' not in signal.symbol:
                return False
            
            if signal.quantity <= 0:
                return False
            
            # Check signal strength threshold
            if signal.strength < 0.5:  # Pairs trading requires stronger signals
                return False
            
            # Check pair exists in our analysis
            pair_id = signal.symbol
            if pair_id not in self.pair_data:
                logger.warning(f"No analysis data for pair {pair_id}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating pairs signal: {e}")
            return False
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get detailed strategy information"""
        active_pairs = len(self.active_pairs)
        analyzed_pairs = len(self.pair_data)
        
        return {
            'strategy_name': 'Pairs Trading Strategy',
            'description': 'Statistical arbitrage strategy using cointegrated asset pairs',
            'parameters': {
                'lookback_period': self.lookback_period,
                'entry_threshold': self.entry_threshold,
                'exit_threshold': self.exit_threshold,
                'stop_loss_threshold': self.stop_loss_threshold,
                'min_correlation': self.min_correlation,
                'min_cointegration': self.min_cointegration,
                'rebalance_frequency': self.rebalance_frequency
            },
            'pairs_analysis': {
                'analyzed_pairs': analyzed_pairs,
                'active_pairs': active_pairs,
                'pair_symbols': self.pair_symbols
            },
            'indicators_used': ['Correlation', 'Cointegration', 'Hedge Ratio', 'Z-Score'],
            'timeframe': self.timeframe,
            'risk_level': self.config.risk_level.value,
            'position_sizing': 'Dollar-neutral with hedge ratio adjustment',
            'typical_hold_time': 'Several days to weeks',
            'entry_conditions': 'Z-score beyond entry threshold with valid cointegration'
        }


# Factory function to create pairs trading strategy
def create_pairs_trading_strategy(
    strategy_id: str,
    pair_symbols: List[str],
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    **kwargs
) -> PairsTradingStrategy:
    """Factory function to create pairs trading strategy"""
    
    config = StrategyConfig(
        strategy_id=strategy_id,
        strategy_type=StrategyType.PAIRS_TRADING,
        name="Pairs Trading Strategy",
        description="Statistical arbitrage strategy using cointegrated asset pairs",
        parameters={
            'pair_symbols': pair_symbols,
            'entry_threshold': entry_threshold,
            'exit_threshold': exit_threshold,
            'stop_loss_threshold': kwargs.get('stop_loss_threshold', 3.0),
            'lookback_period': kwargs.get('lookback_period', 252),
            'min_correlation': kwargs.get('min_correlation', 0.7),
            'min_cointegration': kwargs.get('min_cointegration', 0.05),
            'rebalance_frequency': kwargs.get('rebalance_frequency', 24)
        },
        risk_level=RiskLevel.LOW,  # Pairs trading is generally lower risk due to hedging
        symbols=sum([pair.split(':') for pair in pair_symbols], []),  # Flatten symbols
        max_position_size=Decimal(kwargs.get('max_position_size', '100000')),
        max_daily_loss=Decimal(kwargs.get('max_daily_loss', '8000'))
    )
    
    return PairsTradingStrategy(config)


# Example usage and testing
if __name__ == "__main__":
    async def test_pairs_trading_strategy():
        # Create strategy with some example pairs
        strategy = create_pairs_trading_strategy(
            strategy_id="pairs_001",
            pair_symbols=['AAPL:MSFT', 'GOOGL:AMZN'],
            entry_threshold=2.0,
            exit_threshold=0.5
        )
        
        # Mock context for testing
        class MockContext:
            async def get_market_data(self, symbol, timeframe):
                return strategy._generate_mock_data(symbol)
            
            async def get_current_price(self, symbol):
                return Decimal('100.0')  # Simplified
        
        strategy.set_context(MockContext())
        
        # Generate signals
        signals = await strategy.generate_signals()
        
        print(f"Generated {len(signals)} pairs trading signals:")
        for signal in signals:
            print(f"  {signal.signal_type.value} {signal.symbol} (strength: {signal.strength:.2f})")
        
        # Get strategy info
        info = strategy.get_strategy_info()
        print("\nStrategy Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    # Run test
    import asyncio
    asyncio.run(test_pairs_trading_strategy())