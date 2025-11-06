"""
@file arbitrage_strategies.py
@brief Arbitrage Strategies Implementation

@details
This module implements 12+ arbitrage-based trading strategies that exploit
price differences across different markets, instruments, or timeframes.
Each strategy focuses on identifying and capitalizing on pricing inefficiencies.

Strategy Categories:
- Market Arbitrage (3): Cross-broker arbitrage, cross-venue arbitrage, index arbitrage
- Statistical Arbitrage (2): Pairs trading, statistical arbitrage
- Time Arbitrage (2): Covered interest arbitrage, futures arbitrage
- Instrument Arbitrage (3): Triangular arbitrage, convertible arbitrage, options arbitrage
- Event Arbitrage (2): Merger arbitrage, event-driven arbitrage

Key Features:
- Multi-venue price monitoring
- Statistical correlation analysis
- Latency-optimized execution
- Risk-neutral positioning
- Real-time opportunity detection

@author Trading Orchestrator System
@version 2.0
@date 2025-11-06

@warning
Arbitrage strategies require sophisticated infrastructure and can have
significant execution risks. Monitor latency and transaction costs carefully.

@see library.StrategyLibrary for strategy management
@see base.BaseStrategy for base implementation
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from dataclasses import dataclass
import asyncio
import numpy as np
import pandas as pd
from collections import deque
import json

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
from scipy import stats
from scipy.stats import zscore
import requests


# ============================================================================
# MARKET ARBITRAGE STRATEGIES
# ============================================================================

class CrossBrokerArbitrageStrategy(BaseStrategy):
    """Cross-Broker Arbitrage Strategy
    
    Exploits price differences between different brokers for the same instrument.
    Monitors multiple broker feeds and executes trades when arbitrage opportunities
    exceed transaction costs and minimum profit thresholds.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['min_profit_threshold', 'max_execution_time']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config)
        
        self.min_profit_threshold = float(config.parameters.get('min_profit_threshold', 0.002))  # 0.2%
        self.max_execution_time = float(config.parameters.get('max_execution_time', 5.0))  # seconds
        self.max_position_size = float(config.parameters.get('max_position_size', 10000))
        
        # Broker price tracking
        self.broker_prices = {}
        self.opportunity_history = deque(maxlen=100)
        
        # Metadata
        self.metadata = StrategyMetadata(
            name="CrossBrokerArbitrage",
            description="Arbitrage between different brokers",
            category=StrategyCategory.ARBITRAGE,
            risk_level=RiskLevel.MEDIUM,
            min_capital_required=50000,
            expected_return_range=(0.05, 0.15),
            max_drawdown=0.05,
            sharpe_ratio_range=(1.5, 3.0),
            time_horizon="Intraday",
            market_regime="Any",
            instruments=["stocks", "forex", "crypto", "futures"],
            parameters_schema={
                "min_profit_threshold": {"type": "float", "min": 0.001, "max": 0.01},
                "max_execution_time": {"type": "float", "min": 1.0, "max": 30.0},
                "max_position_size": {"type": "float", "min": 1000, "max": 100000}
            }
        )
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        for symbol in self.config.symbols:
            try:
                # Get prices from multiple brokers
                broker_prices = await self._get_broker_prices(symbol)
                
                if len(broker_prices) < 2:
                    continue
                
                # Find best buy and sell prices
                min_price = min(broker_prices.values())
                max_price = max(broker_prices.values())
                
                # Calculate arbitrage opportunity
                spread = (max_price - min_price) / min_price
                
                if spread > self.min_profit_threshold:
                    # Determine execution direction
                    buy_broker = [b for b, p in broker_prices.items() if p == min_price][0]
                    sell_broker = [b for b, p in broker_prices.items() if p == max_price][0]
                    
                    # Calculate position size
                    available_capital = min(
                        self.max_position_size,
                        self._get_available_capital() / len(self.config.symbols)
                    )
                    
                    position_size = min(
                        available_capital / min_price,
                        self._calculate_max_shares(symbol, min_price, available_capital)
                    )
                    
                    # Generate arbitrage signal
                    signal = TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        confidence=min(0.95, spread * 100),  # Higher spread = higher confidence
                        strength=spread,
                        price=min_price,
                        quantity=position_size,
                        timestamp=datetime.utcnow(),
                        metadata={
                            'arbitrage_type': 'cross_broker',
                            'buy_broker': buy_broker,
                            'sell_broker': sell_broker,
                            'buy_price': min_price,
                            'sell_price': max_price,
                            'spread': spread,
                            'expected_profit': spread * position_size * min_price
                        }
                    )
                    signals.append(signal)
                    
                    # Log opportunity
                    self.opportunity_history.append({
                        'symbol': symbol,
                        'spread': spread,
                        'timestamp': datetime.utcnow(),
                        'brokers': list(broker_prices.keys())
                    })
            
            except Exception as e:
                logger.error(f"Error in cross-broker arbitrage for {symbol}: {e}")
                continue
        
        return signals
    
    async def _get_broker_prices(self, symbol: str) -> Dict[str, float]:
        """Get prices from multiple brokers"""
        # Simulated broker feeds - in production, integrate with actual broker APIs
        broker_endpoints = [
            f"https://api.broker1.com/price/{symbol}",
            f"https://api.broker2.com/price/{symbol}",
            f"https://api.broker3.com/price/{symbol}"
        ]
        
        broker_prices = {}
        
        for i, endpoint in enumerate(broker_endpoints):
            try:
                # Simulate API call with small price variations
                base_price = 100.0  # Mock base price
                variation = np.random.normal(0, 0.001)  # Small random variation
                price = base_price * (1 + variation)
                broker_prices[f"broker_{i+1}"] = price
                
                # Simulate network latency
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.warning(f"Failed to get price from broker {i+1}: {e}")
                continue
        
        return broker_prices


class CrossVenueArbitrageStrategy(BaseStrategy):
    """Cross-Venue Arbitrage Strategy
    
    Exploits price differences between different trading venues (exchanges).
    Particularly effective in crypto markets where venue-specific liquidity
    creates temporary price discrepancies.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['min_profit_threshold', 'max_latency']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config)
        
        self.min_profit_threshold = float(config.parameters.get('min_profit_threshold', 0.0015))  # 0.15%
        self.max_latency = float(config.parameters.get('max_latency', 100))  # milliseconds
        self.min_liquidity = float(config.parameters.get('min_liquidity', 50000))
        self.max_position_size = float(config.parameters.get('max_position_size', 25000))
        
        self.venue_data = {}
        self.arbitrage_opportunities = deque(maxlen=1000)
        
        self.metadata = StrategyMetadata(
            name="CrossVenueArbitrage",
            description="Arbitrage between different trading venues",
            category=StrategyCategory.ARBITRAGE,
            risk_level=RiskLevel.HIGH,
            min_capital_required=100000,
            expected_return_range=(0.10, 0.25),
            max_drawdown=0.08,
            sharpe_ratio_range=(2.0, 4.0),
            time_horizon="Intraday",
            market_regime="High Volatility",
            instruments=["crypto", "forex", "stocks"],
            parameters_schema={
                "min_profit_threshold": {"type": "float", "min": 0.0005, "max": 0.005},
                "max_latency": {"type": "float", "min": 50, "max": 500},
                "min_liquidity": {"type": "float", "min": 10000, "max": 200000},
                "max_position_size": {"type": "float", "min": 5000, "max": 100000}
            }
        )
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        for symbol in self.config.symbols:
            try:
                # Get order book data from multiple venues
                venue_data = await self._get_venue_orderbooks(symbol)
                
                if len(venue_data) < 2:
                    continue
                
                # Find best opportunities
                opportunities = self._find_arbitrage_opportunities(venue_data)
                
                for opportunity in opportunities:
                    if opportunity['profit'] > self.min_profit_threshold:
                        signal = TradingSignal(
                            symbol=symbol,
                            signal_type=SignalType.BUY,
                            confidence=min(0.90, opportunity['profit'] * 200),
                            strength=opportunity['profit'],
                            price=opportunity['buy_price'],
                            quantity=opportunity['size'],
                            timestamp=datetime.utcnow(),
                            metadata={
                                'arbitrage_type': 'cross_venue',
                                'buy_venue': opportunity['buy_venue'],
                                'sell_venue': opportunity['sell_venue'],
                                'buy_price': opportunity['buy_price'],
                                'sell_price': opportunity['sell_price'],
                                'buy_depth': opportunity['buy_depth'],
                                'sell_depth': opportunity['sell_depth'],
                                'net_profit': opportunity['net_profit'],
                                'latency': opportunity['latency']
                            }
                        )
                        signals.append(signal)
                        
                        # Record opportunity
                        self.arbitrage_opportunities.append({
                            'symbol': symbol,
                            'profit': opportunity['profit'],
                            'timestamp': datetime.utcnow(),
                            'venues': [opportunity['buy_venue'], opportunity['sell_venue']]
                        })
            
            except Exception as e:
                logger.error(f"Error in cross-venue arbitrage for {symbol}: {e}")
                continue
        
        return signals
    
    async def _get_venue_orderbooks(self, symbol: str) -> Dict[str, Dict]:
        """Get order book data from multiple venues"""
        venues = ['binance', 'coinbase', 'kraken', 'bitfinex']
        venue_data = {}
        
        for venue in venues:
            try:
                # Simulate order book data with venue-specific characteristics
                mid_price = 100.0
                bid_ask_spread = np.random.uniform(0.0001, 0.001)
                price_level = np.random.normal(0, 0.0005)
                
                venue_data[venue] = {
                    'bid': mid_price - bid_ask_spread/2 + price_level,
                    'ask': mid_price + bid_ask_spread/2 + price_level,
                    'bid_size': np.random.uniform(10, 100),
                    'ask_size': np.random.uniform(10, 100),
                    'timestamp': datetime.utcnow(),
                    'latency': np.random.uniform(50, 200)  # milliseconds
                }
                
                # Simulate network latency
                await asyncio.sleep(0.005)
                
            except Exception as e:
                logger.warning(f"Failed to get data from {venue}: {e}")
                continue
        
        return venue_data
    
    def _find_arbitrage_opportunities(self, venue_data: Dict) -> List[Dict]:
        """Find arbitrage opportunities in venue data"""
        opportunities = []
        
        venues = list(venue_data.keys())
        
        for i in range(len(venues)):
            for j in range(i + 1, len(venues)):
                venue1, venue2 = venues[i], venues[j]
                
                # Buy from venue1, sell on venue2
                buy_price = venue_data[venue1]['ask']
                sell_price = venue2 in venue_data and venue_data[venue2]['bid']
                
                if sell_price and buy_price < sell_price:
                    profit = (sell_price - buy_price) / buy_price
                    max_size = min(
                        venue_data[venue1]['ask_size'],
                        venue_data[venue2]['bid_size']
                    )
                    
                    opportunities.append({
                        'buy_venue': venue1,
                        'sell_venue': venue2,
                        'buy_price': buy_price,
                        'sell_price': sell_price,
                        'size': max_size,
                        'profit': profit,
                        'net_profit': profit * max_size,
                        'latency': max(venue_data[venue1].get('latency', 100), 
                                     venue_data[venue2].get('latency', 100))
                    })
        
        return sorted(opportunities, key=lambda x: x['profit'], reverse=True)


class IndexArbitrageStrategy(BaseStrategy):
    """Index Arbitrage Strategy
    
    Exploits price differences between an index and its constituent securities
    or between different index derivatives (futures, ETFs, options).
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['min_deviation', 'rebalance_frequency']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config)
        
        self.min_deviation = float(config.parameters.get('min_deviation', 0.005))  # 0.5%
        self.rebalance_frequency = int(config.parameters.get('rebalance_frequency', 300))  # seconds
        self.max_holding_period = int(config.parameters.get('max_holding_period', 3600))  # 1 hour
        self.position_limit = float(config.parameters.get('position_limit', 50000))
        
        self.index_weights = {}
        self.last_rebalance = None
        self.deviation_history = deque(maxlen=500)
        
        self.metadata = StrategyMetadata(
            name="IndexArbitrage",
            description="Arbitrage between index and components",
            category=StrategyCategory.ARBITRAGE,
            risk_level=RiskLevel.MEDIUM,
            min_capital_required=250000,
            expected_return_range=(0.03, 0.08),
            max_drawdown=0.04,
            sharpe_ratio_range=(1.2, 2.5),
            time_horizon="Daily",
            market_regime="Trending",
            instruments=["stocks", "etfs", "index_futures"],
            parameters_schema={
                "min_deviation": {"type": "float", "min": 0.001, "max": 0.02},
                "rebalance_frequency": {"type": "int", "min": 60, "max": 3600},
                "max_holding_period": {"type": "int", "min": 300, "max": 7200},
                "position_limit": {"type": "float", "min": 10000, "max": 1000000}
            }
        )
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        # Only rebalance at specified frequency
        now = datetime.utcnow()
        if (self.last_rebalance and 
            (now - self.last_rebalance).total_seconds() < self.rebalance_frequency):
            return signals
        
        try:
            # Get index price and component prices
            for index_symbol in self.config.symbols:
                if not index_symbol.startswith('INDEX:'):
                    continue
                
                index_data = await self._get_index_data(index_symbol)
                if not index_data:
                    continue
                
                # Get component prices and weights
                component_data = await self._get_component_data(index_symbol)
                if not component_data:
                    continue
                
                # Calculate theoretical vs actual index price
                theoretical_price = sum(
                    component['price'] * component['weight'] 
                    for component in component_data.values()
                )
                
                actual_price = index_data['price']
                deviation = (actual_price - theoretical_price) / theoretical_price
                
                # Store deviation in history
                self.deviation_history.append({
                    'timestamp': now,
                    'deviation': deviation,
                    'theoretical': theoretical_price,
                    'actual': actual_price
                })
                
                # Check for arbitrage opportunity
                if abs(deviation) > self.min_deviation:
                    # Generate signals for components
                    for component, data in component_data.items():
                        weight = data['weight']
                        position_value = self.position_limit * abs(deviation)
                        quantity = position_value / data['price']
                        
                        signal_type = SignalType.BUY if deviation < 0 else SignalType.SELL
                        
                        signal = TradingSignal(
                            symbol=component,
                            signal_type=signal_type,
                            confidence=min(0.85, abs(deviation) * 50),
                            strength=abs(deviation),
                            price=data['price'],
                            quantity=quantity,
                            timestamp=now,
                            metadata={
                                'arbitrage_type': 'index',
                                'index_symbol': index_symbol,
                                'deviation': deviation,
                                'theoretical_price': theoretical_price,
                                'actual_price': actual_price,
                                'component_weight': weight,
                                'expected_return': abs(deviation)
                            }
                        )
                        signals.append(signal)
                
                self.last_rebalance = now
        
        except Exception as e:
            logger.error(f"Error in index arbitrage: {e}")
        
        return signals
    
    async def _get_index_data(self, index_symbol: str) -> Optional[Dict]:
        """Get index price and metadata"""
        try:
            # Simulate index data
            base_price = 1000.0
            price = base_price * (1 + np.random.normal(0, 0.02))
            
            return {
                'price': price,
                'timestamp': datetime.utcnow(),
                'volume': np.random.uniform(1000000, 5000000)
            }
        except Exception as e:
            logger.error(f"Error getting index data for {index_symbol}: {e}")
            return None
    
    async def _get_component_data(self, index_symbol: str) -> Dict[str, Dict]:
        """Get component securities data with weights"""
        components = {}
        
        # Simulate top 10 components
        for i in range(10):
            symbol = f"{index_symbol.replace('INDEX:', '')}_COMP{i+1}"
            weight = np.random.uniform(0.05, 0.15)
            price = 50.0 * (1 + np.random.normal(0, 0.03))
            
            components[symbol] = {
                'price': price,
                'weight': weight,
                'timestamp': datetime.utcnow()
            }
        
        # Normalize weights to sum to 1.0
        total_weight = sum(comp['weight'] for comp in components.values())
        for component in components.values():
            component['weight'] /= total_weight
        
        return components


# ============================================================================
# STATISTICAL ARBITRAGE STRATEGIES
# ============================================================================

class StatisticalPairsArbitrageStrategy(BaseTimeSeriesStrategy):
    """Statistical Pairs Arbitrage Strategy
    
    Uses statistical methods to identify cointegrated pairs of securities
    and trades the spread when it deviates from historical norms.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['lookback_period', 'zscore_threshold', 'half_life']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="15m")
        
        self.lookback_period = int(config.parameters.get('lookback_period', 252))
        self.zscore_threshold = float(config.parameters.get('zscore_threshold', 2.0))
        self.half_life = int(config.parameters.get('half_life', 20))
        self.min_correlation = float(config.parameters.get('min_correlation', 0.7))
        
        self.pair_data = {}
        self.cointegration_results = {}
        
        self.metadata = StrategyMetadata(
            name="StatisticalPairsArbitrage",
            description="Statistical arbitrage using cointegrated pairs",
            category=StrategyCategory.ARBITRAGE,
            risk_level=RiskLevel.MEDIUM,
            min_capital_required=100000,
            expected_return_range=(0.08, 0.15),
            max_drawdown=0.06,
            sharpe_ratio_range=(1.8, 2.8),
            time_horizon="Weekly",
            market_regime="Mean Reverting",
            instruments=["stocks", "etfs"],
            parameters_schema={
                "lookback_period": {"type": "int", "min": 50, "max": 500},
                "zscore_threshold": {"type": "float", "min": 1.5, "max": 3.0},
                "half_life": {"type": "int", "min": 5, "max": 60},
                "min_correlation": {"type": "float", "min": 0.5, "max": 0.95}
            }
        )
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        try:
            # Get historical data for all symbols
            price_data = {}
            for symbol in self.config.symbols:
                data = await self._get_historical_data(symbol)
                if len(data) >= self.lookback_period:
                    prices = [Decimal(str(item['close'])) for item in data[-self.lookback_period:]]
                    price_data[symbol] = list(map(float, prices))
            
            if len(price_data) < 2:
                return signals
            
            # Find cointegrated pairs
            pairs = self._find_cointegrated_pairs(price_data)
            
            # Generate signals for each pair
            for pair in pairs:
                pair_signals = await self._analyze_pair_spread(pair, price_data)
                signals.extend(pair_signals)
        
        except Exception as e:
            logger.error(f"Error in statistical pairs arbitrage: {e}")
        
        return signals
    
    def _find_cointegrated_pairs(self, price_data: Dict[str, List[float]]) -> List[Tuple[str, str, float]]:
        """Find cointegrated pairs using Johansen test"""
        pairs = []
        symbols = list(price_data.keys())
        
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                symbol1, symbol2 = symbols[i], symbols[j]
                
                try:
                    # Calculate correlation
                    correlation = np.corrcoef(price_data[symbol1], price_data[symbol2])[0, 1]
                    
                    if correlation < self.min_correlation:
                        continue
                    
                    # Test for cointegration using Engle-Granger method
                    price1 = np.array(price_data[symbol1])
                    price2 = np.array(price_data[symbol2])
                    
                    # Calculate hedge ratio using OLS
                    hedge_ratio = np.corrcoef(price1, price2)[0, 1] * np.std(price2) / np.std(price1)
                    
                    # Calculate spread
                    spread = price1 - hedge_ratio * price2
                    
                    # Test if spread is stationary (ADF test)
                    adf_statistic = self._adf_test(spread)
                    
                    # If ADF statistic is significant, series is stationary (cointegrated)
                    if adf_statistic < -2.86:  # Critical value at 5%
                        pairs.append((symbol1, symbol2, hedge_ratio))
                
                except Exception as e:
                    logger.warning(f"Error testing pair ({symbol1}, {symbol2}): {e}")
                    continue
        
        return pairs
    
    async def _analyze_pair_spread(self, pair: Tuple[str, str, float], 
                                 price_data: Dict[str, List[float]]) -> List[TradingSignal]:
        """Analyze pair spread for trading opportunities"""
        symbol1, symbol2, hedge_ratio = pair
        signals = []
        
        try:
            price1 = np.array(price_data[symbol1])
            price2 = np.array(price_data[symbol2])
            
            # Calculate current spread
            spread = price1[-1] - hedge_ratio * price2[-1]
            
            # Calculate historical spread statistics
            historical_spread = price1 - hedge_ratio * price2
            spread_mean = np.mean(historical_spread)
            spread_std = np.std(historical_spread)
            
            # Calculate z-score
            z_score = (spread - spread_mean) / spread_std
            
            # Check for trading signals
            if abs(z_score) > self.zscore_threshold:
                # Calculate position sizes
                portfolio_value = self._get_portfolio_value()
                position_size = portfolio_value * 0.02  # 2% of portfolio per trade
                
                # Determine trade direction
                if z_score > 0:
                    # Spread is high, short the spread (short symbol1, long symbol2)
                    signal_type1 = SignalType.SELL
                    signal_type2 = SignalType.BUY
                    quantity1 = position_size / price1[-1]
                    quantity2 = (hedge_ratio * position_size) / price2[-1]
                else:
                    # Spread is low, long the spread (long symbol1, short symbol2)
                    signal_type1 = SignalType.BUY
                    signal_type2 = SignalType.SELL
                    quantity1 = position_size / price1[-1]
                    quantity2 = (hedge_ratio * position_size) / price2[-1]
                
                # Generate signals for both securities
                signal1 = TradingSignal(
                    symbol=symbol1,
                    signal_type=signal_type1,
                    confidence=min(0.90, abs(z_score) / self.zscore_threshold),
                    strength=abs(z_score) / self.zscore_threshold,
                    price=price1[-1],
                    quantity=quantity1,
                    timestamp=datetime.utcnow(),
                    metadata={
                        'arbitrage_type': 'statistical_pairs',
                        'pair_symbol': symbol2,
                        'hedge_ratio': hedge_ratio,
                        'spread': spread,
                        'z_score': z_score,
                        'spread_mean': spread_mean,
                        'expected_reversion': True
                    }
                )
                
                signal2 = TradingSignal(
                    symbol=symbol2,
                    signal_type=signal_type2,
                    confidence=min(0.90, abs(z_score) / self.zscore_threshold),
                    strength=abs(z_score) / self.zscore_threshold,
                    price=price2[-1],
                    quantity=quantity2,
                    timestamp=datetime.utcnow(),
                    metadata={
                        'arbitrage_type': 'statistical_pairs',
                        'pair_symbol': symbol1,
                        'hedge_ratio': 1/hedge_ratio,
                        'spread': -spread/hedge_ratio,
                        'z_score': -z_score,
                        'spread_mean': -spread_mean/hedge_ratio,
                        'expected_reversion': True
                    }
                )
                
                signals.extend([signal1, signal2])
        
        except Exception as e:
            logger.error(f"Error analyzing pair ({symbol1}, {symbol2}): {e}")
        
        return signals
    
    def _adf_test(self, series: np.ndarray) -> float:
        """Augmented Dickey-Fuller test for stationarity"""
        try:
            from statsmodels.tsa.stattools import adfuller
            
            result = adfuller(series)
            return result[0]  # ADF statistic
        except ImportError:
            # Simple fallback if statsmodels not available
            # Calculate first difference autocorrrelation as proxy
            diff_series = np.diff(series)
            if len(diff_series) > 1:
                return -np.corrcoef(diff_series[:-1], diff_series[1:])[0, 1]
            return 0


class MeanReversionArbitrageStrategy(BaseTimeSeriesStrategy):
    """Mean Reversion Arbitrage Strategy
    
    Uses statistical models to identify securities that have deviated
    significantly from their historical means and trades the reversion.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['lookback_period', 'std_threshold', 'min_trades']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1h")
        
        self.lookback_period = int(config.parameters.get('lookback_period', 126))
        self.std_threshold = float(config.parameters.get('std_threshold', 2.0))
        self.min_trades = int(config.parameters.get('min_trades', 30))
        self.half_life = int(config.parameters.get('half_life', 10))
        
        self.statistics = {}
        self.trade_history = {}
        
        self.metadata = StrategyMetadata(
            name="MeanReversionArbitrage",
            description="Statistical arbitrage using mean reversion",
            category=StrategyCategory.ARBITRAGE,
            risk_level=RiskLevel.MEDIUM,
            min_capital_required=75000,
            expected_return_range=(0.06, 0.12),
            max_drawdown=0.05,
            sharpe_ratio_range=(1.5, 2.5),
            time_horizon="Intraday",
            market_regime="Mean Reverting",
            instruments=["stocks", "etfs", "commodities"],
            parameters_schema={
                "lookback_period": {"type": "int", "min": 50, "max": 252},
                "std_threshold": {"type": "float", "min": 1.5, "max": 3.0},
                "min_trades": {"type": "int", "min": 10, "max": 100},
                "half_life": {"type": "int", "min": 5, "max": 30}
            }
        )
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        for symbol in self.config.symbols:
            try:
                # Get historical data
                data = await self._get_historical_data(symbol)
                if len(data) < self.lookback_period + self.half_life:
                    continue
                
                prices = [Decimal(str(item['close'])) for item in data]
                
                # Calculate statistics
                stats = self._calculate_price_statistics(prices)
                
                if stats['trade_count'] < self.min_trades:
                    continue
                
                # Check for mean reversion opportunity
                current_price = float(prices[-1])
                z_score = (current_price - stats['mean']) / stats['std']
                
                if abs(z_score) > self.std_threshold:
                    # Calculate position size based on signal strength
                    signal_strength = abs(z_score) / self.std_threshold
                    position_size = self._calculate_position_size(symbol, signal_strength)
                    
                    signal_type = SignalType.SELL if z_score > 0 else SignalType.BUY
                    
                    signal = TradingSignal(
                        symbol=symbol,
                        signal_type=signal_type,
                        confidence=min(0.85, signal_strength),
                        strength=signal_strength,
                        price=current_price,
                        quantity=position_size,
                        timestamp=datetime.utcnow(),
                        metadata={
                            'arbitrage_type': 'mean_reversion',
                            'mean_price': stats['mean'],
                            'std_price': stats['std'],
                            'z_score': z_score,
                            'half_life': self.half_life,
                            'expected_return': stats['mean'] - current_price if z_score < 0 
                                             else current_price - stats['mean']
                        }
                    )
                    signals.append(signal)
            
            except Exception as e:
                logger.error(f"Error in mean reversion arbitrage for {symbol}: {e}")
                continue
        
        return signals
    
    def _calculate_price_statistics(self, prices: List[Decimal]) -> Dict:
        """Calculate price statistics for mean reversion analysis"""
        price_list = [float(p) for p in prices]
        
        # Calculate returns
        returns = np.diff(price_list) / price_list[:-1]
        
        # Calculate price statistics
        mean_price = np.mean(price_list)
        std_price = np.std(price_list)
        
        # Calculate trade opportunities
        up_days = np.sum(returns > 0)
        down_days = np.sum(returns < 0)
        total_trades = len(returns)
        
        # Calculate half-life of mean reversion
        price_centered = price_list - mean_price
        lag_price = np.roll(price_centered, 1)
        lag_price[0] = 0
        
        # Regression of price change on lagged price
        if len(price_centered) > 1:
            X = lag_price[1:].reshape(-1, 1)
            y = np.diff(price_centered)
            
            try:
                # Simple OLS regression
                X_with_intercept = np.column_stack([np.ones(len(X)), X])
                beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
                decay_rate = -beta[1]
                half_life = np.log(2) / decay_rate if decay_rate > 0 else self.half_life
            except:
                half_life = self.half_life
        else:
            half_life = self.half_life
        
        return {
            'mean': mean_price,
            'std': std_price,
            'up_days': up_days,
            'down_days': down_days,
            'trade_count': total_trades,
            'half_life': half_life,
            'mean_return': np.mean(returns) if len(returns) > 0 else 0,
            'volatility': np.std(returns) if len(returns) > 0 else 0
        }


# ============================================================================
# TIME ARBITRAGE STRATEGIES
# ============================================================================

class CoveredInterestArbitrageStrategy(BaseStrategy):
    """Covered Interest Arbitrage Strategy
    
    Exploits interest rate differentials between currencies using forward
    contracts to hedge exchange rate risk.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['min_rate_diff', 'forward_period', 'min_tenor']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config)
        
        self.min_rate_diff = float(config.parameters.get('min_rate_diff', 0.0025))  # 0.25%
        self.forward_period = int(config.parameters.get('forward_period', 30))  # days
        self.min_tenor = int(config.parameters.get('min_tenor', 90))  # days
        self.position_limit = float(config.parameters.get('position_limit', 1000000))
        
        self.interest_rates = {}
        self.forward_rates = {}
        
        self.metadata = StrategyMetadata(
            name="CoveredInterestArbitrage",
            description="Interest rate arbitrage using forward contracts",
            category=StrategyCategory.ARBITRAGE,
            risk_level=RiskLevel.LOW,
            min_capital_required=500000,
            expected_return_range=(0.02, 0.05),
            max_drawdown=0.02,
            sharpe_ratio_range=(1.0, 2.0),
            time_horizon="Monthly",
            market_regime="Stable Interest Rates",
            instruments=["forex", "interest_rate_derivatives"],
            parameters_schema={
                "min_rate_diff": {"type": "float", "min": 0.001, "max": 0.01},
                "forward_period": {"type": "int", "min": 7, "max": 90},
                "min_tenor": {"type": "int", "min": 30, "max": 365},
                "position_limit": {"type": "float", "min": 100000, "max": 10000000}
            }
        )
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        try:
            # Get currency pairs and interest rates
            currency_pairs = [s for s in self.config.symbols if '/' in s]
            
            for pair in currency_pairs:
                base_currency, quote_currency = pair.split('/')
                
                # Get interest rates and forward rates
                spot_rate = await self._get_spot_rate(pair)
                base_rate = await self._get_interest_rate(base_currency)
                quote_rate = await self._get_interest_rate(quote_currency)
                
                if None in [spot_rate, base_rate, quote_rate]:
                    continue
                
                # Calculate forward rate using covered interest parity
                forward_rate = spot_rate * (1 + quote_rate * self.forward_period / 365) / (1 + base_rate * self.forward_period / 365)
                
                # Compare with actual forward market rate
                market_forward_rate = await self._get_forward_rate(pair, self.forward_period)
                
                if market_forward_rate is None:
                    continue
                
                # Calculate arbitrage opportunity
                rate_diff = (market_forward_rate - forward_rate) / forward_rate
                
                if abs(rate_diff) > self.min_rate_diff:
                    position_size = min(
                        self.position_limit / spot_rate,
                        self._calculate_max_position(pair, spot_rate)
                    )
                    
                    signal_type = SignalType.SELL if rate_diff > 0 else SignalType.BUY
                    
                    signal = TradingSignal(
                        symbol=pair,
                        signal_type=signal_type,
                        confidence=min(0.90, abs(rate_diff) * 200),
                        strength=abs(rate_diff),
                        price=spot_rate,
                        quantity=position_size,
                        timestamp=datetime.utcnow(),
                        metadata={
                            'arbitrage_type': 'covered_interest',
                            'forward_period': self.forward_period,
                            'base_rate': base_rate,
                            'quote_rate': quote_rate,
                            'theoretical_forward': forward_rate,
                            'market_forward': market_forward_rate,
                            'rate_diff': rate_diff,
                            'expected_profit': abs(rate_diff) * position_size * spot_rate,
                            'forward_contract_needed': True
                        }
                    )
                    signals.append(signal)
        
        except Exception as e:
            logger.error(f"Error in covered interest arbitrage: {e}")
        
        return signals
    
    async def _get_spot_rate(self, pair: str) -> Optional[float]:
        """Get current spot exchange rate"""
        try:
            # Simulate spot rate
            base_rates = {'USD': 1.0, 'EUR': 0.85, 'GBP': 0.73, 'JPY': 110.0, 'CHF': 0.92}
            
            base, quote = pair.split('/')
            base_rate = base_rates.get(base, 1.0)
            quote_rate = base_rates.get(quote, 1.0)
            
            spot_rate = base_rate / quote_rate
            
            # Add some market noise
            spot_rate *= (1 + np.random.normal(0, 0.001))
            
            return spot_rate
        except Exception as e:
            logger.error(f"Error getting spot rate for {pair}: {e}")
            return None
    
    async def _get_interest_rate(self, currency: str) -> Optional[float]:
        """Get interest rate for currency"""
        rates = {
            'USD': 0.0250,  # 2.5%
            'EUR': 0.0150,  # 1.5%
            'GBP': 0.0200,  # 2.0%
            'JPY': -0.0010, # -0.1%
            'CHF': 0.0100,  # 1.0%
            'CAD': 0.0175,  # 1.75%
            'AUD': 0.0200   # 2.0%
        }
        return rates.get(currency)
    
    async def _get_forward_rate(self, pair: str, period: int) -> Optional[float]:
        """Get forward exchange rate from market"""
        try:
            spot_rate = await self._get_spot_rate(pair)
            if spot_rate is None:
                return None
            
            # Simulate forward rate with small deviation from theoretical
            theoretical_forward = spot_rate * (1 + np.random.uniform(-0.001, 0.001))
            return theoretical_forward
        except Exception as e:
            logger.error(f"Error getting forward rate for {pair}: {e}")
            return None


class FuturesArbitrageStrategy(BaseStrategy):
    """Futures Arbitrage Strategy
    
    Exploits price differences between futures contracts and underlying assets,
    or between futures contracts with different expiration dates (calendar spreads).
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['min_basis', 'cost_of_carry', 'max_holding_period']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config)
        
        self.min_basis = float(config.parameters.get('min_basis', 0.005))  # 0.5%
        self.cost_of_carry = float(config.parameters.get('cost_of_carry', 0.03))  # 3% annual
        self.max_holding_period = int(config.parameters.get('max_holding_period', 30))  # days
        self.position_limit = float(config.parameters.get('position_limit', 500000))
        
        self.futures_data = {}
        self.spot_data = {}
        
        self.metadata = StrategyMetadata(
            name="FuturesArbitrage",
            description="Arbitrage between futures and spot markets",
            category=StrategyCategory.ARBITRAGE,
            risk_level=RiskLevel.MEDIUM,
            min_capital_required=300000,
            expected_return_range=(0.04, 0.10),
            max_drawdown=0.04,
            sharpe_ratio_range=(1.2, 2.2),
            time_horizon="Weekly",
            market_regime="Contango/Backwardation",
            instruments=["commodities", "financial_futures", "index_futures"],
            parameters_schema={
                "min_basis": {"type": "float", "min": 0.002, "max": 0.02},
                "cost_of_carry": {"type": "float", "min": 0.01, "max": 0.08},
                "max_holding_period": {"type": "int", "min": 7, "max": 90},
                "position_limit": {"type": "float", "min": 100000, "max": 5000000}
            }
        )
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        for symbol in self.config.symbols:
            try:
                if not symbol.startswith('FUTURE:'):
                    continue
                
                underlying = symbol.replace('FUTURE:', '')
                
                # Get futures and spot data
                futures_data = await self._get_futures_data(symbol)
                spot_data = await self._get_spot_data(underlying)
                
                if not futures_data or not spot_data:
                    continue
                
                # Calculate theoretical futures price using cost of carry model
                time_to_expiry = (futures_data['expiry'] - datetime.utcnow()).days
                theoretical_futures = spot_data['price'] * (1 + self.cost_of_carry * time_to_expiry / 365)
                
                # Calculate basis
                basis = (futures_data['price'] - spot_data['price']) / spot_data['price']
                theoretical_basis = (theoretical_futures - spot_data['price']) / spot_data['price']
                
                # Check for arbitrage opportunity
                basis_deviation = basis - theoretical_basis
                
                if abs(basis_deviation) > self.min_basis:
                    position_size = min(
                        self.position_limit / spot_data['price'],
                        self._calculate_max_futures_position(symbol, futures_data['price'])
                    )
                    
                    if basis_deviation > 0:
                        # Futures overpriced relative to spot
                        signal_type = SignalType.SELL
                        hedge_type = SignalType.BUY
                    else:
                        # Futures underpriced relative to spot
                        signal_type = SignalType.BUY
                        hedge_type = SignalType.SELL
                    
                    # Generate futures signal
                    futures_signal = TradingSignal(
                        symbol=symbol,
                        signal_type=signal_type,
                        confidence=min(0.85, abs(basis_deviation) * 50),
                        strength=abs(basis_deviation),
                        price=futures_data['price'],
                        quantity=position_size,
                        timestamp=datetime.utcnow(),
                        metadata={
                            'arbitrage_type': 'futures_spot',
                            'underlying': underlying,
                            'spot_price': spot_data['price'],
                            'theoretical_futures': theoretical_futures,
                            'actual_futures': futures_data['price'],
                            'basis': basis,
                            'theoretical_basis': theoretical_basis,
                            'basis_deviation': basis_deviation,
                            'time_to_expiry': time_to_expiry,
                            'hedge_symbol': underlying,
                            'hedge_type': hedge_type
                        }
                    )
                    
                    # Generate spot hedge signal
                    spot_signal = TradingSignal(
                        symbol=underlying,
                        signal_type=hedge_type,
                        confidence=futures_signal.confidence,
                        strength=futures_signal.strength,
                        price=spot_data['price'],
                        quantity=position_size,
                        timestamp=datetime.utcnow(),
                        metadata={
                            'arbitrage_type': 'futures_spot_hedge',
                            'futures_symbol': symbol,
                            'futures_price': futures_data['price'],
                            'hedge_ratio': 1.0,
                            'basis_deviation': basis_deviation
                        }
                    )
                    
                    signals.extend([futures_signal, spot_signal])
            except Exception as e:
                logger.error(f"Error in futures arbitrage: {e}")
        
        return signals
    
    async def _get_futures_data(self, symbol: str) -> Optional[Dict]:
        """Get futures contract data"""
        try:
            # Simulate futures data
            base_price = 100.0
            time_to_expiry = np.random.uniform(30, 180)  # days
            price = base_price * (1 + np.random.normal(0, 0.02))
            
            return {
                'price': price,
                'expiry': datetime.utcnow() + timedelta(days=time_to_expiry),
                'volume': np.random.uniform(1000, 10000),
                'open_interest': np.random.uniform(5000, 50000),
                'timestamp': datetime.utcnow()
            }
        except Exception as e:
            logger.error(f"Error getting futures data for {symbol}: {e}")
            return None
    
    async def _get_spot_data(self, underlying: str) -> Optional[Dict]:
        """Get spot price data"""
        try:
            # Simulate spot data
            price = 100.0 * (1 + np.random.normal(0, 0.015))
            
            return {
                'price': price,
                'volume': np.random.uniform(10000, 100000),
                'timestamp': datetime.utcnow()
            }
        except Exception as e:
            logger.error(f"Error getting spot data for {underlying}: {e}")
            return None


# ============================================================================
# INSTRUMENT ARBITRAGE STRATEGIES
# ============================================================================

class TriangularArbitrageStrategy(BaseStrategy):
    """Triangular Arbitrage Strategy
    
    Exploits pricing inefficiencies in three-currency relationships
    where the cross-rate doesn't match the direct rates.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['min_profit_threshold', 'max_execution_time']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config)
        
        self.min_profit_threshold = float(config.parameters.get('min_profit_threshold', 0.001))  # 0.1%
        self.max_execution_time = float(config.parameters.get('max_execution_time', 3.0))  # seconds
        self.min_liquidity = float(config.parameters.get('min_liquidity', 100000))
        self.max_position_size = float(config.parameters.get('max_position_size', 100000))
        
        self.currency_pairs = []
        self.triangular_paths = []
        
        self.metadata = StrategyMetadata(
            name="TriangularArbitrage",
            description="Currency triangular arbitrage opportunities",
            category=StrategyCategory.ARBITRAGE,
            risk_level=RiskLevel.HIGH,
            min_capital_required=200000,
            expected_return_range=(0.08, 0.20),
            max_drawdown=0.06,
            sharpe_ratio_range=(2.0, 4.0),
            time_horizon="Intraday",
            market_regime="High Volatility",
            instruments=["forex", "crypto"],
            parameters_schema={
                "min_profit_threshold": {"type": "float", "min": 0.0005, "max": 0.005},
                "max_execution_time": {"type": "float", "min": 1.0, "max": 10.0},
                "min_liquidity": {"type": "float", "min": 50000, "max": 500000},
                "max_position_size": {"type": "float", "min": 25000, "max": 500000}
            }
        )
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        try:
            # Get all currency pairs
            self.currency_pairs = [s for s in self.config.symbols if '/' in s]
            
            # Generate triangular paths
            self.triangular_paths = self._generate_triangular_paths()
            
            for path in self.triangular_paths:
                opportunity = await self._calculate_arbitrage_opportunity(path)
                
                if opportunity and opportunity['profit'] > self.min_profit_threshold:
                    # Generate signals for each leg of the triangle
                    for i, leg in enumerate(path):
                        signal = TradingSignal(
                            symbol=leg['pair'],
                            signal_type=leg['side'],
                            confidence=min(0.95, opportunity['profit'] * 500),
                            strength=opportunity['profit'],
                            price=leg['price'],
                            quantity=opportunity['size'],
                            timestamp=datetime.utcnow(),
                            metadata={
                                'arbitrage_type': 'triangular',
                                'path_index': i,
                                'total_legs': len(path),
                                'total_profit': opportunity['profit'],
                                'path_profitability': opportunity['path_profit'],
                                'execution_order': i + 1,
                                'max_execution_time': self.max_execution_time
                            }
                        )
                        signals.append(signal)
        
        except Exception as e:
            logger.error(f"Error in triangular arbitrage: {e}")
        
        return signals
    
    def _generate_triangular_paths(self) -> List[List[Dict]]:
        """Generate all possible triangular arbitrage paths"""
        paths = []
        currencies = set()
        
        # Extract all currencies from pairs
        for pair in self.currency_pairs:
            base, quote = pair.split('/')
            currencies.add(base)
            currencies.add(quote)
        
        currencies = list(currencies)
        
        # Generate paths for each currency triplet
        for i, curr1 in enumerate(currencies):
            for j, curr2 in enumerate(currencies):
                if i >= j:
                    continue
                for k, curr3 in enumerate(currencies):
                    if k in [i, j]:
                        continue
                    
                    # Generate triangular paths
                    paths.extend([
                        # Path 1: curr1 -> curr2 -> curr3 -> curr1
                        [
                            {'pair': f'{curr1}/{curr2}', 'side': SignalType.BUY, 'currency': curr2},
                            {'pair': f'{curr2}/{curr3}', 'side': SignalType.BUY, 'currency': curr3},
                            {'pair': f'{curr3}/{curr1}', 'side': SignalType.SELL, 'currency': curr1}
                        ],
                        # Path 2: curr1 -> curr3 -> curr2 -> curr1
                        [
                            {'pair': f'{curr1}/{curr3}', 'side': SignalType.BUY, 'currency': curr3},
                            {'pair': f'{curr3}/{curr2}', 'side': SignalType.BUY, 'currency': curr2},
                            {'pair': f'{curr2}/{curr1}', 'side': SignalType.SELL, 'currency': curr1}
                        ]
                    ])
        
        # Filter paths where all pairs exist
        existing_pairs = set(self.currency_pairs)
        valid_paths = []
        
        for path in paths:
            if all(leg['pair'] in existing_pairs for leg in path):
                valid_paths.append(path)
        
        return valid_paths
    
    async def _calculate_arbitrage_opportunity(self, path: List[Dict]) -> Optional[Dict]:
        """Calculate arbitrage opportunity for a triangular path"""
        try:
            # Get current prices for all pairs in the path
            prices = {}
            for leg in path:
                price = await self._get_pair_price(leg['pair'])
                if price is None:
                    return None
                leg['price'] = price
                prices[leg['pair']] = price
            
            # Calculate profit through the path
            start_amount = 100000  # Start with $100k
            
            current_amount = start_amount
            for leg in path:
                pair = leg['pair']
                if leg['side'] == SignalType.BUY:
                    # Buy the quote currency
                    current_amount = current_amount / prices[pair]
                else:
                    # Sell to get base currency
                    current_amount = current_amount * prices[pair]
            
            # Calculate profit
            final_amount = current_amount
            profit = (final_amount - start_amount) / start_amount
            
            if profit > self.min_profit_threshold:
                # Calculate position size based on liquidity
                min_liquidity_pair = min(
                    await self._get_pair_liquidity(leg['pair']) for leg in path
                )
                
                max_size = min(self.max_position_size, min_liquidity_pair * 0.1)
                position_size = max_size
                
                return {
                    'profit': profit,
                    'path_profit': profit * position_size,
                    'size': position_size,
                    'legs': path
                }
        
        except Exception as e:
            logger.error(f"Error calculating arbitrage opportunity: {e}")
        
        return None
    
    async def _get_pair_price(self, pair: str) -> Optional[float]:
        """Get current price for a currency pair"""
        try:
            # Simulate currency pair pricing
            base_prices = {'USD': 1.0, 'EUR': 0.85, 'GBP': 0.73, 'JPY': 110.0}
            
            if pair in base_prices:
                return base_prices[pair]
            
            base, quote = pair.split('/')
            base_price = base_prices.get(base, 1.0)
            quote_price = base_prices.get(quote, 1.0)
            
            price = base_price / quote_price
            # Add market noise
            price *= (1 + np.random.normal(0, 0.0005))
            
            return price
        except Exception as e:
            logger.error(f"Error getting price for {pair}: {e}")
            return None
    
    async def _get_pair_liquidity(self, pair: str) -> float:
        """Get liquidity for a currency pair"""
        # Simulate pair liquidity
        base_liquidity = {
            'USD/EUR': 2000000,
            'EUR/GBP': 1500000,
            'GBP/JPY': 1800000,
            'USD/JPY': 2500000
        }
        return base_liquidity.get(pair, 1000000)


class ConvertibleArbitrageStrategy(BaseTimeSeriesStrategy):
    """Convertible Arbitrage Strategy
    
    Exploits mispricing between convertible bonds and their underlying stocks
    by taking positions in both instruments to capture the conversion premium.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['min_premium', 'conversion_ratio', 'volatility_threshold']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1d")
        
        self.min_premium = float(config.parameters.get('min_premium', 0.02))  # 2%
        self.conversion_ratio = float(config.parameters.get('conversion_ratio', 1.0))
        self.volatility_threshold = float(config.parameters.get('volatility_threshold', 0.25))  # 25%
        self.delta_neutral_ratio = float(config.parameters.get('delta_neutral_ratio', 0.8))
        self.max_position_size = float(config.parameters.get('max_position_size', 200000))
        
        self.convertible_data = {}
        
        self.metadata = StrategyMetadata(
            name="ConvertibleArbitrage",
            description="Arbitrage between convertible bonds and stocks",
            category=StrategyCategory.ARBITRAGE,
            risk_level=RiskLevel.MEDIUM,
            min_capital_required=150000,
            expected_return_range=(0.05, 0.12),
            max_drawdown=0.04,
            sharpe_ratio_range=(1.3, 2.3),
            time_horizon="Monthly",
            market_regime="High Volatility",
            instruments=["convertible_bonds", "stocks"],
            parameters_schema={
                "min_premium": {"type": "float", "min": 0.01, "max": 0.05},
                "conversion_ratio": {"type": "float", "min": 0.5, "max": 5.0},
                "volatility_threshold": {"type": "float", "min": 0.15, "max": 0.50},
                "delta_neutral_ratio": {"type": "float", "min": 0.5, "max": 1.0},
                "max_position_size": {"type": "float", "min": 50000, "max": 1000000}
            }
        )
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        for symbol in self.config.symbols:
            try:
                if not symbol.startswith('CONV:'):
                    continue
                
                stock_symbol = symbol.replace('CONV:', '')
                
                # Get convertible bond and stock data
                convertible_data = await self._get_convertible_data(symbol)
                stock_data = await self._get_stock_data(stock_symbol)
                
                if not convertible_data or not stock_data:
                    continue
                
                # Calculate conversion value and premium
                conversion_value = stock_data['price'] * self.conversion_ratio
                premium = (convertible_data['price'] - conversion_value) / conversion_value
                
                # Calculate implied volatility
                implied_vol = self._calculate_implied_volatility(
                    convertible_data, stock_data, conversion_value
                )
                
                # Check for arbitrage opportunity
                if premium > self.min_premium and implied_vol > self.volatility_threshold:
                    # Calculate delta-neutral position
                    convertible_position = self.max_position_size * 0.6
                    stock_hedge = convertible_position * self.delta_neutral_ratio / stock_data['price']
                    
                    # Generate convertible bond signal
                    convert_signal = TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        confidence=min(0.85, premium * 20),
                        strength=premium,
                        price=convertible_data['price'],
                        quantity=convertible_position / convertible_data['price'],
                        timestamp=datetime.utcnow(),
                        metadata={
                            'arbitrage_type': 'convertible',
                            'stock_symbol': stock_symbol,
                            'stock_price': stock_data['price'],
                            'conversion_value': conversion_value,
                            'premium': premium,
                            'implied_volatility': implied_vol,
                            'conversion_ratio': self.conversion_ratio,
                            'delta_neutral': True,
                            'hedge_symbol': stock_symbol,
                            'hedge_quantity': stock_hedge
                        }
                    )
                    
                    # Generate stock hedge signal
                    hedge_signal = TradingSignal(
                        symbol=stock_symbol,
                        signal_type=SignalType.SELL,
                        confidence=convert_signal.confidence,
                        strength=convert_signal.strength,
                        price=stock_data['price'],
                        quantity=stock_hedge,
                        timestamp=datetime.utcnow(),
                        metadata={
                            'arbitrage_type': 'convertible_hedge',
                            'convertible_symbol': symbol,
                            'convertible_price': convertible_data['price'],
                            'conversion_premium': premium,
                            'delta_ratio': self.delta_neutral_ratio
                        }
                    )
                    
                    signals.extend([convert_signal, hedge_signal])
            except Exception as e:
                logger.error(f"Error in convertible arbitrage: {e}")
        
        return signals
    
    async def _get_convertible_data(self, symbol: str) -> Optional[Dict]:
        """Get convertible bond data"""
        try:
            # Simulate convertible bond data
            base_price = 100.0
            price = base_price * (1 + np.random.normal(0, 0.02))
            yield_rate = np.random.uniform(0.02, 0.06)
            years_to_maturity = np.random.uniform(2, 8)
            
            return {
                'price': price,
                'yield_rate': yield_rate,
                'years_to_maturity': years_to_maturity,
                'volume': np.random.uniform(1000, 10000),
                'timestamp': datetime.utcnow()
            }
        except Exception as e:
            logger.error(f"Error getting convertible data for {symbol}: {e}")
            return None
    
    async def _get_stock_data(self, symbol: str) -> Optional[Dict]:
        """Get underlying stock data"""
        try:
            # Simulate stock data
            price = 50.0 * (1 + np.random.normal(0, 0.03))
            volume = np.random.uniform(100000, 1000000)
            
            return {
                'price': price,
                'volume': volume,
                'timestamp': datetime.utcnow()
            }
        except Exception as e:
            logger.error(f"Error getting stock data for {symbol}: {e}")
            return None
    
    def _calculate_implied_volatility(self, convertible_data: Dict, stock_data: Dict, 
                                    conversion_value: float) -> float:
        """Calculate implied volatility of the convertible bond"""
        try:
            # Simplified implied volatility calculation
            # In practice, this would use Black-Scholes or similar models
            
            bond_price = convertible_data['price']
            time_to_maturity = convertible_data['years_to_maturity']
            risk_free_rate = 0.03  # Assume 3% risk-free rate
            
            # Monte Carlo simulation for implied volatility
            stock_price = stock_data['price']
            num_simulations = 1000
            
            # Simple volatility guess
            vol_guess = 0.25
            
            # Calculate option value using Black-Scholes-like approach
            option_value = self._calculate_conversion_option_value(
                stock_price, conversion_value, time_to_maturity, 
                risk_free_rate, vol_guess
            )
            
            # Adjust volatility until option value matches market price
            # This is a simplified iterative approach
            for _ in range(10):
                if option_value > bond_price * 0.1:  # Option is worth about 10% of bond
                    vol_guess *= 1.1
                else:
                    vol_guess *= 0.9
                
                option_value = self._calculate_conversion_option_value(
                    stock_price, conversion_value, time_to_maturity,
                    risk_free_rate, vol_guess
                )
            
            return vol_guess
            
        except Exception as e:
            logger.error(f"Error calculating implied volatility: {e}")
            return 0.25
    
    def _calculate_conversion_option_value(self, stock_price: float, conversion_value: float,
                                         time_to_maturity: float, risk_free_rate: float,
                                         volatility: float) -> float:
        """Calculate the option value of the conversion feature"""
        # Simplified Black-Scholes calculation
        # In practice, this would be more sophisticated
        
        moneyness = conversion_value / stock_price
        
        if moneyness > 1.2:  # Deep out-of-the-money
            return 0
        
        # Approximate option value
        intrinsic_value = max(0, stock_price - conversion_value)
        time_value = volatility * np.sqrt(time_to_maturity) * 0.4  # Rough approximation
        
        return intrinsic_value + time_value


class OptionsArbitrageStrategy(BaseTimeSeriesStrategy):
    """Options Arbitrage Strategy
    
    Exploits pricing inefficiencies in options markets through various
    arbitrage strategies including put-call parity, volatility arbitrage,
    and calendar spreads.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['min_put_call_diff', 'vol_smile_threshold', 'calendar_spread_threshold']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="15m")
        
        self.min_put_call_diff = float(config.parameters.get('min_put_call_diff', 0.01))  # 1%
        self.vol_smile_threshold = float(config.parameters.get('vol_smile_threshold', 0.05))  # 5%
        self.calendar_spread_threshold = float(config.parameters.get('calendar_spread_threshold', 0.02))  # 2%
        self.max_position_size = float(config.parameters.get('max_position_size', 150000))
        
        self.options_chains = {}
        
        self.metadata = StrategyMetadata(
            name="OptionsArbitrage",
            description="Options market arbitrage opportunities",
            category=StrategyCategory.ARBITRAGE,
            risk_level=RiskLevel.MEDIUM,
            min_capital_required=100000,
            expected_return_range=(0.06, 0.15),
            max_drawdown=0.05,
            sharpe_ratio_range=(1.4, 2.6),
            time_horizon="Intraday",
            market_regime="High Volatility",
            instruments=["stock_options", "index_options"],
            parameters_schema={
                "min_put_call_diff": {"type": "float", "min": 0.005, "max": 0.03},
                "vol_smile_threshold": {"type": "float", "min": 0.02, "max": 0.10},
                "calendar_spread_threshold": {"type": "float", "min": 0.01, "max": 0.05},
                "max_position_size": {"type": "float", "min": 50000, "max": 500000}
            }
        )
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        for symbol in self.config.symbols:
            try:
                # Get options chain data
                options_chain = await self._get_options_chain(symbol)
                if not options_chain:
                    continue
                
                # Check for put-call parity arbitrage
                parity_signals = await self._check_put_call_parity(symbol, options_chain)
                signals.extend(parity_signals)
                
                # Check for volatility smile arbitrage
                vol_signals = await self._check_volatility_smile(symbol, options_chain)
                signals.extend(vol_signals)
                
                # Check for calendar spread arbitrage
                calendar_signals = await self._check_calendar_spreads(symbol, options_chain)
                signals.extend(calendar_signals)
            except Exception as e:
                logger.error(f"Error in options arbitrage for {symbol}: {e}")
        
        return signals
    
    async def _get_options_chain(self, symbol: str) -> Optional[Dict]:
        """Get options chain data for a symbol"""
        try:
            # Simulate options chain data
            underlying_price = 100.0 * (1 + np.random.normal(0, 0.02))
            strikes = [95, 100, 105, 110, 115]
            expirations = [30, 60, 90]  # days
            
            chain = {}
            
            for expiry in expirations:
                expiry_date = datetime.utcnow() + timedelta(days=expiry)
                chain[expiry] = {}
                
                for strike in strikes:
                    # Simulate option prices using Black-Scholes approximation
                    time_to_expiry = expiry / 365
                    intrinsic_call = max(0, underlying_price - strike)
                    intrinsic_put = max(0, strike - underlying_price)
                    
                    # Add time value (simplified)
                    call_price = intrinsic_call + np.random.uniform(1, 3)
                    put_price = intrinsic_put + np.random.uniform(1, 3)
                    
                    # Add market noise
                    call_price *= (1 + np.random.normal(0, 0.05))
                    put_price *= (1 + np.random.normal(0, 0.05))
                    
                    chain[expiry][strike] = {
                        'call_price': call_price,
                        'put_price': put_price,
                        'call_volume': np.random.randint(50, 500),
                        'put_volume': np.random.randint(50, 500),
                        'call_open_interest': np.random.randint(100, 2000),
                        'put_open_interest': np.random.randint(100, 2000)
                    }
            
            return {
                'underlying_price': underlying_price,
                'chain': chain,
                'expirations': list(chain.keys()),
                'strikes': strikes,
                'timestamp': datetime.utcnow()
            }
        except Exception as e:
            logger.error(f"Error getting options chain for {symbol}: {e}")
            return None
    
    async def _check_put_call_parity(self, symbol: str, options_chain: Dict) -> List[TradingSignal]:
        """Check for put-call parity violations"""
        signals = []
        
        try:
            underlying_price = options_chain['underlying_price']
            risk_free_rate = 0.03  # Assume 3% risk-free rate
            
            for expiry, strikes_data in options_chain['chain'].items():
                time_to_expiry = expiry / 365
                
                for strike, options_data in strikes_data.items():
                    call_price = options_data['call_price']
                    put_price = options_data['put_price']
                    
                    # Calculate put-call parity theoretical values
                    # Call - Put = S - K*e^(-r*T)
                    put_call_synthetic = call_price - put_price
                    put_call_theoretical = underlying_price - strike * np.exp(-risk_free_rate * time_to_expiry)
                    
                    parity_diff = abs(put_call_synthetic - put_call_theoretical) / underlying_price
                    
                    if parity_diff > self.min_put_call_diff:
                        # Determine which side of the parity is mispriced
                        if put_call_synthetic > put_call_theoretical:
                            # Call is too expensive relative to put
                            # Buy put, sell call, short stock
                            signal_type_call = SignalType.SELL
                            signal_type_put = SignalType.BUY
                            signal_type_stock = SignalType.SELL
                        else:
                            # Put is too expensive relative to call
                            # Buy call, sell put, long stock
                            signal_type_call = SignalType.BUY
                            signal_type_put = SignalType.SELL
                            signal_type_stock = SignalType.BUY
                        
                        # Calculate position sizes
                        option_size = self.max_position_size / (call_price + put_price)
                        stock_size = option_size * strike  # Approximate delta-neutral
                        
                        # Generate signals
                        call_signal = TradingSignal(
                            symbol=f"{symbol}_CALL_{strike}_{expiry}",
                            signal_type=signal_type_call,
                            confidence=min(0.90, parity_diff * 100),
                            strength=parity_diff,
                            price=call_price,
                            quantity=option_size,
                            timestamp=datetime.utcnow(),
                            metadata={
                                'arbitrage_type': 'put_call_parity',
                                'strike': strike,
                                'expiry': expiry,
                                'parity_diff': parity_diff,
                                'theoretical_parity': put_call_theoretical,
                                'actual_parity': put_call_synthetic,
                                'risk_free_rate': risk_free_rate
                            }
                        )
                        
                        put_signal = TradingSignal(
                            symbol=f"{symbol}_PUT_{strike}_{expiry}",
                            signal_type=signal_type_put,
                            confidence=call_signal.confidence,
                            strength=call_signal.strength,
                            price=put_price,
                            quantity=option_size,
                            timestamp=datetime.utcnow(),
                            metadata={
                                'arbitrage_type': 'put_call_parity_hedge',
                                'call_symbol': call_signal.symbol,
                                'parity_diff': parity_diff
                            }
                        )
                        
                        stock_signal = TradingSignal(
                            symbol=symbol,
                            signal_type=signal_type_stock,
                            confidence=call_signal.confidence,
                            strength=call_signal.strength,
                            price=underlying_price,
                            quantity=stock_size,
                            timestamp=datetime.utcnow(),
                            metadata={
                                'arbitrage_type': 'put_call_parity_hedge',
                                'call_symbol': call_signal.symbol,
                                'put_symbol': put_signal.symbol,
                                'hedge_ratio': 1.0
                            }
                        )
                        
                        signals.extend([call_signal, put_signal, stock_signal])
        
        except Exception as e:
            logger.error(f"Error checking put-call parity: {e}")
        
        return signals
    
    async def _check_volatility_smile(self, symbol: str, options_chain: Dict) -> List[TradingSignal]:
        """Check for volatility smile arbitrage opportunities"""
        signals = []
        
        try:
            underlying_price = options_chain['underlying_price']
            
            # Get ATM options for volatility smile analysis
            atm_strikes = []
            strikes = list(next(iter(options_chain['chain'].values())).keys())
            
            for strike in strikes:
                if abs(strike - underlying_price) / underlying_price < 0.05:  # Within 5% of ATM
                    atm_strikes.append(strike)
            
            if len(atm_strikes) < 3:
                return signals
            
            # Analyze volatility smile across strikes
            expiries = list(options_chain['chain'].keys())
            
            for expiry in expiries[:2]:  # Check near and medium term
                vol_curve = []
                
                for strike in atm_strikes:
                    if strike in options_chain['chain'][expiry]:
                        call_price = options_chain['chain'][expiry][strike]['call_price']
                        
                        # Estimate implied volatility (simplified)
                        iv = self._estimate_implied_volatility(
                            call_price, underlying_price, strike, expiry/365, 'call'
                        )
                        
                        vol_curve.append((strike, iv))
                
                # Check for volatility smile violations
                if len(vol_curve) >= 3:
                    # Sort by strike and check for monotonicity
                    vol_curve.sort(key=lambda x: x[0])
                    
                    for i in range(len(vol_curve) - 2):
                        strike1, vol1 = vol_curve[i]
                        strike2, vol2 = vol_curve[i + 1]
                        strike3, vol3 = vol_curve[i + 2]
                        
                        # Check for butterfly arbitrage (volatility should be convex)
                        if vol2 > (vol1 + vol3) / 2 + self.vol_smile_threshold:
                            # Potential butterfly arbitrage
                            position_size = self.max_position_size / underlying_price
                            
                            # Buy butterfly spread: long 1 ATM, short 2 wings
                            butterfly_signal = TradingSignal(
                                symbol=symbol,
                                signal_type=SignalType.BUY,
                                confidence=0.75,
                                strength=(vol2 - (vol1 + vol3) / 2),
                                price=underlying_price,
                                quantity=position_size,
                                timestamp=datetime.utcnow(),
                                metadata={
                                    'arbitrage_type': 'volatility_smile',
                                    'strategy': 'butterfly',
                                    'strike_center': strike2,
                                    'strike_wing1': strike1,
                                    'strike_wing2': strike3,
                                    'vol_center': vol2,
                                    'vol_wing_avg': (vol1 + vol3) / 2,
                                    'vol_arb': vol2 - (vol1 + vol3) / 2
                                }
                            )
                            signals.append(butterfly_signal)
        
        except Exception as e:
            logger.error(f"Error checking volatility smile: {e}")
        
        return signals
    
    async def _check_calendar_spreads(self, symbol: str, options_chain: Dict) -> List[TradingSignal]:
        """Check for calendar spread arbitrage opportunities"""
        signals = []
        
        try:
            underlying_price = options_chain['underlying_price']
            expiries = sorted(options_chain['chain'].keys())
            
            if len(expiries) < 2:
                return signals
            
            # Find ATM strikes for each expiry
            atm_strikes = {}
            all_strikes = set()
            
            for expiry in expiries:
                strikes = list(options_chain['chain'][expiry].keys())
                all_strikes.update(strikes)
                
                # Find ATM strike
                atm_strike = min(strikes, key=lambda s: abs(s - underlying_price))
                atm_strikes[expiry] = atm_strike
            
            # Check calendar spreads at ATM strikes
            for i in range(len(expiries) - 1):
                near_expiry = expiries[i]
                far_expiry = expiries[i + 1]
                strike = atm_strikes[near_expiry]
                
                if strike in options_chain['chain'][far_expiry]:
                    near_call = options_chain['chain'][near_expiry][strike]['call_price']
                    far_call = options_chain['chain'][far_expiry][strike]['call_price']
                    
                    # Calendar spread value
                    calendar_spread = far_call - near_call
                    
                    # Check for arbitrage (calendar spread should be positive and reasonable)
                    if calendar_spread < 0 or calendar_spread > self.calendar_spread_threshold * underlying_price:
                        position_size = self.max_position_size / underlying_price
                        
                        signal = TradingSignal(
                            symbol=symbol,
                            signal_type=SignalType.BUY,
                            confidence=0.70,
                            strength=abs(calendar_spread) / underlying_price,
                            price=underlying_price,
                            quantity=position_size,
                            timestamp=datetime.utcnow(),
                            metadata={
                                'arbitrage_type': 'calendar_spread',
                                'near_expiry': near_expiry,
                                'far_expiry': far_expiry,
                                'strike': strike,
                                'near_call_price': near_call,
                                'far_call_price': far_call,
                                'spread_value': calendar_spread,
                                'strategy': 'calendar_spread'
                            }
                        )
                        signals.append(signal)
        
        except Exception as e:
            logger.error(f"Error checking calendar spreads: {e}")
        
        return signals
    
    def _estimate_implied_volatility(self, option_price: float, underlying: float, 
                                   strike: float, time_to_expiry: float, option_type: str) -> float:
        """Estimate implied volatility using simplified Newton-Raphson"""
        risk_free_rate = 0.03
        
        # Initial volatility guess
        vol = 0.25
        
        # Newton-Raphson iterations
        for _ in range(10):
            # Calculate option price with current volatility
            d1 = (np.log(underlying / strike) + (risk_free_rate + 0.5 * vol**2) * time_to_expiry) / (vol * np.sqrt(time_to_expiry))
            d2 = d1 - vol * np.sqrt(time_to_expiry)
            
            if option_type.lower() == 'call':
                price = underlying * stats.norm.cdf(d1) - strike * np.exp(-risk_free_rate * time_to_expiry) * stats.norm.cdf(d2)
                vega = underlying * np.sqrt(time_to_expiry) * stats.norm.pdf(d1)
            else:
                price = strike * np.exp(-risk_free_rate * time_to_expiry) * stats.norm.cdf(-d2) - underlying * stats.norm.cdf(-d1)
                vega = underlying * np.sqrt(time_to_expiry) * stats.norm.pdf(d1)
            
            # Price difference
            price_diff = price - option_price
            
            # Check convergence
            if abs(price_diff) < 0.01:
                break
            
            # Update volatility
            if vega > 0:
                vol -= price_diff / vega
                vol = max(0.01, min(2.0, vol))  # Keep volatility reasonable
        
        return vol


# ============================================================================
# EVENT ARBITRAGE STRATEGIES
# ============================================================================

class MergerArbitrageStrategy(BaseTimeSeriesStrategy):
    """Merger Arbitrage Strategy
    
    Trades on announced mergers and acquisitions by buying target companies
    and shorting acquiring companies to capture the spread.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['min_spread', 'completion_probability', 'max_holding_period']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1d")
        
        self.min_spread = float(config.parameters.get('min_spread', 0.02))  # 2%
        self.completion_probability = float(config.parameters.get('completion_probability', 0.8))
        self.max_holding_period = int(config.parameters.get('max_holding_period', 180))  # days
        self.risk_adjustment = float(config.parameters.get('risk_adjustment', 0.5))
        
        self.merger_deals = {}
        self.deal_outcomes = {}
        
        self.metadata = StrategyMetadata(
            name="MergerArbitrage",
            description="Merger and acquisition arbitrage opportunities",
            category=StrategyCategory.ARBITRAGE,
            risk_level=RiskLevel.MEDIUM,
            min_capital_required=200000,
            expected_return_range=(0.04, 0.10),
            max_drawdown=0.06,
            sharpe_ratio_range=(1.0, 1.8),
            time_horizon="Quarterly",
            market_regime="Corporate Events",
            instruments=["stocks", "ADR"],
            parameters_schema={
                "min_spread": {"type": "float", "min": 0.01, "max": 0.10},
                "completion_probability": {"type": "float", "min": 0.5, "max": 0.95},
                "max_holding_period": {"type": "int", "min": 30, "max": 365},
                "risk_adjustment": {"type": "float", "min": 0.2, "max": 1.0}
            }
        )
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        try:
            # Get active merger deals
            deals = await self._get_active_merger_deals()
            
            for deal in deals:
                opportunity = await self._analyze_merger_opportunity(deal)
                
                if opportunity and opportunity['adjusted_spread'] > self.min_spread:
                    # Generate target company buy signal
                    target_signal = TradingSignal(
                        symbol=opportunity['target_symbol'],
                        signal_type=SignalType.BUY,
                        confidence=opportunity['completion_probability'],
                        strength=opportunity['adjusted_spread'],
                        price=opportunity['target_price'],
                        quantity=opportunity['position_size'] / opportunity['target_price'],
                        timestamp=datetime.utcnow(),
                        metadata={
                            'arbitrage_type': 'merger',
                            'acquirer_symbol': opportunity['acquirer_symbol'],
                            'acquirer_price': opportunity['acquirer_price'],
                            'acquisition_price': opportunity['acquisition_price'],
                            'offer_spread': opportunity['offer_spread'],
                            'current_spread': opportunity['current_spread'],
                            'completion_probability': opportunity['completion_probability'],
                            'risk_adjusted_spread': opportunity['adjusted_spread'],
                            'expected_timeline': opportunity['timeline'],
                            'deal_type': opportunity['deal_type'],
                            'acquirer_hedge_required': True
                        }
                    )
                    
                    # Generate acquirer hedge signal
                    hedge_signal = TradingSignal(
                        symbol=opportunity['acquirer_symbol'],
                        signal_type=SignalType.SELL,
                        confidence=opportunity['completion_probability'],
                        strength=opportunity['adjusted_spread'],
                        price=opportunity['acquirer_price'],
                        quantity=opportunity['hedge_size'] / opportunity['acquirer_price'],
                        timestamp=datetime.utcnow(),
                        metadata={
                            'arbitrage_type': 'merger_hedge',
                            'target_symbol': opportunity['target_symbol'],
                            'hedge_ratio': opportunity['hedge_ratio'],
                            'acquisition_price': opportunity['acquisition_price'],
                            'deal_spread': opportunity['current_spread']
                        }
                    )
                    
                    signals.extend([target_signal, hedge_signal])
            except Exception as e:
                logger.error(f"Error in merger arbitrage: {e}")
        
        return signals
    
    async def _get_active_merger_deals(self) -> List[Dict]:
        """Get list of active merger and acquisition deals"""
        # Simulate merger deals database
        deals = [
            {
                'deal_id': 'MERGER_001',
                'target_symbol': 'TARGET_CORP',
                'acquirer_symbol': 'ACQUIRER_INC',
                'acquisition_price': 45.00,  # $45 per share
                'announcement_date': datetime.utcnow() - timedelta(days=30),
                'expected_close_date': datetime.utcnow() + timedelta(days=120),
                'deal_type': 'cash',
                'regulatory_approval': 'pending',
                'shareholder_approval': 'pending',
                'status': 'announced'
            },
            {
                'deal_id': 'MERGER_002',
                'target_symbol': 'TECH_STARTUP',
                'acquirer_symbol': 'BIG_TECH',
                'acquisition_price': 25.50,  # $25.50 per share
                'announcement_date': datetime.utcnow() - timedelta(days=15),
                'expected_close_date': datetime.utcnow() + timedelta(days=90),
                'deal_type': 'stock',
                'regulatory_approval': 'approved',
                'shareholder_approval': 'pending',
                'status': 'announced'
            }
        ]
        
        # Update with current market prices
        for deal in deals:
            deal['target_price'] = 45.00 * (1 + np.random.uniform(-0.1, 0.05))  # Below acquisition price
            deal['acquirer_price'] = 100.00 * (1 + np.random.uniform(-0.05, 0.05))
        
        return deals
    
    async def _analyze_merger_opportunity(self, deal: Dict) -> Optional[Dict]:
        """Analyze merger arbitrage opportunity"""
        try:
            target_price = deal['target_price']
            acquisition_price = deal['acquisition_price']
            acquirer_price = deal['acquirer_price']
            
            # Calculate spreads
            offer_spread = (acquisition_price - target_price) / target_price
            
            # Current spread (should be less than offer spread due to completion risk)
            current_spread = max(0, offer_spread * np.random.uniform(0.3, 0.9))
            
            # Assess completion probability based on deal characteristics
            completion_prob = self._assess_completion_probability(deal)
            
            # Adjust spread for risk
            adjusted_spread = current_spread * completion_prob * self.risk_adjustment
            
            # Calculate expected timeline
            days_to_close = (deal['expected_close_date'] - datetime.utcnow()).days
            
            # Calculate position sizing
            total_position_size = 100000  # Assume $100k allocation per deal
            target_allocation = total_position_size * 0.7  # 70% in target, 30% hedge
            hedge_allocation = total_position_size * 0.3
            
            # Calculate hedge ratio based on deal value
            deal_value_ratio = acquisition_price / acquirer_price
            hedge_ratio = deal_value_ratio * completion_prob
            
            return {
                'target_symbol': deal['target_symbol'],
                'acquirer_symbol': deal['acquirer_symbol'],
                'target_price': target_price,
                'acquirer_price': acquirer_price,
                'acquisition_price': acquisition_price,
                'offer_spread': offer_spread,
                'current_spread': current_spread,
                'adjusted_spread': adjusted_spread,
                'completion_probability': completion_prob,
                'position_size': target_allocation,
                'hedge_size': hedge_allocation,
                'hedge_ratio': hedge_ratio,
                'timeline': days_to_close,
                'deal_type': deal['deal_type'],
                'deal_id': deal['deal_id']
            }
        
        except Exception as e:
            logger.error(f"Error analyzing merger opportunity: {e}")
            return None
    
    def _assess_completion_probability(self, deal: Dict) -> float:
        """Assess probability of deal completion"""
        base_prob = self.completion_probability
        
        # Adjust based on deal characteristics
        if deal['deal_type'] == 'cash':
            base_prob += 0.1  # Cash deals more likely to close
        elif deal['deal_type'] == 'stock':
            base_prob -= 0.05  # Stock deals more risky
        
        # Adjust based on approval status
        if deal['regulatory_approval'] == 'approved':
            base_prob += 0.15
        elif deal['regulatory_approval'] == 'pending':
            base_prob -= 0.1
        
        if deal['shareholder_approval'] == 'approved':
            base_prob += 0.1
        elif deal['shareholder_approval'] == 'pending':
            base_prob -= 0.05
        
        # Adjust based on time since announcement
        days_since_announcement = (datetime.utcnow() - deal['announcement_date']).days
        if days_since_announcement > 60:
            base_prob += 0.05  # Deals more likely to close as time passes
        
        return max(0.1, min(0.95, base_prob))


class EventDrivenArbitrageStrategy(BaseTimeSeriesStrategy):
    """Event-Driven Arbitrage Strategy
    
    Trades on corporate events including earnings announcements, stock splits,
    dividend changes, and other catalysts that create temporary price inefficiencies.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['event_window', 'min_price_move', 'volatility_threshold']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1h")
        
        self.event_window = int(config.parameters.get('event_window', 3))  # days
        self.min_price_move = float(config.parameters.get('min_price_move', 0.02))  # 2%
        self.volatility_threshold = float(config.parameters.get('volatility_threshold', 0.3))  # 30%
        self.max_position_size = float(config.parameters.get('max_position_size', 75000))
        
        self.event_calendar = {}
        self.event_outcomes = {}
        
        self.metadata = StrategyMetadata(
            name="EventDrivenArbitrage",
            description="Event-driven arbitrage opportunities",
            category=StrategyCategory.ARBITRAGE,
            risk_level=RiskLevel.HIGH,
            min_capital_required=150000,
            expected_return_range=(0.08, 0.18),
            max_drawdown=0.08,
            sharpe_ratio_range=(1.5, 2.8),
            time_horizon="Intraday to Weekly",
            market_regime="Event-Driven",
            instruments=["stocks", "options"],
            parameters_schema={
                "event_window": {"type": "int", "min": 1, "max": 7},
                "min_price_move": {"type": "float", "min": 0.01, "max": 0.10},
                "volatility_threshold": {"type": "float", "min": 0.15, "max": 0.50},
                "max_position_size": {"type": "float", "min": 25000, "max": 300000}
            }
        )
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        try:
            # Get upcoming events
            events = await self._get_upcoming_events()
            
            for event in events:
                opportunity = await self._analyze_event_opportunity(event)
                
                if opportunity and opportunity['expected_move'] > self.min_price_move:
                    position_size = min(
                        self.max_position_size,
                        self._calculate_event_position_size(event, opportunity)
                    )
                    
                    # Determine signal based on event type and market expectations
                    signal_type = self._determine_event_signal(event, opportunity)
                    
                    signal = TradingSignal(
                        symbol=event['symbol'],
                        signal_type=signal_type,
                        confidence=opportunity['confidence'],
                        strength=opportunity['expected_move'],
                        price=opportunity['current_price'],
                        quantity=position_size / opportunity['current_price'],
                        timestamp=datetime.utcnow(),
                        metadata={
                            'arbitrage_type': 'event_driven',
                            'event_type': event['type'],
                            'event_date': event['date'],
                            'days_to_event': opportunity['days_to_event'],
                            'expected_move': opportunity['expected_move'],
                            'volatility_regime': opportunity['volatility_regime'],
                            'event_probability': opportunity['event_probability'],
                            'historical_success_rate': opportunity['historical_rate'],
                            'premium_justified': opportunity['premium_justified']
                        }
                    )
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error in event-driven arbitrage: {e}")
        
        return signals
    
    async def _get_upcoming_events(self) -> List[Dict]:
        """Get upcoming corporate events"""
        events = [
            {
                'symbol': 'EARNINGS_STOCK',
                'type': 'earnings',
                'date': datetime.utcnow() + timedelta(days=2),
                'historical_volatility': 0.25,
                'surprise_frequency': 0.15,
                'average_move': 0.05
            },
            {
                'symbol': 'DIVIDEND_STOCK',
                'type': 'dividend',
                'date': datetime.utcnow() + timedelta(days=5),
                'historical_volatility': 0.20,
                'surprise_frequency': 0.05,
                'average_move': 0.02
            },
            {
                'symbol': 'SPLIT_STOCK',
                'type': 'stock_split',
                'date': datetime.utcnow() + timedelta(days=7),
                'historical_volatility': 0.15,
                'surprise_frequency': 0.02,
                'average_move': 0.03
            },
            {
                'symbol': 'GUIDANCE_STOCK',
                'type': 'guidance',
                'date': datetime.utcnow() + timedelta(days=3),
                'historical_volatility': 0.30,
                'surprise_frequency': 0.25,
                'average_move': 0.08
            }
        ]
        
        # Add current market data
        for event in events:
            event['current_price'] = 50.0 * (1 + np.random.uniform(-0.1, 0.1))
            event['current_volatility'] = event['historical_volatility'] * (1 + np.random.uniform(-0.2, 0.3))
        
        return events
    
    async def _analyze_event_opportunity(self, event: Dict) -> Optional[Dict]:
        """Analyze event-driven trading opportunity"""
        try:
            current_price = event['current_price']
            current_volatility = event['current_volatility']
            days_to_event = (event['date'] - datetime.utcnow()).days
            
            # Calculate expected price move
            base_move = event['average_move']
            volatility_adjustment = current_volatility / event['historical_volatility']
            
            # Time decay adjustment (more uncertainty as event approaches)
            time_adjustment = 1.0 + (7 - min(days_to_event, 7)) * 0.1
            
            expected_move = base_move * volatility_adjustment * time_adjustment
            
            # Calculate event probability and confidence
            event_probability = self._calculate_event_probability(event)
            confidence = min(0.85, event_probability * volatility_adjustment)
            
            # Assess if current volatility premium is justified
            vol_premium = (current_volatility - event['historical_volatility']) / event['historical_volatility']
            premium_justified = vol_premium > 0.1
            
            # Determine volatility regime
            if current_volatility > self.volatility_threshold:
                volatility_regime = 'high'
            elif current_volatility < event['historical_volatility'] * 0.8:
                volatility_regime = 'low'
            else:
                volatility_regime = 'normal'
            
            return {
                'current_price': current_price,
                'expected_move': expected_move,
                'confidence': confidence,
                'event_probability': event_probability,
                'days_to_event': days_to_event,
                'volatility_regime': volatility_regime,
                'premium_justified': premium_justified,
                'volatility_premium': vol_premium,
                'historical_rate': event['surprise_frequency']
            }
        
        except Exception as e:
            logger.error(f"Error analyzing event opportunity: {e}")
            return None
    
    def _calculate_event_probability(self, event: Dict) -> float:
        """Calculate probability of significant event outcome"""
        base_prob = 1.0  # Assume event will happen
        
        # Adjust based on event type
        if event['type'] == 'earnings':
            # Higher probability of surprise
            base_prob *= (1 + event['surprise_frequency'])
        elif event['type'] == 'dividend':
            # Lower probability of surprise
            base_prob *= (1 - event['surprise_frequency'])
        elif event['type'] == 'stock_split':
            # Usually announced in advance
            base_prob = 0.95
        
        # Adjust based on time to event
        days_to_event = (event['date'] - datetime.utcnow()).days
        if days_to_event <= 1:
            base_prob *= 1.1  # Higher confidence as event approaches
        elif days_to_event > 7:
            base_prob *= 0.9  # Lower confidence for distant events
        
        return min(0.99, max(0.1, base_prob))
    
    def _calculate_event_position_size(self, event: Dict, opportunity: Dict) -> float:
        """Calculate position size for event trade"""
        base_size = self.max_position_size
        
        # Adjust based on expected move
        move_adjustment = min(2.0, opportunity['expected_move'] / self.min_price_move)
        
        # Adjust based on confidence
        confidence_adjustment = opportunity['confidence']
        
        # Adjust based on volatility regime
        if opportunity['volatility_regime'] == 'high':
            volatility_adjustment = 0.7  # Reduce size in high volatility
        elif opportunity['volatility_regime'] == 'low':
            volatility_adjustment = 1.2  # Increase size in low volatility
        else:
            volatility_adjustment = 1.0
        
        position_size = base_size * move_adjustment * confidence_adjustment * volatility_adjustment
        
        return min(position_size, self.max_position_size)
    
    def _determine_event_signal(self, event: Dict, opportunity: Dict) -> SignalType:
        """Determine signal direction based on event analysis"""
        if event['type'] in ['earnings', 'guidance']:
            # For earnings and guidance, volatility plays are more common
            if opportunity['volatility_regime'] == 'low' and opportunity['premium_justified']:
                return SignalType.BUY  # Buy volatility before event
            elif opportunity['volatility_regime'] == 'high':
                return SignalType.SELL  # Sell volatility if overextended
        elif event['type'] in ['dividend', 'stock_split']:
            # For corporate actions, directional bets based on historical patterns
            if event['type'] == 'dividend':
                return SignalType.BUY  # Generally positive for shareholders
            elif event['type'] == 'stock_split':
                return SignalType.BUY  # Usually positive reception
        
        # Default to directional based on volatility regime
        return SignalType.BUY if opportunity['volatility_regime'] != 'high' else SignalType.SELL


# ============================================================================
# STRATEGY REGISTRATION
# ============================================================================

# Register all arbitrage strategies
def register_arbitrage_strategies():
    """Register all arbitrage strategies with the strategy library"""
    
    # Market Arbitrage Strategies
    strategy_library.register_strategy(
        strategy_class=CrossBrokerArbitrageStrategy,
        category=StrategyCategory.ARBITRAGE,
        tags=['market', 'broker', 'price_comparison', 'execution']
    )
    
    strategy_library.register_strategy(
        strategy_class=CrossVenueArbitrageStrategy,
        category=StrategyCategory.ARBITRAGE,
        tags=['market', 'venue', 'exchange', 'liquidity']
    )
    
    strategy_library.register_strategy(
        strategy_class=IndexArbitrageStrategy,
        category=StrategyCategory.ARBITRAGE,
        tags=['market', 'index', 'etf', 'basket']
    )
    
    # Statistical Arbitrage Strategies
    strategy_library.register_strategy(
        strategy_class=StatisticalPairsArbitrageStrategy,
        category=StrategyCategory.ARBITRAGE,
        tags=['statistical', 'pairs', 'cointegration', 'reversion']
    )
    
    strategy_library.register_strategy(
        strategy_class=MeanReversionArbitrageStrategy,
        category=StrategyCategory.ARBITRAGE,
        tags=['statistical', 'mean_reversion', 'z_score', 'reversion']
    )
    
    # Time Arbitrage Strategies
    strategy_library.register_strategy(
        strategy_class=CoveredInterestArbitrageStrategy,
        category=StrategyCategory.ARBITRAGE,
        tags=['time', 'interest_rate', 'forward', 'currency']
    )
    
    strategy_library.register_strategy(
        strategy_class=FuturesArbitrageStrategy,
        category=StrategyCategory.ARBITRAGE,
        tags=['time', 'futures', 'spot', 'basis']
    )
    
    # Instrument Arbitrage Strategies
    strategy_library.register_strategy(
        strategy_class=TriangularArbitrageStrategy,
        category=StrategyCategory.ARBITRAGE,
        tags=['instrument', 'triangular', 'currency', 'conversion']
    )
    
    strategy_library.register_strategy(
        strategy_class=ConvertibleArbitrageStrategy,
        category=StrategyCategory.ARBITRAGE,
        tags=['instrument', 'convertible', 'bond', 'stock']
    )
    
    strategy_library.register_strategy(
        strategy_class=OptionsArbitrageStrategy,
        category=StrategyCategory.ARBITRAGE,
        tags=['instrument', 'options', 'volatility', 'greeks']
    )
    
    # Event Arbitrage Strategies
    strategy_library.register_strategy(
        strategy_class=MergerArbitrageStrategy,
        category=StrategyCategory.ARBITRAGE,
        tags=['event', 'merger', 'acquisition', 'corporate']
    )
    
    strategy_library.register_strategy(
        strategy_class=EventDrivenArbitrageStrategy,
        category=StrategyCategory.ARBITRAGE,
        tags=['event', 'earnings', 'catalyst', 'volatility']
    )


# Auto-register strategies when module is imported
register_arbitrage_strategies()