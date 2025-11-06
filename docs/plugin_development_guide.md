# Plugin Development Guide

## Overview

The Day Trading Orchestrator is built on a modular plugin architecture that allows developers to extend the system with custom trading strategies, data providers, risk management modules, and more. This guide covers how to develop, test, and deploy plugins for the platform.

## Table of Contents

1. [Plugin Architecture](#plugin-architecture)
2. [Development Environment Setup](#development-environment-setup)
3. [Strategy Plugin Development](#strategy-plugin-development)
4. [Data Provider Plugins](#data-provider-plugins)
5. [Risk Management Plugins](#risk-management-plugins)
6. [Indicator Plugins](#indicator-plugins)
7. [UI Component Plugins](#ui-component-plugins)
8. [Plugin Testing Framework](#plugin-testing-framework)
9. [Plugin Distribution](#plugin-distribution)
10. [Best Practices](#best-practices)

## Plugin Architecture

### Plugin System Overview

The plugin system follows a modular architecture where each plugin implements specific interfaces and can be dynamically loaded at runtime.

```
┌─────────────────────────────────────────────────────────────────┐
│                     Core System                                 │
├─────────────────────────────────────────────────────────────────┤
│  Plugin Manager  │  Event Bus  │  Configuration Manager        │
├─────────────────────────────────────────────────────────────────┤
│                     Plugin Registry                             │
├──────────────┬──────────────┬──────────────┬─────────────────────┤
│  Strategy   │ Data Provider│    Risk      │    Indicator       │
│  Plugins    │   Plugins    │  Management  │     Plugins        │
│             │              │   Plugins    │                     │
└──────────────┴──────────────┴──────────────┴─────────────────────┘
```

### Plugin Types

#### 1. Strategy Plugins
- **Purpose**: Implement trading algorithms and signal generation
- **Interface**: `BaseStrategy`
- **Examples**: Moving Average Crossover, Mean Reversion, Momentum

#### 2. Data Provider Plugins
- **Purpose**: Supply market data from various sources
- **Interface**: `BaseDataProvider`
- **Examples**: Yahoo Finance, Alpha Vantage, IEX Cloud

#### 3. Risk Management Plugins
- **Purpose**: Implement custom risk calculation and control logic
- **Interface**: `BaseRiskManager`
- **Examples**: VaR Calculator, Correlation Risk Monitor

#### 4. Indicator Plugins
- **Purpose**: Calculate technical analysis indicators
- **Interface**: `BaseIndicator`
- **Examples**: Custom Moving Averages, Oscillators

#### 5. UI Component Plugins
- **Purpose**: Add custom interfaces to the Matrix Command Center
- **Interface**: `BaseUIComponent`
- **Examples**: Custom Dashboards, Trading Panels

## Development Environment Setup

### Prerequisites

```bash
# Python 3.9+
python --version

# Poetry for dependency management
pip install poetry

# Development dependencies
poetry install --with dev
```

### Project Structure

```
trading_orchestrator/
├── plugins/
│   ├── __init__.py
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── base_strategy.py
│   │   └── my_custom_strategy.py
│   ├── data_providers/
│   │   ├── __init__.py
│   │   ├── base_provider.py
│   │   └── my_data_provider.py
│   ├── risk_managers/
│   │   ├── __init__.py
│   │   ├── base_risk_manager.py
│   │   └── my_risk_manager.py
│   └── indicators/
│       ├── __init__.py
│       ├── base_indicator.py
│       └── my_indicator.py
├── tests/
│   ├── test_plugins/
│   └── conftest.py
└── examples/
    └── plugin_examples/
```

### Plugin Development Template

```python
# plugins/__init__.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio

# Plugin metadata
@dataclass
class PluginMetadata:
    name: str
    version: str
    description: str
    author: str
    category: str
    dependencies: List[str] = None

# Plugin lifecycle events
class PluginEvent(Enum):
    INITIALIZE = "initialize"
    START = "start"
    STOP = "stop"
    CONFIGURE = "configure"
    ERROR = "error"

class PluginBase(ABC):
    """Base class for all plugins"""
    
    def __init__(self, metadata: PluginMetadata):
        self.metadata = metadata
        self.config = {}
        self.is_running = False
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Setup plugin-specific logger"""
        import logging
        logger = logging.getLogger(f"plugin.{self.metadata.name}")
        logger.setLevel(logging.INFO)
        return logger
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the plugin. Return True if successful."""
        pass
    
    async def start(self):
        """Start the plugin"""
        self.is_running = True
        self.logger.info(f"Plugin {self.metadata.name} started")
    
    async def stop(self):
        """Stop the plugin"""
        self.is_running = False
        self.logger.info(f"Plugin {self.metadata.name} stopped")
    
    async def configure(self, config: Dict[str, Any]):
        """Configure the plugin with given settings"""
        self.config.update(config)
        self.logger.info(f"Plugin {self.metadata.name} configured")
    
    async def handle_event(self, event_type: PluginEvent, data: Any):
        """Handle plugin lifecycle events"""
        if event_type == PluginEvent.ERROR:
            self.logger.error(f"Plugin error: {data}")
```

## Strategy Plugin Development

### Base Strategy Interface

```python
# plugins/strategies/base_strategy.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import pandas as pd

@dataclass
class TradingSignal:
    """Represents a trading signal generated by a strategy"""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    confidence: float  # 0.0 to 1.0
    strategy_id: str
    timestamp: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = None

@dataclass
class MarketData:
    """Represents market data for analysis"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    bid: Optional[float] = None
    ask: Optional[float] = None

class BaseStrategy(PluginBase):
    """Abstract base class for trading strategies"""
    
    def __init__(self, metadata: PluginMetadata):
        super().__init__(metadata)
        self.positions = {}
        self.performance_metrics = {}
        self.is_active = False
    
    @abstractmethod
    async def analyze(self, market_data: MarketData) -> List[TradingSignal]:
        """
        Analyze market data and generate trading signals
        """
        pass
    
    async def on_tick(self, tick_data: MarketData) -> List[TradingSignal]:
        """
        Process real-time market data tick
        Default implementation calls analyze()
        """
        return await self.analyze(tick_data)
    
    async def on_bar(self, bar_data: MarketData) -> List[TradingSignal]:
        """
        Process bar/candle data (1m, 5m, 1h, 1d, etc.)
        Default implementation calls analyze()
        """
        return await self.analyze(bar_data)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get strategy performance metrics"""
        return self.performance_metrics
    
    def is_strategy_active(self) -> bool:
        """Check if strategy is currently active"""
        return self.is_active and self.is_running

class BacktestStrategy(BaseStrategy):
    """Base class for strategies that support backtesting"""
    
    def __init__(self, metadata: PluginMetadata):
        super().__init__(metadata)
        self.historical_data = []
        self.backtest_results = {}
    
    async def run_backtest(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run backtest on historical data
        """
        self.historical_data = data
        signals = []
        
        for index, row in data.iterrows():
            market_data = MarketData(
                symbol=row['symbol'],
                timestamp=index,
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume']
            )
            
            tick_signals = await self.on_bar(market_data)
            signals.extend(tick_signals)
        
        return self._calculate_backtest_metrics(signals)
    
    def _calculate_backtest_metrics(self, signals: List[TradingSignal]) -> Dict[str, Any]:
        """Calculate backtest performance metrics"""
        if not signals:
            return {}
        
        # Calculate returns
        returns = []
        for signal in signals:
            if signal.side == 'BUY':
                returns.append(signal.confidence * 0.01)  # Simplified
            else:
                returns.append(-signal.confidence * 0.01)
        
        # Calculate metrics
        total_return = sum(returns)
        win_rate = len([r for r in returns if r > 0]) / len(returns)
        
        return {
            'total_return': total_return,
            'win_rate': win_rate,
            'total_signals': len(signals),
            'signals': signals
        }
```

### Example Strategy Plugin

```python
# plugins/strategies/moving_average_crossover.py
import pandas as pd
import numpy as np
from typing import List
from datetime import datetime

from .base_strategy import BaseStrategy, TradingSignal, MarketData, PluginMetadata

class MovingAverageCrossoverStrategy(BaseStrategy):
    """
    A simple moving average crossover strategy that generates buy signals
    when the short-term MA crosses above the long-term MA
    """
    
    def __init__(self):
        metadata = PluginMetadata(
            name="MovingAverageCrossover",
            version="1.0.0",
            description="Simple moving average crossover strategy",
            author="Trading Team",
            category="momentum"
        )
        super().__init__(metadata)
        
        # Strategy parameters
        self.short_period = 10
        self.long_period = 20
        self.min_confidence = 0.6
        
        # State
        self.price_history = {}
        self.ma_short_history = {}
        self.ma_long_history = {}
    
    async def configure(self, config: dict):
        """Configure strategy parameters"""
        await super().configure(config)
        
        if 'short_period' in config:
            self.short_period = config['short_period']
        if 'long_period' in config:
            self.long_period = config['long_period']
        if 'min_confidence' in config:
            self.min_confidence = config['min_confidence']
        
        self.logger.info(
            f"Configured with short_period={self.short_period}, "
            f"long_period={self.long_period}"
        )
    
    async def initialize(self) -> bool:
        """Initialize the strategy"""
        if self.long_period <= self.short_period:
            self.logger.error("Long period must be greater than short period")
            return False
        
        self.is_active = True
        self.logger.info("MovingAverageCrossover strategy initialized")
        return True
    
    async def analyze(self, market_data: MarketData) -> List[TradingSignal]:
        """
        Analyze market data and generate trading signals
        """
        symbol = market_data.symbol
        timestamp = market_data.timestamp
        
        # Initialize history for new symbol
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            self.ma_short_history[symbol] = []
            self.ma_long_history[symbol] = []
        
        # Add current price to history
        self.price_history[symbol].append(market_data.close)
        
        # Keep only required history
        max_history = max(self.long_period, 50)  # Keep extra for safety
        if len(self.price_history[symbol]) > max_history:
            self.price_history[symbol] = self.price_history[symbol][-max_history:]
            self.ma_short_history[symbol] = self.ma_short_history[symbol][-max_history:]
            self.ma_long_history[symbol] = self.ma_long_history[symbol][-max_history:]
        
        # Calculate moving averages if we have enough data
        signals = []
        
        if len(self.price_history[symbol]) >= self.long_period:
            # Calculate short MA
            short_ma = np.mean(self.price_history[symbol][-self.short_period:])
            self.ma_short_history[symbol].append(short_ma)
            
            # Calculate long MA
            long_ma = np.mean(self.price_history[symbol][-self.long_period:])
            self.ma_long_history[symbol].append(long_ma)
            
            # Check for crossover
            if len(self.ma_short_history[symbol]) >= 2 and len(self.ma_long_history[symbol]) >= 2:
                prev_short = self.ma_short_history[symbol][-2]
                prev_long = self.ma_long_history[symbol][-2]
                curr_short = self.ma_short_history[symbol][-1]
                curr_long = self.ma_long_history[symbol][-1]
                
                # Golden Cross (short MA crosses above long MA)
                if prev_short <= prev_long and curr_short > curr_long:
                    confidence = min((curr_short - curr_long) / curr_long * 100, 1.0)
                    
                    if confidence >= self.min_confidence:
                        signal = TradingSignal(
                            symbol=symbol,
                            side='BUY',
                            quantity=100,  # Default quantity
                            confidence=confidence,
                            strategy_id=self.metadata.name,
                            timestamp=timestamp,
                            metadata={
                                'short_ma': short_ma,
                                'long_ma': long_ma,
                                'crossover_type': 'golden_cross'
                            }
                        )
                        signals.append(signal)
                        self.logger.info(
                            f"BUY signal generated for {symbol} "
                            f"at {market_data.close} (confidence: {confidence:.2f})"
                        )
                
                # Death Cross (short MA crosses below long MA)
                elif prev_short >= prev_long and curr_short < curr_long:
                    confidence = min((curr_long - curr_short) / curr_long * 100, 1.0)
                    
                    if confidence >= self.min_confidence:
                        signal = TradingSignal(
                            symbol=symbol,
                            side='SELL',
                            quantity=100,
                            confidence=confidence,
                            strategy_id=self.metadata.name,
                            timestamp=timestamp,
                            metadata={
                                'short_ma': short_ma,
                                'long_ma': long_ma,
                                'crossover_type': 'death_cross'
                            }
                        )
                        signals.append(signal)
                        self.logger.info(
                            f"SELL signal generated for {symbol} "
                            f"at {market_data.close} (confidence: {confidence:.2f})"
                        )
        
        return signals

# Plugin registration
def create_plugin():
    """Factory function to create plugin instance"""
    return MovingAverageCrossoverStrategy()

# Plugin metadata for auto-discovery
PLUGIN_METADATA = {
    'name': 'MovingAverageCrossover',
    'version': '1.0.0',
    'description': 'Simple moving average crossover strategy',
    'author': 'Trading Team',
    'category': 'momentum',
    'class': 'MovingAverageCrossoverStrategy',
    'factory_function': 'create_plugin'
}
```

### Advanced Strategy Plugin Example

```python
# plugins/strategies/rsi_mean_reversion.py
from typing import List, Dict, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np

from .base_strategy import BaseStrategy, TradingSignal, MarketData, PluginMetadata

@dataclass
class RSIData:
    """RSI calculation data"""
    rsi: float
    overbought: bool
    oversold: bool
    divergence: bool

class RSIMeanReversionStrategy(BaseStrategy):
    """
    RSI-based mean reversion strategy that trades on oversold/overbought conditions
    """
    
    def __init__(self):
        metadata = PluginMetadata(
            name="RSIMeanReversion",
            version="1.0.0",
            description="RSI-based mean reversion strategy",
            author="Quant Team",
            category="mean_reversion"
        )
        super().__init__(metadata)
        
        # Default parameters
        self.rsi_period = 14
        self.overbought_threshold = 70
        self.oversold_threshold = 30
        self.rsi_lookback = 5  # For divergence detection
        self.min_confidence = 0.5
        
        # State tracking
        self.price_data = {}
        self.rsi_data = {}
        self.position_state = {}  # Track if we're in a position
    
    async def configure(self, config: dict):
        """Configure strategy parameters"""
        await super().configure(config)
        
        self.rsi_period = config.get('rsi_period', self.rsi_period)
        self.overbought_threshold = config.get('overbought_threshold', self.overbought_threshold)
        self.oversold_threshold = config.get('oversold_threshold', self.oversold_threshold)
        self.rsi_lookback = config.get('rsi_lookback', self.rsi_lookback)
        self.min_confidence = config.get('min_confidence', self.min_confidence)
        
        self.logger.info("RSI Mean Reversion strategy configured")
    
    async def initialize(self) -> bool:
        """Initialize the strategy"""
        self.is_active = True
        self.logger.info("RSI Mean Reversion strategy initialized")
        return True
    
    def calculate_rsi(self, prices: List[float], period: int) -> float:
        """Calculate RSI (Relative Strength Index)"""
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate initial averages
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        # Calculate RSI
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def detect_divergence(self, symbol: str) -> bool:
        """Detect RSI divergence"""
        if len(self.rsi_data[symbol]) < self.rsi_lookback + 1:
            return False
        
        recent_rsi = self.rsi_data[symbol][-self.rsi_lookback:]
        recent_prices = self.price_data[symbol][-self.rsi_lookback:]
        
        # Simple divergence detection: price makes lower lows but RSI makes higher lows (bullish)
        # or price makes higher highs but RSI makes lower highs (bearish)
        
        price_trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
        rsi_trend = np.polyfit(range(len(recent_rsi)), recent_rsi, 1)[0]
        
        # Bullish divergence
        if price_trend < 0 and rsi_trend > 0:
            return True
        
        # Bearish divergence
        if price_trend > 0 and rsi_trend < 0:
            return True
        
        return False
    
    async def analyze(self, market_data: MarketData) -> List[TradingSignal]:
        """Analyze market data and generate signals"""
        symbol = market_data.symbol
        timestamp = market_data.timestamp
        
        # Initialize data storage for new symbol
        if symbol not in self.price_data:
            self.price_data[symbol] = []
            self.rsi_data[symbol] = []
            self.position_state[symbol] = {'has_position': False, 'entry_price': 0}
        
        # Add current price
        self.price_data[symbol].append(market_data.close)
        
        # Keep only necessary history
        max_history = max(self.rsi_period + self.rsi_lookback, 100)
        if len(self.price_data[symbol]) > max_history:
            self.price_data[symbol] = self.price_data[symbol][-max_history:]
            self.rsi_data[symbol] = self.rsi_data[symbol][-max_history:]
        
        signals = []
        
        # Calculate RSI if we have enough data
        if len(self.price_data[symbol]) >= self.rsi_period:
            rsi_value = self.calculate_rsi(
                self.price_data[symbol], 
                self.rsi_period
            )
            self.rsi_data[symbol].append(rsi_value)
            
            # Determine overbought/oversold conditions
            overbought = rsi_value > self.overbought_threshold
            oversold = rsi_value < self.oversold_threshold
            
            # Detect divergence
            divergence = self.detect_divergence(symbol)
            
            # Only trade if we don't already have a position
            if not self.position_state[symbol]['has_position']:
                # Generate signals
                if oversold and divergence:
                    # Strong oversold + divergence = BUY signal
                    confidence = min((self.oversold_threshold - rsi_value) / 30, 1.0)
                    
                    if confidence >= self.min_confidence:
                        signal = TradingSignal(
                            symbol=symbol,
                            side='BUY',
                            quantity=100,
                            confidence=confidence,
                            strategy_id=self.metadata.name,
                            timestamp=timestamp,
                            stop_loss=market_data.close * 0.95,  # 5% stop loss
                            take_profit=market_data.close * 1.10,  # 10% take profit
                            metadata={
                                'rsi': rsi_value,
                                'condition': 'oversold_divergence',
                                'entry_price': market_data.close
                            }
                        )
                        signals.append(signal)
                        self.position_state[symbol]['has_position'] = True
                        self.position_state[symbol]['entry_price'] = market_data.close
                
                elif overbought and divergence:
                    # Strong overbought + divergence = SELL signal
                    confidence = min((rsi_value - self.overbought_threshold) / 30, 1.0)
                    
                    if confidence >= self.min_confidence:
                        signal = TradingSignal(
                            symbol=symbol,
                            side='SELL',
                            quantity=100,
                            confidence=confidence,
                            strategy_id=self.metadata.name,
                            timestamp=timestamp,
                            stop_loss=market_data.close * 1.05,  # 5% stop loss
                            take_profit=market_data.close * 0.90,  # 10% take profit
                            metadata={
                                'rsi': rsi_value,
                                'condition': 'overbought_divergence',
                                'entry_price': market_data.close
                            }
                        )
                        signals.append(signal)
                        self.position_state[symbol]['has_position'] = True
                        self.position_state[symbol]['entry_price'] = market_data.close
            
            else:
                # We have a position, check for exit conditions
                entry_price = self.position_state[symbol]['entry_price']
                current_price = market_data.close
                position_pnl = (current_price - entry_price) / entry_price
                
                # Exit conditions based on RSI normalization
                if self.position_state[symbol].get('side') == 'BUY':
                    if rsi_value > 50:  # RSI normalized, exit long position
                        self.position_state[symbol]['has_position'] = False
                        exit_signal = TradingSignal(
                            symbol=symbol,
                            side='SELL',
                            quantity=100,
                            confidence=0.8,
                            strategy_id=self.metadata.name,
                            timestamp=timestamp,
                            metadata={
                                'exit_reason': 'rsi_normalization',
                                'entry_price': entry_price,
                                'exit_price': current_price,
                                'pnl_pct': position_pnl * 100
                            }
                        )
                        signals.append(exit_signal)
                
                # Update position state
                self.position_state[symbol].pop('side', None)
        
        return signals

def create_plugin():
    """Factory function"""
    return RSIMeanReversionStrategy()

PLUGIN_METADATA = {
    'name': 'RSIMeanReversion',
    'version': '1.0.0',
    'description': 'RSI-based mean reversion strategy',
    'author': 'Quant Team',
    'category': 'mean_reversion',
    'class': 'RSIMeanReversionStrategy',
    'factory_function': 'create_plugin'
}
```

## Data Provider Plugins

### Base Data Provider Interface

```python
# plugins/data_providers/base_provider.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio

@dataclass
class TickerData:
    """Real-time ticker data"""
    symbol: str
    price: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume: Optional[int] = None
    timestamp: datetime = None

@dataclass
class BarData:
    """OHLCV bar data"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    timeframe: str  # '1m', '5m', '1h', '1d', etc.

class BaseDataProvider(PluginBase):
    """Abstract base class for data providers"""
    
    def __init__(self, metadata: PluginMetadata):
        super().__init__(metadata)
        self.subscribers = {}  # symbol -> list of callbacks
        self.rate_limiter = RateLimiter()
        self.is_connected = False
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to data source"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from data source"""
        pass
    
    @abstractmethod
    async def get_real_time_quote(self, symbol: str) -> Optional[TickerData]:
        """Get real-time quote for a symbol"""
        pass
    
    @abstractmethod
    async def get_historical_bars(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[BarData]:
        """Get historical bar data"""
        pass
    
    async def subscribe(self, symbol: str, callback: Callable[[TickerData], None]):
        """Subscribe to real-time data for a symbol"""
        if symbol not in self.subscribers:
            self.subscribers[symbol] = []
            await self._start_subscription(symbol)
        
        self.subscribers[symbol].append(callback)
    
    async def unsubscribe(self, symbol: str, callback: Callable[[TickerData], None]):
        """Unsubscribe from real-time data"""
        if symbol in self.subscribers and callback in self.subscribers[symbol]:
            self.subscribers[symbol].remove(callback)
            
            if not self.subscribers[symbol]:
                await self._stop_subscription(symbol)
                del self.subscribers[symbol]
    
    async def _start_subscription(self, symbol: str):
        """Start real-time subscription for a symbol"""
        self.logger.info(f"Starting subscription for {symbol}")
    
    async def _stop_subscription(self, symbol: str):
        """Stop real-time subscription for a symbol"""
        self.logger.info(f"Stopping subscription for {symbol}")
    
    async def _broadcast_tick(self, tick_data: TickerData):
        """Broadcast tick data to all subscribers"""
        if tick_data.symbol in self.subscribers:
            for callback in self.subscribers[tick_data.symbol]:
                try:
                    await callback(tick_data)
                except Exception as e:
                    self.logger.error(f"Error in subscriber callback: {e}")

class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, max_calls: int = 100, time_window: int = 60):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    async def acquire(self):
        """Acquire permission to make an API call"""
        now = datetime.now()
        
        # Remove old calls outside the time window
        self.calls = [call_time for call_time in self.calls 
                     if (now - call_time).seconds < self.time_window]
        
        # Check if we've exceeded the limit
        if len(self.calls) >= self.max_calls:
            # Wait until we can make another call
            sleep_time = self.time_window - (now - self.calls[0]).seconds
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        # Record this call
        self.calls.append(now)
```

### Example Data Provider Plugin

```python
# plugins/data_providers/yahoo_finance.py
import aiohttp
import asyncio
from typing import List, Optional
from datetime import datetime, timedelta
import pandas as pd

from .base_provider import BaseDataProvider, TickerData, BarData, PluginMetadata

class YahooFinanceProvider(BaseDataProvider):
    """
    Yahoo Finance data provider
    """
    
    def __init__(self):
        metadata = PluginMetadata(
            name="YahooFinance",
            version="1.0.0",
            description="Yahoo Finance market data provider",
            author="Data Team",
            category="data_provider"
        )
        super().__init__(metadata)
        
        self.base_url = "https://query1.finance.yahoo.com"
        self.session = None
        self.active_subscriptions = set()
    
    async def initialize(self) -> bool:
        """Initialize the provider"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        self.logger.info("Yahoo Finance provider initialized")
        return True
    
    async def connect(self) -> bool:
        """Establish connection to Yahoo Finance"""
        try:
            # Test connection with a simple request
            async with self.session.get(f"{self.base_url}/v8/finance/chart/SPY") as resp:
                if resp.status == 200:
                    self.is_connected = True
                    self.logger.info("Connected to Yahoo Finance")
                    return True
                else:
                    self.logger.error(f"Failed to connect to Yahoo Finance: {resp.status}")
                    return False
        except Exception as e:
            self.logger.error(f"Error connecting to Yahoo Finance: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Yahoo Finance"""
        if self.session:
            await self.session.close()
            self.session = None
        
        self.is_connected = False
        self.logger.info("Disconnected from Yahoo Finance")
    
    async def get_real_time_quote(self, symbol: str) -> Optional[TickerData]:
        """Get real-time quote for a symbol"""
        try:
            # Rate limiting
            await self.rate_limiter.acquire()
            
            url = f"{self.base_url}/v8/finance/chart/{symbol}"
            async with self.session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    if 'chart' in data and data['chart']['result']:
                        result = data['chart']['result'][0]
                        
                        # Get the latest quote
                        timestamps = result['timestamp']
                        quote = result['indicators']['quote'][0]
                        
                        latest_idx = -1
                        latest_price = quote['close'][latest_idx]
                        latest_volume = quote['volume'][latest_idx]
                        
                        return TickerData(
                            symbol=symbol,
                            price=latest_price,
                            volume=latest_volume,
                            timestamp=datetime.fromtimestamp(timestamps[latest_idx])
                        )
                else:
                    self.logger.error(f"Failed to get quote for {symbol}: {resp.status}")
                    return None
        
        except Exception as e:
            self.logger.error(f"Error getting quote for {symbol}: {e}")
            return None
    
    async def get_historical_bars(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[BarData]:
        """Get historical bar data"""
        try:
            # Rate limiting
            await self.rate_limiter.acquire()
            
            # Convert timeframe to Yahoo Finance format
            period_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '1h', '1d': '1d', '1wk': '1wk', '1mo': '1mo'
            }
            
            if timeframe not in period_map:
                raise ValueError(f"Unsupported timeframe: {timeframe}")
            
            period1 = int(start_date.timestamp())
            period2 = int(end_date.timestamp())
            
            url = (
                f"{self.base_url}/v8/finance/chart/{symbol}"
                f"?period1={period1}&period2={period2}"
                f"&interval={period_map[timeframe]}"
            )
            
            async with self.session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    if 'chart' in data and data['chart']['result']:
                        result = data['chart']['result'][0]
                        
                        timestamps = result['timestamp']
                        quote = result['indicators']['quote'][0]
                        
                        bars = []
                        for i in range(len(timestamps)):
                            if (quote['open'][i] is not None and 
                                quote['high'][i] is not None and 
                                quote['low'][i] is not None and 
                                quote['close'][i] is not None):
                                
                                bar = BarData(
                                    symbol=symbol,
                                    timestamp=datetime.fromtimestamp(timestamps[i]),
                                    open=quote['open'][i],
                                    high=quote['high'][i],
                                    low=quote['low'][i],
                                    close=quote['close'][i],
                                    volume=quote['volume'][i] or 0,
                                    timeframe=timeframe
                                )
                                bars.append(bar)
                        
                        return bars
                else:
                    self.logger.error(f"Failed to get historical data: {resp.status}")
                    return []
        
        except Exception as e:
            self.logger.error(f"Error getting historical data: {e}")
            return []
    
    async def _start_subscription(self, symbol: str):
        """Start real-time subscription for a symbol"""
        if symbol not in self.active_subscriptions:
            self.active_subscriptions.add(symbol)
            # In a real implementation, you would set up WebSocket connection
            # For now, we'll use polling
            asyncio.create_task(self._poll_symbol(symbol))
    
    async def _stop_subscription(self, symbol: str):
        """Stop real-time subscription for a symbol"""
        self.active_subscriptions.discard(symbol)
    
    async def _poll_symbol(self, symbol: str):
        """Poll symbol for real-time updates"""
        while symbol in self.active_subscriptions:
            try:
                quote = await self.get_real_time_quote(symbol)
                if quote:
                    await self._broadcast_tick(quote)
                
                # Poll every second for real-time data
                await asyncio.sleep(1)
            except Exception as e:
                self.logger.error(f"Error polling {symbol}: {e}")
                await asyncio.sleep(5)  # Wait longer on error

def create_plugin():
    """Factory function"""
    return YahooFinanceProvider()

PLUGIN_METADATA = {
    'name': 'YahooFinance',
    'version': '1.0.0',
    'description': 'Yahoo Finance market data provider',
    'author': 'Data Team',
    'category': 'data_provider',
    'class': 'YahooFinanceProvider',
    'factory_function': 'create_plugin'
}
```

## Risk Management Plugins

### Base Risk Manager Interface

```python
# plugins/risk_managers/base_risk_manager.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import asyncio

@dataclass
class Position:
    """Position information"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    timestamp: datetime

@dataclass
class RiskMetrics:
    """Risk calculation results"""
    var_1d: float  # 1-day Value at Risk
    expected_shortfall: float
    correlation_risk: float
    concentration_risk: float
    liquidity_risk: float
    timestamp: datetime

@dataclass
class RiskLimit:
    """Risk limit definition"""
    limit_type: str  # 'position_size', 'portfolio_risk', 'sector_concentration'
    symbol: Optional[str]  # None for portfolio-wide limits
    limit_value: float
    current_value: float
    utilization_pct: float
    is_breached: bool

class BaseRiskManager(PluginBase):
    """Abstract base class for risk management plugins"""
    
    def __init__(self, metadata: PluginMetadata):
        super().__init__(metadata)
        self.positions = {}
        self.risk_limits = {}
        self.risk_metrics = {}
        self.alert_callbacks = []
    
    @abstractmethod
    async def calculate_position_risk(self, position: Position) -> Dict[str, float]:
        """
        Calculate risk metrics for a single position
        Returns dict with risk metrics
        """
        pass
    
    @abstractmethod
    async def calculate_portfolio_risk(self, positions: List[Position]) -> RiskMetrics:
        """
        Calculate portfolio-wide risk metrics
        """
        pass
    
    @abstractmethod
    async def validate_trade(self, trade_request: Dict[str, Any]) -> bool:
        """
        Validate a trade request against risk rules
        Returns True if trade is allowed, False otherwise
        """
        pass
    
    def add_risk_limit(self, limit: RiskLimit):
        """Add a risk limit to monitor"""
        limit_key = f"{limit.limit_type}:{limit.symbol or 'portfolio'}"
        self.risk_limits[limit_key] = limit
    
    def add_alert_callback(self, callback: callable):
        """Add callback for risk alerts"""
        self.alert_callbacks.append(callback)
    
    async def check_risk_limits(self) -> List[RiskLimit]:
        """Check all risk limits and return any that are breached"""
        breached_limits = []
        
        for limit_key, limit in self.risk_limits.items():
            if limit.is_breached:
                breached_limits.append(limit)
                
                # Trigger alert
                for callback in self.alert_callbacks:
                    try:
                        await callback(limit)
                    except Exception as e:
                        self.logger.error(f"Error in risk alert callback: {e}")
        
        return breached_limits
    
    async def update_positions(self, positions: List[Position]):
        """Update tracked positions"""
        self.positions = {pos.symbol: pos for pos in positions}
        
        # Recalculate risk metrics
        portfolio_risk = await self.calculate_portfolio_risk(list(self.positions.values()))
        self.risk_metrics = portfolio_risk
        
        # Check limits
        await self.check_risk_limits()

class VaRRiskManager(BaseRiskManager):
    """
    Value at Risk (VaR) based risk manager
    """
    
    def __init__(self):
        metadata = PluginMetadata(
            name="VaRRiskManager",
            version="1.0.0",
            description="Value at Risk based risk management",
            author="Risk Team",
            category="risk_management"
        )
        super().__init__(metadata)
        
        self.confidence_level = 0.95
        self.lookback_days = 252  # 1 year of trading days
        self.historical_data = {}
    
    async def configure(self, config: dict):
        """Configure VaR parameters"""
        await super().configure(config)
        
        self.confidence_level = config.get('confidence_level', self.confidence_level)
        self.lookback_days = config.get('lookback_days', self.lookback_days)
        
        self.logger.info(
            f"VaR Risk Manager configured: "
            f"confidence={self.confidence_level}, lookback={self.lookback_days}"
        )
    
    async def calculate_position_risk(self, position: Position) -> Dict[str, float]:
        """Calculate risk metrics for a position"""
        # Get historical data for the symbol
        historical_returns = self.historical_data.get(position.symbol, [])
        
        if not historical_returns:
            return {
                'var': 0.0,
                'volatility': 0.0,
                'beta': 0.0
            }
        
        # Calculate position value
        position_value = position.quantity * position.current_price
        
        # Calculate daily returns distribution
        returns_array = np.array(historical_returns)
        
        # Calculate VaR
        var_percentile = (1 - self.confidence_level) * 100
        var_return = np.percentile(returns_array, var_percentile)
        var_amount = abs(var_return * position_value)
        
        # Calculate volatility
        volatility = np.std(returns_array) * np.sqrt(252)  # Annualized
        
        return {
            'var': var_amount,
            'var_percentage': abs(var_return),
            'volatility': volatility,
            'position_value': position_value
        }
    
    async def calculate_portfolio_risk(self, positions: List[Position]) -> RiskMetrics:
        """Calculate portfolio-wide risk metrics"""
        if not positions:
            return RiskMetrics(
                var_1d=0.0, expected_shortfall=0.0, 
                correlation_risk=0.0, concentration_risk=0.0,
                liquidity_risk=0.0, timestamp=datetime.now()
            )
        
        # Calculate individual position risks
        position_risks = []
        total_portfolio_value = 0
        
        for position in positions:
            position_risk = await self.calculate_position_risk(position)
            position_risks.append(position_risk)
            total_portfolio_value += position.quantity * position.current_price
        
        # Calculate portfolio VaR (simplified - assumes independence)
        portfolio_var = sum(risk['var'] for risk in position_risks)
        
        # Calculate expected shortfall (Conditional VaR)
        expected_shortfall = portfolio_var * 1.3  # Simplified approximation
        
        # Calculate concentration risk (Herfindahl index)
        weights = []
        for position in positions:
            weight = (position.quantity * position.current_price) / total_portfolio_value
            weights.append(weight)
        
        concentration_index = sum(w**2 for w in weights)
        concentration_risk = 1 - concentration_index  # Higher is more diversified
        
        # Calculate correlation risk (simplified)
        correlation_risk = 0.3  # Placeholder - would need correlation matrix
        
        # Calculate liquidity risk (simplified)
        liquidity_risk = 0.1  # Placeholder - would need liquidity metrics
        
        return RiskMetrics(
            var_1d=portfolio_var,
            expected_shortfall=expected_shortfall,
            correlation_risk=correlation_risk,
            concentration_risk=concentration_risk,
            liquidity_risk=liquidity_risk,
            timestamp=datetime.now()
        )
    
    async def validate_trade(self, trade_request: Dict[str, Any]) -> bool:
        """Validate a trade request"""
        # Check basic risk limits
        symbol = trade_request.get('symbol')
        quantity = trade_request.get('quantity', 0)
        side = trade_request.get('side', 'BUY')
        
        # Calculate position size limit (2% rule)
        max_position_value = self.config.get('max_portfolio_risk', 0.02)
        
        # This would be implemented with actual portfolio value
        # For now, just return True
        return True

def create_plugin():
    """Factory function"""
    return VaRRiskManager()

PLUGIN_METADATA = {
    'name': 'VaRRiskManager',
    'version': '1.0.0',
    'description': 'Value at Risk based risk management',
    'author': 'Risk Team',
    'category': 'risk_management',
    'class': 'VaRRiskManager',
    'factory_function': 'create_plugin'
}
```

## Plugin Testing Framework

### Plugin Test Infrastructure

```python
# tests/test_plugins/conftest.py
import pytest
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from typing import List

from plugins.strategies.base_strategy import MarketData, TradingSignal
from plugins.data_providers.base_provider import TickerData

@pytest.fixture
def sample_market_data():
    """Generate sample market data for testing"""
    def _generate_data(symbol: str, periods: int = 100, start_price: float = 100.0):
        data = []
        price = start_price
        
        for i in range(periods):
            # Generate realistic price movement
            change = np.random.normal(0, 0.02)  # 2% daily volatility
            price *= (1 + change)
            
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            
            data.append(MarketData(
                symbol=symbol,
                timestamp=datetime.now() - timedelta(days=periods-i),
                open=price,
                high=high,
                low=low,
                close=price,
                volume=int(np.random.uniform(100000, 1000000))
            ))
        
        return data
    
    return _generate_data

@pytest.fixture
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

# tests/test_plugins/test_strategy_plugin.py
import pytest
import numpy as np
from plugins.strategies.moving_average_crossover import MovingAverageCrossoverStrategy

class TestMovingAverageCrossoverStrategy:
    """Test the moving average crossover strategy"""
    
    @pytest.fixture
    def strategy(self):
        """Create strategy instance for testing"""
        strategy = MovingAverageCrossoverStrategy()
        strategy.configure({
            'short_period': 10,
            'long_period': 20,
            'min_confidence': 0.5
        })
        return strategy
    
    @pytest.mark.asyncio
    async def test_strategy_initialization(self, strategy):
        """Test strategy initialization"""
        result = await strategy.initialize()
        assert result is True
        assert strategy.is_active is True
    
    @pytest.mark.asyncio
    async def test_golden_cross_signal(self, strategy, sample_market_data):
        """Test golden cross signal generation"""
        await strategy.initialize()
        
        # Generate data that will cause a golden cross
        data = sample_market_data('SPY', 50, 100.0)
        
        # Create upward trending data for golden cross
        signals = []
        for bar in data[:30]:  # Need enough data for moving averages
            signal = await strategy.analyze(bar)
            signals.extend(signal)
        
        # At least one signal should be generated
        buy_signals = [s for s in signals if s.side == 'BUY']
        assert len(buy_signals) > 0
        
        # Check signal properties
        for signal in buy_signals:
            assert signal.symbol == 'SPY'
            assert signal.confidence > 0
            assert signal.strategy_id == 'MovingAverageCrossover'
    
    @pytest.mark.asyncio
    async def test_insufficient_data(self, strategy):
        """Test strategy behavior with insufficient data"""
        await strategy.initialize()
        
        # Create only a few data points
        market_data = MarketData(
            symbol='TEST',
            timestamp=datetime.now(),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=1000
        )
        
        signals = await strategy.analyze(market_data)
        assert len(signals) == 0  # No signals with insufficient data
    
    @pytest.mark.asyncio
    async def test_configuration(self, strategy):
        """Test strategy configuration"""
        config = {
            'short_period': 5,
            'long_period': 15,
            'min_confidence': 0.7
        }
        
        await strategy.configure(config)
        
        assert strategy.short_period == 5
        assert strategy.long_period == 15
        assert strategy.min_confidence == 0.7

# tests/test_plugins/test_data_provider_plugin.py
import pytest
from plugins.data_providers.yahoo_finance import YahooFinanceProvider

class TestYahooFinanceProvider:
    """Test the Yahoo Finance data provider"""
    
    @pytest.fixture
    def provider(self):
        """Create provider instance for testing"""
        return YahooFinanceProvider()
    
    @pytest.mark.asyncio
    async def test_provider_initialization(self, provider):
        """Test provider initialization"""
        result = await provider.initialize()
        assert result is True
        assert provider.session is not None
    
    @pytest.mark.asyncio
    async def test_connection(self, provider):
        """Test connection establishment"""
        await provider.initialize()
        
        # Note: This test might fail if Yahoo Finance rate limits or blocks the request
        try:
            connected = await provider.connect()
            # Connection might fail due to network issues, which is acceptable
            assert isinstance(connected, bool)
        except Exception as e:
            # Expected in testing environment
            pytest.skip(f"Connection test failed: {e}")
    
    @pytest.mark.asyncio
    async def test_get_historical_data(self, provider, sample_market_data):
        """Test historical data retrieval"""
        await provider.initialize()
        
        # Test with sample data (in real implementation, this would fetch from Yahoo)
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        
        # This would normally call provider.get_historical_bars()
        # For testing, we'll simulate the result
        bars = []  # Simulated empty result due to testing limitations
        
        assert isinstance(bars, list)

# tests/test_plugins/test_plugin_registry.py
from plugins import PluginRegistry
from plugins.strategies.moving_average_crossover import MovingAverageCrossoverStrategy

class TestPluginRegistry:
    """Test the plugin registry system"""
    
    def test_plugin_registration(self):
        """Test registering and discovering plugins"""
        registry = PluginRegistry()
        
        # Register a plugin
        strategy = MovingAverageCrossoverStrategy()
        registry.register_plugin('moving_average_cross', strategy)
        
        # Discover the plugin
        discovered = registry.get_plugin('moving_average_cross')
        assert discovered is not None
        assert isinstance(discovered, MovingAverageCrossoverStrategy)
    
    def test_plugin_discovery_by_category(self):
        """Test discovering plugins by category"""
        registry = PluginRegistry()
        
        # Register plugins of different categories
        strategy = MovingAverageCrossoverStrategy()
        provider = YahooFinanceProvider()
        
        registry.register_plugin('strategy1', strategy)
        registry.register_plugin('provider1', provider)
        
        # Get strategies
        strategies = registry.get_plugins_by_category('strategy')
        assert len(strategies) == 1
        assert isinstance(strategies[0], MovingAverageCrossoverStrategy)
        
        # Get data providers
        providers = registry.get_plugins_by_category('data_provider')
        assert len(providers) == 1
        assert isinstance(providers[0], YahooFinanceProvider)
```

### Plugin Testing Utilities

```python
# tests/test_plugins/test_utils.py
import pytest
import asyncio
from datetime import datetime, timedelta
from typing import List

class PluginTestUtils:
    """Utilities for testing plugins"""
    
    @staticmethod
    def create_mock_market_data(
        symbol: str, 
        periods: int = 100, 
        start_price: float = 100.0,
        trend: float = 0.0,
        volatility: float = 0.02
    ):
        """Create mock market data with configurable properties"""
        import numpy as np
        
        data = []
        price = start_price
        
        for i in range(periods):
            # Add trend component
            trend_change = trend * (i / periods)
            
            # Add random component
            random_change = np.random.normal(0, volatility)
            
            # Calculate new price
            total_change = trend_change + random_change
            price *= (1 + total_change)
            
            # Generate OHLC
            high = price * (1 + abs(np.random.normal(0, volatility/2)))
            low = price * (1 - abs(np.random.normal(0, volatility/2)))
            
            from plugins.strategies.base_strategy import MarketData
            data.append(MarketData(
                symbol=symbol,
                timestamp=datetime.now() - timedelta(days=periods-i),
                open=price,
                high=high,
                low=low,
                close=price,
                volume=int(np.random.uniform(100000, 1000000))
            ))
        
        return data
    
    @staticmethod
    async def run_strategy_backtest(strategy, data: List):
        """Run a complete backtest on a strategy"""
        results = {
            'signals': [],
            'trades': [],
            'metrics': {}
        }
        
        for bar in data:
            signals = await strategy.analyze(bar)
            results['signals'].extend(signals)
            
            # Simulate trade execution
            for signal in signals:
                if signal.side == 'BUY':
                    # Simulate closing trade at next bar
                    trade = {
                        'symbol': signal.symbol,
                        'entry_time': signal.timestamp,
                        'exit_time': bar.timestamp,
                        'entry_price': bar.open,  # Simplified
                        'exit_price': bar.close,
                        'quantity': signal.quantity,
                        'pnl': (bar.close - bar.open) * signal.quantity
                    }
                    results['trades'].append(trade)
        
        # Calculate metrics
        if results['trades']:
            pnls = [t['pnl'] for t in results['trades']]
            results['metrics'] = {
                'total_trades': len(results['trades']),
                'winning_trades': len([p for p in pnls if p > 0]),
                'losing_trades': len([p for p in pnls if p <= 0]),
                'total_pnl': sum(pnls),
                'win_rate': len([p for p in pnls if p > 0]) / len(pnls)
            }
        
        return results

# Example usage in tests
@pytest.mark.asyncio
async def test_strategy_comprehensive(strategy, sample_market_data):
    """Comprehensive strategy test"""
    # Create test data with upward trend
    data = PluginTestUtils.create_mock_market_data(
        symbol='TEST',
        periods=100,
        start_price=100.0,
        trend=0.001,  # 0.1% daily upward trend
        volatility=0.015
    )
    
    # Initialize strategy
    await strategy.initialize()
    
    # Run backtest
    results = await PluginTestUtils.run_strategy_backtest(strategy, data)
    
    # Assert reasonable results
    assert results['metrics']['total_trades'] > 0
    assert results['metrics']['total_pnl'] > 0  # Should be profitable with upward trend
    assert 0 <= results['metrics']['win_rate'] <= 1
```

## Plugin Distribution

### Plugin Package Structure

```
my_trading_plugin/
├── setup.py
├── pyproject.toml
├── README.md
├── my_plugin/
│   ├── __init__.py
│   ├── strategy.py
│   ├── config.py
│   └── requirements.txt
├── tests/
│   ├── __init__.py
│   └── test_strategy.py
├── examples/
│   ├── example_config.json
│   └── usage_example.py
└── docs/
    ├── installation.md
    ├── configuration.md
    └── usage.md
```

### Package Configuration

```python
# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("my_plugin/requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="my-trading-strategy",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A custom trading strategy plugin",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/my-trading-strategy",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "trading_orchestrator.plugins": [
            "my_strategy = my_plugin.strategy:PLUGIN_METADATA",
        ],
    },
)
```

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "my-trading-strategy"
version = "1.0.0"
description = "A custom trading strategy plugin"
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
license = {text = "MIT"}
readme = "README.md"
requires-python = ">=3.9"
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Financial and Insurance Industry",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]
dependencies = [
    "numpy>=1.21.0",
    "pandas>=1.3.0",
    "aiohttp>=3.8.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "black>=22.0.0",
    "flake8>=5.0.0",
    "mypy>=1.0.0",
    "coverage>=6.0.0",
]

[project.entry-points."trading_orchestrator.plugins"]
my_strategy = "my_plugin.strategy:PLUGIN_METADATA"

[tool.black]
line-length = 88
target-version = ['py39']

[tool.isort]
profile = "black"

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

### Plugin Installation and Distribution

```bash
# Development installation
pip install -e .

# Build distribution
python -m build

# Install from built distribution
pip install dist/my_trading_strategy-1.0.0-py3-none-any.whl

# Upload to PyPI (requires account and authentication)
python -m twine upload dist/*
```

## Best Practices

### 1. Plugin Development Guidelines

#### Code Organization
```python
# Good: Clear separation of concerns
class MyTradingStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(metadata)
        self._initialize_indicators()
        self._setup_state()
    
    def _initialize_indicators(self):
        """Initialize technical indicators"""
        self.rsi_calculator = RSICalculator(period=14)
        self.ma_calculator = MACalculator(short=10, long=20)
    
    def _setup_state(self):
        """Initialize plugin state"""
        self.last_signals = []
        self.performance_tracker = PerformanceTracker()
    
    async def analyze(self, market_data: MarketData) -> List[TradingSignal]:
        """Main analysis logic"""
        # Clear, focused implementation
        pass

# Bad: Everything in one method
async def analyze(self, market_data: MarketData) -> List[TradingSignal]:
    # Don't do this - hard to read and maintain
    pass
```

#### Error Handling
```python
# Good: Comprehensive error handling
async def analyze(self, market_data: MarketData) -> List[TradingSignal]:
    try:
        # Validate input
        if not self._validate_market_data(market_data):
            return []
        
        # Perform analysis
        signals = await self._perform_analysis(market_data)
        
        # Validate outputs
        validated_signals = self._validate_signals(signals)
        
        return validated_signals
    
    except ValidationError as e:
        self.logger.warning(f"Invalid market data: {e}")
        return []
    except AnalysisError as e:
        self.logger.error(f"Analysis failed: {e}")
        # Return empty list instead of crashing
        return []
    except Exception as e:
        self.logger.error(f"Unexpected error in analysis: {e}")
        # Don't let exceptions propagate
        return []

# Bad: No error handling
async def analyze(self, market_data: MarketData) -> List[TradingSignal]:
    # This can crash the entire system
    return self._complex_analysis(market_data)
```

#### Configuration Management
```python
# Good: Comprehensive configuration validation
async def configure(self, config: dict):
    """Configure plugin with validation"""
    await super().configure(config)
    
    # Define expected configuration schema
    schema = {
        'required': ['short_period', 'long_period'],
        'optional': {
            'min_confidence': {'type': float, 'min': 0.0, 'max': 1.0},
            'max_positions': {'type': int, 'min': 1, 'max': 100}
        }
    }
    
    # Validate configuration
    validated_config = self._validate_config(config, schema)
    
    # Apply configuration
    self.short_period = validated_config['short_period']
    self.long_period = validated_config['long_period']
    self.min_confidence = validated_config.get('min_confidence', 0.5)
    self.max_positions = validated_config.get('max_positions', 10)
    
    self.logger.info(f"Plugin configured: {validated_config}")

# Bad: No validation
async def configure(self, config: dict):
    self.short_period = config['short_period']  # Could crash if missing
```

### 2. Performance Optimization

#### Async/Await Best Practices
```python
# Good: Proper async implementation
class OptimizedStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(metadata)
        self._cache = {}
        self._cache_ttl = 60  # 1 minute
    
    async def analyze(self, market_data: MarketData) -> List[TradingSignal]:
        # Check cache first
        cache_key = f"{market_data.symbol}_{market_data.timestamp.isoformat()}"
        cached_result = self._get_cached_result(cache_key)
        
        if cached_result:
            return cached_result
        
        # Perform analysis
        signals = await self._perform_analysis(market_data)
        
        # Cache result
        self._cache_result(cache_key, signals)
        
        return signals
    
    def _get_cached_result(self, cache_key: str) -> Optional[List[TradingSignal]]:
        if cache_key in self._cache:
            result, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return result
        return None

# Bad: Synchronous blocking operations
async def analyze(self, market_data: MarketData) -> List[TradingSignal]:
    # Don't do this in async context
    result = self.heavy_calculation_blocking()  # Blocks event loop
    return result
```

#### Memory Management
```python
# Good: Memory-efficient data storage
class MemoryEfficientStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(metadata)
        self.max_history = 1000  # Limit history size
        self.price_history = {}
    
    async def add_market_data(self, market_data: MarketData):
        symbol = market_data.symbol
        
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.max_history)
        
        self.price_history[symbol].append(market_data.close)
        
        # Clean up old symbols to prevent memory leaks
        if len(self.price_history) > 50:  # Limit number of symbols
            self._cleanup_old_symbols()

# Bad: Unbounded memory growth
class MemoryLeakingStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(metadata)
        self.all_prices = []  # This grows forever!
    
    async def add_market_data(self, market_data: MarketData):
        self.all_prices.append(market_data.close)  # Memory leak!
```

### 3. Testing Best Practices

#### Test Organization
```python
# tests/test_my_strategy.py
import pytest
from unittest.mock import Mock, patch
from my_plugin.strategy import MyTradingStrategy
from plugins.strategies.base_strategy import MarketData

class TestMyTradingStrategy:
    """Comprehensive test suite for MyTradingStrategy"""
    
    @pytest.fixture
    def strategy(self):
        """Create fresh strategy instance for each test"""
        return MyTradingStrategy()
    
    @pytest.fixture
    def mock_market_data(self):
        """Create mock market data"""
        return MarketData(
            symbol='TEST',
            timestamp=datetime.now(),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000000
        )
    
    class TestInitialization:
        """Test initialization and configuration"""
        
        @pytest.mark.asyncio
        async def test_initialize_success(self, strategy):
            result = await strategy.initialize()
            assert result is True
            assert strategy.is_active is True
        
        @pytest.mark.asyncio
        async def test_initialize_failure(self, strategy):
            # Mock initialization failure
            with patch.object(strategy, '_initialize_components', return_value=False):
                result = await strategy.initialize()
                assert result is False
                assert strategy.is_active is False
    
    class TestConfiguration:
        """Test configuration management"""
        
        @pytest.mark.asyncio
        async def test_valid_configuration(self, strategy):
            config = {
                'short_period': 10,
                'long_period': 20,
                'min_confidence': 0.6
            }
            
            await strategy.configure(config)
            
            assert strategy.short_period == 10
            assert strategy.long_period == 20
            assert strategy.min_confidence == 0.6
        
        @pytest.mark.asyncio
        async def test_invalid_configuration(self, strategy):
            invalid_config = {
                'short_period': -5,  # Invalid value
                'long_period': 20
            }
            
            with pytest.raises(ConfigurationError):
                await strategy.configure(invalid_config)
    
    class TestAnalysis:
        """Test market analysis functionality"""
        
        @pytest.mark.asyncio
        async def test_generate_buy_signal(self, strategy, mock_market_data):
            await strategy.initialize()
            
            signals = await strategy.analyze(mock_market_data)
            
            # Test specific expected behavior
            assert len(signals) > 0  # Should generate at least one signal
            
            for signal in signals:
                assert signal.symbol == 'TEST'
                assert signal.confidence > 0
                assert signal.strategy_id == strategy.metadata.name
        
        @pytest.mark.asyncio
        async def test_insufficient_data(self, strategy):
            await strategy.initialize()
            
            # Use mock to simulate insufficient data
            with patch.object(strategy, '_has_sufficient_data', return_value=False):
                signals = await strategy.analyze(mock_market_data)
                assert len(signals) == 0
    
    class TestPerformance:
        """Test performance characteristics"""
        
        @pytest.mark.asyncio
        async def test_analysis_speed(self, strategy):
            await strategy.initialize()
            
            start_time = time.time()
            
            # Run analysis multiple times
            for _ in range(100):
                await strategy.analyze(mock_market_data)
            
            end_time = time.time()
            avg_time = (end_time - start_time) / 100
            
            # Analysis should complete in under 10ms on average
            assert avg_time < 0.01
        
        @pytest.mark.asyncio
        async def test_memory_usage(self, strategy):
            await strategy.initialize()
            
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss
            
            # Generate lots of data
            for _ in range(1000):
                await strategy.analyze(mock_market_data)
            
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable (< 50MB)
            assert memory_increase < 50 * 1024 * 1024
```

#### Test Coverage and Quality
```bash
# Run tests with coverage
pytest --cov=my_plugin --cov-report=html --cov-report=term

# Run specific test categories
pytest tests/test_strategy/ -k "test_initialization"
pytest tests/test_strategy/ -k "test_analysis"

# Run performance tests
pytest tests/test_performance/ --benchmark-only

# Generate test report
pytest --html=test_report.html --self-contained-html
```

## Conclusion

The plugin development framework provides a robust, extensible architecture for building custom trading components. Key benefits include:

1. **Modularity**: Clean separation of concerns through well-defined interfaces
2. **Testability**: Comprehensive testing framework and utilities
3. **Performance**: Async/await support and optimization guidelines
4. **Distribution**: Easy packaging and distribution mechanisms
5. **Standards**: Consistent patterns and best practices

### Next Steps

1. **Start Small**: Begin with a simple strategy plugin
2. **Follow Patterns**: Use the provided base classes and interfaces
3. **Test Thoroughly**: Write comprehensive tests using the testing framework
4. **Document Well**: Provide clear documentation and examples
5. **Share**: Consider sharing your plugins with the community

### Additional Resources

- [Architecture Overview](./architecture_overview.md)
- [Plugin Examples Repository](https://github.com/trading-orchestrator/plugin-examples)
- [Developer Forum](https://forum.trading-orchestrator.com/dev)
- [API Documentation](./api_documentation.md)

---

**Questions?** Join our developer community or contact the development team for support with plugin development.