# Plugin Development Guide

## Table of Contents
- [Plugin Architecture Overview](#plugin-architecture-overview)
- [Development Environment Setup](#development-environment-setup)
- [Plugin Types and Interfaces](#plugin-types-and-interfaces)
- [Strategy Plugin Development](#strategy-plugin-development)
- [Data Provider Plugin Development](#data-provider-plugin-development)
- [Risk Management Plugin Development](#risk-management-plugin-development)
- [Indicator Plugin Development](#indicator-plugin-development)
- [Broker Plugin Development](#broker-plugin-development)
- [UI Plugin Development](#ui-plugin-development)
- [Plugin Testing Framework](#plugin-testing-framework)
- [Plugin Packaging and Distribution](#plugin-packaging-and-distribution)
- [Plugin Security and Validation](#plugin-security-and-validation)
- [Performance Optimization](#performance-optimization)
- [Debugging and Profiling](#debugging-and-profiling)
- [Plugin Examples](#plugin-examples)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Plugin Architecture Overview

The Day Trading Orchestrator uses a modular plugin architecture that allows for flexible extension of core functionality. Plugins are self-contained modules that can be loaded at runtime to add new features without modifying the core system.

### Plugin System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Plugin System Layer                       │
├─────────────────────────────────────────────────────────────┤
│  Plugin Manager    │  Plugin Loader  │  Plugin Registry     │
├─────────────────────────────────────────────────────────────┤
│  Plugin Validator  │  Plugin Monitor │  Plugin Updater      │
├─────────────────────────────────────────────────────────────┤
│  Security Manager  │  Config Manager │  Logging Manager     │
├─────────────────────────────────────────────────────────────┤
│                     Core System                            │
│  Order Manager     │  Risk Manager   │  Data Manager        │
│  Strategy Engine   │  Broker Manager │  UI Manager          │
└─────────────────────────────────────────────────────────────┘
```

### Plugin Lifecycle

1. **Discovery**: System scans plugin directories for valid plugins
2. **Validation**: Plugin metadata and code are validated
3. **Loading**: Plugin code is loaded into memory
4. **Initialization**: Plugin initialization methods are called
5. **Registration**: Plugin registers its capabilities with the system
6. **Execution**: Plugin methods are called during system operation
7. **Monitoring**: Plugin performance and health are monitored
8. **Unloading**: Plugin is cleanly unloaded when no longer needed

### Plugin Dependencies

```python
# Core dependencies for all plugins
"""Required dependencies for plugin system"""
day_trading_orchestrator >= 1.0.0
numpy >= 1.20.0
pandas >= 1.3.0
pydantic >= 1.8.0
loguru >= 0.6.0
```

## Development Environment Setup

### Prerequisites

- Python 3.8 or higher
- Day Trading Orchestrator core system
- Git for version control
- IDE with Python support (PyCharm, VS Code, etc.)

### Project Structure

```
my-plugin/
├── setup.py                 # Package setup
├── requirements.txt         # Dependencies
├── plugin_manifest.json     # Plugin metadata
├── my_plugin/
│   ├── __init__.py
│   ├── plugin.py           # Main plugin class
│   ├── strategies/         # Strategy implementations
│   ├── indicators/         # Custom indicators
│   ├── utils/             # Utility functions
│   └── tests/             # Test files
├── examples/               # Usage examples
├── docs/                  # Documentation
└── README.md             # Plugin description
```

### Development Setup

```bash
# 1. Create plugin development directory
mkdir -p /opt/trading-orchestrator/plugins/development
cd /opt/trading-orchestrator/plugins/development

# 2. Create plugin template
python -m day_trading_orchestrator.plugins.create_template my-plugin

# 3. Install development dependencies
pip install -e .
pip install pytest pytest-cov black flake8 mypy

# 4. Set up pre-commit hooks
pre-commit install

# 5. Run tests to verify setup
pytest tests/ -v
```

## Plugin Types and Interfaces

### Plugin Base Class

All plugins inherit from the base Plugin class:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from day_trading_orchestrator.plugins import Plugin
from day_trading_orchestrator.core import MarketData, Order, Position

class BasePlugin(Plugin, ABC):
    """Base class for all plugins"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = self.get_logger(__name__)
    
    @abstractmethod
    def get_plugin_info(self) -> Dict[str, Any]:
        """Return plugin metadata"""
        pass
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the plugin"""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up plugin resources"""
        pass
```

### Plugin Types

1. **Strategy Plugins**: Implement trading strategies
2. **Data Provider Plugins**: Supply market data
3. **Risk Management Plugins**: Enforce risk controls
4. **Indicator Plugins**: Provide technical indicators
5. **Broker Plugins**: Interface with brokers
6. **UI Plugins**: Extend the user interface
7. **Utility Plugins**: Provide utility functions

## Strategy Plugin Development

### Strategy Plugin Interface

```python
from day_trading_orchestrator.plugins.strategies import StrategyPlugin
from day_trading_orchestrator.core import Signal, MarketData, Position
from typing import List, Dict, Any, Optional

class MyStrategyPlugin(StrategyPlugin):
    """Example strategy plugin"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.short_period = config.get('short_period', 5)
        self.long_period = config.get('long_period', 20)
        self.symbols = config.get('symbols', [])
        self.data_buffer = {}
        
    def get_plugin_info(self) -> Dict[str, Any]:
        return {
            'name': 'My Moving Average Strategy',
            'version': '1.0.0',
            'author': 'Your Name',
            'description': 'Moving average crossover strategy',
            'type': 'strategy',
            'parameters': {
                'short_period': {'type': 'int', 'default': 5, 'min': 1, 'max': 50},
                'long_period': {'type': 'int', 'default': 20, 'min': 5, 'max': 200},
                'symbols': {'type': 'list', 'default': [], 'item_type': 'str'}
            }
        }
    
    def initialize(self) -> bool:
        """Initialize the strategy"""
        try:
            # Validate parameters
            if self.short_period >= self.long_period:
                raise ValueError("Short period must be less than long period")
            
            # Initialize data buffers
            for symbol in self.symbols:
                self.data_buffer[symbol] = []
            
            self.logger.info("Strategy initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize strategy: {e}")
            return False
    
    def process_market_data(self, data: MarketData) -> List[Signal]:
        """Process market data and generate signals"""
        try:
            # Add data to buffer
            self.data_buffer[data.symbol].append({
                'timestamp': data.timestamp,
                'price': data.close,
                'volume': data.volume
            })
            
            # Keep only necessary data
            max_length = max(self.short_period, self.long_period) * 2
            if len(self.data_buffer[data.symbol]) > max_length:
                self.data_buffer[data.symbol] = self.data_buffer[data.symbol][-max_length:]
            
            # Generate signal if enough data
            if len(self.data_buffer[data.symbol]) >= self.long_period:
                return self._generate_signal(data.symbol)
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error processing market data: {e}")
            return []
    
    def _generate_signal(self, symbol: str) -> List[Signal]:
        """Generate trading signal based on strategy logic"""
        try:
            data = self.data_buffer[symbol]
            
            # Calculate moving averages
            short_ma = self._calculate_sma(data, self.short_period)
            long_ma = self._calculate_sma(data, self.long_period)
            
            # Generate signal based on crossover
            if short_ma > long_ma and data[-1]['price'] > short_ma:
                return [Signal(
                    symbol=symbol,
                    action='buy',
                    strength=0.8,
                    metadata={'short_ma': short_ma, 'long_ma': long_ma}
                )]
            elif short_ma < long_ma and data[-1]['price'] < short_ma:
                return [Signal(
                    symbol=symbol,
                    action='sell',
                    strength=0.8,
                    metadata={'short_ma': short_ma, 'long_ma': long_ma}
                )]
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            return []
    
    def _calculate_sma(self, data: List[Dict], period: int) -> float:
        """Calculate Simple Moving Average"""
        prices = [d['price'] for d in data[-period:]]
        return sum(prices) / len(prices)
    
    def get_positions(self) -> List[Position]:
        """Get current positions managed by this strategy"""
        # Implementation depends on strategy requirements
        return []
    
    def cleanup(self) -> None:
        """Clean up resources"""
        self.data_buffer.clear()
        self.logger.info("Strategy plugin cleaned up")
```

### Strategy Plugin Configuration

```json
{
  "plugin_config": {
    "name": "my_moving_average_strategy",
    "type": "strategy",
    "enabled": true,
    "parameters": {
      "short_period": 5,
      "long_period": 20,
      "symbols": ["AAPL", "MSFT", "GOOGL"],
      "risk_limit": 0.02
    },
    "dependencies": [],
    "version": "1.0.0"
  }
}
```

## Data Provider Plugin Development

### Data Provider Interface

```python
from day_trading_orchestrator.plugins.data_providers import DataProviderPlugin
from day_trading_orchestrator.core import MarketData, Bar, Tick
from typing import Iterator, List, Dict, Any, Optional
from datetime import datetime

class MyDataProviderPlugin(DataProviderPlugin):
    """Example data provider plugin"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('api_key')
        self.api_secret = config.get('api_secret')
        self.base_url = config.get('base_url')
        self.symbols = config.get('symbols', [])
        self.rate_limit = config.get('rate_limit', 100)  # requests per minute
        
    def get_plugin_info(self) -> Dict[str, Any]:
        return {
            'name': 'My Data Provider',
            'version': '1.0.0',
            'type': 'data_provider',
            'description': 'Custom data provider for market data',
            'supported_data_types': ['realtime', 'historical', 'tick'],
            'rate_limit': self.rate_limit
        }
    
    def initialize(self) -> bool:
        """Initialize data provider"""
        try:
            # Test API connection
            if not self._test_connection():
                return False
            
            # Initialize rate limiter
            self.rate_limiter = self._create_rate_limiter(self.rate_limit)
            
            self.logger.info("Data provider initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize data provider: {e}")
            return False
    
    def get_realtime_data(self, symbol: str) -> Iterator[MarketData]:
        """Get real-time market data for a symbol"""
        try:
            self.rate_limiter.acquire()
            
            # Make API request
            response = self._api_request('GET', f'/realtime/{symbol}')
            
            for data_point in response['data']:
                yield MarketData(
                    symbol=symbol,
                    timestamp=datetime.fromisoformat(data_point['timestamp']),
                    open=data_point['open'],
                    high=data_point['high'],
                    low=data_point['low'],
                    close=data_point['close'],
                    volume=data_point['volume']
                )
                
        except Exception as e:
            self.logger.error(f"Error getting realtime data for {symbol}: {e}")
    
    def get_historical_data(self, symbol: str, 
                          start_date: datetime, 
                          end_date: datetime,
                          frequency: str = '1d') -> List[Bar]:
        """Get historical market data"""
        try:
            self.rate_limiter.acquire()
            
            params = {
                'symbol': symbol,
                'start': start_date.isoformat(),
                'end': end_date.isoformat(),
                'frequency': frequency
            }
            
            response = self._api_request('GET', '/historical', params)
            
            bars = []
            for data_point in response['data']:
                bars.append(Bar(
                    symbol=symbol,
                    timestamp=datetime.fromisoformat(data_point['timestamp']),
                    open=data_point['open'],
                    high=data_point['high'],
                    low=data_point['low'],
                    close=data_point['close'],
                    volume=data_point['volume']
                ))
            
            return bars
            
        except Exception as e:
            self.logger.error(f"Error getting historical data: {e}")
            return []
    
    def get_tick_data(self, symbol: str) -> Iterator[Tick]:
        """Get tick-by-tick data"""
        try:
            self.rate_limiter.acquire()
            
            # Make WebSocket or streaming API request
            for tick_data in self._stream_ticks(symbol):
                yield Tick(
                    symbol=symbol,
                    timestamp=datetime.fromisoformat(tick_data['timestamp']),
                    price=tick_data['price'],
                    size=tick_data['size'],
                    side=tick_data['side']
                )
                
        except Exception as e:
            self.logger.error(f"Error getting tick data for {symbol}: {e}")
    
    def _test_connection(self) -> bool:
        """Test API connection"""
        try:
            response = self._api_request('GET', '/ping')
            return response.get('status') == 'ok'
        except Exception:
            return False
    
    def _api_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make API request with authentication"""
        # Implementation depends on the specific API
        pass
    
    def cleanup(self) -> None:
        """Clean up resources"""
        # Close any open connections
        self.logger.info("Data provider plugin cleaned up")
```

## Risk Management Plugin Development

### Risk Management Interface

```python
from day_trading_orchestrator.plugins.risk_management import RiskManagementPlugin
from day_trading_orchestrator.core import Order, Position, Account
from typing import Dict, Any, List, Optional, Tuple
from decimal import Decimal

class MyRiskManagementPlugin(RiskManagementPlugin):
    """Example risk management plugin"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.max_position_size = Decimal(str(config.get('max_position_size', 10000)))
        self.max_daily_loss = Decimal(str(config.get('max_daily_loss', 1000)))
        self.max_portfolio_risk = Decimal(str(config.get('max_portfolio_risk', 0.02)))
        self.correlation_limit = config.get('correlation_limit', 0.7)
        
    def get_plugin_info(self) -> Dict[str, Any]:
        return {
            'name': 'My Risk Management',
            'version': '1.0.0',
            'type': 'risk_management',
            'description': 'Advanced risk management with correlation limits',
            'risk_controls': [
                'position_size_limit',
                'daily_loss_limit',
                'portfolio_risk_limit',
                'correlation_risk'
            ]
        }
    
    def initialize(self) -> bool:
        """Initialize risk management system"""
        try:
            # Initialize risk metrics tracking
            self.daily_pnl = Decimal('0')
            self.position_correlations = {}
            self.risk_metrics = {
                'var': Decimal('0'),  # Value at Risk
                'max_drawdown': Decimal('0'),
                'sharpe_ratio': Decimal('0')
            }
            
            self.logger.info("Risk management initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize risk management: {e}")
            return False
    
    def validate_order(self, order: Order, 
                      current_positions: List[Position], 
                      account: Account) -> Tuple[bool, str]:
        """Validate an order against risk rules"""
        try:
            # Check position size limit
            if order.quantity * order.price > self.max_position_size:
                return False, f"Position size {order.quantity * order.price} exceeds limit {self.max_position_size}"
            
            # Check available margin
            available_margin = account.margin_available
            required_margin = order.quantity * order.price * Decimal('0.2')  # 20% margin requirement
            
            if required_margin > available_margin:
                return False, f"Insufficient margin. Required: {required_margin}, Available: {available_margin}"
            
            # Check correlation risk
            if not self._check_correlation_risk(order, current_positions):
                return False, f"Order would create excessive correlation risk"
            
            # Check daily loss limit
            if self.daily_pnl - (order.quantity * order.price) < -self.max_daily_loss:
                return False, f"Order would exceed daily loss limit"
            
            return True, "Order approved"
            
        except Exception as e:
            self.logger.error(f"Error validating order: {e}")
            return False, f"Risk validation error: {e}"
    
    def check_portfolio_risk(self, positions: List[Position], 
                           account: Account) -> Dict[str, Any]:
        """Check overall portfolio risk"""
        try:
            total_value = account.total_value
            total_risk = self._calculate_portfolio_risk(positions)
            
            risk_percentage = total_risk / total_value
            
            return {
                'portfolio_risk': total_risk,
                'risk_percentage': risk_percentage,
                'limit': float(self.max_portfolio_risk),
                'within_limits': risk_percentage <= self.max_portfolio_risk,
                'var': self._calculate_var(positions),
                'correlation_risk': self._analyze_correlation_risk(positions)
            }
            
        except Exception as e:
            self.logger.error(f"Error checking portfolio risk: {e}")
            return {'error': str(e)}
    
    def update_risk_metrics(self, pnl_change: Decimal) -> None:
        """Update risk metrics based on P&L change"""
        try:
            self.daily_pnl += pnl_change
            
            # Update VaR (simplified calculation)
            self.risk_metrics['var'] = self._calculate_rolling_var()
            
            # Update maximum drawdown
            if pnl_change < 0:
                self.risk_metrics['max_drawdown'] = min(
                    self.risk_metrics['max_drawdown'],
                    pnl_change
                )
            
        except Exception as e:
            self.logger.error(f"Error updating risk metrics: {e}")
    
    def _check_correlation_risk(self, order: Order, 
                              positions: List[Position]) -> bool:
        """Check if order would create excessive correlation risk"""
        try:
            # Get correlation between order symbol and existing positions
            correlations = []
            for position in positions:
                if position.symbol != order.symbol:
                    corr = self._get_correlation(order.symbol, position.symbol)
                    correlations.append(corr)
            
            if not correlations:
                return True  # No existing positions
            
            avg_correlation = sum(abs(c) for c in correlations) / len(correlations)
            
            return avg_correlation <= self.correlation_limit
            
        except Exception as e:
            self.logger.error(f"Error checking correlation risk: {e}")
            return True  # Default to allowing the order
    
    def _calculate_portfolio_risk(self, positions: List[Position]) -> Decimal:
        """Calculate total portfolio risk"""
        # Simplified risk calculation
        total_risk = Decimal('0')
        
        for position in positions:
            # Use position value as risk proxy (simplified)
            position_risk = position.quantity * position.price * position.volatility
            total_risk += position_risk
        
        return total_risk
    
    def _calculate_var(self, positions: List[Position]) -> Decimal:
        """Calculate Value at Risk"""
        # Simplified VaR calculation
        portfolio_value = sum(p.quantity * p.price for p in positions)
        return portfolio_value * Decimal('0.05')  # 5% VaR
    
    def _get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols"""
        # Implementation depends on data availability
        return 0.5  # Placeholder
    
    def _analyze_correlation_risk(self, positions: List[Position]) -> Dict[str, Any]:
        """Analyze correlation risk in portfolio"""
        if len(positions) < 2:
            return {'correlation_risk': 'low'}
        
        # Calculate pairwise correlations
        correlations = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                corr = self._get_correlation(
                    positions[i].symbol, 
                    positions[j].symbol
                )
                correlations.append(corr)
        
        avg_correlation = sum(abs(c) for c in correlations) / len(correlations)
        
        if avg_correlation > 0.8:
            return {'correlation_risk': 'high', 'avg_correlation': avg_correlation}
        elif avg_correlation > 0.5:
            return {'correlation_risk': 'medium', 'avg_correlation': avg_correlation}
        else:
            return {'correlation_risk': 'low', 'avg_correlation': avg_correlation}
    
    def cleanup(self) -> None:
        """Clean up resources"""
        self.daily_pnl = Decimal('0')
        self.position_correlations.clear()
        self.logger.info("Risk management plugin cleaned up")
```

## Indicator Plugin Development

### Indicator Interface

```python
from day_trading_orchestrator.plugins.indicators import IndicatorPlugin
from day_trading_orchestrator.core import MarketData, Bar
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

class MyCustomIndicator(IndicatorPlugin):
    """Example custom indicator plugin"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.period = config.get('period', 14)
        self.threshold = config.get('threshold', 1.5)
        self.data_buffer = []
        
    def get_plugin_info(self) -> Dict[str, Any]:
        return {
            'name': 'My Custom Momentum Indicator',
            'version': '1.0.0',
            'type': 'indicator',
            'description': 'Custom momentum indicator with dynamic threshold',
            'category': 'momentum',
            'parameters': {
                'period': {'type': 'int', 'default': 14, 'min': 2, 'max': 100},
                'threshold': {'type': 'float', 'default': 1.5, 'min': 0.1, 'max': 10.0}
            },
            'outputs': ['value', 'signal', 'strength']
        }
    
    def initialize(self) -> bool:
        """Initialize indicator"""
        try:
            # Reset data buffer
            self.data_buffer = []
            
            self.logger.info(f"Indicator initialized with period={self.period}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize indicator: {e}")
            return False
    
    def calculate(self, data: MarketData) -> Dict[str, Any]:
        """Calculate indicator value"""
        try:
            # Add data to buffer
            self.data_buffer.append({
                'timestamp': data.timestamp,
                'close': data.close,
                'high': data.high,
                'low': data.low,
                'volume': data.volume
            })
            
            # Keep only necessary data
            if len(self.data_buffer) > self.period * 2:
                self.data_buffer = self.data_buffer[-self.period * 2:]
            
            # Calculate indicator if enough data
            if len(self.data_buffer) >= self.period:
                return self._calculate_indicator()
            else:
                return {'value': None, 'signal': 'waiting', 'strength': 0.0}
            
        except Exception as e:
            self.logger.error(f"Error calculating indicator: {e}")
            return {'value': None, 'signal': 'error', 'strength': 0.0}
    
    def calculate_batch(self, bars: List[Bar]) -> List[Dict[str, Any]]:
        """Calculate indicator for multiple bars"""
        try:
            results = []
            temp_buffer = []
            
            for bar in bars:
                temp_buffer.append({
                    'timestamp': bar.timestamp,
                    'close': bar.close,
                    'high': bar.high,
                    'low': bar.low,
                    'volume': bar.volume
                })
                
                if len(temp_buffer) >= self.period:
                    value = self._calculate_indicator_from_buffer(temp_buffer)
                    signal = self._generate_signal(value)
                    strength = self._calculate_strength(value)
                    
                    results.append({
                        'timestamp': bar.timestamp,
                        'value': value,
                        'signal': signal,
                        'strength': strength
                    })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error calculating indicator batch: {e}")
            return []
    
    def _calculate_indicator(self) -> Dict[str, Any]:
        """Calculate indicator from data buffer"""
        try:
            data = self.data_buffer[-self.period:]
            
            # Example: Custom momentum calculation
            prices = [d['close'] for d in data]
            volumes = [d['volume'] for d in data]
            
            # Calculate price change
            price_change = (prices[-1] - prices[0]) / prices[0]
            
            # Calculate volume-weighted average
            vwap = sum(p * v for p, v in zip(prices, volumes)) / sum(volumes)
            
            # Calculate momentum value
            momentum = (prices[-1] - vwap) / vwap
            
            # Determine signal
            signal = self._generate_signal(momentum)
            
            # Calculate strength
            strength = self._calculate_strength(momentum)
            
            return {
                'value': momentum,
                'signal': signal,
                'strength': strength,
                'components': {
                    'price_change': price_change,
                    'vwap': vwap,
                    'current_price': prices[-1]
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in indicator calculation: {e}")
            return {'value': None, 'signal': 'error', 'strength': 0.0}
    
    def _calculate_indicator_from_buffer(self, buffer: List[Dict]) -> float:
        """Calculate indicator from a specific buffer"""
        prices = [d['close'] for d in buffer]
        volumes = [d['volume'] for d in buffer]
        
        price_change = (prices[-1] - prices[0]) / prices[0]
        vwap = sum(p * v for p, v in zip(prices, volumes)) / sum(volumes)
        momentum = (prices[-1] - vwap) / vwap
        
        return momentum
    
    def _generate_signal(self, value: Optional[float]) -> str:
        """Generate trading signal from indicator value"""
        if value is None:
            return 'neutral'
        
        if value > self.threshold:
            return 'buy'
        elif value < -self.threshold:
            return 'sell'
        else:
            return 'hold'
    
    def _calculate_strength(self, value: Optional[float]) -> float:
        """Calculate signal strength (0.0 to 1.0)"""
        if value is None:
            return 0.0
        
        # Normalize strength based on threshold
        abs_value = abs(value)
        return min(abs_value / self.threshold, 1.0)
    
    def get_required_data_length(self) -> int:
        """Return required data length for calculation"""
        return self.period
    
    def cleanup(self) -> None:
        """Clean up resources"""
        self.data_buffer.clear()
        self.logger.info("Indicator plugin cleaned up")
```

## Plugin Testing Framework

### Unit Testing

```python
import pytest
from unittest.mock import Mock, patch
from my_plugin import MyStrategyPlugin
from day_trading_orchestrator.core import MarketData, Signal
from datetime import datetime

class TestMyStrategyPlugin:
    """Test cases for MyStrategyPlugin"""
    
    @pytest.fixture
    def strategy_config(self):
        return {
            'short_period': 5,
            'long_period': 20,
            'symbols': ['AAPL']
        }
    
    @pytest.fixture
    def strategy_plugin(self, strategy_config):
        return MyStrategyPlugin(strategy_config)
    
    def test_plugin_info(self, strategy_plugin):
        """Test plugin information"""
        info = strategy_plugin.get_plugin_info()
        
        assert info['name'] == 'My Moving Average Strategy'
        assert info['type'] == 'strategy'
        assert 'parameters' in info
    
    def test_initialization_success(self, strategy_plugin):
        """Test successful initialization"""
        assert strategy_plugin.initialize() is True
        assert 'AAPL' in strategy_plugin.data_buffer
    
    def test_initialization_validation_error(self):
        """Test initialization with invalid parameters"""
        config = {
            'short_period': 20,  # Should be less than long_period
            'long_period': 10,
            'symbols': []
        }
        
        plugin = MyStrategyPlugin(config)
        assert plugin.initialize() is False
    
    def test_process_market_data_insufficient_data(self, strategy_plugin):
        """Test processing with insufficient data"""
        strategy_plugin.initialize()
        
        # Add data less than required period
        data = MarketData(
            symbol='AAPL',
            timestamp=datetime.now(),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000
        )
        
        signals = strategy_plugin.process_market_data(data)
        assert len(signals) == 0
    
    @patch('my_plugin.MyStrategyPlugin._calculate_sma')
    def test_signal_generation(self, mock_calculate_sma, strategy_plugin):
        """Test signal generation"""
        strategy_plugin.initialize()
        
        # Mock the SMA calculation
        mock_calculate_sma.return_value = 101.0
        
        # Add enough data for signal generation
        for i in range(25):  # More than long_period
            data = MarketData(
                symbol='AAPL',
                timestamp=datetime.now(),
                open=100.0 + i,
                high=101.0 + i,
                low=99.0 + i,
                close=100.5 + i,
                volume=1000
            )
            signals = strategy_plugin.process_market_data(data)
        
        # Should generate at least one signal
        assert len(signals) > 0
        assert isinstance(signals[0], Signal)
    
    def test_cleanup(self, strategy_plugin):
        """Test cleanup functionality"""
        strategy_plugin.initialize()
        assert 'AAPL' in strategy_plugin.data_buffer
        
        strategy_plugin.cleanup()
        assert len(strategy_plugin.data_buffer) == 0
```

### Integration Testing

```python
import pytest
from day_trading_orchestrator.plugins import PluginManager
from my_plugin import MyStrategyPlugin

class TestPluginIntegration:
    """Integration tests for plugin system"""
    
    @pytest.fixture
    def plugin_manager(self):
        return PluginManager()
    
    def test_plugin_loading(self, plugin_manager):
        """Test plugin loading"""
        # Load plugin
        plugin = plugin_manager.load_plugin('my_plugin')
        
        assert plugin is not None
        assert isinstance(plugin, MyStrategyPlugin)
        assert plugin.get_plugin_info()['type'] == 'strategy'
    
    def test_plugin_registration(self, plugin_manager):
        """Test plugin registration"""
        plugin = plugin_manager.load_plugin('my_plugin')
        
        # Register plugin
        plugin_manager.register_plugin(plugin)
        
        # Check if plugin is registered
        assert 'my_plugin' in plugin_manager.get_registered_plugins()
    
    def test_plugin_execution(self, plugin_manager):
        """Test plugin execution"""
        plugin = plugin_manager.load_plugin('my_plugin')
        plugin.initialize()
        
        # Process market data
        from day_trading_orchestrator.core import MarketData
        from datetime import datetime
        
        data = MarketData(
            symbol='AAPL',
            timestamp=datetime.now(),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000
        )
        
        signals = plugin.process_market_data(data)
        
        # Plugin should handle the data without errors
        assert isinstance(signals, list)
    
    def test_plugin_unloading(self, plugin_manager):
        """Test plugin unloading"""
        plugin = plugin_manager.load_plugin('my_plugin')
        plugin_manager.register_plugin(plugin)
        
        # Unload plugin
        plugin_manager.unload_plugin('my_plugin')
        
        # Check if plugin is unregistered
        assert 'my_plugin' not in plugin_manager.get_registered_plugins()
```

### Performance Testing

```python
import pytest
import time
from my_plugin import MyStrategyPlugin
from day_trading_orchestrator.core import MarketData
from datetime import datetime

class TestPluginPerformance:
    """Performance tests for plugins"""
    
    @pytest.mark.performance
    def test_strategy_processing_speed(self):
        """Test strategy processing speed"""
        config = {
            'short_period': 5,
            'long_period': 20,
            'symbols': ['AAPL', 'MSFT', 'GOOGL']
        }
        
        plugin = MyStrategyPlugin(config)
        plugin.initialize()
        
        # Generate test data
        test_data = []
        for i in range(1000):
            for symbol in config['symbols']:
                test_data.append(MarketData(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    open=100.0 + i,
                    high=101.0 + i,
                    low=99.0 + i,
                    close=100.5 + i,
                    volume=1000
                ))
        
        # Measure processing time
        start_time = time.time()
        
        for data in test_data:
            plugin.process_market_data(data)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process 1000 data points in reasonable time
        assert processing_time < 5.0  # Less than 5 seconds
    
    @pytest.mark.performance
    def test_memory_usage(self):
        """Test memory usage"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        config = {
            'short_period': 5,
            'long_period': 20,
            'symbols': ['AAPL']
        }
        
        plugin = MyStrategyPlugin(config)
        plugin.initialize()
        
        # Generate and process data
        for i in range(10000):
            data = MarketData(
                symbol='AAPL',
                timestamp=datetime.now(),
                open=100.0 + i,
                high=101.0 + i,
                low=99.0 + i,
                close=100.5 + i,
                volume=1000
            )
            plugin.process_market_data(data)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024  # 100MB
        
        plugin.cleanup()
```

## Plugin Packaging and Distribution

### Package Structure

```python
# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="day-trading-orchestrator-my-plugin",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="My custom plugin for Day Trading Orchestrator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/day-trading-orchestrator-my-plugin",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "day_trading_orchestrator.plugins": [
            "my_plugin = my_plugin.plugin:MyPluginClass",
        ],
    },
    include_package_data=True,
    package_data={
        "my_plugin": ["config/*", "examples/*"],
    },
)
```

### Plugin Manifest

```json
{
  "manifest_version": "1.0",
  "plugin_info": {
    "name": "my-plugin",
    "version": "1.0.0",
    "display_name": "My Custom Plugin",
    "description": "A custom plugin for advanced trading strategies",
    "author": {
      "name": "Your Name",
      "email": "your.email@example.com",
      "website": "https://github.com/yourusername"
    },
    "license": "MIT",
    "homepage": "https://github.com/yourusername/day-trading-orchestrator-my-plugin",
    "repository": "https://github.com/yourusername/day-trading-orchestrator-my-plugin.git",
    "keywords": ["trading", "strategy", "plugin"],
    "category": "strategy"
  },
  "compatibility": {
    "min_platform_version": "1.0.0",
    "max_platform_version": "2.0.0",
    "python_version": ">=3.8",
    "supported_oses": ["linux", "windows", "macos"]
  },
  "dependencies": {
    "required": [
      "day-trading-orchestrator>=1.0.0",
      "numpy>=1.20.0",
      "pandas>=1.3.0"
    ],
    "optional": [
      "scipy>=1.7.0",
      "scikit-learn>=1.0.0"
    ]
  },
  "configuration": {
    "parameters": {
      "short_period": {
        "type": "integer",
        "default": 5,
        "min": 1,
        "max": 50,
        "description": "Short moving average period"
      },
      "long_period": {
        "type": "integer", 
        "default": 20,
        "min": 5,
        "max": 200,
        "description": "Long moving average period"
      },
      "symbols": {
        "type": "array",
        "default": [],
        "item_type": "string",
        "description": "List of trading symbols"
      }
    }
  },
  "permissions": {
    "required": [
      "read_market_data",
      "place_orders",
      "read_account_info"
    ],
    "optional": [
      "access_external_apis",
      "send_notifications"
    ]
  },
  "security": {
    "checksum": "sha256:...",
    "signature": "...",
    "trusted_publisher": false
  }
}
```

### Distribution Script

```bash
#!/bin/bash
# build_and_distribute.sh

set -e

echo "Building plugin distribution..."

# Clean previous builds
rm -rf build/ dist/ *.egg-info/

# Run tests
echo "Running tests..."
pytest tests/ -v --cov=my_plugin

# Run linting
echo "Running linting..."
flake8 my_plugin/
mypy my_plugin/
black --check my_plugin/

# Build distribution
echo "Building distribution..."
python setup.py sdist bdist_wheel

# Verify package
echo "Verifying package..."
twine check dist/*

# Create checksum
echo "Creating checksums..."
sha256sum dist/* > checksums.txt

echo "Distribution built successfully!"
echo "Files created:"
ls -la dist/
echo
echo "Checksums:"
cat checksums.txt
```

## Plugin Security and Validation

### Security Manager

```python
from day_trading_orchestrator.plugins.security import SecurityManager
from typing import Dict, Any, List, Optional
import hashlib
import json
import importlib.util
from pathlib import Path

class PluginSecurityManager(SecurityManager):
    """Security manager for plugin validation"""
    
    def __init__(self):
        self.allowed_imports = {
            'numpy', 'pandas', 'scipy', 'sklearn',
            'day_trading_orchestrator', 'day_trading_orchestrator.core',
            'day_trading_orchestrator.plugins'
        }
        self.dangerous_functions = {
            'exec', 'eval', 'compile', '__import__',
            'subprocess', 'os.system', 'os.popen'
        }
    
    def validate_plugin(self, plugin_path: Path) -> Dict[str, Any]:
        """Validate plugin for security issues"""
        try:
            validation_result = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'checksums': {}
            }
            
            # 1. Check file integrity
            validation_result['checksums'] = self._calculate_checksums(plugin_path)
            
            # 2. Analyze code for security issues
            security_issues = self._analyze_security(plugin_path)
            validation_result['errors'].extend(security_issues['errors'])
            validation_result['warnings'].extend(security_issues['warnings'])
            
            # 3. Validate imports
            import_issues = self._validate_imports(plugin_path)
            validation_result['errors'].extend(import_issues)
            
            # 4. Check manifest
            manifest_issues = self._validate_manifest(plugin_path)
            validation_result['errors'].extend(manifest_issues)
            
            # Determine overall validity
            validation_result['valid'] = len(validation_result['errors']) == 0
            
            return validation_result
            
        except Exception as e:
            return {
                'valid': False,
                'errors': [f"Validation error: {e}"],
                'warnings': [],
                'checksums': {}
            }
    
    def _calculate_checksums(self, plugin_path: Path) -> Dict[str, str]:
        """Calculate checksums for plugin files"""
        checksums = {}
        
        for file_path in plugin_path.rglob('*.py'):
            with open(file_path, 'rb') as f:
                content = f.read()
                checksum = hashlib.sha256(content).hexdigest()
                checksums[str(file_path.relative_to(plugin_path))] = checksum
        
        return checksums
    
    def _analyze_security(self, plugin_path: Path) -> Dict[str, List[str]]:
        """Analyze plugin code for security issues"""
        errors = []
        warnings = []
        
        for file_path in plugin_path.rglob('*.py'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for dangerous function calls
                for func in self.dangerous_functions:
                    if func in content:
                        errors.append(
                            f"Dangerous function '{func}' found in {file_path}"
                        )
                
                # Check for import statements
                for line in content.split('\n'):
                    line = line.strip()
                    if line.startswith('import ') or line.startswith('from '):
                        module = line.split()[1].split('.')[0]
                        if module not in self.allowed_imports:
                            warnings.append(
                                f"Potentially unsafe import '{module}' in {file_path}"
                            )
            
            except Exception as e:
                errors.append(f"Error analyzing {file_path}: {e}")
        
        return {'errors': errors, 'warnings': warnings}
    
    def _validate_imports(self, plugin_path: Path) -> List[str]:
        """Validate plugin imports"""
        errors = []
        
        try:
            # Try to import the plugin module safely
            plugin_main = plugin_path / 'plugin.py'
            if plugin_main.exists():
                spec = importlib.util.spec_from_file_location("plugin", plugin_main)
                if spec and spec.loader:
                    # This is a basic check - in production, use more sophisticated sandboxing
                    pass
        
        except Exception as e:
            errors.append(f"Import validation failed: {e}")
        
        return errors
    
    def _validate_manifest(self, plugin_path: Path) -> List[str]:
        """Validate plugin manifest"""
        errors = []
        
        manifest_path = plugin_path / 'plugin_manifest.json'
        
        if not manifest_path.exists():
            errors.append("Missing plugin_manifest.json")
            return errors
        
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            # Validate required fields
            required_fields = ['plugin_info', 'compatibility', 'configuration']
            for field in required_fields:
                if field not in manifest:
                    errors.append(f"Missing required field in manifest: {field}")
            
            # Validate plugin info
            if 'plugin_info' in manifest:
                info = manifest['plugin_info']
                required_info_fields = ['name', 'version', 'type']
                for field in required_info_fields:
                    if field not in info:
                        errors.append(f"Missing required plugin info field: {field}")
        
        except json.JSONDecodeError:
            errors.append("Invalid JSON in plugin_manifest.json")
        except Exception as e:
            errors.append(f"Error validating manifest: {e}")
        
        return errors
    
    def generate_signature(self, plugin_path: Path) -> str:
        """Generate digital signature for plugin"""
        # This would use proper cryptographic signing in production
        combined_hash = hashlib.sha256()
        
        for file_path in sorted(plugin_path.rglob('*')):
            if file_path.is_file():
                with open(file_path, 'rb') as f:
                    combined_hash.update(f.read())
        
        return combined_hash.hexdigest()
```

### Plugin Validation Example

```python
from day_trading_orchestrator.plugins.security import PluginSecurityManager
from pathlib import Path

def validate_plugin_security(plugin_path: str) -> Dict[str, Any]:
    """Validate plugin security"""
    security_manager = PluginSecurityManager()
    plugin_path = Path(plugin_path)
    
    result = security_manager.validate_plugin(plugin_path)
    
    if result['valid']:
        print("✓ Plugin validation passed")
        if result['warnings']:
            print("⚠ Warnings:")
            for warning in result['warnings']:
                print(f"  - {warning}")
    else:
        print("✗ Plugin validation failed")
        print("Errors:")
        for error in result['errors']:
            print(f"  - {error}")
    
    return result

# Usage
plugin_path = "/opt/trading-orchestrator/plugins/my-plugin"
result = validate_plugin_security(plugin_path)

if result['valid']:
    print("Plugin is safe to load")
else:
    print("Plugin has security issues and will not be loaded")
```

## Best Practices

### Code Quality

1. **Follow PEP 8**: Use consistent code formatting
2. **Type Hints**: Use type annotations for better code clarity
3. **Docstrings**: Document all public methods and classes
4. **Error Handling**: Implement comprehensive error handling
5. **Logging**: Use structured logging for debugging

### Performance

1. **Efficient Algorithms**: Use optimized algorithms for calculations
2. **Memory Management**: Avoid memory leaks and excessive allocations
3. **Caching**: Implement caching for expensive calculations
4. **Rate Limiting**: Respect API rate limits
5. **Profiling**: Use profiling tools to identify bottlenecks

### Security

1. **Input Validation**: Validate all input parameters
2. **Sandboxing**: Run plugins in restricted environments
3. **Code Review**: Conduct security reviews of all plugins
4. **Updates**: Keep dependencies up to date
5. **Monitoring**: Monitor plugin behavior for anomalies

### Testing

1. **Unit Tests**: Write comprehensive unit tests
2. **Integration Tests**: Test plugin integration with core system
3. **Performance Tests**: Benchmark plugin performance
4. **Security Tests**: Test for security vulnerabilities
5. **End-to-End Tests**: Test complete trading workflows

## Troubleshooting

### Common Issues

1. **Import Errors**: Check Python path and dependencies
2. **Configuration Errors**: Validate plugin configuration
3. **Performance Issues**: Profile plugin code
4. **Memory Leaks**: Monitor memory usage
5. **Security Warnings**: Review security validation results

### Debug Techniques

1. **Logging**: Use detailed logging for debugging
2. **Debugging Tools**: Use Python debugging tools (pdb, IDE debuggers)
3. **Profiling**: Use profiling tools (cProfile, line_profiler)
4. **Unit Tests**: Write tests to isolate issues
5. **Monitoring**: Monitor plugin metrics

---

This plugin development guide provides comprehensive coverage of plugin development for the Day Trading Orchestrator system. Follow the examples and best practices to create secure, performant, and maintainable plugins.