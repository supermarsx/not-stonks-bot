"""
Exit Strategies Test Suite
Comprehensive test suite for all exit strategy modules.

Provides:
- Test fixtures and utilities for exit strategy testing
- Mock data generators for various market scenarios
- Test data for backtesting scenarios
- Common test patterns and assertions
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from trading_orchestrator.strategies.exit_strategies.base_exit_strategy import (
    BaseExitStrategy,
    ExitReason,
    ExitType,
    ExitStatus,
    ExitCondition,
    ExitSignal,
    ExitMetrics,
    ExitConfiguration,
    ExitRule,
    ExitStrategyRegistry,
    exit_strategy_registry
)

# Test utilities and fixtures

@pytest.fixture
def mock_exit_context():
    """Mock exit context for testing"""
    context = AsyncMock()
    context.get_current_price = AsyncMock(return_value=Decimal('150.00'))
    context.get_position = AsyncMock(return_value={
        'position_id': 'pos_001',
        'symbol': 'AAPL',
        'quantity': Decimal('100'),
        'entry_price': Decimal('140.00'),
        'entry_time': datetime.utcnow() - timedelta(hours=2),
        'created_at': datetime.utcnow() - timedelta(hours=2)
    })
    context.get_historical_data = AsyncMock(return_value=[
        {'timestamp': datetime.utcnow() - timedelta(minutes=i), 'close': Decimal(str(140 + i*0.5))}
        for i in range(30, 0, -1)
    ])
    context.calculate_volatility = AsyncMock(return_value=Decimal('0.025'))
    context.submit_exit_order = AsyncMock(return_value={'success': True, 'order_id': 'order_123'})
    context.get_portfolio_value = AsyncMock(return_value=Decimal('1000000'))
    context.get_risk_metrics = AsyncMock(return_value={'max_drawdown': 0.05, 'var_99': Decimal('5000')})
    context.get_positions = AsyncMock(return_value=[
        {
            'position_id': 'pos_001',
            'symbol': 'AAPL',
            'quantity': Decimal('100'),
            'entry_price': Decimal('140.00'),
            'entry_time': datetime.utcnow() - timedelta(hours=2),
            'created_at': datetime.utcnow() - timedelta(hours=2)
        }
    ])
    return context


@pytest.fixture
def sample_exit_configuration():
    """Sample exit configuration for testing"""
    return ExitConfiguration(
        strategy_id="test_strategy_001",
        strategy_type=ExitType.TRAILING_STOP,
        name="Test Trailing Stop",
        description="Test configuration for unit testing",
        parameters={
            'trailing_distance': Decimal('0.03'),
            'initial_stop': Decimal('0.95'),
            'update_frequency': 60,
            'profit_threshold': Decimal('0.02')
        },
        conditions=[
            ExitCondition(
                condition_id="test_condition_1",
                name="Test Price Condition",
                condition_type="price",
                threshold_value=Decimal('145.00'),
                comparison_operator="<=",
                priority=1
            )
        ],
        max_exit_size=Decimal('100000'),
        min_confidence=0.80,
        monitoring_interval=30
    )


@pytest.fixture
def sample_position():
    """Sample position data for testing"""
    return {
        'position_id': 'pos_001',
        'symbol': 'AAPL',
        'quantity': Decimal('100'),
        'entry_price': Decimal('140.00'),
        'entry_time': datetime.utcnow() - timedelta(hours=2),
        'created_at': datetime.utcnow() - timedelta(hours=2),
        'current_price': Decimal('150.00'),
        'unrealized_pnl': Decimal('1000.00')
    }


@pytest.fixture
def sample_exit_signal():
    """Sample exit signal for testing"""
    return ExitSignal(
        signal_id="signal_001",
        strategy_id="test_strategy_001",
        position_id="pos_001",
        symbol="AAPL",
        exit_reason=ExitReason.TRAILING_STOP,
        exit_price=Decimal('145.00'),
        exit_quantity=Decimal('100'),
        confidence=0.90,
        urgency=0.75,
        metadata={'trigger_type': 'price_hit'},
        created_at=datetime.utcnow()
    )


@pytest.fixture
def sample_historical_prices():
    """Generate sample historical price data"""
    base_price = Decimal('140.00')
    return [
        {
            'timestamp': datetime.utcnow() - timedelta(hours=30-i),
            'open': base_price + Decimal(str(i * 0.2)),
            'high': base_price + Decimal(str(i * 0.2 + 2)),
            'low': base_price + Decimal(str(i * 0.2 - 1)),
            'close': base_price + Decimal(str(i * 0.5)),
            'volume': 1000000 + i * 1000
        }
        for i in range(30)
    ]


@pytest.fixture
def volatile_market_data():
    """Generate volatile market data for stress testing"""
    base_price = Decimal('150.00')
    data = []
    for i in range(100):
        # Create volatile price movements
        change = (i % 5 - 2) * 0.5  # -1.0 to +1.0 with some volatility
        price = base_price + Decimal(str(change * (1 + i % 3)))
        data.append({
            'timestamp': datetime.utcnow() - timedelta(hours=100-i),
            'close': price,
            'volume': 1000000 + abs(change) * 500000
        })
    return data


# Async test utilities
async def wait_for_condition(condition_func, timeout=5.0, interval=0.1):
    """Wait for a condition to become true"""
    start_time = datetime.utcnow()
    while (datetime.utcnow() - start_time).total_seconds() < timeout:
        if await condition_func():
            return True
        await asyncio.sleep(interval)
    return False


def assert_exit_signal_valid(signal: ExitSignal):
    """Assert that an exit signal has valid structure"""
    assert signal.signal_id is not None
    assert signal.strategy_id is not None
    assert signal.position_id is not None
    assert signal.symbol is not None
    assert signal.exit_reason is not None
    assert signal.exit_price > 0
    assert signal.exit_quantity > 0
    assert 0.0 <= signal.confidence <= 1.0
    assert 0.0 <= signal.urgency <= 1.0
    assert isinstance(signal.created_at, datetime)


def assert_metrics_reasonable(metrics: ExitMetrics):
    """Assert that exit metrics have reasonable values"""
    assert metrics.strategy_id is not None
    assert metrics.total_exits >= 0
    assert metrics.successful_exits >= 0
    assert metrics.failed_exits >= 0
    assert 0.0 <= metrics.success_rate <= 1.0
    assert 0.0 <= metrics.win_rate <= 1.0
    assert isinstance(metrics.last_updated, datetime)


# Market scenario generators
def generate_trending_data(length=50, direction='up', volatility=0.01):
    """Generate trending market data"""
    base_price = Decimal('100.00')
    data = []
    current_price = base_price
    
    for i in range(length):
        # Add trend component
        trend_change = 0.1 if direction == 'up' else -0.1
        
        # Add random component with volatility
        import random
        random_change = random.uniform(-volatility, volatility)
        
        current_price = current_price + Decimal(str(trend_change + random_change))
        
        data.append({
            'timestamp': datetime.utcnow() - timedelta(hours=length-i),
            'close': current_price,
            'volume': 1000000 + i * 1000
        })
    
    return data


def generate_mean_reverting_data(length=50, amplitude=5.0):
    """Generate mean-reverting market data"""
    base_price = Decimal('100.00')
    data = []
    
    for i in range(length):
        # Sine wave pattern for mean reversion
        import math
        change = amplitude * math.sin(i * math.pi / 10) * 0.1
        price = base_price + Decimal(str(change))
        
        data.append({
            'timestamp': datetime.utcnow() - timedelta(hours=length-i),
            'close': price,
            'volume': 1000000 + i * 1000
        })
    
    return data


def generate_volatile_data(length=50, volatility=0.05):
    """Generate high volatility market data"""
    base_price = Decimal('100.00')
    data = []
    
    for i in range(length):
        # High volatility movements
        import random
        change = random.uniform(-volatility, volatility) * 10
        price = base_price + Decimal(str(change))
        
        data.append({
            'timestamp': datetime.utcnow() - timedelta(hours=length-i),
            'close': price,
            'volume': 1000000 + random.randint(0, 500000)
        })
    
    return data


# Test data constants
TEST_SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
TEST_POSITIONS = ['pos_001', 'pos_002', 'pos_003']
TEST_STRATEGIES = ['test_001', 'test_002', 'test_003']

# Export all utilities
__all__ = [
    'mock_exit_context',
    'sample_exit_configuration', 
    'sample_position',
    'sample_exit_signal',
    'sample_historical_prices',
    'volatile_market_data',
    'wait_for_condition',
    'assert_exit_signal_valid',
    'assert_metrics_reasonable',
    'generate_trending_data',
    'generate_mean_reverting_data', 
    'generate_volatile_data',
    'TEST_SYMBOLS',
    'TEST_POSITIONS',
    'TEST_STRATEGIES'
]
