"""Test Fixtures and Common Utilities for Exit Strategy Tests

Provides shared test fixtures, mock data generators, and common assertion
utilities used across all exit strategy test modules.

Features:
- Position and market data fixtures
- Mock configuration generators
- Common test utilities
- Assertion helpers
- Data validation tools
- Performance benchmarking utilities

Author: Trading System Development Team
Version: 1.0.0
Date: 2024-12-19
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional, Tuple, Union
from unittest.mock import AsyncMock, MagicMock
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()

# Standard test symbols
TEST_SYMBOLS = [
    'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 
    'META', 'NFLX', 'AMD', 'INTC', 'CRM', 'ORCL'
]

# Test timeframes
TEST_TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']

@dataclass
class MockPosition:
    """Mock position for testing"""
    position_id: str
    symbol: str
    quantity: Decimal
    entry_price: Decimal
    current_price: Decimal
    side: str = 'long'  # 'long' or 'short'
    entry_time: datetime = field(default_factory=datetime.utcnow)
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    trailing_stop: Optional[Decimal] = None
    
    @property
    def unrealized_pnl(self) -> Decimal:
        """Calculate unrealized P&L"""
        if self.side == 'long':
            return (self.current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.current_price) * self.quantity
    
    @property
    def pnl_percentage(self) -> Decimal:
        """Calculate P&L percentage"""
        if self.side == 'long':
            return ((self.current_price - self.entry_price) / self.entry_price) * Decimal('100')
        else:
            return ((self.entry_price - self.current_price) / self.entry_price) * Decimal('100')

@dataclass
class MockMarketData:
    """Mock market data for testing"""
    symbol: str
    timestamp: datetime
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    volume: int = 1000000
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    
    def __post_init__(self):
        if self.bid is None:
            self.bid = self.close_price * Decimal('0.999')  # 0.1% spread
        if self.ask is None:
            self.ask = self.close_price * Decimal('1.001')  # 0.1% spread

class TestDataGenerator:
    """Generate realistic test data for various scenarios"""
    
    @staticmethod
    def generate_position(
        symbol: Optional[str] = None,
        side: str = 'long',
        quantity: Optional[Decimal] = None,
        entry_price: Optional[Decimal] = None,
        current_price: Optional[Decimal] = None
    ) -> MockPosition:
        """Generate a mock position"""
        symbol = symbol or fake.random_element(TEST_SYMBOLS)
        quantity = quantity or Decimal(str(fake.random_int(min=1, max=1000)))
        entry_price = entry_price or Decimal(str(round(fake.random.uniform(50, 500), 2)))
        
        if current_price is None:
            price_change = fake.random.uniform(-0.2, 0.2)  # ±20% price change
            if side == 'long':
                current_price = entry_price * Decimal(str(1 + price_change))
            else:
                current_price = entry_price * Decimal(str(1 - price_change))
        
        return MockPosition(
            position_id=fake.uuid4(),
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            current_price=current_price,
            side=side
        )
    
    @staticmethod
    def generate_market_data(
        symbol: Optional[str] = None,
        base_price: Optional[Decimal] = None,
        timestamp: Optional[datetime] = None,
        volatility: float = 0.02
    ) -> MockMarketData:
        """Generate mock market data"""
        symbol = symbol or fake.random_element(TEST_SYMBOLS)
        base_price = base_price or Decimal(str(round(fake.random.uniform(50, 500), 2)))
        timestamp = timestamp or datetime.utcnow()
        
        # Generate realistic OHLC data
        open_price = base_price
        price_change = fake.random.uniform(-volatility, volatility)
        close_price = base_price * Decimal(str(1 + price_change))
        
        high_offset = fake.random.uniform(0, 0.01) * close_price
        low_offset = fake.random.uniform(0, 0.01) * close_price
        
        high_price = max(open_price, close_price) + high_offset
        low_price = min(open_price, close_price) - low_offset
        
        volume = fake.random_int(min=100000, max=10000000)
        
        return MockMarketData(
            symbol=symbol,
            timestamp=timestamp,
            open_price=open_price,
            high_price=high_price,
            low_price=low_price,
            close_price=close_price,
            volume=volume
        )
    
    @staticmethod
    def generate_historical_data(
        symbol: Optional[str] = None,
        days: int = 30,
        timeframe: str = '1d',
        base_price: Optional[Decimal] = None
    ) -> List[MockMarketData]:
        """Generate historical market data"""
        symbol = symbol or fake.random_element(TEST_SYMBOLS)
        base_price = base_price or Decimal(str(round(fake.random.uniform(50, 500), 2)))
        
        data = []
        current_price = base_price
        current_time = datetime.utcnow()
        
        for i in range(days):
            # Calculate timeframe intervals
            if timeframe == '1d':
                time_delta = timedelta(days=1)
            elif timeframe == '4h':
                time_delta = timedelta(hours=4)
            elif timeframe == '1h':
                time_delta = timedelta(hours=1)
            elif timeframe == '15m':
                time_delta = timedelta(minutes=15)
            else:
                time_delta = timedelta(days=1)
            
            current_time -= time_delta
            
            # Generate market data
            market_data = TestDataGenerator.generate_market_data(
                symbol=symbol,
                base_price=current_price,
                timestamp=current_time
            )
            
            data.append(market_data)
            current_price = market_data.close_price
        
        return data[::-1]  # Reverse to get chronological order
    
    @staticmethod
    def generate_portfolio(
        num_positions: int = 5,
        symbols: Optional[List[str]] = None
    ) -> List[MockPosition]:
        """Generate a portfolio of positions"""
        symbols = symbols or fake.random_elements(TEST_SYMBOLS, length=num_positions, unique=True)
        
        positions = []
        for symbol in symbols:
            side = fake.random_element(['long', 'short'])
            position = TestDataGenerator.generate_position(symbol=symbol, side=side)
            positions.append(position)
        
        return positions

# Pytest Fixtures

@pytest.fixture
def test_symbols():
    """Provide test symbols"""
    return TEST_SYMBOLS

@pytest.fixture
def test_timeframes():
    """Provide test timeframes"""
    return TEST_TIMEFRAMES

@pytest.fixture
def mock_position():
    """Generate a single mock position"""
    return TestDataGenerator.generate_position()

@pytest.fixture
def mock_portfolio():
    """Generate a mock portfolio"""
    return TestDataGenerator.generate_portfolio()

@pytest.fixture
def mock_market_data():
    """Generate mock market data"""
    return TestDataGenerator.generate_market_data()

@pytest.fixture
def historical_data():
    """Generate historical market data"""
    return TestDataGenerator.generate_historical_data()

@pytest.fixture
def long_position():
    """Generate a long position"""
    return TestDataGenerator.generate_position(side='long')

@pytest.fixture
def short_position():
    """Generate a short position"""
    return TestDataGenerator.generate_position(side='short')

@pytest.fixture
def profitable_position():
    """Generate a profitable position"""
    entry_price = Decimal('100.00')
    current_price = Decimal('120.00')  # 20% profit
    return TestDataGenerator.generate_position(
        entry_price=entry_price,
        current_price=current_price
    )

@pytest.fixture
def losing_position():
    """Generate a losing position"""
    entry_price = Decimal('100.00')
    current_price = Decimal('80.00')  # 20% loss
    return TestDataGenerator.generate_position(
        entry_price=entry_price,
        current_price=current_price
    )

@pytest.fixture
def volatile_market_data():
    """Generate volatile market data"""
    return TestDataGenerator.generate_market_data(volatility=0.05)  # 5% volatility

@pytest.fixture
def stable_market_data():
    """Generate stable market data"""
    return TestDataGenerator.generate_market_data(volatility=0.005)  # 0.5% volatility

# Mock Context Fixtures

@pytest.fixture
def mock_trading_context():
    """Create a mock trading context"""
    context = AsyncMock()
    
    # Mock common context methods
    context.get_current_price = AsyncMock(return_value=Decimal('150.00'))
    context.get_position = AsyncMock()
    context.get_historical_data = AsyncMock()
    context.get_portfolio_value = AsyncMock(return_value=Decimal('1000000'))
    context.calculate_volatility = AsyncMock(return_value=Decimal('0.025'))
    context.submit_exit_order = AsyncMock(return_value={'success': True, 'order_id': 'order_123'})
    context.get_risk_metrics = AsyncMock(return_value={
        'max_drawdown': 0.05,
        'var_99': Decimal('5000'),
        'sharpe_ratio': 1.2
    })
    context.get_positions = AsyncMock(return_value=[])
    
    return context

@pytest.fixture
def mock_broker_context():
    """Create a mock broker context"""
    context = AsyncMock()
    
    # Mock broker-specific methods
    context.get_account_info = AsyncMock(return_value={
        'account_id': 'test_account',
        'cash': Decimal('100000'),
        'equity': Decimal('100000'),
        'buying_power': Decimal('200000')
    })
    
    context.place_order = AsyncMock(return_value={
        'success': True,
        'order_id': 'order_456',
        'status': 'filled'
    })
    
    context.get_order_status = AsyncMock(return_value={
        'order_id': 'order_456',
        'status': 'filled',
        'filled_quantity': Decimal('100'),
        'filled_price': Decimal('150.00')
    })
    
    return context

# Configuration Fixtures

@pytest.fixture
def minimal_exit_config():
    """Create minimal exit configuration"""
    from trading_orchestrator.strategies.exit_strategies.base_exit_strategy import ExitConfiguration, ExitType
    
    return ExitConfiguration(
        strategy_id="test_exit_001",
        strategy_type=ExitType.STOP_LOSS,
        name="Test Exit Strategy",
        description="Test configuration",
        parameters={}
    )

@pytest.fixture
def complete_exit_config():
    """Create complete exit configuration"""
    from trading_orchestrator.strategies.exit_strategies.base_exit_strategy import ExitConfiguration, ExitType
    
    return ExitConfiguration(
        strategy_id="test_exit_002",
        strategy_type=ExitType.STOP_LOSS,
        name="Complete Test Exit Strategy",
        description="Complete test configuration with all parameters",
        parameters={
            'stop_loss_percentage': 5.0,
            'trailing_stop': True,
            'trailing_distance': 2.0,
            'time_based_exit': False,
            'profit_target': 15.0,
            'risk_reward_ratio': 3.0,
            'max_hold_days': 30,
            'early_exit_conditions': ['news', 'volatility_spike'],
            'partial_exits': True,
            'exit_splits': [0.3, 0.4, 0.3]
        }
    )

# Assertion Helpers

class ExitStrategyAssertions:
    """Custom assertions for exit strategy testing"""
    
    @staticmethod
    def assert_valid_exit_signal(exit_signal, expected_reason=None):
        """Assert that an exit signal is valid"""
        assert exit_signal is not None, "Exit signal should not be None"
        assert hasattr(exit_signal, 'signal_id'), "Exit signal should have signal_id"
        assert hasattr(exit_signal, 'strategy_id'), "Exit signal should have strategy_id"
        assert hasattr(exit_signal, 'position_id'), "Exit signal should have position_id"
        assert hasattr(exit_signal, 'symbol'), "Exit signal should have symbol"
        assert hasattr(exit_signal, 'exit_reason'), "Exit signal should have exit_reason"
        assert hasattr(exit_signal, 'exit_price'), "Exit signal should have exit_price"
        assert hasattr(exit_signal, 'exit_quantity'), "Exit signal should have exit_quantity"
        assert hasattr(exit_signal, 'confidence'), "Exit signal should have confidence"
        assert hasattr(exit_signal, 'urgency'), "Exit signal should have urgency"
        
        if expected_reason:
            assert exit_signal.exit_reason == expected_reason, f"Exit reason should be {expected_reason}"
        
        # Validate ranges
        assert 0.0 <= exit_signal.confidence <= 1.0, "Confidence should be between 0 and 1"
        assert 0.0 <= exit_signal.urgency <= 1.0, "Urgency should be between 0 and 1"
        assert exit_signal.exit_quantity >= 0, "Exit quantity should be non-negative"
    
    @staticmethod
    def assert_position_exited_correctly(position: MockPosition, exit_price: Decimal):
        """Assert that position was exited correctly"""
        if position.side == 'long':
            assert exit_price <= position.current_price, "Long position should exit at or below current price"
        else:
            assert exit_price >= position.current_price, "Short position should exit at or above current price"
    
    @staticmethod
    def assert_pnl_within_tolerance(actual_pnl: Decimal, expected_pnl: Decimal, tolerance: Decimal = Decimal('0.01')):
        """Assert that P&L is within tolerance"""
        pnl_diff = abs(actual_pnl - expected_pnl)
        assert pnl_diff <= tolerance, f"P&L difference {pnl_diff} exceeds tolerance {tolerance}"
    
    @staticmethod
    def assert_timing_realistic(exit_time: datetime, entry_time: datetime, min_hold_time: timedelta = timedelta(minutes=1)):
        """Assert that timing is realistic"""
        hold_duration = exit_time - entry_time
        assert hold_duration >= min_hold_time, f"Hold time {hold_duration} is less than minimum {min_hold_time}"
        assert exit_time >= entry_time, "Exit time should be after entry time"

# Performance Testing Utilities

class PerformanceBenchmark:
    """Performance benchmarking utilities"""
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.start_time = None
        self.end_time = None
        self.metrics = {}
    
    def __enter__(self):
        self.start_time = datetime.utcnow()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.utcnow()
        if exc_type is None:
            self.metrics['success'] = True
        else:
            self.metrics['success'] = False
            self.metrics['error'] = str(exc_val)
        
        self.metrics['duration'] = (self.end_time - self.start_time).total_seconds()
    
    def add_metric(self, name: str, value: Any):
        """Add custom metric"""
        self.metrics[name] = value
    
    def get_duration(self) -> float:
        """Get execution duration"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    def assert_performance(self, max_duration: float):
        """Assert that performance meets requirements"""
        duration = self.get_duration()
        assert duration <= max_duration, f"Test {self.test_name} took {duration:.4f}s, expected <= {max_duration}s"

@pytest.fixture
def performance_benchmark():
    """Provide performance benchmarking fixture"""
    def _benchmark(test_name: str):
        return PerformanceBenchmark(test_name)
    return _benchmark

# Data Validation Utilities

class DataValidator:
    """Data validation utilities"""
    
    @staticmethod
    def validate_market_data(market_data: MockMarketData):
        """Validate market data integrity"""
        assert market_data.high_price >= market_data.low_price, "High price should be >= low price"
        assert market_data.high_price >= market_data.open_price, "High should be >= open"
        assert market_data.high_price >= market_data.close_price, "High should be >= close"
        assert market_data.low_price <= market_data.open_price, "Low should be <= open"
        assert market_data.low_price <= market_data.close_price, "Low should be <= close"
        assert market_data.volume > 0, "Volume should be positive"
        assert market_data.bid < market_data.ask, "Bid should be less than ask"
        assert market_data.bid > 0, "Bid should be positive"
        assert market_data.ask > 0, "Ask should be positive"
    
    @staticmethod
    def validate_position(position: MockPosition):
        """Validate position data"""
        assert position.position_id, "Position should have ID"
        assert position.symbol, "Position should have symbol"
        assert position.quantity > 0, "Position quantity should be positive"
        assert position.entry_price > 0, "Entry price should be positive"
        assert position.current_price > 0, "Current price should be positive"
        assert position.side in ['long', 'short'], "Position side should be 'long' or 'short'"
    
    @staticmethod
    def validate_historical_data(historical_data: List[MockMarketData]):
        """Validate historical data sequence"""
        assert len(historical_data) > 0, "Historical data should not be empty"
        
        for i in range(1, len(historical_data)):
            prev_data = historical_data[i-1]
            curr_data = historical_data[i]
            
            assert curr_data.timestamp > prev_data.timestamp, "Data should be chronologically ordered"
            DataValidator.validate_market_data(curr_data)

# Test Environment Setup

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment before each test"""
    # Set test environment variables
    import os
    os.environ['TESTING'] = 'true'
    os.environ['LOG_LEVEL'] = 'DEBUG'
    
    yield
    
    # Clean up after test
    # Reset any global state if needed

@pytest.fixture
def cleanup_temp_files():
    """Clean up temporary files after test"""
    temp_files = []
    
    def add_temp_file(file_path: str):
        temp_files.append(file_path)
    
    yield add_temp_file
    
    # Cleanup
    import os
    for file_path in temp_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass

# Export commonly used items
__all__ = [
    'TEST_SYMBOLS',
    'TEST_TIMEFRAMES',
    'MockPosition',
    'MockMarketData',
    'TestDataGenerator',
    'ExitStrategyAssertions',
    'PerformanceBenchmark',
    'DataValidator'
]
