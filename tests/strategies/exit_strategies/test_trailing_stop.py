"""
Test Trailing Stop Exit Strategies

Tests various trailing stop exit strategies including:
- TrailingStopStrategy base class
- ATR-based trailing stops
- Fixed percentage trailing stops
- Dynamic trailing distance adjustment
- Multiple trailing stop modes
- Integration with position management
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch

from trading_orchestrator.strategies.exit_strategies.trailing_stop import (
    TrailingStopStrategy,
    TrailingStopConfig,
    ATRTrailingStop,
    FixedTrailingStop,
    ChandelierExit
)

from trading_orchestrator.strategies.exit_strategies.base_exit_strategy import (
    ExitReason,
    ExitType,
    ExitCondition,
    ExitConfiguration
)


# Test fixture for trailing stop configuration
@pytest.fixture
def trailing_stop_config():
    """Sample trailing stop configuration"""
    return ExitConfiguration(
        strategy_id="trailing_test_001",
        strategy_type=ExitType.TRAILING_STOP,
        name="Test Trailing Stop",
        description="Test configuration for trailing stop",
        parameters={
            'initial_stop': Decimal('0.95'),
            'trailing_distance': Decimal('0.03'),
            'min_trailing_distance': Decimal('0.01'),
            'max_trailing_distance': Decimal('0.10'),
            'trailing_mode': 'percentage',
            'update_frequency': 60,
            'price_change_threshold': Decimal('0.001'),
            'volatility_adjustment': True,
            'profit_threshold': Decimal('0.02'),
            'activation_delay': 30,
            'reset_on_loss': False,
            'multiple_trails': False,
            'atr_period': 14,
            'atr_multiplier': Decimal('2.0'),
            'trailing_sensitivity': Decimal('1.0')
        }
    )


# Test implementation of TrailingStopStrategy
class TestTrailingStopStrategy(TrailingStopStrategy):
    """Test implementation for unit testing"""
    
    def __init__(self, config):
        super().__init__(config)
        self.calculate_trailing_stop_calls = []
        self.validate_trail_level_calls = []
        self.exit_triggered = False
    
    async def _calculate_trailing_stop(self, current_price: Decimal, trail_data: Dict[str, Any], position: Dict[str, Any]) -> Decimal:
        """Mock implementation for testing"""
        self.calculate_trailing_stop_calls.append({
            'current_price': current_price,
            'trail_data': trail_data,
            'position': position
        })
        
        # Simple test implementation: trail at 3% below current price for longs
        if trail_data.get('is_long', True):
            return current_price * Decimal('0.97')
        else:
            return current_price * Decimal('1.03')
    
    async def _validate_trail_level(self, new_trail: Decimal, trail_data: Dict[str, Any], position: Dict[str, Any]) -> Decimal:
        """Mock implementation for testing"""
        self.validate_trail_level_calls.append({
            'new_trail': new_trail,
            'trail_data': trail_data,
            'position': position
        })
        
        # Simple validation: ensure trail is within min/max bounds
        min_trail = position.get('entry_price', Decimal('0')) * self.trailing_config.min_trailing_distance
        max_trail = position.get('entry_price', Decimal('0')) * self.trailing_config.max_trailing_distance
        
        if new_trail < min_trail:
            return min_trail
        elif new_trail > max_trail:
            return max_trail
        else:
            return new_trail
    
    async def _check_exit_trigger(self, position: Dict[str, Any], trail_data: Dict[str, Any]) -> bool:
        """Mock implementation for testing"""
        return self.exit_triggered


# Test implementation of ATRTrailingStop
class TestATRTrailingStop(ATRTrailingStop):
    """Test implementation for ATR trailing stop"""
    
    async def _calculate_atr(self, symbol: str, period: int) -> Decimal:
        """Mock ATR calculation for testing"""
        return Decimal('2.5')  # Mock ATR value


# Mock trailing stop context
@pytest.fixture
def trailing_stop_context():
    """Mock context for trailing stop testing"""
    context = AsyncMock()
    context.get_current_price = AsyncMock(return_value=Decimal('150.00'))
    context.get_position = AsyncMock(return_value={
        'position_id': 'pos_001',
        'symbol': 'AAPL',
        'quantity': Decimal('100'),
        'entry_price': Decimal('140.00'),
        'side': 'long',
        'entry_time': datetime.utcnow() - timedelta(hours=2),
        'created_at': datetime.utcnow() - timedelta(hours=2)
    })
    context.get_historical_data = AsyncMock(return_value=[
        {
            'timestamp': datetime.utcnow() - timedelta(hours=i),
            'open': Decimal(str(140 + i * 0.5)),
            'high': Decimal(str(142 + i * 0.5)),
            'low': Decimal(str(138 + i * 0.5)),
            'close': Decimal(str(140 + i * 0.5)),
            'volume': 1000000
        }
        for i in range(30, 0, -1)
    ])
    context.calculate_volatility = AsyncMock(return_value=Decimal('0.025'))
    context.submit_exit_order = AsyncMock(return_value={'success': True, 'order_id': 'order_123'})
    context.get_portfolio_value = AsyncMock(return_value=Decimal('1000000'))
    context.get_risk_metrics = AsyncMock(return_value={'max_drawdown': 0.05, 'var_99': Decimal('5000')})
    context.get_positions = AsyncMock(return_value=[{
        'position_id': 'pos_001',
        'symbol': 'AAPL',
        'quantity': Decimal('100'),
        'entry_price': Decimal('140.00'),
        'side': 'long',
        'current_price': Decimal('150.00'),
        'entry_time': datetime.utcnow() - timedelta(hours=2),
        'created_at': datetime.utcnow() - timedelta(hours=2)
    }])
    return context


class TestTrailingStopStrategy:
    """Test TrailingStopStrategy base class"""
    
    def test_initialization(self, trailing_stop_config):
        """Test proper initialization of TrailingStopStrategy"""
        strategy = TestTrailingStopStrategy(trailing_stop_config)
        
        # Test inherited attributes
        assert strategy.config == trailing_stop_config
        assert strategy.status.value == "initializing"
        
        # Test trailing stop specific attributes
        assert isinstance(strategy.trailing_config, TrailingStopConfig)
        assert strategy.trailing_config.initial_stop == Decimal('0.95')
        assert strategy.trailing_config.trailing_distance == Decimal('0.03')
        assert strategy.trailing_config.trailing_mode == 'percentage'
        assert strategy.trailing_config.update_frequency == 60
        
        # Test tracking attributes
        assert len(strategy.position_trails) == 0
        assert len(strategy.last_trail_updates) == 0
        assert len(strategy.trail_adjustments) == 0
        assert len(strategy.price_history) == 0
        assert len(strategy.volatility_cache) == 0
    
    def test_trailing_config_creation(self):
        """Test TrailingStopConfig creation and validation"""
        config = TrailingStopConfig(
            initial_stop=Decimal('0.95'),
            trailing_distance=Decimal('0.03'),
            min_trailing_distance=Decimal('0.01'),
            max_trailing_distance=Decimal('0.10'),
            trailing_mode='percentage',
            update_frequency=60,
            price_change_threshold=Decimal('0.001'),
            volatility_adjustment=True,
            profit_threshold=Decimal('0.02'),
            activation_delay=30,
            reset_on_loss=False,
            multiple_trails=False,
            atr_period=14,
            atr_multiplier=Decimal('2.0'),
            trailing_sensitivity=Decimal('1.0')
        )
        
        assert config.initial_stop == Decimal('0.95')
        assert config.trailing_distance == Decimal('0.03')
        assert config.trailing_mode == 'percentage'
        assert config.update_frequency == 60
        assert config.volatility_adjustment is True
        assert config.profit_threshold == Decimal('0.02')
        assert config.atr_period == 14
        assert config.atr_multiplier == Decimal('2.0')
        
        # Test default values
        assert config.min_trailing_distance == Decimal('0.005')
        assert config.max_trailing_distance == Decimal('0.10')
        assert config.price_change_threshold == Decimal('0.001')
        assert config.activation_delay == 0
        assert config.reset_on_loss is False
        assert config.multiple_trails is False
        assert config.trailing_sensitivity == Decimal('1.0')
    
    @pytest.mark.asyncio
    async def test_is_position_eligible(self, trailing_stop_config, trailing_stop_context):
        """Test position eligibility checking"""
        strategy = TestTrailingStopStrategy(trailing_stop_config)
        strategy.set_context(trailing_stop_context)
        
        # Test valid position
        valid_position = {
            'position_id': 'pos_001',
            'symbol': 'AAPL',
            'entry_price': Decimal('140.00'),
            'created_at': datetime.utcnow() - timedelta(hours=1)
        }
        
        # Should be eligible without activation delay or profit threshold
        result = await strategy._is_position_eligible(valid_position)
        assert result is True
        
        # Test with activation delay
        strategy.trailing_config.activation_delay = 60  # 1 minute
        new_position = {
            'position_id': 'pos_002',
            'symbol': 'AAPL',
            'entry_price': Decimal('140.00'),
            'created_at': datetime.utcnow() - timedelta(seconds=30)  # Only 30 seconds old
        }
        
        result = await strategy._is_position_eligible(new_position)
        assert result is False
        
        # Test with profit threshold (failing)
        old_position = {
            'position_id': 'pos_003',
            'symbol': 'AAPL',
            'entry_price': Decimal('140.00'),
            'created_at': datetime.utcnow() - timedelta(hours=1)
        }
        
        trailing_stop_context.get_current_price.return_value = Decimal('141.00')  # Only 0.7% profit
        strategy.trailing_config.profit_threshold = Decimal('0.02')  # 2% required
        
        result = await strategy._is_position_eligible(old_position)
        assert result is False
        
        # Test with profit threshold (passing)
        trailing_stop_context.get_current_price.return_value = Decimal('145.00')  # 3.6% profit
        
        result = await strategy._is_position_eligible(old_position)
        assert result is True
        
        # Test position without required data
        invalid_position = {
            'position_id': 'pos_004'
            # Missing entry_price or symbol
        }
        
        result = await strategy._is_position_eligible(invalid_position)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_or_initialize_trail(self, trailing_stop_config, trailing_stop_context):
        """Test trail initialization for positions"""
        strategy = TestTrailingStopStrategy(trailing_stop_config)
        strategy.set_context(trailing_stop_context)
        
        # Test long position
        long_position = {
            'position_id': 'pos_001',
            'symbol': 'AAPL',
            'entry_price': Decimal('140.00'),
            'side': 'long'
        }
        
        trail_data = await strategy._get_or_initialize_trail(long_position)
        
        assert 'position_id' in trail_data
        assert trail_data['position_id'] == 'pos_001'
        assert trail_data['initial_trail'] == Decimal('0.95')
        assert trail_data['current_trail'] == Decimal('0.95')
        assert trail_data['highest_price'] == Decimal('140.00')  # Entry price for long
        assert trail_data['lowest_price'] == Decimal('0')
        assert trail_data['is_long'] is True
        assert trail_data['adjustments'] == []
        assert isinstance(trail_data['activated_at'], datetime)
        assert isinstance(trail_data['last_update'], datetime)
        
        # Test short position
        short_position = {
            'position_id': 'pos_002',
            'symbol': 'AAPL',
            'entry_price': Decimal('140.00'),
            'side': 'short'
        }
        
        trail_data_short = await strategy._get_or_initialize_trail(short_position)
        
        assert trail_data_short['is_long'] is False
        assert trail_data_short['highest_price'] == Decimal('0')
        assert trail_data_short['lowest_price'] == Decimal('140.00')  # Entry price for short
        
        # Test getting existing trail
        existing_trail = await strategy._get_or_initialize_trail(long_position)
        assert existing_trail == trail_data  # Same object
        
        # Check that position was added to tracking
        assert 'pos_001' in strategy.position_trails
        assert strategy.position_trails['pos_001'] == trail_data
    
    @pytest.mark.asyncio
    async def test_update_trailing_stop(self, trailing_stop_config, trailing_stop_context):
        """Test trailing stop level updates"""
        strategy = TestTrailingStopStrategy(trailing_stop_config)
        strategy.set_context(trailing_stop_context)
        
        position = {
            'position_id': 'pos_001',
            'symbol': 'AAPL',
            'entry_price': Decimal('140.00'),
            'side': 'long'
        }
        
        # Initialize trail
        trail_data = await strategy._get_or_initialize_trail(position)
        
        # Mock current price
        trailing_stop_context.get_current_price.return_value = Decimal('150.00')
        
        # Update trailing stop
        await strategy._update_trailing_stop(position, trail_data)
        
        # Check that calculations were called
        assert len(strategy.calculate_trailing_stop_calls) == 1
        assert len(strategy.validate_trail_level_calls) == 1
        
        # Check that trail was updated
        assert trail_data['current_trail'] != Decimal('0.95')  # Should be updated
        assert trail_data['last_update'] is not None
        
        # Check that adjustment was recorded
        assert len(trail_data['adjustments']) == 1
        adjustment = trail_data['adjustments'][0]
        assert 'timestamp' in adjustment
        assert 'old_trail' in adjustment
        assert 'new_trail' in adjustment
        assert 'current_price' in adjustment
        assert 'reason' in adjustment
        
        # Test update frequency throttling
        await strategy._update_trailing_stop(position, trail_data)
        # Should not add another adjustment due to frequency throttling
        assert len(trail_data['adjustments']) == 1
    
    @pytest.mark.asyncio
    async def test_evaluate_exit_conditions(self, trailing_stop_config, trailing_stop_context):
        """Test exit condition evaluation"""
        strategy = TestTrailingStopStrategy(trailing_stop_config)
        strategy.set_context(trailing_stop_context)
        
        position = {
            'position_id': 'pos_001',
            'symbol': 'AAPL',
            'entry_price': Decimal('140.00'),
            'side': 'long'
        }
        
        # Test when exit is not triggered
        strategy.exit_triggered = False
        result = await strategy.evaluate_exit_conditions(position)
        assert result is False
        
        # Test when exit is triggered
        strategy.exit_triggered = True
        result = await strategy.evaluate_exit_conditions(position)
        assert result is True
        
        # Test with invalid position
        invalid_position = {}
        result = await strategy.evaluate_exit_conditions(invalid_position)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_generate_exit_signal(self, trailing_stop_config, trailing_stop_context):
        """Test exit signal generation"""
        strategy = TestTrailingStopStrategy(trailing_stop_config)
        strategy.set_context(trailing_stop_context)
        
        position = {
            'position_id': 'pos_001',
            'symbol': 'AAPL',
            'quantity': Decimal('100'),
            'entry_price': Decimal('140.00'),
            'side': 'long'
        }
        
        # Set up trail data
        trail_data = {
            'current_trail': Decimal('145.50'),
            'is_long': True,
            'adjustments': [{'timestamp': datetime.utcnow(), 'old_trail': Decimal('142.00'), 'new_trail': Decimal('145.50')}],
            'activated_at': datetime.utcnow() - timedelta(hours=1)
        }
        strategy.position_trails['pos_001'] = trail_data
        
        # Generate exit signal
        exit_signal = await strategy.generate_exit_signal(position, ExitReason.TRAILING_STOP)
        
        assert exit_signal is not None
        assert exit_signal.strategy_id == trailing_stop_config.strategy_id
        assert exit_signal.position_id == 'pos_001'
        assert exit_signal.symbol == 'AAPL'
        assert exit_signal.exit_reason == ExitReason.TRAILING_STOP
        assert exit_signal.exit_quantity == Decimal('100')
        assert 0.0 <= exit_signal.confidence <= 1.0
        assert 0.0 <= exit_signal.urgency <= 1.0
        assert 'trail_level' in exit_signal.metadata
        assert 'entry_price' in exit_signal.metadata
        assert 'current_price' in exit_signal.metadata
        assert 'trail_type' in exit_signal.metadata
        assert 'trail_adjustments' in exit_signal.metadata
        
        # Test with missing trail data
        strategy.position_trails.clear()
        exit_signal = await strategy.generate_exit_signal(position, ExitReason.TRAILING_STOP)
        assert exit_signal is None
    
    @pytest.mark.asyncio
    async def test_calculate_exit_confidence(self, trailing_stop_config, trailing_stop_context):
        """Test exit confidence calculation"""
        strategy = TestTrailingStopStrategy(trailing_stop_config)
        strategy.set_context(trailing_stop_context)
        
        position = {
            'position_id': 'pos_001',
            'symbol': 'AAPL',
            'entry_price': Decimal('140.00'),
            'side': 'long'
        }
        
        trail_data = {
            'current_trail': Decimal('145.00'),
            'highest_price': Decimal('155.00'),
            'is_long': True,
            'adjustments': [{'timestamp': datetime.utcnow()}, {'timestamp': datetime.utcnow()}]
        }
        
        # Mock high price scenario
        trailing_stop_context.get_current_price.return_value = Decimal('150.00')
        
        # The actual implementation would calculate confidence based on various factors
        confidence = await strategy._calculate_exit_confidence(position, trail_data)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
    
    def test_calculate_exit_urgency(self, trailing_stop_config):
        """Test exit urgency calculation"""
        strategy = TestTrailingStopStrategy(trailing_stop_config)
        
        # Test when price is very close to trail (high urgency)
        current_price = Decimal('145.10')
        trail_level = Decimal('145.00')
        urgency = strategy._calculate_exit_urgency(current_price, trail_level)
        assert urgency > 0.8  # Should be high urgency
        
        # Test when price is far from trail (low urgency)
        current_price = Decimal('150.00')
        trail_level = Decimal('145.00')
        urgency = strategy._calculate_exit_urgency(current_price, trail_level)
        assert urgency < 0.3  # Should be low urgency
        
        # Test equal prices
        current_price = Decimal('145.00')
        trail_level = Decimal('145.00')
        urgency = strategy._calculate_exit_urgency(current_price, trail_level)
        assert urgency == 1.0  # Maximum urgency when equal
    
    @pytest.mark.asyncio
    async def test_estimate_market_impact(self, trailing_stop_config, trailing_stop_context):
        """Test market impact estimation"""
        strategy = TestTrailingStopStrategy(trailing_stop_config)
        strategy.set_context(trailing_stop_context)
        
        # Test with large quantity (high impact)
        impact_large = await strategy._estimate_market_impact('AAPL', Decimal('10000'))
        assert isinstance(impact_large, Decimal)
        assert impact_large > 0
        
        # Test with small quantity (low impact)
        impact_small = await strategy._estimate_market_impact('AAPL', Decimal('100'))
        assert isinstance(impact_small, Decimal)
        assert impact_small >= 0
        
        # Large quantity should generally have higher impact
        assert impact_large >= impact_small


class TestATRTrailingStop:
    """Test ATRTrailingStop implementation"""
    
    def test_atr_trailing_stop_initialization(self, trailing_stop_config):
        """Test ATRTrailingStop initialization"""
        # Update config for ATR mode
        trailing_stop_config.parameters['trailing_mode'] = 'atr'
        trailing_stop_config.parameters['atr_period'] = 14
        trailing_stop_config.parameters['atr_multiplier'] = Decimal('2.5')
        
        strategy = TestATRTrailingStop(trailing_stop_config)
        
        assert strategy.trailing_config.trailing_mode == 'atr'
        assert strategy.trailing_config.atr_period == 14
        assert strategy.trailing_config.atr_multiplier == Decimal('2.5')
    
    @pytest.mark.asyncio
    async def test_atr_calculation(self, trailing_stop_config, trailing_stop_context):
        """Test ATR calculation for trailing stop"""
        strategy = TestATRTrailingStop(trailing_stop_config)
        strategy.set_context(trailing_stop_context)
        
        # Mock historical data for ATR calculation
        atr_data = [
            {'high': Decimal('152'), 'low': Decimal('148'), 'close': Decimal('150')},
            {'high': Decimal('153'), 'low': Decimal('149'), 'close': Decimal('151')},
            {'high': Decimal('154'), 'low': Decimal('150'), 'close': Decimal('152')},
            {'high': Decimal('155'), 'low': Decimal('151'), 'close': Decimal('153')},
            {'high': Decimal('156'), 'low': Decimal('152'), 'close': Decimal('154')}
        ]
        
        trailing_stop_context.get_historical_data.return_value = atr_data
        
        atr_value = await strategy._calculate_atr('AAPL', 14)
        assert isinstance(atr_value, Decimal)
        assert atr_value > 0


class TestFixedTrailingStop:
    """Test FixedTrailingStop implementation"""
    
    def test_fixed_trailing_stop_initialization(self, trailing_stop_config):
        """Test FixedTrailingStop initialization"""
        # Update config for fixed mode
        trailing_stop_config.parameters['trailing_mode'] = 'fixed'
        trailing_stop_config.parameters['trailing_distance'] = Decimal('5.00')  # Fixed $5
        
        strategy = FixedTrailingStop(trailing_stop_config)
        
        assert strategy.trailing_config.trailing_mode == 'fixed'
        assert strategy.trailing_config.trailing_distance == Decimal('5.00')


class TestChandelierExit:
    """Test ChandelierExit implementation"""
    
    def test_chandelier_exit_initialization(self, trailing_stop_config):
        """Test ChandelierExit initialization"""
        # Update config for chandelier exit
        trailing_stop_config.parameters['trailing_mode'] = 'chandelier'
        trailing_stop_config.parameters['atr_period'] = 22
        trailing_stop_config.parameters['atr_multiplier'] = Decimal('3.0')
        
        strategy = ChandelierExit(trailing_stop_config)
        
        assert strategy.trailing_config.trailing_mode == 'chandelier'
        assert strategy.trailing_config.atr_period == 22
        assert strategy.trailing_config.atr_multiplier == Decimal('3.0')


class TestTrailingStopScenarios:
    """Test various market scenarios for trailing stops"""
    
    @pytest.mark.asyncio
    async def test_uptrending_market(self, trailing_stop_config, trailing_stop_context):
        """Test trailing stop behavior in uptrending market"""
        strategy = TestTrailingStopStrategy(trailing_stop_config)
        strategy.set_context(trailing_stop_context)
        
        # Simulate uptrending prices
        uptrend_prices = [Decimal('140'), Decimal('142'), Decimal('145'), Decimal('148'), Decimal('150')]
        
        position = {
            'position_id': 'pos_001',
            'symbol': 'AAPL',
            'entry_price': Decimal('140.00'),
            'side': 'long'
        }
        
        trail_data = await strategy._get_or_initialize_trail(position)
        
        for price in uptrend_prices:
            trailing_stop_context.get_current_price.return_value = price
            await strategy._update_trailing_stop(position, trail_data)
            
            # Trail should generally move up in uptrend
            assert trail_data['current_trail'] >= trail_data['initial_trail']
        
        assert len(trail_data['adjustments']) > 0
    
    @pytest.mark.asyncio
    async def test_sideways_market(self, trailing_stop_config, trailing_stop_context):
        """Test trailing stop behavior in sideways market"""
        strategy = TestTrailingStopStrategy(trailing_stop_config)
        strategy.set_context(trailing_stop_context)
        
        # Simulate sideways prices
        sideways_prices = [Decimal('145'), Decimal('144'), Decimal('146'), Decimal('145'), Decimal('147')]
        
        position = {
            'position_id': 'pos_001',
            'symbol': 'AAPL',
            'entry_price': Decimal('145.00'),
            'side': 'long'
        }
        
        trail_data = await strategy._get_or_initialize_trail(position)
        
        initial_adjustments = len(trail_data['adjustments'])
        
        for price in sideways_prices:
            trailing_stop_context.get_current_price.return_value = price
            await strategy._update_trailing_stop(position, trail_data)
        
        # Should have minimal adjustments in sideways market
        final_adjustments = len(trail_data['adjustments'])
        assert final_adjustments - initial_adjustments <= 3  # Allow for some noise
    
    @pytest.mark.asyncio
    async def test_downtrending_market(self, trailing_stop_config, trailing_stop_context):
        """Test trailing stop behavior in downtrending market"""
        strategy = TestTrailingStopStrategy(trailing_stop_config)
        strategy.set_context(trailing_stop_context)
        
        # Simulate downtrending prices
        downtrend_prices = [Decimal('150'), Decimal('148'), Decimal('145'), Decimal('142'), Decimal('140')]
        
        position = {
            'position_id': 'pos_001',
            'symbol': 'AAPL',
            'entry_price': Decimal('150.00'),
            'side': 'long'
        }
        
        trail_data = await strategy._get_or_initialize_trail(position)
        
        for price in downtrend_prices:
            trailing_stop_context.get_current_price.return_value = price
            
            # Check if exit would be triggered
            exit_triggered = await strategy._check_exit_trigger(position, trail_data)
            
            # When price falls significantly below trail, exit should be triggered
            if price < trail_data['current_trail'] * Decimal('0.95'):
                assert exit_triggered is True


class TestTrailingStopEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_extreme_trailing_distances(self, trailing_stop_config):
        """Test extreme trailing distance values"""
        # Test very small trailing distance
        config_small = TrailingStopConfig(
            initial_stop=Decimal('0.99'),
            trailing_distance=Decimal('0.001'),  # 0.1%
            min_trailing_distance=Decimal('0.0001'),
            max_trailing_distance=Decimal('0.05')
        )
        
        strategy = TestTrailingStopStrategy(trailing_stop_config)
        strategy.trailing_config = config_small
        
        assert strategy.trailing_config.trailing_distance == Decimal('0.001')
        
        # Test very large trailing distance
        config_large = TrailingStopConfig(
            initial_stop=Decimal('0.80'),
            trailing_distance=Decimal('0.20'),  # 20%
            min_trailing_distance=Decimal('0.05'),
            max_trailing_distance=Decimal('0.50')
        )
        
        strategy.trailing_config = config_large
        assert strategy.trailing_config.trailing_distance == Decimal('0.20')
    
    @pytest.mark.asyncio
    async def test_missing_market_data(self, trailing_stop_config):
        """Test behavior with missing market data"""
        strategy = TestTrailingStopStrategy(trailing_stop_config)
        
        # Mock context that returns None for prices
        context = AsyncMock()
        context.get_current_price.return_value = None
        context.get_historical_data.return_value = []
        context.calculate_volatility.return_value = None
        context.submit_exit_order.return_value = {'success': True}
        context.get_portfolio_value.return_value = Decimal('1000000')
        context.get_risk_metrics.return_value = {}
        context.get_positions.return_value = []
        
        strategy.set_context(context)
        
        position = {
            'position_id': 'pos_001',
            'symbol': 'AAPL',
            'entry_price': Decimal('140.00')
        }
        
        # Should handle missing data gracefully
        trail_data = await strategy._get_or_initialize_trail(position)
        assert trail_data is not None
        
        # Exit evaluation should handle missing data
        result = await strategy.evaluate_exit_conditions(position)
        assert isinstance(result, bool)
    
    @pytest.mark.asyncio
    async def test_concurrent_position_updates(self, trailing_stop_config, trailing_stop_context):
        """Test concurrent position updates"""
        strategy = TestTrailingStopStrategy(trailing_stop_config)
        strategy.set_context(trailing_stop_context)
        
        # Create multiple positions
        positions = [
            {
                'position_id': f'pos_{i:03d}',
                'symbol': 'AAPL',
                'entry_price': Decimal('140.00') + Decimal(str(i)),
                'side': 'long'
            }
            for i in range(10)
        ]
        
        # Initialize trails for all positions
        for position in positions:
            trail_data = await strategy._get_or_initialize_trail(position)
            assert trail_data is not None
        
        # All positions should be tracked
        assert len(strategy.position_trails) == 10
        
        # Update all positions concurrently
        tasks = []
        for position in positions:
            trail_data = strategy.position_trails[position['position_id']]
            task = strategy._update_trailing_stop(position, trail_data)
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        # All positions should have been updated
        for position in positions:
            trail_data = strategy.position_trails[position['position_id']]
            assert len(trail_data['adjustments']) > 0
    
    def test_invalid_trailing_mode(self):
        """Test behavior with invalid trailing mode"""
        config = TrailingStopConfig(
            initial_stop=Decimal('0.95'),
            trailing_distance=Decimal('0.03'),
            trailing_mode='invalid_mode'
        )
        
        # Should still create config, but may cause issues later
        assert config.trailing_mode == 'invalid_mode'


if __name__ == "__main__":
    pytest.main([__file__])
