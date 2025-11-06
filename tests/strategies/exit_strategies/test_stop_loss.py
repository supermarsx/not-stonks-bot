"""
Test Stop Loss Exit Strategies

Tests various stop loss exit strategies including:
- StopLossStrategy base class
- Percentage-based stop losses
- Volatility-adjusted stop losses
- Fixed price stop losses
- Stop loss adjustment capabilities
- Stop activation and management
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from trading_orchestrator.strategies.exit_strategies.stop_loss import (
    StopLossStrategy,
    StopLossConfig,
    PercentageStopLoss,
    VolatilityStopLoss,
    FixedStopLoss
)

from trading_orchestrator.strategies.exit_strategies.base_exit_strategy import (
    ExitReason,
    ExitType,
    ExitConfiguration,
    ExitSignal
)


# Test fixture for stop loss configuration
@pytest.fixture
def stop_loss_config():
    """Sample stop loss configuration"""
    return ExitConfiguration(
        strategy_id="stop_loss_001",
        strategy_type=ExitType.STOP_LOSS,
        name="Test Stop Loss",
        description="Test configuration for stop loss strategy",
        parameters={
            'stop_percentage': Decimal('0.05'),  # 5% stop
            'stop_price': None,
            'stop_mode': 'percentage',
            'activation_delay': 300,  # 5 minutes
            'activation_threshold': Decimal('0.01'),  # 1% movement
            'stop_adjustment': False,
            'stop_increment': Decimal('0.01'),
            'max_stop_distance': None,
            'stop_reset_on_profit': False,
            'volatility_adjustment': False,
            'atr_period': 14,
            'atr_multiplier': Decimal('2.0')
        }
    )


# Test fixture for volatility-adjusted stop loss
@pytest.fixture
def volatility_stop_loss_config():
    """Configuration for volatility-adjusted stop loss"""
    return ExitConfiguration(
        strategy_id="vol_stop_loss_001",
        strategy_type=ExitType.STOP_LOSS,
        name="Volatility Stop Loss",
        description="Test configuration for volatility stop loss",
        parameters={
            'stop_percentage': None,
            'stop_price': None,
            'stop_mode': 'volatility',
            'activation_delay': 0,
            'activation_threshold': Decimal('0'),
            'stop_adjustment': True,
            'stop_increment': Decimal('0.01'),
            'max_stop_distance': Decimal('0.20'),  # 20% max
            'stop_reset_on_profit': True,
            'volatility_adjustment': True,
            'atr_period': 14,
            'atr_multiplier': Decimal('2.5')
        }
    )


# Test implementation of StopLossStrategy
class TestStopLossStrategy(StopLossStrategy):
    """Test implementation for unit testing"""
    
    def __init__(self, config):
        super().__init__(config)
        self.stop_calculation_calls = []
        self.adjustment_calls = []
        self.exit_triggered = False
    
    async def _calculate_stop_level(self, position: Dict[str, Any], current_price: Decimal) -> Decimal:
        """Mock implementation for testing"""
        self.stop_calculation_calls.append({
            'position': position,
            'current_price': current_price
        })
        
        entry_price = Decimal(str(position.get('entry_price', 0)))
        side = position.get('side', 'long')
        
        if self.stop_config.stop_mode == 'percentage' and self.stop_config.stop_percentage:
            if side == 'long':
                return entry_price * (Decimal('1') - self.stop_config.stop_percentage)
            else:
                return entry_price * (Decimal('1') + self.stop_config.stop_percentage)
        elif self.stop_config.stop_price:
            return self.stop_config.stop_price
        else:
            return entry_price * Decimal('0.95')  # Default 5% stop
    
    async def _validate_stop_level(self, stop_level: Decimal, position: Dict[str, Any]) -> bool:
        """Mock implementation for testing"""
        if not self.stop_config.max_stop_distance:
            return True
        
        entry_price = Decimal(str(position.get('entry_price', 0)))
        distance = abs(stop_level - entry_price) / entry_price
        
        return distance <= self.stop_config.max_stop_distance
    
    async def _should_activate_stop(self, position: Dict[str, Any], stop_level: Decimal) -> bool:
        """Mock implementation for testing"""
        return self.exit_triggered
    
    async def evaluate_exit_conditions(self, position: Dict[str, Any]) -> bool:
        """Simple test implementation"""
        current_price = position.get('current_price', Decimal('0'))
        if not current_price:
            return False
        
        stop_level = await self._calculate_stop_level(position, current_price)
        return await self._should_activate_stop(position, stop_level)
    
    async def generate_exit_signal(
        self,
        position: Dict[str, Any],
        exit_reason: ExitReason
    ) -> Optional[ExitSignal]:
        """Generate test exit signal"""
        position_id = position.get('position_id', '')
        symbol = position.get('symbol', '')
        quantity = Decimal(str(position.get('quantity', 0)))
        current_price = position.get('current_price', Decimal('0'))
        
        if not all([position_id, symbol, quantity, current_price]):
            return None
        
        return ExitSignal(
            signal_id=f"stop_{position_id}_{datetime.utcnow().timestamp()}",
            strategy_id=self.config.strategy_id,
            position_id=position_id,
            symbol=symbol,
            exit_reason=exit_reason,
            exit_price=current_price,
            exit_quantity=quantity,
            confidence=0.95,
            urgency=0.90,
            metadata={
                'stop_type': exit_reason.value,
                'trigger_price': float(current_price),
                'stop_level': float(await self._calculate_stop_level(position, current_price))
            }
        )


# Mock context for stop loss testing
@pytest.fixture
def stop_loss_context():
    """Mock context for stop loss testing"""
    context = AsyncMock()
    context.get_current_price = AsyncMock(return_value=Decimal('145.00'))
    context.get_position = AsyncMock(return_value={
        'position_id': 'pos_001',
        'symbol': 'AAPL',
        'quantity': Decimal('100'),
        'entry_price': Decimal('150.00'),
        'side': 'long',
        'entry_time': datetime.utcnow() - timedelta(hours=1),
        'created_at': datetime.utcnow() - timedelta(hours=1)
    })
    context.get_historical_data = AsyncMock(return_value=[
        {
            'timestamp': datetime.utcnow() - timedelta(hours=i),
            'high': Decimal(str(150 + i * 0.5)),
            'low': Decimal(str(148 + i * 0.5)),
            'close': Decimal(str(149 + i * 0.5)),
            'volume': 1000000
        }
        for i in range(30, 0, -1)
    ])
    context.calculate_volatility = AsyncMock(return_value=Decimal('0.030'))
    context.submit_exit_order = AsyncMock(return_value={'success': True, 'order_id': 'order_123'})
    context.get_portfolio_value = AsyncMock(return_value=Decimal('1000000'))
    context.get_risk_metrics = AsyncMock(return_value={'max_drawdown': 0.05, 'var_99': Decimal('5000')})
    context.get_positions = AsyncMock(return_value=[{
        'position_id': 'pos_001',
        'symbol': 'AAPL',
        'quantity': Decimal('100'),
        'entry_price': Decimal('150.00'),
        'side': 'long',
        'current_price': Decimal('145.00'),
        'entry_time': datetime.utcnow() - timedelta(hours=1),
        'created_at': datetime.utcnow() - timedelta(hours=1)
    }])
    return context


class TestStopLossConfig:
    """Test StopLossConfig data class"""
    
    def test_stop_loss_config_creation(self):
        """Test StopLossConfig creation and initialization"""
        config = StopLossConfig(
            stop_percentage=Decimal('0.05'),
            stop_price=None,
            stop_mode='percentage',
            activation_delay=300,
            activation_threshold=Decimal('0.01'),
            stop_adjustment=False,
            stop_increment=Decimal('0.01'),
            max_stop_distance=None,
            stop_reset_on_profit=False,
            volatility_adjustment=False,
            atr_period=14,
            atr_multiplier=Decimal('2.0')
        )
        
        assert config.stop_percentage == Decimal('0.05')
        assert config.stop_price is None
        assert config.stop_mode == 'percentage'
        assert config.activation_delay == 300
        assert config.activation_threshold == Decimal('0.01')
        assert config.stop_adjustment is False
        assert config.stop_increment == Decimal('0.01')
        assert config.max_stop_distance is None
        assert config.stop_reset_on_profit is False
        assert config.volatility_adjustment is False
        assert config.atr_period == 14
        assert config.atr_multiplier == Decimal('2.0')
        
        # Test default values
        assert config.stop_increment == Decimal('0.01')
        assert config.atr_period == 14
        assert config.atr_multiplier == Decimal('2.0')
    
    def test_stop_loss_config_fixed_price(self):
        """Test StopLossConfig with fixed price"""
        config = StopLossConfig(
            stop_percentage=None,
            stop_price=Decimal('145.00'),
            stop_mode='fixed',
            activation_delay=0,
            activation_threshold=Decimal('0'),
            stop_adjustment=True,
            volatility_adjustment=False
        )
        
        assert config.stop_percentage is None
        assert config.stop_price == Decimal('145.00')
        assert config.stop_mode == 'fixed'
        assert config.stop_adjustment is True
        assert config.volatility_adjustment is False
    
    def test_stop_loss_config_volatility(self):
        """Test StopLossConfig with volatility adjustment"""
        config = StopLossConfig(
            stop_percentage=Decimal('0.03'),
            stop_mode='volatility',
            activation_delay=60,
            activation_threshold=Decimal('0.005'),
            stop_adjustment=True,
            stop_reset_on_profit=True,
            volatility_adjustment=True,
            atr_period=20,
            atr_multiplier=Decimal('2.5')
        )
        
        assert config.stop_mode == 'volatility'
        assert config.volatility_adjustment is True
        assert config.atr_period == 20
        assert config.atr_multiplier == Decimal('2.5')
        assert config.stop_reset_on_profit is True


class TestStopLossStrategy:
    """Test StopLossStrategy base class"""
    
    def test_initialization(self, stop_loss_config):
        """Test proper initialization of StopLossStrategy"""
        strategy = TestStopLossStrategy(stop_loss_config)
        
        # Test inherited attributes
        assert strategy.config == stop_loss_config
        assert strategy.status.value == "initializing"
        
        # Test stop loss specific attributes
        assert isinstance(strategy.stop_config, StopLossConfig)
        assert strategy.stop_config.stop_percentage == Decimal('0.05')
        assert strategy.stop_config.stop_mode == 'percentage'
        assert strategy.stop_config.activation_delay == 300
        
        # Test tracking attributes
        assert len(strategy.position_stops) == 0
        assert len(strategy.stop_adjustments) == 0
        assert len(strategy.activation_history) == 0
        assert len(strategy.stop_history) == 0
    
    @pytest.mark.asyncio
    async def test_calculate_stop_level_percentage(self, stop_loss_config, stop_loss_context):
        """Test stop level calculation for percentage mode"""
        strategy = TestStopLossStrategy(stop_loss_config)
        strategy.set_context(stop_loss_context)
        
        # Test long position
        long_position = {
            'position_id': 'pos_long',
            'symbol': 'AAPL',
            'entry_price': Decimal('150.00'),
            'side': 'long'
        }
        
        current_price = Decimal('155.00')
        stop_level = await strategy._calculate_stop_level(long_position, current_price)
        
        # Should be 5% below entry price: 150 * (1 - 0.05) = 142.50
        assert stop_level == Decimal('142.50')
        
        # Test short position
        short_position = {
            'position_id': 'pos_short',
            'symbol': 'AAPL',
            'entry_price': Decimal('150.00'),
            'side': 'short'
        }
        
        stop_level_short = await strategy._calculate_stop_level(short_position, current_price)
        
        # Should be 5% above entry price for short: 150 * (1 + 0.05) = 157.50
        assert stop_level_short == Decimal('157.50')
    
    @pytest.mark.asyncio
    async def test_calculate_stop_level_fixed(self, stop_loss_config):
        """Test stop level calculation for fixed price mode"""
        # Update config for fixed price mode
        stop_loss_config.parameters['stop_mode'] = 'fixed'
        stop_loss_config.parameters['stop_price'] = Decimal('145.00')
        
        strategy = TestStopLossStrategy(stop_loss_config)
        
        position = {
            'position_id': 'pos_fixed',
            'symbol': 'AAPL',
            'entry_price': Decimal('150.00'),
            'side': 'long'
        }
        
        current_price = Decimal('155.00')
        stop_level = await strategy._calculate_stop_level(position, current_price)
        
        # Should return fixed price regardless of position
        assert stop_level == Decimal('145.00')
    
    @pytest.mark.asyncio
    async def test_validate_stop_level(self, stop_loss_config):
        """Test stop level validation"""
        strategy = TestStopLossStrategy(stop_loss_config)
        
        # Test with no max distance constraint
        no_max_config = StopLossConfig(
            stop_percentage=Decimal('0.10'),
            stop_mode='percentage'
        )
        strategy.stop_config = no_max_config
        
        position = {'entry_price': Decimal('100.00')}
        stop_level = Decimal('90.00')  # 10% stop
        
        is_valid = await strategy._validate_stop_level(stop_level, position)
        assert is_valid is True
        
        # Test with max distance constraint
        max_distance_config = StopLossConfig(
            stop_percentage=Decimal('0.30'),  # 30% stop
            stop_mode='percentage',
            max_stop_distance=Decimal('0.20')  # Max 20% distance
        )
        strategy.stop_config = max_distance_config
        
        is_valid = await strategy._validate_stop_level(stop_level, position)
        assert is_valid is False  # 30% > 20% max
    
    @pytest.mark.asyncio
    async def test_should_activate_stop(self, stop_loss_config, stop_loss_context):
        """Test stop activation logic"""
        strategy = TestStopLossStrategy(stop_loss_config)
        strategy.set_context(stop_loss_context)
        
        position = {
            'position_id': 'pos_test',
            'symbol': 'AAPL',
            'entry_price': Decimal('150.00'),
            'side': 'long'
        }
        
        # Test when exit is not triggered
        strategy.exit_triggered = False
        stop_level = Decimal('142.50')
        
        should_activate = await strategy._should_activate_stop(position, stop_level)
        assert should_activate is False
        
        # Test when exit is triggered
        strategy.exit_triggered = True
        should_activate = await strategy._should_activate_stop(position, stop_level)
        assert should_activate is True
    
    @pytest.mark.asyncio
    async def test_check_activation_delay(self, stop_loss_config):
        """Test activation delay checking"""
        # Config with activation delay
        delay_config = StopLossConfig(
            stop_percentage=Decimal('0.05'),
            stop_mode='percentage',
            activation_delay=300  # 5 minutes
        )
        stop_loss_config.parameters.update(delay_config.__dict__)
        strategy = TestStopLossStrategy(stop_loss_config)
        
        # Test position within delay
        recent_position = {
            'created_at': datetime.utcnow() - timedelta(minutes=2)  # 2 minutes old
        }
        
        activation_allowed = await strategy._check_activation_delay(recent_position)
        assert activation_allowed is False
        
        # Test position beyond delay
        old_position = {
            'created_at': datetime.utcnow() - timedelta(minutes=10)  # 10 minutes old
        }
        
        activation_allowed = await strategy._check_activation_delay(old_position)
        assert activation_allowed is True
        
        # Test with no activation delay
        no_delay_config = StopLossConfig(
            stop_percentage=Decimal('0.05'),
            stop_mode='percentage',
            activation_delay=0
        )
        stop_loss_config.parameters.update(no_delay_config.__dict__)
        strategy.stop_config = no_delay_config
        
        activation_allowed = await strategy._check_activation_delay(recent_position)
        assert activation_allowed is True
    
    @pytest.mark.asyncio
    async def test_evaluate_exit_conditions(self, stop_loss_config, stop_loss_context):
        """Test exit condition evaluation"""
        strategy = TestStopLossStrategy(stop_loss_config)
        strategy.set_context(stop_loss_context)
        
        position = {
            'position_id': 'pos_001',
            'symbol': 'AAPL',
            'entry_price': Decimal('150.00'),
            'current_price': Decimal('145.00'),
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
        
        # Test with missing current price
        position_no_price = {
            'position_id': 'pos_002',
            'symbol': 'AAPL',
            'entry_price': Decimal('150.00'),
            # Missing current_price
            'side': 'long'
        }
        
        result = await strategy.evaluate_exit_conditions(position_no_price)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_generate_exit_signal(self, stop_loss_config, stop_loss_context):
        """Test exit signal generation"""
        strategy = TestStopLossStrategy(stop_loss_config)
        strategy.set_context(stop_loss_context)
        
        position = {
            'position_id': 'pos_001',
            'symbol': 'AAPL',
            'quantity': Decimal('100'),
            'entry_price': Decimal('150.00'),
            'current_price': Decimal('142.00'),
            'side': 'long'
        }
        
        exit_signal = await strategy.generate_exit_signal(position, ExitReason.STOP_LOSS)
        
        assert exit_signal is not None
        assert exit_signal.strategy_id == stop_loss_config.strategy_id
        assert exit_signal.position_id == 'pos_001'
        assert exit_signal.symbol == 'AAPL'
        assert exit_signal.exit_reason == ExitReason.STOP_LOSS
        assert exit_signal.exit_price == Decimal('142.00')
        assert exit_signal.exit_quantity == Decimal('100')
        assert exit_signal.confidence == 0.95
        assert exit_signal.urgency == 0.90
        assert 'stop_type' in exit_signal.metadata
        assert 'trigger_price' in exit_signal.metadata
        assert 'stop_level' in exit_signal.metadata
        
        # Test with incomplete position data
        incomplete_position = {
            'position_id': 'pos_002'
            # Missing other required fields
        }
        
        exit_signal = await strategy.generate_exit_signal(incomplete_position, ExitReason.STOP_LOSS)
        assert exit_signal is None
    
    @pytest.mark.asyncio
    async def test_adjust_stop_level(self, stop_loss_config, stop_loss_context):
        """Test stop level adjustment functionality"""
        # Config with stop adjustment enabled
        adjust_config = StopLossConfig(
            stop_percentage=Decimal('0.05'),
            stop_mode='percentage',
            stop_adjustment=True,
            stop_reset_on_profit=True
        )
        stop_loss_config.parameters.update(adjust_config.__dict__)
        strategy = TestStopLossStrategy(stop_loss_config)
        strategy.set_context(stop_loss_context)
        
        position = {
            'position_id': 'pos_001',
            'symbol': 'AAPL',
            'entry_price': Decimal('150.00'),
            'side': 'long'
        }
        
        # Initialize stop
        initial_stop = await strategy._calculate_stop_level(position, Decimal('155.00'))
        assert initial_stop == Decimal('142.50')
        
        # Simulate profit movement
        profit_position = position.copy()
        profit_position['current_price'] = Decimal('160.00')  # 6.7% profit
        
        # Adjust stop upward for profit
        new_stop = await strategy._calculate_stop_level(profit_position, Decimal('160.00'))
        
        # Should have same percentage stop relative to new price
        expected_stop = Decimal('160.00') * Decimal('0.95')  # 5% below current
        assert abs(new_stop - expected_stop) < Decimal('0.01')


class TestPercentageStopLoss:
    """Test PercentageStopLoss implementation"""
    
    def test_percentage_stop_loss_initialization(self, stop_loss_config):
        """Test PercentageStopLoss initialization"""
        stop_loss_config.parameters['stop_mode'] = 'percentage'
        stop_loss_config.parameters['stop_percentage'] = Decimal('0.03')
        
        strategy = PercentageStopLoss(stop_loss_config)
        
        assert strategy.stop_config.stop_mode == 'percentage'
        assert strategy.stop_config.stop_percentage == Decimal('0.03')
    
    @pytest.mark.asyncio
    async def test_calculate_percentage_stop(self, stop_loss_config, stop_loss_context):
        """Test percentage stop calculation"""
        strategy = PercentageStopLoss(stop_loss_config)
        strategy.set_context(stop_loss_context)
        
        position = {
            'position_id': 'pos_percentage',
            'symbol': 'AAPL',
            'entry_price': Decimal('100.00'),
            'side': 'long'
        }
        
        # Test long position
        long_stop = await strategy._calculate_stop_level(position, Decimal('105.00'))
        assert long_stop == Decimal('95.00')  # 5% below entry
        
        # Test short position
        position['side'] = 'short'
        short_stop = await strategy._calculate_stop_level(position, Decimal('105.00'))
        assert short_stop == Decimal('105.00')  # 5% above entry


class TestVolatilityStopLoss:
    """Test VolatilityStopLoss implementation"""
    
    def test_volatility_stop_loss_initialization(self, volatility_stop_loss_config):
        """Test VolatilityStopLoss initialization"""
        strategy = VolatilityStopLoss(volatility_stop_loss_config)
        
        assert strategy.stop_config.stop_mode == 'volatility'
        assert strategy.stop_config.volatility_adjustment is True
        assert strategy.stop_config.atr_multiplier == Decimal('2.5')
    
    @pytest.mark.asyncio
    async def test_calculate_volatility_stop(self, volatility_stop_loss_config, stop_loss_context):
        """Test volatility stop calculation"""
        strategy = VolatilityStopLoss(volatility_stop_loss_config)
        strategy.set_context(stop_loss_context)
        
        # Mock ATR calculation
        with patch.object(strategy, '_calculate_atr', return_value=Decimal('2.0')):
            position = {
                'position_id': 'pos_volatility',
                'symbol': 'AAPL',
                'entry_price': Decimal('100.00'),
                'side': 'long'
            }
            
            current_price = Decimal('102.00')
            stop_level = await strategy._calculate_stop_level(position, current_price)
            
            # Should be entry_price - (ATR * multiplier)
            expected_stop = Decimal('100.00') - (Decimal('2.0') * Decimal('2.5'))
            assert abs(stop_level - expected_stop) < Decimal('0.01')


class TestFixedStopLoss:
    """Test FixedStopLoss implementation"""
    
    @pytest.mark.asyncio
    async def test_calculate_fixed_stop(self, stop_loss_context):
        """Test fixed stop calculation"""
        # Create config with fixed stop price
        config = ExitConfiguration(
            strategy_id="fixed_stop_001",
            strategy_type=ExitType.STOP_LOSS,
            name="Fixed Stop Loss",
            description="Test fixed stop loss",
            parameters={
                'stop_price': Decimal('145.00'),
                'stop_mode': 'fixed'
            }
        )
        
        strategy = FixedStopLoss(config)
        strategy.set_context(stop_loss_context)
        
        position = {
            'position_id': 'pos_fixed',
            'symbol': 'AAPL',
            'entry_price': Decimal('150.00'),
            'side': 'long'
        }
        
        current_price = Decimal('155.00')
        stop_level = await strategy._calculate_stop_level(position, current_price)
        
        # Should always return the fixed price
        assert stop_level == Decimal('145.00')


class TestStopLossScenarios:
    """Test various market scenarios for stop losses"""
    
    @pytest.mark.asyncio
    async def test_hit_stop_loss(self, stop_loss_config, stop_loss_context):
        """Test scenario where stop loss is hit"""
        strategy = TestStopLossStrategy(stop_loss_config)
        strategy.set_context(stop_loss_context)
        
        position = {
            'position_id': 'pos_hit',
            'symbol': 'AAPL',
            'quantity': Decimal('100'),
            'entry_price': Decimal('150.00'),
            'current_price': Decimal('142.00'),  # Below 5% stop (142.50)
            'side': 'long'
        }
        
        # Calculate expected stop level
        stop_level = await strategy._calculate_stop_level(position, Decimal('142.00'))
        assert stop_level == Decimal('142.50')
        
        # Position price is below stop level, so stop should trigger
        strategy.exit_triggered = True
        exit_triggered = await strategy.evaluate_exit_conditions(position)
        assert exit_triggered is True
    
    @pytest.mark.asyncio
    async def test_stop_not_hit(self, stop_loss_config, stop_loss_context):
        """Test scenario where stop loss is not hit"""
        strategy = TestStopLossStrategy(stop_loss_config)
        strategy.set_context(stop_loss_context)
        
        position = {
            'position_id': 'pos_safe',
            'symbol': 'AAPL',
            'quantity': Decimal('100'),
            'entry_price': Decimal('150.00'),
            'current_price': Decimal('147.00'),  # Above 5% stop (142.50)
            'side': 'long'
        }
        
        strategy.exit_triggered = False
        exit_triggered = await strategy.evaluate_exit_conditions(position)
        assert exit_triggered is False
    
    @pytest.mark.asyncio
    async def test_volatile_market_behavior(self, stop_loss_config, stop_loss_context):
        """Test stop loss behavior in volatile market"""
        strategy = TestStopLossStrategy(stop_loss_config)
        strategy.set_context(stop_loss_context)
        
        # Simulate volatile price movements
        volatile_prices = [Decimal('150'), Decimal('148'), Decimal('152'), Decimal('145'), Decimal('151')]
        
        position = {
            'position_id': 'pos_volatile',
            'symbol': 'AAPL',
            'entry_price': Decimal('150.00'),
            'side': 'long'
        }
        
        for price in volatile_prices:
            position['current_price'] = price
            stop_level = await strategy._calculate_stop_level(position, price)
            
            # Stop level should remain constant for non-adjusting stops
            if not strategy.stop_config.stop_adjustment:
                assert stop_level == Decimal('142.50')  # Initial 5% stop
    
    @pytest.mark.asyncio
    async def test_profitable_adjustment(self, volatility_stop_loss_config, stop_loss_context):
        """Test stop adjustment in profitable position"""
        strategy = TestStopLossStrategy(volatility_stop_loss_config)
        strategy.set_context(stop_loss_context)
        
        position = {
            'position_id': 'pos_profit',
            'symbol': 'AAPL',
            'entry_price': Decimal('100.00'),
            'side': 'long'
        }
        
        # Initial stop
        initial_stop = await strategy._calculate_stop_level(position, Decimal('100.00'))
        
        # Profitable movement
        position['current_price'] = Decimal('110.00')
        profit_stop = await strategy._calculate_stop_level(position, Decimal('110.00'))
        
        # With stop reset on profit, stop should move up
        if strategy.stop_config.stop_reset_on_profit:
            assert profit_stop >= initial_stop


class TestStopLossEdgeCases:
    """Test edge cases and error conditions"""
    
    @pytest.mark.asyncio
    async def test_zero_stop_percentage(self, stop_loss_context):
        """Test behavior with zero stop percentage"""
        config = ExitConfiguration(
            strategy_id="zero_stop",
            strategy_type=ExitType.STOP_LOSS,
            name="Zero Stop",
            description="Test zero stop percentage",
            parameters={
                'stop_percentage': Decimal('0'),
                'stop_mode': 'percentage'
            }
        )
        
        strategy = TestStopLossStrategy(config)
        strategy.set_context(stop_loss_context)
        
        position = {
            'position_id': 'pos_zero',
            'symbol': 'AAPL',
            'entry_price': Decimal('100.00'),
            'side': 'long'
        }
        
        stop_level = await strategy._calculate_stop_level(position, Decimal('100.00'))
        # Should be entry price * (1 - 0) = entry price
        assert stop_level == Decimal('100.00')
    
    @pytest.mark.asyncio
    async def test_very_large_stop_percentage(self, stop_loss_context):
        """Test behavior with very large stop percentage"""
        config = ExitConfiguration(
            strategy_id="large_stop",
            strategy_type=ExitType.STOP_LOSS,
            name="Large Stop",
            description="Test large stop percentage",
            parameters={
                'stop_percentage': Decimal('0.50'),  # 50% stop
                'stop_mode': 'percentage'
            }
        )
        
        strategy = TestStopLossStrategy(config)
        strategy.set_context(stop_loss_context)
        
        position = {
            'position_id': 'pos_large',
            'symbol': 'AAPL',
            'entry_price': Decimal('100.00'),
            'side': 'long'
        }
        
        stop_level = await strategy._calculate_stop_level(position, Decimal('100.00'))
        # Should be entry price * (1 - 0.5) = 50% of entry
        assert stop_level == Decimal('50.00')
    
    @pytest.mark.asyncio
    async def test_negative_prices(self, stop_loss_context):
        """Test behavior with negative prices"""
        config = ExitConfiguration(
            strategy_id="negative_test",
            strategy_type=ExitType.STOP_LOSS,
            name="Negative Test",
            description="Test negative prices",
            parameters={
                'stop_percentage': Decimal('0.05'),
                'stop_mode': 'percentage'
            }
        )
        
        strategy = TestStopLossStrategy(config)
        strategy.set_context(stop_loss_context)
        
        position = {
            'position_id': 'pos_negative',
            'symbol': 'TEST',
            'entry_price': Decimal('-100.00'),  # Negative entry
            'side': 'long'
        }
        
        stop_level = await strategy._calculate_stop_level(position, Decimal('-90.00'))
        # Should handle negative prices gracefully
        assert isinstance(stop_level, Decimal)
    
    @pytest.mark.asyncio
    async def test_missing_entry_price(self, stop_loss_context):
        """Test behavior with missing entry price"""
        config = ExitConfiguration(
            strategy_id="missing_entry",
            strategy_type=ExitType.STOP_LOSS,
            name="Missing Entry",
            description="Test missing entry price",
            parameters={
                'stop_percentage': Decimal('0.05'),
                'stop_mode': 'percentage'
            }
        )
        
        strategy = TestStopLossStrategy(config)
        strategy.set_context(stop_loss_context)
        
        position = {
            'position_id': 'pos_missing',
            'symbol': 'AAPL',
            # Missing entry_price
            'side': 'long'
        }
        
        stop_level = await strategy._calculate_stop_level(position, Decimal('100.00'))
        # Should handle missing data gracefully, using default behavior
        assert isinstance(stop_level, Decimal)
    
    def test_stop_modes_validation(self):
        """Test different stop mode configurations"""
        # Valid percentage mode
        percent_config = StopLossConfig(
            stop_percentage=Decimal('0.05'),
            stop_mode='percentage'
        )
        assert percent_config.stop_percentage is not None
        
        # Valid fixed mode
        fixed_config = StopLossConfig(
            stop_price=Decimal('145.00'),
            stop_mode='fixed'
        )
        assert fixed_config.stop_price is not None
        
        # Valid volatility mode
        volatility_config = StopLossConfig(
            stop_mode='volatility',
            volatility_adjustment=True
        )
        assert volatility_config.stop_mode == 'volatility'


if __name__ == "__main__":
    pytest.main([__file__])
