"""
Test Fixed Target Exit Strategies

Tests various fixed target exit strategies including:
- FixedTargetStrategy base class
- Profit target monitoring
- Loss target monitoring
- Partial exit capabilities
- Multiple target level support
- Time-based target management
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from trading_orchestrator.strategies.exit_strategies.fixed_target import (
    FixedTargetStrategy,
    FixedTargetConfig
)

from trading_orchestrator.strategies.exit_strategies.base_exit_strategy import (
    ExitReason,
    ExitType,
    ExitConfiguration,
    ExitSignal
)


# Test fixture for fixed target configuration
@pytest.fixture
def fixed_target_config():
    """Sample fixed target configuration"""
    return ExitConfiguration(
        strategy_id="fixed_target_001",
        strategy_type=ExitType.FIXED_TARGET,
        name="Test Fixed Target",
        description="Test configuration for fixed target strategy",
        parameters={
            'profit_target': Decimal('0.10'),  # 10% profit
            'loss_target': Decimal('0.05'),    # 5% loss
            'partial_exits': False,
            'profit_levels': [],
            'loss_levels': [],
            'target_timeout': timedelta(days=30),
            'early_exit_enabled': True,
            'time_decay_factor': Decimal('1.0'),
            'activation_delay': 0,
            'min_profit_hold_time': 0,
            'max_hold_time': None
        }
    )


# Test fixture for partial exit configuration
@pytest.fixture
def partial_exit_config():
    """Configuration with partial exits enabled"""
    return ExitConfiguration(
        strategy_id="partial_target_001",
        strategy_type=ExitType.FIXED_TARGET,
        name="Partial Exit Strategy",
        description="Test configuration for partial exits",
        parameters={
            'profit_target': Decimal('0.15'),
            'loss_target': Decimal('0.08'),
            'partial_exits': True,
            'profit_levels': [Decimal('0.05'), Decimal('0.10'), Decimal('0.15')],
            'partial_sizes': [Decimal('0.25'), Decimal('0.25'), Decimal('0.50')],
            'loss_levels': [Decimal('0.03'), Decimal('0.05'), Decimal('0.08')],
            'level_priorities': [1, 2, 3],
            'target_timeout': timedelta(days=60),
            'early_exit_enabled': True,
            'time_decay_factor': Decimal('0.95'),
            'activation_delay': 300,  # 5 minutes
            'min_profit_hold_time': 600,  # 10 minutes
            'max_hold_time': timedelta(days=90)
        }
    )


# Test implementation of FixedTargetStrategy
class TestFixedTargetStrategy(FixedTargetStrategy):
    """Test implementation for unit testing"""
    
    def __init__(self, config):
        super().__init__(config)
        self.target_calculation_calls = []
        self.exit_decisions = []
    
    async def evaluate_exit_conditions(self, position: Dict[str, Any]) -> bool:
        """Simple test implementation"""
        current_price = position.get('current_price', Decimal('0'))
        entry_price = position.get('entry_price', Decimal('0'))
        side = position.get('side', 'long')
        
        if not current_price or not entry_price:
            return False
        
        # Calculate profit/loss percentage
        if side == 'long':
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price
        
        # Check profit target
        if pnl_pct >= self.target_config.profit_target:
            return True
        
        # Check loss target
        if pnl_pct <= -self.target_config.loss_target:
            return True
        
        return False
    
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
            signal_id=f"target_{position_id}_{datetime.utcnow().timestamp()}",
            strategy_id=self.config.strategy_id,
            position_id=position_id,
            symbol=symbol,
            exit_reason=exit_reason,
            exit_price=current_price,
            exit_quantity=quantity,
            confidence=0.90,
            urgency=0.80,
            metadata={
                'target_type': exit_reason.value,
                'current_price': float(current_price),
                'entry_price': float(position.get('entry_price', 0))
            }
        )


# Mock context for fixed target testing
@pytest.fixture
def fixed_target_context():
    """Mock context for fixed target testing"""
    context = AsyncMock()
    context.get_current_price = AsyncMock(return_value=Decimal('150.00'))
    context.get_position = AsyncMock(return_value={
        'position_id': 'pos_001',
        'symbol': 'AAPL',
        'quantity': Decimal('100'),
        'entry_price': Decimal('140.00'),
        'side': 'long',
        'entry_time': datetime.utcnow() - timedelta(hours=1),
        'created_at': datetime.utcnow() - timedelta(hours=1)
    })
    context.get_historical_data = AsyncMock(return_value=[])
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
        'entry_time': datetime.utcnow() - timedelta(hours=1),
        'created_at': datetime.utcnow() - timedelta(hours=1)
    }])
    return context


class TestFixedTargetConfig:
    """Test FixedTargetConfig data class"""
    
    def test_fixed_target_config_creation(self):
        """Test FixedTargetConfig creation and initialization"""
        config = FixedTargetConfig(
            profit_target=Decimal('0.10'),
            loss_target=Decimal('0.05'),
            partial_exits=False,
            target_timeout=timedelta(days=30),
            early_exit_enabled=True,
            time_decay_factor=Decimal('1.0'),
            activation_delay=0,
            min_profit_hold_time=0,
            max_hold_time=None
        )
        
        assert config.profit_target == Decimal('0.10')
        assert config.loss_target == Decimal('0.05')
        assert config.partial_exits is False
        assert config.profit_levels == []
        assert config.loss_levels == []
        assert config.level_priorities == []
        assert config.partial_sizes == []
        assert config.target_timeout == timedelta(days=30)
        assert config.early_exit_enabled is True
        assert config.time_decay_factor == Decimal('1.0')
        assert config.activation_delay == 0
        assert config.min_profit_hold_time == 0
        assert config.max_hold_time is None
    
    def test_fixed_target_config_with_partial_exits(self):
        """Test FixedTargetConfig with partial exits"""
        config = FixedTargetConfig(
            profit_target=Decimal('0.15'),
            loss_target=Decimal('0.08'),
            partial_exits=True,
            profit_levels=[Decimal('0.05'), Decimal('0.10'), Decimal('0.15')],
            loss_levels=[Decimal('0.03'), Decimal('0.05'), Decimal('0.08')],
            partial_sizes=[Decimal('0.25'), Decimal('0.25'), Decimal('0.50')],
            level_priorities=[1, 2, 3],
            target_timeout=timedelta(days=60),
            time_decay_factor=Decimal('0.95'),
            activation_delay=300,
            min_profit_hold_time=600,
            max_hold_time=timedelta(days=90)
        )
        
        assert config.partial_exits is True
        assert config.profit_levels == [Decimal('0.05'), Decimal('0.10'), Decimal('0.15')]
        assert config.loss_levels == [Decimal('0.03'), Decimal('0.05'), Decimal('0.08')]
        assert config.partial_sizes == [Decimal('0.25'), Decimal('0.25'), Decimal('0.50')]
        assert config.level_priorities == [1, 2, 3]
        assert config.target_timeout == timedelta(days=60)
        assert config.time_decay_factor == Decimal('0.95')
        assert config.activation_delay == 300
        assert config.min_profit_hold_time == 600
        assert config.max_hold_time == timedelta(days=90)
    
    def test_partial_sizes_validation(self):
        """Test partial sizes sum validation"""
        # Valid partial sizes (sum to 1.0)
        valid_config = FixedTargetConfig(
            profit_target=Decimal('0.10'),
            loss_target=Decimal('0.05'),
            partial_exits=True,
            partial_sizes=[Decimal('0.30'), Decimal('0.40'), Decimal('0.30')]  # Sum = 1.0
        )
        
        assert sum(valid_config.partial_sizes) == Decimal('1.0')
        
        # Invalid partial sizes (don't sum to 1.0)
        invalid_config = FixedTargetConfig(
            profit_target=Decimal('0.10'),
            loss_target=Decimal('0.05'),
            partial_exits=True,
            partial_sizes=[Decimal('0.30'), Decimal('0.40'), Decimal('0.20')]  # Sum = 0.9
        )
        
        assert sum(invalid_config.partial_sizes) == Decimal('0.9')  # Should not auto-correct
    
    def test_target_levels_consistency(self):
        """Test target levels consistency"""
        config = FixedTargetConfig(
            profit_target=Decimal('0.15'),
            loss_target=Decimal('0.08'),
            profit_levels=[Decimal('0.05'), Decimal('0.10'), Decimal('0.15')],  # Includes main target
            loss_levels=[Decimal('0.03'), Decimal('0.05'), Decimal('0.08')]   # Includes main target
        )
        
        # Main targets should be included in levels
        assert config.profit_target in config.profit_levels
        assert config.loss_target in config.loss_levels
        
        # Levels should be sorted
        assert config.profit_levels == sorted(config.profit_levels)
        assert config.loss_levels == sorted(config.loss_levels)


class TestFixedTargetStrategy:
    """Test FixedTargetStrategy base class"""
    
    def test_initialization(self, fixed_target_config):
        """Test proper initialization of FixedTargetStrategy"""
        strategy = TestFixedTargetStrategy(fixed_target_config)
        
        # Test inherited attributes
        assert strategy.config == fixed_target_config
        assert strategy.status.value == "initializing"
        
        # Test fixed target specific attributes
        assert isinstance(strategy.target_config, FixedTargetConfig)
        assert strategy.target_config.profit_target == Decimal('0.10')
        assert strategy.target_config.loss_target == Decimal('0.05')
        assert strategy.target_config.partial_exits is False
        assert strategy.target_config.target_timeout == timedelta(days=30)
        
        # Test tracking attributes
        assert len(strategy.position_targets) == 0
        assert len(strategy.partial_exits) == 0
        assert len(strategy.target_history) == 0
        assert len(strategy.level_achievements) == 0
    
    @pytest.mark.asyncio
    async def test_evaluate_exit_conditions_simple(self, fixed_target_config, fixed_target_context):
        """Test exit condition evaluation for simple targets"""
        strategy = TestFixedTargetStrategy(fixed_target_config)
        strategy.set_context(fixed_target_context)
        
        # Test profit target hit
        profit_position = {
            'position_id': 'pos_profit',
            'symbol': 'AAPL',
            'quantity': Decimal('100'),
            'entry_price': Decimal('100.00'),
            'current_price': Decimal('110.00'),  # 10% profit
            'side': 'long'
        }
        
        result = await strategy.evaluate_exit_conditions(profit_position)
        assert result is True
        
        # Test loss target hit
        loss_position = {
            'position_id': 'pos_loss',
            'symbol': 'AAPL',
            'quantity': Decimal('100'),
            'entry_price': Decimal('100.00'),
            'current_price': Decimal('95.00'),  # 5% loss
            'side': 'long'
        }
        
        result = await strategy.evaluate_exit_conditions(loss_position)
        assert result is True
        
        # Test target not hit
        holding_position = {
            'position_id': 'pos_hold',
            'symbol': 'AAPL',
            'quantity': Decimal('100'),
            'entry_price': Decimal('100.00'),
            'current_price': Decimal('103.00'),  # 3% profit
            'side': 'long'
        }
        
        result = await strategy.evaluate_exit_conditions(holding_position)
        assert result is False
        
        # Test short position
        short_position = {
            'position_id': 'pos_short',
            'symbol': 'AAPL',
            'quantity': Decimal('100'),
            'entry_price': Decimal('100.00'),
            'current_price': Decimal('110.00'),  # 10% profit for short
            'side': 'short'
        }
        
        result = await strategy.evaluate_exit_conditions(short_position)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_evaluate_exit_conditions_partial(self, partial_exit_config, fixed_target_context):
        """Test exit condition evaluation with partial exits"""
        strategy = TestFixedTargetStrategy(partial_exit_config)
        strategy.set_context(fixed_target_context)
        
        # Test first profit level
        position_level1 = {
            'position_id': 'pos_level1',
            'symbol': 'AAPL',
            'quantity': Decimal('100'),
            'entry_price': Decimal('100.00'),
            'current_price': Decimal('105.00'),  # 5% profit - first level
            'side': 'long'
        }
        
        # The test implementation uses main targets, so this should not trigger
        result = await strategy.evaluate_exit_conditions(position_level1)
        assert result is False  # 5% < 15% main target
        
        # Test main profit target
        position_main = {
            'position_id': 'pos_main',
            'symbol': 'AAPL',
            'quantity': Decimal('100'),
            'entry_price': Decimal('100.00'),
            'current_price': Decimal('115.00'),  # 15% profit - main target
            'side': 'long'
        }
        
        result = await strategy.evaluate_exit_conditions(position_main)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_generate_exit_signal(self, fixed_target_config, fixed_target_context):
        """Test exit signal generation"""
        strategy = TestFixedTargetStrategy(fixed_target_config)
        strategy.set_context(fixed_target_context)
        
        position = {
            'position_id': 'pos_001',
            'symbol': 'AAPL',
            'quantity': Decimal('100'),
            'entry_price': Decimal('140.00'),
            'current_price': Decimal('154.00'),  # 10% profit
            'side': 'long'
        }
        
        exit_signal = await strategy.generate_exit_signal(position, ExitReason.PROFIT_TARGET)
        
        assert exit_signal is not None
        assert exit_signal.strategy_id == fixed_target_config.strategy_id
        assert exit_signal.position_id == 'pos_001'
        assert exit_signal.symbol == 'AAPL'
        assert exit_signal.exit_reason == ExitReason.PROFIT_TARGET
        assert exit_signal.exit_price == Decimal('154.00')
        assert exit_signal.exit_quantity == Decimal('100')
        assert 0.0 <= exit_signal.confidence <= 1.0
        assert 0.0 <= exit_signal.urgency <= 1.0
        assert 'target_type' in exit_signal.metadata
        assert 'current_price' in exit_signal.metadata
        assert 'entry_price' in exit_signal.metadata
        
        # Test with incomplete position data
        incomplete_position = {
            'position_id': 'pos_002'
            # Missing other required fields
        }
        
        exit_signal = await strategy.generate_exit_signal(incomplete_position, ExitReason.STOP_LOSS)
        assert exit_signal is None
    
    @pytest.mark.asyncio
    async def test_calculate_profit_percentage(self, fixed_target_config):
        """Test profit percentage calculation"""
        strategy = TestFixedTargetStrategy(fixed_target_config)
        
        # Test long position profit
        long_position = {
            'entry_price': Decimal('100.00'),
            'current_price': Decimal('110.00'),
            'side': 'long'
        }
        
        profit_pct = strategy._calculate_profit_percentage(long_position)
        assert profit_pct == Decimal('0.10')  # 10% profit
        
        # Test long position loss
        loss_position = {
            'entry_price': Decimal('100.00'),
            'current_price': Decimal('95.00'),
            'side': 'long'
        }
        
        profit_pct = strategy._calculate_profit_percentage(loss_position)
        assert profit_pct == Decimal('-0.05')  # 5% loss
        
        # Test short position profit (price drop)
        short_profit_position = {
            'entry_price': Decimal('100.00'),
            'current_price': Decimal('90.00'),
            'side': 'short'
        }
        
        profit_pct = strategy._calculate_profit_percentage(short_profit_position)
        assert profit_pct == Decimal('0.10')  # 10% profit for short
        
        # Test short position loss (price rise)
        short_loss_position = {
            'entry_price': Decimal('100.00'),
            'current_price': Decimal('105.00'),
            'side': 'short'
        }
        
        profit_pct = strategy._calculate_profit_percentage(short_loss_position)
        assert profit_pct == Decimal('-0.05')  # 5% loss for short
    
    def test_calculate_exit_confidence(self, fixed_target_config):
        """Test exit confidence calculation"""
        strategy = TestFixedTargetStrategy(fixed_target_config)
        
        # Test high confidence scenario (large profit)
        high_profit_position = {
            'profit_percentage': Decimal('0.15'),  # 15% profit (50% above target)
            'time_to_target': timedelta(hours=1)  # Quick target hit
        }
        
        confidence = strategy._calculate_exit_confidence(high_profit_position, ExitReason.PROFIT_TARGET)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        
        # Test low confidence scenario (just barely hit target)
        low_profit_position = {
            'profit_percentage': Decimal('0.10'),  # Exactly at target
            'time_to_target': timedelta(days=29)  # Almost timeout
        }
        
        confidence = strategy._calculate_exit_confidence(low_profit_position, ExitReason.PROFIT_TARGET)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        
        # Test loss scenario
        loss_position = {
            'profit_percentage': Decimal('-0.08'),  # Beyond loss target
            'time_to_target': timedelta(minutes=30)
        }
        
        confidence = strategy._calculate_exit_confidence(loss_position, ExitReason.STOP_LOSS)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
    
    def test_calculate_exit_urgency(self, fixed_target_config):
        """Test exit urgency calculation"""
        strategy = TestFixedTargetStrategy(fixed_target_config)
        
        # Test high urgency (target significantly exceeded)
        current_price = Decimal('165.00')  # Well above target
        target_price = Decimal('154.00')   # 10% above entry
        entry_price = Decimal('140.00')
        
        urgency = strategy._calculate_exit_urgency(current_price, target_price, entry_price)
        assert isinstance(urgency, float)
        assert urgency >= 0.5  # Should be high urgency
        
        # Test low urgency (barely at target)
        current_price = Decimal('154.01')  # Just barely above target
        target_price = Decimal('154.00')
        entry_price = Decimal('140.00')
        
        urgency = strategy._calculate_exit_urgency(current_price, target_price, entry_price)
        assert isinstance(urgency, float)
        assert urgency >= 0.0
    
    @pytest.mark.asyncio
    async def test_check_target_timeout(self, fixed_target_config):
        """Test target timeout checking"""
        strategy = TestFixedTargetStrategy(fixed_target_config)
        
        # Test with timeout configured
        config_with_timeout = FixedTargetConfig(
            profit_target=Decimal('0.10'),
            loss_target=Decimal('0.05'),
            target_timeout=timedelta(hours=1)
        )
        strategy.target_config = config_with_timeout
        
        # Test position within timeout
        position_recent = {
            'created_at': datetime.utcnow() - timedelta(minutes=30)
        }
        
        timeout_reached = await strategy._check_target_timeout(position_recent)
        assert timeout_reached is False
        
        # Test position beyond timeout
        position_old = {
            'created_at': datetime.utcnow() - timedelta(hours=2)
        }
        
        timeout_reached = await strategy._check_target_timeout(position_old)
        assert timeout_reached is True
        
        # Test with no timeout configured
        config_no_timeout = FixedTargetConfig(
            profit_target=Decimal('0.10'),
            loss_target=Decimal('0.05'),
            target_timeout=None
        )
        strategy.target_config = config_no_timeout
        
        timeout_reached = await strategy._check_target_timeout(position_old)
        assert timeout_reached is False
    
    @pytest.mark.asyncio
    async def test_apply_time_decay(self, fixed_target_config):
        """Test time decay factor application"""
        strategy = TestFixedTargetStrategy(fixed_target_config)
        
        # Test with decay factor
        decay_config = FixedTargetConfig(
            profit_target=Decimal('0.10'),
            loss_target=Decimal('0.05'),
            time_decay_factor=Decimal('0.95')
        )
        strategy.target_config = decay_config
        
        original_profit_target = Decimal('0.10')
        entry_time = datetime.utcnow() - timedelta(days=10)
        
        decayed_target = await strategy._apply_time_decay(original_profit_target, entry_time)
        
        # Should be reduced due to time decay
        assert decayed_target < original_profit_target
        assert decayed_target > 0
        
        # Test with no decay
        no_decay_config = FixedTargetConfig(
            profit_target=Decimal('0.10'),
            loss_target=Decimal('0.05'),
            time_decay_factor=Decimal('1.0')
        )
        strategy.target_config = no_decay_config
        
        decayed_target_no_decay = await strategy._apply_time_decay(original_profit_target, entry_time)
        assert decayed_target_no_decay == original_profit_target


class TestFixedTargetScenarios:
    """Test various market scenarios for fixed targets"""
    
    @pytest.mark.asyncio
    async def test_profitable_position(self, fixed_target_config, fixed_target_context):
        """Test fixed target behavior with profitable position"""
        strategy = TestFixedTargetStrategy(fixed_target_config)
        strategy.set_context(fixed_target_context)
        
        profitable_position = {
            'position_id': 'pos_profit',
            'symbol': 'AAPL',
            'quantity': Decimal('100'),
            'entry_price': Decimal('100.00'),
            'current_price': Decimal('112.00'),  # 12% profit
            'side': 'long'
        }
        
        # Should trigger exit due to profit target
        exit_triggered = await strategy.evaluate_exit_conditions(profitable_position)
        assert exit_triggered is True
        
        # Should generate profit target signal
        exit_signal = await strategy.generate_exit_signal(profitable_position, ExitReason.PROFIT_TARGET)
        assert exit_signal is not None
        assert exit_signal.exit_reason == ExitReason.PROFIT_TARGET
    
    @pytest.mark.asyncio
    async def test_losing_position(self, fixed_target_config, fixed_target_context):
        """Test fixed target behavior with losing position"""
        strategy = TestFixedTargetStrategy(fixed_target_config)
        strategy.set_context(fixed_target_context)
        
        losing_position = {
            'position_id': 'pos_loss',
            'symbol': 'AAPL',
            'quantity': Decimal('100'),
            'entry_price': Decimal('100.00'),
            'current_price': Decimal('93.00'),  # 7% loss
            'side': 'long'
        }
        
        # Should trigger exit due to loss target
        exit_triggered = await strategy.evaluate_exit_conditions(losing_position)
        assert exit_triggered is True
        
        # Should generate stop loss signal
        exit_signal = await strategy.generate_exit_signal(losing_position, ExitReason.STOP_LOSS)
        assert exit_signal is not None
        assert exit_signal.exit_reason == ExitReason.STOP_LOSS
    
    @pytest.mark.asyncio
    async def test_breakeven_position(self, fixed_target_config, fixed_target_context):
        """Test fixed target behavior with breakeven position"""
        strategy = TestFixedTargetStrategy(fixed_target_config)
        strategy.set_context(fixed_target_context)
        
        breakeven_position = {
            'position_id': 'pos_breakeven',
            'symbol': 'AAPL',
            'quantity': Decimal('100'),
            'entry_price': Decimal('100.00'),
            'current_price': Decimal('100.50'),  # 0.5% profit
            'side': 'long'
        }
        
        # Should not trigger exit (neither profit nor loss target hit)
        exit_triggered = await strategy.evaluate_exit_conditions(breakeven_position)
        assert exit_triggered is False
    
    @pytest.mark.asyncio
    async def test_extreme_profit_scenario(self, fixed_target_config, fixed_target_context):
        """Test with extremely profitable position"""
        strategy = TestFixedTargetStrategy(fixed_target_config)
        strategy.set_context(fixed_target_context)
        
        extreme_profit_position = {
            'position_id': 'pos_extreme',
            'symbol': 'AAPL',
            'quantity': Decimal('100'),
            'entry_price': Decimal('100.00'),
            'current_price': Decimal('200.00'),  # 100% profit
            'side': 'long'
        }
        
        # Should definitely trigger exit
        exit_triggered = await strategy.evaluate_exit_conditions(extreme_profit_position)
        assert exit_triggered is True
        
        # Should generate signal with high confidence
        exit_signal = await strategy.generate_exit_signal(extreme_profit_position, ExitReason.PROFIT_TARGET)
        assert exit_signal is not None
        assert exit_signal.confidence > 0.8  # High confidence for extreme profit


class TestFixedTargetEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_extreme_target_values(self):
        """Test extreme target values"""
        # Very tight targets
        tight_config = FixedTargetConfig(
            profit_target=Decimal('0.001'),  # 0.1% profit
            loss_target=Decimal('0.0005')    # 0.05% loss
        )
        
        assert tight_config.profit_target == Decimal('0.001')
        assert tight_config.loss_target == Decimal('0.0005')
        
        # Very wide targets
        wide_config = FixedTargetConfig(
            profit_target=Decimal('1.00'),   # 100% profit
            loss_target=Decimal('0.50')      # 50% loss
        )
        
        assert wide_config.profit_target == Decimal('1.00')
        assert wide_config.loss_target == Decimal('0.50')
    
    @pytest.mark.asyncio
    async def test_zero_quantity_position(self, fixed_target_config, fixed_target_context):
        """Test behavior with zero quantity position"""
        strategy = TestFixedTargetStrategy(fixed_target_config)
        strategy.set_context(fixed_target_context)
        
        zero_qty_position = {
            'position_id': 'pos_zero',
            'symbol': 'AAPL',
            'quantity': Decimal('0'),
            'entry_price': Decimal('100.00'),
            'current_price': Decimal('110.00'),
            'side': 'long'
        }
        
        # Should handle gracefully
        exit_triggered = await strategy.evaluate_exit_conditions(zero_qty_position)
        assert isinstance(exit_triggered, bool)
    
    @pytest.mark.asyncio
    async def test_negative_prices(self, fixed_target_config):
        """Test behavior with negative prices"""
        strategy = TestFixedTargetStrategy(fixed_target_config)
        
        negative_price_position = {
            'position_id': 'pos_negative',
            'symbol': 'TEST',
            'entry_price': Decimal('-100.00'),  # Negative entry price
            'current_price': Decimal('-90.00'),  # Negative current price
            'side': 'long'
        }
        
        # Should handle negative prices gracefully
        profit_pct = strategy._calculate_profit_percentage(negative_price_position)
        # Result may be unexpected but should not crash
        assert isinstance(profit_pct, Decimal)
    
    @pytest.mark.asyncio
    async def test_missing_price_data(self, fixed_target_config):
        """Test behavior with missing price data"""
        strategy = TestFixedTargetStrategy(fixed_target_config)
        
        missing_price_position = {
            'position_id': 'pos_missing',
            'symbol': 'AAPL',
            'entry_price': Decimal('100.00'),
            # Missing current_price
            'side': 'long'
        }
        
        # Should handle missing data gracefully
        exit_triggered = await strategy.evaluate_exit_conditions(missing_price_position)
        assert exit_triggered is False
    
    def test_partial_exit_size_validation(self):
        """Test partial exit size validation logic"""
        strategy = TestFixedTargetStrategy(ExitConfiguration(
            strategy_id="test",
            strategy_type=ExitType.FIXED_TARGET,
            name="Test",
            description="Test",
            parameters={}
        ))
        
        # Test valid partial sizes
        valid_sizes = [Decimal('0.30'), Decimal('0.40'), Decimal('0.30')]
        is_valid = strategy._validate_partial_sizes(valid_sizes)
        assert is_valid is True
        
        # Test invalid partial sizes (don't sum to 1.0)
        invalid_sizes = [Decimal('0.30'), Decimal('0.40'), Decimal('0.20')]
        is_valid = strategy._validate_partial_sizes(invalid_sizes)
        assert is_valid is False
        
        # Test empty partial sizes
        empty_sizes = []
        is_valid = strategy._validate_partial_sizes(empty_sizes)
        assert is_valid is False
        
        # Test negative partial sizes
        negative_sizes = [Decimal('0.50'), Decimal('0.30'), Decimal('0.20')]
        negative_sizes[0] = Decimal('-0.50')  # Make first negative
        is_valid = strategy._validate_partial_sizes(negative_sizes)
        assert is_valid is False


if __name__ == "__main__":
    pytest.main([__file__])
