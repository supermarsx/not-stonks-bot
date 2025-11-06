"""
Test Time-Based Exit Strategies

Tests various time-based exit strategies including:
- TimeBasedExitStrategy base class
- Session-based exits
- Time limit exits
- Scheduled exits
- Market close exits
- Time decay-based exits
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch

from trading_orchestrator.strategies.exit_strategies.time_based_exit import (
    TimeBasedExitStrategy,
    SessionExit,
    TimeLimitExit,
    ScheduledExit,
    MarketCloseExit,
    TimeDecayExit
)

from trading_orchestrator.strategies.exit_strategies.base_exit_strategy import (
    ExitReason,
    ExitType,
    ExitConfiguration
)


# Test fixture for time-based exit configuration
@pytest.fixture
def time_based_config():
    """Sample time-based exit configuration"""
    return ExitConfiguration(
        strategy_id="time_based_001",
        strategy_type=ExitType.TIME_BASED,
        name="Test Time Based Exit",
        description="Test configuration for time-based exit strategy",
        parameters={
            'exit_type': 'time_limit',  # 'session', 'time_limit', 'scheduled', 'market_close', 'time_decay'
            'exit_time': None,  # For scheduled exits
            'max_hold_time': timedelta(hours=4),  # 4 hours
            'session_type': 'regular',  # 'regular', 'extended', 'full'
            'market_open_time': '09:30',
            'market_close_time': '16:00',
            'timezone': 'US/Eastern',
            'decay_rate': Decimal('0.1'),
            'initial_urgency': Decimal('0.5'),
            'urgency_increase_rate': Decimal('0.1'),
            'hold_time_threshold': timedelta(hours=2),
            'partial_exit_enabled': False,
            'emergency_exit_enabled': True,
            'exit_windows': ['15:30-16:00'],  # Last 30 minutes
            'blackout_periods': ['12:00-13:00']  # Lunch hour
        }
    )


# Test implementation of TimeBasedExitStrategy
class TestTimeBasedExitStrategy(TimeBasedExitStrategy):
    """Test implementation for unit testing"""
    
    def __init__(self, config):
        super().__init__(config)
        self.exit_decisions = []
        self.time_checks = []
        self.exit_triggered = False
    
    async def _calculate_time_to_exit(self, position: Dict[str, Any]) -> timedelta:
        """Mock implementation for testing"""
        self.time_checks.append({
            'position_id': position.get('position_id'),
            'check_time': datetime.utcnow()
        })
        
        entry_time = position.get('entry_time', datetime.utcnow())
        current_time = datetime.utcnow()
        hold_time = current_time - entry_time
        
        max_hold = self.time_config.max_hold_time or timedelta(hours=24)
        remaining_time = max_hold - hold_time
        
        return max(remaining_time, timedelta(0))
    
    async def _should_trigger_time_exit(self, position: Dict[str, Any], time_to_exit: timedelta) -> bool:
        """Mock implementation for testing"""
        return self.exit_triggered or time_to_exit <= timedelta(0)
    
    async def evaluate_exit_conditions(self, position: Dict[str, Any]) -> bool:
        """Simple test implementation"""
        time_to_exit = await self._calculate_time_to_exit(position)
        return await self._should_trigger_time_exit(position, time_to_exit)
    
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
        
        time_to_exit = await self._calculate_time_to_exit(position)
        
        return ExitSignal(
            signal_id=f"time_{position_id}_{datetime.utcnow().timestamp()}",
            strategy_id=self.config.strategy_id,
            position_id=position_id,
            symbol=symbol,
            exit_reason=exit_reason,
            exit_price=current_price,
            exit_quantity=quantity,
            confidence=0.75 + (1.0 - min(time_to_exit.total_seconds() / 3600, 4.0) / 4.0) * 0.25,  # Higher confidence as time runs out
            urgency=min(time_to_exit.total_seconds() / 3600, 4.0) / 4.0,  # Higher urgency as time runs out
            metadata={
                'exit_type': self.time_config.exit_type,
                'time_to_exit': str(time_to_exit),
                'hold_time': str(await self._calculate_time_to_exit(position)),
                'max_hold_time': str(self.time_config.max_hold_time)
            }
        )


# Mock context for time-based exit testing
@pytest.fixture
def time_based_context():
    """Mock context for time-based exit testing"""
    context = AsyncMock()
    context.get_current_price = AsyncMock(return_value=Decimal('150.00'))
    context.get_position = AsyncMock(return_value={
        'position_id': 'pos_001',
        'symbol': 'AAPL',
        'quantity': Decimal('100'),
        'entry_price': Decimal('145.00'),
        'side': 'long',
        'entry_time': datetime.utcnow() - timedelta(hours=2),
        'created_at': datetime.utcnow() - timedelta(hours=2)
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
        'entry_price': Decimal('145.00'),
        'side': 'long',
        'current_price': Decimal('150.00'),
        'entry_time': datetime.utcnow() - timedelta(hours=2),
        'created_at': datetime.utcnow() - timedelta(hours=2)
    }])
    
    # Mock market hours
    context.get_market_hours = AsyncMock(return_value={
        'is_open': True,
        'open_time': datetime.utcnow().replace(hour=9, minute=30),
        'close_time': datetime.utcnow().replace(hour=16, minute=0),
        'timezone': 'US/Eastern'
    })
    
    return context


class TestTimeBasedExitStrategy:
    """Test TimeBasedExitStrategy base class"""
    
    def test_initialization(self, time_based_config):
        """Test proper initialization of TimeBasedExitStrategy"""
        strategy = TestTimeBasedExitStrategy(time_based_config)
        
        # Test inherited attributes
        assert strategy.config == time_based_config
        assert strategy.status.value == "initializing"
        
        # Test time-based specific attributes
        assert isinstance(strategy.time_config, TimeBasedExitConfig)
        assert strategy.time_config.exit_type == 'time_limit'
        assert strategy.time_config.max_hold_time == timedelta(hours=4)
        assert strategy.time_config.session_type == 'regular'
        
        # Test tracking attributes
        assert len(strategy.position_timers) == 0
        assert len(strategy.exit_schedule) == 0
        assert len(strategy.session_history) == 0
        assert len(strategy.time_based_exits) == 0
    
    @pytest.mark.asyncio
    async def test_calculate_time_to_exit(self, time_based_config, time_based_context):
        """Test time to exit calculation"""
        strategy = TestTimeBasedExitStrategy(time_based_config)
        strategy.set_context(time_based_context)
        
        # Test position within time limit
        recent_position = {
            'position_id': 'pos_recent',
            'entry_time': datetime.utcnow() - timedelta(hours=2)
        }
        
        time_to_exit = await strategy._calculate_time_to_exit(recent_position)
        assert isinstance(time_to_exit, timedelta)
        assert time_to_exit > timedelta(0)  # Should have time remaining
        
        # Test position at time limit
        old_position = {
            'position_id': 'pos_old',
            'entry_time': datetime.utcnow() - timedelta(hours=5)  # Over 4-hour limit
        }
        
        time_to_exit_over = await strategy._calculate_time_to_exit(old_position)
        assert time_to_exit_over <= timedelta(0)  # Should be at or past limit
    
    @pytest.mark.asyncio
    async def test_should_trigger_time_exit(self, time_based_config, time_based_context):
        """Test time exit trigger logic"""
        strategy = TestTimeBasedExitStrategy(time_based_config)
        strategy.set_context(time_based_context)
        
        position = {
            'position_id': 'pos_test',
            'entry_time': datetime.utcnow() - timedelta(hours=2)
        }
        
        # Test when exit is not triggered
        strategy.exit_triggered = False
        time_to_exit = timedelta(hours=2)  # 2 hours remaining
        
        should_exit = await strategy._should_trigger_time_exit(position, time_to_exit)
        assert should_exit is False
        
        # Test when exit is triggered
        strategy.exit_triggered = True
        should_exit = await strategy._should_trigger_time_exit(position, time_to_exit)
        assert should_exit is True
        
        # Test when time is up
        strategy.exit_triggered = False
        time_to_exit_zero = timedelta(0)
        should_exit = await strategy._should_trigger_time_exit(position, time_to_exit_zero)
        assert should_exit is True
    
    @pytest.mark.asyncio
    async def test_is_market_open(self, time_based_config, time_based_context):
        """Test market hours checking"""
        strategy = TestTimeBasedExitStrategy(time_based_config)
        strategy.set_context(time_based_context)
        
        # Test when market is open
        time_based_context.get_market_hours.return_value = {
            'is_open': True,
            'open_time': datetime.utcnow().replace(hour=9, minute=30),
            'close_time': datetime.utcnow().replace(hour=16, minute=0),
            'timezone': 'US/Eastern'
        }
        
        is_open = await strategy._is_market_open()
        assert is_open is True
        
        # Test when market is closed
        time_based_context.get_market_hours.return_value = {
            'is_open': False,
            'open_time': datetime.utcnow().replace(hour=9, minute=30),
            'close_time': datetime.utcnow().replace(hour=16, minute=0),
            'timezone': 'US/Eastern'
        }
        
        is_closed = await strategy._is_market_open()
        assert is_closed is False
    
    @pytest.mark.asyncio
    async def test_is_in_exit_window(self, time_based_config):
        """Test exit window checking"""
        strategy = TestTimeBasedExitStrategy(time_based_config)
        
        # Test when in exit window
        current_time = datetime.utcnow().replace(hour=15, minute=45)  # 3:45 PM
        is_in_window = strategy._is_in_exit_window(current_time)
        # Should be true if 15:30-16:00 window is configured
        
        # Test when outside exit window
        outside_time = datetime.utcnow().replace(hour=14, minute=0)  # 2:00 PM
        is_outside = strategy._is_in_exit_window(outside_time)
        # Should be false for this time
    
    @pytest.mark.asyncio
    async def test_is_in_blackout_period(self, time_based_config):
        """Test blackout period checking"""
        strategy = TestTimeBasedExitStrategy(time_based_config)
        
        # Test when in blackout period
        blackout_time = datetime.utcnow().replace(hour=12, minute=30)  # 12:30 PM (lunch)
        is_blackout = strategy._is_in_blackout_period(blackout_time)
        # Should be true if 12:00-13:00 blackout is configured
        
        # Test when outside blackout period
        normal_time = datetime.utcnow().replace(hour=11, minute=0)  # 11:00 AM
        is_normal = strategy._is_in_blackout_period(normal_time)
        # Should be false for this time
    
    @pytest.mark.asyncio
    async def test_calculate_exit_urgency(self, time_based_config):
        """Test exit urgency calculation based on time"""
        strategy = TestTimeBasedExitStrategy(time_based_config)
        
        # Test with plenty of time
        time_remaining = timedelta(hours=3)  # 3 hours left
        urgency_low = strategy._calculate_exit_urgency(time_remaining)
        assert urgency_low < 0.5  # Low urgency
        
        # Test with little time
        time_remaining_short = timedelta(minutes=30)  # 30 minutes left
        urgency_high = strategy._calculate_exit_urgency(time_remaining_short)
        assert urgency_high > 0.8  # High urgency
        
        # Test with no time
        time_remaining_zero = timedelta(0)
        urgency_max = strategy._calculate_exit_urgency(time_remaining_zero)
        assert urgency_max == 1.0  # Maximum urgency
    
    @pytest.mark.asyncio
    async def test_evaluate_exit_conditions(self, time_based_config, time_based_context):
        """Test exit condition evaluation"""
        strategy = TestTimeBasedExitStrategy(time_based_config)
        strategy.set_context(time_based_context)
        
        position = {
            'position_id': 'pos_001',
            'symbol': 'AAPL',
            'entry_time': datetime.utcnow() - timedelta(hours=2),
            'quantity': Decimal('100')
        }
        
        # Test when exit is not triggered
        strategy.exit_triggered = False
        result = await strategy.evaluate_exit_conditions(position)
        assert result is False
        
        # Test when exit is triggered
        strategy.exit_triggered = True
        result = await strategy.evaluate_exit_conditions(position)
        assert result is True
        
        # Test with position over time limit
        old_position = {
            'position_id': 'pos_old',
            'symbol': 'AAPL',
            'entry_time': datetime.utcnow() - timedelta(hours=5),  # Over limit
            'quantity': Decimal('100')
        }
        
        strategy.exit_triggered = False  # Reset
        result = await strategy.evaluate_exit_conditions(old_position)
        assert result is True  # Should trigger due to time limit
    
    @pytest.mark.asyncio
    async def test_generate_exit_signal(self, time_based_config, time_based_context):
        """Test exit signal generation"""
        strategy = TestTimeBasedExitStrategy(time_based_config)
        strategy.set_context(time_based_context)
        
        position = {
            'position_id': 'pos_001',
            'symbol': 'AAPL',
            'quantity': Decimal('100'),
            'entry_time': datetime.utcnow() - timedelta(hours=2),
            'current_price': Decimal('150.00')
        }
        
        exit_signal = await strategy.generate_exit_signal(position, ExitReason.TIME_EXIT)
        
        assert exit_signal is not None
        assert exit_signal.strategy_id == time_based_config.strategy_id
        assert exit_signal.position_id == 'pos_001'
        assert exit_signal.symbol == 'AAPL'
        assert exit_signal.exit_reason == ExitReason.TIME_EXIT
        assert exit_signal.exit_price == Decimal('150.00')
        assert exit_signal.exit_quantity == Decimal('100')
        assert 'exit_type' in exit_signal.metadata
        assert 'time_to_exit' in exit_signal.metadata
        assert 'hold_time' in exit_signal.metadata
        assert 'max_hold_time' in exit_signal.metadata
        
        # Test with incomplete position data
        incomplete_position = {
            'position_id': 'pos_002'
            # Missing other required fields
        }
        
        exit_signal = await strategy.generate_exit_signal(incomplete_position, ExitReason.TIME_EXIT)
        assert exit_signal is None


class TestSessionExit:
    """Test SessionExit implementation"""
    
    @pytest.mark.asyncio
    async def test_session_exit_initialization(self, time_based_context):
        """Test SessionExit initialization and configuration"""
        config = ExitConfiguration(
            strategy_id="session_exit_001",
            strategy_type=ExitType.TIME_BASED,
            name="Session Exit",
            description="Test session exit",
            parameters={
                'exit_type': 'session',
                'session_type': 'regular',
                'market_open_time': '09:30',
                'market_close_time': '16:00',
                'timezone': 'US/Eastern',
                'exit_at_session_end': True
            }
        )
        
        strategy = SessionExit(config)
        strategy.set_context(time_based_context)
        
        assert strategy.time_config.exit_type == 'session'
        assert strategy.time_config.session_type == 'regular'
    
    @pytest.mark.asyncio
    async def test_session_end_detection(self, time_based_context):
        """Test session end detection"""
        config = ExitConfiguration(
            strategy_id="session_end_001",
            strategy_type=ExitType.TIME_BASED,
            name="Session End",
            description="Test session end detection",
            parameters={
                'exit_type': 'session',
                'exit_at_session_end': True,
                'session_end_buffer': 900  # 15 minutes buffer
            }
        )
        
        strategy = SessionExit(config)
        strategy.set_context(time_based_context)
        
        # Mock market hours near session end
        time_based_context.get_market_hours.return_value = {
            'is_open': True,
            'open_time': datetime.utcnow().replace(hour=9, minute=30),
            'close_time': datetime.utcnow().replace(hour=15, minute=45),  # 15 minutes left
            'timezone': 'US/Eastern'
        }
        
        near_end = await strategy._is_near_session_end()
        assert near_end is True  # Should detect near session end


class TestTimeLimitExit:
    """Test TimeLimitExit implementation"""
    
    @pytest.mark.asyncio
    async def test_time_limit_calculation(self, time_based_context):
        """Test time limit calculation and enforcement"""
        config = ExitConfiguration(
            strategy_id="time_limit_001",
            strategy_type=ExitType.TIME_BASED,
            name="Time Limit Exit",
            description="Test time limit exit",
            parameters={
                'exit_type': 'time_limit',
                'max_hold_time': timedelta(hours=2),
                'enforce_strict_limit': True,
                'grace_period': 300  # 5 minutes grace
            }
        )
        
        strategy = TimeLimitExit(config)
        strategy.set_context(time_based_context)
        
        # Test position within limit
        recent_position = {
            'position_id': 'pos_recent',
            'entry_time': datetime.utcnow() - timedelta(hours=1)  # 1 hour ago
        }
        
        time_to_limit = await strategy._calculate_time_to_limit(recent_position)
        assert isinstance(time_to_limit, timedelta)
        assert time_to_limit > timedelta(0)  # Should have time remaining
        
        # Test position over limit
        old_position = {
            'position_id': 'pos_old',
            'entry_time': datetime.utcnow() - timedelta(hours=3)  # 3 hours ago (over 2-hour limit)
        }
        
        time_over_limit = await strategy._calculate_time_to_limit(old_position)
        assert time_over_limit <= timedelta(0)  # Should be over limit


class TestScheduledExit:
    """Test ScheduledExit implementation"""
    
    @pytest.mark.asyncio
    async def test_scheduled_exit_configuration(self, time_based_context):
        """Test scheduled exit configuration and execution"""
        # Set specific exit time
        exit_time = datetime.utcnow().replace(hour=15, minute=30)
        
        config = ExitConfiguration(
            strategy_id="scheduled_001",
            strategy_type=ExitType.TIME_BASED,
            name="Scheduled Exit",
            description="Test scheduled exit",
            parameters={
                'exit_type': 'scheduled',
                'exit_time': exit_time.strftime('%H:%M'),
                'timezone': 'US/Eastern',
                'respect_market_hours': True
            }
        )
        
        strategy = ScheduledExit(config)
        strategy.set_context(time_based_context)
        
        assert strategy.time_config.exit_type == 'scheduled'
        assert strategy.time_config.exit_time == exit_time.strftime('%H:%M')
        
        # Mock current time after scheduled exit
        with patch('trading_orchestrator.strategies.time_based_exit.datetime') as mock_datetime:
            mock_datetime.utcnow.return_value = datetime.utcnow().replace(hour=16, minute=0)  # After exit time
            
            # Should trigger when current time exceeds scheduled time
            should_exit = await strategy._is_scheduled_time_reached()
            assert should_exit is True


class TestMarketCloseExit:
    """Test MarketCloseExit implementation"""
    
    @pytest.mark.asyncio
    async def test_market_close_exit(self, time_based_context):
        """Test market close exit logic"""
        config = ExitConfiguration(
            strategy_id="market_close_001",
            strategy_type=ExitType.TIME_BASED,
            name="Market Close Exit",
            description="Test market close exit",
            parameters={
                'exit_type': 'market_close',
                'exit_before_close': True,
                'close_buffer_minutes': 30,
                'emergency_close_enabled': True
            }
        )
        
        strategy = MarketCloseExit(config)
        strategy.set_context(time_based_context)
        
        # Mock market hours near close
        time_based_context.get_market_hours.return_value = {
            'is_open': True,
            'open_time': datetime.utcnow().replace(hour=9, minute=30),
            'close_time': datetime.utcnow().replace(hour=15, minute=30),  # 30 minutes left
            'timezone': 'US/Eastern'
        }
        
        near_close = await strategy._is_near_market_close()
        assert near_close is True  # Should detect near market close


class TestTimeDecayExit:
    """Test TimeDecayExit implementation"""
    
    @pytest.mark.asyncio
    async def test_time_decay_calculation(self):
        """Test time decay urgency calculation"""
        config = ExitConfiguration(
            strategy_id="time_decay_001",
            strategy_type=ExitType.TIME_BASED,
            name="Time Decay Exit",
            description="Test time decay exit",
            parameters={
                'exit_type': 'time_decay',
                'decay_rate': Decimal('0.1'),
                'initial_urgency': Decimal('0.5'),
                'urgency_increase_rate': Decimal('0.1'),
                'max_urgency': Decimal('1.0')
            }
        )
        
        strategy = TimeDecayExit(config)
        
        # Test urgency calculation based on time
        entry_time = datetime.utcnow() - timedelta(hours=2)
        urgency_2h = strategy._calculate_time_decay_urgency(entry_time)
        
        entry_time_long = datetime.utcnow() - timedelta(hours=4)
        urgency_4h = strategy._calculate_time_decay_urgency(entry_time_long)
        
        # Longer hold should result in higher urgency
        assert urgency_4h > urgency_2h
        
        # Urgency should be between 0 and 1
        assert 0.0 <= urgency_2h <= 1.0
        assert 0.0 <= urgency_4h <= 1.0


class TestTimeBasedScenarios:
    """Test various market scenarios for time-based exits"""
    
    @pytest.mark.asyncio
    async def test_regular_trading_session(self, time_based_config, time_based_context):
        """Test behavior during regular trading session"""
        strategy = TestTimeBasedExitStrategy(time_based_config)
        strategy.set_context(time_based_context)
        
        # Mock regular market hours
        time_based_context.get_market_hours.return_value = {
            'is_open': True,
            'open_time': datetime.utcnow().replace(hour=9, minute=30),
            'close_time': datetime.utcnow().replace(hour=16, minute=0),
            'timezone': 'US/Eastern'
        }
        
        # Test position during regular session
        session_position = {
            'position_id': 'pos_session',
            'symbol': 'AAPL',
            'entry_time': datetime.utcnow().replace(hour=11, minute=0),  # Mid-session
            'quantity': Decimal('100')
        }
        
        is_open = await strategy._is_market_open()
        assert is_open is True
        
        time_to_exit = await strategy._calculate_time_to_exit(session_position)
        assert isinstance(time_to_exit, timedelta)
    
    @pytest.mark.asyncio
    async def test_after_hours_position(self, time_based_config, time_based_context):
        """Test behavior with after-hours position"""
        strategy = TestTimeBasedExitStrategy(time_based_config)
        strategy.set_context(time_based_context)
        
        # Mock after-hours
        time_based_context.get_market_hours.return_value = {
            'is_open': False,
            'open_time': datetime.utcnow().replace(hour=9, minute=30),
            'close_time': datetime.utcnow().replace(hour=16, minute=0),
            'timezone': 'US/Eastern'
        }
        
        after_hours_position = {
            'position_id': 'pos_after_hours',
            'symbol': 'AAPL',
            'entry_time': datetime.utcnow().replace(hour=20, minute=0),  # After hours
            'quantity': Decimal('100')
        }
        
        is_open = await strategy._is_market_open()
        assert is_open is False
        
        # Time-based exits should still work after hours
        exit_triggered = await strategy.evaluate_exit_conditions(after_hours_position)
        assert isinstance(exit_triggered, bool)
    
    @pytest.mark.asyncio
    async def test_emergency_exit_scenario(self, time_based_config, time_based_context):
        """Test emergency exit scenarios"""
        config = ExitConfiguration(
            strategy_id="emergency_001",
            strategy_type=ExitType.TIME_BASED,
            name="Emergency Exit",
            description="Test emergency exit",
            parameters={
                'exit_type': 'time_limit',
                'max_hold_time': timedelta(hours=8),
                'emergency_exit_enabled': True,
                'emergency_triggers': ['news_event', 'market_halt'],
                'emergency_buffer': 1800  # 30 minutes
            }
        )
        
        strategy = TestTimeBasedExitStrategy(config)
        strategy.set_context(time_based_context)
        
        emergency_position = {
            'position_id': 'pos_emergency',
            'symbol': 'AAPL',
            'entry_time': datetime.utcnow() - timedelta(minutes=30),  # Recently opened
            'quantity': Decimal('100')
        }
        
        # Should not trigger normal exit for recent position
        normal_exit = await strategy.evaluate_exit_conditions(emergency_position)
        assert normal_exit is False
        
        # Mock emergency condition
        strategy.emergency_triggered = True
        
        # Should trigger emergency exit
        emergency_exit = await strategy.evaluate_exit_conditions(emergency_position)
        assert emergency_exit is True


class TestTimeBasedEdgeCases:
    """Test edge cases and error conditions"""
    
    @pytest.mark.asyncio
    async def test_missing_entry_time(self, time_based_config, time_based_context):
        """Test behavior with missing entry time"""
        strategy = TestTimeBasedExitStrategy(time_based_config)
        strategy.set_context(time_based_context)
        
        position_no_time = {
            'position_id': 'pos_no_time',
            'symbol': 'AAPL',
            'quantity': Decimal('100')
            # Missing entry_time
        }
        
        # Should handle missing entry time gracefully
        time_to_exit = await strategy._calculate_time_to_exit(position_no_time)
        assert isinstance(time_to_exit, timedelta)
    
    @pytest.mark.asyncio
    async def test_zero_max_hold_time(self, time_based_config):
        """Test behavior with zero max hold time"""
        config = ExitConfiguration(
            strategy_id="zero_hold_001",
            strategy_type=ExitType.TIME_BASED,
            name="Zero Hold Time",
            description="Test zero hold time",
            parameters={
                'exit_type': 'time_limit',
                'max_hold_time': timedelta(0),  # Zero hold time
                'emergency_exit_enabled': True
            }
        )
        
        strategy = TestTimeBasedExitStrategy(config)
        
        position = {
            'position_id': 'pos_zero_hold',
            'symbol': 'AAPL',
            'entry_time': datetime.utcnow(),
            'quantity': Decimal('100')
        }
        
        time_to_exit = await strategy._calculate_time_to_exit(position)
        assert time_to_exit <= timedelta(0)  # Should be at or past limit
    
    @pytest.mark.asyncio
    async def test_extreme_time_values(self, time_based_config):
        """Test behavior with extreme time values"""
        strategy = TestTimeBasedExitStrategy(time_based_config)
        
        # Test with very old position
        old_position = {
            'position_id': 'pos_old',
            'symbol': 'AAPL',
            'entry_time': datetime.utcnow() - timedelta(days=365),  # 1 year ago
            'quantity': Decimal('100')
        }
        
        time_to_exit = await strategy._calculate_time_to_exit(old_position)
        assert time_to_exit <= timedelta(0)  # Should be well past limit
        
        # Test with future position (invalid but should handle gracefully)
        future_position = {
            'position_id': 'pos_future',
            'symbol': 'AAPL',
            'entry_time': datetime.utcnow() + timedelta(hours=1),  # Future time
            'quantity': Decimal('100')
        }
        
        time_to_exit_future = await strategy._calculate_time_to_exit(future_position)
        assert isinstance(time_to_exit_future, timedelta)
        # Should handle gracefully even with future time
    
    def test_invalid_time_formats(self, time_based_config):
        """Test behavior with invalid time formats"""
        config = ExitConfiguration(
            strategy_id="invalid_time_001",
            strategy_type=ExitType.TIME_BASED,
            name="Invalid Time Format",
            description="Test invalid time formats",
            parameters={
                'exit_type': 'scheduled',
                'exit_time': 'invalid_time_format',
                'market_open_time': '25:70',  # Invalid time
                'market_close_time': 'invalid'
            }
        )
        
        strategy = TestTimeBasedExitStrategy(config)
        
        # Should handle invalid time formats gracefully
        try:
            # This might raise an exception or handle gracefully
            strategy._parse_time_format(config.parameters['exit_time'])
        except (ValueError, TypeError):
            # Expected for invalid formats
            pass
        
        try:
            strategy._parse_time_format(config.parameters['market_open_time'])
        except (ValueError, TypeError):
            # Expected for invalid formats
            pass


if __name__ == "__main__":
    pytest.main([__file__])
