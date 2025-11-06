"""
Test Base Exit Strategy Framework

Tests the foundational classes and interfaces for exit strategies including:
- BaseExitStrategy abstract class
- ExitReason, ExitType, ExitStatus enums
- ExitCondition, ExitSignal, ExitMetrics data classes
- ExitConfiguration, ExitRule data classes
- ExitStrategyRegistry class
- Core functionality and edge cases
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

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
    ExitContext,
    ExitStrategyRegistry,
    exit_strategy_registry
)

# Test subclasses for abstract class testing
class TestExitStrategy(BaseExitStrategy):
    """Test implementation of BaseExitStrategy for testing"""
    
    def __init__(self, config):
        super().__init__(config)
        self.exit_triggered = False
    
    async def evaluate_exit_conditions(self, position: Dict[str, Any]) -> bool:
        """Simple test implementation"""
        return self.exit_triggered
    
    async def generate_exit_signal(self, position: Dict[str, Any], exit_reason: ExitReason) -> Optional[ExitSignal]:
        """Generate test exit signal"""
        if self.exit_triggered:
            return ExitSignal(
                signal_id=str(uuid.uuid4()),
                strategy_id=self.config.strategy_id,
                position_id=position.get('position_id', ''),
                symbol=position.get('symbol', ''),
                exit_reason=exit_reason,
                exit_price=Decimal('145.00'),
                exit_quantity=Decimal('100'),
                confidence=0.9,
                urgency=0.8
            )
        return None


class TestExitContext(ExitContext):
    """Test implementation of ExitContext"""
    
    def __init__(self):
        self.current_prices = {'AAPL': Decimal('150.00'), 'GOOGL': Decimal('2800.00')}
        self.positions = {}
        self.historical_data = {}
        self.volatility = {'AAPL': Decimal('0.025'), 'GOOGL': Decimal('0.030')}
        self.exit_orders = []
        self.portfolio_value = Decimal('1000000')
        self.risk_metrics = {'max_drawdown': 0.05, 'var_99': Decimal('5000')}
    
    async def get_current_price(self, symbol: str) -> Decimal:
        return self.current_prices.get(symbol, Decimal('100.00'))
    
    async def get_position(self, position_id: str) -> Optional[Dict[str, Any]]:
        return self.positions.get(position_id)
    
    async def get_historical_data(self, symbol: str, timeframe: str, limit: int) -> List[Dict[str, Any]]:
        key = f"{symbol}_{timeframe}_{limit}"
        return self.historical_data.get(key, [])
    
    async def calculate_volatility(self, symbol: str, period: int) -> Decimal:
        return self.volatility.get(symbol, Decimal('0.02'))
    
    async def submit_exit_order(self, position_id: str, symbol: str, quantity: Decimal, exit_price: Decimal, order_type: str = "market") -> Dict[str, Any]:
        order = {
            'position_id': position_id,
            'symbol': symbol,
            'quantity': quantity,
            'exit_price': exit_price,
            'order_type': order_type,
            'timestamp': datetime.utcnow()
        }
        self.exit_orders.append(order)
        return {'success': True, 'order_id': str(uuid.uuid4())}
    
    async def get_portfolio_value(self) -> Decimal:
        return self.portfolio_value
    
    async def get_risk_metrics(self) -> Dict[str, Any]:
        return self.risk_metrics


class TestExitStrategyComplex(TestExitStrategy):
    """More complex test implementation with realistic behavior"""
    
    async def evaluate_exit_conditions(self, position: Dict[str, Any]) -> bool:
        """Realistic exit condition evaluation"""
        current_price = position.get('current_price', Decimal('0'))
        entry_price = position.get('entry_price', Decimal('0'))
        
        if not current_price or not entry_price:
            return False
        
        # Simple profit-based exit
        profit_pct = (current_price - entry_price) / entry_price
        return profit_pct > 0.05  # Exit at 5% profit
    
    async def generate_exit_signal(self, position: Dict[str, Any], exit_reason: ExitReason) -> ExitSignal:
        """Generate realistic exit signal"""
        return ExitSignal(
            signal_id=str(uuid.uuid4()),
            strategy_id=self.config.strategy_id,
            position_id=position.get('position_id', ''),
            symbol=position.get('symbol', ''),
            exit_reason=exit_reason,
            exit_price=position.get('current_price', Decimal('0')),
            exit_quantity=position.get('quantity', Decimal('0')),
            confidence=0.85 + abs(position.get('current_price', Decimal('0')) - position.get('entry_price', Decimal('0'))) / position.get('entry_price', Decimal('1')) * 0.1,
            urgency=0.7,
            estimated_execution_time=timedelta(seconds=5)
        )


class TestBaseExitStrategy:
    """Test BaseExitStrategy class"""
    
    def test_initialization(self, sample_exit_configuration):
        """Test proper initialization of BaseExitStrategy"""
        strategy = TestExitStrategy(sample_exit_configuration)
        
        assert strategy.config == sample_exit_configuration
        assert strategy.status == ExitStatus.INITIALIZING
        assert strategy.context is None
        assert isinstance(strategy.metrics, ExitMetrics)
        assert strategy.metrics.strategy_id == sample_exit_configuration.strategy_id
        assert len(strategy.exit_signals) == 0
        assert len(strategy.exit_history) == 0
        assert len(strategy.monitored_positions) == 0
        assert len(strategy.conditions) == len(sample_exit_configuration.conditions)
        assert len(strategy.rules) == 0
        assert isinstance(strategy.start_time, datetime)
        assert strategy.last_evaluation_time is None
        assert strategy.total_evaluations == 0
        assert strategy.total_signals_generated == 0
        assert strategy.total_exits_executed == 0
        assert len(strategy.active_exits) == 0
        assert len(strategy.pending_exits) == 0
    
    def test_set_context(self, sample_exit_configuration):
        """Test setting exit strategy context"""
        strategy = TestExitStrategy(sample_exit_configuration)
        context = TestExitContext()
        
        strategy.set_context(context)
        
        assert strategy.context == context
    
    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self, sample_exit_configuration, mock_exit_context):
        """Test strategy start and stop lifecycle"""
        strategy = TestExitStrategy(sample_exit_configuration)
        strategy.set_context(mock_exit_context)
        
        # Test start
        start_task = asyncio.create_task(strategy.start())
        await asyncio.sleep(0.1)  # Let it start
        
        assert strategy.status == ExitStatus.ACTIVE
        
        # Test stop
        await strategy.stop()
        await asyncio.sleep(0.1)  # Let it stop
        
        assert strategy.status == ExitStatus.STOPPED
    
    @pytest.mark.asyncio
    async def test_pause_resume(self, sample_exit_configuration):
        """Test pause and resume functionality"""
        strategy = TestExitStrategy(sample_exit_configuration)
        
        # Test pause
        await strategy.pause()
        assert strategy.status == ExitStatus.PAUSED
        
        # Test resume
        await strategy.resume()
        assert strategy.status == ExitStatus.ACTIVE
    
    @pytest.mark.asyncio
    async def test_get_status(self, sample_exit_configuration):
        """Test status reporting"""
        strategy = TestExitStrategy(sample_exit_configuration)
        
        status = await strategy.get_status()
        
        assert 'strategy_id' in status
        assert 'name' in status
        assert 'type' in status
        assert 'status' in status
        assert 'runtime_seconds' in status
        assert 'total_evaluations' in status
        assert 'total_signals_generated' in status
        assert 'total_exits_executed' in status
        assert 'monitored_positions' in status
        assert 'active_exits' in status
        assert 'pending_exits' in status
        assert 'metrics' in status
        assert 'last_updated' in status
        
        assert status['strategy_id'] == sample_exit_configuration.strategy_id
        assert status['name'] == sample_exit_configuration.name
        assert status['type'] == sample_exit_configuration.strategy_type.value
        assert status['status'] == ExitStatus.INITIALIZING.value
    
    @pytest.mark.asyncio
    async def test_position_monitoring(self, sample_exit_configuration, mock_exit_context):
        """Test position monitoring functionality"""
        strategy = TestExitStrategy(sample_exit_configuration)
        strategy.set_context(mock_exit_context)
        
        test_position = {
            'position_id': 'pos_001',
            'symbol': 'AAPL',
            'quantity': Decimal('100'),
            'entry_price': Decimal('140.00'),
            'entry_time': datetime.utcnow(),
            'created_at': datetime.utcnow()
        }
        
        mock_exit_context.get_positions.return_value = [test_position]
        
        await strategy.monitor_positions()
        
        assert 'pos_001' in strategy.monitored_positions
        assert strategy.monitored_positions['pos_001'] == test_position
    
    @pytest.mark.asyncio
    async def test_evaluate_position(self, sample_exit_configuration, mock_exit_context):
        """Test position evaluation for exit conditions"""
        strategy = TestExitStrategy(sample_exit_configuration)
        strategy.set_context(mock_exit_context)
        
        test_position = {
            'position_id': 'pos_001',
            'symbol': 'AAPL',
            'quantity': Decimal('100'),
            'entry_price': Decimal('140.00'),
            'entry_time': datetime.utcnow(),
            'created_at': datetime.utcnow()
        }
        
        # Test with exit triggered
        strategy.exit_triggered = True
        await strategy.evaluate_position(test_position)
        
        assert strategy.total_evaluations == 1
        assert strategy.last_evaluation_time is not None
    
    @pytest.mark.asyncio
    async def test_should_evaluate_position(self, sample_exit_configuration):
        """Test position evaluation eligibility checks"""
        strategy = TestExitStrategy(sample_exit_configuration)
        
        # Normal position
        normal_position = {
            'position_id': 'pos_001',
            'created_at': datetime.utcnow() - timedelta(hours=1)
        }
        
        # New position (too recent)
        new_position = {
            'position_id': 'pos_002',
            'created_at': datetime.utcnow() - timedelta(seconds=30)
        }
        
        # Position in active exits
        strategy.active_exits['pos_003'] = {}
        
        # Test evaluation eligibility
        assert await strategy.should_evaluate_position(normal_position) is True
        
        # Test new position with activation delay
        strategy.config.parameters['activation_delay'] = 60  # 60 seconds
        assert await strategy.should_evaluate_position(new_position) is False
        
        # Test position in active exits
        assert await strategy.should_evaluate_position({'position_id': 'pos_003'}) is False
    
    @pytest.mark.asyncio
    async def test_determine_exit_reason(self, sample_exit_configuration):
        """Test exit reason determination"""
        strategy = TestExitStrategy(sample_exit_configuration)
        
        position = {'position_id': 'pos_001'}
        
        # Test with profit target condition
        profit_condition = ExitCondition(
            condition_id="profit_001",
            name="Profit Target",
            condition_type="price_profit",
            threshold_value=Decimal('0.05'),
            comparison_operator=">=",
            is_active=True,
            priority=1
        )
        strategy.conditions = [profit_condition]
        
        exit_reason = await strategy.determine_exit_reason(position)
        assert exit_reason == ExitReason.PROFIT_TARGET
    
    def test_compare_values(self, sample_exit_configuration):
        """Test value comparison logic"""
        strategy = TestExitStrategy(sample_exit_configuration)
        
        # Test greater than
        assert strategy._compare_values(Decimal('10'), Decimal('5'), ">") is True
        assert strategy._compare_values(Decimal('5'), Decimal('10'), ">") is False
        
        # Test less than
        assert strategy._compare_values(Decimal('5'), Decimal('10'), "<") is True
        assert strategy._compare_values(Decimal('10'), Decimal('5'), "<") is False
        
        # Test equality
        assert strategy._compare_values(Decimal('10'), Decimal('10'), "==") is True
        assert strategy._compare_values(Decimal('10'), Decimal('5'), "==") is False
        
        # Test with floats
        assert strategy._compare_values(10.5, 5.2, ">") is True
        assert strategy._compare_values(5.2, 10.5, "<") is True
        
        # Test with mixed types
        assert strategy._compare_values(10, 5, ">") is True
        assert strategy._compare_values(Decimal('10'), 5, ">") is True
    
    def test_calculate_rsi(self, sample_exit_configuration):
        """Test RSI calculation"""
        strategy = TestExitStrategy(sample_exit_configuration)
        
        # Test with valid data
        prices = [Decimal('100'), Decimal('101'), Decimal('102'), Decimal('101'), Decimal('100')]
        rsi = strategy._calculate_rsi(prices, 3)
        
        assert isinstance(rsi, float)
        assert 0.0 <= rsi <= 100.0
        
        # Test with insufficient data
        short_prices = [Decimal('100')]
        rsi_short = strategy._calculate_rsi(short_prices, 3)
        assert rsi_short == 50.0  # Neutral RSI
    
    def test_calculate_sma(self, sample_exit_configuration):
        """Test Simple Moving Average calculation"""
        strategy = TestExitStrategy(sample_exit_configuration)
        
        prices = [Decimal('100'), Decimal('102'), Decimal('104'), Decimal('103'), Decimal('105')]
        sma = strategy._calculate_sma(prices, 3)
        
        assert isinstance(sma, float)
        assert abs(sma - 104.0) < 0.001
        
        # Test with insufficient data
        short_prices = [Decimal('100')]
        sma_short = strategy._calculate_sma(short_prices, 3)
        assert sma_short == 0.0
    
    def test_calculate_ema(self, sample_exit_configuration):
        """Test Exponential Moving Average calculation"""
        strategy = TestExitStrategy(sample_exit_configuration)
        
        prices = [Decimal('100'), Decimal('102'), Decimal('104'), Decimal('103'), Decimal('105')]
        ema = strategy._calculate_ema(prices, 3)
        
        assert isinstance(ema, float)
        assert ema > 0.0
        
        # Test with empty data
        empty_prices = []
        ema_empty = strategy._calculate_ema(empty_prices, 3)
        assert ema_empty == 0.0


class TestExitEnums:
    """Test exit strategy enums"""
    
    def test_exit_reason_enum(self):
        """Test ExitReason enum values"""
        assert ExitReason.PROFIT_TARGET.value == "profit_target"
        assert ExitReason.STOP_LOSS.value == "stop_loss"
        assert ExitReason.TRAILING_STOP.value == "trailing_stop"
        assert ExitReason.VOLATILITY_STOP.value == "volatility_stop"
        assert ExitReason.TIME_EXIT.value == "time_exit"
        assert ExitReason.MARKET_CONDITION.value == "market_condition"
        assert ExitReason.AI_SIGNAL.value == "ai_signal"
        assert ExitReason.MANUAL.value == "manual"
        assert ExitReason.SYSTEM_ERROR.value == "system_error"
        assert ExitReason.RISK_LIMIT.value == "risk_limit"
        
        # Test enum iteration
        all_reasons = list(ExitReason)
        assert len(all_reasons) >= 10
        assert ExitReason.PROFIT_TARGET in all_reasons
    
    def test_exit_type_enum(self):
        """Test ExitType enum values"""
        assert ExitType.TRAILING_STOP.value == "trailing_stop"
        assert ExitType.FIXED_TARGET.value == "fixed_target"
        assert ExitType.STOP_LOSS.value == "stop_loss"
        assert ExitType.VOLATILITY_STOP.value == "volatility_stop"
        assert ExitType.TIME_BASED.value == "time_based"
        assert ExitType.CONDITIONAL.value == "conditional"
        assert ExitType.AI_DRIVEN.value == "ai_driven"
        
        # Test enum iteration
        all_types = list(ExitType)
        assert len(all_types) >= 7
        assert ExitType.TRAILING_STOP in all_types
    
    def test_exit_status_enum(self):
        """Test ExitStatus enum values"""
        assert ExitStatus.INITIALIZING.value == "initializing"
        assert ExitStatus.ACTIVE.value == "active"
        assert ExitStatus.TRIGGERED.value == "triggered"
        assert ExitStatus.COMPLETED.value == "completed"
        assert ExitStatus.PAUSED.value == "paused"
        assert ExitStatus.STOPPED.value == "stopped"
        assert ExitStatus.ERROR.value == "error"
        
        # Test enum iteration
        all_statuses = list(ExitStatus)
        assert len(all_status) >= 7
        assert ExitStatus.INITIALIZING in all_status


class TestExitCondition:
    """Test ExitCondition data class"""
    
    def test_exit_condition_creation(self):
        """Test ExitCondition creation and initialization"""
        condition = ExitCondition(
            condition_id="test_001",
            name="Test Condition",
            condition_type="price",
            threshold_value=Decimal('150.00'),
            comparison_operator=">=",
            priority=1
        )
        
        assert condition.condition_id == "test_001"
        assert condition.name == "Test Condition"
        assert condition.condition_type == "price"
        assert condition.threshold_value == Decimal('150.00')
        assert condition.comparison_operator == ">="
        assert condition.is_active is True
        assert condition.priority == 1
        assert isinstance(condition.created_at, datetime)
        assert condition.last_evaluated is None
        assert condition.evaluation_count == 0
        assert condition.parameters == {}
    
    def test_exit_condition_with_parameters(self):
        """Test ExitCondition with additional parameters"""
        condition = ExitCondition(
            condition_id="test_002",
            name="Complex Condition",
            condition_type="volatility",
            threshold_value=Decimal('0.025'),
            comparison_operator=">=",
            parameters={
                'period': 14,
                'method': 'ATR',
                'multiplier': 2.0
            },
            priority=2,
            is_active=False
        )
        
        assert condition.parameters == {
            'period': 14,
            'method': 'ATR', 
            'multiplier': 2.0
        }
        assert condition.priority == 2
        assert condition.is_active is False


class TestExitSignal:
    """Test ExitSignal data class"""
    
    def test_exit_signal_creation(self):
        """Test ExitSignal creation and initialization"""
        signal = ExitSignal(
            signal_id="signal_001",
            strategy_id="strategy_001",
            position_id="position_001",
            symbol="AAPL",
            exit_reason=ExitReason.TRAILING_STOP,
            exit_price=Decimal('145.00'),
            exit_quantity=Decimal('100'),
            confidence=0.90,
            urgency=0.80
        )
        
        assert signal.signal_id == "signal_001"
        assert signal.strategy_id == "strategy_001"
        assert signal.position_id == "position_001"
        assert signal.symbol == "AAPL"
        assert signal.exit_reason == ExitReason.TRAILING_STOP
        assert signal.exit_price == Decimal('145.00')
        assert signal.exit_quantity == Decimal('100')
        assert signal.confidence == 0.90
        assert signal.urgency == 0.80
        assert signal.estimated_execution_time is None
        assert signal.market_impact is None
        assert signal.metadata == {}
        assert isinstance(signal.created_at, datetime)
        assert signal.expires_at is None
    
    def test_exit_signal_with_metadata(self):
        """Test ExitSignal with metadata"""
        signal = ExitSignal(
            signal_id="signal_002",
            strategy_id="strategy_002",
            position_id="position_002",
            symbol="GOOGL",
            exit_reason=ExitReason.STOP_LOSS,
            exit_price=Decimal('2750.00'),
            exit_quantity=Decimal('50'),
            confidence=0.95,
            urgency=0.90,
            estimated_execution_time=timedelta(seconds=10),
            market_impact=Decimal('0.001'),
            metadata={
                'trigger_price': Decimal('2745.00'),
                'volatility': Decimal('0.030'),
                'volume_spike': True
            },
            expires_at=datetime.utcnow() + timedelta(minutes=5)
        )
        
        assert signal.estimated_execution_time == timedelta(seconds=10)
        assert signal.market_impact == Decimal('0.001')
        assert signal.metadata == {
            'trigger_price': Decimal('2745.00'),
            'volatility': Decimal('0.030'),
            'volume_spike': True
        }
        assert isinstance(signal.expires_at, datetime)
    
    def test_exit_signal_validation(self):
        """Test ExitSignal field validation"""
        # Valid signal
        valid_signal = ExitSignal(
            signal_id="signal_valid",
            strategy_id="strategy_valid",
            position_id="position_valid", 
            symbol="AAPL",
            exit_reason=ExitReason.MANUAL,
            exit_price=Decimal('150.00'),
            exit_quantity=Decimal('100'),
            confidence=0.5,
            urgency=0.5
        )
        
        # Test confidence bounds
        assert valid_signal.confidence == 0.5
        
        # Test invalid confidence (should be allowed in construction but logic should handle)
        try:
            invalid_signal = ExitSignal(
                signal_id="signal_invalid",
                strategy_id="strategy_invalid",
                position_id="position_invalid",
                symbol="TEST",
                exit_reason=ExitReason.MANUAL,
                exit_price=Decimal('100.00'),
                exit_quantity=Decimal('50'),
                confidence=1.5,  # Invalid - should be between 0 and 1
                urgency=0.5
            )
            # Construction should work, but validation should happen elsewhere
            assert invalid_signal.confidence == 1.5
        except ValueError:
            # If there's validation, that's fine too
            pass


class TestExitMetrics:
    """Test ExitMetrics data class"""
    
    def test_exit_metrics_creation(self):
        """Test ExitMetrics creation and initialization"""
        metrics = ExitMetrics(strategy_id="test_strategy_001")
        
        assert metrics.strategy_id == "test_strategy_001"
        assert metrics.total_exits == 0
        assert metrics.successful_exits == 0
        assert metrics.failed_exits == 0
        assert metrics.profit_exits == 0
        assert metrics.loss_exits == 0
        assert metrics.success_rate == 0.0
        assert metrics.win_rate == 0.0
        assert metrics.total_profit == 0.0
        assert metrics.total_loss == 0.0
        assert metrics.net_profit == 0.0
        assert metrics.avg_exit_time == 0.0
        assert metrics.median_exit_time == 0.0
        assert metrics.avg_profit_per_exit == 0.0
        assert metrics.avg_loss_per_exit == 0.0
        assert metrics.profit_factor == 0.0
        assert metrics.largest_profit == 0.0
        assert metrics.largest_loss == 0.0
        assert metrics.avg_confidence == 0.0
        assert metrics.avg_urgency == 0.0
        assert metrics.consecutive_wins == 0
        assert metrics.consecutive_losses == 0
        assert metrics.longest_win_streak == 0
        assert metrics.longest_loss_streak == 0
        assert metrics.sharpe_ratio == 0.0
        assert metrics.max_drawdown == 0.0
        assert metrics.calmar_ratio == 0.0
        assert metrics.sortino_ratio == 0.0
        assert isinstance(metrics.last_updated, datetime)
    
    def test_exit_metrics_with_data(self):
        """Test ExitMetrics with performance data"""
        metrics = ExitMetrics(
            strategy_id="test_strategy_002",
            total_exits=100,
            successful_exits=85,
            profit_exits=65,
            total_profit=25000.0,
            total_loss=-8000.0,
            avg_confidence=0.85,
            avg_urgency=0.70
        )
        
        assert metrics.total_exits == 100
        assert metrics.successful_exits == 85
        assert metrics.failed_exits == 15  # Calculated
        assert metrics.profit_exits == 65
        assert metrics.loss_exits == 20  # Not explicitly set
        assert metrics.success_rate == 0.85
        assert metrics.win_rate == 0.65  # 65 profitable out of 100 total
        assert metrics.total_profit == 25000.0
        assert metrics.total_loss == -8000.0
        assert metrics.net_profit == 17000.0
        assert metrics.avg_confidence == 0.85
        assert metrics.avg_urgency == 0.70
        assert metrics.profit_factor == 3.125  # 25000 / 8000


class TestExitConfiguration:
    """Test ExitConfiguration data class"""
    
    def test_exit_configuration_creation(self):
        """Test ExitConfiguration creation and initialization"""
        config = ExitConfiguration(
            strategy_id="test_config_001",
            strategy_type=ExitType.TRAILING_STOP,
            name="Test Trailing Stop",
            description="Test configuration for unit testing",
            parameters={
                'trailing_distance': Decimal('0.03'),
                'initial_stop': Decimal('0.95'),
                'update_frequency': 60
            }
        )
        
        assert config.strategy_id == "test_config_001"
        assert config.strategy_type == ExitType.TRAILING_STOP
        assert config.name == "Test Trailing Stop"
        assert config.description == "Test configuration for unit testing"
        assert config.parameters == {
            'trailing_distance': Decimal('0.03'),
            'initial_stop': Decimal('0.95'),
            'update_frequency': 60
        }
        assert config.conditions == []
        assert config.is_active is True
        assert config.max_exit_size == Decimal('100000')
        assert config.max_exit_time is None
        assert config.min_confidence == 0.50
        assert config.max_loss_threshold == Decimal('10000')
        assert config.monitoring_interval == 60
        assert config.execution_timeout == timedelta(seconds=60)
        assert config.slippage_tolerance == Decimal('0.001')
        assert config.retry_attempts == 3
        assert isinstance(config.created_at, datetime)
    
    def test_exit_configuration_with_conditions(self):
        """Test ExitConfiguration with exit conditions"""
        condition = ExitCondition(
            condition_id="cond_001",
            name="Test Condition",
            condition_type="price",
            threshold_value=Decimal('145.00'),
            comparison_operator="<="
        )
        
        config = ExitConfiguration(
            strategy_id="test_config_002",
            strategy_type=ExitType.STOP_LOSS,
            name="Test Stop Loss",
            description="Test with conditions",
            parameters={'stop_distance': Decimal('0.05')},
            conditions=[condition],
            min_confidence=0.80,
            monitoring_interval=30,
            max_exit_size=Decimal('50000')
        )
        
        assert len(config.conditions) == 1
        assert config.conditions[0] == condition
        assert config.min_confidence == 0.80
        assert config.monitoring_interval == 30
        assert config.max_exit_size == Decimal('50000')


class TestExitRule:
    """Test ExitRule data class"""
    
    def test_exit_rule_creation(self):
        """Test ExitRule creation and initialization"""
        condition = ExitCondition(
            condition_id="cond_001",
            name="Price Condition",
            condition_type="price",
            threshold_value=Decimal('145.00'),
            comparison_operator="<="
        )
        
        rule = ExitRule(
            rule_id="rule_001",
            name="Price Drop Rule",
            conditions=[condition],
            logical_operator="AND",
            threshold=Decimal('0.03'),
            priority=1
        )
        
        assert rule.rule_id == "rule_001"
        assert rule.name == "Price Drop Rule"
        assert rule.conditions == [condition]
        assert rule.logical_operator == "AND"
        assert rule.threshold == Decimal('0.03')
        assert rule.priority == 1
        assert rule.is_active is True
        assert rule.evaluation_count == 0
        assert rule.success_count == 0
        assert isinstance(rule.created_at, datetime)
    
    def test_exit_rule_with_multiple_conditions(self):
        """Test ExitRule with multiple conditions"""
        condition1 = ExitCondition(
            condition_id="cond_001",
            name="Price Condition",
            condition_type="price",
            threshold_value=Decimal('145.00'),
            comparison_operator="<="
        )
        
        condition2 = ExitCondition(
            condition_id="cond_002",
            name="Volume Condition", 
            condition_type="volume",
            threshold_value=1000000,
            comparison_operator=">="
        )
        
        rule = ExitRule(
            rule_id="rule_002",
            name="Price and Volume Rule",
            conditions=[condition1, condition2],
            logical_operator="OR",
            priority=2
        )
        
        assert len(rule.conditions) == 2
        assert rule.logical_operator == "OR"
        assert rule.priority == 2
        assert rule.threshold is None


class TestExitStrategyRegistry:
    """Test ExitStrategyRegistry class"""
    
    def test_registry_initialization(self):
        """Test registry initialization"""
        registry = ExitStrategyRegistry()
        
        assert len(registry._strategies) == 0
        assert len(registry._strategy_types) == 0
        assert registry._is_registered is False
    
    def test_register_strategy_type(self):
        """Test registering strategy types"""
        registry = ExitStrategyRegistry()
        
        registry.register_strategy_type(ExitType.TRAILING_STOP, TestExitStrategy)
        
        assert "trailing_stop" in registry._strategy_types
        assert registry._strategy_types["trailing_stop"] == TestExitStrategy
    
    def test_create_strategy(self, sample_exit_configuration):
        """Test creating strategy from registry"""
        registry = ExitStrategyRegistry()
        registry.register_strategy_type(ExitType.TRAILING_STOP, TestExitStrategy)
        
        strategy = registry.create_strategy(ExitType.TRAILING_STOP, sample_exit_configuration)
        
        assert isinstance(strategy, TestExitStrategy)
        assert strategy.config == sample_exit_configuration
    
    def test_create_strategy_unknown_type(self, sample_exit_configuration):
        """Test creating strategy with unknown type"""
        registry = ExitStrategyRegistry()
        
        with pytest.raises(ValueError, match="Unknown exit strategy type"):
            registry.create_strategy(ExitType.TRAILING_STOP, sample_exit_configuration)
    
    def test_register_unregister_instance(self, sample_exit_configuration):
        """Test registering and unregistering strategy instances"""
        registry = ExitStrategyRegistry()
        strategy = TestExitStrategy(sample_exit_configuration)
        
        # Register instance
        registry.register_instance(strategy)
        assert sample_exit_configuration.strategy_id in registry._strategies
        assert registry.get_strategy(sample_exit_configuration.strategy_id) == strategy
        
        # Unregister instance
        registry.unregister_strategy(sample_exit_configuration.strategy_id)
        assert sample_exit_configuration.strategy_id not in registry._strategies
        assert registry.get_strategy(sample_exit_configuration.strategy_id) is None
    
    def test_get_all_strategies(self, sample_exit_configuration):
        """Test getting all registered strategies"""
        registry = ExitStrategyRegistry()
        
        strategy1 = TestExitStrategy(sample_exit_configuration)
        strategy2 = TestExitStrategy(sample_exit_configuration)
        strategy2.config.strategy_id = "strategy_002"
        
        registry.register_instance(strategy1)
        registry.register_instance(strategy2)
        
        all_strategies = registry.get_all_strategies()
        
        assert len(all_strategies) == 2
        assert strategy1 in all_strategies.values()
        assert strategy2 in all_strategies.values()
        
        # Should return copy, not reference
        all_strategies.clear()
        assert len(registry._strategies) == 2
    
    def test_get_available_types(self):
        """Test getting available strategy types"""
        registry = ExitStrategyRegistry()
        
        # Initially no types registered
        available_types = registry.get_available_types()
        assert len(available_types) == 0
        
        # Register some types
        registry.register_strategy_type(ExitType.TRAILING_STOP, TestExitStrategy)
        registry.register_strategy_type(ExitType.STOP_LOSS, TestExitStrategy)
        
        available_types = registry.get_available_types()
        
        assert len(available_types) == 2
        assert ExitType.TRAILING_STOP in available_types
        assert ExitType.STOP_LOSS in available_types
    
    def test_clear_registry(self, sample_exit_configuration):
        """Test clearing all registered strategies"""
        registry = ExitStrategyRegistry()
        
        strategy1 = TestExitStrategy(sample_exit_configuration)
        strategy2 = TestExitStrategy(sample_exit_configuration)
        strategy2.config.strategy_id = "strategy_002"
        
        registry.register_instance(strategy1)
        registry.register_instance(strategy2)
        registry.register_strategy_type(ExitType.TRAILING_STOP, TestExitStrategy)
        
        # Clear registry
        registry.clear()
        
        assert len(registry._strategies) == 0
        assert len(registry._strategy_types) == 1  # Strategy types not cleared
    
    def test_global_registry(self):
        """Test global registry instance"""
        assert isinstance(exit_strategy_registry, ExitStrategyRegistry)


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_exit_condition_comparison_errors(self, sample_exit_configuration):
        """Test error handling in condition comparisons"""
        strategy = TestExitStrategy(sample_exit_configuration)
        
        # Test with invalid operator
        result = strategy._compare_values(Decimal('10'), Decimal('5'), "invalid_operator")
        assert result is False
        
        # Test with None values
        result = strategy._compare_values(None, Decimal('5'), ">")
        # Should handle gracefully
        assert isinstance(result, bool)
    
    def test_exit_strategy_error_handling(self, sample_exit_configuration):
        """Test error handling in exit strategy operations"""
        strategy = TestExitStrategy(sample_exit_configuration)
        
        # Test position evaluation with invalid position
        invalid_position = {}  # Missing required fields
        
        # Should not crash
        asyncio.run(strategy.evaluate_position(invalid_position))
        
        assert strategy.total_evaluations == 0  # Should not increment on error
    
    def test_concurrent_operations(self, sample_exit_configuration):
        """Test concurrent operations on strategy"""
        strategy = TestExitStrategy(sample_exit_configuration)
        
        # Test multiple pause/resume calls
        asyncio.run(strategy.pause())
        asyncio.run(strategy.pause())  # Should be safe
        assert strategy.status == ExitStatus.PAUSED
        
        asyncio.run(strategy.resume())
        asyncio.run(strategy.resume())  # Should be safe
        assert strategy.status == ExitStatus.ACTIVE
    
    def test_exit_signal_with_extreme_values(self):
        """Test exit signal with extreme values"""
        signal = ExitSignal(
            signal_id="extreme_signal",
            strategy_id="extreme_strategy",
            position_id="extreme_position",
            symbol="EXTREME",
            exit_reason=ExitReason.SYSTEM_ERROR,
            exit_price=Decimal('999999999.99'),
            exit_quantity=Decimal('0.0001'),
            confidence=0.0,
            urgency=1.0
        )
        
        assert signal.exit_price == Decimal('999999999.99')
        assert signal.exit_quantity == Decimal('0.0001')
        assert signal.confidence == 0.0
        assert signal.urgency == 1.0


if __name__ == "__main__":
    pytest.main([__file__])
