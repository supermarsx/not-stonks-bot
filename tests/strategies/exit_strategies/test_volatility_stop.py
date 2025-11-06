"""
Test Volatility-Based Exit Strategies

Tests various volatility-based exit strategies including:
- VolatilityStopStrategy base class
- ATR-based volatility stops
- Bollinger band exits
- Standard deviation stops
- Volatility-adjusted exits
- Market condition-based volatility exits
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from trading_orchestrator.strategies.exit_strategies.volatility_stop import (
    VolatilityStopStrategy,
    ATRVolatilityStop,
    BollingerBandExit,
    StandardDeviationStop,
    VolatilityThresholdExit,
    VolatilityStopConfig
)

from trading_orchestrator.strategies.exit_strategies.base_exit_strategy import (
    ExitReason,
    ExitType,
    ExitConfiguration,
    ExitCondition,
    ExitSignal
)


# Test fixture for volatility stop configuration
@pytest.fixture
def volatility_stop_config():
    """Sample volatility stop configuration"""
    return ExitConfiguration(
        strategy_id="volatility_stop_001",
        strategy_type=ExitType.VOLATILITY_STOP,
        name="Test Volatility Stop",
        description="Test configuration for volatility stop strategy",
        parameters={
            'volatility_method': 'ATR',  # 'ATR', 'Bollinger', 'StdDev', 'Threshold'
            'lookback_period': 20,
            'volatility_multiplier': Decimal('2.0'),
            'upper_threshold': Decimal('0.05'),
            'lower_threshold': Decimal('0.01'),
            'band_width_threshold': Decimal('0.10'),
            'squeeze_factor': Decimal('0.5'),
            'volatility_regime': 'adaptive',
            'min_periods': 14,
            'max_periods': 50,
            'adaptation_rate': Decimal('0.1'),
            'current_regime': 'normal',
            'regime_thresholds': {
                'low': Decimal('0.015'),
                'normal': Decimal('0.025'),
                'high': Decimal('0.035'),
                'extreme': Decimal('0.050')
            }
        }
    )


# Test implementation of VolatilityStopStrategy
class TestVolatilityStopStrategy(VolatilityStopStrategy):
    """Test implementation for unit testing"""
    
    def __init__(self, config):
        super().__init__(config)
        self.volatility_calculation_calls = []
        self.exit_decisions = []
        self.exit_triggered = False
    
    async def _calculate_volatility(self, symbol: str, prices: List[Dict[str, Any]]) -> Decimal:
        """Mock implementation for testing"""
        self.volatility_calculation_calls.append({
            'symbol': symbol,
            'price_count': len(prices),
            'method': self.volatility_config.volatility_method
        })
        
        if not prices:
            return Decimal('0.02')  # Default volatility
        
        closes = [Decimal(str(p.get('close', 0))) for p in prices]
        
        if self.volatility_config.volatility_method == 'ATR':
            # Mock ATR calculation
            return Decimal('2.5')
        elif self.volatility_config.volatility_method == 'StdDev':
            # Mock standard deviation calculation
            if len(closes) < 2:
                return Decimal('0')
            
            mean = sum(closes) / len(closes)
            variance = sum((price - mean) ** 2 for price in closes) / (len(closes) - 1)
            return variance.sqrt()
        else:
            # Default simple volatility
            return Decimal('0.025')
    
    async def _determine_exit_trigger(self, position: Dict[str, Any], volatility: Decimal) -> bool:
        """Mock implementation for testing"""
        return self.exit_triggered
    
    async def evaluate_exit_conditions(self, position: Dict[str, Any]) -> bool:
        """Simple test implementation"""
        current_price = position.get('current_price', Decimal('0'))
        if not current_price:
            return False
        
        symbol = position.get('symbol', '')
        
        # Get mock price data
        mock_prices = [
            {'close': Decimal(str(140 + i * 0.5))}
            for i in range(self.volatility_config.lookback_period)
        ]
        
        volatility = await self._calculate_volatility(symbol, mock_prices)
        return await self._determine_exit_trigger(position, volatility)
    
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
            signal_id=f"vol_{position_id}_{datetime.utcnow().timestamp()}",
            strategy_id=self.config.strategy_id,
            position_id=position_id,
            symbol=symbol,
            exit_reason=exit_reason,
            exit_price=current_price,
            exit_quantity=quantity,
            confidence=0.88,
            urgency=0.85,
            metadata={
                'volatility_method': self.volatility_config.volatility_method,
                'volatility_value': float(current_price * Decimal('0.025')),  # Mock volatility
                'threshold': self.volatility_config.upper_threshold,
                'regime': self.volatility_config.current_regime
            }
        )


# Mock context for volatility stop testing
@pytest.fixture
def volatility_stop_context():
    """Mock context for volatility stop testing"""
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
    context.get_historical_data = AsyncMock(return_value=[
        {
            'timestamp': datetime.utcnow() - timedelta(hours=i),
            'open': Decimal(str(145 + i * 0.3)),
            'high': Decimal(str(147 + i * 0.3)),
            'low': Decimal(str(143 + i * 0.3)),
            'close': Decimal(str(145 + i * 0.3)),
            'volume': 1000000
        }
        for i in range(30, 0, -1)
    ])
    context.calculate_volatility = AsyncMock(return_value=Decimal('0.028'))
    context.submit_exit_order = AsyncMock(return_value={'success': True, 'order_id': 'order_123'})
    context.get_portfolio_value = AsyncMock(return_value=Decimal('1000000'))
    context.get_risk_metrics = AsyncMock(return_value={'max_drawdown': 0.05, 'var_99': Decimal('5000')})
    context.get_positions = AsyncAsyncMock(return_value=[{
        'position_id': 'pos_001',
        'symbol': 'AAPL',
        'quantity': Decimal('100'),
        'entry_price': Decimal('145.00'),
        'side': 'long',
        'current_price': Decimal('150.00'),
        'entry_time': datetime.utcnow() - timedelta(hours=2),
        'created_at': datetime.utcnow() - timedelta(hours=2)
    }])
    return context


# Mock class to fix async issue
class AsyncAsyncMock(AsyncMock):
    """Fixed AsyncMock to handle async methods properly"""
    pass


class TestVolatilityStopStrategy:
    """Test VolatilityStopStrategy base class"""
    
    def test_initialization(self, volatility_stop_config):
        """Test proper initialization of VolatilityStopStrategy"""
        strategy = TestVolatilityStopStrategy(volatility_stop_config)
        
        # Test inherited attributes
        assert strategy.config == volatility_stop_config
        assert strategy.status.value == "initializing"
        
        # Test volatility stop specific attributes
        assert isinstance(strategy.volatility_config, VolatilityStopConfig)
        assert strategy.volatility_config.volatility_method == 'ATR'
        assert strategy.volatility_config.lookback_period == 20
        assert strategy.volatility_config.volatility_multiplier == Decimal('2.0')
        
        # Test tracking attributes
        assert len(strategy.volatility_cache) == 0
        assert len(strategy.regime_history) == 0
        assert len(strategy.exit_signals) == 0
        assert len(strategy.trigger_history) == 0
    
    @pytest.mark.asyncio
    async def test_calculate_volatility_atr(self, volatility_stop_config, volatility_stop_context):
        """Test ATR volatility calculation"""
        strategy = TestVolatilityStopStrategy(volatility_stop_config)
        strategy.set_context(volatility_stop_context)
        
        # Update config for ATR method
        volatility_stop_config.parameters['volatility_method'] = 'ATR'
        
        symbol = 'AAPL'
        mock_prices = [
            {'high': Decimal('152'), 'low': Decimal('148'), 'close': Decimal('150')},
            {'high': Decimal('153'), 'low': Decimal('149'), 'close': Decimal('151')},
            {'high': Decimal('154'), 'low': Decimal('150'), 'close': Decimal('152')},
            {'high': Decimal('155'), 'low': Decimal('151'), 'close': Decimal('153')},
            {'high': Decimal('156'), 'low': Decimal('152'), 'close': Decimal('154')}
        ]
        
        volatility = await strategy._calculate_volatility(symbol, mock_prices)
        
        assert isinstance(volatility, Decimal)
        assert volatility > 0
        assert len(strategy.volatility_calculation_calls) == 1
    
    @pytest.mark.asyncio
    async def test_calculate_volatility_stddev(self, volatility_stop_config):
        """Test standard deviation volatility calculation"""
        # Update config for StdDev method
        volatility_stop_config.parameters['volatility_method'] = 'StdDev'
        
        strategy = TestVolatilityStopStrategy(volatility_stop_config)
        
        symbol = 'AAPL'
        mock_prices = [
            {'close': Decimal('100')},
            {'close': Decimal('102')},
            {'close': Decimal('101')},
            {'close': Decimal('103')},
            {'close': Decimal('99')}
        ]
        
        volatility = await strategy._calculate_volatility(symbol, mock_prices)
        
        assert isinstance(volatility, Decimal)
        assert volatility >= 0
        
        # Test with insufficient data
        insufficient_prices = [{'close': Decimal('100')}]
        volatility_small = await strategy._calculate_volatility(symbol, insufficient_prices)
        assert volatility_small == Decimal('0')
    
    @pytest.mark.asyncio
    async def test_calculate_volatility_empty_data(self, volatility_stop_config):
        """Test volatility calculation with empty data"""
        strategy = TestVolatilityStopStrategy(volatility_stop_config)
        
        symbol = 'AAPL'
        empty_prices = []
        
        volatility = await strategy._calculate_volatility(symbol, empty_prices)
        
        assert isinstance(volatility, Decimal)
        # Should return default volatility for empty data
        assert volatility == Decimal('0.02')
    
    @pytest.mark.asyncio
    async def test_determine_volatility_regime(self, volatility_stop_config):
        """Test volatility regime determination"""
        strategy = TestVolatilityStopStrategy(volatility_stop_config)
        
        # Test low volatility regime
        low_volatility = Decimal('0.015')
        regime = await strategy._determine_volatility_regime(low_volatility)
        assert regime == 'low'
        
        # Test normal volatility regime
        normal_volatility = Decimal('0.025')
        regime = await strategy._determine_volatility_regime(normal_volatility)
        assert regime == 'normal'
        
        # Test high volatility regime
        high_volatility = Decimal('0.035')
        regime = await strategy._determine_volatility_regime(high_volatility)
        assert regime == 'high'
        
        # Test extreme volatility regime
        extreme_volatility = Decimal('0.055')
        regime = await strategy._determine_volatility_regime(extreme_volatility)
        assert regime == 'extreme'
    
    @pytest.mark.asyncio
    async def test_adapt_volatility_parameters(self, volatility_stop_config):
        """Test volatility parameter adaptation"""
        strategy = TestVolatilityStopStrategy(volatility_stop_config)
        
        # Test adaptation to higher volatility
        original_multiplier = strategy.volatility_config.volatility_multiplier
        original_period = strategy.volatility_config.lookback_period
        
        new_regime = 'high'
        await strategy._adapt_volatility_parameters(new_regime)
        
        # Should adapt parameters for high volatility regime
        assert strategy.volatility_config.volatility_multiplier >= original_multiplier
        assert strategy.volatility_config.lookback_period >= original_period
        
        # Test adaptation to lower volatility
        new_regime = 'low'
        await strategy._adapt_volatility_parameters(new_regime)
        
        # Should adapt parameters for low volatility regime
        assert strategy.volatility_config.volatility_multiplier <= original_multiplier
        assert strategy.volatility_config.lookback_period <= original_period
    
    @pytest.mark.asyncio
    async def test_evaluate_exit_conditions(self, volatility_stop_config, volatility_stop_context):
        """Test exit condition evaluation"""
        strategy = TestVolatilityStopStrategy(volatility_stop_config)
        strategy.set_context(volatility_stop_context)
        
        position = {
            'position_id': 'pos_001',
            'symbol': 'AAPL',
            'quantity': Decimal('100'),
            'entry_price': Decimal('145.00'),
            'current_price': Decimal('152.00'),
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
            'quantity': Decimal('100'),
            'entry_price': Decimal('145.00'),
            # Missing current_price
            'side': 'long'
        }
        
        result = await strategy.evaluate_exit_conditions(position_no_price)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_generate_exit_signal(self, volatility_stop_config, volatility_stop_context):
        """Test exit signal generation"""
        strategy = TestVolatilityStopStrategy(volatility_stop_config)
        strategy.set_context(volatility_stop_context)
        
        position = {
            'position_id': 'pos_001',
            'symbol': 'AAPL',
            'quantity': Decimal('100'),
            'entry_price': Decimal('145.00'),
            'current_price': Decimal('152.00'),
            'side': 'long'
        }
        
        exit_signal = await strategy.generate_exit_signal(position, ExitReason.VOLATILITY_STOP)
        
        assert exit_signal is not None
        assert exit_signal.strategy_id == volatility_stop_config.strategy_id
        assert exit_signal.position_id == 'pos_001'
        assert exit_signal.symbol == 'AAPL'
        assert exit_signal.exit_reason == ExitReason.VOLATILITY_STOP
        assert exit_signal.exit_price == Decimal('152.00')
        assert exit_signal.exit_quantity == Decimal('100')
        assert 0.0 <= exit_signal.confidence <= 1.0
        assert 0.0 <= exit_signal.urgency <= 1.0
        assert 'volatility_method' in exit_signal.metadata
        assert 'volatility_value' in exit_signal.metadata
        assert 'threshold' in exit_signal.metadata
        assert 'regime' in exit_signal.metadata
        
        # Test with incomplete position data
        incomplete_position = {
            'position_id': 'pos_002'
            # Missing other required fields
        }
        
        exit_signal = await strategy.generate_exit_signal(incomplete_position, ExitReason.VOLATILITY_STOP)
        assert exit_signal is None
    
    @pytest.mark.asyncio
    async def test_calculate_volatility_score(self, volatility_stop_config):
        """Test volatility score calculation"""
        strategy = TestVolatilityStopStrategy(volatility_stop_config)
        
        current_volatility = Decimal('0.030')
        historical_volatility = Decimal('0.025')
        regime = 'high'
        
        score = strategy._calculate_volatility_score(current_volatility, historical_volatility, regime)
        assert isinstance(score, float)
        assert score >= 0.0
        
        # Test with different regime
        score_low = strategy._calculate_volatility_score(current_volatility, historical_volatility, 'low')
        assert isinstance(score_low, float)
        assert score_low != score  # Different regimes should give different scores
    
    def test_validate_volatility_parameters(self, volatility_stop_config):
        """Test volatility parameter validation"""
        strategy = TestVolatilityStopStrategy(volatility_stop_config)
        
        # Test valid parameters
        valid_params = {
            'lookback_period': 20,
            'volatility_multiplier': Decimal('2.0'),
            'upper_threshold': Decimal('0.05'),
            'lower_threshold': Decimal('0.01')
        }
        
        is_valid = strategy._validate_volatility_parameters(valid_params)
        assert is_valid is True
        
        # Test invalid parameters
        invalid_params = {
            'lookback_period': 5,  # Too small
            'volatility_multiplier': Decimal('0.5'),  # Too small
            'upper_threshold': Decimal('0.01'),  # Lower than lower threshold
            'lower_threshold': Decimal('0.05')
        }
        
        is_valid = strategy._validate_volatility_parameters(invalid_params)
        assert is_valid is False


class TestATRVolatilityStop:
    """Test ATRVolatilityStop implementation"""
    
    @pytest.mark.asyncio
    async def test_calculate_atr_volatility(self, volatility_stop_context):
        """Test ATR volatility calculation"""
        config = ExitConfiguration(
            strategy_id="atr_vol_001",
            strategy_type=ExitType.VOLATILITY_STOP,
            name="ATR Volatility Stop",
            description="Test ATR volatility stop",
            parameters={
                'volatility_method': 'ATR',
                'lookback_period': 14,
                'volatility_multiplier': Decimal('2.0'),
                'atr_period': 14
            }
        )
        
        strategy = ATRVolatilityStop(config)
        strategy.set_context(volatility_stop_context)
        
        # Mock historical data for ATR calculation
        atr_data = [
            {'high': Decimal('152'), 'low': Decimal('148'), 'close': Decimal('150')},
            {'high': Decimal('153'), 'low': Decimal('149'), 'close': Decimal('151')},
            {'high': Decimal('154'), 'low': Decimal('150'), 'close': Decimal('152')},
            {'high': Decimal('155'), 'low': Decimal('151'), 'close': Decimal('153')},
            {'high': Decimal('156'), 'low': Decimal('152'), 'close': Decimal('154')}
        ]
        
        volatility = await strategy._calculate_volatility('AAPL', atr_data)
        assert isinstance(volatility, Decimal)
        assert volatility > 0


class TestBollingerBandExit:
    """Test BollingerBandExit implementation"""
    
    @pytest.mark.asyncio
    async def test_calculate_bollinger_volatility(self, volatility_stop_context):
        """Test Bollinger band volatility calculation"""
        config = ExitConfiguration(
            strategy_id="bollinger_001",
            strategy_type=ExitType.VOLATILITY_STOP,
            name="Bollinger Band Exit",
            description="Test Bollinger band exit",
            parameters={
                'volatility_method': 'Bollinger',
                'lookback_period': 20,
                'volatility_multiplier': Decimal('2.0'),
                'squeeze_factor': Decimal('0.5')
            }
        )
        
        strategy = BollingerBandExit(config)
        strategy.set_context(volatility_stop_context)
        
        # Mock price data for Bollinger calculation
        bollinger_data = [
            {'close': Decimal(str(150 + i * 0.5))}
            for i in range(20)
        ]
        
        volatility = await strategy._calculate_volatility('AAPL', bollinger_data)
        assert isinstance(volatility, Decimal)
        assert volatility >= 0


class TestStandardDeviationStop:
    """Test StandardDeviationStop implementation"""
    
    @pytest.mark.asyncio
    async def test_calculate_stddev_volatility(self):
        """Test standard deviation volatility calculation"""
        config = ExitConfiguration(
            strategy_id="stddev_001",
            strategy_type=ExitType.VOLATILITY_STOP,
            name="Standard Deviation Stop",
            description="Test standard deviation stop",
            parameters={
                'volatility_method': 'StdDev',
                'lookback_period': 30,
                'volatility_multiplier': Decimal('1.5')
            }
        )
        
        strategy = StandardDeviationStop(config)
        
        # Mock price data
        stddev_data = [
            {'close': Decimal(str(150 + i))}
            for i in range(30)
        ]
        
        volatility = await strategy._calculate_volatility('AAPL', stddev_data)
        assert isinstance(volatility, Decimal)
        assert volatility > 0


class TestVolatilityThresholdExit:
    """Test VolatilityThresholdExit implementation"""
    
    def test_threshold_exit_initialization(self):
        """Test threshold exit initialization"""
        config = ExitConfiguration(
            strategy_id="threshold_001",
            strategy_type=ExitType.VOLATILITY_STOP,
            name="Threshold Exit",
            description="Test threshold exit",
            parameters={
                'volatility_method': 'Threshold',
                'upper_threshold': Decimal('0.05'),
                'lower_threshold': Decimal('0.01'),
                'band_width_threshold': Decimal('0.10')
            }
        )
        
        strategy = VolatilityThresholdExit(config)
        assert strategy.volatility_config.volatility_method == 'Threshold'
        assert strategy.volatility_config.upper_threshold == Decimal('0.05')
        assert strategy.volatility_config.lower_threshold == Decimal('0.01')


class TestVolatilityStopScenarios:
    """Test various market scenarios for volatility stops"""
    
    @pytest.mark.asyncio
    async def test_low_volatility_regime(self, volatility_stop_config):
        """Test behavior in low volatility regime"""
        strategy = TestVolatilityStopStrategy(volatility_stop_config)
        
        # Simulate low volatility environment
        low_vol_prices = [
            {'close': Decimal('150.00') + Decimal(str(i * 0.1))}  # Low volatility
            for i in range(20)
        ]
        
        volatility = await strategy._calculate_volatility('AAPL', low_vol_prices)
        regime = await strategy._determine_volatility_regime(volatility)
        
        assert regime in ['low', 'normal']  # Should be low or normal
        
        # Check that parameters are adjusted for low volatility
        original_params = {
            'multiplier': strategy.volatility_config.volatility_multiplier,
            'period': strategy.volatility_config.lookback_period
        }
        
        await strategy._adapt_volatility_parameters(regime)
        
        # Should have tighter stops in low volatility
        assert strategy.volatility_config.volatility_multiplier <= original_params['multiplier']
    
    @pytest.mark.asyncio
    async def test_high_volatility_regime(self, volatility_stop_config):
        """Test behavior in high volatility regime"""
        strategy = TestVolatilityStopStrategy(volatility_stop_config)
        
        # Simulate high volatility environment
        high_vol_prices = [
            {'close': Decimal('150.00') + Decimal(str(i * 2.0))}  # High volatility
            for i in range(20)
        ]
        
        volatility = await strategy._calculate_volatility('AAPL', high_vol_prices)
        regime = await strategy._determine_volatility_regime(volatility)
        
        assert regime in ['high', 'extreme']  # Should be high or extreme
        
        # Check that parameters are adjusted for high volatility
        original_params = {
            'multiplier': strategy.volatility_config.volatility_multiplier,
            'period': strategy.volatility_config.lookback_period
        }
        
        await strategy._adapt_volatility_parameters(regime)
        
        # Should have wider stops in high volatility
        assert strategy.volatility_config.volatility_multiplier >= original_params['multiplier']
    
    @pytest.mark.asyncio
    async def test_volatility_squeeze_scenario(self, volatility_stop_config):
        """Test volatility squeeze detection and response"""
        config = ExitConfiguration(
            strategy_id="squeeze_001",
            strategy_type=ExitType.VOLATILITY_STOP,
            name="Volatility Squeeze",
            description="Test volatility squeeze scenario",
            parameters={
                'volatility_method': 'Bollinger',
                'lookback_period': 20,
                'squeeze_factor': Decimal('0.3'),  # Low squeeze threshold
                'band_width_threshold': Decimal('0.05')
            }
        )
        
        strategy = TestVolatilityStopStrategy(config)
        
        # Simulate squeeze pattern (low volatility followed by potential breakout)
        squeeze_prices = [
            {'close': Decimal('150.00') + Decimal(str(0.5 * (1 if i < 10 else 3.0)))}  # Squeeze then expansion
            for i in range(20)
        ]
        
        volatility = await strategy._calculate_volatility('AAPL', squeeze_prices)
        regime = await strategy._determine_volatility_regime(volatility)
        
        # Should detect low volatility during squeeze
        assert regime in ['low', 'normal']
        
        # Parameters should adapt to squeeze conditions
        await strategy._adapt_volatility_parameters(regime)
        assert strategy.volatility_config.lookback_period >= config.parameters['lookback_period']
    
    @pytest.mark.asyncio
    async def test_extreme_volatility_event(self, volatility_stop_config):
        """Test response to extreme volatility events"""
        strategy = TestVolatilityStopStrategy(volatility_stop_config)
        
        # Simulate extreme volatility event
        extreme_prices = [
            {'close': Decimal('150.00') + Decimal(str(i * 5.0))}  # Very high volatility
            for i in range(20)
        ]
        
        volatility = await strategy._calculate_volatility('AAPL', extreme_prices)
        regime = await strategy._determine_volatility_regime(volatility)
        
        # Should detect extreme volatility
        assert regime == 'extreme'
        
        # Should adapt parameters significantly for extreme volatility
        original_multiplier = strategy.volatility_config.volatility_multiplier
        await strategy._adapt_volatility_parameters(regime)
        
        # Should significantly widen stops for extreme volatility
        assert strategy.volatility_config.volatility_multiplier > original_multiplier


class TestVolatilityStopEdgeCases:
    """Test edge cases and error conditions"""
    
    @pytest.mark.asyncio
    async def test_zero_volatility(self, volatility_stop_config):
        """Test behavior with zero volatility"""
        strategy = TestVolatilityStopStrategy(volatility_stop_config)
        
        # Set config for zero volatility scenario
        volatility_stop_config.parameters['lower_threshold'] = Decimal('0')
        volatility_stop_config.parameters['upper_threshold'] = Decimal('0.001')
        
        # Mock zero volatility prices
        zero_vol_prices = [
            {'close': Decimal('150.00')}  # Constant price
            for _ in range(20)
        ]
        
        volatility = await strategy._calculate_volatility('AAPL', zero_vol_prices)
        regime = await strategy._determine_volatility_regime(volatility)
        
        # Should handle zero or near-zero volatility gracefully
        assert isinstance(regime, str)
        assert regime in ['low', 'normal', 'high', 'extreme']
    
    @pytest.mark.asyncio
    async def test_very_high_volatility(self, volatility_stop_config):
        """Test behavior with extremely high volatility"""
        strategy = TestVolatilityStopStrategy(volatility_stop_config)
        
        # Mock extremely volatile prices
        extreme_vol_prices = [
            {'close': Decimal('150.00') + Decimal(str(i * 50.0))}  # Extreme volatility
            for i in range(20)
        ]
        
        volatility = await strategy._calculate_volatility('AAPL', extreme_vol_prices)
        regime = await strategy._determine_volatility_regime(volatility)
        
        # Should detect extreme volatility
        assert regime == 'extreme'
    
    @pytest.mark.asyncio
    async def test_insufficient_data(self, volatility_stop_config):
        """Test behavior with insufficient historical data"""
        strategy = TestVolatilityStopStrategy(volatility_stop_config)
        
        # Mock insufficient data
        insufficient_prices = [
            {'close': Decimal('150.00')} for _ in range(2)  # Too few data points
        ]
        
        volatility = await strategy._calculate_volatility('AAPL', insufficient_prices)
        assert isinstance(volatility, Decimal)
        assert volatility >= 0  # Should return some valid value
    
    def test_invalid_regime_thresholds(self, volatility_stop_config):
        """Test behavior with invalid regime thresholds"""
        strategy = TestVolatilityStopStrategy(volatility_stop_config)
        
        # Set invalid thresholds
        invalid_thresholds = {
            'low': Decimal('0.05'),
            'normal': Decimal('0.03'),  # Lower than low
            'high': Decimal('0.02'),     # Lower than normal
            'extreme': Decimal('0.01')   # Lower than high
        }
        strategy.volatility_config.regime_thresholds = invalid_thresholds
        
        # Should handle invalid thresholds gracefully
        volatility = Decimal('0.025')
        regime = strategy._determine_volatility_regime(volatility)
        assert isinstance(regime, str)  # Should still return some regime
    
    @pytest.mark.asyncio
    async def test_missing_price_data(self, volatility_stop_config):
        """Test behavior with missing price data"""
        strategy = TestVolatilityStopStrategy(volatility_stop_config)
        
        # Mock missing data
        missing_prices = [
            {'open': Decimal('150.00')},  # Missing close price
            {'high': Decimal('152.00')},   # Missing close price
            {}                             # Completely empty
        ]
        
        volatility = await strategy._calculate_volatility('AAPL', missing_prices)
        assert isinstance(volatility, Decimal)
        assert volatility >= 0  # Should handle missing data gracefully


if __name__ == "__main__":
    pytest.main([__file__])
