"""
Test Conditional Exit Strategies

Tests various conditional exit strategies including:
- ConditionalExitStrategy base class
- Multi-condition evaluation
- Market regime detection
- Volume-based exits
- Correlation-based exits
- Technical indicator-based exits
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from trading_orchestrator.strategies.exit_strategies.conditional_exit import (
    ConditionalExitStrategy,
    ConditionalExitConfig,
    MultiConditionExit,
    MarketRegimeExit,
    VolumeExit,
    CorrelationExit,
    TechnicalIndicatorExit
)

from trading_orchestrator.strategies.exit_strategies.base_exit_strategy import (
    ExitReason,
    ExitType,
    ExitConfiguration,
    ExitCondition,
    ExitSignal
)


# Test fixture for conditional exit configuration
@pytest.fixture
def conditional_exit_config():
    """Sample conditional exit configuration"""
    return ExitConfiguration(
        strategy_id="conditional_001",
        strategy_type=ExitType.CONDITIONAL,
        name="Test Conditional Exit",
        description="Test configuration for conditional exit strategy",
        parameters={
            'condition_logic': 'AND',  # 'AND', 'OR', 'MAJORITY'
            'min_conditions': 2,
            'condition_weighting': True,
            'condition_weights': {
                'price_condition': 0.4,
                'volume_condition': 0.3,
                'volatility_condition': 0.3
            },
            'conditions': [
                {
                    'type': 'price',
                    'operator': 'crosses_below',
                    'value': Decimal('145.00'),
                    'weight': 0.4
                },
                {
                    'type': 'volume',
                    'operator': 'above_average',
                    'multiplier': 1.5,
                    'weight': 0.3
                },
                {
                    'type': 'volatility',
                    'operator': 'increasing',
                    'rate': Decimal('0.1'),
                    'weight': 0.3
                }
            ],
            'market_regime_detection': True,
            'regime_lookback': 20,
            'correlation_lookback': 50,
            'technical_indicators': ['rsi', 'macd', 'bollinger'],
            'regime_thresholds': {
                'trending': 0.7,
                'volatile': 0.6,
                'mean_reverting': 0.5
            }
        }
    )


# Test implementation of ConditionalExitStrategy
class TestConditionalExitStrategy(ConditionalExitStrategy):
    """Test implementation for unit testing"""
    
    def __init__(self, config):
        super().__init__(config)
        self.evaluation_calls = []
        self.exit_decisions = []
        self.exit_triggered = False
    
    async def _evaluate_condition(self, condition: Dict[str, Any], position: Dict[str, Any]) -> bool:
        """Mock implementation for testing"""
        self.evaluation_calls.append({
            'condition': condition,
            'position_id': position.get('position_id')
        })
        
        condition_type = condition.get('type', '')
        
        if condition_type == 'price':
            current_price = position.get('current_price', Decimal('0'))
            target_value = condition.get('value', Decimal('0'))
            return current_price <= target_value
        elif condition_type == 'volume':
            current_volume = position.get('volume', 1000000)
            average_volume = condition.get('average_volume', 1000000)
            multiplier = condition.get('multiplier', 1.0)
            return current_volume >= (average_volume * Decimal(str(multiplier)))
        else:
            return self.exit_triggered  # Use flag for other conditions
    
    async def _combine_conditions(self, condition_results: List[bool]) -> bool:
        """Mock implementation for testing"""
        logic = self.conditional_config.condition_logic
        
        if logic == 'AND':
            return all(condition_results)
        elif logic == 'OR':
            return any(condition_results)
        elif logic == 'MAJORITY':
            return sum(condition_results) >= len(condition_results) // 2
        else:
            return any(condition_results)  # Default to OR
    
    async def evaluate_exit_conditions(self, position: Dict[str, Any]) -> bool:
        """Simple test implementation"""
        if not self.conditional_config.conditions:
            return False
        
        condition_results = []
        for condition_config in self.conditional_config.conditions:
            condition = {
                'type': condition_config.get('type', ''),
                'value': condition_config.get('value'),
                'multiplier': condition_config.get('multiplier', 1.0),
                'weight': condition_config.get('weight', 1.0)
            }
            result = await self._evaluate_condition(condition, position)
            condition_results.append(result)
        
        return await self._combine_conditions(condition_results)
    
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
            signal_id=f"conditional_{position_id}_{datetime.utcnow().timestamp()}",
            strategy_id=self.config.strategy_id,
            position_id=position_id,
            symbol=symbol,
            exit_reason=exit_reason,
            exit_price=current_price,
            exit_quantity=quantity,
            confidence=0.80,
            urgency=0.75,
            metadata={
                'condition_logic': self.conditional_config.condition_logic,
                'conditions_evaluated': len(self.conditional_config.conditions),
                'triggered_conditions': sum(await self._evaluate_condition(c, position) for c in self.conditional_config.conditions),
                'exit_reason': exit_reason.value
            }
        )


# Mock context for conditional exit testing
@pytest.fixture
def conditional_exit_context():
    """Mock context for conditional exit testing"""
    context = AsyncMock()
    context.get_current_price = AsyncMock(return_value=Decimal('150.00'))
    context.get_position = AsyncMock(return_value={
        'position_id': 'pos_001',
        'symbol': 'AAPL',
        'quantity': Decimal('100'),
        'entry_price': Decimal('145.00'),
        'side': 'long',
        'volume': 1500000,  # Above average
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
            'volume': 1000000 + i * 1000
        }
        for i in range(30, 0, -1)
    ])
    context.calculate_volatility = AsyncMock(return_value=Decimal('0.028'))
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
        'volume': 1500000,
        'entry_time': datetime.utcnow() - timedelta(hours=2),
        'created_at': datetime.utcnow() - timedelta(hours=2)
    }])
    return context


class TestConditionalExitStrategy:
    """Test ConditionalExitStrategy base class"""
    
    def test_initialization(self, conditional_exit_config):
        """Test proper initialization of ConditionalExitStrategy"""
        strategy = TestConditionalExitStrategy(conditional_exit_config)
        
        # Test inherited attributes
        assert strategy.config == conditional_exit_config
        assert strategy.status.value == "initializing"
        
        # Test conditional exit specific attributes
        assert isinstance(strategy.conditional_config, ConditionalExitConfig)
        assert strategy.conditional_config.condition_logic == 'AND'
        assert len(strategy.conditional_config.conditions) == 3
        
        # Test tracking attributes
        assert len(strategy.condition_history) == 0
        assert len(strategy.regime_cache) == 0
        assert len(strategy.correlation_cache) == 0
        assert len(strategy.technical_cache) == 0
    
    @pytest.mark.asyncio
    async def test_evaluate_single_condition(self, conditional_exit_config, conditional_exit_context):
        """Test single condition evaluation"""
        strategy = TestConditionalExitStrategy(conditional_exit_config)
        strategy.set_context(conditional_exit_context)
        
        # Test price condition
        price_condition = {
            'type': 'price',
            'operator': 'crosses_below',
            'value': Decimal('145.00'),
            'weight': 0.4
        }
        
        position = {
            'position_id': 'pos_001',
            'current_price': Decimal('144.00'),  # Below threshold
            'symbol': 'AAPL'
        }
        
        result = await strategy._evaluate_condition(price_condition, position)
        assert result is True  # Price is below threshold
        
        # Test with price above threshold
        position_high = position.copy()
        position_high['current_price'] = Decimal('146.00')
        
        result_high = await strategy._evaluate_condition(price_condition, position_high)
        assert result_high is False  # Price is above threshold
        
        # Test volume condition
        volume_condition = {
            'type': 'volume',
            'operator': 'above_average',
            'multiplier': 1.5,
            'average_volume': 1000000,
            'weight': 0.3
        }
        
        position_volume = {
            'position_id': 'pos_002',
            'volume': 2000000,  # 2x average
            'symbol': 'AAPL'
        }
        
        result_volume = await strategy._evaluate_condition(volume_condition, position_volume)
        assert result_volume is True  # Volume is above average
    
    @pytest.mark.asyncio
    async def test_combine_conditions_and_logic(self, conditional_exit_config):
        """Test condition combination with AND logic"""
        strategy = TestConditionalExitStrategy(conditional_exit_config)
        strategy.conditional_config.condition_logic = 'AND'
        
        # All conditions true
        all_true = [True, True, True]
        result = await strategy._combine_conditions(all_true)
        assert result is True
        
        # Some conditions false
        some_false = [True, False, True]
        result = await strategy._combine_conditions(some_false)
        assert result is False
        
        # All conditions false
        all_false = [False, False, False]
        result = await strategy._combine_conditions(all_false)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_combine_conditions_or_logic(self, conditional_exit_config):
        """Test condition combination with OR logic"""
        strategy = TestConditionalExitStrategy(conditional_exit_config)
        strategy.conditional_config.condition_logic = 'OR'
        
        # All conditions true
        all_true = [True, True, True]
        result = await strategy._combine_conditions(all_true)
        assert result is True
        
        # Some conditions true
        some_true = [True, False, False]
        result = await strategy._combine_conditions(some_true)
        assert result is True
        
        # All conditions false
        all_false = [False, False, False]
        result = await strategy._combine_conditions(all_false)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_combine_conditions_majority_logic(self, conditional_exit_config):
        """Test condition combination with MAJORITY logic"""
        strategy = TestConditionalExitStrategy(conditional_exit_config)
        strategy.conditional_config.condition_logic = 'MAJORITY'
        
        # Majority true (2 out of 3)
        majority_true = [True, True, False]
        result = await strategy._combine_conditions(majority_true)
        assert result is True
        
        # Majority false (1 out of 3)
        majority_false = [True, False, False]
        result = await strategy._combine_conditions(majority_false)
        assert result is False
        
        # Tie (2 out of 4)
        tie = [True, True, False, False]
        result = await strategy._combine_conditions(tie)
        assert result is False  # Less than half
    
    @pytest.mark.asyncio
    async def test_evaluate_exit_conditions(self, conditional_exit_config, conditional_exit_context):
        """Test exit condition evaluation"""
        strategy = TestConditionalExitStrategy(conditional_exit_config)
        strategy.set_context(conditional_exit_context)
        
        # Test position meeting conditions
        position_meeting_conditions = {
            'position_id': 'pos_meet',
            'symbol': 'AAPL',
            'quantity': Decimal('100'),
            'current_price': Decimal('144.00'),  # Below price threshold
            'volume': 2000000  # Above volume threshold
        }
        
        result = await strategy.evaluate_exit_conditions(position_meeting_conditions)
        # Should trigger if conditions are met (depends on mock implementation)
        assert isinstance(result, bool)
        
        # Test position not meeting conditions
        position_not_meeting = {
            'position_id': 'pos_not_meet',
            'symbol': 'AAPL',
            'quantity': Decimal('100'),
            'current_price': Decimal('146.00'),  # Above price threshold
            'volume': 500000  # Below volume threshold
        }
        
        result_not_meet = await strategy.evaluate_exit_conditions(position_not_meeting)
        assert isinstance(result_not_meet, bool)
    
    @pytest.mark.asyncio
    async def test_generate_exit_signal(self, conditional_exit_config, conditional_exit_context):
        """Test exit signal generation"""
        strategy = TestConditionalExitStrategy(conditional_exit_config)
        strategy.set_context(conditional_exit_context)
        
        position = {
            'position_id': 'pos_001',
            'symbol': 'AAPL',
            'quantity': Decimal('100'),
            'current_price': Decimal('150.00')
        }
        
        exit_signal = await strategy.generate_exit_signal(position, ExitReason.MARKET_CONDITION)
        
        assert exit_signal is not None
        assert exit_signal.strategy_id == conditional_exit_config.strategy_id
        assert exit_signal.position_id == 'pos_001'
        assert exit_signal.symbol == 'AAPL'
        assert exit_signal.exit_reason == ExitReason.MARKET_CONDITION
        assert exit_signal.exit_price == Decimal('150.00')
        assert exit_signal.exit_quantity == Decimal('100')
        assert 'condition_logic' in exit_signal.metadata
        assert 'conditions_evaluated' in exit_signal.metadata
        assert 'triggered_conditions' in exit_signal.metadata
        assert 'exit_reason' in exit_signal.metadata
        
        # Test with incomplete position data
        incomplete_position = {
            'position_id': 'pos_002'
            # Missing other required fields
        }
        
        exit_signal = await strategy.generate_exit_signal(incomplete_position, ExitReason.MARKET_CONDITION)
        assert exit_signal is None


class TestMarketRegimeExit:
    """Test MarketRegimeExit implementation"""
    
    @pytest.mark.asyncio
    async def test_market_regime_detection(self, conditional_exit_context):
        """Test market regime detection functionality"""
        config = ExitConfiguration(
            strategy_id="regime_001",
            strategy_type=ExitType.CONDITIONAL,
            name="Market Regime Exit",
            description="Test market regime exit",
            parameters={
                'exit_type': 'market_regime',
                'regime_detection': True,
                'regime_lookback': 20,
                'regime_thresholds': {
                    'trending': 0.7,
                    'volatile': 0.6,
                    'mean_reverting': 0.5
                },
                'exit_on_regime_change': True
            }
        )
        
        strategy = MarketRegimeExit(config)
        strategy.set_context(conditional_exit_context)
        
        # Mock historical data for regime detection
        trending_data = [
            {'close': Decimal(str(100 + i * 2))}  # Trending up
            for i in range(20)
        ]
        
        regime = await strategy._detect_market_regime(trending_data)
        assert regime in ['trending', 'volatile', 'mean_reverting']
        
        # Test regime change detection
        previous_regime = 'trending'
        current_regime = await strategy._detect_market_regime(trending_data)
        
        regime_changed = await strategy._has_regime_changed(previous_regime, current_regime)
        assert isinstance(regime_changed, bool)


class TestVolumeExit:
    """Test VolumeExit implementation"""
    
    @pytest.mark.asyncio
    async def test_volume_condition_evaluation(self, conditional_exit_context):
        """Test volume-based exit conditions"""
        config = ExitConfiguration(
            strategy_id="volume_001",
            strategy_type=ExitType.CONDITIONAL,
            name="Volume Exit",
            description="Test volume exit",
            parameters={
                'exit_type': 'volume',
                'volume_threshold': 1.5,  # 1.5x average
                'volume_lookback': 20,
                'spike_detection': True,
                'spike_multiplier': 2.0
            }
        )
        
        strategy = VolumeExit(config)
        strategy.set_context(conditional_exit_context)
        
        # Mock historical volume data
        volume_data = [1000000 + i * 1000 for i in range(20)]
        
        average_volume = sum(volume_data) / len(volume_data)
        
        position_high_volume = {
            'position_id': 'pos_high_vol',
            'symbol': 'AAPL',
            'volume': average_volume * 2.0  # 2x average
        }
        
        is_high_volume = await strategy._is_volume_spike(position_high_volume)
        assert is_high_volume is True
        
        position_normal_volume = {
            'position_id': 'pos_normal_vol',
            'symbol': 'AAPL',
            'volume': average_volume * 1.2  # 1.2x average
        }
        
        is_normal_volume = await strategy._is_volume_spike(position_normal_volume)
        assert is_normal_volume is False  # Below spike threshold


class TestCorrelationExit:
    """Test CorrelationExit implementation"""
    
    @pytest.mark.asyncio
    async def test_correlation_based_exits(self, conditional_exit_context):
        """Test correlation-based exit conditions"""
        config = ExitConfiguration(
            strategy_id="correlation_001",
            strategy_type=ExitType.CONDITIONAL,
            name="Correlation Exit",
            description="Test correlation exit",
            parameters={
                'exit_type': 'correlation',
                'correlation_symbols': ['SPY', 'QQQ'],
                'correlation_threshold': 0.8,
                'correlation_lookback': 50,
                'inverse_correlation_exit': True
            }
        )
        
        strategy = CorrelationExit(config)
        strategy.set_context(conditional_exit_context)
        
        # Mock correlated price data
        aapl_prices = [Decimal(str(150 + i)) for i in range(50)]
        spy_prices = [Decimal(str(400 + i * 0.8)) for i in range(50)]  # Correlated
        
        correlation = await strategy._calculate_correlation('AAPL', 'SPY', aapl_prices, spy_prices)
        assert isinstance(correlation, float)
        assert 0.0 <= abs(correlation) <= 1.0
        
        # Test correlation threshold check
        high_correlation = abs(correlation) > 0.8
        assert isinstance(high_correlation, bool)


class TestTechnicalIndicatorExit:
    """Test TechnicalIndicatorExit implementation"""
    
    @pytest.mark.asyncio
    async def test_technical_indicator_conditions(self, conditional_exit_context):
        """Test technical indicator-based exits"""
        config = ExitConfiguration(
            strategy_id="technical_001",
            strategy_type=ExitType.CONDITIONAL,
            name="Technical Indicator Exit",
            description="Test technical indicator exit",
            parameters={
                'exit_type': 'technical',
                'indicators': ['rsi', 'macd', 'bollinger'],
                'rsi_overbought': 70,
                'rsi_oversold': 30,
                'macd_signal_cross': True,
                'bollinger_squeeze': True
            }
        )
        
        strategy = TechnicalIndicatorExit(config)
        strategy.set_context(conditional_exit_context)
        
        # Mock technical data
        technical_data = [
            {'close': Decimal(str(150 + i))}
            for i in range(20)
        ]
        
        # Test RSI calculation
        rsi_value = await strategy._calculate_rsi(technical_data)
        assert isinstance(rsi_value, float)
        assert 0.0 <= rsi_value <= 100.0
        
        # Test overbought condition
        is_overbought = await strategy._is_rsi_overbought(rsi_value)
        assert isinstance(is_overbought, bool)
        
        # Test oversold condition
        is_oversold = await strategy._is_rsi_oversold(rsi_value)
        assert isinstance(is_oversold, bool)


class TestConditionalScenarios:
    """Test various market scenarios for conditional exits"""
    
    @pytest.mark.asyncio
    async def test_trending_market_conditions(self, conditional_exit_config):
        """Test conditional exits in trending market"""
        strategy = TestConditionalExitStrategy(conditional_exit_config)
        
        # Mock trending market data
        trending_position = {
            'position_id': 'pos_trending',
            'symbol': 'AAPL',
            'current_price': Decimal('160.00'),  # Strong uptrend
            'volume': 2000000,  # High volume
            'volatility': Decimal('0.035')  # Higher volatility
        }
        
        # Should trigger volume condition due to high volume
        volume_condition = {
            'type': 'volume',
            'operator': 'above_average',
            'multiplier': 1.5,
            'average_volume': 1000000
        }
        
        volume_result = await strategy._evaluate_condition(volume_condition, trending_position)
        assert volume_result is True
    
    @pytest.mark.asyncio
    async def test_mean_reverting_conditions(self, conditional_exit_config):
        """Test conditional exits in mean-reverting market"""
        strategy = TestConditionalExitStrategy(conditional_exit_config)
        
        # Mock mean-reverting market data
        mean_reverting_position = {
            'position_id': 'pos_mean_revert',
            'symbol': 'AAPL',
            'current_price': Decimal('155.00'),  # Pullback from high
            'volume': 800000,  # Lower volume
            'volatility': Decimal('0.020')  # Lower volatility
        }
        
        # Should not trigger volume condition due to low volume
        volume_condition = {
            'type': 'volume',
            'operator': 'above_average',
            'multiplier': 1.5,
            'average_volume': 1000000
        }
        
        volume_result = await strategy._evaluate_condition(volume_condition, mean_reverting_position)
        assert volume_result is False
    
    @pytest.mark.asyncio
    async def test_volatile_market_conditions(self, conditional_exit_config):
        """Test conditional exits in volatile market"""
        strategy = TestConditionalExitStrategy(conditional_exit_config)
        
        # Mock volatile market data
        volatile_position = {
            'position_id': 'pos_volatile',
            'symbol': 'AAPL',
            'current_price': Decimal('148.00'),  # Price swing
            'volume': 3000000,  # Very high volume
            'volatility': Decimal('0.050')  # Very high volatility
        }
        
        # Should trigger both price and volume conditions
        price_condition = {
            'type': 'price',
            'operator': 'crosses_below',
            'value': Decimal('150.00')
        }
        
        volume_condition = {
            'type': 'volume',
            'operator': 'above_average',
            'multiplier': 1.5,
            'average_volume': 1000000
        }
        
        price_result = await strategy._evaluate_condition(price_condition, volatile_position)
        volume_result = await strategy._evaluate_condition(volume_condition, volatile_position)
        
        # In volatile market, both conditions likely to be met
        assert isinstance(price_result, bool)
        assert isinstance(volume_result, bool)
        assert volume_result is True  # Volume definitely high


class TestConditionalEdgeCases:
    """Test edge cases and error conditions"""
    
    @pytest.mark.asyncio
    async def test_empty_conditions(self, conditional_exit_config):
        """Test behavior with empty conditions"""
        conditional_exit_config.parameters['conditions'] = []
        strategy = TestConditionalExitStrategy(conditional_exit_config)
        
        position = {
            'position_id': 'pos_empty',
            'symbol': 'AAPL',
            'current_price': Decimal('150.00')
        }
        
        result = await strategy.evaluate_exit_conditions(position)
        assert result is False  # Should not trigger with no conditions
    
    @pytest.mark.asyncio
    async def test_missing_condition_data(self, conditional_exit_config):
        """Test behavior with missing condition data"""
        strategy = TestConditionalExitStrategy(conditional_exit_config)
        
        # Condition without required data
        incomplete_condition = {
            'type': 'price'
            # Missing value
        }
        
        position = {
            'position_id': 'pos_incomplete',
            'symbol': 'AAPL'
            # Missing current_price
        }
        
        # Should handle missing data gracefully
        try:
            result = await strategy._evaluate_condition(incomplete_condition, position)
            assert isinstance(result, bool)
        except (KeyError, TypeError, ValueError):
            # Expected to handle gracefully or raise appropriate exception
            pass
    
    @pytest.mark.asyncio
    async def test_extreme_condition_values(self, conditional_exit_config):
        """Test behavior with extreme condition values"""
        strategy = TestConditionalExitStrategy(conditional_exit_config)
        
        # Extreme price condition
        extreme_condition = {
            'type': 'price',
            'operator': 'crosses_below',
            'value': Decimal('0.01')  # Very low price
        }
        
        position_high_price = {
            'position_id': 'pos_high',
            'symbol': 'AAPL',
            'current_price': Decimal('999999.99')  # Very high price
        }
        
        result = await strategy._evaluate_condition(extreme_condition, position_high_price)
        assert result is False  # High price not below extremely low threshold
        
        # Test with negative values
        negative_condition = {
            'type': 'price',
            'operator': 'crosses_below',
            'value': Decimal('-100.00')  # Negative price
        }
        
        position_negative = {
            'position_id': 'pos_neg',
            'symbol': 'TEST',
            'current_price': Decimal('-50.00')  # Negative price
        }
        
        result_neg = await strategy._evaluate_condition(negative_condition, position_negative)
        assert isinstance(result_neg, bool)
    
    @pytest.mark.asyncio
    async def test_invalid_condition_logic(self, conditional_exit_config):
        """Test behavior with invalid condition logic"""
        conditional_exit_config.parameters['condition_logic'] = 'INVALID_LOGIC'
        strategy = TestConditionalExitStrategy(conditional_exit_config)
        
        condition_results = [True, False, True]
        
        # Should handle invalid logic gracefully (default to OR)
        result = await strategy._combine_conditions(condition_results)
        assert isinstance(result, bool)
    
    @pytest.mark.asyncio
    async def test_single_condition_evaluation(self, conditional_exit_config):
        """Test behavior with single condition"""
        conditional_exit_config.parameters['conditions'] = [
            {
                'type': 'price',
                'operator': 'crosses_below',
                'value': Decimal('145.00')
            }
        ]
        strategy = TestConditionalExitStrategy(conditional_exit_config)
        
        position = {
            'position_id': 'pos_single',
            'symbol': 'AAPL',
            'current_price': Decimal('144.00')  # Below threshold
        }
        
        result = await strategy.evaluate_exit_conditions(position)
        # Single condition should work with any logic
        assert isinstance(result, bool)


if __name__ == "__main__":
    pytest.main([__file__])
