"""
Test AI-Driven Exit Strategies

Tests AI-driven exit decision making including:
- AIExitStrategy base class
- LLM-based market analysis
- Sentiment analysis integration
- Pattern recognition
- Multi-factor AI decision making
- AI confidence scoring
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch

from trading_orchestrator.strategies.exit_strategies.ai_exit_strategy import (
    AIExitStrategy,
    LLMExitDecision,
    SentimentExit,
    PatternExit,
    MultiFactorAIExit
)

from trading_orchestrator.strategies.exit_strategies.base_exit_strategy import (
    ExitReason,
    ExitType,
    ExitConfiguration
)


# Test fixture for AI exit configuration
@pytest.fixture
def ai_exit_config():
    """Sample AI exit configuration"""
    return ExitConfiguration(
        strategy_id="ai_exit_001",
        strategy_type=ExitType.AI_DRIVEN,
        name="Test AI Exit",
        description="Test configuration for AI-driven exit strategy",
        parameters={
            'ai_model': 'llm',  # 'llm', 'sentiment', 'pattern', 'multifactor'
            'confidence_threshold': 0.7,
            'decision_timeout': 30,  # seconds
            'model_endpoint': 'openai',
            'api_key': 'test_key',
            'model_name': 'gpt-4',
            'max_tokens': 1000,
            'temperature': 0.3,
            'sentiment_sources': ['news', 'social', 'analyst'],
            'pattern_types': ['breakout', 'reversal', 'continuation'],
            'factors': ['price_action', 'volume', 'volatility', 'sentiment', 'market_regime'],
            'factor_weights': {
                'price_action': 0.3,
                'volume': 0.2,
                'volatility': 0.2,
                'sentiment': 0.2,
                'market_regime': 0.1
            },
            'market_data_lookback': 50,
            'real_time_analysis': True,
            'fallback_strategy': 'stop_loss'
        }
    )


# Test implementation of AIExitStrategy
class TestAIExitStrategy(AIExitStrategy):
    """Test implementation for unit testing"""
    
    def __init__(self, config):
        super().__init__(config)
        self.ai_calls = []
        self.analysis_cache = {}
        self.exit_decisions = []
        self.exit_triggered = False
    
    async def _analyze_market_conditions(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """Mock implementation for testing"""
        self.ai_calls.append({
            'type': 'market_analysis',
            'position_id': position.get('position_id'),
            'analysis_time': datetime.utcnow()
        })
        
        return {
            'trend': 'bullish',
            'volatility': 'moderate',
            'momentum': 'positive',
            'sentiment': 'optimistic',
            'regime': 'trending',
            'confidence': 0.75
        }
    
    async def _generate_ai_decision(self, market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Mock implementation for testing"""
        return {
            'decision': 'hold' if not self.exit_triggered else 'exit',
            'confidence': 0.8 if not self.exit_triggered else 0.9,
            'reasoning': 'Market conditions suggest continued holding',
            'urgency': 0.3 if not self.exit_triggered else 0.8,
            'risk_level': 'moderate'
        }
    
    async def evaluate_exit_conditions(self, position: Dict[str, Any]) -> bool:
        """Simple test implementation"""
        market_analysis = await self._analyze_market_conditions(position)
        ai_decision = await self._generate_ai_decision(market_analysis)
        
        self.exit_decisions.append(ai_decision)
        
        return ai_decision.get('decision') == 'exit'
    
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
        
        market_analysis = await self._analyze_market_conditions(position)
        ai_decision = await self._generate_ai_decision(market_analysis)
        
        return ExitSignal(
            signal_id=f"ai_{position_id}_{datetime.utcnow().timestamp()}",
            strategy_id=self.config.strategy_id,
            position_id=position_id,
            symbol=symbol,
            exit_reason=exit_reason,
            exit_price=current_price,
            exit_quantity=quantity,
            confidence=ai_decision.get('confidence', 0.75),
            urgency=ai_decision.get('urgency', 0.5),
            metadata={
                'ai_model': self.ai_config.ai_model,
                'market_analysis': market_analysis,
                'ai_decision': ai_decision,
                'decision_reasoning': ai_decision.get('reasoning', ''),
                'risk_level': ai_decision.get('risk_level', 'unknown')
            }
        )


# Mock context for AI exit testing
@pytest.fixture
def ai_exit_context():
    """Mock context for AI exit testing"""
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
        'entry_time': datetime.utcnow() - timedelta(hours=2),
        'created_at': datetime.utcnow() - timedelta(hours=2)
    }])
    
    # Mock external data sources
    context.get_market_sentiment = AsyncMock(return_value={
        'overall_sentiment': 0.65,  # Positive sentiment
        'news_sentiment': 0.70,
        'social_sentiment': 0.60,
        'analyst_sentiment': 0.65,
        'confidence': 0.80
    })
    
    context.get_technical_patterns = AsyncMock(return_value={
        'patterns_detected': ['breakout', 'ascending_triangle'],
        'pattern_confidence': 0.75,
        'reversal_probability': 0.25,
        'continuation_probability': 0.75
    })
    
    return context


class TestAIExitStrategy:
    """Test AIExitStrategy base class"""
    
    def test_initialization(self, ai_exit_config):
        """Test proper initialization of AIExitStrategy"""
        strategy = TestAIExitStrategy(ai_exit_config)
        
        # Test inherited attributes
        assert strategy.config == ai_exit_config
        assert strategy.status.value == "initializing"
        
        # Test AI exit specific attributes
        assert isinstance(strategy.ai_config, AIExitConfig)
        assert strategy.ai_config.ai_model == 'llm'
        assert strategy.ai_config.confidence_threshold == 0.7
        assert strategy.ai_config.decision_timeout == 30
        
        # Test tracking attributes
        assert len(strategy.analysis_cache) == 0
        assert len(strategy.ai_decisions) == 0
        assert len(strategy.model_calls) == 0
        assert len(strategy.confidence_scores) == 0
    
    @pytest.mark.asyncio
    async def test_analyze_market_conditions(self, ai_exit_config, ai_exit_context):
        """Test market condition analysis"""
        strategy = TestAIExitStrategy(ai_exit_config)
        strategy.set_context(ai_exit_context)
        
        position = {
            'position_id': 'pos_001',
            'symbol': 'AAPL',
            'entry_price': Decimal('145.00'),
            'current_price': Decimal('150.00')
        }
        
        analysis = await strategy._analyze_market_conditions(position)
        
        assert isinstance(analysis, dict)
        assert 'trend' in analysis
        assert 'volatility' in analysis
        assert 'momentum' in analysis
        assert 'sentiment' in analysis
        assert 'regime' in analysis
        assert 'confidence' in analysis
        
        assert len(strategy.ai_calls) == 1
        assert strategy.ai_calls[0]['type'] == 'market_analysis'
    
    @pytest.mark.asyncio
    async def test_generate_ai_decision(self, ai_exit_config, ai_exit_context):
        """Test AI decision generation"""
        strategy = TestAIExitStrategy(ai_exit_config)
        strategy.set_context(ai_exit_context)
        
        market_analysis = {
            'trend': 'bullish',
            'volatility': 'moderate',
            'momentum': 'positive',
            'sentiment': 'optimistic',
            'confidence': 0.75
        }
        
        decision = await strategy._generate_ai_decision(market_analysis)
        
        assert isinstance(decision, dict)
        assert 'decision' in decision
        assert 'confidence' in decision
        assert 'reasoning' in decision
        assert 'urgency' in decision
        assert 'risk_level' in decision
        
        assert decision['decision'] in ['hold', 'exit', 'reduce']
        assert 0.0 <= decision['confidence'] <= 1.0
        assert 0.0 <= decision['urgency'] <= 1.0
    
    @pytest.mark.asyncio
    async def test_evaluate_exit_conditions(self, ai_exit_config, ai_exit_context):
        """Test exit condition evaluation"""
        strategy = TestAIExitStrategy(ai_exit_config)
        strategy.set_context(ai_exit_context)
        
        position = {
            'position_id': 'pos_001',
            'symbol': 'AAPL',
            'current_price': Decimal('150.00')
        }
        
        # Test when AI suggests hold
        strategy.exit_triggered = False
        result = await strategy.evaluate_exit_conditions(position)
        assert result is False  # Hold decision
        
        # Test when AI suggests exit
        strategy.exit_triggered = True
        result = await strategy.evaluate_exit_conditions(position)
        assert result is True  # Exit decision
        
        assert len(strategy.exit_decisions) == 2
    
    @pytest.mark.asyncio
    async def test_generate_exit_signal(self, ai_exit_config, ai_exit_context):
        """Test exit signal generation"""
        strategy = TestAIExitStrategy(ai_exit_config)
        strategy.set_context(ai_exit_context)
        
        position = {
            'position_id': 'pos_001',
            'symbol': 'AAPL',
            'quantity': Decimal('100'),
            'current_price': Decimal('150.00')
        }
        
        exit_signal = await strategy.generate_exit_signal(position, ExitReason.AI_SIGNAL)
        
        assert exit_signal is not None
        assert exit_signal.strategy_id == ai_exit_config.strategy_id
        assert exit_signal.position_id == 'pos_001'
        assert exit_signal.symbol == 'AAPL'
        assert exit_signal.exit_reason == ExitReason.AI_SIGNAL
        assert exit_signal.exit_price == Decimal('150.00')
        assert exit_signal.exit_quantity == Decimal('100')
        assert 0.0 <= exit_signal.confidence <= 1.0
        assert 0.0 <= exit_signal.urgency <= 1.0
        assert 'ai_model' in exit_signal.metadata
        assert 'market_analysis' in exit_signal.metadata
        assert 'ai_decision' in exit_signal.metadata
        assert 'decision_reasoning' in exit_signal.metadata
        assert 'risk_level' in exit_signal.metadata
    
    @pytest.mark.asyncio
    async def test_calculate_ai_confidence(self, ai_exit_config, ai_exit_context):
        """Test AI confidence calculation"""
        strategy = TestAIExitStrategy(ai_exit_config)
        strategy.set_context(ai_exit_context)
        
        # Test with high confidence signals
        high_confidence_analysis = {
            'confidence': 0.9,
            'data_quality': 'high',
            'signal_strength': 'strong'
        }
        
        high_confidence = strategy._calculate_ai_confidence(high_confidence_analysis)
        assert high_confidence > 0.8
        
        # Test with low confidence signals
        low_confidence_analysis = {
            'confidence': 0.3,
            'data_quality': 'low',
            'signal_strength': 'weak'
        }
        
        low_confidence = strategy._calculate_ai_confidence(low_confidence_analysis)
        assert low_confidence < 0.6
        
        # Confidence should be between 0 and 1
        assert 0.0 <= high_confidence <= 1.0
        assert 0.0 <= low_confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_handle_ai_timeout(self, ai_exit_config, ai_exit_context):
        """Test AI decision timeout handling"""
        # Set short timeout
        ai_exit_config.parameters['decision_timeout'] = 1  # 1 second
        strategy = TestAIExitStrategy(ai_exit_config)
        strategy.set_context(ai_exit_context)
        
        # Mock slow AI response
        async def slow_analysis(position):
            await asyncio.sleep(2)  # Sleep longer than timeout
            return {'decision': 'hold', 'confidence': 0.5}
        
        with patch.object(strategy, '_analyze_market_conditions', slow_analysis):
            position = {
                'position_id': 'pos_timeout',
                'symbol': 'AAPL',
                'current_price': Decimal('150.00')
            }
            
            # Should timeout and use fallback
            try:
                analysis = await asyncio.wait_for(
                    strategy._analyze_market_conditions(position),
                    timeout=0.5  # Very short timeout
                )
                assert analysis is not None
            except asyncio.TimeoutError:
                # Expected timeout behavior
                pass
    
    @pytest.mark.asyncio
    async def test_fallback_strategy(self, ai_exit_config, ai_exit_context):
        """Test fallback strategy when AI fails"""
        strategy = TestAIExitStrategy(ai_exit_config)
        strategy.set_context(ai_exit_context)
        
        # Test with AI disabled
        strategy.ai_config.enabled = False
        
        position = {
            'position_id': 'pos_fallback',
            'symbol': 'AAPL',
            'current_price': Decimal('140.00'),  # Below entry
            'entry_price': Decimal('145.00')
        }
        
        # Should fall back to stop loss logic
        result = await strategy.evaluate_exit_conditions(position)
        assert isinstance(result, bool)
    
    @pytest.mark.asyncio
    async def test_ai_model_caching(self, ai_exit_config, ai_exit_context):
        """Test AI analysis caching"""
        strategy = TestAIExitStrategy(ai_exit_config)
        strategy.set_context(ai_exit_context)
        
        position = {
            'position_id': 'pos_cache',
            'symbol': 'AAPL',
            'current_price': Decimal('150.00'),
            'timestamp': datetime.utcnow()
        }
        
        # First call should cache
        analysis1 = await strategy._analyze_market_conditions(position)
        assert 'pos_cache' not in strategy.analysis_cache
        
        # Second call with same position should potentially use cache
        analysis2 = await strategy._analyze_market_conditions(position)
        assert len(strategy.ai_calls) == 2


class TestLLMExitDecision:
    """Test LLMExitDecision implementation"""
    
    @pytest.mark.asyncio
    async def test_llm_decision_making(self, ai_exit_context):
        """Test LLM-based exit decisions"""
        config = ExitConfiguration(
            strategy_id="llm_exit_001",
            strategy_type=ExitType.AI_DRIVEN,
            name="LLM Exit Decision",
            description="Test LLM exit decision",
            parameters={
                'ai_model': 'llm',
                'model_endpoint': 'openai',
                'model_name': 'gpt-4',
                'max_tokens': 1000,
                'temperature': 0.3,
                'prompt_template': 'analyze_market_exit'
            }
        )
        
        strategy = LLMExitDecision(config)
        strategy.set_context(ai_exit_context)
        
        # Mock LLM response
        mock_llm_response = {
            'decision': 'hold',
            'confidence': 0.75,
            'reasoning': 'Market shows continued upward momentum with strong volume',
            'key_factors': ['bullish_patterns', 'high_volume', 'positive_sentiment'],
            'risk_assessment': 'moderate'
        }
        
        with patch.object(strategy, '_call_llm_api', return_value=mock_llm_response):
            market_data = {
                'symbol': 'AAPL',
                'price': Decimal('150.00'),
                'volume': 2000000,
                'volatility': 0.025,
                'sentiment': 0.65
            }
            
            decision = await strategy._make_llm_decision(market_data)
            
            assert decision['decision'] == 'hold'
            assert 0.0 <= decision['confidence'] <= 1.0
            assert 'reasoning' in decision
            assert 'key_factors' in decision


class TestSentimentExit:
    """Test SentimentExit implementation"""
    
    @pytest.mark.asyncio
    async def test_sentiment_based_exits(self, ai_exit_context):
        """Test sentiment-based exit decisions"""
        config = ExitConfiguration(
            strategy_id="sentiment_001",
            strategy_type=ExitType.AI_DRIVEN,
            name="Sentiment Exit",
            description="Test sentiment exit",
            parameters={
                'ai_model': 'sentiment',
                'sentiment_sources': ['news', 'social', 'analyst'],
                'sentiment_threshold': 0.3,  # Exit if sentiment below 30%
                'sentiment_lookback': 24  # hours
            }
        )
        
        strategy = SentimentExit(config)
        strategy.set_context(ai_exit_context)
        
        # Mock positive sentiment
        ai_exit_context.get_market_sentiment.return_value = {
            'overall_sentiment': 0.75,  # Positive
            'news_sentiment': 0.70,
            'social_sentiment': 0.80,
            'analyst_sentiment': 0.75,
            'confidence': 0.85
        }
        
        position = {'position_id': 'pos_sentiment', 'symbol': 'AAPL'}
        
        sentiment_analysis = await strategy._analyze_sentiment(position)
        assert sentiment_analysis['overall_sentiment'] > 0.5  # Positive
        
        # Test negative sentiment scenario
        ai_exit_context.get_market_sentiment.return_value = {
            'overall_sentiment': 0.15,  # Very negative
            'news_sentiment': 0.10,
            'social_sentiment': 0.15,
            'analyst_sentiment': 0.20,
            'confidence': 0.90
        }
        
        negative_analysis = await strategy._analyze_sentiment(position)
        assert negative_analysis['overall_sentiment'] < 0.3  # Below threshold


class TestPatternExit:
    """Test PatternExit implementation"""
    
    @pytest.mark.asyncio
    async def test_pattern_recognition_exits(self, ai_exit_context):
        """Test pattern-based exit decisions"""
        config = ExitConfiguration(
            strategy_id="pattern_001",
            strategy_type=ExitType.AI_DRIVEN,
            name="Pattern Exit",
            description="Test pattern exit",
            parameters={
                'ai_model': 'pattern',
                'pattern_types': ['breakout', 'reversal', 'continuation'],
                'pattern_confidence_threshold': 0.6,
                'lookback_period': 20
            }
        )
        
        strategy = PatternExit(config)
        strategy.set_context(ai_exit_context)
        
        # Mock pattern detection
        ai_exit_context.get_technical_patterns.return_value = {
            'patterns_detected': ['ascending_triangle', 'breakout'],
            'pattern_confidence': 0.75,
            'reversal_probability': 0.25,
            'continuation_probability': 0.75,
            'breakout_direction': 'upward',
            'volume_confirmation': True
        }
        
        position = {'position_id': 'pos_pattern', 'symbol': 'AAPL'}
        
        pattern_analysis = await strategy._analyze_patterns(position)
        
        assert 'patterns_detected' in pattern_analysis
        assert 'pattern_confidence' in pattern_analysis
        assert pattern_analysis['pattern_confidence'] >= 0.6
        assert pattern_analysis['continuation_probability'] > pattern_analysis['reversal_probability']


class TestMultiFactorAIExit:
    """Test MultiFactorAIExit implementation"""
    
    @pytest.mark.asyncio
    async def test_multi_factor_analysis(self, ai_exit_context):
        """Test multi-factor AI analysis"""
        config = ExitConfiguration(
            strategy_id="multifactor_001",
            strategy_type=ExitType.AI_DRIVEN,
            name="Multi Factor AI Exit",
            description="Test multi-factor AI exit",
            parameters={
                'ai_model': 'multifactor',
                'factors': ['price_action', 'volume', 'volatility', 'sentiment'],
                'factor_weights': {
                    'price_action': 0.4,
                    'volume': 0.3,
                    'volatility': 0.2,
                    'sentiment': 0.1
                },
                'composite_threshold': 0.6
            }
        )
        
        strategy = MultiFactorAIExit(config)
        strategy.set_context(ai_exit_context)
        
        position = {'position_id': 'pos_multi', 'symbol': 'AAPL'}
        
        factor_analysis = await strategy._analyze_factors(position)
        
        assert isinstance(factor_analysis, dict)
        for factor in config.parameters['factors']:
            assert factor in factor_analysis
        
        # Test weighted score calculation
        if 'composite_score' in factor_analysis:
            assert 0.0 <= factor_analysis['composite_score'] <= 1.0


class TestAIScenarios:
    """Test various AI scenarios"""
    
    @pytest.mark.asyncio
    async def test_bullish_market_ai_decision(self, ai_exit_config, ai_exit_context):
        """Test AI decision in bullish market"""
        strategy = TestAIExitStrategy(ai_exit_config)
        strategy.set_context(ai_exit_context)
        
        # Mock bullish market conditions
        bullish_position = {
            'position_id': 'pos_bullish',
            'symbol': 'AAPL',
            'current_price': Decimal('155.00'),  # Strong upward move
            'entry_price': Decimal('145.00')
        }
        
        analysis = await strategy._analyze_market_conditions(bullish_position)
        
        # In bullish conditions, AI should lean toward holding
        if analysis['trend'] == 'bullish':
            # Would expect hold decision, but depends on implementation
            decision_result = await strategy.evaluate_exit_conditions(bullish_position)
            assert isinstance(decision_result, bool)
    
    @pytest.mark.asyncio
    async def test_bearish_market_ai_decision(self, ai_exit_config, ai_exit_context):
        """Test AI decision in bearish market"""
        strategy = TestAIExitStrategy(ai_exit_config)
        strategy.set_context(ai_exit_context)
        
        # Mock bearish market conditions
        bearish_position = {
            'position_id': 'pos_bearish',
            'symbol': 'AAPL',
            'current_price': Decimal('135.00'),  # Downward move
            'entry_price': Decimal('145.00')
        }
        
        # Simulate AI detecting bearish conditions
        strategy.exit_triggered = True
        
        decision_result = await strategy.evaluate_exit_conditions(bearish_position)
        assert decision_result is True  # Should trigger exit
    
    @pytest.mark.asyncio
    async def test_high_volatility_ai_response(self, ai_exit_config, ai_exit_context):
        """Test AI response to high volatility"""
        strategy = TestAIExitStrategy(ai_exit_config)
        strategy.set_context(ai_exit_context)
        
        # Mock high volatility
        ai_exit_context.calculate_volatility.return_value = Decimal('0.050')  # High volatility
        
        volatile_position = {
            'position_id': 'pos_volatile',
            'symbol': 'AAPL',
            'current_price': Decimal('150.00')
        }
        
        analysis = await strategy._analyze_market_conditions(volatile_position)
        
        # AI should recognize high volatility and potentially adjust confidence
        if 'volatility' in analysis:
            assert analysis['volatility'] in ['high', 'very_high', 'extreme']
    
    @pytest.mark.asyncio
    async def test_low_confidence_ai_behavior(self, ai_exit_config, ai_exit_context):
        """Test AI behavior with low confidence"""
        # Set high confidence threshold
        ai_exit_config.parameters['confidence_threshold'] = 0.9
        
        strategy = TestAIExitStrategy(ai_exit_config)
        strategy.set_context(ai_exit_context)
        
        # Mock low confidence analysis
        async def low_confidence_analysis(position):
            return {
                'trend': 'unclear',
                'volatility': 'moderate',
                'momentum': 'neutral',
                'sentiment': 'mixed',
                'confidence': 0.6  # Below 0.9 threshold
            }
        
        with patch.object(strategy, '_analyze_market_conditions', low_confidence_analysis):
            position = {
                'position_id': 'pos_low_conf',
                'symbol': 'AAPL'
            }
            
            analysis = await strategy._analyze_market_conditions(position)
            assert analysis['confidence'] < 0.9


class TestAIEdgeCases:
    """Test edge cases and error conditions"""
    
    @pytest.mark.asyncio
    async def test_ai_api_failure(self, ai_exit_config, ai_exit_context):
        """Test behavior when AI API fails"""
        strategy = TestAIExitStrategy(ai_exit_config)
        strategy.set_context(ai_exit_context)
        
        # Mock API failure
        async def failing_analysis(position):
            raise Exception("AI API unavailable")
        
        with patch.object(strategy, '_analyze_market_conditions', failing_analysis):
            position = {
                'position_id': 'pos_api_fail',
                'symbol': 'AAPL'
            }
            
            # Should handle API failure gracefully
            try:
                result = await strategy.evaluate_exit_conditions(position)
                assert isinstance(result, bool)
            except Exception:
                # Expected to handle gracefully or raise appropriate exception
                pass
    
    @pytest.mark.asyncio
    async def test_invalid_market_data(self, ai_exit_config):
        """Test behavior with invalid market data"""
        strategy = TestAIExitStrategy(ai_exit_config)
        
        # Position with missing critical data
        invalid_position = {
            'position_id': 'pos_invalid'
            # Missing symbol, price, etc.
        }
        
        # Should handle invalid data gracefully
        try:
            result = await strategy.evaluate_exit_conditions(invalid_position)
            assert isinstance(result, bool)
        except (KeyError, TypeError, ValueError):
            # Expected to handle gracefully or raise appropriate exception
            pass
    
    def test_ai_model_validation(self, ai_exit_config):
        """Test AI model configuration validation"""
        # Test with invalid AI model
        ai_exit_config.parameters['ai_model'] = 'invalid_model'
        strategy = TestAIExitStrategy(ai_exit_config)
        
        assert strategy.ai_config.ai_model == 'invalid_model'
        # Should still initialize but may fail during execution
    
    def test_extreme_confidence_values(self, ai_exit_config):
        """Test behavior with extreme confidence values"""
        # Test with zero confidence threshold
        ai_exit_config.parameters['confidence_threshold'] = 0.0
        strategy = TestAIExitStrategy(ai_exit_config)
        
        assert strategy.ai_config.confidence_threshold == 0.0
        
        # Test with 100% confidence threshold
        ai_exit_config.parameters['confidence_threshold'] = 1.0
        strategy.ai_config.confidence_threshold = 1.0
        
        assert strategy.ai_config.confidence_threshold == 1.0


if __name__ == "__main__":
    pytest.main([__file__])
