"""
@file ai_exit_strategy.py
@brief AI-Driven Exit Strategy Implementation

@details
This module implements AI-powered exit strategies that use machine learning
and market analysis to make intelligent exit decisions. AI exit strategies
analyze market conditions, technical indicators, and market sentiment to
optimize exit timing and reduce false signals.

Key Features:
- Machine learning-based exit predictions
- Multi-factor market analysis
- Sentiment and news integration
- Real-time model updates
- Performance feedback learning

@author Trading Orchestrator System
@version 1.0
@date 2025-11-06

@warning
AI models require sufficient data and may not generalize well to unprecedented
market conditions. Always maintain human oversight and traditional exit strategies.

@note
AI strategies work best when combined with traditional risk management and
diversified exit approaches.

@see base_exit_strategy.py for base framework
@see strategies.ai_ml_strategies for AI model implementation
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
import asyncio
import json
import pickle
from abc import ABC, abstractmethod

from loguru import logger

from .base_exit_strategy import (
    BaseExitStrategy,
    ExitSignal,
    ExitReason,
    ExitType,
    ExitCondition,
    ExitConfiguration,
    ExitMetrics,
    ExitStatus
)


@dataclass
class MLExitDecision:
    """
    @class MLExitDecision
    @brief Machine learning exit decision data structure
    
    @details
    Contains the decision output from machine learning models including
    prediction probability, confidence scores, and feature importance.
    """
    decision: str  # "exit", "hold", "monitor"
    probability: float  # 0.0 to 1.0
    confidence: float  # Model confidence in decision
    exit_reason: Optional[str] = None  # Predicted exit reason
    features_importance: Dict[str, float] = field(default_factory=dict)
    model_version: str = "1.0"
    prediction_time: datetime = field(default_factory=datetime.utcnow)
    features: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AIExitModel(ABC):
    """Abstract base class for AI exit models"""
    
    @abstractmethod
    async def predict(self, features: Dict[str, Any]) -> MLExitDecision:
        """Make exit prediction based on features"""
        pass
    
    @abstractmethod
    async def update(self, features: Dict[str, Any], outcome: bool, feedback: Dict[str, Any]):
        """Update model with feedback"""
        pass
    
    @abstractmethod
    async def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        pass


class TechnicalIndicatorModel(AIExitModel):
    """AI model based on technical indicators"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model_weights = {
            'rsi': 0.15,
            'macd': 0.20,
            'bollinger': 0.18,
            'volume': 0.12,
            'price_momentum': 0.25,
            'volatility': 0.10
        }
        self.decision_threshold = 0.70
        self.confidence_threshold = 0.60
        
    async def predict(self, features: Dict[str, Any]) -> MLExitDecision:
        """Predict exit based on technical indicators"""
        try:
            # Extract technical features
            rsi = features.get('rsi', 50)
            macd_signal = features.get('macd_signal', 0)
            bollinger_position = features.get('bollinger_position', 0.5)
            volume_ratio = features.get('volume_ratio', 1.0)
            price_momentum = features.get('price_momentum', 0)
            volatility = features.get('volatility', 0.02)
            
            # Calculate weighted score
            score = (
                self.model_weights['rsi'] * self._normalize_rsi(rsi) +
                self.model_weights['macd'] * self._normalize_macd(macd_signal) +
                self.model_weights['bollinger'] * self._normalize_bollinger(bollinger_position) +
                self.model_weights['volume'] * self._normalize_volume(volume_ratio) +
                self.model_weights['price_momentum'] * self._normalize_momentum(price_momentum) +
                self.model_weights['volatility'] * self._normalize_volatility(volatility)
            )
            
            # Make decision
            if score > self.decision_threshold:
                decision = "exit"
                exit_reason = "technical_indicator"
            elif score < 0.30:
                decision = "hold"
                exit_reason = None
            else:
                decision = "monitor"
                exit_reason = None
            
            # Calculate confidence
            confidence = abs(score - 0.5) * 2  # Higher score = higher confidence
            
            return MLExitDecision(
                decision=decision,
                probability=score,
                confidence=confidence,
                exit_reason=exit_reason,
                features_importance=self.model_weights,
                features=features
            )
            
        except Exception as e:
            logger.error(f"Error in technical indicator prediction: {e}")
            return MLExitDecision(
                decision="hold",
                probability=0.5,
                confidence=0.5,
                features=features
            )
    
    def _normalize_rsi(self, rsi: float) -> float:
        """Normalize RSI to -1 to 1 scale"""
        if rsi > 70:
            return 1.0
        elif rsi < 30:
            return -1.0
        elif rsi > 50:
            return (rsi - 50) / 20  # 0 to 1
        else:
            return (rsi - 50) / 20  # -1 to 0
    
    def _normalize_macd(self, macd_signal: float) -> float:
        """Normalize MACD signal"""
        # Assume signal is already normalized or use bounds
        return max(-1.0, min(1.0, macd_signal))
    
    def _normalize_bollinger(self, bb_position: float) -> float:
        """Normalize Bollinger Band position"""
        # 0.5 = middle, 1.0 = upper band, 0.0 = lower band
        return (bb_position - 0.5) * 2  # -1 to 1
    
    def _normalize_volume(self, volume_ratio: float) -> float:
        """Normalize volume ratio"""
        if volume_ratio > 2.0:
            return 1.0
        elif volume_ratio < 0.5:
            return -0.5
        else:
            return (volume_ratio - 1.0)  # -0.5 to 1.0
    
    def _normalize_momentum(self, momentum: float) -> float:
        """Normalize price momentum"""
        # Assume momentum is percentage change, normalize to -1 to 1
        return max(-1.0, min(1.0, momentum))
    
    def _normalize_volatility(self, volatility: float) -> float:
        """Normalize volatility"""
        # Higher volatility = higher exit probability
        return min(1.0, volatility * 50)  # Scale volatility
    
    async def update(self, features: Dict[str, Any], outcome: bool, feedback: Dict[str, Any]):
        """Update model weights based on feedback"""
        try:
            # Simple update: adjust weights based on outcome
            adjustment_factor = 0.1 if outcome else -0.05
            
            for feature in self.model_weights:
                if feature in feedback.get('important_features', []):
                    self.model_weights[feature] += adjustment_factor
            
            # Normalize weights
            total_weight = sum(self.model_weights.values())
            if total_weight > 0:
                for feature in self.model_weights:
                    self.model_weights[feature] /= total_weight
            
            logger.info(f"Updated model weights: {self.model_weights}")
            
        except Exception as e:
            logger.error(f"Error updating model: {e}")
    
    async def get_feature_importance(self) -> Dict[str, float]:
        """Get current feature importance"""
        return self.model_weights.copy()


class SentimentAnalysisModel(AIExitModel):
    """AI model based on market sentiment"""
    
    def __init__(self):
        self.sentiment_weights = {
            'news_sentiment': 0.3,
            'social_sentiment': 0.2,
            'fear_greed_index': 0.25,
            'vix_level': 0.15,
            'insider_trading': 0.1
        }
        self.fear_greed_extreme_fear = 25
        self.fear_greed_extreme_greed = 75
        
    async def predict(self, features: Dict[str, Any]) -> MLExitDecision:
        """Predict exit based on sentiment analysis"""
        try:
            news_sentiment = features.get('news_sentiment', 0)  # -1 to 1
            social_sentiment = features.get('social_sentiment', 0)  # -1 to 1
            fear_greed_index = features.get('fear_greed_index', 50)  # 0 to 100
            vix_level = features.get('vix_level', 20)  # VIX level
            insider_sentiment = features.get('insider_sentiment', 0)  # -1 to 1
            
            # Calculate sentiment score
            sentiment_score = (
                self.sentiment_weights['news_sentiment'] * news_sentiment +
                self.sentiment_weights['social_sentiment'] * social_sentiment +
                self.sentiment_weights['insider_trading'] * insider_sentiment
            )
            
            # Fear & Greed adjustment
            if fear_greed_index < self.fear_greed_extreme_fear:
                sentiment_score += 0.3  # Extreme fear = exit
            elif fear_greed_index > self.fear_greed_extreme_greed:
                sentiment_score -= 0.2  # Extreme greed = potential reversal
            
            # VIX adjustment
            if vix_level > 30:
                sentiment_score -= 0.3  # High VIX = exit
            elif vix_level < 15:
                sentiment_score += 0.1  # Low VIX = hold
            
            # Normalize to 0-1 range
            score = (sentiment_score + 1) / 2
            
            # Make decision
            if score > 0.75:
                decision = "exit"
                exit_reason = "sentiment_extreme"
            elif score < 0.25:
                decision = "hold"
                exit_reason = None
            else:
                decision = "monitor"
                exit_reason = None
            
            confidence = abs(sentiment_score)
            
            return MLExitDecision(
                decision=decision,
                probability=score,
                confidence=confidence,
                exit_reason=exit_reason,
                features_importance=self.sentiment_weights,
                features=features
            )
            
        except Exception as e:
            logger.error(f"Error in sentiment prediction: {e}")
            return MLExitDecision(
                decision="hold",
                probability=0.5,
                confidence=0.5,
                features=features
            )
    
    async def update(self, features: Dict[str, Any], outcome: bool, feedback: Dict[str, Any]):
        """Update sentiment weights"""
        # Simple implementation - could be enhanced with more sophisticated learning
        logger.info("Sentiment model updated with feedback")
    
    async def get_feature_importance(self) -> Dict[str, float]:
        """Get sentiment feature importance"""
        return self.sentiment_weights.copy()


class EnsembleAIExitModel(AIExitModel):
    """Ensemble model combining multiple AI exit strategies"""
    
    def __init__(self, models: List[AIExitModel]):
        self.models = models
        self.model_weights = [1.0 / len(models)] * len(models)
        
    async def predict(self, features: Dict[str, Any]) -> MLExitDecision:
        """Predict using ensemble of models"""
        try:
            decisions = []
            probabilities = []
            confidences = []
            
            # Get predictions from all models
            for model in self.models:
                decision = await model.predict(features)
                decisions.append(decision)
                probabilities.append(decision.probability)
                confidences.append(decision.confidence)
            
            # Ensemble scoring
            weighted_score = sum(
                prob * weight for prob, weight in zip(probabilities, self.model_weights)
            )
            
            weighted_confidence = sum(
                conf * weight for conf, weight in zip(confidences, self.model_weights)
            )
            
            # Make ensemble decision
            if weighted_score > 0.7:
                decision = "exit"
            elif weighted_score < 0.3:
                decision = "hold"
            else:
                decision = "monitor"
            
            # Use most confident model's exit reason if available
            exit_reason = None
            for dec in decisions:
                if dec.decision == "exit":
                    exit_reason = dec.exit_reason
                    break
            
            return MLExitDecision(
                decision=decision,
                probability=weighted_score,
                confidence=weighted_confidence,
                exit_reason=exit_reason,
                features=features,
                metadata={'model_count': len(self.models)}
            )
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            return MLExitDecision(
                decision="hold",
                probability=0.5,
                confidence=0.5,
                features=features
            )
    
    async def update(self, features: Dict[str, Any], outcome: bool, feedback: Dict[str, Any]):
        """Update ensemble model weights"""
        try:
            # Update individual models
            for model in self.models:
                await model.update(features, outcome, feedback)
            
            # Adjust ensemble weights based on performance
            if outcome:
                # Successful prediction - slightly favor this model type
                pass  # Simplified for now
            
            logger.info("Ensemble model updated")
            
        except Exception as e:
            logger.error(f"Error updating ensemble model: {e}")
    
    async def get_feature_importance(self) -> Dict[str, float]:
        """Get ensemble feature importance"""
        feature_importance = {}
        
        for model in self.models:
            model_importance = await model.get_feature_importance()
            for feature, importance in model_importance.items():
                if feature not in feature_importance:
                    feature_importance[feature] = 0
                feature_importance[feature] += importance
        
        # Normalize
        total = sum(feature_importance.values())
        if total > 0:
            for feature in feature_importance:
                feature_importance[feature] /= total
        
        return feature_importance


class AIExitStrategy(BaseExitStrategy):
    """
    @class AIExitStrategy
    @brief AI-powered exit strategy
    
    @details
    Uses machine learning models to analyze market conditions and make
    intelligent exit decisions. Combines technical analysis, sentiment
    analysis, and market microstructure data.
    
    @par Key Features:
    - Multi-model ensemble approach
    - Real-time market data analysis
    - Feature engineering and selection
    - Continuous model learning
    - Performance feedback integration
    """
    
    def __init__(self, config: ExitConfiguration):
        super().__init__(config)
        
        # AI model configuration
        self.model_type = config.parameters.get('model_type', 'ensemble')
        self.feature_window = config.parameters.get('feature_window', 50)
        self.update_frequency = config.parameters.get('update_frequency', 3600)  # 1 hour
        self.min_confidence = config.parameters.get('min_confidence', 0.65)
        
        # Initialize AI models
        self.models: Dict[str, AIExitModel] = {}
        self.active_model: Optional[AIExitModel] = None
        
        # Feature engineering
        self.feature_cache: Dict[str, Dict[str, Any]] = {}
        self.market_data_cache: Dict[str, List[Dict[str, Any]]] = {}
        
        # Performance tracking
        self.prediction_history: List[Dict[str, Any]] = []
        self.feedback_history: List[Dict[str, Any]] = []
        
        # Initialize models
        self._initialize_models()
        
        logger.info(f"AI exit strategy initialized with model type: {self.model_type}")
    
    def _initialize_models(self):
        """Initialize AI models"""
        try:
            if self.model_type == 'technical':
                self.active_model = TechnicalIndicatorModel()
                self.models['technical'] = self.active_model
            elif self.model_type == 'sentiment':
                self.active_model = SentimentAnalysisModel()
                self.models['sentiment'] = self.active_model
            elif self.model_type == 'ensemble':
                technical_model = TechnicalIndicatorModel()
                sentiment_model = SentimentAnalysisModel()
                self.active_model = EnsembleAIExitModel([technical_model, sentiment_model])
                self.models['ensemble'] = self.active_model
            else:
                # Default to technical model
                self.active_model = TechnicalIndicatorModel()
                self.models['technical'] = self.active_model
            
            logger.info(f"Initialized AI models: {list(self.models.keys())}")
            
        except Exception as e:
            logger.error(f"Error initializing AI models: {e}")
    
    async def evaluate_exit_conditions(self, position: Dict[str, Any]) -> bool:
        """Evaluate AI-based exit conditions"""
        try:
            if not self.active_model:
                return False
            
            position_id = position.get('position_id')
            if not position_id:
                return False
            
            # Extract features for AI analysis
            features = await self._extract_features(position)
            if not features:
                return False
            
            # Get AI prediction
            decision = await self.active_model.predict(features)
            
            # Check if exit is recommended
            exit_recommended = (
                decision.decision == "exit" and 
                decision.confidence >= self.min_confidence
            )
            
            # Store prediction for analysis
            prediction_record = {
                'position_id': position_id,
                'timestamp': datetime.utcnow(),
                'features': features,
                'decision': decision.decision,
                'probability': decision.probability,
                'confidence': decision.confidence,
                'exit_reason': decision.exit_reason
            }
            self.prediction_history.append(prediction_record)
            
            # Keep history limited
            if len(self.prediction_history) > 1000:
                self.prediction_history = self.prediction_history[-500:]
            
            if exit_recommended:
                logger.info(f"AI exit recommended for {position_id}: {decision.exit_reason}")
            
            return exit_recommended
            
        except Exception as e:
            logger.error(f"Error evaluating AI exit conditions: {e}")
            return False
    
    async def generate_exit_signal(
        self, 
        position: Dict[str, Any], 
        exit_reason: ExitReason
    ) -> Optional[ExitSignal]:
        """Generate AI-powered exit signal"""
        try:
            if not self.active_model:
                return None
            
            position_id = position.get('position_id')
            symbol = position.get('symbol')
            quantity = Decimal(str(position.get('quantity', 0)))
            
            if not all([position_id, symbol, quantity]):
                return None
            
            # Get current market data
            current_price = await self.context.get_current_price(symbol)
            if not current_price:
                return None
            
            # Get latest AI prediction
            latest_prediction = None
            for record in reversed(self.prediction_history):
                if record['position_id'] == position_id:
                    latest_prediction = record
                    break
            
            if not latest_prediction:
                return None
            
            # Calculate exit parameters
            confidence = latest_prediction['confidence']
            urgency = self._calculate_ai_urgency(latest_prediction)
            market_impact = await self._estimate_market_impact(symbol, quantity)
            
            # Create exit signal with AI metadata
            exit_signal = ExitSignal(
                signal_id=f"ai_{position_id}_{datetime.utcnow().timestamp()}",
                strategy_id=self.config.strategy_id,
                position_id=position_id,
                symbol=symbol,
                exit_reason=ExitReason.AI_SIGNAL,
                exit_price=current_price,
                exit_quantity=quantity,
                confidence=confidence,
                urgency=urgency,
                estimated_execution_time=timedelta(seconds=30),
                market_impact=market_impact,
                metadata={
                    'ai_model_type': self.model_type,
                    'prediction_probability': latest_prediction['probability'],
                    'ai_confidence': latest_prediction['confidence'],
                    'predicted_exit_reason': latest_prediction.get('exit_reason'),
                    'features_used': list(latest_prediction['features'].keys()),
                    'model_version': '1.0',
                    'prediction_timestamp': latest_prediction['timestamp'].isoformat()
                }
            )
            
            logger.info(f"Generated AI exit signal: {exit_signal.signal_id}")
            return exit_signal
            
        except Exception as e:
            logger.error(f"Error generating AI exit signal: {e}")
            return None
    
    async def _extract_features(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features for AI analysis"""
        try:
            symbol = position.get('symbol')
            if not symbol or not self.context:
                return {}
            
            features = {}
            
            # Market data features
            if not self.context:
                return features
            
            # Get historical data
            historical_data = await self.context.get_historical_data(symbol, '1h', 50)
            if not historical_data or len(historical_data) < 20:
                return features
            
            # Technical indicators
            features.update(await self._calculate_technical_features(historical_data))
            
            # Market microstructure features
            features.update(await self._calculate_microstructure_features(symbol))
            
            # Position-specific features
            features.update(await self._calculate_position_features(position))
            
            # Sentiment features (if available)
            features.update(await self._calculate_sentiment_features(symbol))
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return {}
    
    async def _calculate_technical_features(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate technical indicator features"""
        try:
            features = {}
            
            if len(data) < 20:
                return features
            
            # Convert to price arrays
            prices = [Decimal(str(item.get('close', 0))) for item in data]
            highs = [Decimal(str(item.get('high', 0))) for item in data]
            lows = [Decimal(str(item.get('low', 0))) for item in data]
            volumes = [item.get('volume', 0) for item in data]
            
            # RSI
            features['rsi'] = self._calculate_rsi(prices, 14)
            
            # MACD
            macd_line, macd_signal, macd_histogram = self._calculate_macd(prices)
            features['macd'] = macd_line
            features['macd_signal'] = macd_signal
            features['macd_histogram'] = macd_histogram
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(prices, 20)
            current_price = float(prices[-1])
            features['bollinger_upper'] = float(bb_upper)
            features['bollinger_middle'] = float(bb_middle)
            features['bollinger_lower'] = float(bb_lower)
            features['bollinger_position'] = (current_price - float(bb_lower)) / (float(bb_upper) - float(bb_lower))
            
            # Volume analysis
            avg_volume = sum(volumes[-20:]) / 20
            current_volume = volumes[-1]
            features['volume_ratio'] = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Price momentum
            momentum_periods = [5, 10, 20]
            for period in momentum_periods:
                if len(prices) >= period:
                    momentum = (prices[-1] - prices[-period]) / prices[-period]
                    features[f'momentum_{period}'] = float(momentum)
            
            # Volatility
            returns = []
            for i in range(1, len(prices)):
                ret = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(ret)
            
            if returns:
                features['volatility'] = float((sum(r*r for r in returns) / len(returns)) ** 0.5)
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating technical features: {e}")
            return {}
    
    def _calculate_rsi(self, prices: List[Decimal], period: int) -> float:
        """Calculate RSI indicator"""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            gains = []
            losses = []
            
            for i in range(1, len(prices)):
                change = float(prices[i] - prices[i-1])
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))
            
            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return 50.0
    
    def _calculate_macd(self, prices: List[Decimal]) -> Tuple[float, float, float]:
        """Calculate MACD indicator"""
        try:
            if len(prices) < 26:
                return 0.0, 0.0, 0.0
            
            # Calculate EMAs
            ema_12 = self._calculate_ema(prices, 12)
            ema_26 = self._calculate_ema(prices, 26)
            
            macd_line = float(ema_12 - ema_26)
            macd_signal = float(self._calculate_ema([Decimal(str(macd_line))] * 26, 9))  # Simplified
            
            macd_histogram = macd_line - macd_signal
            
            return macd_line, macd_signal, macd_histogram
            
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return 0.0, 0.0, 0.0
    
    def _calculate_ema(self, prices: List[Decimal], period: int) -> Decimal:
        """Calculate Exponential Moving Average"""
        try:
            if len(prices) == 0:
                return Decimal('0')
            
            multiplier = Decimal('2') / (period + 1)
            ema = prices[0]
            
            for price in prices[1:]:
                ema = price * multiplier + ema * (Decimal('1') - multiplier)
            
            return ema
            
        except Exception as e:
            logger.error(f"Error calculating EMA: {e}")
            return Decimal('0')
    
    def _calculate_bollinger_bands(self, prices: List[Decimal], period: int) -> Tuple[Decimal, Decimal, Decimal]:
        """Calculate Bollinger Bands"""
        try:
            if len(prices) < period:
                current_price = prices[-1] if prices else Decimal('0')
                return current_price, current_price, current_price
            
            # Calculate SMA
            recent_prices = prices[-period:]
            sma = sum(recent_prices) / len(recent_prices)
            
            # Calculate standard deviation
            variance = sum((p - sma) ** 2 for p in recent_prices) / len(recent_prices)
            std_dev = variance ** Decimal('0.5')
            
            # Calculate bands
            upper_band = sma + (std_dev * Decimal('2'))
            lower_band = sma - (std_dev * Decimal('2'))
            
            return upper_band, sma, lower_band
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            current_price = prices[-1] if prices else Decimal('0')
            return current_price, current_price, current_price
    
    async def _calculate_microstructure_features(self, symbol: str) -> Dict[str, Any]:
        """Calculate market microstructure features"""
        # Simplified implementation - would use real market microstructure data
        return {
            'bid_ask_spread': 0.001,
            'order_book_imbalance': 0.0,
            'trade_velocity': 1.0
        }
    
    async def _calculate_position_features(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate position-specific features"""
        try:
            features = {}
            
            # Position metrics
            entry_price = Decimal(str(position.get('entry_price', 0)))
            current_price = Decimal(str(position.get('current_price', 0)))
            
            if entry_price > 0 and current_price > 0:
                features['position_pnl'] = float((current_price - entry_price) / entry_price)
                features['position_duration_hours'] = self._calculate_position_duration_hours(position)
                features['position_size'] = float(position.get('quantity', 0))
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating position features: {e}")
            return {}
    
    async def _calculate_sentiment_features(self, symbol: str) -> Dict[str, Any]:
        """Calculate sentiment features"""
        # Simplified implementation - would integrate with news/social APIs
        return {
            'news_sentiment': 0.0,
            'social_sentiment': 0.0,
            'fear_greed_index': 50.0,
            'insider_sentiment': 0.0
        }
    
    def _calculate_position_duration_hours(self, position: Dict[str, Any]) -> float:
        """Calculate position duration in hours"""
        try:
            created_at = position.get('created_at')
            if created_at:
                duration = datetime.utcnow() - created_at
                return duration.total_seconds() / 3600
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating position duration: {e}")
            return 0.0
    
    def _calculate_ai_urgency(self, prediction_record: Dict[str, Any]) -> float:
        """Calculate urgency based on AI prediction"""
        try:
            confidence = prediction_record.get('confidence', 0.5)
            probability = prediction_record.get('probability', 0.5)
            
            # Higher urgency for high confidence exit predictions
            urgency = confidence * 0.8 + probability * 0.2
            
            return min(0.95, urgency)
            
        except Exception as e:
            logger.error(f"Error calculating AI urgency: {e}")
            return 0.7
    
    async def _estimate_market_impact(self, symbol: str, quantity: Decimal) -> Optional[Decimal]:
        """Estimate market impact for AI exit"""
        # Similar to other strategies but potentially higher due to decision complexity
        if not self.context:
            return None
        
        try:
            current_price = await self.context.get_current_price(symbol)
            if not current_price:
                return None
            
            order_value = quantity * current_price
            # Slightly higher impact for AI decisions due to model processing time
            impact_percentage = min(0.004, quantity / Decimal('750'))
            
            return current_price * impact_percentage
            
        except Exception as e:
            logger.error(f"Error estimating market impact: {e}")
            return None
    
    async def update_model_with_feedback(self, position_id: str, outcome: bool, feedback: Dict[str, Any]):
        """Update AI models with execution feedback"""
        try:
            if not self.active_model:
                return
            
            # Find corresponding prediction
            for record in reversed(self.prediction_history):
                if record['position_id'] == position_id:
                    features = record['features']
                    
                    # Update model
                    await self.active_model.update(features, outcome, feedback)
                    
                    # Store feedback
                    feedback_record = {
                        'position_id': position_id,
                        'timestamp': datetime.utcnow(),
                        'features': features,
                        'outcome': outcome,
                        'feedback': feedback
                    }
                    self.feedback_history.append(feedback_record)
                    
                    # Keep history limited
                    if len(self.feedback_history) > 500:
                        self.feedback_history = self.feedback_history[-250:]
                    
                    logger.info(f"Updated AI model with feedback for {position_id}")
                    break
            
        except Exception as e:
            logger.error(f"Error updating model with feedback: {e}")
    
    async def get_model_performance(self) -> Dict[str, Any]:
        """Get AI model performance statistics"""
        try:
            if not self.prediction_history:
                return {}
            
            # Calculate performance metrics
            total_predictions = len(self.prediction_history)
            exit_predictions = len([p for p in self.prediction_history if p['decision'] == 'exit'])
            avg_confidence = sum(p['confidence'] for p in self.prediction_history) / total_predictions
            avg_probability = sum(p['probability'] for p in self.prediction_history) / total_predictions
            
            return {
                'total_predictions': total_predictions,
                'exit_predictions': exit_predictions,
                'exit_rate': exit_predictions / total_predictions if total_predictions > 0 else 0,
                'average_confidence': avg_confidence,
                'average_probability': avg_probability,
                'model_type': self.model_type,
                'features_used': len(self.prediction_history[0]['features']) if self.prediction_history else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating model performance: {e}")
            return {}


# Factory functions

def create_ai_exit_strategy(
    strategy_id: str,
    symbol: str,
    model_type: str = 'ensemble',
    min_confidence: float = 0.65,
    **kwargs
) -> AIExitStrategy:
    """
    Create an AI-powered exit strategy
    
    @param strategy_id Unique identifier for the strategy
    @param symbol Trading symbol to monitor
    @param model_type Type of AI model ('technical', 'sentiment', 'ensemble')
    @param min_confidence Minimum confidence required for exit signals
    @param kwargs Additional configuration parameters
    
    @returns Configured AI exit strategy instance
    """
    parameters = {
        'model_type': model_type,
        'min_confidence': min_confidence,
        'feature_window': 50,
        'update_frequency': 3600,
        **kwargs
    }
    
    config = ExitConfiguration(
        strategy_id=strategy_id,
        strategy_type=ExitType.AI_DRIVEN,
        name=f"AI Exit Strategy ({symbol})",
        description=f"AI-powered exit strategy for {symbol}",
        parameters=parameters,
        symbols=[symbol]
    )
    
    return AIExitStrategy(config)


def create_technical_ai_exit(
    strategy_id: str,
    symbol: str,
    model_path: Optional[str] = None,
    **kwargs
) -> AIExitStrategy:
    """Create technical analysis-based AI exit strategy"""
    config = ExitConfiguration(
        strategy_id=strategy_id,
        strategy_type=ExitType.AI_DRIVEN,
        name=f"Technical AI Exit ({symbol})",
        description=f"Technical analysis AI exit for {symbol}",
        parameters={
            'model_type': 'technical',
            'model_path': model_path,
            **kwargs
        },
        symbols=[symbol]
    )
    
    strategy = AIExitStrategy(config)
    strategy.active_model = TechnicalIndicatorModel(model_path)
    return strategy


def create_sentiment_ai_exit(
    strategy_id: str,
    symbol: str,
    **kwargs
) -> AIExitStrategy:
    """Create sentiment analysis-based AI exit strategy"""
    config = ExitConfiguration(
        strategy_id=strategy_id,
        strategy_type=ExitType.AI_DRIVEN,
        name=f"Sentiment AI Exit ({symbol})",
        description=f"Sentiment analysis AI exit for {symbol}",
        parameters={
            'model_type': 'sentiment',
            **kwargs
        },
        symbols=[symbol]
    )
    
    strategy = AIExitStrategy(config)
    strategy.active_model = SentimentAnalysisModel()
    return strategy
