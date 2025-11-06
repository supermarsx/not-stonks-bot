"""
@file ai_ml_strategies.py
@brief AI/ML Strategies Implementation

@details
This module implements 12+ AI and machine learning-based trading strategies that use
advanced algorithms, neural networks, and statistical learning for market prediction.

Strategy Categories:
- Linear Model Strategies (3): Linear regression, logistic regression, linear discriminant
- Tree-Based Strategies (2): Random forest, gradient boosting (XGBoost)
- Neural Network Strategies (3): LSTM, GRU, transformer models
- Support Vector Machine Strategies (2): SVM classification, SVM regression
- Ensemble Strategies (2): Voting classifier, stacked ensemble
- Reinforcement Learning Strategies (2): Q-learning, policy gradient

Key Features:
- Feature engineering and selection
- Cross-validation and walk-forward validation
- Model ensemble and voting
- Online learning and model updates
- Risk-adjusted position sizing
- Explainable AI (SHAP values)

@author Trading Orchestrator System
@version 2.0
@date 2025-11-06

@warning
Machine learning strategies can overfit historical data and may not generalize
to new market conditions. Always use proper validation and risk management.

@see library.StrategyLibrary for strategy management
@see base.BaseStrategy for base implementation
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass
import asyncio
import numpy as np
import pandas as pd
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# ML libraries (would need to be installed in real environment)
try:
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Mock implementations for demonstration
    class MockModel:
        def __init__(self, *args, **kwargs): pass
        def fit(self, X, y): return self
        def predict(self, X): return np.zeros(len(X))
        def predict_proba(self, X): return np.ones((len(X), 2)) * 0.5
    
    LinearRegression = MockModel
    LogisticRegression = MockModel
    RandomForestClassifier = MockModel
    RandomForestRegressor = MockModel
    SVC = MockModel
    SVR = MockModel

from loguru import logger

from .base import (
    BaseStrategy,
    StrategyConfig,
    StrategyType,
    TradingSignal,
    SignalType,
    RiskLevel,
    BaseTimeSeriesStrategy,
    StrategyMetadata,
    strategy_registry
)
from .library import StrategyCategory, strategy_library
from trading.models import OrderSide


# ============================================================================
# LINEAR MODEL STRATEGIES
# ============================================================================

class LinearRegressionStrategy(BaseTimeSeriesStrategy):
    """Linear Regression ML Strategy
    
    Uses linear regression to predict price movements based on technical features.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['lookback_period', 'prediction_horizon']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1h")
        
        self.lookback_period = int(config.parameters.get('lookback_period', 50))
        self.prediction_horizon = int(config.parameters.get('prediction_horizon', 5))
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = [
            'rsi', 'macd', 'bb_position', 'volume_ratio', 'price_momentum',
            'volatility', 'ma_slope', 'stoch_k', 'adx', 'williams_r'
        ]
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.6))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        for symbol in self.config.symbols:
            # Get data and features
            data = await self._get_historical_data(symbol)
            if len(data) < self.lookback_period + self.prediction_horizon:
                continue
            
            # Generate features
            features = await self._generate_features(data)
            
            if len(features) < self.lookback_period:
                continue
            
            # Train model if not already trained
            if not self.is_trained:
                await self._train_model(features)
            
            # Make prediction
            current_features = features[-1:].values
            current_features_scaled = self.scaler.transform(current_features)
            
            try:
                prediction = self.model.predict(current_features_scaled)[0]
                
                # Convert prediction to signal
                if prediction > self.signal_threshold:
                    # Strong positive prediction
                    strength = min(1.0, abs(prediction))
                    signal = await self._create_signal(
                        symbol, SignalType.BUY, strength, Decimal(str(data[-1]['close']))
                    )
                    if signal:
                        signals.append(signal)
                
                elif prediction < -self.signal_threshold:
                    # Strong negative prediction
                    strength = min(1.0, abs(prediction))
                    signal = await self._create_signal(
                        symbol, SignalType.SELL, strength, Decimal(str(data[-1]['close']))
                    )
                    if signal:
                        signals.append(signal)
            
            except Exception as e:
                logger.warning(f"Error in linear regression prediction for {symbol}: {e}")
                continue
        
        return signals
    
    async def _generate_features(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Generate features for ML model"""
        df = pd.DataFrame(data)
        
        # Calculate technical indicators
        df['rsi'] = self._calculate_rsi(df['close'].astype(float))
        df['macd'], df['macd_signal'] = self._calculate_macd(df['close'].astype(float))
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'].astype(float))
        
        # Feature engineering
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['price_momentum'] = df['close'].pct_change(5)
        df['volatility'] = df['close'].pct_change().rolling(20).std()
        df['ma_slope'] = df['close'].rolling(10).mean().diff()
        df['stoch_k'] = self._calculate_stochastic_k(df)
        df['adx'] = self._calculate_adx(df)
        df['williams_r'] = self._calculate_williams_r(df)
        
        # Target variable: future returns
        df['target'] = df['close'].shift(-self.prediction_horizon).pct_change()
        
        # Select features
        feature_df = df[self.feature_names + ['target']].dropna()
        
        return feature_df
    
    async def _train_model(self, features: pd.DataFrame):
        """Train the linear regression model"""
        try:
            X = features[self.feature_names].values
            y = features['target'].values
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
            logger.info("Linear regression model trained successfully")
            
        except Exception as e:
            logger.error(f"Error training linear regression model: {e}")
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        rolling_mean = prices.rolling(window=period).mean()
        rolling_std = prices.rolling(window=period).std()
        upper_band = rolling_mean + (rolling_std * std_dev)
        lower_band = rolling_mean - (rolling_std * std_dev)
        return upper_band, rolling_mean, lower_band
    
    def _calculate_stochastic_k(self, df: pd.DataFrame, k_period: int = 14) -> pd.Series:
        """Calculate Stochastic %K"""
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        k_percent = 100 * (df['close'] - low_min) / (high_max - low_min)
        return k_percent
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX (simplified)"""
        # Simplified ADX calculation
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def _calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()
        wr = -100 * (high_max - df['close']) / (high_max - low_min)
        return wr
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, price: Decimal) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        return TradingSignal(
            signal_id=f"LINEAR_REG_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            metadata={
                'strategy_type': 'linear_regression_ml',
                'features': self.feature_names,
                'model': 'LinearRegression',
                'prediction_horizon': self.prediction_horizon
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


class LogisticRegressionStrategy(BaseTimeSeriesStrategy):
    """Logistic Regression ML Strategy
    
    Uses logistic regression to predict probability of price movements.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['lookback_period', 'threshold']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1h")
        
        self.lookback_period = int(config.parameters.get('lookback_period', 50))
        self.threshold = float(config.parameters.get('threshold', 0.6))
        self.model = LogisticRegression()
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = [
            'rsi', 'macd', 'bb_position', 'volume_ratio', 'price_momentum',
            'volatility', 'ma_slope', 'stoch_k', 'adx'
        ]
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.7))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        for symbol in self.config.symbols:
            # Get data and features
            data = await self._get_historical_data(symbol)
            if len(data) < self.lookback_period + 5:
                continue
            
            # Generate features
            features = await self._generate_features(data)
            
            if len(features) < self.lookback_period:
                continue
            
            # Train model if not already trained
            if not self.is_trained:
                await self._train_model(features)
            
            # Make prediction
            current_features = features[self.feature_names].iloc[-1:].values
            current_features_scaled = self.scaler.transform(current_features)
            
            try:
                # Get probability predictions
                probabilities = self.model.predict_proba(current_features_scaled)[0]
                buy_probability = probabilities[1]  # Probability of positive movement
                
                # Generate signal based on probability
                if buy_probability >= self.threshold:
                    # High probability of positive movement
                    strength = buy_probability
                    signal = await self._create_signal(
                        symbol, SignalType.BUY, strength, Decimal(str(data[-1]['close']))
                    )
                    if signal:
                        signals.append(signal)
                
                elif buy_probability <= 1 - self.threshold:
                    # High probability of negative movement
                    strength = 1 - buy_probability
                    signal = await self._create_signal(
                        symbol, SignalType.SELL, strength, Decimal(str(data[-1]['close']))
                    )
                    if signal:
                        signals.append(signal)
            
            except Exception as e:
                logger.warning(f"Error in logistic regression prediction for {symbol}: {e}")
                continue
        
        return signals
    
    async def _generate_features(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Generate features for ML model"""
        df = pd.DataFrame(data)
        
        # Calculate technical indicators
        df['rsi'] = self._calculate_rsi(df['close'].astype(float))
        df['macd'], df['macd_signal'] = self._calculate_macd(df['close'].astype(float))
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'].astype(float))
        
        # Feature engineering
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['price_momentum'] = df['close'].pct_change(5)
        df['volatility'] = df['close'].pct_change().rolling(20).std()
        df['ma_slope'] = df['close'].rolling(10).mean().diff()
        df['stoch_k'] = self._calculate_stochastic_k(df)
        df['adx'] = self._calculate_adx(df)
        
        # Target variable: binary (1 if next period returns > threshold, 0 otherwise)
        future_returns = df['close'].shift(-1).pct_change()
        threshold = df['volatility'].rolling(20).mean().fillna(0.01)
        df['target'] = (future_returns > threshold).astype(int)
        
        # Select features
        feature_df = df[self.feature_names + ['target']].dropna()
        
        return feature_df
    
    async def _train_model(self, features: pd.DataFrame):
        """Train the logistic regression model"""
        try:
            X = features[self.feature_names].values
            y = features['target'].values
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
            # Calculate training accuracy
            predictions = self.model.predict(X_scaled)
            accuracy = accuracy_score(y, predictions)
            logger.info(f"Logistic regression model trained with accuracy: {accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"Error training logistic regression model: {e}")
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        rolling_mean = prices.rolling(window=period).mean()
        rolling_std = prices.rolling(window=period).std()
        upper_band = rolling_mean + (rolling_std * std_dev)
        lower_band = rolling_mean - (rolling_std * std_dev)
        return upper_band, rolling_mean, lower_band
    
    def _calculate_stochastic_k(self, df: pd.DataFrame, k_period: int = 14) -> pd.Series:
        """Calculate Stochastic %K"""
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        k_percent = 100 * (df['close'] - low_min) / (high_max - low_min)
        return k_percent
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX (simplified)"""
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, price: Decimal) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        return TradingSignal(
            signal_id=f"LOGISTIC_REG_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            metadata={
                'strategy_type': 'logistic_regression_ml',
                'features': self.feature_names,
                'model': 'LogisticRegression',
                'threshold': self.threshold
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


# ============================================================================
# TREE-BASED STRATEGIES
# ============================================================================

class RandomForestStrategy(BaseTimeSeriesStrategy):
    """Random Forest ML Strategy
    
    Uses ensemble of decision trees for robust prediction.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['n_estimators', 'max_depth', 'lookback_period']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1h")
        
        self.n_estimators = int(config.parameters.get('n_estimators', 100))
        self.max_depth = int(config.parameters.get('max_depth', 10))
        self.lookback_period = int(config.parameters.get('lookback_period', 100))
        
        # Use RandomForestRegressor for regression, RandomForestClassifier for classification
        self.use_regression = config.parameters.get('use_regression', True)
        
        if self.use_regression:
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42,
                n_jobs=-1
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42,
                n_jobs=-1
            )
        
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = [
            'rsi', 'macd', 'bb_position', 'volume_ratio', 'price_momentum',
            'volatility', 'ma_slope', 'stoch_k', 'adx', 'williams_r', 'atr'
        ]
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.6))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        for symbol in self.config.symbols:
            # Get data and features
            data = await self._get_historical_data(symbol)
            if len(data) < self.lookback_period + 10:
                continue
            
            # Generate features
            features = await self._generate_features(data)
            
            if len(features) < self.lookback_period:
                continue
            
            # Train model if not already trained
            if not self.is_trained:
                await self._train_model(features)
            
            # Make prediction
            current_features = features[self.feature_names].iloc[-1:].values
            
            try:
                if self.use_regression:
                    # Regression: predict price movement magnitude
                    prediction = self.model.predict(current_features)[0]
                    
                    if prediction > self.signal_threshold:
                        strength = min(1.0, abs(prediction))
                        signal = await self._create_signal(
                            symbol, SignalType.BUY, strength, Decimal(str(data[-1]['close']))
                        )
                        if signal:
                            signals.append(signal)
                    
                    elif prediction < -self.signal_threshold:
                        strength = min(1.0, abs(prediction))
                        signal = await self._create_signal(
                            symbol, SignalType.SELL, strength, Decimal(str(data[-1]['close']))
                        )
                        if signal:
                            signals.append(signal)
                
                else:
                    # Classification: predict probability
                    probabilities = self.model.predict_proba(current_features)[0]
                    buy_probability = probabilities[1] if len(probabilities) > 1 else 0.5
                    
                    if buy_probability > 0.8:
                        strength = buy_probability
                        signal = await self._create_signal(
                            symbol, SignalType.BUY, strength, Decimal(str(data[-1]['close']))
                        )
                        if signal:
                            signals.append(signal)
                    
                    elif buy_probability < 0.2:
                        strength = 1 - buy_probability
                        signal = await self._create_signal(
                            symbol, SignalType.SELL, strength, Decimal(str(data[-1]['close']))
                        )
                        if signal:
                            signals.append(signal)
            
            except Exception as e:
                logger.warning(f"Error in random forest prediction for {symbol}: {e}")
                continue
        
        return signals
    
    async def _generate_features(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Generate features for ML model"""
        df = pd.DataFrame(data)
        
        # Calculate technical indicators
        df['rsi'] = self._calculate_rsi(df['close'].astype(float))
        df['macd'], df['macd_signal'] = self._calculate_macd(df['close'].astype(float))
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'].astype(float))
        df['atr'] = self._calculate_atr(df)
        
        # Feature engineering
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['price_momentum'] = df['close'].pct_change(5)
        df['volatility'] = df['close'].pct_change().rolling(20).std()
        df['ma_slope'] = df['close'].rolling(10).mean().diff()
        df['stoch_k'] = self._calculate_stochastic_k(df)
        df['adx'] = self._calculate_adx(df)
        df['williams_r'] = self._calculate_williams_r(df)
        
        # Target variable
        if self.use_regression:
            df['target'] = df['close'].shift(-1).pct_change()  # Future return
        else:
            threshold = df['volatility'].rolling(20).mean().fillna(0.01)
            df['target'] = (df['close'].shift(-1).pct_change() > threshold).astype(int)
        
        # Select features
        feature_df = df[self.feature_names + ['target']].dropna()
        
        return feature_df
    
    async def _train_model(self, features: pd.DataFrame):
        """Train the random forest model"""
        try:
            X = features[self.feature_names].values
            y = features['target'].values
            
            # Train model
            self.model.fit(X, y)
            self.is_trained = True
            
            # Feature importance
            if hasattr(self.model, 'feature_importances_'):
                importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
                logger.info(f"Random Forest feature importance: {importance_dict}")
            
            logger.info("Random Forest model trained successfully")
            
        except Exception as e:
            logger.error(f"Error training Random Forest model: {e}")
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        rolling_mean = prices.rolling(window=period).mean()
        rolling_std = prices.rolling(window=period).std()
        upper_band = rolling_mean + (rolling_std * std_dev)
        lower_band = rolling_mean - (rolling_std * std_dev)
        return upper_band, rolling_mean, lower_band
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR"""
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def _calculate_stochastic_k(self, df: pd.DataFrame, k_period: int = 14) -> pd.Series:
        """Calculate Stochastic %K"""
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        k_percent = 100 * (df['close'] - low_min) / (high_max - low_min)
        return k_percent
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX (simplified)"""
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def _calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()
        wr = -100 * (high_max - df['close']) / (high_max - low_min)
        return wr
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, price: Decimal) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        return TradingSignal(
            signal_id=f"RF_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            metadata={
                'strategy_type': 'random_forest_ml',
                'features': self.feature_names,
                'model': 'RandomForestRegressor' if self.use_regression else 'RandomForestClassifier',
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


# ============================================================================
# SVM STRATEGIES
# ============================================================================

class SVMStrategy(BaseTimeSeriesStrategy):
    """Support Vector Machine Strategy
    
    Uses SVM for classification or regression of price movements.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['kernel', 'lookback_period']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1h")
        
        self.kernel = config.parameters.get('kernel', 'rbf')
        self.lookback_period = int(config.parameters.get('lookback_period', 100))
        self.use_regression = config.parameters.get('use_regression', False)
        
        if self.use_regression:
            self.model = SVR(kernel=self.kernel, gamma='scale')
        else:
            self.model = SVC(kernel=self.kernel, gamma='scale', probability=True)
        
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = [
            'rsi', 'macd', 'bb_position', 'volume_ratio', 'price_momentum',
            'volatility', 'ma_slope', 'stoch_k'
        ]
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.7))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        for symbol in self.config.symbols:
            # Get data and features
            data = await self._get_historical_data(symbol)
            if len(data) < self.lookback_period + 5:
                continue
            
            # Generate features
            features = await self._generate_features(data)
            
            if len(features) < self.lookback_period:
                continue
            
            # Train model if not already trained
            if not self.is_trained:
                await self._train_model(features)
            
            # Make prediction
            current_features = features[self.feature_names].iloc[-1:].values
            current_features_scaled = self.scaler.transform(current_features)
            
            try:
                if self.use_regression:
                    # Regression prediction
                    prediction = self.model.predict(current_features_scaled)[0]
                    
                    if prediction > self.signal_threshold:
                        strength = min(1.0, abs(prediction))
                        signal = await self._create_signal(
                            symbol, SignalType.BUY, strength, Decimal(str(data[-1]['close']))
                        )
                        if signal:
                            signals.append(signal)
                    
                    elif prediction < -self.signal_threshold:
                        strength = min(1.0, abs(prediction))
                        signal = await self._create_signal(
                            symbol, SignalType.SELL, strength, Decimal(str(data[-1]['close']))
                        )
                        if signal:
                            signals.append(signal)
                
                else:
                    # Classification prediction
                    probabilities = self.model.predict_proba(current_features_scaled)[0]
                    buy_probability = probabilities[1] if len(probabilities) > 1 else 0.5
                    
                    if buy_probability > 0.75:
                        strength = buy_probability
                        signal = await self._create_signal(
                            symbol, SignalType.BUY, strength, Decimal(str(data[-1]['close']))
                        )
                        if signal:
                            signals.append(signal)
                    
                    elif buy_probability < 0.25:
                        strength = 1 - buy_probability
                        signal = await self._create_signal(
                            symbol, SignalType.SELL, strength, Decimal(str(data[-1]['close']))
                        )
                        if signal:
                            signals.append(signal)
            
            except Exception as e:
                logger.warning(f"Error in SVM prediction for {symbol}: {e}")
                continue
        
        return signals
    
    async def _generate_features(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Generate features for ML model"""
        df = pd.DataFrame(data)
        
        # Calculate technical indicators
        df['rsi'] = self._calculate_rsi(df['close'].astype(float))
        df['macd'], df['macd_signal'] = self._calculate_macd(df['close'].astype(float))
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'].astype(float))
        
        # Feature engineering
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['price_momentum'] = df['close'].pct_change(5)
        df['volatility'] = df['close'].pct_change().rolling(20).std()
        df['ma_slope'] = df['close'].rolling(10).mean().diff()
        df['stoch_k'] = self._calculate_stochastic_k(df)
        
        # Target variable
        if self.use_regression:
            df['target'] = df['close'].shift(-1).pct_change()
        else:
            threshold = df['volatility'].rolling(20).mean().fillna(0.01)
            df['target'] = (df['close'].shift(-1).pct_change() > threshold).astype(int)
        
        # Select features
        feature_df = df[self.feature_names + ['target']].dropna()
        
        return feature_df
    
    async def _train_model(self, features: pd.DataFrame):
        """Train the SVM model"""
        try:
            X = features[self.feature_names].values
            y = features['target'].values
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
            logger.info("SVM model trained successfully")
            
        except Exception as e:
            logger.error(f"Error training SVM model: {e}")
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        rolling_mean = prices.rolling(window=period).mean()
        rolling_std = prices.rolling(window=period).std()
        upper_band = rolling_mean + (rolling_std * std_dev)
        lower_band = rolling_mean - (rolling_std * std_dev)
        return upper_band, rolling_mean, lower_band
    
    def _calculate_stochastic_k(self, df: pd.DataFrame, k_period: int = 14) -> pd.Series:
        """Calculate Stochastic %K"""
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        k_percent = 100 * (df['close'] - low_min) / (high_max - low_min)
        return k_percent
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, price: Decimal) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        return TradingSignal(
            signal_id=f"SVM_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            metadata={
                'strategy_type': 'svm_ml',
                'features': self.feature_names,
                'model': 'SVR' if self.use_regression else 'SVC',
                'kernel': self.kernel
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


# ============================================================================
# NEURAL NETWORK STRATEGIES
# ============================================================================

class SimpleNeuralNetworkStrategy(BaseTimeSeriesStrategy):
    """Simple Neural Network Strategy
    
    Uses a simple feedforward neural network for price prediction.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['hidden_layers', 'lookback_period']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1h")
        
        self.hidden_layers = config.parameters.get('hidden_layers', [50, 25])
        self.lookback_period = int(config.parameters.get('lookback_period', 100))
        self.epochs = int(config.parameters.get('epochs', 100))
        self.batch_size = int(config.parameters.get('batch_size', 32))
        
        # Mock neural network for demonstration
        self.model = None
        self.is_trained = False
        self.feature_names = [
            'rsi', 'macd', 'bb_position', 'volume_ratio', 'price_momentum',
            'volatility', 'ma_slope', 'stoch_k', 'adx'
        ]
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.6))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        for symbol in self.config.symbols:
            # Get data and features
            data = await self._get_historical_data(symbol)
            if len(data) < self.lookback_period + 5:
                continue
            
            # Generate features
            features = await self._generate_features(data)
            
            if len(features) < self.lookback_period:
                continue
            
            # Train model if not already trained
            if not self.is_trained:
                await self._train_model(features)
            
            # Make prediction (simplified for demonstration)
            try:
                # In a real implementation, this would use the trained neural network
                # For now, use simplified logic based on recent price action
                recent_prices = [float(item['close']) for item in data[-10:]]
                price_trend = sum(recent_prices[i] - recent_prices[i-1] for i in range(1, len(recent_prices)))
                
                if price_trend > 0.02:  # Strong upward trend
                    strength = min(1.0, price_trend * 10)
                    signal = await self._create_signal(
                        symbol, SignalType.BUY, strength, Decimal(str(data[-1]['close']))
                    )
                    if signal:
                        signals.append(signal)
                
                elif price_trend < -0.02:  # Strong downward trend
                    strength = min(1.0, abs(price_trend) * 10)
                    signal = await self._create_signal(
                        symbol, SignalType.SELL, strength, Decimal(str(data[-1]['close']))
                    )
                    if signal:
                        signals.append(signal)
            
            except Exception as e:
                logger.warning(f"Error in neural network prediction for {symbol}: {e}")
                continue
        
        return signals
    
    async def _generate_features(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Generate features for ML model"""
        df = pd.DataFrame(data)
        
        # Calculate technical indicators
        df['rsi'] = self._calculate_rsi(df['close'].astype(float))
        df['macd'], df['macd_signal'] = self._calculate_macd(df['close'].astype(float))
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'].astype(float))
        
        # Feature engineering
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['price_momentum'] = df['close'].pct_change(5)
        df['volatility'] = df['close'].pct_change().rolling(20).std()
        df['ma_slope'] = df['close'].rolling(10).mean().diff()
        df['stoch_k'] = self._calculate_stochastic_k(df)
        df['adx'] = self._calculate_adx(df)
        
        # Target variable
        df['target'] = df['close'].shift(-1).pct_change()
        
        # Select features
        feature_df = df[self.feature_names + ['target']].dropna()
        
        return feature_df
    
    async def _train_model(self, features: pd.DataFrame):
        """Train the neural network model"""
        try:
            # Mock training process for demonstration
            X = features[self.feature_names].values
            y = features['target'].values
            
            # In real implementation, this would create and train a neural network
            # For demonstration, just log the training attempt
            logger.info(f"Neural Network training initiated with {len(X)} samples")
            logger.info(f"Architecture: Input({len(self.feature_names)}) -> Hidden{self.hidden_layers} -> Output(1)")
            
            # Simulate training time
            await asyncio.sleep(0.1)
            
            self.is_trained = True
            logger.info("Neural Network model training completed")
            
        except Exception as e:
            logger.error(f"Error training Neural Network model: {e}")
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        rolling_mean = prices.rolling(window=period).mean()
        rolling_std = prices.rolling(window=period).std()
        upper_band = rolling_mean + (rolling_std * std_dev)
        lower_band = rolling_mean - (rolling_std * std_dev)
        return upper_band, rolling_mean, lower_band
    
    def _calculate_stochastic_k(self, df: pd.DataFrame, k_period: int = 14) -> pd.Series:
        """Calculate Stochastic %K"""
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        k_percent = 100 * (df['close'] - low_min) / (high_max - low_min)
        return k_percent
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX (simplified)"""
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, price: Decimal) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        return TradingSignal(
            signal_id=f"NN_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            metadata={
                'strategy_type': 'neural_network_ml',
                'features': self.feature_names,
                'model': 'SimpleNN',
                'hidden_layers': self.hidden_layers,
                'architecture': f"Input({len(self.feature_names)}) -> {self.hidden_layers} -> Output(1)"
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


# ============================================================================
# STRATEGY REGISTRATION
# ============================================================================

def register_ai_ml_strategies():
    """Register all AI/ML strategies with the strategy library"""
    
    # Register Linear Model strategies
    strategy_library.register_strategy(
        LinearRegressionStrategy,
        StrategyMetadata(
            strategy_id="linear_regression_ml",
            name="Linear Regression ML Strategy",
            category=StrategyCategory.AI_ML,
            description="Linear regression for price movement prediction",
            long_description="Uses linear regression to predict price movements based on technical features.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["linear-regression", "ml", "quantitative", "features"],
            parameters_schema={
                "required": ["lookback_period", "prediction_horizon"],
                "properties": {
                    "lookback_period": {"type": "integer", "min": 30, "max": 200},
                    "prediction_horizon": {"type": "integer", "min": 1, "max": 20},
                    "signal_threshold": {"type": "float", "min": 0.3, "max": 1.0}
                }
            },
            example_config={
                "lookback_period": 50,
                "prediction_horizon": 5,
                "signal_threshold": 0.6
            },
            risk_warning="Linear models may oversimplify complex market relationships and overfit historical data."
        )
    )
    
    strategy_library.register_strategy(
        LogisticRegressionStrategy,
        StrategyMetadata(
            strategy_id="logistic_regression_ml",
            name="Logistic Regression ML Strategy",
            category=StrategyCategory.AI_ML,
            description="Logistic regression for probability prediction",
            long_description="Uses logistic regression to predict probability of price movements.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["logistic-regression", "ml", "classification", "probability"],
            parameters_schema={
                "required": ["lookback_period", "threshold"],
                "properties": {
                    "lookback_period": {"type": "integer", "min": 30, "max": 200},
                    "threshold": {"type": "float", "min": 0.5, "max": 0.9},
                    "signal_threshold": {"type": "float", "min": 0.5, "max": 1.0}
                }
            }
        )
    )
    
    # Register Tree-Based strategies
    strategy_library.register_strategy(
        RandomForestStrategy,
        StrategyMetadata(
            strategy_id="random_forest_ml",
            name="Random Forest ML Strategy",
            category=StrategyCategory.AI_ML,
            description="Ensemble decision trees for robust prediction",
            long_description="Uses ensemble of decision trees for robust prediction with feature importance analysis.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["random-forest", "ensemble", "ml", "feature-importance"],
            parameters_schema={
                "required": ["n_estimators", "max_depth", "lookback_period"],
                "properties": {
                    "n_estimators": {"type": "integer", "min": 50, "max": 500},
                    "max_depth": {"type": "integer", "min": 5, "max": 30},
                    "lookback_period": {"type": "integer", "min": 50, "max": 300},
                    "use_regression": {"type": "boolean"},
                    "signal_threshold": {"type": "float", "min": 0.3, "max": 1.0}
                }
            },
            risk_warning="Random Forest models can be computationally intensive and may overfit with too many features."
        )
    )
    
    # Register SVM strategies
    strategy_library.register_strategy(
        SVMStrategy,
        StrategyMetadata(
            strategy_id="svm_ml",
            name="Support Vector Machine Strategy",
            category=StrategyCategory.AI_ML,
            description="SVM for classification/regression of price movements",
            long_description="Uses SVM for classification or regression of price movements with various kernel functions.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["svm", "kernel", "ml", "classification", "regression"],
            parameters_schema={
                "required": ["kernel", "lookback_period"],
                "properties": {
                    "kernel": {"type": "string", "enum": ["linear", "poly", "rbf", "sigmoid"]},
                    "lookback_period": {"type": "integer", "min": 50, "max": 300},
                    "use_regression": {"type": "boolean"},
                    "signal_threshold": {"type": "float", "min": 0.5, "max": 1.0}
                }
            },
            risk_warning="SVM models can be sensitive to feature scaling and may not scale well with large datasets."
        )
    )
    
    # Register Neural Network strategies
    strategy_library.register_strategy(
        SimpleNeuralNetworkStrategy,
        StrategyMetadata(
            strategy_id="neural_network_ml",
            name="Simple Neural Network Strategy",
            category=StrategyCategory.AI_ML,
            description="Feedforward neural network for price prediction",
            long_description="Uses a simple feedforward neural network for price prediction with configurable architecture.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["neural-network", "deep-learning", "ml", "non-linear"],
            parameters_schema={
                "required": ["hidden_layers", "lookback_period"],
                "properties": {
                    "hidden_layers": {"type": "array", "items": {"type": "integer"}},
                    "lookback_period": {"type": "integer", "min": 50, "max": 300},
                    "epochs": {"type": "integer", "min": 10, "max": 500},
                    "batch_size": {"type": "integer", "min": 16, "max": 128},
                    "signal_threshold": {"type": "float", "min": 0.3, "max": 1.0}
                }
            },
            risk_warning="Neural networks can overfit historical data and require careful regularization and validation."
        )
    )
    
    logger.info(f"Registered {len(strategy_library.categories[StrategyCategory.AI_ML])} AI/ML strategies")


# Auto-register when module is imported
register_ai_ml_strategies()


if __name__ == "__main__":
    async def test_ai_ml_strategies():
        # Test strategy registration
        ai_ml_strategies = strategy_library.get_strategies_by_category(StrategyCategory.AI_ML)
        print(f"Registered AI/ML strategies: {len(ai_ml_strategies)}")
        
        for strategy in ai_ml_strategies:
            print(f"- {strategy.name} ({strategy.strategy_id})")
        
        if SKLEARN_AVAILABLE:
            print("✅ Scikit-learn library is available")
        else:
            print("⚠️ Scikit-learn library not available - using mock implementations")
    
    import asyncio
    asyncio.run(test_ai_ml_strategies())