"""
@file library.py
@brief Comprehensive Trading Strategies Library

@details
This module provides a comprehensive library of 50+ trading strategies across
multiple categories, with user-selectable options, configuration management,
and performance tracking capabilities.

Strategy Categories:
- Momentum Strategies (15+): Moving averages, MACD, RSI momentum, etc.
- Mean Reversion Strategies (15+): Pairs trading, value investing, etc.
- Arbitrage Strategies (10+): Cross-broker, cross-venue, statistical arbitrage
- Volatility Strategies (10+): VIX-based, volatility breakout, etc.
- News-Based Strategies (5+): Sentiment analysis, event-driven trading
- AI/ML Strategies (10+): Machine learning predictions, neural networks

Key Features:
- Strategy registration and management
- Configuration validation and optimization
- Performance comparison and ranking
- Backtesting integration
- Real-time monitoring and alerts
- Community strategy sharing
- Strategy marketplace features

@author Trading Orchestrator System
@version 2.0
@date 2025-11-06
"""

from typing import Dict, Any, List, Optional, Type, Union
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import json
import asyncio
import uuid
from abc import ABC, abstractmethod

from loguru import logger

from .base import (
    BaseStrategy,
    StrategyConfig,
    StrategyType,
    TradingSignal,
    SignalType,
    RiskLevel,
    BaseTimeSeriesStrategy,
    strategy_registry,
    validate_strategy_config
)


class StrategyCategory(Enum):
    """Strategy categories for organization"""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    ARBITRAGE = "arbitrage"
    VOLATILITY = "volatility"
    NEWS_BASED = "news_based"
    AI_ML = "ai_ml"


@dataclass
class StrategyMetadata:
    """Metadata for strategy documentation and marketplace"""
    strategy_id: str
    name: str
    category: StrategyCategory
    description: str
    long_description: str
    author: str
    version: str
    tags: List[str] = field(default_factory=list)
    parameters_schema: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    downloads: int = 0
    rating: float = 0.0
    reviews: List[Dict[str, Any]] = field(default_factory=list)
    is_premium: bool = False
    requires_subscription: bool = False
    documentation_url: Optional[str] = None
    example_config: Dict[str, Any] = field(default_factory=dict)
    risk_warning: Optional[str] = None


@dataclass
class BacktestResults:
    """Backtest results for strategy evaluation"""
    strategy_id: str
    start_date: datetime
    end_date: datetime
    initial_capital: Decimal
    final_capital: Decimal
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: timedelta
    volatility: float
    beta: float
    alpha: float
    var_95: float
    cvar_95: float
    calibration_score: float
    out_of_sample_score: float
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class OptimizationResult:
    """Parameter optimization results"""
    strategy_id: str
    parameter_set: Dict[str, Any]
    score: float
    performance_metrics: Dict[str, float]
    optimization_method: str
    optimization_iterations: int
    created_at: datetime = field(default_factory=datetime.utcnow)


class StrategyLibrary:
    """Comprehensive strategy library management"""
    
    def __init__(self):
        self.strategies: Dict[str, Type[BaseStrategy]] = {}
        self.metadata: Dict[str, StrategyMetadata] = {}
        self.categories: Dict[StrategyCategory, List[str]] = {
            StrategyCategory.MOMENTUM: [],
            StrategyCategory.MEAN_REVERSION: [],
            StrategyCategory.ARBITRAGE: [],
            StrategyCategory.VOLATILITY: [],
            StrategyCategory.NEWS_BASED: [],
            StrategyCategory.AI_ML: []
        }
        self.backtest_results: Dict[str, List[BacktestResults]] = {}
        self.optimization_results: Dict[str, List[OptimizationResult]] = {}
        self.strategy_performance: Dict[str, Dict[str, float]] = {}
        
        # Initialize built-in strategies
        self._register_builtin_strategies()
        
        logger.info(f"Strategy Library initialized with {len(self.strategies)} strategies")
    
    def _register_builtin_strategies(self):
        """Register all built-in strategies"""
        try:
            # Import all strategy modules to register them
            from . import momentum
            from . import mean_reversion
            from . import arbitrage
            from . import volatility
            from . import news_based
            from . import ai_ml
            
            logger.info("Built-in strategies registered")
            
        except ImportError as e:
            logger.warning(f"Could not import some strategy modules: {e}")
    
    def register_strategy(
        self,
        strategy_class: Type[BaseStrategy],
        metadata: StrategyMetadata
    ):
        """Register a strategy in the library"""
        strategy_id = metadata.strategy_id
        
        # Validate strategy class
        if not issubclass(strategy_class, BaseStrategy):
            raise ValueError(f"Strategy class must inherit from BaseStrategy")
        
        # Store strategy class and metadata
        self.strategies[strategy_id] = strategy_class
        self.metadata[strategy_id] = metadata
        self.categories[metadata.category].append(strategy_id)
        
        logger.info(f"Strategy registered: {strategy_id} ({metadata.name})")
    
    def get_strategy_class(self, strategy_id: str) -> Optional[Type[BaseStrategy]]:
        """Get strategy class by ID"""
        return self.strategies.get(strategy_id)
    
    def get_strategy_metadata(self, strategy_id: str) -> Optional[StrategyMetadata]:
        """Get strategy metadata by ID"""
        return self.metadata.get(strategy_id)
    
    def get_strategies_by_category(self, category: StrategyCategory) -> List[StrategyMetadata]:
        """Get all strategies in a category"""
        strategy_ids = self.categories.get(category, [])
        return [self.metadata[sid] for sid in strategy_ids if sid in self.metadata]
    
    def get_all_strategies(self) -> List[StrategyMetadata]:
        """Get all registered strategies"""
        return list(self.metadata.values())
    
    def search_strategies(
        self,
        query: Optional[str] = None,
        category: Optional[StrategyCategory] = None,
        tags: Optional[List[str]] = None,
        min_rating: Optional[float] = None,
        limit: int = 50
    ) -> List[StrategyMetadata]:
        """Search strategies based on criteria"""
        results = []
        
        for metadata in self.metadata.values():
            # Filter by query
            if query and query.lower() not in metadata.name.lower() and \
               query.lower() not in metadata.description.lower():
                continue
            
            # Filter by category
            if category and metadata.category != category:
                continue
            
            # Filter by tags
            if tags:
                if not any(tag.lower() in [t.lower() for t in metadata.tags] for tag in tags):
                    continue
            
            # Filter by rating
            if min_rating and metadata.rating < min_rating:
                continue
            
            results.append(metadata)
            
            if len(results) >= limit:
                break
        
        # Sort by rating and downloads
        results.sort(key=lambda x: (x.rating, x.downloads), reverse=True)
        
        return results
    
    def create_strategy_instance(
        self,
        strategy_id: str,
        config: StrategyConfig,
        **kwargs
    ) -> Optional[BaseStrategy]:
        """Create a strategy instance from library"""
        strategy_class = self.get_strategy_class(strategy_id)
        if not strategy_class:
            logger.error(f"Strategy class not found: {strategy_id}")
            return None
        
        try:
            # Validate configuration
            errors = validate_strategy_config(config)
            if errors:
                logger.error(f"Configuration validation failed: {errors}")
                return None
            
            # Create instance
            instance = strategy_class(config, **kwargs)
            
            # Register with global registry
            strategy_registry.register_strategy(instance)
            
            logger.info(f"Strategy instance created: {strategy_id}")
            return instance
            
        except Exception as e:
            logger.error(f"Error creating strategy instance: {e}")
            return None
    
    def get_strategy_performance(
        self,
        strategy_id: str,
        time_period: str = "1m"
    ) -> Optional[Dict[str, float]]:
        """Get strategy performance metrics"""
        if strategy_id not in self.strategy_performance:
            return None
        
        return self.strategy_performance[strategy_id].get(time_period)
    
    def update_strategy_performance(
        self,
        strategy_id: str,
        performance_data: Dict[str, Dict[str, float]]
    ):
        """Update strategy performance metrics"""
        if strategy_id not in self.strategy_performance:
            self.strategy_performance[strategy_id] = {}
        
        self.strategy_performance[strategy_id].update(performance_data)
        
        # Update metadata rating based on performance
        if strategy_id in self.metadata:
            metadata = self.metadata[strategy_id]
            # Calculate rating based on performance metrics
            if 'sharpe_ratio' in performance_data.get('1m', {}):
                sharpe = performance_data['1m'].get('sharpe_ratio', 0)
                # Convert Sharpe ratio to 0-5 star rating
                rating = max(1.0, min(5.0, sharpe / 2.0))
                metadata.rating = rating
    
    def export_strategy_config(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Export strategy configuration template"""
        metadata = self.get_strategy_metadata(strategy_id)
        if not metadata:
            return None
        
        return {
            "strategy_id": strategy_id,
            "name": metadata.name,
            "description": metadata.description,
            "parameters": metadata.parameters_schema,
            "example_config": metadata.example_config,
            "risk_warning": metadata.risk_warning,
            "documentation_url": metadata.documentation_url
        }
    
    def validate_strategy_parameters(
        self,
        strategy_id: str,
        parameters: Dict[str, Any]
    ) -> List[str]:
        """Validate strategy parameters against schema"""
        metadata = self.get_strategy_metadata(strategy_id)
        if not metadata:
            return ["Strategy not found"]
        
        schema = metadata.parameters_schema
        errors = []
        
        # Validate required parameters
        required_params = schema.get('required', [])
        for param in required_params:
            if param not in parameters:
                errors.append(f"Missing required parameter: {param}")
        
        # Validate parameter types
        param_types = schema.get('properties', {})
        for param, value in parameters.items():
            if param in param_types:
                expected_type = param_types[param].get('type')
                if not self._validate_type(value, expected_type):
                    errors.append(f"Parameter {param} should be of type {expected_type}")
        
        # Validate parameter ranges
        param_ranges = schema.get('ranges', {})
        for param, value in parameters.items():
            if param in param_ranges:
                range_config = param_ranges[param]
                if 'min' in range_config and value < range_config['min']:
                    errors.append(f"Parameter {param} should be >= {range_config['min']}")
                if 'max' in range_config and value > range_config['max']:
                    errors.append(f"Parameter {param} should be <= {range_config['max']}")
        
        return errors
    
    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate value type against expected type"""
        type_map = {
            'integer': int,
            'number': (int, float),
            'float': float,
            'string': str,
            'boolean': bool,
            'array': list,
            'object': dict
        }
        
        expected_python_type = type_map.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)
        
        return True  # Unknown type, assume valid
    
    def get_strategy_recommendations(
        self,
        user_preferences: Dict[str, Any],
        limit: int = 5
    ) -> List[StrategyMetadata]:
        """Get personalized strategy recommendations"""
        recommendations = []
        
        # Get user's preferred categories and risk levels
        preferred_categories = user_preferences.get('preferred_categories', [])
        risk_tolerance = user_preference.get('risk_tolerance', 'medium')
        trading_experience = user_preferences.get('trading_experience', 'intermediate')
        
        # Calculate preference scores for each strategy
        for metadata in self.metadata.values():
            score = 0.0
            
            # Category preference score
            if metadata.category.value in preferred_categories:
                score += 2.0
            
            # Risk level score
            risk_score = self._get_risk_score(metadata, risk_tolerance)
            score += risk_score
            
            # Experience level score
            exp_score = self._get_experience_score(metadata, trading_experience)
            score += exp_score
            
            # Performance score
            perf_score = self._get_performance_score(metadata)
            score += perf_score
            
            # Popularity score (downloads and rating)
            pop_score = (metadata.downloads / 1000) * 0.1 + metadata.rating * 0.2
            score += pop_score
            
            recommendations.append((metadata, score))
        
        # Sort by score and return top recommendations
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return [rec[0] for rec in recommendations[:limit]]
    
    def _get_risk_score(self, metadata: StrategyMetadata, risk_tolerance: str) -> float:
        """Get risk compatibility score"""
        # This would analyze strategy metadata to determine risk level
        # For now, return a simple score based on tags
        if 'low-risk' in [t.lower() for t in metadata.tags]:
            return 2.0 if risk_tolerance == 'low' else 1.0
        elif 'high-risk' in [t.lower() for t in metadata.tags]:
            return 2.0 if risk_tolerance == 'high' else 1.0
        else:
            return 1.5  # Medium risk strategies
    
    def _get_experience_score(self, metadata: StrategyMetadata, experience: str) -> float:
        """Get experience compatibility score"""
        if 'beginner' in [t.lower() for t in metadata.tags]:
            return 2.0 if experience == 'beginner' else 1.0
        elif 'advanced' in [t.lower() for t in metadata.tags]:
            return 2.0 if experience == 'advanced' else 1.0
        else:
            return 1.5  # Intermediate strategies
    
    def _get_performance_score(self, metadata: StrategyMetadata) -> float:
        """Get performance-based score"""
        # Use performance metrics to calculate score
        if not metadata.performance_metrics:
            return 1.0
        
        score = 0.0
        metrics = metadata.performance_metrics
        
        # Sharpe ratio contribution
        if 'sharpe_ratio' in metrics:
            score += min(2.0, metrics['sharpe_ratio'] / 2.0)
        
        # Max drawdown contribution (lower is better)
        if 'max_drawdown' in metrics:
            score += max(0.0, 2.0 - abs(metrics['max_drawdown']) / 10.0)
        
        # Win rate contribution
        if 'win_rate' in metrics:
            score += metrics['win_rate'] / 50.0  # Scale to 0-2 range
        
        return score
    
    def create_strategy_ensemble(
        self,
        strategy_ids: List[str],
        weights: Optional[List[float]] = None,
        ensemble_type: str = "weighted_average"
    ) -> Dict[str, Any]:
        """Create strategy ensemble configuration"""
        if len(strategy_ids) < 2:
            raise ValueError("Ensemble requires at least 2 strategies")
        
        if weights is None:
            # Equal weights
            weights = [1.0 / len(strategy_ids)] * len(strategy_ids)
        
        if len(weights) != len(strategy_ids):
            raise ValueError("Number of weights must match number of strategies")
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        ensemble_config = {
            "ensemble_id": f"ensemble_{uuid.uuid4().hex[:8]}",
            "strategy_ids": strategy_ids,
            "weights": normalized_weights,
            "ensemble_type": ensemble_type,
            "created_at": datetime.utcnow().isoformat(),
            "description": f"Ensemble of {len(strategy_ids)} strategies"
        }
        
        logger.info(f"Strategy ensemble created: {ensemble_config['ensemble_id']}")
        return ensemble_config
    
    def compare_strategies(
        self,
        strategy_ids: List[str],
        metrics: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Compare multiple strategies across performance metrics"""
        if metrics is None:
            metrics = [
                'total_return', 'sharpe_ratio', 'max_drawdown',
                'win_rate', 'profit_factor', 'total_trades'
            ]
        
        comparison = {}
        
        for strategy_id in strategy_ids:
            metadata = self.get_strategy_metadata(strategy_id)
            if not metadata:
                continue
            
            comparison[strategy_id] = {
                'name': metadata.name,
                'category': metadata.category.value,
                'rating': metadata.rating,
                'downloads': metadata.downloads
            }
            
            # Add performance metrics
            performance = self.get_strategy_performance(strategy_id)
            if performance:
                for metric in metrics:
                    comparison[strategy_id][metric] = performance.get(metric, 0.0)
            else:
                for metric in metrics:
                    comparison[strategy_id][metric] = 0.0
        
        return comparison
    
    def get_leaderboard(self, category: Optional[StrategyCategory] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get strategy performance leaderboard"""
        if category:
            strategies = self.get_strategies_by_category(category)
        else:
            strategies = self.get_all_strategies()
        
        # Sort by rating and performance
        leaderboard = []
        for metadata in strategies:
            performance = self.get_strategy_performance(metadata.strategy_id)
            if performance:
                score = self._calculate_leaderboard_score(metadata, performance)
                leaderboard.append({
                    'strategy_id': metadata.strategy_id,
                    'name': metadata.name,
                    'category': metadata.category.value,
                    'score': score,
                    'rating': metadata.rating,
                    'downloads': metadata.downloads,
                    'performance': performance
                })
        
        # Sort by score
        leaderboard.sort(key=lambda x: x['score'], reverse=True)
        
        return leaderboard[:limit]
    
    def _calculate_leaderboard_score(self, metadata: StrategyMetadata, performance: Dict[str, float]) -> float:
        """Calculate leaderboard score for strategy"""
        score = 0.0
        
        # Rating score (0-5 scale)
        score += metadata.rating
        
        # Performance metrics (normalized)
        if 'sharpe_ratio' in performance:
            score += min(2.0, performance['sharpe_ratio'] / 2.0)
        
        if 'total_return' in performance:
            score += min(2.0, performance['total_return'] / 20.0)  # 20% = 2 points
        
        if 'max_drawdown' in performance:
            # Lower drawdown is better
            score += max(0.0, 2.0 - abs(performance['max_drawdown']) / 10.0)
        
        # Popularity bonus
        score += min(1.0, metadata.downloads / 10000)
        
        return score
    
    def export_strategy_report(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Export comprehensive strategy report"""
        metadata = self.get_strategy_metadata(strategy_id)
        if not metadata:
            return None
        
        performance = self.get_strategy_performance(strategy_id)
        backtests = self.backtest_results.get(strategy_id, [])
        
        report = {
            "strategy_info": {
                "strategy_id": metadata.strategy_id,
                "name": metadata.name,
                "category": metadata.category.value,
                "description": metadata.description,
                "author": metadata.author,
                "version": metadata.version,
                "created_at": metadata.created_at.isoformat(),
                "tags": metadata.tags
            },
            "performance": performance or {},
            "backtest_results": [
                {
                    "start_date": bt.start_date.isoformat(),
                    "end_date": bt.end_date.isoformat(),
                    "total_return": bt.total_return,
                    "sharpe_ratio": bt.sharpe_ratio,
                    "max_drawdown": bt.max_drawdown,
                    "win_rate": bt.win_rate
                }
                for bt in backtests[-5:]  # Last 5 backtests
            ],
            "configuration": self.export_strategy_config(strategy_id),
            "risk_information": {
                "risk_level": metadata.risk_warning,
                "max_drawdown_risk": performance.get('max_drawdown', 0) if performance else 0,
                "volatility_risk": performance.get('volatility', 0) if performance else 0
            }
        }
        
        return report
    
    def save_library_state(self, filepath: str):
        """Save library state to file"""
        state = {
            "strategies": list(self.strategies.keys()),
            "metadata": {
                sid: {
                    "strategy_id": md.strategy_id,
                    "name": md.name,
                    "category": md.category.value,
                    "description": md.description,
                    "author": md.author,
                    "version": md.version,
                    "tags": md.tags,
                    "rating": md.rating,
                    "downloads": md.downloads,
                    "created_at": md.created_at.isoformat(),
                    "performance_metrics": md.performance_metrics
                }
                for sid, md in self.metadata.items()
            },
            "performance_data": self.strategy_performance,
            "last_updated": datetime.utcnow().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Library state saved to {filepath}")


# Global strategy library instance
strategy_library = StrategyLibrary()


# Utility functions for strategy development
def create_strategy_template(
    name: str,
    category: StrategyCategory,
    description: str,
    parameters_schema: Dict[str, Any]
) -> str:
    """Generate strategy template code"""
    
    template = f'''"""
@file {name.lower().replace(" ", "_")}.py
@brief {name} Strategy Implementation

@details
{description}

Strategy Logic:
1. [Add strategy logic steps here]

@author Generated Strategy Template
@version 1.0
@date {datetime.utcnow().strftime("%Y-%m-%d")}

@warning
[Add strategy-specific warnings here]

@note
[Add strategy-specific notes here]
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from decimal import Decimal

from loguru import logger

from .base import (
    BaseStrategy,
    StrategyConfig,
    StrategyType,
    TradingSignal,
    SignalType,
    RiskLevel,
    BaseTimeSeriesStrategy
)
from trading.models import OrderSide


class {name.replace(" ", "").replace("-", "")}Strategy(BaseTimeSeriesStrategy):
    """
    {name} Strategy
    
    {description}
    
    Parameters:
    """
    
    def __init__(self, config: StrategyConfig):
        # Validate required parameters
        required_params = {list(parameters_schema.get('required', []))}
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {{param}}")
        
        super().__init__(config, timeframe="1h")
        
        # Extract parameters with defaults
        # TODO: Add parameter extraction here
        
        logger.info(f"{name} Strategy initialized")
    
    async def generate_signals(self) -> List[TradingSignal]:
        """Generate {name} signals"""
        signals = []
        
        try:
            # Process each configured symbol
            for symbol in self.config.symbols:
                signal = await self._analyze_symbol(symbol)
                if signal:
                    signals.append(signal)
            
            logger.debug(f"Generated {len(signals)} {name} signals")
            
        except Exception as e:
            logger.error(f"Error generating {name} signals: {{e}}")
        
        return signals
    
    async def _analyze_symbol(self, symbol: str) -> Optional[TradingSignal]:
        """Analyze symbol for {name} signals"""
        # TODO: Implement strategy-specific logic
        return None
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        """Validate {name} signal"""
        # TODO: Implement signal validation
        return True
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get detailed strategy information"""
        return {{
            'strategy_name': '{name}',
            'description': '{description}',
            'category': '{category.value}',
            'parameters': {{}},
            'indicators_used': [],
            'timeframe': self.timeframe,
            'risk_level': self.config.risk_level.value
        }}


# Factory function
def create_{name.lower().replace(" ", "_")}_strategy(
    strategy_id: str,
    symbols: List[str],
    **kwargs
) -> {name.replace(" ", "").replace("-", "")}Strategy:
    """Factory function to create {name} strategy"""
    
    config = StrategyConfig(
        strategy_id=strategy_id,
        strategy_type=StrategyType.{category.name.upper()},
        name="{name}",
        description="{description}",
        parameters=kwargs,
        risk_level=RiskLevel.MEDIUM,
        symbols=symbols
    )
    
    return {name.replace(" ", "").replace("-", "")}Strategy(config)
'''
    
    return template


# Example usage
if __name__ == "__main__":
    async def test_strategy_library():
        # Get all strategies
        all_strategies = strategy_library.get_all_strategies()
        print(f"Total strategies in library: {len(all_strategies)}")
        
        # Get strategies by category
        momentum_strategies = strategy_library.get_strategies_by_category(StrategyCategory.MOMENTUM)
        print(f"Momentum strategies: {len(momentum_strategies)}")
        
        # Search strategies
        search_results = strategy_library.search_strategies(query="moving average")
        print(f"Search results for 'moving average': {len(search_results)}")
        
        # Get recommendations
        recommendations = strategy_library.get_strategy_recommendations({
            'preferred_categories': ['momentum', 'trend_following'],
            'risk_tolerance': 'medium',
            'trading_experience': 'intermediate'
        })
        print(f"Recommendations: {len(recommendations)}")
    
    asyncio.run(test_strategy_library())