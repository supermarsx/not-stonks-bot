"""
Trading Strategies Framework
Comprehensive trading strategy system with optimization and backtesting

Provides:
- Base strategy classes and interfaces
- Individual strategy implementations (trend following, mean reversion, pairs trading, arbitrage)
- Parameter optimization system
- Comprehensive backtesting framework
- Strategy selection and management
"""

from datetime import timedelta
from typing import Dict, Any
from decimal import Decimal

from loguru import logger

# Base classes and interfaces
from .base import (
    # Core classes
    BaseStrategy,
    BaseTimeSeriesStrategy,
    StrategyRegistry,
    
    # Enums
    StrategyType,
    StrategyStatus,
    SignalType,
    RiskLevel,
    
    # Data structures
    TradingSignal,
    StrategyMetrics,
    StrategyConfig,
    StrategyContext,
    
    # Global registry
    strategy_registry
)

# Individual strategy implementations
from .trend_following import (
    TrendFollowingStrategy,
    create_trend_following_strategy
)

from .mean_reversion import (
    MeanReversionStrategy,
    create_mean_reversion_strategy
)

from .pairs_trading import (
    PairsTradingStrategy,
    create_pairs_trading_strategy
)

from .arbitrage import (
    CrossVenueArbitrageStrategy,
    ArbitrageType,
    VenueType,
    VenuePrice,
    ArbitrageOpportunity,
    create_arbitrage_strategy
)

# Import comprehensive strategy modules
from .momentum import (
    # SMA Strategies
    SMACrossoverStrategy,
    SMASlopeStrategy,
    SMABreakoutStrategy,
    
    # EMA Strategies
    EMACrossoverStrategy,
    EMARibbonStrategy,
    EMASlopeStrategy,
    
    # MACD Strategies
    MACDClassicStrategy,
    MACDDivergenceStrategy,
    
    # RSI Strategies
    RSIMomentumStrategy,
    RSIDivergenceStrategy,
    RSIOversoldOverboughtStrategy,
    
    # Price Momentum Strategies
    PriceMomentumStrategy,
    ROCStrategy,
    
    # Volume-weighted Strategies
    VolumeMomentumStrategy,
    VolumeROCStrategy
)

from .volatility_strategies import (
    # VIX-based Strategies
    VIXBreakoutStrategy,
    VIXMeanReversionStrategy,
    VIXFuturesContangoStrategy,
    
    # Volatility Breakout Strategies
    BollingerSqueezeBreakoutStrategy,
    VolatilityExpansionBreakoutStrategy,
    
    # Realized Volatility Strategies
    GARCHVolatilityStrategy,
    
    # Cross-Asset Volatility Strategies
    StockBondVolatilityStrategy
)

from .news_based_strategies import (
    # Sentiment Analysis Strategies
    NewsSentimentStrategy,
    SocialMediaSentimentStrategy,
    
    # Earnings-based Strategies
    EarningsSurpriseStrategy,
    
    # Economic Calendar Strategies
    EconomicCalendarStrategy,
    
    # Media Sentiment Strategies
    AnalystRatingStrategy,
    
    # Market Psychology Strategies
    FearGreedIndexStrategy
)

from .ai_ml_strategies import (
    # Linear Model Strategies
    LinearRegressionStrategy,
    LogisticRegressionStrategy,
    
    # Tree-based Strategies
    RandomForestStrategy,
    
    # SVM Strategies
    SVMStrategy,
    
    # Neural Network Strategies
    SimpleNeuralNetworkStrategy
)

# Optimization system
from .optimization import (
    # Optimization classes
    ParameterOptimizer,
    OptimizationStrategy,
    GridSearchOptimizer,
    RandomSearchOptimizer,
    GeneticAlgorithmOptimizer,
    
    # Enums
    OptimizationMethod,
    OptimizationObjective,
    
    # Data structures
    ParameterRange,
    OptimizationResult,
    
    # Utilities
    create_parameter_ranges
)

# Backtesting framework
from .backtesting import (
    # Core classes
    BacktestEngine,
    BacktestContext,
    
    # Data structures
    Trade,
    Position,
    BacktestMetrics,
    BacktestResult,
    
    # Enums
    BacktestStatus,
    
    # Utilities
    compare_backtests
)

__all__ = [
    # Base framework
    'BaseStrategy',
    'BaseTimeSeriesStrategy',
    'StrategyRegistry',
    'strategy_registry',
    
    # Enums
    'StrategyType',
    'StrategyStatus',
    'SignalType',
    'RiskLevel',
    'OptimizationMethod',
    'OptimizationObjective',
    'BacktestStatus',
    'ArbitrageType',
    'VenueType',
    
    # Data structures
    'TradingSignal',
    'StrategyMetrics',
    'StrategyConfig',
    'StrategyContext',
    'ParameterRange',
    'OptimizationResult',
    'Trade',
    'Position',
    'BacktestMetrics',
    'BacktestResult',
    'VenuePrice',
    'ArbitrageOpportunity',
    
    # Strategy implementations
    'TrendFollowingStrategy',
    'MeanReversionStrategy',
    'PairsTradingStrategy',
    'CrossVenueArbitrageStrategy',
    
    # Momentum Strategies
    'SMACrossoverStrategy',
    'SMASlopeStrategy',
    'SMABreakoutStrategy',
    'EMACrossoverStrategy',
    'EMARibbonStrategy',
    'EMASlopeStrategy',
    'MACDClassicStrategy',
    'MACDDivergenceStrategy',
    'RSIMomentumStrategy',
    'RSIDivergenceStrategy',
    'RSIOversoldOverboughtStrategy',
    'PriceMomentumStrategy',
    'ROCStrategy',
    'VolumeMomentumStrategy',
    'VolumeROCStrategy',
    
    # Volatility Strategies
    'VIXBreakoutStrategy',
    'VIXMeanReversionStrategy',
    'VIXFuturesContangoStrategy',
    'BollingerSqueezeBreakoutStrategy',
    'VolatilityExpansionBreakoutStrategy',
    'GARCHVolatilityStrategy',
    'StockBondVolatilityStrategy',
    
    # News-based Strategies
    'NewsSentimentStrategy',
    'SocialMediaSentimentStrategy',
    'EarningsSurpriseStrategy',
    'EconomicCalendarStrategy',
    'AnalystRatingStrategy',
    'FearGreedIndexStrategy',
    
    # AI/ML Strategies
    'LinearRegressionStrategy',
    'LogisticRegressionStrategy',
    'RandomForestStrategy',
    'SVMStrategy',
    'SimpleNeuralNetworkStrategy',
    
    # Factory functions
    'create_trend_following_strategy',
    'create_mean_reversion_strategy',
    'create_pairs_trading_strategy',
    'create_arbitrage_strategy',
    'create_parameter_ranges',
    
    # Core engines
    'ParameterOptimizer',
    'BacktestEngine',
    
    # Optimization algorithms
    'OptimizationStrategy',
    'GridSearchOptimizer',
    'RandomSearchOptimizer',
    'GeneticAlgorithmOptimizer',
    
    # Utilities
    'compare_backtests'
]

# Quick start examples
def quick_start_example():
    """
    Example of how to use the trading strategies framework
    """
    return """
# Quick Start Example

from strategies import (
    create_trend_following_strategy,
    ParameterOptimizer,
    BacktestEngine,
    OptimizationMethod,
    OptimizationObjective
)
from datetime import datetime

async def main():
    # 1. Create a strategy
    strategy = create_trend_following_strategy(
        strategy_id="my_strategy",
        symbols=['AAPL', 'GOOGL'],
        fast_period=10,
        slow_period=30
    )
    
    # 2. Optimize parameters
    optimizer = ParameterOptimizer()
    
    # Define parameter ranges
    from strategies.optimization import ParameterRange
    parameter_ranges = [
        ParameterRange("fast_period", "int", 5, 20, 1),
        ParameterRange("slow_period", "int", 20, 50, 5),
        ParameterRange("signal_threshold", "float", 0.3, 0.8, 0.1)
    ]
    
    # Run optimization
    result = await optimizer.optimize_parameters(
        method=OptimizationMethod.RANDOM_SEARCH,
        objective=OptimizationObjective.MAXIMIZE_SHARPE_RATIO,
        parameter_ranges=parameter_ranges,
        objective_function=lambda params: 0.5  # Dummy objective
    )
    
    # 3. Update strategy with best parameters
    strategy.config.parameters.update(result.best_parameters)
    
    # 4. Backtest the optimized strategy
    engine = BacktestEngine()
    
    backtest_result = await engine.run_backtest(
        strategy=strategy,
        symbols=['AAPL', 'GOOGL'],
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
        initial_capital=100000
    )
    
    # 5. View results
    print(f"Total Return: {backtest_result.metrics.total_return:.2%}")
    print(f"Sharpe Ratio: {backtest_result.metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {backtest_result.metrics.max_drawdown:.2%}")

# Run the example
import asyncio
asyncio.run(main())
"""

# Strategy management utilities
async def create_strategy_from_config(config_data: Dict[str, Any]):
    """
    Create a strategy from configuration dictionary
    
    Args:
        config_data: Configuration dictionary with strategy parameters
        
    Returns:
        Configured strategy instance
    """
    strategy_type = StrategyType(config_data.get('type', 'trend_following'))
    
    if strategy_type == StrategyType.TREND_FOLLOWING:
        return create_trend_following_strategy(**config_data)
    elif strategy_type == StrategyType.MEAN_REVERSION:
        return create_mean_reversion_strategy(**config_data)
    elif strategy_type == StrategyType.PAIRS_TRADING:
        return create_pairs_trading_strategy(**config_data)
    elif strategy_type == StrategyType.ARBITRAGE:
        return create_arbitrage_strategy(**config_data)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

async def batch_optimize_strategies(
    strategies: List[BaseStrategy],
    objective: OptimizationObjective,
    method: OptimizationMethod = OptimizationMethod.RANDOM_SEARCH,
    **kwargs
) -> List[OptimizationResult]:
    """
    Optimize multiple strategies in batch
    
    Args:
        strategies: List of strategies to optimize
        objective: Optimization objective
        method: Optimization method to use
        **kwargs: Additional optimization parameters
        
    Returns:
        List of optimization results
    """
    optimizer = ParameterOptimizer()
    results = []
    
    for strategy in strategies:
        try:
            # Create parameter ranges based on strategy type
            parameter_ranges = create_parameter_ranges(strategy.config.strategy_type)
            
            # Create objective function
            objective_function = optimizer.create_strategy_objective_function(
                type(strategy),
                strategy.config.symbols[0] if strategy.config.symbols else 'AAPL',
                datetime.utcnow() - timedelta(days=365),
                datetime.utcnow(),
                objective
            )
            
            # Run optimization
            result = await optimizer.optimize_parameters(
                method=method,
                objective=objective,
                parameter_ranges=parameter_ranges,
                objective_function=objective_function,
                **kwargs
            )
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error optimizing strategy {strategy.config.strategy_id}: {e}")
            continue
    
    return results

async def run_strategy_comparison(
    strategies: List[BaseStrategy],
    start_date: datetime,
    end_date: datetime,
    initial_capital: Decimal = Decimal('100000')
) -> Dict[str, Any]:
    """
    Run comparison backtest for multiple strategies
    
    Args:
        strategies: List of strategies to compare
        start_date: Backtest start date
        end_date: Backtest end date
        initial_capital: Starting capital
        
    Returns:
        Comparison results
    """
    engine = BacktestEngine()
    results = []
    
    for strategy in strategies:
        try:
            result = await engine.run_backtest(
                strategy=strategy,
                symbols=strategy.config.symbols,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital
            )
            
            if result.status == BacktestStatus.COMPLETED:
                results.append(result)
            
        except Exception as e:
            logger.error(f"Error backtesting strategy {strategy.config.strategy_id}: {e}")
            continue
    
    return compare_backtests(results)

# Export utility functions
__all__.extend([
    'create_strategy_from_config',
    'batch_optimize_strategies',
    'run_strategy_comparison',
    'quick_start_example'
])