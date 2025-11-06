"""
Exit Strategies Framework
Comprehensive exit strategy system with AI-driven decision making

Provides:
- Base exit strategy framework
- Profit-taking strategies (trailing stops, fixed targets, percentage-based)
- Loss-cutting strategies (stop loss, volatility-based, time-based)
- Conditional exit strategies based on market conditions
- AI-driven exit decision making
- Exit strategy configuration and backtesting
- Exit strategy monitoring and optimization
- Performance analytics integration
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Protocol
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum

from loguru import logger

# Base exit strategy framework
from .base_exit_strategy import (
    # Core classes
    BaseExitStrategy,
    ExitCondition,
    ExitSignal,
    ExitContext,
    
    # Enums
    ExitReason,
    ExitType,
    ExitStatus,
    
    # Data structures
    ExitMetrics,
    ExitConfiguration,
    ExitRule,
    
    # Global registry
    exit_strategy_registry
)

# Exit strategy implementations
from .trailing_stop import (
    TrailingStopStrategy,
    ATRTrailingStop,
    FixedTrailingStop,
    create_trailing_stop_strategy
)

from .fixed_target import (
    FixedTargetStrategy,
    create_fixed_target_strategy
)

from .stop_loss import (
    StopLossStrategy,
    PercentageStopLoss,
    VolatilityStopLoss,
    create_stop_loss_strategy
)

from .volatility_stop import (
    VolatilityStopStrategy,
    ATRVolatilityStop,
    create_volatility_stop_strategy
)

from .ai_exit_strategy import (
    AIExitStrategy,
    MLExitDecision,
    create_ai_exit_strategy
)

from .time_based_exit import (
    TimeBasedExitStrategy,
    create_time_based_exit_strategy
)

from .conditional_exit import (
    ConditionalExitStrategy,
    create_conditional_exit_strategy
)

# Backtesting and optimization
from .backtesting import (
    ExitStrategyBacktestEngine,
    ExitStrategyBacktestResult,
    run_exit_strategy_backtest
)

from .optimization import (
    ExitStrategyOptimizer,
    ExitParameterRange,
    OptimizeExitStrategyParameters,
    run_exit_strategy_optimization
)

# Monitoring and alerting
from .monitoring import (
    ExitStrategyMonitor,
    ExitAlert,
    ExitStrategyAlertManager
)

__all__ = [
    # Base framework
    'BaseExitStrategy',
    'ExitCondition',
    'ExitSignal',
    'ExitContext',
    'exit_strategy_registry',
    
    # Enums
    'ExitReason',
    'ExitType',
    'ExitStatus',
    
    # Data structures
    'ExitMetrics',
    'ExitConfiguration',
    'ExitRule',
    
    # Exit strategy implementations
    'TrailingStopStrategy',
    'FixedTargetStrategy',
    'StopLossStrategy',
    'VolatilityStopStrategy',
    'AIExitStrategy',
    'TimeBasedExitStrategy',
    'ConditionalExitStrategy',
    'ATRTrailingStop',
    'FixedTrailingStop',
    'PercentageStopLoss',
    'VolatilityStopLoss',
    'MLExitDecision',
    
    # Factory functions
    'create_trailing_stop_strategy',
    'create_fixed_target_strategy',
    'create_stop_loss_strategy',
    'create_volatility_stop_strategy',
    'create_ai_exit_strategy',
    'create_time_based_exit_strategy',
    'create_conditional_exit_strategy',
    
    # Backtesting and optimization
    'ExitStrategyBacktestEngine',
    'ExitStrategyBacktestResult',
    'run_exit_strategy_backtest',
    'ExitStrategyOptimizer',
    'ExitParameterRange',
    'OptimizeExitStrategyParameters',
    'run_exit_strategy_optimization',
    
    # Monitoring and alerting
    'ExitStrategyMonitor',
    'ExitAlert',
    'ExitStrategyAlertManager'
]

# Quick start example
def quick_start_exit_strategies_example():
    """
    Example of how to use the exit strategies framework
    """
    return """
# Quick Start Exit Strategies Example

from strategies.exit_strategies import (
    create_trailing_stop_strategy,
    create_fixed_target_strategy,
    create_stop_loss_strategy,
    run_exit_strategy_backtest,
    ExitReason,
    ExitType
)
from decimal import Decimal

async def main():
    # 1. Create trailing stop strategy
    trailing_stop = create_trailing_stop_strategy(
        strategy_id="ts_001",
        symbol="AAPL",
        trailing_distance=Decimal('0.05'),  # 5% trailing stop
        min_stop_distance=Decimal('0.02')   # Minimum 2% stop
    )
    
    # 2. Create fixed target strategy
    fixed_target = create_fixed_target_strategy(
        strategy_id="ft_001",
        symbol="AAPL",
        target_profit=Decimal('0.10'),      # 10% profit target
        target_loss=Decimal('0.05')         # 5% loss threshold
    )
    
    # 3. Create stop loss strategy
    stop_loss = create_stop_loss_strategy(
        strategy_id="sl_001",
        symbol="AAPL",
        stop_percentage=Decimal('0.03'),    # 3% stop loss
        activation_delay=60                  # 60 seconds before activation
    )
    
    # 4. Test exit strategies on historical data
    from strategies.backtesting import run_exit_strategy_backtest
    
    backtest_result = await run_exit_strategy_backtest(
        strategies=[trailing_stop, fixed_target, stop_loss],
        symbol="AAPL",
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
        initial_price=Decimal('150.00')
    )
    
    # 5. View results
    print(f"Trailing Stop Returns: {trailing_stop.metrics.total_return:.2%}")
    print(f"Fixed Target Returns: {fixed_target.metrics.total_return:.2%}")
    print(f"Stop Loss Protection: {stop_loss.metrics.success_rate:.2%}")
    
    # 6. Compare performance
    for strategy in [trailing_stop, fixed_target, stop_loss]:
        print(f"{strategy.config.name}:")
        print(f"  - Total Return: {strategy.metrics.total_return:.2%}")
        print(f"  - Max Drawdown: {strategy.metrics.max_drawdown:.2%}")
        print(f"  - Win Rate: {strategy.metrics.win_rate:.2%}")

# Run the example
import asyncio
asyncio.run(main())
"""

async def create_exit_strategy_from_config(config_data: Dict[str, Any]):
    """
    Create an exit strategy from configuration dictionary
    
    Args:
        config_data: Configuration dictionary with exit strategy parameters
        
    Returns:
        Configured exit strategy instance
    """
    strategy_type = ExitType(config_data.get('type', 'trailing_stop'))
    
    if strategy_type == ExitType.TRAILING_STOP:
        return create_trailing_stop_strategy(**config_data)
    elif strategy_type == ExitType.FIXED_TARGET:
        return create_fixed_target_strategy(**config_data)
    elif strategy_type == ExitType.STOP_LOSS:
        return create_stop_loss_strategy(**config_data)
    elif strategy_type == ExitType.VOLATILITY_STOP:
        return create_volatility_stop_strategy(**config_data)
    elif strategy_type == ExitType.AI_DRIVEN:
        return create_ai_exit_strategy(**config_data)
    elif strategy_type == ExitType.TIME_BASED:
        return create_time_based_exit_strategy(**config_data)
    elif strategy_type == ExitType.CONDITIONAL:
        return create_conditional_exit_strategy(**config_data)
    else:
        raise ValueError(f"Unknown exit strategy type: {strategy_type}")

async def run_comprehensive_exit_analysis(
    strategies: List[BaseExitStrategy],
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    initial_price: Decimal,
    benchmark_data: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Run comprehensive analysis comparing multiple exit strategies
    
    Args:
        strategies: List of exit strategies to analyze
        symbol: Trading symbol
        start_date: Analysis start date
        end_date: Analysis end date
        initial_price: Initial asset price
        benchmark_data: Optional benchmark performance data
        
    Returns:
        Comprehensive analysis results
    """
    try:
        # Run backtests for each strategy
        backtest_results = []
        for strategy in strategies:
            result = await run_exit_strategy_backtest(
                strategies=[strategy],
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                initial_price=initial_price
            )
            backtest_results.append(result)
        
        # Compile comparison metrics
        comparison = {
            'symbol': symbol,
            'analysis_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'initial_price': float(initial_price)
            },
            'strategies': [],
            'summary': {
                'best_return': None,
                'lowest_drawdown': None,
                'highest_sharpe': None,
                'best_win_rate': None
            }
        }
        
        best_return = float('-inf')
        lowest_drawdown = float('inf')
        highest_sharpe = float('-inf')
        best_win_rate = float('-inf')
        
        for i, (strategy, result) in enumerate(zip(strategies, backtest_results)):
            strategy_data = {
                'strategy_id': strategy.config.strategy_id,
                'name': strategy.config.name,
                'type': strategy.config.exit_type.value,
                'metrics': {
                    'total_return': result.strategies[strategy.config.strategy_id].metrics.total_return,
                    'max_drawdown': result.strategies[strategy.config.strategy_id].metrics.max_drawdown,
                    'sharpe_ratio': result.strategies[strategy.config.strategy_id].metrics.sharpe_ratio,
                    'win_rate': result.strategies[strategy.config.strategy_id].metrics.win_rate,
                    'total_exits': result.strategies[strategy.config.strategy_id].metrics.total_exits,
                    'avg_exit_time': result.strategies[strategy.config.strategy_id].metrics.avg_exit_time
                }
            }
            
            comparison['strategies'].append(strategy_data)
            
            # Update best metrics
            metrics = strategy_data['metrics']
            if metrics['total_return'] > best_return:
                best_return = metrics['total_return']
                comparison['summary']['best_return'] = strategy.config.strategy_id
            
            if metrics['max_drawdown'] < lowest_drawdown:
                lowest_drawdown = metrics['max_drawdown']
                comparison['summary']['lowest_drawdown'] = strategy.config.strategy_id
            
            if metrics['sharpe_ratio'] > highest_sharpe:
                highest_sharpe = metrics['sharpe_ratio']
                comparison['summary']['highest_sharpe'] = strategy.config.strategy_id
            
            if metrics['win_rate'] > best_win_rate:
                best_win_rate = metrics['win_rate']
                comparison['summary']['best_win_rate'] = strategy.config.strategy_id
        
        # Add benchmark comparison if provided
        if benchmark_data:
            comparison['benchmark'] = benchmark_data
        
        logger.info(f"Comprehensive exit analysis completed for {symbol}")
        return comparison
        
    except Exception as e:
        logger.error(f"Error in comprehensive exit analysis: {e}")
        return {}

# Export utility functions
__all__.extend([
    'create_exit_strategy_from_config',
    'run_comprehensive_exit_analysis',
    'quick_start_exit_strategies_example'
])
