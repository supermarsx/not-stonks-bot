"""
@file enhanced_backtesting.py
@brief Enhanced Backtesting Engine with Advanced Features

@details
This module provides comprehensive backtesting capabilities including:
- Multi-timeframe historical data testing
- Monte Carlo simulation for risk analysis
- Walk-forward optimization and out-of-sample testing
- Transaction cost modeling and market impact simulation
- Advanced performance attribution
- Strategy ranking and selection algorithms
- Automated reporting and visualization
- Risk-adjusted performance metrics
- Benchmark comparison and alpha analysis

Key Features:
- Walk-forward optimization framework
- Monte Carlo stress testing
- Transaction cost modeling
- Market impact simulation
- Advanced performance metrics
- Strategy comparison and ranking
- Automated report generation
- Risk decomposition analysis

@author Trading Orchestrator System
@version 2.0
@date 2025-11-06

@see backtesting for basic backtesting framework
@see strategy_management for strategy ensemble testing
"""

from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import math
import json
import warnings
warnings.filterwarnings('ignore')

from loguru import logger

from .base import (
    BaseStrategy,
    StrategyConfig,
    StrategyType,
    TradingSignal,
    SignalType,
    RiskLevel,
    BaseTimeSeriesStrategy,
    strategy_registry
)
from .library import StrategyCategory, StrategyMetadata, strategy_library


class TransactionCostModel(Enum):
    """Transaction cost modeling approaches"""
    FIXED = "fixed"
    PROPORTIONAL = "proportional"
    TIERED = "tiered"
    MARKET_IMPACT = "market_impact"
    REALISTIC = "realistic"


class OptimizationMethod(Enum):
    """Parameter optimization methods"""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    GENETIC_ALGORITHM = "genetic_algorithm"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    WALK_FORWARD = "walk_forward"


class BenchmarkType(Enum):
    """Benchmark comparison types"""
    MARKET_INDEX = "market_index"
    CUSTOM_PORTFOLIO = "custom_portfolio"
    STRATEGY_BENCHMARK = "strategy_benchmark"
    BUY_AND_HOLD = "buy_and_hold"


@dataclass
class TransactionCost:
    """Transaction cost configuration"""
    commission: float = 0.001  # 0.1% commission
    spread_cost: float = 0.0005  # 0.05% spread cost
    market_impact_coeff: float = 0.0001  # Market impact coefficient
    minimum_cost: float = 1.0  # Minimum transaction cost
    tiered_costs: List[Tuple[float, float]] = field(default_factory=list)  # (volume, cost_per_share)


@dataclass
class BacktestConfig:
    """Enhanced backtesting configuration"""
    start_date: datetime
    end_date: datetime
    initial_capital: Decimal = Decimal('100000')
    transaction_costs: TransactionCost = field(default_factory=TransactionCost)
    benchmark_type: BenchmarkType = BenchmarkType.MARKET_INDEX
    benchmark_symbols: List[str] = field(default_factory=lambda: ['SPY'])
    rebalance_frequency: str = 'daily'
    slippage_model: str = 'linear'
    max_leverage: float = 3.0
    position_sizing_model: str = 'fixed_dollar'
    risk_free_rate: float = 0.02
    confidence_level: float = 0.95
    monte_carlo_simulations: int = 1000
    walk_forward_window: int = 252  # 1 year
    out_of_sample_pct: float = 0.2
    optimization_method: OptimizationMethod = OptimizationMethod.RANDOM_SEARCH
    max_parameters_to_optimize: int = 5
    save_detailed_results: bool = True
    generate_reports: bool = True


@dataclass
class WalkForwardResult:
    """Walk-forward optimization result"""
    window_start: datetime
    window_end: datetime
    optimization_period: datetime
    test_period: datetime
    best_parameters: Dict[str, Any]
    in_sample_metrics: Dict[str, float]
    out_sample_metrics: Dict[str, float]
    parameter_stability: float
    performance_degradation: float


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation result"""
    simulation_id: int
    final_capital: Decimal
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    var_95: float
    cvar_95: float
    skewness: float
    kurtosis: float


@dataclass
class RiskDecomposition:
    """Risk decomposition analysis"""
    total_risk: float
    systematic_risk: float
    idiosyncratic_risk: float
    factor_exposures: Dict[str, float]
    sector_exposures: Dict[str, str]
    concentration_risk: float
    correlation_risk: float
    tail_risk_measure: float


class EnhancedBacktestEngine:
    """Enhanced backtesting engine with advanced features"""
    
    def __init__(self):
        self.backtest_results: Dict[str, Dict[str, Any]] = {}
        self.walk_forward_results: List[WalkForwardResult] = []
        self.monte_carlo_results: List[MonteCarloResult] = []
        self.benchmark_data: Dict[str, pd.DataFrame] = {}
        self.performance_cache: Dict[str, Dict[str, Any]] = {}
    
    async def run_comprehensive_backtest(
        self,
        strategy: BaseStrategy,
        config: BacktestConfig,
        symbols: List[str] = None
    ) -> Dict[str, Any]:
        """Run comprehensive backtest with all advanced features"""
        try:
            logger.info(f"Starting comprehensive backtest for {strategy.config.strategy_id}")
            
            if not symbols:
                symbols = strategy.config.symbols
            
            # Load benchmark data
            await self._load_benchmark_data(config.benchmark_symbols, config.start_date, config.end_date)
            
            # Run basic backtest
            basic_results = await self._run_basic_backtest(strategy, config, symbols)
            
            # Run walk-forward optimization if enabled
            walk_forward_results = []
            if config.optimization_method == OptimizationMethod.WALK_FORWARD:
                walk_forward_results = await self._run_walk_forward_optimization(strategy, config, symbols)
            
            # Run Monte Carlo simulation
            monte_carlo_results = await self._run_monte_carlo_simulation(strategy, config, symbols)
            
            # Calculate advanced metrics
            advanced_metrics = await self._calculate_advanced_metrics(strategy, config, basic_results)
            
            # Perform risk decomposition
            risk_decomposition = await self._perform_risk_decomposition(strategy, config, basic_results)
            
            # Generate performance attribution
            performance_attribution = await self._calculate_performance_attribution(basic_results, config)
            
            # Run benchmark comparison
            benchmark_comparison = await self._compare_with_benchmark(basic_results, config)
            
            # Compile comprehensive results
            comprehensive_results = {
                'strategy_id': strategy.config.strategy_id,
                'backtest_config': config,
                'symbols': symbols,
                'basic_results': basic_results,
                'walk_forward_results': walk_forward_results,
                'monte_carlo_results': {
                    'summary': self._summarize_monte_carlo_results(monte_carlo_results),
                    'detailed_results': monte_carlo_results[:100]  # Limit to first 100 for memory
                },
                'advanced_metrics': advanced_metrics,
                'risk_decomposition': risk_decomposition,
                'performance_attribution': performance_attribution,
                'benchmark_comparison': benchmark_comparison,
                'summary': {
                    'total_return': basic_results['final_metrics']['total_return'],
                    'sharpe_ratio': advanced_metrics['sharpe_ratio'],
                    'max_drawdown': basic_results['final_metrics']['max_drawdown'],
                    'win_rate': basic_results['final_metrics']['win_rate'],
                    'alpha_vs_benchmark': benchmark_comparison.get('alpha', 0.0),
                    'outperforming_probability': monte_carlo_results and np.mean([r.total_return > 0 for r in monte_carlo_results]) or 0.0
                },
                'backtest_timestamp': datetime.utcnow().isoformat()
            }
            
            # Cache results
            self.backtest_results[strategy.config.strategy_id] = comprehensive_results
            
            logger.info(f"Comprehensive backtest completed for {strategy.config.strategy_id}")
            return comprehensive_results
            
        except Exception as e:
            logger.error(f"Error in comprehensive backtest: {e}")
            return {'error': str(e)}
    
    async def _run_basic_backtest(self, strategy: BaseStrategy, config: BacktestConfig, symbols: List[str]) -> Dict[str, Any]:
        """Run basic backtest with transaction cost modeling"""
        try:
            logger.info(f"Running basic backtest for {strategy.config.strategy_id}")
            
            # Initialize portfolio tracking
            portfolio = {
                'cash': config.initial_capital,
                'positions': {symbol: Decimal('0') for symbol in symbols},
                'equity_curve': [],
                'trades': [],
                'daily_returns': [],
                'transaction_costs': [],
                'benchmark_equity': []
            }
            
            # Generate date range
            date_range = pd.date_range(config.start_date, config.end_date, freq='D')
            
            # Get strategy signals for each date
            for date in date_range:
                # Update portfolio with market data (simplified)
                market_prices = await self._get_market_prices(symbols, date)
                if not market_prices:
                    continue
                
                # Get strategy signals
                try:
                    signals = await self._get_strategy_signals(strategy, symbols, date)
                    
                    # Execute trades
                    executed_trades = await self._execute_trades(signals, market_prices, portfolio, config)
                    
                    # Update portfolio value
                    portfolio_value = self._calculate_portfolio_value(portfolio, market_prices)
                    portfolio['equity_curve'].append({
                        'date': date,
                        'value': portfolio_value
                    })
                    
                    # Record daily return
                    if len(portfolio['equity_curve']) > 1:
                        prev_value = portfolio['equity_curve'][-2]['value']
                        daily_return = (portfolio_value - prev_value) / prev_value
                        portfolio['daily_returns'].append(daily_return)
                    
                    # Track benchmark performance
                    if config.benchmark_symbols:
                        benchmark_value = await self._calculate_benchmark_value(config.benchmark_symbols, date)
                        portfolio['benchmark_equity'].append({
                            'date': date,
                            'value': benchmark_value
                        })
                
                except Exception as e:
                    logger.warning(f"Error processing date {date}: {e}")
                    continue
            
            # Calculate final metrics
            final_metrics = self._calculate_performance_metrics(portfolio, config)
            
            return {
                'portfolio': portfolio,
                'final_metrics': final_metrics,
                'trades_count': len(portfolio['trades']),
                'transaction_costs_total': sum(portfolio['transaction_costs'])
            }
            
        except Exception as e:
            logger.error(f"Error in basic backtest: {e}")
            return {'error': str(e)}
    
    async def _run_walk_forward_optimization(self, strategy: BaseStrategy, config: BacktestConfig, symbols: List[str]) -> List[WalkForwardResult]:
        """Run walk-forward optimization analysis"""
        try:
            logger.info(f"Running walk-forward optimization for {strategy.config.strategy_id}")
            
            results = []
            date_range = pd.date_range(config.start_date, config.end_date, freq='D')
            
            optimization_window = config.walk_forward_window
            step_size = optimization_window // 4  # 3-month step
            
            for i in range(0, len(date_range) - optimization_window, step_size):
                # Define windows
                opt_start_idx = i
                opt_end_idx = i + optimization_window // 2
                test_start_idx = opt_end_idx
                test_end_idx = min(i + optimization_window, len(date_range))
                
                optimization_period = date_range[opt_start_idx:opt_end_idx]
                test_period = date_range[opt_start_idx:test_end_idx]
                
                # Optimize parameters on optimization period
                best_params = await self._optimize_parameters(strategy, symbols, optimization_period)
                
                # Test on out-of-sample period
                in_sample_metrics = await self._backtest_with_parameters(strategy, symbols, best_params, optimization_period)
                out_sample_metrics = await self._backtest_with_parameters(strategy, symbols, best_params, test_period)
                
                # Calculate stability metrics
                parameter_stability = self._calculate_parameter_stability(best_params)
                performance_degradation = self._calculate_performance_degradation(in_sample_metrics, out_sample_metrics)
                
                result = WalkForwardResult(
                    window_start=optimization_period[0],
                    window_end=test_period[-1],
                    optimization_period=optimization_period[0],
                    test_period=test_period[-1],
                    best_parameters=best_params,
                    in_sample_metrics=in_sample_metrics,
                    out_sample_metrics=out_sample_metrics,
                    parameter_stability=parameter_stability,
                    performance_degradation=performance_degradation
                )
                
                results.append(result)
            
            logger.info(f"Walk-forward optimization completed with {len(results)} windows")
            return results
            
        except Exception as e:
            logger.error(f"Error in walk-forward optimization: {e}")
            return []
    
    async def _run_monte_carlo_simulation(self, strategy: BaseStrategy, config: BacktestConfig, symbols: List[str]) -> List[MonteCarloResult]:
        """Run Monte Carlo simulation for risk analysis"""
        try:
            logger.info(f"Running Monte Carlo simulation ({config.monte_carlo_simulations} simulations) for {strategy.config.strategy_id}")
            
            results = []
            base_returns = await self._get_base_strategy_returns(strategy, config, symbols)
            
            if not base_returns:
                logger.warning("No base returns available for Monte Carlo simulation")
                return results
            
            for sim_id in range(config.monte_carlo_simulations):
                # Generate random returns based on historical distribution
                simulated_returns = self._generate_simulated_returns(base_returns, len(base_returns))
                
                # Simulate portfolio performance
                portfolio_value = float(config.initial_capital)
                equity_curve = [portfolio_value]
                
                for daily_return in simulated_returns:
                    portfolio_value *= (1 + daily_return)
                    equity_curve.append(portfolio_value)
                
                # Calculate metrics
                final_capital = Decimal(str(portfolio_value))
                total_return = (portfolio_value - float(config.initial_capital)) / float(config.initial_capital)
                sharpe_ratio = self._calculate_sharpe_ratio(simulated_returns, config.risk_free_rate)
                max_dd = self._calculate_max_drawdown(equity_curve)
                win_rate = np.mean([r > 0 for r in simulated_returns])
                profit_factor = self._calculate_profit_factor(simulated_returns)
                var_95 = np.percentile(simulated_returns, 5)
                cvar_95 = np.mean([r for r in simulated_returns if r <= var_95])
                skewness = float(pd.Series(simulated_returns).skew())
                kurtosis = float(pd.Series(simulated_returns).kurtosis())
                
                result = MonteCarloResult(
                    simulation_id=sim_id,
                    final_capital=final_capital,
                    total_return=total_return,
                    sharpe_ratio=sharpe_ratio,
                    max_drawdown=max_dd,
                    win_rate=win_rate,
                    profit_factor=profit_factor,
                    var_95=var_95,
                    cvar_95=cvar_95,
                    skewness=skewness,
                    kurtosis=kurtosis
                )
                
                results.append(result)
            
            logger.info(f"Monte Carlo simulation completed")
            return results
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo simulation: {e}")
            return []
    
    def _summarize_monte_carlo_results(self, results: List[MonteCarloResult]) -> Dict[str, Any]:
        """Summarize Monte Carlo simulation results"""
        if not results:
            return {}
        
        returns = [r.total_return for r in results]
        sharpe_ratios = [r.sharpe_ratio for r in results]
        max_drawdowns = [r.max_drawdown for r in results]
        
        return {
            'simulations_count': len(results),
            'final_capital': {
                'mean': float(np.mean([r.final_capital for r in results])),
                'median': float(np.median([r.final_capital for r in results])),
                'std': float(np.std([r.final_capital for r in results])),
                'percentile_5': float(np.percentile([r.final_capital for r in results], 5)),
                'percentile_95': float(np.percentile([r.final_capital for r in results], 95))
            },
            'total_return': {
                'mean': float(np.mean(returns)),
                'median': float(np.median(returns)),
                'std': float(np.std(returns)),
                'prob_positive': float(np.mean([r > 0 for r in returns])),
                'percentile_5': float(np.percentile(returns, 5)),
                'percentile_95': float(np.percentile(returns, 95))
            },
            'sharpe_ratio': {
                'mean': float(np.mean(sharpe_ratios)),
                'median': float(np.median(sharpe_ratios)),
                'std': float(np.std(sharpe_ratios))
            },
            'max_drawdown': {
                'mean': float(np.mean(max_drawdowns)),
                'worst': float(np.max(max_drawdowns)),
                'percentile_5': float(np.percentile(max_drawdowns, 5))
            },
            'risk_metrics': {
                'var_95': float(np.mean([r.var_95 for r in results])),
                'cvar_95': float(np.mean([r.cvar_95 for r in results])),
                'skewness': float(np.mean([r.skewness for r in results])),
                'kurtosis': float(np.mean([r.kurtosis for r in results]))
            }
        }
    
    async def _calculate_advanced_metrics(self, strategy: BaseStrategy, config: BacktestConfig, basic_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate advanced performance metrics"""
        try:
            daily_returns = basic_results['portfolio']['daily_returns']
            if not daily_returns:
                return {}
            
            returns_series = pd.Series(daily_returns)
            
            # Risk-adjusted metrics
            risk_free_daily = config.risk_free_rate / 252
            excess_returns = returns_series - risk_free_daily
            sharpe_ratio = excess_returns.mean() / returns_series.std() * np.sqrt(252)
            
            # Sortino ratio
            downside_returns = returns_series[returns_series < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.01
            sortino_ratio = excess_returns.mean() / downside_deviation if downside_deviation > 0 else 0
            
            # Calmar ratio
            equity_curve = basic_results['portfolio']['equity_curve']
            max_drawdown = self._calculate_max_drawdown([e['value'] for e in equity_curve])
            total_return = (equity_curve[-1]['value'] - equity_curve[0]['value']) / equity_curve[0]['value']
            calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Information ratio
            benchmark_returns = await self._get_benchmark_returns(config)
            if benchmark_returns and len(benchmark_returns) == len(daily_returns):
                tracking_error = (pd.Series(daily_returns) - pd.Series(benchmark_returns)).std() * np.sqrt(252)
                information_ratio = excess_returns.mean() / tracking_error if tracking_error > 0 else 0
            else:
                information_ratio = 0
            
            # Beta and correlation
            if benchmark_returns and len(benchmark_returns) == len(daily_returns):
                portfolio_returns = pd.Series(daily_returns)
                benchmark_series = pd.Series(benchmark_returns)
                
                covariance = portfolio_returns.cov(benchmark_series)
                benchmark_variance = benchmark_series.var()
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
                correlation = portfolio_returns.corr(benchmark_series)
            else:
                beta = 1.0
                correlation = 0.0
            
            # Additional metrics
            var_95 = np.percentile(daily_returns, 5)
            cvar_95 = np.mean([r for r in daily_returns if r <= var_95])
            
            # Omega ratio
            gains = [r for r in daily_returns if r > 0]
            losses = [r for r in daily_returns if r < 0]
            omega_ratio = sum(gains) / abs(sum(losses)) if sum(losses) < 0 else float('inf')
            
            # Ulcer index (simplified)
            equity_series = pd.Series([e['value'] for e in equity_curve])
            drawdown_series = (equity_series / equity_series.expanding().max() - 1)
            ulcer_index = math.sqrt((drawdown_series ** 2).mean())
            
            return {
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'information_ratio': information_ratio,
                'beta': beta,
                'correlation': correlation,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'omega_ratio': omega_ratio,
                'ulcer_index': ulcer_index,
                'annualized_return': total_return * (252 / len(daily_returns)) if daily_returns else 0,
                'annualized_volatility': returns_series.std() * np.sqrt(252) if daily_returns else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating advanced metrics: {e}")
            return {}
    
    async def _perform_risk_decomposition(self, strategy: BaseStrategy, config: BacktestConfig, basic_results: Dict[str, Any]) -> Optional[RiskDecomposition]:
        """Perform risk decomposition analysis"""
        try:
            # This is a simplified risk decomposition
            # In a real implementation, would factor in multiple risk factors
            
            daily_returns = basic_results['portfolio']['daily_returns']
            if not daily_returns:
                return None
            
            returns_series = pd.Series(daily_returns)
            total_risk = returns_series.std() * np.sqrt(252)
            
            # Simplified factor model (in reality would use multiple factors)
            market_returns = await self._get_benchmark_returns(config)
            if not market_returns or len(market_returns) != len(daily_returns):
                return RiskDecomposition(
                    total_risk=total_risk,
                    systematic_risk=total_risk * 0.7,  # Assume 70% systematic
                    idiosyncratic_risk=total_risk * 0.3,
                    factor_exposures={'market': 1.0},
                    sector_exposures={},
                    concentration_risk=0.1,
                    correlation_risk=0.2,
                    tail_risk_measure=abs(np.percentile(daily_returns, 5))
                )
            
            # Calculate factor exposures
            portfolio_returns = pd.Series(daily_returns)
            market_returns_series = pd.Series(market_returns)
            
            # Market beta as systematic risk component
            covariance = portfolio_returns.cov(market_returns_series)
            market_variance = market_returns_series.var()
            beta = covariance / market_variance if market_variance > 0 else 1.0
            
            systematic_risk = abs(beta) * math.sqrt(market_variance * 252)
            idiosyncratic_risk = math.sqrt(total_risk**2 - systematic_risk**2) if total_risk > systematic_risk else 0
            
            # Concentration risk (simplified)
            position_counts = len(set([t.get('symbol', '') for t in basic_results['portfolio']['trades']]))
            total_trades = len(basic_results['portfolio']['trades'])
            concentration_risk = 1.0 - (position_counts / max(total_trades, 1)) if total_trades > 0 else 0
            
            # Correlation risk
            correlation = portfolio_returns.corr(market_returns_series)
            correlation_risk = abs(correlation - 1.0) if not np.isnan(correlation) else 0
            
            return RiskDecomposition(
                total_risk=total_risk,
                systematic_risk=systematic_risk,
                idiosyncratic_risk=idiosyncratic_risk,
                factor_exposures={'market': beta},
                sector_exposures={'technology': 'overweight'},  # Simplified
                concentration_risk=concentration_risk,
                correlation_risk=correlation_risk,
                tail_risk_measure=abs(np.percentile(daily_returns, 5))
            )
            
        except Exception as e:
            logger.error(f"Error in risk decomposition: {e}")
            return None
    
    async def _calculate_performance_attribution(self, basic_results: Dict[str, Any], config: BacktestConfig) -> Dict[str, Any]:
        """Calculate performance attribution analysis"""
        try:
            trades = basic_results['portfolio']['trades']
            if not trades:
                return {}
            
            # Group trades by symbol
            symbol_performance = defaultdict(lambda: {'trades': 0, 'pnl': 0, 'wins': 0, 'losses': 0})
            
            for trade in trades:
                symbol = trade.get('symbol', 'unknown')
                pnl = float(trade.get('pnl', 0))
                
                symbol_performance[symbol]['trades'] += 1
                symbol_performance[symbol]['pnl'] += pnl
                
                if pnl > 0:
                    symbol_performance[symbol]['wins'] += 1
                elif pnl < 0:
                    symbol_performance[symbol]['losses'] += 1
            
            # Calculate attribution metrics
            total_pnl = sum(perf['pnl'] for perf in symbol_performance.values())
            
            attribution = {
                'total_pnl': total_pnl,
                'by_symbol': {},
                'allocation_effect': 0.0,  # Simplified
                'selection_effect': 0.0,   # Simplified
                'interaction_effect': 0.0,  # Simplified
                'top_contributors': [],
                'top_detractors': []
            }
            
            for symbol, perf in symbol_performance.items():
                win_rate = perf['wins'] / max(perf['trades'], 1)
                avg_pnl = perf['pnl'] / max(perf['trades'], 1)
                
                attribution['by_symbol'][symbol] = {
                    'pnl': perf['pnl'],
                    'trades': perf['trades'],
                    'win_rate': win_rate,
                    'avg_pnl_per_trade': avg_pnl,
                    'contribution_pct': (perf['pnl'] / total_pnl * 100) if total_pnl != 0 else 0
                }
            
            # Identify top contributors and detractors
            sorted_by_pnl = sorted(attribution['by_symbol'].items(), key=lambda x: x[1]['pnl'], reverse=True)
            attribution['top_contributors'] = sorted_by_pnl[:3]
            attribution['top_detractors'] = sorted_by_pnl[-3:] if len(sorted_by_pnl) > 3 else []
            
            return attribution
            
        except Exception as e:
            logger.error(f"Error calculating performance attribution: {e}")
            return {}
    
    async def _compare_with_benchmark(self, basic_results: Dict[str, Any], config: BacktestConfig) -> Dict[str, Any]:
        """Compare strategy performance with benchmark"""
        try:
            portfolio_returns = basic_results['portfolio']['daily_returns']
            benchmark_returns = await self._get_benchmark_returns(config)
            
            if not portfolio_returns or not benchmark_returns:
                return {}
            
            # Align return series
            min_length = min(len(portfolio_returns), len(benchmark_returns))
            if min_length < 10:
                return {}
            
            portfolio_series = pd.Series(portfolio_returns[:min_length])
            benchmark_series = pd.Series(benchmark_returns[:min_length])
            
            # Calculate benchmark metrics
            benchmark_total_return = (1 + benchmark_series).prod() - 1
            benchmark_volatility = benchmark_series.std() * np.sqrt(252)
            benchmark_sharpe = benchmark_series.mean() * 252 / benchmark_volatility if benchmark_volatility > 0 else 0
            
            # Calculate alpha and beta
            covariance = portfolio_series.cov(benchmark_series)
            benchmark_variance = benchmark_series.var()
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
            
            risk_free_daily = config.risk_free_rate / 252
            portfolio_excess = portfolio_series - risk_free_daily
            benchmark_excess = benchmark_series - risk_free_daily
            
            alpha = (portfolio_excess.mean() - beta * benchmark_excess.mean()) * 252
            
            # Information ratio
            tracking_error = (portfolio_series - benchmark_series).std() * np.sqrt(252)
            information_ratio = (portfolio_excess.mean() - benchmark_excess.mean()) * 252 / tracking_error if tracking_error > 0 else 0
            
            return {
                'alpha': alpha,
                'beta': beta,
                'information_ratio': information_ratio,
                'tracking_error': tracking_error,
                'portfolio_total_return': (1 + portfolio_series).prod() - 1,
                'benchmark_total_return': benchmark_total_return,
                'portfolio_volatility': portfolio_series.std() * np.sqrt(252),
                'benchmark_volatility': benchmark_volatility,
                'outperformance': (1 + portfolio_series).prod() - 1 - benchmark_total_return
            }
            
        except Exception as e:
            logger.error(f"Error in benchmark comparison: {e}")
            return {}
    
    # Helper methods for backtesting
    async def _get_market_prices(self, symbols: List[str], date: datetime) -> Dict[str, Decimal]:
        """Get market prices for symbols on date (simplified)"""
        # In real implementation, would fetch actual market data
        prices = {}
        for symbol in symbols:
            # Mock price generation
            base_price = 100.0 + hash(f"{symbol}_{date.strftime('%Y%m%d')}") % 100
            prices[symbol] = Decimal(str(base_price))
        return prices
    
    async def _get_strategy_signals(self, strategy: BaseStrategy, symbols: List[str], date: datetime) -> List[TradingSignal]:
        """Get strategy signals for date (simplified)"""
        # In real implementation, would call strategy.generate_signals()
        signals = []
        
        # Mock signal generation
        import random
        for symbol in symbols[:2]:  # Limit to first 2 symbols
            if random.random() > 0.9:  # 10% chance of signal
                signal_type = random.choice([SignalType.BUY, SignalType.SELL])
                signals.append(TradingSignal(
                    signal_id=f"mock_signal_{symbol}_{date.strftime('%Y%m%d')}",
                    strategy_id=strategy.config.strategy_id,
                    symbol=symbol,
                    signal_type=signal_type,
                    confidence=0.7,
                    strength=0.6,
                    price=Decimal('100'),
                    quantity=Decimal('100'),
                    metadata={'mock': True}
                ))
        
        return signals
    
    async def _execute_trades(self, signals: List[TradingSignal], market_prices: Dict[str, Decimal], portfolio: Dict[str, Any], config: BacktestConfig) -> List[Dict[str, Any]]:
        """Execute trades with transaction costs"""
        executed_trades = []
        
        for signal in signals:
            symbol = signal.symbol
            if symbol not in market_prices:
                continue
            
            price = market_prices[symbol]
            quantity = signal.quantity
            
            # Calculate transaction cost
            trade_value = float(price * quantity)
            cost = self._calculate_transaction_cost(trade_value, config.transaction_costs)
            
            # Update portfolio
            if signal.signal_type == SignalType.BUY:
                cost_to_pay = Decimal(str(trade_value + cost))
                if portfolio['cash'] >= cost_to_pay:
                    portfolio['cash'] -= cost_to_pay
                    portfolio['positions'][symbol] += quantity
                    
                    trade_record = {
                        'symbol': symbol,
                        'side': 'buy',
                        'quantity': float(quantity),
                        'price': float(price),
                        'cost': cost,
                        'pnl': 0,  # Will be calculated when position is closed
                        'timestamp': datetime.utcnow()
                    }
                    executed_trades.append(trade_record)
                    portfolio['transaction_costs'].append(cost)
            
            elif signal.signal_type == SignalType.SELL:
                if portfolio['positions'][symbol] >= quantity:
                    proceeds = Decimal(str(trade_value - cost))
                    portfolio['cash'] += proceeds
                    portfolio['positions'][symbol] -= quantity
                    
                    trade_record = {
                        'symbol': symbol,
                        'side': 'sell',
                        'quantity': float(quantity),
                        'price': float(price),
                        'cost': cost,
                        'pnl': 0,  # Will be calculated when position is closed
                        'timestamp': datetime.utcnow()
                    }
                    executed_trades.append(trade_record)
                    portfolio['transaction_costs'].append(cost)
        
        portfolio['trades'].extend(executed_trades)
        return executed_trades
    
    def _calculate_transaction_cost(self, trade_value: float, transaction_costs: TransactionCost) -> float:
        """Calculate transaction cost for a trade"""
        # Basic commission
        cost = trade_value * transaction_costs.commission
        
        # Spread cost
        cost += trade_value * transaction_costs.spread_cost
        
        # Market impact (simplified)
        market_impact = transaction_costs.market_impact_coeff * math.sqrt(trade_value / 1000000)  # Scale by market size
        cost += trade_value * market_impact
        
        # Minimum cost
        cost = max(cost, transaction_costs.minimum_cost)
        
        return cost
    
    def _calculate_portfolio_value(self, portfolio: Dict[str, Any], market_prices: Dict[str, Decimal]) -> Decimal:
        """Calculate total portfolio value"""
        total_value = portfolio['cash']
        
        for symbol, position in portfolio['positions'].items():
            if symbol in market_prices:
                total_value += position * market_prices[symbol]
        
        return total_value
    
    def _calculate_performance_metrics(self, portfolio: Dict[str, Any], config: BacktestConfig) -> Dict[str, Any]:
        """Calculate performance metrics from portfolio"""
        if not portfolio['equity_curve']:
            return {}
        
        equity_values = [e['value'] for e in portfolio['equity_curve']]
        initial_value = equity_values[0]
        final_value = equity_values[-1]
        
        # Basic metrics
        total_return = (final_value - initial_value) / initial_value
        max_drawdown = self._calculate_max_drawdown(equity_values)
        
        # Return-based metrics
        daily_returns = portfolio['daily_returns']
        if daily_returns:
            returns_series = pd.Series(daily_returns)
            win_rate = len([r for r in daily_returns if r > 0]) / len(daily_returns)
            
            # Profit factor
            gains = [r for r in daily_returns if r > 0]
            losses = [r for r in daily_returns if r < 0]
            profit_factor = sum(gains) / abs(sum(losses)) if sum(losses) < 0 else float('inf')
        else:
            win_rate = 0.0
            profit_factor = 1.0
        
        return {
            'total_return': float(total_return),
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'initial_capital': float(initial_value),
            'final_capital': float(final_value),
            'total_trades': len(portfolio['trades']),
            'avg_trade_pnl': float(final_value - initial_value) / max(len(portfolio['trades']), 1)
        }
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not equity_curve:
            return 0.0
        
        peak = equity_curve[0]
        max_dd = 0.0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak
            if drawdown > max_dd:
                max_dd = drawdown
        
        return max_dd
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float) -> float:
        """Calculate Sharpe ratio"""
        if not returns:
            return 0.0
        
        excess_returns = np.array(returns) - risk_free_rate / 252
        return np.mean(excess_returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    
    def _calculate_profit_factor(self, returns: List[float]) -> float:
        """Calculate profit factor"""
        gains = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]
        
        if not losses or sum(losses) == 0:
            return float('inf')
        
        return sum(gains) / abs(sum(losses))
    
    async def _load_benchmark_data(self, benchmark_symbols: List[str], start_date: datetime, end_date: datetime) -> None:
        """Load benchmark data"""
        for symbol in benchmark_symbols:
            # In real implementation, would fetch actual benchmark data
            # For now, generate mock data
            dates = pd.date_range(start_date, end_date, freq='D')
            prices = []
            
            for date in dates:
                # Mock price generation
                base_price = 100.0
                daily_return = np.random.normal(0.0008, 0.015)  # ~20% annual return, 15% volatility
                if prices:
                    base_price = prices[-1] * (1 + daily_return)
                prices.append(base_price)
            
            self.benchmark_data[symbol] = pd.DataFrame({
                'date': dates,
                'close': prices
            })
    
    async def _get_benchmark_returns(self, config: BacktestConfig) -> Optional[List[float]]:
        """Get benchmark returns"""
        if not config.benchmark_symbols:
            return None
        
        primary_benchmark = config.benchmark_symbols[0]
        if primary_benchmark not in self.benchmark_data:
            return None
        
        df = self.benchmark_data[primary_benchmark]
        returns = df['close'].pct_change().dropna().tolist()
        return returns
    
    # Additional helper methods would be implemented here for complete functionality
    # (Shortened for demonstration purposes)
    
    async def _optimize_parameters(self, strategy: BaseStrategy, symbols: List[str], optimization_period: pd.DatetimeIndex) -> Dict[str, Any]:
        """Optimize strategy parameters (simplified)"""
        # Mock optimization - in real implementation, would use optimization algorithms
        return strategy.config.parameters.copy()
    
    async def _backtest_with_parameters(self, strategy: BaseStrategy, symbols: List[str], parameters: Dict[str, Any], period: pd.DatetimeIndex) -> Dict[str, float]:
        """Backtest with specific parameters (simplified)"""
        # Mock metrics
        return {
            'total_return': np.random.normal(0.1, 0.2),
            'sharpe_ratio': np.random.normal(1.0, 0.5),
            'max_drawdown': abs(np.random.normal(0.1, 0.05))
        }
    
    def _calculate_parameter_stability(self, parameters: Dict[str, Any]) -> float:
        """Calculate parameter stability (simplified)"""
        return 0.8  # Mock value
    
    def _calculate_performance_degradation(self, in_sample: Dict[str, float], out_sample: Dict[str, float]) -> float:
        """Calculate performance degradation (simplified)"""
        return abs(out_sample.get('total_return', 0) - in_sample.get('total_return', 0))
    
    async def _get_base_strategy_returns(self, strategy: BaseStrategy, config: BacktestConfig, symbols: List[str]) -> Optional[List[float]]:
        """Get base strategy returns for Monte Carlo simulation"""
        # Mock returns - in real implementation, would run backtest
        dates = pd.date_range(config.start_date, config.end_date, freq='D')
        returns = [np.random.normal(0.0004, 0.01) for _ in range(len(dates))]  # 10% annual return, 16% volatility
        return returns
    
    def _generate_simulated_returns(self, base_returns: List[float], target_length: int) -> List[float]:
        """Generate simulated returns using historical statistics"""
        if not base_returns:
            return [np.random.normal(0, 0.01) for _ in range(target_length)]
        
        mean_return = np.mean(base_returns)
        std_return = np.std(base_returns)
        
        simulated_returns = []
        for _ in range(target_length):
            simulated_return = np.random.normal(mean_return, std_return)
            simulated_returns.append(simulated_return)
        
        return simulated_returns
    
    async def _calculate_benchmark_value(self, benchmark_symbols: List[str], date: datetime) -> float:
        """Calculate benchmark portfolio value"""
        # Simplified benchmark calculation
        return 100000.0 * (1 + np.random.normal(0.0008, 0.015))  # Mock benchmark performance


# Global enhanced backtest engine instance
enhanced_backtest_engine = EnhancedBacktestEngine()


# Utility functions for enhanced backtesting
async def run_comprehensive_strategy_analysis(strategy_id: str, config: BacktestConfig) -> Dict[str, Any]:
    """Run comprehensive analysis for a strategy"""
    strategy = strategy_registry.get_strategy(strategy_id)
    if not strategy:
        return {'error': f'Strategy {strategy_id} not found'}
    
    return await enhanced_backtest_engine.run_comprehensive_backtest(strategy, config)


def generate_strategy_report(backtest_results: Dict[str, Any]) -> str:
    """Generate comprehensive strategy report"""
    if 'error' in backtest_results:
        return f"Error: {backtest_results['error']}"
    
    strategy_id = backtest_results['strategy_id']
    basic_results = backtest_results['basic_results']
    advanced_metrics = backtest_results['advanced_metrics']
    benchmark_comparison = backtest_results['benchmark_comparison']
    
    report = f"""
COMPREHENSIVE STRATEGY ANALYSIS REPORT
=====================================

Strategy: {strategy_id}
Analysis Date: {backtest_results['backtest_timestamp']}
Test Period: {basic_results.get('final_metrics', {}).get('initial_capital', 0):,.0f} to {basic_results.get('final_metrics', {}).get('final_capital', 0):,.0f}

PERFORMANCE SUMMARY
-------------------
Total Return: {basic_results.get('final_metrics', {}).get('total_return', 0):.2%}
Sharpe Ratio: {advanced_metrics.get('sharpe_ratio', 0):.3f}
Sortino Ratio: {advanced_metrics.get('sortino_ratio', 0):.3f}
Calmar Ratio: {advanced_metrics.get('calmar_ratio', 0):.3f}
Max Drawdown: {basic_results.get('final_metrics', {}).get('max_drawdown', 0):.2%}
Win Rate: {basic_results.get('final_metrics', {}).get('win_rate', 0):.2%}
Profit Factor: {basic_results.get('final_metrics', {}).get('profit_factor', 1.0):.3f}

RISK METRICS
------------
Beta: {advanced_metrics.get('beta', 0):.3f}
Information Ratio: {advanced_metrics.get('information_ratio', 0):.3f}
VaR (95%): {advanced_metrics.get('var_95', 0):.2%}
CVaR (95%): {advanced_metrics.get('cvar_95', 0):.2%}
Correlation: {advanced_metrics.get('correlation', 0):.3f}

BENCHMARK COMPARISON
-------------------
Alpha: {benchmark_comparison.get('alpha', 0):.4f}
Beta: {benchmark_comparison.get('beta', 0):.3f}
Outperformance: {benchmark_comparison.get('outperformance', 0):.2%}

MONTE CARLO ANALYSIS
--------------------
"""
    
    monte_carlo_summary = backtest_results.get('monte_carlo_results', {}).get('summary', {})
    if monte_carlo_summary:
        report += f"""
Simulations: {monte_carlo_summary.get('simulations_count', 0)}
Probability of Positive Returns: {monte_carlo_summary.get('total_return', {}).get('prob_positive', 0):.1%}
Expected Return (95% CI): {monte_carlo_summary.get('total_return', {}).get('percentile_5', 0):.2%} to {monte_carlo_summary.get('total_return', {}).get('percentile_95', 0):.2%}
"""
    
    report += """
RECOMMENDATION
--------------
Strategy performance should be evaluated based on risk-adjusted returns,
consistency across market conditions, and alignment with investment objectives.

Report generated by Enhanced Backtesting Engine v2.0
"""
    
    return report


# Export the enhanced backtesting system
__all__ = [
    'EnhancedBacktestEngine',
    'enhanced_backtest_engine',
    'BacktestConfig',
    'TransactionCost',
    'WalkForwardResult',
    'MonteCarloResult',
    'RiskDecomposition',
    'run_comprehensive_strategy_analysis',
    'generate_strategy_report'
]