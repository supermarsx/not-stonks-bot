"""
Performance Analytics Module

Comprehensive portfolio performance analysis including attribution,
benchmarking, risk-adjusted metrics, and performance decomposition.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..core.config import AnalyticsConfig

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics container"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    information_ratio: float
    tracking_error: float
    beta: float
    alpha: float
    correlation: float
    var_95: float
    cvar_95: float
    skewness: float
    kurtosis: float
    win_rate: float
    profit_factor: float


@dataclass
class PerformanceAttribution:
    """Performance attribution results"""
    total_attribution: float
    selection_effect: float
    allocation_effect: float
    interaction_effect: float
    sector_attribution: Dict[str, float]
    security_selection_attribution: float
    timing_attribution: float
    currency_effect: float


@dataclass
class RollingMetrics:
    """Rolling performance metrics"""
    window: int
    sharpe_series: pd.Series
    max_drawdown_series: pd.Series
    var_series: pd.Series
    correlation_series: pd.Series
    beta_series: pd.Series


class PerformanceAnalytics:
    """
    Advanced Performance Analytics
    
    Provides comprehensive portfolio performance analysis including:
    - Portfolio performance attribution analysis
    - Strategy performance decomposition
    - Risk-adjusted returns analysis
    - Rolling performance metrics
    - Performance benchmarking vs market indices
    - Seasonal and cyclical analysis
    - Outperformance analysis
    """
    
    def __init__(self, config: AnalyticsConfig):
        """Initialize performance analytics"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Performance cache for faster calculations
        self._performance_cache = {}
        
        # Default benchmark indices
        self.benchmarks = config.benchmark_indices or [
            'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VEA', 'VWO'
        ]
        
        # Risk-free rate
        self.risk_free_rate = config.risk_free_rate
        
        self.logger.info("Performance Analytics initialized")
    
    async def analyze_portfolio_performance(self,
                                          portfolio_id: str,
                                          period: str = "1Y") -> Dict[str, Any]:
        """
        Comprehensive portfolio performance analysis
        
        Args:
            portfolio_id: Portfolio identifier
            period: Analysis period (1D, 1W, 1M, 3M, 6M, 1Y, YTD, ALL)
            
        Returns:
            Comprehensive performance analysis
        """
        try:
            self.logger.info(f"Analyzing portfolio performance: {portfolio_id}, period: {period}")
            
            # Get portfolio data
            portfolio_data = await self._get_portfolio_data(portfolio_id, period)
            benchmark_data = await self._get_benchmark_data(period)
            
            # Calculate basic performance metrics
            metrics = await self._calculate_performance_metrics(portfolio_data, benchmark_data)
            
            # Calculate attribution analysis
            attribution = await self._calculate_performance_attribution(portfolio_data, benchmark_data)
            
            # Calculate rolling metrics
            rolling_metrics = await self._calculate_rolling_metrics(portfolio_data, benchmark_data)
            
            # Calculate seasonal analysis
            seasonal_analysis = await self._calculate_seasonal_analysis(portfolio_data)
            
            # Calculate strategy decomposition
            strategy_decomposition = await self._calculate_strategy_decomposition(portfolio_data)
            
            # Calculate outperformance analysis
            outperformance = await self._calculate_outperformance_analysis(portfolio_data, benchmark_data)
            
            return {
                'portfolio_id': portfolio_id,
                'period': period,
                'performance_metrics': metrics.__dict__,
                'attribution': attribution.__dict__,
                'rolling_metrics': {
                    'sharpe_ratio_series': rolling_metrics.sharpe_series.to_dict(),
                    'max_drawdown_series': rolling_metrics.max_drawdown_series.to_dict(),
                    'var_series': rolling_metrics.var_series.to_dict(),
                    'correlation_series': rolling_metrics.correlation_series.to_dict(),
                    'beta_series': rolling_metrics.beta_series.to_dict()
                },
                'seasonal_analysis': seasonal_analysis,
                'strategy_decomposition': strategy_decomposition,
                'outperformance_analysis': outperformance,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing portfolio performance: {e}")
            raise
    
    async def _get_portfolio_data(self, portfolio_id: str, period: str) -> pd.DataFrame:
        """Get portfolio historical data"""
        # Mock implementation - replace with actual data retrieval
        # In real implementation, this would fetch from database
        
        date_range = self._get_date_range(period)
        dates = pd.date_range(start=date_range[0], end=date_range[1], freq='D')
        
        # Generate mock portfolio returns
        np.random.seed(42)  # For reproducible results
        n_days = len(dates)
        returns = np.random.normal(0.001, 0.02, n_days)  # ~0.25% daily return, 10% annualized volatility
        
        portfolio_data = pd.DataFrame({
            'date': dates,
            'portfolio_return': returns,
            'portfolio_value': 100000 * (1 + returns).cumprod(),
            'benchmark_return': np.random.normal(0.0008, 0.015, n_days),
            'portfolio_pnl': returns * 100000
        })
        
        return portfolio_data
    
    async def _get_benchmark_data(self, period: str) -> pd.DataFrame:
        """Get benchmark indices data"""
        date_range = self._get_date_range(period)
        dates = pd.date_range(start=date_range[0], end=date_range[1], freq='D')
        
        # Mock benchmark data
        benchmark_data = {}
        for benchmark in self.benchmarks:
            benchmark_data[benchmark] = np.random.normal(0.0008, 0.015, len(dates))
        
        df = pd.DataFrame(benchmark_data, index=dates)
        df['date'] = dates
        return df
    
    def _get_date_range(self, period: str) -> Tuple[datetime, datetime]:
        """Get date range for period"""
        end_date = datetime.now()
        
        if period == "1D":
            start_date = end_date - timedelta(days=1)
        elif period == "1W":
            start_date = end_date - timedelta(weeks=1)
        elif period == "1M":
            start_date = end_date - timedelta(days=30)
        elif period == "3M":
            start_date = end_date - timedelta(days=90)
        elif period == "6M":
            start_date = end_date - timedelta(days=180)
        elif period == "1Y":
            start_date = end_date - timedelta(days=365)
        elif period == "YTD":
            start_date = datetime(end_date.year, 1, 1)
        else:
            start_date = end_date - timedelta(days=365 * 5)  # 5 years default
        
        return start_date, end_date
    
    async def _calculate_performance_metrics(self,
                                           portfolio_data: pd.DataFrame,
                                           benchmark_data: pd.DataFrame) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        try:
            returns = portfolio_data['portfolio_return']
            benchmark_returns = portfolio_data['benchmark_return']
            
            # Basic metrics
            total_return = (portfolio_data['portfolio_value'].iloc[-1] / portfolio_data['portfolio_value'].iloc[0]) - 1
            
            # Annualized return (assuming daily data)
            periods_per_year = 252
            annualized_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1
            
            # Volatility
            volatility = returns.std() * np.sqrt(periods_per_year)
            
            # Sharpe ratio
            excess_returns = returns - self.risk_free_rate / periods_per_year
            sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)
            
            # Sortino ratio
            downside_returns = returns[returns < 0]
            downside_volatility = downside_returns.std() * np.sqrt(periods_per_year)
            sortino_ratio = annualized_return / downside_volatility if downside_volatility > 0 else np.inf
            
            # Maximum drawdown
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else np.inf
            
            # Information ratio
            active_returns = returns - benchmark_returns
            tracking_error = active_returns.std() * np.sqrt(periods_per_year)
            information_ratio = active_returns.mean() / active_returns.std() * np.sqrt(periods_per_year)
            
            # Alpha and Beta
            covariance = np.cov(returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
            
            # CAPM alpha
            expected_return = self.risk_free_rate + beta * (benchmark_returns.mean() * periods_per_year - self.risk_free_rate)
            alpha = annualized_return - expected_return
            
            # Correlation
            correlation = np.corrcoef(returns, benchmark_returns)[0, 1]
            
            # VaR and CVaR (95% confidence)
            var_95 = np.percentile(returns, 5)
            cvar_95 = returns[returns <= var_95].mean()
            
            # Higher moments
            skewness = stats.skew(returns)
            kurtosis = stats.kurtosis(returns)
            
            # Trading metrics (simplified)
            win_rate = (returns > 0).mean()
            
            # Profit factor
            gross_profit = returns[returns > 0].sum()
            gross_loss = abs(returns[returns < 0].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
            
            return PerformanceMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                calmar_ratio=calmar_ratio,
                information_ratio=information_ratio,
                tracking_error=tracking_error,
                beta=beta,
                alpha=alpha,
                correlation=correlation,
                var_95=var_95,
                cvar_95=cvar_95,
                skewness=skewness,
                kurtosis=kurtosis,
                win_rate=win_rate,
                profit_factor=profit_factor
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            raise
    
    async def _calculate_performance_attribution(self,
                                               portfolio_data: pd.DataFrame,
                                               benchmark_data: pd.DataFrame) -> PerformanceAttribution:
        """Calculate Brinson-Fachler performance attribution"""
        try:
            # Mock sector data for demonstration
            portfolio_weights = {
                'Technology': 0.3,
                'Healthcare': 0.2,
                'Financials': 0.15,
                'Consumer_Discretionary': 0.15,
                'Industrials': 0.1,
                'Energy': 0.05,
                'Materials': 0.05
            }
            
            benchmark_weights = {
                'Technology': 0.25,
                'Healthcare': 0.15,
                'Financials': 0.12,
                'Consumer_Discretionary': 0.12,
                'Industrials': 0.13,
                'Energy': 0.08,
                'Materials': 0.15
            }
            
            # Mock sector returns
            sector_returns = {
                'Technology': 0.15,
                'Healthcare': 0.12,
                'Financials': 0.08,
                'Consumer_Discretionary': 0.10,
                'Industrials': 0.06,
                'Energy': -0.02,
                'Materials': 0.04
            }
            
            # Calculate attribution effects
            selection_effect = 0.0
            allocation_effect = 0.0
            
            for sector in portfolio_weights.keys():
                if sector in benchmark_weights and sector in sector_returns:
                    w_p = portfolio_weights[sector]
                    w_b = benchmark_weights[sector]
                    r_s = sector_returns[sector]
                    r_b = np.mean(list(sector_returns.values()))  # Benchmark average return
                    
                    selection_effect += w_p * (r_s - r_b)
                    allocation_effect += (w_p - w_b) * r_b
            
            interaction_effect = 0.0  # Simplified
            currency_effect = 0.001   # Mock small currency effect
            timing_attribution = 0.002  # Mock timing effect
            security_selection_attribution = selection_effect * 0.8  # Portion from security selection
            
            total_attribution = selection_effect + allocation_effect + interaction_effect + currency_effect
            
            return PerformanceAttribution(
                total_attribution=total_attribution,
                selection_effect=selection_effect,
                allocation_effect=allocation_effect,
                interaction_effect=interaction_effect,
                sector_attribution={
                    sector: portfolio_weights[sector] * (sector_returns.get(sector, 0) - 0.08)
                    for sector in portfolio_weights.keys()
                },
                security_selection_attribution=security_selection_attribution,
                timing_attribution=timing_attribution,
                currency_effect=currency_effect
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating performance attribution: {e}")
            raise
    
    async def _calculate_rolling_metrics(self,
                                       portfolio_data: pd.DataFrame,
                                       benchmark_data: pd.DataFrame,
                                       window: int = 60) -> RollingMetrics:
        """Calculate rolling performance metrics"""
        try:
            returns = portfolio_data['portfolio_return']
            benchmark_returns = portfolio_data['benchmark_return']
            
            # Rolling Sharpe ratio
            rolling_mean = returns.rolling(window=window).mean()
            rolling_std = returns.rolling(window=window).std()
            sharpe_series = (rolling_mean / rolling_std) * np.sqrt(252)
            
            # Rolling maximum drawdown
            def calculate_rolling_drawdown(returns_series):
                cumulative = (1 + returns_series).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                return drawdown
            
            max_drawdown_series = returns.rolling(window=window).apply(
                lambda x: calculate_rolling_drawdown(x).min(), raw=False
            )
            
            # Rolling VaR
            var_series = returns.rolling(window=window).quantile(0.05)
            
            # Rolling correlation with benchmark
            rolling_corr = returns.rolling(window=window).corr(benchmark_returns)
            correlation_series = rolling_corr
            
            # Rolling beta
            def calculate_rolling_beta(port_returns, bench_returns):
                if len(port_returns) < 2:
                    return 1.0
                covariance = np.cov(port_returns, bench_returns)[0, 1]
                bench_variance = np.var(bench_returns)
                return covariance / bench_variance if bench_variance > 0 else 1.0
            
            beta_series = returns.rolling(window=window).apply(
                lambda x: calculate_rolling_beta(x, benchmark_returns.iloc[x.index[-window:]]),
                raw=False
            )
            
            return RollingMetrics(
                window=window,
                sharpe_series=sharpe_series,
                max_drawdown_series=max_drawdown_series,
                var_series=var_series,
                correlation_series=correlation_series,
                beta_series=beta_series
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating rolling metrics: {e}")
            raise
    
    async def _calculate_seasonal_analysis(self, portfolio_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate seasonal and cyclical analysis"""
        try:
            df = portfolio_data.copy()
            df['month'] = df['date'].dt.month
            df['day_of_week'] = df['date'].dt.dayofweek
            df['quarter'] = df['date'].dt.quarter
            
            # Monthly seasonality
            monthly_returns = df.groupby('month')['portfolio_return'].mean().to_dict()
            
            # Day of week seasonality
            dow_returns = df.groupby('day_of_week')['portfolio_return'].mean().to_dict()
            
            # Quarterly seasonality
            quarterly_returns = df.groupby('quarter')['portfolio_return'].mean().to_dict()
            
            # Calendar effects analysis
            first_trading_day_month = df.groupby(df['date'].dt.date)['portfolio_return'].first()
            last_trading_day_month = df.groupby(df['date'].dt.date)['portfolio_return'].last()
            
            month_effect = {
                'first_trading_day': first_trading_day_month.mean(),
                'last_trading_day': last_trading_day_month.mean(),
                'first_day_premium': first_trading_day_month.mean() - df['portfolio_return'].mean(),
                'last_day_premium': last_trading_day_month.mean() - df['portfolio_return'].mean()
            }
            
            return {
                'monthly_returns': monthly_returns,
                'day_of_week_returns': dow_returns,
                'quarterly_returns': quarterly_returns,
                'calendar_effects': month_effect,
                'seasonality_strength': np.std(list(monthly_returns.values())) / df['portfolio_return'].std()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating seasonal analysis: {e}")
            raise
    
    async def _calculate_strategy_decomposition(self, portfolio_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate strategy performance decomposition"""
        try:
            # Mock strategy performance decomposition
            strategies = {
                'Momentum_Strategy': {
                    'allocation': 0.4,
                    'return': 0.18,
                    'risk': 0.15,
                    'sharpe_ratio': 1.2
                },
                'Mean_Reversion_Strategy': {
                    'allocation': 0.3,
                    'return': 0.12,
                    'risk': 0.10,
                    'sharpe_ratio': 1.1
                },
                'Arbitrage_Strategy': {
                    'allocation': 0.2,
                    'return': 0.08,
                    'risk': 0.05,
                    'sharpe_ratio': 1.6
                },
                'Trend_Following_Strategy': {
                    'allocation': 0.1,
                    'return': 0.15,
                    'risk': 0.18,
                    'sharpe_ratio': 0.83
                }
            }
            
            # Calculate weighted contributions
            total_return = 0.0
            total_risk = 0.0
            strategy_contributions = {}
            
            for strategy_name, strategy_data in strategies.items():
                allocation = strategy_data['allocation']
                strategy_return = strategy_data['return']
                strategy_risk = strategy_data['risk']
                
                contribution = allocation * strategy_return
                risk_contribution = allocation * strategy_risk
                
                strategy_contributions[strategy_name] = {
                    'return_contribution': contribution,
                    'risk_contribution': risk_contribution,
                    'allocation': allocation,
                    'strategy_return': strategy_return,
                    'strategy_risk': strategy_risk
                }
                
                total_return += contribution
                total_risk += risk_contribution
            
            # Correlation analysis between strategies
            strategy_returns = pd.DataFrame({
                name: np.random.normal(data['return']/252, data['risk']/np.sqrt(252), 252)
                for name, data in strategies.items()
            })
            
            correlation_matrix = strategy_returns.corr().to_dict()
            
            return {
                'strategy_performance': strategy_contributions,
                'total_contributed_return': total_return,
                'total_contributed_risk': total_risk,
                'correlation_matrix': correlation_matrix,
                'attribution_summary': {
                    'momentum_contribution': strategies['Momentum_Strategy']['allocation'] * strategies['Momentum_Strategy']['return'],
                    'mean_reversion_contribution': strategies['Mean_Reversion_Strategy']['allocation'] * strategies['Mean_Reversion_Strategy']['return'],
                    'arbitrage_contribution': strategies['Arbitrage_Strategy']['allocation'] * strategies['Arbitrage_Strategy']['return'],
                    'trend_following_contribution': strategies['Trend_Following_Strategy']['allocation'] * strategies['Trend_Following_Strategy']['return']
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating strategy decomposition: {e}")
            raise
    
    async def _calculate_outperformance_analysis(self,
                                               portfolio_data: pd.DataFrame,
                                               benchmark_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate outperformance analysis vs benchmark"""
        try:
            returns = portfolio_data['portfolio_return']
            benchmark_returns = portfolio_data['benchmark_return']
            
            # Active returns
            active_returns = returns - benchmark_returns
            
            # Win rate vs benchmark
            outperformance_rate = (active_returns > 0).mean()
            
            # Information ratio
            active_mean = active_returns.mean()
            active_std = active_returns.std()
            information_ratio = active_mean / active_std * np.sqrt(252)
            
            # Up/down capture ratios
            up_market_days = benchmark_returns > 0
            down_market_days = benchmark_returns <= 0
            
            if up_market_days.sum() > 0:
                up_capture = (returns[up_market_days] / benchmark_returns[up_market_days]).mean()
            else:
                up_capture = 1.0
            
            if down_market_days.sum() > 0:
                down_capture = (returns[down_market_days] / benchmark_returns[down_market_days]).mean()
            else:
                down_capture = 1.0
            
            # Outperformance persistence
            consecutive_wins = self._calculate_consecutive_wins(active_returns)
            outperformance_persistence = np.mean(consecutive_wins)
            
            # Alpha analysis
            periods_per_year = 252
            covariance = np.cov(returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
            
            expected_return = self.risk_free_rate + beta * (benchmark_returns.mean() * periods_per_year - self.risk_free_rate)
            actual_return = returns.mean() * periods_per_year
            alpha = actual_return - expected_return
            
            return {
                'outperformance_rate': outperformance_rate,
                'information_ratio': information_ratio,
                'up_capture_ratio': up_capture,
                'down_capture_ratio': down_capture,
                'outperformance_persistence': outperformance_persistence,
                'alpha': alpha,
                'beta': beta,
                'tracking_error': active_std * np.sqrt(periods_per_year),
                'active_returns_statistics': {
                    'mean': active_mean,
                    'std': active_std,
                    'min': active_returns.min(),
                    'max': active_returns.max(),
                    'skewness': stats.skew(active_returns),
                    'kurtosis': stats.kurtosis(active_returns)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating outperformance analysis: {e}")
            raise
    
    def _calculate_consecutive_wins(self, active_returns: pd.Series) -> List[int]:
        """Calculate consecutive outperformance periods"""
        consecutive_wins = []
        current_streak = 0
        
        for ret in active_returns:
            if ret > 0:
                current_streak += 1
            else:
                if current_streak > 0:
                    consecutive_wins.append(current_streak)
                current_streak = 0
        
        if current_streak > 0:
            consecutive_wins.append(current_streak)
        
        return consecutive_wins if consecutive_wins else [0]
    
    async def update_real_time_metrics(self):
        """Update real-time performance metrics"""
        try:
            # Update cached performance metrics for real-time dashboard
            self.logger.debug("Updating real-time performance metrics")
            
            # This would typically update a Redis cache or similar
            # For now, just log the update
            pass
            
        except Exception as e:
            self.logger.error(f"Error updating real-time metrics: {e}")
    
    async def run_comprehensive_analysis(self):
        """Run comprehensive portfolio analysis for batch processing"""
        try:
            self.logger.info("Running comprehensive performance analysis")
            
            # This would run full analysis on all portfolios
            # For now, just log the operation
            result = {
                'analysis_type': 'comprehensive_performance',
                'portfolios_analyzed': 1,  # Mock
                'execution_time': datetime.now().isoformat(),
                'status': 'completed'
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive analysis: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for performance analytics"""
        try:
            return {
                'status': 'healthy',
                'last_update': datetime.now().isoformat(),
                'cache_size': len(self._performance_cache),
                'benchmarks_available': len(self.benchmarks)
            }
        except Exception as e:
            self.logger.error(f"Error in performance analytics health check: {e}")
            return {'status': 'error', 'error': str(e)}