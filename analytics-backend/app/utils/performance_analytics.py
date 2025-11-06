"""
Performance Analytics Calculations
Institutional-grade mathematical functions for portfolio performance analysis
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PerformanceAnalytics:
    """Advanced performance analytics calculations"""
    
    @staticmethod
    def brinson_fachler_attribution(
        portfolio_weights: Dict[str, float],
        benchmark_weights: Dict[str, float], 
        portfolio_returns: Dict[str, float],
        benchmark_returns: Dict[str, float],
        portfolio_return: float,
        benchmark_return: float
    ) -> Dict[str, Dict[str, float]]:
        """
        Brinson-Fachler Performance Attribution Analysis
        
        Decomposes portfolio outperformance into:
        - Asset Allocation Effect
        - Security Selection Effect  
        - Interaction Effect
        """
        
        # Ensure all assets are present in all dictionaries
        all_assets = set(portfolio_weights.keys()) | set(benchmark_weights.keys())
        
        attribution = {}
        total_allocation_effect = 0
        total_selection_effect = 0
        total_interaction_effect = 0
        
        for asset in all_assets:
            pw = portfolio_weights.get(asset, 0.0)  # Portfolio weight
            bw = benchmark_weights.get(asset, 0.0)  # Benchmark weight
            pr = portfolio_returns.get(asset, 0.0)  # Portfolio asset return
            br = benchmark_returns.get(asset, 0.0)  # Benchmark asset return
            
            # Brinson-Fachler Attribution Effects
            allocation_effect = (pw - bw) * br
            selection_effect = bw * (pr - br) 
            interaction_effect = (pw - bw) * (pr - br)
            
            attribution[asset] = {
                'allocation_effect': allocation_effect,
                'selection_effect': selection_effect,
                'interaction_effect': interaction_effect,
                'total_effect': allocation_effect + selection_effect + interaction_effect,
                'portfolio_weight': pw,
                'benchmark_weight': bw,
                'portfolio_return': pr,
                'benchmark_return': br
            }
            
            total_allocation_effect += allocation_effect
            total_selection_effect += selection_effect
            total_interaction_effect += interaction_effect
        
        # Summary
        attribution['TOTAL'] = {
            'allocation_effect': total_allocation_effect,
            'selection_effect': total_selection_effect, 
            'interaction_effect': total_interaction_effect,
            'total_attribution': total_allocation_effect + total_selection_effect + total_interaction_effect,
            'active_return': portfolio_return - benchmark_return,
            'explained_active_return': total_allocation_effect + total_selection_effect + total_interaction_effect
        }
        
        return attribution
    
    @staticmethod
    def risk_adjusted_metrics(returns: pd.Series, risk_free_rate: float = 0.02) -> Dict[str, float]:
        """
        Calculate comprehensive risk-adjusted performance metrics
        
        Returns:
        - Sharpe Ratio: (Return - Risk Free) / Volatility
        - Sortino Ratio: (Return - Risk Free) / Downside Deviation
        - Calmar Ratio: Annual Return / Maximum Drawdown
        - Information Ratio: Active Return / Tracking Error
        - Treynor Ratio: (Return - Risk Free) / Beta
        """
        
        if len(returns) < 2:
            return {metric: 0.0 for metric in ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'information_ratio']}
        
        # Annualized metrics (assuming daily returns)
        annual_return = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)
        
        # Sharpe Ratio
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
        
        # Sortino Ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else annual_volatility
        sortino_ratio = (annual_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Maximum Drawdown and Calmar Ratio
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        # Win Rate
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
        
        # Information Ratio (assumes benchmark is 0 for active returns)
        tracking_error = returns.std() * np.sqrt(252)
        information_ratio = annual_return / tracking_error if tracking_error > 0 else 0
        
        return {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(returns),
            'average_return': returns.mean(),
            'best_day': returns.max(),
            'worst_day': returns.min()
        }
    
    @staticmethod
    def rolling_performance_metrics(
        returns: pd.Series, 
        window: int = 252,
        risk_free_rate: float = 0.02
    ) -> pd.DataFrame:
        """Calculate rolling performance metrics over time"""
        
        if len(returns) < window:
            return pd.DataFrame()
        
        rolling_data = []
        
        for i in range(window, len(returns) + 1):
            period_returns = returns.iloc[i-window:i]
            metrics = PerformanceAnalytics.risk_adjusted_metrics(period_returns, risk_free_rate)
            metrics['date'] = returns.index[i-1]
            rolling_data.append(metrics)
        
        return pd.DataFrame(rolling_data).set_index('date')
    
    @staticmethod
    def calculate_var_cvar(
        returns: pd.Series,
        confidence_levels: List[float] = [0.95, 0.99]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate Value at Risk (VaR) and Conditional Value at Risk (CVaR)
        
        VaR: Maximum expected loss at given confidence level
        CVaR: Expected loss beyond VaR threshold
        """
        
        var_cvar = {}
        
        for confidence in confidence_levels:
            alpha = 1 - confidence
            
            # Historical VaR (percentile method)
            var = np.percentile(returns, alpha * 100)
            
            # CVaR (expected shortfall)
            cvar = returns[returns <= var].mean()
            
            # Annualized values
            var_annual = var * np.sqrt(252)
            cvar_annual = cvar * np.sqrt(252)
            
            var_cvar[f"{confidence:.0%}"] = {
                'var_daily': var,
                'cvar_daily': cvar,
                'var_annual': var_annual,
                'cvar_annual': cvar_annual,
                'var_percentage': var * 100,
                'cvar_percentage': cvar * 100
            }
        
        return var_cvar
    
    @staticmethod
    def performance_summary_report(
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.02
    ) -> Dict[str, any]:
        """Generate comprehensive performance summary report"""
        
        # Basic performance metrics
        perf_metrics = PerformanceAnalytics.risk_adjusted_metrics(returns, risk_free_rate)
        
        # VaR/CVaR analysis
        var_cvar = PerformanceAnalytics.calculate_var_cvar(returns)
        
        # Benchmark comparison if provided
        benchmark_comparison = {}
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            active_returns = returns - benchmark_returns
            benchmark_metrics = PerformanceAnalytics.risk_adjusted_metrics(benchmark_returns, risk_free_rate)
            active_metrics = PerformanceAnalytics.risk_adjusted_metrics(active_returns, 0)  # No risk-free for active
            
            benchmark_comparison = {
                'benchmark_metrics': benchmark_metrics,
                'active_metrics': active_metrics,
                'tracking_error': active_returns.std() * np.sqrt(252),
                'information_ratio': active_metrics['annual_return'] / (active_returns.std() * np.sqrt(252)) if active_returns.std() > 0 else 0,
                'beta': np.cov(returns, benchmark_returns)[0, 1] / np.var(benchmark_returns) if np.var(benchmark_returns) > 0 else 1.0
            }
        
        # Monthly/Quarterly returns breakdown
        returns_df = pd.DataFrame({'returns': returns})
        monthly_returns = returns_df.resample('M')['returns'].apply(lambda x: (1 + x).prod() - 1)
        quarterly_returns = returns_df.resample('Q')['returns'].apply(lambda x: (1 + x).prod() - 1)
        
        return {
            'performance_metrics': perf_metrics,
            'risk_metrics': var_cvar,
            'benchmark_comparison': benchmark_comparison,
            'monthly_returns': monthly_returns.to_dict(),
            'quarterly_returns': quarterly_returns.to_dict(),
            'summary_stats': {
                'total_return': (1 + returns).prod() - 1,
                'total_days': len(returns),
                'profitable_days': (returns > 0).sum(),
                'loss_days': (returns < 0).sum(),
                'flat_days': (returns == 0).sum(),
                'largest_gain': returns.max(),
                'largest_loss': returns.min(),
                'average_gain': returns[returns > 0].mean() if (returns > 0).any() else 0,
                'average_loss': returns[returns < 0].mean() if (returns < 0).any() else 0
            }
        }