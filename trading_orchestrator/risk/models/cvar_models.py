"""
Conditional Value at Risk (CVaR) Models

Implements Expected Shortfall and tail risk analysis:
- Conditional VaR (CVaR) / Expected Shortfall
- Tail Risk analysis
- Coherent risk measures
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from scipy import stats, optimize
import logging
import asyncio

from database.models.trading import Position, Trade
from database.models.market_data import PriceData

logger = logging.getLogger(__name__)


class BaseCVaRModel:
    """Base class for CVaR calculations."""
    
    def __init__(self, confidence_level: float = 0.95, horizon_days: int = 1):
        """
        Initialize CVaR model.
        
        Args:
            confidence_level: CVaR confidence level (e.g., 0.95 for 95%)
            horizon_days: Risk horizon in days
        """
        self.confidence_level = confidence_level
        self.horizon_days = horizon_days
        
    def calculate_cvar(self, returns: pd.Series, current_value: float) -> Dict[str, float]:
        """
        Calculate Conditional Value at Risk.
        
        Args:
            returns: Historical returns series
            current_value: Current portfolio/market value
            
        Returns:
            Dictionary with VaR, CVaR and other statistics
        """
        raise NotImplementedError
        
    def validate_data(self, returns: pd.Series) -> bool:
        """Validate return data for calculations."""
        return len(returns) >= 30 and returns.std() > 0


class ExpectedShortfall(BaseCVaRModel):
    """
    Conditional Value at Risk (CVaR) / Expected Shortfall calculation.
    
    CVaR represents the expected loss conditional on the loss
    exceeding the VaR threshold. It's a coherent risk measure.
    """
    
    def __init__(self, confidence_level: float = 0.95, horizon_days: int = 1,
                 method: str = 'historical'):
        """
        Initialize Expected Shortfall model.
        
        Args:
            confidence_level: CVaR confidence level
            horizon_days: Risk horizon in days
            method: Calculation method ('historical', 'parametric', 'monte_carlo')
        """
        super().__init__(confidence_level, horizon_days)
        self.method = method
        
    async def calculate_portfolio_cvar(self, positions: List[Dict[str, Any]], 
                                     returns_data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        Calculate CVaR for entire portfolio.
        
        Args:
            positions: List of position dictionaries
            returns_data: Historical returns data by symbol
            
        Returns:
            Portfolio CVaR breakdown and statistics
        """
        try:
            portfolio_cvar = {
                'cvar': 0.0,
                'var': 0.0,
                'position_contributions': {},
                'statistics': {},
                'tail_risk_metrics': {},
                'method': self.method,
                'confidence_level': self.confidence_level
            }
            
            if self.method == 'historical':
                result = await self._calculate_historical_cvar(positions, returns_data)
            elif self.method == 'parametric':
                result = await self._calculate_parametric_cvar(positions, returns_data)
            elif self.method == 'monte_carlo':
                result = await self._calculate_monte_carlo_cvar(positions, returns_data)
            else:
                raise ValueError(f"Unsupported CVaR method: {self.method}")
            
            portfolio_cvar.update(result)
            return portfolio_cvar
            
        except Exception as e:
            logger.error(f"Portfolio CVaR calculation error: {str(e)}")
            return {'error': str(e), 'cvar': 0.0, 'var': 0.0}
    
    async def _calculate_historical_cvar(self, positions: List[Dict[str, Any]], 
                                       returns_data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Calculate CVaR using historical data."""
        try:
            # Calculate portfolio returns for each day
            portfolio_returns = self._calculate_portfolio_returns(positions, returns_data)
            
            if len(portfolio_returns) == 0:
                return {'cvar': 0.0, 'var': 0.0}
            
            # Sort returns to find VaR and CVaR
            sorted_returns = portfolio_returns.sort_values()
            
            # Calculate VaR (alpha quantile)
            alpha = 1 - self.confidence_level
            var_index = int(alpha * len(sorted_returns))
            var_return = sorted_returns.iloc[var_index] if var_index < len(sorted_returns) else sorted_returns.iloc[-1]
            
            # Calculate CVaR (average of worst alpha% returns)
            tail_returns = sorted_returns.iloc[:var_index + 1]
            cvar_return = tail_returns.mean()
            
            # Convert to monetary amounts
            portfolio_value = sum(abs(pos.get('quantity', 0) * pos.get('current_price', 100)) 
                                for pos in positions)
            
            var_amount = abs(var_return) * portfolio_value
            cvar_amount = abs(cvar_return) * portfolio_value
            
            # Calculate additional tail risk metrics
            tail_metrics = self._calculate_tail_metrics(sorted_returns, alpha)
            
            return {
                'var': var_amount,
                'cvar': cvar_amount,
                'var_return': var_return,
                'cvar_return': cvar_return,
                'portfolio_value': portfolio_value,
                'tail_risk_metrics': tail_metrics,
                'tail_returns_count': len(tail_returns),
                'tail_returns_percentage': (len(tail_returns) / len(sorted_returns)) * 100
            }
            
        except Exception as e:
            logger.error(f"Historical CVaR calculation error: {str(e)}")
            return {'error': str(e)}
    
    async def _calculate_parametric_cvar(self, positions: List[Dict[str, Any]], 
                                       returns_data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Calculate CVaR using parametric (normal) distribution."""
        try:
            # Calculate portfolio parameters
            portfolio_value, portfolio_mean, portfolio_std = self._calculate_portfolio_parameters(positions, returns_data)
            
            # Calculate VaR
            var_return = stats.norm.ppf(1 - self.confidence_level, 
                                      loc=portfolio_mean * self.horizon_days,
                                      scale=portfolio_std * np.sqrt(self.horizon_days))
            
            # Calculate CVaR for normal distribution
            alpha = 1 - self.confidence_level
            var_threshold = stats.norm.ppf(alpha)
            
            # CVaR for normal distribution: -phi(var_threshold) / alpha
            # where phi is the standard normal density
            phi_var_threshold = stats.norm.pdf(var_threshold)
            cvar_return = -(phi_var_threshold / alpha)
            
            # Convert to monetary amounts
            var_amount = abs(var_return) * portfolio_value
            cvar_amount = cvar_return * portfolio_value
            
            # Calculate additional metrics
            tail_metrics = {
                'expected_shortfall_ratio': cvar_amount / var_amount if var_amount > 0 else 0,
                'tail_loss_probability': alpha,
                'extreme_value_theory_estimate': cvar_return
            }
            
            return {
                'var': var_amount,
                'cvar': cvar_amount,
                'var_return': var_return,
                'cvar_return': cvar_return,
                'portfolio_value': portfolio_value,
                'portfolio_mean': portfolio_mean,
                'portfolio_std': portfolio_std,
                'tail_risk_metrics': tail_metrics
            }
            
        except Exception as e:
            logger.error(f"Parametric CVaR calculation error: {str(e)}")
            return {'error': str(e)}
    
    async def _calculate_monte_carlo_cvar(self, positions: List[Dict[str, Any]], 
                                        returns_data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Calculate CVaR using Monte Carlo simulation."""
        try:
            num_simulations = 100000
            
            # Calculate portfolio parameters
            portfolio_value, portfolio_mean, portfolio_std = self._calculate_portfolio_parameters(positions, returns_data)
            
            # Generate simulated returns
            simulated_returns = np.random.normal(
                loc=portfolio_mean * self.horizon_days,
                scale=portfolio_std * np.sqrt(self.horizon_days),
                size=num_simulations
            )
            
            # Calculate VaR and CVaR
            var_return = np.percentile(simulated_returns, (1 - self.confidence_level) * 100)
            
            # CVaR is the mean of returns worse than VaR
            tail_mask = simulated_returns <= var_return
            cvar_return = np.mean(simulated_returns[tail_mask])
            
            # Convert to monetary amounts
            var_amount = abs(var_return) * portfolio_value
            cvar_amount = abs(cvar_return) * portfolio_value
            
            # Calculate simulation statistics
            tail_metrics = self._calculate_monte_carlo_metrics(simulated_returns)
            
            return {
                'var': var_amount,
                'cvar': cvar_amount,
                'var_return': var_return,
                'cvar_return': cvar_return,
                'portfolio_value': portfolio_value,
                'simulations_run': num_simulations,
                'tail_risk_metrics': tail_metrics
            }
            
        except Exception as e:
            logger.error(f"Monte Carlo CVaR calculation error: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_portfolio_returns(self, positions: List[Dict[str, Any]], 
                                   returns_data: Dict[str, pd.Series]) -> pd.Series:
        """Calculate portfolio returns from position and return data."""
        try:
            # Get common date index
            common_dates = None
            for symbol, returns in returns_data.items():
                if common_dates is None:
                    common_dates = returns.index
                else:
                    common_dates = common_dates.intersection(returns.index)
            
            if common_dates is None or len(common_dates) < 30:
                return pd.Series()
            
            # Calculate weighted portfolio returns
            total_weight = 0.0
            weighted_returns = pd.Series(0.0, index=common_dates)
            
            for position in positions:
                symbol = position['symbol']
                if symbol in returns_data and symbol in common_dates:
                    quantity = position['quantity']
                    current_price = position.get('current_price', 100.0)
                    
                    # Calculate position weight (simplified)
                    position_value = abs(quantity * current_price)
                    
                    # Weight by position value (assuming portfolio value of 1 for normalization)
                    weight = position_value
                    weighted_returns += returns_data[symbol] * weight
                    total_weight += weight
            
            if total_weight > 0:
                weighted_returns = weighted_returns / total_weight
            
            return weighted_returns.dropna()
            
        except Exception as e:
            logger.error(f"Portfolio returns calculation error: {str(e)}")
            return pd.Series()
    
    def _calculate_portfolio_parameters(self, positions: List[Dict[str, Any]], 
                                      returns_data: Dict[str, pd.Series]) -> Tuple[float, float, float]:
        """Calculate portfolio mean and standard deviation."""
        try:
            total_value = 0.0
            weighted_mean = 0.0
            weighted_var = 0.0
            
            # Calculate portfolio statistics (simplified - assumes independence)
            for position in positions:
                symbol = position['symbol']
                if symbol in returns_data:
                    quantity = position['quantity']
                    current_price = position.get('current_price', 100.0)
                    position_value = abs(quantity * current_price)
                    returns = returns_data[symbol].dropna()
                    
                    if len(returns) > 0:
                        mean_return = returns.mean()
                        std_return = returns.std()
                        
                        weight = position_value
                        total_value += position_value
                        
                        weighted_mean += weight * mean_return
                        weighted_var += weight * weight * std_return * std_return
            
            if total_value > 0:
                # Normalize weights
                portfolio_mean = weighted_mean / total_value
                portfolio_var = weighted_var / (total_value * total_value)
                portfolio_std = np.sqrt(portfolio_var)
            else:
                portfolio_mean = 0.0
                portfolio_std = 0.0
            
            return total_value, portfolio_mean, portfolio_std
            
        except Exception as e:
            logger.error(f"Portfolio parameters calculation error: {str(e)}")
            return 0.0, 0.0, 0.0
    
    def _calculate_tail_metrics(self, returns: pd.Series, alpha: float) -> Dict[str, float]:
        """Calculate additional tail risk metrics."""
        try:
            sorted_returns = returns.sort_values()
            
            # Extreme losses (worst 5% of days)
            extreme_threshold = alpha * 0.2  # 20% of the tail
            extreme_count = max(1, int(extreme_threshold * len(sorted_returns)))
            extreme_returns = sorted_returns.iloc[:extreme_count]
            
            return {
                'extreme_var': abs(extreme_returns.iloc[-1]) if len(extreme_returns) > 0 else 0,
                'extreme_cvar': abs(extreme_returns.mean()) if len(extreme_returns) > 0 else 0,
                'extreme_events_count': len(extreme_returns),
                'extreme_events_frequency': len(extreme_returns) / len(sorted_returns),
                'worst_day_return': sorted_returns.iloc[0] if len(sorted_returns) > 0 else 0,
                'second_worst_return': sorted_returns.iloc[1] if len(sorted_returns) > 1 else 0,
                'tail_skewness': stats.skew(sorted_returns.iloc[:int(alpha * len(sorted_returns))]),
                'tail_kurtosis': stats.kurtosis(sorted_returns.iloc[:int(alpha * len(sorted_returns))])
            }
            
        except Exception as e:
            logger.error(f"Tail metrics calculation error: {str(e)}")
            return {}
    
    def _calculate_monte_carlo_metrics(self, simulated_returns: np.ndarray) -> Dict[str, float]:
        """Calculate metrics for Monte Carlo simulation results."""
        try:
            return {
                'min_simulated_return': np.min(simulated_returns),
                'max_simulated_return': np.max(simulated_returns),
                'mean_simulated_return': np.mean(simulated_returns),
                'std_simulated_return': np.std(simulated_returns),
                'skewness': stats.skew(simulated_returns),
                'kurtosis': stats.kurtosis(simulated_returns),
                'tail_probability': np.mean(simulated_returns < 0),  # Probability of loss
                'extreme_loss_probability': np.mean(simulated_returns < -0.05)  # 5%+ loss probability
            }
            
        except Exception as e:
            logger.error(f"Monte Carlo metrics calculation error: {str(e)}")
            return {}


class TailRiskAnalyzer:
    """
    Advanced tail risk analysis beyond basic CVaR.
    
    Provides sophisticated analysis of extreme events and tail dependence.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize Tail Risk Analyzer.
        
        Args:
            confidence_level: Analysis confidence level
        """
        self.confidence_level = confidence_level
        
    async def analyze_tail_dependence(self, returns_data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        Analyze tail dependence between assets.
        
        Args:
            returns_data: Historical returns by symbol
            
        Returns:
            Tail dependence analysis results
        """
        try:
            symbols = list(returns_data.keys())
            tail_dependence_matrix = {}
            
            # Calculate pairwise tail dependence
            for i, symbol1 in enumerate(symbols):
                tail_dependence_matrix[symbol1] = {}
                for j, symbol2 in enumerate(symbols):
                    if i != j:
                        tau = self._calculate_tail_dependence(returns_data[symbol1], returns_data[symbol2])
                        tail_dependence_matrix[symbol1][symbol2] = tau
            
            # Calculate portfolio tail risk
            portfolio_tail_risk = await self._calculate_portfolio_tail_risk(returns_data)
            
            return {
                'tail_dependence_matrix': tail_dependence_matrix,
                'portfolio_tail_risk': portfolio_tail_risk,
                'symbols_analyzed': symbols,
                'confidence_level': self.confidence_level
            }
            
        except Exception as e:
            logger.error(f"Tail dependence analysis error: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_tail_dependence(self, returns1: pd.Series, returns2: pd.Series) -> float:
        """Calculate tail dependence coefficient between two return series."""
        try:
            # Calculate exceedances in left tail
            threshold1 = np.percentile(returns1, 5)  # 5th percentile
            threshold2 = np.percentile(returns2, 5)
            
            exceed1 = returns1 <= threshold1
            exceed2 = returns2 <= threshold2
            
            # Joint exceedances
            joint_exceed = exceed1 & exceed2
            
            # Tail dependence coefficient
            prob_exceed1 = exceed1.mean()
            prob_joint = joint_exceed.mean()
            
            if prob_exceed1 > 0:
                tail_dependence = prob_joint / prob_exceed1
            else:
                tail_dependence = 0.0
            
            return tail_dependence
            
        except Exception:
            return 0.0
    
    async def _calculate_portfolio_tail_risk(self, returns_data: Dict[str, pd.Series]) -> Dict[str, float]:
        """Calculate portfolio-level tail risk metrics."""
        try:
            # Create portfolio return series
            portfolio_returns = self._create_equal_weight_portfolio(returns_data)
            
            if len(portfolio_returns) == 0:
                return {}
            
            # Calculate tail risk metrics
            tail_risk_metrics = {
                'max_drawdown': self._calculate_max_drawdown(portfolio_returns),
                'calmar_ratio': self._calculate_calmar_ratio(portfolio_returns),
                'sortino_ratio': self._calculate_sortino_ratio(portfolio_returns),
                'skewness': stats.skew(portfolio_returns),
                'kurtosis': stats.kurtosis(portfolio_returns),
                'jarque_bera_stat': self._jarque_bera_statistic(portfolio_returns)
            }
            
            return tail_risk_metrics
            
        except Exception as e:
            logger.error(f"Portfolio tail risk calculation error: {str(e)}")
            return {}
    
    def _create_equal_weight_portfolio(self, returns_data: Dict[str, pd.Series]) -> pd.Series:
        """Create equal-weighted portfolio returns."""
        try:
            # Get common dates
            common_dates = None
            for returns in returns_data.values():
                if common_dates is None:
                    common_dates = returns.index
                else:
                    common_dates = common_dates.intersection(returns.index)
            
            if common_dates is None or len(common_dates) < 30:
                return pd.Series()
            
            # Calculate equal-weighted returns
            num_assets = len(returns_data)
            portfolio_returns = pd.Series(0.0, index=common_dates)
            
            for returns in returns_data.values():
                portfolio_returns += returns.reindex(common_dates).fillna(0)
            
            portfolio_returns = portfolio_returns / num_assets
            
            return portfolio_returns.dropna()
            
        except Exception as e:
            logger.error(f"Portfolio creation error: {str(e)}")
            return pd.Series()
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        try:
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            
            return abs(drawdown.min())
            
        except Exception:
            return 0.0
    
    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)."""
        try:
            annual_return = returns.mean() * 252  # Annualized
            max_dd = self._calculate_max_drawdown(returns)
            
            return annual_return / max_dd if max_dd > 0 else 0
            
        except Exception:
            return 0.0
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio (return / downside deviation)."""
        try:
            annual_return = returns.mean() * 252  # Annualized
            
            # Downside deviation
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            
            return annual_return / downside_std if downside_std > 0 else 0
            
        except Exception:
            return 0.0
    
    def _jarque_bera_statistic(self, returns: pd.Series) -> float:
        """Calculate Jarque-Bera test statistic for normality."""
        try:
            skewness = stats.skew(returns)
            kurtosis = stats.kurtosis(returns)
            n = len(returns)
            
            jb_stat = (n / 6) * (skewness**2 + (kurtosis**2) / 4)
            
            return jb_stat
            
        except Exception:
            return 0.0


class CVaRFactory:
    """Factory class for creating CVaR models."""
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> BaseCVaRModel:
        """
        Create CVaR model instance.
        
        Args:
            model_type: Type of CVaR model ('expected_shortfall')
            **kwargs: Model-specific parameters
            
        Returns:
            CVaR model instance
        """
        model_types = {
            'expected_shortfall': ExpectedShortfall,
            'cvar': ExpectedShortfall,  # Alias
            'es': ExpectedShortfall     # Alias
        }
        
        if model_type.lower() not in model_types:
            raise ValueError(f"Unsupported CVaR model type: {model_type}")
        
        return model_types[model_type.lower()](**kwargs)
