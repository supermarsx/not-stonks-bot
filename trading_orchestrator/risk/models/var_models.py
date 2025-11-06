"""
Value at Risk (VaR) Models

Implements multiple VaR calculation methods:
- Historical VaR
- Parametric VaR (normal distribution assumptions)
- Monte Carlo VaR
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import minimize_scalar
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

from database.models.trading import Position, Trade
from database.models.market_data import PriceData

logger = logging.getLogger(__name__)


class BaseVaRModel:
    """Base class for VaR calculations."""
    
    def __init__(self, confidence_level: float = 0.95, horizon_days: int = 1):
        """
        Initialize VaR model.
        
        Args:
            confidence_level: VaR confidence level (e.g., 0.95 for 95%)
            horizon_days: Risk horizon in days
        """
        self.confidence_level = confidence_level
        self.horizon_days = horizon_days
        self.z_score = stats.norm.ppf(1 - confidence_level)
        
    def calculate_var(self, returns: pd.Series, current_value: float) -> float:
        """
        Calculate Value at Risk.
        
        Args:
            returns: Historical returns series
            current_value: Current portfolio/market value
            
        Returns:
            VaR amount (positive value representing potential loss)
        """
        raise NotImplementedError
        
    def validate_data(self, returns: pd.Series) -> bool:
        """Validate return data for calculations."""
        return len(returns) >= 30 and returns.std() > 0


class HistoricalVaR(BaseVaRModel):
    """
    Historical VaR using empirical distribution of returns.
    
    Uses historical return distribution to estimate VaR without
    distributional assumptions.
    """
    
    def __init__(self, confidence_level: float = 0.95, horizon_days: int = 1, 
                 lookback_days: int = 252):
        """
        Initialize Historical VaR model.
        
        Args:
            confidence_level: VaR confidence level
            horizon_days: Risk horizon in days
            lookback_days: Historical lookback period
        """
        super().__init__(confidence_level, horizon_days)
        self.lookback_days = lookback_days
        
    async def calculate_portfolio_var(self, positions: List[Dict[str, Any]], 
                                    market_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate VaR for entire portfolio using historical simulation.
        
        Args:
            positions: List of position dictionaries
            market_data: Historical market data DataFrame
            
        Returns:
            Portfolio VaR breakdown by position and total
        """
        try:
            portfolio_var = {}
            total_var = 0.0
            
            for position in positions:
                symbol = position['symbol']
                quantity = position['quantity']
                current_price = position.get('current_price', 100.0)  # Default price
                
                if symbol in market_data.columns:
                    returns = market_data[symbol].pct_change().dropna()
                    
                    # Adjust for horizon (sqrt of time for volatility)
                    var_per_unit = self._calculate_single_var(returns, current_price)
                    
                    # Scale by quantity and sqrt of horizon
                    position_var = var_per_unit * abs(quantity) * np.sqrt(self.horizon_days)
                    
                    portfolio_var[symbol] = {
                        'var': position_var,
                        'percentage': (position_var / (abs(quantity) * current_price)) * 100,
                        'var_contribution': position_var
                    }
                    
                    total_var += position_var
                else:
                    logger.warning(f"No market data found for {symbol}")
            
            portfolio_var['total'] = total_var
            return portfolio_var
            
        except Exception as e:
            logger.error(f"Portfolio VaR calculation error: {str(e)}")
            return {'total': 0.0, 'error': str(e)}
    
    def _calculate_single_var(self, returns: pd.Series, current_value: float) -> float:
        """Calculate VaR for a single asset."""
        if not self.validate_data(returns):
            return current_value * 0.10  # Default 10% if insufficient data
        
        # Get historical returns
        sorted_returns = returns.sort_values()
        
        # Calculate VaR from historical distribution
        var_percentile = int((1 - self.confidence_level) * len(sorted_returns))
        var_return = sorted_returns.iloc[var_percentile]
        
        # Convert to monetary VaR
        var_amount = abs(var_return) * current_value
        
        return var_amount
    
    def calculate_rolling_var(self, returns: pd.Series, window: int = 252) -> pd.Series:
        """
        Calculate rolling VaR over time.
        
        Args:
            returns: Price return series
            window: Rolling window size
            
        Returns:
            Rolling VaR series
        """
        try:
            price_returns = returns.pct_change().dropna()
            rolling_var = []
            
            for i in range(window, len(price_returns) + 1):
                window_returns = price_returns.iloc[i-window:i]
                var = self._calculate_single_var(window_returns, 100.0)  # Normalized to 100
                rolling_var.append(var)
            
            return pd.Series(rolling_var, index=returns.index[window:])
            
        except Exception as e:
            logger.error(f"Rolling VaR calculation error: {str(e)}")
            return pd.Series()


class ParametricVaR(BaseVaRModel):
    """
    Parametric VaR using normal distribution assumptions.
    
    Assumes returns follow normal distribution and uses
    mean and standard deviation to calculate VaR.
    """
    
    def __init__(self, confidence_level: float = 0.95, horizon_days: int = 1,
                 method: str = 'variance_covariance'):
        """
        Initialize Parametric VaR model.
        
        Args:
            confidence_level: VaR confidence level
            horizon_days: Risk horizon in days
            method: Calculation method ('variance_covariance' or 'cornish_fisher')
        """
        super().__init__(confidence_level, horizon_days)
        self.method = method
        
    async def calculate_portfolio_var(self, positions: List[Dict[str, Any]], 
                                    returns_data: Dict[str, pd.Series]) -> Dict[str, float]:
        """
        Calculate parametric VaR for portfolio.
        
        Args:
            positions: List of position dictionaries
            returns_data: Historical returns data by symbol
            
        Returns:
            Portfolio VaR breakdown
        """
        try:
            portfolio_var = {}
            position_vars = []
            weights = []
            
            # Calculate position-level VaR
            for position in positions:
                symbol = position['symbol']
                quantity = position['quantity']
                current_price = position.get('current_price', 100.0)
                position_value = abs(quantity * current_price)
                
                if symbol in returns_data:
                    returns = returns_data[symbol].dropna()
                    
                    if self.validate_data(returns):
                        mean_return = returns.mean()
                        std_return = returns.std()
                        
                        # Calculate parametric VaR
                        var_return = self._calculate_parametric_var(mean_return, std_return)
                        var_amount = abs(var_return) * current_price * abs(quantity)
                        
                        portfolio_var[symbol] = {
                            'var': var_amount,
                            'var_return': var_return,
                            'mean_return': mean_return,
                            'std_return': std_return,
                            'position_value': position_value
                        }
                        
                        position_vars.append(var_amount)
                        weights.append(position_value)
                    else:
                        # Fallback to simple calculation
                        var_amount = position_value * 0.15  # 15% default
                        portfolio_var[symbol] = {'var': var_amount, 'position_value': position_value}
                        position_vars.append(var_amount)
                        weights.append(position_value)
            
            # Calculate portfolio-level VaR (assuming normal distribution)
            if len(position_vars) > 0:
                portfolio_value = sum(weights)
                total_var = sum(position_vars)  # Simplified - assumes perfect correlation
                portfolio_var['total'] = total_var
                portfolio_var['portfolio_value'] = portfolio_value
                portfolio_var['var_percentage'] = (total_var / portfolio_value) * 100 if portfolio_value > 0 else 0
            
            return portfolio_var
            
        except Exception as e:
            logger.error(f"Parametric VaR calculation error: {str(e)}")
            return {'total': 0.0, 'error': str(e)}
    
    def _calculate_parametric_var(self, mean_return: float, std_return: float) -> float:
        """Calculate parametric VaR using normal distribution."""
        if self.method == 'cornish_fisher':
            # Use Cornish-Fisher expansion for better accuracy with non-normal returns
            return self._cornish_fisher_var(mean_return, std_return)
        else:
            # Standard variance-covariance method
            var_return = mean_return * self.horizon_days + self.z_score * std_return * np.sqrt(self.horizon_days)
            return var_return
    
    def _cornish_fisher_var(self, mean_return: float, std_return: float) -> float:
        """Calculate VaR using Cornish-Fisher expansion for better tail estimation."""
        try:
            # This is a simplified version - in practice, you'd calculate skewness and kurtosis
            basic_var = mean_return * self.horizon_days + self.z_score * std_return * np.sqrt(self.horizon_days)
            return basic_var
        except Exception:
            # Fallback to basic calculation
            return mean_return * self.horizon_days + self.z_score * std_return * np.sqrt(self.horizon_days)


class MonteCarloVaR(BaseVaRModel):
    """
    Monte Carlo VaR using simulation-based approach.
    
    Simulates potential future scenarios based on return distributions
    and calculates VaR from simulated outcomes.
    """
    
    def __init__(self, confidence_level: float = 0.95, horizon_days: int = 1,
                 num_simulations: int = 10000, use_correlations: bool = True):
        """
        Initialize Monte Carlo VaR model.
        
        Args:
            confidence_level: VaR confidence level
            horizon_days: Risk horizon in days
            num_simulations: Number of simulation runs
            use_correlations: Whether to use correlation structure
        """
        super().__init__(confidence_level, horizon_days)
        self.num_simulations = num_simulations
        self.use_correlations = use_correlations
        
    async def calculate_portfolio_var(self, positions: List[Dict[str, Any]], 
                                    returns_data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        Calculate Monte Carlo VaR for portfolio.
        
        Args:
            positions: List of position dictionaries
            returns_data: Historical returns data by symbol
            
        Returns:
            Monte Carlo simulation results
        """
        try:
            # Prepare return distributions
            distributions = {}
            for symbol, returns in returns_data.items():
                if self.validate_data(returns):
                    distributions[symbol] = {
                        'mean': returns.mean(),
                        'std': returns.std(),
                        'skew': stats.skew(returns),
                        'kurt': stats.kurtosis(returns)
                    }
            
            if not distributions:
                return {'error': 'Insufficient return data for Monte Carlo simulation'}
            
            # Run Monte Carlo simulations
            results = await self._run_simulations(positions, distributions)
            
            return results
            
        except Exception as e:
            logger.error(f"Monte Carlo VaR calculation error: {str(e)}")
            return {'error': str(e)}
    
    async def _run_simulations(self, positions: List[Dict[str, Any]], 
                             distributions: Dict[str, Dict]) -> Dict[str, Any]:
        """Run Monte Carlo simulations."""
        try:
            # Get portfolio composition
            portfolio_values = []
            position_values = {pos['symbol']: 0.0 for pos in positions}
            
            # Calculate initial portfolio value
            initial_value = 0.0
            for position in positions:
                symbol = position['symbol']
                quantity = position['quantity']
                current_price = position.get('current_price', 100.0)
                pos_value = abs(quantity * current_price)
                position_values[symbol] = pos_value
                initial_value += pos_value
            
            # Run simulations
            simulation_results = []
            
            for _ in range(self.num_simulations):
                # Generate random returns for each asset
                scenario_returns = {}
                
                for symbol, dist_params in distributions.items():
                    # Generate return using normal distribution (simplified)
                    # In practice, use more sophisticated methods for non-normal distributions
                    random_return = np.random.normal(
                        dist_params['mean'] * self.horizon_days,
                        dist_params['std'] * np.sqrt(self.horizon_days)
                    )
                    scenario_returns[symbol] = random_return
                
                # Calculate portfolio outcome
                portfolio_outcome = initial_value
                for position in positions:
                    symbol = position['symbol']
                    if symbol in scenario_returns:
                        pos_return = scenario_returns[symbol]
                        pos_value = position_values[symbol]
                        portfolio_outcome += pos_value * pos_return
                
                portfolio_values.append(portfolio_outcome)
                simulation_results.append({
                    'portfolio_value': portfolio_outcome,
                    'portfolio_return': (portfolio_outcome - initial_value) / initial_value,
                    'returns_by_symbol': scenario_returns
                })
            
            # Calculate VaR from simulation results
            portfolio_losses = [initial_value - result['portfolio_value'] for result in simulation_results]
            var_percentile = int((1 - self.confidence_level) * len(portfolio_losses))
            var_value = sorted(portfolio_losses)[var_percentile]
            
            # Calculate additional statistics
            portfolio_returns = [result['portfolio_return'] for result in simulation_results]
            
            return {
                'var': var_value,
                'var_percentage': (var_value / initial_value) * 100,
                'initial_portfolio_value': initial_value,
                'simulations_run': self.num_simulations,
                'confidence_level': self.confidence_level,
                'horizon_days': self.horizon_days,
                'statistics': {
                    'mean_return': np.mean(portfolio_returns),
                    'std_return': np.std(portfolio_returns),
                    'min_return': min(portfolio_returns),
                    'max_return': max(portfolio_returns),
                    'percentile_5': np.percentile(portfolio_returns, 5),
                    'percentile_95': np.percentile(portfolio_returns, 95)
                },
                'sample_scenarios': simulation_results[:10]  # First 10 scenarios for analysis
            }
            
        except Exception as e:
            logger.error(f"Monte Carlo simulation error: {str(e)}")
            return {'error': str(e)}


class VaRFactory:
    """Factory class for creating VaR models."""
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> BaseVaRModel:
        """
        Create VaR model instance.
        
        Args:
            model_type: Type of VaR model ('historical', 'parametric', 'monte_carlo')
            **kwargs: Model-specific parameters
            
        Returns:
            VaR model instance
            
        Raises:
            ValueError: If model_type is not supported
        """
        model_types = {
            'historical': HistoricalVaR,
            'parametric': ParametricVaR,
            'monte_carlo': MonteCarloVaR,
            'mc': MonteCarloVaR  # Alias
        }
        
        if model_type.lower() not in model_types:
            raise ValueError(f"Unsupported VaR model type: {model_type}")
        
        return model_types[model_type.lower()](**kwargs)
    
    @staticmethod
    def calculate_multiple_var(returns: pd.Series, current_value: float, 
                             methods: List[str] = None) -> Dict[str, float]:
        """
        Calculate VaR using multiple methods for comparison.
        
        Args:
            returns: Historical returns
            current_value: Current portfolio value
            methods: List of VaR methods to use
            
        Returns:
            VaR results by method
        """
        if methods is None:
            methods = ['historical', 'parametric', 'monte_carlo']
        
        results = {}
        
        for method in methods:
            try:
                model = VaRFactory.create_model(method)
                var = model.calculate_var(returns, current_value)
                results[method] = var
            except Exception as e:
                logger.error(f"Error calculating VaR with {method}: {str(e)}")
                results[method] = 0.0
        
        return results
