"""
Portfolio Optimization Module

Advanced portfolio optimization including:
- Modern Portfolio Theory (MPT) implementation
- Black-Litterman model for portfolio construction
- Risk parity optimization
- Maximum Sharpe ratio optimization
- Minimum variance portfolios
- Robust optimization techniques
- Multi-objective optimization (risk/return/ESG)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import asyncio
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm

from database.models.trading import Position
from database.models.market_data import PriceData

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Portfolio optimization result."""
    weights: np.ndarray
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    objective_value: float
    optimization_method: str
    constraints_satisfied: bool
    convergence_achieved: bool
    solver_info: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OptimizationConstraints:
    """Portfolio optimization constraints."""
    # Weight constraints
    min_weight: float = 0.0
    max_weight: float = 1.0
    max_single_position: float = 0.4  # 40% max in single asset
    
    # Budget constraint
    total_weight_sum: float = 1.0
    
    # Sector/Geographic constraints
    max_sector_weight: float = 0.6
    max_country_weight: float = 0.8
    
    # Risk constraints
    max_volatility: float = 0.25  # 25% annualized volatility
    max_var: float = 0.15  # 15% VaR
    max_drawdown_limit: float = 0.20  # 20% max drawdown
    
    # Liquidity constraints
    min_liquidity_score: float = 0.5
    
    # ESG constraints
    min_esg_score: float = 0.0
    max_esg_risk: float = 1.0
    
    # Transaction cost constraints
    max_turnover: float = 0.50  # 50% turnover limit


@dataclass
class OptimizationParameters:
    """Portfolio optimization parameters."""
    optimization_objective: str = "max_sharpe"  # Options: max_sharpe, min_variance, max_return, risk_parity
    risk_free_rate: float = 0.02
    optimization_horizon: int = 252  # Trading days
    confidence_level: float = 0.95
    expected_returns_method: str = "historical"  # historical, risk_adjusted, black_litterman
    covariance_matrix_method: str = "sample"  # sample, shrinkage, factor_based
    transaction_costs: float = 0.001  # 10 basis points
    rebalancing_frequency: str = "monthly"  # daily, weekly, monthly, quarterly
    
    # Black-Litterman parameters
    risk_aversion: float = 3.0
    view_uncertainty: float = 0.07
    confidence_threshold: float = 0.5
    
    # Robust optimization parameters
    robustness_level: float = 0.1  # Robust optimization uncertainty set size
    worst_case_optimization: bool = False
    
    # Multi-objective parameters
    return_weight: float = 0.4
    risk_weight: float = 0.4
    esg_weight: float = 0.2
    
    # Additional constraints
    constraints: Optional[OptimizationConstraints] = None


class BaseOptimizer(ABC):
    """Abstract base class for portfolio optimizers."""
    
    def __init__(self, returns_data: pd.DataFrame, parameters: OptimizationParameters):
        """
        Initialize optimizer.
        
        Args:
            returns_data: Historical return data
            parameters: Optimization parameters
        """
        self.returns_data = returns_data
        self.parameters = parameters
        self.expected_returns = None
        self.covariance_matrix = None
        self.assets = returns_data.columns.tolist()
        self.n_assets = len(self.assets)
        
    @abstractmethod
    async def optimize(self) -> OptimizationResult:
        """Run portfolio optimization."""
        pass
    
    def _calculate_expected_returns(self) -> np.ndarray:
        """Calculate expected returns."""
        if self.expected_returns is not None:
            return self.expected_returns
        
        method = self.parameters.expected_returns_method
        
        if method == "historical":
            # Simple historical average returns
            self.expected_returns = self.returns_data.mean().values
        elif method == "risk_adjusted":
            # Risk-adjusted returns using Sharpe ratio
            excess_returns = self.returns_data.mean() - self.parameters.risk_free_rate
            volatility = self.returns_data.std()
            self.expected_returns = (excess_returns / volatility).values
        elif method == "black_litterman":
            # Black-Litterman expected returns
            self.expected_returns = self._black_litterman_returns()
        else:
            # Default to historical
            self.expected_returns = self.returns_data.mean().values
        
        return self.expected_returns
    
    def _calculate_covariance_matrix(self) -> np.ndarray:
        """Calculate covariance matrix."""
        if self.covariance_matrix is not None:
            return self.covariance_matrix
        
        method = self.parameters.covariance_matrix_method
        
        if method == "sample":
            self.covariance_matrix = self.returns_data.cov().values
        elif method == "shrinkage":
            self.covariance_matrix = self._shrinkage_covariance()
        elif method == "factor_based":
            self.covariance_matrix = self._factor_based_covariance()
        else:
            self.covariance_matrix = self.returns_data.cov().values
        
        return self.covariance_matrix
    
    def _black_litterman_returns(self) -> np.ndarray:
        """Calculate Black-Litterman expected returns."""
        try:
            # Prior returns (market equilibrium)
            market_returns = self.returns_data.mean().values
            market_cap_weights = np.ones(self.n_assets) / self.n_assets  # Equal weights as prior
            
            # Market-implied returns
            risk_aversion = self.parameters.risk_aversion
            cov_matrix = self.returns_data.cov().values
            pi = risk_aversion * cov_matrix @ market_cap_weights
            
            # Views (simplified - in practice would be analyst views)
            views = np.zeros(self.n_assets)  # No specific views for now
            
            # View uncertainty
            omega = np.eye(self.n_assets) * self.parameters.view_uncertainty
            
            # Black-Litterman formula
            tau = 0.025  # Scaling factor (typical value)
            
            # Combine prior and views
            try:
                M1 = np.linalg.inv(tau * cov_matrix)
                M2 = np.linalg.inv(omega)
                
                # Expected returns
                BL_returns = np.linalg.inv(
                    M1 + M2.T @ M2
                ) @ (M1 @ pi + M2.T @ M2 @ views)
                
                return BL_returns
                
            except np.linalg.LinAlgError:
                # Fallback to prior if matrix inversion fails
                logger.warning("Black-Litterman matrix inversion failed, using prior returns")
                return pi
                
        except Exception as e:
            logger.error(f"Black-Litterman calculation error: {str(e)}")
            return self.returns_data.mean().values
    
    def _shrinkage_covariance(self) -> np.ndarray:
        """Calculate shrinkage covariance matrix."""
        try:
            sample_cov = self.returns_data.cov().values
            
            # Simple shrinkage target (diagonal matrix with sample variances)
            shrinkage_target = np.diag(np.diag(sample_cov))
            shrinkage_intensity = 0.2  # 20% shrinkage
            
            shrunk_cov = (1 - shrinkage_intensity) * sample_cov + shrinkage_intensity * shrinkage_target
            
            return shrunk_cov
            
        except Exception as e:
            logger.error(f"Shrinkage covariance error: {str(e)}")
            return self.returns_data.cov().values
    
    def _factor_based_covariance(self) -> np.ndarray:
        """Calculate factor-based covariance matrix."""
        try:
            # Simplified factor model
            # In practice, would use PCA or fundamental factors
            
            sample_cov = self.returns_data.cov().values
            
            # Extract first few principal components
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(5, self.n_assets))
            pca.fit(self.returns_data)
            
            # Reconstruct covariance from factors
            explained_variance = pca.explained_variance_ratio_
            factors = pca.components_
            
            # Factor-based covariance
            factor_cov = np.zeros((self.n_assets, self.n_assets))
            for i in range(len(explained_variance)):
                factor_cov += explained_variance[i] * np.outer(factors[i], factors[i])
            
            # Idiosyncratic component
            residual_variance = np.diag(sample_cov) - np.diag(factor_cov)
            factor_cov += np.diag(np.maximum(residual_variance, 0))
            
            return factor_cov
            
        except Exception as e:
            logger.error(f"Factor-based covariance error: {str(e)}")
            return self.returns_data.cov().values
    
    def _portfolio_metrics(self, weights: np.ndarray) -> Dict[str, float]:
        """Calculate portfolio metrics for given weights."""
        try:
            expected_returns = self._calculate_expected_returns()
            cov_matrix = self._calculate_covariance_matrix()
            
            # Expected return
            portfolio_return = np.dot(weights, expected_returns)
            
            # Volatility
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            # Sharpe ratio
            excess_return = portfolio_return - self.parameters.risk_free_rate
            sharpe_ratio = excess_return / portfolio_vol if portfolio_vol > 0 else 0
            
            # VaR (assumes normal distribution)
            portfolio_std_daily = portfolio_vol / np.sqrt(252)
            var_95 = norm.ppf(1 - self.parameters.confidence_level) * portfolio_std_daily
            
            # CVaR (Expected Shortfall)
            cvar_95 = (1 / (1 - self.parameters.confidence_level)) * norm.pdf(norm.ppf(1 - self.parameters.confidence_level)) * portfolio_std_daily
            
            # Maximum drawdown (simplified estimate)
            # In practice, would use historical portfolio returns
            max_drawdown_estimate = 2.33 * portfolio_std_daily  # Approximation
            
            return {
                'expected_return': portfolio_return,
                'expected_volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio,
                'var_95': abs(var_95),
                'cvar_95': abs(cvar_95),
                'max_drawdown': max_drawdown_estimate
            }
            
        except Exception as e:
            logger.error(f"Portfolio metrics calculation error: {str(e)}")
            return {
                'expected_return': 0.0,
                'expected_volatility': 0.0,
                'sharpe_ratio': 0.0,
                'var_95': 0.0,
                'cvar_95': 0.0,
                'max_drawdown': 0.0
            }


class ModernPortfolioTheoryOptimizer(BaseOptimizer):
    """
    Modern Portfolio Theory (Markowitz) Optimizer.
    
    Implements the classic Markowitz mean-variance optimization
    with various objective functions.
    """
    
    async def optimize(self) -> OptimizationResult:
        """Run MPT optimization."""
        try:
            expected_returns = self._calculate_expected_returns()
            cov_matrix = self._calculate_covariance_matrix()
            
            # Set up optimization problem
            constraints = self._setup_constraints()
            bounds = self._setup_bounds()
            
            # Define objective function based on optimization type
            if self.parameters.optimization_objective == "max_sharpe":
                objective_func = self._sharpe_ratio_objective
            elif self.parameters.optimization_objective == "min_variance":
                objective_func = self._variance_objective
            elif self.parameters.optimization_objective == "max_return":
                objective_func = self._return_objective
            elif self.parameters.optimization_objective == "risk_parity":
                objective_func = self._risk_parity_objective
            else:
                objective_func = self._sharpe_ratio_objective  # Default
            
            # Solve optimization
            result = minimize(
                objective_func,
                x0=np.ones(self.n_assets) / self.n_assets,  # Equal weights starting point
                args=(expected_returns, cov_matrix),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            # Calculate portfolio metrics
            if result.success:
                weights = result.x
                metrics = self._portfolio_metrics(weights)
                
                # Normalize weights to sum to 1
                weights = weights / np.sum(weights)
                metrics = self._portfolio_metrics(weights)
                
                return OptimizationResult(
                    weights=weights,
                    expected_return=metrics['expected_return'],
                    expected_volatility=metrics['expected_volatility'],
                    sharpe_ratio=metrics['sharpe_ratio'],
                    max_drawdown=metrics['max_drawdown'],
                    var_95=metrics['var_95'],
                    cvar_95=metrics['cvar_95'],
                    objective_value=result.fun,
                    optimization_method="Modern Portfolio Theory",
                    constraints_satisfied=True,
                    convergence_achieved=True,
                    solver_info={
                        'success': result.success,
                        'message': result.message,
                        'iterations': result.nit,
                        'function_evaluations': result.nfev
                    }
                )
            else:
                # Return equal weights if optimization fails
                weights = np.ones(self.n_assets) / self.n_assets
                metrics = self._portfolio_metrics(weights)
                
                return OptimizationResult(
                    weights=weights,
                    expected_return=metrics['expected_return'],
                    expected_volatility=metrics['expected_volatility'],
                    sharpe_ratio=metrics['sharpe_ratio'],
                    max_drawdown=metrics['max_drawdown'],
                    var_95=metrics['var_95'],
                    cvar_95=metrics['cvar_95'],
                    objective_value=float('inf'),
                    optimization_method="Modern Portfolio Theory (Fallback)",
                    constraints_satisfied=False,
                    convergence_achieved=False,
                    solver_info={
                        'success': False,
                        'message': result.message,
                        'error': 'Optimization failed'
                    }
                )
            
        except Exception as e:
            logger.error(f"MPT optimization error: {str(e)}")
            # Return equal weights on error
            weights = np.ones(self.n_assets) / self.n_assets
            metrics = self._portfolio_metrics(weights)
            
            return OptimizationResult(
                weights=weights,
                expected_return=metrics['expected_return'],
                expected_volatility=metrics['expected_volatility'],
                sharpe_ratio=metrics['sharpe_ratio'],
                max_drawdown=metrics['max_drawdown'],
                var_95=metrics['var_95'],
                cvar_95=metrics['cvar_95'],
                objective_value=float('inf'),
                optimization_method="Modern Portfolio Theory (Error Fallback)",
                constraints_satisfied=False,
                convergence_achieved=False,
                solver_info={'error': str(e)}
            )
    
    def _setup_constraints(self) -> List[Dict[str, Any]]:
        """Set up optimization constraints."""
        constraints = []
        
        # Budget constraint: weights sum to 1
        constraints.append({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0
        })
        
        # Additional constraints from parameters
        if self.parameters.constraints:
            cons = self.parameters.constraints
            
            # Max volatility constraint
            if cons.max_volatility < 1.0:
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda w, cov=cov_matrix: cons.max_volatility - np.sqrt(np.dot(w.T, np.dot(cov, w)))
                })
        
        return constraints
    
    def _setup_bounds(self) -> List[Tuple[float, float]]:
        """Set up variable bounds."""
        if self.parameters.constraints:
            min_w = self.parameters.constraints.min_weight
            max_w = self.parameters.constraints.max_weight
        else:
            min_w = 0.0
            max_w = 1.0
        
        return [(min_w, max_w) for _ in range(self.n_assets)]
    
    def _sharpe_ratio_objective(self, weights: np.ndarray, expected_returns: np.ndarray, 
                              cov_matrix: np.ndarray) -> float:
        """Negative Sharpe ratio objective function."""
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        if portfolio_vol == 0:
            return float('inf')
        
        sharpe_ratio = (portfolio_return - self.parameters.risk_free_rate) / portfolio_vol
        return -sharpe_ratio  # Minimize negative Sharpe = maximize Sharpe
    
    def _variance_objective(self, weights: np.ndarray, expected_returns: np.ndarray, 
                          cov_matrix: np.ndarray) -> float:
        """Portfolio variance objective function."""
        return np.dot(weights.T, np.dot(cov_matrix, weights))
    
    def _return_objective(self, weights: np.ndarray, expected_returns: np.ndarray, 
                        cov_matrix: np.ndarray) -> float:
        """Negative portfolio return objective function."""
        return -np.dot(weights, expected_returns)
    
    def _risk_parity_objective(self, weights: np.ndarray, expected_returns: np.ndarray, 
                             cov_matrix: np.ndarray) -> float:
        """Risk parity objective function."""
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        if portfolio_vol == 0:
            return float('inf')
        
        # Calculate risk contributions
        marginal_contrib = np.dot(cov_matrix, weights)
        risk_contrib = weights * marginal_contrib / portfolio_vol
        
        # Target equal risk contribution
        target_risk_contrib = 1.0 / self.n_assets
        
        # Minimize deviation from equal risk contribution
        return np.sum((risk_contrib - target_risk_contrib) ** 2)


class BlackLittermanOptimizer(BaseOptimizer):
    """
    Black-Litterman Model Optimizer.
    
    Implements the Black-Litterman model combining market equilibrium
    with investor views for portfolio optimization.
    """
    
    async def optimize(self) -> OptimizationResult:
        """Run Black-Litterman optimization."""
        try:
            # Calculate Black-Litterman expected returns
            expected_returns = self._black_litterman_returns()
            cov_matrix = self._calculate_covariance_matrix()
            
            # Set up constraints
            constraints = self._setup_constraints()
            bounds = self._setup_bounds()
            
            # Black-Litterman objective (maximize utility)
            def bl_objective(weights, mu, Sigma):
                # Utility function: return - 0.5 * risk_aversion * variance
                portfolio_return = np.dot(weights, mu)
                portfolio_variance = np.dot(weights.T, np.dot(Sigma, weights))
                utility = portfolio_return - 0.5 * self.parameters.risk_aversion * portfolio_variance
                return -utility  # Minimize negative utility = maximize utility
            
            # Solve optimization
            result = minimize(
                bl_objective,
                x0=np.ones(self.n_assets) / self.n_assets,
                args=(expected_returns, cov_matrix),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            # Process results
            if result.success:
                weights = result.x / np.sum(result.x)  # Normalize
                metrics = self._portfolio_metrics(weights)
                
                return OptimizationResult(
                    weights=weights,
                    expected_return=metrics['expected_return'],
                    expected_volatility=metrics['expected_volatility'],
                    sharpe_ratio=metrics['sharpe_ratio'],
                    max_drawdown=metrics['max_drawdown'],
                    var_95=metrics['var_95'],
                    cvar_95=metrics['cvar_95'],
                    objective_value=result.fun,
                    optimization_method="Black-Litterman",
                    constraints_satisfied=True,
                    convergence_achieved=True,
                    solver_info={
                        'success': result.success,
                        'message': result.message,
                        'risk_aversion': self.parameters.risk_aversion
                    }
                )
            else:
                # Fallback to market capitalization weights
                market_weights = np.ones(self.n_assets) / self.n_assets
                metrics = self._portfolio_metrics(market_weights)
                
                return OptimizationResult(
                    weights=market_weights,
                    expected_return=metrics['expected_return'],
                    expected_volatility=metrics['expected_volatility'],
                    sharpe_ratio=metrics['sharpe_ratio'],
                    max_drawdown=metrics['max_drawdown'],
                    var_95=metrics['var_95'],
                    cvar_95=metrics['cvar_95'],
                    objective_value=float('inf'),
                    optimization_method="Black-Litterman (Fallback)",
                    constraints_satisfied=False,
                    convergence_achieved=False,
                    solver_info={'error': result.message}
                )
            
        except Exception as e:
            logger.error(f"Black-Litterman optimization error: {str(e)}")
            # Return equal weights
            weights = np.ones(self.n_assets) / self.n_assets
            metrics = self._portfolio_metrics(weights)
            
            return OptimizationResult(
                weights=weights,
                expected_return=metrics['expected_return'],
                expected_volatility=metrics['expected_volatility'],
                sharpe_ratio=metrics['sharpe_ratio'],
                max_drawdown=metrics['max_drawdown'],
                var_95=metrics['var_95'],
                cvar_95=metrics['cvar_95'],
                objective_value=float('inf'),
                optimization_method="Black-Litterman (Error Fallback)",
                constraints_satisfied=False,
                convergence_achieved=False,
                solver_info={'error': str(e)}
            )


class RobustOptimizer(BaseOptimizer):
    """
    Robust Portfolio Optimizer.
    
    Implements robust optimization techniques that account for
    parameter uncertainty and model risk.
    """
    
    async def optimize(self) -> OptimizationResult:
        """Run robust optimization."""
        try:
            expected_returns = self._calculate_expected_returns()
            cov_matrix = self._calculate_covariance_matrix()
            
            # Create uncertainty sets
            uncertainty_sets = self._create_uncertainty_sets(expected_returns, cov_matrix)
            
            # Robust optimization objective
            def robust_objective(weights):
                if self.parameters.worst_case_optimization:
                    # Worst-case scenario
                    worst_return = self._worst_case_return(weights, uncertainty_sets['returns'])
                    worst_variance = self._worst_case_variance(weights, uncertainty_sets['covariance'])
                    return -(worst_return - 0.5 * self.parameters.risk_aversion * worst_variance)
                else:
                    # Robust optimization with uncertainty sets
                    portfolio_return = np.dot(weights, expected_returns)
                    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
                    
                    # Add robustness penalty
                    robustness_penalty = self._robustness_penalty(weights, uncertainty_sets)
                    
                    return -(portfolio_return - 0.5 * self.parameters.risk_aversion * portfolio_variance + robustness_penalty)
            
            # Set up constraints
            constraints = self._setup_constraints()
            bounds = self._setup_bounds()
            
            # Solve using differential evolution for global optimization
            result = differential_evolution(
                robust_objective,
                bounds=bounds,
                seed=42,
                maxiter=1000,
                popsize=15
            )
            
            # Process results
            if result.success:
                weights = result.x / np.sum(result.x)
                metrics = self._portfolio_metrics(weights)
                
                return OptimizationResult(
                    weights=weights,
                    expected_return=metrics['expected_return'],
                    expected_volatility=metrics['expected_volatility'],
                    sharpe_ratio=metrics['sharpe_ratio'],
                    max_drawdown=metrics['max_drawdown'],
                    var_95=metrics['var_95'],
                    cvar_95=metrics['cvar_95'],
                    objective_value=result.fun,
                    optimization_method="Robust Optimization",
                    constraints_satisfied=True,
                    convergence_achieved=True,
                    solver_info={
                        'success': result.success,
                        'message': result.message,
                        'robustness_level': self.parameters.robustness_level
                    }
                )
            else:
                # Fallback to standard MPT
                fallback_optimizer = ModernPortfolioTheoryOptimizer(self.returns_data, self.parameters)
                return await fallback_optimizer.optimize()
            
        except Exception as e:
            logger.error(f"Robust optimization error: {str(e)}")
            # Fallback to equal weights
            weights = np.ones(self.n_assets) / self.n_assets
            metrics = self._portfolio_metrics(weights)
            
            return OptimizationResult(
                weights=weights,
                expected_return=metrics['expected_return'],
                expected_volatility=metrics['expected_volatility'],
                sharpe_ratio=metrics['sharpe_ratio'],
                max_drawdown=metrics['max_drawdown'],
                var_95=metrics['var_95'],
                cvar_95=metrics['cvar_95'],
                objective_value=float('inf'),
                optimization_method="Robust Optimization (Error Fallback)",
                constraints_satisfied=False,
                convergence_achieved=False,
                solver_info={'error': str(e)}
            )
    
    def _create_uncertainty_sets(self, expected_returns: np.ndarray, 
                               cov_matrix: np.ndarray) -> Dict[str, np.ndarray]:
        """Create uncertainty sets for robust optimization."""
        try:
            robustness_level = self.parameters.robustness_level
            
            # Return uncertainty set
            return_uncertainty = robustness_level * np.std(expected_returns) * np.ones(len(expected_returns))
            
            # Covariance uncertainty set (simplified)
            cov_uncertainty = robustness_level * np.diag(cov_matrix)
            
            return {
                'returns': return_uncertainty,
                'covariance': cov_uncertainty
            }
            
        except Exception as e:
            logger.error(f"Uncertainty set creation error: {str(e)}")
            return {
                'returns': np.zeros(len(expected_returns)),
                'covariance': np.zeros(len(expected_returns))
            }
    
    def _worst_case_return(self, weights: np.ndarray, return_uncertainty: np.ndarray) -> float:
        """Calculate worst-case return."""
        # Assume returns can be as low as mean - uncertainty
        return np.dot(weights, -np.abs(return_uncertainty))
    
    def _worst_case_variance(self, weights: np.ndarray, cov_uncertainty: np.ndarray) -> float:
        """Calculate worst-case variance."""
        # Assume covariance can be as high as mean + uncertainty
        worst_cov = np.diag(np.abs(cov_uncertainty))
        return np.dot(weights.T, np.dot(worst_cov, weights))
    
    def _robustness_penalty(self, weights: np.ndarray, uncertainty_sets: Dict[str, np.ndarray]) -> float:
        """Calculate robustness penalty."""
        # Simplified robustness penalty based on concentration
        concentration_penalty = np.sum(weights ** 2) - 1.0 / len(weights)
        return self.parameters.robustness_level * concentration_penalty


class MultiObjectiveOptimizer(BaseOptimizer):
    """
    Multi-Objective Portfolio Optimizer.
    
    Optimizes multiple objectives simultaneously including
    return, risk, and ESG factors.
    """
    
    async def optimize(self) -> OptimizationResult:
        """Run multi-objective optimization."""
        try:
            expected_returns = self._calculate_expected_returns()
            cov_matrix = self._calculate_covariance_matrix()
            
            # Define multi-objective function
            def multi_objective(weights):
                metrics = self._portfolio_metrics(weights)
                
                # Normalize objectives
                return_objective = -metrics['expected_return']  # Maximize return
                risk_objective = metrics['expected_volatility   # Minimize risk
                
                # ESG objective (simplified - assume we have ESG scores)
                esg_score = self._calculate_esg_score(weights) if hasattr(self, '_esg_data') else 0.5
                esg_objective = -esg_score  # Maximize ESG score
                
                # Weighted combination
                total_objective = (
                    self.parameters.return_weight * return_objective +
                    self.parameters.risk_weight * risk_objective +
                    self.parameters.esg_weight * esg_objective
                )
                
                return total_objective
            
            # Set up constraints
            constraints = self._setup_constraints()
            bounds = self._setup_bounds()
            
            # Solve optimization
            result = minimize(
                multi_objective,
                x0=np.ones(self.n_assets) / self.n_assets,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            # Process results
            if result.success:
                weights = result.x / np.sum(result.x)
                metrics = self._portfolio_metrics(weights)
                
                # Calculate ESG score
                esg_score = self._calculate_esg_score(weights) if hasattr(self, '_esg_data') else 0.5
                
                return OptimizationResult(
                    weights=weights,
                    expected_return=metrics['expected_return'],
                    expected_volatility=metrics['expected_volatility'],
                    sharpe_ratio=metrics['sharpe_ratio'],
                    max_drawdown=metrics['max_drawdown'],
                    var_95=metrics['var_95'],
                    cvar_95=metrics['cvar_95'],
                    objective_value=result.fun,
                    optimization_method="Multi-Objective",
                    constraints_satisfied=True,
                    convergence_achieved=True,
                    solver_info={
                        'success': result.success,
                        'message': result.message,
                        'return_weight': self.parameters.return_weight,
                        'risk_weight': self.parameters.risk_weight,
                        'esg_weight': self.parameters.esg_weight,
                        'esg_score': esg_score
                    }
                )
            else:
                # Fallback to standard optimization
                fallback_optimizer = ModernPortfolioTheoryOptimizer(self.returns_data, self.parameters)
                result = await fallback_optimizer.optimize()
                result.optimization_method = "Multi-Objective (Fallback)"
                return result
            
        except Exception as e:
            logger.error(f"Multi-objective optimization error: {str(e)}")
            # Fallback to equal weights
            weights = np.ones(self.n_assets) / self.n_assets
            metrics = self._portfolio_metrics(weights)
            
            return OptimizationResult(
                weights=weights,
                expected_return=metrics['expected_return'],
                expected_volatility=metrics['expected_volatility'],
                sharpe_ratio=metrics['sharpe_ratio'],
                max_drawdown=metrics['max_drawdown'],
                var_95=metrics['var_95'],
                cvar_95=metrics['cvar_95'],
                objective_value=float('inf'),
                optimization_method="Multi-Objective (Error Fallback)",
                constraints_satisfied=False,
                convergence_achieved=False,
                solver_info={'error': str(e)}
            )
    
    def _calculate_esg_score(self, weights: np.ndarray) -> float:
        """Calculate portfolio ESG score."""
        try:
            if not hasattr(self, '_esg_data'):
                return 0.5  # Neutral ESG score
            
            return np.dot(weights, self._esg_data)
            
        except Exception:
            return 0.5


class PortfolioOptimizer:
    """
    Main portfolio optimization interface.
    
    Coordinates different optimization methods and provides
    a unified interface for portfolio optimization.
    """
    
    def __init__(self, returns_data: pd.DataFrame):
        """
        Initialize portfolio optimizer.
        
        Args:
            returns_data: Historical return data
        """
        self.returns_data = returns_data
        self.optimization_results = {}
        
    async def optimize_portfolio(self, parameters: OptimizationParameters) -> OptimizationResult:
        """
        Optimize portfolio using specified method.
        
        Args:
            parameters: Optimization parameters
            
        Returns:
            Optimization result
        """
        try:
            # Choose optimizer based on objective
            optimizer_type = parameters.optimization_objective
            
            if optimizer_type == "black_litterman":
                optimizer = BlackLittermanOptimizer(self.returns_data, parameters)
            elif optimizer_type in ["robust", "robust_optimization"]:
                optimizer = RobustOptimizer(self.returns_data, parameters)
            elif optimizer_type in ["multi_objective", "esg_optimization"]:
                optimizer = MultiObjectiveOptimizer(self.returns_data, parameters)
            else:
                # Default to standard MPT
                optimizer = ModernPortfolioTheoryOptimizer(self.returns_data, parameters)
            
            # Run optimization
            result = await optimizer.optimize()
            
            # Store result
            self.optimization_results[datetime.now()] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Portfolio optimization error: {str(e)}")
            raise
    
    async def compare_optimization_methods(self, methods: List[str]) -> Dict[str, OptimizationResult]:
        """
        Compare multiple optimization methods.
        
        Args:
            methods: List of optimization methods to compare
            
        Returns:
            Dictionary of results by method
        """
        try:
            results = {}
            
            for method in methods:
                try:
                    params = OptimizationParameters(optimization_objective=method)
                    result = await self.optimize_portfolio(params)
                    results[method] = result
                except Exception as e:
                    logger.error(f"Optimization method {method} failed: {str(e)}")
                    results[method] = None
            
            return results
            
        except Exception as e:
            logger.error(f"Optimization comparison error: {str(e)}")
            return {}
    
    def calculate_efficient_frontier(self, num_portfolios: int = 100) -> Dict[str, Any]:
        """
        Calculate efficient frontier using Monte Carlo simulation.
        
        Args:
            num_portfolios: Number of random portfolios to generate
            
        Returns:
            Efficient frontier data
        """
        try:
            returns_data = self.returns_data
            n_assets = len(returns_data.columns)
            
            # Generate random weights
            weights_array = np.random.random((num_portfolios, n_assets))
            weights_array = weights_array / weights_array.sum(axis=1)[:, np.newaxis]
            
            # Calculate portfolio metrics
            portfolio_returns = []
            portfolio_volatilities = []
            portfolio_sharpe_ratios = []
            
            for weights in weights_array:
                portfolio_return = np.sum(returns_data.mean() * weights) * 252  # Annualized
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns_data.cov() * 252, weights)))
                
                portfolio_returns.append(portfolio_return)
                portfolio_volatilities.append(portfolio_volatility)
                
                sharpe_ratio = (portfolio_return - 0.02) / portfolio_volatility if portfolio_volatility > 0 else 0
                portfolio_sharpe_ratios.append(sharpe_ratio)
            
            # Find efficient frontier
            max_sharpe_idx = np.argmax(portfolio_sharpe_ratios)
            min_vol_idx = np.argmin(portfolio_volatilities)
            
            return {
                'portfolio_returns': portfolio_returns,
                'portfolio_volatilities': portfolio_volatilities,
                'portfolio_sharpe_ratios': portfolio_sharpe_ratios,
                'max_sharpe_portfolio': {
                    'return': portfolio_returns[max_sharpe_idx],
                    'volatility': portfolio_volatilities[max_sharpe_idx],
                    'sharpe_ratio': portfolio_sharpe_ratios[max_sharpe_idx],
                    'weights': weights_array[max_sharpe_idx].tolist()
                },
                'min_volatility_portfolio': {
                    'return': portfolio_returns[min_vol_idx],
                    'volatility': portfolio_volatilities[min_vol_idx],
                    'sharpe_ratio': portfolio_sharpe_ratios[min_vol_idx],
                    'weights': weights_array[min_vol_idx].tolist()
                },
                'assets': returns_data.columns.tolist(),
                'num_portfolios': num_portfolios
            }
            
        except Exception as e:
            logger.error(f"Efficient frontier calculation error: {str(e)}")
            return {}
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get summary of optimization results.
        
        Returns:
            Summary of all optimization results
        """
        try:
            if not self.optimization_results:
                return {'message': 'No optimization results available'}
            
            results = list(self.optimization_results.values())
            
            return {
                'total_optimizations': len(results),
                'latest_optimization': {
                    'timestamp': max(self.optimization_results.keys()),
                    'expected_return': results[-1].expected_return,
                    'expected_volatility': results[-1].expected_volatility,
                    'sharpe_ratio': results[-1].sharpe_ratio,
                    'optimization_method': results[-1].optimization_method
                },
                'best_performance': {
                    'highest_sharpe': max(results, key=lambda x: x.sharpe_ratio),
                    'lowest_volatility': min(results, key=lambda x: x.expected_volatility),
                    'highest_return': max(results, key=lambda x: x.expected_return)
                },
                'optimization_history': [
                    {
                        'timestamp': timestamp,
                        'method': result.optimization_method,
                        'sharpe_ratio': result.sharpe_ratio,
                        'expected_return': result.expected_return,
                        'expected_volatility': result.expected_volatility
                    }
                    for timestamp, result in self.optimization_results.items()
                ]
            }
            
        except Exception as e:
            logger.error(f"Optimization summary error: {str(e)}")
            return {'error': str(e)}


class PortfolioOptimizationFactory:
    """Factory class for creating portfolio optimization models."""
    
    @staticmethod
    def create_optimizer(optimization_type: str, returns_data: pd.DataFrame, 
                        parameters: OptimizationParameters = None) -> BaseOptimizer:
        """
        Create portfolio optimizer instance.
        
        Args:
            optimization_type: Type of optimization
            returns_data: Historical return data
            parameters: Optimization parameters
            
        Returns:
            Portfolio optimizer instance
        """
        if parameters is None:
            parameters = OptimizationParameters()
        
        optimizer_types = {
            'mpt': ModernPortfolioTheoryOptimizer,
            'markowitz': ModernPortfolioTheoryOptimizer,
            'black_litterman': BlackLittermanOptimizer,
            'robust': RobustOptimizer,
            'multi_objective': MultiObjectiveOptimizer,
            'esg': MultiObjectiveOptimizer
        }
        
        if optimization_type.lower() not in optimizer_types:
            raise ValueError(f"Unsupported optimization type: {optimization_type}")
        
        return optimizer_types[optimization_type.lower()](returns_data, parameters)
