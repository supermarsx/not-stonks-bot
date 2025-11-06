"""
Portfolio Optimization Analytics
Advanced portfolio optimization including Black-Litterman model and efficient frontier
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
from scipy.optimize import minimize, minimize_scalar
from scipy import linalg
import warnings
warnings.filterwarnings('ignore')

class PortfolioOptimization:
    """Advanced portfolio optimization and allocation strategies"""
    
    @staticmethod
    def mean_variance_optimization(
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_aversion: float = 1.0,
        weight_bounds: Optional[Tuple[float, float]] = None,
        target_return: Optional[float] = None
    ) -> Dict[str, any]:
        """
        Mean-Variance Optimization (Markowitz)
        
        Finds optimal portfolio weights given expected returns and covariance
        """
        
        n_assets = len(expected_returns)
        
        if weight_bounds is None:
            weight_bounds = (0.0, 1.0)  # Long-only portfolio
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]  # Weights sum to 1
        
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: np.dot(w, expected_returns) - target_return
            })
        
        # Bounds for each weight
        bounds = [weight_bounds] * n_assets
        
        # Objective function
        if target_return is None:
            # Maximize utility: return - (risk_aversion/2) * variance
            def objective(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
                return -(portfolio_return - 0.5 * risk_aversion * portfolio_variance)
        else:
            # Minimize variance for target return
            def objective(weights):
                return np.dot(weights.T, np.dot(covariance_matrix, weights))
        
        # Initial guess (equal weights)
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # Optimization
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'disp': False}
        )
        
        if result.success:
            optimal_weights = result.x
            portfolio_return = np.dot(optimal_weights, expected_returns)
            portfolio_variance = np.dot(optimal_weights.T, np.dot(covariance_matrix, optimal_weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
            
            return {
                'optimal_weights': optimal_weights.tolist(),
                'portfolio_return': portfolio_return,
                'portfolio_volatility': portfolio_volatility,
                'portfolio_variance': portfolio_variance,
                'sharpe_ratio': sharpe_ratio,
                'optimization_success': True,
                'objective_value': result.fun
            }
        else:
            return {
                'optimal_weights': initial_weights.tolist(),
                'optimization_success': False,
                'error_message': result.message
            }
    
    @staticmethod
    def black_litterman_optimization(
        market_caps: np.ndarray,
        returns_data: pd.DataFrame,
        investor_views: Optional[Dict[str, float]] = None,
        view_confidence: float = 0.25,
        risk_aversion: float = 3.0,
        tau: float = 0.025
    ) -> Dict[str, any]:
        """
        Black-Litterman Portfolio Optimization
        
        Combines market equilibrium with investor views
        """
        
        # Calculate market equilibrium
        market_weights = market_caps / np.sum(market_caps)
        
        # Historical covariance matrix
        cov_matrix = returns_data.cov().values
        
        # Implied equilibrium returns (reverse optimization)
        implied_returns = risk_aversion * np.dot(cov_matrix, market_weights)
        
        # If no views provided, return market portfolio
        if not investor_views:
            return {
                'optimal_weights': market_weights.tolist(),
                'implied_returns': implied_returns.tolist(),
                'portfolio_return': np.dot(market_weights, implied_returns),
                'portfolio_volatility': np.sqrt(np.dot(market_weights.T, np.dot(cov_matrix, market_weights))),
                'views_incorporated': False,
                'market_equilibrium': True
            }
        
        # Process investor views
        view_assets = list(investor_views.keys())
        view_returns = list(investor_views.values())
        asset_names = list(returns_data.columns)
        
        # Create picking matrix P (maps views to assets)
        P = np.zeros((len(view_assets), len(asset_names)))
        for i, asset in enumerate(view_assets):
            if asset in asset_names:
                asset_idx = asset_names.index(asset)
                P[i, asset_idx] = 1
        
        # View vector Q
        Q = np.array(view_returns)
        
        # Uncertainty matrix Omega (diagonal matrix of view uncertainties)
        Omega = np.eye(len(view_assets)) * view_confidence
        
        # Black-Litterman formula
        # Precision of prior (tau * covariance)^-1
        tau_cov_inv = linalg.inv(tau * cov_matrix)
        
        # Precision of views P^T * Omega^-1 * P
        view_precision = np.dot(P.T, np.dot(linalg.inv(Omega), P))
        
        # New expected returns
        bl_precision = tau_cov_inv + view_precision
        bl_cov = linalg.inv(bl_precision)
        
        # Blended expected returns
        prior_term = np.dot(tau_cov_inv, implied_returns)
        view_term = np.dot(P.T, np.dot(linalg.inv(Omega), Q))
        bl_returns = np.dot(bl_cov, prior_term + view_term)
        
        # New covariance matrix
        bl_cov_matrix = bl_cov + cov_matrix
        
        # Optimize with Black-Litterman inputs
        optimization_result = PortfolioOptimization.mean_variance_optimization(
            bl_returns, bl_cov_matrix, risk_aversion
        )
        
        # Calculate changes from market portfolio
        weight_changes = np.array(optimization_result['optimal_weights']) - market_weights
        
        return {
            **optimization_result,
            'implied_returns': implied_returns.tolist(),
            'bl_returns': bl_returns.tolist(),
            'market_weights': market_weights.tolist(),
            'weight_changes': weight_changes.tolist(),
            'views_incorporated': True,
            'investor_views': investor_views,
            'view_confidence': view_confidence,
            'tau': tau,
            'risk_aversion': risk_aversion
        }
    
    @staticmethod
    def efficient_frontier(
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        num_portfolios: int = 100,
        weight_bounds: Optional[Tuple[float, float]] = None
    ) -> Dict[str, any]:
        """
        Generate efficient frontier portfolios
        """
        
        if weight_bounds is None:
            weight_bounds = (0.0, 1.0)
        
        # Calculate range of target returns
        min_return = np.min(expected_returns)
        max_return = np.max(expected_returns)
        target_returns = np.linspace(min_return, max_return, num_portfolios)
        
        efficient_portfolios = []
        
        for target_return in target_returns:
            result = PortfolioOptimization.mean_variance_optimization(
                expected_returns, covariance_matrix, 
                target_return=target_return, weight_bounds=weight_bounds
            )
            
            if result['optimization_success']:
                efficient_portfolios.append({
                    'target_return': target_return,
                    'portfolio_return': result['portfolio_return'],
                    'portfolio_volatility': result['portfolio_volatility'],
                    'sharpe_ratio': result['sharpe_ratio'],
                    'weights': result['optimal_weights']
                })
        
        if not efficient_portfolios:
            return {'error': 'Failed to generate efficient frontier'}
        
        # Find special portfolios
        # Minimum variance portfolio
        min_var_portfolio = min(efficient_portfolios, key=lambda p: p['portfolio_volatility'])
        
        # Maximum Sharpe ratio portfolio
        max_sharpe_portfolio = max(efficient_portfolios, key=lambda p: p['sharpe_ratio'])
        
        # Maximum return portfolio
        max_return_portfolio = max(efficient_portfolios, key=lambda p: p['portfolio_return'])
        
        return {
            'efficient_portfolios': efficient_portfolios,
            'min_variance_portfolio': min_var_portfolio,
            'max_sharpe_portfolio': max_sharpe_portfolio,
            'max_return_portfolio': max_return_portfolio,
            'frontier_statistics': {
                'num_portfolios': len(efficient_portfolios),
                'return_range': [min_return, max_return],
                'volatility_range': [
                    min(p['portfolio_volatility'] for p in efficient_portfolios),
                    max(p['portfolio_volatility'] for p in efficient_portfolios)
                ],
                'sharpe_range': [
                    min(p['sharpe_ratio'] for p in efficient_portfolios),
                    max(p['sharpe_ratio'] for p in efficient_portfolios)
                ]
            }
        }
    
    @staticmethod
    def risk_parity_optimization(
        covariance_matrix: np.ndarray,
        target_volatility: Optional[float] = None
    ) -> Dict[str, any]:
        """
        Risk Parity Portfolio Optimization
        
        Allocates capital so each asset contributes equally to portfolio risk
        """
        
        n_assets = covariance_matrix.shape[0]
        
        # Objective function: minimize sum of squared differences in risk contributions
        def risk_parity_objective(weights):
            portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Calculate risk contributions
            marginal_contrib = np.dot(covariance_matrix, weights)
            risk_contrib = weights * marginal_contrib / portfolio_volatility if portfolio_volatility > 0 else np.zeros(n_assets)
            
            # Target is equal risk contribution (1/n for each asset)
            target_contrib = portfolio_variance / n_assets
            
            # Minimize squared deviations from equal risk
            return np.sum((risk_contrib - target_contrib) ** 2)
        
        # Constraints and bounds
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]  # Weights sum to 1
        bounds = [(0.01, 0.99)] * n_assets  # Small bounds to avoid numerical issues
        
        # Initial guess (equal weights)
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # Optimization
        result = minimize(
            risk_parity_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'disp': False}
        )
        
        if result.success:
            optimal_weights = result.x
            portfolio_variance = np.dot(optimal_weights.T, np.dot(covariance_matrix, optimal_weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Calculate final risk contributions
            marginal_contrib = np.dot(covariance_matrix, optimal_weights)
            risk_contrib = optimal_weights * marginal_contrib / portfolio_volatility if portfolio_volatility > 0 else np.zeros(n_assets)
            risk_contrib_pct = (risk_contrib / np.sum(risk_contrib)) * 100 if np.sum(risk_contrib) > 0 else np.zeros(n_assets)
            
            # Scale to target volatility if provided
            if target_volatility is not None:
                scaling_factor = target_volatility / portfolio_volatility
                scaled_weights = optimal_weights * scaling_factor
                scaled_volatility = portfolio_volatility * scaling_factor
            else:
                scaled_weights = optimal_weights
                scaled_volatility = portfolio_volatility
            
            return {
                'optimal_weights': scaled_weights.tolist(),
                'portfolio_volatility': scaled_volatility,
                'risk_contributions': risk_contrib.tolist(),
                'risk_contributions_pct': risk_contrib_pct.tolist(),
                'optimization_success': True,
                'objective_value': result.fun,
                'target_volatility': target_volatility,
                'scaling_applied': target_volatility is not None
            }
        else:
            return {
                'optimal_weights': initial_weights.tolist(),
                'optimization_success': False,
                'error_message': result.message
            }
    
    @staticmethod
    def tactical_allocation_signals(
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        returns_momentum: Dict[str, float],
        volatility_regime: Dict[str, float],
        rebalance_threshold: float = 0.05
    ) -> Dict[str, any]:
        """
        Generate tactical allocation signals and rebalancing recommendations
        """
        
        allocation_signals = {}
        rebalance_actions = []
        total_drift = 0
        
        for asset in current_weights.keys():
            current_weight = current_weights[asset]
            target_weight = target_weights.get(asset, 0)
            momentum = returns_momentum.get(asset, 0)
            volatility = volatility_regime.get(asset, 1)
            
            # Calculate drift
            weight_drift = current_weight - target_weight
            drift_percentage = abs(weight_drift) / target_weight if target_weight > 0 else abs(weight_drift)
            total_drift += abs(weight_drift)
            
            # Generate signals
            signal_strength = 0
            signal_direction = "HOLD"
            
            # Drift signal
            if drift_percentage > rebalance_threshold:
                if weight_drift > 0:
                    signal_direction = "REDUCE"
                    signal_strength = min(100, drift_percentage * 200)
                else:
                    signal_direction = "INCREASE"
                    signal_strength = min(100, drift_percentage * 200)
            
            # Momentum adjustment
            momentum_adjustment = momentum * 20  # Scale momentum to signal strength
            if momentum > 0.05:  # Strong positive momentum
                if signal_direction == "REDUCE":
                    signal_strength *= 0.7  # Reduce selling pressure
                elif signal_direction == "INCREASE":
                    signal_strength *= 1.3  # Increase buying signal
            elif momentum < -0.05:  # Strong negative momentum
                if signal_direction == "REDUCE":
                    signal_strength *= 1.3  # Increase selling pressure
                elif signal_direction == "INCREASE":
                    signal_strength *= 0.7  # Reduce buying signal
            
            # Volatility adjustment
            volatility_multiplier = 1 + (volatility - 1) * 0.5  # Moderate volatility impact
            signal_strength *= volatility_multiplier
            
            # Cap signal strength
            signal_strength = min(100, max(0, signal_strength))
            
            allocation_signals[asset] = {
                'current_weight': current_weight,
                'target_weight': target_weight,
                'weight_drift': weight_drift,
                'drift_percentage': drift_percentage,
                'signal_direction': signal_direction,
                'signal_strength': signal_strength,
                'momentum': momentum,
                'volatility_regime': volatility,
                'rebalance_needed': drift_percentage > rebalance_threshold
            }
            
            # Generate rebalance actions
            if drift_percentage > rebalance_threshold:
                action_size = abs(weight_drift)
                rebalance_actions.append({
                    'asset': asset,
                    'action': signal_direction,
                    'current_weight_pct': current_weight * 100,
                    'target_weight_pct': target_weight * 100,
                    'adjustment_pct': weight_drift * 100,
                    'action_size': action_size,
                    'priority': signal_strength
                })
        
        # Sort rebalance actions by priority
        rebalance_actions.sort(key=lambda x: x['priority'], reverse=True)
        
        # Portfolio-level analysis
        portfolio_analysis = {
            'total_drift': total_drift,
            'drift_percentage': (total_drift / sum(target_weights.values())) * 100 if target_weights else 0,
            'rebalance_recommended': total_drift > rebalance_threshold,
            'number_of_actions': len(rebalance_actions),
            'high_priority_actions': len([a for a in rebalance_actions if a['priority'] > 70]),
            'portfolio_momentum': np.mean(list(returns_momentum.values())),
            'average_volatility': np.mean(list(volatility_regime.values()))
        }
        
        return {
            'allocation_signals': allocation_signals,
            'rebalance_actions': rebalance_actions,
            'portfolio_analysis': portfolio_analysis,
            'recommendations': PortfolioOptimization._generate_rebalance_recommendations(
                portfolio_analysis, rebalance_actions
            )
        }
    
    @staticmethod
    def _generate_rebalance_recommendations(
        portfolio_analysis: Dict,
        rebalance_actions: List[Dict]
    ) -> List[str]:
        """Generate human-readable rebalancing recommendations"""
        
        recommendations = []
        
        if not portfolio_analysis['rebalance_recommended']:
            recommendations.append("✅ Portfolio is well-balanced. No immediate rebalancing required.")
            return recommendations
        
        recommendations.append(f"🔄 Portfolio rebalancing recommended (Total drift: {portfolio_analysis['drift_percentage']:.1f}%)")
        
        # High priority actions
        high_priority = [a for a in rebalance_actions if a['priority'] > 70]
        if high_priority:
            recommendations.append(f"🚨 {len(high_priority)} high-priority adjustments needed:")
            for action in high_priority[:3]:  # Top 3
                recommendations.append(
                    f"   • {action['action']} {action['asset']}: "
                    f"{action['current_weight_pct']:.1f}% → {action['target_weight_pct']:.1f}% "
                    f"({action['adjustment_pct']:+.1f}%)"
                )
        
        # Market timing considerations
        if portfolio_analysis['portfolio_momentum'] > 0.05:
            recommendations.append("📈 Consider momentum: Market trends may delay some rebalancing")
        elif portfolio_analysis['portfolio_momentum'] < -0.05:
            recommendations.append("📉 Market weakness detected: Prioritize defensive rebalancing")
        
        if portfolio_analysis['average_volatility'] > 1.5:
            recommendations.append("⚠️ High volatility regime: Consider gradual rebalancing approach")
        
        return recommendations