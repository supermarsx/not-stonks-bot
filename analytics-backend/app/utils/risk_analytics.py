"""
Risk Analytics and Management
Advanced risk calculations including VaR, stress testing, and scenario analysis
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class RiskAnalytics:
    """Advanced risk analytics and stress testing"""
    
    @staticmethod
    def monte_carlo_var(
        returns: pd.Series,
        confidence_levels: List[float] = [0.95, 0.99],
        num_simulations: int = 10000,
        time_horizon: int = 1
    ) -> Dict[str, Dict[str, float]]:
        """
        Monte Carlo Value at Risk simulation
        
        Uses historical return distribution to simulate future scenarios
        """
        
        # Fit distribution to historical returns
        mu = returns.mean()
        sigma = returns.std()
        
        # Generate random scenarios
        random_returns = np.random.normal(mu, sigma, (num_simulations, time_horizon))
        
        # Calculate portfolio returns for each scenario
        if time_horizon == 1:
            scenario_returns = random_returns.flatten()
        else:
            # Compound returns over time horizon
            scenario_returns = np.prod(1 + random_returns, axis=1) - 1
        
        var_results = {}
        
        for confidence in confidence_levels:
            alpha = 1 - confidence
            
            # Calculate VaR and CVaR
            var = np.percentile(scenario_returns, alpha * 100)
            cvar = scenario_returns[scenario_returns <= var].mean()
            
            var_results[f"{confidence:.0%}"] = {
                'var_return': var,
                'cvar_return': cvar,
                'var_percentage': var * 100,
                'cvar_percentage': cvar * 100,
                'scenarios_at_risk': len(scenario_returns[scenario_returns <= var]),
                'worst_case_scenario': scenario_returns.min(),
                'best_case_scenario': scenario_returns.max(),
                'mean_scenario': scenario_returns.mean(),
                'scenario_volatility': scenario_returns.std()
            }
        
        return var_results
    
    @staticmethod
    def parametric_var(
        portfolio_value: float,
        expected_return: float,
        volatility: float,
        confidence_levels: List[float] = [0.95, 0.99],
        time_horizon: int = 1
    ) -> Dict[str, Dict[str, float]]:
        """
        Parametric (Analytical) VaR calculation
        
        Assumes normal distribution of returns
        """
        
        var_results = {}
        
        # Adjust for time horizon
        horizon_return = expected_return * time_horizon
        horizon_volatility = volatility * np.sqrt(time_horizon)
        
        for confidence in confidence_levels:
            alpha = 1 - confidence
            z_score = stats.norm.ppf(alpha)
            
            # Calculate VaR
            var_return = horizon_return + z_score * horizon_volatility
            var_dollar = portfolio_value * var_return
            
            # CVaR calculation for normal distribution
            cvar_return = horizon_return - horizon_volatility * stats.norm.pdf(z_score) / alpha
            cvar_dollar = portfolio_value * cvar_return
            
            var_results[f"{confidence:.0%}"] = {
                'var_return': var_return,
                'var_dollar': var_dollar,
                'cvar_return': cvar_return,
                'cvar_dollar': cvar_dollar,
                'var_percentage': var_return * 100,
                'cvar_percentage': cvar_return * 100,
                'z_score': z_score,
                'expected_return': horizon_return,
                'volatility': horizon_volatility
            }
        
        return var_results
    
    @staticmethod
    def stress_testing(
        portfolio_weights: Dict[str, float],
        asset_returns: Dict[str, pd.Series],
        stress_scenarios: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Stress testing analysis with predefined scenarios
        
        stress_scenarios format:
        {
            'scenario_name': {'asset1': shock_return, 'asset2': shock_return, ...}
        }
        """
        
        stress_results = {}
        
        for scenario_name, shocks in stress_scenarios.items():
            scenario_return = 0
            scenario_impact = {}
            
            for asset, weight in portfolio_weights.items():
                shock_return = shocks.get(asset, 0)
                asset_impact = weight * shock_return
                scenario_return += asset_impact
                
                scenario_impact[asset] = {
                    'weight': weight,
                    'shock_return': shock_return,
                    'contribution': asset_impact,
                    'contribution_percentage': asset_impact * 100
                }
            
            stress_results[scenario_name] = {
                'total_scenario_return': scenario_return,
                'total_scenario_percentage': scenario_return * 100,
                'asset_contributions': scenario_impact,
                'largest_contributor': max(scenario_impact.items(), key=lambda x: abs(x[1]['contribution']))[0] if scenario_impact else None,
                'largest_contribution': max(abs(impact['contribution']) for impact in scenario_impact.values()) if scenario_impact else 0
            }
        
        return stress_results
    
    @staticmethod
    def correlation_analysis(
        returns_data: Dict[str, pd.Series]
    ) -> Dict[str, any]:
        """
        Portfolio correlation and diversification analysis
        """
        
        # Create combined dataframe
        df = pd.DataFrame(returns_data)
        
        # Calculate correlation matrix
        correlation_matrix = df.corr()
        
        # Calculate average correlations
        avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
        
        # Find highest and lowest correlations
        correlation_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                asset1 = correlation_matrix.columns[i]
                asset2 = correlation_matrix.columns[j]
                corr_value = correlation_matrix.iloc[i, j]
                correlation_pairs.append({
                    'asset1': asset1,
                    'asset2': asset2,
                    'correlation': corr_value
                })
        
        # Sort by correlation
        highest_correlation = max(correlation_pairs, key=lambda x: x['correlation'])
        lowest_correlation = min(correlation_pairs, key=lambda x: x['correlation'])
        
        # Diversification ratio calculation
        # Diversification Ratio = Portfolio Volatility / Weighted Average Asset Volatility
        asset_volatilities = df.std()
        weights = {asset: 1/len(returns_data) for asset in returns_data.keys()}  # Equal weights assumption
        
        portfolio_returns = sum(df[asset] * weights[asset] for asset in weights.keys())
        portfolio_volatility = portfolio_returns.std()
        weighted_avg_volatility = sum(asset_volatilities[asset] * weights[asset] for asset in weights.keys())
        
        diversification_ratio = portfolio_volatility / weighted_avg_volatility if weighted_avg_volatility > 0 else 1
        
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'average_correlation': avg_correlation,
            'highest_correlation': highest_correlation,
            'lowest_correlation': lowest_correlation,
            'diversification_ratio': diversification_ratio,
            'portfolio_volatility': portfolio_volatility,
            'asset_volatilities': asset_volatilities.to_dict(),
            'correlation_summary': {
                'highly_correlated_pairs': [pair for pair in correlation_pairs if pair['correlation'] > 0.7],
                'negatively_correlated_pairs': [pair for pair in correlation_pairs if pair['correlation'] < -0.3],
                'diversification_score': max(0, min(100, (1 - avg_correlation) * 100))  # 0-100 scale
            }
        }
    
    @staticmethod
    def tail_risk_analysis(
        returns: pd.Series,
        tail_percentile: float = 0.05
    ) -> Dict[str, float]:
        """
        Tail risk analysis including skewness, kurtosis, and tail behavior
        """
        
        # Basic statistics
        mean_return = returns.mean()
        volatility = returns.std()
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Tail statistics
        left_tail = returns.quantile(tail_percentile)
        right_tail = returns.quantile(1 - tail_percentile)
        
        # Tail ratio (right tail / abs(left tail))
        tail_ratio = right_tail / abs(left_tail) if left_tail != 0 else 0
        
        # Expected shortfall (CVaR) for tail
        tail_returns = returns[returns <= left_tail]
        expected_shortfall = tail_returns.mean() if len(tail_returns) > 0 else left_tail
        
        # Maximum consecutive losses
        consecutive_losses = 0
        max_consecutive_losses = 0
        for ret in returns:
            if ret < 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0
        
        return {
            'mean_return': mean_return,
            'volatility': volatility,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'excess_kurtosis': kurtosis - 3,  # Excess over normal distribution
            'left_tail_5pct': left_tail,
            'right_tail_95pct': right_tail,
            'tail_ratio': tail_ratio,
            'expected_shortfall': expected_shortfall,
            'max_consecutive_losses': max_consecutive_losses,
            'tail_risk_score': RiskAnalytics._calculate_tail_risk_score(skewness, kurtosis, expected_shortfall),
            'distribution_assessment': RiskAnalytics._assess_distribution(skewness, kurtosis)
        }
    
    @staticmethod
    def portfolio_concentration_risk(
        portfolio_weights: Dict[str, float],
        asset_returns: Optional[Dict[str, pd.Series]] = None
    ) -> Dict[str, any]:
        """
        Analyze portfolio concentration and diversification
        """
        
        weights = list(portfolio_weights.values())
        assets = list(portfolio_weights.keys())
        
        # Concentration metrics
        largest_position = max(weights)
        top_3_positions = sum(sorted(weights, reverse=True)[:3])
        top_5_positions = sum(sorted(weights, reverse=True)[:5])
        
        # Herfindahl-Hirschman Index (HHI)
        hhi = sum(w**2 for w in weights)
        
        # Effective number of positions
        effective_positions = 1 / hhi if hhi > 0 else 0
        
        # Concentration ratio
        concentration_ratio = largest_position
        
        # Risk contribution analysis (if returns provided)
        risk_contributions = {}
        if asset_returns:
            # Calculate marginal contributions to risk
            portfolio_returns = pd.Series(dtype=float)
            for asset, weight in portfolio_weights.items():
                if asset in asset_returns:
                    if len(portfolio_returns) == 0:
                        portfolio_returns = weight * asset_returns[asset]
                    else:
                        portfolio_returns += weight * asset_returns[asset]
            
            portfolio_volatility = portfolio_returns.std()
            
            for asset, weight in portfolio_weights.items():
                if asset in asset_returns:
                    # Marginal contribution to risk
                    marginal_var = asset_returns[asset].cov(portfolio_returns) / portfolio_volatility if portfolio_volatility > 0 else 0
                    component_var = weight * marginal_var
                    risk_contributions[asset] = {
                        'weight': weight,
                        'marginal_var': marginal_var,
                        'component_var': component_var,
                        'risk_contribution_pct': (component_var / portfolio_volatility * 100) if portfolio_volatility > 0 else 0
                    }
        
        return {
            'concentration_metrics': {
                'largest_position_pct': largest_position * 100,
                'top_3_positions_pct': top_3_positions * 100,
                'top_5_positions_pct': top_5_positions * 100,
                'herfindahl_index': hhi,
                'effective_positions': effective_positions,
                'concentration_score': RiskAnalytics._calculate_concentration_score(hhi, largest_position)
            },
            'diversification_assessment': {
                'diversification_level': RiskAnalytics._assess_diversification(effective_positions),
                'concentration_warning': largest_position > 0.25,  # Warning if any position > 25%
                'risk_level': RiskAnalytics._assess_concentration_risk(hhi)
            },
            'position_details': [
                {'asset': asset, 'weight_pct': weight * 100, 'rank': i+1}
                for i, (asset, weight) in enumerate(sorted(portfolio_weights.items(), key=lambda x: x[1], reverse=True))
            ],
            'risk_contributions': risk_contributions
        }
    
    @staticmethod
    def _calculate_tail_risk_score(skewness: float, kurtosis: float, expected_shortfall: float) -> float:
        """Calculate composite tail risk score (0-100, higher = more risk)"""
        
        # Normalize components
        skew_score = min(50, abs(skewness) * 20)  # Cap at 50
        kurt_score = min(30, max(0, kurtosis - 3) * 10)  # Excess kurtosis, cap at 30
        shortfall_score = min(20, abs(expected_shortfall) * 400)  # Cap at 20
        
        return skew_score + kurt_score + shortfall_score
    
    @staticmethod
    def _assess_distribution(skewness: float, kurtosis: float) -> str:
        """Assess return distribution characteristics"""
        
        if abs(skewness) < 0.5 and abs(kurtosis - 3) < 1:
            return "Approximately Normal"
        elif skewness < -0.5:
            return "Negatively Skewed (Left Tail Risk)"
        elif skewness > 0.5:
            return "Positively Skewed (Right Tail Opportunity)"
        elif kurtosis > 5:
            return "Fat Tails (High Kurtosis)"
        elif kurtosis < 1:
            return "Thin Tails (Low Kurtosis)"
        else:
            return "Non-Normal Distribution"
    
    @staticmethod
    def _calculate_concentration_score(hhi: float, largest_position: float) -> float:
        """Calculate concentration score (0-100, higher = more concentrated)"""
        
        # HHI component (0-50)
        hhi_score = min(50, hhi * 50)
        
        # Largest position component (0-50)
        position_score = min(50, largest_position * 100)
        
        return hhi_score + position_score
    
    @staticmethod
    def _assess_diversification(effective_positions: float) -> str:
        """Assess diversification level"""
        
        if effective_positions >= 10:
            return "Well Diversified"
        elif effective_positions >= 5:
            return "Moderately Diversified"
        elif effective_positions >= 3:
            return "Concentrated"
        else:
            return "Highly Concentrated"
    
    @staticmethod
    def _assess_concentration_risk(hhi: float) -> str:
        """Assess concentration risk level"""
        
        if hhi < 0.1:
            return "Low Concentration Risk"
        elif hhi < 0.25:
            return "Moderate Concentration Risk"
        elif hhi < 0.5:
            return "High Concentration Risk"
        else:
            return "Very High Concentration Risk"