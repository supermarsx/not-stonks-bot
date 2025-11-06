"""
Stress Testing Framework

Comprehensive stress testing system including:
- Historical scenario replay
- Monte Carlo stress testing
- Sensitivity analysis
- Correlation breakdown scenarios
- Liquidity stress testing
- Black swan event simulation
- Portfolio heat maps
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
import asyncio
from scipy import stats, optimize
from concurrent.futures import ThreadPoolExecutor
import json

from database.models.trading import Position, Trade
from database.models.market_data import PriceData

logger = logging.getLogger(__name__)


@dataclass
class StressTestScenario:
    """Stress test scenario definition."""
    scenario_name: str
    description: str
    scenario_type: str  # 'historical', 'monte_carlo', 'custom', 'shock'
    parameters: Dict[str, Any]
    probability: float  # Scenario probability (0-1)
    severity_level: str  # 'low', 'medium', 'high', 'extreme'
    market_impact: Dict[str, float]  # Asset class impacts
    duration_days: int


@dataclass
class StressTestResult:
    """Stress test result."""
    scenario_name: str
    portfolio_impact: Dict[str, Any]
    position_impacts: Dict[str, float]
    risk_metrics: Dict[str, float]
    stress_score: float
    survival_probability: float
    recommended_actions: List[str]


class ScenarioEngine:
    """
    Stress testing scenario engine.
    
    Manages and executes various stress testing scenarios
    for portfolio risk assessment.
    """
    
    def __init__(self, positions: List[Dict[str, Any]]):
        """
        Initialize scenario engine.
        
        Args:
            positions: List of position dictionaries
        """
        self.positions = positions
        self.portfolio_value = self._calculate_portfolio_value()
        self.scenarios = []
        
    async def run_historical_stress_test(self, scenario_name: str, 
                                       market_returns: Dict[str, pd.Series]) -> StressTestResult:
        """
        Run historical stress test using past market events.
        
        Args:
            scenario_name: Name of historical scenario
            market_returns: Historical market return series by asset
            
        Returns:
            Stress test results
        """
        try:
            # Define historical scenarios
            historical_scenarios = {
                'black_monday_1987': {
                    'start_date': '1987-10-19',
                    'returns': {'SPY': -0.22, 'QQQ': -0.25, 'TLT': 0.05, 'GLD': 0.03},
                    'description': 'Black Monday market crash'
                },
                'dot_com_bubble': {
                    'start_date': '2000-03-24',
                    'returns': {'SPY': -0.49, 'QQQ': -0.83, 'XLF': -0.45, 'XLK': -0.62},
                    'description': 'Dot-com bubble burst'
                },
                'financial_crisis_2008': {
                    'start_date': '2008-09-15',
                    'returns': {'SPY': -0.57, 'XLF': -0.80, 'TLT': 0.18, 'DXY': 0.15},
                    'description': '2008 Financial Crisis'
                },
                'covid_crash_2020': {
                    'start_date': '2020-03-16',
                    'returns': {'SPY': -0.34, 'QQQ': -0.42, 'XLV': -0.45, 'TLT': -0.15},
                    'description': 'COVID-19 Market Crash'
                },
                'european_debt_crisis': {
                    'start_date': '2011-08-05',
                    'returns': {'SPY': -0.19, 'EWZ': -0.23, 'TLT': 0.08, 'EURUSD': -0.12},
                    'description': 'European Debt Crisis'
                }
            }
            
            if scenario_name not in historical_scenarios:
                return StressTestResult(
                    scenario_name=scenario_name,
                    portfolio_impact={},
                    position_impacts={},
                    risk_metrics={},
                    stress_score=0.0,
                    survival_probability=0.0,
                    recommended_actions=[f"Unknown scenario: {scenario_name}"]
                )
            
            scenario = historical_scenarios[scenario_name]
            
            # Calculate portfolio impact
            portfolio_impact = self._calculate_historical_scenario_impact(scenario['returns'])
            
            # Calculate position-level impacts
            position_impacts = self._calculate_position_impacts(scenario['returns'])
            
            # Calculate risk metrics
            risk_metrics = self._calculate_stress_risk_metrics(portfolio_impact)
            
            # Calculate stress score
            stress_score = self._calculate_stress_score(portfolio_impact)
            
            # Survival probability
            survival_probability = self._calculate_survival_probability(portfolio_impact)
            
            # Generate recommendations
            recommendations = self._generate_stress_recommendations(portfolio_impact, stress_score)
            
            return StressTestResult(
                scenario_name=scenario_name,
                portfolio_impact=portfolio_impact,
                position_impacts=position_impacts,
                risk_metrics=risk_metrics,
                stress_score=stress_score,
                survival_probability=survival_probability,
                recommended_actions=recommendations
            )
            
        except Exception as e:
            logger.error(f"Historical stress test error: {str(e)}")
            return StressTestResult(
                scenario_name=scenario_name,
                portfolio_impact={'error': str(e)},
                position_impacts={},
                risk_metrics={},
                stress_score=0.0,
                survival_probability=0.0,
                recommended_actions=[f"Test failed: {str(e)}"]
            )
    
    async def run_monte_carlo_stress_test(self, num_simulations: int = 10000,
                                        stress_parameters: Dict[str, Any] = None) -> StressTestResult:
        """
        Run Monte Carlo stress test.
        
        Args:
            num_simulations: Number of simulation runs
            stress_parameters: Stress testing parameters
            
        Returns:
            Monte Carlo stress test results
        """
        try:
            if stress_parameters is None:
                stress_parameters = {
                    'correlation_breakdown': 0.8,
                    'volatility_multiplier': 3.0,
                    'mean_shift': -0.1
                }
            
            # Run simulations
            portfolio_outcomes = await self._run_monte_carlo_simulations(
                num_simulations, stress_parameters
            )
            
            # Analyze results
            portfolio_impact = self._analyze_monte_carlo_results(portfolio_outcomes)
            
            # Calculate position impacts
            position_impacts = self._analyze_position_stress_impacts(portfolio_outcomes)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_monte_carlo_risk_metrics(portfolio_outcomes)
            
            # Calculate stress score
            stress_score = self._calculate_monte_carlo_stress_score(portfolio_outcomes)
            
            # Survival probability
            survival_probability = self._calculate_monte_carlo_survival(portfolio_outcomes)
            
            # Generate recommendations
            recommendations = self._generate_monte_carlo_recommendations(
                portfolio_outcomes, stress_score
            )
            
            return StressTestResult(
                scenario_name='Monte Carlo Stress',
                portfolio_impact=portfolio_impact,
                position_impacts=position_impacts,
                risk_metrics=risk_metrics,
                stress_score=stress_score,
                survival_probability=survival_probability,
                recommended_actions=recommendations
            )
            
        except Exception as e:
            logger.error(f"Monte Carlo stress test error: {str(e)}")
            return StressTestResult(
                scenario_name='Monte Carlo Stress',
                portfolio_impact={'error': str(e)},
                position_impacts={},
                risk_metrics={},
                stress_score=0.0,
                survival_probability=0.0,
                recommended_actions=[f"Test failed: {str(e)}"]
            )
    
    async def run_custom_shock_scenario(self, asset_shocks: Dict[str, float],
                                      correlation_adjustment: float = 0.5) -> StressTestResult:
        """
        Run custom asset shock scenario.
        
        Args:
            asset_shocks: Dictionary of asset symbols and their shock magnitudes
            correlation_adjustment: Correlation adjustment factor (0-1)
            
        Returns:
            Custom shock scenario results
        """
        try:
            # Apply asset shocks
            shock_impact = self._apply_asset_shocks(asset_shocks)
            
            # Apply correlation breakdown
            correlation_impact = self._apply_correlation_breakdown(correlation_adjustment)
            
            # Combine impacts
            portfolio_impact = {
                'shock_only_pnl': shock_impact['portfolio_pnl'],
                'correlation_breakdown_pnl': correlation_impact['portfolio_pnl'],
                'combined_impact': shock_impact['portfolio_pnl'] + correlation_impact['portfolio_pnl'],
                'shock_method': 'custom_shock',
                'correlation_adjustment': correlation_adjustment
            }
            
            # Position impacts
            position_impacts = shock_impact['position_impacts']
            
            # Calculate metrics
            risk_metrics = {
                'max_loss': abs(min(0, portfolio_impact['combined_impact'])),
                'shock_severity': max(abs(shock) for shock in asset_shocks.values()),
                'correlation_breakdown_severity': abs(correlation_adjustment - 1.0)
            }
            
            # Stress score
            stress_score = min(100, abs(portfolio_impact['combined_impact']) / self.portfolio_value * 100)
            
            # Survival probability
            survival_probability = 1.0 if portfolio_impact['combined_impact'] > -0.5 * self.portfolio_value else 0.5
            
            recommendations = self._generate_shock_recommendations(asset_shocks, stress_score)
            
            return StressTestResult(
                scenario_name='Custom Shock Scenario',
                portfolio_impact=portfolio_impact,
                position_impacts=position_impacts,
                risk_metrics=risk_metrics,
                stress_score=stress_score,
                survival_probability=survival_probability,
                recommended_actions=recommendations
            )
            
        except Exception as e:
            logger.error(f"Custom shock scenario error: {str(e)}")
            return StressTestResult(
                scenario_name='Custom Shock Scenario',
                portfolio_impact={'error': str(e)},
                position_impacts={},
                risk_metrics={},
                stress_score=0.0,
                survival_probability=0.0,
                recommended_actions=[f"Test failed: {str(e)}"]
            )
    
    def _calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        try:
            total_value = 0.0
            for position in self.positions:
                quantity = position.get('quantity', 0)
                price = position.get('current_price', 100.0)
                total_value += abs(quantity * price)
            
            return total_value
        except Exception:
            return 0.0
    
    def _calculate_historical_scenario_impact(self, scenario_returns: Dict[str, float]) -> Dict[str, Any]:
        """Calculate portfolio impact from historical scenario."""
        try:
            portfolio_pnl = 0.0
            position_details = []
            
            for position in self.positions:
                symbol = position['symbol']
                quantity = position['quantity']
                current_price = position.get('current_price', 100.0)
                position_value = abs(quantity * current_price)
                
                # Apply asset-specific shock
                asset_return = scenario_returns.get(symbol, 0.0)
                
                # Apply shock
                if symbol in scenario_returns:
                    position_pnl = position_value * asset_return
                    portfolio_pnl += position_pnl
                    
                    position_details.append({
                        'symbol': symbol,
                        'position_value': position_value,
                        'asset_return': asset_return,
                        'position_pnl': position_pnl,
                        'pnl_percentage': (position_pnl / position_value * 100) if position_value > 0 else 0
                    })
            
            return {
                'portfolio_pnl': portfolio_pnl,
                'portfolio_pnl_percentage': (portfolio_pnl / self.portfolio_value * 100) if self.portfolio_value > 0 else 0,
                'position_details': position_details,
                'scenario_returns': scenario_returns
            }
            
        except Exception as e:
            logger.error(f"Historical scenario impact calculation error: {str(e)}")
            return {'portfolio_pnl': 0.0, 'portfolio_pnl_percentage': 0.0}
    
    def _calculate_position_impacts(self, scenario_returns: Dict[str, float]) -> Dict[str, float]:
        """Calculate position-level impacts."""
        try:
            impacts = {}
            
            for position in self.positions:
                symbol = position['symbol']
                quantity = position['quantity']
                price = position.get('current_price', 100.0)
                
                if symbol in scenario_returns:
                    asset_return = scenario_returns[symbol]
                    position_pnl = quantity * price * asset_return
                    impacts[symbol] = position_pnl
                else:
                    impacts[symbol] = 0.0
            
            return impacts
            
        except Exception as e:
            logger.error(f"Position impacts calculation error: {str(e)}")
            return {}
    
    def _calculate_stress_risk_metrics(self, portfolio_impact: Dict[str, Any]) -> Dict[str, float]:
        """Calculate risk metrics from stress scenario."""
        try:
            portfolio_pnl = portfolio_impact.get('portfolio_pnl', 0.0)
            
            return {
                'loss_amount': abs(min(0, portfolio_pnl)),
                'loss_percentage': abs(min(0, portfolio_pnl / self.portfolio_value * 100)) if self.portfolio_value > 0 else 0,
                'worst_position_loss': self._get_worst_position_loss(portfolio_impact),
                'diversification_benefit': self._calculate_diversification_benefit(portfolio_impact),
                'stress_loss_to_var_ratio': abs(portfolio_pnl) / (0.05 * self.portfolio_value)  # Assume 5% VaR
            }
            
        except Exception as e:
            logger.error(f"Stress risk metrics calculation error: {str(e)}")
            return {}
    
    def _calculate_stress_score(self, portfolio_impact: Dict[str, Any]) -> float:
        """Calculate overall stress score."""
        try:
            portfolio_pnl = portfolio_impact.get('portfolio_pnl', 0.0)
            loss_percentage = abs(min(0, portfolio_pnl / self.portfolio_value * 100)) if self.portfolio_value > 0 else 0
            
            # Stress score based on loss severity
            if loss_percentage >= 30:
                return 90 + (loss_percentage - 30) * 2  # 90-100 for extreme stress
            elif loss_percentage >= 20:
                return 70 + (loss_percentage - 20) * 2  # 70-90 for high stress
            elif loss_percentage >= 10:
                return 50 + (loss_percentage - 10) * 2  # 50-70 for medium stress
            elif loss_percentage >= 5:
                return 30 + (loss_percentage - 5) * 4   # 30-50 for low stress
            else:
                return loss_percentage * 6              # 0-30 for minimal stress
                
        except Exception:
            return 0.0
    
    def _calculate_survival_probability(self, portfolio_impact: Dict[str, Any]) -> float:
        """Calculate survival probability under stress scenario."""
        try:
            portfolio_pnl = portfolio_impact.get('portfolio_pnl', 0.0)
            loss_percentage = abs(min(0, portfolio_pnl / self.portfolio_value * 100)) if self.portfolio_value > 0 else 0
            
            # Survival probability decreases with loss severity
            if loss_percentage >= 50:
                return 0.1  # 10% survival probability
            elif loss_percentage >= 30:
                return 0.3  # 30% survival probability
            elif loss_percentage >= 20:
                return 0.5  # 50% survival probability
            elif loss_percentage >= 10:
                return 0.7  # 70% survival probability
            elif loss_percentage >= 5:
                return 0.85  # 85% survival probability
            else:
                return 0.95  # 95% survival probability
                
        except Exception:
            return 0.5
    
    def _get_worst_position_loss(self, portfolio_impact: Dict[str, Any]) -> float:
        """Get worst position loss from portfolio impact."""
        try:
            position_details = portfolio_impact.get('position_details', [])
            if not position_details:
                return 0.0
            
            losses = [abs(min(0, detail.get('position_pnl', 0))) for detail in position_details]
            return max(losses) if losses else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_diversification_benefit(self, portfolio_impact: Dict[str, Any]) -> float:
        """Calculate diversification benefit under stress."""
        try:
            position_details = portfolio_impact.get('position_details', [])
            if not position_details:
                return 0.0
            
            # Calculate weighted average loss
            total_weight = sum(detail.get('position_value', 0) for detail in position_details)
            if total_weight == 0:
                return 0.0
            
            weighted_avg_loss = sum(
                detail.get('position_value', 0) * abs(min(0, detail.get('position_pnl', 0)))
                for detail in position_details
            ) / total_weight
            
            # Calculate maximum potential loss if all perfectly correlated
            max_possible_loss = max(
                abs(min(0, detail.get('position_pnl', 0)))
                for detail in position_details
            )
            
            if max_possible_loss == 0:
                return 0.0
            
            # Diversification benefit as reduction in worst-case loss
            diversification_benefit = (max_possible_loss - weighted_avg_loss) / max_possible_loss * 100
            
            return max(0, diversification_benefit)
            
        except Exception:
            return 0.0
    
    def _generate_stress_recommendations(self, portfolio_impact: Dict[str, Any], 
                                       stress_score: float) -> List[str]:
        """Generate stress test recommendations."""
        try:
            recommendations = []
            
            portfolio_pnl = portfolio_impact.get('portfolio_pnl', 0.0)
            loss_percentage = abs(min(0, portfolio_pnl / self.portfolio_value * 100)) if self.portfolio_value > 0 else 0
            
            # Portfolio-level recommendations
            if stress_score >= 80:
                recommendations.append("URGENT: Reduce portfolio risk immediately")
                recommendations.append("Consider liquidating high-risk positions")
            elif stress_score >= 60:
                recommendations.append("High stress detected - review position sizing")
                recommendations.append("Consider hedging strategies")
            elif stress_score >= 40:
                recommendations.append("Moderate stress - monitor positions closely")
                recommendations.append("Consider rebalancing portfolio")
            
            # Position-specific recommendations
            position_details = portfolio_impact.get('position_details', [])
            worst_position = max(position_details, 
                               key=lambda x: abs(min(0, x.get('position_pnl', 0))), 
                               default=None)
            
            if worst_position and abs(min(0, worst_position.get('position_pnl', 0))) > 0.1 * self.portfolio_value:
                recommendations.append(f"Reduce exposure to {worst_position['symbol']} - highest stress contributor")
            
            # General recommendations
            if loss_percentage > 20:
                recommendations.append("Implement emergency risk controls")
                recommendations.append("Prepare liquidity contingency plans")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Stress recommendations generation error: {str(e)}")
            return ["Review stress test results manually"]
    
    async def _run_monte_carlo_simulations(self, num_simulations: int, 
                                         stress_parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run Monte Carlo simulations for stress testing."""
        try:
            simulations = []
            
            # Extract stress parameters
            correlation_breakdown = stress_parameters.get('correlation_breakdown', 0.8)
            volatility_multiplier = stress_parameters.get('volatility_multiplier', 3.0)
            mean_shift = stress_parameters.get('mean_shift', -0.1)
            
            for i in range(num_simulations):
                simulation_result = {
                    'simulation_id': i,
                    'portfolio_outcome': 0.0,
                    'position_outcomes': {},
                    'stress_factors': {}
                }
                
                # Generate correlated asset shocks
                asset_shocks = self._generate_correlated_shocks(correlation_breakdown)
                
                # Apply volatility scaling
                for symbol in asset_shocks:
                    asset_shocks[symbol] *= volatility_multiplier
                
                # Apply mean shift
                for symbol in asset_shocks:
                    asset_shocks[symbol] += mean_shift
                
                # Calculate portfolio outcome
                portfolio_outcome = 0.0
                
                for position in self.positions:
                    symbol = position['symbol']
                    quantity = position['quantity']
                    price = position.get('current_price', 100.0)
                    
                    if symbol in asset_shocks:
                        shock_return = asset_shocks[symbol]
                        position_outcome = quantity * price * shock_return
                        simulation_result['position_outcomes'][symbol] = position_outcome
                        portfolio_outcome += position_outcome
                    
                simulation_result['portfolio_outcome'] = portfolio_outcome
                simulation_result['stress_factors'] = asset_shocks
                
                simulations.append(simulation_result)
            
            return simulations
            
        except Exception as e:
            logger.error(f"Monte Carlo simulations error: {str(e)}")
            return []
    
    def _generate_correlated_shocks(self, correlation_breakdown: float) -> Dict[str, float]:
        """Generate correlated asset shocks."""
        try:
            # Common market factor
            market_shock = np.random.normal(-0.15, 0.25)  # Market drawdown with volatility
            
            # Asset-specific shocks with correlation structure
            asset_shocks = {}
            
            for position in self.positions:
                symbol = position['symbol']
                
                # Correlation with market factor
                market_correlation = 0.7 if 'SPY' in symbol.upper() else 0.5  # Simplified
                
                # Asset-specific component
                    # Asset-specific component
                idiosyncratic_shock = np.random.normal(0, 0.15)
                
                # Combine market and idiosyncratic components
                combined_shock = (
                    market_correlation * market_shock + 
                    np.sqrt(1 - market_correlation**2) * idiosyncratic_shock
                )
                
                # Apply correlation breakdown
                breakdown_shock = np.random.normal(0, 0.1)
                final_shock = (
                    correlation_breakdown * combined_shock + 
                    (1 - correlation_breakdown) * breakdown_shock
                )
                
                asset_shocks[symbol] = final_shock
            
            return asset_shocks
            
        except Exception as e:
            logger.error(f"Correlated shocks generation error: {str(e)}")
            return {symbol: np.random.normal(-0.1, 0.2) for position in self.positions for symbol in [position['symbol']]}
    
    def _analyze_monte_carlo_results(self, simulations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze Monte Carlo simulation results."""
        try:
            if not simulations:
                return {}
            
            portfolio_outcomes = [sim['portfolio_outcome'] for sim in simulations]
            
            return {
                'mean_outcome': np.mean(portfolio_outcomes),
                'std_outcome': np.std(portfolio_outcomes),
                'percentile_5': np.percentile(portfolio_outcomes, 5),
                'percentile_95': np.percentile(portfolio_outcomes, 95),
                'worst_case': min(portfolio_outcomes),
                'best_case': max(portfolio_outcomes),
                'loss_probability': np.mean(np.array(portfolio_outcomes) < 0),
                'extreme_loss_probability': np.mean(np.array(portfolio_outcomes) < -0.2 * self.portfolio_value),
                'mean_loss_given_loss': np.mean([x for x in portfolio_outcomes if x < 0])
            }
            
        except Exception as e:
            logger.error(f"Monte Carlo results analysis error: {str(e)}")
            return {}
    
    def _analyze_position_stress_impacts(self, simulations: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze position-level stress impacts."""
        try:
            if not simulations:
                return {}
            
            position_impacts = {}
            
            # Collect all symbols
            all_symbols = set()
            for sim in simulations:
                all_symbols.update(sim['position_outcomes'].keys())
            
            for symbol in all_symbols:
                position_outcomes = [sim['position_outcomes'].get(symbol, 0) for sim in simulations]
                position_impacts[symbol] = {
                    'mean_impact': np.mean(position_outcomes),
                    'worst_impact': min(position_outcomes),
                    'loss_probability': np.mean(np.array(position_outcomes) < 0)
                }
            
            return position_impacts
            
        except Exception as e:
            logger.error(f"Position stress impact analysis error: {str(e)}")
            return {}
    
    def _calculate_monte_carlo_risk_metrics(self, simulations: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate risk metrics from Monte Carlo simulations."""
        try:
            if not simulations:
                return {}
            
            portfolio_outcomes = [sim['portfolio_outcome'] for sim in simulations]
            
            return {
                'var_95': abs(np.percentile(portfolio_outcomes, 5)),
                'cvar_95': abs(np.mean([x for x in portfolio_outcomes if x <= np.percentile(portfolio_outcomes, 5)])),
                'expected_shortfall': abs(np.mean([x for x in portfolio_outcomes if x < 0])),
                'maximum_drawdown': abs(min(portfolio_outcomes)),
                'tail_risk_ratio': abs(np.mean([x for x in portfolio_outcomes if x < 0])) / (0.05 * self.portfolio_value)
            }
            
        except Exception as e:
            logger.error(f"Monte Carlo risk metrics calculation error: {str(e)}")
            return {}
    
    def _calculate_monte_carlo_stress_score(self, simulations: List[Dict[str, Any]]) -> float:
        """Calculate stress score from Monte Carlo results."""
        try:
            if not simulations:
                return 0.0
            
            portfolio_outcomes = [sim['portfolio_outcome'] for sim in simulations]
            worst_loss = abs(min(portfolio_outcomes))
            loss_percentage = worst_loss / self.portfolio_value * 100 if self.portfolio_value > 0 else 0
            
            return min(100, loss_percentage * 2)  # Scale to 0-100
            
        except Exception:
            return 0.0
    
    def _calculate_monte_carlo_survival(self, simulations: List[Dict[str, Any]]) -> float:
        """Calculate survival probability from Monte Carlo results."""
        try:
            if not simulations:
                return 0.0
            
            portfolio_outcomes = [sim['portfolio_outcome'] for sim in simulations]
            
            # Survival defined as portfolio value > 50% of initial
            survival_threshold = -0.5 * self.portfolio_value
            survival_count = sum(1 for outcome in portfolio_outcomes if outcome > survival_threshold)
            
            return survival_count / len(simulations)
            
        except Exception:
            return 0.0
    
    def _generate_monte_carlo_recommendations(self, simulations: List[Dict[str, Any]], 
                                            stress_score: float) -> List[str]:
        """Generate Monte Carlo stress test recommendations."""
        try:
            if not simulations:
                return ["No simulation data available"]
            
            portfolio_outcomes = [sim['portfolio_outcome'] for sim in simulations]
            loss_probability = np.mean(np.array(portfolio_outcomes) < 0)
            extreme_loss_prob = np.mean(np.array(portfolio_outcomes) < -0.2 * self.portfolio_value)
            
            recommendations = []
            
            if loss_probability > 0.7:
                recommendations.append("High probability of losses detected - reduce overall risk")
            elif loss_probability > 0.5:
                recommendations.append("Moderate loss probability - consider hedging strategies")
            
            if extreme_loss_prob > 0.2:
                recommendations.append("High probability of extreme losses - implement circuit breakers")
            elif extreme_loss_prob > 0.1:
                recommendations.append("Some probability of extreme losses - prepare contingency plans")
            
            if stress_score > 80:
                recommendations.append("Extreme stress scenarios likely - emergency risk controls required")
            elif stress_score > 60:
                recommendations.append("High stress scenarios possible - review risk limits")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Monte Carlo recommendations generation error: {str(e)}")
            return ["Review Monte Carlo results manually"]
    
    def _apply_asset_shocks(self, asset_shocks: Dict[str, float]) -> Dict[str, Any]:
        """Apply asset-specific shocks to portfolio."""
        try:
            portfolio_pnl = 0.0
            position_impacts = {}
            
            for position in self.positions:
                symbol = position['symbol']
                quantity = position['quantity']
                price = position.get('current_price', 100.0)
                
                if symbol in asset_shocks:
                    shock_return = asset_shocks[symbol]
                    position_pnl = quantity * price * shock_return
                    position_impacts[symbol] = position_pnl
                    portfolio_pnl += position_pnl
                else:
                    position_impacts[symbol] = 0.0
            
            return {
                'portfolio_pnl': portfolio_pnl,
                'position_impacts': position_impacts
            }
            
        except Exception as e:
            logger.error(f"Asset shock application error: {str(e)}")
            return {'portfolio_pnl': 0.0, 'position_impacts': {}}
    
    def _apply_correlation_breakdown(self, correlation_adjustment: float) -> Dict[str, Any]:
        """Apply correlation breakdown impact."""
        try:
            # Simplified correlation breakdown model
            # In practice, this would involve complex correlation matrix adjustments
            
            baseline_correlation = 0.5
            adjusted_correlation = correlation_adjustment * baseline_correlation
            
            # Calculate portfolio impact from correlation breakdown
            # This is a simplified model - in practice, would use actual correlation matrices
            
            portfolio_impact = 0.0
            
            for position in self.positions:
                symbol = position['symbol']
                quantity = position['quantity']
                price = position.get('current_price', 100.0)
                
                # Simulate correlation breakdown impact
                breakdown_shock = np.random.normal(0, 0.1) * (1 - adjusted_correlation)
                position_impact = quantity * price * breakdown_shock
                portfolio_impact += position_impact
            
            return {
                'portfolio_pnl': portfolio_impact,
                'correlation_adjustment': correlation_adjustment
            }
            
        except Exception as e:
            logger.error(f"Correlation breakdown application error: {str(e)}")
            return {'portfolio_pnl': 0.0, 'correlation_adjustment': correlation_adjustment}
    
    def _generate_shock_recommendations(self, asset_shocks: Dict[str, float], 
                                      stress_score: float) -> List[str]:
        """Generate recommendations for shock scenarios."""
        try:
            recommendations = []
            
            # Identify most stressed assets
            worst_shock = min(asset_shocks.values()) if asset_shocks else 0
            most_stressed_asset = None
            
            for symbol, shock in asset_shocks.items():
                if shock == worst_shock:
                    most_stressed_asset = symbol
                    break
            
            if worst_shock < -0.2:  # Asset down more than 20%
                recommendations.append(f"Reduce exposure to {most_stressed_asset} - severe stress")
            elif worst_shock < -0.1:  # Asset down more than 10%
                recommendations.append(f"Monitor {most_stressed_asset} closely - moderate stress")
            
            if stress_score > 70:
                recommendations.append("High stress scenario - consider reducing portfolio beta")
                recommendations.append("Implement additional hedging measures")
            
            # Liquidity recommendations
            shocked_assets = [symbol for symbol, shock in asset_shocks.items() if shock < -0.15]
            if shocked_assets:
                recommendations.append(f"Monitor liquidity for: {', '.join(shocked_assets)}")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Shock recommendations generation error: {str(e)}")
            return ["Review shock scenario results"]


class HistoricalScenarios:
    """
    Predefined historical market stress scenarios.
    
    Contains detailed historical market events and their impact
    on various asset classes for stress testing.
    """
    
    def __init__(self):
        """Initialize historical scenarios."""
        self.scenarios = self._load_historical_scenarios()
    
    def _load_historical_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined historical stress scenarios."""
        return {
            'black_monday_1987': {
                'name': 'Black Monday 1987',
                'date': '1987-10-19',
                'description': 'Global stock market crash on October 19, 1987',
                'duration_days': 3,
                'asset_returns': {
                    'US_Equity': -0.226,
                    'Intl_Equity': -0.187,
                    'US_Bonds': 0.058,
                    'Gold': 0.038,
                    'Commodities': -0.12,
                    'USD_Index': 0.041
                },
                'correlation_changes': {
                    'US_Intl_Equity': 0.89,
                    'Equity_Bond': -0.78,
                    'Gold_Equity': -0.45
                }
            },
            'dot_com_bubble': {
                'name': 'Dot-com Bubble Burst',
                'date': '2000-03-24',
                'description': 'Tech stock bubble burst and market decline',
                'duration_days': 365,
                'asset_returns': {
                    'US_Equity': -0.49,
                    'Tech_Equity': -0.83,
                    'US_Bonds': 0.086,
                    'REIT': -0.15,
                    'Commodities': 0.47,
                    'USD_Index': 0.032
                },
                'correlation_changes': {
                    'Tech_Bond': 0.12,
                    'Equity_Commodity': -0.34,
                    'US_Intl_Equity': 0.91
                }
            },
            'financial_crisis_2008': {
                'name': 'Global Financial Crisis 2008',
                'date': '2008-09-15',
                'description': 'Lehman Brothers collapse and global financial crisis',
                'duration_days': 545,
                'asset_returns': {
                    'US_Equity': -0.57,
                    'Financial_Equity': -0.80,
                    'US_Bonds': 0.183,
                    'REIT': -0.68,
                    'Gold': 0.048,
                    'USD_Index': 0.15,
                    'Commodities': -0.35
                },
                'correlation_changes': {
                    'Equity_Bond': 0.85,
                    'Financial_Equity': -0.91,
                    'Gold_Equity': -0.67
                }
            },
            'covid_crash_2020': {
                'name': 'COVID-19 Market Crash',
                'date': '2020-03-16',
                'description': 'COVID-19 pandemic market impact',
                'duration_days': 34,
                'asset_returns': {
                    'US_Equity': -0.34,
                    'Tech_Equity': -0.42,
                    'Health_Equity': -0.45,
                    'US_Bonds': -0.015,
                    'REIT': -0.42,
                    'Gold': 0.006,
                    'USD_Index': 0.025,
                    'Commodities': -0.28
                },
                'correlation_changes': {
                    'US_Intl_Equity': 0.94,
                    'Equity_Bond': -0.23,
                    'Tech_Health_Equity': 0.91
                }
            },
            'european_debt_crisis': {
                'name': 'European Debt Crisis',
                'date': '2011-08-05',
                'description': 'European sovereign debt crisis impact',
                'duration_days': 180,
                'asset_returns': {
                    'US_Equity': -0.19,
                    'EU_Equity': -0.23,
                    'EM_Equity': -0.15,
                    'US_Bonds': 0.085,
                    'EU_Bonds': -0.12,
                    'EURUSD': -0.12,
                    'Gold': 0.18
                },
                'correlation_changes': {
                    'US_EU_Equity': 0.87,
                    'US_EU_Bond': 0.76,
                    'EURUSD_Gold': -0.34
                }
            },
            'flash_crash_2010': {
                'name': 'Flash Crash 2010',
                'date': '2010-05-06',
                'description': 'Intraday flash crash in US equity markets',
                'duration_days': 1,
                'asset_returns': {
                    'US_Equity': -0.099,
                    'Tech_Equity': -0.12,
                    'US_Bonds': 0.023,
                    'USD_Index': 0.008
                },
                'correlation_changes': {
                    'US_Equity_Bond': -0.65,
                    'Tech_US_Equity': 0.95
                }
            },
            'taper_tantrum_2013': {
                'name': 'Taper Tantrum 2013',
                'date': '2013-05-22',
                'description': 'Fed tapering announcement impact on markets',
                'duration_days': 90,
                'asset_returns': {
                    'US_Equity': 0.083,
                    'EM_Equity': -0.17,
                    'US_Bonds': -0.06,
                    'EM_Bonds': -0.13,
                    'USD_Index': 0.035
                },
                'correlation_changes': {
                    'US_EM_Equity': 0.23,
                    'US_EM_Bond': 0.45,
                    'Equity_Bond': -0.12
                }
            }
        }
    
    def get_scenario(self, scenario_name: str) -> Optional[Dict[str, Any]]:
        """Get specific historical scenario."""
        return self.scenarios.get(scenario_name)
    
    def get_all_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """Get all available scenarios."""
        return self.scenarios
    
    def get_scenarios_by_type(self, scenario_type: str) -> Dict[str, Dict[str, Any]]:
        """Get scenarios by type (crisis, correction, event)."""
        type_mapping = {
            'crisis': ['black_monday_1987', 'financial_crisis_2008', 'covid_crash_2020'],
            'correction': ['dot_com_bubble', 'european_debt_crisis'],
            'event': ['flash_crash_2010', 'taper_tantrum_2013']
        }
        
        filtered_scenarios = {}
        scenario_names = type_mapping.get(scenario_type, [])
        
        for name in scenario_names:
            if name in self.scenarios:
                filtered_scenarios[name] = self.scenarios[name]
        
        return filtered_scenarios
    
    def create_custom_scenario(self, name: str, description: str, 
                             asset_returns: Dict[str, float],
                             duration_days: int = 30) -> Dict[str, Any]:
        """Create custom scenario based on historical patterns."""
        return {
            'name': name,
            'description': description,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'duration_days': duration_days,
            'asset_returns': asset_returns,
            'correlation_changes': {},
            'custom': True
        }


class MonteCarloStressTest:
    """
    Monte Carlo stress testing engine.
    
    Advanced Monte Carlo simulation for stress testing
    with sophisticated modeling of market dynamics.
    """
    
    def __init__(self, positions: List[Dict[str, Any]], 
                 asset_returns: Dict[str, pd.Series]):
        """
        Initialize Monte Carlo stress tester.
        
        Args:
            positions: Portfolio positions
            asset_returns: Historical asset return series
        """
        self.positions = positions
        self.asset_returns = asset_returns
        self.symbols = list(asset_returns.keys())
        
    async def run_comprehensive_stress_test(self, num_scenarios: int = 10000) -> Dict[str, Any]:
        """Run comprehensive Monte Carlo stress test."""
        try:
            # Define stress parameters
            stress_scenarios = [
                {'name': 'Market_Crash', 'severity': 'extreme', 'prob': 0.05},
                {'name': 'Market_Correction', 'severity': 'high', 'prob': 0.15},
                {'name': 'Volatility_Spike', 'severity': 'medium', 'prob': 0.30},
                {'name': 'Correlation_Breakdown', 'severity': 'high', 'prob': 0.20},
                {'name': 'Liquidity_Crisis', 'severity': 'extreme', 'prob': 0.10},
                {'name': 'Normal_Stress', 'severity': 'low', 'prob': 0.20}
            ]
            
            all_results = []
            
            for scenario in stress_scenarios:
                scenario_results = await self._run_stress_scenario(
                    scenario['name'], scenario['severity'], 
                    int(num_scenarios * scenario['prob'])
                )
                all_results.append(scenario_results)
            
            # Aggregate results
            aggregated_results = self._aggregate_stress_results(all_results)
            
            return aggregated_results
            
        except Exception as e:
            logger.error(f"Comprehensive stress test error: {str(e)}")
            return {'error': str(e)}
    
    async def _run_stress_scenario(self, scenario_name: str, severity: str, 
                                 num_simulations: int) -> Dict[str, Any]:
        """Run specific stress scenario."""
        try:
            # Define scenario parameters
            scenario_params = {
                'Market_Crash': {
                    'mean_shift': -0.3,
                    'volatility_multiplier': 4.0,
                    'correlation_breakdown': 0.3,
                    'liquidity_factor': 0.2
                },
                'Market_Correction': {
                    'mean_shift': -0.15,
                    'volatility_multiplier': 2.5,
                    'correlation_breakdown': 0.5,
                    'liquidity_factor': 0.5
                },
                'Volatility_Spike': {
                    'mean_shift': -0.05,
                    'volatility_multiplier': 3.0,
                    'correlation_breakdown': 0.7,
                    'liquidity_factor': 0.7
                },
                'Correlation_Breakdown': {
                    'mean_shift': -0.1,
                    'volatility_multiplier': 1.5,
                    'correlation_breakdown': 0.1,
                    'liquidity_factor': 0.8
                },
                'Liquidity_Crisis': {
                    'mean_shift': -0.25,
                    'volatility_multiplier': 3.5,
                    'correlation_breakdown': 0.4,
                    'liquidity_factor': 0.1
                },
                'Normal_Stress': {
                    'mean_shift': -0.08,
                    'volatility_multiplier': 2.0,
                    'correlation_breakdown': 0.6,
                    'liquidity_factor': 0.6
                }
            }
            
            params = scenario_params.get(scenario_name, scenario_params['Normal_Stress'])
            
            # Run simulations
            simulations = await self._run_advanced_simulations(num_simulations, params)
            
            # Analyze results
            return self._analyze_stress_scenario(scenario_name, severity, simulations)
            
        except Exception as e:
            logger.error(f"Stress scenario error: {str(e)}")
            return {'error': str(e), 'scenario': scenario_name}
    
    async def _run_advanced_simulations(self, num_simulations: int, 
                                      params: Dict[str, float]) -> List[Dict[str, Any]]:
        """Run advanced Monte Carlo simulations."""
        try:
            simulations = []
            
            for i in range(num_simulations):
                # Generate market scenario
                market_shock = self._generate_market_shock(params)
                
                # Generate liquidity impact
                liquidity_impact = self._generate_liquidity_impact(params)
                
                # Calculate portfolio impact
                portfolio_impact = self._calculate_advanced_portfolio_impact(
                    market_shock, liquidity_impact
                )
                
                simulations.append({
                    'simulation_id': i,
                    'portfolio_impact': portfolio_impact,
                    'market_shock': market_shock,
                    'liquidity_impact': liquidity_impact,
                    'params': params
                })
            
            return simulations
            
        except Exception as e:
            logger.error(f"Advanced simulations error: {str(e)}")
            return []
    
    def _generate_market_shock(self, params: Dict[str, float]) -> Dict[str, float]:
        """Generate market shock based on stress parameters."""
        try:
            # Market-wide shock
            market_component = np.random.normal(
                params['mean_shift'], 
                params['volatility_multiplier'] * 0.2
            )
            
            # Asset-specific shocks with correlation
            shocks = {}
            for symbol in self.symbols:
                # Correlation with market factor
                market_correlation = np.random.uniform(0.3, 0.8)
                
                # Idiosyncratic component
                idiosyncratic = np.random.normal(0, params['volatility_multiplier'] * 0.15)
                
                # Combine components
                total_shock = (
                    market_correlation * market_component + 
                    np.sqrt(1 - market_correlation**2) * idiosyncratic
                )
                
                shocks[symbol] = total_shock
            
            return shocks
            
        except Exception as e:
            logger.error(f"Market shock generation error: {str(e)}")
            return {symbol: np.random.normal(-0.1, 0.2) for symbol in self.symbols}
    
    def _generate_liquidity_impact(self, params: Dict[str, float]) -> Dict[str, float]:
        """Generate liquidity impact on portfolio."""
        try:
            liquidity_factor = params['liquidity_factor']
            
            # Simulate bid-ask spread widening and market impact
            liquidity_impacts = {}
            
            for position in self.positions:
                symbol = position['symbol']
                
                # Base liquidity impact (wider spreads during stress)
                base_impact = np.random.uniform(0.02, 0.08) * (1 - liquidity_factor)
                
                # Market size impact (larger positions have higher market impact)
                position_size = abs(position['quantity'] * position.get('current_price', 100))
                size_impact = min(0.05, position_size / 1000000)  # 5% max for $1M+ positions
                
                total_impact = base_impact + size_impact
                liquidity_impacts[symbol] = total_impact
            
            return liquidity_impacts
            
        except Exception as e:
            logger.error(f"Liquidity impact generation error: {str(e)}")
            return {}
    
    def _calculate_advanced_portfolio_impact(self, market_shock: Dict[str, float],
                                           liquidity_impact: Dict[str, float]) -> Dict[str, float]:
        """Calculate advanced portfolio impact including liquidity effects."""
        try:
            total_impact = 0.0
            position_impacts = {}
            
            for position in self.positions:
                symbol = position['symbol']
                quantity = position['quantity']
                price = position.get('current_price', 100.0)
                
                # Market shock impact
                market_impact = quantity * price * market_shock.get(symbol, 0)
                
                # Liquidity impact (additional cost)
                liquidity_cost = abs(quantity * price) * liquidity_impact.get(symbol, 0)
                
                # Total position impact
                total_position_impact = market_impact - liquidity_cost
                
                position_impacts[symbol] = total_position_impact
                total_impact += total_position_impact
            
            return {
                'total_impact': total_impact,
                'position_impacts': position_impacts,
                'liquidity_cost': sum(abs(pos * liq) for pos, liq in 
                                    zip([pos * pos.get('current_price', 100) for pos in self.positions],
                                        [liquidity_impact.get(pos['symbol'], 0) for pos in self.positions]))
            }
            
        except Exception as e:
            logger.error(f"Advanced portfolio impact calculation error: {str(e)}")
            return {'total_impact': 0.0, 'position_impacts': {}, 'liquidity_cost': 0.0}
    
    def _analyze_stress_scenario(self, scenario_name: str, severity: str, 
                               simulations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze stress scenario results."""
        try:
            if not simulations:
                return {'error': 'No simulations to analyze'}
            
            portfolio_impacts = [sim['portfolio_impact']['total_impact'] for sim in simulations]
            
            return {
                'scenario_name': scenario_name,
                'severity': severity,
                'num_simulations': len(simulations),
                'statistics': {
                    'mean_impact': np.mean(portfolio_impacts),
                    'std_impact': np.std(portfolio_impacts),
                    'percentile_5': np.percentile(portfolio_impacts, 5),
                    'percentile_95': np.percentile(portfolio_impacts, 95),
                    'worst_case': min(portfolio_impacts),
                    'best_case': max(portfolio_impacts)
                },
                'risk_metrics': {
                    'var_95': abs(np.percentile(portfolio_impacts, 5)),
                    'cvar_95': abs(np.mean([x for x in portfolio_impacts if x <= np.percentile(portfolio_impacts, 5)])),
                    'loss_probability': np.mean(np.array(portfolio_impacts) < 0),
                    'extreme_loss_probability': np.mean(np.array(portfolio_impacts) < -0.2 * sum(abs(pos.get('quantity', 0) * pos.get('current_price', 100)) for pos in self.positions))
                }
            }
            
        except Exception as e:
            logger.error(f"Stress scenario analysis error: {str(e)}")
            return {'error': str(e), 'scenario': scenario_name}
    
    def _aggregate_stress_results(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate stress test results across all scenarios."""
        try:
            all_portfolio_impacts = []
            scenario_weights = []
            
            for result in all_results:
                if 'statistics' in result:
                    # Extract simulation results (approximation)
                    mean_impact = result['statistics']['mean_impact']
                    std_impact = result['statistics']['std_impact']
                    num_sims = result['num_simulations']
                    
                    # Generate representative sample
                    scenario_impacts = np.random.normal(mean_impact, std_impact, num_sims)
                    all_portfolio_impacts.extend(scenario_impacts)
                    scenario_weights.extend([1.0] * num_sims)
            
            if not all_portfolio_impacts:
                return {'error': 'No valid results to aggregate'}
            
            # Calculate aggregate statistics
            portfolio_impacts = np.array(all_portfolio_impacts)
            
            return {
                'aggregate_statistics': {
                    'mean_impact': np.mean(portfolio_impacts),
                    'std_impact': np.std(portfolio_impacts),
                    'percentile_1': np.percentile(portfolio_impacts, 1),
                    'percentile_5': np.percentile(portfolio_impacts, 5),
                    'percentile_95': np.percentile(portfolio_impacts, 95),
                    'percentile_99': np.percentile(portfolio_impacts, 99),
                    'worst_case': min(portfolio_impacts),
                    'best_case': max(portfolio_impacts)
                },
                'aggregate_risk_metrics': {
                    'var_95': abs(np.percentile(portfolio_impacts, 5)),
                    'var_99': abs(np.percentile(portfolio_impacts, 1)),
                    'cvar_95': abs(np.mean([x for x in portfolio_impacts if x <= np.percentile(portfolio_impacts, 5)])),
                    'cvar_99': abs(np.mean([x for x in portfolio_impacts if x <= np.percentile(portfolio_impacts, 1)])),
                    'total_loss_probability': np.mean(portfolio_impacts < 0),
                    'extreme_loss_probability': np.mean(portfolio_impacts < -0.3 * sum(abs(pos.get('quantity', 0) * pos.get('current_price', 100)) for pos in self.positions))
                },
                'scenario_breakdown': all_results,
                'stress_test_summary': {
                    'total_scenarios': len(all_results),
                    'average_stress_level': np.mean([self._get_severity_score(r.get('severity', 'low')) for r in all_results]),
                    'portfolio_survival_probability': np.mean(np.array(all_portfolio_impacts) > -0.5 * sum(abs(pos.get('quantity', 0) * pos.get('current_price', 100)) for pos in self.positions))
                }
            }
            
        except Exception as e:
            logger.error(f"Stress results aggregation error: {str(e)}")
            return {'error': str(e)}
    
    def _get_severity_score(self, severity: str) -> float:
        """Convert severity level to numeric score."""
        severity_scores = {
            'low': 1.0,
            'medium': 2.0,
            'high': 3.0,
            'extreme': 4.0
        }
        return severity_scores.get(severity, 2.0)


class StressTestFactory:
    """Factory class for creating stress testing scenarios."""
    
    @staticmethod
    def create_stress_test(test_type: str, positions: List[Dict[str, Any]], **kwargs) -> Any:
        """
        Create stress test instance.
        
        Args:
            test_type: Type of stress test ('scenario', 'monte_carlo', 'historical')
            positions: Portfolio positions
            **kwargs: Test-specific parameters
            
        Returns:
            Stress test instance
        """
        test_types = {
            'scenario': ScenarioEngine,
            'monte_carlo': MonteCarloStressTest,
            'historical': HistoricalScenarios
        }
        
        if test_type.lower() not in test_types:
            raise ValueError(f"Unsupported stress test type: {test_type}")
        
        if test_type.lower() == 'historical':
            return HistoricalScenarios()
        else:
            return test_types[test_type.lower()](positions, **kwargs)
