"""
Risk Reporting Dashboard Generator

Provides comprehensive risk reporting dashboards including:
- Real-time risk dashboards
- VaR and CVaR monitoring
- Stress testing results visualization
- Concentration risk monitoring
- Correlation matrix analysis
- Risk decomposition charts
- Regulatory reporting dashboards
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from scipy import stats
import json

from ..core.config import AnalyticsConfig

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Risk metrics container"""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    expected_shortfall: float
    volatility: float
    maximum_drawdown: float
    skewness: float
    kurtosis: float
    tail_ratio: float


@dataclass
class ConcentrationRisk:
    """Concentration risk metrics"""
    largest_position: float
    top_5_concentration: float
    herfindahl_index: float
    sector_concentration: Dict[str, float]
    geographic_concentration: Dict[str, float]
    liquidity_concentration: float


@dataclass
class StressTestResult:
    """Stress test results"""
    scenario_name: str
    scenario_description: str
    estimated_impact: float
    probability: float
    time_horizon: str
    affected_positions: List[str]
    recovery_time: Optional[float]


@dataclass
class CorrelationMatrix:
    """Correlation analysis"""
    correlation_matrix: pd.DataFrame
    eigen_decomposition: Dict[str, Any]
    principal_components: Dict[str, Any]
    correlation_concentration: float
    regime_dependent_correlations: Dict[str, pd.DataFrame]


@dataclass
class RegulatoryMetrics:
    """Regulatory reporting metrics"""
    capital_requirement: float
    leverage_ratio: float
    liquidity_coverage_ratio: float
    net_stable_funding_ratio: float
    concentration_limit_utilization: Dict[str, float]
    compliance_status: Dict[str, bool]


class RiskDashboardGenerator:
    """
    Advanced Risk Reporting Dashboard Generator
    
    Provides comprehensive risk reporting capabilities including:
    - Real-time risk dashboards
    - VaR and CVaR monitoring
    - Stress testing results visualization
    - Concentration risk monitoring
    - Correlation matrix analysis
    - Risk decomposition charts
    - Regulatory reporting dashboards
    """
    
    def __init__(self, config: AnalyticsConfig):
        """Initialize risk dashboard generator"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Risk limits and thresholds
        self.risk_limits = {
            'var_95_limit': 0.05,  # 5% of portfolio value
            'var_99_limit': 0.08,  # 8% of portfolio value
            'volatility_limit': 0.25,  # 25% annualized
            'max_drawdown_limit': 0.15,  # 15%
            'largest_position_limit': 0.10,  # 10%
            'concentration_limit': 0.25  # HHI limit
        }
        
        # Stress test scenarios
        self.stress_scenarios = {
            'Market_Crash_2008': {
                'description': 'Similar to 2008 financial crisis',
                'market_shock': -0.40,
                'credit_spread_widening': 0.05,
                'volatility_spike': 3.0,
                'probability': 0.02
            },
            'COVID_March_2020': {
                'description': 'COVID-19 pandemic shock',
                'market_shock': -0.35,
                'sector_rotation': {'Technology': 0.15, 'Energy': -0.30},
                'volatility_spike': 4.0,
                'probability': 0.05
            },
            'Interest_Rate_Shock': {
                'description': '200bps interest rate increase',
                'rate_shock': 0.02,
                'duration_impact': -0.15,
                'credit_impact': -0.10,
                'probability': 0.10
            },
            'Recession_Scenario': {
                'description': 'Economic recession',
                'gdp_contraction': -0.03,
                'earnings_decline': -0.20,
                'unemployment_rise': 0.03,
                'probability': 0.15
            }
        }
        
        # Regulatory frameworks
        self.regulatory_frameworks = {
            'Basel_III': {
                'capital_adequacy': {'min_ratio': 0.08, 'risk_weighted': True},
                'leverage_ratio': {'min_ratio': 0.03},
                'liquidity_coverage': {'min_ratio': 1.0},
                'net_stable_funding': {'min_ratio': 1.0}
            },
            'MiFID_II': {
                'transaction_reporting': True,
                'best_execution': True,
                'product_governance': True
            },
            'Dodd_Frank': {
                'volcker_rule': True,
                'clearing_requirement': True,
                'margin_requirements': True
            }
        }
        
        self.logger.info("Risk Dashboard Generator initialized")
    
    async def generate_comprehensive_dashboards(self, portfolio_id: str = None) -> Dict[str, Any]:
        """Generate comprehensive risk dashboards"""
        try:
            self.logger.info("Generating comprehensive risk dashboards")
            
            # Real-time risk dashboard
            real_time_dashboard = await self._generate_real_time_dashboard(portfolio_id)
            
            # VaR monitoring dashboard
            var_dashboard = await self._generate_var_monitoring_dashboard(portfolio_id)
            
            # Stress testing dashboard
            stress_test_dashboard = await self._generate_stress_test_dashboard(portfolio_id)
            
            # Concentration risk dashboard
            concentration_dashboard = await self._generate_concentration_dashboard(portfolio_id)
            
            # Correlation analysis dashboard
            correlation_dashboard = await self._generate_correlation_dashboard(portfolio_id)
            
            # Risk decomposition dashboard
            decomposition_dashboard = await self._generate_decomposition_dashboard(portfolio_id)
            
            # Regulatory reporting dashboard
            regulatory_dashboard = await self._generate_regulatory_dashboard(portfolio_id)
            
            # Executive summary dashboard
            executive_summary = await self._generate_executive_summary(
                real_time_dashboard, var_dashboard, stress_test_dashboard,
                concentration_dashboard, correlation_dashboard, decomposition_dashboard
            )
            
            return {
                'dashboard_generated_at': datetime.now().isoformat(),
                'portfolios_monitored': [portfolio_id] if portfolio_id else ['Portfolio_1', 'Portfolio_2', 'Portfolio_3'],
                'executive_summary': executive_summary,
                'dashboards': {
                    'real_time_risk': real_time_dashboard,
                    'var_monitoring': var_dashboard,
                    'stress_testing': stress_test_dashboard,
                    'concentration_risk': concentration_dashboard,
                    'correlation_analysis': correlation_dashboard,
                    'risk_decomposition': decomposition_dashboard,
                    'regulatory_reporting': regulatory_dashboard
                },
                'risk_alerts': await self._generate_risk_alerts(),
                'dashboard_metadata': {
                    'total_scenarios': len(self.stress_scenarios),
                    'regulatory_frameworks': list(self.regulatory_frameworks.keys()),
                    'risk_limits': self.risk_limits,
                    'last_updated': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive dashboards: {e}")
            raise
    
    async def _generate_real_time_dashboard(self, portfolio_id: str = None) -> Dict[str, Any]:
        """Generate real-time risk dashboard"""
        try:
            # Mock real-time risk metrics
            current_time = datetime.now()
            
            # Generate realistic risk metrics
            np.random.seed(hash(portfolio_id or 'default') % 1000)
            
            base_pnl = np.random.normal(0, 0.02)  # 2% daily volatility
            volatility = 0.18 + np.random.normal(0, 0.02)
            
            # VaR calculations (simplified)
            var_95 = -1.645 * volatility / np.sqrt(252)  # 1-day VaR
            var_99 = -2.33 * volatility / np.sqrt(252)
            cvar_95 = -2.0 * volatility / np.sqrt(252)
            cvar_99 = -3.0 * volatility / np.sqrt(252)
            
            # Risk metrics
            risk_metrics = RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                cvar_99=cvar_99,
                expected_shortfall=cvar_95,
                volatility=volatility,
                maximum_drawdown=max(0, -base_pnl * 1.5),
                skewness=np.random.normal(0, 0.3),
                kurtosis=np.random.normal(3, 0.5),
                tail_ratio=0.8
            )
            
            # Risk limit violations
            violations = []
            if abs(risk_metrics.var_95) > self.risk_limits['var_95_limit']:
                violations.append('VaR_95_Limit_Breach')
            if abs(risk_metrics.var_99) > self.risk_limits['var_99_limit']:
                violations.append('VaR_99_Limit_Breach')
            if risk_metrics.volatility > self.risk_limits['volatility_limit']:
                violations.append('Volatility_Limit_Breach')
            
            # Real-time position updates
            positions_update = {
                'last_update': current_time.isoformat(),
                'total_positions': 25,
                'position_changes': 3,
                'largest_move': np.random.uniform(-0.05, 0.05)
            }
            
            # Risk budget utilization
            risk_budget_utilization = {
                'daily_var_budget': min(100, abs(risk_metrics.var_95) / self.risk_limits['var_95_limit'] * 100),
                'volatility_budget': min(100, risk_metrics.volatility / self.risk_limits['volatility_limit'] * 100),
                'drawdown_budget': min(100, risk_metrics.maximum_drawdown / self.risk_limits['max_drawdown_limit'] * 100)
            }
            
            return {
                'dashboard_type': 'real_time_risk',
                'generated_at': current_time.isoformat(),
                'portfolio_id': portfolio_id or 'default',
                'current_metrics': risk_metrics.__dict__,
                'risk_limit_violations': violations,
                'positions_update': positions_update,
                'risk_budget_utilization': risk_budget_utilization,
                'alert_level': self._determine_alert_level(violations, risk_budget_utilization),
                'chart_specifications': {
                    'pnl_chart': {'type': 'line', 'period': '1D', 'data_points': 24},
                    'var_chart': {'type': 'area', 'confidence_levels': [95, 99]},
                    'volatility_chart': {'type': 'line', 'lookback_period': 30}
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating real-time dashboard: {e}")
            raise
    
    async def _generate_var_monitoring_dashboard(self, portfolio_id: str = None) -> Dict[str, Any]:
        """Generate VaR monitoring dashboard"""
        try:
            # Historical VaR backtesting
            historical_periods = 252  # 1 year
            dates = pd.date_range(end=datetime.now(), periods=historical_periods, freq='D')
            
            # Generate historical returns and VaR
            np.random.seed(42)
            returns = np.random.normal(0.0005, 0.018, historical_periods)
            
            # Historical VaR series
            window = 60
            var_95_series = []
            var_99_series = []
            
            for i in range(window, len(returns)):
                window_returns = returns[i-window:i]
                var_95 = np.percentile(window_returns, 5)
                var_99 = np.percentile(window_returns, 1)
                var_95_series.append(var_95)
                var_99_series.append(var_99)
            
            # VaR exceptions
            exceptions_95 = sum(1 for i in range(window, len(returns)) if returns[i] < var_95_series[i-window])
            exceptions_99 = sum(1 for i in range(window, len(returns)) if returns[i] < var_99_series[i-window])
            
            exception_rate_95 = exceptions_95 / (len(returns) - window)
            exception_rate_99 = exceptions_99 / (len(returns) - window)
            
            # VaR model validation
            kupiec_test_95 = self._kupiec_test(exception_rate_95, 0.05, len(returns) - window)
            kupiec_test_99 = self._kupiec_test(exception_rate_99, 0.01, len(returns) - window)
            
            # Conditional VaR analysis
            conditional_var_95 = np.mean([r for r in returns if r < np.percentile(returns, 5)])
            conditional_var_99 = np.mean([r for r in returns if r < np.percentile(returns, 1)])
            
            # Current VaR estimates
            current_var_95 = np.percentile(returns[-60:], 5)
            current_var_99 = np.percentile(returns[-60:], 1)
            
            # Monte Carlo VaR (simplified)
            monte_carlo_var_95 = current_var_95 * 1.1  # Assume 10% model risk
            monte_carlo_var_99 = current_var_99 * 1.15  # Assume 15% model risk
            
            return {
                'dashboard_type': 'var_monitoring',
                'generated_at': datetime.now().isoformat(),
                'backtesting_period': f'{historical_periods} days',
                'var_statistics': {
                    'current_var_95': current_var_95,
                    'current_var_99': current_var_99,
                    'exception_rate_95': exception_rate_95,
                    'exception_rate_99': exception_rate_99,
                    'expected_exception_rate_95': 0.05,
                    'expected_exception_rate_99': 0.01
                },
                'model_validation': {
                    'kupiec_test_95': {'statistic': kupiec_test_95['statistic'], 'p_value': kupiec_test_95['p_value'], 'model_passes': kupiec_test_95['p_value'] > 0.05},
                    'kupiec_test_99': {'statistic': kupiec_test_99['statistic'], 'p_value': kupiec_test_99['p_value'], 'model_passes': kupiec_test_99['p_value'] > 0.05},
                    'conditional_var_95': conditional_var_95,
                    'conditional_var_99': conditional_var_99
                },
                'monte_carlo_var': {
                    'var_95': monte_carlo_var_95,
                    'var_99': monte_carlo_var_99,
                    'model_risk_adjustment_95': 0.10,
                    'model_risk_adjustment_99': 0.15
                },
                'var_series_data': {
                    'dates': dates[window:].tolist(),
                    'var_95_series': var_95_series,
                    'var_99_series': var_99_series,
                    'returns_series': returns[window:].tolist()
                },
                'alert_thresholds': {
                    'exception_rate_warning': 0.08,
                    'exception_rate_critical': 0.12,
                    'var_level_warning': 0.06,
                    'var_level_critical': 0.10
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating VaR monitoring dashboard: {e}")
            raise
    
    async def _generate_stress_test_dashboard(self, portfolio_id: str = None) -> Dict[str, Any]:
        """Generate stress testing dashboard"""
        try:
            # Mock portfolio data
            portfolio_value = 100000000  # $100M
            positions = {
                'AAPL': {'weight': 0.08, 'sector': 'Technology'},
                'MSFT': {'weight': 0.07, 'sector': 'Technology'},
                'GOOGL': {'weight': 0.05, 'sector': 'Technology'},
                'JNJ': {'weight': 0.04, 'sector': 'Healthcare'},
                'AMZN': {'weight': 0.06, 'sector': 'Consumer_Discretionary'},
                'TSLA': {'weight': 0.03, 'sector': 'Consumer_Discretionary'},
                'JPM': {'weight': 0.04, 'sector': 'Financials'},
                'XOM': {'weight': 0.02, 'sector': 'Energy'}
            }
            
            # Run stress tests
            stress_results = []
            
            for scenario_name, scenario_config in self.stress_scenarios.items():
                scenario_result = await self._run_stress_test(scenario_name, scenario_config, positions, portfolio_value)
                stress_results.append(scenario_result)
            
            # Sort by impact magnitude
            stress_results.sort(key=lambda x: abs(x['estimated_impact']), reverse=True)
            
            # Historical scenario analysis
            historical_scenarios = {
                'Black_Monday_1987': {'impact': -0.22, 'duration_days': 2},
                'Dot_Com_Bubble_2000': {'impact': -0.49, 'duration_days': 550},
                'Financial_Crisis_2008': {'impact': -0.57, 'duration_days': 540},
                'COVID_Crash_2020': {'impact': -0.34, 'duration_days': 33}
            }
            
            # Stress test summary
            total_negative_impact = sum(result['estimated_impact'] for result in stress_results if result['estimated_impact'] < 0)
            worst_case_scenario = min(stress_results, key=lambda x: x['estimated_impact'])
            best_case_scenario = max(stress_results, key=lambda x: x['estimated_impact'])
            
            # Recovery analysis
            recovery_analysis = {
                'average_recovery_time': 180,  # days
                'fastest_recovery': 45,
                'slowest_recovery': 540,
                'probability_weighted_recovery': 120
            }
            
            return {
                'dashboard_type': 'stress_testing',
                'generated_at': datetime.now().isoformat(),
                'portfolio_value': portfolio_value,
                'stress_test_scenarios': self.stress_scenarios,
                'stress_test_results': stress_results,
                'scenario_ranking': [
                    {'scenario': result['scenario_name'], 'impact': result['estimated_impact'], 'rank': i+1}
                    for i, result in enumerate(stress_results)
                ],
                'historical_scenarios': historical_scenarios,
                'summary_statistics': {
                    'total_negative_impact': total_negative_impact,
                    'worst_case_scenario': worst_case_scenario,
                    'best_case_scenario': best_case_scenario,
                    'average_impact': np.mean([result['estimated_impact'] for result in stress_results]),
                    'volatility_of_impacts': np.std([result['estimated_impact'] for result in stress_results])
                },
                'recovery_analysis': recovery_analysis,
                'scenario_probabilities': {
                    name: config['probability'] for name, config in self.stress_scenarios.items()
                },
                'stress_test_alerts': await self._generate_stress_test_alerts(stress_results)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating stress test dashboard: {e}")
            raise
    
    async def _run_stress_test(self, scenario_name: str, scenario_config: Dict[str, Any], 
                             positions: Dict[str, Dict[str, float]], portfolio_value: float) -> StressTestResult:
        """Run individual stress test scenario"""
        try:
            # Calculate sector impacts
            sector_weights = {}
            for symbol, data in positions.items():
                sector = data['sector']
                if sector not in sector_weights:
                    sector_weights[sector] = 0
                sector_weights[sector] += data['weight']
            
            # Calculate total impact
            total_impact = 0
            affected_positions = []
            
            for symbol, data in positions.items():
                weight = data['weight']
                sector = data['sector']
                symbol_impact = 0
                
                if 'market_shock' in scenario_config:
                    symbol_impact += scenario_config['market_shock']
                
                if 'sector_rotation' in scenario_config and sector in scenario_config['sector_rotation']:
                    symbol_impact += scenario_config['sector_rotation'][sector]
                
                if 'earnings_decline' in scenario_config:
                    symbol_impact += scenario_config['earnings_decline'] * 0.5  # Partial earnings impact
                
                # Add random factor for individual stock variation
                symbol_impact += np.random.normal(0, 0.05)
                
                total_impact += weight * symbol_impact
                
                if symbol_impact < -0.10:  # Significant impact
                    affected_positions.append(symbol)
            
            # Adjust for correlation effects
            correlation_adjustment = 0.1 if scenario_name == 'Market_Crash_2008' else 0.05
            total_impact *= (1 + correlation_adjustment)
            
            # Estimate recovery time
            if abs(total_impact) < 0.1:
                recovery_time = np.random.uniform(30, 90)  # 1-3 months
            elif abs(total_impact) < 0.3:
                recovery_time = np.random.uniform(90, 365)  # 3-12 months
            else:
                recovery_time = np.random.uniform(365, 730)  # 1-2 years
            
            return StressTestResult(
                scenario_name=scenario_name,
                scenario_description=scenario_config['description'],
                estimated_impact=total_impact,
                probability=scenario_config['probability'],
                time_horizon='1Y',
                affected_positions=affected_positions,
                recovery_time=recovery_time
            )
            
        except Exception as e:
            self.logger.error(f"Error running stress test {scenario_name}: {e}")
            raise
    
    async def _generate_concentration_dashboard(self, portfolio_id: str = None) -> Dict[str, Any]:
        """Generate concentration risk dashboard"""
        try:
            # Mock portfolio holdings
            holdings = {
                'AAPL': {'weight': 0.085, 'sector': 'Technology', 'liquidity_score': 0.95},
                'MSFT': {'weight': 0.075, 'sector': 'Technology', 'liquidity_score': 0.92},
                'GOOGL': {'weight': 0.065, 'sector': 'Technology', 'liquidity_score': 0.88},
                'JNJ': {'weight': 0.055, 'sector': 'Healthcare', 'liquidity_score': 0.90},
                'UNH': {'weight': 0.045, 'sector': 'Healthcare', 'liquidity_score': 0.85},
                'AMZN': {'weight': 0.040, 'sector': 'Consumer_Discretionary', 'liquidity_score': 0.80},
                'JPM': {'weight': 0.038, 'sector': 'Financials', 'liquidity_score': 0.87},
                'PG': {'weight': 0.035, 'sector': 'Consumer_Staples', 'liquidity_score': 0.85},
                'V': {'weight': 0.032, 'sector': 'Financials', 'liquidity_score': 0.83},
                'MA': {'weight': 0.030, 'sector': 'Financials', 'liquidity_score': 0.82}
            }
            
            # Calculate concentration metrics
            weights = [holding['weight'] for holding in holdings.values()]
            largest_position = max(weights)
            top_5_concentration = sum(sorted(weights, reverse=True)[:5])
            
            # Herfindahl Index
            herfindahl_index = sum(w**2 for w in weights)
            
            # Sector concentration
            sector_weights = {}
            for holding in holdings.values():
                sector = holding['sector']
                if sector not in sector_weights:
                    sector_weights[sector] = 0
                sector_weights[sector] += holding['weight']
            
            # Geographic concentration (mock)
            geographic_concentration = {
                'United_States': 0.65,
                'Europe': 0.20,
                'Asia_Pacific': 0.10,
                'Emerging_Markets': 0.05
            }
            
            # Liquidity concentration
            liquidity_weights = {symbol: holding['weight'] * holding['liquidity_score'] 
                               for symbol, holding in holdings.items()}
            total_weighted_liquidity = sum(liquidity_weights.values())
            liquidity_concentration = total_weighted_liquidity
            
            concentration_risk = ConcentrationRisk(
                largest_position=largest_position,
                top_5_concentration=top_5_concentration,
                herfindahl_index=herfindahl_index,
                sector_concentration=sector_weights,
                geographic_concentration=geographic_concentration,
                liquidity_concentration=liquidity_concentration
            )
            
            # Concentration risk alerts
            alerts = []
            if largest_position > self.risk_limits['largest_position_limit']:
                alerts.append(f'Largest position {largest_position:.1%} exceeds limit {self.risk_limits["largest_position_limit"]:.1%}')
            
            if herfindahl_index > self.risk_limits['concentration_limit']:
                alerts.append(f'Herfindahl Index {herfindahl_index:.3f} exceeds concentration limit')
            
            # Concentration limits utilization
            utilization = {
                'largest_position_utilization': min(100, largest_position / self.risk_limits['largest_position_limit'] * 100),
                'concentration_limit_utilization': min(100, herfindahl_index / self.risk_limits['concentration_limit'] * 100),
                'sector_limit_utilization': {
                    sector: min(100, weight / 0.30 * 100)  # Mock 30% sector limit
                    for sector, weight in sector_weights.items()
                }
            }
            
            # Diversification analysis
            effective_number_of_assets = 1 / herfindahl_index
            optimal_portfolio_diversification = np.sqrt(len(holdings))  # Theoretical maximum
            diversification_ratio = effective_number_of_assets / optimal_portfolio_diversification
            
            return {
                'dashboard_type': 'concentration_risk',
                'generated_at': datetime.now().isoformat(),
                'concentration_metrics': concentration_risk.__dict__,
                'holdings_breakdown': {
                    symbol: {
                        'weight': holding['weight'],
                        'sector': holding['sector'],
                        'liquidity_score': holding['liquidity_score'],
                        'contribution_to_concentration': holding['weight']**2
                    }
                    for symbol, holding in holdings.items()
                },
                'concentration_alerts': alerts,
                'limit_utilization': utilization,
                'diversification_analysis': {
                    'effective_number_of_assets': effective_number_of_assets,
                    'optimal_portfolio_diversification': optimal_portfolio_diversification,
                    'diversification_ratio': diversification_ratio,
                    'diversification_score': min(100, diversification_ratio * 100)
                },
                'rebalancing_recommendations': await self._generate_rebalancing_recommendations(concentration_risk, utilization)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating concentration dashboard: {e}")
            raise
    
    async def _generate_correlation_dashboard(self, portfolio_id: str = None) -> Dict[str, Any]:
        """Generate correlation analysis dashboard"""
        try:
            # Mock asset returns for correlation analysis
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JNJ', 'JPM', 'XOM']
            periods = 252  # 1 year
            
            np.random.seed(42)
            
            # Generate correlated returns
            base_correlation = 0.3
            correlation_matrix_data = []
            
            for i, symbol1 in enumerate(symbols):
                row = []
                for j, symbol2 in enumerate(symbols):
                    if i == j:
                        correlation = 1.0
                    else:
                        # Add some variation to base correlation
                        correlation = base_correlation + np.random.normal(0, 0.15)
                        correlation = max(-0.8, min(0.8, correlation))  # Bound correlations
                    row.append(correlation)
                correlation_matrix_data.append(row)
            
            correlation_df = pd.DataFrame(correlation_matrix_data, index=symbols, columns=symbols)
            
            # Eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eigh(correlation_df.values)
            eigenvalues = np.sort(eigenvalues)[::-1]  # Sort in descending order
            
            # Principal components analysis
            explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
            cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
            
            principal_components = {
                'first_pc_variance_explained': explained_variance_ratio[0],
                'first_two_pcs_variance_explained': cumulative_variance_ratio[1],
                'first_three_pcs_variance_explained': cumulative_variance_ratio[2],
                'number_of_significant_components': sum(1 for ratio in explained_variance_ratio if ratio > 0.1)
            }
            
            # Correlation concentration (maximum correlation off-diagonal)
            off_diagonal_correlations = []
            for i in range(len(symbols)):
                for j in range(len(symbols)):
                    if i != j:
                        off_diagonal_correlations.append(abs(correlation_df.iloc[i, j]))
            
            correlation_concentration = max(off_diagonal_correlations)
            average_correlation = np.mean(off_diagonal_correlations)
            
            # Regime-dependent correlations
            regime_correlations = {
                'Bull_Market': correlation_df * 0.8 + np.eye(len(symbols)) * 0.2,  # Lower correlations
                'Bear_Market': correlation_df * 1.2 + np.eye(len(symbols)) * -0.1,  # Higher correlations
                'Crisis_Period': np.ones((len(symbols), len(symbols))) * 0.85  # Very high correlations
            }
            
            # Dynamic correlation analysis
            correlation_stability = {
                'rolling_correlation_volatility': 0.15,  # How much correlations change over time
                'correlation_regime_shifts': 3,  # Number of significant regime changes
                'diversification_effectiveness': max(0, 1 - average_correlation)  # How much diversification helps
            }
            
            return {
                'dashboard_type': 'correlation_analysis',
                'generated_at': datetime.now().isoformat(),
                'correlation_matrix': correlation_df.to_dict(),
                'eigenvalue_analysis': {
                    'eigenvalues': eigenvalues.tolist(),
                    'explained_variance_ratio': explained_variance_ratio.tolist(),
                    'cumulative_variance_ratio': cumulative_variance_ratio.tolist()
                },
                'principal_components': principal_components,
                'correlation_statistics': {
                    'correlation_concentration': correlation_concentration,
                    'average_correlation': average_correlation,
                    'maximum_correlation': max(off_diagonal_correlations),
                    'minimum_correlation': min(off_diagonal_correlations),
                    'correlation_volatility': np.std(off_diagonal_correlations)
                },
                'regime_dependent_correlations': {
                    regime: corr_matrix.to_dict()
                    for regime, corr_matrix in regime_correlations.items()
                },
                'dynamic_correlation_analysis': correlation_stability,
                'risk_implications': {
                    'diversification_benefit': 1 - np.mean(off_diagonal_correlations),
                    'systematic_risk_level': explained_variance_ratio[0],
                    'concentration_risk': 'High' if correlation_concentration > 0.8 else 'Medium' if correlation_conrelation > 0.6 else 'Low',
                    'correlation_regime': 'High Correlation' if average_correlation > 0.6 else 'Normal Correlation'
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating correlation dashboard: {e}")
            raise
    
    async def _generate_decomposition_dashboard(self, portfolio_id: str = None) -> Dict[str, Any]:
        """Generate risk decomposition dashboard"""
        try:
            # Mock risk decomposition components
            total_risk = 0.18  # 18% annualized volatility
            
            # Systematic vs Idiosyncratic risk
            systematic_risk = 0.12  # 67% of total risk
            idiosyncratic_risk = np.sqrt(total_risk**2 - systematic_risk**2)  # Remaining risk
            
            # Factor risk decomposition
            factor_risks = {
                'Market_Risk': 0.08,      # 44% of total risk
                'Sector_Risk': 0.03,      # 17% of total risk
                'Style_Risk': 0.025,      # 14% of total risk
                'Country_Risk': 0.015,    # 8% of total risk
                'Currency_Risk': 0.01     # 6% of total risk
            }
            
            # Risk by asset class
            asset_class_risk = {
                'Equities': 0.15,
                'Fixed_Income': 0.05,
                'Commodities': 0.08,
                'Alternatives': 0.12
            }
            
            # Geographic risk decomposition
            geographic_risk = {
                'United_States': 0.10,
                'Europe': 0.04,
                'Asia_Pacific': 0.03,
                'Emerging_Markets': 0.02
            }
            
            # Time-based risk decomposition
            time_based_risk = {
                'Short_term_1M': 0.25,    # 1-month VaR
                'Medium_term_3M': 0.20,   # 3-month VaR
                'Long_term_1Y': 0.18      # 1-year volatility
            }
            
            # Risk contribution by position
            position_risk_contributions = {
                'AAPL': 0.025,
                'MSFT': 0.020,
                'GOOGL': 0.018,
                'JNJ': 0.015,
                'AMZN': 0.017
            }
            
            # Calculate risk attribution percentages
            factor_attribution = {
                factor: risk / total_risk * 100
                for factor, risk in factor_risks.items()
            }
            
            # Risk decomposition summary
            risk_summary = {
                'total_risk': total_risk,
                'systematic_risk_percentage': systematic_risk / total_risk * 100,
                'idiosyncratic_risk_percentage': idiosyncratic_risk / total_risk * 100,
                'factor_diversification_ratio': 1 - max(factor_risks.values()) / total_risk,
                'geographic_diversification_ratio': 1 - max(geographic_risk.values()) / total_risk
            }
            
            # Risk forecasting
            risk_forecast = {
                'expected_risk_next_month': total_risk * 1.05,
                'risk_trend': 'Increasing',
                'risk_drivers': ['Market volatility', 'Geopolitical tensions'],
                'risk_mitigation_effectiveness': 0.75
            }
            
            return {
                'dashboard_type': 'risk_decomposition',
                'generated_at': datetime.now().isoformat(),
                'total_risk_metrics': {
                    'total_volatility': total_risk,
                    'systematic_risk': systematic_risk,
                    'idiosyncratic_risk': idiosyncratic_risk
                },
                'factor_risk_decomposition': {
                    'factor_risks': factor_risks,
                    'factor_attribution': factor_attribution
                },
                'asset_class_risk': asset_class_risk,
                'geographic_risk': geographic_risk,
                'time_based_risk': time_based_risk,
                'position_risk_contributions': position_risk_contributions,
                'risk_summary': risk_summary,
                'risk_forecast': risk_forecast,
                'risk_management_recommendations': [
                    'Consider reducing market beta exposure',
                    'Increase geographic diversification',
                    'Monitor sector concentration risk',
                    'Review position sizing methodology'
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error generating decomposition dashboard: {e}")
            raise
    
    async def _generate_regulatory_dashboard(self, portfolio_id: str = None) -> Dict[str, Any]:
        """Generate regulatory reporting dashboard"""
        try:
            # Mock regulatory metrics
            portfolio_value = 1000000000  # $1B AUM
            risk_weighted_assets = 750000000  # $750M RWA
            tier_1_capital = 50000000  # $50M Tier 1 capital
            
            # Basel III metrics
            capital_adequacy_ratio = tier_1_capital / risk_weighted_assets
            leverage_ratio = tier_1_capital / portfolio_value
            liquidity_coverage_ratio = 1.25  # Mock LCR > 100%
            net_stable_funding_ratio = 1.15  # Mock NSFR > 100%
            
            # Concentration limits utilization
            concentration_utilization = {
                'single_counterparty': 0.025,  # 2.5% of portfolio
                'sector_concentration': 0.35,  # 35% Technology sector
                'geographic_concentration': 0.65  # 65% US exposure
            }
            
            # MiFID II compliance
            mifid_compliance = {
                'best_execution': True,
                'transaction_reporting': True,
                'product_governance': True,
                'investor_protection': True
            }
            
            # Dodd-Frank compliance
            dodd_frank_compliance = {
                'volcker_rule': True,
                'clearing_requirement': True,
                'margin_requirements': True,
                'swap_data_repository': True
            }
            
            regulatory_metrics = RegulatoryMetrics(
                capital_requirement=capital_adequacy_ratio,
                leverage_ratio=leverage_ratio,
                liquidity_coverage_ratio=liquidity_coverage_ratio,
                net_stable_funding_ratio=net_stable_funding_ratio,
                concentration_limit_utilization=concentration_utilization,
                compliance_status={
                    **mifid_compliance, **dodd_frank_compliance,
                    'capital_adequacy': capital_adequacy_ratio >= 0.08,
                    'leverage_compliance': leverage_ratio >= 0.03,
                    'liquidity_compliance': liquidity_coverage_ratio >= 1.0
                }
            )
            
            # Regulatory capital requirements
            capital_requirements = {
                'minimum_capital_ratio': 0.08,
                'capital_conservation_buffer': 0.025,
                'countercyclical_buffer': 0.0,  # Currently 0%
                'total_required_ratio': 0.105
            }
            
            # Reporting obligations
            reporting_obligations = {
                'basel_iii_pillar_1': {'frequency': 'Quarterly', 'next_due': '2024-03-31'},
                'basel_iii_pillar_2': {'frequency': 'Annual', 'next_due': '2024-12-31'},
                'basel_iii_pillar_3': {'frequency': 'Quarterly', 'next_due': '2024-03-31'},
                'mifid_ii_transaction_reporting': {'frequency': 'Daily', 'next_due': 'Daily'},
                'stress_testing_report': {'frequency': 'Annual', 'next_due': '2024-06-30'}
            }
            
            # Regulatory violations and remedial actions
            violations = []
            if capital_adequacy_ratio < 0.08:
                violations.append('Capital Adequacy Ratio Below Minimum')
            
            if concentration_utilization['sector_concentration'] > 0.40:
                violations.append('Sector Concentration Limit Breach')
            
            return {
                'dashboard_type': 'regulatory_reporting',
                'generated_at': datetime.now().isoformat(),
                'regulatory_framework': 'Basel III, MiFID II, Dodd-Frank',
                'basel_iii_metrics': {
                    'capital_adequacy_ratio': capital_adequacy_ratio,
                    'leverage_ratio': leverage_ratio,
                    'liquidity_coverage_ratio': liquidity_coverage_ratio,
                    'net_stable_funding_ratio': net_stable_funding_ratio
                },
                'capital_requirements': capital_requirements,
                'concentration_limits': concentration_utilization,
                'compliance_status': regulatory_metrics.compliance_status,
                'reporting_obligations': reporting_obligations,
                'regulatory_violations': violations,
                'remedial_actions': [
                    'Increase Tier 1 capital by $10M to meet 8% requirement',
                    'Reduce Technology sector concentration to below 35%',
                    'Enhance liquidity management procedures'
                ] if violations else [],
                'regulatory_scores': {
                    'overall_compliance_score': 85 if not violations else 65,
                    'capital_adequacy_score': min(100, capital_adequacy_ratio / 0.08 * 100),
                    'liquidity_score': min(100, liquidity_coverage_ratio * 80),
                    'transparency_score': 95
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating regulatory dashboard: {e}")
            raise
    
    async def _generate_executive_summary(self, *dashboards) -> Dict[str, Any]:
        """Generate executive summary dashboard"""
        try:
            # Extract key metrics from all dashboards
            total_alerts = 0
            critical_issues = []
            performance_summary = {}
            
            for dashboard in dashboards:
                if 'risk_limit_violations' in dashboard:
                    total_alerts += len(dashboard['risk_limit_violations'])
                
                if 'alert_level' in dashboard:
                    if dashboard['alert_level'] in ['Critical', 'High']:
                        critical_issues.append(dashboard.get('portfolio_id', 'Unknown'))
            
            # Overall risk assessment
            risk_assessment = {
                'overall_risk_level': 'Medium',
                'var_utilization': 65,  # % of VaR limit
                'stress_test_impact': -0.12,  # Worst case scenario
                'concentration_risk': 'Moderate',
                'regulatory_compliance': 'Good'
            }
            
            # Key performance indicators
            kpis = {
                'sharpe_ratio': 1.45,
                'maximum_drawdown': -0.08,
                'win_rate': 0.65,
                'profit_factor': 1.8,
                'var_ratio': 0.05,
                'information_ratio': 0.78
            }
            
            # Action items
            action_items = [
                'Monitor Technology sector concentration',
                'Review VaR model parameters',
                'Update stress test scenarios',
                'Enhance liquidity monitoring'
            ]
            
            return {
                'executive_summary': {
                    'total_risk_alerts': total_alerts,
                    'critical_issues_count': len(critical_issues),
                    'risk_assessment': risk_assessment,
                    'key_performance_indicators': kpis,
                    'action_items': action_items,
                    'last_updated': datetime.now().isoformat(),
                    'dashboard_generation_time': 2.5,  # seconds
                    'data_quality_score': 95
                },
                'trending_metrics': {
                    'risk_trend': 'Decreasing',
                    'performance_trend': 'Stable',
                    'volatility_trend': 'Stable',
                    'correlation_trend': 'Increasing'
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating executive summary: {e}")
            raise
    
    async def _generate_risk_alerts(self) -> List[Dict[str, Any]]:
        """Generate risk alerts"""
        alerts = []
        
        # Mock risk alerts
        current_time = datetime.now()
        
        alerts.append({
            'alert_id': 'VAR_001',
            'timestamp': current_time.isoformat(),
            'severity': 'Medium',
            'category': 'VaR Monitoring',
            'message': 'VaR utilization above 80% threshold',
            'action_required': 'Monitor position sizes and market conditions',
            'acknowledged': False
        })
        
        alerts.append({
            'alert_id': 'CONC_002',
            'timestamp': current_time.isoformat(),
            'severity': 'Low',
            'category': 'Concentration Risk',
            'message': 'Technology sector concentration at 35%',
            'action_required': 'Review sector allocation',
            'acknowledged': False
        })
        
        return alerts
    
    async def _generate_stress_test_alerts(self, stress_results: List[StressTestResult]) -> List[Dict[str, Any]]:
        """Generate stress test specific alerts"""
        alerts = []
        
        for result in stress_results:
            if abs(result.estimated_impact) > 0.20:  # > 20% impact
                alerts.append({
                    'scenario': result.scenario_name,
                    'impact': result.estimated_impact,
                    'severity': 'High' if abs(result.estimated_impact) > 0.30 else 'Medium',
                    'probability_adjusted_impact': result.estimated_impact * result.probability,
                    'recovery_time': result.recovery_time
                })
        
        return alerts
    
    async def _generate_rebalancing_recommendations(self, concentration_risk: ConcentrationRisk, 
                                                   utilization: Dict[str, Any]) -> List[str]:
        """Generate rebalancing recommendations"""
        recommendations = []
        
        if concentration_risk.largest_position > 0.10:
            recommendations.append(f"Reduce largest position from {concentration_risk.largest_position:.1%} to below 10%")
        
        if concentration_risk.herfindahl_index > 0.25:
            recommendations.append(f"Reduce portfolio concentration - Herfindahl Index is {concentration_risk.herfindahl_index:.3f}")
        
        for sector, util in utilization.get('sector_limit_utilization', {}).items():
            if util > 80:
                recommendations.append(f"Consider reducing {sector} exposure - {util:.1f}% of limit utilized")
        
        return recommendations
    
    def _determine_alert_level(self, violations: List[str], utilization: Dict[str, float]) -> str:
        """Determine overall alert level"""
        if not violations and all(util < 70 for util in utilization.values()):
            return 'Low'
        elif len(violations) <= 1 and all(util < 85 for util in utilization.values()):
            return 'Medium'
        elif len(violations) <= 3:
            return 'High'
        else:
            return 'Critical'
    
    def _kupiec_test(self, observed_exceptions: float, expected_exceptions: float, n_observations: int) -> Dict[str, float]:
        """Perform Kupiec test for VaR model validation"""
        # Simplified Kupiec test
        # LR_statistic = -2 * ln(((1-p)^(N-x) * p^x) / ((1-x/N)^(N-x) * (x/N)^x))
        
        if observed_exceptions == 0 or observed_exceptions == 1:
            return {'statistic': 0, 'p_value': 1.0, 'model_passes': True}
        
        x = observed_exceptions * n_observations
        N = n_observations
        p = expected_exceptions
        
        lr_stat = -2 * np.log(((1 - p)**(N - x) * p**x) / 
                             ((1 - x/N)**(N - x) * (x/N)**x))
        
        # Chi-square test with 1 degree of freedom
        p_value = 1 - stats.chi2.cdf(lr_stat, 1)
        
        return {
            'statistic': lr_stat,
            'p_value': p_value,
            'model_passes': p_value > 0.05
        }
    
    async def get_real_time_dashboard_data(self) -> Dict[str, Any]:
        """Get real-time dashboard data for Matrix integration"""
        try:
            return await self._generate_real_time_dashboard()
        except Exception as e:
            self.logger.error(f"Error getting real-time dashboard data: {e}")
            raise
    
    async def update_real_time_dashboard(self):
        """Update real-time dashboard metrics"""
        try:
            self.logger.debug("Updating real-time risk dashboard metrics")
            # This would typically update Redis cache or push to WebSocket
            pass
        except Exception as e:
            self.logger.error(f"Error updating real-time dashboard: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for risk dashboard generator"""
        try:
            return {
                'status': 'healthy',
                'last_update': datetime.now().isoformat(),
                'stress_scenarios': len(self.stress_scenarios),
                'regulatory_frameworks': len(self.regulatory_frameworks),
                'risk_limits': len(self.risk_limits)
            }
        except Exception as e:
            self.logger.error(f"Error in risk dashboard health check: {e}")
            return {'status': 'error', 'error': str(e)}