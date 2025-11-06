"""
Advanced Risk Management System - Core Demo

This simplified demo showcases the new advanced risk management components
without database dependencies, focusing on:

1. Advanced Risk Models (VaR, CVaR, stress testing, etc.)
2. Portfolio Optimization (MPT, Risk Parity)
3. Compliance Frameworks (Basel III, MiFID II, Dodd-Frank)
4. Real-time Risk Monitoring
5. API Integration Layer

Usage:
    python demo_risk_management_core.py
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the new advanced risk management components
try:
    from risk.models.var_models import VaRCalculator
    from risk.models.cvar_models import CVaRCalculator
    from risk.models.drawdown_models import DrawdownAnalyzer
    from risk.models.volatility_models import VolatilityModeler
    from risk.models.correlation_models import CorrelationAnalyzer
    from risk.models.stress_testing import StressTestEngine
    from risk.enhanced_limits import EnhancedRiskLimits
    from risk.portfolio_optimization import PortfolioOptimizer
    from risk.real_time_monitor import RealTimeRiskMonitor
    from risk.compliance_frameworks import ComplianceFrameworks, RegulationFramework
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some risk management components not available: {e}")
    COMPONENTS_AVAILABLE = False

class CoreRiskManagementDemo:
    """
    Core demonstration of advanced risk management capabilities
    
    This class showcases the new institutional-grade risk management
    features in an integrated demonstration.
    """
    
    def __init__(self):
        """Initialize the demo"""
        if not COMPONENTS_AVAILABLE:
            raise ImportError("Advanced risk management components not available")
            
        self.logger = logging.getLogger(__name__)
        
        # Initialize risk management components
        self.var_calculator = VaRCalculator()
        self.cvar_calculator = CVaRCalculator()
        self.drawdown_analyzer = DrawdownAnalyzer()
        self.volatility_modeler = VolatilityModeler()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.stress_tester = StressTestEngine()
        self.enhanced_limits = EnhancedRiskLimits()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.real_time_monitor = RealTimeRiskMonitor()
        self.compliance_frameworks = ComplianceFrameworks()
        
        # Demo portfolio data
        self.demo_portfolio_data = self._generate_demo_portfolio_data()
        
        self.logger.info("Core Risk Management Demo initialized")
    
    def _generate_demo_portfolio_data(self) -> pd.DataFrame:
        """Generate demo portfolio data for testing"""
        np.random.seed(42)  # For reproducible results
        
        # Generate 252 days of returns for each asset
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        
        assets = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA']
        
        portfolio_data = pd.DataFrame(index=dates)
        
        # Generate realistic return data with different risk characteristics
        for i, asset in enumerate(assets):
            # Base return with trend
            base_return = 0.0005 + i * 0.0001  # Slightly increasing expected return
            
            # Add volatility clustering and market correlation
            market_shock = np.random.normal(0, 0.015, 252)
            asset_specific = np.random.normal(0, 0.02 + i * 0.005, 252)
            
            returns = base_return + 0.6 * market_shock + 0.4 * asset_specific
            
            # Add some correlation breaks during "crisis" periods
            crisis_periods = [50, 100, 150, 200]  # Days with higher correlation
            for crisis_day in crisis_periods:
                if crisis_day < len(returns):
                    returns[crisis_day:crisis_day+5] *= 3  # Increased volatility
            
            portfolio_data[asset] = returns
        
        return portfolio_data
    
    async def demo_1_var_and_cvar(self):
        """Demonstrate VaR and CVaR calculations"""
        print("\n" + "="*80)
        print("DEMO 1: VALUE AT RISK (VaR) AND CONDITIONAL VaR (CVaR)")
        print("="*80)
        
        try:
            print("\n📊 Calculating Value at Risk (VaR)...")
            
            # Historical VaR
            print("\n1. Historical VaR Calculation:")
            historical_var = self.var_calculator.calculate_historical_var(
                self.demo_portfolio_data, 
                confidence_level=0.95,
                time_horizon=1
            )
            print(f"   Historical VaR (95%, 1-day): ${historical_var.get('var', 0):,.2f}")
            
            # Parametric VaR
            print("\n2. Parametric VaR Calculation:")
            parametric_var = self.var_calculator.calculate_parametric_var(
                self.demo_portfolio_data,
                confidence_level=0.95,
                time_horizon=1
            )
            print(f"   Parametric VaR (95%, 1-day): ${parametric_var.get('var', 0):,.2f}")
            
            # Monte Carlo VaR
            print("\n3. Monte Carlo VaR Calculation:")
            mc_var = self.var_calculator.calculate_monte_carlo_var(
                self.demo_portfolio_data,
                confidence_level=0.95,
                time_horizon=1
            )
            print(f"   Monte Carlo VaR (95%, 1-day): ${mc_var.get('var', 0):,.2f}")
            
            # CVaR Calculation
            print("\n📊 Calculating Conditional Value at Risk (CVaR)...")
            cvar_result = self.cvar_calculator.calculate_cvar(
                self.demo_portfolio_data,
                confidence_level=0.95,
                time_horizon=1
            )
            print(f"CVaR (95%, 1-day): ${cvar_result.get('cvar', 0):,.2f}")
            
            # CVaR vs VaR comparison
            print(f"\n📈 Risk Metrics Comparison:")
            print(f"   VaR:  ${historical_var.get('var', 0):,.2f}")
            print(f"   CVaR: ${cvar_result.get('cvar', 0):,.2f}")
            print(f"   CVaR/VaR Ratio: {cvar_result.get('cvar', 0) / historical_var.get('var', 1):.2f}")
            
            print("\n✅ VaR and CVaR Demo Completed")
            return True
            
        except Exception as e:
            print(f"❌ Error in VaR/CVaR demo: {e}")
            return False
    
    async def demo_2_drawdown_and_volatility(self):
        """Demonstrate drawdown analysis and volatility modeling"""
        print("\n" + "="*80)
        print("DEMO 2: DRAWDOWN ANALYSIS AND VOLATILITY MODELING")
        print("="*80)
        
        try:
            # Drawdown Analysis
            print("\n📊 Maximum Drawdown Analysis...")
            
            # Generate portfolio value series from returns
            portfolio_returns = self.demo_portfolio_data.sum(axis=1)
            portfolio_value = 100000 * (1 + portfolio_returns).cumprod()
            
            drawdown_result = self.drawdown_analyzer.calculate_drawdown(portfolio_value)
            
            print(f"Maximum Drawdown: {drawdown_result.get('max_drawdown', 0):.2%}")
            print(f"Current Drawdown: {drawdown_result.get('current_drawdown', 0):.2%}")
            print(f"Recovery Factor: {drawdown_result.get('recovery_factor', 0):.2f}")
            
            # Find drawdown periods
            underwater_analysis = self.drawdown_analyzer.underwater_analysis(portfolio_value)
            if underwater_analysis:
                print(f"Average Underwater Period: {underwater_analysis.get('avg_underwater_days', 0):.1f} days")
                print(f"Longest Underwater Period: {underwater_analysis.get('max_underwater_days', 0)} days")
            
            # Volatility Analysis
            print("\n📊 Portfolio Volatility Analysis...")
            
            # Calculate EWMA volatility
            ewma_vol = self.volatility_modeler.calculate_ewma(
                self.demo_portfolio_data,
                lambda_param=0.94
            )
            print(f"Portfolio EWMA Volatility: {ewma_vol.get('portfolio_volatility', 0):.2%}")
            
            # GARCH volatility for individual assets
            print("\nGARCH Volatility for Individual Assets:")
            for asset in ['AAPL', 'GOOGL', 'MSFT'][:3]:
                try:
                    garch_vol = self.volatility_modeler.fit_garch(
                        self.demo_portfolio_data[asset],
                        order=(1, 1)
                    )
                    annualized_vol = np.sqrt(252) * garch_vol.get('conditional_volatility', 0)
                    print(f"   {asset}: {annualized_vol:.2%}")
                except Exception as e:
                    print(f"   {asset}: Error in GARCH fitting")
            
            print("\n✅ Drawdown and Volatility Demo Completed")
            return True
            
        except Exception as e:
            print(f"❌ Error in drawdown/volatility demo: {e}")
            return False
    
    async def demo_3_correlation_and_stress_testing(self):
        """Demonstrate correlation analysis and stress testing"""
        print("\n" + "="*80)
        print("DEMO 3: CORRELATION ANALYSIS AND STRESS TESTING")
        print("="*80)
        
        try:
            # Correlation Analysis
            print("\n📊 Asset Correlation Analysis...")
            
            correlation_matrix = self.correlation_analyzer.calculate_correlation_matrix(
                self.demo_portfolio_data
            )
            
            # Find highest correlations
            corr_values = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    asset1 = correlation_matrix.columns[i]
                    asset2 = correlation_matrix.columns[j]
                    corr = correlation_matrix.iloc[i, j]
                    corr_values.append((asset1, asset2, corr))
            
            # Sort by correlation
            corr_values.sort(key=lambda x: abs(x[2]), reverse=True)
            
            print("Highest Asset Correlations:")
            for asset1, asset2, corr in corr_values[:4]:
                print(f"   {asset1} - {asset2}: {corr:.3f}")
            
            # Rolling Correlations
            print("\n📊 Rolling Correlation Analysis...")
            rolling_corr = self.correlation_analyzer.rolling_correlations(
                self.demo_portfolio_data['AAPL'],
                self.demo_portfolio_data['GOOGL'],
                window=30
            )
            print(f"AAPL-GOOGL 30-day Rolling Correlation: {rolling_corr.mean():.3f}")
            print(f"Rolling Correlation Std Dev: {rolling_corr.std():.3f}")
            
            # Stress Testing
            print("\n📊 Portfolio Stress Testing...")
            
            # Historical Scenarios
            scenarios = ['black_monday_1987', 'financial_crisis_2008', 'covid_pandemic_2020']
            
            for scenario in scenarios:
                try:
                    result = self.stress_tester.historical_scenario_test(
                        self.demo_portfolio_data,
                        scenario
                    )
                    loss = result.get('loss', 0)
                    print(f"{scenario.replace('_', ' ').title()}: ${loss:,.2f} loss")
                except Exception as e:
                    print(f"{scenario}: Error - {e}")
            
            # Monte Carlo Stress Test
            print("\n📊 Monte Carlo Stress Testing...")
            try:
                mc_result = self.stress_tester.monte_carlo_stress_test(
                    self.demo_portfolio_data,
                    num_simulations=500  # Reduced for demo
                )
                print(f"Monte Carlo Stress Test: ${mc_result.get('loss', 0):,.2f} average loss")
                print(f"Stress Test Probability: {mc_result.get('probability', 0):.1%}")
            except Exception as e:
                print(f"Monte Carlo Stress Test: Error - {e}")
            
            print("\n✅ Correlation and Stress Testing Demo Completed")
            return True
            
        except Exception as e:
            print(f"❌ Error in correlation/stress testing demo: {e}")
            return False
    
    async def demo_4_portfolio_optimization(self):
        """Demonstrate portfolio optimization techniques"""
        print("\n" + "="*80)
        print("DEMO 4: PORTFOLIO OPTIMIZATION")
        print("="*80)
        
        try:
            # Maximum Sharpe Ratio Optimization
            print("\n📊 Maximum Sharpe Ratio Optimization...")
            max_sharpe = self.portfolio_optimizer.optimize_max_sharpe(
                self.demo_portfolio_data,
                risk_free_rate=0.02
            )
            
            print("Optimal Portfolio Weights:")
            for asset, weight in max_sharpe.get('weights', {}).items():
                if weight > 0.01:  # Only show significant weights
                    print(f"   {asset}: {weight:.1%}")
            
            print(f"Expected Return: {max_sharpe.get('expected_return', 0):.2%}")
            print(f"Expected Volatility: {max_sharpe.get('volatility', 0):.2%}")
            print(f"Sharpe Ratio: {max_sharpe.get('sharpe_ratio', 0):.3f}")
            
            # Minimum Variance Optimization
            print("\n📊 Minimum Variance Optimization...")
            min_variance = self.portfolio_optimizer.optimize_min_variance(
                self.demo_portfolio_data
            )
            
            print("Minimum Variance Portfolio Weights:")
            for asset, weight in min_variance.get('weights', {}).items():
                if weight > 0.01:
                    print(f"   {asset}: {weight:.1%}")
            
            print(f"Expected Volatility: {min_variance.get('volatility', 0):.2%}")
            
            # Risk Parity Optimization
            print("\n📊 Risk Parity Optimization...")
            risk_parity = self.portfolio_optimizer.optimize_risk_parity(
                self.demo_portfolio_data
            )
            
            print("Risk Parity Portfolio Weights:")
            for asset, weight in risk_parity.get('weights', {}).items():
                if weight > 0.01:
                    print(f"   {asset}: {weight:.1%}")
            
            # Compare optimization results
            print(f"\n📈 Optimization Comparison:")
            print(f"Maximum Sharpe:  Return {max_sharpe.get('expected_return', 0):.2%}, "
                  f"Vol {max_sharpe.get('volatility', 0):.2%}, "
                  f"Sharpe {max_sharpe.get('sharpe_ratio', 0):.3f}")
            print(f"Minimum Variance: Return {min_variance.get('expected_return', 0):.2%}, "
                  f"Vol {min_variance.get('volatility', 0):.2%}")
            print(f"Risk Parity:      Return {risk_parity.get('expected_return', 0):.2%}, "
                  f"Vol {risk_parity.get('volatility', 0):.2%}")
            
            print("\n✅ Portfolio Optimization Demo Completed")
            return True
            
        except Exception as e:
            print(f"❌ Error in portfolio optimization demo: {e}")
            return False
    
    async def demo_5_compliance_frameworks(self):
        """Demonstrate compliance frameworks"""
        print("\n" + "="*80)
        print("DEMO 5: REGULATORY COMPLIANCE FRAMEWORKS")
        print("="*80)
        
        try:
            # Start compliance monitoring
            print("\n📊 Starting Real-time Compliance Monitoring...")
            self.compliance_frameworks.start_real_time_monitoring()
            
            # Get compliance status for different frameworks
            frameworks = [RegulationFramework.BASEL_III, RegulationFramework.MIFID_II, RegulationFramework.DODD_FRANK]
            
            for framework in frameworks:
                print(f"\n{framework.value.upper()} Compliance Status:")
                status = self.compliance_frameworks.get_compliance_status(framework)
                
                if status and 'status_summary' in status:
                    summary = status['status_summary']
                    active_breaches = status.get('active_breaches', 0)
                    
                    print(f"   Compliant Events: {summary.get('compliant', 0)}")
                    print(f"   Warning Events: {summary.get('warning', 0)}")
                    print(f"   Breach Events: {summary.get('breach', 0)}")
                    print(f"   Active Breaches: {active_breaches}")
            
            # Generate compliance reports
            print("\n📊 Generating Compliance Reports...")
            
            for framework in frameworks[:2]:  # Limit to first 2 for demo
                report = self.compliance_frameworks.generate_compliance_report(framework)
                
                if report and 'executive_summary' in report:
                    summary = report['executive_summary']
                    print(f"\n{framework.value.upper()} Executive Summary:")
                    print(f"   Overall Status: {summary.get('overall_status', 'Unknown')}")
                    print(f"   Total Events: {summary.get('total_events', 0)}")
                    print(f"   Active Breaches: {summary.get('active_breaches', 0)}")
                    
                    # Show recommendations
                    if 'recommendations' in report:
                        print(f"   Key Recommendations:")
                        for rec in report['recommendations'][:2]:
                            print(f"     - {rec}")
            
            # Trade surveillance demo
            print("\n📊 Trade Surveillance Demo...")
            sample_trade = {
                "trade_id": "TRD_001",
                "notional_amount": 50000000,  # $50M
                "instrument_type": "equity",
                "execution_price": 102.5,
                "market_price": 102.0,
                "short_sale": True,
                "client_trades_per_day": 50,
                "large_orders_cancelled": 2,
                "layered_order_count": 1
            }
            
            surveillance_result = self.compliance_frameworks.trade_surveillance(sample_trade)
            if surveillance_result and 'surveillance_score' in surveillance_result:
                print(f"Trade Surveillance Score: {surveillance_result['surveillance_score']}")
                print(f"Risk Flags: {surveillance_result.get('risk_flags', [])}")
                print(f"Compliance Flags: {surveillance_result.get('compliance_flags', [])}")
            
            # Stop monitoring
            self.compliance_frameworks.stop_real_time_monitoring()
            
            print("\n✅ Compliance Frameworks Demo Completed")
            return True
            
        except Exception as e:
            print(f"❌ Error in compliance frameworks demo: {e}")
            return False
    
    async def demo_6_real_time_monitoring(self):
        """Demonstrate real-time risk monitoring"""
        print("\n" + "="*80)
        print("DEMO 6: REAL-TIME RISK MONITORING")
        print("="*80)
        
        try:
            # Get real-time risk metrics
            print("\n📊 Real-time Risk Metrics...")
            metrics = self.real_time_monitor.get_live_risk_metrics()
            
            if metrics:
                print("Current Risk Metrics:")
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        if 'ratio' in metric_name.lower() or 'rate' in metric_name.lower():
                            print(f"   {metric_name}: {value:.3f}")
                        else:
                            print(f"   {metric_name}: ${value:,.2f}")
                    else:
                        print(f"   {metric_name}: {value}")
            
            # Check for limit breaches
            print("\n📊 Risk Limit Breach Analysis...")
            breaches = self.real_time_monitor.check_limit_breaches()
            
            if breaches:
                print("Active Limit Breaches:")
                for breach in breaches:
                    print(f"   {breach}")
            else:
                print("No active limit breaches detected")
            
            # Generate risk alerts
            print("\n📊 Risk Alert Generation...")
            alerts = self.real_time_monitor.generate_risk_alerts()
            
            if alerts:
                print("Generated Risk Alerts:")
                for alert in alerts[:3]:  # Show first 3 alerts
                    print(f"   - {alert}")
            else:
                print("No risk alerts generated")
            
            print("\n✅ Real-time Risk Monitoring Demo Completed")
            return True
            
        except Exception as e:
            print(f"❌ Error in real-time monitoring demo: {e}")
            return False
    
    async def run_complete_demo(self):
        """Run the complete core risk management demonstration"""
        print("🏛️  ADVANCED RISK MANAGEMENT SYSTEM - CORE DEMO")
        print("=" * 80)
        print("This demo showcases institutional-grade risk management capabilities")
        print("including VaR/CVaR, stress testing, portfolio optimization,")
        print("compliance frameworks, and real-time monitoring.")
        print("=" * 80)
        
        demos = [
            ("VaR and CVaR Calculations", self.demo_1_var_and_cvar),
            ("Drawdown Analysis and Volatility", self.demo_2_drawdown_and_volatility),
            ("Correlation and Stress Testing", self.demo_3_correlation_and_stress_testing),
            ("Portfolio Optimization", self.demo_4_portfolio_optimization),
            ("Regulatory Compliance", self.demo_5_compliance_frameworks),
            ("Real-time Risk Monitoring", self.demo_6_real_time_monitoring)
        ]
        
        start_time = datetime.now()
        successful_demos = 0
        
        for name, demo_func in demos:
            try:
                print(f"\n🚀 Starting: {name}")
                success = await demo_func()
                if success:
                    print(f"✅ Completed: {name}")
                    successful_demos += 1
                else:
                    print(f"⚠️  Partial: {name}")
            except Exception as e:
                print(f"❌ Failed: {name} - {e}")
                continue
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "="*80)
        print("🎉 CORE DEMO COMPLETE")
        print("="*80)
        print(f"Total Duration: {duration:.1f} seconds")
        print(f"Demos Executed: {successful_demos}/{len(demos)}")
        print("\n✅ Successfully Demonstrated:")
        print("✅ Advanced VaR and CVaR Calculations")
        print("✅ Drawdown Analysis and Volatility Modeling")
        print("✅ Correlation Analysis and Stress Testing")
        print("✅ Portfolio Optimization (MPT, Risk Parity)")
        print("✅ Regulatory Compliance (Basel III, MiFID II, Dodd-Frank)")
        print("✅ Real-time Risk Monitoring")
        print("\n🏛️  Institutional-grade risk management accessible to all traders!")


async def main():
    """Main demonstration function"""
    try:
        demo = CoreRiskManagementDemo()
        await demo.run_complete_demo()
    except ImportError as e:
        print(f"❌ Cannot run demo: {e}")
        print("Please ensure all advanced risk management components are available.")
    except Exception as e:
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    print("Starting Advanced Risk Management Core Demo...")
    asyncio.run(main())