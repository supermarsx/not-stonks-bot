"""
Advanced Risk Management System - Comprehensive Demo

This script demonstrates the complete advanced risk management system
with institutional-grade features including:

1. Advanced Risk Models (VaR, CVaR, stress testing, etc.)
2. User-Configurable Risk Limits
3. Portfolio Optimization (MPT, Black-Litterman, Risk Parity)
4. Compliance Frameworks (Basel III, MiFID II, Dodd-Frank)
5. Real-time Risk Monitoring
6. API Integration Layer
7. Broker Integration
8. Trade Surveillance

Usage:
    python demo_advanced_risk_management.py

This demo showcases institutional-grade risk management capabilities
accessible for retail traders.
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

# Import advanced risk management components
from risk.models.var_models import VaRCalculator
from risk.models.cvar_models import CVaRCalculator
from risk.models.drawdown_models import DrawdownAnalyzer
from risk.models.volatility_models import VolatilityModeler
from risk.models.correlation_models import CorrelationAnalyzer
from risk.models.stress_testing import StressTestEngine
from risk.models.credit_risk import CreditRiskAnalyzer
from risk.enhanced_limits import EnhancedRiskLimits
from risk.portfolio_optimization import PortfolioOptimizer
from risk.real_time_monitor import RealTimeRiskMonitor
from risk.compliance_frameworks import ComplianceFrameworks, RegulationFramework
from risk.api_integration import RiskManagementAPI
from risk.integration_layer import RiskBrokerIntegration, Order, BrokerType
from risk.engine import RiskManager

class AdvancedRiskManagementDemo:
    """
    Comprehensive demonstration of advanced risk management capabilities
    
    This class showcases all the institutional-grade risk management
    features in an integrated demonstration.
    """
    
    def __init__(self):
        """Initialize the demo"""
        self.logger = logging.getLogger(__name__)
        
        # Initialize all risk management components
        self.var_calculator = VaRCalculator()
        self.cvar_calculator = CVaRCalculator()
        self.drawdown_analyzer = DrawdownAnalyzer()
        self.volatility_modeler = VolatilityModeler()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.stress_tester = StressTestEngine()
        self.credit_risk_analyzer = CreditRiskAnalyzer()
        self.enhanced_limits = EnhancedRiskLimits()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.real_time_monitor = RealTimeRiskMonitor()
        self.compliance_frameworks = ComplianceFrameworks()
        self.risk_manager = RiskManager(user_id=1)  # Demo user
        
        # Demo portfolio data
        self.demo_portfolio_data = self._generate_demo_portfolio_data()
        
        self.logger.info("Advanced Risk Management Demo initialized")
    
    def _generate_demo_portfolio_data(self) -> pd.DataFrame:
        """Generate demo portfolio data for testing"""
        np.random.seed(42)  # For reproducible results
        
        # Generate 252 days of returns for each asset
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        
        assets = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX']
        
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
    
    async def demo_1_basic_risk_models(self):
        """Demonstrate basic risk models (VaR, CVaR, Drawdown)"""
        print("\n" + "="*80)
        print("DEMO 1: ADVANCED RISK MODELS")
        print("="*80)
        
        try:
            # VaR Calculations
            print("\n📊 Calculating Value at Risk (VaR)...")
            
            # Historical VaR
            historical_var = self.var_calculator.calculate_historical_var(
                self.demo_portfolio_data, 
                confidence_level=0.95,
                time_horizon=1
            )
            print(f"Historical VaR (95%, 1-day): ${historical_var.get('var', 0):,.2f}")
            
            # Parametric VaR
            parametric_var = self.var_calculator.calculate_parametric_var(
                self.demo_portfolio_data,
                confidence_level=0.95,
                time_horizon=1
            )
            print(f"Parametric VaR (95%, 1-day): ${parametric_var.get('var', 0):,.2f}")
            
            # Monte Carlo VaR
            mc_var = self.var_calculator.calculate_monte_carlo_var(
                self.demo_portfolio_data,
                confidence_level=0.95,
                time_horizon=1
            )
            print(f"Monte Carlo VaR (95%, 1-day): ${mc_var.get('var', 0):,.2f}")
            
            # CVaR Calculation
            print("\n📊 Calculating Conditional Value at Risk (CVaR)...")
            cvar_result = self.cvar_calculator.calculate_cvar(
                self.demo_portfolio_data,
                confidence_level=0.95,
                time_horizon=1
            )
            print(f"CVaR (95%, 1-day): ${cvar_result.get('cvar', 0):,.2f}")
            
            # Drawdown Analysis
            print("\n📊 Calculating Maximum Drawdown...")
            
            # Generate portfolio value series from returns
            portfolio_value = 100000 * (1 + self.demo_portfolio_data.sum(axis=1)).cumprod()
            drawdown_result = self.drawdown_analyzer.calculate_drawdown(portfolio_value)
            
            print(f"Maximum Drawdown: {drawdown_result.get('max_drawdown', 0):.2%}")
            print(f"Current Drawdown: {drawdown_result.get('current_drawdown', 0):.2%}")
            print(f"Recovery Factor: {drawdown_result.get('recovery_factor', 0):.2f}")
            
            print("\n✅ Basic Risk Models Demo Completed")
            
        except Exception as e:
            print(f"❌ Error in basic risk models demo: {e}")
    
    async def demo_2_advanced_risk_analysis(self):
        """Demonstrate advanced risk analysis (Volatility, Correlation, Stress Testing)"""
        print("\n" + "="*80)
        print("DEMO 2: ADVANCED RISK ANALYSIS")
        print("="*80)
        
        try:
            # Volatility Analysis
            print("\n📊 Analyzing Portfolio Volatility...")
            
            # Fit GARCH model for one asset
            garch_vol = self.volatility_modeler.fit_garch(
                self.demo_portfolio_data['AAPL'],
                order=(1, 1)
            )
            print(f"AAPL GARCH Volatility (annualized): {np.sqrt(252) * garch_vol.get('conditional_volatility', 0):.2%}")
            
            # Calculate EWMA volatility
            ewma_vol = self.volatility_modeler.calculate_ewma(
                self.demo_portfolio_data,
                lambda_param=0.94
            )
            print(f"Portfolio EWMA Volatility: {ewma_vol.get('portfolio_volatility', 0):.2%}")
            
            # Correlation Analysis
            print("\n📊 Analyzing Asset Correlations...")
            
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
            for asset1, asset2, corr in corr_values[:3]:
                print(f"  {asset1} - {asset2}: {corr:.3f}")
            
            # Rolling Correlations
            rolling_corr = self.correlation_analyzer.rolling_correlations(
                self.demo_portfolio_data['AAPL'],
                self.demo_portfolio_data['GOOGL'],
                window=30
            )
            print(f"AAPL-GOOGL 30-day Rolling Correlation: {rolling_corr.mean():.3f}")
            
            print("\n✅ Advanced Risk Analysis Demo Completed")
            
        except Exception as e:
            print(f"❌ Error in advanced risk analysis demo: {e}")
    
    async def demo_3_stress_testing(self):
        """Demonstrate stress testing scenarios"""
        print("\n" + "="*80)
        print("DEMO 3: STRESS TESTING")
        print("="*80)
        
        try:
            # Historical Scenarios
            print("\n📊 Running Historical Stress Tests...")
            
            scenarios = ['black_monday_1987', 'financial_crisis_2008', 'covid_pandemic_2020']
            
            for scenario in scenarios:
                try:
                    result = self.stress_tester.historical_scenario_test(
                        self.demo_portfolio_data,
                        scenario
                    )
                    print(f"{scenario.replace('_', ' ').title()}: ${result.get('loss', 0):,.2f} loss")
                except Exception as e:
                    print(f"{scenario}: Error - {e}")
            
            # Monte Carlo Stress Test
            print("\n📊 Running Monte Carlo Stress Test...")
            mc_result = self.stress_tester.monte_carlo_stress_test(
                self.demo_portfolio_data,
                num_simulations=1000
            )
            print(f"Monte Carlo Stress Test: ${mc_result.get('loss', 0):,.2f} average loss")
            print(f"Stress Test Probability: {mc_result.get('probability', 0):.1%}")
            
            # Sensitivity Analysis
            print("\n📊 Running Sensitivity Analysis...")
            sensitivity = self.stress_tester.sensitivity_analysis(
                self.demo_portfolio_data,
                risk_factors=['market_shock', 'volatility_spike', 'correlation_breakdown']
            )
            print("Sensitivity Analysis Results:")
            for factor, impact in sensitivity.items():
                print(f"  {factor}: {impact:.3f}")
            
            print("\n✅ Stress Testing Demo Completed")
            
        except Exception as e:
            print(f"❌ Error in stress testing demo: {e}")
    
    async def demo_4_portfolio_optimization(self):
        """Demonstrate portfolio optimization techniques"""
        print("\n" + "="*80)
        print("DEMO 4: PORTFOLIO OPTIMIZATION")
        print("="*80)
        
        try:
            # Maximum Sharpe Ratio Optimization
            print("\n📊 Optimizing for Maximum Sharpe Ratio...")
            max_sharpe = self.portfolio_optimizer.optimize_max_sharpe(
                self.demo_portfolio_data,
                risk_free_rate=0.02
            )
            
            print("Maximum Sharpe Portfolio:")
            for asset, weight in max_sharpe.get('weights', {}).items():
                if weight > 0.01:  # Only show significant weights
                    print(f"  {asset}: {weight:.1%}")
            
            print(f"Expected Return: {max_sharpe.get('expected_return', 0):.2%}")
            print(f"Expected Volatility: {max_sharpe.get('volatility', 0):.2%}")
            print(f"Sharpe Ratio: {max_sharpe.get('sharpe_ratio', 0):.3f}")
            
            # Minimum Variance Optimization
            print("\n📊 Optimizing for Minimum Variance...")
            min_variance = self.portfolio_optimizer.optimize_min_variance(
                self.demo_portfolio_data
            )
            
            print("Minimum Variance Portfolio:")
            for asset, weight in min_variance.get('weights', {}).items():
                if weight > 0.01:
                    print(f"  {asset}: {weight:.1%}")
            
            print(f"Expected Volatility: {min_variance.get('volatility', 0):.2%}")
            
            # Risk Parity Optimization
            print("\n📊 Optimizing for Risk Parity...")
            risk_parity = self.portfolio_optimizer.optimize_risk_parity(
                self.demo_portfolio_data
            )
            
            print("Risk Parity Portfolio:")
            for asset, weight in risk_parity.get('weights', {}).items():
                if weight > 0.01:
                    print(f"  {asset}: {weight:.1%}")
            
            print("\n✅ Portfolio Optimization Demo Completed")
            
        except Exception as e:
            print(f"❌ Error in portfolio optimization demo: {e}")
    
    async def demo_5_compliance_monitoring(self):
        """Demonstrate compliance monitoring and reporting"""
        print("\n" + "="*80)
        print("DEMO 5: COMPLIANCE MONITORING")
        print("="*80)
        
        try:
            # Start compliance monitoring
            print("\n📊 Starting Compliance Monitoring...")
            self.compliance_frameworks.start_real_time_monitoring()
            
            # Get compliance status for different frameworks
            frameworks = [RegulationFramework.BASEL_III, RegulationFramework.MIFID_II, RegulationFramework.DODD_FRANK]
            
            for framework in frameworks:
                print(f"\n{framework.value.upper()} Compliance Status:")
                status = self.compliance_frameworks.get_compliance_status(framework)
                
                if status:
                    summary = status.get('status_summary', {})
                    active_breaches = status.get('active_breaches', 0)
                    
                    print(f"  Compliant Events: {summary.get('compliant', 0)}")
                    print(f"  Warning Events: {summary.get('warning', 0)}")
                    print(f"  Breach Events: {summary.get('breach', 0)}")
                    print(f"  Active Breaches: {active_breaches}")
            
            # Generate compliance reports
            print("\n📊 Generating Compliance Reports...")
            
            for framework in frameworks:
                report = self.compliance_frameworks.generate_compliance_report(framework)
                
                if report and 'executive_summary' in report:
                    summary = report['executive_summary']
                    print(f"\n{framework.value.upper()} Report:")
                    print(f"  Overall Status: {summary.get('overall_status', 'Unknown')}")
                    print(f"  Total Events: {summary.get('total_events', 0)}")
                    print(f"  Active Breaches: {summary.get('active_breaches', 0)}")
            
            # Stop monitoring
            self.compliance_frameworks.stop_real_time_monitoring()
            
            print("\n✅ Compliance Monitoring Demo Completed")
            
        except Exception as e:
            print(f"❌ Error in compliance monitoring demo: {e}")
    
    async def demo_6_trade_surveillance(self):
        """Demonstrate trade surveillance and monitoring"""
        print("\n" + "="*80)
        print("DEMO 6: TRADE SURVEILLANCE")
        print("="*80)
        
        try:
            # Sample trades for surveillance
            sample_trades = [
                {
                    "trade_id": "TRD_001",
                    "notional_amount": 50000000,  # $50M
                    "instrument_type": "equity",
                    "execution_price": 102.5,
                    "market_price": 102.0,
                    "short_sale": True,
                    "client_trades_per_day": 50,
                    "large_orders_cancelled": 2,
                    "layered_order_count": 1,
                    "time_between_large_orders_ms": 2000
                },
                {
                    "trade_id": "TRD_002",
                    "notional_amount": 500000,  # $500K
                    "instrument_type": "equity",
                    "execution_price": 150.0,
                    "market_price": 150.5,
                    "short_sale": False,
                    "client_trades_per_day": 20,
                    "large_orders_cancelled": 0,
                    "layered_order_count": 0,
                    "time_between_large_orders_ms": 5000
                }
            ]
            
            print("\n📊 Performing Trade Surveillance...")
            
            for i, trade in enumerate(sample_trades, 1):
                print(f"\nSurveillance for Trade {i}:")
                result = self.compliance_frameworks.trade_surveillance(trade)
                
                if result and 'surveillance_score' in result:
                    print(f"  Surveillance Score: {result['surveillance_score']}")
                    print(f"  Risk Flags: {result.get('risk_flags', [])}")
                    print(f"  Compliance Flags: {result.get('compliance_flags', [])}")
                    
                    # Check if trade triggers any concerns
                    if result['surveillance_score'] > 20:
                        print(f"  ⚠️  High surveillance score - Manual review recommended")
                    else:
                        print(f"  ✅ Normal surveillance profile")
                else:
                    print(f"  Error in surveillance analysis")
            
            print("\n✅ Trade Surveillance Demo Completed")
            
        except Exception as e:
            print(f"❌ Error in trade surveillance demo: {e}")
    
    async def demo_7_integration_layer(self):
        """Demonstrate broker integration and order management"""
        print("\n" + "="*80)
        print("DEMO 7: BROKER INTEGRATION")
        print("="*80)
        
        try:
            # Initialize broker integration
            broker_integration = RiskBrokerIntegration()
            
            print("\n📊 Starting Broker Integration...")
            await broker_integration.start_monitoring()
            
            # Get broker status
            print("\nBroker Status:")
            broker_status = await broker_integration._get_broker_status()  # Simplified call
            
            # Create sample orders for risk checking
            sample_orders = [
                Order(
                    order_id="ORD_001",
                    symbol="AAPL",
                    side="buy",
                    quantity=1000,
                    order_type="limit",
                    limit_price=150.0,
                    strategy_id="MOMENTUM_STRATEGY"
                ),
                Order(
                    order_id="ORD_002",
                    symbol="TSLA",
                    side="sell",
                    quantity=500,
                    order_type="market",
                    strategy_id="MEAN_REVERSION"
                )
            ]
            
            print("\n📊 Risk Checking Sample Orders...")
            
            for order in sample_orders:
                print(f"\nOrder {order.order_id}: {order.side.upper()} {order.quantity} {order.symbol}")
                
                # Perform risk checks
                risk_result = await broker_integration.check_order_risk(order)
                
                print(f"  Risk Action: {risk_result.action.value}")
                print(f"  Risk Score: {risk_result.risk_score:.2f}")
                print(f"  Allowed: {risk_result.allowed}")
                
                if risk_result.violations:
                    print(f"  Violations: {len(risk_result.violations)}")
                    for violation in risk_result.violations[:2]:  # Show first 2
                        print(f"    - {violation}")
                
                if risk_result.recommendations:
                    print(f"  Recommendations: {risk_result.recommendations[0]}")
            
            # Get portfolio summary
            portfolio_summary = broker_integration.get_portfolio_summary()
            print(f"\nPortfolio Summary:")
            print(f"  Total Value: ${portfolio_summary.get('total_value', 0):,.2f}")
            print(f"  Active Brokers: {portfolio_summary.get('active_brokers', 0)}")
            print(f"  Risk Alerts: {portfolio_summary.get('risk_alerts_count', 0)}")
            
            # Stop monitoring
            await broker_integration.stop_monitoring()
            
            print("\n✅ Broker Integration Demo Completed")
            
        except Exception as e:
            print(f"❌ Error in broker integration demo: {e}")
    
    async def demo_8_risk_manager_integration(self):
        """Demonstrate integrated RiskManager with all features"""
        print("\n" + "="*80)
        print("DEMO 8: INTEGRATED RISK MANAGER")
        print("="*80)
        
        try:
            # Start advanced monitoring
            print("\n📊 Starting Advanced Risk Monitoring...")
            monitoring_result = await self.risk_manager.start_advanced_monitoring(api_port=8000)
            print(f"Monitoring Status: {monitoring_result.get('status', 'unknown')}")
            
            # Calculate advanced risk metrics
            print("\n📊 Calculating Advanced Risk Metrics...")
            
            # Portfolio VaR
            var_result = await self.risk_manager.calculate_portfolio_var()
            if 'var_value' in var_result:
                print(f"Portfolio VaR (95%, 1-day): ${var_result['var_value']:,.2f}")
            
            # Portfolio CVaR
            cvar_result = await self.risk_manager.calculate_portfolio_cvar()
            if 'cvar_value' in cvar_result:
                print(f"Portfolio CVaR (95%, 1-day): ${cvar_result['cvar_value']:,.2f}")
            
            # Run stress tests
            print("\n📊 Running Stress Tests...")
            stress_result = await self.risk_manager.run_stress_tests()
            if 'summary' in stress_result:
                summary = stress_result['summary']
                print(f"Stress Test Summary:")
                print(f"  Scenarios Tested: {summary.get('total_scenarios', 0)}")
                print(f"  Worst Case Loss: ${summary.get('worst_case_loss', 0):,.2f}")
                print(f"  Average Loss: ${summary.get('average_loss', 0):,.2f}")
            
            # Generate compliance report
            print("\n📊 Generating Compliance Report...")
            compliance_report = await self.risk_manager.generate_compliance_report("basel_iii")
            if 'executive_summary' in compliance_report:
                summary = compliance_report['executive_summary']
                print(f"Basel III Compliance Status: {summary.get('overall_status', 'Unknown')}")
                print(f"Active Breaches: {summary.get('active_breaches', 0)}")
            
            # Portfolio optimization
            print("\n📊 Optimizing Portfolio...")
            optimization_result = await self.risk_manager.optimize_portfolio("max_sharpe")
            if 'sharpe_ratio' in optimization_result:
                print(f"Optimal Sharpe Ratio: {optimization_result['sharpe_ratio']:.3f}")
            
            # Get comprehensive risk status
            print("\n📊 Getting Comprehensive Risk Status...")
            risk_status = await self.risk_manager.get_advanced_risk_status()
            
            if 'advanced_metrics' in risk_status:
                metrics = risk_status['advanced_metrics']
                print(f"Advanced Risk Metrics:")
                print(f"  VaR (95%, 1-day): ${metrics.get('var_95_1d', 0):,.2f}")
                print(f"  CVaR (95%, 1-day): ${metrics.get('cvar_95_1d', 0):,.2f}")
            
            # Stop monitoring
            await self.risk_manager.stop_advanced_monitoring()
            
            print("\n✅ Integrated Risk Manager Demo Completed")
            
        except Exception as e:
            print(f"❌ Error in integrated risk manager demo: {e}")
    
    async def run_complete_demo(self):
        """Run the complete advanced risk management demonstration"""
        print("🏛️  ADVANCED RISK MANAGEMENT SYSTEM DEMO")
        print("=" * 80)
        print("This demo showcases institutional-grade risk management capabilities")
        print("accessible for retail traders through the Trading Orchestrator.")
        print("=" * 80)
        
        demos = [
            ("Basic Risk Models", self.demo_1_basic_risk_models),
            ("Advanced Risk Analysis", self.demo_2_advanced_risk_analysis),
            ("Stress Testing", self.demo_3_stress_testing),
            ("Portfolio Optimization", self.demo_4_portfolio_optimization),
            ("Compliance Monitoring", self.demo_5_compliance_monitoring),
            ("Trade Surveillance", self.demo_6_trade_surveillance),
            ("Broker Integration", self.demo_7_integration_layer),
            ("Integrated Risk Manager", self.demo_8_risk_manager_integration)
        ]
        
        start_time = datetime.now()
        
        for name, demo_func in demos:
            try:
                print(f"\n🚀 Starting: {name}")
                await demo_func()
                print(f"✅ Completed: {name}")
            except Exception as e:
                print(f"❌ Failed: {name} - {e}")
                continue
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "="*80)
        print("🎉 DEMO COMPLETE")
        print("="*80)
        print(f"Total Duration: {duration:.1f} seconds")
        print(f"Demos Executed: {len(demos)}")
        print("\nKey Features Demonstrated:")
        print("✅ Advanced Risk Models (VaR, CVaR, Drawdown)")
        print("✅ Volatility and Correlation Analysis")
        print("✅ Historical and Monte Carlo Stress Testing")
        print("✅ Portfolio Optimization (MPT, Risk Parity)")
        print("✅ Regulatory Compliance (Basel III, MiFID II, Dodd-Frank)")
        print("✅ Real-time Trade Surveillance")
        print("✅ Broker Integration and Risk Checks")
        print("✅ Integrated Risk Management System")
        print("\n🏛️  Institutional-grade risk management now accessible to retail traders!")
        
        # Cleanup
        try:
            await self.risk_manager.close()
        except:
            pass


async def main():
    """Main demonstration function"""
    demo = AdvancedRiskManagementDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    print("Starting Advanced Risk Management System Demo...")
    asyncio.run(main())