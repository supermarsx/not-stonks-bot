"""
@file demo_frequency_system.py
@brief Comprehensive Demo of Trading Frequency Configuration System

@details
This module provides a complete demonstration of the trading frequency
configuration system, showcasing all features including configuration,
monitoring, analytics, risk management, and UI components.

Demo Features:
- Complete frequency system initialization and configuration
- Real-time frequency monitoring and alerting
- Analytics and optimization recommendations
- Risk management integration
- UI component demonstrations
- End-to-end workflow scenarios

@author Trading Orchestrator System
@version 1.0
@date 2025-11-06

@note
This demo script is designed to showcase all aspects of the frequency
configuration system and can be used for testing and validation.
"""

import asyncio
import sys
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn
from rich.live import Live
from rich.layout import Layout

# Import frequency system components
from config.trading_frequency import (
    FrequencyManager, FrequencySettings, FrequencyType, FrequencyAlertType,
    FrequencyAlert, FrequencyOptimization, initialize_frequency_manager,
    get_frequency_manager
)

from risk.frequency_risk_manager import (
    FrequencyRiskManager, FrequencyRiskAssessment, FrequencyRiskLimit,
    initialize_frequency_risk_manager, get_frequency_risk_manager
)

from analytics.frequency_analytics import (
    FrequencyAnalyticsEngine, FrequencyAnalyticsReport, AnalyticsPeriod,
    OptimizationTarget, FrequencyOptimizationInsight,
    initialize_frequency_analytics_engine, get_frequency_analytics_engine
)

from ui.components.frequency_components import (
    FrequencyConfigurationComponent, FrequencyMonitoringComponent,
    FrequencyAlertsComponent, FrequencyAnalyticsComponent
)


class FrequencySystemDemo:
    """
    @class FrequencySystemDemo
    @brief Comprehensive demo of the frequency configuration system
    
    @details
    Provides interactive demonstration of all frequency system components
    including initialization, configuration, monitoring, analytics, and
    risk management features.
    """
    
    def __init__(self):
        self.console = Console()
        self.frequency_manager = None
        self.frequency_risk_manager = None
        self.analytics_engine = None
        
        # Demo configuration
        self.demo_strategies = [
            "momentum_strategy",
            "mean_reversion_strategy", 
            "arbitrage_strategy",
            "scalping_strategy"
        ]
        
        # Demo data
        self.demo_trade_count = 0
        
    def display_welcome(self):
        """Display welcome screen"""
        welcome_text = Text()
        welcome_text.append("TRADING FREQUENCY CONFIGURATION SYSTEM\n", style="bold bright_green")
        welcome_text.append("=" * 50 + "\n", style="green")
        welcome_text.append("Comprehensive Demo and Testing Platform\n\n", style="bright_cyan")
        welcome_text.append("Features:\n", style="bold")
        welcome_text.append("• Frequency Configuration Management\n")
        welcome_text.append("• Real-time Frequency Monitoring\n")
        welcome_text.append("• Risk-based Frequency Controls\n")
        welcome_text.append("• Analytics and Optimization\n")
        welcome_text.append("• Interactive UI Components\n")
        welcome_text.append("• Cross-strategy Analysis\n\n")
        welcome_text.append(f"Demo Started: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        panel = Panel(
            welcome_text,
            title="[bold]Welcome to Frequency System Demo[/bold]",
            border_style="bright_green",
            padding=(2, 4)
        )
        
        self.console.print(panel)
    
    async def initialize_system(self):
        """Initialize the complete frequency system"""
        self.console.print("\n[bold yellow]Initializing Frequency System...[/bold yellow]\n")
        
        try:
            # 1. Initialize Frequency Manager
            self.console.print("[cyan]1. Initializing Frequency Manager...[/cyan]")
            settings = self._get_demo_frequency_settings()
            self.frequency_manager = initialize_frequency_manager(settings)
            self.console.print("[green]✓ Frequency Manager initialized[/green]")
            
            # 2. Initialize Risk Manager
            self.console.print("[cyan]2. Initializing Risk Manager...[/cyan]")
            base_risk_manager = self._create_mock_base_risk_manager()
            self.frequency_risk_manager = initialize_frequency_risk_manager(base_risk_manager)
            self.console.print("[green]✓ Risk Manager initialized[/green]")
            
            # 3. Initialize Analytics Engine
            self.console.print("[cyan]3. Initializing Analytics Engine...[/cyan]")
            self.analytics_engine = initialize_frequency_analytics_engine(self.frequency_manager)
            self.console.print("[green]✓ Analytics Engine initialized[/green]")
            
            # 4. Setup demo strategies
            await self._setup_demo_strategies()
            
            self.console.print("\n[bold green]✓ Frequency System initialization complete![/bold green]")
            
        except Exception as e:
            self.console.print(f"[red]Error initializing system: {e}[/red]")
            raise
    
    def _get_demo_frequency_settings(self) -> FrequencySettings:
        """Get demo frequency settings"""
        return FrequencySettings(
            frequency_type=FrequencyType.MEDIUM,
            interval_seconds=300,  # 5 minutes
            max_trades_per_minute=5,
            max_trades_per_hour=25,
            max_trades_per_day=120,
            position_size_multiplier=0.9,
            cooldown_periods=120,  # 2 minutes
            market_hours_only=True,
            max_daily_frequency_risk=0.04,
            frequency_volatility_adjustment=True,
            enable_alerts=True,
            auto_optimization=True,
            optimization_period_hours=24
        )
    
    def _create_mock_base_risk_manager(self):
        """Create mock base risk manager"""
        class MockRiskManager:
            def __init__(self):
                self.max_position_size = Decimal("50000")
                self.max_daily_loss = Decimal("5000")
                self.risk_per_trade = Decimal("0.02")
        
        return MockRiskManager()
    
    async def _setup_demo_strategies(self):
        """Setup demo strategies with different configurations"""
        self.console.print("[cyan]Setting up demo strategies...[/cyan]")
        
        # Add risk limits for each strategy
        risk_limits = [
            FrequencyRiskLimit(
                limit_id="limit_1",
                strategy_id="momentum_strategy",
                limit_type="hard",
                max_frequency_rate=8.0,
                max_position_size_multiplier=0.8
            ),
            FrequencyRiskLimit(
                limit_id="limit_2",
                strategy_id="mean_reversion_strategy",
                limit_type="recommended",
                max_frequency_rate=5.0,
                max_position_size_multiplier=1.0
            ),
            FrequencyRiskLimit(
                limit_id="limit_3",
                strategy_id="arbitrage_strategy",
                limit_type="soft",
                max_frequency_rate=15.0,
                max_position_size_multiplier=0.6
            ),
            FrequencyRiskLimit(
                limit_id="limit_4",
                strategy_id="scalping_strategy",
                limit_type="hard",
                max_frequency_rate=20.0,
                max_position_size_multiplier=0.4
            )
        ]
        
        for limit in risk_limits:
            self.frequency_risk_manager.add_risk_limit(limit)
        
        # Simulate some initial trades
        for strategy_id in self.demo_strategies:
            for _ in range(3):  # 3 initial trades per strategy
                await self.frequency_manager.record_trade(strategy_id)
        
        self.console.print("[green]✓ Demo strategies configured[/green]")
    
    async def demonstrate_configuration(self):
        """Demonstrate frequency configuration features"""
        self.console.print("\n[bold yellow]=== FREQUENCY CONFIGURATION DEMO ===[/bold yellow]\n")
        
        # 1. Display current configuration
        self._display_frequency_configuration()
        
        # 2. Demonstrate position sizing calculations
        await self._demonstrate_position_sizing()
        
        # 3. Show optimization recommendations
        await self._demonstrate_optimization_recommendations()
        
        # 4. Test different frequency settings
        await self._demonstrate_different_settings()
    
    def _display_frequency_configuration(self):
        """Display current frequency configuration"""
        if not self.frequency_manager:
            return
        
        settings = self.frequency_manager.settings
        
        table = Table(title="Current Frequency Configuration")
        table.add_column("Setting", style="cyan", no_wrap=True)
        table.add_column("Value", style="white")
        table.add_column("Description", style="green")
        
        table.add_row(
            "Frequency Type",
            settings.frequency_type.value,
            "Overall trading frequency category"
        )
        table.add_row(
            "Time Interval",
            f"{settings.interval_seconds}s ({settings.interval_seconds/60:.1f} min)",
            "Time between trades"
        )
        table.add_row(
            "Max Trades/Minute",
            str(settings.max_trades_per_minute),
            "Maximum trades per minute"
        )
        table.add_row(
            "Max Trades/Hour",
            str(settings.max_trades_per_hour),
            "Maximum trades per hour"
        )
        table.add_row(
            "Max Trades/Day",
            str(settings.max_trades_per_day),
            "Maximum trades per day"
        )
        table.add_row(
            "Position Size Multiplier",
            f"{settings.position_size_multiplier:.2f}",
            "Position size adjustment factor"
        )
        table.add_row(
            "Cooldown Period",
            f"{settings.cooldown_periods}s",
            "Mandatory wait between trades"
        )
        
        self.console.print(table)
    
    async def _demonstrate_position_sizing(self):
        """Demonstrate frequency-based position sizing"""
        self.console.print("\n[bold cyan]Position Sizing Calculations[/bold cyan]\n")
        
        base_size = Decimal("10000")
        test_frequencies = [1.0, 3.0, 6.0, 10.0, 15.0]
        
        table = Table(title="Frequency-Based Position Sizing")
        table.add_column("Frequency Rate", style="cyan")
        table.add_column("Base Size", style="white")
        table.add_column("Adjusted Size", style="green")
        table.add_column("Adjustment %", style="yellow")
        
        for freq_rate in test_frequencies:
            adjusted_size = await self.frequency_manager.calculate_position_size(
                strategy_id="demo_strategy",
                base_position_size=base_size,
                current_frequency_rate=freq_rate,
                market_volatility=0.2
            )
            
            adjustment_pct = ((adjusted_size - base_size) / base_size) * 100
            
            table.add_row(
                f"{freq_rate:.1f}/min",
                f"${base_size:,.2f}",
                f"${adjusted_size:,.2f}",
                f"{adjustment_pct:+.1f}%"
            )
        
        self.console.print(table)
        
        # Explanation
        explanation = Text()
        explanation.append("Position sizing adjusts based on trading frequency to manage risk.\n")
        explanation.append("Higher frequency strategies use smaller positions to reduce exposure.\n")
        explanation.append("Volatility adjustments are also applied when market volatility is high.")
        
        self.console.print(Panel(explanation, title="Position Sizing Logic", border_style="secondary"))
    
    async def _demonstrate_optimization_recommendations(self):
        """Demonstrate optimization recommendation generation"""
        self.console.print("\n[bold cyan]Optimization Recommendations[/bold cyan]\n")
        
        recommendations = await self.frequency_manager.generate_optimization_recommendations(
            "demo_strategy",
            backtest_period_days=30
        )
        
        if recommendations:
            for i, opt in enumerate(recommendations, 1):
                content = Text()
                content.append(f"Recommendation {i}:\n", style="bold")
                content.append(f"• Recommended Interval: {opt.recommended_interval_seconds}s\n")
                content.append(f"• Position Size Multiplier: {opt.recommended_position_size_multiplier:.2f}\n")
                content.append(f"• Confidence Level: {opt.confidence_level:.1%}\n")
                content.append(f"• Expected Improvement: {opt.expected_improvement:+.1f}%\n")
                content.append(f"• Backtest Period: {opt.backtest_period_days} days\n")
                
                panel = Panel(content, border_style="success")
                self.console.print(panel)
        else:
            self.console.print("[yellow]No optimization recommendations available for current settings[/yellow]")
    
    async def _demonstrate_different_settings(self):
        """Demonstrate different frequency settings configurations"""
        self.console.print("\n[bold cyan]Different Frequency Settings[/bold cyan]\n")
        
        settings_scenarios = [
            ("Scalping", FrequencyType.ULTRA_HIGH, 30, 20, 500, 0.3),
            ("Day Trading", FrequencyType.HIGH, 180, 15, 200, 0.6),
            ("Swing Trading", FrequencyType.MEDIUM, 900, 5, 50, 0.9),
            ("Position Trading", FrequencyType.VERY_LOW, 3600, 1, 10, 1.0)
        ]
        
        table = Table(title="Frequency Settings Comparison")
        table.add_column("Strategy Type", style="cyan")
        table.add_column("Frequency", style="white")
        table.add_column("Interval", style="yellow")
        table.add_column("Max/Min", style="green")
        table.add_column("Max/Hour", style="green")
        table.add_column("Max/Day", style="green")
        table.add_column("Position Multiplier", style="magenta")
        
        for name, freq_type, interval, max_min, max_hour, max_day, multiplier in settings_scenarios:
            table.add_row(
                name,
                freq_type.value,
                f"{interval}s",
                str(max_min),
                str(max_hour),
                str(max_day),
                f"{multiplier:.1f}"
            )
        
        self.console.print(table)
    
    async def demonstrate_monitoring(self):
        """Demonstrate real-time monitoring features"""
        self.console.print("\n[bold yellow]=== FREQUENCY MONITORING DEMO ===[/bold yellow]\n")
        
        # 1. Display current metrics
        await self._display_current_metrics()
        
        # 2. Simulate real-time monitoring
        await self._simulate_real_time_monitoring()
        
        # 3. Demonstrate alert system
        await self._demonstrate_alert_system()
    
    async def _display_current_metrics(self):
        """Display current frequency metrics"""
        self.console.print("[bold cyan]Current Frequency Metrics[/bold cyan]\n")
        
        metrics = self.frequency_manager.get_all_metrics()
        
        table = Table(title="Strategy Frequency Metrics")
        table.add_column("Strategy", style="cyan")
        table.add_column("Trades Today", style="white")
        table.add_column("Current Rate", style="yellow")
        table.add_column("Avg Rate", style="green")
        table.add_column("Efficiency", style="magenta")
        table.add_column("Status", style="red")
        
        for strategy_id, strategy_metrics in metrics.items():
            status_indicator = "🟢 Active"
            if strategy_metrics.in_cooldown:
                status_indicator = "🟡 Cooldown"
            if strategy_metrics.threshold_violations > 0:
                status_indicator = "🔴 Violations"
            
            table.add_row(
                strategy_id,
                str(strategy_metrics.trades_today),
                f"{strategy_metrics.current_frequency_rate:.2f}/min",
                f"{strategy_metrics.average_frequency_rate:.2f}/min",
                f"{strategy_metrics.frequency_efficiency:.1%}",
                status_indicator
            )
        
        self.console.print(table)
        
        # Display summary
        total_trades = sum(m.trades_today for m in metrics.values())
        avg_frequency = sum(m.current_frequency_rate for m in metrics.values()) / max(len(metrics), 1)
        
        summary = Text()
        summary.append(f"Total Trades Today: {total_trades}\n")
        summary.append(f"Average Frequency Rate: {avg_frequency:.2f}/min\n")
        summary.append(f"Active Strategies: {len(metrics)}\n")
        
        self.console.print(Panel(summary, title="System Summary", border_style="success"))
    
    async def _simulate_real_time_monitoring(self):
        """Simulate real-time monitoring with live updates"""
        self.console.print("\n[bold cyan]Real-time Monitoring Simulation[/bold cyan]\n")
        
        if not Confirm.ask("Start real-time monitoring simulation?"):
            return
        
        try:
            with Live(self._create_monitoring_layout(), refresh_per_second=1, console=self.console) as live:
                for _ in range(10):  # 10 seconds of simulation
                    # Simulate new trades
                    for strategy in self.demo_strategies:
                        if self.demo_trade_count % 3 == 0:  # Every 3rd iteration
                            await self.frequency_manager.record_trade(strategy)
                    
                    self.demo_trade_count += 1
                    
                    # Update layout
                    live.update(self._create_monitoring_layout())
                    
                    await asyncio.sleep(1)
                    
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Monitoring simulation stopped[/yellow]")
        
        self.console.print("\n[green]Monitoring simulation complete![/green]")
    
    def _create_monitoring_layout(self) -> Layout:
        """Create monitoring dashboard layout"""
        layout = Layout()
        
        # Header
        layout.split_column(
            Layout(name="header", size=4),
            Layout(name="main")
        )
        
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        # Header content
        header_content = Text()
        header_content.append("FREQUENCY MONITORING DASHBOARD\n", style="bold")
        header_content.append(f"Last Updated: {datetime.utcnow().strftime('%H:%M:%S')}\n")
        
        # Metrics content
        metrics = self.frequency_manager.get_all_metrics()
        total_trades = sum(m.trades_today for m in metrics.values())
        avg_rate = sum(m.current_frequency_rate for m in metrics.values()) / max(len(metrics), 1)
        
        metrics_content = Text()
        metrics_content.append(f"Total Trades: {total_trades}\n")
        metrics_content.append(f"Avg Frequency: {avg_rate:.2f}/min\n")
        metrics_content.append(f"Active Strategies: {len(metrics)}\n")
        
        # Alerts content
        active_alerts = self.frequency_manager.get_active_alerts()
        alerts_content = Text()
        alerts_content.append(f"Active Alerts: {len(active_alerts)}\n")
        if active_alerts:
            alerts_content.append("Recent Alerts:\n")
            for alert in active_alerts[:3]:
                alerts_content.append(f"• {alert.alert_type.value}\n")
        
        # Update layout
        layout["header"].update(Panel(header_content, border_style="primary"))
        layout["left"].update(Panel(metrics_content, title="Metrics", border_style="success"))
        layout["right"].update(Panel(alerts_content, title="Alerts", border_style="warning"))
        
        return layout
    
    async def _demonstrate_alert_system(self):
        """Demonstrate frequency alert system"""
        self.console.print("\n[bold cyan]Frequency Alert System[/bold cyan]\n")
        
        # Generate test alerts
        test_alerts = [
            ("threshold_exceeded", "medium", "Frequency rate approaching limit"),
            ("position_size_warning", "high", "Position size too large for frequency"),
            ("risk_limit_reached", "critical", "Daily frequency risk limit reached"),
            ("optimization_suggestion", "low", "Optimization opportunity detected")
        ]
        
        for alert_type, severity, message in test_alerts:
            alert = await self.frequency_manager.generate_alert(
                strategy_id="demo_strategy",
                alert_type=FrequencyAlertType(alert_type),
                message=message,
                severity=severity
            )
            self.console.print(f"[green]Generated alert: {alert_type} ({severity})[/green]")
        
        # Display alerts
        active_alerts = self.frequency_manager.get_active_alerts()
        
        if active_alerts:
            table = Table(title="Active Alerts")
            table.add_column("Type", style="cyan")
            table.add_column("Severity", style="white")
            table.add_column("Message", style="yellow")
            table.add_column("Time", style="green")
            
            for alert in active_alerts:
                table.add_row(
                    alert.alert_type.value,
                    alert.severity.upper(),
                    alert.message[:50] + "..." if len(alert.message) > 50 else alert.message,
                    alert.trigger_time.strftime('%H:%M:%S')
                )
            
            self.console.print(table)
        else:
            self.console.print("[yellow]No active alerts[/yellow]")
        
        # Demonstrate alert acknowledgment
        if active_alerts and Confirm.ask("Acknowledge first alert?"):
            first_alert = active_alerts[0]
            acknowledged = self.frequency_manager.acknowledge_alert(first_alert.alert_id)
            if acknowledged:
                self.console.print(f"[green]Alert {first_alert.alert_id} acknowledged[/green]")
            else:
                self.console.print("[red]Failed to acknowledge alert[/red]")
    
    async def demonstrate_risk_management(self):
        """Demonstrate frequency risk management features"""
        self.console.print("\n[bold yellow]=== FREQUENCY RISK MANAGEMENT DEMO ===[/bold yellow]\n")
        
        # 1. Risk assessment
        await self._demonstrate_risk_assessment()
        
        # 2. Compliance checking
        await self._demonstrate_compliance_checking()
        
        # 3. Cross-strategy risk monitoring
        await self._demonstrate_cross_strategy_monitoring()
    
    async def _demonstrate_risk_assessment(self):
        """Demonstrate frequency risk assessment"""
        self.console.print("[bold cyan]Frequency Risk Assessment[/bold cyan]\n")
        
        test_scenarios = [
            ("Normal Trading", 2.0, Decimal("5000"), 0.15),
            ("High Frequency", 8.0, Decimal("10000"), 0.25),
            ("Excessive Trading", 15.0, Decimal("20000"), 0.4),
            ("Critical Situation", 25.0, Decimal("50000"), 0.6)
        ]
        
        table = Table(title="Risk Assessment Scenarios")
        table.add_column("Scenario", style="cyan")
        table.add_column("Freq Rate", style="yellow")
        table.add_column("Position Size", style="white")
        table.add_column("Volatility", style="magenta")
        table.add_column("Risk Score", style="red")
        table.add_column("Risk Level", style="bold")
        
        for scenario, freq_rate, position_size, volatility in test_scenarios:
            assessment = await self.frequency_risk_manager.assess_frequency_risk(
                strategy_id="risk_demo_strategy",
                current_frequency_rate=freq_rate,
                position_size=position_size,
                market_volatility=volatility
            )
            
            risk_level_style = {
                "low": "green",
                "medium": "yellow", 
                "high": "red",
                "critical": "bold red"
            }.get(assessment.risk_level.value, "white")
            
            table.add_row(
                scenario,
                f"{freq_rate:.1f}/min",
                f"${position_size:,.0f}",
                f"{volatility:.1%}",
                f"{assessment.frequency_risk_score:.2f}",
                f"[{risk_level_style}]{assessment.risk_level.value.upper()}[/{risk_level_style}]"
            )
        
        self.console.print(table)
        
        # Show detailed assessment for one scenario
        detailed_assessment = await self.frequency_risk_manager.assess_frequency_risk(
            strategy_id="demo_strategy",
            current_frequency_rate=5.0,
            position_size=Decimal("8000"),
            market_volatility=0.3
        )
        
        self._display_detailed_assessment(detailed_assessment)
    
    def _display_detailed_assessment(self, assessment: FrequencyRiskAssessment):
        """Display detailed risk assessment"""
        content = Text()
        content.append(f"Strategy: {assessment.strategy_id}\n", style="bold")
        content.append(f"Risk Score: {assessment.frequency_risk_score:.3f}\n")
        content.append(f"Risk Level: {assessment.risk_level.value.upper()}\n\n")
        
        content.append("Risk Components:\n", style="bold")
        content.append(f"• Concentration Risk: {assessment.concentration_risk:.2f}\n")
        content.append(f"• Volatility Risk: {assessment.volatility_risk:.2f}\n")
        content.append(f"• Correlation Risk: {assessment.correlation_risk:.2f}\n")
        content.append(f"• Drawdown Risk: {assessment.drawdown_risk:.2f}\n\n")
        
        if assessment.active_violations:
            content.append("Active Violations:\n", style="bold red")
            for violation in assessment.active_violations:
                content.append(f"• {violation.value}\n")
            content.append("\n")
        
        if assessment.recommendations:
            content.append("Recommendations:\n", style="bold yellow")
            for rec in assessment.recommendations:
                content.append(f"• {rec}\n")
        
        panel = Panel(content, title="Detailed Risk Assessment", border_style="red")
        self.console.print(panel)
    
    async def _demonstrate_compliance_checking(self):
        """Demonstrate frequency compliance checking"""
        self.console.print("\n[bold cyan]Compliance Checking[/bold cyan]\n")
        
        test_trades = [
            {"symbol": "AAPL", "side": "buy", "quantity": 100, "market_volatility": 0.15},
            {"symbol": "GOOGL", "side": "sell", "quantity": 50, "market_volatility": 0.2},
            {"symbol": "TSLA", "side": "buy", "quantity": 200, "market_volatility": 0.4}
        ]
        
        table = Table(title="Trade Compliance Check")
        table.add_column("Symbol", style="cyan")
        table.add_column("Side", style="white")
        table.add_column("Quantity", style="yellow")
        table.add_column("Volatility", style="magenta")
        table.add_column("Allowed", style="bold")
        table.add_column("Violations", style="red")
        
        for trade in test_trades:
            compliance, violations = await self.frequency_risk_manager.check_frequency_risk_compliance(
                strategy_id="compliance_demo",
                proposed_trade=trade
            )
            
            status_style = "green" if compliance else "red"
            status_text = "✓ YES" if compliance else "✗ NO"
            
            violations_text = "; ".join(violations)[:30] + "..." if len("; ".join(violations)) > 30 else "; ".join(violations)
            
            table.add_row(
                trade["symbol"],
                trade["side"],
                str(trade["quantity"]),
                f"{trade['market_volatility']:.1%}",
                f"[{status_style}]{status_text}[/{status_style}]",
                violations_text if violations else "None"
            )
        
        self.console.print(table)
    
    async def _demonstrate_cross_strategy_monitoring(self):
        """Demonstrate cross-strategy risk monitoring"""
        self.console.print("\n[bold cyan]Cross-Strategy Risk Monitoring[/bold cyan]\n")
        
        # Create risk assessments for multiple strategies
        for strategy_id in self.demo_strategies:
            await self.frequency_risk_manager.assess_frequency_risk(
                strategy_id=strategy_id,
                current_frequency_rate=2.0 + hash(strategy_id) % 10,  # Varying frequencies
                position_size=Decimal("5000"),
                market_volatility=0.2
            )
        
        # Monitor cross-strategy risk
        monitoring_result = await self.frequency_risk_manager.monitor_cross_strategy_frequency_risk()
        
        if "risk_summary" in monitoring_result:
            summary = monitoring_result["risk_summary"]
            
            content = Text()
            content.append("Portfolio Risk Summary\n", style="bold")
            content.append(f"Portfolio Frequency Rate: {summary['portfolio_frequency_rate']:.2f}/min\n")
            content.append(f"Portfolio Risk Score: {summary['portfolio_risk_score']:.2f}\n")
            content.append(f"High Risk Strategies: {summary['high_risk_strategies_count']}\n")
            content.append(f"Total Strategies: {summary['total_strategies']}\n\n")
            
            content.append("Risk Factors:\n", style="bold")
            content.append(f"• Concentration Risk: {summary['concentration_risk']:.2f}\n")
            content.append(f"• Correlation Risk: {summary['correlation_risk']:.2f}\n")
            
            if summary.get('recommendations'):
                content.append("\nRecommendations:\n", style="bold yellow")
                for rec in summary['recommendations']:
                    content.append(f"• {rec}\n")
            
            panel = Panel(content, title="Cross-Strategy Risk Analysis", border_style="magenta")
            self.console.print(panel)
    
    async def demonstrate_analytics(self):
        """Demonstrate frequency analytics features"""
        self.console.print("\n[bold yellow]=== FREQUENCY ANALYTICS DEMO ===[/bold yellow]\n")
        
        # 1. Generate analytics report
        await self._demonstrate_analytics_report()
        
        # 2. Trend analysis
        await self._demonstrate_trend_analysis()
        
        # 3. Optimization insights
        await self._demonstrate_optimization_insights()
    
    async def _demonstrate_analytics_report(self):
        """Demonstrate analytics report generation"""
        self.console.print("[bold cyan]Analytics Report Generation[/bold cyan]\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Generating analytics report...", total=None)
            
            report = await self.analytics_engine.generate_analytics_report(
                strategy_id="analytics_demo",
                period=AnalyticsPeriod.DAILY
            )
            
            progress.update(task, description="Report generated!")
        
        # Display report summary
        if report:
            content = Text()
            content.append(f"Analytics Report for {report.strategy_id}\n", style="bold")
            content.append(f"Period: {report.period_start.strftime('%Y-%m-%d')} to {report.period_end.strftime('%Y-%m-%d')}\n\n")
            
            content.append("Summary Metrics:\n", style="bold")
            content.append(f"• Total Trades: {report.total_trades}\n")
            content.append(f"• Average Frequency Rate: {report.average_frequency_rate:.2f}/min\n")
            content.append(f"• Frequency Efficiency: {report.frequency_efficiency:.1%}\n")
            content.append(f"• Optimization Score: {report.optimization_score:.0f}/100\n\n")
            
            content.append("Risk Metrics:\n", style="bold")
            content.append(f"• Frequency VaR: {report.frequency_var:.2f}\n")
            content.append(f"• Frequency Volatility: {report.frequency_volatility:.2f}\n")
            content.append(f"• Max Frequency Drawdown: {report.frequency_drawdown:.1%}\n\n")
            
            if report.top_recommendations:
                content.append("Top Recommendations:\n", style="bold yellow")
                for i, rec in enumerate(report.top_recommendations, 1):
                    content.append(f"{i}. {rec}\n")
            
            panel = Panel(content, title="Analytics Report Summary", border_style="cyan")
            self.console.print(panel)
        else:
            self.console.print("[yellow]No analytics report generated[/yellow]")
    
    async def _demonstrate_trend_analysis(self):
        """Demonstrate frequency trend analysis"""
        self.console.print("\n[bold cyan]Frequency Trend Analysis[/bold cyan]\n")
        
        trends = await self.analytics_engine.analyze_frequency_trends(
            strategy_id="trends_demo",
            lookback_days=14
        )
        
        if "status" in trends:
            if trends["status"] == "insufficient_data":
                self.console.print("[yellow]Insufficient data for trend analysis[/yellow]")
            elif trends["status"] == "error":
                self.console.print(f"[red]Trend analysis error: {trends.get('message', 'Unknown')}[/red]")
            else:
                self.console.print(f"[green]Trend analysis status: {trends['status']}[/green]")
        else:
            # Display trends if available
            content = Text()
            content.append("Frequency Trend Analysis\n", style="bold")
            
            if "frequency_trends" in trends:
                content.append("Frequency Trends:\n", style="cyan")
                for trend in trends["frequency_trends"]:
                    content.append(f"• Trend: {trend.get('trend', 'N/A')}\n")
                    content.append(f"  Slope: {trend.get('slope', 0):.3f}\n")
                    content.append(f"  R²: {trend.get('r_squared', 0):.3f}\n")
                content.append("\n")
            
            if "volatility_trends" in trends:
                content.append("Volatility Trends:\n", style="cyan")
                for trend in trends["volatility_trends"]:
                    content.append(f"• Trend: {trend.get('trend', 'N/A')}\n")
                    content.append(f"  Slope: {trend.get('slope', 0):.3f}\n")
                content.append("\n")
            
            if "summary" in trends:
                summary = trends["summary"]
                content.append("Trend Summary:\n", style="bold")
                content.append(f"• Overall Trend: {summary.get('overall_trend', 'N/A')}\n")
                content.append(f"• Trend Strength: {summary.get('trend_strength', 0):.2f}\n")
            
            panel = Panel(content, title="Trend Analysis Results", border_style="green")
            self.console.print(panel)
    
    async def _demonstrate_optimization_insights(self):
        """Demonstrate optimization insights generation"""
        self.console.print("\n[bold cyan]Optimization Insights[/bold cyan]\n")
        
        targets = [
            OptimizationTarget.MAXIMIZE_SHARPE,
            OptimizationTarget.MINIMIZE_DRAWDOWN,
            OptimizationTarget.MAXIMIZE_FREQUENCY_EFFICIENCY
        ]
        
        for target in targets:
            self.console.print(f"[yellow]Generating insights for: {target.value}[/yellow]")
            
            insights = await self.analytics_engine.generate_optimization_insights(
                strategy_id="insights_demo",
                target=target
            )
            
            if insights:
                for i, insight in enumerate(insights, 1):
                    content = Text()
                    content.append(f"Optimization Insight {i}\n", style="bold")
                    content.append(f"Type: {insight.insight_type}\n")
                    content.append(f"Priority: {insight.priority.upper()}\n")
                    content.append(f"Current Frequency Rate: {insight.current_frequency_rate:.2f}/min\n")
                    content.append(f"Target Frequency Rate: {insight.target_frequency_rate:.2f}/min\n")
                    content.append(f"Expected Improvement: {insight.expected_improvement:+.1f}%\n")
                    content.append(f"Confidence Level: {insight.confidence_level:.1%}\n\n")
                    
                    if insight.recommended_settings:
                        content.append("Recommended Settings:\n", style="bold cyan")
                        for key, value in insight.recommended_settings.items():
                            content.append(f"• {key}: {value}\n")
                        content.append("\n")
                    
                    if insight.implementation_steps:
                        content.append("Implementation Steps:\n", style="bold green")
                        for step in insight.implementation_steps:
                            content.append(f"• {step}\n")
                    
                    panel = Panel(content, title=f"Insight ({insight.priority} priority)", border_style="blue")
                    self.console.print(panel)
                    self.console.print()
            else:
                self.console.print("[yellow]No optimization insights generated[/yellow]")
    
    async def demonstrate_ui_components(self):
        """Demonstrate UI components"""
        self.console.print("\n[bold yellow]=== UI COMPONENTS DEMO ===[/bold yellow]\n")
        
        # 1. Configuration Component
        await self._demonstrate_config_component()
        
        # 2. Monitoring Component
        await self._demonstrate_monitoring_component()
        
        # 3. Alerts Component
        await self._demonstrate_alerts_component()
        
        # 4. Analytics Component
        await self._demonstrate_analytics_component()
    
    async def _demonstrate_config_component(self):
        """Demonstrate frequency configuration component"""
        self.console.print("[bold cyan]Frequency Configuration Component[/bold cyan]\n")
        
        config_component = FrequencyConfigurationComponent(self.console)
        config_component.set_frequency_manager(self.frequency_manager)
        
        # Create configuration panel
        panel = config_component.create_frequency_config_panel("demo_strategy")
        self.console.print(panel)
        
        # Interactive configuration (non-interactive for demo)
        if Confirm.ask("Show interactive configuration interface?"):
            self.console.print("[yellow]Interactive configuration requires user input - skipped in demo[/yellow]")
    
    async def _demonstrate_monitoring_component(self):
        """Demonstrate frequency monitoring component"""
        self.console.print("\n[bold cyan]Frequency Monitoring Component[/bold cyan]\n")
        
        monitoring_component = FrequencyMonitoringComponent(self.console)
        monitoring_component.set_frequency_manager(self.frequency_manager)
        
        # Create monitoring dashboard
        layout = monitoring_component.create_monitoring_dashboard()
        
        panel = Panel(
            layout,
            title="Frequency Monitoring Dashboard",
            border_style="primary"
        )
        self.console.print(panel)
        
        # Demo real-time monitoring (simplified)
        if Confirm.ask("Show real-time monitoring simulation?"):
            self.console.print("[yellow]Real-time monitoring requires live updates - simulation shown[/yellow]")
            # Simulate a few updates
            for i in range(3):
                await self.frequency_manager.record_trade(f"ui_demo_strategy_{i}")
                self.console.print(f"[green]Demo trade recorded for strategy {i}[/green]")
                await asyncio.sleep(1)
    
    async def _demonstrate_alerts_component(self):
        """Demonstrate frequency alerts component"""
        self.console.print("\n[bold cyan]Frequency Alerts Component[/bold cyan]\n")
        
        alerts_component = FrequencyAlertsComponent(self.console)
        alerts_component.set_frequency_manager(self.frequency_manager)
        
        # Generate a test alert first
        await self.frequency_manager.generate_alert(
            strategy_id="ui_demo_strategy",
            alert_type=FrequencyAlertType.OPTIMIZATION_SUGGESTION,
            message="UI Demo Alert - Frequency optimization opportunity",
            severity="medium"
        )
        
        # Create alerts interface
        panel = alerts_component.create_alerts_interface()
        self.console.print(panel)
        
        # Interactive alert management (non-interactive for demo)
        if Confirm.ask("Show interactive alert management?"):
            self.console.print("[yellow]Interactive alert management requires user input - skipped in demo[/yellow]")
    
    async def _demonstrate_analytics_component(self):
        """Demonstrate frequency analytics component"""
        self.console.print("\n[bold cyan]Frequency Analytics Component[/bold cyan]\n")
        
        analytics_component = FrequencyAnalyticsComponent(self.console)
        analytics_component.set_analytics_engine(self.analytics_engine)
        
        # Create analytics interface
        panel = analytics_component.create_analytics_interface("ui_demo_strategy")
        self.console.print(panel)
        
        # Interactive analytics (non-interactive for demo)
        if Confirm.ask("Show interactive analytics interface?"):
            self.console.print("[yellow]Interactive analytics requires user input - skipped in demo[/yellow]")
    
    async def demonstrate_end_to_end_scenarios(self):
        """Demonstrate complete end-to-end scenarios"""
        self.console.print("\n[bold yellow]=== END-TO-END SCENARIOS DEMO ===[/bold yellow]\n")
        
        # Scenario 1: High-frequency trading optimization
        await self._demonstrate_high_frequency_scenario()
        
        # Scenario 2: Risk management in volatile markets
        await self._demonstrate_volatile_market_scenario()
        
        # Scenario 3: Portfolio frequency rebalancing
        await self._demonstrate_portfolio_rebalancing_scenario()
    
    async def _demonstrate_high_frequency_scenario(self):
        """Demonstrate high-frequency trading optimization scenario"""
        self.console.print("[bold cyan]Scenario 1: High-Frequency Trading Optimization[/bold cyan]\n")
        
        scenario_text = Text()
        scenario_text.append("SCENARIO: High-frequency scalping strategy experiencing\n", style="bold")
        scenario_text.append("excessive trading costs and diminishing returns.\n\n")
        scenario_text.append("GOAL: Optimize frequency to improve profitability\n")
        scenario_text.append("while maintaining competitive advantage.\n\n")
        
        self.console.print(Panel(scenario_text, title="High-Frequency Optimization", border_style="yellow"))
        
        # Simulate current state
        current_freq = 15.0  # trades per minute
        base_position = Decimal("1000")
        
        self.console.print(f"[yellow]Current State: {current_freq} trades/min, ${base_position} position size[/yellow]")
        
        # Assess current risk
        assessment = await self.frequency_risk_manager.assess_frequency_risk(
            strategy_id="hft_scenario",
            current_frequency_rate=current_freq,
            position_size=base_position,
            market_volatility=0.35
        )
        
        self.console.print(f"[red]Risk Score: {assessment.frequency_risk_score:.2f} ({assessment.risk_level.value})[/red]")
        
        # Generate optimization recommendations
        recommendations = await self.frequency_manager.generate_optimization_recommendations(
            "hft_scenario",
            backtest_period_days=30
        )
        
        if recommendations:
            opt = recommendations[0]
            self.console.print(f"[green]Recommended interval: {opt.recommended_interval_seconds}s[/green]")
            self.console.print(f"[green]Expected improvement: {opt.expected_improvement:+.1f}%[/green]")
            
            # Calculate new position size
            new_position = await self.frequency_manager.calculate_position_size(
                strategy_id="hft_scenario",
                base_position_size=base_position,
                current_frequency_rate=60.0/opt.recommended_interval_seconds,  # Convert to rate
                market_volatility=0.35
            )
            
            self.console.print(f"[blue]Optimized position size: ${new_position:,.2f}[/blue]")
    
    async def _demonstrate_volatile_market_scenario(self):
        """Demonstrate risk management in volatile markets scenario"""
        self.console.print("\n[bold cyan]Scenario 2: Risk Management in Volatile Markets[/bold cyan]\n")
        
        scenario_text = Text()
        scenario_text.append("SCENARIO: Multiple strategies operating during\n", style="bold")
        scenario_text.append("high market volatility period.\n\n")
        scenario_text.append("GOAL: Implement frequency-based risk controls\n")
        scenario_text.append("to prevent excessive exposure.\n\n")
        
        self.console.print(Panel(scenario_text, title="Volatile Market Risk Management", border_style="red"))
        
        # Simulate volatile market conditions
        high_volatility = 0.5
        
        self.console.print(f"[yellow]Market volatility: {high_volatility:.1%}[/yellow]")
        
        # Check compliance for each strategy in volatile conditions
        for strategy_id in self.demo_strategies[:3]:  # Test first 3 strategies
            compliance, violations = await self.frequency_risk_manager.check_frequency_risk_compliance(
                strategy_id=strategy_id,
                proposed_trade={
                    "symbol": "VOLATILE_STOCK",
                    "side": "buy",
                    "quantity": 100,
                    "market_volatility": high_volatility
                }
            )
            
            status = "✓ COMPLIANT" if compliance else "✗ VIOLATIONS"
            status_color = "green" if compliance else "red"
            
            self.console.print(f"[{status_color}]{strategy_id}: {status}[/{status_color}]")
            
            if violations:
                for violation in violations:
                    self.console.print(f"  • {violation}")
        
        # Cross-strategy monitoring
        monitoring_result = await self.frequency_risk_manager.monitor_cross_strategy_frequency_risk()
        
        if "risk_summary" in monitoring_result:
            summary = monitoring_result["risk_summary"]
            self.console.print(f"\n[red]Portfolio risk score: {summary['portfolio_risk_score']:.2f}[/red]")
            self.console.print(f"[yellow]High-risk strategies: {summary['high_risk_strategies_count']}[/yellow]")
    
    async def _demonstrate_portfolio_rebalancing_scenario(self):
        """Demonstrate portfolio frequency rebalancing scenario"""
        self.console.print("\n[bold cyan]Scenario 3: Portfolio Frequency Rebalancing[/bold cyan]\n")
        
        scenario_text = Text()
        scenario_text.append("SCENARIO: Portfolio with uneven frequency distribution\n", style="bold")
        scenario_text.append("across strategies causing concentration risk.\n\n")
        scenario_text.append("GOAL: Rebalance frequency allocation to optimize\n")
        scenario_text.append("portfolio performance and risk distribution.\n\n")
        
        self.console.print(Panel(scenario_text, title="Portfolio Frequency Rebalancing", border_style="blue"))
        
        # Analyze current frequency distribution
        analytics_result = await self.analytics_engine.analyze_cross_strategy_frequency(
            self.demo_strategies
        )
        
        if "status" in analytics_result:
            if analytics_result["status"] == "insufficient_strategies":
                self.console.print("[yellow]Insufficient strategies for cross-analysis[/yellow]")
            elif analytics_result["status"] == "error":
                self.console.print(f"[red]Analysis error: {analytics_result.get('message', 'Unknown')}[/red]")
            else:
                self.console.print("[green]Cross-strategy analysis completed[/green]")
                
                # Generate rebalancing recommendations
                insights = await self.analytics_engine.generate_optimization_insights(
                    strategy_id="portfolio_demo",
                    target=OptimizationTarget.BALANCE_RISK_RETURN
                )
                
                if insights:
                    self.console.print(f"\n[bold cyan]Rebalancing Recommendations:[/bold cyan]")
                    for insight in insights:
                        if "frequency" in insight.insight_type:
                            self.console.print(f"• {insight.recommended_settings}")
                            self.console.print(f"  Expected improvement: {insight.expected_improvement:+.1f}%")
    
    async def run_complete_demo(self):
        """Run the complete frequency system demonstration"""
        try:
            # Welcome
            self.display_welcome()
            
            # Initialize system
            await self.initialize_system()
            
            # Run demonstrations
            if Confirm.ask("Continue with configuration demo?"):
                await self.demonstrate_configuration()
            
            if Confirm.ask("Continue with monitoring demo?"):
                await self.demonstrate_monitoring()
            
            if Confirm.ask("Continue with risk management demo?"):
                await self.demonstrate_risk_management()
            
            if Confirm.ask("Continue with analytics demo?"):
                await self.demonstrate_analytics()
            
            if Confirm.ask("Continue with UI components demo?"):
                await self.demonstrate_ui_components()
            
            if Confirm.ask("Continue with end-to-end scenarios?"):
                await self.demonstrate_end_to_end_scenarios()
            
            # Final summary
            self.display_demo_summary()
            
        except Exception as e:
            self.console.print(f"[red]Demo error: {e}[/red]")
            raise
        finally:
            self.console.print("\n[bold green]Frequency System Demo Complete![/bold green]")
    
    def display_demo_summary(self):
        """Display demo summary and system status"""
        summary_text = Text()
        summary_text.append("DEMO SUMMARY\n", style="bold bright_green")
        summary_text.append("=" * 40 + "\n\n", style="green")
        
        if self.frequency_manager:
            metrics = self.frequency_manager.get_all_metrics()
            total_trades = sum(m.trades_today for m in metrics.values())
            
            summary_text.append("System Status:\n", style="bold")
            summary_text.append(f"• Frequency Manager: Active\n")
            summary_text.append(f"• Risk Manager: Active\n")
            summary_text.append(f"• Analytics Engine: Active\n")
            summary_text.append(f"• Total Trades Recorded: {total_trades}\n")
            summary_text.append(f"• Active Strategies: {len(metrics)}\n\n")
        
        summary_text.append("Components Demonstrated:\n", style="bold")
        summary_text.append("✓ Frequency Configuration Management\n")
        summary_text.append("✓ Real-time Frequency Monitoring\n")
        summary_text.append("✓ Frequency-based Risk Management\n")
        summary_text.append("✓ Analytics and Optimization\n")
        summary_text.append("✓ UI Components\n")
        summary_text.append("✓ End-to-end Workflows\n\n")
        
        summary_text.append(f"Demo Completed: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}\n", style="bright_cyan")
        
        panel = Panel(
            summary_text,
            title="[bold]Frequency System Demo Summary[/bold]",
            border_style="bright_green",
            padding=(2, 4)
        )
        
        self.console.print(panel)


async def main():
    """Main demo function"""
    demo = FrequencySystemDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    print("Starting Trading Frequency Configuration System Demo...")
    print("This demonstration will showcase all aspects of the frequency system.")
    print("=" * 70)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[yellow]Demo interrupted by user[/yellow]")
    except Exception as e:
        print(f"\n[red]Demo failed with error: {e}[/red]")
        sys.exit(1)