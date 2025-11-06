"""
Automated Trading System Demo

Comprehensive demonstration of the automated trading system showing:
- Market hours detection and session management
- Automated trading engine with perpetual operation
- Autonomous decision making and opportunity detection
- Continuous monitoring and health management
- Automated risk management and controls
- Comprehensive logging and reporting

Run this demo to see the full automated trading system in action.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json
import signal

# Add the trading orchestrator to the path
sys.path.append(str(Path(__file__).parent.parent))

from automated_trading import (
    MarketHoursManager, MarketSession, MarketType,
    AutomatedTradingEngine, AutomationLevel, TradingMode,
    AutonomousDecisionEngine, ContinuousMonitoringSystem,
    AutomatedRiskManager, RiskMode,
    AutomatedTradingConfig, TradingLogger
)
from automated_trading.config import AutomationLevel as AutoLevel
from automated_trading.risk_management import RiskMode as AutoRiskMode


class AutomatedTradingDemo:
    """Demo class for automated trading system"""
    
    def __init__(self):
        self.running = False
        
        # Initialize configuration
        self.config = AutomatedTradingConfig("demo_config.json")
        
        # Initialize core components
        self.market_hours = MarketHoursManager()
        self.trading_logger = TradingLogger(self.config)
        
        # System components (will be initialized in start)
        self.engine: Optional[AutomatedTradingEngine] = None
        self.autonomous_decisions: Optional[AutonomousDecisionEngine] = None
        self.monitoring: Optional[ContinuousMonitoringSystem] = None
        self.risk_manager: Optional[AutomatedRiskManager] = None
        
        print("🤖 Automated Trading System Demo Initialized")
    
    async def start(self):
        """Start the automated trading system demo"""
        try:
            print("\n🚀 Starting Automated Trading System...")
            print("=" * 60)
            
            # Setup signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            # Start logging system
            print("📝 Starting logging system...")
            await self.trading_logger.start()
            
            # Initialize components
            print("⚙️ Initializing system components...")
            
            # Initialize autonomous decisions engine
            self.autonomous_decisions = AutonomousDecisionEngine(self.config)
            await self.autonomous_decisions.initialize()
            
            # Initialize monitoring system
            self.monitoring = ContinuousMonitoringSystem(self.config)
            await self.monitoring.start()
            
            # Initialize risk manager
            self.risk_manager = AutomatedRiskManager(self.config)
            await self.risk_manager.start()
            
            # Initialize main trading engine
            self.engine = AutomatedTradingEngine(
                config=self.config,
                risk_manager=None,  # Would be integrated with actual risk manager
                order_manager=None,  # Would be integrated with actual order manager
                strategy_registry=None  # Would be integrated with actual strategy registry
            )
            
            # Start the engine
            engine_started = await self.engine.start()
            if not engine_started:
                raise Exception("Failed to start trading engine")
            
            self.running = True
            
            print("\n✅ Automated Trading System Started Successfully!")
            print("=" * 60)
            
            # Display initial status
            await self._display_initial_status()
            
            # Run the demo
            await self._run_demo()
            
        except Exception as e:
            print(f"\n❌ Error starting automated trading system: {e}")
            import traceback
            traceback.print_exc()
            await self.stop()
    
    async def stop(self):
        """Stop the automated trading system"""
        try:
            print("\n🛑 Stopping Automated Trading System...")
            
            self.running = False
            
            # Stop components in reverse order
            if self.engine:
                await self.engine.stop()
            
            if self.risk_manager:
                await self.risk_manager.stop()
            
            if self.monitoring:
                await self.monitoring.stop()
            
            if self.autonomous_decisions:
                await self.autonomous_decisions.stop()
            
            if self.trading_logger:
                await self.trading_logger.stop()
            
            print("✅ Automated Trading System stopped successfully")
            
        except Exception as e:
            print(f"❌ Error stopping system: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        asyncio.create_task(self.stop())
        sys.exit(0)
    
    async def _run_demo(self):
        """Run the main demo loop"""
        print("\n🎯 Running Automated Trading Demo...")
        print("Press Ctrl+C to stop the demo\n")
        
        demo_duration = 300  # 5 minutes for demo
        demo_end_time = datetime.utcnow() + timedelta(seconds=demo_duration)
        
        try:
            while self.running and datetime.utcnow() < demo_end_time:
                await asyncio.sleep(10)  # Update every 10 seconds
                
                # Display status updates
                if int((demo_end_time - datetime.utcnow()).total_seconds()) % 30 == 0:
                    await self._display_status_update()
            
            # Demo complete
            print(f"\n🎉 Demo completed! Ran for {demo_duration // 60} minutes")
            
        except asyncio.CancelledError:
            print("\n⚠️ Demo cancelled")
        except KeyboardInterrupt:
            print("\n⚠️ Demo interrupted by user")
        except Exception as e:
            print(f"\n❌ Demo error: {e}")
    
    async def _display_initial_status(self):
        """Display initial system status"""
        print("\n📊 INITIAL SYSTEM STATUS")
        print("-" * 40)
        
        # Display configuration summary
        config_summary = self.config.get_config_summary()
        print(f"🤖 Automation Level: {config_summary['automation_level']}")
        print(f"⚖️ Risk Level: {config_summary['risk_level']}")
        print(f"📈 Enabled Strategies: {config_summary['enabled_strategies']}")
        print(f"💼 Preferred Symbols: {len(config_summary['preferred_symbols'])}")
        print(f"💰 Max Position Size: {config_summary['max_position_size']}")
        print(f"📉 Max Daily Loss: {config_summary['max_daily_loss']}")
        
        # Display market status
        market_summary = self.market_hours.get_market_summary()
        print(f"\n🌍 MARKET STATUS")
        print("-" * 40)
        print(f"Open Exchanges: {market_summary['open_exchanges']}/{market_summary['total_exchanges']}")
        print(f"Markets Open: {'Yes' if market_summary['is_any_market_open'] else 'No'}")
        
        for exchange_name, info in market_summary['exchanges'].items():
            status_icon = "🟢" if info['is_open'] else "🔴"
            print(f"{status_icon} {exchange_name}: {info['session']}")
        
        # Display component status
        print(f"\n⚙️ COMPONENT STATUS")
        print("-" * 40)
        print("✅ Market Hours Manager: Active")
        print("✅ Logging System: Active")
        print("✅ Risk Manager: Active")
        print("✅ Monitoring System: Active")
        print("✅ Autonomous Decisions: Active")
        print("✅ Trading Engine: Active")
        
        print("\n" + "=" * 60)
    
    async def _display_status_update(self):
        """Display periodic status updates"""
        print(f"\n📊 STATUS UPDATE - {datetime.utcnow().strftime('%H:%M:%S UTC')}")
        print("-" * 50)
        
        # Get engine status
        if self.engine:
            engine_status = await self.engine.get_engine_status()
            print(f"🎯 Trading Engine: {engine_status['status']}")
            print(f"   Automation: {engine_status['automation_level']}")
            print(f"   Trading: {'Active' if engine_status['is_trading'] else 'Inactive'}")
            
            metrics = engine_status['metrics']
            print(f"   Trades Executed: {metrics['trades_executed']}")
            print(f"   Win Rate: {metrics['win_rate']:.1%}")
            print(f"   Total P&L: ${metrics['total_pnl']:,.2f}")
            print(f"   Health Score: {metrics['health_score']:.1f}")
        
        # Get monitoring status
        if self.monitoring:
            monitoring_status = await self.monitoring.get_monitoring_status()
            print(f"📈 Monitoring: {len(monitoring_status['active_alerts'])} active alerts")
            
            for alert_id, alert in list(monitoring_status['active_alerts'].items())[:3]:
                print(f"   ⚠️ {alert['title']}: {alert['message']}")
        
        # Get risk status
        if self.risk_manager:
            risk_status = await self.risk_manager.get_risk_status()
            print(f"⚖️ Risk Management: {risk_status['current_risk_mode']}")
            print(f"   Emergency Stop: {'Active' if risk_status['emergency_stop_active'] else 'Inactive'}")
            print(f"   Daily Loss Limit: {'Reached' if risk_status['daily_loss_limit_reached'] else 'OK'}")
            print(f"   Active Positions: {risk_status['active_positions']}")
        
        # Get autonomous decisions status
        if self.autonomous_decisions:
            decision_metrics = await self.autonomous_decisions.get_decision_metrics()
            print(f"🧠 AI Decisions: {decision_metrics['current_regime']}")
            print(f"   Active Opportunities: {decision_metrics['active_opportunities']}")
            print(f"   Symbols Monitored: {decision_metrics['symbols_monitored']}")
        
        # Get recent logs
        recent_logs = await self.trading_logger.get_recent_logs("trading", 3)
        if recent_logs:
            print(f"📝 Recent Activity:")
            for log in recent_logs:
                timestamp = log.timestamp.strftime('%H:%M:%S')
                print(f"   {timestamp} | {log.level} | {log.message}")
        
        print("-" * 50)


async def demonstrate_market_hours():
    """Demonstrate market hours detection"""
    print("\n🌍 MARKET HOURS DEMONSTRATION")
    print("=" * 50)
    
    market_hours = MarketHoursManager()
    
    # Show all exchanges
    all_sessions = market_hours.get_all_current_sessions()
    print(f"Current Market Sessions ({len(all_sessions)} exchanges):")
    
    for exchange_name, session in all_sessions.items():
        status_icon = "🟢" if session.is_market_open else "🔴"
        session_info = session.current_session.value.replace('_', ' ').title()
        
        print(f"{status_icon} {exchange_name:15} | {session_info:15}")
        
        if session.is_market_open and session.minutes_to_close:
            print(f"    └─ Closes in {session.minutes_to_close} minutes")
        elif not session.is_market_open and session.minutes_to_open:
            print(f"    └─ Opens in {session.minutes_to_open} minutes")
    
    # Show global status
    print(f"\nGlobal Status:")
    print(f"Any Market Open: {'Yes' if market_hours.is_any_market_open() else 'No'}")
    print(f"Open Exchanges: {', '.join(market_hours.get_open_exchanges())}")
    
    # Show market summary
    summary = market_hours.get_market_summary()
    print(f"\nDetailed Summary:")
    print(json.dumps(summary, indent=2, default=str))


async def demonstrate_autonomous_decisions():
    """Demonstrate autonomous decision making"""
    print("\n🧠 AUTONOMOUS DECISIONS DEMONSTRATION")
    print("=" * 50)
    
    # Initialize with demo config
    config = AutomatedTradingConfig("demo_config.json")
    decisions_engine = AutonomousDecisionEngine(config)
    await decisions_engine.initialize()
    
    # Perform market analysis
    print("Performing market analysis...")
    analysis = await decisions_engine.analyze_market_conditions()
    
    print(f"Market Analysis Results:")
    print(f"  Regime: {analysis.market_regime.value}")
    print(f"  Sentiment: {analysis.overall_sentiment:.2f} (-1=Very Bearish, +1=Very Bullish)")
    print(f"  Volatility: {analysis.volatility_level:.2f}")
    print(f"  Liquidity: {analysis.liquidity_score:.2f}")
    print(f"  Trend Strength: {analysis.trend_strength:.2f}")
    print(f"  Conditions Acceptable: {analysis.conditions_acceptable}")
    print(f"  Recommended Strategies: {', '.join(analysis.recommended_strategies)}")
    
    # Detect opportunities
    print(f"\nDetecting trading opportunities...")
    opportunities = await decisions_engine.detect_opportunities()
    
    print(f"Found {len(opportunities)} opportunities:")
    for i, opp in enumerate(opportunities[:5], 1):  # Show top 5
        print(f"{i}. {opp.symbol} {opp.action.upper()} | "
              f"Confidence: {opp.confidence:.2f} | "
              f"Expected Return: {opp.expected_return:.1%} | "
              f"Risk Score: {opp.risk_score:.2f}")
        print(f"   Reasoning: {opp.reasoning}")
    
    # Get decision metrics
    metrics = await decisions_engine.get_decision_metrics()
    print(f"\nDecision Engine Metrics:")
    print(f"  Current Regime: {metrics['current_regime']}")
    print(f"  Symbols Monitored: {metrics['symbols_monitored']}")
    print(f"  Regime Duration: {metrics['regime_duration']:.0f} seconds")
    
    await decisions_engine.stop()


async def demonstrate_risk_management():
    """Demonstrate risk management features"""
    print("\n⚖️ RISK MANAGEMENT DEMONSTRATION")
    print("=" * 50)
    
    config = AutomatedTradingConfig("demo_config.json")
    risk_manager = AutomatedRiskManager(config)
    await risk_manager.start()
    
    # Assess current risk
    print("Assessing current risk...")
    assessment = await risk_manager.assess_current_risk()
    
    print(f"Risk Assessment Results:")
    print(f"  Overall Risk Score: {assessment.overall_risk_score:.2f} (0=Low, 1=High)")
    print(f"  Portfolio VaR: {assessment.portfolio_var:.1%}")
    print(f"  Max Position Risk: {assessment.max_position_risk:.2f}")
    print(f"  Concentration Risk: {assessment.concentration_risk:.2f}")
    print(f"  Risk Mode: {assessment.risk_mode.value}")
    print(f"  Confidence: {assessment.confidence_level:.2f}")
    
    # Show recommended actions
    if assessment.recommended_actions:
        print(f"  Recommended Actions: {', '.join([action.value for action in assessment.recommended_actions])}")
    
    # Calculate position size example
    print(f"\nPosition Sizing Example:")
    position_size = await risk_manager.calculate_position_size(
        symbol="AAPL",
        strategy_confidence=0.75,
        market_volatility=0.15
    )
    print(f"  Recommended Position Size for AAPL: ${position_size}")
    
    # Update stop loss example
    print(f"\nStop Loss Example:")
    stop_levels = await risk_manager.update_stop_loss(
        symbol="AAPL",
        current_price=150.0,
        entry_price=145.0,
        position_size=1000.0
    )
    print(f"  Stop Loss: ${stop_levels['stop_loss']:.2f}")
    print(f"  Take Profit: ${stop_levels['take_profit']:.2f}")
    print(f"  Trailing Stop: ${stop_levels['trailing_stop']:.2f}")
    
    # Show risk status
    risk_status = await risk_manager.get_risk_status()
    print(f"\nRisk Management Status:")
    print(f"  Emergency Stop Active: {risk_status['emergency_stop_active']}")
    print(f"  Active Positions: {risk_status['active_positions']}")
    print(f"  Position Size Multiplier: {risk_status['position_size_multiplier']:.2f}")
    
    await risk_manager.stop()


async def demonstrate_continuous_monitoring():
    """Demonstrate continuous monitoring"""
    print("\n📊 CONTINUOUS MONITORING DEMONSTRATION")
    print("=" * 50)
    
    config = AutomatedTradingConfig("demo_config.json")
    monitoring = ContinuousMonitoringSystem(config)
    await monitoring.start()
    
    # Wait a moment for metrics collection
    await asyncio.sleep(5)
    
    # Get monitoring status
    status = await monitoring.get_monitoring_status()
    
    print(f"Monitoring System Status:")
    print(f"  Monitoring Active: {status['monitoring_active']}")
    print(f"  Start Time: {status['start_time']}")
    print(f"  Uptime: {status['monitoring_uptime']:.0f} seconds")
    
    # Show current metrics
    current_metrics = status['current_metrics']
    if current_metrics['system']:
        system_metrics = current_metrics['system']
        print(f"\nSystem Metrics:")
        print(f"  CPU Usage: {system_metrics['cpu_usage']:.1f}%")
        print(f"  Memory Usage: {system_metrics['memory_usage']:.1f}%")
        print(f"  Disk Usage: {system_metrics['disk_usage']:.1f}%")
        print(f"  Health Status: {system_metrics['health_status']}")
    
    if current_metrics['trading']:
        trading_metrics = current_metrics['trading']
        print(f"\nTrading Metrics:")
        print(f"  Total P&L: ${trading_metrics['total_pnl']:,.2f}")
        print(f"  Daily P&L: ${trading_metrics['daily_pnl']:,.2f}")
        print(f"  Active Positions: {trading_metrics['active_positions']}")
        print(f"  Win Rate: {trading_metrics['win_rate']:.1%}")
        print(f"  Current Drawdown: {trading_metrics['current_drawdown']:.1%}")
    
    if current_metrics['risk']:
        risk_metrics = current_metrics['risk']
        print(f"\nRisk Metrics:")
        print(f"  Portfolio VaR: {risk_metrics['portfolio_var']:.1%}")
        print(f"  Concentration Risk: {risk_metrics['concentration_risk']:.2f}")
        print(f"  Leverage Ratio: {risk_metrics['leverage_ratio']:.2f}")
    
    # Show active alerts
    active_alerts = status['active_alerts']
    print(f"\nActive Alerts: {len(active_alerts)}")
    for alert_id, alert in active_alerts.items():
        print(f"  {alert['level'].upper()}: {alert['title']} - {alert['message']}")
    
    # Get performance summary
    performance = await monitoring.get_performance_summary()
    print(f"\nPerformance Summary:")
    print(f"  System Health: {performance['system_health']:.1f}")
    print(f"  Active Alerts: {performance['active_alerts']}")
    print(f"  Monitoring Uptime: {performance['monitoring_uptime']:.0f} seconds")
    
    await monitoring.stop()


async def demonstrate_configuration():
    """Demonstrate configuration system"""
    print("\n⚙️ CONFIGURATION SYSTEM DEMONSTRATION")
    print("=" * 50)
    
    # Create demo configuration
    config = AutomatedTradingConfig("demo_config.json")
    
    # Show current configuration
    summary = config.get_config_summary()
    print("Current Configuration:")
    for key, value in summary.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Validate configuration
    errors = config.validate_config()
    if errors:
        print(f"\nConfiguration Errors:")
        for error in errors:
            print(f"  ❌ {error}")
    else:
        print(f"\n✅ Configuration is valid")
    
    # Demonstrate configuration updates
    print(f"\nDemonstrating Configuration Updates:")
    
    # Update automation level
    print("1. Updating automation level...")
    config.update_automation_level(AutoLevel.SEMI_AUTOMATED)
    print("   ✅ Automation level updated to SEMI_AUTOMATED")
    
    # Update risk level
    print("2. Updating risk level...")
    from strategies.base import RiskLevel
    config.update_risk_level(RiskLevel.CONSERVATIVE)
    print("   ✅ Risk level updated to CONSERVATIVE")
    
    # Update risk limits
    print("3. Updating risk limits...")
    config.update_risk_limits(
        max_daily_loss=3000.0,
        max_position_size=0.08
    )
    print("   ✅ Risk limits updated")
    
    # Show updated configuration
    updated_summary = config.get_config_summary()
    print(f"\nUpdated Configuration:")
    print(f"  Automation Level: {updated_summary['automation_level']}")
    print(f"  Risk Level: {updated_summary['risk_level']}")
    print(f"  Max Position Size: {updated_summary['max_position_size']}")
    print(f"  Max Daily Loss: {updated_summary['max_daily_loss']}")
    
    # Export configuration example
    print(f"\nExporting configuration...")
    config.export_config("demo_config_export.json")
    print("   ✅ Configuration exported to demo_config_export.json")


async def demonstrate_logging():
    """Demonstrate logging system"""
    print("\n📝 LOGGING SYSTEM DEMONSTRATION")
    print("=" * 50)
    
    config = AutomatedTradingConfig("demo_config.json")
    logger = TradingLogger(config)
    await logger.start()
    
    # Log various types of events
    print("Logging various events...")
    
    # Trading events
    await logger.log_trading_event("INFO", "Trade executed", {
        "symbol": "AAPL",
        "action": "BUY",
        "quantity": 100,
        "price": 150.00,
        "pnl": 25.50
    })
    
    await logger.log_trading_event("WARNING", "Position size alert", {
        "symbol": "TSLA",
        "current_size": 0.12,  # 12%
        "max_size": 0.10,      # 10%
        "excess": 0.02
    })
    
    # Risk events
    await logger.log_risk_event("ERROR", "Risk limit breached", {
        "limit_type": "daily_loss",
        "limit_value": 5000.0,
        "current_value": 5200.0,
        "excess": 200.0
    })
    
    # System events
    await logger.log_system_event("INFO", "System health check", {
        "cpu_usage": 65.0,
        "memory_usage": 70.0,
        "health_score": 85.0
    })
    
    # Performance event
    await logger.log_performance_event("INFO", "Performance report generated", {
        "report_type": "daily",
        "total_trades": 25,
        "win_rate": 0.68,
        "total_pnl": 1250.75
    })
    
    # Demonstrate logging statistics
    print("\nLogging Statistics:")
    stats = await logger.get_logging_statistics()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key.replace('_', ' ').title()}:")
            for subkey, subvalue in value.items():
                print(f"    {subkey}: {subvalue}")
        else:
            print(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Generate performance report
    print("\nGenerating performance report...")
    report = await logger.generate_performance_report("daily")
    print(f"Performance Report Generated:")
    print(f"  Report ID: {report.report_id}")
    print(f"  Period: {report.period_start.strftime('%Y-%m-%d')} to {report.period_end.strftime('%Y-%m-%d')}")
    print(f"  Total Trades: {report.total_trades}")
    print(f"  Win Rate: {report.win_rate:.1%}")
    print(f"  Total P&L: ${report.total_pnl:,.2f}")
    print(f"  Max Drawdown: {report.max_drawdown:.1%}")
    print(f"  System Uptime: {report.uptime_seconds / 3600:.1f} hours")
    
    await logger.stop()


async def main():
    """Main demo function"""
    print("🤖 AUTOMATED TRADING SYSTEM - COMPREHENSIVE DEMO")
    print("=" * 70)
    print("This demo showcases the complete automated trading system including:")
    print("• Market hours detection and session management")
    print("• Automated trading engine with perpetual operation")  
    print("• Autonomous decision making with AI integration")
    print("• Continuous monitoring and health management")
    print("• Automated risk management and controls")
    print("• Comprehensive logging and reporting")
    print("• User-configurable automation levels")
    print("=" * 70)
    
    try:
        # Run individual demonstrations
        print("\n📚 Running Individual Component Demonstrations...")
        
        # Market hours demonstration
        await demonstrate_market_hours()
        
        # Configuration demonstration
        await demonstrate_configuration()
        
        # Logging demonstration  
        await demonstrate_logging()
        
        # Autonomous decisions demonstration
        await demonstrate_autonomous_decisions()
        
        # Risk management demonstration
        await demonstrate_risk_management()
        
        # Continuous monitoring demonstration
        await demonstrate_continuous_monitoring()
        
        print("\n🎯 Individual demonstrations completed!")
        
        # Run full system demo
        print("\n🚀 Starting Full System Demo...")
        demo = AutomatedTradingDemo()
        await demo.start()
        
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n👋 Demo completed! Thank you for watching.")


if __name__ == "__main__":
    # Run the comprehensive demo
    asyncio.run(main())