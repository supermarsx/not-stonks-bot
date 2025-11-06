# Automated Trading System

A comprehensive, fully automated trading system that runs perpetually during market hours with AI-driven decision making, autonomous risk management, and continuous monitoring.

## 🚀 Features

### Core Capabilities
- **Perpetual Operation**: Runs continuously during market hours with automatic session management
- **Market Hours Detection**: Support for US, EU, and Asian markets with holiday/weekend detection
- **Multi-Level Automation**: From manual to fully automated with user-configurable controls
- **AI-Driven Decisions**: Autonomous market analysis, opportunity detection, and signal generation
- **Dynamic Risk Management**: Real-time risk assessment with automatic position sizing and stop-loss updates
- **Comprehensive Monitoring**: Real-time system health, performance tracking, and alerting
- **Extensive Logging**: Detailed audit trails, performance reports, and decision logging

### Market Support
- **US Markets**: NYSE, NASDAQ with pre-market and after-hours
- **European Markets**: LSE, DAX, CAC with extended hours
- **Asian Markets**: TSE, HKEX, SSE with regional variations
- **24/7 Markets**: Cryptocurrency and Forex markets
- **Holiday Detection**: Automatic holiday calendar integration
- **Session Transitions**: Seamless pre-market/regular/after-hours management

### Risk Management
- **Dynamic Position Sizing**: Kelly criterion-inspired sizing with confidence adjustments
- **Automated Stop-Loss Management**: Trailing stops with dynamic adjustments
- **Risk-On/Risk-Off Switching**: Automatic risk mode transitions based on market conditions
- **Emergency Stop Mechanisms**: Multiple trigger conditions with automatic position closure
- **Portfolio Rebalancing**: Automatic rebalancing based on risk thresholds
- **Real-Time Risk Monitoring**: Continuous VaR, drawdown, and correlation monitoring

### Automation Levels
1. **Disabled**: No automated trading, manual operations only
2. **Manual**: Human review required for all decisions
3. **Advisory**: AI provides recommendations, human executes
4. **Semi-Automated**: AI executes with human approval for high-risk trades
5. **Fully Automated**: Complete autonomous operation with risk controls

## 📁 System Architecture

```
automated_trading/
├── __init__.py                 # Module initialization
├── market_hours.py             # Market hours detection & session management
├── automated_engine.py         # Core automated trading engine
├── autonomous_decisions.py     # AI-driven decision making system
├── continuous_monitor.py       # Real-time monitoring & health management
├── risk_management.py          # Automated risk management system
├── config.py                   # Configuration management system
├── logging_system.py           # Comprehensive logging & reporting
└── demo.py                     # Comprehensive demonstration script
```

## 🛠️ Installation & Setup

### Prerequisites
```bash
pip install numpy pandas loguru psutil
```

### Basic Setup
```python
from automated_trading import (
    AutomatedTradingEngine, 
    AutomatedTradingConfig,
    AutomationLevel,
    TradingMode
)

# Initialize configuration
config = AutomatedTradingConfig()
config.update_automation_level(AutomationLevel.SEMI_AUTOMATED)

# Create trading engine
engine = AutomatedTradingEngine(config)

# Start the system
await engine.start()
```

## 🎯 Quick Start

### 1. Basic Configuration
```python
from automated_trading.config import AutomatedTradingConfig

# Create configuration
config = AutomatedTradingConfig("my_config.json")

# Configure automation level
config.update_automation_level(AutomationLevel.FULLY_AUTOMATED)

# Set risk tolerance
config.update_risk_limits(
    max_daily_loss=5000.0,
    max_position_size=0.10,
    max_drawdown=0.15
)

# Configure market hours
config.update_market_hours(
    enable_pre_market=True,
    enable_after_hours=True,
    preferred_exchanges=["NYSE", "NASDAQ"]
)

# Save configuration
config.save_config()
```

### 2. Start Automated Trading
```python
from automated_trading import AutomatedTradingEngine

# Initialize engine
engine = AutomatedTradingEngine(config)

# Start trading
success = await engine.start()
if success:
    print("✅ Automated trading started")
else:
    print("❌ Failed to start trading")
```

### 3. Monitor Performance
```python
# Get engine status
status = await engine.get_engine_status()
print(f"Status: {status['status']}")
print(f"Trades: {status['metrics']['trades_executed']}")
print(f"P&L: ${status['metrics']['total_pnl']:,.2f}")

# Emergency stop if needed
if status['metrics']['health_score'] < 25:
    await engine.emergency_stop("Low health score")
```

## 📊 Market Hours System

The system automatically detects market sessions across multiple exchanges:

```python
from automated_trading import MarketHoursManager

market_hours = MarketHoursManager()

# Get current sessions
sessions = market_hours.get_all_current_sessions()
for exchange, session in sessions.items():
    print(f"{exchange}: {session.current_session.value}")
    print(f"  Open: {session.is_market_open}")
    if session.is_market_open:
        print(f"  Closes in: {session.minutes_to_close} minutes")

# Check if any markets are open
if market_hours.is_any_market_open():
    print("Markets are currently open")
```

### Supported Markets
- **US**: NYSE, NASDAQ (9:30 AM - 4:00 PM ET)
- **Europe**: LSE, DAX, CAC (8:00 AM - 4:30 PM GMT)
- **Asia**: TSE, HKEX, SSE (9:00 AM - 11:30 AM local time)
- **24/7**: Cryptocurrency and Forex markets

## 🧠 Autonomous Decision Making

The system includes AI-driven market analysis and opportunity detection:

```python
from automated_trading import AutonomousDecisionEngine

decisions = AutonomousDecisionEngine(config)
await decisions.initialize()

# Analyze market conditions
analysis = await decisions.analyze_market_conditions()
print(f"Market Regime: {analysis.market_regime.value}")
print(f"Sentiment: {analysis.overall_sentiment:.2f}")
print(f"Volatility: {analysis.volatility_level:.2f}")

# Detect opportunities
opportunities = await decisions.detect_opportunities()
for opp in opportunities:
    print(f"{opp.symbol} {opp.action} - Confidence: {opp.confidence:.2f}")
    print(f"  Reasoning: {opp.reasoning}")
```

### Market Regime Detection
- **Bull Trending**: Strong upward momentum
- **Bear Trending**: Strong downward momentum  
- **Sideways**: Range-bound trading
- **High Volatility**: Elevated market volatility
- **Low Volatility**: Stable, low-volatility environment
- **Crisis**: Market stress conditions

### Opportunity Types
- **Trend Following**: Momentum-based entries
- **Mean Reversion**: Price normalization trades
- **Breakout**: Volume-driven breakouts
- **Volatility**: Volatility arbitrage strategies

## ⚖️ Risk Management

Advanced risk management with automatic controls:

```python
from automated_trading import AutomatedRiskManager, RiskMode

risk_manager = AutomatedRiskManager(config)
await risk_manager.start()

# Assess current risk
assessment = await risk_manager.assess_current_risk()
print(f"Overall Risk: {assessment.overall_risk_score:.2f}")
print(f"Recommended Actions: {assessment.recommended_actions}")

# Calculate position size
position_size = await risk_manager.calculate_position_size(
    symbol="AAPL",
    strategy_confidence=0.75,
    market_volatility=0.15
)
print(f"Recommended Size: ${position_size}")

# Update stop loss
stop_levels = await risk_manager.update_stop_loss(
    symbol="AAPL",
    current_price=150.0,
    entry_price=145.0,
    position_size=1000.0
)
print(f"Stop Loss: ${stop_levels['stop_loss']}")
```

### Risk Controls
- **Position Sizing**: Dynamic sizing based on confidence and volatility
- **Stop Loss Management**: Automated trailing stops and dynamic adjustments
- **Risk Mode Switching**: Conservative/Moderate/Aggressive/Emergency modes
- **Emergency Stops**: Automatic triggers for critical risk events
- **Portfolio Limits**: Maximum exposure, concentration, and sector limits

## 📈 Continuous Monitoring

Real-time system monitoring and health management:

```python
from automated_trading import ContinuousMonitoringSystem

monitoring = ContinuousMonitoringSystem(config)
await monitoring.start()

# Get monitoring status
status = await monitoring.get_monitoring_status()
print(f"System Health: {status['current_metrics']['system']['health_status']}")

# Get performance summary
performance = await monitoring.get_performance_summary()
print(f"Total P&L: ${performance['total_pnl']:,.2f}")
print(f"Win Rate: {performance['win_rate']:.1%}")
print(f"Active Alerts: {performance['active_alerts']}")

# Register alert callback
def handle_alert(alert):
    print(f"ALERT: {alert.title} - {alert.message}")

monitoring.register_alert_callback(handle_alert)
```

### Monitoring Metrics
- **System Health**: CPU, memory, disk usage
- **Trading Performance**: P&L, win rate, drawdown
- **Risk Metrics**: VaR, concentration, correlation
- **Position Tracking**: Size, duration, P&L by symbol
- **Alert Management**: Real-time risk and system alerts

## 📝 Logging & Reporting

Comprehensive logging system with detailed audit trails:

```python
from automated_trading import TradingLogger

logger = TradingLogger(config)
await logger.start()

# Log trading events
await logger.log_trading_event("INFO", "Trade executed", {
    "symbol": "AAPL",
    "action": "BUY",
    "quantity": 100,
    "price": 150.00,
    "pnl": 25.50
})

# Log risk alerts
from automated_trading.logging_system import RiskAlert
alert = RiskAlert(
    alert_id="risk_001",
    timestamp=datetime.utcnow(),
    alert_level="warning",
    category="position",
    title="High Position Concentration",
    description="Position exceeds 15% of portfolio",
    metric_value=0.18,
    threshold_value=0.15,
    actions_taken=["position_reduction_recommended"]
)
await logger.log_risk_alert(alert)

# Generate performance report
report = await logger.generate_performance_report("daily")
print(f"Report: {report.total_trades} trades, ${report.total_pnl:,.2f} P&L")
```

### Log Types
- **Trading Logs**: All trading activity and decisions
- **Risk Logs**: Risk events and limit breaches
- **System Logs**: Health checks and performance metrics
- **Decision Logs**: AI decision audit trails
- **Performance Logs**: Performance reports and analysis

## ⚙️ Configuration

The system uses a comprehensive configuration system:

```python
from automated_trading.config import AutomatedTradingConfig, AutomationLevel

config = AutomatedTradingConfig()

# Automation settings
config.automation.default_automation_level = AutomationLevel.SEMI_AUTOMATED
config.automation.max_simultaneous_opportunities = 10

# Risk limits
config.risk.max_daily_loss = 5000.0
config.risk.max_position_size = 0.10
config.risk.default_stop_loss = 0.025

# Market hours
config.market_hours.enable_pre_market = True
config.market_hours.enable_after_hours = True
config.market_hours.preferred_exchanges = ["NYSE", "NASDAQ"]

# Monitoring
config.monitoring.enable_real_time_monitoring = True
config.monitoring.alert_drawdown_threshold = 0.05

# Save configuration
config.save_config()
```

### Configuration Validation
```python
# Validate configuration
errors = config.validate_config()
if errors:
    print("Configuration errors:")
    for error in errors:
        print(f"  - {error}")
else:
    print("Configuration is valid")

# Export/import configuration
config.export_config("backup_config.json")
config.import_config("backup_config.json")
```

## 🚨 Emergency Controls

Multiple layers of emergency controls:

```python
# Emergency stop
await engine.emergency_stop_trigger("Critical system error")

# Risk mode emergency
await risk_manager.switch_risk_mode(RiskMode.EMERGENCY, "Market crisis")

# Manual pause
await engine.pause("Manual intervention required")
```

### Emergency Triggers
- **Daily Loss Limit**: Automatic stop when daily loss exceeds threshold
- **Drawdown Limit**: Emergency stop on maximum drawdown
- **System Health**: Stop on critical system issues
- **Consecutive Losses**: Stop after too many losing trades
- **Manual Override**: User-initiated emergency stops

## 📊 Performance Reports

Automatic generation of comprehensive performance reports:

```python
# Generate daily report
daily_report = await logger.generate_performance_report("daily")

# Generate weekly report  
weekly_report = await logger.generate_performance_report("weekly")

# Custom period report
custom_report = await logger.generate_performance_report("custom")
```

### Report Contents
- **Trading Metrics**: Total trades, win rate, P&L breakdown
- **Risk Metrics**: VaR, drawdown, Sharpe ratio
- **System Metrics**: Uptime, availability, response times
- **Strategy Breakdown**: Performance by strategy type
- **Market Conditions**: Regime analysis and volatility

## 🎯 Demo Script

Run the comprehensive demo to see all features:

```bash
cd automated_trading
python demo.py
```

The demo includes:
- Market hours demonstration
- Configuration management
- Autonomous decision making
- Risk management controls
- Continuous monitoring
- Logging and reporting
- Full system integration

## 🔧 Integration

### With Existing Trading Orchestrator

```python
from trading_orchestrator.main import TradingOrchestratorApp
from automated_trading import AutomatedTradingEngine

# Initialize main orchestrator
orchestrator = TradingOrchestratorApp()
await orchestrator.initialize()

# Add automated trading
auto_engine = AutomatedTradingEngine(config)
await auto_engine.start()

# Integration with existing components
auto_engine.risk_manager = orchestrator.app_config.state.risk_manager
auto_engine.order_manager = orchestrator.app_config.state.order_manager

print("Automated trading integrated with existing system")
```

### Custom Strategy Integration

```python
from automated_trading.strategies.base import BaseStrategy, StrategyConfig

class MyCustomStrategy(BaseStrategy):
    async def generate_signals(self):
        # Custom signal generation logic
        return signals
    
    async def validate_signal(self, signal):
        # Custom signal validation
        return True

# Register with automated system
strategy_config = StrategyConfig(
    strategy_id="my_strategy",
    strategy_type=StrategyType.MOMENTUM,
    name="My Custom Strategy",
    parameters={"param1": "value1"}
)

strategy = MyCustomStrategy(strategy_config)
auto_engine.strategy_registry.register_strategy(strategy)
```

## 📈 Best Practices

### 1. Risk Management
- Start with conservative settings
- Use position sizing limits
- Enable emergency stops
- Monitor drawdown closely

### 2. Automation Levels
- Begin with ADVISORY mode
- Gradually increase automation
- Monitor decision quality
- Adjust confidence thresholds

### 3. Market Hours
- Enable pre-market for opportunities
- Use after-hours for news events
- Respect holiday schedules
- Monitor session transitions

### 4. Monitoring
- Set appropriate alert thresholds
- Review performance daily
- Monitor system health
- Track decision accuracy

### 5. Configuration
- Regular backup of configs
- Test changes in paper trading
- Document customizations
- Validate after updates

## 🐛 Troubleshooting

### Common Issues

1. **Market Hours Not Detected**
   ```python
   # Check exchange configuration
   sessions = market_hours.get_all_current_sessions()
   print(sessions)
   ```

2. **Configuration Errors**
   ```python
   # Validate configuration
   errors = config.validate_config()
   print(errors)
   ```

3. **Risk Limits Breached**
   ```python
   # Check risk status
   risk_status = await risk_manager.get_risk_status()
   print(risk_status)
   ```

4. **System Health Issues**
   ```python
   # Get monitoring status
   status = await monitoring.get_monitoring_status()
   print(status)
   ```

### Debug Logging

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or use the system's logging
config.monitoring.log_level = "DEBUG"
```

## 📚 API Reference

See the individual module documentation for detailed API reference:

- [Market Hours API](market_hours.py)
- [Automated Engine API](automated_engine.py) 
- [Autonomous Decisions API](autonomous_decisions.py)
- [Monitoring System API](continuous_monitor.py)
- [Risk Management API](risk_management.py)
- [Configuration API](config.py)
- [Logging API](logging_system.py)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add comprehensive tests
5. Submit a pull request

## 📄 License

This automated trading system is provided under the MIT License. See LICENSE file for details.

## ⚠️ Disclaimer

This system is for educational and research purposes. Trading involves substantial risk and is not suitable for all investors. Past performance does not guarantee future results. Always:

- Start with paper trading
- Use proper risk management
- Monitor system performance
- Understand the algorithms
- Consult financial advisors
- Comply with regulations

---

**Happy Trading! 🚀📈**