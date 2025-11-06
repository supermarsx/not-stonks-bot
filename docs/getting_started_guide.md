# Getting Started Guide

Welcome to the Day Trading Orchestrator system! This guide will help you get up and running quickly, whether you're new to trading or an experienced trader.

## 🚀 Quick Start (5 Minutes)

### Prerequisites Check
Before we begin, ensure you have:
- ✅ Python 3.8+ installed
- ✅ 4GB+ RAM available
- ✅ Internet connection
- ✅ Terminal with 256-color support

### 1. System Installation

```bash
# Navigate to the trading orchestrator directory
cd /workspace/trading_orchestrator

# Install dependencies
pip install -r requirements.txt

# Run system validation
python validate_system.py
```

### 2. First Launch

```bash
# Start the system (this will create config files automatically)
python main.py

# Or use the startup script
python start_system.py
```

The system will:
- 🔧 Auto-create configuration files
- 🗄️ Initialize the database
- 📊 Launch the Matrix-themed terminal interface
- 🌐 Connect to default broker (if configured)

### 3. Basic Configuration

The system creates `config.json` automatically. Edit it with your settings:

```json
{
  "ai": {
    "trading_mode": "PAPER"  // ALWAYS start with PAPER!
  },
  "brokers": {
    "binance": {
      "enabled": true,
      "api_key": "your_testnet_key",
      "testnet": true  // Use testnet for learning
    }
  },
  "risk": {
    "max_position_size": 1000,  // Start small!
    "max_daily_loss": 100
  }
}
```

## 📚 Learning Path

### Phase 1: System Familiarization (Day 1)
1. **Understand the Interface**
   - Launch the Matrix terminal
   - Explore the dashboard sections
   - Learn keyboard shortcuts

2. **Configure Paper Trading**
   - Set up a testnet broker account
   - Configure small position limits
   - Test basic order placement

3. **Run Your First Demo**
   ```bash
   python demo.py
   ```

### Phase 2: Strategy Basics (Week 1)
1. **Learn Basic Strategies**
   - Mean Reversion
   - Trend Following
   - Simple Pairs Trading

2. **Practice Risk Management**
   - Set position limits
   - Configure stop losses
   - Monitor risk metrics

3. **Backtest Strategies**
   ```bash
   # Run backtests on historical data
   python -c "from strategies.backtesting import BacktestEngine; BacktestEngine().run_demo()"
   ```

### Phase 3: Advanced Features (Month 1)
1. **AI Integration**
   - Configure GPT-4 for market analysis
   - Use AI for strategy selection
   - Enable AI risk assessment

2. **Multi-Broker Setup**
   - Connect multiple broker accounts
   - Compare execution quality
   - Optimize routing

3. **Custom Strategies**
   - Create your own strategies
   - Parameter optimization
   - Performance tuning

## 🎮 Interface Tour

### Matrix Terminal Interface

When you launch the system, you'll see the Matrix-themed terminal with several sections:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         DAY TRADING ORCHESTRATOR v2.0                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│ SYSTEM STATUS: [ONLINE] │ MODE: [PAPER] │ RISK: [LOW] │ BROKERS: [1/7 ACTIVE]  │
├─────────────────┬─────────────────┬─────────────────┬─────────────────────────┤
│ ACCOUNT SUMMARY  │ POSITIONS       │ ACTIVE ORDERS   │ MARKET OVERVIEW          │
├─────────────────┤ ┌───────────────┤ ┌───────────────┤ ┌─────────────────────────┤
│ Balance: $10,000 │ │ AAPL:  +$250  │ │ GOOGL: Buy    │ │ AAPL:   $150.25 ▲     │
│ Equity: $10,250 │ │ BTC:   +$500  │ │ 50 @ $150.20  │ │ GOOGL:  $2,850.10 ▼    │
│ P&L:    +$250   │ │ TSLA:  -$100  │ │ Status: Pending│ │ BTC:   $45,250.00 ▲    │
│                      └───────────────┘ └───────────────┘ └─────────────────────────┘
├─────────────────┬─────────────────┬─────────────────┬─────────────────────────┤
│ RISK METRICS    │ AI INSIGHTS     │ STRATEGY STATUS │ SYSTEM HEALTH           │
├─────────────────┤ ┌───────────────┤ ┌───────────────┤ ┌─────────────────────────┤
│ Portfolio Risk: │ │ Market: BULL │ │ Mean Rev:    │ │ CPU:    45%            │
│ 2.3% (LOW)      │ │ Sentiment:    │ │   ACTIVE      │ │ Memory: 2.1GB          │
│ Max Drawdown:   │ │   Positive    │ │ Trend:       │ │ Uptime: 99.9%          │
│ 1.2%            │ │ AI Score: 8.2 │ │   Monitoring  │ │ Brokers: Connected     │
└─────────────────┘ └───────────────┘ └───────────────┘ └─────────────────────────┘
```

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+C` | Graceful shutdown |
| `F1` | Show help |
| `F2` | Toggle fullscreen |
| `Tab` | Cycle through sections |
| `Enter` | Select/Execute |
| `Esc` | Cancel/Back |
| `Space` | Pause/Resume |
| `R` | Refresh data |
| `Q` | Quit |

### Dashboard Sections

#### 1. Account Summary
- **Balance**: Available cash
- **Equity**: Total account value
- **P&L**: Real-time profit/loss

#### 2. Positions
- **Current holdings**
- **Unrealized P&L**
- **Position size and cost basis**

#### 3. Active Orders
- **Pending orders**
- **Order status**
- **Fill progress**

#### 4. Market Overview
- **Price movements**
- **Volume indicators**
- **Market sentiment**

#### 5. Risk Metrics
- **Portfolio risk level**
- **Maximum drawdown**
- **Risk alerts**

#### 6. AI Insights
- **Market analysis**
- **AI recommendations**
- **Sentiment scores**

#### 7. Strategy Status
- **Active strategies**
- **Performance metrics**
- **Signal generation**

## 🎯 Your First Trading Session

### 1. Start Paper Trading

```bash
# Ensure paper mode is enabled
python -c "
import json
with open('config.json', 'r') as f:
    config = json.load(f)
config['ai']['trading_mode'] = 'PAPER'
with open('config.json', 'w') as f:
    json.dump(config, f, indent=2)
print('Paper trading enabled!')
"
```

### 2. Place Your First Order

From the terminal interface:
1. Press `O` to open the order panel
2. Select symbol (e.g., AAPL)
3. Choose order type (Market/Limit)
4. Enter quantity (start with 1-10 shares)
5. Confirm order

Or via command line:
```python
# Python console in terminal
>>> broker = get_broker('alpaca')
>>> await broker.place_order('AAPL', 'buy', 'market', 10)
```

### 3. Monitor Performance

Watch the dashboard for:
- ✅ Order execution confirmation
- 📊 Position updates
- 💰 P&L changes
- ⚠️ Risk alerts

### 4. Review Results

After execution:
- Check order history
- Analyze fill prices
- Review risk metrics
- Document lessons learned

## 📖 Essential Concepts

### Trading Modes

**PAPER TRADING** 🟢
- Simulated trading with fake money
- Perfect for learning
- No real financial risk
- All features enabled

**LIVE TRADING** 🔴
- Real money trading
- Requires careful preparation
- Full risk exposure
- Start small amounts only

**ANALYSIS MODE** 🟡
- Market analysis only
- No trading executed
- Strategy testing
- Research and development

### Order Types

**Market Orders**
- Execute immediately at current price
- Guaranteed execution
- No price control
- Best for: High liquidity, urgency

**Limit Orders**
- Execute at specified price or better
- Price control
- May not fill
- Best for: Price-sensitive trading

**Stop Orders**
- Trigger at stop price
- Convert to market order
- Risk management tool
- Best for: Stop losses

**Stop-Limit Orders**
- Trigger at stop price
- Execute as limit order
- Price and execution control
- Best for: Risk management

### Risk Management Basics

**Position Sizing**
- Never risk more than 1-2% per trade
- Use position size calculators
- Adjust for volatility

**Stop Losses**
- Always use stop losses
- Set at logical technical levels
- Stick to your plan

**Diversification**
- Don't concentrate in single assets
- Consider correlations
- Spread across strategies

**Daily Limits**
- Set maximum daily loss
- Stop trading if limit hit
- Maintain discipline

## 🛠️ Configuration Basics

### Quick Configuration

The system creates sensible defaults, but you'll want to customize:

```json
{
  "risk": {
    "max_position_size": 1000,        // Start small: $1,000
    "max_daily_loss": 100,            // Daily loss limit
    "stop_loss_percentage": 0.05,     // 5% stop loss
    "take_profit_percentage": 0.10    // 10% take profit
  },
  "strategies": {
    "enabled_strategies": [
      "mean_reversion",              // Good for beginners
      "trend_following"              // Popular strategy
    ]
  }
}
```

### Broker Setup

**For Beginners (Recommended)**

1. **Binance (Crypto)**
   - Easy API setup
   - Testnet available
   - Good for learning

2. **Alpaca (US Stocks)**
   - Paper trading
   - Commission-free
   - US market focus

3. **Trading 212 (EU Stocks)**
   - European markets
   - Practice account
   - User-friendly

### AI Configuration

Start with conservative AI settings:

```json
{
  "ai": {
    "trading_mode": "PAPER",
    "default_model_tier": "fast",          // Faster, lower cost
    "max_tokens_per_request": 2000,        // Conservative limits
    "local_models": {
      "enabled": false                     // Start with cloud models
    }
  }
}
```

## 🎓 Next Steps

### Week 1 Goals
- [ ] Complete paper trading setup
- [ ] Place 10+ test orders
- [ ] Understand the interface
- [ ] Configure basic risk limits
- [ ] Run first backtest

### Week 2 Goals
- [ ] Enable 2-3 basic strategies
- [ ] Monitor performance daily
- [ ] Adjust risk parameters
- [ ] Learn order types
- [ ] Document lessons

### Month 1 Goals
- [ ] Achieve consistent paper profits
- [ ] Master all basic features
- [ ] Understand all strategies
- [ ] Configure multiple brokers
- [ ] Consider live trading (small amounts)

### Resources for Continued Learning

1. **Strategy Guide**: Master 50+ available strategies
2. **Risk Management Guide**: Protect your capital
3. **API Documentation**: Automate your trading
4. **Video Tutorials**: Visual learning resources
5. **Community Forum**: Learn from other traders

## 🆘 Getting Help

### Self-Service Resources

1. **Check the [FAQ Database](faq_database.md)**
   - Most common questions answered
   - Quick solutions

2. **Use the [Troubleshooting Wizard](troubleshooting/troubleshooting_wizard.md)**
   - Interactive problem solving
   - Step-by-step guidance

3. **Review [Common Issues](troubleshooting/common_issues.md)**
   - Solutions for frequent problems
   - Error code reference

### Support Channels

- **Documentation**: Always start here
- **Community Forum**: User discussions
- **GitHub Issues**: Bug reports and feature requests
- **Discord**: Real-time community support

### Emergency Procedures

**If something goes wrong:**

1. **Stop all trading immediately**
   - Press `Ctrl+C` to shutdown
   - Review what happened

2. **Check system status**
   ```bash
   python health_check.py
   ```

3. **Review logs**
   ```bash
   tail -f logs/trading_orchestrator.log
   ```

4. **Contact support** with:
   - Error messages
   - Log files
   - Configuration details
   - Steps to reproduce

## 🎉 Congratulations!

You've completed the Getting Started Guide! You now have:

✅ **Basic understanding** of the trading system
✅ **Functional setup** with paper trading
✅ **Knowledge of the interface** and key features
✅ **Risk management basics** for safe trading
✅ **Support resources** for continued learning

**Ready for the next level?** 
- Continue with the [Complete User Manual](complete_user_manual.md)
- Explore the [Strategy Guide](strategy_guide.md)
- Master [Risk Management](risk_management_guide.md)

---

*Remember: Start slow, think carefully, and never risk more than you can afford to lose. Happy trading!*