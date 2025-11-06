# Usage Guide

This comprehensive guide covers how to use the Day Trading Orchestrator system effectively.

## Table of Contents

- [Getting Started](#getting-started)
- [Basic Operations](#basic-operations)
- [Trading Interface](#trading-interface)
- [Strategy Management](#strategy-management)
- [Risk Controls](#risk-controls)
- [Portfolio Management](#portfolio-management)
- [AI Features](#ai-features)
- [Advanced Features](#advanced-features)
- [Daily Workflow](#daily-workflow)
- [Best Practices](#best-practices)

## Getting Started

### First Launch

```bash
# Start the system
python main.py

# Or use launch scripts
./start.sh        # Linux/macOS
start.bat         # Windows
python run.py     # Cross-platform
```

### Interface Overview

When you first launch the system, you'll see the Matrix-themed terminal interface:

```
╔══════════════════════════════════════════════════════════════╗
║    ██████╗ ██╗   ██╗██████╗ ███╗   ███╗                    ║
║   ██╔═══██╗██║   ██║██╔══██╗████╗ ████║                    ║
║   ██║   ██║██║   ██║██████╔╝██╔████╔██║                    ║
║   ██║   ██║██║   ██║██╔══██╗██║╚██╔╝██║                    ║
║   ╚██████╔╝╚██████╔╝██████╔╝██║ ╚═╝ ██║                    ║
║    ╚═════╝  ╚═════╝ ╚═════╝ ╚═╝     ╚═╝                    ║
║                                                              ║
║           DAY TRADING ORCHESTRATOR SYSTEM                    ║
║                 Multi-Broker AI Trading                     ║
╚══════════════════════════════════════════════════════════════╝

🚀 Initializing Day Trading Orchestrator...
📋 Configuration loaded from config.json
💾 Database initialized
🔌 Broker factory initialized with 3 brokers
🤖 AI models initialized
🛠️ Trading tools initialized
🧠 AI orchestrator initialized
⚡ Risk engine initialized
📋 OMS initialized
📈 Strategy registry initialized
📊 Market data manager initialized
💻 Matrix terminal interface initialized
✅ System initialization complete!

🏁 Starting Day Trading Orchestrator...

Welcome to the Matrix Terminal
Type 'help' for available commands
Type 'status' for system status
Type 'demo' to run demo mode
```

### Command Palette

The system provides a command palette for quick access to all features:

```bash
# Show command palette
Ctrl+P

# Or type 'help' in the terminal
> help
```

Available commands:
- `status` - Show system status
- `connect <broker>` - Connect to broker
- `symbols` - List available symbols
- `strategies` - Manage trading strategies
- `portfolio` - View portfolio
- `orders` - Manage orders
- `risk` - Configure risk limits
- `ai` - AI-powered features
- `demo` - Run demo mode
- `quit` - Exit system

## Basic Operations

### Connecting to Brokers

```bash
# Connect to specific broker
> connect alpaca

# Connect to all enabled brokers
> connect all

# Check broker status
> status brokers
```

### Viewing Market Data

```bash
# View real-time quotes
> quote AAPL
> quote BTC-USD

# View price history
> history AAPL --period 1d --interval 1h

# View market overview
> markets
```

### Basic Order Placement

```bash
# Place market order
> order buy AAPL 100 market

# Place limit order
> order buy AAPL 100 limit 150.50

# Place stop loss order
> order sell AAPL 100 stop 145.00

# View open orders
> orders
```

### Position Management

```bash
# View current positions
> positions

# View position details
> position AAPL

# Close position
> close AAPL

# Close all positions
> close all
```

## Trading Interface

### Real-Time Dashboard

The Matrix interface provides a real-time trading dashboard:

```
┌─ Portfolio Status ─┐┌─ Market Data ─────┐┌─ Active Strategies ─┐
│ Cash: $50,000     ││ AAPL:  $150.25↑   ││ Mean Reversion     │
│ Equity: $75,000   ││ BTC:   $45,200↓   ││ Trend Following    │
│ P&L:   +$2,500    ││ TSLA:  $220.80↑   ││ Pairs Trading      │
│ Margin: $0        ││ SPY:   $425.30→   ││                    │
└───────────────────┘└───────────────────┘└────────────────────┘

┌─ Risk Metrics ─────┐┌─ Active Orders ───┐┌─ System Health ─────┐
│ VaR:    1.2%       ││ AAPL BUY 50 MKT  ││ DB:    ✓ Healthy   │
│ Heat:   15%        ││ BTC SELL 0.5 LTC ││ AI:    ✓ Active    │
│ Drawdown: 2.5%     ││ TSLA BUY 20 LMT  ││ Risk:  ✓ Good      │
│ Sharpe: 1.8        ││                  ││ Brokers: ✓ 3/3     │
└───────────────────┘└───────────────────┘└────────────────────┘
```

### Interactive Charts

```bash
# Open chart for symbol
> chart AAPL

# Chart with indicators
> chart AAPL --indicators RSI,MA20,MA50

# Multiple symbols comparison
> chart AAPL,MSFT,GOOGL --relative
```

### Order Book

```bash
# View order book
> book AAPL

# View with market depth
> book AAPL --depth 10

# View paper trading order book
> book AAPL --paper
```

### Trade Log

```bash
# View recent trades
> trades

# View trades for specific symbol
> trades AAPL --limit 50

# Export trade history
> trades --export csv --file trades_2024.csv
```

## Strategy Management

### Strategy Selection

```bash
# List available strategies
> strategies list

# Enable strategy
> strategy enable mean_reversion

# Disable strategy
> strategy disable pairs_trading

# Configure strategy parameters
> strategy config mean_reversion --lookback 20 --threshold 2.0
```

### Strategy Execution

```bash
# Start strategy
> strategy start mean_reversion

# Stop strategy
> strategy stop mean_reversion

# View strategy status
> strategy status

# View strategy performance
> strategy performance mean_reversion --period 1d
```

### Strategy Types

#### Mean Reversion Strategy

```bash
# Configure mean reversion
> strategy config mean_reversion \
    --lookback 20 \
    --entry_threshold 2.0 \
    --exit_threshold 0.5 \
    --min_confidence 0.75

# Start mean reversion trading
> strategy start mean_reversion
```

#### Trend Following Strategy

```bash
# Configure trend following
> strategy config trend_following \
    --fast_ma 10 \
    --slow_ma 30 \
    --rsi_period 14 \
    --min_trend_strength 0.6

# Start trend following
> strategy start trend_following
```

#### Pairs Trading Strategy

```bash
# Configure pairs trading
> strategy config pairs_trading \
    --correlation_threshold 0.8 \
    --zscore_entry 2.0 \
    --zscore_exit 0.5

# Start pairs trading
> strategy start pairs_trading
```

### Backtesting Strategies

```bash
# Backtest strategy
> backtest mean_reversion --start 2023-01-01 --end 2024-01-01

# Compare strategies
> backtest compare mean_reversion trend_following --period 6m

# View backtest results
> backtest results --strategy mean_reversion --detailed
```

## Risk Controls

### Position Limits

```bash
# Set maximum position size
> risk limit position 10000

# Set daily loss limit
> risk limit daily_loss 5000

# Set maximum number of open orders
> risk limit open_orders 100

# View current limits
> risk limits
```

### Circuit Breakers

```bash
# Enable circuit breakers
> risk circuit enable

# Set daily loss circuit breaker
> risk circuit daily_loss 10000

# Set consecutive loss circuit breaker
> risk circuit consecutive_losses 3

# View circuit breaker status
> risk circuit status
```

### Risk Monitoring

```bash
# View portfolio risk metrics
> risk metrics

# View correlation matrix
> risk correlation

# View sector exposure
> risk sectors

# View risk heatmap
> risk heatmap
```

### Compliance Controls

```bash
# Check pattern day trader status
> compliance pdt

# Verify wash sale compliance
> compliance wash_sale

# Check margin requirements
> compliance margin

# Generate compliance report
> compliance report --format pdf
```

## Portfolio Management

### Portfolio Overview

```bash
# View portfolio summary
> portfolio

# View detailed positions
> portfolio details

# View allocation by asset class
> portfolio allocation

# View performance metrics
> portfolio performance --period 1m
```

### Asset Allocation

```bash
# View current allocation
> allocation

# Rebalance portfolio
> rebalance --target stocks:60% bonds:30% cash:10%

# View sector allocation
> allocation sectors

# View geographic allocation
> allocation regions
```

### Performance Analysis

```bash
# Calculate returns
> returns --period 1y

# View Sharpe ratio
> performance sharpe

# View maximum drawdown
> performance drawdown

# Compare to benchmark
> performance benchmark SPY
```

### Portfolio Optimization

```bash
# Optimize portfolio weights
> optimize --method modern_portfolio_theory

# View efficient frontier
> optimize frontier

# Rebalance based on optimization
> optimize rebalance
```

## AI Features

### Market Analysis

```bash
# Analyze market conditions
> ai analyze AAPL

# Get market sentiment
> ai sentiment AAPL --source news,social

# Technical pattern recognition
> ai patterns AAPL

# Fibonacci levels
> ai fibonacci AAPL
```

### AI Strategy Recommendations

```bash
# Get strategy recommendation
> ai recommend AAPL --confidence_threshold 0.8

# Market regime analysis
> ai regime

# Risk assessment
> ai risk_assessment AAPL

# Portfolio optimization suggestion
> ai optimize_portfolio
```

### News Analysis

```bash
# Analyze news for symbol
> ai news AAPL

# Sentiment analysis
> ai sentiment news AAPL

# Event impact analysis
> ai events AAPL --period 1d

# Earnings announcement analysis
> ai earnings AAPL
```

### Custom AI Prompts

```bash
# Use custom prompt
> ai prompt "Analyze the technical outlook for AAPL based on current market conditions"

# Save custom prompt
> ai prompt save my_analysis "Analyze market volatility"

# Use saved prompt
> ai prompt use my_analysis
```

## Advanced Features

### Order Types

#### Advanced Order Types

```bash
# One-Cancels-the-Other (OCO)
> order oco AAPL \
    --buy_limit 148.00 \
    --sell_stop 145.00

# Bracket Order
> order bracket AAPL \
    --parent_buy 100 \
    --child_stop 145.00 \
    --child_take_profit 155.00

# Trailing Stop
> order sell 50 AAPL \
    --type trailing_stop \
    --trail_amount 5.00

# Iceberg Order
> order buy 1000 AAPL \
    --type iceberg \
    --display_quantity 100
```

#### Conditional Orders

```bash
# If-Then Order
> order if AAPL > 150 then buy AAPL 100 market

# Time-Based Order
> order buy AAPL 100 at_time "2024-01-01 09:30:00"

# Volume-Based Order
> order buy AAPL 100 if_volume > 1000000
```

### Options Trading

```bash
# View options chains
> options chain AAPL --expiration 2024-03-15

# Buy call option
> options call AAPL 150 2024-03-15 buy 1

# Buy put option
> options put AAPL 155 2024-03-15 buy 1

# Complex options strategy
> options straddle AAPL 150 2024-03-15
```

### Futures Trading

```bash
# View futures contracts
> futures contracts ES

# Buy E-mini S&P 500
> futures buy ES 2024-03-15 2

# Set margin requirements
> futures margin ES 5000

# View contract specifications
> futures specs ES 2024-03-15
```

### Forex Trading

```bash# View currency pairs
> forex pairs

# Buy EUR/USD
> forex buy EURUSD 10000

# Set stop loss
> forex stop_loss EURUSD 1.0850

# View major pairs
> forex majors

# Cross currency analysis
> forex cross EURGBP
```

### Cryptocurrency Trading

```bash
# View crypto pairs
> crypto pairs

# Buy Bitcoin
> crypto buy BTCUSDT 0.5

# Set take profit
> crypto take_profit BTCUSDT 50000

# View altcoins
> crypto altcoins --market_cap 1000000000

# DeFi analysis
> crypto defi analyze
```

## Daily Workflow

### Pre-Market Routine

```bash
# Check market status
> market status

# Review overnight news
> news overnight

# Update risk limits
> risk limits review

# Check economic calendar
> calendar today

# Pre-market analysis
> ai analysis --market_brief
```

### Market Hours

```bash
# Monitor positions
> monitor positions

# Watch key levels
> watch AAPL,TSLA,SPY

# Review AI signals
> ai signals --active

# Check risk metrics
> risk check

# Monitor news
> news watch --symbols AAPL
```

### End-of-Day Routine

```bash
# Close positions
> close --time market_close

# Generate reports
> report daily --format pdf

# Backup data
> backup data

# Update strategy parameters
> strategy update

# Review performance
> performance summary --period 1d
```

### Risk Review

```bash
# Daily risk report
> risk report --daily

# P&L analysis
> pnl analysis

# Position concentration
> risk concentration

# Margin utilization
> margin usage

# Compliance check
> compliance check
```

## Best Practices

### Security

1. **API Key Management:**
   - Use environment variables for sensitive data
   - Rotate API keys regularly
   - Use paper trading for testing
   - Enable 2FA on all broker accounts

2. **System Security:**
   - Keep system updated
   - Use secure connections
   - Monitor access logs
   - Regular security audits

### Risk Management

1. **Position Sizing:**
   - Never risk more than 2% per trade
   - Use position size calculators
   - Monitor correlation between positions
   - Diversify across asset classes

2. **Daily Limits:**
   - Set maximum daily loss limits
   - Monitor drawdown closely
   - Use circuit breakers
   - Regular risk reviews

### Strategy Management

1. **Backtesting:**
   - Always backtest strategies
   - Use out-of-sample testing
   - Monitor live vs backtest performance
   - Regular strategy review

2. **Performance Tracking:**
   - Track all strategies separately
   - Monitor Sharpe ratios
   - Compare to benchmarks
   - Regular performance attribution

### System Monitoring

1. **Health Checks:**
   - Run daily health checks
   - Monitor system resources
   - Check broker connectivity
   - Verify data feeds

2. **Alert Setup:**
   - Configure risk alerts
   - Set up email notifications
   - Use system monitoring
   - Emergency contact list

## Keyboard Shortcuts

### Global Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+C` | Exit system |
| `Ctrl+P` | Command palette |
| `Ctrl+Z` | Undo last action |
| `F1` | Show help |
| `F2` | System status |
| `F3` | Quick order |
| `F4` | Portfolio view |
| `F5` | Refresh all data |
| `F10` | Settings menu |

### Trading Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+B` | Buy order |
| `Ctrl+S` | Sell order |
| `Ctrl+O` | Open orders |
| `Ctrl+L` | Close position |
| `Ctrl+I` | Position info |
| `Ctrl+M` | Market data |
| `Ctrl+T` | Trading strategies |
| `Ctrl+R` | Risk controls |

## Troubleshooting Common Issues

### Connection Issues

```bash
# Check broker connectivity
> status brokers

# Test connection
> test broker alpaca

# Restart connection
> restart broker alpaca

# View connection logs
> logs connection --last 1h
```

### Order Execution Issues

```bash
# Check order status
> order status --id ORDER_ID

# View execution logs
> logs orders --symbol AAPL

# Cancel and retry
> order cancel --id ORDER_ID
> order retry --id ORDER_ID
```

### AI Integration Issues

```bash
# Test AI connectivity
> ai test

# Check API limits
> ai limits

# Reset AI session
> ai reset

# View AI logs
> logs ai --last 1h
```

## Getting Help

### Built-in Help

```bash
# General help
> help

# Command-specific help
> help order
> help strategy
> help risk

# Tutorial mode
> tutorial
```

### Support Resources

- **[Documentation](docs/)** - Comprehensive documentation
- **[GitHub Issues](https://github.com/your-username/day-trading-orchestrator/issues)** - Bug reports
- **[Community Forum](https://forum.trading-orchestrator.com)** - Discussions
- **[Discord](https://discord.gg/trading-orchestrator)** - Real-time support
- **[Email Support](mailto:support@trading-orchestrator.com)** - Technical support

## Next Steps

After mastering the basics:

1. **Advanced Strategies** - Develop custom strategies
2. **API Integration** - Build custom integrations
3. **Portfolio Management** - Advanced portfolio techniques
4. **Risk Analytics** - Sophisticated risk models
5. **Machine Learning** - Custom ML models

Remember: Always start with paper trading and small position sizes when testing new strategies or features!