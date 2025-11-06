# Usage Guide

## Getting Started

### System Overview
The Not Stonks Bot is an automated trading system that combines traditional algorithmic trading with AI-powered decision making. It supports multiple brokers, strategies, and risk management features.

### Quick Start

1. **Installation**
   ```bash
   git clone https://github.com/supermarsx/not-stonks-bot.git
   cd not-stonks-bot
   ./install.sh
   ```

2. **Configuration**
   ```bash
   cp config/config.template.json config/config.json
   # Edit config.json with your settings
   ```

3. **Testing**
   ```bash
   ./not-stonks-bot --test-mode --validate-config
   ```

4. **Start Trading**
   ```bash
   ./not-stonks-bot --start
   ```

## Basic Operations

### Starting the System

#### Command Line
```bash
# Start with default config
./not-stonks-bot start

# Start with custom config
./not-stonks-bot --config /path/to/config.json start

# Start in background
./not-stonks-bot start --daemon

# Start with specific log level
./not-stonks-bot start --log-level DEBUG
```

#### Service Management
```bash
# Start as service
sudo systemctl start not-stonks-bot

# Enable auto-start
sudo systemctl enable not-stonks-bot

# Check status
sudo systemctl status not-stonks-bot
```

### Stopping the System

#### Graceful Shutdown
```bash
# Stop trading gracefully
./not-stonks-bot stop

# Force stop
./not-stonks-bot stop --force
```

#### Emergency Stop
```bash
# Immediate stop with position closing
./not-stonks-bot emergency-stop

# Stop without closing positions
./not-stonks-bot emergency-stop --keep-positions
```

### System Status

#### Check Status
```bash
# General status
./not-stonks-bot status

# Detailed status
./not-stonks-bot status --verbose

# JSON output
./not-stonks-bot status --json
```

#### Health Check
```bash
# Basic health check
./not-stonks-bot health

# Comprehensive health check
./not-stonks-bot health --comprehensive

# Broker connectivity check
./not-stonks-bot health --brokers

# Database check
./not-stonks-bot health --database
```

## Trading Interface

### Portfolio Management

#### View Portfolio
```bash
# Current positions
./not-stonks-bot portfolio show

# Portfolio summary
./not-stonks-bot portfolio summary

# Historical performance
./not-stonks-bot portfolio performance

# Risk metrics
./not-stonks-bot portfolio risk
```

#### Manual Trades
```bash
# Place market order
./not-stonks-bot trade place --symbol AAPL --qty 10 --side buy --type market

# Place limit order
./not-stonks-bot trade place --symbol AAPL --qty 10 --side buy --type limit --price 150.00

# Cancel order
./not-stonks-bot trade cancel --order-id 12345

# Close position
./not-stonks-bot trade close --symbol AAPL
```

### Market Data

#### Real-time Data
```bash
# Get quote
./not-stonks-bot market quote AAPL

# Get multiple quotes
./not-stonks-bot market quotes AAPL,GOOGL,MSFT

# Get order book
./not-stonks-bot market orderbook AAPL

# Get recent trades
./not-stonks-bot market trades AAPL
```

#### Historical Data
```bash
# Get historical prices
./not-stonks-bot market history AAPL --period 1d --interval 1m

# Get price with custom range
./not-stonks-bot market history AAPL --start "2024-01-01" --end "2024-01-15"

# Export to CSV
./not-stonks-bot market history AAPL --output data.csv
```

### Order Management

#### View Orders
```bash
# Current orders
./not-stonks-bot orders list

# Orders by status
./not-stonks-bot orders list --status open

# Historical orders
./not-stonks-bot orders history --days 7

# Order details
./not-stonks-bot orders show 12345
```

#### Order Actions
```bash
# Modify order
./not-stonks-bot orders modify 12345 --price 149.50

# Cancel order
./not-stonks-bot orders cancel 12345

# Cancel all orders
./not-stonks-bot orders cancel-all

# Reject order
./not-stonks-bot orders reject 12345 --reason "Insufficient funds"
```

## Strategy Management

### Strategy Configuration

#### Create Strategy
```bash
# Create from template
./not-stonks-bot strategy create my_strategy --template moving_average

# Create custom strategy
./not-stonks-bot strategy create my_strategy --config strategy.json
```

#### Edit Strategy
```bash
# Edit strategy config
./not-stonks-bot strategy edit my_strategy

# Update parameters
./not-stonks-bot strategy update my_strategy --params '{"period": 20, "threshold": 0.02}'
```

#### Strategy Actions
```bash
# Start strategy
./not-stonks-bot strategy start my_strategy

# Stop strategy
./not-stonks-bot strategy stop my_strategy

# Pause strategy
./not-stonks-bot strategy pause my_strategy

# Resume strategy
./not-stonks-bot strategy resume my_strategy
```

### Strategy Monitoring

#### Strategy Status
```bash
# List all strategies
./not-stonks-bot strategy list

# Strategy details
./not-stonks-bot strategy show my_strategy

# Strategy performance
./not-stonks-bot strategy performance my_strategy

# Strategy logs
./not-stonks-bot strategy logs my_strategy --lines 100
```

#### Strategy Analytics
```bash
# Strategy performance report
./not-stonks-bot strategy report my_strategy

# Strategy comparison
./not-stonks-bot strategy compare strategy1,strategy2

# Strategy optimization
./not-stonks-bot strategy optimize my_strategy --iterations 1000
```

## Risk Management

### Position Management

#### View Positions
```bash
# All positions
./not-stonks-bot position list

# Position details
./not-stonks-bot position show AAPL

# Position history
./not-stonks-bot position history AAPL --days 30
```

#### Position Actions
```bash
# Close position
./not-stonks-bot position close AAPL

# Close multiple positions
./not-stonks-bot position close-by-filter --sector technology

# Rebalance portfolio
./not-stonks-bot position rebalance --target-weights file://portfolio.json
```

### Risk Limits

#### Configure Limits
```bash
# Set position limit
./not-stonks-bot risk set-limit position-size 10000

# Set daily loss limit
./not-stonks-bot risk set-limit daily-loss 5000

# Set exposure limit
./not-stonks-bot risk set-limit sector-exposure 0.30
```

#### Risk Monitoring
```bash
# Check current risk metrics
./not-stonks-bot risk check

# Risk report
./not-stonks-bot risk report

# Risk alerts
./not-stonks-bot risk alerts
```

### Risk Actions

#### Emergency Procedures
```bash
# Reduce all positions
./not-stonks-bot risk reduce-all --by 0.5

# Close losing positions
./not-stonks-bot risk close-losers --threshold 0.05

# Emergency hedge
./not-stonks-bot risk hedge --positions AAPL,GOOGL
```

## AI Features

### AI Analysis

#### Market Analysis
```bash
# AI market analysis
./not-stonks-bot ai analyze market AAPL

# Sentiment analysis
./not-stonks-bot ai sentiment AAPL

# Technical analysis
./not-stonks-bot ai technical AAPL

# Fundamental analysis
./not-stonks-bot ai fundamental AAPL
```

#### News Analysis
```bash
# Get AI news analysis
./not-stonks-bot ai news AAPL

# News sentiment
./not-stonks-bot ai news-sentiment AAPL

# Event impact analysis
./not-stonks-bot ai event-impact earnings --symbol AAPL
```

### AI Training

#### Model Training
```bash
# Train prediction model
./not-stonks-bot ai train predict --symbol AAPL --days 365

# Train sentiment model
./not-stonks-bot ai train sentiment --data news_data.csv

# Backtest AI model
./not-stonks-bot ai backtest model.pkl --start 2023-01-01
```

#### Model Management
```bash
# List models
./not-stonks-bot ai models list

# Load model
./not-stonks-bot ai models load model.pkl

# Save model
./not-stonks-bot ai models save model.pkl

# Model performance
./not-stonks-bot ai models performance model.pkl
```

## Advanced Features

### Automation

#### Scheduled Tasks
```bash
# Create scheduled task
./not-stonks-bot schedule create "Daily Rebalance" --cron "0 16 * * 1-5" --command "portfolio rebalance"

# List scheduled tasks
./not-stonks-bot schedule list

# Enable/disable task
./not-stonks-bot schedule disable "Daily Rebalance"
```

#### Event Triggers
```bash
# Create trigger
./not-stonks-bot trigger create price_alert --symbol AAPL --condition "price < 140" --action "close AAPL"

# List triggers
./not-stonks-bot trigger list

# Test trigger
./not-stonks-bot trigger test price_alert
```

### Integration

#### Webhook Setup
```bash
# Configure webhook
./not-stonks-bot webhook configure --url https://your-domain.com/webhook --secret secret123

# Test webhook
./not-stonks-bot webhook test

# Webhook logs
./not-stonks-bot webhook logs
```

#### API Usage
```bash
# Start API server
./not-stonks-bot api start --port 8080

# API health check
curl http://localhost:8080/health

# Get portfolio via API
curl http://localhost:8080/portfolio
```

## Monitoring and Alerts

### Logging

#### Log Management
```bash
# View real-time logs
./not-stonks-bot logs tail

# View specific log level
./not-stonks-bot logs --level ERROR

# Search logs
./not-stonks-bot logs search "order failed"

# Export logs
./not-stonks-bot logs export --start "2024-01-01" --end "2024-01-31" --output logs.zip
```

#### Log Configuration
```bash
# Set log level
./not-stonks-bot logs set-level DEBUG

# Configure log rotation
./not-stonks-bot logs configure --max-size 100MB --backup-count 5

# Add custom logger
./not-stonks-bot logs add-handler syslog --facility local0
```

### Alerts

#### Alert Configuration
```bash
# Create alert
./not-stonks-bot alert create high_loss --condition "portfolio_pnl < -0.05" --action email

# List alerts
./not-stonks-bot alert list

# Test alert
./not-stonks-bot alert test high_loss

# Disable alert
./not-stonks-bot alert disable high_loss
```

#### Notification Channels
```bash
# Configure email alerts
./not-stonks-bot notification email --smtp-server smtp.gmail.com --username user --password pass

# Configure Slack alerts
./not-stonks-bot notification slack --webhook-url https://hooks.slack.com/...

# Configure SMS alerts
./not-stonks-bot notification sms --provider twilio --account-sid xxx --auth-token xxx
```

## Daily Workflow

### Morning Routine

1. **System Check**
   ```bash
   ./not-stonks-bot health --comprehensive
   ```

2. **Portfolio Review**
   ```bash
   ./not-stonks-bot portfolio summary
   ./not-stonks-bot position list --unrealized
   ```

3. **Risk Check**
   ```bash
   ./not-stonks-bot risk check
   ./not-stonks-bot alert list
   ```

4. **Strategy Status**
   ```bash
   ./not-stonks-bot strategy list
   ./not-stonks-bot strategy performance --today
   ```

5. **Market Analysis**
   ```bash
   ./not-stonks-bot ai analyze market
   ./not-stonks-bot market overview
   ```

### During Trading Hours

1. **Monitor Positions**
   ```bash
   # Set up monitoring
   watch -n 60 './not-stonks-bot position list'
   ```

2. **Monitor Strategies**
   ```bash
   # Check strategy performance
   ./not-stonks-bot strategy performance --live
   ```

3. **Handle Alerts**
   ```bash
   # Check for alerts
   ./not-stonks-bot alert list
   ./not-stonks-bot logs --level WARNING --since 1h
   ```

### End of Day

1. **Daily Report**
   ```bash
   ./not-stonks-bot report daily
   ./not-stonks-bot performance summary --period 1d
   ```

2. **Position Review**
   ```bash
   ./not-stonks-bot position history --today
   ./not-stonks-bot orders history --today
   ```

3. **System Health**
   ```bash
   ./not-stonks-bot health --summary
   ./not-stonks-bot logs summary --today
   ```

4. **Data Backup**
   ```bash
   ./not-stonks-bot backup data
   ./not-stonks-bot export portfolio --format csv
   ```

## Best Practices

### Security
- Use strong passwords and API keys
- Enable two-factor authentication
- Regularly rotate credentials
- Monitor access logs
- Use VPN for remote access

### Performance
- Monitor system resources
- Optimize database queries
- Use caching where appropriate
- Regular maintenance tasks
- Performance profiling

### Risk Management
- Set appropriate position limits
- Regular risk assessments
- Diversification strategies
- Stop-loss mechanisms
- Regular strategy reviews

### Maintenance
- Regular system updates
- Database optimization
- Log rotation and cleanup
- Backup verification
- Disaster recovery testing

---

*For detailed documentation on specific features, see the individual guide files in docs/*