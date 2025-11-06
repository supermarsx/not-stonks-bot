# Day Trading Orchestrator

<div align="center">

![Day Trading Orchestrator](https://img.shields.io/badge/Day%20Trading%20Orchestrator-v1.0.0-blue?style=for-the-badge&logo=python)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Matrix-Themed Multi-Broker AI Trading System**

[🚀 Quick Start](#-quick-start) • [📚 Documentation](#-documentation) • [⚙️ Configuration](#-configuration) • [🛠️ API Reference](#-api-reference) • [❓ Support](#-support)

</div>

## 📋 Table of Contents

- [Overview](#-overview)
- [✨ Features](#-features)
- [🏗️ Architecture](#-architecture)
- [🚀 Quick Start](#-quick-start)
- [📚 Documentation](#-documentation)
- [⚙️ Configuration](#-configuration)
- [🔌 Broker Integration](#-broker-integration)
- [🤖 AI Features](#-ai-features)
- [🛡️ Risk Management](#-risk-management)
- [🛠️ API Reference](#-api-reference)
- [❓ Support](#-support)
- [📝 License](#-license)

## 🎯 Overview

The Day Trading Orchestrator is a comprehensive, AI-powered trading system designed for serious traders who need enterprise-grade risk management and multi-broker integration. Built with a Matrix-themed terminal interface, it provides real-time market analysis, automated strategy execution, and intelligent risk controls.

### 🎬 Why This System?

- **Multi-Broker Support**: Trade across 7 different brokers from a single interface
- **AI-Powered Decisions**: Leverage GPT-4 and Claude for market analysis and strategy selection
- **Enterprise Risk Management**: Advanced circuit breakers and position limits
- **Real-Time Execution**: Sub-second order execution with smart routing
- **Matrix Aesthetics**: Stunning terminal interface with real-time visualizations

## ✨ Features

### 🔌 Broker Integration
- **Alpaca Trading** - US Stocks & Crypto with commission-free trading
- **Binance** - Global crypto trading with advanced order types
- **Interactive Brokers** - Global markets access with professional tools
- **Trading 212** - European stocks with competitive fees
- **DEGIRO** - European broker (unofficial API integration)
- **XTB** - Forex & CFDs with advanced charting
- **Trade Republic** - German broker (unofficial API integration)

### 🤖 AI-Powered Features
- **Market Analysis** - GPT-4 powered market sentiment analysis
- **Strategy Selection** - AI-driven strategy selection based on market conditions
- **Risk Assessment** - Real-time risk evaluation using Claude
- **Pattern Recognition** - Advanced technical pattern detection
- **News Analysis** - Real-time news sentiment analysis
- **Portfolio Optimization** - AI-assisted portfolio rebalancing

### 🛡️ Risk Management
- **Circuit Breakers** - Automatic trading halt on significant losses
- **Position Limits** - Configurable maximum position sizes
- **Correlation Analysis** - Prevent over-concentration in correlated assets
- **Real-Time Monitoring** - 24/7 system health monitoring
- **Compliance Controls** - Built-in compliance and regulatory checks
- **Audit Trail** - Complete trade execution audit trail

### 📊 Analytics & Reporting
- **Real-Time Dashboard** - Live performance metrics and P&L
- **Strategy Performance** - Individual strategy analytics
- **Risk Metrics** - Comprehensive risk analysis reports
- **Trade History** - Detailed trade execution logs
- **Backtesting Engine** - Historical strategy validation
- **Performance Attribution** - Detailed performance breakdown

### 🖥️ User Interface
- **Matrix Terminal** - Stunning terminal-based trading interface
- **Real-Time Charts** - Live market data visualization
- **Command Palette** - Quick access to all trading functions
- **Hotkeys** - Keyboard shortcuts for rapid execution
- **Customizable Layout** - Personalized interface configuration
- **Dark Theme** - Eye-friendly dark interface

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Day Trading Orchestrator                 │
├─────────────────────────────────────────────────────────────┤
│  UI Layer              │  AI Layer          │  Risk Layer    │
│  ┌─────────────────┐   │  ┌─────────────┐   │  ┌───────────┐ │
│  │ Matrix Terminal │   │  │ GPT-4      │   │  │ Circuit   │ │
│  │ Real-time Charts│   │  │ Claude     │   │  │ Breakers  │ │
│  │ Command Palette │   │  │ Local LLM  │   │  │ Position  │ │
│  └─────────────────┘   │  └─────────────┘   │  │ Limits    │ │
│                        │                    │  │ Compliance│ │
│  Strategy Layer        │  Broker Layer      │  └───────────┘ │
│  ┌─────────────────┐   │  ┌─────────────┐   │               │
│  │ Mean Reversion  │   │  │ Alpaca     │   │               │
│  │ Trend Following │   │  │ Binance    │   │               │
│  │ Pairs Trading   │   │  │ IBKR       │   │               │
│  │ Arbitrage       │   │  │ ...        │   │               │
│  └─────────────────┘   │  └─────────────┘   │               │
│                        │                    │               │
│  Data Layer            │  OMS Layer         │               │
│  ┌─────────────────┐   │  ┌─────────────┐   │               │
│  │ Market Data     │   │  │ Order Mgmt  │   │               │
│  │ Historical Data │   │  │ Trade Exec  │   │               │
│  │ News Feeds      │   │  │ Settlement  │   │               │
│  │ Options Data    │   │  │ Reconciliation│ │               │
│  └─────────────────┘   │  └─────────────┘   │               │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** - Required for all core functionality
- **4GB RAM minimum** - 8GB+ recommended for optimal performance
- **Internet connection** - Required for market data and broker APIs
- **Broker accounts** - At least one supported broker account

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/your-username/day-trading-orchestrator.git
cd day-trading-orchestrator
```

#### 2. Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r trading_orchestrator/requirements.txt
```

#### 3. Quick Launch Scripts

**Linux/macOS:**
```bash
# Make executable (Linux/macOS)
chmod +x start.sh

# Run normal mode
./start.sh

# Run demo mode
./start.sh demo

# Create default config
./start.sh create-config
```

**Windows:**
```cmd
# Run normal mode
start.bat

# Run demo mode
start.bat demo

# Create default config
start.bat create-config
```

**Python (cross-platform):**
```bash
# Run directly with Python
python run.py

# Run with main module
python main.py

# Create configuration
python main.py --create-config

# Run demo
python main.py --demo
```

### First Time Setup

1. **Run Configuration Setup**:
   ```bash
   python main.py --create-config
   ```

2. **Configure Your Brokers**:
   - Edit `config.json` with your broker API keys
   - Start with paper trading enabled
   - Set appropriate risk limits

3. **Start in Demo Mode**:
   ```bash
   python main.py --demo
   ```

4. **Verify Installation**:
   ```bash
   python health_check.py
   ```

## 📚 Documentation

### Core Documentation
- **[Installation Guide](docs/installation.md)** - Detailed installation instructions
- **[Usage Guide](docs/usage.md)** - How to use the system effectively
- **[Broker Integration](docs/brokers.md)** - Detailed broker setup guides
- **[Troubleshooting](docs/troubleshooting.md)** - Common issues and solutions

### Advanced Topics
- **[AI Integration](docs/ai.md)** - AI model configuration and usage
- **[Risk Management](docs/risk.md)** - Advanced risk controls and settings
- **[Strategy Development](docs/strategies.md)** - Creating custom trading strategies
- **[API Reference](docs/api.md)** - Complete API documentation
- **[Development Guide](docs/development.md)** - Contributing and extending the system

### Configuration Examples
- **[Alpaca Configuration](config.alpaca.example.json)** - Alpaca setup example
- **[Binance Configuration](config.binance.example.json)** - Binance setup example
- **[IBKR Configuration](config.ibkr.example.json)** - Interactive Brokers example

## ⚙️ Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# Copy the example
cp .env.example .env

# Edit with your settings
nano .env
```

### Main Configuration

Create `config.json` based on `config.example.json`:

```json
{
  "database": {
    "url": "sqlite:///trading_orchestrator.db"
  },
  "brokers": {
    "alpaca": {
      "enabled": true,
      "api_key": "YOUR_ALPACA_API_KEY",
      "secret_key": "YOUR_ALPACA_SECRET_KEY",
      "paper": true
    }
  },
  "risk": {
    "max_position_size": 10000,
    "max_daily_loss": 5000
  }
}
```

### Security Configuration

**⚠️ IMPORTANT SECURITY NOTES:**

1. **API Keys**: Never commit API keys to version control
2. **Permissions**: Use least-privilege API permissions
3. **Environment**: Use separate API keys for paper/live trading
4. **Rotation**: Regularly rotate API keys
5. **Monitoring**: Enable audit logging for all activities

## 🔌 Broker Integration

### Supported Brokers

| Broker | Markets | Commission | Paper Trading | Live Trading |
|--------|---------|------------|---------------|--------------|
| **Alpaca** | US Stocks, Crypto | $0 | ✅ | ✅ |
| **Binance** | Crypto | 0.1% | ✅ | ✅ |
| **Interactive Brokers** | Global Markets | Varies | ✅ | ✅ |
| **Trading 212** | EU Stocks | €0 | ✅ | ✅ |
| **DEGIRO** | EU Stocks | Varies | ✅ | ✅ |
| **XTB** | Forex, CFDs | Varies | ✅ | ✅ |
| **Trade Republic** | German Stocks | €0 | ✅ | ✅ |

### Broker Setup Guides

1. **Alpaca Trading**:
   - Sign up at [alpaca.markets](https://alpaca.markets)
   - Generate API keys in your dashboard
   - Start with paper trading for testing

2. **Binance**:
   - Create account at [binance.com](https://binance.com)
   - Enable API access in account settings
   - Use testnet for development: [testnet.binance.vision](https://testnet.binance.vision)

3. **Interactive Brokers**:
   - Download TWS or IBKR Gateway
   - Configure API settings in TWS
   - Use paper trading account for testing

## 🤖 AI Features

### Supported Models

- **OpenAI GPT-4** - Market analysis and strategy generation
- **Anthropic Claude** - Risk assessment and compliance checks
- **Local Models** - Ollama, LM Studio, Transformers integration

### AI Capabilities

- **Market Sentiment Analysis** - Real-time news and social media sentiment
- **Technical Pattern Recognition** - Advanced chart pattern detection
- **Strategy Optimization** - AI-driven parameter optimization
- **Risk Assessment** - Real-time risk scoring using multiple models
- **Portfolio Rebalancing** - Intelligent portfolio optimization

### Configuration

```json
{
  "ai": {
    "trading_mode": "PAPER",
    "openai_api_key": "YOUR_OPENAI_API_KEY",
    "anthropic_api_key": "YOUR_ANTHROPIC_API_KEY",
    "local_models": {
      "enabled": false,
      "model_path": "./models/local"
    }
  }
}
```

## 🛡️ Risk Management

### Circuit Breakers

- **Daily Loss Limit** - Automatic trading halt on daily loss threshold
- **Consecutive Losses** - Stop trading after N consecutive losses
- **Drawdown Protection** - Emergency stop on excessive drawdown
- **Correlation Limits** - Prevent over-concentration in correlated assets

### Position Controls

- **Maximum Position Size** - Limit individual position exposure
- **Portfolio Heat** - Total portfolio risk percentage
- **Sector Limits** - Maximum exposure to single sectors
- **Geographic Limits** - Regional exposure controls

### Compliance Features

- **Pattern Day Trader** - PDT rule compliance (US markets)
- **Wash Sale** - Wash sale rule prevention (US markets)
- **Regulation T** - Margin and credit controls
- **MiFID II** - European regulatory compliance

## 🛠️ API Reference

### REST API

```bash
# Health check
GET /health

# System status
GET /api/status

# Broker connections
GET /api/brokers

# Create order
POST /api/orders
{
  "symbol": "AAPL",
  "side": "buy",
  "quantity": 10,
  "order_type": "market"
}

# Get positions
GET /api/positions

# Get portfolio
GET /api/portfolio
```

### WebSocket API

```javascript
// Connect to real-time data
const ws = new WebSocket('ws://localhost:8000/ws');

// Subscribe to market data
ws.send(JSON.stringify({
  action: 'subscribe',
  symbol: 'AAPL',
  data_type: 'price'
}));
```

## 🧪 Testing

### Health Check

```bash
# Run system health check
python health_check.py

# Check specific components
python health_check.py --broker alpaca
python health_check.py --database
python health_check.py --ai
```

### Integration Testing

```bash
# Run integration tests
python test_integration.py

# Test specific broker
python test_integration.py --broker alpaca --paper

# Run demo mode
python demo.py
```

### Performance Testing

```bash
# Benchmark system performance
python benchmark.py

# Load test order execution
python load_test.py --orders 1000
```

## 📊 Monitoring

### Performance Metrics

- **Order Execution Time** - Sub-second order execution tracking
- **System Uptime** - 99.9%+ uptime monitoring
- **API Response Times** - Real-time latency tracking
- **Memory Usage** - Resource utilization monitoring

### Alerts

- **Risk Threshold Breaches** - Automated risk alerts
- **System Errors** - Immediate error notifications
- **Performance Degradation** - Proactive performance alerts
- **Compliance Violations** - Regulatory compliance monitoring

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-username/day-trading-orchestrator.git

# Create development environment
python setup_dev.py

# Run tests
python -m pytest tests/

# Install pre-commit hooks
pre-commit install
```

### Code Style

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

## ❓ Support

### Getting Help

1. **Documentation**: Check our comprehensive docs first
2. **GitHub Issues**: Report bugs or request features
3. **Discussions**: Join our community discussions
4. **Discord**: Real-time support in our Discord server

### Common Issues

- **Configuration Errors**: Check `config.json` syntax and broker API keys
- **Connection Issues**: Verify internet connection and broker credentials
- **Performance Issues**: Check system resources and database performance
- **AI Integration**: Verify API keys and model availability

### Resources

- **[FAQ](docs/faq.md)** - Frequently Asked Questions
- **[Video Tutorials](https://youtube.com/trading-orchestrator)** - Step-by-step guides
- **[Blog](https://blog.trading-orchestrator.com)** - Latest updates and tutorials
- **[Community Forum](https://forum.trading-orchestrator.com)** - Community discussions

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** - For providing access to GPT-4 API
- **Anthropic** - For Claude API access
- **TradingView** - For charting inspiration
- **Matrix** - For the aesthetic inspiration
- **Open Source Community** - For all the amazing libraries and tools

## ⚠️ Disclaimer

**This software is provided for educational and research purposes only. Trading involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results. You are responsible for your own trading decisions and should consult with a qualified financial advisor before making any investment decisions.**

The authors and contributors of this software are not responsible for any financial losses incurred through the use of this software.

---

<div align="center">

**[⬆ Back to Top](#day-trading-orchestrator)**

Made with ❤️ and ☕ by the Trading Orchestrator Team

</div>