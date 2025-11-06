# 🚀 not-stonks-bot

<div align="center">

**AI-Powered Multi-Broker Trading Platform**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/supermarsx/not-stonks-bot.svg)](https://github.com/supermarsx/not-stonks-bot)
[![GitHub issues](https://img.shields.io/github/issues/supermarsx/not-stonks-bot)](https://github.com/supermarsx/not-stonks-bot/issues)

[⚡ Quick Start](#-quick-start) • [📚 Documentation](#-documentation) • [🛠️ Setup](#-setup) • [🤝 Contributing](#-contributing) • [❓ Support](#-support)

</div>

## 🎯 Overview

not-stonks-bot is a comprehensive, AI-powered trading platform that enables automated trading across multiple brokers. Built with a Matrix-themed terminal interface, it provides real-time market analysis, intelligent strategy execution, and enterprise-grade risk management.

### ✨ Key Features

- **🤖 AI-Powered Trading**: GPT-4 and Claude integration for market analysis and strategy selection
- **🔌 Multi-Broker Support**: Trade across 7 different brokers from a single interface
- **🛡️ Risk Management**: Advanced circuit breakers, position limits, and compliance controls
- **📊 Real-Time Dashboard**: Live performance metrics and P&L tracking
- **⚡ Sub-Second Execution**: Smart order routing and execution optimization
- **🎮 Demo Mode**: Practice trading with simulated data before going live

### 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    not-stonks-bot                          │
├─────────────────────────────────────────────────────────────┤
│  User Interface │  AI Engine      │  Risk Management       │
│  Matrix Terminal │  GPT-4 + Claude │  Circuit Breakers     │
│  Web Dashboard   │  Local Models   │  Position Limits      │
│  API Endpoints   │  Strategy AI    │  Compliance Checks    │
├─────────────────────────────────────────────────────────────┤
│  Trading Engine    │  Broker Layer     │  Data Layer         │
│  Strategy Exec     │  7 Broker APIs    │  Market Data        │
│  Order Management  │  Smart Routing    │  Historical Data    │
│  Portfolio Mgmt    │  Risk Validation  │  News Feeds         │
└─────────────────────────────────────────────────────────────┘
```

## ⚡ Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/supermarsx/not-stonks-bot.git
cd not-stonks-bot

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
```

### 2. Configuration

```bash
# Quick setup with auto-configuration
./start.sh setup  # Linux/macOS
start.bat setup   # Windows
```

### 3. Demo Mode

```bash
# Start in demo mode (recommended for first run)
python main.py --demo

# Or use quick scripts
./start.sh demo  # Linux/macOS
start.bat demo   # Windows
```

## 📚 Documentation

### Quick Links

- **[📖 Installation Guide](docs/getting-started/installation.md)** - Complete setup instructions
- **[🚀 Quick Start Tutorial](docs/getting-started/quick-start.md)** - Hands-on walkthrough
- **[⚙️ Configuration Guide](docs/getting-started/configuration.md)** - All configuration options
- **[🤖 AI Setup](docs/guides/ai-integration.md)** - Configure GPT-4, Claude, and local models
- **[🔌 Broker Setup](docs/guides/brokers.md)** - Setup guides for each supported broker
- **[🛡️ Risk Management](docs/guides/risk-management.md)** - Configure risk controls
- **[📊 API Reference](docs/api/)** - Complete API documentation

### Detailed Documentation

| Category | Description |
|----------|-------------|
| **[Getting Started](docs/getting-started/)** | Installation, setup, and first steps |
| **[Guides](docs/guides/)** | Detailed usage guides and tutorials |
| **[API Reference](docs/api/)** | Complete API documentation with examples |
| **[Development](docs/development/)** | Contributing guidelines and development setup |
| **[Architecture](docs/architecture/)** | System design and component documentation |

## 🛠️ Setup

### Prerequisites

- **Python 3.8+** (3.11+ recommended)
- **4GB+ RAM** (8GB+ recommended for optimal performance)
- **Internet connection** for market data and broker APIs
- **Broker accounts** (optional for demo mode)

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

### AI Integration

- **OpenAI GPT-4** - Market analysis and strategy generation
- **Anthropic Claude** - Risk assessment and compliance checks
- **Local Models** - Ollama, LM Studio, Transformers integration

## 🤖 AI Features

### Market Analysis
- **Real-time sentiment analysis** using news and social media
- **Technical pattern recognition** with advanced chart analysis
- **Strategy optimization** with AI-driven parameter tuning
- **Risk assessment** using multiple AI models

### Strategy Selection
- **Dynamic strategy selection** based on market conditions
- **Backtesting validation** with historical performance analysis
- **Performance attribution** with detailed breakdown

## 🛡️ Risk Management

### Circuit Breakers
- **Daily loss limits** with automatic trading halt
- **Consecutive loss protection** with smart recovery
- **Drawdown monitoring** with emergency stop mechanisms
- **Correlation analysis** to prevent over-concentration

### Compliance
- **Pattern Day Trader (PDT)** rule compliance
- **Wash sale** prevention for tax optimization
- **MiFID II** compliance for European markets
- **Audit trail** with complete trade documentation

## 📊 Monitoring & Analytics

### Real-Time Dashboard
- **Live P&L tracking** with detailed performance metrics
- **Strategy performance** analysis with individual attribution
- **Risk metrics** monitoring with threshold alerts
- **System health** monitoring with uptime tracking

### Alerts & Notifications
- **Risk threshold breaches** with immediate alerts
- **System errors** with detailed error reporting
- **Performance degradation** with proactive notifications
- **Trade confirmations** via multiple channels (Slack, email, SMS)

## 🧪 Testing

### Health Checks
```bash
# Full system health check
python health_check.py --full

# Component-specific tests
python health_check.py --brokers
python health_check.py --ai
python health_check.py --database
```

### Integration Testing
```bash
# Test all broker integrations
python test_integration.py

# Load testing
python test_integration.py --load-test

# Demo mode testing
python main.py --demo
```

## 🛠️ Development

### Development Setup
```bash
# Clone and setup development environment
python setup_dev.py

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Install pre-commit hooks
pre-commit install
```

### Project Structure
```
not-stonks-bot/
├── 📁 trading_orchestrator/     # Core trading system
├── 📁 trading-command-center/   # Web dashboard
├── 📁 crawlers/                 # Market data crawlers
├── 📁 analytics-backend/        # Analytics and reporting
├── 📁 tests/                    # Test suites
├── 📁 docs/                     # Documentation
├── 📁 scripts/                  # Utility and setup scripts
├── 📁 configs/                  # Configuration files
├── 📄 main.py                   # Main application entry
├── 📄 requirements.txt          # Core dependencies
└── 📄 pyproject.toml           # Project configuration
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/development/contributing.md) for details.

### How to Contribute

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Guidelines

- **Code Style**: Black for formatting, isort for imports, flake8 for linting
- **Testing**: Write tests for all new features and bug fixes
- **Documentation**: Update docs for any API changes
- **Type Safety**: Use type hints for all public interfaces

## ❓ Support

### Getting Help

- **📚 Documentation**: Check our comprehensive docs first
- **❓ GitHub Issues**: [Report bugs](https://github.com/supermarsx/not-stonks-bot/issues) or [request features](https://github.com/supermarsx/not-stonks-bot/issues)
- **💬 Discord**: Join our [Discord server](https://discord.gg/not-stonks-bot)
- **📧 Email**: Contact us at support@not-stonks-bot.com

### Common Issues

- **[Configuration Errors](docs/guides/troubleshooting.md#configuration-errors)** - Check config.json syntax and API keys
- **[Connection Issues](docs/guides/troubleshooting.md#connection-issues)** - Verify internet and broker credentials
- **[Performance Issues](docs/guides/troubleshooting.md#performance-issues)** - Check system resources and database
- **[AI Integration](docs/guides/troubleshooting.md#ai-integration)** - Verify API keys and model availability

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**This software is provided for educational and research purposes only. Trading involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results. You are responsible for your own trading decisions and should consult with a qualified financial advisor before making any investment decisions.**

## 🙏 Acknowledgments

- **OpenAI** - For providing access to GPT-4 API
- **Anthropic** - For Claude API access
- **TradingView** - For charting inspiration
- **Matrix** - For the aesthetic inspiration
- **Open Source Community** - For all the amazing libraries and tools

---

<div align="center">

**[⬆ Back to Top](#not-stonks-bot)**

Made with ❤️ and ☕ by the not-stonks-bot Team

</div>
