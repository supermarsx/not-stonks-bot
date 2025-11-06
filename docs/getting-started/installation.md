# Quick Installation Guide

Get not-stonks-bot running in under 5 minutes!

## 📋 Prerequisites

- **🐍 Python 3.8+** (3.11+ recommended)
- **💾 4GB+ RAM** (8GB+ recommended)
- **🌐 Internet connection**
- **💳 Broker account** (optional for demo mode)

## ⚡ Super Quick Start (3 Steps)

### 1. Clone & Setup
```bash
git clone https://github.com/supermarsx/not-stonks-bot.git
cd not-stonks-bot
```

### 2. Auto-Install
```bash
# Linux/macOS
chmod +x start.sh && ./start.sh setup

# Windows
start.bat setup
```

### 3. Start Demo
```bash
# Linux/macOS
./start.sh demo

# Windows
start.bat demo
```

## 🛠️ Manual Installation

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate environment
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
```bash
# Copy environment template
cp .env.example .env

# Create main config
cp configs/config.example.json config.json

# Edit with your API keys
nano .env
nano config.json
```

### 3. Health Check
```bash
python health_check.py --full
```

### 4. Start System
```bash
# Demo mode (recommended for first run)
python main.py --demo

# Or normal mode
python main.py
```

## 🎮 Demo Mode

Perfect for testing without real money:

```bash
# Start demo trading
python main.py --demo

# Or use the quick scripts
./start.sh demo  # Linux/macOS
start.bat demo   # Windows
```

## 📊 Dashboard (Optional)

For the full web interface:

```bash
cd trading-command-center
npm install
npm run dev
```

Then open: http://localhost:5173

## 🔧 Configuration

### Environment Variables (.env)
```bash
# Copy template
cp .env.example .env

# Key settings
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
```

### Broker Configuration (config.json)
```json
{
  "brokers": {
    "alpaca": {
      "enabled": true,
      "api_key": "YOUR_API_KEY",
      "secret_key": "YOUR_SECRET_KEY",
      "paper": true
    }
  }
}
```

## 🧪 Testing

### Health Check
```bash
# Full system check
python health_check.py --full

# Component tests
python health_check.py --brokers
python health_check.py --ai
python health_check.py --database
```

### Integration Test
```bash
# Test all integrations
python test_integration.py

# Load testing
python test_integration.py --load-test
```

## 🆘 Troubleshooting

### Common Issues

**🔑 API Key Errors**
```bash
# Check your .env file
cat .env

# Verify config.json
python -m json.tool config.json
```

**📊 Connection Issues**
```bash
# Test broker connectivity
python health_check.py --brokers

# Check internet connection
ping google.com
```

**🐌 Performance Issues**
```bash
# Check system resources
python health_check.py --performance

# Monitor memory usage
python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"
```

**🤖 AI Integration**
```bash
# Test AI connectivity
python health_check.py --ai

# Check API keys
echo $OPENAI_API_KEY
```

### Getting Help

1. **📚 Documentation**: Check [docs/](docs/) directory
2. **❓ GitHub Issues**: [Report bugs](https://github.com/supermarsx/not-stonks-bot/issues)
3. **💬 Discord**: Join our [Discord server](https://discord.gg/not-stonks-bot)
4. **📋 FAQ**: See [docs/guides/troubleshooting.md](docs/guides/troubleshooting.md)

## 🔄 Updates

### Update System
```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Update configuration if needed
cp .env.example .env
```

### Version Check
```bash
# Check current version
python main.py --version

# Check for updates
git fetch --tags
git describe --tags
```

## 📁 Project Structure

```
not-stonks-bot/
├── 📂 trading_orchestrator/     # Core trading system
├── 📂 trading-command-center/   # Web dashboard
├── 📂 crawlers/                 # Market data crawlers
├── 📂 tests/                    # Test suites
├── 📂 docs/                     # Documentation
├── 📂 scripts/                  # Utility and setup scripts
├── 📂 configs/                  # Configuration files
├── 📄 main.py                   # Main entry point
├── 📄 requirements.txt          # Dependencies
└── 📄 start.sh / start.bat      # Quick start scripts
```

## 🎯 Next Steps

1. **📚 Read the [User Manual](docs/guides/)**
2. **⚙️ Configure your [brokers](docs/guides/brokers.md)**
3. **🤖 Set up [AI integration](docs/guides/ai-integration.md)**
4. **🛡️ Configure [risk management](docs/guides/risk-management.md)**
5. **📊 Explore the [dashboard](trading-command-center/)**

---

<div align="center">

**🎉 Welcome to not-stonks-bot!**

[![GitHub stars](https://img.shields.io/github/stars/supermarsx/not-stonks-bot.svg?style=social)](https://github.com/supermarsx/not-stonks-bot)

</div>