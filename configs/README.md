# Configuration Documentation

Complete guide to configuring not-stonks-bot for your trading needs.

## 📁 Configuration Structure

The configuration system is organized as follows:

```
configs/
├── config.example.json          # Main configuration template
├── config.template.json         # Alternative template
├── brokers/                     # Broker-specific configurations
│   ├── alpaca.json
│   ├── binance.json
│   └── ibkr.json
├── environments/                # Environment-specific configurations
│   ├── development.json
│   ├── testing.json
│   └── production.json
└── examples/                    # Example configurations
    ├── simple_config.json
    └── full_config.json
```

## 🔧 Configuration Loading

The system loads configuration in this order:

1. **Base template** → `config.example.json`
2. **User config** → `config.json` (if exists)
3. **Environment config** → `environments/{env}.json`
4. **Broker config** → `brokers/{broker}.json`
5. **Environment variables** → Override any config values

## 🎯 Quick Start

1. **Copy the template**:
   ```bash
   cp configs/config.example.json config.json
   ```

2. **Update your settings** in `config.json`

3. **Set environment variables** for sensitive data:
   ```bash
   export ALPACA_API_KEY="your_key_here"
   export OPENAI_API_KEY="your_key_here"
   ```

## 📊 Configuration Sections

### Database
```json
{
  "database": {
    "url": "sqlite:///not_stonks_bot.db",
    "echo": false
  }
}
```

### AI Configuration
```json
{
  "ai": {
    "trading_mode": "PAPER",
    "default_model_tier": "fast",
    "openai_api_key": null,
    "anthropic_api_key": null
  }
}
```

### Broker Configuration
```json
{
  "brokers": {
    "alpaca": {
      "enabled": true,
      "api_key": null,
      "secret_key": null,
      "paper": true
    }
  }
}
```

### Risk Management
```json
{
  "risk": {
    "max_position_size": 10000,
    "max_daily_loss": 5000,
    "circuit_breaker": {
      "enabled": true,
      "daily_loss_limit": 10000
    }
  }
}
```

## 🔐 Security Best Practices

### API Keys
- **Never hardcode** API keys in configuration files
- **Use environment variables** for all sensitive data
- **Implement key rotation** regularly
- **Use different keys** for paper and live trading

### Example Environment Setup
```bash
# .env file
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
BINANCE_API_KEY=your_binance_key
BINANCE_SECRET_KEY=your_binance_secret
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
```

### Loading Environment Variables
```python
# In your Python code
import os
from decouple import config

api_key = config('ALPACA_API_KEY', default=None)
```

## 🎮 Trading Modes

### Paper Trading (Recommended for Testing)
```json
{
  "ai": {
    "trading_mode": "PAPER"
  },
  "brokers": {
    "alpaca": {
      "paper": true
    }
  }
}
```

### Live Trading
```json
{
  "ai": {
    "trading_mode": "LIVE"
  },
  "brokers": {
    "alpaca": {
      "paper": false
    }
  }
}
```

## 🔧 Environment-Specific Configurations

### Development
```json
{
  "database": {
    "url": "sqlite:///dev.db",
    "echo": true
  },
  "logging": {
    "level": "DEBUG"
  }
}
```

### Production
```json
{
  "database": {
    "url": "postgresql://user:pass@localhost:5432/trading",
    "echo": false
  },
  "logging": {
    "level": "INFO"
  }
}
```

## 🛡️ Risk Management Configuration

### Circuit Breakers
```json
{
  "risk": {
    "circuit_breaker": {
      "enabled": true,
      "daily_loss_limit": 10000,
      "consecutive_loss_limit": 3,
      "drawdown_limit": 0.1
    }
  }
}
```

### Position Limits
```json
{
  "risk": {
    "max_position_size": 10000,
    "max_daily_loss": 5000,
    "max_portfolio_heat": 0.2,
    "position_sizing_method": "fixed"
  }
}
```

## 📊 Broker-Specific Configuration

### Alpaca Trading
See: `configs/brokers/alpaca.json`

### Binance
See: `configs/brokers/binance.json`

### Interactive Brokers
See: `configs/brokers/ibkr.json`

## 🧪 Testing Configuration

Use the testing environment for safe testing:

```bash
# Use testing config
export CONFIG_ENV=testing
python main.py
```

Or specify directly:
```bash
python main.py --config configs/environments/testing.json
```

## 🔍 Validation

Validate your configuration before running:

```bash
python scripts/validate_config.py
```

## 🆘 Troubleshooting

### Common Issues

**Configuration not found**
```bash
# Ensure config exists
ls -la config.json

# Or specify path
python main.py --config /path/to/config.json
```

**API keys not loading**
```bash
# Check environment variables
echo $ALPACA_API_KEY

# Verify .env file
cat .env
```

**Database connection issues**
```bash
# Test database connection
python scripts/test_database.py
```

### Configuration Templates

If you need to reset your configuration:

```bash
# Backup current config
cp config.json config.json.backup

# Reset to default
cp configs/config.example.json config.json
```

## 📚 Advanced Configuration

### Custom Configuration Loader

You can implement custom configuration loading:

```python
from trading_orchestrator.config import load_config

# Load from custom location
config = load_config("/custom/path/config.json")

# Override specific sections
config = load_config(
    config_file="config.json",
    override={
        "risk": {"max_position_size": 5000}
    }
)
```

### Configuration Inheritance

Configuration files can inherit from each other:

```json
{
  "_extends": "configs/brokers/alpaca.json",
  "local_overrides": {
    "max_position_size": 15000
  }
}
```

## 🚀 Performance Optimization

### Database
- Use PostgreSQL for production
- Enable connection pooling
- Configure proper indexes

### Caching
- Enable Redis caching
- Configure cache TTL
- Use cache for market data

### Memory
- Limit memory usage
- Enable garbage collection
- Monitor memory leaks

---

For more information, see:
- [Broker Setup Guide](guides/brokers.md)
- [AI Integration Guide](guides/ai-integration.md)
- [Risk Management Guide](guides/risk-management.md)