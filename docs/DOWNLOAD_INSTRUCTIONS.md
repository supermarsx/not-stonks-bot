# Download Instructions

## Quick Start Guide

Follow these step-by-step instructions to download and install the Day Trading Orchestrator.

## Step 1: Choose Download Method

### Recommended: Git Clone
```bash
git clone https://github.com/supermarsx/not-stonks-bot.git
cd not-stonks-bot
```

### Alternative: ZIP Download
1. Go to https://github.com/supermarsx/not-stonks-bot
2. Click "Code" → "Download ZIP"
3. Extract the downloaded file
4. Open terminal/command prompt in the extracted directory

## Step 2: Check Requirements

Ensure you have:
- **Python 3.11+**: Run `python --version`
- **Git** (if using clone method): Run `git --version`
- **4GB+ RAM**: Minimum for basic operation
- **Internet connection**: Required for real-time data

## Step 3: Run Setup Script

```bash
# Windows
python scripts\setup_dev.py

# macOS/Linux  
python3 scripts/setup_dev.py
```

The setup script will:
- Create virtual environment
- Install dependencies
- Set up configuration
- Run initial tests

## Step 4: Configure Your Settings

1. **Copy example configuration**:
   ```bash
   cp config.example.json config.json
   ```

2. **Edit configuration with your API keys**:
   ```bash
   # Use your preferred editor
   nano config.json
   # or
   code config.json
   ```

3. **Required broker API keys**:
   - Alpaca: Sign up at https://alpaca.markets
   - Binance: Create account at https://binance.com
   - Interactive Brokers: Open account at https://interactivebrokers.com

## Step 5: Verify Installation

```bash
# Run health check
python scripts/health_check.py

# Test configuration
python scripts/validate_config.py
```

## Step 6: Start Trading

```bash
# Demo mode (no real money)
python main.py --demo

# Live trading (with real money)
python main.py
```

## Troubleshooting

### Common Issues:

**Python not found**:
- Install Python 3.11+ from https://python.org
- Add Python to your system PATH

**Permission errors**:
- Run with administrator/root privileges
- Check file permissions

**Configuration errors**:
- Run: `python scripts/validate_config.py`
- Check API key format and validity

**Connection errors**:
- Verify internet connection
- Check firewall settings
- Confirm API key permissions

### Getting Help:

1. Check logs in `logs/` directory
2. Run health check script
3. Visit: https://github.com/supermarsx/not-stonks-bot/issues
4. Email: support@not-stonks-bot.com

## Next Steps:

- Read the User Guide: `docs/user-guide.md`
- Configure strategies: See strategy documentation
- Set up risk management: Review risk settings
- Access web interface: http://localhost:8000
- View API docs: http://localhost:8000/docs

---

**Remember**: Start with demo mode to test your setup before using real money!