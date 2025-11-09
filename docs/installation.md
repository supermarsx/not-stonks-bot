# Installation Guide

## Overview

This guide provides step-by-step instructions for installing and configuring the Day Trading Orchestrator system. Choose the installation method that best fits your environment and requirements.

## Prerequisites

### System Requirements

- **Operating System**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Python**: Version 3.11 or higher
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB free space (10GB recommended for data)
- **Network**: Internet connection for real-time market data
- **Browser**: Modern web browser (Chrome, Firefox, Safari, Edge)

## Installation Methods

### Method 1: Quick Setup (Recommended)

For most users, the quick setup script provides the fastest installation:

```bash
# Clone or download the repository
git clone https://github.com/supermarsx/not-stonks-bot.git
cd not-stonks-bot

# Run the setup script
python scripts/setup_dev.py

# Follow the interactive prompts
# The script will handle everything automatically
```

### Method 2: Manual Installation

For advanced users who prefer manual control:

1. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # For development
   ```

3. **Set up configuration**
   ```bash
   cp config.example.json config.json
   # Edit config.json with your settings
   ```

## Configuration

### 1. Broker API Keys

Obtain API keys from supported brokers:

#### Alpaca Markets
1. Visit [Alpaca Markets](https://alpaca.markets)
2. Sign up for a free account
3. Go to API Keys section
4. Generate new API key and secret
5. Use paper trading for testing

#### Binance
1. Visit [Binance](https://binance.com)
2. Complete account verification
3. Go to API Management
4. Create new API key with trading permissions
5. Use testnet for testing (testnet.binance.vision)

### 2. Environment Variables

Set up your environment file:

```bash
# Copy example environment file
cp .env.example .env

# Edit with your API keys
nano .env  # or use your preferred editor
```

## Verification

### 1. Health Check

Verify your installation:

```bash
python scripts/health_check.py
```

### 2. Configuration Validation

```bash
python scripts/validate_config.py
```

## Support

- **Email**: support@not-stonks-bot.com
- **Documentation**: https://docs.not-stonks-bot.com
- **GitHub Issues**: https://github.com/supermarsx/not-stonks-bot/issues

**Disclaimer**: This software is for educational purposes. Always consult with financial professionals before making investment decisions.