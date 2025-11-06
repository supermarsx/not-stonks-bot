# Installation Guide

This comprehensive guide will walk you through installing and setting up the Day Trading Orchestrator system.

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
- [Step-by-Step Installation](#step-by-step-installation)
- [Configuration Setup](#configuration-setup)
- [Broker Integration](#broker-integration)
- [AI Setup](#ai-setup)
- [Database Configuration](#database-configuration)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)

## System Requirements

### Minimum Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Windows 10, macOS 10.15, Ubuntu 18.04 | Latest versions |
| **Python** | 3.8+ | 3.10+ |
| **RAM** | 4GB | 8GB+ |
| **Storage** | 2GB free | 10GB+ SSD |
| **Internet** | Broadband | High-speed broadband |
| **Display** | 1024x768 | 1920x1080 or higher |

### Supported Operating Systems

✅ **Fully Supported:**
- Windows 10/11 (64-bit)
- macOS 10.15+ (Catalina and later)
- Ubuntu 18.04+ (LTS versions)
- CentOS 7/8
- Debian 9+

⚠️ **Experimental Support:**
- Windows Server 2019/2022
- Fedora 30+
- Arch Linux
- WSL (Windows Subsystem for Linux)

### Python Version Compatibility

| Python Version | Support Level | Notes |
|----------------|---------------|-------|
| 3.8 | ✅ Full | Minimum supported version |
| 3.9 | ✅ Full | Recommended for better performance |
| 3.10 | ✅ Full | Optimal performance |
| 3.11+ | ✅ Full | Latest features |

❌ **Not Supported:**
- Python 3.7 and below
- Python 2.x (all versions)

## Installation Methods

### Method 1: Quick Start (Recommended)

For most users, we recommend the quick start approach using our provided launch scripts.

#### Linux/macOS

```bash
# Download or clone the repository
git clone https://github.com/your-username/day-trading-orchestrator.git
cd day-trading-orchestrator

# Make the start script executable
chmod +x start.sh

# Run the installation script
./start.sh
```

#### Windows

```cmd
# Download or clone the repository
git clone https://github.com/your-username/day-trading-orchestrator.git
cd day-trading-orchestrator

# Run the installation script
start.bat
```

### Method 2: Manual Installation

For advanced users who want more control over the installation process.

#### Prerequisites Installation

**Python Installation:**

```bash
# Check if Python is installed
python --version
python3 --version

# If not installed, download from https://python.org
# Or use a package manager:

# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip python3-venv

# macOS (with Homebrew)
brew install python

# Windows (with Chocolatey)
choco install python
```

**Git Installation:**

```bash
# Ubuntu/Debian
sudo apt install git

# macOS (with Homebrew)
brew install git

# Windows (download from git-scm.com)
```

#### Step 1: Clone Repository

```bash
# Clone the repository
git clone https://github.com/your-username/day-trading-orchestrator.git
cd day-trading-orchestrator
```

#### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Verify activation
which python
# Should show: .../venv/bin/python
```

#### Step 3: Install Dependencies

```bash
# Install requirements
pip install -r trading_orchestrator/requirements.txt

# Or install with development dependencies
pip install -r trading_orchestrator/requirements-dev.txt
```

## Step-by-Step Installation

### Step 1: Download the System

```bash
# Option 1: Clone from Git
git clone https://github.com/your-username/day-trading-orchestrator.git
cd day-trading-orchestrator

# Option 2: Download ZIP (Windows)
# Download from GitHub and extract to desired location
```

### Step 2: Verify Python Installation

```bash
# Check Python version
python --version
# Expected output: Python 3.8.x or higher

# Check pip availability
pip --version
```

### Step 3: Create Environment

```bash
# Create isolated environment
python -m venv trading_env

# Activate environment
# Windows:
trading_env\Scripts\activate
# macOS/Linux:
source trading_env/bin/activate

# Verify environment
where python  # Windows
which python  # macOS/Linux
```

### Step 4: Install System Dependencies

```bash
# Install required packages
pip install -r trading_orchestrator/requirements.txt

# Optional: Install development dependencies
pip install -r trading_orchestrator/requirements-dev.txt
```

### Step 5: Create Configuration

```bash
# Generate default configuration
python main.py --create-config

# This creates config.json with default values
```

### Step 6: Verify Installation

```bash
# Run health check
python health_check.py

# Expected output: System is healthy
```

## Configuration Setup

### Environment Variables

1. **Create .env file:**

```bash
cp .env.example .env
```

2. **Edit .env with your settings:**

```env
# Core settings
ENVIRONMENT=development
LOG_LEVEL=INFO

# Database
DATABASE_URL=sqlite:///trading_orchestrator.db

# AI Configuration
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

### Main Configuration File

Edit `config.json` generated in Step 5:

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

## Broker Integration

### Alpaca Trading

1. **Create Alpaca Account:**
   - Visit [alpaca.markets](https://alpaca.markets)
   - Sign up for account
   - Complete account verification

2. **Get API Keys:**
   - Log into Alpaca dashboard
   - Navigate to API Keys section
   - Generate new API key
   - Copy API Key and Secret Key

3. **Configure Alpaca:**

```json
{
  "brokers": {
    "alpaca": {
      "enabled": true,
      "api_key": "YOUR_ACTUAL_ALPACA_API_KEY",
      "secret_key": "YOUR_ACTUAL_ALPACA_SECRET_KEY",
      "paper": true
    }
  }
}
```

### Binance

1. **Create Binance Account:**
   - Visit [binance.com](https://binance.com)
   - Complete identity verification
   - Enable 2FA

2. **API Key Setup:**
   - Go to Account > API Management
   - Create new API key
   - Set appropriate permissions (Read Info, Enable Spot & Margin Trading)
   - Save API Key and Secret Key

3. **Enable Testnet (Recommended):**
   - Use [testnet.binance.vision](https://testnet.binance.vision)
   - Testnet has identical API to mainnet
   - No real funds involved

4. **Configure Binance:**

```json
{
  "brokers": {
    "binance": {
      "enabled": true,
      "api_key": "YOUR_BINANCE_API_KEY",
      "secret_key": "YOUR_BINANCE_SECRET_KEY",
      "testnet": true
    }
  }
}
```

### Interactive Brokers

1. **Download TWS or IBKR Gateway:**
   - Visit [interactivebrokers.com](https://interactivebrokers.com)
   - Download TWS (Trader Workstation) or IBKR Gateway
   - Install following their installation guide

2. **Configure TWS for API:**
   - Open TWS
   - Go to Edit > Global Configuration
   - Navigate to API > Settings
   - Enable "Enable ActiveX and Socket Clients"
   - Set socket port (default 7497)
   - Note the port number

3. **Configure IBKR:**

```json
{
  "brokers": {
    "ibkr": {
      "enabled": true,
      "host": "127.0.0.1",
      "port": 7497,
      "client_id": 1,
      "paper": true
    }
  }
}
```

### Other Brokers

For additional broker configurations, see:
- [Trading 212 Configuration](config.trading212.example.json)
- [DEGIRO Configuration](config.degiro.example.json)
- [XTB Configuration](config.xtb.example.json)
- [Trade Republic Configuration](config.trade_republic.example.json)

## AI Setup

### OpenAI Integration

1. **Create OpenAI Account:**
   - Visit [platform.openai.com](https://platform.openai.com)
   - Sign up for account
   - Add payment method

2. **Generate API Key:**
   - Go to API Keys section
   - Create new secret key
   - Copy the key (starts with "sk-")

3. **Configure OpenAI:**

```json
{
  "ai": {
    "openai_api_key": "sk-your-openai-key-here",
    "model": "gpt-4o",
    "max_tokens": 4000
  }
}
```

### Anthropic Claude Integration

1. **Create Anthropic Account:**
   - Visit [console.anthropic.com](https://console.anthropic.com)
   - Sign up for account
   - Complete verification

2. **Get API Key:**
   - Navigate to API Keys
   - Create new key
   - Copy the key

3. **Configure Claude:**

```json
{
  "ai": {
    "anthropic_api_key": "your-claude-api-key-here",
    "model": "claude-3-sonnet-20240229"
  }
}
```

### Local AI Models

For users who prefer local models:

1. **Install Ollama:**
   ```bash
   # macOS
   brew install ollama
   
   # Windows/Linux
   curl https://ollama.ai/install.sh | sh
   ```

2. **Pull Models:**
   ```bash
   ollama pull llama2
   ollama pull codellama
   ```

3. **Configure Local Models:**

```json
{
  "ai": {
    "local_models": {
      "enabled": true,
      "model_backend": "ollama",
      "model_name": "llama2"
    }
  }
}
```

## Database Configuration

### SQLite (Default, Recommended for Testing)

```json
{
  "database": {
    "url": "sqlite:///trading_orchestrator.db"
  }
}
```

### PostgreSQL (Recommended for Production)

```json
{
  "database": {
    "url": "postgresql://username:password@localhost:5432/trading_db"
  }
}
```

### MySQL

```json
{
  "database": {
    "url": "mysql://username:password@localhost:3306/trading_db"
  }
}
```

## Verification

### Run Health Check

```bash
# Run comprehensive health check
python health_check.py

# Check specific components
python health_check.py --database
python health_check.py --brokers
python health_check.py --ai
```

### Test Basic Functionality

```bash
# Run integration tests
python test_integration.py

# Test demo mode
python main.py --demo

# Validate configuration
python validate_config.py
```

### Verify AI Integration

```bash
# Test AI functionality
python -c "
import asyncio
from ai.orchestrator import AITradingOrchestrator
app = AITradingOrchestrator()
result = await app.analyze_market('AAPL')
print(result)
"
```

## Troubleshooting

### Common Installation Issues

#### Python Version Issues

**Problem:** "Python version not supported"
```bash
# Check Python version
python --version

# If version < 3.8, upgrade Python
# Ubuntu/Debian
sudo apt update && sudo apt install python3.10

# macOS (with Homebrew)
brew install python@3.10

# Windows - download from python.org
```

#### Permission Errors

**Problem:** "Permission denied" during installation
```bash
# Use user installation
pip install --user -r requirements.txt

# Or use virtual environment
python -m venv myenv
source myenv/bin/activate  # Linux/macOS
myenv\Scripts\activate     # Windows
pip install -r requirements.txt
```

#### SSL Certificate Issues

**Problem:** "SSL certificate verification failed"
```bash
# Update certificates
pip install --upgrade certifi

# Or disable SSL verification (not recommended)
pip install --trusted-host pypi.org --trusted-host pypi.python.org -r requirements.txt
```

#### Virtual Environment Issues

**Problem:** Virtual environment not working
```bash
# Recreate virtual environment
rm -rf venv
python -m venv venv

# Activate and install
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Broker Connection Issues

#### Alpaca Connection Failed

```bash
# Test connection
curl -H "APCA-API-KEY-ID: YOUR_KEY" \
     -H "APCA-API-SECRET-KEY: YOUR_SECRET" \
     https://paper-api.alpaca.markets/v2/account
```

**Common fixes:**
- Verify API keys are correct
- Check paper trading is enabled
- Ensure account is approved for trading

#### Binance Connection Failed

```bash
# Test testnet connection
curl -H "X-MBX-APIKEY: YOUR_API_KEY" \
     https://testnet.binance.vision/api/v3/account
```

**Common fixes:**
- Enable testnet for testing
- Check IP restrictions
- Verify API permissions

#### IBKR Connection Failed

```bash
# Check if TWS is running
netstat -an | grep 7497

# Test connection
telnet 127.0.0.1 7497
```

**Common fixes:**
- Start TWS or IBKR Gateway
- Enable API in TWS settings
- Check firewall settings

### AI Integration Issues

#### OpenAI API Errors

```bash
# Test API key
curl -H "Authorization: Bearer YOUR_API_KEY" \
     https://api.openai.com/v1/models
```

**Common fixes:**
- Verify API key is valid
- Check account has sufficient credits
- Verify model availability in your region

#### Rate Limiting

```json
{
  "ai": {
    "rate_limits": {
      "requests_per_minute": 60,
      "retry_delay": 1
    }
  }
}
```

## Advanced Configuration

### Performance Optimization

#### Database Optimization

```json
{
  "database": {
    "pool_size": 20,
    "max_overflow": 30,
    "pool_timeout": 30,
    "pool_recycle": 3600
  }
}
```

#### Memory Optimization

```json
{
  "performance": {
    "caching": {
      "enabled": true,
      "cache_size": "1GB",
      "ttl": 3600
    },
    "workers": {
      "max_workers": 8,
      "queue_size": 1000
    }
  }
}
```

### Security Configuration

#### API Key Encryption

```json
{
  "security": {
    "encryption": {
      "enabled": true,
      "algorithm": "AES-256",
      "key_derivation": "PBKDF2"
    }
  }
}
```

#### Network Security

```json
{
  "security": {
    "network": {
      "allowed_hosts": ["localhost", "127.0.0.1"],
      "ssl_verify": true,
      "timeout": 30
    }
  }
}
```

### Monitoring Setup

#### Logging Configuration

```json
{
  "logging": {
    "level": "INFO",
    "handlers": {
      "file": {
        "enabled": true,
        "file_path": "logs/trading_orchestrator.log",
        "max_size": "100MB",
        "backup_count": 5
      },
      "console": {
        "enabled": true,
        "level": "INFO"
      }
    }
  }
}
```

#### Metrics Collection

```json
{
  "monitoring": {
    "prometheus": {
      "enabled": true,
      "port": 8000,
      "path": "/metrics"
    }
  }
}
```

## Next Steps

After successful installation:

1. **Read the [Usage Guide](usage.md)** - Learn how to use the system
2. **Configure Brokers** - Set up your trading accounts
3. **Test Demo Mode** - Practice without real money
4. **Start Paper Trading** - Test with virtual money
5. **Enable Live Trading** - Only when you're comfortable

## Support

If you encounter issues during installation:

1. Check the [Troubleshooting Guide](troubleshooting.md)
2. Search [GitHub Issues](https://github.com/your-username/day-trading-orchestrator/issues)
3. Create a new issue with installation details
4. Join our [Discord community](https://discord.gg/trading-orchestrator)

## Getting Help

- **Documentation**: [docs/](docs/) directory
- **GitHub Issues**: [Create issue](https://github.com/your-username/day-trading-orchestrator/issues/new)
- **Discord**: [Join our server](https://discord.gg/trading-orchestrator)
- **Email**: support@trading-orchestrator.com