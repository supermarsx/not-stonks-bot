# Download Guide

## Overview

This guide provides comprehensive instructions for downloading and setting up the Day Trading Orchestrator system. The system is distributed in multiple formats to accommodate different use cases and technical requirements.

## System Requirements

### Minimum Requirements

- **Operating System**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Python**: Version 3.11 or higher
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB free space
- **Network**: Internet connection for real-time data
- **Dependencies**: Git, pip (Python package manager)

### Recommended Requirements

- **Operating System**: Latest stable versions of supported OS
- **Python**: Version 3.11+
- **Memory**: 16GB RAM
- **Storage**: 10GB SSD
- **Network**: High-speed internet connection
- **Dependencies**: Docker, Node.js (optional for web interface)

## Download Options

### Option 1: Git Clone (Recommended)

The most up-to-date version with the latest features and bug fixes.

```bash
# Clone the repository
git clone https://github.com/supermarsx/not-stonks-bot.git
cd not-stonks-bot

# Switch to stable branch
git checkout main

# Verify installation
python --version
git status
```

**Advantages:**
- Always up-to-date with latest features
- Easy to update to newer versions
- Access to development branches
- Full Git history and commit information

**Disadvantages:**
- Requires Git installation
- Larger download size
- Requires manual setup

### Option 2: Download ZIP Archive

For users who prefer not to use Git or need a specific version.

1. **Visit the repository page** at https://github.com/supermarsx/not-stonks-bot
2. **Click the green "Code" button**
3. **Select "Download ZIP"**
4. **Extract the downloaded archive**
5. **Navigate to the extracted directory**

### Option 3: Package Manager Installation

For users who prefer automated package management.

#### Using pip (Python Package Index)

```bash
pip install trading-orchestrator
```

#### Using conda (Anaconda)

```bash
conda install -c conda-forge trading-orchestrator
```

#### Using Docker

```bash
# Pull the official image
docker pull supermarsx/not-stonks-bot:latest

# Run the container
docker run -d -p 8000:8000 supermarsx/not-stonks-bot
```

## Installation Methods

### Method 1: Automated Setup (Recommended)

Use our automated setup script for a hassle-free installation:

```bash
# Run the automated setup
python scripts/setup_dev.py

# Follow the interactive prompts
# The script will automatically:
# - Create a virtual environment
# - Install all dependencies
# - Set up configuration files
# - Run initial tests
```

### Method 2: Manual Installation

For advanced users who prefer manual control:

1. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # For development
   ```

3. **Set up configuration**
   ```bash
   # Copy example configuration
   cp config.example.json config.json
   
   # Edit configuration with your API keys
   nano config.json  # or your preferred editor
   ```

4. **Verify installation**
   ```bash
   python scripts/health_check.py
   ```

### Method 3: Docker Installation

For users who prefer containerized deployment:

1. **Build the Docker image**
   ```bash
   docker build -t not-stonks-bot .
   ```

2. **Run the container**
   ```bash
   docker run -d \
     --name not-stonks-bot \
     -p 8000:8000 \
     -v $(pwd)/config.json:/app/config.json \
     -v $(pwd)/data:/app/data \
     not-stonks-bot
   ```

3. **Access the application**
   - Web Interface: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

## Platform-Specific Instructions

### Windows Installation

1. **Install Python**
   - Download from https://python.org
   - Ensure "Add Python to PATH" is checked
   - Verify installation: `python --version`

2. **Install Git** (if not using ZIP download)
   - Download from https://git-scm.com
   - Use default installation options
   - Verify installation: `git --version`

3. **Clone or extract the repository**

4. **Run the setup script**
   ```cmd
   python scripts\setup_dev.py
   ```

### macOS Installation

1. **Install Homebrew** (if not already installed)
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install Python and Git**
   ```bash
   brew install python@3.11 git
   ```

3. **Clone the repository**
   ```bash
   git clone https://github.com/supermarsx/not-stonks-bot.git
   cd not-stonks-bot
   ```

4. **Run the setup script**
   ```bash
   python3 scripts/setup_dev.py
   ```

### Linux Installation

1. **Update package manager**
   ```bash
   sudo apt update  # Ubuntu/Debian
   # or
   sudo yum update  # RHEL/CentOS
   ```

2. **Install dependencies**
   ```bash
   sudo apt install python3 python3-pip python3-venv git  # Ubuntu/Debian
   # or
   sudo yum install python3 python3-pip git  # RHEL/CentOS
   ```

3. **Clone and setup**
   ```bash
   git clone https://github.com/supermarsx/not-stonks-bot.git
   cd not-stonks-bot
   python3 scripts/setup_dev.py
   ```

## Configuration Setup

### 1. Broker API Keys

Obtain API keys from supported brokers:

- **Alpaca**: https://alpaca.markets
- **Binance**: https://binance.com
- **Interactive Brokers**: https://interactivebrokers.com
- **Trading 212**: https://trading212.com
- **XTB**: https://xtb.com

### 2. Environment Variables

Set up your environment file:

```bash
# Copy the example environment file
cp .env.example .env

# Edit with your API keys
nano .env
```

### 3. Configuration File

Update your configuration:

```json
{
  "api_keys": {
    "alpaca": {
      "api_key": "your_alpaca_api_key",
      "secret_key": "your_alpaca_secret_key",
      "paper_trading": true
    },
    "binance": {
      "api_key": "your_binance_api_key",
      "secret_key": "your_binance_secret_key",
      "testnet": true
    }
  },
  "trading": {
    "max_position_size": 10000,
    "risk_per_trade": 0.02,
    "stop_loss_percent": 0.05
  },
  "strategies": {
    "enabled": ["mean_reversion", "trend_following"]
  }
}
```

## First Run

### 1. Health Check

Verify your installation:

```bash
python scripts/health_check.py
```

### 2. Demo Mode

Test the system in demo mode:

```bash
python main.py --demo
```

### 3. Full Startup

Start the complete system:

```bash
python main.py
```

### 4. Web Interface

Access the web interface:
- URL: http://localhost:8000
- Default port: 8000
- API docs: http://localhost:8000/docs

## Verification Tests

### Test 1: System Health

```bash
python scripts/health_check.py
```

Expected output: All systems green

### Test 2: Configuration Validation

```bash
python scripts/validate_config.py
```

Expected output: Configuration valid

### Test 3: Broker Connection

```bash
python scripts/test_broker_connection.py
```

Expected output: Connection successful

### Test 4: Strategy Execution

```bash
python main.py --test-strategy
```

Expected output: Strategy runs without errors

## Troubleshooting

### Common Issues

#### Python Version Error

**Problem**: Python version too old

**Solution**:
```bash
# Check Python version
python --version

# Install newer Python (example for Ubuntu)
sudo apt install python3.11
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
```

#### Permission Denied

**Problem**: Insufficient permissions

**Solution**:
```bash
# Fix file permissions
chmod +x scripts/*.sh
chmod +x start.sh
chmod +x start.bat
```

#### Missing Dependencies

**Problem**: Module not found errors

**Solution**:
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### Configuration Errors

**Problem**: Invalid configuration

**Solution**:
```bash
# Validate configuration
python scripts/validate_config.py

# Reset to defaults
cp config.example.json config.json
```

#### Network Issues

**Problem**: Cannot connect to brokers

**Solution**:
- Check internet connection
- Verify firewall settings
- Check API key validity
- Review rate limiting

### Getting Help

If you encounter issues:

1. **Check the troubleshooting section**
2. **Run the health check script**
3. **Review the logs** in the `logs/` directory
4. **Search existing issues** on GitHub
5. **Create a new issue** with detailed information

## Updating

### Update from Git

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Run health check
python scripts/health_check.py
```

### Update Package

```bash
# Update using pip
pip install trading-orchestrator --upgrade

# Update using conda
conda update trading-orchestrator
```

### Update Docker

```bash
# Pull latest image
docker pull supermarsx/not-stonks-bot:latest

# Stop and remove old container
docker stop not-stonks-bot
docker rm not-stonks-bot

# Start new container
docker run -d --name not-stonks-bot -p 8000:8000 supermarsx/not-stonks-bot:latest
```

## Uninstallation

### Complete Removal

1. **Stop all services**
2. **Remove the directory**
3. **Clean up Python environment**
4. **Remove configuration files**

```bash
# Stop services
python main.py --stop

# Remove directory
rm -rf not-stonks-bot

# Remove Python environment
rm -rf venv

# Remove configuration
rm -f config.json .env
```

### Docker Cleanup

```bash
# Stop and remove container
docker stop not-stonks-bot
docker rm not-stonks-bot

# Remove image
docker rmi supermarsx/not-stonks-bot
```

## Support

### Documentation

- **Installation Guide**: This document
- **User Guide**: `docs/user-guide.md`
- **API Documentation**: http://localhost:8000/docs
- **Configuration Reference**: `docs/configuration.md`

### Community

- **GitHub Issues**: https://github.com/supermarsx/not-stonks-bot/issues
- **Discussions**: https://github.com/supermarsx/not-stonks-bot/discussions
- **Discord**: Join our community server
- **Email**: support@not-stonks-bot.com

### Professional Support

For enterprise users:
- **Email**: enterprise@not-stonks-bot.com
- **Documentation**: https://docs.not-stonks-bot.com
- **SLA**: Available for enterprise customers

---

**Note**: This installation guide is regularly updated. Always check for the latest version before installing or upgrading the system.