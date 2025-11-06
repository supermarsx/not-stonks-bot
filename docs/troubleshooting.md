# Troubleshooting Guide

This comprehensive guide helps you resolve common issues with the Day Trading Orchestrator system.

## Table of Contents

- [Quick Diagnosis](#quick-diagnosis)
- [Installation Issues](#installation-issues)
- [Configuration Problems](#configuration-problems)
- [Broker Connection Issues](#broker-connection-issues)
- [AI Integration Problems](#ai-integration-problems)
- [Database Issues](#database-issues)
- [Performance Problems](#performance-problems)
- [UI/Interface Issues](#uiinterface-issues)
- [Trading Errors](#trading-errors)
- [System Monitoring](#system-monitoring)
- [Log Analysis](#log-analysis)
- [Getting Help](#getting-help)

## Quick Diagnosis

### Health Check Script

Run the comprehensive health check to identify issues:

```bash
# Full system health check
python health_check.py

# Check specific components
python health_check.py --database
python health_check.py --brokers
python health_check.py --ai
python health_check.py --network

# Detailed diagnostic
python health_check.py --detailed --output report.json
```

### System Status Commands

```bash
# Check overall system status
> status

# Check broker connections
> status brokers

# Check AI services
> status ai

# Check database connectivity
> status database

# Check system resources
> status system
```

### Common Error Patterns

| Error Type | Quick Fix | Check Command |
|------------|-----------|---------------|
| Connection Timeout | Restart broker | `> restart broker alpaca` |
| Authentication Failed | Verify API keys | `> test broker alpaca` |
| Configuration Error | Validate config | `python validate_config.py` |
| Database Locked | Restart application | `> restart system` |
| AI API Error | Check API keys | `> test ai connectivity` |

## Installation Issues

### Python Version Problems

**Error:** "Python version not supported"
```bash
# Check current version
python --version
python3 --version

# Verify minimum version (3.8+)
python -c "import sys; print(sys.version_info)"

# Upgrade Python if needed
# Ubuntu/Debian
sudo apt update && sudo apt install python3.10

# macOS (with Homebrew)
brew install python@3.10

# Windows - Download from python.org
```

**Error:** "Command 'python' not found"
```bash
# Try python3 instead
python3 --version

# Create alias (Linux/macOS)
echo 'alias python=python3' >> ~/.bashrc
source ~/.bashrc

# Windows - Add Python to PATH or use py command
py --version
```

### Permission Errors

**Error:** "Permission denied" during installation
```bash
# Use user installation
pip install --user -r trading_orchestrator/requirements.txt

# Or use virtual environment
python -m venv myenv
source myenv/bin/activate  # Linux/macOS
myenv\Scripts\activate     # Windows
pip install -r trading_orchestrator/requirements.txt

# Fix permissions on existing installation
chmod +x start.sh  # Linux/macOS
```

### Virtual Environment Issues

**Error:** Virtual environment not working
```bash
# Recreate virtual environment
rm -rf venv trading_env
python -m venv venv

# Activate and verify
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
which python  # Should show venv path

# Reinstall dependencies
pip install --upgrade pip
pip install -r trading_orchestrator/requirements.txt
```

### SSL Certificate Issues

**Error:** "SSL certificate verification failed"
```bash
# Update certificates
pip install --upgrade certifi
pip install --upgrade requests

# Or disable SSL verification (not recommended)
pip install --trusted-host pypi.org --trusted-host pypi.python.org -r requirements.txt

# For corporate networks, configure proxy
pip install --proxy http://user:pass@proxy.company.com:8080 -r requirements.txt
```

### Package Dependencies

**Error:** "No module named 'xyz'"
```bash
# Check installed packages
pip list

# Install missing packages
pip install -r trading_orchestrator/requirements.txt

# Force reinstall problematic packages
pip install --force-reinstall package_name

# Check for version conflicts
pip check
```

## Configuration Problems

### Invalid JSON Syntax

**Error:** "Invalid JSON configuration file"
```bash
# Validate JSON syntax
python -c "import json; json.load(open('config.json'))"

# Or use online JSON validator
# Check for common issues:
# - Trailing commas
# - Missing quotes
# - Unescaped characters
# - Invalid escape sequences
```

### Missing Configuration Files

**Error:** "Configuration file not found"
```bash
# Create default configuration
python main.py --create-config

# Copy example configuration
cp config.example.json config.json

# Check file permissions
ls -la config.json
chmod 644 config.json  # Read/write for owner
```

### Environment Variable Issues

**Error:** "Environment variable not set"
```bash
# Check environment variables
env | grep -E "(ALPACA|BINANCE|OPENAI)"

# Load .env file manually
source .env

# Verify critical variables
python -c "
import os
print('ALPACA_API_KEY:', 'Set' if os.getenv('ALPACA_API_KEY') else 'Missing')
print('BINANCE_API_KEY:', 'Set' if os.getenv('BINANCE_API_KEY') else 'Missing')
"
```

### API Key Validation

**Error:** "Invalid API key format"
```bash
# Test API key format
# Alpaca: Should be alphanumeric
echo $ALPACA_API_KEY | grep -E "^[A-Za-z0-9]+$"

# Binance: Should be 64 characters
echo $BINANCE_API_KEY | wc -c  # Should be 65 (including newline)

# OpenAI: Should start with 'sk-'
echo $OPENAI_API_KEY | grep -E "^sk-"
```

## Broker Connection Issues

### Alpaca Connection Problems

**Error:** "Alpaca connection failed"

```bash
# Test API key
curl -H "APCA-API-KEY-ID: $ALPACA_API_KEY" \
     -H "APCA-API-SECRET-KEY: $ALPACA_SECRET_KEY" \
     https://paper-api.alpaca.markets/v2/account

# Check account status
curl -H "APCA-API-KEY-ID: $ALPACA_API_KEY" \
     -H "APCA-API-SECRET-KEY: $ALPACA_SECRET_KEY" \
     https://paper-api.alpaca.markets/v2/account/status
```

**Solutions:**
- Verify API keys are correct (no extra spaces)
- Check paper trading is enabled
- Ensure account is funded (for live trading)
- Verify IP is allowed (if IP restriction enabled)

**Debug commands:**
```bash
# Test connection programmatically
python -c "
from brokers.alpaca_broker import AlpacaBroker
broker = AlpacaBroker('YOUR_API_KEY', 'YOUR_SECRET_KEY', paper=True)
print('Connection:', broker.test_connection())
"

# Check network connectivity
telnet paper-api.alpaca.markets 443
```

### Binance Connection Problems

**Error:** "Binance connection failed"

```bash
# Test API key
curl -H "X-MBX-APIKEY: $BINANCE_API_KEY" \
     https://api.binance.com/api/v3/account

# Test connectivity
curl https://api.binance.com/api/v3/ping

# Check time synchronization
curl -H "X-MBX-APIKEY: $BINANCE_API_KEY" \
     https://api.binance.com/api/v3/time
```

**Solutions:**
- Verify API key has trading permissions
- Check if IP is whitelisted
- Ensure testnet URL is used for testnet
- Check for rate limiting (429 errors)

**Debug commands:**
```bash
# Check Binance server time
date
curl -s https://api.binance.com/api/v3/time | jq '.serverTime'

# Test order placement (testnet)
python -c "
from brokers.binance_broker import BinanceBroker
broker = BinanceBroker('TEST_API_KEY', 'TEST_SECRET_KEY', testnet=True)
print('Server time:', broker.get_server_time())
"
```

### Interactive Brokers Connection Problems

**Error:** "IBKR connection failed"

```bash
# Check if TWS is running
netstat -an | grep 7497
ps aux | grep "Trader Workstation"

# Test port connectivity
telnet 127.0.0.1 7497

# Check TWS API settings
# 1. Open TWS
# 2. Edit > Global Configuration
# 3. API > Settings
# 4. Verify settings match config.json
```

**Common IBKR Issues:**
- TWS not running or not connected
- API not enabled in TWS settings
- Wrong port number configured
- Client ID already in use
- TWS version incompatible

**Solutions:**
```bash
# Restart TWS and IBKR Gateway
# Verify API settings in TWS
# Check firewall settings
# Ensure client_id is unique

# Test with simple Python script
python -c "
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
result = sock.connect_ex(('127.0.0.1', 7497))
sock.close()
print('Port 7497:', 'Open' if result == 0 else 'Closed')
"
```

### Multiple Broker Conflicts

**Error:** "Broker configuration conflict"

```bash
# Check for duplicate enabled brokers
python -c "
import json
with open('config.json') as f:
    config = json.load(f)
for broker, settings in config['brokers'].items():
    if settings.get('enabled', False):
        print(f'{broker}: enabled')
"

# Verify broker selection logic
python validate_config.py --brokers
```

## AI Integration Problems

### OpenAI API Issues

**Error:** "OpenAI API request failed"

```bash
# Test API key
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models

# Check account limits
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/dashboard/billing/subscription

# Test simple completion
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello"}]}' \
     https://api.openai.com/v1/chat/completions
```

**Common OpenAI Issues:**
- Insufficient credits/quota exceeded
- Invalid API key
- Model access not granted
- Rate limiting (429 errors)
- Network connectivity issues

**Solutions:**
```python
# Test OpenAI integration
python -c "
import openai
import os

client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
try:
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[{'role': 'user', 'content': 'Test message'}]
    )
    print('OpenAI API: Working')
except Exception as e:
    print(f'OpenAI API Error: {e}')
"
```

### Anthropic API Issues

**Error:** "Anthropic API request failed"

```bash
# Test API key
curl -H "x-api-key: $ANTHROPIC_API_KEY" \
     https://api.anthropic.com/v1/messages \
     -H "anthropic-version: 2023-06-01" \
     -H "Content-Type: application/json" \
     -d '{"model": "claude-3-sonnet-20240229", "max_tokens": 1024, "messages": [{"role": "user", "content": "Hello"}]}'
```

### Local Model Issues

**Error:** "Local model not found"

```bash
# Check Ollama installation
ollama --version
ollama list

# Install model
ollama pull llama2

# Test model
ollama run llama2 "Hello"

# Check model path
echo $OLLAMA_BASE_URL
```

## Database Issues

### SQLite Database Problems

**Error:** "Database is locked"

```bash
# Check for database locks
lsof trading_orchestrator.db

# Remove lock file if exists
rm -f trading_orchestrator.db-wal trading_orchestrator.db-shm

# Check database integrity
sqlite3 trading_orchestrator.db "PRAGMA integrity_check;"

# Backup and recreate
cp trading_orchestrator.db trading_orchestrator.db.backup
sqlite3 trading_orchestrator.db "VACUUM;"
```

### PostgreSQL Connection Issues

**Error:** "PostgreSQL connection failed"

```bash
# Test database connection
psql -h localhost -U username -d trading_db -c "SELECT 1;"

# Check database status
sudo systemctl status postgresql

# Check connection logs
sudo tail -f /var/log/postgresql/postgresql-*.log

# Verify credentials
python -c "
from sqlalchemy import create_engine
try:
    engine = create_engine('postgresql://user:pass@localhost/db')
    conn = engine.connect()
    print('PostgreSQL: Connected')
except Exception as e:
    print(f'PostgreSQL Error: {e}')
"
```

### Database Migration Issues

**Error:** "Migration failed"

```bash
# Check migration status
python -c "
from database.migrations import get_migration_status
print(get_migration_status())
"

# Run migrations manually
python -m alembic upgrade head

# Check migration history
python -m alembic history

# Reset migrations (careful!)
python -m alembic downgrade base
```

## Performance Problems

### High CPU Usage

```bash
# Monitor system resources
top -p $(pgrep -f "python.*main.py")

# Check memory usage
free -h
ps aux --sort=-%mem | head

# Profile application
python -m cProfile -s tottime main.py

# Check for infinite loops
strace -p $(pgrep -f "python.*main.py") -c
```

### Memory Leaks

```bash
# Monitor memory growth
watch -n 5 'ps aux | grep python | awk "{print \$4, \$11}"'

# Profile memory usage
python -m memory_profiler main.py

# Check for large objects
python -c "
import tracemalloc
tracemalloc.start()
# Your code here
current, peak = tracemalloc.get_traced_memory()
print(f'Current: {current / 1024 / 1024:.1f} MB')
print(f'Peak: {peak / 1024 / 1024:.1f} MB')
"
```

### Slow Database Queries

```bash
# Enable SQL logging
# In config.json:
{
  "database": {
    "echo": true,
    "pool_size": 20,
    "pool_timeout": 30
  }
}

# Check slow queries
sqlite3 trading_orchestrator.db "EXPLAIN QUERY PLAN SELECT * FROM trades;"

# Analyze table structure
sqlite3 trading_orchestrator.db ".schema trades"

# Create indexes if needed
sqlite3 trading_orchestrator.db "CREATE INDEX idx_trades_symbol ON trades(symbol);"
sqlite3 trading_orchestrator.db "CREATE INDEX idx_trades_timestamp ON trades(timestamp);"
```

### Network Latency

```bash
# Test broker API latency
time curl -s -o /dev/null -w "%{time_total}" https://paper-api.alpaca.markets/v2/account

# Check network connectivity
ping google.com
traceroute api.binance.com

# Monitor real-time latency
while true; do
  echo "$(date): $(time -f '%e' curl -s https://api.binance.com/api/v3/time > /dev/null)"
  sleep 1
done
```

## UI/Interface Issues

### Terminal Display Problems

**Error:** "Terminal rendering broken"

```bash
# Check terminal compatibility
echo $TERM
tput colors

# Set proper terminal type
export TERM=xterm-256color

# Clear screen and reset
clear
reset

# Disable Matrix theme for debugging
python main.py --no-theme
```

### Character Encoding Issues

**Error:** "Unicode characters not displaying"

```bash
# Set UTF-8 encoding
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

# Python encoding fix
python -c "
import sys
sys.stdout.reconfigure(encoding='utf-8')
print('Unicode test: €£¥')
"

# Check locale
locale
```

### Screen Size Issues

**Error:** "Interface doesn't fit screen"

```bash
# Get terminal dimensions
stty size
tput cols
tput lines

# Resize terminal
# Or set minimum size in config:
{
  "ui": {
    "terminal_size": {
      "width": 120,
      "height": 40
    }
  }
}
```

### Cursor Position Errors

**Error:** "Cursor jumping around"

```bash
# Disable cursor auto-hide
export TERM=xterm

# Reset terminal state
tput reset

# Clear terminal buffer
printf '\033[2J\033[H'
```

## Trading Errors

### Order Rejection

**Error:** "Order rejected by broker"

```bash
# Check order details
> order status --id ORDER_ID

# Get rejection reason
> alpaca order history --rejected

# Check account status
> alpaca account

# Verify market hours
> market status

# Check position limits
> risk limits
```

**Common order rejection reasons:**
- Insufficient buying power
- Market closed
- Symbol not tradeable
- Position limit exceeded
- Regulatory restrictions

### Price Slippage

**Error:** "Order filled at worse than expected price"

```bash
# Check order execution details
> order details --id ORDER_ID

# Analyze slippage
python -c "
# Calculate slippage for recent orders
import sqlite3
conn = sqlite3.connect('trading_orchestrator.db')
orders = conn.execute('SELECT symbol, price, filled_price FROM orders WHERE filled_price IS NOT NULL').fetchall()
for symbol, expected, actual in orders:
    slippage = ((actual - expected) / expected) * 100
    print(f'{symbol}: {slippage:.2f}% slippage')
"
```

### Position Sync Issues

**Error:** "Position mismatch between broker and system"

```bash
# Sync positions with broker
> positions sync

# Force position reconciliation
> portfolio reconcile

# Check for orphaned positions
> positions orphaned

# Reset positions (if needed)
> positions reset --force
```

### Market Data Issues

**Error:** "Market data not updating"

```bash
# Check market data connection
> market status

# Restart market data feeds
> restart market_data

# Test data source
> market test binance
> market test alpaca

# Check data subscription
> market subscriptions
```

## System Monitoring

### Health Check Automation

```bash
# Create monitoring script
cat > monitor_system.sh << 'EOF'
#!/bin/bash
while true; do
    python health_check.py --quick
    sleep 300  # Check every 5 minutes
done
EOF

# Run in background
chmod +x monitor_system.sh
nohup ./monitor_system.sh &
```

### Log Rotation

```bash
# Setup logrotate
sudo cat > /etc/logrotate.d/trading-orchestrator << 'EOF'
/path/to/trading-orchestrator/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
EOF
```

### Resource Monitoring

```bash
# Monitor CPU and memory
cat > resource_monitor.py << 'EOF'
#!/usr/bin/env python3
import psutil
import time
import json

def monitor_resources():
    while True:
        data = {
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent
        }
        print(json.dumps(data))
        time.sleep(60)

if __name__ == "__main__":
    monitor_resources()
EOF

# Run monitoring
python resource_monitor.py > resource_log.json
```

## Log Analysis

### Common Log Patterns

```bash
# Find errors in logs
grep -i error logs/trading_orchestrator.log

# Find broker connection issues
grep -i "connection.*failed\|timeout\|refused" logs/*.log

# Find order issues
grep -i "order.*rejected\|rejected\|failed" logs/*.log

# Find AI errors
grep -i "ai.*error\|openai.*error\|claude.*error" logs/*.log
```

### Log Analysis Script

```bash
#!/bin/bash
# analyze_logs.sh - Comprehensive log analysis

echo "=== Last 24 Hours Summary ==="
find logs/ -name "*.log" -mtime -1 -exec wc -l {} \; | awk '{sum+=$1} END {print "Total log lines:", sum}'

echo -e "\n=== Error Count by Type ==="
grep -h -i "error\|exception\|failed" logs/trading_orchestrator.log | \
awk '{tolower($0); if(/broker/) broker++; else if(/ai/) ai++; else if(/database/) db++; else if(/network/) network++; else other++} 
END {print "Broker errors:", broker+0; print "AI errors:", ai+0; print "Database errors:", db+0; print "Network errors:", network+0; print "Other errors:", other+0}'

echo -e "\n=== Performance Issues ==="
grep -i "slow\|timeout\|lag" logs/trading_orchestrator.log | tail -10

echo -e "\n=== Recent Crashes ==="
grep -i "crash\|abort\|segmentation" logs/trading_orchestrator.log | tail -5
```

### Real-time Log Monitoring

```bash
# Monitor logs in real-time
tail -f logs/trading_orchestrator.log | grep -E "(ERROR|CRITICAL|Exception)"

# Monitor specific components
tail -f logs/trading_orchestrator.log | grep -E "(alpaca|binance|ibkr)"

# Alert on errors
tail -f logs/trading_orchestrator.log | while read line; do
    if echo "$line" | grep -qi "error\|exception"; then
        echo "ALERT: $line" | mail -s "Trading Orchestrator Error" admin@example.com
    fi
done
```

## Getting Help

### Self-Diagnosis Checklist

Before seeking help, gather this information:

```bash
# System information
uname -a
python --version
pip list | grep -E "(alpaca|binance|openai|anthropic)"

# Configuration status
python validate_config.py

# Health check report
python health_check.py --detailed --output diagnostic_report.json

# Recent error logs
tail -100 logs/trading_orchestrator.log
```

### Diagnostic Report Template

```json
{
  "system_info": {
    "os": "Ubuntu 20.04",
    "python_version": "3.9.7",
    "architecture": "x86_64"
  },
  "configuration": {
    "config_file_exists": true,
    "env_file_exists": true,
    "api_keys_set": {
      "alpaca": false,
      "binance": false,
      "openai": false
    }
  },
  "connectivity": {
    "internet": true,
    "alpaca_api": false,
    "binance_api": false,
    "openai_api": false
  },
  "errors": [
    "2024-01-15 10:30:15 - ERROR - Connection to Alpaca failed",
    "2024-01-15 10:30:16 - ERROR - Authentication failed for API key"
  ]
}
```

### Support Channels

1. **GitHub Issues** - [Create Issue](https://github.com/your-username/day-trading-orchestrator/issues/new)
   - Include diagnostic report
   - Describe steps to reproduce
   - Attach relevant log files

2. **Discord Community** - [Join Server](https://discord.gg/trading-orchestrator)
   - Real-time help
   - Community discussions
   - Feature requests

3. **Documentation** - [Read Docs](https://docs.trading-orchestrator.com)
   - Comprehensive guides
   - API reference
   - Examples and tutorials

4. **Email Support** - [Contact Support](mailto:support@trading-orchestrator.com)
   - Technical issues
   - Business inquiries
   - Custom development

### Reporting Bugs

When reporting bugs, include:

```markdown
## Bug Description
Brief description of the issue

## Steps to Reproduce
1. Step one
2. Step two
3. See error

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: [e.g., Ubuntu 20.04]
- Python: [e.g., 3.9.7]
- Version: [e.g., v1.0.0]

## Logs
```
[Paste relevant log entries]
```

## Configuration
```json
[Paste relevant config sections]
```
```

### Emergency Contacts

For critical system failures:

- **Emergency Email**: emergency@trading-orchestrator.com
- **Discord**: @Moderator in #emergency channel
- **Phone**: +1-XXX-XXX-XXXX (24/7 for enterprise customers)

## Prevention Tips

### Regular Maintenance

```bash
# Weekly maintenance script
#!/bin/bash
echo "=== Weekly Maintenance ==="
python health_check.py
python -c "import sqlite3; c=sqlite3.connect('trading_orchestrator.db'); c.execute('VACUUM'); print('Database optimized')"
pip list --outdated --format=json | python -c "import sys,json; [print(f'Update {p[\"name\"]}') for p in json.load(sys.stdin)]"
tail -1000 logs/trading_orchestrator.log > logs/weekly_summary.log
```

### Backup Strategy

```bash
# Daily backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backups/$DATE"
mkdir -p $BACKUP_DIR

# Backup database
cp trading_orchestrator.db $BACKUP_DIR/

# Backup configuration
cp config.json $BACKUP_DIR/
cp .env $BACKUP_DIR/

# Backup logs
tar -czf $BACKUP_DIR/logs.tar.gz logs/

# Keep only last 30 days of backups
find backups/ -type d -mtime +30 -exec rm -rf {} \;
```

### Security Checklist

- [ ] API keys stored in environment variables
- [ ] Regular password rotation
- [ ] 2FA enabled on all broker accounts
- [ ] IP whitelisting configured where possible
- [ ] SSL certificates up to date
- [ ] System packages regularly updated
- [ ] Firewall configured appropriately
- [ ] Regular security audits performed

Remember: Prevention is better than cure! Regular maintenance and monitoring can prevent most issues from occurring.