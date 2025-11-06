# Troubleshooting Guide

## Table of Contents
1. [Overview](#overview)
2. [Getting Started with Troubleshooting](#getting-started-with-troubleshooting)
3. [Connection Issues](#connection-issues)
4. [API Errors](#api-errors)
5. [Performance Problems](#performance-problems)
6. [Database Issues](#database-issues)
7. [LLM Provider Problems](#llm-provider-problems)
8. [Authentication Issues](#authentication-issues)
9. [Trading System Errors](#trading-system-errors)
10. [System-Level Issues](#system-level-issues)
11. [Log Analysis Guide](#log-analysis-guide)
12. [Diagnostic Tools](#diagnostic-tools)
13. [Emergency Procedures](#emergency-procedures)

## Overview

This troubleshooting guide helps you quickly identify and resolve common issues with the day trading orchestrator. Use the diagnostic steps systematically to minimize downtime and maintain system reliability.

### Before You Start
- [ ] Check system status using health check endpoints
- [ ] Review recent logs for error patterns
- [ ] Verify network connectivity
- [ ] Check system resource utilization
- [ ] Consult the emergency procedures for critical issues

### Troubleshooting Hierarchy
1. **Immediate Checks**: System status, connectivity, basic functionality
2. **Log Analysis**: Application logs, system logs, error patterns
3. **Configuration Review**: Environment variables, API keys, settings
4. **Service Restart**: Restart affected services in order
5. **Escalation**: Contact technical support if issue persists

---

## Getting Started with Troubleshooting

### Quick System Check
Run these commands to get a quick overview of system health:

```bash
# Check system status
curl -f http://localhost:8000/health

# Check service status
sudo systemctl status trading-orchestrator

# Check system resources
htop
df -h
free -m

# Check recent errors
journalctl -u trading-orchestrator --since "1 hour ago"
```

### Log Locations
- **Application Logs**: `/var/log/trading-orchestrator/app.log`
- **Error Logs**: `/var/log/trading-orchestrator/error.log`
- **System Logs**: `/var/log/syslog`
- **Database Logs**: `/var/log/postgresql/`
- **Web Server Logs**: `/var/log/nginx/`

### Common Error Indicators
| Indicator | Meaning | Immediate Action |
|-----------|---------|------------------|
| 503 Service Unavailable | Service not responding | Check service status and restart |
| 401 Unauthorized | Authentication failed | Check API keys and credentials |
| 429 Too Many Requests | Rate limiting | Wait and check rate limits |
| 500 Internal Server Error | Application error | Check application logs |
| Connection refused | Service not running | Start the service |

---

## Connection Issues

### Broker API Connection Problems

#### Symptoms
- Unable to connect to broker APIs
- Frequent disconnections
- Timeout errors on API calls
- Authentication failures

#### Diagnostic Steps

**Step 1: Verify Network Connectivity**
```bash
# Test basic connectivity
ping api.alpaca.markets
ping api.binance.us
ping ibkr.api.platform.com

# Check DNS resolution
nslookup api.alpaca.markets
nslookup api.binance.us

# Test port connectivity
telnet api.alpaca.markets 443
telnet api.binance.us 443
```

**Step 2: Verify API Credentials**
```bash
# Check environment variables
grep -r "ALPACA_" ~/.bashrc
grep -r "BINANCE_" ~/.bashrc

# Test API connectivity
curl -H "APCA-API-KEY-ID: $ALPACA_KEY" \
     -H "APCA-API-SECRET-KEY: $ALPACA_SECRET" \
     https://paper-api.alpaca.markets/v2/account
```

**Step 3: Check API Rate Limits**
```python
# Check rate limit headers in responses
import requests

response = requests.get("https://api.alpaca.markets/v2/account")
print("Rate Limit:", response.headers.get('X-RateLimit-Limit'))
print("Rate Limit Remaining:", response.headers.get('X-RateLimit-Remaining'))
print("Rate Limit Reset:", response.headers.get('X-RateLimit-Reset'))
```

#### Common Solutions

**Network Firewall Issues**
- [ ] Allow outbound HTTPS (port 443) to broker APIs
- [ ] Configure proxy settings if behind corporate firewall
- [ ] Check VPN connectivity for remote access
- [ ] Verify DNS settings for broker domains

**API Authentication Problems**
- [ ] Verify API key format (no extra spaces/characters)
- [ ] Check API key permissions (trading, data, account access)
- [ ] Ensure API key is not expired or revoked
- [ ] Test with sandbox/paper trading environment first

**Connection Timeout Issues**
- [ ] Increase timeout settings in configuration
- [ ] Check network latency to broker servers
- [ ] Verify SSL/TLS certificate validity
- [ ] Consider using connection pooling

### Database Connection Issues

#### Symptoms
- "Connection refused" errors
- "Too many connections" errors
- Slow database queries
- Intermittent connection failures

#### Diagnostic Steps

**Step 1: Check Database Status**
```bash
# PostgreSQL
sudo systemctl status postgresql
sudo -u postgres psql -c "SELECT version();"

# SQLite
ls -la /path/to/database.db
sqlite3 /path/to/database.db ".tables"
```

**Step 2: Test Database Connectivity**
```python
from sqlalchemy import create_engine
import traceback

try:
    engine = create_engine("postgresql://user:pass@localhost/dbname")
    connection = engine.connect()
    print("Database connection successful")
    connection.close()
except Exception as e:
    print(f"Database connection failed: {e}")
    traceback.print_exc()
```

**Step 3: Check Connection Pool Settings**
```python
# Check connection pool configuration
from sqlalchemy import create_engine

engine = create_engine(
    "postgresql://user:pass@localhost/dbname",
    pool_size=10,           # Max connections
    max_overflow=20,        # Overflow connections
    pool_timeout=30,        # Timeout for connection
    pool_recycle=3600,      # Recycle connections after 1 hour
    pool_pre_ping=True      # Validate connections
)
```

#### Common Solutions

**Connection Pool Exhaustion**
- [ ] Increase `max_connections` in database configuration
- [ ] Optimize connection pool settings
- [ ] Close unused database connections
- [ ] Implement connection timeout handling

**Database Performance Issues**
- [ ] Analyze slow queries with `EXPLAIN ANALYZE`
- [ ] Add missing database indexes
- [ ] Optimize query patterns
- [ ] Consider database maintenance (VACUUM, ANALYZE)

---

## API Errors

### HTTP Error Codes

#### 4xx Client Errors

**401 Unauthorized**
```
Error: Authentication failed
Cause: Invalid API key, expired token, or missing credentials
```
**Solutions:**
- [ ] Verify API key is correct and active
- [ ] Check API key permissions
- [ ] Ensure API key format is correct
- [ ] Regenerate API key if compromised

**403 Forbidden**
```
Error: Access denied
Cause: Insufficient permissions for requested action
```
**Solutions:**
- [ ] Check API key permissions
- [ ] Verify account status
- [ ] Ensure trading permissions are enabled
- [ ] Check IP whitelist configuration

**404 Not Found**
```
Error: Resource not found
Cause: Invalid endpoint or resource ID
```
**Solutions:**
- [ ] Verify API endpoint URL
- [ ] Check resource ID format
- [ ] Ensure correct HTTP method (GET, POST, PUT)
- [ ] Review API documentation

**429 Too Many Requests**
```
Error: Rate limit exceeded
Cause: API rate limit exceeded
```
**Solutions:**
- [ ] Implement rate limiting in application
- [ ] Add delays between API calls
- [ ] Upgrade to higher rate limit tier
- [ ] Use bulk operations where available

#### 5xx Server Errors

**500 Internal Server Error**
```
Error: Server-side error
Cause: Broker API issue or invalid request format
```
**Solutions:**
- [ ] Check request format and parameters
- [ ] Retry with exponential backoff
- [ ] Contact broker API support
- [ ] Monitor broker status page

**502 Bad Gateway**
```
Error: Invalid response from upstream server
Cause: Broker API unavailable or network issue
```
**Solutions:**
- [ ] Check broker API status
- [ ] Verify network connectivity
- [ ] Implement retry logic
- [ ] Use fallback broker if configured

**503 Service Unavailable**
```
Error: Service temporarily unavailable
Cause: Broker maintenance or overload
```
**Solutions:**
- [ ] Check broker status page
- [ ] Implement exponential backoff retry
- [ ] Set up monitoring for broker availability
- [ ] Consider using multiple brokers

### Specific Broker API Issues

#### Alpaca API Issues

**Paper Trading vs Live Trading**
```python
# Check current environment
import alpaca_trade_api as tradeapi

api = tradeapi.REST(
    key_id='your_key',
    secret_key='your_secret',
    base_url='https://paper-api.alpaca.markets'  # Paper trading
    # base_url='https://api.alpaca.markets'     # Live trading
)

account = api.get_account()
print(f"Account status: {account.status}")
```

**Common Issues:**
- [ ] Wrong base URL for environment (paper vs live)
- [ ] Insufficient buying power for orders
- [ ] Market data subscription not configured
- [ ] Order symbol not available in chosen market

#### Binance API Issues

**API Key Configuration**
```python
import binance
from binance.client import Client

# Check API key configuration
client = Client(api_key, api_secret, testnet=True)  # Use testnet for testing
account_info = client.get_account()
print(f"Account type: {account_info['accountType']}")
```

**Common Issues:**
- [ ] Wrong API key (testnet vs mainnet)
- [ ] Insufficient balance for orders
- [ ] Symbol not available for trading
- [ ] Withdrawal permissions not granted

---

## Performance Problems

### Slow Response Times

#### Symptoms
- API calls taking >5 seconds
- User interface sluggish
- Database queries slow
- High CPU/Memory usage

#### Diagnostic Steps

**Step 1: Identify Bottlenecks**
```bash
# Check system performance
top -p $(pgrep -f trading-orchestrator)

# Check I/O performance
iotop

# Check network performance
netstat -i
ss -tuln

# Check database performance
sudo -u postgres psql -c "SELECT * FROM pg_stat_activity;"
```

**Step 2: Analyze Application Performance**
```python
import time
import cProfile
import pstats

def profile_function(func):
    pr = cProfile.Profile()
    pr.enable()
    start_time = time.time()
    result = func()
    pr.disable()
    
    # Save profiling results
    pr.dump_stats('performance_profile.prof')
    
    # Display top time consumers
    stats = pstats.Stats('performance_profile.prof')
    stats.sort_stats('cumulative')
    stats.print_stats(10)
    
    return result
```

**Step 3: Database Performance Analysis**
```sql
-- Check slow queries
SELECT query, mean_time, calls, total_time
FROM pg_stat_statements
ORDER BY total_time DESC
LIMIT 10;

-- Check database locks
SELECT blocked_locks.pid AS blocked_pid,
       blocked_activity.usename AS blocked_user,
       blocking_locks.pid AS blocking_pid,
       blocking_activity.usename AS blocking_user,
       blocked_activity.query AS blocked_statement,
       blocking_activity.query AS blocking_statement
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks ON blocking_locks.locktype = blocked_locks.locktype
JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.granted;
```

#### Common Performance Solutions

**Slow Database Queries**
- [ ] Add appropriate indexes for frequently queried columns
- [ ] Optimize JOIN operations
- [ ] Use query result caching
- [ ] Implement connection pooling
- [ ] Regular database maintenance (VACUUM, ANALYZE)

**High Memory Usage**
- [ ] Profile memory usage with memory_profiler
- [ ] Implement object pooling for expensive objects
- [ ] Use generators for large datasets
- [ ] Optimize data structures
- [ ] Implement garbage collection tuning

**Slow API Calls**
- [ ] Implement request caching
- [ ] Use async/await for I/O operations
- [ ] Optimize request frequency
- [ ] Implement request batching
- [ ] Use persistent connections

### High Resource Usage

#### CPU Optimization
```python
import multiprocessing
import threading

# Use multiprocessing for CPU-intensive tasks
def cpu_intensive_task(data):
    # CPU-intensive processing
    return processed_data

# Use threading for I/O-bound tasks
import requests

def io_bound_task():
    # I/O operations
    response = requests.get('https://api.example.com/data')
    return response.json()
```

#### Memory Management
```python
import gc
import tracemalloc

# Enable memory tracing
tracemalloc.start()

# Manual garbage collection for large objects
def cleanup_large_objects():
    gc.collect()

# Use context managers for resource cleanup
with open('large_file.txt', 'r') as f:
    data = f.read()
    # Process data
    pass  # File automatically closed
```

---

## Database Issues

### Database Connection Problems

#### Symptoms
- "could not connect to server" errors
- Connection timeouts
- Too many connections errors
- "database is being accessed by other users"

#### Diagnostic Steps

**Step 1: Check Database Status**
```bash
# PostgreSQL
sudo systemctl status postgresql
sudo -u postgres psql -c "SELECT version();"

# Check active connections
sudo -u postgres psql -c "SELECT count(*) FROM pg_stat_activity;"
```

**Step 2: Check Connection Limits**
```sql
-- Check current connections
SELECT count(*) FROM pg_stat_activity;

-- Check max connections setting
SHOW max_connections;

-- Check connection limits by user
SELECT usename, count(*) 
FROM pg_stat_activity 
GROUP BY usename;
```

#### Solutions

**Connection Limit Issues**
```sql
-- Increase max connections (requires restart)
ALTER SYSTEM SET max_connections = '200';
SELECT pg_reload_conf();
```

**Database Lock Issues**
```sql
-- Find blocking queries
SELECT 
    blocked_locks.pid AS blocked_pid,
    blocked_activity.usename AS blocked_user,
    blocking_locks.pid AS blocking_pid,
    blocking_activity.usename AS blocking_user,
    blocked_activity.query AS blocked_statement,
    blocking_activity.query AS blocking_statement
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks ON blocking_locks.locktype = blocked_locks.locktype
JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.granted;
```

### Data Integrity Issues

#### Database Corruption
```bash
# PostgreSQL integrity check
sudo -u postgres psql -c "SELECT pg_database_size('database_name');"
sudo -u postgres psql -c "VACUUM ANALYZE;"

# Check for corruption
sudo -u postgres psql -c "SELECT * FROM pg_stat_database WHERE datname = 'database_name';"
```

#### Migration Issues
```bash
# Check migration status
python manage.py db current
python manage.py db history

# Apply pending migrations
python manage.py db upgrade

# Rollback migration (if needed)
python manage.py db downgrade
```

---

## LLM Provider Problems

### OpenAI API Issues

#### Common Error Patterns
```python
import openai
import time

def handle_openai_error(error):
    if isinstance(error, openai.error.RateLimitError):
        print("Rate limit exceeded. Implement exponential backoff.")
        time.sleep(60)  # Wait 1 minute
    elif isinstance(error, openai.error.APIError):
        print(f"OpenAI API error: {error}")
    elif isinstance(error, openai.error.AuthenticationError):
        print("Authentication failed. Check API key.")
    elif isinstance(error, openai.error.InvalidRequestError):
        print("Invalid request. Check parameters.")
```

#### Solutions
- [ ] Implement exponential backoff for rate limits
- [ ] Cache API responses to reduce calls
- [ ] Use appropriate model for task complexity
- [ ] Monitor token usage and costs
- [ ] Implement fallback providers

### Anthropic Claude API Issues

#### Authentication and Configuration
```python
import anthropic

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

# Check API key validity
try:
    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=100,
        messages=[{"role": "user", "content": "Hello"}]
    )
    print("API key valid")
except Exception as e:
    print(f"API error: {e}")
```

### LLM Performance Optimization

#### Response Caching
```python
import hashlib
import json
from functools import lru_cache

def cache_key(prompt, model, max_tokens, temperature):
    content = f"{prompt}_{model}_{max_tokens}_{temperature}"
    return hashlib.md5(content.encode()).hexdigest()

# Simple in-memory cache
llm_cache = {}

def get_llm_response(prompt, **kwargs):
    cache_key = cache_key(prompt, **kwargs)
    if cache_key in llm_cache:
        return llm_cache[cache_key]
    
    # Make API call and cache result
    response = make_api_call(prompt, **kwargs)
    llm_cache[cache_key] = response
    return response
```

---

## Authentication Issues

### JWT Token Problems

#### Token Expiration
```python
import jwt
import datetime

def check_token_expiration(token):
    try:
        payload = jwt.decode(token, secret_key, algorithms=["HS256"])
        exp = datetime.datetime.fromtimestamp(payload['exp'])
        now = datetime.datetime.now()
        
        if exp < now:
            print("Token expired")
            return False
        else:
            print(f"Token expires in: {exp - now}")
            return True
    except jwt.ExpiredSignatureError:
        print("Token signature expired")
        return False
    except jwt.InvalidTokenError:
        print("Invalid token")
        return False
```

### API Key Issues

#### Environment Variable Problems
```bash
# Check if environment variables are set
echo $ALPACA_KEY_ID
echo $ALPACA_SECRET_KEY
echo $OPENAI_API_KEY

# Verify they're not empty
if [ -z "$ALPACA_KEY_ID" ]; then
    echo "ALPACA_KEY_ID is not set"
fi
```

---

## Trading System Errors

### Order Placement Failures

#### Common Error Types
```python
def handle_order_error(error_message):
    error_mapping = {
        "insufficient_buying_power": "Increase account balance or reduce order size",
        "symbol_not_found": "Verify symbol exists on broker",
        "market_hours": "Check if market is open",
        "price_protection": "Price protection triggered",
        "rate_limit": "Too many orders, slow down"
    }
    
    for error_type, solution in error_mapping.items():
        if error_type in error_message.lower():
            return f"Error: {solution}"
    
    return "Unknown error, check broker API documentation"
```

### Risk Management Issues

#### Position Limits
```python
def check_position_limits(orders):
    max_position_size = 10000  # $10,000
    current_positions = get_current_positions()
    
    for order in orders:
        if order.quantity * order.price > max_position_size:
            raise ValueError(f"Order exceeds position limit of ${max_position_size}")
    
    return True
```

---

## System-Level Issues

### Process Management

#### Service Restart
```bash
# Restart trading orchestrator service
sudo systemctl restart trading-orchestrator

# Check service status
sudo systemctl status trading-orchestrator

# Check logs for errors
sudo journalctl -u trading-orchestrator -f
```

#### Process Monitoring
```bash
# Check if process is running
ps aux | grep trading-orchestrator

# Check process tree
pstree -p $(pgrep -f trading-orchestrator)

# Monitor resource usage
top -p $(pgrep -f trading-orchestrator)
```

### File System Issues

#### Disk Space
```bash
# Check disk space
df -h

# Check large files
find /var/log -name "*.log" -size +100M -exec ls -lh {} \;

# Clean up old logs
find /var/log -name "*.log" -mtime +30 -delete
```

#### Permission Issues
```bash
# Check file permissions
ls -la /opt/trading-orchestrator/

# Fix permissions
sudo chown -R trading-orchestrator:trading-orchestrator /opt/trading-orchestrator/
sudo chmod +x /opt/trading-orchestrator/main.py
```

---

## Log Analysis Guide

### Log Patterns to Watch For

#### Error Patterns
```bash
# Find error patterns in logs
grep -i "error" /var/log/trading-orchestrator/app.log | tail -20
grep -i "exception" /var/log/trading-orchestrator/error.log | tail -20
grep -i "failed" /var/log/trading-orchestrator/app.log | tail -20
```

#### Performance Patterns
```bash
# Find slow operations
grep -i "slow" /var/log/trading-orchestrator/app.log | tail -10
grep -i "timeout" /var/log/trading-orchestrator/app.log | tail -10
```

#### Security Patterns
```bash
# Find authentication failures
grep -i "unauthorized" /var/log/trading-orchestrator/app.log
grep -i "forbidden" /var/log/trading-orchestrator/app.log
```

### Log Analysis Tools

#### Using `journalctl` for System Logs
```bash
# Filter by time range
journalctl --since "2024-01-01 10:00:00" --until "2024-01-01 12:00:00"

# Filter by service
journalctl -u trading-orchestrator --since "1 hour ago"

# Follow logs in real-time
journalctl -f -u trading-orchestrator
```

#### Using `grep` for Pattern Matching
```bash
# Find specific error codes
grep -E "ERROR|CRITICAL|FATAL" /var/log/trading-orchestrator/app.log

# Find API call patterns
grep -E "API_CALL|BROKER_REQUEST" /var/log/trading-orchestrator/app.log

# Count error types
grep -i "error" /var/log/trading-orchestrator/app.log | cut -d' ' -f3 | sort | uniq -c | sort -nr
```

---

## Diagnostic Tools

### System Information
```bash
#!/bin/bash
# System diagnostic script

echo "=== System Information ==="
uname -a
uptime
free -h
df -h
ps aux | grep trading-orchestrator | grep -v grep

echo "=== Network Connectivity ==="
ping -c 3 8.8.8.8
nslookup api.alpaca.markets

echo "=== Service Status ==="
systemctl status postgresql
systemctl status nginx

echo "=== Recent Errors ==="
journalctl -u trading-orchestrator --since "1 hour ago" | tail -20
```

### Performance Monitoring
```bash
#!/bin/bash
# Performance monitoring script

echo "=== CPU Usage ==="
top -bn1 | head -20

echo "=== Memory Usage ==="
free -m

echo "=== Disk I/O ==="
iostat -x 1 3

echo "=== Network Connections ==="
ss -tuln

echo "=== Database Connections ==="
sudo -u postgres psql -c "SELECT count(*) FROM pg_stat_activity;"
```

### Database Diagnostics
```sql
-- Database health check script
SELECT 
    'Database Size' as metric,
    pg_size_pretty(pg_database_size(current_database())) as value
UNION ALL
SELECT 
    'Active Connections',
    count(*)::text
FROM pg_stat_activity
UNION ALL
SELECT 
    'Slow Queries',
    count(*)::text
FROM pg_stat_statements
WHERE mean_time > 1000;
```

---

## Emergency Procedures

### System Down Procedure

**Immediate Actions (0-5 minutes)**
1. [ ] Check if service is running: `systemctl status trading-orchestrator`
2. [ ] Check system resources: `top`, `df -h`
3. [ ] Check recent logs: `journalctl -u trading-orchestrator --since "5 minutes ago"`
4. [ ] Attempt service restart: `systemctl restart trading-orchestrator`

**If Service Won't Start**
1. [ ] Check configuration files for syntax errors
2. [ ] Verify environment variables are set
3. [ ] Check database connectivity
4. [ ] Review error logs in detail

**If Database is Unavailable**
1. [ ] Check database service: `systemctl status postgresql`
2. [ ] Check database connectivity: `psql -U postgres -c "SELECT 1;"`
3. [ ] Check database logs: `/var/log/postgresql/`
4. [ ] Restore from backup if database is corrupted

### Data Loss Prevention

**If Data Appears Lost**
1. [ ] Stop all services immediately
2. [ ] Check for database corruption: `pg_dump --schema-only`
3. [ ] Review recent transactions in logs
4. [ ] Restore from latest backup
5. [ ] Verify data integrity after restore

### Security Incident Response

**If Security Breach Suspected**
1. [ ] Disconnect system from network
2. [ ] Preserve all logs and evidence
3. [ ] Change all API keys and passwords
4. [ ] Review access logs for unauthorized access
5. [ ] Contact security team immediately

### Communication Template

**Status Update Template:**
```
Subject: [SERVICE NAME] - Service Status Update

Current Status: [UP/DOWN/DEGRADED]
Issue: [Brief description]
Impact: [Affected services/users]
ETA for Resolution: [Estimated time]
Next Update: [Time for next update]
Contact: [On-call engineer]
```

---

## Quick Reference Cards

### Emergency Contacts
| Issue Type | Contact | Phone | Email |
|------------|---------|--------|--------|
| System Down | On-call Engineer | [Number] | [Email] |
| Database Issues | DBA | [Number] | [Email] |
| Security Issues | Security Team | [Number] | [Email] |
| Broker API Issues | Trading Team | [Number] | [Email] |

### Common Commands
```bash
# Service Management
sudo systemctl start trading-orchestrator
sudo systemctl stop trading-orchestrator
sudo systemctl restart trading-orchestrator
sudo systemctl status trading-orchestrator

# Log Monitoring
tail -f /var/log/trading-orchestrator/app.log
journalctl -f -u trading-orchestrator
grep -i error /var/log/trading-orchestrator/app.log | tail -20

# Database Operations
sudo -u postgres psql -c "SELECT * FROM pg_stat_activity;"
sudo -u postgres psql -c "VACUUM ANALYZE;"

# System Health
htop
df -h
free -m
netstat -tuln
```

### URL Quick Reference
- Health Check: `http://localhost:8000/health`
- API Documentation: `http://localhost:8000/docs`
- Metrics: `http://localhost:8000/metrics`
- Status Dashboard: `http://localhost:8000/status`

---

**Document Version:** 1.0  
**Last Updated:** [Current Date]  
**Next Review Date:** [Review Date]  
**Document Owner:** Technical Support Team  

---

*Keep this guide accessible during operations. Print or save locally for offline reference during system outages.*