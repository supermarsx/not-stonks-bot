# Troubleshooting Wizard

## Table of Contents
- [Quick Diagnostic Procedures](#quick-diagnostic-procedures)
- [Common Issues and Solutions](#common-issues-and-solutions)
- [System Health Monitoring](#system-health-monitoring)
- [Emergency Response Procedures](#emergency-response-procedures)
- [Diagnostic Tools and Utilities](#diagnostic-tools-and-utilities)
- [Log Analysis Guide](#log-analysis-guide)
- [Performance Troubleshooting](#performance-troubleshooting)
- [Network Connectivity Issues](#network-connectivity-issues)
- [Database Troubleshooting](#database-troubleshooting)
- [Broker Connection Issues](#broker-connection-issues)
- [Strategy Execution Problems](#strategy-execution-problems)
- [Risk Management Alerts](#risk-management-alerts)
- [API and WebSocket Issues](#api-and-websocket-issues)
- [Plugin System Problems](#plugin-system-problems)
- [UI and Dashboard Issues](#ui-and-dashboard-issues)
- [Data Feed Problems](#data-feed-problems)
- [Backtesting Troubleshooting](#backtesting-troubleshooting)
- [Report Generation Issues](#report-generation-issues)
- [Automated Testing Problems](#automated-testing-problems)
- [Deployment and Configuration Issues](#deployment-and-configuration-issues)
- [Monitoring and Alerting](#monitoring-and-alerting)
- [Recovery Procedures](#recovery-procedures)

## Quick Diagnostic Procedures

### System Health Check
```bash
# Run comprehensive system health check
./scripts/system-health-check.sh

# Check service status
systemctl status trading-orchestrator
systemctl status postgres
systemctl status redis
systemctl status nginx

# Check disk space
df -h

# Check memory usage
free -h

# Check CPU usage
top -bn1 | head -20
```

### Connectivity Test
```bash
# Test broker connectivity
curl -k https://paper-api.alpaca.markets/v2/account
curl -k https://api.binance.com/api/v3/ping

# Test database connectivity
psql -h localhost -U trading_user -d trading_db -c "SELECT 1;"

# Test Redis connectivity
redis-cli ping

# Test external APIs
curl -s https://api.polygon.io/v2/aggs/ticker/AAPL/prev
```

### Log Review
```bash
# Review recent errors
grep -i error /var/log/trading-orchestrator/app.log | tail -20

# Check for critical alerts
grep -i critical /var/log/trading-orchestrator/app.log | tail -10

# Review broker connection logs
grep -i "broker\|connection\|alpaca\|binance" /var/log/trading-orchestrator/app.log | tail -15

# Check for performance issues
grep -i "slow\|timeout\|latency" /var/log/trading-orchestrator/app.log | tail -15
```

## Common Issues and Solutions

### Issue: System Won't Start

**Symptoms:**
- Service fails to start
- Port already in use errors
- Configuration file errors

**Diagnostic Steps:**
1. Check system requirements
2. Verify configuration files
3. Check port availability
4. Review startup logs

**Solutions:**
```bash
# Check port availability
netstat -tulpn | grep :8080

# Kill processes on required ports
sudo kill -9 $(lsof -ti:8080)

# Validate configuration files
./scripts/validate-config.sh

# Start with debug logging
LOG_LEVEL=DEBUG ./bin/trading-orchestrator
```

### Issue: High CPU/Memory Usage

**Symptoms:**
- System performance degradation
- High resource consumption alerts
- Slow response times

**Diagnostic Steps:**
1. Identify resource-intensive processes
2. Check for memory leaks
3. Review thread/connection pools
4. Analyze performance metrics

**Solutions:**
```bash
# Monitor system resources
top -o %CPU
top -o %MEM

# Check for memory leaks
ps aux --sort=-%mem | head -10

# Restart service to clear memory
systemctl restart trading-orchestrator

# Optimize configuration
./scripts/optimize-performance.sh
```

### Issue: Database Connection Failures

**Symptoms:**
- "Connection refused" errors
- Timeout errors
- Authentication failures

**Diagnostic Steps:**
1. Test database connectivity
2. Check authentication credentials
3. Review connection pool settings
4. Examine network connectivity

**Solutions:**
```bash
# Test database connection
psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c "SELECT version();"

# Check database status
sudo systemctl status postgresql

# Test connection with detailed output
psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c "\conninfo"

# Restart database service
sudo systemctl restart postgresql
```

### Issue: Broker Connection Problems

**Symptoms:**
- Authentication failures
- Rate limit errors
- Market data issues
- Order execution failures

**Diagnostic Steps:**
1. Verify API credentials
2. Check account status
3. Test API connectivity
4. Review rate limit usage

**Solutions:**
```bash
# Test Alpaca connection
curl -H "APCA-API-KEY-ID: $ALPACA_KEY" \
     -H "APCA-API-SECRET-KEY: $ALPACA_SECRET" \
     https://paper-api.alpaca.markets/v2/account

# Test Binance connection
curl -H "X-MBX-APIKEY: $BINANCE_API_KEY" \
     https://api.binance.com/api/v3/account

# Check rate limit headers
curl -I https://paper-api.alpaca.markets/v2/account
```

## System Health Monitoring

### Real-time Monitoring

**CPU and Memory Usage:**
```bash
# Monitor CPU usage continuously
watch -n 1 'top -bn1 | head -20'

# Monitor memory usage
watch -n 1 'free -h'

# Monitor disk I/O
watch -n 1 'iostat -x 1'
```

**Network Connectivity:**
```bash
# Monitor network connections
watch -n 1 'ss -tuln'

# Monitor bandwidth usage
ifconfig eth0

# Test latency to key endpoints
for host in api.alpaca.markets api.binance.com api.polygon.io; do
    ping -c 3 $host
done
```

**Database Health:**
```bash
# Monitor database connections
psql -h localhost -U trading_user -d trading_db -c "
    SELECT count(*) as active_connections 
    FROM pg_stat_activity 
    WHERE state = 'active';
"

# Check database performance
psql -h localhost -U trading_user -d trading_db -c "
    SELECT query, mean_time, calls 
    FROM pg_stat_statements 
    ORDER BY mean_time DESC 
    LIMIT 10;
"
```

### Automated Health Checks

**Create health check script:**
```bash
#!/bin/bash
# /opt/trading-orchestrator/scripts/health-check.sh

echo "=== Trading Orchestrator Health Check ==="
echo "Timestamp: $(date)"
echo

# Check service status
echo "--- Service Status ---"
systemctl is-active trading-orchestrator || echo "Service not running"
systemctl is-active postgresql || echo "PostgreSQL not running"
systemctl is-active redis || echo "Redis not running"
systemctl is-active nginx || echo "Nginx not running"

echo

# Check system resources
echo "--- System Resources ---"
echo "CPU Usage: $(top -bn1 | grep 'Cpu(s)' | awk '{print $2}' | cut -d'%' -f1)%"
echo "Memory Usage: $(free | grep Mem | awk '{printf("%.1f%%"), $3/$2 * 100.0}')"
echo "Disk Usage: $(df -h / | awk 'NR==2{printf "%s", $5}')"

echo

# Check network connectivity
echo "--- Network Connectivity ---"
for endpoint in "https://paper-api.alpaca.markets" "https://api.binance.com" "https://api.polygon.io"; do
    if curl -s --connect-timeout 5 "$endpoint" > /dev/null; then
        echo "✓ $endpoint"
    else
        echo "✗ $endpoint"
    fi
done

echo

# Check database connectivity
echo "--- Database Connectivity ---"
if psql -h localhost -U trading_user -d trading_db -c "SELECT 1;" > /dev/null 2>&1; then
    echo "✓ Database connection successful"
else
    echo "✗ Database connection failed"
fi

echo

# Check recent errors
echo "--- Recent Errors (Last 10 minutes) ---"
find /var/log/trading-orchestrator -name "*.log" -mmin -10 -exec grep -i error {} \; | wc -l
```

**Schedule health checks:**
```bash
# Add to crontab
*/5 * * * * /opt/trading-orchestrator/scripts/health-check.sh >> /var/log/trading-orchestrator/health-check.log 2>&1
```

## Emergency Response Procedures

### System Failure Response

**Immediate Actions:**
1. Stop all trading activities
2. Cancel pending orders
3. Close open positions if necessary
4. Document the incident
5. Notify stakeholders

**Emergency Commands:**
```bash
# Emergency stop all trading
curl -X POST http://localhost:8080/api/v1/emergency/stop

# Cancel all pending orders
curl -X POST http://localhost:8080/api/v1/orders/cancel/all

# Close all positions (use with caution)
curl -X POST http://localhost:8080/api/v1/positions/close/all

# Check system status
curl http://localhost:8080/api/v1/status
```

### Data Loss Recovery

**Database Backup Restoration:**
```bash
# Stop the application
sudo systemctl stop trading-orchestrator

# Restore from latest backup
sudo -u postgres pg_restore -d trading_db /backups/trading_db_$(date +%Y%m%d_%H%M%S).backup

# Verify restoration
psql -h localhost -U trading_user -d trading_db -c "SELECT count(*) FROM trades;"

# Start the application
sudo systemctl start trading-orchestrator
```

### Security Incident Response

**Compromised Credentials:**
```bash
# Immediately revoke API keys (via broker dashboard)
# Change all passwords
# Review access logs
# Enable additional authentication

# Review system access
last -n 20
who
ps aux | grep -E 'ssh|scp|sftp'

# Check for unauthorized processes
ps aux --sort=-%cpu | head -20
netstat -tulpn
```

## Diagnostic Tools and Utilities

### Custom Diagnostic Scripts

**System Information Script:**
```bash
#!/bin/bash
# /opt/trading-orchestrator/scripts/system-info.sh

echo "=== Trading Orchestrator System Information ==="
echo "Hostname: $(hostname)"
echo "OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"
echo "Kernel: $(uname -r)"
echo "Uptime: $(uptime)"
echo

echo "--- System Resources ---"
echo "CPU: $(lscpu | grep 'Model name' | cut -d: -f2 | xargs)"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "Disk: $(df -h / | awk 'NR==2{print $2}')"
echo

echo "--- Network Configuration ---"
ip addr show | grep -E 'inet ' | awk '{print $2}' | head -5
echo

echo "--- Service Status ---"
systemctl list-units --type=service --state=active | grep trading
```

**Performance Analysis Script:**
```bash
#!/bin/bash
# /opt/trading-orchestrator/scripts/performance-analysis.sh

echo "=== Performance Analysis ==="
echo "Timestamp: $(date)"
echo

# CPU analysis
echo "--- CPU Analysis ---"
echo "CPU Usage: $(top -bn1 | grep 'Cpu(s)' | awk '{print $2}' | cut -d'%' -f1)%"
echo "Load Average: $(uptime | awk -F'load average:' '{print $2}')"

# Memory analysis
echo "--- Memory Analysis ---"
free -h

# Disk analysis
echo "--- Disk Analysis ---"
df -h

# Network analysis
echo "--- Network Analysis ---"
ss -tuln | wc -l | xargs echo "Total Network Connections:"
netstat -i | tail -n +3 | awk '{print $1 " RX bytes:" $3 " TX bytes:" $9}'

# Process analysis
echo "--- Process Analysis ---"
echo "Top 5 CPU consumers:"
ps aux --sort=-%cpu | head -6
echo
echo "Top 5 Memory consumers:"
ps aux --sort=-%mem | head -6
```

**Database Diagnostic Script:**
```bash
#!/bin/bash
# /opt/trading-orchestrator/scripts/database-diagnostics.sh

echo "=== Database Diagnostics ==="
echo "Timestamp: $(date)"
echo

# Connection info
echo "--- Connection Information ---"
psql -h localhost -U trading_user -d trading_db -c "
    SELECT 
        count(*) as total_connections,
        count(*) FILTER (WHERE state = 'active') as active_connections,
        count(*) FILTER (WHERE state = 'idle') as idle_connections
    FROM pg_stat_activity;
"

# Database size
echo "--- Database Size ---"
psql -h localhost -U trading_user -d trading_db -c "
    SELECT 
        pg_size_pretty(pg_database_size(current_database())) as db_size;
"

# Table sizes
echo "--- Table Sizes ---"
psql -h localhost -U trading_user -d trading_db -c "
    SELECT 
        schemaname,
        tablename,
        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
    FROM pg_tables 
    WHERE schemaname = 'public'
    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
"

# Slow queries
echo "--- Slow Queries (if pg_stat_statements enabled) ---"
psql -h localhost -U trading_user -d trading_db -c "
    SELECT 
        query,
        calls,
        total_time,
        mean_time
    FROM pg_stat_statements 
    ORDER BY mean_time DESC 
    LIMIT 5;
" 2>/dev/null || echo "pg_stat_statements not enabled"
```

### Log Analysis Tools

**Error Trend Analysis:**
```bash
#!/bin/bash
# /opt/trading-orchestrator/scripts/error-analysis.sh

LOG_FILE="/var/log/trading-orchestrator/app.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "Log file not found: $LOG_FILE"
    exit 1
fi

echo "=== Error Analysis ==="
echo "Time range: $(head -1 "$LOG_FILE") to $(tail -1 "$LOG_FILE")"
echo

# Count errors by type
echo "--- Error Count by Type ---"
grep -i error "$LOG_FILE" | cut -d' ' -f1-3 | sort | uniq -c | sort -rn | head -10
echo

# Count errors by severity
echo "--- Error Count by Severity ---"
grep -i 'CRITICAL\|ERROR\|WARNING\|INFO' "$LOG_FILE" | cut -d' ' -f1-4 | sort | uniq -c
echo

# Recent error patterns
echo "--- Recent Error Patterns (Last 100 lines) ---"
tail -100 "$LOG_FILE" | grep -i error | cut -d' ' -f1-6 | sort | uniq -c | sort -rn | head -10
echo

# Broker-related errors
echo "--- Broker-Related Errors ---"
grep -i "broker\|alpaca\|binance" "$LOG_FILE" | grep -i error | tail -10
echo

# Database-related errors
echo "--- Database-Related Errors ---"
grep -i "database\|postgres\|sql" "$LOG_FILE" | grep -i error | tail -10
```

## Performance Troubleshooting

### Performance Metrics

**Key Performance Indicators:**
- Order execution latency
- API response times
- Database query performance
- Memory usage patterns
- CPU utilization
- Network latency

**Performance Monitoring Commands:**
```bash
# Monitor order execution latency
tail -f /var/log/trading-orchestrator/app.log | grep -E "order_executed|execution_time"

# Monitor API response times
for i in {1..10}; do
    time curl -s https://paper-api.alpaca.markets/v2/account > /dev/null
    sleep 1
done

# Monitor database performance
psql -h localhost -U trading_user -d trading_db -c "
    SELECT 
        query,
        mean_time,
        calls
    FROM pg_stat_statements 
    WHERE mean_time > 100
    ORDER BY mean_time DESC;
"
```

### Performance Optimization

**Database Optimization:**
```sql
-- Analyze query performance
EXPLAIN ANALYZE SELECT * FROM trades WHERE symbol = 'AAPL';

-- Add indexes for frequently queried columns
CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_trades_timestamp ON trades(timestamp);

-- Vacuum and analyze tables
VACUUM ANALYZE trades;
VACUUM ANALYZE orders;
```

**Application Optimization:**
```bash
# Tune JVM settings for better performance
export JAVA_OPTS="-Xmx4g -Xms2g -XX:+UseG1GC -XX:MaxGCPauseMillis=200"

# Tune connection pool settings
export DB_POOL_SIZE=20
export DB_POOL_MAX_WAIT=5000

# Monitor GC performance
jstat -gc $(pgrep -f trading-orchestrator) 5s
```

## Network Connectivity Issues

### Network Diagnostics

**Basic Connectivity Tests:**
```bash
# Test DNS resolution
nslookup api.alpaca.markets
nslookup api.binance.com

# Test network connectivity
ping -c 5 api.alpaca.markets
ping -c 5 api.binance.com

# Test specific ports
nc -zv api.alpaca.markets 443
nc -zv api.binance.com 443
```

**Advanced Network Analysis:**
```bash
# Monitor network traffic
iftop -i eth0

# Analyze connection states
ss -tuln

# Check firewall rules
sudo iptables -L

# Test TLS/SSL connectivity
openssl s_client -connect api.alpaca.markets:443
```

### Network Performance Issues

**High Latency Solutions:**
```bash
# Test latency to different endpoints
for host in api.alpaca.markets api.binance.com api.polygon.io; do
    echo "Testing $host..."
    time curl -s https://$host > /dev/null
done

# Optimize network configuration
# Check MTU settings
ip link show | grep mtu

# Optimize TCP settings
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.wmem_max=134217728
```

## Database Troubleshooting

### Connection Issues

**Database Connection Problems:**
```bash
# Test basic connectivity
psql -h localhost -U trading_user -d trading_db

# Check authentication
psql -h localhost -U trading_user -d trading_db -c "\conninfo"

# Check connection limits
psql -h localhost -U trading_user -d trading_db -c "
    SHOW max_connections;
"

# Check current connections
psql -h localhost -U trading_user -d trading_db -c "
    SELECT count(*) FROM pg_stat_activity;
"
```

**Performance Issues:**
```sql
-- Identify slow queries
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;

-- Check for locks
SELECT * FROM pg_locks WHERE NOT granted;

-- Analyze table statistics
SELECT schemaname, tablename, n_tup_ins, n_tup_upd, n_tup_del 
FROM pg_stat_user_tables 
ORDER BY n_tup_ins DESC;
```

### Database Maintenance

**Regular Maintenance Tasks:**
```bash
#!/bin/bash
# /opt/trading-orchestrator/scripts/db-maintenance.sh

echo "Starting database maintenance..."

# Vacuum and analyze
sudo -u postgres psql -d trading_db << EOF
VACUUM ANALYZE;
REINDEX DATABASE trading_db;
EOF

# Update table statistics
sudo -u postgres psql -d trading_db << EOF
ANALYZE;
EOF

# Check database integrity
sudo -u postgres pg_dump trading_db > /dev/null
echo "Database integrity check passed"

echo "Database maintenance completed"
```

## Broker Connection Issues

### Alpaca Connection Problems

**API Authentication Issues:**
```bash
# Test API key validity
curl -H "APCA-API-KEY-ID: $ALPACA_KEY" \
     -H "APCA-API-SECRET-KEY: $ALPACA_SECRET" \
     https://paper-api.alpaca.markets/v2/account

# Check account status
curl -H "APCA-API-KEY-ID: $ALPACA_KEY" \
     -H "APCA-API-SECRET-KEY: $ALPACA_SECRET" \
     https://paper-api.alpaca.markets/v2/account

# Test paper trading vs live trading
curl -H "APCA-API-KEY-ID: $ALPACA_KEY" \
     -H "APCA-API-SECRET-KEY: $ALPACA_SECRET" \
     https://api.alpaca.markets/v2/account
```

**Rate Limiting Issues:**
```bash
# Check rate limit headers
curl -I -H "APCA-API-KEY-ID: $ALPACA_KEY" \
     -H "APCA-API-SECRET-KEY: $ALPACA_SECRET" \
     https://paper-api.alpaca.markets/v2/account

# Implement exponential backoff
for i in {1..5}; do
    response=$(curl -s -w "%{http_code}" -H "APCA-API-KEY-ID: $ALPACA_KEY" \
             -H "APCA-API-SECRET-KEY: $ALPACA_SECRET" \
             https://paper-api.alpaca.markets/v2/account)
    
    status_code=${response: -3}
    
    if [ "$status_code" = "200" ]; then
        echo "Request successful"
        break
    else
        echo "Rate limited, waiting $((2**i)) seconds..."
        sleep $((2**i))
    fi
done
```

### Binance Connection Problems

**API Key Issues:**
```bash
# Test Binance API key
curl -H "X-MBX-APIKEY: $BINANCE_API_KEY" \
     https://api.binance.com/api/v3/account

# Check server time (required for trading)
curl https://api.binance.com/api/v3/time

# Test with different environments
export BINANCE_BASE_URL="https://testnet.binance.vision"  # For testing
export BINANCE_BASE_URL="https://api.binance.com"  # For production
```

**Symbol and Trading Issues:**
```bash
# Check symbol availability
curl "https://api.binance.com/api/v3/exchangeInfo?symbol=BTCUSDT"

# Test minimum trade amounts
curl -H "X-MBX-APIKEY: $BINANCE_API_KEY" \
     -X POST \
     -d 'symbol=BTCUSDT&side=BUY&type=MARKET&quantity=0.001' \
     https://api.binance.com/api/v3/order/test
```

## Strategy Execution Problems

### Strategy Logic Issues

**Debug Strategy Execution:**
```bash
# Enable debug logging for strategies
export LOG_LEVEL=DEBUG

# Monitor strategy performance
tail -f /var/log/trading-orchestrator/app.log | grep -E "strategy|signal"

# Check strategy configuration
grep -A 20 "strategy" /etc/trading-orchestrator/config.yaml

# Test strategy logic manually
./scripts/test-strategy.sh --strategy=moving_average_crossover --symbol=AAPL
```

**Signal Generation Issues:**
```bash
# Monitor signal generation
tail -f /var/log/trading-orchestrator/app.log | grep "signal generated"

# Check data feed issues
grep -i "data\|market_data\|price" /var/log/trading-orchestrator/app.log | tail -20

# Verify indicators calculation
grep -i "indicator\|calculation" /var/log/trading-orchestrator/app.log | tail -10
```

### Backtesting Issues

**Backtest Performance Problems:**
```bash
# Check backtest configuration
./scripts/run-backtest.sh --help

# Monitor backtest progress
tail -f /var/log/trading-orchestrator/backtest.log

# Analyze backtest results
grep -i "performance\|return\|sharpe" /var/log/trading-orchestrator/backtest.log

# Check for data quality issues
grep -i "missing\|null\|invalid" /var/log/trading-orchestrator/backtest.log
```

## Risk Management Alerts

### Risk Limit Violations

**Position Size Limits:**
```bash
# Check current positions
curl http://localhost:8080/api/v1/positions

# Check position limits
grep -i "position.*limit" /etc/trading-orchestrator/config.yaml

# Monitor risk metrics
grep -i "risk\|exposure" /var/log/trading-orchestrator/app.log | tail -20
```

**Daily Loss Limits:**
```bash
# Check daily P&L
psql -h localhost -U trading_user -d trading_db -c "
    SELECT 
        DATE(timestamp) as trade_date,
        SUM(CASE WHEN side = 'sell' THEN quantity * price - fees 
                 WHEN side = 'buy' THEN -(quantity * price + fees) 
                 ELSE 0 END) as daily_pnl
    FROM trades 
    WHERE DATE(timestamp) = CURRENT_DATE
    GROUP BY DATE(timestamp);
"

# Check loss limits configuration
grep -i "daily.*loss\|stop.*loss" /etc/trading-orchestrator/config.yaml
```

## API and WebSocket Issues

### REST API Problems

**API Response Issues:**
```bash
# Test API endpoints
curl -v http://localhost:8080/api/v1/status
curl -v http://localhost:8080/api/v1/positions
curl -v http://localhost:8080/api/v1/orders

# Check API error responses
curl -s http://localhost:8080/api/v1/invalid-endpoint | jq .

# Monitor API request/response times
time curl http://localhost:8080/api/v1/status
```

**Authentication Issues:**
```bash
# Test API authentication
curl -H "Authorization: Bearer $API_TOKEN" \
     http://localhost:8080/api/v1/status

# Check token expiration
grep -i "token\|expiration" /var/log/trading-orchestrator/app.log | tail -10
```

### WebSocket Connection Problems

**Market Data Streaming Issues:**
```bash
# Test WebSocket connection
wscat -c "wss://stream.data.alpaca.markets/v2/iex" \
      -H "APCA-API-KEY-ID: $ALPACA_KEY" \
      -H "APCA-API-SECRET-KEY: $ALPACA_SECRET"

# Monitor WebSocket logs
grep -i "websocket\|stream" /var/log/trading-orchestrator/app.log | tail -20

# Check connection quality
grep -i "ping\|pong\|disconnect" /var/log/trading-orchestrator/app.log | tail -15
```

## Plugin System Problems

### Plugin Loading Issues

**Plugin Discovery Problems:**
```bash
# List available plugins
ls -la /opt/trading-orchestrator/plugins/

# Check plugin dependencies
pip list | grep -E "plugin|strategy"

# Verify plugin signatures
grep -i "signature\|verify" /var/log/trading-orchestrator/app.log | tail -10

# Test plugin loading manually
./scripts/test-plugin-loader.sh --plugin=strategy_plugins
```

**Plugin Execution Issues:**
```bash
# Monitor plugin execution
grep -i "plugin.*execute\|strategy.*execute" /var/log/trading-orchestrator/app.log | tail -20

# Check plugin resource usage
ps aux | grep -E "plugin|strategy" | head -10

# Validate plugin configuration
grep -A 10 "plugins:" /etc/trading-orchestrator/config.yaml
```

## UI and Dashboard Issues

### Web Interface Problems

**Dashboard Loading Issues:**
```bash
# Check web server status
systemctl status nginx

# Test web interface accessibility
curl -I http://localhost:8080
curl -I http://localhost:8080/dashboard

# Check frontend build
ls -la /opt/trading-orchestrator/frontend/dist/

# Monitor web server logs
tail -f /var/log/nginx/access.log
tail -f /var/log/nginx/error.log
```

**Real-time Updates Not Working:**
```bash
# Check WebSocket connections
ss -tuln | grep 8080

# Monitor WebSocket logs
grep -i "websocket\|socket" /var/log/trading-orchestrator/app.log | tail -20

# Test WebSocket endpoint
wscat -c "ws://localhost:8080/ws" 2>/dev/null || echo "WebSocket not available"
```

## Data Feed Problems

### Market Data Issues

**Data Quality Problems:**
```bash
# Check data feed status
grep -i "data.*feed\|market.*data" /var/log/trading-orchestrator/app.log | tail -20

# Verify data integrity
grep -i "missing.*data\|gap\|hole" /var/log/trading-orchestrator/app.log | tail -10

# Check data vendor status
curl -s https://status.polygon.io/ | head -10
```

**Historical Data Issues:**
```bash
# Check data availability
psql -h localhost -U trading_user -d trading_db -c "
    SELECT 
        symbol,
        COUNT(*) as records,
        MIN(timestamp) as earliest,
        MAX(timestamp) as latest
    FROM market_data 
    GROUP BY symbol 
    ORDER BY records DESC 
    LIMIT 10;
"

# Check for data gaps
grep -i "data.*gap\|missing.*period" /var/log/trading-orchestrator/app.log | tail -10
```

## Report Generation Issues

### Performance Reports

**Report Generation Problems:**
```bash
# Test report generation manually
./scripts/generate-report.sh --type=performance --period=daily

# Check report templates
ls -la /opt/trading-orchestrator/reports/templates/

# Monitor report generation logs
grep -i "report\|export" /var/log/trading-orchestrator/app.log | tail -20

# Check file permissions
ls -la /opt/trading-orchestrator/reports/output/
```

**Email Report Issues:**
```bash
# Test email configuration
./scripts/test-email.sh

# Check email logs
grep -i "email\|smtp" /var/log/trading-orchestrator/app.log | tail -10

# Verify SMTP settings
grep -i "smtp\|email" /etc/trading-orchestrator/config.yaml
```

## Automated Testing Problems

### Test Execution Issues

**Unit Test Failures:**
```bash
# Run specific test suites
python -m pytest tests/test_strategies.py -v
python -m pytest tests/test_brokers.py -v
python -m pytest tests/test_risk_management.py -v

# Check test configuration
cat pytest.ini

# Monitor test results
grep -i "test.*failed\|error" /tmp/test-results.log
```

**Integration Test Issues:**
```bash
# Run integration tests
./scripts/run-integration-tests.sh

# Check broker integration
./scripts/test-broker-integration.sh

# Monitor test environment
grep -i "test.*environment" /var/log/trading-orchestrator/app.log | tail -10
```

## Deployment and Configuration Issues

### Configuration Problems

**Configuration File Issues:**
```bash
# Validate configuration files
./scripts/validate-config.sh

# Check configuration syntax
grep -i "yaml\|json" /etc/trading-orchestrator/config.yaml | head -5

# Compare configuration with defaults
diff /etc/trading-orchestrator/config.yaml /opt/trading-orchestrator/config/default.yaml

# Test configuration reload
grep -i "reload\|config.*change" /var/log/trading-orchestrator/app.log | tail -10
```

**Environment Variable Issues:**
```bash
# Check environment variables
env | grep -E "ALPACA|BINANCE|DB_|REDIS"

# Test environment variable loading
export $(cat /opt/trading-orchestrator/.env | xargs)

# Validate critical environment variables
./scripts/validate-env.sh
```

### Deployment Issues

**Service Deployment Problems:**
```bash
# Check service installation
systemctl list-unit-files | grep trading

# Verify service configuration
systemctl cat trading-orchestrator

# Test service start/stop
systemctl stop trading-orchestrator
systemctl start trading-orchestrator
systemctl status trading-orchestrator
```

**Docker Deployment Issues:**
```bash
# Check Docker containers
docker ps -a | grep trading

# Check Docker logs
docker logs trading-orchestrator-container

# Test Docker network
docker network ls
docker network inspect trading-network

# Check volume mounts
docker inspect trading-orchestrator-container | grep -A 10 Mounts
```

## Monitoring and Alerting

### Alert Configuration

**Alert System Issues:**
```bash
# Check alert configuration
grep -i "alert\|notification" /etc/trading-orchestrator/config.yaml

# Test alert system
./scripts/test-alerts.sh

# Monitor alert logs
grep -i "alert\|notification" /var/log/trading-orchestrator/app.log | tail -20

# Check email/SMS configuration
grep -i "smtp\|webhook" /etc/trading-orchestrator/config.yaml
```

**Monitoring Dashboard Issues:**
```bash
# Check monitoring service status
systemctl status prometheus
systemctl status grafana

# Test monitoring endpoints
curl http://localhost:9090/api/v1/query?query=up
curl http://localhost:3000/api/health

# Check monitoring configuration
cat /etc/prometheus/prometheus.yml
```

## Recovery Procedures

### System Recovery

**Complete System Recovery:**
```bash
#!/bin/bash
# /opt/trading-orchestrator/scripts/emergency-recovery.sh

echo "Starting emergency recovery procedure..."

# Stop all trading activities
echo "Stopping all trading activities..."
curl -X POST http://localhost:8080/api/v1/emergency/stop

# Cancel all pending orders
echo "Canceling all pending orders..."
curl -X POST http://localhost:8080/api/v1/orders/cancel/all

# Backup current state
echo "Creating emergency backup..."
./scripts/create-emergency-backup.sh

# Restore from last known good state
echo "Restoring from backup..."
./scripts/restore-from-backup.sh --latest

# Restart services
echo "Restarting services..."
systemctl restart trading-orchestrator

# Verify system health
echo "Verifying system health..."
./scripts/health-check.sh

echo "Emergency recovery completed"
```

**Database Recovery:**
```bash
# Stop application
sudo systemctl stop trading-orchestrator

# Restore database from backup
sudo -u postgres dropdb trading_db
sudo -u postgres createdb trading_db
sudo -u postgres pg_restore -d trading_db /backups/trading_db_latest.backup

# Start application
sudo systemctl start trading-orchestrator

# Verify data integrity
./scripts/verify-data-integrity.sh
```

**Configuration Recovery:**
```bash
# Restore configuration files
cp /backups/config/trading-orchestrator.config.yaml /etc/trading-orchestrator/config.yaml
cp /backups/config/.env /opt/trading-orchestrator/.env

# Restart services
systemctl restart trading-orchestrator

# Verify configuration
./scripts/validate-config.sh
```

---

This troubleshooting wizard provides comprehensive guidance for diagnosing and resolving common issues in the Day Trading Orchestrator system. Use the diagnostic procedures and tools systematically to identify and fix problems efficiently.