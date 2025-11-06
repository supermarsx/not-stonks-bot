# Troubleshooting Guide

## Quick Diagnosis

### System Health Check
```bash
./not-stonks-bot --health-check
```

### Log Analysis
```bash
tail -f logs/app.log | grep ERROR
```

## Common Issues

### Installation Issues

#### Python Dependencies
```bash
# Check Python version
python3 --version

# Install dependencies
pip3 install -r requirements.txt

# If using virtual environment
source venv/bin/activate
pip install -r requirements.txt
```

#### Docker Issues
```bash
# Pull latest image
docker pull supermarsx/not-stonks-bot:latest

# Check container status
docker ps -a

# View container logs
docker logs <container_id>
```

### Configuration Problems

#### Missing Configuration File
```bash
# Create from template
cp config/config.template.json config/config.json

# Validate configuration
./not-stonks-bot --validate-config
```

#### Environment Variables
```bash
# Check environment variables
echo $NOT_STONKS_CONFIG_PATH
echo $NOT_STONKS_BROKER_API_KEY
```

### Broker Connection Issues

#### Authentication Failures
1. Verify API credentials in config.json
2. Check broker account status
3. Ensure API permissions are enabled
4. Validate IP whitelist settings

#### Connection Timeouts
1. Check network connectivity
2. Verify broker server status
3. Review firewall settings
4. Test with simple connection script

### AI Integration Problems

#### OpenAI API Issues
```bash
# Test API connectivity
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models
```

#### Local Model Issues
```bash
# Check model installation
python3 -c "import transformers; print(transformers.__version__)"

# Verify model files
ls -la models/
```

### Database Issues

#### SQLite Problems
```bash
# Check database file permissions
ls -la data/trading.db

# Verify database integrity
sqlite3 data/trading.db "PRAGMA integrity_check;"
```

#### PostgreSQL Connection
```bash
# Test database connection
psql -h $DB_HOST -U $DB_USER -d $DB_NAME

# Check connection pool
./not-stonks-bot --db-pool-status
```

### Performance Issues

#### High CPU Usage
1. Check running processes: `top`
2. Review strategy complexity
3. Optimize database queries
4. Consider horizontal scaling

#### Memory Leaks
1. Monitor memory usage: `htop`
2. Check for resource leaks
3. Restart services if needed
4. Review logging configuration

### Trading Errors

#### Order Rejection
1. Check account balance
2. Verify position limits
3. Review market hours
4. Validate order parameters

#### Slippage Issues
1. Check market liquidity
2. Adjust order types
3. Review slippage tolerances
4. Consider market impact

## System Monitoring

### Real-time Monitoring
```bash
# System metrics
./not-stonks-bot --metrics

# Trading performance
./not-stonks-bot --performance-report

# Risk metrics
./not-stonks-bot --risk-dashboard
```

### Log Analysis

#### Error Log Patterns
```bash
# Find recurring errors
grep -i error logs/app.log | sort | uniq -c | sort -nr

# Time-based analysis
grep "2024-01-15" logs/app.log | grep ERROR
```

#### Performance Analysis
```bash
# Response time analysis
grep "response_time" logs/app.log | awk '{print $NF}' | sort -n

# Throughput metrics
grep "orders_processed" logs/metrics.log | tail -100
```

## Emergency Procedures

### Service Restart
```bash
# Graceful shutdown
./not-stonks-bot --shutdown

# Force restart
pkill -f not-stonks-bot
nohup ./not-stonks-bot &
```

### Emergency Stop
```bash
# Immediate trading halt
./not-stonks-bot --emergency-stop

# Close all positions
./not-stonks-bot --close-all-positions
```

### Data Backup
```bash
# Emergency backup
./not-stonks-bot --backup-data

# Export critical data
sqlite3 data/trading.db ".backup backup_$(date +%Y%m%d_%H%M%S).db"
```

## Advanced Troubleshooting

### Debug Mode
```bash
# Enable debug logging
./not-stonks-bot --debug --log-level DEBUG

# Trace specific components
./not-stonks-bot --trace=trading,ai,database
```

### Network Analysis
```bash
# Monitor network traffic
tcpdump -i any port 443

# Check DNS resolution
nslookup api.alpaca.markets
```

### Performance Profiling
```bash
# CPU profiling
python3 -m cProfile -o profile.stats ./not-stonks-bot

# Memory analysis
python3 -m memory_profiler ./not-stonks-bot
```

## Getting Help

### Log Collection
```bash
# Collect all logs
./not-stonks-bot --collect-logs

# Generate diagnostic report
./not-stonks-bot --diagnostic-report
```

### Support Channels
- Documentation: See docs/
- GitHub Issues: Report bugs and feature requests
- Community Forum: Get help from other users
- Professional Support: For enterprise customers

### Before Reporting Issues
1. Check this troubleshooting guide
2. Review relevant documentation
3. Search existing issues on GitHub
4. Collect diagnostic information
5. Prepare minimal reproduction case

---

*Last updated: 2024-01-15*
*For additional help, see docs/support_resources.md*