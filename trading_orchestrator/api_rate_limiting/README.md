# API Rate Limiting and Compliance System

A comprehensive API rate limiting and compliance system for broker integrations, designed to ensure optimal performance while respecting all broker API limits and regulatory requirements.

## Overview

This system provides intelligent rate limiting, request management, monitoring, alerting, and compliance features across multiple broker integrations including Binance, Alpaca, Interactive Brokers, DEGIRO, Trading 212, and Trade Republic.

## Key Features

### 🛡️ Core Rate Limiting
- **Token Bucket Algorithm**: Allows burst traffic while maintaining average rate
- **Sliding Window Rate Limiting**: More accurate request tracking
- **Circuit Breaker Pattern**: Prevents cascading failures
- **Adaptive Rate Limiting**: Dynamically adjusts based on performance

### 📊 Broker-Specific Configurations
- **Binance**: Weight-based limits (1200/min), order limits (10/sec)
- **Alpaca**: 200 requests/minute, market data limits
- **IBKR**: 50 requests/second, TWS API limits
- **DEGIRO**: Conservative limits to avoid detection
- **Trading 212**: Strict limits (1-6 requests/minute)
- **Trade Republic**: Minimal safe usage patterns

### 🔄 Request Management
- **Priority Queue**: Critical requests processed first
- **Intelligent Batching**: Combines similar requests
- **Request Coalescing**: Eliminates duplicate operations
- **Dead Letter Queue**: Handles failed requests
- **Exponential Backoff**: Smart retry logic with jitter

### 📈 Monitoring and Alerting
- **Real-time Metrics**: Request rates, response times, error rates
- **Proactive Alerts**: Warns before limits are exceeded
- **Performance Analytics**: Historical usage analysis
- **Health Checks**: System status monitoring
- **Cost Tracking**: API usage cost monitoring

### ⚖️ Compliance Features
- **Audit Logging**: Complete request history
- **Compliance Rules**: Custom regulatory compliance
- **Cost Optimization**: Automated cost reduction suggestions
- **Security Monitoring**: API key validation and rotation
- **Regulatory Reporting**: Automated compliance reports

## Quick Start

### Basic Setup

```python
import asyncio
from api_rate_limiting.manager import APIRateLimitManager
from api_rate_limiting.core.rate_limiter import RequestType, RequestPriority

async def main():
    # Create manager
    manager = APIRateLimitManager()
    
    # Add brokers
    manager.add_broker("binance", is_futures=False)
    manager.add_broker("alpaca", is_paper=True)
    
    # Start the system
    await manager.start()
    
    # Submit requests with rate limiting
    async def get_account_info():
        return {"balance": 1000.0, "currency": "USD"}
    
    request_id = await manager.submit_request(
        broker="binance",
        request_type=RequestType.ACCOUNT_INFO,
        callback=get_account_info,
        priority=RequestPriority.HIGH
    )
    
    # Monitor system health
    health = manager.get_system_health()
    print(f"System Status: {health.status.value}")
    
    await manager.stop()

# Run the example
asyncio.run(main())
```

### Request with Transaction Context

```python
async def main():
    manager = APIRateLimitManager()
    manager.add_broker("binance")
    await manager.start()
    
    # Use transaction context manager
    async with manager.transaction("binance", RequestType.ORDER_PLACE, RequestPriority.CRITICAL) as txn:
        # Your API calls here
        result = await place_order("BTCUSDT", "buy", 0.1)
        txn.result = result
    
    await manager.stop()
```

### Cost Optimization

```python
# Analyze current request patterns
request_patterns = {
    RequestType.MARKET_DATA: 100,
    RequestType.HISTORICAL_DATA: 10,
    RequestType.ACCOUNT_INFO: 5
}

# Get optimization recommendations
optimization = manager.get_cost_optimization(
    broker="binance",
    requests=request_patterns,
    time_window_hours=24
)

print(f"Current Cost: ${optimization['current_cost']:.2f}")
print(f"Potential Savings: ${optimization['estimated_savings']:.2f}")
print("Recommendations:")
for rec in optimization['recommendations']:
    print(f"- {rec}")
```

## Advanced Usage

### Custom Rate Limiting Rules

```python
from api_rate_limiting.brokers.rate_limit_configs import RateLimitRule
from api_rate_limiting.core.rate_limiter import RequestType

# Create custom configuration
custom_config = BinanceRateLimitConfig(is_futures=True)
manager.add_broker("binance_futures", custom_config=custom_config)

# Override specific limits
order_limit = RateLimitRule(
    request_type=RequestType.ORDER_PLACE,
    limit=5,  # More conservative
    window_seconds=1,
    priority=1
)
custom_config.endpoint_limits["orders"] = order_limit
```

### Health Monitoring

```python
from api_rate_limiting.monitoring.health_check import HealthChecker, CheckType

# Create health checker
health_checker = HealthChecker(manager)

# Run specific checks
results = await health_checker.run_all_checks([CheckType.LIVENESS, CheckType.PERFORMANCE])
print(f"System Status: {results['overall_status']}")

# Add custom health check
async def custom_api_check():
    # Your custom health check logic
    return {"status": "ok", "latency": 0.1}

health_checker.add_custom_check(HealthCheck(
    id="custom_api_check",
    name="Custom API Health",
    description="Check custom API endpoints",
    check_type=CheckType.LIVENESS,
    component="custom_api",
    check_function=custom_api_check
))
```

### Compliance Monitoring

```python
# Check compliance status
compliance = manager.get_compliance_status()
print(f"Overall Compliance: {compliance}")

# Generate audit report
audit_report = manager._compliance.get_audit_report(hours=24)
print(f"Violations: {audit_report['compliance_violations']}")
print(f"Rate Limit Hits: {audit_report['rate_limit_exceeded']}")

# Export compliance data
manager.export_data("compliance_data.json", hours=24, include_compliance=True)
```

## Monitoring Dashboard

### Health Check Endpoints

The system provides REST API endpoints for monitoring:

- `GET /health` - Basic health check
- `GET /health/detailed` - Comprehensive health status
- `GET /health/ready` - Readiness check
- `GET /health/live` - Liveness check
- `GET /metrics` - System metrics

### FastAPI Integration

```python
from fastapi import FastAPI
from api_rate_limiting.monitoring.health_check import create_health_check_endpoints

app = FastAPI()
health_checker = HealthChecker(manager)

# Add health check endpoints
health_endpoints = create_health_check_endpoints(health_checker)
for path, endpoint in health_endpoints.items():
    app.get(path)(endpoint)

# Start health monitoring
health_checker.start_monitoring()
```

## Configuration

### Rate Limiting Configuration

```python
config = {
    "monitoring_interval": 10.0,           # Metrics collection interval
    "monitoring_retention_days": 30,       # How long to keep metrics
    "audit_retention_days": 90,           # How long to keep audit logs
    "cost_thresholds": {
        "daily_warning": 50.0,            # Daily cost warning threshold
        "daily_critical": 100.0,          # Daily cost critical threshold
        "monthly_warning": 1000.0,        # Monthly cost warning
        "monthly_critical": 2000.0        # Monthly cost critical
    }
}

manager = APIRateLimitManager(config)
```

### Broker-Specific Settings

Each broker has optimized configurations:

- **Binance**: Weight-based system with burst support
- **Alpaca**: Consistent 200 req/min with circuit breaker
- **IBKR**: Conservative 50/sec with circuit breaker
- **DEGIRO**: Very conservative 20/min (unofficial API)
- **Trading 212**: Strict 6/min with no burst
- **Trade Republic**: Minimal usage patterns

## Error Handling

The system provides comprehensive error handling:

### Rate Limit Exceeded

```python
try:
    result = await manager.submit_request(...)
except RateLimitExceededException as e:
    print(f"Rate limit exceeded. Retry after {e.retry_after}s")
    # Wait and retry
    await asyncio.sleep(e.retry_after)
```

### Circuit Breaker

```python
try:
    result = await manager.submit_request(...)
except CircuitBreakerOpenException as e:
    print(f"Circuit breaker open: {e}")
    # System will automatically recover
```

### Request Timeouts

```python
from api_rate_limiting.core.request_manager import RetryConfig

retry_config = RetryConfig(
    max_retries=3,
    initial_delay=1.0,
    max_delay=30.0,
    timeout=15.0
)

await manager.submit_request(
    broker="binance",
    request_type=RequestType.MARKET_DATA,
    callback=api_call,
    retry_config=retry_config
)
```

## Performance Optimization

### Request Batching

The system automatically batches similar requests:

- Market data requests for the same symbol
- Account info requests
- Historical data requests

### Cost Optimization

```python
# Get optimization suggestions
optimization = manager.get_cost_optimization(broker, requests, 24)
for opportunity in optimization.get('batching_opportunities', []):
    print(f"Batch {opportunity['count']} {opportunity['request_type']} requests")
```

### Adaptive Rate Limiting

The system automatically adapts rates based on:
- API response times
- Error rates
- Broker-specific feedback
- Historical performance

## Security Features

### API Key Management

```python
from api_rate_limiting.utils.helpers import validate_api_key_security

# Validate API key security
security = validate_api_key_security(api_key)
print(f"Security Score: {security['security_score']}/100")

if security['issues']:
    print("Security issues found:")
    for issue in security['issues']:
        print(f"- {issue}")
```

### Audit Logging

All requests are logged with:
- Timestamp
- Broker information
- Request details
- Response status
- Compliance status
- Hash verification

## Deployment

### Requirements

```txt
asyncio
dataclasses
typing
pathlib
collections
datetime
threading
heapq
hashlib
json
```

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run examples
python api_rate_limiting/examples/usage_examples.py
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    API Rate Limit Manager                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐  │
│  │ Rate Limiters   │  │ Request Manager │  │ Compliance   │  │
│  │                 │  │                 │  │ Engine       │  │
│  │ • Token Bucket  │  │ • Priority Queue│  │              │  │
│  │ • Sliding Window│  │ • Batching      │  │ • Audit Log  │  │
│  │ • Circuit Breaker│  │ • Retry Logic  │  │ • Cost Track │  │
│  └─────────────────┘  └─────────────────┘  └──────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    Monitoring System                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐  │
│  │ Rate Monitor    │  │ Alert System    │  │ Health Check│  │
│  │                 │  │                 │  │              │  │
│  │ • Metrics Col.  │  │ • Real-time Alerts│  │ • Liveness  │  │
│  │ • Analytics     │  │ • Threshold Rules│  │ • Readiness │  │
│  │ • Prediction    │  │ • Notifications │  │ • Performance│  │
│  └─────────────────┘  └─────────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Request → Rate Limiter Check → Queue/Process → Response
     ↓              ↓              ↓             ↓
  Compliance   Monitoring    Dead Letter   Audit Log
  Check        Metrics       Queue         Record
```

## Best Practices

### 1. Request Prioritization
- Use `RequestPriority.CRITICAL` for trading operations
- Use `RequestPriority.HIGH` for account management
- Use `RequestPriority.NORMAL` for market data
- Use `RequestPriority.LOW` for background tasks

### 2. Retry Configuration
```python
# Conservative retry for trading
trading_retry = RetryConfig(
    max_retries=3,
    initial_delay=2.0,
    max_delay=30.0,
    timeout=15.0
)

# Aggressive retry for market data
market_data_retry = RetryConfig(
    max_retries=1,
    initial_delay=0.5,
    max_delay=5.0,
    timeout=5.0
)
```

### 3. Monitoring Setup
- Start health monitoring in production
- Set up alerting for critical alerts
- Monitor cost thresholds
- Regular compliance reporting

### 4. Broker Management
- Use appropriate configurations for each broker
- Monitor individual broker health
- Implement fallback strategies
- Regular API key rotation

## Troubleshooting

### Common Issues

**Rate limits consistently exceeded**
- Check current usage vs. broker limits
- Implement request batching
- Consider request caching
- Review request patterns

**High error rates**
- Check circuit breaker status
- Verify API credentials
- Review broker-specific issues
- Adjust timeout values

**Performance issues**
- Monitor queue depth
- Check processing times
- Review concurrent request limits
- Optimize request batching

### Debug Commands

```python
# Check system health
health = manager.get_system_health()
print(health.to_dict())

# Check broker status
for broker in manager.get_broker_list():
    status = manager.get_rate_limit_status(broker)
    print(f"{broker}: {status}")

# Check active alerts
alerts = manager.get_active_alerts()
for alert in alerts:
    print(f"{alert.severity}: {alert.title}")

# Export debug data
manager.export_data("debug_export.json", include_metrics=True)
```

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure compliance with rate limits
5. Submit pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Check the examples directory
- Review the health check endpoints
- Monitor system logs
- Create GitHub issues for bugs

---

**Important**: Always respect broker API limits and terms of service. This system is designed to help ensure compliance but users are responsible for proper usage.