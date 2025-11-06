# Day Trading Orchestrator - API Reference

## Overview

The Day Trading Orchestrator provides a comprehensive REST API and Python SDK for automated trading operations. This document covers all available endpoints, authentication methods, and usage examples.

## Base URL

```
Production: https://api.not-stonks-bot.com
Development: http://localhost:8000
```

## Authentication

All API requests require authentication using one of the following methods:

### API Key Authentication

```http
Authorization: Bearer YOUR_API_KEY
```

### JWT Token Authentication

```http
Authorization: Bearer YOUR_JWT_TOKEN
```

## Core Endpoints

### Trading Operations

#### Execute Trade

```http
POST /api/v1/trade
```

**Request Body:**
```json
{
  "symbol": "AAPL",
  "action": "buy",
  "quantity": 100,
  "order_type": "limit",
  "price": 150.00,
  "strategy": "mean_reversion",
  "metadata": {
    "confidence": 0.85,
    "risk_score": 0.3
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "order_id": "ORD-2024-001",
    "symbol": "AAPL",
    "action": "buy",
    "quantity": 100,
    "price": 150.00,
    "status": "pending",
    "timestamp": "2024-11-07T00:43:39Z"
  }
}
```

#### Get Portfolio

```http
GET /api/v1/portfolio
```

**Response:**
```json
{
  "success": true,
  "data": {
    "total_value": 125000.00,
    "cash_balance": 25000.00,
    "positions": [
      {
        "symbol": "AAPL",
        "quantity": 100,
        "avg_price": 145.00,
        "current_price": 150.00,
        "unrealized_pnl": 500.00,
        "unrealized_pnl_percent": 3.45
      }
    ],
    "performance": {
      "daily_pnl": 1250.00,
      "total_return": 12500.00,
      "total_return_percent": 11.11
    }
  }
}
```

#### Get Account Status

```http
GET /api/v1/account/status
```

**Response:**
```json
{
  "success": true,
  "data": {
    "account_id": "ACC-2024-001",
    "status": "active",
    "buying_power": 50000.00,
    "margin_used": 0.00,
    "margin_available": 50000.00,
    "pattern_day_trader": false,
    "trading_restrictions": []
  }
}
```

### Strategy Management

#### List Strategies

```http
GET /api/v1/strategies
```

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "id": "mean_reversion",
      "name": "Mean Reversion",
      "description": "Trading strategy based on mean reversion patterns",
      "status": "active",
      "parameters": {
        "lookback_period": 20,
        "entry_threshold": 2.0,
        "exit_threshold": 0.5
      },
      "performance": {
        "total_trades": 150,
        "win_rate": 0.65,
        "avg_return": 0.02,
        "sharpe_ratio": 1.2
      }
    }
  ]
}
```

#### Create Strategy

```http
POST /api/v1/strategies
```

**Request Body:**
```json
{
  "name": "Custom Strategy",
  "description": "My custom trading strategy",
  "strategy_type": "trend_following",
  "parameters": {
    "fast_ma_period": 10,
    "slow_ma_period": 30,
    "rsi_period": 14
  },
  "risk_management": {
    "max_position_size": 10000,
    "stop_loss_percent": 0.05,
    "take_profit_percent": 0.10
  }
}
```

### Market Data

#### Get Real-time Quote

```http
GET /api/v1/market/quote/{symbol}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "symbol": "AAPL",
    "price": 150.25,
    "bid": 150.20,
    "ask": 150.30,
    "volume": 50000000,
    "timestamp": "2024-11-07T00:43:39Z",
    "change": 2.15,
    "change_percent": 1.45
  }
}
```

#### Get Historical Data

```http
GET /api/v1/market/historical/{symbol}?period=1d&interval=1h
```

**Parameters:**
- `period`: 1d, 5d, 1m, 3m, 6m, 1y, 2y, 5y
- `interval`: 1m, 5m, 15m, 30m, 1h, 1d, 1w, 1mo

**Response:**
```json
{
  "success": true,
  "data": {
    "symbol": "AAPL",
    "period": "1d",
    "interval": "1h",
    "data": [
      {
        "timestamp": "2024-11-07T00:00:00Z",
        "open": 148.00,
        "high": 151.00,
        "low": 147.50,
        "close": 150.25,
        "volume": 5000000
      }
    ]
  }
}
```

### Risk Management

#### Get Risk Metrics

```http
GET /api/v1/risk/metrics
```

**Response:**
```json
{
  "success": true,
  "data": {
    "portfolio_risk": 0.15,
    "var_95": -2500.00,
    "max_drawdown": 0.08,
    "sharpe_ratio": 1.2,
    "sortino_ratio": 1.5,
    "beta": 0.9,
    "correlation": 0.7,
    "position_concentration": 0.25,
    "leverage_ratio": 1.1
  }
}
```

#### Set Risk Limits

```http
POST /api/v1/risk/limits
```

**Request Body:**
```json
{
  "max_position_size": 15000,
  "max_daily_loss": 5000,
  "max_portfolio_risk": 0.20,
  "stop_loss_global": 0.05,
  "concentration_limit": 0.30
}
```

### System Health

#### Health Check

```http
GET /api/v1/health
```

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "timestamp": "2024-11-07T00:43:39Z",
    "uptime": 86400,
    "version": "1.0.0",
    "services": {
      "database": "healthy",
      "broker_connections": "healthy",
      "data_feeds": "healthy",
      "ml_models": "healthy"
    },
    "metrics": {
      "cpu_usage": 0.25,
      "memory_usage": 0.60,
      "disk_usage": 0.45
    }
  }
}
```

#### System Logs

```http
GET /api/v1/system/logs?level=info&limit=100
```

**Response:**
```json
{
  "success": true,
  "data": {
    "logs": [
      {
        "timestamp": "2024-11-07T00:43:39Z",
        "level": "info",
        "message": "Trade executed successfully",
        "metadata": {
          "order_id": "ORD-2024-001",
          "symbol": "AAPL"
        }
      }
    ],
    "total": 1500,
    "has_more": true
  }
}
```

## Error Handling

### Standard Error Response

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": {
      "field": "quantity",
      "reason": "must be a positive integer"
    }
  },
  "timestamp": "2024-11-07T00:43:39Z"
}
```

### Common Error Codes

- `AUTHENTICATION_ERROR`: Invalid or missing authentication
- `AUTHORIZATION_ERROR`: Insufficient permissions
- `VALIDATION_ERROR`: Invalid request parameters
- `INSUFFICIENT_FUNDS`: Account lacks sufficient buying power
- `MARKET_CLOSED`: Market is currently closed
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `STRATEGY_ERROR`: Strategy execution error
- `BROKER_ERROR`: Broker connection or execution error
- `INTERNAL_ERROR`: Internal system error

## Rate Limiting

- **Authentication endpoints**: 10 requests per minute
- **Trading endpoints**: 60 requests per minute
- **Market data**: 300 requests per minute
- **System endpoints**: 100 requests per minute

## SDK Examples

### Python SDK

```python
from trading_orchestrator import TradingOrchestrator

# Initialize client
client = TradingOrchestrator(api_key="your_api_key")

# Execute a trade
trade = client.trade.execute(
    symbol="AAPL",
    action="buy",
    quantity=100,
    order_type="limit",
    price=150.00,
    strategy="mean_reversion"
)

# Get portfolio
portfolio = client.portfolio.get()

# Subscribe to real-time data
client.market.subscribe("AAPL", callback=handle_quote)
```

### JavaScript SDK

```javascript
import { TradingOrchestrator } from '@not-stonks-bot/sdk';

// Initialize client
const client = new TradingOrchestrator({
  apiKey: 'your_api_key'
});

// Execute a trade
const trade = await client.trade.execute({
  symbol: 'AAPL',
  action: 'buy',
  quantity: 100,
  orderType: 'limit',
  price: 150.00,
  strategy: 'mean_reversion'
});

// Get portfolio
const portfolio = await client.portfolio.get();
```

## Webhooks

### Webhook Events

- `trade.executed`: Trade has been executed
- `trade.filled`: Order has been filled
- `order.cancelled`: Order has been cancelled
- `position.updated`: Position has been updated
- `risk.alert`: Risk limit has been breached
- `system.error`: System error occurred

### Webhook Configuration

```http
POST /api/v1/webhooks
```

**Request Body:**
```json
{
  "url": "https://your-app.com/webhook",
  "events": ["trade.executed", "risk.alert"],
  "secret": "your_webhook_secret"
}
```

## Pagination

For endpoints that return lists, use the following parameters:

- `page`: Page number (starting from 1)
- `limit`: Items per page (max 100)
- `sort`: Sort field
- `order`: Sort order (asc, desc)

```http
GET /api/v1/trades?page=2&limit=50&sort=timestamp&order=desc
```

## WebSocket API

### Connection

```javascript
const ws = new WebSocket('wss://api.not-stonks-bot.com/ws');

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'auth',
    token: 'your_api_key'
  }));
};
```

### Subscriptions

```javascript
// Subscribe to market data
ws.send(JSON.stringify({
  type: 'subscribe',
  channel: 'market_data',
  symbols: ['AAPL', 'GOOGL']
}));

// Subscribe to trade updates
ws.send(JSON.stringify({
  type: 'subscribe',
  channel: 'trades'
}));
```

## SDK Installation

### Python

```bash
pip install trading-orchestrator-sdk
```

### JavaScript/TypeScript

```bash
npm install @not-stonks-bot/sdk
```

### Go

```bash
go get github.com/not-stonks-bot/go-sdk
```

## Testing

### Sandbox Environment

Use the sandbox environment for testing without real money:

```
Sandbox: https://sandbox-api.not-stonks-bot.com
```

### Test Data

All test endpoints return realistic but fake data suitable for testing strategies and integrations.

## Support

- **Documentation**: https://docs.not-stonks-bot.com
- **Support Email**: support@not-stonks-bot.com
- **Community**: https://community.not-stonks-bot.com
- **Status Page**: https://status.not-stonks-bot.com