# API Documentation

Complete REST API reference for the Day Trading Orchestrator system. This documentation covers all endpoints, request/response formats, authentication, and integration examples.

## 📚 Table of Contents

1. [API Overview](#api-overview)
2. [Authentication](#authentication)
3. [Rate Limiting](#rate-limiting)
4. [Error Handling](#error-handling)
5. [Core Endpoints](#core-endpoints)
6. [Broker Management](#broker-management)
7. [Order Management](#order-management)
8. [Position Management](#position-management)
9. [Portfolio Management](#portfolio-management)
10. [Market Data](#market-data)
11. [Strategy Management](#strategy-management)
12. [Risk Management](#risk-management)
13. [AI Integration](#ai-integration)
14. [System Monitoring](#system-monitoring)
15. [WebSocket API](#websocket-api)
16. [SDKs and Examples](#sdks-and-examples)

---

## 🚀 API Overview

### Base URL
```
Production: https://api.trading-orchestrator.com/v1
Development: http://localhost:8000/v1
```

### API Versioning
- Current version: v1
- Backward compatibility maintained for v1.x
- Breaking changes require new version

### Content Types
- **Request**: `application/json`
- **Response**: `application/json`
- **Compression**: gzip supported

### Date/Time Format
All timestamps use ISO 8601 format:
```json
{
  "timestamp": "2025-11-06T14:30:00Z",
  "date": "2025-11-06"
}
```

### Pagination
All list endpoints support pagination:
```json
{
  "data": [...],
  "pagination": {
    "page": 1,
    "per_page": 50,
    "total": 150,
    "pages": 3
  }
}
```

---

## 🔐 Authentication

### API Key Authentication

**Header-based Authentication:**
```http
GET /api/v1/account
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json
```

**Query Parameter Authentication:**
```http
GET /api/v1/account?api_key=YOUR_API_KEY
```

### JWT Token Authentication

**Obtaining JWT Token:**
```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "your_password"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "refresh_token": "refresh_token_here"
}
```

**Using JWT Token:**
```http
GET /api/v1/account
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### OAuth 2.0 Integration

**Authorization Flow:**
```http
GET /api/v1/auth/authorize?response_type=code&client_id=CLIENT_ID&redirect_uri=REDIRECT_URI
```

**Token Exchange:**
```http
POST /api/v1/auth/token
Content-Type: application/x-www-form-urlencoded

grant_type=authorization_code&
code=AUTH_CODE&
client_id=CLIENT_ID&
client_secret=CLIENT_SECRET&
redirect_uri=REDIRECT_URI
```

---

## ⚡ Rate Limiting

### Rate Limit Headers
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995200
X-RateLimit-Window: 60
```

### Rate Limit Tiers

| Tier | Requests/Minute | Burst Limit | Cost/Month |
|------|----------------|-------------|------------|
| **Free** | 60 | 10 | $0 |
| **Basic** | 300 | 50 | $29 |
| **Professional** | 1000 | 200 | $99 |
| **Enterprise** | 5000 | 1000 | $299 |

### Rate Limit Error Response
```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Try again in 30 seconds.",
    "details": {
      "limit": 1000,
      "window": 60,
      "reset_time": "2025-11-06T14:31:00Z"
    }
  }
}
```

### Rate Limit Best Practices

1. **Implement Exponential Backoff**
```python
import time
import requests

def api_request_with_backoff(url, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate limited
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                continue
            else:
                raise
    raise Exception("Max retries exceeded")
```

2. **Use WebSocket for Real-time Data**
3. **Cache Frequently Accessed Data**
4. **Batch Multiple Requests**

---

## ❌ Error Handling

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| **200** | OK | Request successful |
| **201** | Created | Resource created successfully |
| **400** | Bad Request | Invalid request format |
| **401** | Unauthorized | Authentication required |
| **403** | Forbidden | Access denied |
| **404** | Not Found | Resource not found |
| **429** | Too Many Requests | Rate limit exceeded |
| **500** | Internal Server Error | Server error |
| **503** | Service Unavailable | Service temporarily unavailable |

### Error Response Format

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": {
      "field": "Additional error details",
      "correlation_id": "req_1234567890"
    }
  }
}
```

### Common Error Codes

| Code | Description | Resolution |
|------|-------------|------------|
| **INVALID_SYMBOL** | Trading symbol not found | Verify symbol format |
| **INSUFFICIENT_FUNDS** | Account balance insufficient | Deposit funds or reduce order size |
| **MARKET_CLOSED** | Market is closed | Wait for market open |
| **RISK_LIMIT_EXCEEDED** | Risk limit would be violated | Reduce position size |
| **BROKER_CONNECTION_ERROR** | Broker API unavailable | Check broker status |
| **ORDER_REJECTED** | Order rejected by broker | Review order parameters |

### Error Handling Best Practices

```python
def handle_api_error(response):
    """Handle API errors gracefully"""
    
    if response.status_code == 401:
        raise AuthenticationError("Please check your API key")
    elif response.status_code == 403:
        raise PermissionError("Access denied")
    elif response.status_code == 429:
        raise RateLimitError("Rate limit exceeded. Please wait.")
    elif response.status_code >= 500:
        raise ServerError("Server error. Please try again later.")
    else:
        error_data = response.json()
        raise APIError(error_data['error']['message'])
```

---

## 🏗️ Core Endpoints

### System Health

#### GET /health
Check system health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-06T14:30:00Z",
  "uptime": 86400,
  "version": "2.0.0",
  "components": {
    "database": "healthy",
    "brokers": "healthy",
    "ai_services": "healthy",
    "risk_management": "healthy"
  }
}
```

#### GET /api/v1/status
Get detailed system status.

**Response:**
```json
{
  "system_status": "operational",
  "trading_mode": "PAPER",
  "active_brokers": ["alpaca", "binance"],
  "total_positions": 5,
  "portfolio_value": 100000.00,
  "daily_pnl": 250.75,
  "risk_level": "LOW",
  "alerts": []
}
```

### User Management

#### GET /api/v1/user/profile
Get user profile information.

**Response:**
```json
{
  "user_id": "usr_123456",
  "email": "user@example.com",
  "first_name": "John",
  "last_name": "Doe",
  "account_type": "premium",
  "subscription_status": "active",
  "created_at": "2025-01-01T00:00:00Z",
  "last_login": "2025-11-06T14:00:00Z",
  "preferences": {
    "timezone": "UTC",
    "language": "en",
    "notifications": {
      "email": true,
      "sms": false,
      "push": true
    }
  }
}
```

#### PUT /api/v1/user/profile
Update user profile.

**Request:**
```json
{
  "first_name": "Jane",
  "last_name": "Smith",
  "preferences": {
    "timezone": "America/New_York",
    "language": "en"
  }
}
```

**Response:**
```json
{
  "user_id": "usr_123456",
  "updated_at": "2025-11-06T14:30:00Z",
  "message": "Profile updated successfully"
}
```

---

## 🔗 Broker Management

### List Brokers

#### GET /api/v1/brokers
Get list of available brokers.

**Response:**
```json
{
  "brokers": [
    {
      "broker_id": "alpaca",
      "name": "Alpaca Trading",
      "status": "connected",
      "supported_markets": ["stocks", "crypto"],
      "paper_trading": true,
      "commission_type": "commission_free",
      "rate_limit": 200,
      "last_heartbeat": "2025-11-06T14:29:30Z"
    },
    {
      "broker_id": "binance",
      "name": "Binance",
      "status": "connected",
      "supported_markets": ["crypto"],
      "paper_trading": true,
      "commission_type": "percentage",
      "rate_limit": 1200,
      "last_heartbeat": "2025-11-06T14:29:45Z"
    }
  ]
}
```

### Broker Connection

#### POST /api/v1/brokers/connect
Connect to a broker.

**Request:**
```json
{
  "broker_id": "alpaca",
  "credentials": {
    "api_key": "YOUR_API_KEY",
    "secret_key": "YOUR_SECRET_KEY"
  },
  "paper_mode": true,
  "config": {
    "base_url": "https://paper-api.alpaca.markets"
  }
}
```

**Response:**
```json
{
  "broker_id": "alpaca",
  "status": "connected",
  "account_info": {
    "account_id": "acc_123456",
    "currency": "USD",
    "buying_power": 100000.00,
    "regt_buying_power": 100000.00,
    "daytrading_buying_power": 400000.00
  },
  "connection_time": "2025-11-06T14:30:00Z"
}
```

### Broker Status

#### GET /api/v1/brokers/{broker_id}/status
Get broker connection status.

**Response:**
```json
{
  "broker_id": "alpaca",
  "status": "connected",
  "connected_at": "2025-11-06T14:00:00Z",
  "last_ping": "2025-11-06T14:29:30Z",
  "api_calls_today": 145,
  "rate_limit_remaining": 55,
  "server_time": "2025-11-06T14:30:00Z"
}
```

---

## 📋 Order Management

### Create Order

#### POST /api/v1/orders
Create a new trading order.

**Request:**
```json
{
  "symbol": "AAPL",
  "side": "buy",
  "type": "market",
  "quantity": 100,
  "time_in_force": "day",
  "broker_id": "alpaca",
  "metadata": {
    "strategy_id": "mean_reversion_001",
    "source": "api"
  }
}
```

**Alternative for Limit Order:**
```json
{
  "symbol": "AAPL",
  "side": "buy",
  "type": "limit",
  "quantity": 100,
  "limit_price": 150.00,
  "time_in_force": "gtc",
  "broker_id": "alpaca"
}
```

**Response:**
```json
{
  "order_id": "ord_123456789",
  "broker_order_id": "alpac_order_789",
  "symbol": "AAPL",
  "side": "buy",
  "type": "market",
  "quantity": 100,
  "filled_quantity": 0,
  "remaining_quantity": 100,
  "status": "accepted",
  "submitted_at": "2025-11-06T14:30:00Z",
  "broker_id": "alpaca",
  "estimated_cost": 15000.00,
  "estimated_commission": 0.00
}
```

### List Orders

#### GET /api/v1/orders
Get list of orders with filtering options.

**Query Parameters:**
- `status`: Filter by order status (open, closed, all)
- `symbol`: Filter by symbol
- `side`: Filter by side (buy, sell)
- `limit`: Maximum number of orders (default: 50, max: 100)
- `offset`: Offset for pagination
- `from_date`: Filter orders from date
- `to_date`: Filter orders to date

**Example:**
```http
GET /api/v1/orders?status=open&symbol=AAPL&limit=10
```

**Response:**
```json
{
  "orders": [
    {
      "order_id": "ord_123456789",
      "symbol": "AAPL",
      "side": "buy",
      "type": "limit",
      "quantity": 100,
      "filled_quantity": 50,
      "remaining_quantity": 50,
      "status": "partially_filled",
      "limit_price": 150.00,
      "avg_fill_price": 149.95,
      "submitted_at": "2025-11-06T14:25:00Z",
      "filled_at": "2025-11-06T14:27:30Z",
      "broker_id": "alpaca"
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 10,
    "total": 1,
    "pages": 1
  }
}
```

### Get Order Details

#### GET /api/v1/orders/{order_id}
Get detailed order information.

**Response:**
```json
{
  "order_id": "ord_123456789",
  "symbol": "AAPL",
  "side": "buy",
  "type": "market",
  "quantity": 100,
  "filled_quantity": 100,
  "remaining_quantity": 0,
  "status": "filled",
  "avg_fill_price": 149.85,
  "total_fill_value": 14985.00,
  "commission": 0.00,
  "submitted_at": "2025-11-06T14:30:00Z",
  "filled_at": "2025-11-06T14:30:01Z",
  "broker_id": "alpaca",
  "broker_order_id": "alpac_order_789",
  "fills": [
    {
      "fill_id": "fill_123",
      "quantity": 50,
      "price": 149.85,
      "commission": 0.00,
      "timestamp": "2025-11-06T14:30:01Z"
    },
    {
      "fill_id": "fill_124",
      "quantity": 50,
      "price": 149.85,
      "commission": 0.00,
      "timestamp": "2025-11-06T14:30:01Z"
    }
  ]
}
```

### Modify Order

#### PUT /api/v1/orders/{order_id}
Modify existing order (limit price changes only).

**Request:**
```json
{
  "limit_price": 150.50,
  "quantity": 150
}
```

**Response:**
```json
{
  "order_id": "ord_123456789",
  "status": "accepted",
  "updated_at": "2025-11-06T14:35:00Z",
  "message": "Order updated successfully"
}
```

### Cancel Order

#### DELETE /api/v1/orders/{order_id}
Cancel pending order.

**Response:**
```json
{
  "order_id": "ord_123456789",
  "status": "cancelled",
  "cancelled_at": "2025-11-06T14:35:00Z",
  "message": "Order cancelled successfully"
}
```

### Bulk Order Operations

#### POST /api/v1/orders/bulk
Create multiple orders in a single request.

**Request:**
```json
{
  "orders": [
    {
      "symbol": "AAPL",
      "side": "buy",
      "type": "market",
      "quantity": 100
    },
    {
      "symbol": "GOOGL",
      "side": "buy",
      "type": "limit",
      "quantity": 50,
      "limit_price": 2800.00
    }
  ],
  "broker_id": "alpaca"
}
```

**Response:**
```json
{
  "bulk_order_id": "bulk_123456789",
  "orders": [
    {
      "order_id": "ord_123456789",
      "symbol": "AAPL",
      "status": "accepted"
    },
    {
      "order_id": "ord_123456790",
      "symbol": "GOOGL",
      "status": "accepted"
    }
  ],
  "submission_time": "2025-11-06T14:30:00Z"
}
```

---

## 📊 Position Management

### Get Positions

#### GET /api/v1/positions
Get current positions across all brokers.

**Response:**
```json
{
  "positions": [
    {
      "position_id": "pos_123456789",
      "symbol": "AAPL",
      "side": "long",
      "quantity": 100,
      "avg_entry_price": 149.50,
      "current_price": 150.25,
      "market_value": 15025.00,
      "unrealized_pnl": 75.00,
      "unrealized_pnl_percent": 0.50,
      "realized_pnl": 0.00,
      "cost_basis": 14950.00,
      "broker_id": "alpaca",
      "opened_at": "2025-11-06T10:00:00Z",
      "last_updated": "2025-11-06T14:30:00Z"
    }
  ],
  "total_unrealized_pnl": 75.00,
  "total_market_value": 15025.00,
  "position_count": 1
}
```

### Get Position by Symbol

#### GET /api/v1/positions/{symbol}
Get position for specific symbol.

**Response:**
```json
{
  "symbol": "AAPL",
  "side": "long",
  "quantity": 100,
  "avg_entry_price": 149.50,
  "current_price": 150.25,
  "market_value": 15025.00,
  "unrealized_pnl": 75.00,
  "day_change": 25.00,
  "day_change_percent": 0.17
}
```

### Close Position

#### DELETE /api/v1/positions/{symbol}
Close position for symbol.

**Query Parameters:**
- `quantity`: Specific quantity to close (optional, closes all if not specified)
- `order_type`: Order type for closing (market, limit)

**Response:**
```json
{
  "symbol": "AAPL",
  "closed_quantity": 100,
  "execution_price": 150.30,
  "realized_pnl": 80.00,
  "commission": 0.00,
  "closed_at": "2025-11-06T14:35:00Z",
  "order_id": "ord_123456791"
}
```

---

## 💼 Portfolio Management

### Get Portfolio

#### GET /api/v1/portfolio
Get complete portfolio information.

**Response:**
```json
{
  "account_id": "acc_123456",
  "currency": "USD",
  "cash": 85000.00,
  "total_equity": 100075.00,
  "buying_power": 170000.00,
  "initial_margin": 0.00,
  "maintenance_margin": 0.00,
  "daytrade_count": 0,
  "account_status": "ACTIVE",
  "broker_balances": {
    "alpaca": {
      "cash": 50000.00,
      "equity": 50250.00,
      "buying_power": 100500.00
    },
    "binance": {
      "cash": 35000.00,
      "equity": 49825.00,
      "buying_power": 69500.00
    }
  },
  "daytrade_buying_power": 201000.00,
  "last_updated": "2025-11-06T14:30:00Z"
}
```

### Get Portfolio History

#### GET /api/v1/portfolio/history
Get portfolio value history.

**Query Parameters:**
- `period`: Time period (1D, 1W, 1M, 3M, 6M, 1Y, all)
- `timeframe`: Data frequency (1min, 5min, 15min, 1H, 1D)
- `start_date`: Start date for custom period
- `end_date`: End date for custom period

**Example:**
```http
GET /api/v1/portfolio/history?period=1M&timeframe=1D
```

**Response:**
```json
{
  "portfolio_value": [
    {
      "timestamp": "2025-10-06T00:00:00Z",
      "equity": 98000.00,
      "cash": 85000.00,
      "positions_value": 13000.00
    },
    {
      "timestamp": "2025-10-07T00:00:00Z",
      "equity": 98500.00,
      "cash": 85500.00,
      "positions_value": 13000.00
    }
  ],
  "performance": {
    "total_return": 2.56,
    "period_return": 2.56,
    "volatility": 12.5,
    "sharpe_ratio": 1.45,
    "max_drawdown": -5.2
  }
}
```

### Rebalance Portfolio

#### POST /api/v1/portfolio/rebalance
Rebalance portfolio to target allocations.

**Request:**
```json
{
  "target_allocations": {
    "AAPL": 0.25,
    "GOOGL": 0.25,
    "TSLA": 0.20,
    "MSFT": 0.20,
    "AMZN": 0.10
  },
  "tolerance": 0.02,
  "order_type": "market"
}
```

**Response:**
```json
{
  "rebalance_id": "rebal_123456789",
  "status": "in_progress",
  "orders_created": [
    {
      "symbol": "TSLA",
      "side": "buy",
      "quantity": 15,
      "order_id": "ord_123456792"
    }
  ],
  "estimated_cost": 7500.00,
  "rebalance_time": "2025-11-06T14:35:00Z"
}
```

---

## 📈 Market Data

### Get Quote

#### GET /api/v1/market/quote/{symbol}
Get real-time quote for symbol.

**Response:**
```json
{
  "symbol": "AAPL",
  "timestamp": "2025-11-06T14:30:00Z",
  "bid": 150.24,
  "ask": 150.26,
  "bid_size": 100,
  "ask_size": 150,
  "last": 150.25,
  "volume": 25478932,
  "high": 151.00,
  "low": 149.80,
  "open": 150.10,
  "previous_close": 149.95,
  "change": 0.30,
  "change_percent": 0.20
}
```

### Get Historical Data

#### GET /api/v1/market/history/{symbol}
Get historical price data.

**Query Parameters:**
- `timeframe`: Data frequency (1min, 5min, 15min, 1H, 1D, 1W, 1M)
- `start_date`: Start date (ISO format)
- `end_date`: End date (ISO format)
- `limit`: Maximum number of candles

**Example:**
```http
GET /api/v1/market/history/AAPL?timeframe=1H&start_date=2025-11-01&limit=168
```

**Response:**
```json
{
  "symbol": "AAPL",
  "timeframe": "1H",
  "data": [
    {
      "timestamp": "2025-11-06T13:00:00Z",
      "open": 150.10,
      "high": 150.45,
      "low": 150.05,
      "close": 150.30,
      "volume": 1250000
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 100,
    "total": 168,
    "pages": 2
  }
}
```

### Get Multiple Quotes

#### POST /api/v1/market/quotes
Get quotes for multiple symbols.

**Request:**
```json
{
  "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA"]
}
```

**Response:**
```json
{
  "quotes": [
    {
      "symbol": "AAPL",
      "last": 150.25,
      "change": 0.30,
      "change_percent": 0.20
    }
  ],
  "timestamp": "2025-11-06T14:30:00Z"
}
```

### Get Market Summary

#### GET /api/v1/market/summary
Get overall market summary.

**Response:**
```json
{
  "timestamp": "2025-11-06T14:30:00Z",
  "market_status": "open",
  "next_close": "2025-11-06T21:00:00Z",
  "indices": {
    "S&P_500": {
      "value": 4250.75,
      "change": 12.30,
      "change_percent": 0.29
    },
    "NASDAQ": {
      "value": 13250.80,
      "change": -25.40,
      "change_percent": -0.19
    }
  },
  "sectors": {
    "technology": 0.45,
    "healthcare": -0.12,
    "financial": 0.23
  },
  "sentiment": {
    "fear_greed_index": 65,
    "vix": 18.5,
    "put_call_ratio": 0.85
  }
}
```

---

## 🧠 Strategy Management

### List Strategies

#### GET /api/v1/strategies
Get list of available strategies.

**Response:**
```json
{
  "strategies": [
    {
      "strategy_id": "mean_reversion_001",
      "name": "Bollinger Bands Mean Reversion",
      "category": "mean_reversion",
      "description": "Classic Bollinger Bands strategy for ranging markets",
      "risk_level": "medium",
      "timeframes": ["1H", "4H", "1D"],
      "markets": ["stocks", "crypto"],
      "is_active": true,
      "performance": {
        "total_return": 15.2,
        "sharpe_ratio": 1.45,
        "max_drawdown": -8.5,
        "win_rate": 0.67
      },
      "created_at": "2025-01-01T00:00:00Z"
    }
  ]
}
```

### Strategy Details

#### GET /api/v1/strategies/{strategy_id}
Get detailed strategy information.

**Response:**
```json
{
  "strategy_id": "mean_reversion_001",
  "name": "Bollinger Bands Mean Reversion",
  "category": "mean_reversion",
  "description": "Classic Bollinger Bands strategy for ranging markets",
  "parameters": {
    "bb_period": 20,
    "bb_std": 2.0,
    "rsi_period": 14,
    "stop_loss": 0.03,
    "take_profit": 0.05
  },
  "risk_management": {
    "max_position_size": 10000,
    "max_daily_loss": 5000,
    "stop_loss_percentage": 0.03
  },
  "performance": {
    "backtest_period": "2023-01-01 to 2025-11-01",
    "total_return": 15.2,
    "annualized_return": 7.8,
    "volatility": 12.5,
    "sharpe_ratio": 1.45,
    "max_drawdown": -8.5,
    "win_rate": 0.67,
    "profit_factor": 1.85
  }
}
```

### Create Strategy

#### POST /api/v1/strategies
Create new custom strategy.

**Request:**
```json
{
  "name": "My Custom Strategy",
  "category": "momentum",
  "description": "Custom momentum strategy",
  "parameters": {
    "fast_period": 10,
    "slow_period": 30,
    "signal_threshold": 0.6
  },
  "risk_management": {
    "max_position_size": 5000,
    "max_daily_loss": 2500
  }
}
```

**Response:**
```json
{
  "strategy_id": "custom_001",
  "name": "My Custom Strategy",
  "status": "created",
  "created_at": "2025-11-06T14:30:00Z"
}
```

### Enable/Disable Strategy

#### PUT /api/v1/strategies/{strategy_id}/status
Enable or disable strategy.

**Request:**
```json
{
  "enabled": true
}
```

**Response:**
```json
{
  "strategy_id": "mean_reversion_001",
  "enabled": true,
  "updated_at": "2025-11-06T14:30:00Z"
}
```

### Strategy Performance

#### GET /api/v1/strategies/{strategy_id}/performance
Get strategy performance metrics.

**Query Parameters:**
- `period`: Performance period (1D, 1W, 1M, 3M, 6M, 1Y, all)
- `benchmark`: Benchmark symbol for comparison

**Response:**
```json
{
  "strategy_id": "mean_reversion_001",
  "period": "1M",
  "benchmark": "SPY",
  "returns": {
    "strategy": 2.45,
    "benchmark": 1.85,
    "alpha": 0.60,
    "excess_return": 0.60
  },
  "risk_metrics": {
    "volatility": 12.5,
    "sharpe_ratio": 1.45,
    "max_drawdown": -3.2,
    "beta": 0.85,
    "correlation": 0.75
  },
  "trades": {
    "total_trades": 25,
    "winning_trades": 17,
    "losing_trades": 8,
    "win_rate": 0.68,
    "avg_win": 1.2,
    "avg_loss": -0.8
  }
}
```

---

## ⚠️ Risk Management

### Get Risk Status

#### GET /api/v1/risk/status
Get current portfolio risk status.

**Response:**
```json
{
  "risk_level": "LOW",
  "portfolio_risk": 0.15,
  "max_position_weight": 0.20,
  "correlation_risk": 0.45,
  "daily_var_95": 2500.00,
  "concentration_risk": 0.12,
  "regulatory_compliance": {
    "pdt_status": "compliant",
    "wash_sale_risk": "low",
    "reg_t_status": "compliant"
  },
  "risk_limits": {
    "max_daily_loss": 5000.00,
    "max_position_size": 10000.00,
    "max_portfolio_risk": 0.25
  },
  "current_exposure": {
    "daily_loss_used": 0.50,
    "position_limit_used": 0.75,
    "portfolio_risk_used": 0.60
  }
}
```

### Risk Limits

#### GET /api/v1/risk/limits
Get current risk limits.

**Response:**
```json
{
  "position_limits": {
    "max_position_size": 10000.00,
    "max_portfolio_weight": 0.20,
    "max_sector_weight": 0.30
  },
  "daily_limits": {
    "max_daily_loss": 5000.00,
    "max_daily_gain": 15000.00
  },
  "correlation_limits": {
    "max_correlation": 0.70,
    "rebalance_threshold": 0.80
  },
  "volatility_limits": {
    "max_portfolio_volatility": 0.25,
    "position_volatility_cap": 0.40
  }
}
```

#### PUT /api/v1/risk/limits
Update risk limits.

**Request:**
```json
{
  "position_limits": {
    "max_position_size": 15000.00
  },
  "daily_limits": {
    "max_daily_loss": 7500.00
  }
}
```

### Risk Alerts

#### GET /api/v1/risk/alerts
Get active risk alerts.

**Response:**
```json
{
  "alerts": [
    {
      "alert_id": "alert_123456",
      "type": "warning",
      "category": "position_concentration",
      "message": "AAPL position approaching 20% portfolio limit",
      "severity": "medium",
      "created_at": "2025-11-06T14:25:00Z",
      "acknowledged": false
    }
  ],
  "alert_count": 1
}
```

#### POST /api/v1/risk/alerts/{alert_id}/acknowledge
Acknowledge risk alert.

### Stress Testing

#### POST /api/v1/risk/stress-test
Run portfolio stress test.

**Request:**
```json
{
  "scenario": "market_crash_2008",
  "portfolio_value": 100000.00
}
```

**Response:**
```json
{
  "scenario": "market_crash_2008",
  "shock_description": "2008 Financial Crisis (-37% equity markets)",
  "original_value": 100000.00,
  "stressed_value": 63000.00,
  "loss_amount": -37000.00,
  "loss_percentage": -37.0,
  "recovery_time_estimate": "18 months",
  "confidence_level": 0.95
}
```

---

## 🤖 AI Integration

### Market Analysis

#### POST /api/v1/ai/analyze-market
Get AI-powered market analysis.

**Request:**
```json
{
  "symbols": ["AAPL", "GOOGL", "MSFT"],
  "analysis_type": "comprehensive",
  "timeframe": "1D",
  "use_reasoning_model": true
}
```

**Response:**
```json
{
  "analysis_id": "analysis_123456",
  "symbols": ["AAPL", "GOOGL", "MSFT"],
  "market_sentiment": {
    "overall": "bullish",
    "confidence": 0.75,
    "factors": ["strong_earnings", "technical_breakout"]
  },
  "symbol_analysis": [
    {
      "symbol": "AAPL",
      "sentiment": "bullish",
      "confidence": 0.82,
      "price_target": 155.00,
      "risk_score": 0.25,
      "key_factors": ["earnings_growth", "product_launch"]
    }
  ],
  "ai_recommendations": [
    {
      "action": "buy",
      "symbols": ["AAPL"],
      "confidence": 0.78,
      "reasoning": "Strong earnings beat with positive guidance"
    }
  ],
  "generated_at": "2025-11-06T14:30:00Z"
}
```

### Strategy Recommendation

#### POST /api/v1/ai/recommend-strategy
Get AI strategy recommendations.

**Request:**
```json
{
  "market_conditions": {
    "volatility": 0.25,
    "trend_strength": 0.60,
    "correlation": 0.30,
    "volume_profile": "high"
  },
  "risk_profile": "moderate",
  "investment_horizon": "short_term"
}
```

**Response:**
```json
{
  "recommendations": [
    {
      "strategy_id": "mean_reversion_001",
      "strategy_name": "Bollinger Bands Mean Reversion",
      "confidence": 0.85,
      "expected_return": 0.12,
      "risk_level": "medium",
      "allocation": 0.40,
      "reasoning": "Current market volatility and range-bound conditions favor mean reversion"
    },
    {
      "strategy_id": "momentum_002",
      "strategy_name": "Moving Average Crossover",
      "confidence": 0.72,
      "expected_return": 0.15,
      "risk_level": "medium-high",
      "allocation": 0.35,
      "reasoning": "Strong trend signals with volume confirmation"
    }
  ],
  "portfolio_allocation": {
    "total_allocation": 0.75,
    "cash_allocation": 0.25,
    "expected_portfolio_return": 0.13,
    "expected_portfolio_risk": 0.18
  }
}
```

### Risk Assessment

#### POST /api/v1/ai/assess-risk
Get AI-powered risk assessment.

**Request:**
```json
{
  "trade_request": {
    "symbol": "AAPL",
    "side": "buy",
    "quantity": 100,
    "order_type": "limit",
    "limit_price": 150.00
  },
  "portfolio_context": {
    "current_positions": [
      {
        "symbol": "GOOGL",
        "quantity": 50,
        "avg_price": 2800.00
      }
    ],
    "total_portfolio_value": 100000.00
  }
}
```

**Response:**
```json
{
  "risk_assessment": {
    "overall_risk_score": 0.35,
    "risk_level": "medium",
    "risk_factors": [
      {
        "factor": "position_concentration",
        "score": 0.45,
        "description": "Moderate concentration in large-cap tech"
      },
      {
        "factor": "correlation_risk",
        "score": 0.25,
        "description": "Low correlation with existing positions"
      }
    ],
    "recommendations": [
      {
        "type": "approve",
        "conditions": ["monitor_correlation", "set_stop_loss"],
        "reasoning": "Low correlation with existing positions and reasonable position size"
      }
    ]
  },
  "ai_reasoning": "Based on current market conditions and portfolio composition, this trade represents a balanced risk-reward opportunity. The position size is appropriate relative to portfolio value, and correlation analysis shows low overlap with existing holdings.",
  "confidence": 0.78
}
```

### Backtesting with AI

#### POST /api/v1/ai/backtest
Run AI-enhanced backtesting.

**Request:**
```json
{
  "strategy": "mean_reversion_001",
  "parameters": {
    "bb_period": 20,
    "bb_std": 2.0
  },
  "backtest_period": {
    "start_date": "2023-01-01",
    "end_date": "2025-11-01"
  },
  "initial_capital": 100000,
  "ai_enhancement": true
}
```

**Response:**
```json
{
  "backtest_id": "bt_123456",
  "strategy": "mean_reversion_001",
  "period": "2023-01-01 to 2025-11-01",
  "results": {
    "total_return": 18.5,
    "annualized_return": 8.2,
    "volatility": 11.8,
    "sharpe_ratio": 1.52,
    "max_drawdown": -6.2,
    "win_rate": 0.68
  },
  "ai_enhancements": {
    "parameter_optimization": {
      "original_sharpe": 1.45,
      "optimized_sharpe": 1.52,
      "improvement": 0.07
    },
    "signal_enhancement": {
      "original_signals": 145,
      "enhanced_signals": 138,
      "quality_improvement": 0.15
    }
  }
}
```

---

## 🔧 System Monitoring

### System Metrics

#### GET /api/v1/monitoring/metrics
Get system performance metrics.

**Response:**
```json
{
  "timestamp": "2025-11-06T14:30:00Z",
  "system": {
    "cpu_usage": 0.34,
    "memory_usage": 0.45,
    "disk_usage": 0.25,
    "network_io": {
      "bytes_sent": 1024000,
      "bytes_received": 2048000
    }
  },
  "application": {
    "uptime": 86400,
    "total_requests": 15420,
    "active_connections": 12,
    "average_response_time": 145
  },
  "trading": {
    "orders_processed": 145,
    "positions_tracked": 25,
    "brokers_connected": 3,
    "data_feeds_active": 8
  }
}
```

### Health Checks

#### GET /api/v1/monitoring/health
Detailed health check status.

**Response:**
```json
{
  "overall_status": "healthy",
  "timestamp": "2025-11-06T14:30:00Z",
  "checks": {
    "database": {
      "status": "healthy",
      "response_time": 15,
      "last_check": "2025-11-06T14:29:59Z"
    },
    "brokers": {
      "status": "healthy",
      "connected_brokers": 3,
      "total_brokers": 7,
      "last_heartbeat": "2025-11-06T14:29:45Z"
    },
    "ai_services": {
      "status": "healthy",
      "models_available": 3,
      "response_time": 250
    },
    "market_data": {
      "status": "healthy",
      "feeds_active": 8,
      "data_lag": 0
    }
  }
}
```

### Audit Logs

#### GET /api/v1/monitoring/audit-logs
Get audit trail logs.

**Query Parameters:**
- `event_type`: Filter by event type
- `user_id`: Filter by user
- `start_date`: Start date filter
- `end_date`: End date filter
- `limit`: Number of logs (max 100)

**Response:**
```json
{
  "logs": [
    {
      "log_id": "audit_123456789",
      "timestamp": "2025-11-06T14:30:00Z",
      "event_type": "trade_executed",
      "user_id": "usr_123456",
      "details": {
        "order_id": "ord_123456789",
        "symbol": "AAPL",
        "side": "buy",
        "quantity": 100,
        "price": 149.85
      },
      "ip_address": "192.168.1.100",
      "user_agent": "TradingApp/1.0"
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 50,
    "total": 1250,
    "pages": 25
  }
}
```

---

## 🔌 WebSocket API

### Connection

**WebSocket Endpoint:**
```
wss://api.trading-orchestrator.com/v1/ws
```

**Authentication:**
```javascript
const ws = new WebSocket('wss://api.trading-orchestrator.com/v1/ws');

ws.onopen = function() {
    ws.send(JSON.stringify({
        action: 'auth',
        token: 'YOUR_API_KEY'
    }));
};
```

### Message Format

**Client to Server:**
```json
{
  "action": "subscribe",
  "channel": "market_data",
  "symbol": "AAPL"
}
```

**Server to Client:**
```json
{
  "type": "market_data",
  "symbol": "AAPL",
  "timestamp": "2025-11-06T14:30:00Z",
  "data": {
    "bid": 150.24,
    "ask": 150.26,
    "last": 150.25,
    "volume": 25478932
  }
}
```

### Subscription Channels

#### Market Data
```json
{
  "action": "subscribe",
  "channel": "market_data",
  "symbols": ["AAPL", "GOOGL", "MSFT"]
}
```

#### Order Updates
```json
{
  "action": "subscribe",
  "channel": "order_updates"
}
```

#### Position Updates
```json
{
  "action": "subscribe",
  "channel": "position_updates"
}
```

#### Portfolio Updates
```json
{
  "action": "subscribe",
  "channel": "portfolio_updates"
}
```

#### Risk Alerts
```json
{
  "action": "subscribe",
  "channel": "risk_alerts"
}
```

### Unsubscription

```json
{
  "action": "unsubscribe",
  "channel": "market_data",
  "symbols": ["AAPL"]
}
```

---

## 🛠️ SDKs and Examples

### Python SDK

**Installation:**
```bash
pip install trading-orchestrator-sdk
```

**Basic Usage:**
```python
from orchestrator_sdk import TradingOrchestrator

# Initialize client
client = TradingOrchestrator(api_key="your_api_key")

# Get account info
account = client.account.get()
print(f"Account equity: ${account['total_equity']:,.2f}")

# Place order
order = client.orders.create({
    "symbol": "AAPL",
    "side": "buy",
    "type": "market",
    "quantity": 100
})

# Get positions
positions = client.positions.get()
for pos in positions['positions']:
    print(f"{pos['symbol']}: {pos['quantity']} shares, P&L: ${pos['unrealized_pnl']}")
```

**Advanced Example - Automated Strategy:**
```python
from orchestrator_sdk import TradingOrchestrator
from datetime import datetime

class MeanReversionBot:
    def __init__(self, api_key):
        self.client = TradingOrchestrator(api_key)
        self.symbols = ["AAPL", "GOOGL", "MSFT"]
    
    async def run_strategy(self):
        # Get market data
        quotes = self.client.market.get_quotes(self.symbols)
        
        # Analyze with AI
        analysis = self.client.ai.analyze_market(
            symbols=self.symbols,
            analysis_type="comprehensive"
        )
        
        # Execute trades based on signals
        for recommendation in analysis['ai_recommendations']:
            if recommendation['action'] == 'buy':
                await self.place_buy_order(
                    recommendation['symbols'][0],
                    100
                )
    
    async def place_buy_order(self, symbol, quantity):
        try:
            order = self.client.orders.create({
                "symbol": symbol,
                "side": "buy",
                "type": "market",
                "quantity": quantity
            })
            print(f"Order placed: {order['order_id']}")
        except Exception as e:
            print(f"Order failed: {e}")

# Run the bot
bot = MeanReversionBot("your_api_key")
await bot.run_strategy()
```

### JavaScript SDK

**Installation:**
```bash
npm install trading-orchestrator-sdk
```

**Basic Usage:**
```javascript
import { TradingOrchestrator } from 'trading-orchestrator-sdk';

const client = new TradingOrchestrator({
  apiKey: 'your_api_key',
  environment: 'production' // or 'development'
});

// Get account info
const account = await client.account.get();
console.log(`Account equity: $${account.total_equity.toLocaleString()}`);

// Place order
const order = await client.orders.create({
  symbol: 'AAPL',
  side: 'buy',
  type: 'market',
  quantity: 100
});

console.log(`Order placed: ${order.order_id}`);
```

**Real-time Trading:**
```javascript
import { WebSocketClient } from 'trading-orchestrator-sdk';

const ws = new WebSocketClient({
  apiKey: 'your_api_key',
  url: 'wss://api.trading-orchestrator.com/v1/ws'
});

ws.on('market_data', (data) => {
  console.log(`Price update for ${data.symbol}: ${data.data.last}`);
  
  // Check for trading signals
  if (data.data.change_percent > 2.0) {
    // Place buy order for momentum
    client.orders.create({
      symbol: data.symbol,
      side: 'buy',
      type: 'market',
      quantity: 100
    });
  }
});

// Subscribe to market data
ws.subscribe('market_data', ['AAPL', 'GOOGL', 'MSFT']);
```

### cURL Examples

**Get Account Information:**
```bash
curl -X GET "https://api.trading-orchestrator.com/v1/account" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json"
```

**Place Order:**
```bash
curl -X POST "https://api.trading-orchestrator.com/v1/orders" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "side": "buy",
    "type": "market",
    "quantity": 100
  }'
```

**Get Positions:**
```bash
curl -X GET "https://api.trading-orchestrator.com/v1/positions" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**AI Market Analysis:**
```bash
curl -X POST "https://api.trading-orchestrator.com/v1/ai/analyze-market" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL", "GOOGL"],
    "analysis_type": "comprehensive"
  }'
```

---

## 📞 API Support

### Getting Help

1. **Documentation**: Refer to this documentation first
2. **SDK Examples**: Check the SDK documentation for code samples
3. **Community Forum**: [forum.trading-orchestrator.com](https://forum.trading-orchestrator.com)
4. **Support Email**: api-support@trading-orchestrator.com

### Rate Limit Optimization

1. **Use WebSocket** for real-time data instead of polling
2. **Batch requests** when possible
3. **Cache data** locally to reduce API calls
4. **Implement exponential backoff** for retries

### Best Practices

1. **Always handle errors** gracefully
2. **Use appropriate timeouts** for requests
3. **Implement request retry logic** for transient failures
4. **Monitor your API usage** to avoid rate limits
5. **Use pagination** for large datasets
6. **Cache static data** (broker info, etc.) locally

This comprehensive API documentation provides everything needed to integrate with and extend the Day Trading Orchestrator system. The system is designed to be developer-friendly with clear error handling, comprehensive logging, and extensive documentation.
