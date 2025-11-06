"""
Trading Orchestrator Strategy API

A comprehensive REST API and WebSocket service for managing trading strategies,
backtesting, performance analysis, and real-time monitoring.

## Architecture

The API is built with FastAPI and provides:

- **Strategy Management**: CRUD operations for trading strategies
- **Backtesting Engine**: Historical strategy testing with Monte Carlo simulation
- **Performance Analytics**: Real-time and historical performance metrics
- **WebSocket Streaming**: Live strategy signals, performance updates, and alerts
- **Authentication & Authorization**: JWT-based auth with role-based access control
- **Database Integration**: SQLAlchemy models for persistent data storage

## API Structure

```
api/
├── main.py                    # FastAPI application entry point
├── auth/                      # Authentication & authorization
│   ├── authentication.py      # JWT token management
│   └── authorization.py       # Role-based access control
├── database/                  # Database models
│   └── models.py              # SQLAlchemy strategy models
├── routers/                   # API route handlers
│   ├── dependencies.py        # Common dependencies
│   └── strategies.py          # Strategy CRUD endpoints
├── schemas/                   # Pydantic request/response models
│   └── __init__.py            # Strategy schemas
├── utils/                     # Utility functions
│   ├── error_handlers.py      # Centralized error handling
│   ├── json_encoder.py        # Custom JSON encoding
│   └── response_formatter.py  # Response formatting
└── websocket/                 # WebSocket endpoints
    ├── manager.py             # Connection management
    ├── router.py              # WebSocket route handlers
    └── strategy_websocket.py  # Strategy-specific WebSocket
```

## Authentication

### JWT Token Authentication

All API endpoints (except health checks) require authentication via JWT tokens.

**Token Format:**
```
Authorization: Bearer <jwt_token>
```

**User Roles:**
- `ADMIN`: Full system access
- `TRADER`: Strategy management and execution
- `VIEWER`: Read-only access
- `ANALYST`: Read access with analytics features

### Permissions

Available permissions:
- `READ_STRATEGIES`: View strategy information
- `WRITE_STRATEGIES`: Create/update strategies
- `EXECUTE_STRATEGIES`: Start/stop strategy execution
- `DELETE_STRATEGIES`: Remove strategies
- `MANAGE_USERS`: User management (admin only)
- `VIEW_PERFORMANCE`: Access performance metrics
- `RUN_BACKTESTS`: Execute backtesting
- `MANAGE_SYSTEM`: System configuration

## Strategy Management

### Create Strategy

```http
POST /api/strategies
Content-Type: application/json
Authorization: Bearer <token>

{
  "name": "Moving Average Crossover",
  "description": "Classic MA crossover strategy",
  "category": "momentum",
  "parameters": {
    "fast_period": 10,
    "slow_period": 20,
    "symbol": "AAPL"
  },
  "risk_level": "medium",
  "tags": ["momentum", "crossover"],
  "timeframe": "1h",
  "symbols": ["AAPL", "GOOGL"]
}
```

### List Strategies

```http
GET /api/strategies?page=1&size=20&category=momentum&status=active
Authorization: Bearer <token>
```

### Get Strategy Details

```http
GET /api/strategies/{strategy_id}
Authorization: Bearer <token>
```

### Update Strategy

```http
PUT /api/strategies/{strategy_id}
Content-Type: application/json
Authorization: Bearer <token>

{
  "name": "Updated Strategy Name",
  "parameters": {
    "fast_period": 15,
    "slow_period": 30
  }
}
```

### Delete Strategy

```http
DELETE /api/strategies/{strategy_id}
Authorization: Bearer <token>
```

### Start/Stop Strategy

```http
POST /api/strategies/{strategy_id}/start
Authorization: Bearer <token>

POST /api/strategies/{strategy_id}/stop
Authorization: Bearer <token>
```

## Performance Analytics

### Get Performance Metrics

```http
GET /api/strategies/{strategy_id}/performance?period=30d
Authorization: Bearer <token>
```

**Response:**
```json
{
  "success": true,
  "data": {
    "total_return": 15.2,
    "sharpe_ratio": 1.8,
    "max_drawdown": -5.3,
    "win_rate": 0.65,
    "total_trades": 142,
    "profit_factor": 1.4,
    "volatility": 0.12,
    "beta": 1.1,
    "alpha": 0.08
  },
  "message": "Strategy performance retrieved successfully"
}
```

### Strategy Comparison

```http
POST /api/strategies/compare?period=30d&metrics=sharpe_ratio,total_return
Content-Type: application/json
Authorization: Bearer <token>

{
  "strategy_ids": ["strat_1", "strat_2", "strat_3"]
}
```

## Backtesting

### Run Backtest

```http
POST /api/strategies/{strategy_id}/backtest
Content-Type: application/json
Authorization: Bearer <token>

{
  "start_date": "2023-01-01T00:00:00Z",
  "end_date": "2023-12-31T23:59:59Z",
  "initial_capital": 100000,
  "commission": 0.001,
  "slippage": 0.0005,
  "symbols": ["AAPL", "GOOGL"],
  "include_monte_carlo": true,
  "monte_carlo_runs": 1000
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "backtest_id": "bt_12345",
    "status": "started",
    "strategy_id": "strat_1",
    "message": "Backtest started. Use the backtest ID to check status."
  },
  "message": "Backtest started successfully"
}
```

### Get Backtest Results

```http
GET /api/strategies/backtest/{backtest_id}
Authorization: Bearer <token>
```

## Strategy Signals

### Get Recent Signals

```http
GET /api/strategies/{strategy_id}/signals?limit=50
Authorization: Bearer <token>
```

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "id": "sig_123",
      "strategy_id": "strat_1",
      "symbol": "AAPL",
      "signal_type": "buy",
      "confidence": 0.85,
      "price": 150.25,
      "quantity": 100,
      "timestamp": "2023-12-01T10:30:00Z",
      "metadata": {
        "source": "ma_crossover",
        "fast_ma": 149.5,
        "slow_ma": 148.2
      }
    }
  ],
  "message": "Strategy signals retrieved successfully"
}
```

## Strategy Templates

### Get Templates

```http
GET /api/strategies/templates?category=momentum&risk_level=medium
Authorization: Bearer <token>
```

## Strategy Ensembles

### Create Ensemble

```http
POST /api/strategies/ensemble
Content-Type: application/json
Authorization: Bearer <token>

{
  "name": "Diversified Portfolio",
  "description": "Ensemble of momentum and mean reversion strategies",
  "strategy_weights": {
    "strat_momentum_1": 0.4,
    "strat_momentum_2": 0.3,
    "strat_mean_reversion_1": 0.3
  },
  "rebalance_frequency": "daily",
  "risk_management": {
    "max_portfolio_risk": 0.02,
    "stop_loss": 0.05
  }
}
```

## WebSocket Endpoints

### Strategy Signals (Real-time)

```javascript
const ws = new WebSocket('ws://api.example.com/ws/strategies/strat_123/signals?user_id=user_1&token=jwt_token');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'new_signal') {
    console.log('New signal:', data.data);
  }
};

// Send ping
ws.send(JSON.stringify({ type: 'ping' }));
```

### Strategy Performance (Real-time)

```javascript
const ws = new WebSocket('ws://api.example.com/ws/strategies/strat_123/performance?user_id=user_1&token=jwt_token&frequency=5');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'performance_update') {
    console.log('Performance update:', data.data);
  }
};
```

### Strategy Alerts

```javascript
const ws = new WebSocket('ws://api.example.com/ws/strategies/alerts?user_id=user_1&token=jwt_token&strategy_ids=strat_123,strat_456');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'strategy_alert') {
    console.log('Alert:', data.data);
  }
};
```

### System Notifications

```javascript
const ws = new WebSocket('ws://api.example.com/ws/system/notifications?user_id=user_1&token=jwt_token&types=system,strategy');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'system_notification') {
    console.log('Notification:', data.data);
  }
};
```

## Error Handling

### Standard Error Response

```json
{
  "success": false,
  "message": "Bad request",
  "detail": "Strategy name cannot be empty",
  "errors": [
    {
      "field": "name",
      "message": "Strategy name cannot be empty",
      "type": "value_error"
    }
  ],
  "timestamp": "2023-12-01T10:30:00Z"
}
```

### Common HTTP Status Codes

- `200 OK`: Successful request
- `201 Created`: Resource created successfully
- `204 No Content`: Resource deleted successfully
- `400 Bad Request`: Invalid request data
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Server error

## Rate Limiting

API endpoints are rate-limited to prevent abuse:

- **General endpoints**: 1000 requests per hour per user
- **Strategy execution**: 100 requests per hour per user
- **Backtesting**: 10 requests per hour per user
- **WebSocket connections**: 10 concurrent connections per user

Rate limit headers:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995200
```

## Pagination

List endpoints support pagination:

**Query Parameters:**
- `page`: Page number (default: 1)
- `size`: Items per page (default: 20, max: 100)
- `sort_by`: Sort field
- `sort_order`: Sort direction (asc/desc)

**Response:**
```json
{
  "success": true,
  "data": [...],
  "pagination": {
    "total": 150,
    "page": 1,
    "size": 20,
    "pages": 8,
    "has_next": true,
    "has_prev": false
  },
  "message": "Data retrieved successfully"
}
```

## Filtering and Search

### Strategy Filters

```http
GET /api/strategies?category=momentum&status=active&risk_level=medium&tags=trend&search=crossover
Authorization: Bearer <token>
```

**Available Filters:**
- `category`: Strategy category (momentum, mean_reversion, etc.)
- `status`: Strategy status (active, inactive, testing, error)
- `risk_level`: Risk level (low, medium, high)
- `tags`: Filter by tags (comma-separated)
- `search`: Search in name, description, and tags
- `created_by`: Filter by creator

### Sort Options

- `created_at`: Creation date (default)
- `updated_at`: Last update
- `name`: Strategy name
- `total_return`: Performance (requires performance data)
- `sharpe_ratio`: Risk-adjusted returns
- `max_drawdown`: Maximum drawdown

## WebSocket Message Types

### Outgoing Messages

**Connection Acknowledgment:**
```json
{
  "type": "connected",
  "data": {
    "connection_id": "conn_123",
    "status": "connected",
    "message": "WebSocket connection established"
  },
  "timestamp": "2023-12-01T10:30:00Z"
}
```

**Strategy Signal:**
```json
{
  "type": "new_signal",
  "strategy_id": "strat_123",
  "data": {
    "id": "sig_456",
    "symbol": "AAPL",
    "signal_type": "buy",
    "confidence": 0.85,
    "price": 150.25
  },
  "timestamp": "2023-12-01T10:30:00Z"
}
```

**Performance Update:**
```json
{
  "type": "performance_update",
  "strategy_id": "strat_123",
  "data": {
    "total_return": 15.2,
    "sharpe_ratio": 1.8,
    "max_drawdown": -5.3,
    "timestamp": "2023-12-01T10:30:00Z"
  },
  "timestamp": "2023-12-01T10:30:00Z"
}
```

**Alert Notification:**
```json
{
  "type": "strategy_alert",
  "data": {
    "strategy_id": "strat_123",
    "alert_type": "drawdown_limit",
    "severity": "warning",
    "message": "Drawdown limit of 5% reached",
    "data": {
      "current_drawdown": -5.2,
      "limit": -5.0
    }
  },
  "timestamp": "2023-12-01T10:30:00Z"
}
```

### Incoming Messages

**Ping/Pong:**
```json
{
  "type": "ping"
}
```

**Request Performance Data:**
```json
{
  "type": "request_performance",
  "strategy_id": "strat_123"
}
```

**Subscribe/Unsubscribe:**
```json
{
  "type": "subscribe",
  "subscriptions": ["performance", "signals"]
}
```

## Health Checks

### System Health

```http
GET /api/health
```

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "components": {
      "database": "healthy",
      "redis": "healthy",
      "trading_engine": "healthy",
      "websocket_manager": "healthy"
    },
    "timestamp": "2023-12-01T10:30:00Z"
  }
}
```

### System Overview

```http
GET /api/system/overview
Authorization: Bearer <token>
```

**Response:**
```json
{
  "success": true,
  "data": {
    "total_strategies": 25,
    "active_strategies": 12,
    "total_trades_today": 145,
    "total_portfolio_value": 2500000,
    "daily_return": 0.85,
    "system_uptime": "7d 12h 30m",
    "connected_websockets": 8
  },
  "message": "System overview retrieved successfully"
}
```

## Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/trading_db

# Authentication
JWT_SECRET_KEY=your-secret-key
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# Redis (for caching and sessions)
REDIS_URL=redis://localhost:6379/0

# External APIs
ALPHA_VANTAGE_API_KEY=your-api-key
BINANCE_API_KEY=your-binance-key
BINANCE_SECRET_KEY=your-binance-secret

# WebSocket
WS_MAX_CONNECTIONS=1000
WS_PING_INTERVAL=30
WS_PING_TIMEOUT=10

# Rate Limiting
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW=3600
```

### API Configuration

```python
# main.py
app = FastAPI(
    title="Trading Orchestrator Strategy API",
    description="Comprehensive API for trading strategy management",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://app.example.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
app.add_middleware(
    RateLimitMiddleware,
    calls=1000,
    period=3600  # 1 hour
)
```

## Database Schema

### Strategy Model

```sql
CREATE TABLE strategies (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    category VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'inactive',
    risk_level VARCHAR(10) DEFAULT 'medium',
    tags JSON DEFAULT '[]',
    timeframe VARCHAR(20) DEFAULT '1h',
    symbols JSON DEFAULT '[]',
    parameters JSON DEFAULT '{}',
    created_by VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_executed TIMESTAMP WITH TIME ZONE,
    execution_count INTEGER DEFAULT 0
);

CREATE INDEX idx_strategies_created_by ON strategies(created_by);
CREATE INDEX idx_strategies_category_status ON strategies(category, status);
CREATE INDEX idx_strategies_created_at ON strategies(created_at);
```

### Performance Model

```sql
CREATE TABLE strategy_performance (
    id VARCHAR(50) PRIMARY KEY,
    strategy_id VARCHAR(50) UNIQUE NOT NULL,
    total_return FLOAT DEFAULT 0.0,
    sharpe_ratio FLOAT DEFAULT 0.0,
    max_drawdown FLOAT DEFAULT 0.0,
    win_rate FLOAT DEFAULT 0.0,
    total_trades INTEGER DEFAULT 0,
    calculated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### Signals Model

```sql
CREATE TABLE strategy_signals (
    id VARCHAR(50) PRIMARY KEY,
    strategy_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    signal_type VARCHAR(10) NOT NULL,
    confidence FLOAT DEFAULT 0.0,
    price FLOAT,
    signal_time TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_signals_strategy_time ON strategy_signals(strategy_id, signal_time);
CREATE INDEX idx_signals_symbol_time ON strategy_signals(symbol, signal_time);
```

## Testing

### API Testing

```bash
# Install testing dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/api/

# Run with coverage
pytest --cov=api tests/api/
```

### Example Test

```python
# tests/test_strategies.py
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_create_strategy():
    async with AsyncClient(base_url="http://test") as client:
        response = await client.post(
            "/api/strategies",
            json={
                "name": "Test Strategy",
                "category": "momentum",
                "parameters": {}
            },
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert data["data"]["name"] == "Test Strategy"
```

### WebSocket Testing

```python
# tests/test_websocket.py
import pytest
from websockets.sync.client import connect

def test_strategy_signals_websocket():
    with connect("ws://test/ws/strategies/strat_123/signals?user_id=test") as websocket:
        # Send ping
        websocket.send('{"type": "ping"}')
        
        # Receive pong
        response = websocket.recv(timeout=5)
        data = json.loads(response)
        assert data["type"] == "pong"
```

## Deployment

### Docker

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/trading_db
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    
  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=trading_db
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
    
  redis:
    image: redis:7-alpine
    
volumes:
  postgres_data:
```

### Production Deployment

```bash
# Build and deploy
docker-compose -f docker-compose.prod.yml up -d

# Scale API instances
docker-compose -f docker-compose.prod.yml up -d --scale api=3
```

## Monitoring

### Application Metrics

The API exposes Prometheus metrics at `/api/metrics`:

- `api_requests_total`: Total API requests
- `api_request_duration_seconds`: Request duration
- `websocket_connections_active`: Active WebSocket connections
- `strategies_active`: Number of active strategies
- `backtests_running`: Number of running backtests

### Logging

Structured logging with JSON format:

```json
{
  "timestamp": "2023-12-01T10:30:00Z",
  "level": "INFO",
  "logger": "api.routers.strategies",
  "message": "Strategy created successfully",
  "request_id": "req_12345",
  "user_id": "user_123",
  "strategy_id": "strat_456",
  "action": "create_strategy"
}
```

## Security

### API Security

- JWT token authentication
- Role-based access control (RBAC)
- Rate limiting
- CORS protection
- Input validation and sanitization
- SQL injection protection
- XSS protection

### WebSocket Security

- Token-based authentication
- Connection rate limiting
- Message validation
- Automatic reconnection handling

### Data Protection

- Encrypted API communications (HTTPS/WSS)
- Secure password hashing (bcrypt)
- Database connection encryption
- Sensitive data masking in logs

## Support

For support and questions:

- **Documentation**: `/api/docs` (Swagger UI) and `/api/redoc` (ReDoc)
- **API Status**: `/api/health`
- **Email**: support@example.com
- **GitHub Issues**: Create an issue in the project repository

## Changelog

### v1.0.0 (2023-12-01)

- Initial API release
- Strategy CRUD operations
- WebSocket real-time updates
- Backtesting engine
- Performance analytics
- Authentication & authorization
- Rate limiting and security
- Comprehensive documentation
