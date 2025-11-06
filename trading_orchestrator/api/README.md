# Trading Strategy API Integration Layer

## Overview

The Trading Strategy API Integration Layer provides a comprehensive REST API and WebSocket endpoints for managing trading strategies, real-time monitoring, backtesting, and performance analysis. This layer integrates seamlessly with the existing strategy framework and provides a modern, scalable interface for strategy management.

## Architecture

### Core Components

1. **REST API Endpoints** (`/api/strategies/*`)
   - Complete CRUD operations for strategies
   - Backtesting and performance analysis
   - Strategy comparison and ranking
   - Strategy templates and ensembles

2. **WebSocket Endpoints** (`/ws/*`)
   - Real-time strategy signal streaming
   - Live performance metrics updates
   - Strategy alert notifications
   - System notifications

3. **Authentication & Authorization**
   - JWT-based authentication
   - Role-based access control (RBAC)
   - Strategy-specific permissions

4. **Data Models**
   - Pydantic schemas for validation
   - SQLAlchemy models for persistence
   - Complex type encoding/decoding

## API Endpoints

### Strategy Management

#### List Strategies
```
GET /api/strategies
```
**Query Parameters:**
- `page`: Page number (default: 1)
- `size`: Items per page (default: 20, max: 100)
- `category`: Filter by strategy category
- `status`: Filter by strategy status
- `risk_level`: Filter by risk level
- `search`: Search in name and description
- `tags`: Filter by tags
- `sort_by`: Sort field (default: created_at)
- `sort_order`: Sort order (asc/desc)

#### Get Strategy Details
```
GET /api/strategies/{strategy_id}
```
Returns detailed information about a specific strategy including configuration, performance metrics, and recent signals.

#### Create Strategy
```
POST /api/strategies
```
**Request Body:**
```json
{
  "config": {
    "name": "Momentum Strategy",
    "description": "Price momentum based strategy",
    "category": "momentum",
    "parameters": {
      "fast_period": 12,
      "slow_period": 26,
      "signal_period": 9
    },
    "risk_level": "medium",
    "tags": ["momentum", "technical"],
    "timeframe": "1h",
    "symbols": ["AAPL", "GOOGL", "MSFT"]
  },
  "auto_start": false,
  "validate_parameters": true
}
```

#### Update Strategy
```
PUT /api/strategies/{strategy_id}
```
Update strategy configuration, parameters, or status.

#### Delete Strategy
```
DELETE /api/strategies/{strategy_id}
```
Permanently delete a strategy (requires ownership or admin privileges).

#### Strategy Control
```
POST /api/strategies/{strategy_id}/start
POST /api/strategies/{strategy_id}/stop
```
Start or stop strategy execution.

### Performance & Analysis

#### Get Strategy Signals
```
GET /api/strategies/{strategy_id}/signals
```
**Query Parameters:**
- `limit`: Maximum number of signals (default: 50, max: 500)
- `since`: Get signals since timestamp

#### Get Performance Metrics
```
GET /api/strategies/{strategy_id}/performance
```
**Query Parameters:**
- `period`: Performance period (1d, 7d, 30d, 90d, 1y, all)

#### Run Backtest
```
POST /api/strategies/{strategy_id}/backtest
```
**Request Body:**
```json
{
  "start_date": "2023-01-01T00:00:00Z",
  "end_date": "2023-12-31T23:59:59Z",
  "initial_capital": 100000,
  "commission": 0.001,
  "slippage": 0.0005,
  "include_monte_carlo": true,
  "monte_carlo_runs": 1000,
  "walk_forward": false
}
```

#### Compare Strategies
```
POST /api/strategies/compare
```
**Query Parameters:**
- `strategy_ids`: List of strategy IDs to compare (2-10 strategies)
- `period`: Comparison period (7d, 30d, 90d, 1y, all)
- `metrics`: Specific metrics to compare

### Strategy History & Templates

#### Get Strategy History
```
GET /api/strategies/{strategy_id}/history
```
**Query Parameters:**
- `limit`: Maximum number of history records (default: 100)
- `event_type`: Filter by event type
- `since`: Get history since timestamp

#### Get Strategy Templates
```
GET /api/strategies/templates
```
**Query Parameters:**
- `category`: Filter templates by category
- `risk_level`: Filter templates by risk level

#### Create Strategy Ensemble
```
POST /api/strategies/ensemble
```
**Request Body:**
```json
{
  "name": "Multi-Strategy Portfolio",
  "description": "Balanced portfolio of multiple strategies",
  "strategy_weights": {
    "strategy_id_1": 0.4,
    "strategy_id_2": 0.3,
    "strategy_id_3": 0.3
  },
  "rebalance_frequency": "daily",
  "risk_management": {
    "max_drawdown": 0.1,
    "stop_loss": 0.05
  }
}
```

### Categories & Statistics

#### Get Strategy Categories
```
GET /api/strategies/categories
```
Returns all available strategy categories with usage statistics.

### System Endpoints

#### Health Check
```
GET /api/health
```
System health monitoring endpoint.

#### System Overview
```
GET /api/system/overview
```
Comprehensive system status and statistics (requires authentication).

## WebSocket Endpoints

### Strategy Signals
```
ws://server/ws/strategies/{strategy_id}/signals
```
**Query Parameters:**
- `user_id`: User ID for authentication
- `token`: Authentication token

**Features:**
- Real-time signal streaming
- Historical signals on connection
- Signal filtering support

### Strategy Performance
```
ws://server/ws/strategies/{strategy_id}/performance
```
**Query Parameters:**
- `user_id`: User ID for authentication
- `token`: Authentication token
- `frequency`: Update frequency in seconds (1-60)

**Features:**
- Live performance metrics updates
- Configurable update frequency
- Real-time status monitoring

### Strategy Alerts
```
ws://server/ws/strategies/alerts
```
**Query Parameters:**
- `user_id`: User ID for authentication
- `token`: Authentication token
- `strategy_ids`: Comma-separated strategy IDs

**Features:**
- Strategy status change notifications
- Performance alerts
- Risk warnings
- Strategy error notifications

### System Notifications
```
ws://server/ws/system/notifications
```
**Query Parameters:**
- `user_id`: User ID for authentication
- `token`: Authentication token
- `types`: Comma-separated notification types

**Features:**
- System-wide notifications
- API status updates
- Maintenance notifications

## Authentication & Authorization

### Authentication
The API uses JWT-based authentication. Include the token in the Authorization header:

```
Authorization: Bearer <your-jwt-token>
```

### Demo Credentials
For testing purposes, a demo user is available:
- **Username:** demo@example.com
- **Password:** demo123
- **Token:** Generated automatically upon login

### Roles & Permissions

#### Admin Role
- Full access to all strategies
- User management capabilities
- System administration features
- Can execute, modify, and delete any strategy

#### Trader Role
- Create and manage own strategies
- Execute and modify strategies
- View performance metrics
- Create ensembles
- Place trades

#### Analyst Role
- View strategy performance
- Run backtests
- Compare strategies
- View trade history

#### Viewer Role
- Read-only access to strategies
- View performance metrics
- Limited access to strategy details

### Permission System
Strategies implement granular permissions:
- **read**: View strategy details and performance
- **write**: Modify strategy configuration
- **execute**: Start/stop strategy execution
- **delete**: Permanently delete strategy
- **share**: Share strategy with other users

## Data Models

### Strategy Configuration
```python
class StrategyConfig(BaseModel):
    name: str
    description: Optional[str]
    category: StrategyCategory
    parameters: Dict[str, Any]
    risk_level: RiskLevel
    tags: List[str]
    timeframe: str
    symbols: List[str]
```

### Performance Metrics
```python
class StrategyPerformance(BaseModel):
    total_return: float
    sharpe_ratio: float
    sortino_ratio: Optional[float]
    calmar_ratio: Optional[float]
    max_drawdown: float
    win_rate: float
    profit_factor: Optional[float]
    total_trades: int
    winning_trades: int
    losing_trades: int
    # ... additional metrics
```

### Backtest Request
```python
class BacktestRequest(BaseModel):
    strategy_id: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    commission: float = 0.0
    slippage: float = 0.0
    symbols: Optional[List[str]]
    include_monte_carlo: bool = False
    monte_carlo_runs: int = 1000
    walk_forward: bool = False
    walk_forward_window: int = 252
```

## Error Handling

### Standard Error Response Format
```json
{
  "success": false,
  "message": "Error description",
  "detail": "Detailed error information",
  "errors": ["Specific error details"],
  "timestamp": "2023-12-01T12:00:00Z"
}
```

### HTTP Status Codes
- **200**: Success
- **201**: Created
- **400**: Bad Request (validation errors)
- **401**: Unauthorized (authentication required)
- **403**: Forbidden (insufficient permissions)
- **404**: Not Found (resource doesn't exist)
- **422**: Unprocessable Entity (validation failed)
- **429**: Too Many Requests (rate limited)
- **500**: Internal Server Error

### Error Types
- `VALIDATION_ERROR`: Request validation failed
- `AUTHENTICATION_ERROR`: Invalid or missing authentication
- `AUTHORIZATION_ERROR`: Insufficient permissions
- `NOT_FOUND`: Resource not found
- `CONFLICT`: Resource conflict
- `DATABASE_ERROR`: Database operation failed
- `EXTERNAL_SERVICE_ERROR`: External API/service error

## Database Schema

The API uses SQLAlchemy models for data persistence:

### Core Models
- **User**: User authentication and profile
- **StrategyDB**: Strategy configuration and metadata
- **StrategyPerformanceDB**: Performance metrics
- **StrategySignalDB**: Trading signals
- **BacktestDB**: Backtest results
- **StrategyHistoryDB**: Event history
- **UserStrategyPermission**: Strategy access permissions

### Indexes & Constraints
- Optimized indexes for common query patterns
- Foreign key constraints for data integrity
- Unique constraints for business rules
- Composite indexes for complex queries

## Configuration

### Environment Variables
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=false
API_BASE_URL=http://localhost:8000

# Security
API_CORS_ORIGINS=http://localhost:3000,http://localhost:8080
API_ALLOWED_HOSTS=localhost,127.0.0.1
ENVIRONMENT=development

# Logging
LOG_LEVEL=INFO
```

### API Settings
```python
# config/settings.py
class APISettings(BaseSettings):
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = False
    api_base_url: str = "http://localhost:8000"
    api_cors_origins: str = "*"
    api_allowed_hosts: str = "localhost"
    token_expiry_hours: int = 24
    max_sessions_per_user: int = 5
```

## Integration Examples

### Python Client Example
```python
import requests
import websockets
import json

class TradingStrategyAPI:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {token}"}
    
    def get_strategies(self, category=None, status=None):
        params = {}
        if category:
            params["category"] = category
        if status:
            params["status"] = status
        
        response = requests.get(
            f"{self.base_url}/api/strategies",
            headers=self.headers,
            params=params
        )
        return response.json()
    
    def create_strategy(self, config):
        response = requests.post(
            f"{self.base_url}/api/strategies",
            headers=self.headers,
            json=config
        )
        return response.json()
    
    async def stream_signals(self, strategy_id, callback):
        uri = f"ws://{self.base_url}/ws/strategies/{strategy_id}/signals"
        
        async with websockets.connect(uri) as websocket:
            async for message in websocket:
                data = json.loads(message)
                if data["type"] == "new_signal":
                    callback(data["data"])
```

### JavaScript Client Example
```javascript
class TradingStrategyAPI {
    constructor(baseUrl, token) {
        this.baseUrl = baseUrl;
        this.headers = {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
        };
    }
    
    async getStrategies(filters = {}) {
        const params = new URLSearchParams(filters);
        const response = await fetch(`${this.baseUrl}/api/strategies?${params}`, {
            headers: this.headers
        });
        return response.json();
    }
    
    async createStrategy(config) {
        const response = await fetch(`${this.baseUrl}/api/strategies`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify(config)
        });
        return response.json();
    }
    
    connectSignals(strategyId, onSignal) {
        const ws = new WebSocket(`ws://${this.baseUrl}/ws/strategies/${strategyId}/signals`);
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'new_signal') {
                onSignal(data.data);
            }
        };
        
        return ws;
    }
}
```

## Monitoring & Logging

### Health Checks
- `/api/health`: Basic health check
- Component-specific health monitoring
- Database connectivity checks
- External service status

### Logging
- Structured logging with JSON format
- Different log levels (DEBUG, INFO, WARNING, ERROR)
- Performance metrics logging
- Error tracking and alerting

### Metrics
- API request/response times
- WebSocket connection counts
- Strategy execution statistics
- Backtest completion rates

## Deployment

### Development
```bash
# Start API server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Access documentation
open http://localhost:8000/api/docs
```

### Production
```bash
# Start with production settings
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

# With SSL
uvicorn api.main:app --host 0.0.0.0 --port 443 --ssl-keyfile=key.pem --ssl-certfile=cert.pem
```

### Docker
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Testing

### API Testing
```python
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_get_strategies():
    response = client.get("/api/strategies")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "pagination" in data

def test_create_strategy():
    strategy_config = {
        "name": "Test Strategy",
        "category": "momentum",
        "parameters": {}
    }
    
    response = client.post("/api/strategies", json=strategy_config)
    assert response.status_code == 201
```

### WebSocket Testing
```python
import pytest
from fastapi.testclient import TestClient
import websocket

def test_websocket_signals():
    client = TestClient(app)
    
    # Create strategy first
    strategy_id = create_test_strategy()
    
    # Test WebSocket connection
    ws = websocket.create_connection(f"ws://localhost/ws/strategies/{strategy_id}/signals")
    
    # Receive signal
    message = ws.recv()
    data = json.loads(message)
    assert data["type"] == "recent_signals"
    
    ws.close()
```

## Security Considerations

### Authentication
- JWT token-based authentication
- Token expiration and refresh
- Session management
- Secure token storage

### Authorization
- Role-based access control
- Strategy-specific permissions
- Resource ownership validation
- Admin privilege separation

### Input Validation
- Pydantic model validation
- SQL injection prevention
- XSS protection
- Rate limiting

### Data Protection
- Sensitive data encryption
- Secure communication (HTTPS/WSS)
- Data privacy compliance
- Audit logging

## Performance Optimization

### Caching
- Strategy configuration caching
- Performance metrics caching
- WebSocket connection pooling
- Database query optimization

### Scalability
- Horizontal scaling support
- Load balancing ready
- Async/await patterns
- Background task processing

### Monitoring
- Real-time performance metrics
- API response time tracking
- Error rate monitoring
- Resource utilization tracking

## Future Enhancements

### Planned Features
- Strategy marketplace
- Community sharing
- Advanced analytics
- Machine learning integration
- Mobile app support

### API Versioning
- Version management
- Backward compatibility
- Deprecation handling
- Migration guides

This comprehensive API integration layer provides a robust foundation for strategy management, real-time monitoring, and performance analysis, enabling both simple and advanced use cases while maintaining security, scalability, and developer experience.
