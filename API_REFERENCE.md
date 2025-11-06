# Trading Orchestrator API Reference

## Overview

This document provides comprehensive API reference documentation for the Trading Orchestrator system. It covers all major modules, classes, and methods with detailed descriptions, parameters, return values, and usage examples.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Broker Integration](#broker-integration)
3. [AI Orchestrator](#ai-orchestrator)
4. [Risk Management](#risk-management)
5. [Order Management System](#order-management-system)
6. [Trading Strategies](#trading-strategies)
7. [Configuration](#configuration)
8. [User Interface](#user-interface)
9. [Utilities](#utilities)
10. [Database Models](#database-models)

---

## System Architecture

### Core Components

The Trading Orchestrator is built with a modular architecture:

- **Broker Factory**: Multi-broker service management
- **AI Orchestrator**: AI-driven trading coordination
- **Risk Manager**: Portfolio risk monitoring and control
- **Order Manager**: Order execution and tracking
- **Strategy Engine**: Trading strategy execution
- **Terminal UI**: Real-time monitoring interface
- **Database Layer**: Persistent data storage

### Module Dependencies

```mermaid
graph TD
    A[Application Core] --> B[Broker Factory]
    A --> C[AI Orchestrator]
    A --> D[Risk Manager]
    A --> E[Order Manager]
    
    C --> F[Strategy Engine]
    C --> G[AI Models Manager]
    C --> H[Trading Tools]
    
    B --> I[Base Broker Interface]
    I --> J[Binance Broker]
    I --> K[Alpaca Broker]
    I --> L[IBKR Broker]
    
    D --> M[Risk Components]
    E --> N[OMS Components]
    
    F --> O[Strategy Base]
    O --> P[Trend Following]
    O --> Q[Mean Reversion]
    O --> R[Pairs Trading]
```

---

## Broker Integration

### BaseBroker (Abstract Base Class)

**Location**: `brokers/base.py`

**Description**: Abstract base class that defines the unified interface for all broker implementations.

#### Key Methods

##### `connect() -> bool`
- **Purpose**: Establish connection to broker API
- **Returns**: `True` if connection successful
- **Throws**: `ConnectionError` if connection fails
- **Example**:
```python
broker = AlpacaBroker(config)
success = await broker.connect()
```

##### `get_account() -> AccountInfo`
- **Purpose**: Retrieve account information
- **Returns**: `AccountInfo` with balance, equity, and trading permissions
- **Example**:
```python
account = await broker.get_account()
print(f"Balance: ${account.balance}")
```

##### `place_order(symbol, side, order_type, quantity, **kwargs) -> OrderInfo`
- **Purpose**: Place a new trading order
- **Parameters**:
  - `symbol`: Trading symbol (e.g., "AAPL")
  - `side`: "buy" or "sell"
  - `order_type`: "market", "limit", "stop"
  - `quantity`: Number of shares/units
  - `limit_price`: Price for limit orders
  - `stop_price`: Price for stop orders
- **Returns**: `OrderInfo` with order details
- **Example**:
```python
order = await broker.place_order(
    symbol="AAPL",
    side="buy",
    order_type="limit",
    quantity=100,
    limit_price=150.00
)
```

#### Data Classes

##### AccountInfo
- `account_id`: Unique account identifier
- `broker_name`: Broker name
- `balance`: Available cash balance
- `equity`: Total account equity
- `buying_power`: Available buying power
- `is_pattern_day_trader`: PDT status

##### PositionInfo
- `symbol`: Trading symbol
- `side`: "long" or "short"
- `quantity`: Position size
- `avg_entry_price`: Average entry price
- `current_price`: Current market price
- `unrealized_pnl`: Unrealized profit/loss
- `unrealized_pnl_percent`: P&L as percentage

##### OrderInfo
- `order_id`: Unique order identifier
- `symbol`: Trading symbol
- `order_type`: Order type
- `side`: Buy or sell side
- `quantity`: Order quantity
- `filled_quantity`: Executed quantity
- `status`: Order status (pending, filled, cancelled)
- `limit_price`: Order limit price
- `submitted_at`: Order submission time

### BinanceBroker

**Location**: `brokers/binance_broker.py`

**Description**: Binance cryptocurrency exchange integration

**Features**:
- REST API for account, orders, and market data
- WebSocket streaming for real-time quotes
- Testnet support for paper trading
- Rate limit compliance
- Multi-asset trading support

**Configuration**:
```python
config = BrokerConfig(
    broker_name="binance",
    api_key="your_api_key",
    api_secret="your_api_secret",
    is_paper=True,
    config={"testnet": True}
)
```

**Supported Markets**:
- Spot trading pairs (BTC/USDT, ETH/USDT, etc.)
- Real-time market data
- Order book data
- Trade history

### AlpacaBroker

**Location**: `brokers/alpaca_broker.py`

**Description**: Alpaca Markets US equities and crypto trading

**Features**:
- US equities and ETFs trading
- Paper trading environment
- Fractional share support
- Extended hours trading
- Real-time market data

**Supported Markets**:
- US stocks and ETFs
- Cryptocurrency pairs
- Paper trading simulation

---

## AI Orchestrator

### AITradingOrchestrator

**Location**: `ai/orchestrator.py`

**Description**: Central AI-driven trading coordinator

#### Trading Modes

##### TradingMode Enum
- `PAPER`: Paper trading with simulation
- `LIVE`: Live trading with real money
- `ANALYSIS`: Analysis only, no trading
- `BACKTEST`: Backtest mode

##### StrategyType Enum
- `MOMENTUM`: Trend following
- `MEAN_REVERSION`: Buy low, sell high
- `PAIRS_TRADING`: Statistical arbitrage
- `CROSS_VENUE`: Cross-exchange arbitrage
- `AI_DISCRETIONARY`: Full AI decision making
- `HYBRID`: AI + rule-based

#### Key Methods

##### `analyze_market(symbols, analysis_type, use_reasoning_model) -> Dict`
- **Purpose**: Analyze market conditions for symbols
- **Parameters**:
  - `symbols`: List of symbols to analyze
  - `analysis_type`: "quick", "comprehensive", "deep"
  - `use_reasoning_model`: Use high-quality reasoning model
- **Returns**: Market analysis results with insights
- **Example**:
```python
orchestrator = AITradingOrchestrator(...)
analysis = await orchestrator.analyze_market(
    symbols=["AAPL", "GOOGL"],
    analysis_type="comprehensive"
)
```

##### `generate_trading_strategy(strategy_type, symbols, parameters) -> Dict`
- **Purpose**: Generate AI-powered trading strategy
- **Parameters**:
  - `strategy_type`: Type of strategy to generate
  - `symbols`: Symbols to trade
  - `parameters`: Optional strategy parameters
- **Returns**: Generated strategy configuration
- **Example**:
```python
strategy = await orchestrator.generate_trading_strategy(
    strategy_type=StrategyType.MOMENTUM,
    symbols=["AAPL", "MSFT"],
    parameters={"period": 20, "threshold": 0.02}
)
```

##### `evaluate_trading_opportunity(symbol, opportunity_type, context) -> Dict`
- **Purpose**: Evaluate specific trading opportunity
- **Parameters**:
  - `symbol`: Symbol for opportunity
  - `opportunity_type`: Type (breakout, reversal, arbitrage)
  - `context`: Additional context information
- **Returns**: Evaluation with recommendation
- **Example**:
```python
evaluation = await orchestrator.evaluate_trading_opportunity(
    symbol="AAPL",
    opportunity_type="breakout",
    context={"price": 150.0, "volume": "high"}
)
```

##### `execute_ai_trade(symbol, side, reasoning, risk_check) -> Dict`
- **Purpose**: Execute trade with AI reasoning and risk checks
- **Parameters**:
  - `symbol`: Symbol to trade
  - `side`: "buy" or "sell"
  - `reasoning`: AI reasoning for trade
  - `risk_check`: Perform risk checks
- **Returns**: Trade execution result
- **Example**:
```python
result = await orchestrator.execute_ai_trade(
    symbol="AAPL",
    side="buy",
    reasoning="Strong momentum with volume confirmation",
    risk_check=True
)
```

---

## Risk Management

### RiskManager

**Location**: `risk/engine.py`

**Description**: Central risk management coordinator

#### Risk Levels

##### RiskLevel Enum
- `LOW`: Conservative strategies with minimal risk
- `MEDIUM`: Moderate risk with balanced risk/reward
- `HIGH`: Aggressive strategies with higher risk
- `CRITICAL`: Highest risk requiring special monitoring

#### Key Components

##### RiskLimitChecker
- Portfolio limit enforcement
- Position size validation
- Daily loss limits
- Concentration checks

##### PolicyEngine
- Trading policy validation
- Regulatory compliance
- Rule-based restrictions

##### CircuitBreakerManager
- Market disruption handling
- Automatic trading halts
- Volatility controls

##### ComplianceEngine
- Regulatory monitoring
- Audit trail maintenance
- Compliance reporting

#### Key Methods

##### `check_order_risk(symbol, side, quantity, order_type) -> RiskResult`
- **Purpose**: Validate proposed trade against risk limits
- **Parameters**:
  - `symbol`: Trading symbol
  - `side`: Buy or sell
  - `quantity`: Proposed quantity
  - `order_type`: Order type
- **Returns**: Risk assessment result with approval/rejection
- **Example**:
```python
risk_manager = RiskManager(user_id=123)
result = await risk_manager.check_order_risk(
    symbol="AAPL",
    side="buy",
    quantity=100,
    order_type="market"
)
```

##### `get_risk_status() -> Dict`
- **Purpose**: Get current portfolio risk status
- **Returns**: Risk metrics and status information
- **Example**:
```python
status = await risk_manager.get_risk_status()
print(f"Risk Level: {status['risk_level']}")
print(f"Portfolio Risk: {status['portfolio_risk']:.2%}")
```

##### `start_monitoring()`
- **Purpose**: Start continuous risk monitoring
- **Details**: Begins background monitoring of portfolio risk
- **Example**:
```python
await risk_manager.start_monitoring()
```

---

## Order Management System

### OrderManagementSystem

**Location**: `oms/engine.py`

**Description**: Central order management and execution coordinator

#### Order States

##### OrderStatus Enum
- `PENDING`: Order submitted, awaiting validation
- `VALIDATED`: Order passed validation
- `ROUTED`: Order sent to broker
- `PARTIALLY_FILLED`: Partial execution
- `FILLED`: Fully executed
- `CANCELLED`: Cancelled by user/system
- `REJECTED`: Rejected by broker/risk system

#### Key Components

##### OrderRouter
- Multi-broker order routing
- Best execution selection
- Fallback handling

##### OrderValidator
- Pre-trade validation
- Risk limit checking
- Compliance verification

##### ExecutionMonitor
- Real-time order tracking
- Fill monitoring
- Execution quality measurement

##### PositionManager
- Position state management
- P&L tracking
- Reconciliation

#### Key Methods

##### `submit_order(order_request) -> OrderResult`
- **Purpose**: Submit order for execution
- **Parameters**:
  - `order_request`: Order details dict
- **Returns**: Order submission result with ID
- **Example**:
```python
oms = OrderManagementSystem(user_id=123)
order_request = {
    'symbol': 'AAPL',
    'side': 'buy',
    'quantity': 100,
    'order_type': 'market',
    'time_in_force': 'day'
}
result = await oms.submit_order(order_request)
```

##### `get_order_status(order_id) -> Dict`
- **Purpose**: Get current order status
- **Parameters**:
  - `order_id`: Order identifier
- **Returns**: Order status and details
- **Example**:
```python
status = await oms.get_order_status(result.order_id)
print(f"Status: {status['status']}")
print(f"Filled: {status['filled_quantity']}")
```

##### `get_positions() -> List[Position]`
- **Purpose**: Get current positions across all brokers
- **Returns**: List of position objects
- **Example**:
```python
positions = await oms.get_positions()
for pos in positions:
    print(f"{pos.symbol}: {pos.quantity} @ ${pos.avg_entry_price}")
```

##### `get_execution_analytics() -> Dict`
- **Purpose**: Get execution performance analytics
- **Returns**: Performance metrics and analytics
- **Example**:
```python
analytics = await oms.get_execution_analytics()
print(f"Total Slippage: {analytics['total_slippage']:.4f}")
print(f"Execution Speed: {analytics['avg_fill_time']}ms")
```

---

## Trading Strategies

### BaseStrategy (Abstract Base Class)

**Location**: `strategies/base.py`

**Description**: Abstract base class for all trading strategies

#### Strategy Types

##### StrategyType Enum
- `TREND_FOLLOWING`: Trend identification and trading
- `MEAN_REVERSION`: Price deviation exploitation
- `PAIRS_TRADING`: Statistical arbitrage
- `ARBITRAGE`: Risk-free profit opportunities
- `MOMENTUM`: Price momentum trading
- `BREAKOUT`: Range breakout trading
- `SCALPING`: Short-term profit capture
- `SWING_TRADING`: Medium-term price swings

#### Signal Types

##### SignalType Enum
- `BUY`: Initiate long position
- `SELL`: Close long or initiate short
- `HOLD`: Maintain current position
- `CLOSE`: Exit position completely
- `REDUCE`: Decrease position size
- `INCREASE`: Add to existing position

#### Data Classes

##### TradingSignal
```python
@dataclass
class TradingSignal:
    signal_id: str
    strategy_id: str
    symbol: str
    signal_type: SignalType
    confidence: float  # 0.0 to 1.0
    strength: float    # 0.0 to 1.0
    price: Decimal
    quantity: Decimal
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    time_horizon: Optional[timedelta] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
```

##### StrategyMetrics
```python
@dataclass
class StrategyMetrics:
    strategy_id: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    average_trade: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
```

##### StrategyConfig
```python
@dataclass
class StrategyConfig:
    strategy_id: str
    strategy_type: StrategyType
    name: str
    description: str
    parameters: Dict[str, Any]
    risk_level: RiskLevel
    max_position_size: Decimal = Decimal('100000')
    max_daily_loss: Decimal = Decimal('10000')
    max_trades_per_day: int = 50
    symbols: List[str] = field(default_factory=list)
    enabled: bool = True
```

#### Key Methods

##### `generate_signals() -> List[TradingSignal]`
- **Purpose**: Generate trading signals based on strategy logic (abstract)
- **Returns**: List of trading signals
- **Example**:
```python
signals = await strategy.generate_signals()
for signal in signals:
    print(f"{signal.symbol}: {signal.signal_type.value}")
```

##### `validate_signal(signal) -> bool`
- **Purpose**: Validate trading signal (abstract)
- **Parameters**:
  - `signal`: Signal to validate
- **Returns**: `True` if signal is valid
- **Example**:
```python
is_valid = await strategy.validate_signal(signal)
```

##### `run()`
- **Purpose**: Execute the trading strategy
- **Details**: Main strategy execution loop
- **Example**:
```python
await strategy.run()
```

##### `get_status() -> Dict`
- **Purpose**: Get current strategy status
- **Returns**: Status information and metrics
- **Example**:
```python
status = await strategy.get_status()
print(f"Status: {status['status']}")
```

### TrendFollowingStrategy

**Location**: `strategies/trend_following.py`

**Description**: Moving average crossover strategy with momentum confirmation

#### Parameters
- `fast_period`: Fast moving average period (default: 10)
- `slow_period`: Slow moving average period (default: 30)
- `momentum_period`: Momentum calculation period (default: 14)
- `volume_threshold`: Volume confirmation threshold (default: 1.2)
- `signal_threshold`: Minimum signal strength (default: 0.6)
- `stop_loss`: Stop loss percentage (default: 0.02)
- `take_profit`: Take profit percentage (default: 0.06)

#### Usage Example
```python
config = StrategyConfig(
    strategy_id="trend_001",
    strategy_type=StrategyType.TREND_FOLLOWING,
    name="Moving Average Crossover",
    description="Trend following with momentum confirmation",
    parameters={
        "fast_period": 12,
        "slow_period": 26,
        "momentum_period": 14,
        "volume_threshold": 1.2,
        "signal_threshold": 0.6,
        "stop_loss": 0.02,
        "take_profit": 0.06
    },
    risk_level=RiskLevel.MEDIUM,
    symbols=["AAPL", "GOOGL", "MSFT"]
)

strategy = TrendFollowingStrategy(config)
strategy.set_context(execution_context)
await strategy.run()
```

### MeanReversionStrategy

**Location**: `strategies/mean_reversion.py`

**Description**: Mean reversion strategy using Bollinger Bands and RSI

#### Parameters
- `bb_period`: Bollinger Bands period (default: 20)
- `bb_std`: Bollinger Bands standard deviations (default: 2.0)
- `rsi_period`: RSI period (default: 14)
- `rsi_oversold`: RSI oversold threshold (default: 30)
- `rsi_overbought`: RSI overbought threshold (default: 70)

### PairsTradingStrategy

**Location**: `strategies/pairs_trading.py`

**Description**: Statistical arbitrage between correlated instrument pairs

#### Parameters
- `lookback_period`: Price correlation lookback (default: 252)
- `entry_threshold`: Z-score entry threshold (default: 2.0)
- `exit_threshold`: Z-score exit threshold (default: 0.5)
- `min_correlation`: Minimum correlation for pairs (default: 0.7)

---

## Configuration

### Application Settings

**Location**: `config/settings.py`

**Description**: Centralized application configuration management

#### Configuration Categories

##### Database Settings
```python
DATABASE_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "trading_orchestrator",
    "username": "user",
    "password": "password",
    "pool_size": 10,
    "max_overflow": 20
}
```

##### Broker Settings
```python
BROKER_CONFIGS = {
    "binance": {
        "api_key": "your_api_key",
        "api_secret": "your_api_secret",
        "testnet": True,
        "rate_limit": 1200  # requests per minute
    },
    "alpaca": {
        "api_key": "your_api_key",
        "api_secret": "your_api_secret",
        "paper": True,
        "base_url": "https://paper-api.alpaca.markets"
    }
}
```

##### AI Model Settings
```python
AI_MODEL_CONFIG = {
    "reasoning_model": {
        "provider": "openai",
        "model": "gpt-4",
        "temperature": 0.1,
        "max_tokens": 2000
    },
    "fast_model": {
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        "temperature": 0.2,
        "max_tokens": 1000
    }
}
```

##### Risk Settings
```python
RISK_CONFIG = {
    "max_position_size": 100000,
    "max_daily_loss": 10000,
    "max_portfolio_risk": 0.02,  # 2%
    "circuit_breaker_threshold": 0.05,  # 5%
    "correlation_limit": 0.7
}
```

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost/trading_db

# API Keys
BINANCE_API_KEY=your_binance_key
BINANCE_API_SECRET=your_binance_secret
ALPACA_API_KEY=your_alpaca_key
ALPACA_API_SECRET=your_alpaca_secret

# AI Models
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Security
SECRET_KEY=your_secret_key
ENCRYPTION_KEY=your_encryption_key
```

---

## User Interface

### TerminalUI

**Location**: `ui/terminal.py`

**Description**: Matrix-themed terminal interface for monitoring

#### Features
- Real-time dashboard updates
- Account and position monitoring
- Order management interface
- Risk metrics display
- System health monitoring

#### Key Methods

##### `start_monitoring(refresh_interval=1.0)`
- **Purpose**: Start terminal UI monitoring
- **Parameters**:
  - `refresh_interval`: Update frequency in seconds
- **Example**:
```python
ui = TerminalUI()
await ui.start_monitoring(refresh_interval=2.0)
```

##### `update_positions(positions_data)`
- **Purpose**: Update positions display
- **Parameters**:
  - `positions_data`: List of position information
- **Example**:
```python
ui.update_positions([
    {"symbol": "AAPL", "quantity": 100, "pnl": 250.50}
])
```

##### `update_orders(orders_data)`
- **Purpose**: Update orders display
- **Parameters**:
  - `orders_data`: List of order information
- **Example**:
```python
ui.update_orders([
    {"symbol": "GOOGL", "side": "buy", "quantity": 50, "status": "pending"}
])
```

##### `export_display(filename)`
- **Purpose**: Export current display to file
- **Parameters**:
  - `filename`: Output filename
- **Example**:
```python
ui.export_display("dashboard_export.txt")
```

---

## Utilities

### Logger

**Location**: `utils/logger.py`

**Description**: Matrix-themed logging with trading event tracking

#### Log Levels

##### LogLevel Enum
- `CRITICAL`: System-breaking errors (🟥)
- `ERROR`: Significant errors (🟧)
- `WARNING`: Important conditions (🟨)
- `INFO`: Operational information (🟩)
- `DEBUG`: Debugging information (🟦)
- `TRACE`: Very detailed tracing (⬜)

#### Trading Events

##### TradingEventType Enum
- `ORDER_SUBMITTED`: New order submission
- `ORDER_FILLED`: Order execution
- `ORDER_CANCELLED`: Order cancellation
- `POSITION_OPENED`: Position initiation
- `POSITION_CLOSED`: Position closure
- `RISK_VIOLATION`: Risk limit violation
- `CIRCUIT_TRIGGERED`: Circuit breaker activation
- `MARKET_UPDATE`: Market data update
- `STRATEGY_SIGNAL`: Trading signal generation
- `SYSTEM_HEALTH`: System health update

#### Usage Examples

##### Basic Logging
```python
from utils.logger import logger

logger.info("Trading session started")
logger.warning("Risk limit approaching")
logger.error("Order execution failed")
```

##### Trading Event Logging
```python
from utils.logger import TradingEventType, log_trading_event

log_trading_event(
    event_type=TradingEventType.ORDER_SUBMITTED,
    symbol="AAPL",
    side="buy",
    quantity=100,
    price=150.00
)
```

##### Structured Logging
```python
from utils.logger import structured_log

structured_log(
    level="INFO",
    event="trade_executed",
    data={
        "symbol": "AAPL",
        "pnl": 250.50,
        "duration": "2h 15m"
    }
)
```

---

## Database Models

### Trading Models

#### Order Model
```python
class Order(Base):
    __tablename__ = "orders"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)  # buy/sell
    quantity = Column(Numeric, nullable=False)
    order_type = Column(String, nullable=False)  # market/limit/stop
    status = Column(String, default="pending")
    filled_quantity = Column(Numeric, default=0)
    avg_fill_price = Column(Numeric)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
```

#### Position Model
```python
class Position(Base):
    __tablename__ = "positions"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    symbol = Column(String, nullable=False)
    quantity = Column(Numeric, nullable=False)
    avg_entry_price = Column(Numeric, nullable=False)
    current_price = Column(Numeric)
    unrealized_pnl = Column(Numeric, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
```

#### Trade Model
```python
class Trade(Base):
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True)
    order_id = Column(Integer, ForeignKey("orders.id"))
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)
    quantity = Column(Numeric, nullable=False)
    price = Column(Numeric, nullable=False)
    commission = Column(Numeric, default=0)
    executed_at = Column(DateTime, default=datetime.utcnow)
```

### Risk Models

#### RiskLimit Model
```python
class RiskLimit(Base):
    __tablename__ = "risk_limits"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    limit_type = Column(String, nullable=False)  # position/daily/drawdown
    symbol = Column(String)  # NULL for portfolio limits
    max_value = Column(Numeric, nullable=False)
    current_value = Column(Numeric, default=0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
```

#### RiskEvent Model
```python
class RiskEvent(Base):
    __tablename__ = "risk_events"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    event_type = Column(String, nullable=False)
    symbol = Column(String)
    severity = Column(String)  # low/medium/high/critical
    description = Column(Text)
    triggered_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime)
    status = Column(String, default="open")  # open/resolved/ignored
```

---

## Error Handling

### Common Exceptions

#### Trading Errors
```python
class TradingError(Exception):
    """Base class for trading-related errors"""
    pass

class OrderExecutionError(TradingError):
    """Order execution failure"""
    pass

class InsufficientFundsError(TradingError):
    """Insufficient account balance"""
    pass

class RiskLimitViolationError(TradingError):
    """Risk limit exceeded"""
    pass

class MarketDataError(TradingError):
    """Market data retrieval failure"""
    pass
```

#### Broker Errors
```python
class BrokerConnectionError(Exception):
    """Broker connection failure"""
    pass

class BrokerAPIError(Exception):
    """Broker API error response"""
    pass

class AuthenticationError(Exception):
    """Authentication failure"""
    pass
```

#### Strategy Errors
```python
class StrategyError(Exception):
    """Base class for strategy errors"""
    pass

class SignalGenerationError(StrategyError):
    """Signal generation failure"""
    pass

class StrategyConfigurationError(StrategyError):
    """Invalid strategy configuration"""
    pass
```

### Error Handling Patterns

#### Try-Catch with Logging
```python
try:
    order = await broker.place_order(...)
except OrderExecutionError as e:
    logger.error(f"Order execution failed: {e}")
    # Handle order failure
except InsufficientFundsError as e:
    logger.warning(f"Insufficient funds: {e}")
    # Notify user or reduce order size
except Exception as e:
    logger.critical(f"Unexpected error: {e}")
    # Handle unexpected errors
```

#### Retry with Backoff
```python
import asyncio

async def retry_with_backoff(func, max_retries=3, base_delay=1.0):
    for attempt in range(max_retries):
        try:
            return await func()
        except (BrokerConnectionError, MarketDataError) as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
            await asyncio.sleep(delay)
```

---

## Best Practices

### Security

1. **API Key Management**
   - Store API keys in environment variables
   - Use separate keys for paper and live trading
   - Rotate keys regularly
   - Use least-privilege principles

2. **Data Protection**
   - Encrypt sensitive data at rest
   - Use secure communication (HTTPS/WSS)
   - Implement proper access controls
   - Log security events

3. **Risk Management**
   - Set appropriate position limits
   - Use stop losses consistently
   - Monitor correlation between positions
   - Implement circuit breakers

### Performance

1. **Database Optimization**
   - Use connection pooling
   - Index frequently queried columns
   - Implement proper pagination
   - Monitor query performance

2. **Memory Management**
   - Clear caches periodically
   - Use generators for large datasets
   - Monitor memory usage
   - Implement proper cleanup

3. **Network Optimization**
   - Implement rate limiting
   - Use connection pooling
   - Cache frequently accessed data
   - Monitor network latency

### Testing

1. **Unit Testing**
   - Test all strategy logic
   - Mock external dependencies
   - Test error conditions
   - Validate input parameters

2. **Integration Testing**
   - Test broker integrations
   - Test database operations
   - Test end-to-end workflows
   - Validate configuration

3. **Paper Trading**
   - Always test in paper mode first
   - Validate strategy performance
   - Test risk management
   - Monitor for edge cases

### Monitoring

1. **Logging**
   - Use structured logging
   - Log all trading events
   - Monitor system health
   - Set up alerting

2. **Metrics**
   - Track strategy performance
   - Monitor execution quality
   - Measure system performance
   - Track user activity

3. **Alerting**
   - Set up risk alerts
   - Monitor system failures
   - Track performance degradation
   - Implement escalation procedures

---

## Deployment

### Production Checklist

#### Configuration
- [ ] All API keys configured
- [ ] Database connection tested
- [ ] Risk limits set appropriately
- [ ] Broker connections validated
- [ ] Logging configured

#### Security
- [ ] SSL/TLS certificates installed
- [ ] Firewall rules configured
- [ ] Access controls implemented
- [ ] Backup systems tested
- [ ] Security monitoring enabled

#### Monitoring
- [ ] Log aggregation configured
- [ ] Metrics collection enabled
- [ ] Alerting rules set up
- [ ] Health checks implemented
- [ ] Performance monitoring active

#### Testing
- [ ] Paper trading validated
- [ ] Risk management tested
- [ ] Broker integrations verified
- [ ] Database operations tested
- [ ] Error handling validated

### Scaling Considerations

#### Horizontal Scaling
- Use load balancers for API endpoints
- Implement stateless service design
- Use message queues for async processing
- Consider microservices architecture

#### Database Scaling
- Use read replicas for queries
- Implement database sharding
- Use caching layers (Redis)
- Monitor database performance

#### Network Scaling
- Use CDN for static assets
- Implement API rate limiting
- Use connection pooling
- Monitor network performance

---

## Conclusion

This API reference provides comprehensive documentation for the Trading Orchestrator system. For additional information or questions:

- Review the source code in the `/trading_orchestrator` directory
- Check the `README.md` for setup instructions
- Refer to the `docs/` directory for detailed guides
- Examine the example configurations in the root directory

The system is designed to be modular and extensible, allowing for easy customization and enhancement of trading strategies and risk management components.
