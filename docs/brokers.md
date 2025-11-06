# Broker Integration Guide

## Overview

This guide covers integration with supported brokers and trading platforms.

## Supported Brokers

### 1. Alpaca Markets

#### Configuration
```json
{
  "brokers": {
    "alpaca": {
      "enabled": true,
      "api_key": "YOUR_API_KEY",
      "secret_key": "YOUR_SECRET_KEY",
      "base_url": "https://paper-api.alpaca.markets",
      "account_id": "YOUR_ACCOUNT_ID",
      "webhook_url": "https://your-domain.com/webhook/alpaca"
    }
  }
}
```

#### Features
- Real-time market data
- Paper trading support
- RESTful and WebSocket APIs
- Webhook notifications
- Comprehensive order types

#### Setup Steps
1. Create Alpaca account
2. Generate API keys
3. Configure webhook endpoints
4. Test connection with sample orders

#### API Usage Example
```python
from brokers.alpaca_client import AlpacaClient

client = AlpacaClient(
    api_key='your_key',
    secret_key='your_secret',
    base_url='https://paper-api.alpaca.markets'
)

# Get account info
account = client.get_account()
print(f"Cash: {account.cash}")
print(f"Portfolio Value: {account.portfolio_value}")

# Place order
order = client.place_order(
    symbol='AAPL',
    qty=10,
    side='buy',
    type='market',
    time_in_force='day'
)
```

### 2. Binance

#### Configuration
```json
{
  "brokers": {
    "binance": {
      "enabled": true,
      "api_key": "YOUR_API_KEY",
      "secret_key": "YOUR_SECRET_KEY",
      "base_url": "https://testnet.binance.vision",
      "sandbox_mode": true,
      "recv_window": 5000
    }
  }
}
```

#### Features
- Cryptocurrency trading
- Spot and futures markets
- Real-time WebSocket data
- Advanced order types
- Testnet support

#### Setup Steps
1. Create Binance account
2. Enable API trading
3. Generate API keys
4. Configure IP restrictions
5. Test on testnet first

#### API Usage Example
```python
from brokers.binance_client import BinanceClient

client = BinanceClient(
    api_key='your_key',
    secret_key='your_secret',
    base_url='https://testnet.binance.vision'
)

# Get account info
account_info = client.get_account()
balances = account_info['balances']

# Place order
order = client.create_order(
    symbol='BTCUSDT',
    side='BUY',
    type='MARKET',
    quantity=0.001
)
```

### 3. Interactive Brokers

#### Configuration
```json
{
  "brokers": {
    "interactive_brokers": {
      "enabled": true,
      "host": "127.0.0.1",
      "port": 7497,
      "client_id": 1,
      "account_id": "YOUR_ACCOUNT_ID",
      "currency": "USD"
    }
  }
}
```

#### Features
- Global market access
- Advanced order management
- Real-time data feeds
- Options and futures trading
- Professional-grade tools

#### Setup Steps
1. Install Trader Workstation (TWS)
2. Enable API access in TWS
3. Configure API port and settings
4. Set up firewall rules
5. Test connection

#### API Usage Example
```python
from brokers.ib_client import IBClient

client = IBClient(
    host='127.0.0.1',
    port=7497,
    client_id=1
)

# Connect
client.connect()

# Get positions
positions = client.get_positions()

# Place order
contract = Contract()
contract.symbol = 'AAPL'
contract.exchange = 'SMART'
contract.currency = 'USD'

order = Order()
order.action = 'BUY'
order.totalQuantity = 10
order.orderType = 'MKT'

client.place_order(contract, order)
```

### 4. Trading 212

#### Configuration
```json
{
  "brokers": {
    "trading212": {
      "enabled": true,
      "api_token": "YOUR_API_TOKEN",
      "base_url": "https://live.trading212.com/api/v0",
      "account_id": "YOUR_ACCOUNT_ID"
    }
  }
}
```

#### Features
- Fractional shares
- Commission-free trading
- ISA and SIPP accounts
- Wide instrument range
- Mobile-first platform

#### Setup Steps
1. Create Trading 212 account
2. Apply for API access
3. Generate API token
4. Configure account permissions
5. Test API connectivity

#### API Usage Example
```python
from brokers.trading212_client import Trading212Client

client = Trading212Client(
    api_token='your_token',
    base_url='https://live.trading212.com/api/v0'
)

# Get account info
account = client.get_account()

# Get positions
positions = client.get_positions()

# Place order
order = client.create_order(
    symbol='AAPL',
    quantity=0.5,
    side='buy',
    order_type='market'
)
```

### 5. DEGIRO

#### Configuration
```json
{
  "brokers": {
    "degiro": {
      "enabled": true,
      "username": "YOUR_USERNAME",
      "password": "YOUR_PASSWORD",
      "base_url": "https://trader.degiro.nl",
      "product_browser_url": "https://producttrader.degiro.nl"
    }
  }
}
```

#### Features
- European markets access
- Low-cost trading
- Wide product range
- Multiple account types
- Strong research tools

#### Setup Steps
1. Create DEGIRO account
2. Verify account details
3. Enable API access
4. Configure security settings
5. Test login credentials

#### API Usage Example
```python
from brokers.degiro_client import DegiroClient

client = DegiroClient(
    username='your_username',
    password='your_password',
    base_url='https://trader.degiro.nl'
)

# Login
client.login()

# Get account info
account_info = client.get_account()

# Get portfolio
portfolio = client.get_portfolio()

# Place order
order = client.create_order(
    symbol='AAPL',
    quantity=10,
    side='buy',
    order_type='market'
)
```

### 6. XTB

#### Configuration
```json
{
  "brokers": {
    "xtb": {
      "enabled": true,
      "api_login": "YOUR_LOGIN",
      "api_password": "YOUR_PASSWORD",
      "server": "demo",
      "app_name": "YourAppName"
    }
  }
}
```

#### Features
- FX and CFD trading
- Advanced charting
- Social trading features
- Multiple account types
- Educational resources

#### Setup Steps
1. Create XTB account
2. Choose demo or live trading
3. Generate API credentials
4. Configure app registration
5. Test connection

#### API Usage Example
```python
from brokers.xtb_client import XTBClient

client = XTBClient(
    api_login='your_login',
    api_password='your_password',
    server='demo'
)

# Login
client.login()

# Get account info
account_info = client.get_account()

# Get trading instruments
instruments = client.get_instruments()

# Place order
trade = client.create_trade(
    action='buy',
    symbol='AAPL',
    volume=0.1,
    type='market'
)
```

### 7. Trade Republic

#### Configuration
```json
{
  "brokers": {
    "trade_republic": {
      "enabled": true,
      "username": "YOUR_USERNAME",
      "password": "YOUR_PASSWORD",
      "base_url": "https://app.traderepublic.com"
    }
  }
}
```

#### Features
- German broker
- Cost-effective trading
- ETF savings plans
- Mobile trading app
- Advanced order types

#### Setup Steps
1. Create Trade Republic account
2. Verify identity
3. Enable API access
4. Configure 2FA
5. Test connection

#### API Usage Example
```python
from brokers.trade_republic_client import TradeRepublicClient

client = TradeRepublicClient(
    username='your_username',
    password='your_password',
    base_url='https://app.traderepublic.com'
)

# Login
client.login()

# Get account info
account_info = client.get_account()

# Get portfolio
portfolio = client.get_portfolio()

# Place order
order = client.create_order(
    symbol='AAPL',
    quantity=5,
    side='buy',
    order_type='limit',
    price=150.00
)
```

## Best Practices

### Security
- Store API keys securely
- Use environment variables
- Enable IP whitelisting
- Rotate credentials regularly
- Monitor API usage

### Error Handling
- Implement retry logic
- Handle rate limits
- Log all API calls
- Set up alerts for failures
- Have backup brokers

### Performance
- Use connection pooling
- Implement caching
- Batch API calls
- Monitor latency
- Optimize order routing

### Testing
- Use paper trading first
- Test all order types
- Verify data accuracy
- Test error scenarios
- Monitor for drift

## Multi-Broker Setup

### Configuration
```json
{
  "brokers": {
    "primary": "alpaca",
    "fallback": [
      "interactive_brokers",
      "trading212"
    ],
    "routing_rules": {
      "equities": "alpaca",
      "crypto": "binance",
      "options": "interactive_brokers"
    }
  }
}
```

### Usage
```python
from brokers.broker_manager import BrokerManager

# Initialize broker manager
broker_manager = BrokerManager(config)

# Route to appropriate broker
broker = broker_manager.get_broker('alpaca')

# Use broker
account = broker.get_account()
order = broker.place_order(order_params)
```

## Troubleshooting

### Common Issues
- Authentication failures
- Connection timeouts
- Rate limit exceeded
- Invalid order parameters
- Insufficient funds

### Debug Tools
- API test endpoints
- Connection diagnostics
- Order status tracking
- Balance verification
- Log analysis

---

*For additional support, see docs/troubleshooting.md*