# Broker Integration Guide

This guide provides detailed information on integrating and configuring different brokers with the Day Trading Orchestrator system.

## Table of Contents

- [Supported Brokers Overview](#supported-brokers-overview)
- [Alpaca Trading Integration](#alpaca-trading-integration)
- [Binance Integration](#binance-integration)
- [Interactive Brokers Integration](#interactive-brokers-integration)
- [Trading 212 Integration](#trading-212-integration)
- [DEGIRO Integration](#degiro-integration)
- [XTB Integration](#xtb-integration)
- [Trade Republic Integration](#trade-republic-integration)
- [Multi-Broker Configuration](#multi-broker-configuration)
- [Broker Selection Guide](#broker-selection-guide)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Supported Brokers Overview

| Broker | Markets | Minimum | Commission | API Quality | Risk Level |
|--------|---------|---------|------------|-------------|------------|
| **Alpaca** | US Stocks, Crypto | $0 | $0 | Excellent | Low |
| **Binance** | Crypto | Varies | 0.1% | Excellent | Medium |
| **Interactive Brokers** | Global | $10 | $0.005 | Good | Low |
| **Trading 212** | EU Stocks | £1 | £0 | Good | Low |
| **DEGIRO** | EU Stocks | €1 | Varies | Limited | Medium |
| **XTB** | Forex, CFDs | $1 | Spread | Good | High |
| **Trade Republic** | German Stocks | €1 | €0 | Limited | Medium |

## Alpaca Trading Integration

### Overview

Alpaca offers commission-free trading for US stocks and cryptocurrencies with excellent API support. Perfect for beginners and experienced traders.

### Account Setup

1. **Create Alpaca Account:**
   - Visit [alpaca.markets](https://alpaca.markets)
   - Complete account registration
   - Verify identity (KYC process)
   - Fund account (minimum $0 for cash account)

2. **Enable Paper Trading:**
   - Automatically created with account
   - $100,000 virtual money
   - Real market data
   - Same API as live trading

### API Key Generation

1. **Log into Dashboard:**
   - Go to [dashboard.alpaca.markets](https://dashboard.alpaca.markets)
   - Navigate to "API Keys" section

2. **Create API Key:**
   - Click "Generate API Key"
   - Copy API Key ID and Secret Key
   - **Important:** Save Secret Key immediately (won't be shown again)

3. **API Permissions:**
   ```
   ✅ Trading (Required)
   ✅ Data (Required)
   ✅ Account (Required)
   ❌ Transfer (Optional)
   ❌ Clearing (Not recommended)
   ```

### Configuration

#### Basic Configuration

```json
{
  "brokers": {
    "alpaca": {
      "enabled": true,
      "api_key": "YOUR_ACTUAL_ALPACA_API_KEY",
      "secret_key": "YOUR_ACTUAL_ALPACA_SECRET_KEY",
      "paper": true,
      "base_url": "https://paper-api.alpaca.markets"
    }
  }
}
```

#### Advanced Configuration

```json
{
  "brokers": {
    "alpaca": {
      "enabled": true,
      "api_key": "YOUR_API_KEY",
      "secret_key": "YOUR_SECRET_KEY",
      "paper": true,
      "base_url": "https://paper-api.alpaca.markets",
      
      "rate_limits": {
        "requests_per_minute": 200,
        "burst_limit": 50,
        "cool_down_seconds": 1
      },
      
      "trading_settings": {
        "default_order_type": "market",
        "default_time_in_force": "day",
        "fractional_shares": true,
        "min_order_size": 1.0,
        "max_order_size": 1000000.0
      },
      
      "market_data": {
        "real_time_streaming": true,
        "subscription": {
          "stocks": true,
          "crypto": true,
          "forex": false
        },
        "data_feed": "iex"
      },
      
      "order_types": {
        "market": {"enabled": true},
        "limit": {"enabled": true},
        "stop": {"enabled": true},
        "stop_limit": {"enabled": true}
      },
      
      "features": {
        "short_selling": true,
        "margin_trading": false,
        "after_hours_trading": false,
        "pre_market_trading": false
      }
    }
  }
}
```

### Testing Alpaca Integration

```bash
# Test connection
> test alpaca connection

# Get account info
> alpaca account

# Get positions
> alpaca positions

# Test order placement
> alpaca order buy AAPL 1 market --paper
```

### Supported Markets

#### US Stocks
- **Exchanges:** NYSE, NASDAQ, AMEX
- **Tickers:** All listed US stocks
- **Trading Hours:** 9:30 AM - 4:00 PM ET
- **Extended Hours:** Pre-market (4:00 AM - 9:30 AM), After-hours (4:00 PM - 8:00 PM)

#### Cryptocurrencies
- **Major Pairs:** BTC, ETH, LTC, BCH, USDT
- **Trading Hours:** 24/7
- **Settlement:** Instant

### Best Practices for Alpaca

1. **Use Paper Trading:** Always test strategies in paper mode first
2. **Fractional Shares:** Take advantage of fractional share trading for better diversification
3. **Extended Hours:** Be aware of extended trading sessions have different liquidity
4. **Market Orders:** Use limit orders for better price control
5. **Real-time Data:** Consider upgrading to premium data feeds

## Binance Integration

### Overview

Binance is the world's largest cryptocurrency exchange with advanced trading features and excellent API support.

### Account Setup

1. **Create Binance Account:**
   - Visit [binance.com](https://binance.com)
   - Complete registration
   - Enable 2FA (mandatory)
   - Verify identity (varies by region)

2. **Security Best Practices:**
   - Enable email verification
   - Use strong, unique password
   - Enable withdrawal whitelist
   - Use API restrictions

### API Key Setup

1. **Access API Management:**
   - Log into Binance account
   - Go to Account > API Management
   - Click "Create API"

2. **Configure API Key:**
   ```
   API Name: TradingOrchestrator
   Permissions:
   ✅ Read Info (Required)
   ✅ Enable Spot & Margin Trading (for spot trading)
   ✅ Enable Futures (for futures trading)
   ❌ Enable Withdrawals (Not recommended)
   ```

3. **IP Restrictions:**
   - Add your IP address for enhanced security
   - Use VPS IP if trading from cloud

### Testnet Setup (Recommended)

Binance provides a comprehensive testnet for development:

1. **Testnet Account:**
   - Visit [testnet.binance.vision](https://testnet.binance.vision)
   - Create test account
   - Get testnet API keys

2. **Testnet Benefits:**
   - Identical API to mainnet
   - Virtual balance (10 BTC, 100 ETH)
   - Real-time market data
   - No real funds at risk

### Configuration

#### Basic Configuration

```json
{
  "brokers": {
    "binance": {
      "enabled": true,
      "api_key": "YOUR_BINANCE_API_KEY",
      "secret_key": "YOUR_BINANCE_SECRET_KEY",
      "testnet": true,
      "base_url": "https://testnet.binance.vision"
    }
  }
}
```

#### Advanced Configuration

```json
{
  "brokers": {
    "binance": {
      "enabled": true,
      "api_key": "YOUR_API_KEY",
      "secret_key": "YOUR_SECRET_KEY",
      "testnet": true,
      
      "endpoints": {
        "base_url": "https://testnet.binance.vision",
        "websocket": "wss://testnet.binance.vision/ws"
      },
      
      "rate_limits": {
        "requests_per_minute": 1200,
        "orders_per_second": 10,
        "weight_per_minute": 6000
      },
      
      "trading_settings": {
        "default_order_type": "limit",
        "default_time_in_force": "GTC",
        "min_notional": 10.0,
        "max_notional": 900000.0
      },
      
      "symbols": {
        "spot_trading": {
          "enabled": true,
          "base_assets": ["BTC", "ETH", "BNB", "ADA", "DOT"],
          "quote_assets": ["USDT", "BUSD", "USDC"]
        }
      },
      
      "order_types": {
        "limit": {"enabled": true},
        "market": {"enabled": true},
        "stop_loss": {"enabled": true},
        "take_profit": {"enabled": true},
        "oco": {"enabled": true}
      },
      
      "market_data": {
        "streaming": {
          "enabled": true,
          "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
          "intervals": ["1m", "5m", "15m", "1h"]
        }
      }
    }
  }
}
```

### Testing Binance Integration

```bash
# Test connection
> test binance connection

# Get account info
> binance account

# Get balance
> binance balance

# Test order placement
> binance order buy BTCUSDT 0.01 limit 45000 --testnet
```

### Binance-Specific Features

#### Spot Trading
- **Market Pairs:** 300+ trading pairs
- **Order Types:** Market, Limit, Stop Loss, Take Profit, OCO
- **Trading Fees:** 0.1% (maker) / 0.1% (taker)

#### Futures Trading
- **Contracts:** USDT-M and COIN-M futures
- **Leverage:** Up to 125x (varies by pair)
- **Risk Management:** Position and account liquidation

#### Advanced Order Types
- **OCO (One-Cancels-Other):** Conditional orders
- **Post Only:** Maker-only orders
- **Time in Force:** GTC, IOC, FOK
- **Iceberg Orders:** Large orders broken into smaller pieces

### Security Best Practices

1. **API Restrictions:**
   - Disable withdrawals for trading bots
   - Set IP whitelist restrictions
   - Use separate API keys for different purposes

2. **Rate Limiting:**
   - Respect Binance rate limits
   - Implement exponential backoff
   - Monitor request weights

3. **Market Hours:**
   - Crypto markets are 24/7
   - Plan for volatility spikes
   - Use stop-losses extensively

## Interactive Brokers Integration

### Overview

Interactive Brokers provides access to global markets with professional-grade tools and competitive pricing.

### Account Setup

1. **Create IBKR Account:**
   - Visit [interactivebrokers.com](https://interactivebrokers.com)
   - Choose account type (Individual, Advisor, Corporate)
   - Complete application process
   - Minimum funding varies by account type

2. **Download TWS:**
   - Download Trader Workstation (TWS)
   - Install and configure
   - Enable API access in TWS settings

### TWS Configuration

1. **API Settings:**
   - Open TWS
   - Go to Edit > Global Configuration
   - Navigate to API > Settings
   - Configure settings:

```yaml
API Settings:
  Enable ActiveX and Socket Clients: ✅
  Socket Port: 7497
  Bond Retail Order Entry: ✅
  Allow connections from local host only: ✅
  Create API message log file: ✅
```

2. **Account Settings:**
   - Ensure paper trading account is active
   - Note account ID for configuration
   - Configure base currency

### Configuration

```json
{
  "brokers": {
    "ibkr": {
      "enabled": true,
      "host": "127.0.0.1",
      "port": 7497,
      "client_id": 1,
      "paper": true,
      "account_id": "",
      "base_currency": "USD"
    }
  }
}
```

#### Advanced Configuration

```json
{
  "brokers": {
    "ibkr": {
      "enabled": true,
      "connection": {
        "host": "127.0.0.1",
        "port": 7497,
        "client_id": 1,
        "timeout": 60,
        "retry_attempts": 3
      },
      
      "order_settings": {
        "default_order_type": "LMT",
        "default_time_in_force": "DAY",
        "default_lmqty": 100,
        "transmit": true
      },
      
      "supported_markets": {
        "stocks": {
          "enabled": true,
          "exchanges": ["SMART", "ARCA", "BATS", "NYSE", "NASDAQ"]
        },
        "options": {"enabled": true},
        "futures": {"enabled": true},
        "forex": {"enabled": true}
      },
      
      "order_types": {
        "market": {"enabled": true},
        "limit": {"enabled": true},
        "stop": {"enabled": true},
        "stop_limit": {"enabled": true},
        "bracket": {"enabled": true}
      }
    }
  }
}
```

### Testing IBKR Integration

```bash
# Check TWS connection
> test ibkr connection

# Get managed accounts
> ibkr accounts

# Test order placement
> ibkr order buy AAPL 1 market --testnet
```

### IBKR-Specific Features

#### Order Types
- **Market:** Immediate execution
- **Limit:** Specified price or better
- **Stop:** Becomes market when triggered
- **Stop Limit:** Becomes limit when triggered
- **Bracket:** Parent order with stop and limit children
- **Trailing:** Dynamic stop based on price movement

#### Market Data
- **Real-time:** Market data subscriptions required
- **Historical:** 1-minute to daily bars available
- **Tick Data:** Level 1 and Level 2 data
- **Options:** Greeks and theoretical values

## Trading 212 Integration

### Overview

Trading 212 offers commission-free trading for European stocks with a user-friendly platform.

### Account Setup

1. **Create Account:**
   - Visit [trading212.com](https://trading212.com)
   - Complete registration
   - Verify identity
   - Fund account

2. **API Access:**
   - Contact customer support for API access
   - Provide business use case
   - Complete additional verification

### Configuration

```json
{
  "brokers": {
    "trading212": {
      "enabled": true,
      "api_key": "YOUR_TRADING212_API_KEY",
      "practice": true,
      "base_url": "https://practice.trading212.com",
      "rate_limit": 120
    }
  }
}
```

### Supported Markets
- **UK Stocks:** LSE listed securities
- **EU Stocks:** Major European exchanges
- **ETFs:** Diversification through ETFs
- **Indices:** Track major market indices

## DEGIRO Integration (Unofficial)

### Overview

DEGIRO provides access to European markets but doesn't offer official API support. Integration is through unofficial methods.

### ⚠️ Warning
This integration uses unofficial methods and may violate DEGIRO's Terms of Service. Use at your own risk.

### Account Setup

1. **Create DEGIRO Account:**
   - Visit [degiro.nl](https://degiro.nl)
   - Complete registration
   - Verify identity
   - Fund account

2. **Security Setup:**
   - Enable 2FA
   - Use strong password
   - Monitor account regularly

### Configuration

```json
{
  "brokers": {
    "degiro": {
      "enabled": false,
      "username": "YOUR_DEGIRO_USERNAME",
      "password": "YOUR_DEGIRO_PASSWORD",
      "mfa": false,
      "session": null,
      "use_proxy": false,
      "proxy_url": ""
    }
  }
}
```

### Risk Considerations

1. **Account Risk:** Unofficial access could result in account suspension
2. **Compliance:** May violate regulatory requirements
3. **Security:** Credentials stored locally
4. **Support:** No official support available

## XTB Integration

### Overview

XTB specializes in Forex and CFDs with competitive spreads and advanced trading tools.

### Account Setup

1. **Create XTB Account:**
   - Visit [xtb.com](https://xtb.com)
   - Choose account type (Standard, Pro)
   - Complete verification
   - Fund account

2. **Demo Account:**
   - Practice trading with virtual money
   - Real market data
   - Same interface as live account

### Configuration

```json
{
  "brokers": {
    "xtb": {
      "enabled": true,
      "api_key": "YOUR_XTB_API_KEY",
      "secret_key": "YOUR_XTB_SECRET_KEY",
      "demo": true,
      "server": "demo",
      "max_instruments": 1000
    }
  }
}
```

### Trading Features
- **Forex:** 50+ currency pairs
- **CFDs:** Indices, commodities, cryptocurrencies
- **Spreads:** Competitive spreads starting from 0.1 pips
- **Leverage:** Up to 500:1 (regulatory dependent)

## Trade Republic Integration (Unofficial)

### Overview

Trade Republic offers commission-free German stocks but uses unofficial API access methods.

### ⚠️ Risk Warning
This integration is unofficial and carries significant risks.

### Configuration

```json
{
  "brokers": {
    "trade_republic": {
      "enabled": false,
      "username": "YOUR_TR_USERNAME",
      "password": "YOUR_TR_PASSWORD",
      "pin": "YOUR_TR_PIN",
      "device_id": null,
      "secure_session": true
    }
  }
}
```

## Multi-Broker Configuration

### Setting Up Multiple Brokers

```json
{
  "brokers": {
    "alpaca": {
      "enabled": true,
      "api_key": "YOUR_ALPACA_API_KEY",
      "secret_key": "YOUR_ALPACA_SECRET_KEY",
      "paper": true,
      "markets": ["US_STOCKS", "CRYPTO"]
    },
    
    "binance": {
      "enabled": true,
      "api_key": "YOUR_BINANCE_API_KEY",
      "secret_key": "YOUR_BINANCE_SECRET_KEY",
      "testnet": true,
      "markets": ["CRYPTO"]
    },
    
    "ibkr": {
      "enabled": true,
      "host": "127.0.0.1",
      "port": 7497,
      "paper": true,
      "markets": ["GLOBAL_STOCKS", "OPTIONS", "FUTURES"]
    }
  },
  
  "broker_routing": {
    "default_broker": "alpaca",
    "crypto_broker": "binance",
    "forex_broker": "ibkr",
    "fallback_broker": "alpaca"
  }
}
```

### Broker Selection Strategy

```json
{
  "broker_selection": {
    "alphabetical": {
      "alpaca": {
        "markets": ["US_STOCKS", "CRYPTO"],
        "priority": 1,
        "commission_free": true
      },
      "binance": {
        "markets": ["CRYPTO"],
        "priority": 2,
        "liquidity": "high"
      },
      "ibkr": {
        "markets": ["GLOBAL_STOCKS", "OPTIONS"],
        "priority": 3,
        "institutional_features": true
      }
    },
    
    "by_market": {
      "US_STOCKS": "alpaca",
      "CRYPTO": "binance",
      "OPTIONS": "ibkr",
      "FUTURES": "ibkr",
      "FOREX": "ibkr"
    },
    
    "by_cost": {
      "commission_free": ["alpaca", "trading212"],
      "low_cost": ["ibkr", "xtb"],
      "avoid_high_cost": ["degiro"]
    }
  }
}
```

## Broker Selection Guide

### For Beginners

**Recommended:** Alpaca + Binance (Testnet)

**Reasons:**
- No commissions (Alpaca)
- Easy to use APIs
- Good documentation
- Paper trading available
- Educational resources

### For Crypto Traders

**Recommended:** Binance + Alpaca

**Reasons:**
- Largest crypto exchange
- Advanced order types
- Deep liquidity
- US stock access (Alpaca)

### For Global Markets

**Recommended:** Interactive Brokers

**Reasons:**
- Global market access
- Professional tools
- Competitive pricing
- Advanced order types

### For European Traders

**Recommended:** Trading 212 + Interactive Brokers

**Reasons:**
- Commission-free EU stocks (Trading 212)
- Professional features (IBKR)
- Currency hedging options

### For Options Traders

**Recommended:** Interactive Brokers

**Reasons:**
- Comprehensive options platform
- Complex strategies supported
- Greeks calculation
- Options chains

## Best Practices

### Security

1. **API Key Management:**
   - Store in environment variables
   - Never commit to version control
   - Regular key rotation
   - Least privilege principle

2. **Environment Isolation:**
   - Separate paper/live credentials
   - Use different accounts
   - Regular security audits

### Error Handling

1. **Connection Management:**
   - Implement retry logic
   - Exponential backoff
   - Circuit breaker patterns
   - Graceful degradation

2. **Rate Limiting:**
   - Respect broker rate limits
   - Implement rate limiters
   - Monitor usage patterns
   - Queue management

### Testing

1. **Paper Trading:**
   - Always test in paper mode first
   - Validate order execution
   - Check position management
   - Verify risk controls

2. **Integration Testing:**
   - Test each broker separately
   - Verify data accuracy
   - Check latency
   - Monitor resource usage

## Troubleshooting

### Common Issues

#### Connection Timeouts

```bash
# Check broker status
> status brokers

# Test connection
> test broker alpaca

# Restart connection
> restart broker alpaca

# Check logs
> logs broker alpaca --level error
```

#### Authentication Errors

```bash
# Verify API keys
> validate config

# Test API key
> test alpaca api_key

# Check permissions
> alpaca permissions
```

#### Order Rejection

```bash
# Check order status
> order status --id ORDER_ID

# View rejection reason
> alpaca order history --rejected

# Check account limits
> alpaca limits
```

### Debug Mode

```python
# Enable debug logging
import logging
logging.getLogger('trading_orchestrator.brokers').setLevel(logging.DEBUG)

# Run with debug
python main.py --debug

# View detailed logs
tail -f logs/trading_orchestrator.log | grep DEBUG
```

### Getting Support

1. **Documentation:** Check broker-specific docs
2. **GitHub Issues:** Report bugs and problems
3. **Community:** Ask questions in Discord
4. **Broker Support:** Contact broker directly for API issues

## Next Steps

After broker integration:

1. **Configure Risk Management** - Set position limits and circuit breakers
2. **Test Strategies** - Paper trade with multiple brokers
3. **Monitor Performance** - Track execution quality and costs
4. **Optimize Routing** - Improve order execution with smart routing

Remember: Always start with paper trading and thoroughly test your integration before going live!