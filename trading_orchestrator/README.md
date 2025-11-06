# Day Trading Orchestrator System

**Production-Grade Multi-Broker Trading Platform with AI/LLM Integration**

A comprehensive day trading orchestration system featuring multiple broker integrations, real-time market data, AI-powered trading strategies, risk management, and a Matrix-themed terminal interface.

## Features

### Core Capabilities
- **Multi-Broker Support**: Unified interface for multiple trading platforms
- **Real-Time Data**: WebSocket streaming for live quotes and market data
- **AI/LLM Orchestration**: Integrated AI models for trading strategy generation
- **Risk Management**: Circuit breakers, position limits, compliance monitoring
- **Order Management**: Advanced order types, execution tracking, audit logging
- **Terminal UI**: Matrix-style real-time dashboard interface

### Database Architecture
- **Embedded SQLite** with PostgreSQL upgrade path
- Comprehensive schema for users, brokers, trading data, risk, and AI systems
- Full audit logging for compliance and troubleshooting

### Broker API Integration Status

#### ✅ Officially Supported (Implemented/Planned)

1. **Binance** - IMPLEMENTED
   - Official REST + WebSocket API
   - Testnet support for paper trading
   - Comprehensive market data and order types
   - Rate limit compliance built-in

2. **Interactive Brokers (IBKR)** - PLANNED
   - TWS API (socket-based)
   - Professional-grade trading platform
   - Broad market access and asset classes

3. **Alpaca** - PLANNED
   - Official REST + WebSocket API
   - US equities and crypto
   - Commission-free trading with paper mode

4. **Trading 212** - PLANNED (with warnings)
   - Official Public API (beta)
   - Currently limited to market orders in live mode
   - Full order types available in practice mode

#### ⚠️ Not Supported - Important Notices

5. **XTB (xAPI)** - DISCONTINUED
   - **Status**: API service disabled as of March 14, 2025
   - **Impact**: All historical endpoints and WebSocket streams are non-functional
   - **Migration Required**: Users must switch to alternative broker APIs
   - **See Documentation**: `docs/BROKER_WARNINGS.md` for migration guidance

6. **DEGIRO** - NO OFFICIAL API
   - **Status**: No official public API available
   - **Risk**: Only unofficial reverse-engineered clients exist
   - **Legal Exposure**: Use violates Terms and Conditions
   - **Recommendation**: DO NOT USE for production trading
   - **See Documentation**: `research/degiro/degiro_unofficial_api_analysis.md`

7. **Trade Republic** - NO OFFICIAL API
   - **Status**: No official public API available
   - **Risk**: Unofficial clients violate Customer Agreement
   - **Contract Risk**: "Extraordinary termination" clause for unauthorized access
   - **Recommendation**: DO NOT USE for production trading
   - **See Documentation**: `research/trade_republic/trade_republic_unofficial_api_analysis.md`

## Installation

### Prerequisites
- Python 3.11 or higher
- pip or uv package manager
- Terminal with 256 color support (for UI)

### Setup

1. **Clone Repository**
```bash
cd /workspace/trading_orchestrator
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
# OR using uv (faster)
uv pip install -r requirements.txt
```

3. **Validate System** (Recommended)
```bash
# Run comprehensive system validation
python validate_system.py

# This checks:
# - Environment and dependencies
# - Configuration and database setup
# - AI components and broker connections
# - UI components and system integration
```

4. **Configure Environment**

Create `.env` file in the project root:

```bash
# Application
ENVIRONMENT=development
DEBUG=false
LOG_LEVEL=INFO

# Database (SQLite by default)
DB_TYPE=sqlite
DB_PATH=./trading_orchestrator.db

# Security
SECRET_KEY=your-secret-key-change-in-production
ENCRYPTION_KEY=your-encryption-key-32-chars

# Binance API (Testnet)
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret
BINANCE_TESTNET=true

# IBKR (when implemented)
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=1

# Alpaca (when implemented)
ALPACA_API_KEY=your_alpaca_key
ALPACA_API_SECRET=your_alpaca_secret
ALPACA_PAPER=true

# Trading 212 (when implemented)
TRADING212_API_KEY=your_trading212_key
TRADING212_PRACTICE=true

# AI/LLM (optional)
AI_PROVIDER=openai
OPENAI_API_KEY=your_openai_key
```

4. **Initialize Database**
```bash
python main.py
# Database will be automatically initialized on first run
```

## Usage

### Start the System

```bash
# Recommended: Use the startup script with validation
python start_system.py

# Or run main directly
python main.py

# Validation only
python start_system.py --validate-only

# Skip validation for faster startup
python start_system.py --skip-validation
```

The terminal interface will launch with real-time dashboard showing:
- Account balances and equity
- Open positions with P&L
- Active orders and order history
- Risk metrics and compliance status
- System health and broker connections
- AI-powered market analysis and insights

### Keyboard Shortcuts

- **Ctrl+C**: Graceful shutdown
- **Arrow Keys**: Navigate (when interactive mode enabled)
- **Tab**: Switch panels (when interactive mode enabled)

## Architecture

```
trading_orchestrator/
├── main.py                    # Application entry point (complete integration)
├── start_system.py            # System launcher with validation
├── validate_system.py         # System validation and diagnostics
├── config/
│   ├── settings.py            # Configuration management
│   ├── database.py            # Database setup
│   └── application.py         # System lifecycle management
├── database/
│   ├── models/                # SQLAlchemy ORM models
│   │   ├── user.py           # User & authentication
│   │   ├── broker.py         # Broker connections
│   │   ├── trading.py        # Positions, orders, trades
│   │   ├── risk.py           # Risk management
│   │   └── ai.py             # AI/LLM system
│   └── migrations/            # Database migrations
├── brokers/
│   ├── base.py                # Abstract broker interface
│   ├── factory.py             # Broker instantiation
│   ├── binance_broker.py      # Binance implementation
│   ├── alpaca_broker.py       # Alpaca implementation
│   ├── ibkr_broker.py         # IBKR implementation
│   └── trading212_broker.py   # Trading 212 implementation
├── ai/
│   ├── orchestrator.py        # Main AI trading orchestrator
│   ├── models/                # AI model management
│   │   └── ai_models_manager.py  # Multi-tier LLM support
│   ├── tools/                 # Trading tools for LLM
│   │   └── trading_tools.py   # Market analysis & risk tools
│   ├── prompts/               # Prompt templates
│   └── schemas/               # JSON schemas
├── strategies/                # Trading strategies
├── ui/
│   ├── terminal.py            # Terminal interface with real-time updates
│   └── components/
│       └── dashboard.py       # Dashboard manager with live data
├── risk/
│   └── manager.py             # Comprehensive risk management
├── oms/
│   └── manager.py             # Order management system
├── tests/
│   └── test_integration.py    # Complete system integration tests
└── docs/                      # Documentation
```

## Security

### Credential Management
- All API keys are encrypted in database using AES-256
- Environment variables for development
- Production: Use secrets management (Vault, AWS Secrets Manager, etc.)

### Authentication
- JWT-based API authentication
- Session management with expiration
- Audit logging for all sensitive operations

### Best Practices
- Never commit `.env` files to version control
- Rotate API keys regularly
- Use paper/testnet mode for development
- Enable 2FA on broker accounts

## Risk Management

### Built-in Protections
- **Position Size Limits**: Maximum position size per symbol
- **Daily Loss Limits**: Circuit breakers for daily P&L thresholds
- **Order Limits**: Maximum open orders per account
- **Compliance Rules**: Regulatory and internal rule enforcement
- **Emergency Halt**: Kill switch for immediate trading suspension

### Configuration
Edit risk parameters in `.env`:
```bash
MAX_POSITION_SIZE=10000.0
MAX_DAILY_LOSS=1000.0
MAX_OPEN_ORDERS=50
RISK_PER_TRADE=0.02
```

## AI/LLM Integration

### Supported Models
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude-3)
- Local models via compatible APIs

### Trading Tools
- `get_features()`: Real-time market data features
- `backtest()`: Strategy backtesting engine
- `rag()`: Knowledge base retrieval
- `news_sentiment()`: News analysis
- `risk_limits()`: Risk management checks

### Strategy Types
- Trend following
- Mean reversion
- Pairs trading
- Cross-venue arbitrage

## Testing

### Run Tests
```bash
# Run integration tests (comprehensive system validation)
python -m pytest tests/test_integration.py -v

# Run with coverage
python -m pytest tests/test_integration.py --cov=trading_orchestrator --cov-report=html

# Run specific integration test
python -m pytest tests/test_integration.py::TestSystemIntegration::test_complete_system_startup -v

# Traditional unit tests
pytest tests/
pytest tests/ --cov=trading_orchestrator
```

### Integration Test Coverage

The integration tests validate the complete system:

- **System Startup**: Full initialization and component integration
- **AI Orchestration**: Market analysis and trading decisions
- **Risk Management**: Pre-trade validation and monitoring
- **Order Management**: Multi-broker routing and execution
- **Real-time UI**: Dashboard updates with live data
- **Health Monitoring**: System status and performance metrics
- **End-to-End Workflow**: Complete trading scenarios

### Paper Trading
All supported brokers provide paper/testnet modes for safe testing:
- Binance: Set `BINANCE_TESTNET=true`
- IBKR: Connect to paper trading port (7497)
- Alpaca: Set `ALPACA_PAPER=true`
- Trading 212: Use practice environment

## Monitoring & Logging

### Logs
- Location: `./logs/trading_orchestrator.log`
- Rotation: Daily
- Retention: 30 days
- Levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

### Metrics (Prometheus)
- Port: 9090 (configurable)
- Endpoints: `/metrics`
- Dashboards: Grafana templates included

## Production Deployment

### PostgreSQL Migration
1. Install PostgreSQL
2. Create database
3. Update `.env`:
```bash
DB_TYPE=postgresql
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trading_orchestrator
DB_USER=your_user
DB_PASSWORD=your_password
```

### Systemd Service
See `docs/DEPLOYMENT.md` for systemd service configuration

### Docker
See `docs/DOCKER.md` for containerization guide

## Troubleshooting

### Common Issues

**Problem**: Broker connection fails
- **Solution**: Verify API keys in `.env`
- **Solution**: Check paper/live mode configuration
- **Solution**: Verify network connectivity

**Problem**: Database errors
- **Solution**: Delete `trading_orchestrator.db` and restart (dev only)
- **Solution**: Check database permissions
- **Solution**: Review migration logs

**Problem**: Terminal UI not rendering
- **Solution**: Verify terminal supports 256 colors
- **Solution**: Update Rich library: `pip install --upgrade rich`

## Contributing

See `docs/CONTRIBUTING.md` for contribution guidelines

## License

This project is provided as-is for educational and development purposes.

**IMPORTANT DISCLAIMERS**:
- This software is for informational purposes only
- Not financial advice
- Trading involves substantial risk of loss
- Test thoroughly in paper mode before live trading
- Verify all broker API terms and compliance requirements

## Support & Resources

### Documentation
- Full documentation: `docs/`
- Broker API research: `research/`
- Architecture diagrams: `docs/ARCHITECTURE.md`

### External Resources
- Binance API Docs: https://developers.binance.com/
- IBKR API Docs: https://www.interactivebrokers.com/api/
- Alpaca API Docs: https://docs.alpaca.markets/

## Acknowledgments

- Binance for comprehensive official API
- Interactive Brokers for professional TWS API
- Alpaca for developer-friendly API
- Rich library for terminal interface
- FastAPI for high-performance API framework

---

**Version**: 1.0.0  
**Last Updated**: 2025-11-05  
**Status**: Production-Ready (with noted broker limitations)