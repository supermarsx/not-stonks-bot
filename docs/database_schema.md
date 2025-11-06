# Database Schema Documentation

## Overview

The Day Trading Orchestrator uses a comprehensive database schema to manage trading data, user information, strategy configurations, and system state. This documentation covers the complete database design, relationships, and optimization strategies.

## Table of Contents

1. [Database Architecture](#database-architecture)
2. [Core Entities](#core-entities)
3. [User Management](#user-management)
4. [Trading Data](#trading-data)
5. [Strategy Management](#strategy-management)
6. [Risk Management](#risk-management)
7. [System Monitoring](#system-monitoring)
8. [Performance Optimization](#performance-optimization)
9. [Data Retention](#data-retention)
10. [Migration Strategy](#migration-strategy)

## Database Architecture

### Supported Database Systems

The platform supports multiple database systems with appropriate optimizations:

#### Primary Database (PostgreSQL)
- **Primary Use**: Transactional data, user management, configuration
- **Advantages**: ACID compliance, rich data types, excellent performance
- **Use Cases**: User accounts, orders, positions, strategies

#### Time-Series Database (InfluxDB)
- **Primary Use**: Market data, performance metrics, system monitoring
- **Advantages**: Optimized for time-series data, high write throughput
- **Use Cases**: Price data, trading signals, system metrics

#### Cache Database (Redis)
- **Primary Use**: Session management, real-time data caching
- **Advantages**: In-memory storage, pub/sub capabilities
- **Use Cases**: User sessions, rate limiting, real-time quotes

#### Document Database (MongoDB)
- **Primary Use**: Flexible schema data, logs, analytics
- **Advantages**: Schema flexibility, document storage
- **Use Cases**: Strategy configurations, audit logs, reports

### Database Selection Matrix

| Data Type | Primary DB | Secondary DB | Rationale |
|-----------|------------|--------------|-----------|
| User Data | PostgreSQL | - | ACID requirements, complex relationships |
| Orders | PostgreSQL | Redis (cache) | Transaction integrity, real-time updates |
| Market Data | InfluxDB | PostgreSQL (archive) | Time-series optimization |
| Strategies | PostgreSQL | MongoDB (configs) | Structured data, flexible configs |
| Logs | MongoDB | - | Document flexibility, query performance |
| Metrics | InfluxDB | PostgreSQL (aggregation) | Time-series analysis |

## Core Entities

### 1. Users and Authentication

```sql
-- Users table
CREATE TABLE users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    phone VARCHAR(20),
    date_of_birth DATE,
    country_code VARCHAR(3),
    timezone VARCHAR(50) DEFAULT 'UTC',
    
    -- Account status
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false,
    verification_token VARCHAR(255),
    verification_expires TIMESTAMP,
    
    -- Account settings
    preferred_currency VARCHAR(3) DEFAULT 'USD',
    risk_tolerance VARCHAR(20) DEFAULT 'MODERATE', -- CONSERVATIVE, MODERATE, AGGRESSIVE
    max_daily_loss DECIMAL(15,2) DEFAULT 1000.00,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    deleted_at TIMESTAMP
);

-- User sessions
CREATE TABLE user_sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    refresh_token VARCHAR(255) UNIQUE,
    ip_address INET,
    user_agent TEXT,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_user_sessions_user_id (user_id),
    INDEX idx_user_sessions_token (session_token),
    INDEX idx_user_sessions_expires (expires_at)
);

-- API keys for programmatic access
CREATE TABLE user_api_keys (
    api_key_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    key_name VARCHAR(100) NOT NULL,
    api_key_hash VARCHAR(255) NOT NULL, -- Hashed API key
    api_key_prefix VARCHAR(20) NOT NULL, -- First few chars for identification
    
    -- Permissions
    can_read BOOLEAN DEFAULT true,
    can_write BOOLEAN DEFAULT false,
    can_trade BOOLEAN DEFAULT false,
    rate_limit_per_hour INTEGER DEFAULT 1000,
    
    -- Restrictions
    allowed_ips INET[],
    allowed_endpoints TEXT[],
    
    -- Status
    is_active BOOLEAN DEFAULT true,
    last_used_at TIMESTAMP,
    expires_at TIMESTAMP,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(user_id, key_name),
    INDEX idx_api_keys_user (user_id),
    INDEX idx_api_keys_hash (api_key_hash)
);
```

### 2. Broker Accounts

```sql
-- Broker configuration
CREATE TABLE brokers (
    broker_id VARCHAR(50) PRIMARY KEY,
    broker_name VARCHAR(100) NOT NULL,
    broker_type VARCHAR(20) NOT NULL, -- EQUITY, FOREX, CRYPTO, FUTURES
    base_url VARCHAR(255),
    documentation_url VARCHAR(255),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User broker accounts
CREATE TABLE user_broker_accounts (
    account_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    broker_id VARCHAR(50) NOT NULL REFERENCES brokers(broker_id),
    
    -- Account identification
    account_name VARCHAR(100),
    broker_account_id VARCHAR(100) NOT NULL, -- Account ID from broker
    
    -- Encrypted credentials (encrypted as a whole)
    encrypted_credentials BYTEA, -- Encrypted JSON with API keys, secrets
    
    -- Account metadata
    account_type VARCHAR(20) DEFAULT 'INDIVIDUAL', -- INDIVIDUAL, JOINT, CORPORATE
    currency VARCHAR(3) DEFAULT 'USD',
    base_currency VARCHAR(3) DEFAULT 'USD',
    
    -- Connection status
    is_connected BOOLEAN DEFAULT false,
    last_connected_at TIMESTAMP,
    connection_error TEXT,
    
    -- Limits and settings
    max_position_size DECIMAL(15,2),
    max_daily_volume DECIMAL(15,2),
    allowed_symbols TEXT[], -- NULL means all symbols allowed
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP,
    
    UNIQUE(user_id, broker_id, broker_account_id),
    INDEX idx_broker_accounts_user (user_id),
    INDEX idx_broker_accounts_broker (broker_id),
    INDEX idx_broker_accounts_connected (is_connected)
);

-- Account positions
CREATE TABLE account_positions (
    position_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id UUID NOT NULL REFERENCES user_broker_accounts(account_id) ON DELETE CASCADE,
    
    -- Position details
    symbol VARCHAR(20) NOT NULL,
    quantity DECIMAL(20,8) NOT NULL, -- Supports crypto precision
    average_price DECIMAL(15,8) NOT NULL,
    current_price DECIMAL(15,8),
    
    -- Position metadata
    position_type VARCHAR(10) NOT NULL, -- LONG, SHORT
    sector VARCHAR(50),
    asset_class VARCHAR(20), -- EQUITY, FOREX, CRYPTO, COMMODITY
    
    -- Timestamps
    opened_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    closed_at TIMESTAMP,
    
    INDEX idx_positions_account (account_id),
    INDEX idx_positions_symbol (symbol),
    INDEX idx_positions_open (opened_at),
    INDEX idx_positions_type (position_type)
);

-- Account balances
CREATE TABLE account_balances (
    balance_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id UUID NOT NULL REFERENCES user_broker_accounts(account_id) ON DELETE CASCADE,
    
    -- Balance details
    currency VARCHAR(3) NOT NULL,
    cash_balance DECIMAL(20,8) DEFAULT 0,
    buying_power DECIMAL(20,8) DEFAULT 0,
    portfolio_value DECIMAL(20,8) DEFAULT 0,
    
    -- Margin details
    margin_used DECIMAL(20,8) DEFAULT 0,
    margin_available DECIMAL(20,8) DEFAULT 0,
    
    -- Timestamps
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(account_id, currency, recorded_at),
    INDEX idx_balances_account (account_id),
    INDEX idx_balances_recorded (recorded_at)
);
```

## Trading Data

### 3. Orders and Executions

```sql
-- Orders table
CREATE TABLE orders (
    order_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(user_id),
    account_id UUID NOT NULL REFERENCES user_broker_accounts(account_id),
    
    -- Order identification
    client_order_id VARCHAR(100), -- Client-generated order ID
    broker_order_id VARCHAR(100), -- Broker-generated order ID
    strategy_id UUID, -- References strategy if order is strategy-generated
    
    -- Order details
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(4) NOT NULL, -- BUY, SELL
    order_type VARCHAR(10) NOT NULL, -- MARKET, LIMIT, STOP, STOP_LIMIT
    quantity DECIMAL(20,8) NOT NULL,
    filled_quantity DECIMAL(20,8) DEFAULT 0,
    
    -- Price information
    limit_price DECIMAL(15,8),
    stop_price DECIMAL(15,8),
    average_fill_price DECIMAL(15,8),
    
    -- Order constraints
    time_in_force VARCHAR(10) DEFAULT 'DAY', -- DAY, GTC, IOC, FOK
    good_till_date TIMESTAMP,
    
    -- Order status
    status VARCHAR(20) NOT NULL, -- NEW, PARTIALLY_FILLED, FILLED, CANCELLED, REJECTED, EXPIRED
    rejection_reason TEXT,
    
    -- Metadata
    order_source VARCHAR(20) DEFAULT 'MANUAL', -- MANUAL, STRATEGY, API
    commission DECIMAL(10,4) DEFAULT 0,
    
    -- Timestamps
    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    filled_at TIMESTAMP,
    cancelled_at TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_orders_user (user_id),
    INDEX idx_orders_account (account_id),
    INDEX idx_orders_symbol (symbol),
    INDEX idx_orders_status (status),
    INDEX idx_orders_submitted (submitted_at),
    INDEX idx_orders_strategy (strategy_id)
);

-- Order fills (executions)
CREATE TABLE order_fills (
    fill_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id UUID NOT NULL REFERENCES orders(order_id) ON DELETE CASCADE,
    
    -- Fill details
    fill_quantity DECIMAL(20,8) NOT NULL,
    fill_price DECIMAL(15,8) NOT NULL,
    commission DECIMAL(10,4) DEFAULT 0,
    
    -- Liquidity information
    liquidity_indicator VARCHAR(1), -- 'P' for passive, 'A' for aggressive
    
    -- Fill metadata
    trade_id VARCHAR(100), -- Exchange trade ID
    exchange_time TIMESTAMP, -- When the trade occurred on exchange
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_fills_order (order_id),
    INDEX idx_fills_trade_id (trade_id)
);
```

### 4. Market Data

```sql
-- Market symbols universe
CREATE TABLE market_symbols (
    symbol_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(20) UNIQUE NOT NULL,
    
    -- Symbol details
    company_name VARCHAR(255),
    exchange VARCHAR(50) NOT NULL,
    sector VARCHAR(50),
    industry VARCHAR(100),
    asset_class VARCHAR(20) NOT NULL, -- EQUITY, FOREX, CRYPTO, COMMODITY
    
    -- Trading details
    currency VARCHAR(3) NOT NULL,
    tick_size DECIMAL(10,8),
    contract_size DECIMAL(15,8), -- For derivatives
    is_active BOOLEAN DEFAULT true,
    
    -- Market hours
    trading_hours JSONB, -- Store trading hours as JSON
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_symbols_exchange (exchange),
    INDEX idx_symbols_asset_class (asset_class),
    INDEX idx_symbols_active (is_active)
);

-- Historical price data (stored in separate partitions by symbol/time)
CREATE TABLE market_data_1m (
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    
    -- OHLCV data
    open_price DECIMAL(15,8) NOT NULL,
    high_price DECIMAL(15,8) NOT NULL,
    low_price DECIMAL(15,8) NOT NULL,
    close_price DECIMAL(15,8) NOT NULL,
    volume BIGINT DEFAULT 0,
    
    -- Additional metrics
    vwap DECIMAL(15,8), -- Volume weighted average price
    bid DECIMAL(15,8),
    ask DECIMAL(15,8),
    bid_size INTEGER,
    ask_size INTEGER,
    
    PRIMARY KEY (symbol, timestamp),
    INDEX idx_market_data_1m_timestamp (timestamp),
    INDEX idx_market_data_1m_symbol_time (symbol, timestamp)
);

-- InfluxDB schema for high-frequency data
-- Schema designed for InfluxDB (line protocol format)
-- Measurement: prices
-- Tags: symbol, exchange
-- Fields: open, high, low, close, volume, bid, ask
-- Time: timestamp
```

### 5. Trading Signals

```sql
-- Trading strategies
CREATE TABLE strategies (
    strategy_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    
    -- Strategy identification
    strategy_name VARCHAR(100) NOT NULL,
    strategy_type VARCHAR(50) NOT NULL, -- MOMENTUM, MEAN_REVERSION, ARBITRAGE, etc.
    strategy_category VARCHAR(30), -- DAY_TRADING, SWING_TRADING, POSITION_TRADING
    
    -- Configuration
    parameters JSONB, -- Strategy-specific parameters
    risk_settings JSONB, -- Risk management settings
    
    -- Strategy metadata
    description TEXT,
    author VARCHAR(100),
    version VARCHAR(20) DEFAULT '1.0.0',
    
    -- Status
    is_active BOOLEAN DEFAULT false,
    is_public BOOLEAN DEFAULT false, -- For strategy marketplace
    
    -- Performance tracking
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    total_pnl DECIMAL(20,8) DEFAULT 0,
    max_drawdown DECIMAL(10,4) DEFAULT 0,
    sharpe_ratio DECIMAL(10,4),
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_run_at TIMESTAMP,
    
    INDEX idx_strategies_user (user_id),
    INDEX idx_strategies_type (strategy_type),
    INDEX idx_strategies_active (is_active),
    INDEX idx_strategies_public (is_public)
);

-- Trading signals generated by strategies
CREATE TABLE trading_signals (
    signal_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_id UUID NOT NULL REFERENCES strategies(strategy_id) ON DELETE CASCADE,
    
    -- Signal details
    symbol VARCHAR(20) NOT NULL,
    signal_type VARCHAR(10) NOT NULL, -- BUY, SELL, CLOSE
    signal_strength DECIMAL(5,4), -- 0.0 to 1.0 confidence
    
    -- Signal metadata
    signal_reason TEXT, -- Why this signal was generated
    technical_indicators JSONB, -- Snapshot of indicators at signal time
    
    -- Market context
    market_conditions JSONB, -- Market state at signal time
    volatility DECIMAL(10,6),
    volume_ratio DECIMAL(10,4),
    
    -- Execution details
    recommended_quantity DECIMAL(20,8),
    suggested_entry_price DECIMAL(15,8),
    stop_loss_price DECIMAL(15,8),
    take_profit_price DECIMAL(15,8),
    
    -- Signal status
    status VARCHAR(20) DEFAULT 'ACTIVE', -- ACTIVE, EXECUTED, EXPIRED, CANCELLED
    executed_at TIMESTAMP,
    expires_at TIMESTAMP,
    
    -- Performance tracking
    actual_entry_price DECIMAL(15,8),
    actual_exit_price DECIMAL(15,8),
    realized_pnl DECIMAL(20,8),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_signals_strategy (strategy_id),
    INDEX idx_signals_symbol (symbol),
    INDEX idx_signals_created (created_at),
    INDEX idx_signals_status (status)
);

-- Strategy executions (actual trades based on signals)
CREATE TABLE strategy_executions (
    execution_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    signal_id UUID NOT NULL REFERENCES trading_signals(signal_id),
    order_id UUID NOT NULL REFERENCES orders(order_id),
    
    -- Execution details
    execution_type VARCHAR(20) NOT NULL, -- ENTRY, EXIT, ADJUSTMENT
    execution_price DECIMAL(15,8) NOT NULL,
    execution_quantity DECIMAL(20,8) NOT NULL,
    
    -- Performance
    pnl DECIMAL(20,8),
    pnl_percentage DECIMAL(10,6),
    
    -- Execution metadata
    slippage DECIMAL(10,6), -- Price slippage from signal price
    latency_ms INTEGER, -- Time from signal to execution
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_executions_signal (signal_id),
    INDEX idx_executions_order (order_id),
    INDEX idx_executions_type (execution_type)
);
```

## Strategy Management

### 6. Strategy Configurations

```sql
-- Strategy templates (pre-built strategies)
CREATE TABLE strategy_templates (
    template_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Template details
    template_name VARCHAR(100) NOT NULL,
    template_type VARCHAR(50) NOT NULL,
    description TEXT,
    
    -- Default configuration
    default_parameters JSONB NOT NULL,
    default_risk_settings JSONB,
    
    -- Template metadata
    author VARCHAR(100),
    version VARCHAR(20) DEFAULT '1.0.0',
    category VARCHAR(30),
    tags TEXT[],
    
    -- Licensing
    is_premium BOOLEAN DEFAULT false,
    license_type VARCHAR(20) DEFAULT 'FREE', -- FREE, PREMIUM, SUBSCRIPTION
    
    -- Status
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false,
    verification_date TIMESTAMP,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_templates_type (template_type),
    INDEX idx_templates_active (is_active),
    INDEX idx_templates_premium (is_premium)
);

-- User strategy instances
CREATE TABLE user_strategies (
    user_strategy_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    template_id UUID REFERENCES strategy_templates(template_id),
    
    -- Strategy instance details
    strategy_name VARCHAR(100) NOT NULL,
    is_active BOOLEAN DEFAULT false,
    
    -- Configuration (user overrides)
    parameters JSONB,
    risk_settings JSONB,
    
    -- Schedule settings
    trading_schedule JSONB, -- When strategy should run
    
    -- Performance tracking
    total_trades INTEGER DEFAULT 0,
    total_pnl DECIMAL(20,8) DEFAULT 0,
    win_rate DECIMAL(5,4),
    max_drawdown DECIMAL(10,4) DEFAULT 0,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_run_at TIMESTAMP,
    
    INDEX idx_user_strategies_user (user_id),
    INDEX idx_user_strategies_active (is_active),
    INDEX idx_user_strategies_template (template_id)
);

-- Strategy backtests
CREATE TABLE strategy_backtests (
    backtest_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_id UUID NOT NULL REFERENCES strategies(strategy_id),
    
    -- Backtest configuration
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    initial_capital DECIMAL(20,8) NOT NULL,
    commission_rate DECIMAL(6,4) DEFAULT 0.001,
    
    -- Results
    final_value DECIMAL(20,8) NOT NULL,
    total_return DECIMAL(10,6) NOT NULL,
    annualized_return DECIMAL(10,6),
    volatility DECIMAL(10,6),
    sharpe_ratio DECIMAL(10,6),
    max_drawdown DECIMAL(10,6),
    win_rate DECIMAL(5,4),
    profit_factor DECIMAL(10,4),
    
    -- Trade statistics
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    avg_win DECIMAL(15,8),
    avg_loss DECIMAL(15,8),
    largest_win DECIMAL(20,8),
    largest_loss DECIMAL(20,8),
    
    -- Metadata
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_backtests_strategy (strategy_id),
    INDEX idx_backtests_period (start_date, end_date)
);
```

## Risk Management

### 7. Risk Monitoring

```sql
-- Risk profiles
CREATE TABLE risk_profiles (
    profile_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    
    -- Profile details
    profile_name VARCHAR(100) NOT NULL,
    description TEXT,
    
    -- Risk parameters
    max_position_size DECIMAL(15,8), -- Maximum position size in USD
    max_sector_concentration DECIMAL(5,4) DEFAULT 0.30, -- 30%
    max_correlation_exposure DECIMAL(5,4) DEFAULT 0.50, -- 50%
    max_daily_loss DECIMAL(15,8),
    max_portfolio_risk DECIMAL(5,4) DEFAULT 0.02, -- 2%
    
    -- Position limits
    max_long_positions INTEGER DEFAULT 20,
    max_short_positions INTEGER DEFAULT 10,
    min_cash_percentage DECIMAL(5,4) DEFAULT 0.05, -- 5%
    
    -- Stop loss settings
    default_stop_loss_percentage DECIMAL(5,4) DEFAULT 0.02, -- 2%
    trailing_stop_enabled BOOLEAN DEFAULT true,
    trailing_stop_percentage DECIMAL(5,4) DEFAULT 0.01, -- 1%
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_risk_profiles_user (user_id)
);

-- Risk calculations
CREATE TABLE risk_calculations (
    calculation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(user_id),
    account_id UUID REFERENCES user_broker_accounts(account_id),
    
    -- Portfolio snapshot
    portfolio_value DECIMAL(20,8) NOT NULL,
    cash_balance DECIMAL(20,8) NOT NULL,
    invested_amount DECIMAL(20,8) NOT NULL,
    
    -- Risk metrics
    portfolio_var_1d DECIMAL(20,8), -- 1-day Value at Risk
    portfolio_var_5d DECIMAL(20,8), -- 5-day Value at Risk
    expected_shortfall DECIMAL(20,8),
    portfolio_beta DECIMAL(10,6),
    
    -- Concentration metrics
    sector_concentrations JSONB, -- Sector allocation percentages
    symbol_concentrations JSONB, -- Top symbol concentrations
    correlation_risk DECIMAL(10,6),
    
    -- Exposure metrics
    long_exposure DECIMAL(20,8),
    short_exposure DECIMAL(20,8),
    net_exposure DECIMAL(20,8),
    
    -- Liquidity metrics
    illiquid_positions JSONB, -- Positions with low liquidity
    portfolio_turnover DECIMAL(10,6),
    
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_risk_calculations_user (user_id),
    INDEX idx_risk_calculations_account (account_id),
    INDEX idx_risk_calculations_time (calculated_at)
);

-- Risk alerts
CREATE TABLE risk_alerts (
    alert_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(user_id),
    
    -- Alert details
    alert_type VARCHAR(50) NOT NULL, -- POSITION_LIMIT, DRAWDOWN, CORRELATION
    severity VARCHAR(20) NOT NULL, -- LOW, MEDIUM, HIGH, CRITICAL
    title VARCHAR(255) NOT NULL,
    message TEXT NOT NULL,
    
    -- Alert context
    symbol VARCHAR(20),
    current_value DECIMAL(20,8),
    threshold_value DECIMAL(20,8),
    violation_percentage DECIMAL(10,6),
    
    -- Risk details
    risk_metrics JSONB, -- Snapshot of relevant risk metrics
    
    -- Alert status
    is_acknowledged BOOLEAN DEFAULT false,
    acknowledged_at TIMESTAMP,
    acknowledged_by UUID, -- References users.user_id if auto-acknowledged
    
    -- Resolution
    resolution_action TEXT,
    resolved_at TIMESTAMP,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_risk_alerts_user (user_id),
    INDEX idx_risk_alerts_type (alert_type),
    INDEX idx_risk_alerts_severity (severity),
    INDEX idx_risk_alerts_created (created_at),
    INDEX idx_risk_alerts_acknowledged (is_acknowledged)
);
```

## System Monitoring

### 8. Performance Metrics

```sql
-- System performance metrics
CREATE TABLE system_metrics (
    metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Metric identification
    metric_name VARCHAR(100) NOT NULL,
    metric_category VARCHAR(50) NOT NULL, -- SYSTEM, TRADING, RISK, API
    
    -- Metric values
    metric_value DECIMAL(20,8),
    metric_unit VARCHAR(20), -- ms, %, count, USD, etc.
    
    -- Tags for filtering
    service_name VARCHAR(50),
    endpoint VARCHAR(100),
    broker_id VARCHAR(50),
    
    -- Timestamp
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_metrics_name (metric_name),
    INDEX idx_metrics_category (metric_category),
    INDEX idx_metrics_service (service_name),
    INDEX idx_metrics_recorded (recorded_at)
);

-- API usage tracking
CREATE TABLE api_usage (
    usage_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(user_id),
    api_key_id UUID REFERENCES user_api_keys(api_key_id),
    
    -- API details
    endpoint VARCHAR(200) NOT NULL,
    http_method VARCHAR(10) NOT NULL, -- GET, POST, PUT, DELETE
    
    -- Usage metrics
    response_time_ms INTEGER,
    status_code INTEGER,
    request_size_bytes INTEGER,
    response_size_bytes INTEGER,
    
    -- Rate limiting
    rate_limit_remaining INTEGER,
    rate_limit_reset TIMESTAMP,
    
    -- Context
    ip_address INET,
    user_agent TEXT,
    
    requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_api_usage_user (user_id),
    INDEX idx_api_usage_endpoint (endpoint),
    INDEX idx_api_usage_requested (requested_at),
    INDEX idx_api_usage_key (api_key_id)
);

-- Audit log
CREATE TABLE audit_log (
    log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Actor information
    user_id UUID,
    api_key_id UUID,
    session_id UUID,
    
    -- Action details
    action VARCHAR(100) NOT NULL, -- LOGIN, LOGOUT, CREATE_ORDER, CANCEL_ORDER, etc.
    resource_type VARCHAR(50), -- USER, ORDER, STRATEGY, ACCOUNT
    resource_id VARCHAR(100),
    
    -- Action metadata
    old_values JSONB,
    new_values JSONB,
    metadata JSONB, -- Additional context
    
    -- Request details
    ip_address INET,
    user_agent TEXT,
    endpoint VARCHAR(200),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_audit_user (user_id),
    INDEX idx_audit_action (action),
    INDEX idx_audit_resource (resource_type, resource_id),
    INDEX idx_audit_created (created_at)
);
```

## Performance Optimization

### 9. Indexing Strategy

```sql
-- Composite indexes for common query patterns
CREATE INDEX CONCURRENTLY idx_orders_user_status_date 
ON orders(user_id, status, submitted_at);

CREATE INDEX CONCURRENTLY idx_positions_account_symbol 
ON account_positions(account_id, symbol);

CREATE INDEX CONCURRENTLY idx_signals_strategy_status_time 
ON trading_signals(strategy_id, status, created_at);

-- Partial indexes for filtered queries
CREATE INDEX CONCURRENTLY idx_orders_pending 
ON orders(submitted_at) 
WHERE status IN ('NEW', 'PARTIALLY_FILLED');

CREATE INDEX CONCURRENTLY idx_positions_open 
ON account_positions(symbol, opened_at) 
WHERE closed_at IS NULL;

-- Expression indexes for calculated fields
CREATE INDEX CONCURRENTLY idx_orders_pnl 
ON orders(user_id, (filled_quantity * average_fill_price)) 
WHERE status = 'FILLED';

-- GIN indexes for JSONB columns
CREATE INDEX CONCURRENTLY idx_strategies_parameters_gin 
ON strategies USING GIN (parameters);

CREATE INDEX CONCURRENTLY idx_risk_calculations_metrics_gin 
ON risk_calculations USING GIN (risk_metrics);
```

### 10. Partitioning Strategy

```sql
-- Partitioning by time for large tables
CREATE TABLE orders_y2024m01 PARTITION OF orders
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE orders_y2024m02 PARTITION OF orders
FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- Add monthly partitions for the current year
DO $$
DECLARE
    month_start DATE;
    month_end DATE;
    i INTEGER;
BEGIN
    FOR i IN 1..12 LOOP
        month_start := DATE('2024-01-01') + ((i-1) || ' months')::INTERVAL;
        month_end := month_start + '1 month'::INTERVAL;
        
        EXECUTE format(
            'CREATE TABLE orders_y2024m%2s PARTITION OF orders FOR VALUES FROM (%L) TO (%L)',
            LPAD(i::TEXT, 2, '0'),
            month_start,
            month_end
        );
    END LOOP;
END $$;

-- Automatic partition creation function
CREATE OR REPLACE FUNCTION create_monthly_partitions(table_name TEXT)
RETURNS void AS $$
DECLARE
    next_month_start DATE;
    next_month_end DATE;
    partition_name TEXT;
BEGIN
    -- Get next month dates
    next_month_start := (DATE_TRUNC('month', CURRENT_DATE) + '1 month'::INTERVAL)::DATE;
    next_month_end := (next_month_start + '1 month'::INTERVAL)::DATE;
    
    -- Create partition name
    partition_name := table_name || '_y' || 
                     EXTRACT(year FROM next_month_start) || 
                     'm' || LPAD(EXTRACT(month FROM next_month_start)::TEXT, 2, '0');
    
    -- Create partition
    EXECUTE format(
        'CREATE TABLE %I PARTITION OF %I FOR VALUES FROM (%L) TO (%L)',
        partition_name,
        table_name,
        next_month_start,
        next_month_end
    );
END;
$$ LANGUAGE plpgsql;
```

## Data Retention

### 11. Retention Policies

```sql
-- Retention configuration table
CREATE TABLE data_retention_policies (
    policy_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    table_name VARCHAR(100) NOT NULL,
    retention_period INTERVAL NOT NULL,
    archive_after INTERVAL NOT NULL,
    delete_after INTERVAL NOT NULL,
    
    -- Policy metadata
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(table_name)
);

-- Default retention policies
INSERT INTO data_retention_policies (table_name, retention_period, archive_after, delete_after) VALUES
('orders', '2 years', '1 year', '2 years'),
('order_fills', '2 years', '1 year', '2 years'),
('trading_signals', '1 year', '6 months', '1 year'),
('system_metrics', '30 days', '7 days', '30 days'),
('api_usage', '90 days', '30 days', '90 days'),
('audit_log', '7 years', '3 years', '7 years');

-- Archival function
CREATE OR REPLACE FUNCTION archive_old_data()
RETURNS INTEGER AS $$
DECLARE
    policy RECORD;
    archived_count INTEGER := 0;
BEGIN
    FOR policy IN SELECT * FROM data_retention_policies WHERE is_active LOOP
        -- Move old data to archive tables
        EXECUTE format(
            'INSERT INTO %I_archive 
             SELECT * FROM %I 
             WHERE created_at < NOW() - %L',
            policy.table_name,
            policy.table_name,
            policy.archive_after
        );
        
        GET DIAGNOSTICS archived_count = ROW_COUNT;
        
        -- Delete archived data from main table
        EXECUTE format(
            'DELETE FROM %I 
             WHERE created_at < NOW() - %L',
            policy.table_name,
            policy.archive_after
        );
    END LOOP;
    
    RETURN archived_count;
END;
$$ LANGUAGE plpgsql;

-- Cleanup function for permanent deletion
CREATE OR REPLACE FUNCTION cleanup_expired_data()
RETURNS INTEGER AS $$
DECLARE
    policy RECORD;
    deleted_count INTEGER := 0;
BEGIN
    FOR policy IN SELECT * FROM data_retention_policies WHERE is_active LOOP
        -- Delete expired data from archive tables
        EXECUTE format(
            'DELETE FROM %I_archive 
             WHERE created_at < NOW() - %L',
            policy.table_name,
            policy.delete_after
        );
        
        GET DIAGNOSTICS deleted_count = deleted_count + ROW_COUNT;
    END LOOP;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;
```

## Migration Strategy

### 12. Migration Framework

```sql
-- Migration tracking
CREATE TABLE schema_migrations (
    version VARCHAR(50) PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT,
    checksum VARCHAR(64),
    applied_by VARCHAR(100)
);

-- Sample migration
-- Migration: 20241201_add_strategy_performance_indexes
-- Description: Add indexes for strategy performance queries

BEGIN;

-- Create performance indexes
CREATE INDEX CONCURRENTLY idx_strategies_performance 
ON strategies(user_id, total_pnl DESC, total_trades DESC) 
WHERE is_active = true;

CREATE INDEX CONCURRENTLY idx_strategy_executions_performance 
ON strategy_executions(execution_type, created_at, pnl) 
WHERE pnl IS NOT NULL;

-- Update version
INSERT INTO schema_migrations (version, description, applied_by) 
VALUES ('20241201_add_strategy_performance_indexes', 
        'Add indexes for strategy performance queries', 
        'admin');

COMMIT;
```

### 13. Data Migration Scripts

```sql
-- Data migration: Normalize user risk settings
-- Migration: 20241202_normalize_risk_settings

BEGIN;

-- Create default risk profiles for users without one
INSERT INTO risk_profiles (user_id, profile_name, description)
SELECT 
    u.user_id,
    'Default Risk Profile',
    'Automatically generated default risk profile'
FROM users u
LEFT JOIN risk_profiles rp ON u.user_id = rp.user_id
WHERE rp.profile_id IS NULL;

-- Migrate risk settings from users table to risk_profiles
UPDATE risk_profiles rp
SET 
    max_daily_loss = u.max_daily_loss,
    max_position_size = u.max_daily_loss * 0.10, -- 10% of daily loss limit
    max_portfolio_risk = CASE 
        WHEN u.risk_tolerance = 'CONSERVATIVE' THEN 0.01  -- 1%
        WHEN u.risk_tolerance = 'MODERATE' THEN 0.02     -- 2%
        WHEN u.risk_tolerance = 'AGGRESSIVE' THEN 0.05   -- 5%
        ELSE 0.02
    END
FROM users u
WHERE rp.user_id = u.user_id 
  AND u.max_daily_loss IS NOT NULL;

-- Update schema version
INSERT INTO schema_migrations (version, description, applied_by) 
VALUES ('20241202_normalize_risk_settings', 
        'Normalize user risk settings into risk_profiles table', 
        'admin');

COMMIT;
```

### 14. Backup and Recovery

```sql
-- Backup configuration
CREATE TABLE backup_config (
    config_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    table_name VARCHAR(100) NOT NULL,
    backup_type VARCHAR(20) NOT NULL, -- FULL, INCREMENTAL, DIFFERENTIAL
    frequency VARCHAR(20) NOT NULL, -- DAILY, WEEKLY, MONTHLY
    retention_count INTEGER DEFAULT 7,
    
    -- Backup settings
    compress BOOLEAN DEFAULT true,
    encrypt BOOLEAN DEFAULT true,
    destination_path TEXT,
    
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Default backup configuration
INSERT INTO backup_config (table_name, backup_type, frequency, retention_count) VALUES
('users', 'FULL', 'WEEKLY', 4),
('orders', 'INCREMENTAL', 'DAILY', 30),
('user_broker_accounts', 'FULL', 'WEEKLY', 4),
('strategies', 'FULL', 'WEEKLY', 4),
('trading_signals', 'INCREMENTAL', 'DAILY', 7);

-- Recovery log
CREATE TABLE recovery_log (
    recovery_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    backup_id VARCHAR(100), -- Reference to backup file
    table_name VARCHAR(100) NOT NULL,
    recovery_type VARCHAR(20) NOT NULL, -- FULL, PARTIAL, POINT_IN_TIME
    recovered_records INTEGER,
    recovery_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) NOT NULL, -- SUCCESS, FAILED, IN_PROGRESS
    notes TEXT
);
```

## MongoDB Collections

### 15. Document Data

```javascript
// Strategy configurations (MongoDB)
db.strategy_configs.insertOne({
    _id: ObjectId(),
    strategy_id: UUID(),
    user_id: UUID(),
    
    // Configuration parameters
    parameters: {
        moving_averages: {
            short_period: 10,
            long_period: 20,
            ma_type: "SMA" // SMA, EMA, WMA
        },
        entry_conditions: {
            min_confidence: 0.6,
            volume_threshold: 1000000,
            volatility_max: 0.05
        },
        risk_management: {
            stop_loss_percentage: 0.02,
            take_profit_percentage: 0.04,
            position_sizing: "FIXED_PERCENT"
        }
    },
    
    // Backtesting configuration
    backtest_config: {
        start_date: ISODate("2024-01-01"),
        end_date: ISODate("2024-12-31"),
        initial_capital: 100000,
        commission_rate: 0.001,
        slippage: 0.0005
    },
    
    // Created/updated timestamps
    created_at: ISODate(),
    updated_at: ISODate(),
    
    // Status
    is_active: true,
    version: "1.0.0"
});

// System logs (MongoDB)
db.system_logs.createIndex({"timestamp": 1});
db.system_logs.createIndex({"level": 1, "service": 1});
db.system_logs.createIndex({"user_id": 1});

db.system_logs.insertOne({
    _id: ObjectId(),
    timestamp: ISODate(),
    level: "INFO", // DEBUG, INFO, WARN, ERROR
    service: "trading-engine",
    message: "Strategy execution completed",
    user_id: UUID(),
    strategy_id: UUID(),
    metadata: {
        execution_time_ms: 150,
        signals_generated: 3,
        orders_submitted: 2
    }
});

// Analytics data (MongoDB)
db.analytics_data.insertOne({
    _id: ObjectId(),
    date: ISODate("2024-12-01"),
    
    // Trading statistics
    trading_stats: {
        total_trades: 1250,
        winning_trades: 720,
        losing_trades: 530,
        total_pnl: 15420.50,
        avg_trade_pnl: 12.34,
        win_rate: 0.576,
        profit_factor: 1.45
    },
    
    // System statistics
    system_stats: {
        api_requests: 45670,
        avg_response_time_ms: 85,
        uptime_percentage: 99.98,
        error_rate: 0.001
    },
    
    // User statistics
    user_stats: {
        active_users: 1250,
        new_registrations: 45,
        strategy_executions: 890,
        total_commission: 2340.75
    },
    
    created_at: ISODate()
});
```

## Performance Considerations

### 16. Query Optimization

```sql
-- Optimize common queries with prepared statements

-- 1. User portfolio summary
PREPARE user_portfolio_summary(UUID) AS
SELECT 
    u.user_id,
    u.email,
    COUNT(DISTINCT uba.account_id) as broker_accounts,
    SUM(CASE WHEN ap.closed_at IS NULL THEN ap.quantity * ap.current_price ELSE 0 END) as portfolio_value,
    SUM(CASE WHEN ap.closed_at IS NULL THEN ab.cash_balance ELSE 0 END) as total_cash,
    COUNT(CASE WHEN ap.closed_at IS NULL THEN 1 END) as open_positions
FROM users u
LEFT JOIN user_broker_accounts uba ON u.user_id = uba.user_id
LEFT JOIN account_balances ab ON uba.account_id = ab.account_id
LEFT JOIN account_positions ap ON uba.account_id = ap.account_id
WHERE u.user_id = $1
  AND u.is_active = true
GROUP BY u.user_id, u.email;

-- 2. Strategy performance summary
PREPARE strategy_performance(UUID) AS
SELECT 
    s.strategy_id,
    s.strategy_name,
    s.strategy_type,
    COUNT(DISTINCT se.execution_id) as total_executions,
    COALESCE(SUM(se.pnl), 0) as total_pnl,
    COUNT(CASE WHEN se.pnl > 0 THEN 1 END)::DECIMAL / 
    NULLIF(COUNT(se.execution_id), 0) as win_rate,
    COALESCE(MAX(se.pnl), 0) as best_trade,
    COALESCE(MIN(se.pnl), 0) as worst_trade
FROM strategies s
LEFT JOIN strategy_executions se ON s.strategy_id = se.strategy_id
WHERE s.user_id = $1
  AND s.is_active = true
GROUP BY s.strategy_id, s.strategy_name, s.strategy_type
ORDER BY total_pnl DESC;

-- 3. Recent trading activity
PREPARE recent_trading_activity(UUID, TIMESTAMP) AS
SELECT 
    o.order_id,
    o.symbol,
    o.side,
    o.order_type,
    o.quantity,
    o.filled_quantity,
    o.average_fill_price,
    o.status,
    o.submitted_at,
    b.broker_name
FROM orders o
JOIN user_broker_accounts uba ON o.account_id = uba.account_id
JOIN brokers b ON uba.broker_id = b.broker_id
WHERE o.user_id = $1
  AND o.submitted_at >= $2
ORDER BY o.submitted_at DESC
LIMIT 100;
```

### 17. Maintenance Procedures

```sql
-- Automated maintenance procedures

-- 1. Update table statistics
CREATE OR REPLACE FUNCTION update_table_statistics()
RETURNS void AS $$
BEGIN
    ANALYZE users;
    ANALYZE orders;
    ANALYZE trading_signals;
    ANALYZE strategies;
    ANALYZE account_positions;
    
    -- Update statistics for partitions
    DO $$
    DECLARE
        partition_name TEXT;
    BEGIN
        FOR partition_name IN 
            SELECT schemaname || '.' || tablename 
            FROM pg_tables 
            WHERE tablename LIKE 'orders_y%'
        LOOP
            EXECUTE format('ANALYZE %I', partition_name);
        END LOOP;
    END $$;
END;
$$ LANGUAGE plpgsql;

-- 2. Reindex fragmented indexes
CREATE OR REPLACE FUNCTION reindex_fragmented_indexes()
RETURNS INTEGER AS $$
DECLARE
    idx_record RECORD;
    reindexed_count INTEGER := 0;
BEGIN
    FOR idx_record IN
        SELECT schemaname, indexname, indexdef
        FROM pg_indexes
        WHERE schemaname = 'public'
          AND indexname NOT LIKE '%_pkey'
    LOOP
        -- Check index size and fragmentation (simplified)
        IF random() < 0.1 THEN -- Reindex 10% randomly for demo
            EXECUTE format('REINDEX INDEX %I.%I', 
                          idx_record.schemaname, 
                          idx_record.indexname);
            reindexed_count := reindexed_count + 1;
        END IF;
    END LOOP;
    
    RETURN reindexed_count;
END;
$$ LANGUAGE plpgsql;

-- 3. Schedule maintenance
CREATE EXTENSION IF NOT EXISTS pg_cron;

-- Run maintenance daily at 2 AM
SELECT cron.schedule('daily-maintenance', '0 2 * * *', $$
    SELECT update_table_statistics();
    SELECT reindex_fragmented_indexes();
    SELECT archive_old_data();
$$);

-- Run cleanup weekly on Sunday at 3 AM
SELECT cron.schedule('weekly-cleanup', '0 3 * * 0', $$
    SELECT cleanup_expired_data();
    SELECT create_monthly_partitions('orders');
$$);
```

## Conclusion

The database schema is designed for:

1. **Scalability**: Horizontal and vertical scaling capabilities
2. **Performance**: Optimized indexes and query patterns
3. **Flexibility**: Support for multiple database technologies
4. **Reliability**: ACID compliance and data integrity
5. **Maintainability**: Clear structure and documentation

### Key Design Principles

- **Normalization**: Third normal form for transactional data
- **Denormalization**: Strategic denormalization for performance
- **Partitioning**: Time-based partitioning for large tables
- **Indexing**: Comprehensive indexing strategy
- **Retention**: Automated data lifecycle management
- **Backup**: Robust backup and recovery procedures

### Next Steps

1. **Database Setup**: Follow the deployment guide for initial setup
2. **Performance Tuning**: Monitor query performance and optimize as needed
3. **Scaling**: Implement read replicas and connection pooling
4. **Monitoring**: Set up database performance monitoring
5. **Backup Strategy**: Implement and test backup procedures

---

**Need Help?** Contact the development team for database optimization assistance or custom schema modifications.