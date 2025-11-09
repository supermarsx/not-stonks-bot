-- Migration: Trading Frequency Management System
-- Version: 2
-- Created: 2025-11-06

-- Trading Frequency Settings
CREATE TABLE IF NOT EXISTS frequency_settings (
    id VARCHAR(50) PRIMARY KEY,
    strategy_id VARCHAR(50) NOT NULL,
    frequency_type VARCHAR(20) NOT NULL,
    interval_seconds INTEGER NOT NULL DEFAULT 300,
    max_trades_per_minute INTEGER DEFAULT 1,
    max_trades_per_hour INTEGER DEFAULT 10,
    max_trades_per_day INTEGER DEFAULT 100,
    position_size_multiplier REAL DEFAULT 1.0,
    frequency_based_sizing BOOLEAN DEFAULT 1,
    cooldown_periods INTEGER DEFAULT 0,
    market_hours_only BOOLEAN DEFAULT 0,
    max_daily_frequency_risk REAL DEFAULT 0.05,
    frequency_volatility_adjustment BOOLEAN DEFAULT 1,
    correlation_limits TEXT DEFAULT '{}',
    enable_alerts BOOLEAN DEFAULT 1,
    alert_thresholds TEXT DEFAULT '{}',
    auto_optimization BOOLEAN DEFAULT 0,
    optimization_period_hours INTEGER DEFAULT 24,
    custom_intervals TEXT DEFAULT '[]',
    time_window_limits TEXT DEFAULT '[]',
    strategy_overrides TEXT DEFAULT '{}',
    is_active BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (strategy_id) REFERENCES strategies (id)
);

-- Indexes for frequency settings
CREATE INDEX IF NOT EXISTS idx_frequency_settings_strategy ON frequency_settings(strategy_id);
CREATE INDEX IF NOT EXISTS idx_frequency_settings_active ON frequency_settings(is_active);
CREATE INDEX IF NOT EXISTS idx_frequency_settings_type ON frequency_settings(frequency_type);

-- Frequency Metrics Tracking
CREATE TABLE IF NOT EXISTS frequency_metrics (
    id VARCHAR(50) PRIMARY KEY,
    strategy_id VARCHAR(50) NOT NULL,
    trades_in_last_minute INTEGER DEFAULT 0,
    trades_in_last_hour INTEGER DEFAULT 0,
    trades_today INTEGER DEFAULT 0,
    trades_this_week INTEGER DEFAULT 0,
    trades_this_month INTEGER DEFAULT 0,
    current_frequency_rate REAL DEFAULT 0.0,
    average_frequency_rate REAL DEFAULT 0.0,
    target_frequency_rate REAL DEFAULT 0.0,
    frequency_efficiency REAL DEFAULT 0.0,
    frequency_sharpe REAL DEFAULT 0.0,
    frequency_drawdown REAL DEFAULT 0.0,
    first_trade_today TIMESTAMP,
    last_trade_time TIMESTAMP,
    cooldown_end_time TIMESTAMP,
    in_cooldown BOOLEAN DEFAULT 0,
    alerts_triggered_count INTEGER DEFAULT 0,
    threshold_violations INTEGER DEFAULT 0,
    measurement_start TIMESTAMP NOT NULL,
    measurement_end TIMESTAMP NOT NULL,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (strategy_id) REFERENCES strategies (id)
);

-- Indexes for frequency metrics
CREATE INDEX IF NOT EXISTS idx_frequency_metrics_strategy_time ON frequency_metrics(strategy_id, recorded_at);
CREATE INDEX IF NOT EXISTS idx_frequency_metrics_measurement ON frequency_metrics(measurement_start, measurement_end);

-- Frequency Alerts
CREATE TABLE IF NOT EXISTS frequency_alerts (
    id VARCHAR(50) PRIMARY KEY,
    strategy_id VARCHAR(50) NOT NULL,
    alert_type VARCHAR(30) NOT NULL,
    severity VARCHAR(10) NOT NULL,
    message TEXT NOT NULL,
    threshold_value REAL,
    current_value REAL,
    acknowledged BOOLEAN DEFAULT 0,
    auto_resolve BOOLEAN DEFAULT 1,
    metadata TEXT DEFAULT '{}',
    trigger_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    acknowledged_at TIMESTAMP,
    resolved_at TIMESTAMP,
    FOREIGN KEY (strategy_id) REFERENCES strategies (id)
);

-- Indexes for frequency alerts
CREATE INDEX IF NOT EXISTS idx_frequency_alerts_strategy_time ON frequency_alerts(strategy_id, trigger_time);
CREATE INDEX IF NOT EXISTS idx_frequency_alerts_type_severity ON frequency_alerts(alert_type, severity);
CREATE INDEX IF NOT EXISTS idx_frequency_alerts_status ON frequency_alerts(acknowledged, resolved_at);

-- Frequency Optimization Recommendations
CREATE TABLE IF NOT EXISTS frequency_optimizations (
    id VARCHAR(50) PRIMARY KEY,
    strategy_id VARCHAR(50) NOT NULL,
    recommended_interval_seconds INTEGER NOT NULL,
    recommended_position_size_multiplier REAL DEFAULT 1.0,
    confidence_level REAL DEFAULT 0.0,
    expected_improvement REAL DEFAULT 0.0,
    historical_sharpe REAL DEFAULT 0.0,
    expected_sharpe REAL DEFAULT 0.0,
    max_drawdown_reduction REAL DEFAULT 0.0,
    win_rate_improvement REAL DEFAULT 0.0,
    backtest_period_days INTEGER DEFAULT 30,
    analysis_data TEXT DEFAULT '{}',
    implemented BOOLEAN DEFAULT 0,
    implementation_date TIMESTAMP,
    performance_after_implementation TEXT DEFAULT '{}',
    status VARCHAR(20) DEFAULT 'pending',
    optimization_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (strategy_id) REFERENCES strategies (id)
);

-- Indexes for frequency optimizations
CREATE INDEX IF NOT EXISTS idx_frequency_optimizations_strategy ON frequency_optimizations(strategy_id);
CREATE INDEX IF NOT EXISTS idx_frequency_optimizations_status ON frequency_optimizations(status);
CREATE INDEX IF NOT EXISTS idx_frequency_optimizations_date ON frequency_optimizations(optimization_date);

-- Individual Trade Frequency Records
CREATE TABLE IF NOT EXISTS trade_frequency_records (
    id VARCHAR(50) PRIMARY KEY,
    strategy_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    trade_type VARCHAR(10) NOT NULL,
    quantity REAL,
    price REAL,
    trades_in_last_minute_before INTEGER DEFAULT 0,
    trades_in_last_hour_before INTEGER DEFAULT 0,
    trades_in_last_day_before INTEGER DEFAULT 0,
    trade_time TIMESTAMP NOT NULL,
    previous_trade_time TIMESTAMP,
    time_since_last_trade_seconds INTEGER,
    position_size_before REAL,
    position_size_after REAL,
    position_change REAL,
    new_frequency_rate REAL,
    risk_impact_score REAL,
    strategy_frequency_type VARCHAR(20),
    cooldown_active BOOLEAN DEFAULT 0,
    cooldown_end_time TIMESTAMP,
    market_volatility REAL,
    market_regime VARCHAR(20),
    metadata TEXT DEFAULT '{}',
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (strategy_id) REFERENCES strategies (id)
);

-- Indexes for trade frequency records
CREATE INDEX IF NOT EXISTS idx_trade_frequency_strategy_time ON trade_frequency_records(strategy_id, trade_time);
CREATE INDEX IF NOT EXISTS idx_trade_frequency_symbol_time ON trade_frequency_records(symbol, trade_time);
CREATE INDEX IF NOT EXISTS idx_trade_frequency_rate ON trade_frequency_records(new_frequency_rate);
CREATE INDEX IF NOT EXISTS idx_trade_frequency_impact ON trade_frequency_records(risk_impact_score);

-- Frequency Constraints and Limits
CREATE TABLE IF NOT EXISTS frequency_constraints (
    id VARCHAR(50) PRIMARY KEY,
    strategy_id VARCHAR(50) NOT NULL,
    constraint_type VARCHAR(20) NOT NULL,
    constraint_name VARCHAR(50) NOT NULL,
    description TEXT,
    max_trades_per_minute INTEGER,
    max_trades_per_hour INTEGER,
    max_trades_per_day INTEGER,
    min_interval_seconds INTEGER,
    max_interval_seconds INTEGER,
    time_window_start VARCHAR(10),
    time_window_end VARCHAR(10),
    time_window_timezone VARCHAR(50) DEFAULT 'UTC',
    applicable_days TEXT DEFAULT '[]',
    max_position_size REAL,
    min_position_size REAL,
    position_size_formula TEXT,
    max_daily_risk_percentage REAL,
    max_weekly_risk_percentage REAL,
    max_drawdown_percentage REAL,
    is_active BOOLEAN DEFAULT 1,
    priority INTEGER DEFAULT 0,
    metadata TEXT DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    effective_from TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    effective_until TIMESTAMP,
    FOREIGN KEY (strategy_id) REFERENCES strategies (id)
);

-- Indexes for frequency constraints
CREATE INDEX IF NOT EXISTS idx_frequency_constraints_strategy ON frequency_constraints(strategy_id);
CREATE INDEX IF NOT EXISTS idx_frequency_constraints_type ON frequency_constraints(constraint_type);
CREATE INDEX IF NOT EXISTS idx_frequency_constraints_active ON frequency_constraints(is_active, priority);
CREATE INDEX IF NOT EXISTS idx_frequency_constraints_effective ON frequency_constraints(effective_from, effective_until);

-- Frequency Analytics and Reporting
CREATE TABLE IF NOT EXISTS frequency_analytics (
    id VARCHAR(50) PRIMARY KEY,
    strategy_id VARCHAR(50) NOT NULL,
    period_type VARCHAR(20) NOT NULL,
    period_start TIMESTAMP NOT NULL,
    period_end TIMESTAMP NOT NULL,
    total_trades INTEGER DEFAULT 0,
    avg_trades_per_day REAL DEFAULT 0.0,
    max_trades_in_hour INTEGER DEFAULT 0,
    min_trades_in_hour INTEGER DEFAULT 0,
    std_trades_per_hour REAL DEFAULT 0.0,
    trades_by_hour TEXT DEFAULT '{}',
    trades_by_day_of_week TEXT DEFAULT '{}',
    trades_by_frequency_type TEXT DEFAULT '{}',
    avg_frequency_rate REAL DEFAULT 0.0,
    frequency_efficiency REAL DEFAULT 0.0,
    frequency_sharpe_ratio REAL DEFAULT 0.0,
    max_frequency_drawdown REAL DEFAULT 0.0,
    frequency_var_95 REAL DEFAULT 0.0,
    frequency_volatility REAL DEFAULT 0.0,
    frequency_beta REAL DEFAULT 0.0,
    frequency_correlation REAL DEFAULT 0.0,
    optimal_frequency_range TEXT DEFAULT '{}',
    recommended_adjustments TEXT DEFAULT '{}',
    performance_impact REAL DEFAULT 0.0,
    trades_change_percentage REAL DEFAULT 0.0,
    frequency_efficiency_change REAL DEFAULT 0.0,
    performance_change REAL DEFAULT 0.0,
    detailed_analytics TEXT DEFAULT '{}',
    charts_data TEXT DEFAULT '{}',
    analysis_version VARCHAR(20) DEFAULT '1.0',
    calculated_by VARCHAR(50),
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (strategy_id) REFERENCES strategies (id)
);

-- Indexes for frequency analytics
CREATE INDEX IF NOT EXISTS idx_frequency_analytics_strategy_period ON frequency_analytics(strategy_id, period_start);
CREATE INDEX IF NOT EXISTS idx_frequency_analytics_period_type ON frequency_analytics(period_type, period_start);
CREATE INDEX IF NOT EXISTS idx_frequency_analytics_calculated ON frequency_analytics(calculated_at);

-- Update strategy permissions to include frequency management
ALTER TABLE user_strategy_permissions ADD COLUMN can_manage_frequency BOOLEAN DEFAULT 0;
ALTER TABLE user_strategy_permissions ADD COLUMN can_view_frequency_analytics BOOLEAN DEFAULT 1;