"""
Database models for strategy API

Provides SQLAlchemy models for:
- Strategy configurations
- Performance metrics
- Backtest results
- User strategy associations
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from sqlalchemy import (
    Column, String, Text, JSON, DateTime, Float, Integer, Boolean, 
    ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func


Base = declarative_base()


class StrategyStatus(str, Enum):
    """Strategy status values"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    TESTING = "testing"
    ERROR = "error"


class SignalType(str, Enum):
    """Signal type values"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


class RiskLevel(str, Enum):
    """Risk level values"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class User(Base):
    """User model for authentication and authorization"""
    __tablename__ = "users"
    
    id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), default="trader", index=True)
    permissions = Column(JSON, default=list)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    last_login = Column(DateTime(timezone=True))
    
    # Relationships
    strategies = relationship("StrategyDB", back_populates="creator", cascade="all, delete-orphan")
    strategy_permissions = relationship("UserStrategyPermission", back_populates="user", cascade="all, delete-orphan")


class StrategyDB(Base):
    """Strategy configuration model"""
    __tablename__ = "strategies"
    
    id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=False, index=True)
    description = Column(Text)
    category = Column(String(50), nullable=False, index=True)
    status = Column(String(20), default="inactive", index=True)
    risk_level = Column(String(10), default="medium", index=True)
    tags = Column(JSON, default=list)
    timeframe = Column(String(20), default="1h")
    symbols = Column(JSON, default=list)
    parameters = Column(JSON, default=dict)
    
    # Ownership
    created_by = Column(String(50), ForeignKey("users.id"), nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    
    # Execution status
    last_executed = Column(DateTime(timezone=True))
    execution_count = Column(Integer, default=0)
    
    # Relationships
    creator = relationship("User", back_populates="strategies")
    performance = relationship("StrategyPerformanceDB", back_populates="strategy", uselist=False, cascade="all, delete-orphan")
    signals = relationship("StrategySignalDB", back_populates="strategy", cascade="all, delete-orphan")
    backtests = relationship("BacktestDB", back_populates="strategy", cascade="all, delete-orphan")
    history = relationship("StrategyHistoryDB", back_populates="strategy", cascade="all, delete-orphan")
    user_permissions = relationship("UserStrategyPermission", back_populates="strategy", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_strategies_created_by', 'created_by'),
        Index('idx_strategies_category_status', 'category', 'status'),
        Index('idx_strategies_created_at', 'created_at'),
    )


class StrategyPerformanceDB(Base):
    """Strategy performance metrics model"""
    __tablename__ = "strategy_performance"
    
    id = Column(String(50), primary_key=True)
    strategy_id = Column(String(50), ForeignKey("strategies.id"), unique=True, nullable=False)
    
    # Performance metrics
    total_return = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    sortino_ratio = Column(Float)
    calmar_ratio = Column(Float)
    max_drawdown = Column(Float, default=0.0)
    win_rate = Column(Float, default=0.0)
    profit_factor = Column(Float)
    
    # Trade statistics
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    average_win = Column(Float)
    average_loss = Column(Float)
    best_trade = Column(Float)
    worst_trade = Column(Float)
    
    # Risk metrics
    volatility = Column(Float)
    beta = Column(Float)
    alpha = Column(Float)
    var_95 = Column(Float)
    cvar_95 = Column(Float)
    
    # Period info
    period_start = Column(DateTime(timezone=True))
    period_end = Column(DateTime(timezone=True))
    
    # Timestamps
    calculated_at = Column(DateTime(timezone=True), default=func.now(), index=True)
    last_updated = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    
    # Relationships
    strategy = relationship("StrategyDB", back_populates="performance")
    
    # Indexes
    __table_args__ = (
        Index('idx_performance_strategy_id', 'strategy_id'),
        Index('idx_performance_calculated_at', 'calculated_at'),
    )


class StrategySignalDB(Base):
    """Strategy signal model"""
    __tablename__ = "strategy_signals"
    
    id = Column(String(50), primary_key=True)
    strategy_id = Column(String(50), ForeignKey("strategies.id"), nullable=False, index=True)
    
    # Signal details
    symbol = Column(String(20), nullable=False, index=True)
    signal_type = Column(String(10), nullable=False, index=True)
    confidence = Column(Float, default=0.0)
    price = Column(Float)
    quantity = Column(Float)
    metadata = Column(JSON, default=dict)
    
    # Timing
    signal_time = Column(DateTime(timezone=True), default=func.now(), index=True)
    expires_at = Column(DateTime(timezone=True))
    
    # Relationships
    strategy = relationship("StrategyDB", back_populates="signals")
    
    # Indexes
    __table_args__ = (
        Index('idx_signals_strategy_time', 'strategy_id', 'signal_time'),
        Index('idx_signals_symbol_time', 'symbol', 'signal_time'),
    )


class BacktestDB(Base):
    """Backtest results model"""
    __tablename__ = "backtests"
    
    id = Column(String(50), primary_key=True)
    strategy_id = Column(String(50), ForeignKey("strategies.id"), nullable=False, index=True)
    
    # Backtest configuration
    start_date = Column(DateTime(timezone=True), nullable=False, index=True)
    end_date = Column(DateTime(timezone=True), nullable=False, index=True)
    initial_capital = Column(Float, nullable=False)
    final_capital = Column(Float, nullable=False)
    total_return = Column(Float, nullable=False)
    
    # Configuration
    commission = Column(Float, default=0.0)
    slippage = Column(Float, default=0.0)
    symbols = Column(JSON, default=list)
    include_monte_carlo = Column(Boolean, default=False)
    monte_carlo_runs = Column(Integer)
    walk_forward = Column(Boolean, default=False)
    walk_forward_window = Column(Integer)
    
    # Results
    performance_data = Column(JSON, default=dict)
    trades = Column(JSON, default=list)
    equity_curve = Column(JSON, default=list)
    drawdown_periods = Column(JSON, default=list)
    monte_carlo_results = Column(JSON)
    walk_forward_results = Column(JSON)
    
    # Status
    status = Column(String(20), default="completed", index=True)
    error_message = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=func.now(), index=True)
    completed_at = Column(DateTime(timezone=True))
    
    # Relationships
    strategy = relationship("StrategyDB", back_populates="backtests")
    
    # Indexes
    __table_args__ = (
        Index('idx_backtests_strategy_created', 'strategy_id', 'created_at'),
        Index('idx_backtests_status', 'status'),
    )


class StrategyHistoryDB(Base):
    """Strategy history and events model"""
    __tablename__ = "strategy_history"
    
    id = Column(String(50), primary_key=True)
    strategy_id = Column(String(50), ForeignKey("strategies.id"), nullable=False, index=True)
    
    # Event details
    event_type = Column(String(50), nullable=False, index=True)
    event_data = Column(JSON, default=dict)
    description = Column(Text)
    
    # Status changes
    old_status = Column(String(20))
    new_status = Column(String(20))
    
    # User who performed action
    performed_by = Column(String(50))
    
    # Timing
    event_time = Column(DateTime(timezone=True), default=func.now(), index=True)
    
    # Relationships
    strategy = relationship("StrategyDB", back_populates="history")
    
    # Indexes
    __table_args__ = (
        Index('idx_history_strategy_time', 'strategy_id', 'event_time'),
        Index('idx_history_event_type', 'event_type'),
    )


class UserStrategyPermission(Base):
    """User permissions for specific strategies"""
    __tablename__ = "user_strategy_permissions"
    
    id = Column(String(50), primary_key=True)
    user_id = Column(String(50), ForeignKey("users.id"), nullable=False, index=True)
    strategy_id = Column(String(50), ForeignKey("strategies.id"), nullable=False, index=True)
    
    # Permissions
    can_read = Column(Boolean, default=True)
    can_write = Column(Boolean, default=False)
    can_execute = Column(Boolean, default=False)
    can_delete = Column(Boolean, default=False)
    can_share = Column(Boolean, default=False)
    
    # Timestamps
    granted_at = Column(DateTime(timezone=True), default=func.now())
    granted_by = Column(String(50))
    
    # Relationships
    user = relationship("User", back_populates="user_permissions")
    strategy = relationship("StrategyDB", back_populates="user_permissions")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('user_id', 'strategy_id', name='uq_user_strategy_permission'),
        Index('idx_user_strategy_permissions_user', 'user_id'),
        Index('idx_user_strategy_permissions_strategy', 'strategy_id'),
    )


class StrategyEnsembleDB(Base):
    """Strategy ensemble configuration"""
    __tablename__ = "strategy_ensembles"
    
    id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    created_by = Column(String(50), ForeignKey("users.id"), nullable=False)
    
    # Ensemble configuration
    strategy_weights = Column(JSON, default=dict)  # {"strategy_id": weight}
    rebalance_frequency = Column(String(20), default="daily")
    risk_management = Column(JSON, default=dict)
    
    # Status
    is_active = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    last_rebalanced = Column(DateTime(timezone=True))
    
    # Performance
    performance_data = Column(JSON, default=dict)
    
    # Indexes
    __table_args__ = (
        Index('idx_ensembles_created_by', 'created_by'),
        Index('idx_ensembles_active', 'is_active'),
    )


# Strategy templates for common use cases
class StrategyTemplateDB(Base):
    """Pre-configured strategy templates"""
    __tablename__ = "strategy_templates"
    
    id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    category = Column(String(50), nullable=False, index=True)
    risk_level = Column(String(10), default="medium", index=True)
    
    # Template configuration
    default_parameters = Column(JSON, default=dict)
    required_symbols = Column(JSON, default=list)
    timeframe = Column(String(20), default="1h")
    tags = Column(JSON, default=list)
    
    # Usage statistics
    usage_count = Column(Integer, default=0)
    rating = Column(Float, default=0.0)
    
    # Metadata
    is_public = Column(Boolean, default=True)
    created_by = Column(String(50), ForeignKey("users.id"))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_templates_category', 'category'),
        Index('idx_templates_risk_level', 'risk_level'),
        Index('idx_templates_public', 'is_public'),
    )


# Trading Frequency Management Models
class FrequencySettingsDB(Base):
    """Frequency configuration settings model"""
    __tablename__ = "frequency_settings"
    
    id = Column(String(50), primary_key=True)
    strategy_id = Column(String(50), ForeignKey("strategies.id"), nullable=False, index=True)
    
    # Frequency configuration
    frequency_type = Column(String(20), nullable=False, index=True)
    interval_seconds = Column(Integer, nullable=False, default=300)
    
    # Trading limits
    max_trades_per_minute = Column(Integer, default=1)
    max_trades_per_hour = Column(Integer, default=10)
    max_trades_per_day = Column(Integer, default=100)
    
    # Position sizing
    position_size_multiplier = Column(Float, default=1.0)
    frequency_based_sizing = Column(Boolean, default=True)
    
    # Cooldown and timing
    cooldown_periods = Column(Integer, default=0)
    market_hours_only = Column(Boolean, default=False)
    
    # Risk management
    max_daily_frequency_risk = Column(Float, default=0.05)
    frequency_volatility_adjustment = Column(Boolean, default=True)
    correlation_limits = Column(JSON, default=dict)
    
    # Alerting
    enable_alerts = Column(Boolean, default=True)
    alert_thresholds = Column(JSON, default=dict)
    
    # Optimization
    auto_optimization = Column(Boolean, default=False)
    optimization_period_hours = Column(Integer, default=24)
    
    # Custom settings
    custom_intervals = Column(JSON, default=list)
    time_window_limits = Column(JSON, default=list)
    strategy_overrides = Column(JSON, default=dict)
    
    # Status
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_frequency_settings_strategy', 'strategy_id'),
        Index('idx_frequency_settings_active', 'is_active'),
        Index('idx_frequency_settings_type', 'frequency_type'),
    )


class FrequencyMetricsDB(Base):
    """Frequency metrics tracking model"""
    __tablename__ = "frequency_metrics"
    
    id = Column(String(50), primary_key=True)
    strategy_id = Column(String(50), ForeignKey("strategies.id"), nullable=False, index=True)
    
    # Trade counts
    trades_in_last_minute = Column(Integer, default=0)
    trades_in_last_hour = Column(Integer, default=0)
    trades_today = Column(Integer, default=0)
    trades_this_week = Column(Integer, default=0)
    trades_this_month = Column(Integer, default=0)
    
    # Rates
    current_frequency_rate = Column(Float, default=0.0)
    average_frequency_rate = Column(Float, default=0.0)
    target_frequency_rate = Column(Float, default=0.0)
    
    # Performance metrics
    frequency_efficiency = Column(Float, default=0.0)
    frequency_sharpe = Column(Float, default=0.0)
    frequency_drawdown = Column(Float, default=0.0)
    
    # Time tracking
    first_trade_today = Column(DateTime(timezone=True))
    last_trade_time = Column(DateTime(timezone=True))
    
    # Cooldown tracking
    cooldown_end_time = Column(DateTime(timezone=True))
    in_cooldown = Column(Boolean, default=False)
    
    # Alert tracking
    alerts_triggered_count = Column(Integer, default=0)
    threshold_violations = Column(Integer, default=0)
    
    # Measurement period
    measurement_start = Column(DateTime(timezone=True), nullable=False, index=True)
    measurement_end = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # Timestamps
    recorded_at = Column(DateTime(timezone=True), default=func.now(), index=True)
    
    # Indexes
    __table_args__ = (
        Index('idx_frequency_metrics_strategy_time', 'strategy_id', 'recorded_at'),
        Index('idx_frequency_metrics_measurement', 'measurement_start', 'measurement_end'),
    )


class FrequencyAlertDB(Base):
    """Frequency alerts model"""
    __tablename__ = "frequency_alerts"
    
    id = Column(String(50), primary_key=True)
    strategy_id = Column(String(50), ForeignKey("strategies.id"), nullable=False, index=True)
    
    # Alert details
    alert_type = Column(String(30), nullable=False, index=True)
    severity = Column(String(10), nullable=False, index=True)
    message = Column(Text, nullable=False)
    
    #пп threshold values
    threshold_value = Column(Float)
    current_value = Column(Float)
    
    # Status
    acknowledged = Column(Boolean, default=False)
    auto_resolve = Column(Boolean, default=True)
    
    # Metadata
    metadata = Column(JSON, default=dict)
    
    # Timestamps
    trigger_time = Column(DateTime(timezone=True), default=func.now(), index=True)
    acknowledged_at = Column(DateTime(timezone=True))
    resolved_at = Column(DateTime(timezone=True))
    
    # Indexes
    __table_args__ = (
        Index('idx_frequency_alerts_strategy_time', 'strategy_id', 'trigger_time'),
        Index('idx_frequency_alerts_type_severity', 'alert_type', 'severity'),
        Index('idx_frequency_alerts_status', 'acknowledged', 'resolved_at'),
    )


class FrequencyOptimizationDB(Base):
    """Frequency optimization recommendations model"""
    __tablename__ = "frequency_optimizations"
    
    id = Column(String(50), primary_key=True)
    strategy_id = Column(String(50), ForeignKey("strategies.id"), nullable=False, index=True)
    
    # Optimization recommendations
    recommended_interval_seconds = Column(Integer, nullable=False)
    recommended_position_size_multiplier = Column(Float, default=1.0)
    confidence_level = Column(Float, default=0.0)
    expected_improvement = Column(Float, default=0.0)
    
    # Performance metrics
    historical_sharpe = Column(Float, default=0.0)
    expected_sharpe = Column(Float, default=0.0)
    max_drawdown_reduction = Column(Float, default=0.0)
    win_rate_improvement = Column(Float, default=0.0)
    
    # Analysis details
    backtest_period_days = Column(Integer, default=30)
    analysis_data = Column(JSON, default=dict)
    
    # Implementation tracking
    implemented = Column(Boolean, default=False)
    implementation_date = Column(DateTime(timezone=True))
    performance_after_implementation = Column(JSON, default=dict)
    
    # Status
    status = Column(String(20), default="pending", index=True)  # pending, implemented, rejected
    
    # Timestamps
    optimization_date = Column(DateTime(timezone=True), default=func.now(), index=True)
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_frequency_optimizations_strategy', 'strategy_id'),
        Index('idx_frequency_optimizations_status', 'status'),
        Index('idx_frequency_optimizations_date', 'optimization_date'),
    )


class TradeFrequencyDB(Base):
    """Individual trade frequency tracking model"""
    __tablename__ = "trade_frequency_records"
    
    id = Column(String(50), primary_key=True)
    strategy_id = Column(String(50), ForeignKey("strategies.id"), nullable=False, index=True)
    
    # Trade details
    symbol = Column(String(20), nullable=False, index=True)
    trade_type = Column(String(10), nullable=False)  # buy, sell, close
    quantity = Column(Float)
    price = Column(Float)
    
    # Frequency context
    trades_in_last_minute_before = Column(Integer, default=0)
    trades_in_last_hour_before = Column(Integer, default=0)
    trades_in_last_day_before = Column(Integer, default=0)
    
    # Timing
    trade_time = Column(DateTime(timezone=True), nullable=False, index=True)
    previous_trade_time = Column(DateTime(timezone=True))
    time_since_last_trade_seconds = Column(Integer)
    
    # Position context
    position_size_before = Column(Float)
    position_size_after = Column(Float)
    position_change = Column(Float)
    
    # Frequency impact
    new_frequency_rate = Column(Float)
    risk_impact_score = Column(Float)
    
    # Strategy state
    strategy_frequency_type = Column(String(20))
    cooldown_active = Column(Boolean, default=False)
    cooldown_end_time = Column(DateTime(timezone=True))
    
    # Market context
    market_volatility = Column(Float)
    market_regime = Column(String(20))
    
    # Metadata
    metadata = Column(JSON, default=dict)
    
    # Timestamps
    recorded_at = Column(DateTime(timezone=True), default=func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_trade_frequency_strategy_time', 'strategy_id', 'trade_time'),
        Index('idx_trade_frequency_symbol_time', 'symbol', 'trade_time'),
        Index('idx_trade_frequency_rate', 'new_frequency_rate'),
        Index('idx_trade_frequency_impact', 'risk_impact_score'),
    )


class FrequencyConstraintsDB(Base):
    """Frequency constraints and limits model"""
    __tablename__ = "frequency_constraints"
    
    id = Column(String(50), primary_key=True)
    strategy_id = Column(String(50), ForeignKey("strategies.id"), nullable=False, index=True)
    
    # Constraint details
    constraint_type = Column(String(20), nullable=False, index=True)  # hard_limit, soft_limit, recommended, minimum
    constraint_name = Column(String(50), nullable=False)
    description = Column(Text)
    
    # Constraint values
    max_trades_per_minute = Column(Integer)
    max_trades_per_hour = Column(Integer)
    max_trades_per_day = Column(Integer)
    min_interval_seconds = Column(Integer)
    max_interval_seconds = Column(Integer)
    
    # Time window constraints
    time_window_start = Column(String(10))  # HH:MM format
    time_window_end = Column(String(10))    # HH:MM format
    time_window_timezone = Column(String(50), default="UTC")
    applicable_days = Column(JSON, default=list)  # ["monday", "tuesday", ...]
    
    # Position size constraints
    max_position_size = Column(Float)
    min_position_size = Column(Float)
    position_size_formula = Column(Text)
    
    # Risk constraints
    max_daily_risk_percentage = Column(Float)
    max_weekly_risk_percentage = Column(Float)
    max_drawdown_percentage = Column(Float)
    
    # Status
    is_active = Column(Boolean, default=True)
    priority = Column(Integer, default=0)  # Higher numbers = higher priority
    
    # Metadata
    metadata = Column(JSON, default=dict)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    effective_from = Column(DateTime(timezone=True), default=func.now())
    effective_until = Column(DateTime(timezone=True))
    
    # Indexes
    __table_args__ = (
        Index('idx_frequency_constraints_strategy', 'strategy_id'),
        Index('idx_frequency_constraints_type', 'constraint_type'),
        Index('idx_frequency_constraints_active', 'is_active', 'priority'),
        Index('idx_frequency_constraints_effective', 'effective_from', 'effective_until'),
    )


class FrequencyAnalyticsDB(Base):
    """Frequency analytics and reporting model"""
    __tablename__ = "frequency_analytics"
    
    id = Column(String(50), primary_key=True)
    strategy_id = Column(String(50), ForeignKey("strategies.id"), nullable=False, index=True)
    
    # Analytics period
    period_type = Column(String(20), nullable=False, index=True)  # daily, weekly, monthly
    period_start = Column(DateTime(timezone=True), nullable=False, index=True)
    period_end = Column(DateTime(timezone=True), nullable=False)
    
    # Trade frequency analytics
    total_trades = Column(Integer, default=0)
    avg_trades_per_day = Column(Float, default=0.0)
    max_trades_in_hour = Column(Integer, default=0)
    min_trades_in_hour = Column(Integer, default=0)
    std_trades_per_hour = Column(Float, default=0.0)
    
    # Frequency distribution
    trades_by_hour = Column(JSON, default=dict)  # {"0": count, "1": count, ...}
    trades_by_day_of_week = Column(JSON, default=dict)  # {"monday": count, ...}
    trades_by_frequency_type = Column(JSON, default=dict)
    
    # Performance analytics
    avg_frequency_rate = Column(Float, default=0.0)
    frequency_efficiency = Column(Float, default=0.0)
    frequency_sharpe_ratio = Column(Float, default=0.0)
    max_frequency_drawdown = Column(Float, default=0.0)
    
    # Risk analytics
    frequency_var_95 = Column(Float, default=0.0)
    frequency_volatility = Column(Float, default=0.0)
    frequency_beta = Column(Float, default=0.0)
    frequency_correlation = Column(Float, default=0.0)
    
    # Optimization metrics
    optimal_frequency_range = Column(JSON, default=dict)  # {"min": value, "max": value}
    recommended_adjustments = Column(JSON, default=dict)
    performance_impact = Column(Float, default=0.0)
    
    # Comparison metrics (vs previous period)
    trades_change_percentage = Column(Float, default=0.0)
    frequency_efficiency_change = Column(Float, default=0.0)
    performance_change = Column(Float, default=0.0)
    
    # Analytics data
    detailed_analytics = Column(JSON, default=dict)
    charts_data = Column(JSON, default=dict)
    
    # Metadata
    analysis_version = Column(String(20), default="1.0")
    calculated_by = Column(String(50))  # User or system who initiated analysis
    
    # Timestamps
    calculated_at = Column(DateTime(timezone=True), default=func.now(), index=True)
    
    # Indexes
    __table_args__ = (
        Index('idx_frequency_analytics_strategy_period', 'strategy_id', 'period_start'),
        Index('idx_frequency_analytics_period_type', 'period_type', 'period_start'),
        Index('idx_frequency_analytics_calculated', 'calculated_at'),
    )
