"""
Risk Management Models
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, JSON, Float, Text, Enum as SQLEnum
from sqlalchemy.sql import func
from config.database import Base
from enum import Enum


class RiskEventType(str, Enum):
    POSITION_LIMIT_BREACH = "position_limit_breach"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    ORDER_LIMIT_BREACH = "order_limit_breach"
    CIRCUIT_BREAKER = "circuit_breaker"
    MARGIN_CALL = "margin_call"
    EXPOSURE_LIMIT = "exposure_limit"
    VOLATILITY_SPIKE = "volatility_spike"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskLimit(Base):
    """Risk limits and compliance rules"""
    __tablename__ = "risk_limits"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    
    # Limit details
    limit_type = Column(String(50), nullable=False, index=True)
    limit_name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    
    # Limit values
    limit_value = Column(Float, nullable=False)
    current_value = Column(Float, default=0.0, nullable=False)
    warning_threshold = Column(Float, nullable=True)  # e.g., 80% of limit
    
    # Scope
    scope = Column(String(50), default="global", nullable=False)  # global, per_symbol, per_strategy
    scope_target = Column(String(100), nullable=True)  # symbol or strategy ID if scoped
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    is_breached = Column(Boolean, default=False, nullable=False)
    
    # Actions on breach
    breach_action = Column(String(50), nullable=False)  # block, warn, liquidate, cancel_orders
    
    # Metadata
    config_metadata = Column(JSON, default=dict)
    
    # Timestamps
    last_checked_at = Column(DateTime(timezone=True), nullable=True)
    last_breached_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<RiskLimit(id={self.id}, type='{self.limit_type}', limit={self.limit_value}, current={self.current_value})>"


class RiskEvent(Base):
    """Risk events and circuit breaker triggers"""
    __tablename__ = "risk_events"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    
    # Event details
    event_type = Column(SQLEnum(RiskEventType), nullable=False, index=True)
    risk_level = Column(SQLEnum(RiskLevel), nullable=False, index=True)
    
    # Context
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    
    # Related entities
    related_order_id = Column(Integer, nullable=True)
    related_position_id = Column(Integer, nullable=True)
    related_limit_id = Column(Integer, nullable=True)
    
    # Event data
    trigger_value = Column(Float, nullable=True)
    limit_value = Column(Float, nullable=True)
    
    # Actions taken
    action_taken = Column(String(100), nullable=True)
    action_result = Column(Text, nullable=True)
    
    # Status
    is_resolved = Column(Boolean, default=False, nullable=False)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    resolution_notes = Column(Text, nullable=True)
    
    # Metadata
    config_metadata = Column(JSON, default=dict)
    
    # Timestamp
    occurred_at = Column(DateTime(timezone=True), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    def __repr__(self):
        return f"<RiskEvent(id={self.id}, type='{self.event_type}', level='{self.risk_level}')>"


class ComplianceRule(Base):
    """Compliance and regulatory rules"""
    __tablename__ = "compliance_rules"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Rule details
    rule_code = Column(String(50), unique=True, nullable=False, index=True)
    rule_name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    category = Column(String(50), nullable=False)  # regulatory, internal, exchange
    
    # Scope
    applies_to = Column(JSON, default=list)  # List of user_ids, symbols, or "all"
    
    # Rule configuration
    rule_config = Column(JSON, default=dict)
    validation_function = Column(String(255), nullable=True)  # Function name to call
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    enforcement_level = Column(String(20), default="strict", nullable=False)  # strict, warn, audit
    
    # Metadata
    config_metadata = Column(JSON, default=dict)
    
    # Timestamps
    effective_from = Column(DateTime(timezone=True), nullable=True)
    effective_until = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<ComplianceRule(id={self.id}, code='{self.rule_code}', name='{self.rule_name}')>"


class CircuitBreaker(Base):
    """Circuit breaker status for emergency trading halts"""
    __tablename__ = "circuit_breakers"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=True, index=True)  # Null for global breakers
    
    # Breaker details
    breaker_name = Column(String(100), nullable=False, index=True)
    breaker_type = Column(String(50), nullable=False)  # account, symbol, strategy, global
    scope_target = Column(String(100), nullable=True)
    
    # Status
    is_active = Column(Boolean, default=False, nullable=False, index=True)
    triggered_by = Column(String(100), nullable=True)  # user_id or system
    
    # Trigger conditions
    trigger_condition = Column(Text, nullable=True)
    trigger_value = Column(Float, nullable=True)
    
    # Actions
    halt_new_orders = Column(Boolean, default=True, nullable=False)
    cancel_existing_orders = Column(Boolean, default=False, nullable=False)
    close_positions = Column(Boolean, default=False, nullable=False)
    
    # Metadata
    config_metadata = Column(JSON, default=dict)
    
    # Timestamps
    triggered_at = Column(DateTime(timezone=True), nullable=True)
    cleared_at = Column(DateTime(timezone=True), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<CircuitBreaker(id={self.id}, name='{self.breaker_name}', active={self.is_active})>"
