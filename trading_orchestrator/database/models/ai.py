"""
AI/LLM System Models
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, JSON, Float, Text, Enum as SQLEnum
from sqlalchemy.sql import func
from config.database import Base
from enum import Enum


class ModelTier(str, Enum):
    FAST = "fast"  # Quick queries
    MEDIUM = "medium"  # Standard analysis
    ADVANCED = "advanced"  # Complex reasoning
    SPECIALIZED = "specialized"  # Domain-specific


class ToolCallStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


class AIModel(Base):
    """AI/LLM model configurations"""
    __tablename__ = "ai_models"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Model identification
    model_name = Column(String(100), nullable=False, index=True)
    model_id = Column(String(100), nullable=False, index=True)  # e.g., gpt-4, claude-3
    provider = Column(String(50), nullable=False)  # openai, anthropic, etc.
    
    # Classification
    tier = Column(SQLEnum(ModelTier), nullable=False)
    category = Column(String(50), nullable=False)  # general, trading, analysis
    
    # Capabilities
    max_tokens = Column(Integer, nullable=False)
    supports_function_calling = Column(Boolean, default=False)
    supports_streaming = Column(Boolean, default=False)
    supports_vision = Column(Boolean, default=False)
    
    # Performance metrics
    avg_latency_ms = Column(Float, nullable=True)
    avg_cost_per_1k_tokens = Column(Float, nullable=True)
    
    # Configuration
    default_temperature = Column(Float, default=0.7)
    default_max_tokens = Column(Integer, nullable=True)
    model_config = Column(JSON, default=dict)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    is_available = Column(Boolean, default=True, nullable=False)
    
    # Metadata
    config_metadata = Column(JSON, default=dict)
    
    # Timestamps
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<AIModel(id={self.id}, name='{self.model_name}', provider='{self.provider}', tier='{self.tier}')>"


class AIPrompt(Base):
    """Prompt templates for AI interactions"""
    __tablename__ = "ai_prompts"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Prompt identification
    prompt_name = Column(String(100), unique=True, nullable=False, index=True)
    prompt_type = Column(String(50), nullable=False)  # system, user, function
    category = Column(String(50), nullable=False, index=True)  # trading, analysis, risk
    
    # Template
    template = Column(Text, nullable=False)
    variables = Column(JSON, default=list)  # List of expected variables
    
    # Usage
    version = Column(String(20), default="1.0.0", nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Metadata
    description = Column(Text, nullable=True)
    config_metadata = Column(JSON, default=dict)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<AIPrompt(id={self.id}, name='{self.prompt_name}', type='{self.prompt_type}')>"


class AIToolCall(Base):
    """AI tool/function call logs"""
    __tablename__ = "ai_tool_calls"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), nullable=False, index=True)
    user_id = Column(Integer, nullable=True, index=True)
    
    # Model context
    model_id = Column(Integer, nullable=True)
    model_name = Column(String(100), nullable=True)
    
    # Tool details
    tool_name = Column(String(100), nullable=False, index=True)
    tool_function = Column(String(255), nullable=False)
    
    # Call details
    input_parameters = Column(JSON, default=dict)
    output_result = Column(JSON, default=dict)
    error_message = Column(Text, nullable=True)
    
    # Status
    status = Column(SQLEnum(ToolCallStatus), nullable=False, index=True)
    
    # Performance
    latency_ms = Column(Float, nullable=True)
    tokens_used = Column(Integer, nullable=True)
    
    # Metadata
    config_metadata = Column(JSON, default=dict)
    
    # Timestamps
    called_at = Column(DateTime(timezone=True), nullable=False, index=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    def __repr__(self):
        return f"<AIToolCall(id={self.id}, tool='{self.tool_name}', status='{self.status}')>"


class TradingStrategy(Base):
    """AI-driven trading strategies"""
    __tablename__ = "trading_strategies"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    
    # Strategy identification
    strategy_name = Column(String(100), nullable=False, index=True)
    strategy_type = Column(String(50), nullable=False)  # trend, mean_reversion, arbitrage, etc.
    
    # Configuration
    symbols = Column(JSON, default=list)
    timeframes = Column(JSON, default=list)
    parameters = Column(JSON, default=dict)
    
    # AI model
    ai_model_id = Column(Integer, nullable=True)
    uses_ai = Column(Boolean, default=False, nullable=False)
    
    # Risk parameters
    max_position_size = Column(Float, nullable=True)
    stop_loss_percent = Column(Float, nullable=True)
    take_profit_percent = Column(Float, nullable=True)
    
    # Status
    is_active = Column(Boolean, default=False, nullable=False)
    is_backtested = Column(Boolean, default=False, nullable=False)
    
    # Performance metrics
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    total_pnl = Column(Float, default=0.0)
    win_rate = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    
    # Metadata
    description = Column(Text, nullable=True)
    config_metadata = Column(JSON, default=dict)
    
    # Timestamps
    last_executed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<TradingStrategy(id={self.id}, name='{self.strategy_name}', type='{self.strategy_type}', active={self.is_active})>"


class BacktestResult(Base):
    """Backtesting results for strategies"""
    __tablename__ = "backtest_results"
    
    id = Column(Integer, primary_key=True, index=True)
    strategy_id = Column(Integer, nullable=False, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    
    # Test parameters
    start_date = Column(DateTime(timezone=True), nullable=False)
    end_date = Column(DateTime(timezone=True), nullable=False)
    initial_capital = Column(Float, nullable=False)
    
    # Performance metrics
    final_capital = Column(Float, nullable=False)
    total_return = Column(Float, nullable=False)
    total_return_percent = Column(Float, nullable=False)
    sharpe_ratio = Column(Float, nullable=True)
    sortino_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    max_drawdown_percent = Column(Float, nullable=True)
    
    # Trade statistics
    total_trades = Column(Integer, nullable=False)
    winning_trades = Column(Integer, nullable=False)
    losing_trades = Column(Integer, nullable=False)
    win_rate = Column(Float, nullable=False)
    avg_win = Column(Float, nullable=True)
    avg_loss = Column(Float, nullable=True)
    profit_factor = Column(Float, nullable=True)
    
    # Detailed results
    equity_curve = Column(JSON, default=list)
    trade_log = Column(JSON, default=list)
    
    # Metadata
    config_metadata = Column(JSON, default=dict)
    
    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    def __repr__(self):
        return f"<BacktestResult(id={self.id}, strategy_id={self.strategy_id}, return={self.total_return_percent}%)>"
