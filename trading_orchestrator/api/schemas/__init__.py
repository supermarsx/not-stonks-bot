"""
Pydantic schemas for strategy management API

Provides request/response models for:
- Strategy configuration
- Strategy performance
- Backtesting requests and responses
- Strategy comparison
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid


class StrategyCategory(str, Enum):
    """Strategy categories"""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    ARBITRAGE = "arbitrage"
    VOLATILITY = "volatility"
    NEWS_BASED = "news_based"
    AI_ML = "ai_ml"
    COMPOSITE = "composite"


class StrategyStatus(str, Enum):
    """Strategy status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    TESTING = "testing"
    ERROR = "error"


class SignalType(str, Enum):
    """Trading signal types"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


class RiskLevel(str, Enum):
    """Risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class StrategyFilter(BaseModel):
    """Strategy filtering parameters"""
    category: Optional[StrategyCategory] = None
    status: Optional[StrategyStatus] = None
    risk_level: Optional[RiskLevel] = None
    min_performance: Optional[float] = Field(None, ge=0, le=100)
    max_drawdown: Optional[float] = Field(None, ge=0, le=100)
    tags: Optional[List[str]] = None
    search: Optional[str] = Field(None, description="Search in name, description, tags")


class StrategyConfig(BaseModel):
    """Strategy configuration model"""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    category: StrategyCategory
    parameters: Dict[str, Any] = Field(default_factory=dict)
    risk_level: RiskLevel = RiskLevel.MEDIUM
    tags: List[str] = Field(default_factory=list)
    timeframe: str = Field(default="1h", description="Timeframe for strategy execution")
    symbols: List[str] = Field(default_factory=list, description="Target symbols/trading pairs")
    
    @validator('name')
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError("Name cannot be empty")
        return v.strip()
    
    @validator('symbols')
    def validate_symbols(cls, v):
        if v:
            for symbol in v:
                if not symbol.strip():
                    raise ValueError("Symbols cannot be empty")
        return [s.strip().upper() for s in v if s.strip()]


class StrategyResponse(BaseModel):
    """Strategy response model"""
    id: str
    name: str
    description: Optional[str]
    category: StrategyCategory
    status: StrategyStatus
    risk_level: RiskLevel
    tags: List[str]
    timeframe: str
    symbols: List[str]
    parameters: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    created_by: str
    performance: Optional['StrategyPerformance'] = None
    
    class Config:
        orm_mode = True


class StrategyCreateRequest(BaseModel):
    """Strategy creation request"""
    config: StrategyConfig
    auto_start: bool = False
    validate_parameters: bool = True


class StrategyUpdateRequest(BaseModel):
    """Strategy update request"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    parameters: Optional[Dict[str, Any]] = None
    risk_level: Optional[RiskLevel] = None
    tags: Optional[List[str]] = None
    timeframe: Optional[str] = None
    symbols: Optional[List[str]] = None
    status: Optional[StrategyStatus] = None
    
    @validator('name')
    def validate_name(cls, v):
        if v and not v.strip():
            raise ValueError("Name cannot be empty")
        return v.strip() if v else v


class StrategySignal(BaseModel):
    """Trading signal model"""
    id: str
    strategy_id: str
    symbol: str
    signal_type: SignalType
    confidence: float = Field(..., ge=0, le=1)
    price: Optional[float] = None
    quantity: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime
    expires_at: Optional[datetime] = None


class StrategyPerformance(BaseModel):
    """Strategy performance metrics"""
    total_return: float = Field(..., description="Total return percentage")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    sortino_ratio: Optional[float] = Field(None, description="Sortino ratio")
    calmar_ratio: Optional[float] = Field(None, description="Calmar ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown percentage")
    win_rate: float = Field(..., ge=0, le=1, description="Win rate percentage")
    profit_factor: Optional[float] = Field(None, description="Profit factor")
    total_trades: int = Field(..., ge=0, description="Total number of trades")
    winning_trades: int = Field(..., ge=0, description="Number of winning trades")
    losing_trades: int = Field(..., ge=0, description="Number of losing trades")
    average_win: Optional[float] = Field(None, description="Average winning trade")
    average_loss: Optional[float] = Field(None, description="Average losing trade")
    best_trade: Optional[float] = Field(None, description="Best single trade")
    worst_trade: Optional[float] = Field(None, description="Worst single trade")
    volatility: Optional[float] = Field(None, description="Strategy volatility")
    beta: Optional[float] = Field(None, description="Beta relative to market")
    alpha: Optional[float] = Field(None, description="Alpha relative to market")
    var_95: Optional[float] = Field(None, description="Value at Risk (95% confidence)")
    cvar_95: Optional[float] = Field(None, description="Conditional Value at Risk")
    last_updated: datetime
    
    class Config:
        orm_mode = True


class BacktestRequest(BaseModel):
    """Backtest request model"""
    strategy_id: str
    start_date: datetime
    end_date: datetime
    initial_capital: float = Field(..., gt=0)
    commission: float = Field(0.0, ge=0, le=0.01, description="Commission rate (0.1% = 0.001)")
    slippage: float = Field(0.0, ge=0, le=0.01, description="Slippage rate")
    symbols: Optional[List[str]] = None
    include_monte_carlo: bool = False
    monte_carlo_runs: int = Field(1000, ge=100, le=10000, description="Monte Carlo simulation runs")
    walk_forward: bool = False
    walk_forward_window: int = Field(252, ge=30, le=1000, description="Walk-forward window in days")
    benchmark_symbol: Optional[str] = None


class BacktestResponse(BaseModel):
    """Backtest response model"""
    id: str
    strategy_id: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    performance: StrategyPerformance
    trades: List[Dict[str, Any]] = Field(default_factory=list)
    equity_curve: List[Dict[str, Any]] = Field(default_factory=list)
    drawdown_periods: List[Dict[str, Any]] = Field(default_factory=list)
    monte_carlo: Optional[Dict[str, Any]] = None
    walk_forward_results: Optional[List[Dict[str, Any]]] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


class StrategyComparison(BaseModel):
    """Strategy comparison model"""
    strategies: List[StrategyResponse]
    comparison_metrics: Dict[str, Dict[str, Any]] = Field(description="Performance metrics for each strategy")
    ranking: List[Dict[str, Any]] = Field(description="Strategy ranking by performance")
    correlation_matrix: Optional[List[List[float]]] = None
    recommendation: Optional[str] = Field(None, description="AI recommendation for strategy selection")


class StrategyEnsemble(BaseModel):
    """Strategy ensemble configuration"""
    name: str
    description: Optional[str]
    strategy_weights: Dict[str, float] = Field(..., description="Strategy ID to weight mapping")
    rebalance_frequency: str = Field("daily", description="Rebalance frequency")
    risk_management: Optional[Dict[str, Any]] = None
    created_at: datetime


class APIResponse(BaseModel):
    """Generic API response model"""
    success: bool
    message: str
    data: Optional[Any] = None
    errors: Optional[List[str]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        orm_mode = True


class PaginationParams(BaseModel):
    """Pagination parameters"""
    page: int = Field(1, ge=1, description="Page number")
    size: int = Field(20, ge=1, le=100, description="Items per page")
    
    @property
    def offset(self) -> int:
        return (self.page - 1) * self.size


class PaginatedResponse(BaseModel):
    """Paginated response model"""
    items: List[Any]
    total: int
    page: int
    size: int
    pages: int
    has_next: bool
    has_prev: bool


class WebSocketMessage(BaseModel):
    """WebSocket message model"""
    type: str
    strategy_id: Optional[str] = None
    data: Any
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Update forward references
StrategyResponse.model_rebuild()
