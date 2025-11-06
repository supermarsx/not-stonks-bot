"""
Common Data Models
Shared Pydantic models for analytics backend API
"""
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum

# Enums for standardized values
class OptimizationMethod(str, Enum):
    MEAN_VARIANCE = "mean_variance"
    BLACK_LITTERMAN = "black_litterman"
    RISK_PARITY = "risk_parity"
    MINIMUM_VARIANCE = "minimum_variance"

class ReportFormat(str, Enum):
    JSON = "json"
    EXCEL = "excel"
    PDF = "pdf"

class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"  
    HIGH = "high"
    VERY_HIGH = "very_high"

class ExecutionSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

# Base response model
class BaseResponse(BaseModel):
    success: bool = True
    timestamp: datetime = Field(default_factory=datetime.now)
    message: Optional[str] = None

# Asset and Portfolio Models
class Asset(BaseModel):
    symbol: str = Field(..., description="Asset symbol/identifier")
    name: Optional[str] = Field(None, description="Asset name")
    asset_class: Optional[str] = Field(None, description="Asset class (equity, bond, etc.)")
    sector: Optional[str] = Field(None, description="Sector classification")
    weight: Optional[float] = Field(None, ge=0, le=1, description="Portfolio weight")

class Portfolio(BaseModel):
    name: str = Field(..., description="Portfolio identifier")
    value: float = Field(..., gt=0, description="Total portfolio value")
    assets: List[Asset] = Field(..., description="Portfolio assets")
    benchmark: Optional[str] = Field(None, description="Benchmark identifier")
    created_at: datetime = Field(default_factory=datetime.now)

    @validator('assets')
    def validate_weights_sum_to_one(cls, v):
        """Validate that weights sum to approximately 1.0"""
        weights = [asset.weight for asset in v if asset.weight is not None]
        if weights and abs(sum(weights) - 1.0) > 0.01:
            raise ValueError("Asset weights must sum to approximately 1.0")
        return v

# Performance Models
class PerformanceMetrics(BaseModel):
    total_return: float = Field(..., description="Total return")
    annual_return: float = Field(..., description="Annualized return")
    volatility: float = Field(..., description="Annualized volatility")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    sortino_ratio: float = Field(..., description="Sortino ratio")
    calmar_ratio: float = Field(..., description="Calmar ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown")
    win_rate: float = Field(..., ge=0, le=1, description="Win rate")

class AttributionAnalysis(BaseModel):
    allocation_effect: float = Field(..., description="Asset allocation effect")
    selection_effect: float = Field(..., description="Security selection effect")  
    interaction_effect: float = Field(..., description="Interaction effect")
    total_effect: float = Field(..., description="Total attribution effect")

# Risk Models
class VaRMetrics(BaseModel):
    confidence_level: float = Field(..., ge=0.5, le=0.999, description="Confidence level")
    var_daily: float = Field(..., description="Daily VaR")
    var_annual: float = Field(..., description="Annualized VaR")
    cvar_daily: float = Field(..., description="Daily CVaR")
    cvar_annual: float = Field(..., description="Annualized CVaR")

class StressScenario(BaseModel):
    name: str = Field(..., description="Scenario name")
    description: Optional[str] = Field(None, description="Scenario description")
    asset_shocks: Dict[str, float] = Field(..., description="Asset return shocks")
    portfolio_impact: Optional[float] = Field(None, description="Portfolio impact")

class ConcentrationMetrics(BaseModel):
    largest_position_pct: float = Field(..., description="Largest position percentage")
    top_5_positions_pct: float = Field(..., description="Top 5 positions percentage")
    herfindahl_index: float = Field(..., description="Herfindahl-Hirschman Index")
    effective_positions: float = Field(..., description="Effective number of positions")
    risk_level: RiskLevel = Field(..., description="Concentration risk level")

# Execution Models  
class TradeExecution(BaseModel):
    trade_id: str = Field(..., description="Trade identifier")
    symbol: str = Field(..., description="Asset symbol")
    side: ExecutionSide = Field(..., description="Trade side")
    quantity: float = Field(..., gt=0, description="Trade quantity")
    decision_price: float = Field(..., gt=0, description="Decision price")
    arrival_price: float = Field(..., gt=0, description="Arrival price")
    execution_price: float = Field(..., gt=0, description="Execution price")
    commission: float = Field(0, ge=0, description="Commission cost")
    timestamp: datetime = Field(..., description="Execution timestamp")
    venue: Optional[str] = Field(None, description="Execution venue")

class ExecutionMetrics(BaseModel):
    implementation_shortfall_bps: float = Field(..., description="Implementation shortfall in bps")
    market_impact_bps: float = Field(..., description="Market impact in bps")
    timing_cost_bps: float = Field(..., description="Timing cost in bps")
    commission_cost_bps: float = Field(..., description="Commission cost in bps")
    vwap_difference_bps: float = Field(..., description="VWAP difference in bps")
    execution_grade: str = Field(..., description="Execution quality grade")

# Optimization Models
class OptimizationConstraints(BaseModel):
    min_weight: float = Field(0.0, ge=0, le=1, description="Minimum asset weight")
    max_weight: float = Field(1.0, ge=0, le=1, description="Maximum asset weight")
    max_turnover: Optional[float] = Field(None, ge=0, description="Maximum turnover")
    sector_limits: Optional[Dict[str, float]] = Field(None, description="Sector weight limits")
    
    @validator('max_weight')
    def max_weight_greater_than_min(cls, v, values):
        """Validate max weight is greater than min weight"""
        if 'min_weight' in values and v <= values['min_weight']:
            raise ValueError("max_weight must be greater than min_weight")
        return v

class OptimizationResult(BaseModel):
    method: OptimizationMethod = Field(..., description="Optimization method used")
    optimal_weights: Dict[str, float] = Field(..., description="Optimal asset weights")
    expected_return: float = Field(..., description="Expected portfolio return")
    expected_volatility: float = Field(..., description="Expected portfolio volatility")
    sharpe_ratio: float = Field(..., description="Expected Sharpe ratio")
    optimization_success: bool = Field(..., description="Optimization success flag")
    objective_value: Optional[float] = Field(None, description="Objective function value")

# Reporting Models
class ReportMetadata(BaseModel):
    report_type: str = Field(..., description="Type of report")
    portfolio_name: str = Field(..., description="Portfolio name")
    period: str = Field(..., description="Reporting period")
    generated_at: datetime = Field(default_factory=datetime.now)
    generated_by: Optional[str] = Field(None, description="Report generator")

class ReportSection(BaseModel):
    title: str = Field(..., description="Section title")
    content: Dict[str, Any] = Field(..., description="Section content")
    charts: Optional[List[str]] = Field(None, description="Chart filenames")
    tables: Optional[List[Dict[str, Any]]] = Field(None, description="Table data")

class AnalyticsReport(BaseModel):
    metadata: ReportMetadata = Field(..., description="Report metadata")
    executive_summary: Dict[str, str] = Field(..., description="Executive summary")
    sections: List[ReportSection] = Field(..., description="Report sections")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
    export_formats: List[ReportFormat] = Field(default_factory=list, description="Available export formats")

# Request/Response wrapper models
class AnalyticsRequest(BaseModel):
    request_id: Optional[str] = Field(None, description="Request identifier")
    timestamp: datetime = Field(default_factory=datetime.now)

class AnalyticsResponse(BaseResponse):
    data: Optional[Any] = Field(None, description="Response data")
    request_id: Optional[str] = Field(None, description="Request identifier")

# Batch processing models
class BatchRequest(BaseModel):
    requests: List[Dict[str, Any]] = Field(..., description="Batch of requests")
    batch_id: Optional[str] = Field(None, description="Batch identifier")

class BatchResponse(BaseResponse):
    batch_id: Optional[str] = Field(None, description="Batch identifier")
    results: List[AnalyticsResponse] = Field(..., description="Batch results")
    summary: Dict[str, Any] = Field(..., description="Batch summary")

# Error models
class AnalyticsError(BaseModel):
    error_code: str = Field(..., description="Error code")
    error_message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.now)