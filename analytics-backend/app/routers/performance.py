"""
Performance Analytics API Router
FastAPI endpoints for institutional-grade performance analysis
"""
from fastapi import APIRouter, HTTPException, Depends, Query, Body
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from ..utils.performance_analytics import PerformanceAnalytics
from ..utils.report_generator import ReportGenerator

router = APIRouter(prefix="/api/analytics/performance", tags=["Performance Analytics"])

# Pydantic models for request/response validation
class PortfolioReturnsRequest(BaseModel):
    returns: List[float] = Field(..., description="Daily portfolio returns")
    dates: List[str] = Field(..., description="Corresponding dates in YYYY-MM-DD format")
    risk_free_rate: Optional[float] = Field(0.02, description="Annual risk-free rate")
    benchmark_returns: Optional[List[float]] = Field(None, description="Benchmark returns for comparison")

class AttributionRequest(BaseModel):
    portfolio_weights: Dict[str, float] = Field(..., description="Portfolio weights by asset")
    benchmark_weights: Dict[str, float] = Field(..., description="Benchmark weights by asset")
    portfolio_returns: Dict[str, float] = Field(..., description="Portfolio asset returns")
    benchmark_returns: Dict[str, float] = Field(..., description="Benchmark asset returns")
    portfolio_return: float = Field(..., description="Total portfolio return")
    benchmark_return: float = Field(..., description="Total benchmark return")

class PerformanceMetricsResponse(BaseModel):
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    best_day: float
    worst_day: float

class AttributionResponse(BaseModel):
    total_attribution: Dict[str, float]
    asset_contributions: Dict[str, Dict[str, float]]
    summary: Dict[str, Any]

class VaRCVaRResponse(BaseModel):
    confidence_95: Dict[str, float]
    confidence_99: Dict[str, float]

@router.post("/metrics", response_model=PerformanceMetricsResponse)
async def calculate_performance_metrics(request: PortfolioReturnsRequest):
    """
    Calculate comprehensive performance metrics for a portfolio
    
    Returns risk-adjusted performance metrics including Sharpe, Sortino, Calmar ratios,
    maximum drawdown, win rate, and other key statistics.
    """
    try:
        # Convert to pandas Series
        returns_series = pd.Series(
            request.returns,
            index=pd.to_datetime(request.dates)
        )
        
        # Calculate metrics
        metrics = PerformanceAnalytics.risk_adjusted_metrics(
            returns_series, 
            request.risk_free_rate
        )
        
        return PerformanceMetricsResponse(**metrics)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Performance calculation failed: {str(e)}")

@router.post("/attribution", response_model=AttributionResponse)
async def brinson_fachler_attribution(request: AttributionRequest):
    """
    Perform Brinson-Fachler performance attribution analysis
    
    Decomposes portfolio outperformance into allocation effect, 
    security selection effect, and interaction effect.
    """
    try:
        attribution_result = PerformanceAnalytics.brinson_fachler_attribution(
            request.portfolio_weights,
            request.benchmark_weights,
            request.portfolio_returns,
            request.benchmark_returns,
            request.portfolio_return,
            request.benchmark_return
        )
        
        # Separate total from asset-level results
        total_attribution = attribution_result.pop('TOTAL')
        asset_contributions = attribution_result
        
        return AttributionResponse(
            total_attribution=total_attribution,
            asset_contributions=asset_contributions,
            summary={
                "total_allocation_effect": total_attribution['allocation_effect'],
                "total_selection_effect": total_attribution['selection_effect'],
                "total_interaction_effect": total_attribution['interaction_effect'],
                "active_return": total_attribution['active_return'],
                "explained_return": total_attribution['explained_active_return']
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Attribution analysis failed: {str(e)}")

@router.post("/var-cvar", response_model=VaRCVaRResponse)
async def calculate_var_cvar(request: PortfolioReturnsRequest):
    """
    Calculate Value at Risk (VaR) and Conditional Value at Risk (CVaR)
    
    Provides risk metrics at 95% and 99% confidence levels using historical method.
    """
    try:
        returns_series = pd.Series(request.returns)
        
        var_cvar_result = PerformanceAnalytics.calculate_var_cvar(
            returns_series,
            confidence_levels=[0.95, 0.99]
        )
        
        return VaRCVaRResponse(
            confidence_95=var_cvar_result['95%'],
            confidence_99=var_cvar_result['99%']
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"VaR/CVaR calculation failed: {str(e)}")

@router.post("/rolling-metrics")
async def calculate_rolling_metrics(
    request: PortfolioReturnsRequest,
    window_days: int = Query(252, description="Rolling window in days (default: 1 year)")
):
    """
    Calculate rolling performance metrics over time
    
    Provides time series of performance metrics using a rolling window approach.
    """
    try:
        returns_series = pd.Series(
            request.returns,
            index=pd.to_datetime(request.dates)
        )
        
        rolling_metrics = PerformanceAnalytics.rolling_performance_metrics(
            returns_series,
            window=window_days,
            risk_free_rate=request.risk_free_rate
        )
        
        # Convert to JSON-serializable format
        result = rolling_metrics.reset_index().to_dict('records')
        
        return {
            "rolling_metrics": result,
            "window_days": window_days,
            "data_points": len(result)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Rolling metrics calculation failed: {str(e)}")

@router.post("/summary-report")
async def generate_performance_summary(request: PortfolioReturnsRequest):
    """
    Generate comprehensive performance summary report
    
    Combines all performance metrics, risk analysis, and benchmark comparison
    into a single comprehensive report.
    """
    try:
        returns_series = pd.Series(
            request.returns,
            index=pd.to_datetime(request.dates)
        )
        
        # Benchmark comparison if provided
        benchmark_series = None
        if request.benchmark_returns:
            benchmark_series = pd.Series(
                request.benchmark_returns,
                index=pd.to_datetime(request.dates)
            )
        
        summary_report = PerformanceAnalytics.performance_summary_report(
            returns_series,
            benchmark_series,
            request.risk_free_rate
        )
        
        return {
            "report": summary_report,
            "generated_at": datetime.now().isoformat(),
            "period_analyzed": {
                "start_date": request.dates[0] if request.dates else None,
                "end_date": request.dates[-1] if request.dates else None,
                "total_days": len(request.dates)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Summary report generation failed: {str(e)}")

@router.get("/demo-data")
async def get_demo_performance_data():
    """
    Generate demonstration performance data for testing
    
    Returns sample portfolio and benchmark data for testing analytics endpoints.
    """
    try:
        # Generate synthetic performance data
        np.random.seed(42)  # For reproducible results
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        
        # Portfolio returns (slightly outperforming with higher volatility)
        portfolio_returns = np.random.normal(0.0008, 0.015, len(dates))  # ~20% annual return, 23% volatility
        
        # Benchmark returns (market-like)
        benchmark_returns = np.random.normal(0.0005, 0.012, len(dates))  # ~13% annual return, 19% volatility
        
        # Add some correlation
        for i in range(1, len(portfolio_returns)):
            portfolio_returns[i] += 0.3 * benchmark_returns[i-1]
            benchmark_returns[i] += 0.2 * benchmark_returns[i-1]
        
        return {
            "dates": [d.strftime('%Y-%m-%d') for d in dates],
            "portfolio_returns": portfolio_returns.tolist(),
            "benchmark_returns": benchmark_returns.tolist(),
            "description": "Synthetic daily returns for 1 year period",
            "stats": {
                "portfolio_annual_return": np.mean(portfolio_returns) * 252,
                "portfolio_annual_volatility": np.std(portfolio_returns) * np.sqrt(252),
                "benchmark_annual_return": np.mean(benchmark_returns) * 252,
                "benchmark_annual_volatility": np.std(benchmark_returns) * np.sqrt(252),
                "correlation": np.corrcoef(portfolio_returns, benchmark_returns)[0, 1]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Demo data generation failed: {str(e)}")

@router.post("/export-report")
async def export_performance_report(
    request: PortfolioReturnsRequest,
    format: str = Query("json", description="Export format: json, excel, pdf"),
    filename: Optional[str] = Query(None, description="Custom filename")
):
    """
    Export performance analysis to various formats
    
    Generates comprehensive performance report and exports to specified format.
    """
    try:
        # Generate report data
        returns_series = pd.Series(
            request.returns,
            index=pd.to_datetime(request.dates)
        )
        
        benchmark_series = None
        if request.benchmark_returns:
            benchmark_series = pd.Series(
                request.benchmark_returns,
                index=pd.to_datetime(request.dates)
            )
        
        # Create report generator
        report_gen = ReportGenerator(theme="matrix")
        
        # Prepare portfolio data for report
        portfolio_data = PerformanceAnalytics.performance_summary_report(
            returns_series, benchmark_series, request.risk_free_rate
        )
        
        benchmark_data = None
        if benchmark_series is not None:
            benchmark_data = PerformanceAnalytics.performance_summary_report(
                benchmark_series, None, request.risk_free_rate
            )
        
        # Generate report
        report_data = report_gen.generate_performance_report(
            portfolio_data, benchmark_data, "Custom Period"
        )
        
        # Export based on format
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"performance_report_{timestamp}"
        
        if format.lower() == "json":
            export_path = f"{filename}.json"
            result = report_gen.export_to_json(report_data, export_path)
        elif format.lower() == "excel":
            export_path = f"{filename}.xlsx"
            result = report_gen.export_to_excel(report_data, export_path)
        elif format.lower() == "pdf":
            export_path = f"{filename}.pdf"
            result = report_gen.export_to_pdf(report_data, export_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported format. Use: json, excel, or pdf")
        
        return {
            "export_result": result,
            "filename": export_path,
            "format": format,
            "report_metadata": report_data["metadata"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report export failed: {str(e)}")