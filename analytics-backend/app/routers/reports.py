"""
Reports API Router
FastAPI endpoints for generating and exporting comprehensive analytics reports
"""
from fastapi import APIRouter, HTTPException, File, UploadFile, Body
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import io
import json
from pathlib import Path

from ..utils.report_generator import ReportGenerator
from ..utils.performance_analytics import PerformanceAnalytics
from ..utils.execution_analytics import ExecutionAnalytics
from ..utils.risk_analytics import RiskAnalytics
from ..utils.portfolio_optimization import PortfolioOptimization

router = APIRouter(prefix="/api/analytics/reports", tags=["Analytics Reports"])

# Pydantic models for request/response validation
class PerformanceReportRequest(BaseModel):
    portfolio_name: str = Field(..., description="Portfolio identifier")
    returns: List[float] = Field(..., description="Portfolio returns")
    dates: List[str] = Field(..., description="Corresponding dates")
    benchmark_returns: Optional[List[float]] = Field(None, description="Benchmark returns")
    benchmark_name: Optional[str] = Field("Benchmark", description="Benchmark identifier")
    period: Optional[str] = Field("Custom", description="Reporting period")
    risk_free_rate: Optional[float] = Field(0.02, description="Risk-free rate")

class RiskReportRequest(BaseModel):
    portfolio_name: str = Field(..., description="Portfolio identifier")
    portfolio_weights: Dict[str, float] = Field(..., description="Portfolio weights")
    portfolio_returns: List[float] = Field(..., description="Portfolio returns")
    asset_returns: Optional[Dict[str, List[float]]] = Field(None, description="Asset returns")
    stress_scenarios: Optional[Dict[str, Dict[str, float]]] = Field(None, description="Stress scenarios")
    portfolio_value: Optional[float] = Field(1000000, description="Portfolio value")

class ExecutionReportRequest(BaseModel):
    portfolio_name: str = Field(..., description="Portfolio identifier")
    trades: List[Dict[str, Any]] = Field(..., description="Trade execution data")
    period: Optional[str] = Field("1M", description="Reporting period")
    benchmark_type: Optional[str] = Field("VWAP", description="Execution benchmark")

class OptimizationReportRequest(BaseModel):
    current_portfolio: Dict[str, float] = Field(..., description="Current allocation")
    optimization_results: Dict[str, Any] = Field(..., description="Optimization results")
    optimization_method: str = Field(..., description="Optimization method used")
    expected_returns: List[float] = Field(..., description="Expected returns")
    asset_names: List[str] = Field(..., description="Asset identifiers")

class ReportExportRequest(BaseModel):
    report_data: Dict[str, Any] = Field(..., description="Report data to export")
    format: str = Field("json", description="Export format: json, excel, pdf")
    filename: Optional[str] = Field(None, description="Custom filename")

@router.post("/performance")
async def generate_performance_report(request: PerformanceReportRequest):
    """
    Generate comprehensive performance analysis report
    
    Creates detailed performance report including risk-adjusted metrics,
    attribution analysis, and benchmark comparison.
    """
    try:
        # Convert to pandas Series
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
        
        # Generate performance data
        portfolio_data = PerformanceAnalytics.performance_summary_report(
            returns_series, benchmark_series, request.risk_free_rate
        )
        
        # Add portfolio metadata
        portfolio_data['name'] = request.portfolio_name
        portfolio_data['period'] = request.period
        portfolio_data['returns_series'] = {
            date.strftime('%Y-%m-%d'): value 
            for date, value in returns_series.items()
        }
        
        benchmark_data = None
        if benchmark_series is not None:
            benchmark_data = PerformanceAnalytics.performance_summary_report(
                benchmark_series, None, request.risk_free_rate
            )
            benchmark_data['name'] = request.benchmark_name
            benchmark_data['returns_series'] = {
                date.strftime('%Y-%m-%d'): value 
                for date, value in benchmark_series.items()
            }
        
        # Create report generator
        report_gen = ReportGenerator(theme="matrix")
        
        # Generate comprehensive report
        report = report_gen.generate_performance_report(
            portfolio_data, benchmark_data, request.period
        )
        
        return {
            "report": report,
            "generation_info": {
                "generated_at": datetime.now().isoformat(),
                "report_type": "Performance Analysis",
                "data_points": len(request.returns),
                "period_analyzed": f"{request.dates[0]} to {request.dates[-1]}" if request.dates else "Unknown",
                "benchmark_included": benchmark_data is not None
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Performance report generation failed: {str(e)}")

@router.post("/risk")
async def generate_risk_report(request: RiskReportRequest):
    """
    Generate comprehensive risk analysis report
    
    Creates detailed risk report including VaR analysis, stress testing,
    concentration metrics, and correlation analysis.
    """
    try:
        returns_series = pd.Series(request.portfolio_returns)
        
        # Risk analytics calculations
        risk_analytics = {}
        
        # VaR analysis
        try:
            var_results = RiskAnalytics.monte_carlo_var(returns_series)
            risk_analytics['var_analysis'] = var_results
        except Exception as e:
            risk_analytics['var_analysis'] = {"error": str(e)}
        
        # Tail risk analysis
        try:
            tail_risk = RiskAnalytics.tail_risk_analysis(returns_series)
            risk_analytics['tail_risk'] = tail_risk
        except Exception as e:
            risk_analytics['tail_risk'] = {"error": str(e)}
        
        # Concentration analysis
        try:
            asset_returns_series = None
            if request.asset_returns:
                asset_returns_series = {
                    asset: pd.Series(returns) 
                    for asset, returns in request.asset_returns.items()
                }
            
            concentration = RiskAnalytics.portfolio_concentration_risk(
                request.portfolio_weights, asset_returns_series
            )
            risk_analytics['concentration_analysis'] = concentration
        except Exception as e:
            risk_analytics['concentration_analysis'] = {"error": str(e)}
        
        # Correlation analysis (if asset returns provided)
        if request.asset_returns:
            try:
                correlation = RiskAnalytics.correlation_analysis(
                    {asset: pd.Series(returns) for asset, returns in request.asset_returns.items()}
                )
                risk_analytics['correlation_analysis'] = correlation
            except Exception as e:
                risk_analytics['correlation_analysis'] = {"error": str(e)}
        
        # Stress testing (if scenarios provided)
        stress_results = {}
        if request.stress_scenarios and request.asset_returns:
            try:
                stress_results = RiskAnalytics.stress_testing(
                    request.portfolio_weights,
                    {asset: pd.Series(returns) for asset, returns in request.asset_returns.items()},
                    request.stress_scenarios
                )
            except Exception as e:
                stress_results = {"error": str(e)}
        
        # Portfolio data for report
        portfolio_data = {
            'name': request.portfolio_name,
            'weights': request.portfolio_weights,
            'value': request.portfolio_value,
            'returns': request.portfolio_returns
        }
        
        # Create report generator
        report_gen = ReportGenerator(theme="matrix")
        
        # Generate risk report
        report = report_gen.generate_risk_report(
            portfolio_data, risk_analytics, stress_results
        )
        
        return {
            "report": report,
            "generation_info": {
                "generated_at": datetime.now().isoformat(),
                "report_type": "Risk Analysis",
                "portfolio_value": request.portfolio_value,
                "number_of_positions": len(request.portfolio_weights),
                "stress_scenarios_included": len(request.stress_scenarios) if request.stress_scenarios else 0,
                "data_points": len(request.portfolio_returns)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Risk report generation failed: {str(e)}")

@router.post("/execution")
async def generate_execution_report(request: ExecutionReportRequest):
    """
    Generate trade execution quality report
    
    Analyzes trade execution performance including implementation shortfall,
    VWAP analysis, cost breakdown, and execution recommendations.
    """
    try:
        # Generate execution analytics
        execution_analytics = ExecutionAnalytics.execution_quality_scorecard(request.trades)
        
        # Add venue and timing analysis
        venue_analysis = _analyze_execution_venues(request.trades)
        timing_analysis = _analyze_execution_timing(request.trades)
        
        execution_analytics['venue_analysis'] = venue_analysis
        execution_analytics['timing_analysis'] = timing_analysis
        
        # Create report generator
        report_gen = ReportGenerator(theme="matrix")
        
        # Generate execution report
        report = report_gen.generate_execution_report(
            request.trades, execution_analytics, request.period
        )
        
        return {
            "report": report,
            "generation_info": {
                "generated_at": datetime.now().isoformat(),
                "report_type": "Execution Quality Analysis",
                "total_trades": len(request.trades),
                "period": request.period,
                "benchmark": request.benchmark_type,
                "execution_grade": execution_analytics.get('execution_quality_grade', 'N/A')
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Execution report generation failed: {str(e)}")

@router.post("/optimization")
async def generate_optimization_report(request: OptimizationReportRequest):
    """
    Generate portfolio optimization analysis report
    
    Compares current vs optimized allocation, shows expected improvements,
    and provides implementation recommendations.
    """
    try:
        # Prepare optimized portfolio data
        optimized_portfolio = {
            'weights': {
                request.asset_names[i]: weight 
                for i, weight in enumerate(request.optimization_results.get('optimal_weights', []))
            },
            'expected_return': request.optimization_results.get('portfolio_return', 0),
            'volatility': request.optimization_results.get('portfolio_volatility', 0),
            'sharpe_ratio': request.optimization_results.get('sharpe_ratio', 0)
        }
        
        # Current portfolio metrics
        current_weights = np.array([request.current_portfolio.get(asset, 0) for asset in request.asset_names])
        expected_returns_array = np.array(request.expected_returns)
        
        current_return = np.dot(current_weights, expected_returns_array)
        current_portfolio_data = {
            'weights': request.current_portfolio,
            'expected_return': current_return,
            'name': 'Current Portfolio'
        }
        
        # Create report generator
        report_gen = ReportGenerator(theme="matrix")
        
        # Add optimization method info to results
        optimization_results = {
            **request.optimization_results,
            'method': request.optimization_method
        }
        
        # Generate optimization report
        report = report_gen.generate_optimization_report(
            current_portfolio_data, optimized_portfolio, optimization_results
        )
        
        return {
            "report": report,
            "generation_info": {
                "generated_at": datetime.now().isoformat(),
                "report_type": "Portfolio Optimization",
                "optimization_method": request.optimization_method,
                "number_of_assets": len(request.asset_names),
                "optimization_success": request.optimization_results.get('optimization_success', False)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Optimization report generation failed: {str(e)}")

@router.post("/export")
async def export_report(request: ReportExportRequest):
    """
    Export report to specified format (JSON, Excel, PDF)
    
    Takes generated report data and exports to downloadable file format.
    """
    try:
        # Generate filename if not provided
        if request.filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_type = request.report_data.get('metadata', {}).get('report_type', 'report')
            request.filename = f"{report_type.lower().replace(' ', '_')}_{timestamp}"
        
        # Create report generator for export
        report_gen = ReportGenerator(theme="matrix")
        
        # Export based on format
        if request.format.lower() == "json":
            export_path = f"/tmp/{request.filename}.json"
            result = report_gen.export_to_json(request.report_data, export_path)
            return {"export_result": result, "download_url": f"/api/analytics/reports/download/{request.filename}.json"}
            
        elif request.format.lower() == "excel":
            export_path = f"/tmp/{request.filename}.xlsx"
            result = report_gen.export_to_excel(request.report_data, export_path)
            return {"export_result": result, "download_url": f"/api/analytics/reports/download/{request.filename}.xlsx"}
            
        elif request.format.lower() == "pdf":
            export_path = f"/tmp/{request.filename}.pdf"
            result = report_gen.export_to_pdf(request.report_data, export_path)
            return {"export_result": result, "download_url": f"/api/analytics/reports/download/{request.filename}.pdf"}
            
        else:
            raise HTTPException(status_code=400, detail="Unsupported format. Use: json, excel, or pdf")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report export failed: {str(e)}")

@router.get("/download/{filename}")
async def download_report(filename: str):
    """
    Download exported report file
    
    Serves the exported report file for download.
    """
    try:
        file_path = Path(f"/tmp/{filename}")
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Determine media type based on extension
        if filename.endswith('.json'):
            media_type = 'application/json'
        elif filename.endswith('.xlsx'):
            media_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        elif filename.endswith('.pdf'):
            media_type = 'application/pdf'
        else:
            media_type = 'application/octet-stream'
        
        return FileResponse(
            path=file_path,
            media_type=media_type,
            filename=filename
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File download failed: {str(e)}")

@router.post("/batch-reports")
async def generate_batch_reports(
    reports_config: List[Dict[str, Any]] = Body(..., description="List of report configurations")
):
    """
    Generate multiple reports in batch
    
    Processes multiple report requests simultaneously and returns
    batch results with individual report data.
    """
    try:
        batch_results = []
        
        for i, config in enumerate(reports_config):
            try:
                report_type = config.get('type', 'performance')
                
                if report_type == 'performance':
                    request = PerformanceReportRequest(**config['data'])
                    result = await generate_performance_report(request)
                    
                elif report_type == 'risk':
                    request = RiskReportRequest(**config['data'])
                    result = await generate_risk_report(request)
                    
                elif report_type == 'execution':
                    request = ExecutionReportRequest(**config['data'])
                    result = await generate_execution_report(request)
                    
                elif report_type == 'optimization':
                    request = OptimizationReportRequest(**config['data'])
                    result = await generate_optimization_report(request)
                    
                else:
                    result = {"error": f"Unsupported report type: {report_type}"}
                
                batch_results.append({
                    "report_index": i,
                    "report_type": report_type,
                    "status": "success" if "error" not in result else "error",
                    "result": result
                })
                
            except Exception as e:
                batch_results.append({
                    "report_index": i,
                    "report_type": config.get('type', 'unknown'),
                    "status": "error",
                    "error": str(e)
                })
        
        return {
            "batch_results": batch_results,
            "summary": {
                "total_reports": len(reports_config),
                "successful": len([r for r in batch_results if r['status'] == 'success']),
                "failed": len([r for r in batch_results if r['status'] == 'error']),
                "generated_at": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch report generation failed: {str(e)}")

@router.get("/templates")
async def get_report_templates():
    """
    Get available report templates and configurations
    
    Returns predefined report templates that can be customized
    for common reporting scenarios.
    """
    return {
        "templates": {
            "monthly_performance": {
                "description": "Standard monthly performance review",
                "type": "performance",
                "required_fields": ["portfolio_name", "returns", "dates"],
                "optional_fields": ["benchmark_returns", "benchmark_name"],
                "default_config": {
                    "period": "1M",
                    "risk_free_rate": 0.02
                }
            },
            "quarterly_risk": {
                "description": "Comprehensive quarterly risk assessment",
                "type": "risk",
                "required_fields": ["portfolio_name", "portfolio_weights", "portfolio_returns"],
                "optional_fields": ["asset_returns", "stress_scenarios"],
                "default_config": {
                    "portfolio_value": 1000000
                }
            },
            "execution_review": {
                "description": "Trade execution quality review",
                "type": "execution",
                "required_fields": ["portfolio_name", "trades"],
                "optional_fields": ["period", "benchmark_type"],
                "default_config": {
                    "period": "1M",
                    "benchmark_type": "VWAP"
                }
            },
            "rebalancing_analysis": {
                "description": "Portfolio rebalancing and optimization analysis",
                "type": "optimization",
                "required_fields": ["current_portfolio", "optimization_results", "optimization_method"],
                "optional_fields": ["expected_returns", "asset_names"],
                "default_config": {
                    "optimization_method": "mean_variance"
                }
            }
        },
        "custom_sections": {
            "executive_summary": "High-level overview and key findings",
            "performance_metrics": "Detailed performance statistics",
            "risk_analysis": "Risk metrics and stress testing results",
            "attribution_analysis": "Performance attribution breakdown",
            "recommendations": "Actionable insights and suggestions",
            "appendix": "Supporting data and methodology"
        },
        "export_formats": ["json", "excel", "pdf"],
        "styling_options": ["matrix", "professional", "minimal"]
    }

# Helper functions
def _analyze_execution_venues(trades: List[Dict]) -> Dict[str, Any]:
    """Analyze execution venues from trade data"""
    
    venues = {}
    total_volume = 0
    
    for trade in trades:
        venue = trade.get('venue', 'Unknown')
        volume = trade.get('volume', 0)
        
        if venue not in venues:
            venues[venue] = {'volume': 0, 'trades': 0, 'total_value': 0}
        
        venues[venue]['volume'] += volume
        venues[venue]['trades'] += 1
        venues[venue]['total_value'] += trade.get('trade_value', 0)
        total_volume += volume
    
    # Calculate percentages
    for venue_data in venues.values():
        venue_data['volume_percentage'] = (venue_data['volume'] / total_volume * 100) if total_volume > 0 else 0
    
    return {
        'venue_breakdown': venues,
        'primary_venue': max(venues.keys(), key=lambda x: venues[x]['volume']) if venues else 'Unknown',
        'venue_concentration': max(venues[v]['volume_percentage'] for v in venues) if venues else 0
    }

def _analyze_execution_timing(trades: List[Dict]) -> Dict[str, Any]:
    """Analyze execution timing patterns"""
    
    if not trades:
        return {}
    
    # Extract execution times
    execution_times = []
    for trade in trades:
        if 'timestamp' in trade:
            try:
                timestamp = datetime.fromisoformat(trade['timestamp'])
                execution_times.append(timestamp.hour)
            except:
                continue
    
    if not execution_times:
        return {"error": "No valid timestamps found"}
    
    # Analyze timing patterns
    hour_distribution = {}
    for hour in execution_times:
        hour_distribution[hour] = hour_distribution.get(hour, 0) + 1
    
    peak_hour = max(hour_distribution.keys(), key=lambda x: hour_distribution[x])
    
    return {
        'hour_distribution': hour_distribution,
        'peak_execution_hour': peak_hour,
        'execution_spread': max(execution_times) - min(execution_times),
        'average_execution_hour': np.mean(execution_times)
    }