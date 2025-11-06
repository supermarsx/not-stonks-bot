"""
Execution Quality Analytics API Router
FastAPI endpoints for trade execution analysis and cost measurement
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from ..utils.execution_analytics import ExecutionAnalytics

router = APIRouter(prefix="/api/analytics/execution", tags=["Execution Analytics"])

# Pydantic models for request/response validation
class TradeExecutionRequest(BaseModel):
    decision_price: float = Field(..., description="Price when decision was made")
    arrival_price: float = Field(..., description="Price when order arrived at market")
    execution_price: float = Field(..., description="Actual execution price")
    trade_value: float = Field(..., description="Total trade value")
    commission: Optional[float] = Field(0.0, description="Commission paid")
    market_impact: Optional[float] = Field(0.0, description="Estimated market impact")

class VWAPAnalysisRequest(BaseModel):
    execution_price: float = Field(..., description="Actual execution price")
    vwap_price: float = Field(..., description="VWAP benchmark price")
    trade_volume: float = Field(..., description="Trade volume")
    total_volume: float = Field(..., description="Total market volume")
    side: str = Field("buy", description="Trade side: 'buy' or 'sell'")

class TWAPAnalysisRequest(BaseModel):
    execution_times: List[str] = Field(..., description="Execution timestamps")
    execution_prices: List[float] = Field(..., description="Execution prices")
    execution_volumes: List[float] = Field(..., description="Execution volumes")
    benchmark_start: str = Field(..., description="Benchmark period start")
    benchmark_end: str = Field(..., description="Benchmark period end")
    benchmark_prices: List[float] = Field(..., description="Benchmark prices")
    benchmark_times: List[str] = Field(..., description="Benchmark timestamps")

class MarketImpactRequest(BaseModel):
    pre_trade_price: float = Field(..., description="Price before trade")
    post_trade_price: float = Field(..., description="Price after trade")
    trade_size: float = Field(..., description="Trade size in shares")
    average_daily_volume: float = Field(..., description="Average daily volume")
    trade_duration_minutes: Optional[float] = Field(1.0, description="Trade duration in minutes")

class ExecutionScorecardRequest(BaseModel):
    trades: List[Dict[str, Any]] = Field(..., description="List of trade execution data")

class ImplementationShortfallResponse(BaseModel):
    decision_price: float
    arrival_price: float
    execution_price: float
    paper_return: float
    execution_return: float
    implementation_shortfall_bps: float
    market_impact_bps: float
    timing_cost_bps: float
    commission_cost_bps: float
    trade_value: float

class VWAPResponse(BaseModel):
    execution_price: float
    vwap_price: float
    vwap_difference: float
    vwap_difference_pct: float
    vwap_difference_bps: float
    volume_participation_pct: float
    performance_vs_vwap: str

class MarketImpactResponse(BaseModel):
    price_impact: float
    price_impact_pct: float
    price_impact_bps: float
    participation_rate_pct: float
    impact_per_share: float
    temporary_impact: float
    permanent_impact: float
    temporary_impact_bps: float
    permanent_impact_bps: float

@router.post("/implementation-shortfall", response_model=ImplementationShortfallResponse)
async def calculate_implementation_shortfall(request: TradeExecutionRequest):
    """
    Calculate Implementation Shortfall for a trade execution
    
    Measures the difference between paper portfolio return and actual return,
    decomposing into market impact, timing cost, and commission components.
    """
    try:
        result = ExecutionAnalytics.implementation_shortfall(
            request.decision_price,
            request.arrival_price,
            request.execution_price,
            request.trade_value,
            request.commission,
            request.market_impact
        )
        
        return ImplementationShortfallResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Implementation shortfall calculation failed: {str(e)}")

@router.post("/vwap-analysis", response_model=VWAPResponse)
async def analyze_vwap_performance(request: VWAPAnalysisRequest):
    """
    Analyze execution performance against VWAP benchmark
    
    Compares execution price to Volume Weighted Average Price and
    calculates performance metrics including basis point differences.
    """
    try:
        result = ExecutionAnalytics.vwap_analysis(
            request.execution_price,
            request.vwap_price,
            request.trade_volume,
            request.total_volume,
            request.side
        )
        
        return VWAPResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"VWAP analysis failed: {str(e)}")

@router.post("/twap-analysis")
async def analyze_twap_performance(request: TWAPAnalysisRequest):
    """
    Analyze execution performance against TWAP benchmark
    
    Compares execution against Time Weighted Average Price benchmark
    over the specified time period.
    """
    try:
        # Convert string timestamps to datetime objects
        execution_times = [datetime.fromisoformat(t) for t in request.execution_times]
        benchmark_times = [datetime.fromisoformat(t) for t in request.benchmark_times]
        benchmark_start = datetime.fromisoformat(request.benchmark_start)
        benchmark_end = datetime.fromisoformat(request.benchmark_end)
        
        result = ExecutionAnalytics.twap_analysis(
            execution_times,
            request.execution_prices,
            request.execution_volumes,
            benchmark_start,
            benchmark_end,
            request.benchmark_prices,
            benchmark_times
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"TWAP analysis failed: {str(e)}")

@router.post("/market-impact", response_model=MarketImpactResponse)
async def analyze_market_impact(request: MarketImpactRequest):
    """
    Analyze market impact from trading activity
    
    Measures price impact, participation rate, and estimates
    temporary vs permanent impact components.
    """
    try:
        result = ExecutionAnalytics.market_impact_analysis(
            request.pre_trade_price,
            request.post_trade_price,
            request.trade_size,
            request.average_daily_volume,
            request.trade_duration_minutes
        )
        
        return MarketImpactResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Market impact analysis failed: {str(e)}")

@router.post("/execution-scorecard")
async def generate_execution_scorecard(request: ExecutionScorecardRequest):
    """
    Generate comprehensive execution quality scorecard
    
    Analyzes multiple trades to provide aggregate execution quality metrics,
    cost analysis, and performance grading.
    """
    try:
        result = ExecutionAnalytics.execution_quality_scorecard(request.trades)
        
        return {
            "scorecard": result,
            "generated_at": datetime.now().isoformat(),
            "analysis_period": {
                "total_trades": len(request.trades),
                "first_trade": min(trade.get('timestamp', '2024-01-01') for trade in request.trades),
                "last_trade": max(trade.get('timestamp', '2024-01-01') for trade in request.trades)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Execution scorecard generation failed: {str(e)}")

@router.get("/demo-trades")
async def generate_demo_execution_data():
    """
    Generate demonstration trade execution data for testing
    
    Returns sample trade execution data with realistic market scenarios
    for testing execution analytics endpoints.
    """
    try:
        np.random.seed(42)  # For reproducible results
        
        # Generate sample trades
        demo_trades = []
        base_price = 100.0
        
        for i in range(20):
            # Random trade parameters
            decision_price = base_price + np.random.normal(0, 0.5)
            arrival_lag = np.random.exponential(0.02)  # Price movement during arrival
            arrival_price = decision_price + np.random.normal(0, 0.1) + arrival_lag
            
            execution_lag = np.random.exponential(0.01)  # Additional movement during execution
            execution_price = arrival_price + np.random.normal(0, 0.05) + execution_lag
            
            trade_value = np.random.uniform(10000, 100000)
            commission = trade_value * np.random.uniform(0.0001, 0.005)  # 1-50 bps commission
            
            # VWAP and volume data
            vwap = decision_price + np.random.normal(0, 0.08)
            trade_volume = np.random.uniform(100, 10000)
            total_volume = trade_volume * np.random.uniform(5, 50)
            
            side = np.random.choice(['buy', 'sell'])
            
            trade_data = {
                'trade_id': f'TRADE_{i+1:03d}',
                'decision_price': decision_price,
                'arrival_price': arrival_price,
                'execution_price': execution_price,
                'trade_value': trade_value,
                'commission': commission,
                'vwap': vwap,
                'volume': trade_volume,
                'total_volume': total_volume,
                'side': side,
                'timestamp': (datetime.now() - timedelta(days=i)).isoformat()
            }
            
            demo_trades.append(trade_data)
            base_price += np.random.normal(0, 0.2)  # Drift in base price
        
        return {
            "demo_trades": demo_trades,
            "description": "Sample trade execution data for testing",
            "stats": {
                "total_trades": len(demo_trades),
                "average_trade_value": np.mean([t['trade_value'] for t in demo_trades]),
                "average_commission_bps": np.mean([t['commission']/t['trade_value']*10000 for t in demo_trades]),
                "buy_trades": len([t for t in demo_trades if t['side'] == 'buy']),
                "sell_trades": len([t for t in demo_trades if t['side'] == 'sell'])
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Demo data generation failed: {str(e)}")

@router.post("/batch-analysis")
async def batch_execution_analysis(trades: List[Dict[str, Any]]):
    """
    Perform batch analysis on multiple trade executions
    
    Analyzes a batch of trades to provide comprehensive execution metrics
    including implementation shortfall, VWAP performance, and cost breakdown.
    """
    try:
        batch_results = {
            "implementation_shortfall": [],
            "vwap_analysis": [],
            "market_impact": [],
            "summary_stats": {}
        }
        
        for trade in trades:
            # Implementation shortfall for each trade
            if all(k in trade for k in ['decision_price', 'arrival_price', 'execution_price', 'trade_value']):
                is_result = ExecutionAnalytics.implementation_shortfall(
                    trade['decision_price'],
                    trade['arrival_price'],
                    trade['execution_price'],
                    trade['trade_value'],
                    trade.get('commission', 0)
                )
                batch_results["implementation_shortfall"].append({
                    "trade_id": trade.get('trade_id', 'unknown'),
                    **is_result
                })
            
            # VWAP analysis for each trade
            if all(k in trade for k in ['execution_price', 'vwap', 'volume', 'total_volume']):
                vwap_result = ExecutionAnalytics.vwap_analysis(
                    trade['execution_price'],
                    trade['vwap'],
                    trade['volume'],
                    trade['total_volume'],
                    trade.get('side', 'buy')
                )
                batch_results["vwap_analysis"].append({
                    "trade_id": trade.get('trade_id', 'unknown'),
                    **vwap_result
                })
        
        # Calculate summary statistics
        if batch_results["implementation_shortfall"]:
            shortfall_bps = [r['implementation_shortfall_bps'] for r in batch_results["implementation_shortfall"]]
            batch_results["summary_stats"]["avg_shortfall_bps"] = np.mean(shortfall_bps)
            batch_results["summary_stats"]["median_shortfall_bps"] = np.median(shortfall_bps)
            batch_results["summary_stats"]["best_execution_bps"] = min(shortfall_bps)
            batch_results["summary_stats"]["worst_execution_bps"] = max(shortfall_bps)
        
        if batch_results["vwap_analysis"]:
            vwap_diffs = [r['vwap_difference_bps'] for r in batch_results["vwap_analysis"]]
            batch_results["summary_stats"]["avg_vwap_performance_bps"] = np.mean(vwap_diffs)
            batch_results["summary_stats"]["vwap_outperformance_rate"] = (
                len([d for d in vwap_diffs if d < 0]) / len(vwap_diffs) * 100
            )
        
        batch_results["summary_stats"]["total_trades_analyzed"] = len(trades)
        batch_results["summary_stats"]["analysis_timestamp"] = datetime.now().isoformat()
        
        return batch_results
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch analysis failed: {str(e)}")

@router.get("/execution-benchmarks")
async def get_execution_benchmarks():
    """
    Get industry standard execution quality benchmarks
    
    Returns typical performance ranges for various execution metrics
    to help contextualize execution quality results.
    """
    return {
        "implementation_shortfall_benchmarks": {
            "excellent": "< 5 bps",
            "good": "5-15 bps", 
            "average": "15-30 bps",
            "poor": "> 30 bps"
        },
        "vwap_performance_benchmarks": {
            "excellent": "< 3 bps difference",
            "good": "3-8 bps difference",
            "average": "8-15 bps difference", 
            "poor": "> 15 bps difference"
        },
        "commission_benchmarks": {
            "excellent": "< 2 bps",
            "good": "2-5 bps",
            "average": "5-10 bps",
            "poor": "> 10 bps"
        },
        "market_impact_benchmarks": {
            "low_impact": "< 5 bps",
            "moderate_impact": "5-15 bps",
            "high_impact": "15-30 bps",
            "very_high_impact": "> 30 bps"
        },
        "participation_rate_guidelines": {
            "conservative": "< 5% of daily volume",
            "moderate": "5-15% of daily volume",
            "aggressive": "15-25% of daily volume",
            "very_aggressive": "> 25% of daily volume"
        }
    }