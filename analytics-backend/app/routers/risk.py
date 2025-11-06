"""
Risk Analytics API Router
FastAPI endpoints for advanced risk analysis and management
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from ..utils.risk_analytics import RiskAnalytics

router = APIRouter(prefix="/api/analytics/risk", tags=["Risk Analytics"])

# Pydantic models for request/response validation
class MonteCarloVaRRequest(BaseModel):
    returns: List[float] = Field(..., description="Historical portfolio returns")
    confidence_levels: Optional[List[float]] = Field([0.95, 0.99], description="VaR confidence levels")
    num_simulations: Optional[int] = Field(10000, description="Number of Monte Carlo simulations")
    time_horizon: Optional[int] = Field(1, description="Time horizon in days")

class ParametricVaRRequest(BaseModel):
    portfolio_value: float = Field(..., description="Total portfolio value")
    expected_return: float = Field(..., description="Expected daily return")
    volatility: float = Field(..., description="Daily volatility")
    confidence_levels: Optional[List[float]] = Field([0.95, 0.99], description="VaR confidence levels")
    time_horizon: Optional[int] = Field(1, description="Time horizon in days")

class StressTestRequest(BaseModel):
    portfolio_weights: Dict[str, float] = Field(..., description="Portfolio weights by asset")
    asset_returns: Dict[str, List[float]] = Field(..., description="Historical returns by asset")
    stress_scenarios: Dict[str, Dict[str, float]] = Field(..., description="Stress scenario shocks")

class CorrelationAnalysisRequest(BaseModel):
    returns_data: Dict[str, List[float]] = Field(..., description="Returns data by asset")
    dates: Optional[List[str]] = Field(None, description="Date indices for returns")

class TailRiskRequest(BaseModel):
    returns: List[float] = Field(..., description="Portfolio returns for tail analysis")
    tail_percentile: Optional[float] = Field(0.05, description="Tail percentile (default 5%)")

class ConcentrationRiskRequest(BaseModel):
    portfolio_weights: Dict[str, float] = Field(..., description="Portfolio weights by asset")
    asset_returns: Optional[Dict[str, List[float]]] = Field(None, description="Asset returns for risk contribution")

class VaRResponse(BaseModel):
    confidence_95: Dict[str, float]
    confidence_99: Dict[str, float]

class StressTestResponse(BaseModel):
    scenario_results: Dict[str, Dict[str, Any]]
    worst_case_scenario: str
    best_case_scenario: str
    summary_stats: Dict[str, float]

@router.post("/monte-carlo-var", response_model=VaRResponse)
async def calculate_monte_carlo_var(request: MonteCarloVaRRequest):
    """
    Calculate Value at Risk using Monte Carlo simulation
    
    Uses historical return distribution to simulate future portfolio scenarios
    and calculate VaR and CVaR at specified confidence levels.
    """
    try:
        returns_series = pd.Series(request.returns)
        
        var_results = RiskAnalytics.monte_carlo_var(
            returns_series,
            request.confidence_levels,
            request.num_simulations,
            request.time_horizon
        )
        
        return VaRResponse(
            confidence_95=var_results.get('95%', {}),
            confidence_99=var_results.get('99%', {})
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Monte Carlo VaR calculation failed: {str(e)}")

@router.post("/parametric-var")
async def calculate_parametric_var(request: ParametricVaRRequest):
    """
    Calculate parametric (analytical) VaR assuming normal distribution
    
    Fast VaR calculation based on portfolio statistics assuming 
    normal distribution of returns.
    """
    try:
        var_results = RiskAnalytics.parametric_var(
            request.portfolio_value,
            request.expected_return,
            request.volatility,
            request.confidence_levels,
            request.time_horizon
        )
        
        return {
            "parametric_var": var_results,
            "assumptions": {
                "distribution": "Normal",
                "portfolio_value": request.portfolio_value,
                "expected_return": request.expected_return,
                "volatility": request.volatility,
                "time_horizon": request.time_horizon
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Parametric VaR calculation failed: {str(e)}")

@router.post("/stress-testing", response_model=StressTestResponse)
async def perform_stress_testing(request: StressTestRequest):
    """
    Perform portfolio stress testing with predefined scenarios
    
    Analyzes portfolio performance under various stress scenarios
    including market crashes, sector shocks, and custom scenarios.
    """
    try:
        # Convert returns data to pandas Series
        asset_returns = {}
        for asset, returns in request.asset_returns.items():
            asset_returns[asset] = pd.Series(returns)
        
        stress_results = RiskAnalytics.stress_testing(
            request.portfolio_weights,
            asset_returns,
            request.stress_scenarios
        )
        
        # Find worst and best case scenarios
        scenario_returns = {name: result['total_scenario_return'] 
                          for name, result in stress_results.items()}
        
        worst_case = min(scenario_returns.keys(), key=lambda x: scenario_returns[x])
        best_case = max(scenario_returns.keys(), key=lambda x: scenario_returns[x])
        
        summary_stats = {
            "worst_case_return": scenario_returns[worst_case],
            "best_case_return": scenario_returns[best_case],
            "average_scenario_return": np.mean(list(scenario_returns.values())),
            "scenario_volatility": np.std(list(scenario_returns.values())),
            "number_of_scenarios": len(stress_results)
        }
        
        return StressTestResponse(
            scenario_results=stress_results,
            worst_case_scenario=worst_case,
            best_case_scenario=best_case,
            summary_stats=summary_stats
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Stress testing failed: {str(e)}")

@router.post("/correlation-analysis")
async def analyze_correlations(request: CorrelationAnalysisRequest):
    """
    Perform comprehensive correlation and diversification analysis
    
    Analyzes asset correlations, diversification ratios, and identifies
    concentration risks in portfolio correlations.
    """
    try:
        # Convert to pandas Series
        returns_data = {}
        for asset, returns in request.returns_data.items():
            returns_data[asset] = pd.Series(returns)
        
        correlation_results = RiskAnalytics.correlation_analysis(returns_data)
        
        return {
            "correlation_analysis": correlation_results,
            "generated_at": datetime.now().isoformat(),
            "analysis_summary": {
                "number_of_assets": len(request.returns_data),
                "analysis_period": len(list(request.returns_data.values())[0]) if request.returns_data else 0,
                "diversification_score": correlation_results['correlation_summary']['diversification_score']
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Correlation analysis failed: {str(e)}")

@router.post("/tail-risk-analysis")
async def analyze_tail_risk(request: TailRiskRequest):
    """
    Perform tail risk analysis including skewness, kurtosis, and extreme events
    
    Analyzes the distribution characteristics of returns focusing on
    tail behavior and extreme risk scenarios.
    """
    try:
        returns_series = pd.Series(request.returns)
        
        tail_risk_results = RiskAnalytics.tail_risk_analysis(
            returns_series,
            request.tail_percentile
        )
        
        return {
            "tail_risk_analysis": tail_risk_results,
            "generated_at": datetime.now().isoformat(),
            "analysis_parameters": {
                "tail_percentile": request.tail_percentile,
                "sample_size": len(request.returns),
                "analysis_period": f"{len(request.returns)} observations"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Tail risk analysis failed: {str(e)}")

@router.post("/concentration-risk")
async def analyze_concentration_risk(request: ConcentrationRiskRequest):
    """
    Analyze portfolio concentration and diversification risks
    
    Evaluates position concentration, calculates Herfindahl index,
    and analyzes risk contributions if returns data provided.
    """
    try:
        # Convert returns data if provided
        asset_returns = None
        if request.asset_returns:
            asset_returns = {}
            for asset, returns in request.asset_returns.items():
                asset_returns[asset] = pd.Series(returns)
        
        concentration_results = RiskAnalytics.portfolio_concentration_risk(
            request.portfolio_weights,
            asset_returns
        )
        
        return {
            "concentration_analysis": concentration_results,
            "generated_at": datetime.now().isoformat(),
            "portfolio_summary": {
                "number_of_positions": len(request.portfolio_weights),
                "largest_position": max(request.portfolio_weights.values()),
                "smallest_position": min(request.portfolio_weights.values()),
                "position_range": max(request.portfolio_weights.values()) - min(request.portfolio_weights.values())
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Concentration risk analysis failed: {str(e)}")

@router.post("/comprehensive-risk-report")
async def generate_comprehensive_risk_report(
    portfolio_weights: Dict[str, float],
    portfolio_returns: List[float],
    asset_returns: Optional[Dict[str, List[float]]] = None,
    portfolio_value: float = 1000000,
    stress_scenarios: Optional[Dict[str, Dict[str, float]]] = None
):
    """
    Generate comprehensive risk analysis report
    
    Combines all risk analytics into a single comprehensive report including
    VaR analysis, stress testing, correlation analysis, and concentration metrics.
    """
    try:
        report = {
            "metadata": {
                "report_type": "Comprehensive Risk Analysis",
                "generated_at": datetime.now().isoformat(),
                "portfolio_value": portfolio_value,
                "analysis_period": len(portfolio_returns)
            }
        }
        
        returns_series = pd.Series(portfolio_returns)
        
        # 1. VaR Analysis (Monte Carlo)
        try:
            mc_var = RiskAnalytics.monte_carlo_var(returns_series)
            report["var_analysis"] = mc_var
        except Exception as e:
            report["var_analysis"] = {"error": str(e)}
        
        # 2. Tail Risk Analysis
        try:
            tail_risk = RiskAnalytics.tail_risk_analysis(returns_series)
            report["tail_risk"] = tail_risk
        except Exception as e:
            report["tail_risk"] = {"error": str(e)}
        
        # 3. Concentration Analysis
        try:
            concentration = RiskAnalytics.portfolio_concentration_risk(
                portfolio_weights,
                {asset: pd.Series(returns) for asset, returns in asset_returns.items()} if asset_returns else None
            )
            report["concentration_risk"] = concentration
        except Exception as e:
            report["concentration_risk"] = {"error": str(e)}
        
        # 4. Correlation Analysis (if asset returns provided)
        if asset_returns:
            try:
                correlation = RiskAnalytics.correlation_analysis(
                    {asset: pd.Series(returns) for asset, returns in asset_returns.items()}
                )
                report["correlation_analysis"] = correlation
            except Exception as e:
                report["correlation_analysis"] = {"error": str(e)}
        
        # 5. Stress Testing (if scenarios provided)
        if stress_scenarios and asset_returns:
            try:
                stress_results = RiskAnalytics.stress_testing(
                    portfolio_weights,
                    {asset: pd.Series(returns) for asset, returns in asset_returns.items()},
                    stress_scenarios
                )
                report["stress_testing"] = stress_results
            except Exception as e:
                report["stress_testing"] = {"error": str(e)}
        
        # 6. Risk Summary
        report["risk_summary"] = {
            "overall_risk_level": _assess_overall_risk_level(report),
            "key_risk_factors": _identify_key_risks(report),
            "recommendations": _generate_risk_recommendations(report)
        }
        
        return report
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comprehensive risk report generation failed: {str(e)}")

@router.get("/default-stress-scenarios")
async def get_default_stress_scenarios():
    """
    Get predefined stress testing scenarios
    
    Returns common market stress scenarios for testing portfolio resilience
    including market crashes, sector rotations, and crisis scenarios.
    """
    return {
        "market_crash_scenarios": {
            "2008_financial_crisis": {
                "equities": -0.35,
                "bonds": -0.15,
                "real_estate": -0.45,
                "commodities": -0.25,
                "cash": 0.0
            },
            "covid_march_2020": {
                "equities": -0.30,
                "bonds": 0.05,
                "real_estate": -0.20,
                "commodities": -0.40,
                "cash": 0.0
            },
            "dot_com_crash": {
                "tech_stocks": -0.60,
                "growth_stocks": -0.45,
                "value_stocks": -0.15,
                "bonds": 0.10,
                "cash": 0.0
            }
        },
        "sector_rotation_scenarios": {
            "tech_selloff": {
                "technology": -0.25,
                "financials": 0.10,
                "healthcare": 0.05,
                "utilities": 0.15,
                "energy": -0.10
            },
            "inflation_spike": {
                "growth_stocks": -0.20,
                "value_stocks": 0.10,
                "commodities": 0.25,
                "real_estate": 0.15,
                "bonds": -0.15
            }
        },
        "geopolitical_scenarios": {
            "trade_war": {
                "exporters": -0.30,
                "domestic": 0.05,
                "commodities": -0.15,
                "safe_havens": 0.20
            },
            "oil_shock": {
                "energy": 0.40,
                "airlines": -0.25,
                "industrials": -0.15,
                "consumer_discretionary": -0.10
            }
        },
        "interest_rate_scenarios": {
            "rate_spike": {
                "bonds": -0.20,
                "utilities": -0.15,
                "reits": -0.25,
                "banks": 0.15,
                "tech": -0.10
            },
            "rate_cut": {
                "bonds": 0.10,
                "growth_stocks": 0.15,
                "banks": -0.10,
                "utilities": 0.05
            }
        }
    }

@router.get("/risk-limits")
async def get_risk_limits():
    """
    Get recommended risk limits and thresholds
    
    Returns industry standard risk limits for various risk metrics
    to help establish risk management frameworks.
    """
    return {
        "var_limits": {
            "conservative": "1% of portfolio value (95% VaR)",
            "moderate": "2% of portfolio value (95% VaR)",
            "aggressive": "3% of portfolio value (95% VaR)"
        },
        "concentration_limits": {
            "single_position": "5-10% maximum",
            "top_5_positions": "40% maximum",
            "single_sector": "25% maximum",
            "herfindahl_index": "< 0.25 for diversified portfolio"
        },
        "correlation_limits": {
            "average_correlation": "< 0.7 for diversification",
            "max_pairwise_correlation": "< 0.9",
            "minimum_diversification_ratio": "> 0.8"
        },
        "drawdown_limits": {
            "maximum_drawdown": "10-15% for conservative",
            "peak_to_trough": "20% maximum for moderate risk",
            "recovery_time": "< 12 months target"
        },
        "leverage_limits": {
            "gross_exposure": "150% for moderate leverage",
            "net_exposure": "100% base case",
            "sector_leverage": "200% maximum in any sector"
        }
    }

def _assess_overall_risk_level(report: Dict) -> str:
    """Assess overall portfolio risk level based on metrics"""
    
    risk_scores = []
    
    # VaR score
    if "var_analysis" in report and "95%" in report["var_analysis"]:
        var_95 = report["var_analysis"]["95%"].get("var_percentage", 0)
        if abs(var_95) < 2:
            risk_scores.append("Low")
        elif abs(var_95) < 5:
            risk_scores.append("Moderate")
        else:
            risk_scores.append("High")
    
    # Concentration score
    if "concentration_risk" in report:
        conc_metrics = report["concentration_risk"].get("concentration_metrics", {})
        largest_position = conc_metrics.get("largest_position_pct", 0)
        if largest_position < 10:
            risk_scores.append("Low")
        elif largest_position < 25:
            risk_scores.append("Moderate")
        else:
            risk_scores.append("High")
    
    # Determine overall level
    if not risk_scores:
        return "Unknown"
    
    high_count = risk_scores.count("High")
    moderate_count = risk_scores.count("Moderate")
    
    if high_count > len(risk_scores) / 2:
        return "High"
    elif moderate_count > 0:
        return "Moderate"
    else:
        return "Low"

def _identify_key_risks(report: Dict) -> List[str]:
    """Identify key risk factors from analysis"""
    
    risks = []
    
    # Check concentration risk
    if "concentration_risk" in report:
        conc_metrics = report["concentration_risk"].get("concentration_metrics", {})
        if conc_metrics.get("largest_position_pct", 0) > 20:
            risks.append("High position concentration")
    
    # Check tail risk
    if "tail_risk" in report:
        tail_metrics = report["tail_risk"]
        if tail_metrics.get("tail_risk_score", 0) > 50:
            risks.append("Significant tail risk")
    
    return risks if risks else ["No major risks identified"]

def _generate_risk_recommendations(report: Dict) -> List[str]:
    """Generate risk management recommendations"""
    
    recommendations = []
    
    # Based on concentration
    if "concentration_risk" in report:
        conc_assessment = report["concentration_risk"].get("diversification_assessment", {})
        if conc_assessment.get("concentration_warning", False):
            recommendations.append("Reduce position concentration through diversification")
    
    # Based on VaR
    if "var_analysis" in report:
        var_95 = report["var_analysis"].get("95%", {}).get("var_percentage", 0)
        if abs(var_95) > 5:
            recommendations.append("Consider reducing portfolio risk exposure")
    
    return recommendations if recommendations else ["Current risk profile appears appropriate"]