"""
Portfolio Optimization API Router
FastAPI endpoints for advanced portfolio optimization and allocation strategies
"""
from fastapi import APIRouter, HTTPException, Depends, Body
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from ..utils.portfolio_optimization import PortfolioOptimization

router = APIRouter(prefix="/api/analytics/optimization", tags=["Portfolio Optimization"])

# Pydantic models for request/response validation
class MeanVarianceRequest(BaseModel):
    expected_returns: List[float] = Field(..., description="Expected returns for each asset")
    covariance_matrix: List[List[float]] = Field(..., description="Covariance matrix (n x n)")
    asset_names: List[str] = Field(..., description="Asset identifiers")
    risk_aversion: Optional[float] = Field(1.0, description="Risk aversion parameter")
    weight_bounds: Optional[Tuple[float, float]] = Field((0.0, 1.0), description="Weight bounds (min, max)")
    target_return: Optional[float] = Field(None, description="Target return for minimum variance optimization")

class BlackLittermanRequest(BaseModel):
    market_caps: List[float] = Field(..., description="Market capitalizations")
    returns_data: Dict[str, List[float]] = Field(..., description="Historical returns by asset")
    asset_names: List[str] = Field(..., description="Asset identifiers")
    investor_views: Optional[Dict[str, float]] = Field(None, description="Investor views on expected returns")
    view_confidence: Optional[float] = Field(0.25, description="Confidence in investor views")
    risk_aversion: Optional[float] = Field(3.0, description="Risk aversion parameter")
    tau: Optional[float] = Field(0.025, description="Scaling factor for prior uncertainty")

class EfficientFrontierRequest(BaseModel):
    expected_returns: List[float] = Field(..., description="Expected returns for each asset")
    covariance_matrix: List[List[float]] = Field(..., description="Covariance matrix (n x n)")
    asset_names: List[str] = Field(..., description="Asset identifiers")
    num_portfolios: Optional[int] = Field(100, description="Number of portfolios on frontier")
    weight_bounds: Optional[Tuple[float, float]] = Field((0.0, 1.0), description="Weight bounds")

class RiskParityRequest(BaseModel):
    covariance_matrix: List[List[float]] = Field(..., description="Covariance matrix (n x n)")
    asset_names: List[str] = Field(..., description="Asset identifiers")
    target_volatility: Optional[float] = Field(None, description="Target portfolio volatility")

class TacticalAllocationRequest(BaseModel):
    current_weights: Dict[str, float] = Field(..., description="Current portfolio weights")
    target_weights: Dict[str, float] = Field(..., description="Strategic target weights")
    returns_momentum: Dict[str, float] = Field(..., description="Recent momentum indicators")
    volatility_regime: Dict[str, float] = Field(..., description="Volatility regime indicators")
    rebalance_threshold: Optional[float] = Field(0.05, description="Rebalancing threshold")

class OptimizationResponse(BaseModel):
    optimal_weights: List[float]
    portfolio_return: float
    portfolio_volatility: float
    sharpe_ratio: float
    optimization_success: bool

class EfficientFrontierResponse(BaseModel):
    efficient_portfolios: List[Dict[str, Any]]
    min_variance_portfolio: Dict[str, Any]
    max_sharpe_portfolio: Dict[str, Any]
    max_return_portfolio: Dict[str, Any]
    frontier_statistics: Dict[str, Any]

@router.post("/mean-variance", response_model=OptimizationResponse)
async def optimize_mean_variance(request: MeanVarianceRequest):
    """
    Perform Mean-Variance (Markowitz) portfolio optimization
    
    Finds optimal portfolio weights that maximize utility (return - risk penalty)
    or minimize variance for a target return level.
    """
    try:
        expected_returns = np.array(request.expected_returns)
        covariance_matrix = np.array(request.covariance_matrix)
        
        # Validate dimensions
        if len(expected_returns) != len(request.asset_names):
            raise ValueError("Expected returns length must match number of assets")
        
        if covariance_matrix.shape != (len(request.asset_names), len(request.asset_names)):
            raise ValueError("Covariance matrix dimensions must match number of assets")
        
        # Perform optimization
        result = PortfolioOptimization.mean_variance_optimization(
            expected_returns,
            covariance_matrix,
            request.risk_aversion,
            request.weight_bounds,
            request.target_return
        )
        
        if result['optimization_success']:
            return OptimizationResponse(
                optimal_weights=result['optimal_weights'],
                portfolio_return=result['portfolio_return'],
                portfolio_volatility=result['portfolio_volatility'],
                sharpe_ratio=result['sharpe_ratio'],
                optimization_success=True
            )
        else:
            raise HTTPException(status_code=400, detail=f"Optimization failed: {result.get('error_message', 'Unknown error')}")
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Mean-variance optimization failed: {str(e)}")

@router.post("/black-litterman")
async def optimize_black_litterman(request: BlackLittermanRequest):
    """
    Perform Black-Litterman portfolio optimization
    
    Combines market equilibrium with investor views to generate
    optimized portfolio weights that blend market consensus with personal insights.
    """
    try:
        market_caps = np.array(request.market_caps)
        
        # Convert returns data to DataFrame
        returns_df = pd.DataFrame(request.returns_data)
        
        # Ensure asset order consistency
        if set(returns_df.columns) != set(request.asset_names):
            raise ValueError("Returns data assets must match provided asset names")
        
        returns_df = returns_df[request.asset_names]  # Reorder columns
        
        result = PortfolioOptimization.black_litterman_optimization(
            market_caps,
            returns_df,
            request.investor_views,
            request.view_confidence,
            request.risk_aversion,
            request.tau
        )
        
        if result['optimization_success']:
            return {
                "optimization_result": result,
                "asset_mapping": {name: i for i, name in enumerate(request.asset_names)},
                "interpretation": {
                    "views_impact": "Views incorporated" if request.investor_views else "Market equilibrium only",
                    "confidence_level": request.view_confidence,
                    "risk_aversion": request.risk_aversion,
                    "tau_parameter": request.tau
                }
            }
        else:
            raise HTTPException(status_code=400, detail=f"Black-Litterman optimization failed: {result.get('error_message', 'Unknown error')}")
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Black-Litterman optimization failed: {str(e)}")

@router.post("/efficient-frontier", response_model=EfficientFrontierResponse)
async def generate_efficient_frontier(request: EfficientFrontierRequest):
    """
    Generate efficient frontier for portfolio optimization
    
    Calculates the set of optimal portfolios offering the highest expected return
    for each level of risk, creating the efficient frontier curve.
    """
    try:
        expected_returns = np.array(request.expected_returns)
        covariance_matrix = np.array(request.covariance_matrix)
        
        # Validate dimensions
        if len(expected_returns) != len(request.asset_names):
            raise ValueError("Expected returns length must match number of assets")
        
        if covariance_matrix.shape != (len(request.asset_names), len(request.asset_names)):
            raise ValueError("Covariance matrix dimensions must match number of assets")
        
        result = PortfolioOptimization.efficient_frontier(
            expected_returns,
            covariance_matrix,
            request.num_portfolios,
            request.weight_bounds
        )
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        # Add asset names to portfolio weights
        for portfolio in result['efficient_portfolios']:
            portfolio['asset_weights'] = {
                request.asset_names[i]: weight 
                for i, weight in enumerate(portfolio['weights'])
            }
        
        for special_portfolio in ['min_variance_portfolio', 'max_sharpe_portfolio', 'max_return_portfolio']:
            if special_portfolio in result:
                result[special_portfolio]['asset_weights'] = {
                    request.asset_names[i]: weight 
                    for i, weight in enumerate(result[special_portfolio]['weights'])
                }
        
        return EfficientFrontierResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Efficient frontier generation failed: {str(e)}")

@router.post("/risk-parity")
async def optimize_risk_parity(request: RiskParityRequest):
    """
    Perform Risk Parity portfolio optimization
    
    Allocates capital so that each asset contributes equally to portfolio risk,
    creating a more balanced risk exposure across positions.
    """
    try:
        covariance_matrix = np.array(request.covariance_matrix)
        
        # Validate dimensions
        if covariance_matrix.shape != (len(request.asset_names), len(request.asset_names)):
            raise ValueError("Covariance matrix dimensions must match number of assets")
        
        result = PortfolioOptimization.risk_parity_optimization(
            covariance_matrix,
            request.target_volatility
        )
        
        if result['optimization_success']:
            # Add asset mapping
            asset_weights = {
                request.asset_names[i]: weight 
                for i, weight in enumerate(result['optimal_weights'])
            }
            
            risk_contributions = {
                request.asset_names[i]: contrib 
                for i, contrib in enumerate(result['risk_contributions'])
            }
            
            return {
                "optimization_result": result,
                "asset_weights": asset_weights,
                "risk_contributions": risk_contributions,
                "risk_contributions_pct": {
                    request.asset_names[i]: contrib 
                    for i, contrib in enumerate(result['risk_contributions_pct'])
                },
                "interpretation": {
                    "equal_risk_contribution": "Each asset contributes equally to portfolio risk",
                    "target_volatility": request.target_volatility,
                    "scaling_applied": result.get('scaling_applied', False)
                }
            }
        else:
            raise HTTPException(status_code=400, detail=f"Risk parity optimization failed: {result.get('error_message', 'Unknown error')}")
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Risk parity optimization failed: {str(e)}")

@router.post("/tactical-allocation")
async def generate_tactical_signals(request: TacticalAllocationRequest):
    """
    Generate tactical allocation signals and rebalancing recommendations
    
    Analyzes current vs target allocations considering momentum and volatility
    to provide actionable rebalancing signals and implementation guidance.
    """
    try:
        result = PortfolioOptimization.tactical_allocation_signals(
            request.current_weights,
            request.target_weights,
            request.returns_momentum,
            request.volatility_regime,
            request.rebalance_threshold
        )
        
        return {
            "tactical_signals": result,
            "generated_at": datetime.now().isoformat(),
            "market_conditions": {
                "average_momentum": np.mean(list(request.returns_momentum.values())),
                "average_volatility": np.mean(list(request.volatility_regime.values())),
                "momentum_regime": "Positive" if np.mean(list(request.returns_momentum.values())) > 0 else "Negative",
                "volatility_regime": "High" if np.mean(list(request.volatility_regime.values())) > 1.2 else "Normal"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Tactical allocation analysis failed: {str(e)}")

@router.post("/portfolio-comparison")
async def compare_portfolios(
    portfolio_a: Dict[str, float] = Body(..., description="First portfolio weights"),
    portfolio_b: Dict[str, float] = Body(..., description="Second portfolio weights"),
    expected_returns: List[float] = Body(..., description="Expected returns for assets"),
    covariance_matrix: List[List[float]] = Body(..., description="Covariance matrix"),
    asset_names: List[str] = Body(..., description="Asset identifiers")
):
    """
    Compare two portfolio allocations across multiple metrics
    
    Analyzes differences in allocation, expected performance, risk metrics,
    and provides recommendations for improvement.
    """
    try:
        expected_returns_array = np.array(expected_returns)
        cov_matrix = np.array(covariance_matrix)
        
        # Ensure portfolios have same assets
        all_assets = set(asset_names)
        portfolio_a = {asset: portfolio_a.get(asset, 0.0) for asset in all_assets}
        portfolio_b = {asset: portfolio_b.get(asset, 0.0) for asset in all_assets}
        
        # Convert to arrays in consistent order
        weights_a = np.array([portfolio_a[asset] for asset in asset_names])
        weights_b = np.array([portfolio_b[asset] for asset in asset_names])
        
        # Calculate portfolio metrics
        def calculate_portfolio_metrics(weights):
            portfolio_return = np.dot(weights, expected_returns_array)
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
            return {
                "return": portfolio_return,
                "volatility": portfolio_volatility,
                "sharpe_ratio": sharpe_ratio,
                "variance": portfolio_variance
            }
        
        metrics_a = calculate_portfolio_metrics(weights_a)
        metrics_b = calculate_portfolio_metrics(weights_b)
        
        # Calculate differences
        weight_differences = {
            asset: portfolio_b[asset] - portfolio_a[asset] 
            for asset in asset_names
        }
        
        # Identify largest changes
        largest_increases = sorted(
            [(asset, diff) for asset, diff in weight_differences.items() if diff > 0],
            key=lambda x: x[1], reverse=True
        )[:3]
        
        largest_decreases = sorted(
            [(asset, diff) for asset, diff in weight_differences.items() if diff < 0],
            key=lambda x: x[1]
        )[:3]
        
        return {
            "portfolio_comparison": {
                "portfolio_a_metrics": metrics_a,
                "portfolio_b_metrics": metrics_b,
                "performance_differences": {
                    "return_difference": metrics_b["return"] - metrics_a["return"],
                    "volatility_difference": metrics_b["volatility"] - metrics_a["volatility"],
                    "sharpe_difference": metrics_b["sharpe_ratio"] - metrics_a["sharpe_ratio"]
                },
                "weight_analysis": {
                    "weight_differences": weight_differences,
                    "largest_increases": largest_increases,
                    "largest_decreases": largest_decreases,
                    "total_turnover": sum(abs(diff) for diff in weight_differences.values()) / 2
                },
                "summary": {
                    "better_performer": "Portfolio B" if metrics_b["sharpe_ratio"] > metrics_a["sharpe_ratio"] else "Portfolio A",
                    "risk_level": "Portfolio B is riskier" if metrics_b["volatility"] > metrics_a["volatility"] else "Portfolio A is riskier",
                    "major_changes": len([d for d in weight_differences.values() if abs(d) > 0.05])
                }
            },
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Portfolio comparison failed: {str(e)}")

@router.get("/demo-optimization-data")
async def get_demo_optimization_data():
    """
    Generate demonstration data for portfolio optimization
    
    Returns sample expected returns, covariance matrix, and market data
    for testing optimization endpoints.
    """
    try:
        np.random.seed(42)  # For reproducible results
        
        # Asset universe
        asset_names = ["US_Equity", "EU_Equity", "EM_Equity", "US_Bonds", "EU_Bonds", "Commodities", "REITs", "Cash"]
        n_assets = len(asset_names)
        
        # Expected returns (annual)
        expected_returns = [0.08, 0.07, 0.09, 0.03, 0.02, 0.05, 0.06, 0.01]
        
        # Generate realistic covariance matrix
        # Start with correlation matrix
        base_correlations = np.array([
            [1.00, 0.75, 0.65, -0.20, -0.15, 0.30, 0.60, 0.10],  # US Equity
            [0.75, 1.00, 0.70, -0.10, -0.05, 0.25, 0.55, 0.05],  # EU Equity
            [0.65, 0.70, 1.00, -0.25, -0.20, 0.40, 0.50, 0.00],  # EM Equity
            [-0.20, -0.10, -0.25, 1.00, 0.80, -0.10, -0.30, 0.20],  # US Bonds
            [-0.15, -0.05, -0.20, 0.80, 1.00, -0.05, -0.25, 0.15],  # EU Bonds
            [0.30, 0.25, 0.40, -0.10, -0.05, 1.00, 0.35, -0.05],  # Commodities
            [0.60, 0.55, 0.50, -0.30, -0.25, 0.35, 1.00, 0.00],  # REITs
            [0.10, 0.05, 0.00, 0.20, 0.15, -0.05, 0.00, 1.00]   # Cash
        ])
        
        # Volatilities (annual)
        volatilities = [0.18, 0.20, 0.25, 0.05, 0.06, 0.22, 0.20, 0.01]
        
        # Convert to covariance matrix
        vol_matrix = np.outer(volatilities, volatilities)
        covariance_matrix = base_correlations * vol_matrix
        
        # Market capitalizations (for Black-Litterman)
        market_caps = [25000, 15000, 8000, 20000, 12000, 3000, 2000, 5000]  # Billions USD
        
        # Sample historical returns (for Black-Litterman)
        returns_data = {}
        for i, asset in enumerate(asset_names):
            # Generate correlated returns
            base_return = expected_returns[i] / 252  # Daily
            daily_vol = volatilities[i] / np.sqrt(252)
            returns_data[asset] = np.random.normal(base_return, daily_vol, 252).tolist()
        
        # Sample investor views
        investor_views = {
            "US_Equity": 0.10,  # Bullish on US equities
            "EM_Equity": 0.06,  # Cautious on emerging markets
            "Commodities": 0.08  # Positive on commodities
        }
        
        return {
            "asset_names": asset_names,
            "expected_returns": expected_returns,
            "covariance_matrix": covariance_matrix.tolist(),
            "market_caps": market_caps,
            "historical_returns": returns_data,
            "sample_investor_views": investor_views,
            "portfolio_examples": {
                "equal_weight": {asset: 1/n_assets for asset in asset_names},
                "market_cap_weight": {
                    asset: cap/sum(market_caps) 
                    for asset, cap in zip(asset_names, market_caps)
                },
                "conservative": {
                    "US_Equity": 0.30, "EU_Equity": 0.20, "EM_Equity": 0.10,
                    "US_Bonds": 0.25, "EU_Bonds": 0.10, "Cash": 0.05
                }
            },
            "description": "Sample 8-asset universe with realistic correlations and expected returns"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Demo data generation failed: {str(e)}")

@router.get("/optimization-strategies")
async def get_optimization_strategies():
    """
    Get overview of available optimization strategies and their use cases
    
    Returns detailed information about when to use each optimization approach
    and their relative strengths and weaknesses.
    """
    return {
        "strategies": {
            "mean_variance": {
                "description": "Classic Markowitz optimization maximizing return per unit of risk",
                "best_for": ["Long-term strategic allocation", "Clear return expectations", "Stable correlation structures"],
                "limitations": ["Sensitive to input estimation errors", "May produce concentrated portfolios", "Assumes normal distributions"],
                "parameters": ["expected_returns", "covariance_matrix", "risk_aversion"],
                "use_case": "Foundation of modern portfolio theory, best when you have strong views on expected returns"
            },
            "black_litterman": {
                "description": "Bayesian approach combining market equilibrium with investor views",
                "best_for": ["Incorporating market consensus", "Expressing specific investment views", "Reducing estimation error"],
                "limitations": ["Complex parameter tuning", "Requires market cap data", "View specification can be subjective"],
                "parameters": ["market_caps", "investor_views", "view_confidence", "tau"],
                "use_case": "When you want to tilt away from market consensus based on specific insights"
            },
            "risk_parity": {
                "description": "Equal risk contribution from each asset rather than equal dollar amounts",
                "best_for": ["Risk budgeting", "Diversification focus", "Avoiding concentration"],
                "limitations": ["May underweight high-return assets", "Ignores expected returns", "Can be volatile"],
                "parameters": ["covariance_matrix", "target_volatility"],
                "use_case": "When diversification and risk control are primary objectives"
            },
            "efficient_frontier": {
                "description": "Set of optimal portfolios for each level of risk",
                "best_for": ["Understanding risk-return tradeoffs", "Comparing portfolio efficiency", "Setting risk targets"],
                "limitations": ["Computationally intensive", "Still subject to estimation error", "Historical relationships may not persist"],
                "parameters": ["expected_returns", "covariance_matrix", "num_portfolios"],
                "use_case": "Strategic analysis and understanding the efficient set of possible portfolios"
            }
        },
        "selection_guide": {
            "high_confidence_forecasts": "Mean-Variance Optimization",
            "market_consensus_plus_views": "Black-Litterman",
            "focus_on_diversification": "Risk Parity",
            "risk_budgeting": "Risk Parity or Mean-Variance with constraints",
            "tactical_rebalancing": "Tactical Allocation Signals",
            "strategic_long_term": "Black-Litterman or Mean-Variance"
        },
        "implementation_tips": {
            "estimation_error": "Use robust estimation techniques and consider shrinkage estimators",
            "parameter_sensitivity": "Perform sensitivity analysis on key inputs",
            "transaction_costs": "Include transaction costs in optimization objective",
            "constraints": "Add realistic constraints (position limits, sector limits, etc.)",
            "backtesting": "Test strategies on historical data before implementation",
            "monitoring": "Regularly review and update model inputs and assumptions"
        }
    }