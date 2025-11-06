"""
Advanced Analytics Backend API
FastAPI application for institutional-grade financial analytics and reporting
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
from datetime import datetime
import logging

# Import routers
from .routers import performance, execution, risk, optimization, reports

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    
    # Startup
    logger.info("Starting Advanced Analytics Backend API")
    logger.info("Initializing mathematical libraries and services...")
    
    # Verify critical dependencies
    try:
        import numpy as np
        import pandas as pd
        import scipy
        logger.info("✓ Mathematical libraries loaded successfully")
    except ImportError as e:
        logger.error(f"✗ Critical dependency missing: {e}")
        raise
    
    # Optional dependencies check
    optional_deps = {
        'matplotlib': 'Chart generation',
        'reportlab': 'PDF export',
        'openpyxl': 'Excel export'
    }
    
    for dep, purpose in optional_deps.items():
        try:
            __import__(dep)
            logger.info(f"✓ {dep} available ({purpose})")
        except ImportError:
            logger.warning(f"⚠ {dep} not available - {purpose} disabled")
    
    logger.info("Analytics Backend API ready for institutional-grade calculations")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Advanced Analytics Backend API")

# Create FastAPI application
app = FastAPI(
    title="Advanced Analytics Backend API",
    description="""
    ## Institutional-Grade Financial Analytics & Reporting System
    
    This API provides comprehensive financial analytics capabilities including:
    
    ### 📊 Performance Analytics
    - Brinson-Fachler performance attribution
    - Risk-adjusted performance metrics (Sharpe, Sortino, Calmar ratios)
    - Rolling performance analysis
    - VaR/CVaR calculations
    
    ### 🎯 Execution Quality Analytics  
    - Implementation shortfall analysis
    - VWAP/TWAP performance benchmarking
    - Market impact measurement
    - Trade execution scorecards
    
    ### ⚠️ Risk Analytics
    - Monte Carlo VaR simulations
    - Stress testing and scenario analysis
    - Correlation and diversification analysis
    - Portfolio concentration metrics
    
    ### 🎯 Portfolio Optimization
    - Mean-variance optimization (Markowitz)
    - Black-Litterman model with investor views
    - Risk parity allocation
    - Efficient frontier generation
    
    ### 📈 Advanced Reporting
    - Multi-format exports (PDF, Excel, JSON)
    - Customizable report templates
    - Batch report generation
    - Interactive visualizations
    
    **Built for institutional use** with rigorous mathematical foundations and production-grade reliability.
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(performance.router)
app.include_router(execution.router)
app.include_router(risk.router)
app.include_router(optimization.router)
app.include_router(reports.router)

# Root endpoint
@app.get("/", tags=["System"])
async def root():
    """
    API Health Check and System Information
    
    Returns basic system information and API status.
    """
    return {
        "service": "Advanced Analytics Backend API",
        "version": "1.0.0",
        "status": "operational",
        "description": "Institutional-grade financial analytics and reporting system",
        "timestamp": datetime.now().isoformat(),
        "capabilities": {
            "performance_analytics": "Brinson-Fachler attribution, risk-adjusted metrics",
            "execution_analytics": "Implementation shortfall, VWAP/TWAP analysis",
            "risk_analytics": "VaR/CVaR, stress testing, concentration analysis",
            "portfolio_optimization": "Mean-variance, Black-Litterman, risk parity",
            "reporting": "Multi-format exports, custom templates"
        },
        "endpoints": {
            "performance": "/api/analytics/performance",
            "execution": "/api/analytics/execution", 
            "risk": "/api/analytics/risk",
            "optimization": "/api/analytics/optimization",
            "reports": "/api/analytics/reports"
        },
        "documentation": {
            "interactive_docs": "/docs",
            "redoc": "/redoc",
            "openapi_spec": "/openapi.json"
        }
    }

@app.get("/health", tags=["System"])
async def health_check():
    """
    Detailed Health Check
    
    Performs comprehensive system health verification including
    mathematical libraries and calculation capabilities.
    """
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system_checks": {},
        "library_checks": {},
        "calculation_tests": {}
    }
    
    # System checks
    try:
        import psutil
        health_status["system_checks"]["memory_usage"] = f"{psutil.virtual_memory().percent}%"
        health_status["system_checks"]["cpu_usage"] = f"{psutil.cpu_percent()}%"
    except ImportError:
        health_status["system_checks"]["system_monitoring"] = "psutil not available"
    
    # Library availability checks
    libraries = {
        'numpy': 'Core mathematical operations',
        'pandas': 'Data manipulation and analysis',
        'scipy': 'Scientific computing and optimization',
        'matplotlib': 'Chart generation (optional)',
        'reportlab': 'PDF generation (optional)',
        'openpyxl': 'Excel export (optional)'
    }
    
    for lib_name, description in libraries.items():
        try:
            __import__(lib_name)
            health_status["library_checks"][lib_name] = "available"
        except ImportError:
            health_status["library_checks"][lib_name] = "not available"
            if lib_name in ['numpy', 'pandas', 'scipy']:
                health_status["status"] = "degraded"
    
    # Basic calculation tests
    try:
        import numpy as np
        import pandas as pd
        
        # Test basic operations
        test_array = np.array([1, 2, 3, 4, 5])
        test_mean = np.mean(test_array)
        test_std = np.std(test_array)
        
        test_series = pd.Series([0.01, -0.02, 0.015, -0.005, 0.008])
        test_sharpe = test_series.mean() / test_series.std() if test_series.std() > 0 else 0
        
        health_status["calculation_tests"]["numpy_operations"] = "passing"
        health_status["calculation_tests"]["pandas_operations"] = "passing"
        health_status["calculation_tests"]["basic_analytics"] = "passing"
        
    except Exception as e:
        health_status["calculation_tests"]["error"] = str(e)
        health_status["status"] = "unhealthy"
    
    return health_status

@app.get("/api/analytics/capabilities", tags=["System"])
async def get_analytics_capabilities():
    """
    Get Detailed Analytics Capabilities
    
    Returns comprehensive information about available analytics functions,
    their parameters, and use cases.
    """
    
    return {
        "performance_analytics": {
            "description": "Comprehensive performance measurement and attribution",
            "functions": {
                "risk_adjusted_metrics": {
                    "description": "Calculate Sharpe, Sortino, Calmar, Information ratios",
                    "inputs": ["returns", "risk_free_rate"],
                    "outputs": ["sharpe_ratio", "sortino_ratio", "calmar_ratio", "max_drawdown"]
                },
                "brinson_fachler_attribution": {
                    "description": "Decompose performance into allocation, selection, and interaction effects",
                    "inputs": ["portfolio_weights", "benchmark_weights", "returns"],
                    "outputs": ["allocation_effect", "selection_effect", "interaction_effect"]
                },
                "var_cvar_analysis": {
                    "description": "Value at Risk and Conditional Value at Risk calculations",
                    "inputs": ["returns", "confidence_levels"],
                    "outputs": ["var_estimates", "cvar_estimates", "tail_statistics"]
                }
            }
        },
        "execution_analytics": {
            "description": "Trade execution quality measurement and cost analysis",
            "functions": {
                "implementation_shortfall": {
                    "description": "Measure execution cost vs. paper portfolio",
                    "inputs": ["decision_price", "arrival_price", "execution_price", "trade_value"],
                    "outputs": ["shortfall_bps", "market_impact", "timing_cost", "commission_cost"]
                },
                "vwap_analysis": {
                    "description": "Compare execution against volume-weighted average price",
                    "inputs": ["execution_price", "vwap_price", "trade_volume", "total_volume"],
                    "outputs": ["vwap_difference_bps", "performance_vs_vwap"]
                },
                "market_impact_analysis": {
                    "description": "Measure price impact from trading activity",
                    "inputs": ["pre_trade_price", "post_trade_price", "trade_size", "daily_volume"],
                    "outputs": ["temporary_impact", "permanent_impact", "participation_rate"]
                }
            }
        },
        "risk_analytics": {
            "description": "Comprehensive risk measurement and stress testing",
            "functions": {
                "monte_carlo_var": {
                    "description": "Simulate portfolio scenarios for VaR calculation",
                    "inputs": ["returns", "num_simulations", "confidence_levels"],
                    "outputs": ["var_scenarios", "cvar_estimates", "worst_case_scenarios"]
                },
                "stress_testing": {
                    "description": "Analyze portfolio under predefined stress scenarios",
                    "inputs": ["portfolio_weights", "asset_returns", "stress_scenarios"],
                    "outputs": ["scenario_returns", "worst_case", "best_case"]
                },
                "correlation_analysis": {
                    "description": "Analyze asset correlations and diversification",
                    "inputs": ["asset_returns"],
                    "outputs": ["correlation_matrix", "diversification_ratio", "concentration_metrics"]
                }
            }
        },
        "portfolio_optimization": {
            "description": "Advanced portfolio optimization and allocation strategies",
            "functions": {
                "mean_variance_optimization": {
                    "description": "Markowitz mean-variance optimization",
                    "inputs": ["expected_returns", "covariance_matrix", "risk_aversion"],
                    "outputs": ["optimal_weights", "expected_return", "expected_volatility"]
                },
                "black_litterman": {
                    "description": "Bayesian optimization combining market views with equilibrium",
                    "inputs": ["market_caps", "returns_data", "investor_views", "confidence"],
                    "outputs": ["bl_weights", "implied_returns", "expected_performance"]
                },
                "efficient_frontier": {
                    "description": "Generate set of optimal portfolios for different risk levels",
                    "inputs": ["expected_returns", "covariance_matrix", "num_portfolios"],
                    "outputs": ["frontier_portfolios", "min_variance", "max_sharpe"]
                }
            }
        },
        "reporting": {
            "description": "Comprehensive report generation and export capabilities",
            "functions": {
                "performance_report": {
                    "description": "Generate detailed performance analysis report",
                    "inputs": ["portfolio_data", "benchmark_data", "period"],
                    "outputs": ["executive_summary", "metrics", "charts", "recommendations"]
                },
                "risk_report": {
                    "description": "Generate comprehensive risk analysis report",
                    "inputs": ["portfolio_data", "risk_analytics", "stress_scenarios"],
                    "outputs": ["risk_summary", "var_analysis", "stress_results", "recommendations"]
                },
                "multi_format_export": {
                    "description": "Export reports in multiple formats",
                    "inputs": ["report_data", "format"],
                    "outputs": ["json", "excel", "pdf"]
                }
            }
        }
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler for unhandled errors
    """
    logger.error(f"Unhandled exception: {str(exc)}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred during processing",
            "timestamp": datetime.now().isoformat(),
            "support": "Contact system administrator with timestamp for assistance"
        }
    )

# Run application
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )