"""
Advanced Analytics & Reporting System for Day Trading Orchestrator

This module provides comprehensive analytics, reporting, and performance analysis
tools for professional trading operations.
"""

from .core.analytics_engine import AnalyticsEngine
from .performance.performance_analytics import PerformanceAnalytics
from .attribution.attribution_analysis import AttributionAnalysis
from .execution.execution_quality import ExecutionQualityAnalyzer
from .impact.market_impact import MarketImpactAnalyzer
from .reporting.automated_reporter import AutomatedReporter
from .optimization.portfolio_optimizer import PortfolioOptimizer
from .risk.risk_dashboards import RiskDashboardGenerator
from .integration.matrix_integration import MatrixIntegration

__all__ = [
    'AnalyticsEngine',
    'PerformanceAnalytics', 
    'AttributionAnalysis',
    'ExecutionQualityAnalyzer',
    'MarketImpactAnalyzer',
    'AutomatedReporter',
    'PortfolioOptimizer',
    'RiskDashboardGenerator',
    'MatrixIntegration'
]

__version__ = "1.0.0"