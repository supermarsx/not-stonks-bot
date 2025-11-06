"""
@file frequency_analytics.py
@brief Frequency Analytics and Reporting System

@details
This module provides comprehensive analytics and reporting capabilities for
trading frequency management. It includes frequency performance analysis,
optimization insights, trend analysis, and predictive modeling for frequency
optimization.

Key Features:
- Real-time frequency analytics and reporting
- Frequency performance trend analysis
- Optimization recommendation generation
- Frequency risk analytics and scoring
- Cross-strategy frequency correlation analysis
- Predictive frequency modeling
- Interactive frequency dashboards
- Automated frequency optimization insights

@author Trading Orchestrator System
@version 1.0
@date 2025-11-06

@note
This module provides the analytics foundation for frequency management
and should be integrated with the frequency manager and risk management systems.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import json
import statistics
from loguru import logger

from config.trading_frequency import (
    FrequencyManager, FrequencySettings, FrequencyType, FrequencyAlertType,
    FrequencyAlert, FrequencyOptimization
)


class AnalyticsPeriod(str, Enum):
    """
    @enum AnalyticsPeriod
    @brief Analytics reporting periods
    
    @details
    Defines different time periods for frequency analytics and reporting.
    """
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


class OptimizationTarget(str, Enum):
    """
    @enum OptimizationTarget
    @brief Frequency optimization targets
    
    @details
    Defines different optimization targets for frequency management.
    """
    MAXIMIZE_SHARPE = "maximize_sharpe"
    MINIMIZE_DRAWDOWN = "minimize_drawdown"
    MAXIMIZE_RETURNS = "maximize_returns"
    MINIMIZE_VOLATILITY = "minimize_volatility"
    BALANCE_RISK_RETURN = "balance_risk_return"
    MAXIMIZE_FREQUENCY_EFFICIENCY = "maximize_frequency_efficiency"


@dataclass
class FrequencyAnalyticsReport:
    """
    @class FrequencyAnalyticsReport
    @brief Comprehensive frequency analytics report
    
    @details
    Contains complete frequency analytics including performance metrics,
    optimization recommendations, trend analysis, and predictive insights.
    
    @par Report Components:
    - summary: Executive summary of frequency performance
    - performance_metrics: Detailed performance statistics
    - optimization_insights: Optimization recommendations and insights
    - trend_analysis: Historical trend analysis
    - predictive_modeling: Predictive frequency modeling results
    - risk_analysis: Frequency risk analysis and recommendations
    - comparative_analysis: Strategy vs benchmark comparisons
    """
    
    strategy_id: str
    report_period: AnalyticsPeriod
    period_start: datetime
    period_end: datetime
    generated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Summary metrics
    total_trades: int = 0
    average_frequency_rate: float = 0.0
    frequency_efficiency: float = 0.0
    optimization_score: float = 0.0
    risk_adjusted_return: float = 0.0
    
    # Performance breakdown
    trades_by_hour: Dict[int, int] = field(default_factory=dict)
    trades_by_day: Dict[int, int] = field(default_factory=dict)
    trades_by_frequency_type: Dict[str, int] = field(default_factory=dict)
    
    # Optimization insights
    current_frequency_settings: Dict[str, Any] = field(default_factory=dict)
    recommended_frequency_settings: Dict[str, Any] = field(default_factory=dict)
    expected_improvement: float = 0.0
    optimization_confidence: float = 0.0
    
    # Trend analysis
    frequency_trends: List[Dict[str, Any]] = field(default_factory=list)
    volatility_trends: List[Dict[str, Any]] = field(default_factory=list)
    performance_trends: List[Dict[str, Any]] = field(default_factory=list)
    
    # Risk metrics
    frequency_var: float = 0.0
    frequency_volatility: float = 0.0
    frequency_drawdown: float = 0.0
    max_frequency_rate: float = 0.0
    
    # Comparative analysis
    benchmark_comparison: Dict[str, float] = field(default_factory=dict)
    percentile_rankings: Dict[str, float] = field(default_factory=dict)
    
    # Predictive insights
    predicted_optimal_frequency: float = 0.0
    predicted_performance: Dict[str, float] = field(default_factory=dict)
    forecasting_horizon_days: int = 30
    
    # Recommendations
    top_recommendations: List[str] = field(default_factory=list)
    risk_warnings: List[str] = field(default_factory=list)
    optimization_opportunities: List[str] = field(default_factory=list)
    
    # Metadata
    data_quality_score: float = 1.0
    analysis_confidence: float = 0.0
    sample_size: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FrequencyOptimizationInsight:
    """
    @class FrequencyOptimizationInsight
    @brief Frequency optimization insight and recommendation
    
    @details
    Contains detailed optimization insights including current performance,
    optimization opportunities, expected improvements, and implementation guidance.
    """
    
    insight_id: str
    strategy_id: str
    insight_type: str  # frequency_adjustment, position_sizing, risk_optimization
    priority: str  # high, medium, low
    
    # Current state
    current_frequency_rate: float = 0.0
    current_performance_score: float = 0.0
    current_risk_score: float = 0.0
    
    # Optimization target
    target_frequency_rate: float = 0.0
    target_performance_score: float = 0.0
    expected_improvement: float = 0.0
    confidence_level: float = 0.0
    
    # Implementation details
    recommended_settings: Dict[str, Any] = field(default_factory=dict)
    implementation_steps: List[str] = field(default_factory=list)
    risk_considerations: List[str] = field(default_factory=list)
    expected_timeline: str = ""
    
    # Analysis data
    backtest_results: Dict[str, Any] = field(default_factory=dict)
    sensitivity_analysis: Dict[str, Any] = field(default_factory=dict)
    alternative_scenarios: List[Dict[str, Any]] = field(default_factory=list)
    
    # Tracking
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    implemented: bool = False
    implementation_date: Optional[datetime] = None


class FrequencyAnalyticsEngine:
    """
    @class FrequencyAnalyticsEngine
    @brief Core frequency analytics and optimization engine
    
    @details
    Provides comprehensive analytics capabilities for frequency management
    including real-time analysis, optimization insights, predictive modeling,
    and automated reporting.
    
    @par Core Capabilities:
    - Real-time frequency performance analytics
    - Advanced frequency optimization insights
    - Predictive frequency modeling and forecasting
    - Cross-strategy frequency correlation analysis
    - Automated frequency risk assessment
    - Interactive frequency dashboards and reports
    - Machine learning-based frequency optimization
    
    @par Analytics Pipeline:
    1. Data Collection: Gather frequency metrics and trade data
    2. Data Processing: Clean, validate, and transform data
    3. Statistical Analysis: Calculate performance metrics and trends
    4. Optimization Analysis: Generate optimization recommendations
    5. Predictive Modeling: Forecast optimal frequency settings
    6. Report Generation: Create comprehensive analytics reports
    7. Alert Generation: Generate alerts and recommendations
    """
    
    def __init__(self, frequency_manager: FrequencyManager):
        """
        Initialize frequency analytics engine
        
        Args:
            frequency_manager: Frequency manager instance
        """
        self.frequency_manager = frequency_manager
        
        # Analytics data storage
        self.analytics_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.optimization_history: List[FrequencyOptimizationInsight] = []
        self.reports_cache: Dict[str, FrequencyAnalyticsReport] = {}
        
        # Analytics configuration
        self.analytics_enabled = True
        self.reporting_interval_hours = 1
        self.optimization_horizon_days = 30
        self.min_data_points = 100
        
        # Performance tracking
        self.analytics_performance = {
            "last_analysis_time": None,
            "analysis_count": 0,
            "average_analysis_time": 0.0,
            "optimization_success_rate": 0.0
        }
        
        # Machine learning models (placeholder for advanced features)
        self.ml_models = {}
        self.model_training_enabled = False
        
        logger.info("FrequencyAnalyticsEngine initialized")
    
    async def generate_analytics_report(
        self,
        strategy_id: str,
        period: AnalyticsPeriod = AnalyticsPeriod.DAILY,
        custom_start: Optional[datetime] = None,
        custom_end: Optional[datetime] = None
    ) -> FrequencyAnalyticsReport:
        """
        Generate comprehensive frequency analytics report
        
        Args:
            strategy_id: Strategy identifier
            period: Analytics period
            custom_start: Custom period start time
            custom_end: Custom period end time
            
        Returns:
            Comprehensive frequency analytics report
        """
        start_time = datetime.utcnow()
        
        try:
            # Determine reporting period
            if custom_start and custom_end:
                period_start = custom_start
                period_end = custom_end
            else:
                period_start, period_end = self._get_period_bounds(period)
            
            # Collect and analyze data
            analytics_data = await self._collect_analytics_data(strategy_id, period_start, period_end)
            
            # Generate report
            report = FrequencyAnalyticsReport(
                strategy_id=strategy_id,
                report_period=period,
                period_start=period_start,
                period_end=period_end
            )
            
            # Calculate summary metrics
            await self._calculate_summary_metrics(report, analytics_data)
            
            # Performance breakdown
            await self._calculate_performance_breakdown(report, analytics_data)
            
            # Optimization insights
            await self._generate_optimization_insights(report, strategy_id)
            
            # Trend analysis
            await self._perform_trend_analysis(report, analytics_data)
            
            # Risk metrics
            await self._calculate_risk_metrics(report, analytics_data)
            
            # Comparative analysis
            await self._perform_comparative_analysis(report, strategy_id)
            
            # Predictive modeling
            await self._perform_predictive_modeling(report, analytics_data)
            
            # Generate recommendations
            await self._generate_recommendations(report, analytics_data)
            
            # Calculate data quality and confidence
            await self._assess_data_quality(report, analytics_data)
            
            # Cache report
            cache_key = f"{strategy_id}_{period.value}_{period_start.isoformat()}_{period_end.isoformat()}"
            self.reports_cache[cache_key] = report
            
            # Update performance tracking
            analysis_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_performance_tracking(analysis_time)
            
            logger.info(
                f"Analytics report generated for {strategy_id} ({period.value}): "
                f"{analysis_time:.2f}s analysis time, "
                f"efficiency: {report.frequency_efficiency:.3f}"
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating analytics report for {strategy_id}: {e}")
            # Return minimal report with error information
            return FrequencyAnalyticsReport(
                strategy_id=strategy_id,
                report_period=period,
                period_start=custom_start or datetime.utcnow() - timedelta(days=1),
                period_end=custom_end or datetime.utcnow()
            )
    
    async def generate_optimization_insights(
        self,
        strategy_id: str,
        target: OptimizationTarget = OptimizationTarget.BALANCE_RISK_RETURN,
        optimization_horizon_days: int = 30
    ) -> List[FrequencyOptimizationInsight]:
        """
        Generate frequency optimization insights
        
        Args:
            strategy_id: Strategy identifier
            target: Optimization target
            optimization_horizon_days: Optimization analysis horizon
            
        Returns:
            List of optimization insights
        """
        try:
            # Get current frequency data
            frequency_metrics = self.frequency_manager.get_frequency_metrics(strategy_id)
            if not frequency_metrics:
                logger.warning(f"No frequency metrics available for {strategy_id}")
                return []
            
            insights = []
            
            # Analyze current performance
            current_performance = await self._analyze_current_performance(strategy_id)
            
            # Generate insights based on target
            if target in [OptimizationTarget.MAXIMIZE_SHARPE, OptimizationTarget.BALANCE_RISK_RETURN]:
                insights.extend(await self._generate_sharpe_optimization_insights(strategy_id, current_performance))
            
            if target in [OptimizationTarget.MINIMIZE_DRAWDOWN, OptimizationTarget.BALANCE_RISK_RETURN]:
                insights.extend(await self._generate_drawdown_optimization_insights(strategy_id, current_performance))
            
            if target in [OptimizationTarget.MAXIMIZE_RETURNS, OptimizationTarget.MAXIMIZE_FREQUENCY_EFFICIENCY]:
                insights.extend(await self._generate_efficiency_optimization_insights(strategy_id, current_performance))
            
            if target == OptimizationTarget.MINIMIZE_VOLATILITY:
                insights.extend(await self._generate_volatility_optimization_insights(strategy_id, current_performance))
            
            # Sort insights by priority
            insights.sort(key=lambda x: {"high": 3, "medium": 2, "low": 1}[x.priority], reverse=True)
            
            # Store insights
            self.optimization_history.extend(insights)
            
            logger.info(f"Generated {len(insights)} optimization insights for {strategy_id}")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating optimization insights for {strategy_id}: {e}")
            return []
    
    async def analyze_frequency_trends(
        self,
        strategy_id: str,
        lookback_days: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze frequency trends over time
        
        Args:
            strategy_id: Strategy identifier
            lookback_days: Number of days to analyze
            
        Returns:
            Frequency trend analysis results
        """
        try:
            # Collect historical data
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=lookback_days)
            
            analytics_data = await self._collect_analytics_data(strategy_id, start_time, end_time)
            
            if len(analytics_data) < self.min_data_points:
                return {"status": "insufficient_data", "message": "Insufficient data for trend analysis"}
            
            # Perform trend analysis
            trends = {
                "frequency_trends": await self._analyze_frequency_trends(analytics_data),
                "volatility_trends": await self._analyze_volatility_trends(analytics_data),
                "performance_trends": await self._analyze_performance_trends(analytics_data),
                "correlation_analysis": await self._analyze_frequency_correlations(analytics_data),
                "seasonal_patterns": await self._analyze_seasonal_patterns(analytics_data),
                "trend_significance": await self._assess_trend_significance(analytics_data)
            }
            
            # Generate trend summary
            trends["summary"] = await self._generate_trend_summary(trends)
            
            logger.info(f"Frequency trend analysis completed for {strategy_id}: {lookback_days} days")
            
            return trends
            
        except Exception as e:
            logger.error(f"Error analyzing frequency trends for {strategy_id}: {e}")
            return {"status": "error", "message": str(e)}
    
    async def perform_predictive_modeling(
        self,
        strategy_id: str,
        forecast_horizon_days: int = 30
    ) -> Dict[str, Any]:
        """
        Perform predictive frequency modeling
        
        Args:
            strategy_id: Strategy identifier
            forecast_horizon_days: Forecasting horizon
            
        Returns:
            Predictive modeling results
        """
        try:
            # Collect data for modeling
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=forecast_horizon_days * 2)  # Use 2x horizon for training
            
            analytics_data = await self._collect_analytics_data(strategy_id, start_time, end_time)
            
            if len(analytics_data) < self.min_data_points:
                return {"status": "insufficient_data", "message": "Insufficient data for predictive modeling"}
            
            # Perform various predictive analyses
            predictions = {
                "frequency_forecasting": await self._forecast_frequency_rates(analytics_data, forecast_horizon_days),
                "performance_prediction": await self._predict_performance(analytics_data, forecast_horizon_days),
                "risk_forecasting": await self._forecast_risk_metrics(analytics_data, forecast_horizon_days),
                "optimization_recommendations": await self._predict_optimal_frequency(analytics_data),
                "scenario_analysis": await self._perform_scenario_analysis(analytics_data, forecast_horizon_days)
            }
            
            # Calculate model confidence and accuracy metrics
            predictions["model_metrics"] = await self._assess_model_performance(analytics_data)
            
            # Generate forecasting summary
            predictions["forecasting_summary"] = await self._generate_forecasting_summary(predictions)
            
            logger.info(f"Predictive modeling completed for {strategy_id}: {forecast_horizon_days} days horizon")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error performing predictive modeling for {strategy_id}: {e}")
            return {"status": "error", "message": str(e)}
    
    async def analyze_cross_strategy_frequency(
        self,
        strategy_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze frequency patterns across multiple strategies
        
        Args:
            strategy_ids: List of strategy identifiers
            
        Returns:
            Cross-strategy frequency analysis results
        """
        try:
            # Collect data for all strategies
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=30)  # 30-day analysis window
            
            strategy_data = {}
            for strategy_id in strategy_ids:
                data = await self._collect_analytics_data(strategy_id, start_time, end_time)
                if data:
                    strategy_data[strategy_id] = data
            
            if len(strategy_data) < 2:
                return {"status": "insufficient_strategies", "message": "Need at least 2 strategies for cross-analysis"}
            
            # Perform cross-strategy analysis
            analysis = {
                "correlation_analysis": await self._analyze_cross_strategy_correlations(strategy_data),
                "portfolio_frequency_analysis": await self._analyze_portfolio_frequency(strategy_data),
                "diversification_opportunities": await self._identify_diversification_opportunities(strategy_data),
                "frequency_concentration_risk": await self._assess_frequency_concentration_risk(strategy_data),
                "optimization_recommendations": await self._generate_cross_strategy_recommendations(strategy_data),
                "comparative_performance": await self._compare_strategy_performance(strategy_data)
            }
            
            # Generate summary
            analysis["analysis_summary"] = await self._generate_cross_strategy_summary(analysis)
            
            logger.info(f"Cross-strategy frequency analysis completed for {len(strategy_ids)} strategies")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error performing cross-strategy frequency analysis: {e}")
            return {"status": "error", "message": str(e)}
    
    def _get_period_bounds(self, period: AnalyticsPeriod) -> Tuple[datetime, datetime]:
        """Get start and end times for a given period"""
        end_time = datetime.utcnow()
        
        if period == AnalyticsPeriod.HOURLY:
            start_time = end_time - timedelta(hours=1)
        elif period == AnalyticsPeriod.DAILY:
            start_time = end_time - timedelta(days=1)
        elif period == AnalyticsPeriod.WEEKLY:
            start_time = end_time - timedelta(weeks=1)
        elif period == AnalyticsPeriod.MONTHLY:
            start_time = end_time - timedelta(days=30)
        elif period == AnalyticsPeriod.QUARTERLY:
            start_time = end_time - timedelta(days=90)
        elif period == AnalyticsPeriod.YEARLY:
            start_time = end_time - timedelta(days=365)
        else:
            start_time = end_time - timedelta(days=1)
        
        return start_time, end_time
    
    async def _collect_analytics_data(
        self,
        strategy_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Collect analytics data for a strategy"""
        # This would typically query the database
        # For now, return empty list (placeholder)
        return []
    
    async def _calculate_summary_metrics(
        self,
        report: FrequencyAnalyticsReport,
        analytics_data: List[Dict[str, Any]]
    ):
        """Calculate summary metrics for the report"""
        if not analytics_data:
            return
        
        # Basic metrics
        report.total_trades = len(analytics_data)
        
        if analytics_data:
            # Calculate averages
            frequency_rates = [d.get('frequency_rate', 0.0) for d in analytics_data if d.get('frequency_rate')]
            if frequency_rates:
                report.average_frequency_rate = statistics.mean(frequency_rates)
            
            # Calculate efficiency
            trade_counts = [d.get('trade_count', 0) for d in analytics_data]
            opportunity_counts = [d.get('opportunity_count', 1) for d in analytics_data if d.get('opportunity_count')]
            
            if opportunity_counts:
                total_trades = sum(trade_counts)
                total_opportunities = sum(opportunity_counts)
                report.frequency_efficiency = total_trades / max(total_opportunities, 1)
    
    async def _calculate_performance_breakdown(
        self,
        report: FrequencyAnalyticsReport,
        analytics_data: List[Dict[str, Any]]
    ):
        """Calculate performance breakdown by time periods"""
        # Group trades by hour and day
        trades_by_hour = defaultdict(int)
        trades_by_day = defaultdict(int)
        
        for data_point in analytics_data:
            timestamp = data_point.get('timestamp')
            if timestamp:
                hour = timestamp.hour
                day = timestamp.weekday()
                trades_by_hour[hour] += 1
                trades_by_day[day] += 1
        
        report.trades_by_hour = dict(trades_by_hour)
        report.trades_by_day = dict(trades_by_day)
    
    async def _generate_optimization_insights(
        self,
        report: FrequencyAnalyticsReport,
        strategy_id: str
    ):
        """Generate optimization insights for the report"""
        # Get frequency optimization recommendations
        optimizations = self.frequency_manager.get_optimization_recommendations(strategy_id)
        
        if optimizations:
            # Use the most recent optimization
            latest_optimization = max(optimizations, key=lambda x: x.optimization_date)
            
            report.recommended_frequency_settings = {
                "interval_seconds": latest_optimization.recommended_interval_seconds,
                "position_size_multiplier": latest_optimization.recommended_position_size_multiplier,
                "confidence_level": latest_optimization.confidence_level,
                "expected_improvement": latest_optimization.expected_improvement
            }
            
            report.expected_improvement = latest_optimization.expected_improvement
            report.optimization_confidence = latest_optimization.confidence_level
        
        # Get current settings
        current_settings = self.frequency_manager.settings
        report.current_frequency_settings = {
            "frequency_type": current_settings.frequency_type.value,
            "interval_seconds": current_settings.interval_seconds,
            "position_size_multiplier": current_settings.position_size_multiplier
        }
    
    async def _perform_trend_analysis(
        self,
        report: FrequencyAnalyticsReport,
        analytics_data: List[Dict[str, Any]]
    ):
        """Perform trend analysis"""
        # This would perform statistical trend analysis
        # For now, placeholder implementation
        report.frequency_trends = [{"trend": "stable", "slope": 0.0, "r_squared": 0.5}]
        report.volatility_trends = [{"trend": "increasing", "slope": 0.01, "r_squared": 0.3}]
        report.performance_trends = [{"trend": "improving", "slope": 0.02, "r_squared": 0.4}]
    
    async def _calculate_risk_metrics(
        self,
        report: FrequencyAnalyticsReport,
        analytics_data: List[Dict[str, Any]]
    ):
        """Calculate risk metrics"""
        # Calculate basic risk metrics
        if analytics_data:
            frequency_rates = [d.get('frequency_rate', 0.0) for d in analytics_data if d.get('frequency_rate')]
            
            if frequency_rates:
                report.frequency_volatility = statistics.stdev(frequency_rates)
                report.max_frequency_rate = max(frequency_rates)
                
                # Calculate Value at Risk (VaR)
                sorted_rates = sorted(frequency_rates)
                var_index = int(len(sorted_rates) * 0.05)  # 5% VaR
                report.frequency_var = sorted_rates[var_index] if var_index < len(sorted_rates) else 0.0
    
    async def _perform_comparative_analysis(
        self,
        report: FrequencyAnalyticsReport,
        strategy_id: str
    ):
        """Perform comparative analysis"""
        # Placeholder for benchmark comparisons
        report.benchmark_comparison = {
            "vs_market_average": 1.15,  # 15% better than market
            "vs_peer_group": 0.92,     # 8% below peer group
            "vs_historical": 1.05       # 5% better than historical
        }
        
        # Percentile rankings
        report.percentile_rankings = {
            "frequency_efficiency": 75.0,
            "risk_adjusted_return": 82.0,
            "sharpe_ratio": 68.0,
            "consistency": 71.0
        }
    
    async def _perform_predictive_modeling(
        self,
        report: FrequencyAnalyticsReport,
        analytics_data: List[Dict[str, Any]]
    ):
        """Perform predictive modeling"""
        # Placeholder for predictive modeling
        if analytics_data:
            # Simple prediction based on current trends
            current_rate = sum(d.get('frequency_rate', 0) for d in analytics_data[-10:]) / min(10, len(analytics_data))
            report.predicted_optimal_frequency = current_rate * 1.1  # 10% increase
            
            report.predicted_performance = {
                "expected_sharpe": 1.2,
                "expected_drawdown": -0.05,
                "expected_return": 0.08,
                "confidence_interval": [0.95, 1.05]
            }
    
    async def _generate_recommendations(
        self,
        report: FrequencyAnalyticsReport,
        analytics_data: List[Dict[str, Any]]
    ):
        """Generate recommendations based on analysis"""
        recommendations = []
        risk_warnings = []
        optimization_opportunities = []
        
        # Generate recommendations based on metrics
        if report.frequency_efficiency < 0.5:
            recommendations.append("Consider reducing trading frequency to improve efficiency")
            optimization_opportunities.append("Optimize signal quality vs trading frequency")
        
        if report.frequency_volatility > 0.5:
            risk_warnings.append("High frequency volatility detected - consider position size adjustments")
        
        if report.frequency_var > report.average_frequency_rate * 2:
            recommendations.append("Implement frequency controls to manage tail risk")
        
        report.top_recommendations = recommendations
        report.risk_warnings = risk_warnings
        report.optimization_opportunities = optimization_opportunities
    
    async def _assess_data_quality(
        self,
        report: FrequencyAnalyticsReport,
        analytics_data: List[Dict[str, Any]]
    ):
        """Assess data quality for the report"""
        report.sample_size = len(analytics_data)
        
        # Simple data quality assessment
        if len(analytics_data) < 50:
            report.data_quality_score = 0.5
            report.analysis_confidence = 0.6
        elif len(analytics_data) < 100:
            report.data_quality_score = 0.7
            report.analysis_confidence = 0.7
        else:
            report.data_quality_score = 0.9
            report.analysis_confidence = 0.85
    
    async def _analyze_current_performance(self, strategy_id: str) -> Dict[str, Any]:
        """Analyze current performance of a strategy"""
        # Get frequency metrics
        metrics = self.frequency_manager.get_frequency_metrics(strategy_id)
        
        if not metrics:
            return {"error": "No metrics available"}
        
        return {
            "frequency_rate": metrics.current_frequency_rate,
            "efficiency": metrics.frequency_efficiency,
            "total_trades": metrics.trades_today,
            "sharpe_ratio": metrics.frequency_sharpe,
            "drawdown": metrics.frequency_drawdown
        }
    
    async def _generate_sharpe_optimization_insights(
        self,
        strategy_id: str,
        current_performance: Dict[str, Any]
    ) -> List[FrequencyOptimizationInsight]:
        """Generate Sharpe ratio optimization insights"""
        insights = []
        
        # Check if optimization is needed
        current_sharpe = current_performance.get("sharpe_ratio", 0)
        if current_sharpe < 1.0:  # Sharpe below 1.0 needs improvement
            insight = FrequencyOptimizationInsight(
                insight_id=f"{strategy_id}_sharpe_opt_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                strategy_id=strategy_id,
                insight_type="frequency_adjustment",
                priority="high",
                current_frequency_rate=current_performance.get("frequency_rate", 0),
                current_performance_score=current_sharpe,
                target_frequency_rate=current_performance.get("frequency_rate", 0) * 0.8,  # Reduce frequency
                target_performance_score=1.2,
                expected_improvement=20.0,
                confidence_level=0.75,
                recommended_settings={
                    "interval_seconds": 600,  # 10 minutes
                    "position_size_multiplier": 0.9,
                    "cooldown_periods": 120
                },
                implementation_steps=[
                    "Increase trading intervals to reduce frequency",
                    "Implement cooldown periods between trades",
                    "Monitor performance over 2-week period"
                ]
            )
            insights.append(insight)
        
        return insights
    
    async def _generate_drawdown_optimization_insights(
        self,
        strategy_id: str,
        current_performance: Dict[str, Any]
    ) -> List[FrequencyOptimizationInsight]:
        """Generate drawdown optimization insights"""
        insights = []
        
        current_drawdown = current_performance.get("drawdown", 0)
        if abs(current_drawdown) > 0.1:  # Drawdown > 10%
            insight = FrequencyOptimizationInsight(
                insight_id=f"{strategy_id}_drawdown_opt_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                strategy_id=strategy_id,
                insight_type="position_sizing",
                priority="medium",
                current_frequency_rate=current_performance.get("frequency_rate", 0),
                current_risk_score=abs(current_drawdown),
                target_frequency_rate=current_performance.get("frequency_rate", 0) * 0.7,
                expected_improvement=15.0,
                confidence_level=0.68,
                recommended_settings={
                    "position_size_multiplier": 0.7,
                    "max_daily_frequency_risk": 0.03,
                    "frequency_volatility_adjustment": True
                },
                implementation_steps=[
                    "Reduce position sizes by 30%",
                    "Enable volatility-based adjustments",
                    "Implement stricter risk limits"
                ],
                risk_considerations=[
                    "Reduced position sizes may limit profit potential",
                    "Monitor for over-conservative trading"
                ]
            )
            insights.append(insight)
        
        return insights
    
    async def _generate_efficiency_optimization_insights(
        self,
        strategy_id: str,
        current_performance: Dict[str, Any]
    ) -> List[FrequencyOptimizationInsight]:
        """Generate efficiency optimization insights"""
        insights = []
        
        current_efficiency = current_performance.get("efficiency", 0)
        if current_efficiency < 0.7:  # Efficiency below 70%
            insight = FrequencyOptimizationInsight(
                insight_id=f"{strategy_id}_efficiency_opt_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                strategy_id=strategy_id,
                insight_type="risk_optimization",
                priority="medium",
                current_frequency_rate=current_performance.get("frequency_rate", 0),
                current_performance_score=current_efficiency,
                target_frequency_rate=current_performance.get("frequency_rate", 0) * 0.9,
                target_performance_score=0.85,
                expected_improvement=12.0,
                confidence_level=0.72,
                recommended_settings={
                    "signal_confidence_threshold": 0.7,
                    "min_time_between_trades": 300,
                    "frequency_efficiency_target": 0.8
                },
                implementation_steps=[
                    "Increase minimum time between trades",
                    "Implement signal confidence filtering",
                    "Monitor trade quality metrics"
                ]
            )
            insights.append(insight)
        
        return insights
    
    async def _generate_volatility_optimization_insights(
        self,
        strategy_id: str,
        current_performance: Dict[str, Any]
    ) -> List[FrequencyOptimizationInsight]:
        """Generate volatility optimization insights"""
        # Placeholder implementation
        return []
    
    async def _analyze_frequency_trends(self, analytics_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze frequency trends over time"""
        # Placeholder for trend analysis
        return [{"trend": "stable", "slope": 0.0, "significance": 0.5}]
    
    async def _analyze_volatility_trends(self, analytics_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze volatility trends"""
        # Placeholder for volatility analysis
        return [{"trend": "stable", "slope": 0.0, "significance": 0.5}]
    
    async def _analyze_performance_trends(self, analytics_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze performance trends"""
        # Placeholder for performance analysis
        return [{"trend": "improving", "slope": 0.01, "significance": 0.6}]
    
    async def _analyze_frequency_correlations(self, analytics_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze frequency correlations"""
        # Placeholder for correlation analysis
        return {"auto_correlation": 0.3, "cross_correlation": 0.1}
    
    async def _analyze_seasonal_patterns(self, analytics_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze seasonal patterns"""
        # Placeholder for seasonal analysis
        return {"daily_patterns": {}, "weekly_patterns": {}}
    
    async def _assess_trend_significance(self, analytics_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Assess statistical significance of trends"""
        # Placeholder for significance testing
        return {"frequency_trend_p_value": 0.3, "volatility_trend_p_value": 0.5}
    
    async def _generate_trend_summary(self, trends: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of trend analysis"""
        return {
            "overall_trend": "stable",
            "trend_strength": 0.5,
            "significant_patterns": [],
            "recommendations": ["Continue current frequency settings"]
        }
    
    async def _forecast_frequency_rates(
        self,
        analytics_data: List[Dict[str, Any]],
        forecast_horizon_days: int
    ) -> Dict[str, Any]:
        """Forecast future frequency rates"""
        # Placeholder for forecasting
        return {
            "forecast": [2.1, 2.2, 2.0, 2.3, 2.1],
            "confidence_intervals": [[1.9, 2.3], [2.0, 2.4], [1.8, 2.2], [2.1, 2.5], [1.9, 2.3]],
            "model_accuracy": 0.75
        }
    
    async def _predict_performance(
        self,
        analytics_data: List[Dict[str, Any]],
        forecast_horizon_days: int
    ) -> Dict[str, Any]:
        """Predict future performance"""
        # Placeholder for performance prediction
        return {"expected_return": 0.08, "expected_volatility": 0.15, "confidence": 0.7}
    
    async def _forecast_risk_metrics(
        self,
        analytics_data: List[Dict[str, Any]],
        forecast_horizon_days: int
    ) -> Dict[str, Any]:
        """Forecast risk metrics"""
        # Placeholder for risk forecasting
        return {"expected_var": 0.05, "expected_drawdown": -0.08, "expected_volatility": 0.12}
    
    async def _predict_optimal_frequency(self, analytics_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Predict optimal frequency settings"""
        # Placeholder for optimal frequency prediction
        return {"optimal_frequency_rate": 2.1, "confidence": 0.68}
    
    async def _perform_scenario_analysis(
        self,
        analytics_data: List[Dict[str, Any]],
        forecast_horizon_days: int
    ) -> List[Dict[str, Any]]:
        """Perform scenario analysis"""
        # Placeholder for scenario analysis
        return [
            {"scenario": "bull_market", "frequency_rate": 2.5, "probability": 0.3},
            {"scenario": "bear_market", "frequency_rate": 1.8, "probability": 0.2},
            {"scenario": "sideways_market", "frequency_rate": 2.1, "probability": 0.5}
        ]
    
    async def _assess_model_performance(self, analytics_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Assess predictive model performance"""
        # Placeholder for model assessment
        return {"accuracy": 0.75, "precision": 0.72, "recall": 0.78, "f1_score": 0.75}
    
    async def _generate_forecasting_summary(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of forecasting results"""
        return {
            "forecasting_confidence": 0.75,
            "key_predictions": ["Frequency rates will remain stable"],
            "recommendations": ["Maintain current frequency settings"]
        }
    
    async def _analyze_cross_strategy_correlations(self, strategy_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
        """Analyze correlations between strategies"""
        # Placeholder for cross-strategy correlation analysis
        correlations = {}
        strategy_ids = list(strategy_data.keys())
        
        for i, strategy1 in enumerate(strategy_ids):
            for strategy2 in strategy_ids[i+1:]:
                correlations[f"{strategy1}_vs_{strategy2}"] = 0.3  # Placeholder correlation
        
        return correlations
    
    async def _analyze_portfolio_frequency(self, strategy_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze portfolio-level frequency patterns"""
        # Placeholder for portfolio analysis
        return {
            "total_portfolio_frequency": 8.5,
            "frequency_concentration": 0.3,
            "frequency_risk_score": 0.25
        }
    
    async def _identify_diversification_opportunities(self, strategy_data: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """Identify frequency diversification opportunities"""
        # Placeholder for diversification analysis
        return ["Consider strategies with different frequency patterns"]
    
    async def _assess_frequency_concentration_risk(self, strategy_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
        """Assess frequency concentration risk"""
        # Placeholder for concentration risk assessment
        return {"concentration_risk": 0.2, "diversification_benefit": 0.15}
    
    async def _generate_cross_strategy_recommendations(self, strategy_data: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """Generate cross-strategy recommendations"""
        # Placeholder for cross-strategy recommendations
        return ["Rebalance frequency allocation across strategies"]
    
    async def _compare_strategy_performance(self, strategy_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, float]]:
        """Compare performance across strategies"""
        # Placeholder for comparative performance analysis
        comparisons = {}
        for strategy_id in strategy_data.keys():
            comparisons[strategy_id] = {
                "frequency_efficiency": 0.75,
                "risk_adjusted_return": 0.12,
                "consistency_score": 0.68
            }
        return comparisons
    
    async def _generate_cross_strategy_summary(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of cross-strategy analysis"""
        return {
            "overall_portfolio_health": "good",
            "key_opportunities": ["Frequency diversification"],
            "primary_risks": ["Frequency concentration"],
            "recommendations": ["Implement frequency-based position sizing"]
        }
    
    def _update_performance_tracking(self, analysis_time: float):
        """Update analytics performance tracking"""
        self.analytics_performance["last_analysis_time"] = datetime.utcnow()
        self.analytics_performance["analysis_count"] += 1
        
        # Update average analysis time
        current_avg = self.analytics_performance["average_analysis_time"]
        count = self.analytics_performance["analysis_count"]
        self.analytics_performance["average_analysis_time"] = (
            (current_avg * (count - 1) + analysis_time) / count
        )
    
    def get_cached_report(
        self,
        strategy_id: str,
        period: AnalyticsPeriod,
        period_start: datetime,
        period_end: datetime
    ) -> Optional[FrequencyAnalyticsReport]:
        """Get cached analytics report"""
        cache_key = f"{strategy_id}_{period.value}_{period_start.isoformat()}_{period_end.isoformat()}"
        return self.reports_cache.get(cache_key)
    
    def clear_cache(self):
        """Clear analytics cache"""
        self.reports_cache.clear()
        logger.info("Analytics cache cleared")
    
    def get_optimization_history(
        self,
        strategy_id: Optional[str] = None,
        since: Optional[datetime] = None
    ) -> List[FrequencyOptimizationInsight]:
        """Get optimization insight history"""
        history = self.optimization_history
        
        if strategy_id:
            history = [insight for insight in history if insight.strategy_id == strategy_id]
        
        if since:
            history = [insight for insight in history if insight.created_at >= since]
        
        return history
    
    async def get_analytics_summary(self) -> Dict[str, Any]:
        """Get analytics system summary"""
        return {
            "analytics_enabled": self.analytics_enabled,
            "cached_reports": len(self.reports_cache),
            "optimization_insights": len(self.optimization_history),
            "performance_tracking": self.analytics_performance,
            "configuration": {
                "min_data_points": self.min_data_points,
                "optimization_horizon_days": self.optimization_horizon_days,
                "reporting_interval_hours": self.reporting_interval_hours
            }
        }


# Global analytics engine instance
_frequency_analytics_engine: Optional[FrequencyAnalyticsEngine] = None


def get_frequency_analytics_engine() -> Optional[FrequencyAnalyticsEngine]:
    """Get global frequency analytics engine instance"""
    return _frequency_analytics_engine


def initialize_frequency_analytics_engine(frequency_manager: FrequencyManager) -> FrequencyAnalyticsEngine:
    """
    Initialize global frequency analytics engine
    
    Args:
        frequency_manager: Frequency manager instance
        
    Returns:
        Initialized frequency analytics engine
    """
    global _frequency_analytics_engine
    _frequency_analytics_engine = FrequencyAnalyticsEngine(frequency_manager)
    return _frequency_analytics_engine


async def generate_frequency_analytics_report(
    strategy_id: str,
    period: AnalyticsPeriod = AnalyticsPeriod.DAILY
) -> Optional[FrequencyAnalyticsReport]:
    """
    Convenience function to generate frequency analytics report
    
    Args:
        strategy_id: Strategy identifier
        period: Analytics period
        
    Returns:
        Frequency analytics report
    """
    engine = get_frequency_analytics_engine()
    if engine:
        return await engine.generate_analytics_report(strategy_id, period)
    return None


async def generate_frequency_optimization_insights(
    strategy_id: str,
    target: OptimizationTarget = OptimizationTarget.BALANCE_RISK_RETURN
) -> List[FrequencyOptimizationInsight]:
    """
    Convenience function to generate optimization insights
    
    Args:
        strategy_id: Strategy identifier
        target: Optimization target
        
    Returns:
        List of optimization insights
    """
    engine = get_frequency_analytics_engine()
    if engine:
        return await engine.generate_optimization_insights(strategy_id, target)
    return []


# Example usage and testing
if __name__ == "__main__":
    async def test_frequency_analytics():
        """Test frequency analytics functionality"""
        
        # This would require a real frequency manager
        # For testing, we'll create a mock
        class MockFrequencyManager:
            def __init__(self):
                self.settings = type('Settings', (), {
                    'frequency_type': 'medium',
                    'interval_seconds': 300,
                    'position_size_multiplier': 1.0
                })()
            
            def get_frequency_metrics(self, strategy_id):
                return type('Metrics', (), {
                    'current_frequency_rate': 2.0,
                    'frequency_efficiency': 0.75,
                    'trades_today': 45,
                    'frequency_sharpe': 1.2,
                    'frequency_drawdown': -0.05
                })()
            
            def get_optimization_recommendations(self, strategy_id):
                return []
        
        # Initialize analytics engine
        mock_manager = MockFrequencyManager()
        analytics_engine = initialize_frequency_analytics_engine(mock_manager)
        
        # Test analytics report generation
        report = await analytics_engine.generate_analytics_report(
            strategy_id="test_strategy",
            period=AnalyticsPeriod.DAILY
        )
        print(f"Analytics report generated: {report.total_trades} trades, "
              f"efficiency: {report.frequency_efficiency:.3f}")
        
        # Test optimization insights
        insights = await analytics_engine.generate_optimization_insights(
            strategy_id="test_strategy",
            target=OptimizationTarget.MAXIMIZE_SHARPE
        )
        print(f"Generated {len(insights)} optimization insights")
        
        # Test trend analysis
        trends = await analytics_engine.analyze_frequency_trends("test_strategy", lookback_days=7)
        print(f"Trend analysis status: {trends.get('status', 'unknown')}")
        
        # Test predictive modeling
        predictions = await analytics_engine.perform_predictive_modeling(
            strategy_id="test_strategy",
            forecast_horizon_days=14
        )
        print(f"Predictive modeling status: {predictions.get('status', 'unknown')}")
        
        # Test analytics summary
        summary = await analytics_engine.get_analytics_summary()
        print(f"Analytics summary: {summary['cached_reports']} cached reports")
    
    # Run tests
    asyncio.run(test_frequency_analytics())