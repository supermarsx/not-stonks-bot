"""
Core Analytics Engine

Central orchestrator for all analytics and reporting functionality.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np

from ..performance.performance_analytics import PerformanceAnalytics
from ..attribution.attribution_analysis import AttributionAnalysis
from ..execution.execution_quality import ExecutionQualityAnalyzer
from ..impact.market_impact import MarketImpactAnalyzer
from ..reporting.automated_reporter import AutomatedReporter
from ..optimization.portfolio_optimizer import PortfolioOptimizer
from ..risk.risk_dashboards import RiskDashboardGenerator
from ..integration.matrix_integration import MatrixIntegration
from .config import AnalyticsConfig

logger = logging.getLogger(__name__)


class AnalyticsEngine:
    """
    Advanced Analytics Engine
    
    Central orchestrator for all trading analytics, reporting, and optimization tools.
    Provides real-time analytics, batch processing, and automated reporting capabilities.
    """
    
    def __init__(self, config: AnalyticsConfig = None):
        """Initialize analytics engine"""
        self.config = config or AnalyticsConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize component analyzers
        self.performance_analytics = PerformanceAnalytics(self.config)
        self.attribution_analysis = AttributionAnalysis(self.config)
        self.execution_quality = ExecutionQualityAnalyzer(self.config)
        self.market_impact = MarketImpactAnalyzer(self.config)
        self.automated_reporter = AutomatedReporter(self.config)
        self.portfolio_optimizer = PortfolioOptimizer(self.config)
        self.risk_dashboards = RiskDashboardGenerator(self.config)
        self.matrix_integration = MatrixIntegration(self.config)
        
        # Thread pool for concurrent processing
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_calculations)
        
        # Internal state
        self.is_running = False
        self.last_update = None
        self.analytics_cache = {}
        
        self.logger.info("Analytics Engine initialized")
    
    async def start(self):
        """Start the analytics engine"""
        if self.is_running:
            self.logger.warning("Analytics Engine is already running")
            return
        
        self.is_running = True
        self.logger.info("Starting Analytics Engine")
        
        # Start background tasks
        tasks = []
        if self.config.enable_real_time:
            tasks.append(self._real_time_loop())
        
        if self.config.enable_batch_processing:
            tasks.append(self._batch_processing_loop())
        
        if self.config.enable_automatic_reports:
            tasks.append(self._report_generation_loop())
        
        await asyncio.gather(*tasks)
    
    async def stop(self):
        """Stop the analytics engine"""
        self.is_running = False
        self.logger.info("Analytics Engine stopped")
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
    
    async def _real_time_loop(self):
        """Real-time analytics loop"""
        while self.is_running:
            try:
                await self._update_real_time_analytics()
                await asyncio.sleep(self.config.real_time_update_interval)
            except Exception as e:
                self.logger.error(f"Error in real-time analytics loop: {e}")
                await asyncio.sleep(10)
    
    async def _batch_processing_loop(self):
        """Batch processing loop"""
        while self.is_running:
            try:
                await self._run_batch_analytics()
                await asyncio.sleep(self.config.batch_processing_interval)
            except Exception as e:
                self.logger.error(f"Error in batch processing loop: {e}")
                await asyncio.sleep(60)
    
    async def _report_generation_loop(self):
        """Automated report generation loop"""
        while self.is_running:
            try:
                await self._generate_scheduled_reports()
                await asyncio.sleep(self.config.report_generation_interval)
            except Exception as e:
                self.logger.error(f"Error in report generation loop: {e}")
                await asyncio.sleep(300)
    
    async def _update_real_time_analytics(self):
        """Update real-time analytics"""
        self.logger.debug("Updating real-time analytics")
        
        # Run concurrent analytics updates
        tasks = [
            self.performance_analytics.update_real_time_metrics(),
            self.execution_quality.update_real_time_metrics(),
            self.market_impact.update_real_time_metrics(),
            self.risk_dashboards.update_real_time_dashboard()
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        self.last_update = datetime.now()
        
        # Update Matrix integration
        await self.matrix_integration.push_real_time_updates()
    
    async def _run_batch_analytics(self):
        """Run comprehensive batch analytics"""
        self.logger.info("Running batch analytics processing")
        
        # Run heavy analytics in parallel
        tasks = [
            self.performance_analytics.run_comprehensive_analysis(),
            self.attribution_analysis.run_attribution_analysis(),
            self.portfolio_optimizer.run_optimization_cycle(),
            self.risk_dashboards.generate_comprehensive_dashboards()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Cache results for faster access
        cache_keys = [
            'comprehensive_performance',
            'attribution_analysis',
            'portfolio_optimization',
            'risk_dashboards'
        ]
        
        for i, result in enumerate(results):
            if not isinstance(result, Exception):
                self.analytics_cache[cache_keys[i]] = result
    
    async def _generate_scheduled_reports(self):
        """Generate scheduled reports"""
        current_time = datetime.now()
        
        # Check if it's time for different report types
        reports_to_generate = []
        
        # Daily reports (every day at 6 PM)
        if current_time.hour == 18:
            reports_to_generate.append('daily')
        
        # Weekly reports (every Sunday at 8 PM)
        if current_time.weekday() == 6 and current_time.hour == 20:
            reports_to_generate.append('weekly')
        
        # Monthly reports (every 1st at 9 AM)
        if current_time.day == 1 and current_time.hour == 9:
            reports_to_generate.append('monthly')
        
        if reports_to_generate:
            await self.automated_reporter.generate_scheduled_reports(reports_to_generate)
    
    # Public API Methods
    
    async def get_portfolio_performance(self, 
                                      portfolio_id: str,
                                      period: str = "1Y") -> Dict[str, Any]:
        """
        Get comprehensive portfolio performance analytics
        
        Args:
            portfolio_id: Portfolio identifier
            period: Analysis period (1D, 1W, 1M, 3M, 6M, 1Y, YTD, ALL)
            
        Returns:
            Comprehensive performance metrics and analytics
        """
        try:
            self.logger.info(f"Generating portfolio performance for {portfolio_id}, period: {period}")
            
            # Run performance analysis
            performance_data = await self.performance_analytics.analyze_portfolio_performance(
                portfolio_id, period
            )
            
            # Get attribution analysis
            attribution_data = await self.attribution_analysis.analyze_portfolio_attribution(
                portfolio_id, period
            )
            
            # Get risk metrics
            risk_metrics = await self.risk_dashboards.get_portfolio_risk_metrics(
                portfolio_id, period
            )
            
            # Combine results
            result = {
                'portfolio_id': portfolio_id,
                'period': period,
                'performance': performance_data,
                'attribution': attribution_data,
                'risk_metrics': risk_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating portfolio performance: {e}")
            raise
    
    async def analyze_trade_execution_quality(self,
                                            trade_id: str = None,
                                            strategy_id: str = None,
                                            period: str = "1M") -> Dict[str, Any]:
        """
        Analyze trade execution quality and TCA metrics
        
        Args:
            trade_id: Specific trade ID (optional)
            strategy_id: Strategy identifier (optional)
            period: Analysis period
            
        Returns:
            Trade execution quality metrics
        """
        try:
            self.logger.info(f"Analyzing trade execution quality for trade: {trade_id}")
            
            analysis = await self.execution_quality.analyze_execution_quality(
                trade_id=trade_id,
                strategy_id=strategy_id,
                period=period
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing trade execution quality: {e}")
            raise
    
    async def analyze_market_impact(self,
                                  trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market impact of trades
        
        Args:
            trade_data: Trade information for impact analysis
            
        Returns:
            Market impact analysis results
        """
        try:
            self.logger.info("Analyzing market impact")
            
            impact_analysis = await self.market_impact.analyze_market_impact(trade_data)
            
            return impact_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing market impact: {e}")
            raise
    
    async def generate_custom_report(self,
                                   report_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate custom analytics report
        
        Args:
            report_config: Report configuration including type, parameters, format
            
        Returns:
            Generated report data
        """
        try:
            self.logger.info("Generating custom report")
            
            report = await self.automated_reporter.generate_custom_report(report_config)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating custom report: {e}")
            raise
    
    async def optimize_portfolio(self,
                               portfolio_data: Dict[str, Any],
                               optimization_objectives: List[str] = None) -> Dict[str, Any]:
        """
        Run portfolio optimization
        
        Args:
            portfolio_data: Current portfolio holdings and constraints
            optimization_objectives: Optimization objectives (risk, return, etc.)
            
        Returns:
            Optimization results and recommendations
        """
        try:
            self.logger.info("Running portfolio optimization")
            
            optimization_result = await self.portfolio_optimizer.optimize_portfolio(
                portfolio_data, optimization_objectives
            )
            
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"Error optimizing portfolio: {e}")
            raise
    
    async def get_real_time_dashboard_data(self) -> Dict[str, Any]:
        """
        Get real-time dashboard data for Matrix integration
        
        Returns:
            Real-time analytics data
        """
        try:
            dashboard_data = await self.risk_dashboards.get_real_time_dashboard_data()
            
            # Add Matrix integration specific formatting
            formatted_data = await self.matrix_integration.format_dashboard_data(dashboard_data)
            
            return formatted_data
            
        except Exception as e:
            self.logger.error(f"Error getting real-time dashboard data: {e}")
            raise
    
    def get_cached_analytics(self, cache_key: str) -> Optional[Any]:
        """Get cached analytics result"""
        return self.analytics_cache.get(cache_key)
    
    def clear_cache(self, cache_key: str = None):
        """Clear analytics cache"""
        if cache_key:
            self.analytics_cache.pop(cache_key, None)
        else:
            self.analytics_cache.clear()
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for analytics engine"""
        try:
            health_status = {
                'engine_status': 'running' if self.is_running else 'stopped',
                'last_update': self.last_update.isoformat() if self.last_update else None,
                'components': {
                    'performance_analytics': await self.performance_analytics.health_check(),
                    'attribution_analysis': await self.attribution_analysis.health_check(),
                    'execution_quality': await self.execution_quality.health_check(),
                    'market_impact': await self.market_impact.health_check(),
                    'automated_reporter': await self.automated_reporter.health_check(),
                    'portfolio_optimizer': await self.portfolio_optimizer.health_check(),
                    'risk_dashboards': await self.risk_dashboards.health_check(),
                    'matrix_integration': await self.matrix_integration.health_check()
                }
            }
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Error in health check: {e}")
            return {'status': 'error', 'error': str(e)}