"""
Cost Dashboard - Real-time cost monitoring and visualization system

Provides comprehensive dashboards for cost tracking, monitoring, and analytics.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
import sqlite3
import statistics

from loguru import logger


@dataclass
class DashboardWidget:
    """Dashboard widget configuration"""
    id: str
    title: str
    widget_type: str  # chart, metric, table, alert
    position: Dict[str, int]  # x, y, width, height
    data_source: str
    refresh_interval: int  # seconds
    configuration: Dict[str, Any] = field(default_factory=dict)
    is_visible: bool = True


@dataclass
class DashboardLayout:
    """Dashboard layout configuration"""
    name: str
    widgets: List[DashboardWidget]
    theme: str = 'dark'
    is_default: bool = False
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AlertThreshold:
    """Alert threshold configuration"""
    metric: str
    threshold_value: float
    comparison: str  # greater_than, less_than, equal_to
    severity: str  # info, warning, critical
    enabled: bool = True


class CostDashboard:
    """
    Real-time cost monitoring dashboard system
    
    Features:
    - Real-time cost metrics display
    - Interactive charts and visualizations
    - Alert threshold management
    - Custom dashboard layouts
    - Export and reporting capabilities
    - Mobile-responsive design
    """
    
    def __init__(self, cost_manager, budget_manager, provider_manager, analytics):
        """
        Initialize cost dashboard
        
        Args:
            cost_manager: LLMCostManager instance
            budget_manager: BudgetManager instance
            provider_manager: ProviderManager instance
            analytics: CostAnalytics instance
        """
        self.cost_manager = cost_manager
        self.budget_manager = budget_manager
        self.provider_manager = provider_manager
        self.analytics = analytics
        
        # Dashboard configuration
        self.dashboard_data_path = Path("data/dashboard_configs.json")
        self.dashboard_data_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Dashboard state
        self.dashboards: Dict[str, DashboardLayout] = {}
        self.alert_thresholds: List[AlertThreshold] = []
        self.real_time_updates_enabled = True
        self.update_interval = 30  # seconds
        
        # Current dashboard data cache
        self.current_metrics_cache = {}
        self.last_update = datetime.now()
        
        # Initialize default dashboards
        self._setup_default_dashboards()
        self._setup_default_alert_thresholds()
        
        # Start real-time updates
        asyncio.create_task(self._real_time_update_loop())
        
        logger.info("Cost Dashboard initialized")
    
    def _setup_default_dashboards(self):
        """Setup default dashboard layouts"""
        
        # Main Overview Dashboard
        main_dashboard = DashboardLayout(
            name="Main Overview",
            is_default=True,
            widgets=[
                DashboardWidget(
                    id="current_cost_metric",
                    title="Current Period Cost",
                    widget_type="metric",
                    position={"x": 0, "y": 0, "width": 3, "height": 2},
                    data_source="current_metrics",
                    refresh_interval=30
                ),
                DashboardWidget(
                    id="daily_cost_chart",
                    title="Daily Cost Trend",
                    widget_type="chart",
                    position={"x": 3, "y": 0, "width": 9, "height": 4},
                    data_source="daily_costs",
                    refresh_interval=60,
                    configuration={"chart_type": "line", "time_range": "7d"}
                ),
                DashboardWidget(
                    id="provider_breakdown",
                    title="Cost by Provider",
                    widget_type="chart",
                    position={"x": 0, "y": 2, "width": 6, "height": 4},
                    data_source="provider_costs",
                    refresh_interval=60,
                    configuration={"chart_type": "pie"}
                ),
                DashboardWidget(
                    id="active_alerts",
                    title="Active Alerts",
                    widget_type="alert",
                    position={"x": 9, "y": 2, "width": 3, "height": 4},
                    data_source="alerts",
                    refresh_interval=15
                ),
                DashboardWidget(
                    id="budget_status",
                    title="Budget Status",
                    widget_type="metric",
                    position={"x": 0, "y": 6, "width": 4, "height": 2},
                    data_source="budget_status",
                    refresh_interval=30
                ),
                DashboardWidget(
                    id="top_models",
                    title="Top Models by Cost",
                    widget_type="table",
                    position={"x": 4, "y": 6, "width": 8, "height": 2},
                    data_source="model_performance",
                    refresh_interval=120
                )
            ]
        )
        
        # Analytics Dashboard
        analytics_dashboard = DashboardLayout(
            name="Analytics & Reports",
            widgets=[
                DashboardWidget(
                    id="cost_forecast",
                    title="Cost Forecast",
                    widget_type="chart",
                    position={"x": 0, "y": 0, "width": 12, "height": 4},
                    data_source="cost_forecasts",
                    refresh_interval=300,
                    configuration={"chart_type": "forecast", "horizon_days": 30}
                ),
                DashboardWidget(
                    id="anomaly_detection",
                    title="Anomaly Detection",
                    widget_type="chart",
                    position={"x": 0, "y": 4, "width": 6, "height": 4},
                    data_source="anomalies",
                    refresh_interval=60
                ),
                DashboardWidget(
                    id="performance_benchmarks",
                    title="Performance Benchmarks",
                    widget_type="chart",
                    position={"x": 6, "y": 4, "width": 6, "height": 4},
                    data_source="benchmarks",
                    refresh_interval=600
                ),
                DashboardWidget(
                    id="efficiency_trends",
                    title="Efficiency Trends",
                    widget_type="chart",
                    position={"x": 0, "y": 8, "width": 12, "height": 4},
                    data_source="efficiency_trends",
                    refresh_interval=180
                )
            ]
        )
        
        self.dashboards["Main Overview"] = main_dashboard
        self.dashboards["Analytics & Reports"] = analytics_dashboard
    
    def _setup_default_alert_thresholds(self):
        """Setup default alert thresholds"""
        
        self.alert_thresholds = [
            AlertThreshold(
                metric="current_cost",
                threshold_value=100.0,
                comparison="greater_than",
                severity="critical"
            ),
            AlertThreshold(
                metric="cost_per_hour",
                threshold_value=50.0,
                comparison="greater_than",
                severity="warning"
            ),
            AlertThreshold(
                metric="budget_utilization",
                threshold_value=0.9,  # 90%
                comparison="greater_than",
                severity="warning"
            ),
            AlertThreshold(
                metric="anomaly_count",
                threshold_value=5.0,
                comparison="greater_than",
                severity="info"
            )
        ]
    
    async def _real_time_update_loop(self):
        """Background task for real-time dashboard updates"""
        while True:
            try:
                if self.real_time_updates_enabled:
                    await self._update_dashboard_data()
                
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Dashboard update loop error: {e}")
                await asyncio.sleep(self.update_interval)
    
    async def _update_dashboard_data(self):
        """Update cached dashboard data"""
        try:
            # Update current metrics
            self.current_metrics_cache = {
                'current_metrics': asdict(self.cost_manager.get_current_metrics()),
                'budget_status': await self.budget_manager.get_budget_status(),
                'provider_health': await self.provider_manager.get_provider_health_report(),
                'active_alerts': [asdict(alert) for alert in self.cost_manager.get_active_alerts()],
                'last_update': datetime.now().isoformat()
            }
            
            self.last_update = datetime.now()
            
        except Exception as e:
            logger.error(f"Dashboard data update error: {e}")
    
    async def get_dashboard_data(self, dashboard_name: str = None) -> Dict[str, Any]:
        """
        Get dashboard data for specified dashboard
        
        Args:
            dashboard_name: Name of dashboard to get data for (uses default if None)
            
        Returns:
            Complete dashboard data with widgets and current metrics
        """
        if dashboard_name is None:
            # Get default dashboard
            dashboard_name = next(
                (name for name, layout in self.dashboards.items() if layout.is_default),
                list(self.dashboards.keys())[0]
            )
        
        if dashboard_name not in self.dashboards:
            raise ValueError(f"Dashboard not found: {dashboard_name}")
        
        layout = self.dashboards[dashboard_name]
        
        # Ensure data is fresh
        if datetime.now() - self.last_update > timedelta(seconds=30):
            await self._update_dashboard_data()
        
        dashboard_data = {
            'dashboard_info': {
                'name': layout.name,
                'theme': layout.theme,
                'widget_count': len(layout.widgets),
                'last_updated': self.last_update.isoformat()
            },
            'current_metrics': self.current_metrics_cache,
            'widgets': []
        }
        
        # Get data for each widget
        for widget in layout.widgets:
            if not widget.is_visible:
                continue
            
            try:
                widget_data = await self._get_widget_data(widget)
                dashboard_data['widgets'].append({
                    'widget_id': widget.id,
                    'title': widget.title,
                    'type': widget.widget_type,
                    'position': widget.position,
                    'data': widget_data,
                    'refresh_interval': widget.refresh_interval
                })
            except Exception as e:
                logger.error(f"Error getting widget data for {widget.id}: {e}")
                dashboard_data['widgets'].append({
                    'widget_id': widget.id,
                    'title': widget.title,
                    'type': widget.widget_type,
                    'position': widget.position,
                    'data': {'error': str(e)},
                    'refresh_interval': widget.refresh_interval
                })
        
        return dashboard_data
    
    async def _get_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get data for a specific widget"""
        
        if widget.data_source == "current_metrics":
            return await self._get_current_metrics_data()
        elif widget.data_source == "daily_costs":
            return await self._get_daily_costs_data(widget.configuration)
        elif widget.data_source == "provider_costs":
            return await self._get_provider_costs_data()
        elif widget.data_source == "alerts":
            return await self._get_alerts_data()
        elif widget.data_source == "budget_status":
            return await self._get_budget_status_data()
        elif widget.data_source == "model_performance":
            return await self._get_model_performance_data()
        elif widget.data_source == "cost_forecasts":
            return await self._get_cost_forecasts_data()
        elif widget.data_source == "anomalies":
            return await self._get_anomalies_data()
        elif widget.data_source == "benchmarks":
            return await self._get_benchmarks_data()
        else:
            return {'error': f'Unknown data source: {widget.data_source}'}
    
    async def _get_current_metrics_data(self) -> Dict[str, Any]:
        """Get current metrics for dashboard"""
        metrics = self.cost_manager.get_current_metrics()
        
        return {
            'total_cost': metrics.total_cost,
            'total_tokens': metrics.total_tokens,
            'request_count': metrics.request_count,
            'avg_cost_per_request': metrics.average_cost_per_request,
            'cost_per_1k_tokens': metrics.cost_per_1k_tokens,
            'period_start': metrics.period_start.isoformat(),
            'period_end': metrics.period_end.isoformat(),
            'cost_breakdown': {
                'by_provider': metrics.provider_breakdown,
                'by_model': metrics.model_breakdown
            }
        }
    
    async def _get_daily_costs_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get daily costs data for chart"""
        days_back = int(config.get('time_range', '7d').replace('d', ''))
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Get daily cost data
        daily_metrics = await self.cost_manager.get_historical_metrics(start_date, end_date)
        
        chart_data = []
        for metric in daily_metrics:
            chart_data.append({
                'date': metric.period_start.date().isoformat(),
                'cost': metric.total_cost,
                'tokens': metric.total_tokens,
                'requests': metric.request_count
            })
        
        return {
            'chart_type': 'line',
            'data': chart_data,
            'series': [
                {'name': 'Cost', 'field': 'cost'},
                {'name': 'Tokens', 'field': 'tokens', 'yAxis': 1}
            ]
        }
    
    async def _get_provider_costs_data(self) -> Dict[str, Any]:
        """Get provider cost breakdown data"""
        metrics = self.cost_manager.get_current_metrics()
        
        pie_data = []
        for provider, data in metrics.provider_breakdown.items():
            pie_data.append({
                'name': provider,
                'value': data['total_cost'],
                'tokens': data['total_tokens'],
                'requests': data['request_count']
            })
        
        return {
            'chart_type': 'pie',
            'data': pie_data,
            'total_cost': sum(p['value'] for p in pie_data)
        }
    
    async def _get_alerts_data(self) -> Dict[str, Any]:
        """Get alerts data"""
        active_alerts = self.cost_manager.get_active_alerts()
        
        # Group alerts by severity
        alerts_by_severity = {'info': [], 'warning': [], 'critical': []}
        for alert in active_alerts[:20]:  # Latest 20 alerts
            alerts_by_severity[alert.level.value].append({
                'id': alert.id,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'type': alert.event_type.value
            })
        
        return {
            'total_alerts': len(active_alerts),
            'alerts_by_severity': alerts_by_severity,
            'recent_alerts': [asdict(alert) for alert in active_alerts[:10]]
        }
    
    async def _get_budget_status_data(self) -> Dict[str, Any]:
        """Get budget status data"""
        budget_status = await self.budget_manager.get_budget_status()
        
        return {
            'current_profile': budget_status.get('active_profile', 'Unknown'),
            'tier': budget_status.get('tier', 'unknown'),
            'daily_spent': budget_status.get('daily_spent', '$0.00'),
            'daily_limit': budget_status.get('daily_limit', '$0.00'),
            'daily_percentage': budget_status.get('daily_percentage', '0%'),
            'emergency_mode': budget_status.get('emergency_mode', False),
            'auto_scaling': budget_status.get('auto_scaling', False)
        }
    
    async def _get_model_performance_data(self) -> Dict[str, Any]:
        """Get model performance data"""
        metrics = self.cost_manager.get_current_metrics()
        
        # Convert to table format
        table_data = []
        for model_name, data in metrics.model_breakdown.items():
            table_data.append({
                'model': model_name,
                'cost': f"${data['total_cost']:.2f}",
                'tokens': f"{data['total_tokens']:,}",
                'requests': data['request_count'],
                'avg_cost_per_request': f"${data['total_cost'] / max(data['request_count'], 1):.4f}"
            })
        
        # Sort by cost (descending)
        table_data.sort(key=lambda x: float(x['cost'].replace('$', '')), reverse=True)
        
        return {
            'columns': ['Model', 'Cost', 'Tokens', 'Requests', 'Avg Cost/Request'],
            'data': table_data
        }
    
    async def _get_cost_forecasts_data(self) -> Dict[str, Any]:
        """Get cost forecast data"""
        try:
            # This would call the forecaster
            forecasts = await self.analytics.cost_forecaster.forecast_costs(
                forecast_horizon_days=30
            )
            
            forecast_data = []
            for forecast in forecasts:
                forecast_data.append({
                    'date': forecast.start_date.date().isoformat(),
                    'predicted_cost': forecast.predicted_cost,
                    'confidence_lower': forecast.confidence_interval[0],
                    'confidence_upper': forecast.confidence_interval[1],
                    'confidence_score': forecast.confidence_score
                })
            
            return {
                'chart_type': 'forecast',
                'data': forecast_data,
                'total_projected_cost': sum(f.predicted_cost for f in forecasts)
            }
        except Exception as e:
            return {'error': f'Forecast generation failed: {str(e)}'}
    
    async def _get_anomalies_data(self) -> Dict[str, Any]:
        """Get anomalies data"""
        try:
            # This would call the anomaly detector
            summary = await self.analytics.cost_forecaster.cost_manager.anomaly_detector.get_anomaly_summary()
            
            return summary
        except Exception as e:
            return {'error': f'Anomaly data retrieval failed: {str(e)}'}
    
    async def _get_benchmarks_data(self) -> Dict[str, Any]:
        """Get performance benchmarks data"""
        try:
            benchmarks = await self.analytics.get_performance_benchmark()
            
            return benchmarks
        except Exception as e:
            return {'error': f'Benchmark data retrieval failed: {str(e)}'}
    
    async def check_alert_thresholds(self) -> List[Dict[str, Any]]:
        """Check alert thresholds and return triggered alerts"""
        triggered_alerts = []
        
        for threshold in self.alert_thresholds:
            if not threshold.enabled:
                continue
            
            try:
                current_value = await self._get_metric_value(threshold.metric)
                
                if self._compare_values(current_value, threshold.threshold_value, threshold.comparison):
                    triggered_alerts.append({
                        'metric': threshold.metric,
                        'current_value': current_value,
                        'threshold': threshold.threshold_value,
                        'comparison': threshold.comparison,
                        'severity': threshold.severity,
                        'triggered_at': datetime.now().isoformat()
                    })
            except Exception as e:
                logger.error(f"Error checking threshold {threshold.metric}: {e}")
        
        return triggered_alerts
    
    async def _get_metric_value(self, metric_name: str) -> float:
        """Get current value for a metric"""
        metrics = self.cost_manager.get_current_metrics()
        
        if metric_name == "current_cost":
            return metrics.total_cost
        elif metric_name == "cost_per_hour":
            # Calculate cost per hour (simplified)
            hours_elapsed = max((datetime.now() - metrics.period_start).total_seconds() / 3600, 1)
            return metrics.total_cost / hours_elapsed
        elif metric_name == "budget_utilization":
            budget_status = await self.budget_manager.get_budget_status()
            daily_percentage = float(budget_status.get('daily_percentage', '0%').replace('%', '')) / 100
            return daily_percentage
        else:
            # Default to total cost
            return metrics.total_cost
    
    def _compare_values(self, current: float, threshold: float, comparison: str) -> bool:
        """Compare current value against threshold"""
        if comparison == "greater_than":
            return current > threshold
        elif comparison == "less_than":
            return current < threshold
        elif comparison == "equal_to":
            return abs(current - threshold) < 0.01
        else:
            return False
    
    async def create_custom_dashboard(
        self,
        name: str,
        widgets: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create a custom dashboard"""
        
        dashboard_widgets = []
        for widget_config in widgets:
            widget = DashboardWidget(
                id=widget_config['id'],
                title=widget_config['title'],
                widget_type=widget_config['type'],
                position=widget_config['position'],
                data_source=widget_config['data_source'],
                refresh_interval=widget_config.get('refresh_interval', 60),
                configuration=widget_config.get('configuration', {})
            )
            dashboard_widgets.append(widget)
        
        dashboard = DashboardLayout(
            name=name,
            widgets=dashboard_widgets,
            theme='dark'
        )
        
        self.dashboards[name] = dashboard
        
        # Save configuration
        await self._save_dashboard_configurations()
        
        logger.info(f"Created custom dashboard: {name}")
        
        return {
            'dashboard_id': name,
            'name': name,
            'widget_count': len(dashboard_widgets),
            'message': 'Dashboard created successfully'
        }
    
    async def _save_dashboard_configurations(self):
        """Save dashboard configurations to file"""
        try:
            config_data = {
                'dashboards': {},
                'alert_thresholds': [
                    {
                        'metric': t.metric,
                        'threshold_value': t.threshold_value,
                        'comparison': t.comparison,
                        'severity': t.severity,
                        'enabled': t.enabled
                    }
                    for t in self.alert_thresholds
                ]
            }
            
            for name, layout in self.dashboards.items():
                config_data['dashboards'][name] = {
                    'name': layout.name,
                    'theme': layout.theme,
                    'is_default': layout.is_default,
                    'widgets': [
                        {
                            'id': w.id,
                            'title': w.title,
                            'widget_type': w.widget_type,
                            'position': w.position,
                            'data_source': w.data_source,
                            'refresh_interval': w.refresh_interval,
                            'configuration': w.configuration,
                            'is_visible': w.is_visible
                        }
                        for w in layout.widgets
                    ]
                }
            
            with open(self.dashboard_data_path, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving dashboard configurations: {e}")
    
    async def load_dashboard_configurations(self):
        """Load dashboard configurations from file"""
        try:
            if not self.dashboard_data_path.exists():
                return
            
            with open(self.dashboard_data_path, 'r') as f:
                config_data = json.load(f)
            
            # Load alert thresholds
            if 'alert_thresholds' in config_data:
                self.alert_thresholds = [
                    AlertThreshold(
                        metric=t['metric'],
                        threshold_value=t['threshold_value'],
                        comparison=t['comparison'],
                        severity=t['severity'],
                        enabled=t.get('enabled', True)
                    )
                    for t in config_data['alert_thresholds']
                ]
            
            # Load dashboards (simplified - would need full widget reconstruction)
            
        except Exception as e:
            logger.error(f"Error loading dashboard configurations: {e}")
    
    async def export_dashboard_data(
        self,
        dashboard_name: str,
        format: str = 'json',
        include_raw_data: bool = True
    ) -> Dict[str, Any]:
        """Export dashboard data in specified format"""
        
        dashboard_data = await self.get_dashboard_data(dashboard_name)
        
        if format == 'json':
            export_data = {
                'dashboard_name': dashboard_name,
                'exported_at': datetime.now().isoformat(),
                'dashboard_data': dashboard_data
            }
            
            if include_raw_data:
                # Add raw metrics data
                export_data['raw_data'] = {
                    'cost_events': await self._get_recent_cost_events(),
                    'budget_profiles': self.budget_manager.get_budget_profiles(),
                    'provider_comparison': self.provider_manager.get_provider_comparison()
                }
            
            return export_data
        
        return {'error': f'Unsupported export format: {format}'}
    
    async def _get_recent_cost_events(self) -> List[Dict[str, Any]]:
        """Get recent cost events for export"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            return await self.cost_manager.export_cost_data(start_date, end_date)
        except Exception as e:
            return {'error': str(e)}
    
    def get_dashboard_list(self) -> List[Dict[str, Any]]:
        """Get list of available dashboards"""
        return [
            {
                'name': layout.name,
                'widget_count': len(layout.widgets),
                'is_default': layout.is_default,
                'created_at': layout.created_at.isoformat(),
                'theme': layout.theme
            }
            for layout in self.dashboards.values()
        ]
    
    async def get_real_time_status(self) -> Dict[str, Any]:
        """Get real-time dashboard status"""
        return {
            'real_time_enabled': self.real_time_updates_enabled,
            'update_interval_seconds': self.update_interval,
            'last_update': self.last_update.isoformat(),
            'data_cache_size': len(self.current_metrics_cache),
            'active_dashboards': len(self.dashboards),
            'alert_thresholds_count': len(self.alert_thresholds)
        }