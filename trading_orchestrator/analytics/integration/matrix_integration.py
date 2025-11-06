"""
Matrix Integration Module

Provides integration with React Matrix Command Center including:
- Real-time data visualization
- Interactive charts and graphs
- Drill-down capabilities
- Filter and search functionality
- Export and sharing features
- Mobile-responsive design
- Real-time updates via WebSocket
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import json
import websockets
from websockets.server import serve
import pandas as pd
import numpy as np

from ..core.config import AnalyticsConfig

logger = logging.getLogger(__name__)


@dataclass
class ChartData:
    """Chart data container for Matrix visualization"""
    chart_id: str
    chart_type: str  # 'line', 'bar', 'pie', 'scatter', 'heatmap', 'treemap'
    title: str
    data: List[Dict[str, Any]]
    x_axis: str
    y_axis: str
    color_scheme: str = 'default'
    interactive: bool = True
    drill_down_enabled: bool = False
    real_time: bool = True


@dataclass
class DashboardWidget:
    """Dashboard widget configuration"""
    widget_id: str
    widget_type: str  # 'chart', 'table', 'metric', 'alert'
    title: str
    position: Dict[str, int]  # {'x': 0, 'y': 0, 'width': 4, 'height': 3}
    data_source: str
    refresh_interval: int = 30  # seconds
    config: Dict[str, Any] = None


@dataclass
class FilterConfig:
    """Filter configuration for Matrix dashboard"""
    filter_id: str
    filter_type: str  # 'date_range', 'multi_select', 'slider', 'search'
    field: str
    label: str
    options: List[Dict[str, Any]] = None
    default_value: Any = None
    enabled: bool = True


@dataclass
class RealTimeUpdate:
    """Real-time update message"""
    update_type: str  # 'price', 'pnl', 'position', 'alert'
    timestamp: datetime
    data: Dict[str, Any]
    widget_id: Optional[str] = None
    priority: str = 'normal'  # 'low', 'normal', 'high', 'critical'


class MatrixIntegration:
    """
    Matrix Integration Layer
    
    Provides seamless integration with React Matrix Command Center:
    - Real-time data visualization
    - Interactive charts and graphs
    - Drill-down capabilities
    - Filter and search functionality
    - Export and sharing features
    - Mobile-responsive design
    - Real-time updates via WebSocket
    """
    
    def __init__(self, config: AnalyticsConfig):
        """Initialize Matrix integration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # WebSocket server for real-time updates
        self.websocket_server = None
        self.connected_clients = set()
        
        # Dashboard configurations
        self.dashboard_templates = self._initialize_dashboard_templates()
        
        # Chart configurations
        self.chart_templates = self._initialize_chart_templates()
        
        # Real-time data streams
        self.data_streams = {}
        
        # WebSocket message handlers
        self.message_handlers = {
            'subscribe': self._handle_subscription,
            'unsubscribe': self._handle_unsubscription,
            'filter_update': self._handle_filter_update,
            'export_request': self._handle_export_request,
            'dashboard_config': self._handle_dashboard_config
        }
        
        self.logger.info("Matrix Integration initialized")
    
    def _initialize_dashboard_templates(self) -> Dict[str, List[DashboardWidget]]:
        """Initialize pre-built dashboard templates"""
        templates = {
            'executive_dashboard': [
                DashboardWidget(
                    widget_id='kpi_summary',
                    widget_type='metric',
                    title='Key Performance Indicators',
                    position={'x': 0, 'y': 0, 'width': 12, 'height': 2},
                    data_source='performance_analytics',
                    refresh_interval=60,
                    config={'metrics': ['total_return', 'sharpe_ratio', 'max_drawdown']}
                ),
                DashboardWidget(
                    widget_id='pnl_chart',
                    widget_type='chart',
                    title='Portfolio P&L Trend',
                    position={'x': 0, 'y': 2, 'width': 8, 'height': 4},
                    data_source='real_time_pnl',
                    refresh_interval=30,
                    config={'chart_type': 'line', 'period': '1D'}
                ),
                DashboardWidget(
                    widget_id='risk_metrics',
                    widget_type='chart',
                    title='Risk Metrics',
                    position={'x': 8, 'y': 2, 'width': 4, 'height': 4},
                    data_source='risk_dashboards',
                    refresh_interval=60,
                    config={'chart_type': 'gauge'}
                ),
                DashboardWidget(
                    widget_id='positions_table',
                    widget_type='table',
                    title='Current Positions',
                    position={'x': 0, 'y': 6, 'width': 12, 'height': 4},
                    data_source='positions',
                    refresh_interval=30,
                    config={'sortable': True, 'filterable': True}
                )
            ],
            'risk_dashboard': [
                DashboardWidget(
                    widget_id='var_monitor',
                    widget_type='chart',
                    title='VaR Monitoring',
                    position={'x': 0, 'y': 0, 'width': 6, 'height': 4},
                    data_source='var_analysis',
                    refresh_interval=60,
                    config={'chart_type': 'area', 'confidence_levels': [95, 99]}
                ),
                DashboardWidget(
                    widget_id='stress_test_results',
                    widget_type='chart',
                    title='Stress Test Results',
                    position={'x': 6, 'y': 0, 'width': 6, 'height': 4},
                    data_source='stress_testing',
                    refresh_interval=300,
                    config={'chart_type': 'bar', 'sortable': True}
                ),
                DashboardWidget(
                    widget_id='concentration_risk',
                    widget_type='chart',
                    title='Concentration Risk',
                    position={'x': 0, 'y': 4, 'width': 12, 'height': 4},
                    data_source='concentration_analysis',
                    refresh_interval=300,
                    config={'chart_type': 'treemap'}
                ),
                DashboardWidget(
                    widget_id='correlation_matrix',
                    widget_type='chart',
                    title='Correlation Matrix',
                    position={'x': 0, 'y': 8, 'width': 12, 'height': 4},
                    data_source='correlation_analysis',
                    refresh_interval=300,
                    config={'chart_type': 'heatmap'}
                )
            ],
            'execution_dashboard': [
                DashboardWidget(
                    widget_id='implementation_shortfall',
                    widget_type='chart',
                    title='Implementation Shortfall',
                    position={'x': 0, 'y': 0, 'width': 6, 'height': 4},
                    data_source='execution_quality',
                    refresh_interval=60,
                    config={'chart_type': 'waterfall'}
                ),
                DashboardWidget(
                    widget_id='market_impact',
                    widget_type='chart',
                    title='Market Impact Analysis',
                    position={'x': 6, 'y': 0, 'width': 6, 'height': 4},
                    data_source='market_impact',
                    refresh_interval=60,
                    config={'chart_type': 'scatter'}
                ),
                DashboardWidget(
                    widget_id='execution_algorithms',
                    widget_type='chart',
                    title='Algorithm Performance',
                    position={'x': 0, 'y': 4, 'width': 12, 'height': 4},
                    data_source='execution_quality',
                    refresh_interval=300,
                    config={'chart_type': 'bar', 'grouped': True}
                ),
                DashboardWidget(
                    widget_id='trade_log',
                    widget_type='table',
                    title='Recent Trades',
                    position={'x': 0, 'y': 8, 'width': 12, 'height': 4},
                    data_source='trade_execution',
                    refresh_interval=30,
                    config={'real_time': True, 'exportable': True}
                )
            ]
        }
        
        return templates
    
    def _initialize_chart_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize chart type templates"""
        templates = {
            'line_chart': {
                'default_config': {
                    'responsive': True,
                    'maintainAspectRatio': False,
                    'interaction': {'mode': 'index', 'intersect': False},
                    'scales': {
                        'x': {'display': True, 'title': {'display': True}},
                        'y': {'display': True, 'title': {'display': True}}
                    }
                },
                'animation': {'duration': 750},
                'colors': ['#3B82F6', '#10B981', '#F59E0B', '#EF4444']
            },
            'bar_chart': {
                'default_config': {
                    'responsive': True,
                    'maintainAspectRatio': False,
                    'scales': {
                        'x': {'stacked': False},
                        'y': {'stacked': False, 'beginAtZero': True}
                    }
                },
                'animation': {'duration': 1000},
                'colors': ['#6366F1', '#8B5CF6', '#EC4899', '#F97316']
            },
            'pie_chart': {
                'default_config': {
                    'responsive': True,
                    'maintainAspectRatio': False,
                    'plugins': {
                        'legend': {'position': 'right'},
                        'tooltip': {'callbacks': {'label': 'context.callback'}}
                    }
                },
                'animation': {'duration': 1200},
                'colors': ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899']
            },
            'heatmap': {
                'default_config': {
                    'responsive': True,
                    'maintainAspectRatio': False,
                    'scales': {
                        'x': {'type': 'category'},
                        'y': {'type': 'category'}
                    },
                    'plugins': {
                        'legend': {'display': False},
                        'tooltip': {'enabled': True}
                    }
                },
                'color_scheme': 'RdYlBu_r'
            },
            'treemap': {
                'default_config': {
                    'responsive': True,
                    'maintainAspectRatio': False,
                    'plugins': {
                        'legend': {'display': False},
                        'tooltip': {'callbacks': {'title': 'context.raw.g', 'label': 'context.raw.v'}}
                    }
                }
            }
        }
        
        return templates
    
    async def start_websocket_server(self, host: str = "localhost", port: int = 8765):
        """Start WebSocket server for real-time updates"""
        try:
            self.websocket_server = await serve(self._handle_websocket_connection, host, port)
            self.logger.info(f"Matrix WebSocket server started on {host}:{port}")
            return self.websocket_server
            
        except Exception as e:
            self.logger.error(f"Error starting WebSocket server: {e}")
            raise
    
    async def _handle_websocket_connection(self, websocket, path):
        """Handle WebSocket connection"""
        self.connected_clients.add(websocket)
        self.logger.info(f"Client connected: {websocket.remote_address}")
        
        try:
            async for message in websocket:
                await self._process_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.connected_clients.remove(websocket)
            self.logger.info(f"Client disconnected: {websocket.remote_address}")
    
    async def _process_message(self, websocket, message: str):
        """Process incoming WebSocket message"""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            message_data = data.get('data', {})
            
            if message_type in self.message_handlers:
                response = await self.message_handlers[message_type](websocket, message_data)
                if response:
                    await websocket.send(json.dumps(response))
            else:
                self.logger.warning(f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON message: {e}")
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
    
    async def _handle_subscription(self, websocket, data: Dict[str, Any]):
        """Handle dashboard subscription"""
        subscriptions = data.get('subscriptions', [])
        
        for subscription in subscriptions:
            stream_name = subscription.get('stream')
            if stream_name not in self.data_streams:
                self.data_streams[stream_name] = set()
            self.data_streams[stream_name].add(websocket)
        
        return {
            'type': 'subscription_confirmed',
            'data': {'subscribed_streams': subscriptions}
        }
    
    async def _handle_unsubscription(self, websocket, data: Dict[str, Any]):
        """Handle dashboard unsubscription"""
        unsubscriptions = data.get('unsubscriptions', [])
        
        for unsubscription in unsubscriptions:
            stream_name = unsubscription.get('stream')
            if stream_name in self.data_streams and websocket in self.data_streams[stream_name]:
                self.data_streams[stream_name].remove(websocket)
        
        return {
            'type': 'unsubscription_confirmed',
            'data': {'unsubscribed_streams': unsubscriptions}
        }
    
    async def _handle_filter_update(self, websocket, data: Dict[str, Any]):
        """Handle filter updates"""
        # Broadcast filter updates to all subscribed clients
        await self.broadcast_message({
            'type': 'filter_update',
            'data': data
        })
        
        return {
            'type': 'filter_update_confirmed',
            'data': {'filters_applied': len(data.get('active_filters', []))}
        }
    
    async def _handle_export_request(self, websocket, data: Dict[str, Any]):
        """Handle data export requests"""
        export_config = data.get('export_config', {})
        
        # Mock export generation
        export_result = {
            'export_id': f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'format': export_config.get('format', 'json'),
            'data': self._generate_export_data(export_config),
            'download_url': f"/exports/{export_config.get('export_id', 'unknown')}.{export_config.get('format', 'json')}",
            'expires_at': (datetime.now() + timedelta(hours=24)).isoformat()
        }
        
        return {
            'type': 'export_ready',
            'data': export_result
        }
    
    async def _handle_dashboard_config(self, websocket, data: Dict[str, Any]):
        """Handle dashboard configuration updates"""
        config_type = data.get('config_type')
        config_data = data.get('config_data')
        
        # Save dashboard configuration
        # In practice, this would save to database
        self.logger.info(f"Updated dashboard configuration: {config_type}")
        
        return {
            'type': 'config_saved',
            'data': {'config_type': config_type, 'status': 'success'}
        }
    
    async def push_real_time_updates(self):
        """Push real-time updates to connected clients"""
        try:
            # Generate mock real-time data
            update_data = await self._generate_real_time_data()
            
            for stream_name, clients in self.data_streams.items():
                for client in clients:
                    try:
                        message = {
                            'type': 'real_time_update',
                            'stream': stream_name,
                            'timestamp': datetime.now().isoformat(),
                            'data': update_data.get(stream_name, {})
                        }
                        await client.send(json.dumps(message))
                    except Exception as e:
                        self.logger.error(f"Error sending update to client: {e}")
                        self.connected_clients.discard(client)
                        
        except Exception as e:
            self.logger.error(f"Error pushing real-time updates: {e}")
    
    async def _generate_real_time_data(self) -> Dict[str, Any]:
        """Generate real-time data for different streams"""
        timestamp = datetime.now()
        
        # Portfolio P&L stream
        pnl_data = {
            'timestamp': timestamp.isoformat(),
            'total_pnl': np.random.uniform(-50000, 100000),
            'daily_pnl': np.random.uniform(-10000, 20000),
            'ytd_return': np.random.uniform(0.05, 0.25),
            'positions_count': np.random.randint(20, 50)
        }
        
        # Risk metrics stream
        risk_data = {
            'timestamp': timestamp.isoformat(),
            'var_95': np.random.uniform(-100000, -50000),
            'var_99': np.random.uniform(-150000, -80000),
            'volatility': np.random.uniform(0.15, 0.25),
            'max_drawdown': np.random.uniform(-0.15, -0.05),
            'correlation_avg': np.random.uniform(0.3, 0.7)
        }
        
        # Execution metrics stream
        execution_data = {
            'timestamp': timestamp.isoformat(),
            'trades_today': np.random.randint(10, 100),
            'average_slippage': np.random.uniform(0.001, 0.01),
            'implementation_shortfall': np.random.uniform(0.002, 0.015),
            'market_impact_avg': np.random.uniform(0.001, 0.008),
            'success_rate': np.random.uniform(0.70, 0.95)
        }
        
        # Alert stream
        alert_data = {
            'timestamp': timestamp.isoformat(),
            'new_alerts': np.random.randint(0, 3),
            'critical_alerts': np.random.randint(0, 1),
            'alert_types': ['Risk_Limit', 'Concentration', 'Execution', 'System'],
            'alert_details': [
                {
                    'id': f"ALERT_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                    'type': 'Risk_Limit',
                    'severity': 'Medium',
                    'message': 'VaR utilization above threshold'
                }
            ]
        }
        
        return {
            'portfolio_pnl': pnl_data,
            'risk_metrics': risk_data,
            'execution_quality': execution_data,
            'system_alerts': alert_data
        }
    
    async def format_dashboard_data(self, dashboard_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format data for Matrix dashboard consumption"""
        try:
            formatted_data = {
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'data_version': '1.0',
                    'refresh_interval': 30,
                    'real_time_enabled': True
                },
                'widgets': {},
                'charts': {},
                'filters': self._get_default_filters(),
                'actions': {
                    'export_available': True,
                    'share_enabled': True,
                    'customizable': True
                }
            }
            
            # Format dashboard widgets
            for widget_id, widget_data in dashboard_data.items():
                formatted_data['widgets'][widget_id] = {
                    'id': widget_id,
                    'type': self._determine_widget_type(widget_data),
                    'title': self._generate_widget_title(widget_id),
                    'data': widget_data,
                    'config': self._get_widget_config(widget_id)
                }
            
            # Generate interactive charts
            formatted_data['charts'] = await self._generate_interactive_charts(dashboard_data)
            
            # Add drill-down capabilities
            formatted_data['drill_down'] = await self._setup_drill_down_capabilities(dashboard_data)
            
            return formatted_data
            
        except Exception as e:
            self.logger.error(f"Error formatting dashboard data: {e}")
            raise
    
    def _get_default_filters(self) -> List[FilterConfig]:
        """Get default filter configurations"""
        filters = [
            FilterConfig(
                filter_id='date_range',
                filter_type='date_range',
                field='date',
                label='Date Range',
                default_value={'start': '2024-01-01', 'end': '2024-12-31'}
            ),
            FilterConfig(
                filter_id='portfolio',
                filter_type='multi_select',
                field='portfolio_id',
                label='Portfolio',
                options=[
                    {'value': 'portfolio_1', 'label': 'Main Portfolio'},
                    {'value': 'portfolio_2', 'label': 'Growth Portfolio'},
                    {'value': 'portfolio_3', 'label': 'Income Portfolio'}
                ],
                default_value=['portfolio_1']
            ),
            FilterConfig(
                filter_id='risk_level',
                filter_type='slider',
                field='risk_tolerance',
                label='Risk Tolerance',
                default_value=50
            )
        ]
        
        return filters
    
    async def _generate_interactive_charts(self, data: Dict[str, Any]) -> Dict[str, ChartData]:
        """Generate interactive charts for Matrix visualization"""
        charts = {}
        
        # Portfolio performance chart
        chart_id = 'portfolio_performance_chart'
        performance_data = await self._generate_performance_chart_data()
        charts[chart_id] = ChartData(
            chart_id=chart_id,
            chart_type='line',
            title='Portfolio Performance',
            data=performance_data,
            x_axis='Date',
            y_axis='Return (%)',
            interactive=True,
            drill_down_enabled=True,
            real_time=True
        )
        
        # Risk metrics chart
        chart_id = 'risk_metrics_chart'
        risk_data = await self._generate_risk_chart_data()
        charts[chart_id] = ChartData(
            chart_id=chart_id,
            chart_type='bar',
            title='Risk Metrics',
            data=risk_data,
            x_axis='Metric',
            y_axis='Value',
            interactive=True,
            real_time=True
        )
        
        # Sector allocation chart
        chart_id = 'sector_allocation_chart'
        sector_data = await self._generate_sector_chart_data()
        charts[chart_id] = ChartData(
            chart_id=chart_id,
            chart_type='pie',
            title='Sector Allocation',
            data=sector_data,
            x_axis='Sector',
            y_axis='Weight (%)',
            interactive=True
        )
        
        # Correlation heatmap
        chart_id = 'correlation_heatmap'
        correlation_data = await self._generate_correlation_heatmap_data()
        charts[chart_id] = ChartData(
            chart_id=chart_id,
            chart_type='heatmap',
            title='Correlation Matrix',
            data=correlation_data,
            x_axis='Asset',
            y_axis='Asset',
            interactive=True,
            real_time=True
        )
        
        return charts
    
    async def _generate_performance_chart_data(self) -> List[Dict[str, Any]]:
        """Generate portfolio performance chart data"""
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        
        # Generate realistic performance data
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.015, len(dates))
        cumulative_returns = (1 + returns).cumprod() * 100
        
        return [
            {'x': date.isoformat(), 'y': float(cumulative_return)}
            for date, cumulative_return in zip(dates, cumulative_returns)
        ]
    
    async def _generate_risk_chart_data(self) -> List[Dict[str, Any]]:
        """Generate risk metrics chart data"""
        risk_metrics = {
            'VaR 95%': -75000,
            'VaR 99%': -120000,
            'Volatility': 18.5,
            'Max Drawdown': -12.3,
            'Beta': 0.95,
            'Tracking Error': 4.2
        }
        
        return [
            {'x': metric, 'y': value, 'color': '#EF4444' if value < 0 else '#10B981'}
            for metric, value in risk_metrics.items()
        ]
    
    async def _generate_sector_chart_data(self) -> List[Dict[str, Any]]:
        """Generate sector allocation chart data"""
        sectors = {
            'Technology': 28.5,
            'Healthcare': 18.2,
            'Financials': 15.8,
            'Consumer Discretionary': 12.3,
            'Industrials': 9.7,
            'Energy': 6.2,
            'Materials': 4.1,
            'Utilities': 3.2,
            'Real Estate': 2.0
        }
        
        return [
            {'x': sector, 'y': weight, 'color': f'hsl({i * 40}, 70%, 60%)'}
            for i, (sector, weight) in enumerate(sectors.items())
        ]
    
    async def _generate_correlation_heatmap_data(self) -> List[Dict[str, Any]]:
        """Generate correlation heatmap data"""
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JNJ', 'JPM']
        
        # Generate correlation matrix
        np.random.seed(42)
        correlations = np.random.uniform(0.2, 0.8, (len(symbols), len(symbols)))
        correlations = (correlations + correlations.T) / 2  # Make symmetric
        np.fill_diagonal(correlations, 1.0)  # Perfect self-correlation
        
        # Convert to heatmap format
        heatmap_data = []
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols):
                heatmap_data.append({
                    'x': symbol2,
                    'y': symbol1,
                    'v': float(correlations[i, j]),
                    'color': f'rgba(59, 130, 246, {correlations[i, j]})'
                })
        
        return heatmap_data
    
    async def _setup_drill_down_capabilities(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Setup drill-down capabilities for charts"""
        drill_down_config = {
            'portfolio_performance_chart': {
                'enabled': True,
                'drill_levels': [
                    {'level': 1, 'action': 'show_sector_breakdown'},
                    {'level': 2, 'action': 'show_position_details'},
                    {'level': 3, 'action': 'show_trade_history'}
                ]
            },
            'risk_metrics_chart': {
                'enabled': True,
                'drill_levels': [
                    {'level': 1, 'action': 'show_component_breakdown'},
                    {'level': 2, 'action': 'show_historical_trends'}
                ]
            }
        }
        
        return drill_down_config
    
    def _determine_widget_type(self, data: Dict[str, Any]) -> str:
        """Determine widget type based on data"""
        if 'chart' in str(data).lower():
            return 'chart'
        elif 'table' in str(data).lower() or 'positions' in data:
            return 'table'
        elif 'metric' in str(data).lower() or 'kpi' in str(data).lower():
            return 'metric'
        elif 'alert' in str(data).lower():
            return 'alert'
        else:
            return 'metric'  # Default
    
    def _generate_widget_title(self, widget_id: str) -> str:
        """Generate user-friendly widget title"""
        title_mapping = {
            'pnl_summary': 'Portfolio P&L Summary',
            'risk_metrics': 'Risk Metrics',
            'positions_table': 'Current Positions',
            'performance_chart': 'Performance Chart',
            'execution_summary': 'Execution Summary',
            'attribution_analysis': 'Performance Attribution'
        }
        
        return title_mapping.get(widget_id, widget_id.replace('_', ' ').title())
    
    def _get_widget_config(self, widget_id: str) -> Dict[str, Any]:
        """Get widget configuration"""
        base_config = {
            'refreshable': True,
            'exportable': True,
            'customizable': True,
            'real_time': True
        }
        
        # Widget-specific configurations
        widget_configs = {
            'positions_table': {
                'sortable': True,
                'filterable': True,
                'paginated': True,
                'page_size': 25
            },
            'performance_chart': {
                'zoom_enabled': True,
                'pan_enabled': True,
                'show_legends': True,
                'data_point_hover': True
            }
        }
        
        return {**base_config, **widget_configs.get(widget_id, {})}
    
    def _generate_export_data(self, export_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate data for export"""
        # Mock export data generation
        export_format = export_config.get('format', 'json')
        data_source = export_config.get('data_source', 'all')
        
        if export_format == 'json':
            return {
                'export_type': 'json',
                'data_source': data_source,
                'generated_at': datetime.now().isoformat(),
                'record_count': 1250,
                'file_size': '2.3 MB'
            }
        elif export_format == 'csv':
            # Return CSV content as string (mock)
            return {
                'export_type': 'csv',
                'content': 'timestamp,portfolio_id,pnl,return\n2024-01-01,portfolio_1,15000,0.015\n',
                'file_size': '45 KB'
            }
        elif export_format == 'excel':
            # Return Excel metadata (mock)
            return {
                'export_type': 'excel',
                'sheets': ['Summary', 'Positions', 'Performance', 'Risk'],
                'file_size': '1.8 MB'
            }
        else:
            return {'error': 'Unsupported format'}
    
    async def broadcast_message(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        if not self.connected_clients:
            return
        
        disconnected_clients = set()
        
        for client in self.connected_clients:
            try:
                await client.send(json.dumps(message))
            except Exception as e:
                self.logger.error(f"Error broadcasting to client: {e}")
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.connected_clients -= disconnected_clients
    
    async def get_dashboard_template(self, template_name: str) -> List[DashboardWidget]:
        """Get dashboard template by name"""
        return self.dashboard_templates.get(template_name, [])
    
    async def create_custom_dashboard(self, config: Dict[str, Any]) -> str:
        """Create custom dashboard configuration"""
        dashboard_id = f"custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save custom dashboard configuration
        # In practice, this would save to database
        self.logger.info(f"Created custom dashboard: {dashboard_id}")
        
        return dashboard_id
    
    async def share_dashboard(self, dashboard_id: str, share_config: Dict[str, Any]) -> Dict[str, Any]:
        """Share dashboard with specific users or teams"""
        share_result = {
            'dashboard_id': dashboard_id,
            'shared_with': share_config.get('recipients', []),
            'share_type': share_config.get('type', 'view'),  # 'view', 'edit'
            'permissions': share_config.get('permissions', ['view']),
            'expires_at': (datetime.now() + timedelta(days=30)).isoformat(),
            'share_url': f"/shared/{dashboard_id}",
            'status': 'success'
        }
        
        return share_result
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for Matrix integration"""
        try:
            return {
                'status': 'healthy',
                'last_update': datetime.now().isoformat(),
                'websocket_server': 'running' if self.websocket_server else 'stopped',
                'connected_clients': len(self.connected_clients),
                'dashboard_templates': len(self.dashboard_templates),
                'chart_templates': len(self.chart_templates),
                'data_streams': len(self.data_streams)
            }
        except Exception as e:
            self.logger.error(f"Error in Matrix integration health check: {e}")
            return {'status': 'error', 'error': str(e)}