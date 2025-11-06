"""
Real-time Risk Monitoring System

Advanced real-time risk monitoring including:
- Live risk metrics dashboard
- Real-time alert system
- Risk limit breach notifications
- Automated position sizing based on risk
- Dynamic hedging recommendations
- Risk-adjusted performance attribution
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import asyncio
import websockets
import json
from abc import ABC, abstractmethod

from database.models.trading import Position, Order, Trade
from database.models.risk import RiskEvent, RiskLevel
from database.models.user import User

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class RiskMetricType(Enum):
    """Types of risk metrics."""
    VALUE_AT_RISK = "var"
    CONDITIONAL_VAR = "cvar"
    MAX_DRAWDOWN = "max_drawdown"
    VOLATILITY = "volatility"
    BETA = "beta"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    INFORMATION_RATIO = "information_ratio"
    TRACKING_ERROR = "tracking_error"
    CONCENTRATION_RISK = "concentration_risk"
    LIQUIDITY_RISK = "liquidity_risk"
    CREDIT_RISK = "credit_risk"


@dataclass
class RiskAlert:
    """Risk alert definition."""
    alert_id: str
    alert_type: str
    severity: AlertSeverity
    title: str
    message: str
    metric_value: float
    threshold_value: float
    timestamp: datetime
    user_id: int
    is_acknowledged: bool = False
    is_resolved: bool = False
    resolution_notes: str = ""
    actions_taken: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskMetric:
    """Risk metric definition."""
    metric_type: RiskMetricType
    current_value: float
    previous_value: Optional[float]
    change_pct: float
    trend: str  # 'increasing', 'decreasing', 'stable'
    threshold_warning: float
    threshold_critical: float
    last_updated: datetime
    data_points: pd.Series = field(default_factory=pd.Series)


@dataclass
class RiskDashboard:
    """Real-time risk dashboard data."""
    timestamp: datetime
    user_id: int
    portfolio_value: float
    daily_pnl: float
    daily_pnl_pct: float
    risk_metrics: Dict[str, RiskMetric]
    active_alerts: List[RiskAlert]
    position_summary: Dict[str, Any]
    limit_status: Dict[str, Any]
    performance_metrics: Dict[str, float]
    market_conditions: Dict[str, float]
    recommendations: List[str]


class RiskDataProvider(ABC):
    """Abstract base class for risk data providers."""
    
    @abstractmethod
    async def get_portfolio_data(self, user_id: int) -> Dict[str, Any]:
        """Get current portfolio data."""
        pass
    
    @abstractmethod
    async def get_market_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Get market data for symbols."""
        pass
    
    @abstractmethod
    async def get_positions(self, user_id: int) -> List[Dict[str, Any]]:
        """Get current positions."""
        pass
    
    @abstractmethod
    async def get_orders(self, user_id: int) -> List[Dict[str, Any]]:
        """Get current orders."""
        pass


class RealTimeRiskMonitor:
    """
    Real-time risk monitoring system.
    
    Continuously monitors portfolio risk in real-time and generates
    alerts and recommendations based on predefined thresholds and conditions.
    """
    
    def __init__(self, user_id: int, data_provider: RiskDataProvider):
        """
        Initialize real-time risk monitor.
        
        Args:
            user_id: User identifier
            data_provider: Data provider for portfolio and market data
        """
        self.user_id = user_id
        self.data_provider = data_provider
        self.monitoring_active = False
        self.risk_metrics: Dict[str, RiskMetric] = {}
        self.alerts: List[RiskAlert] = []
        self.alert_callbacks: List[Callable] = []
        self.metric_thresholds = {}
        self.last_update = None
        self.update_interval = 60  # seconds
        
        # Risk calculation engines
        self.var_engine = None
        self.cvar_engine = None
        self.correlation_engine = None
        self.stress_test_engine = None
        
    async def start_monitoring(self, custom_thresholds: Dict[str, float] = None):
        """Start real-time risk monitoring."""
        try:
            self.monitoring_active = True
            
            # Set custom thresholds if provided
            if custom_thresholds:
                self.metric_thresholds.update(custom_thresholds)
            
            logger.info(f"Started real-time risk monitoring for user {self.user_id}")
            
            # Initial data load
            await self._load_initial_data()
            
            # Start monitoring loop
            asyncio.create_task(self._monitoring_loop())
            
        except Exception as e:
            logger.error(f"Risk monitoring start error: {str(e)}")
            raise
    
    async def stop_monitoring(self):
        """Stop real-time risk monitoring."""
        self.monitoring_active = False
        logger.info(f"Stopped real-time risk monitoring for user {self.user_id}")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        try:
            while self.monitoring_active:
                try:
                    # Update risk metrics
                    await self._update_risk_metrics()
                    
                    # Check alerts
                    await self._check_alerts()
                    
                    # Generate recommendations
                    await self._generate_recommendations()
                    
                    # Update dashboard
                    await self._update_dashboard()
                    
                    # Wait for next update
                    await asyncio.sleep(self.update_interval)
                    
                except Exception as e:
                    logger.error(f"Monitoring loop error: {str(e)}")
                    await asyncio.sleep(10)  # Wait before retrying
                    
        except Exception as e:
            logger.error(f"Monitoring loop initialization error: {str(e)}")
    
    async def _load_initial_data(self):
        """Load initial data for monitoring."""
        try:
            # Load portfolio data
            portfolio_data = await self.data_provider.get_portfolio_data(self.user_id)
            
            # Load market data
            positions = await self.data_provider.get_positions(self.user_id)
            symbols = [pos['symbol'] for pos in positions if 'symbol' in pos]
            market_data = await self.data_provider.get_market_data(symbols)
            
            # Initialize risk calculation engines (placeholder)
            self.var_engine = VaREngine()
            self.cvar_engine = CVaREngine()
            self.correlation_engine = CorrelationEngine()
            
            logger.info(f"Initial data loaded for user {self.user_id}")
            
        except Exception as e:
            logger.error(f"Initial data loading error: {str(e)}")
            raise
    
    async def _update_risk_metrics(self):
        """Update all risk metrics."""
        try:
            # Get current data
            portfolio_data = await self.data_provider.get_portfolio_data(self.user_id)
            positions = await self.data_provider.get_positions(self.user_id)
            
            # Update VaR
            await self._update_var_metric(portfolio_data, positions)
            
            # Update CVaR
            await self._update_cvar_metric(portfolio_data, positions)
            
            # Update drawdown
            await self._update_drawdown_metric(portfolio_data)
            
            # Update volatility
            await self._update_volatility_metric(portfolio_data)
            
            # Update concentration risk
            await self._update_concentration_metric(positions)
            
            # Update beta
            await self._update_beta_metric(portfolio_data)
            
            # Update Sharpe ratio
            await self._update_sharpe_metric(portfolio_data)
            
            self.last_update = datetime.now()
            
        except Exception as e:
            logger.error(f"Risk metrics update error: {str(e)}")
    
    async def _update_var_metric(self, portfolio_data: Dict[str, Any], positions: List[Dict[str, Any]]):
        """Update Value at Risk metric."""
        try:
            # Calculate current VaR (simplified)
            portfolio_value = portfolio_data.get('total_value', 0)
            current_var = portfolio_value * 0.05  # Simplified 5% VaR
            
            await self._update_risk_metric(
                RiskMetricType.VALUE_AT_RISK,
                current_var,
                threshold_warning=portfolio_value * 0.03,
                threshold_critical=portfolio_value * 0.07
            )
            
        except Exception as e:
            logger.error(f"VaR metric update error: {str(e)}")
    
    async def _update_cvar_metric(self, portfolio_data: Dict[str, Any], positions: List[Dict[str, Any]]):
        """Update Conditional VaR metric."""
        try:
            portfolio_value = portfolio_data.get('total_value', 0)
            current_cvar = portfolio_value * 0.08  # Simplified 8% CVaR
            
            await self._update_risk_metric(
                RiskMetricType.CONDITIONAL_VAR,
                current_cvar,
                threshold_warning=portfolio_value * 0.05,
                threshold_critical=portfolio_value * 0.10
            )
            
        except Exception as e:
            logger.error(f"CVaR metric update error: {str(e)}")
    
    async def _update_drawdown_metric(self, portfolio_data: Dict[str, Any]):
        """Update maximum drawdown metric."""
        try:
            # Get portfolio history
            portfolio_history = portfolio_data.get('history', pd.Series())
            
            if len(portfolio_history) > 0:
                # Calculate current drawdown
                running_max = portfolio_history.expanding().max()
                current_drawdown = (portfolio_history.iloc[-1] - running_max.iloc[-1]) / running_max.iloc[-1]
                
                await self._update_risk_metric(
                    RiskMetricType.MAX_DRAWDOWN,
                    abs(current_drawdown),
                    threshold_warning=0.05,  # 5%
                    threshold_critical=0.15  # 15%
                )
            
        except Exception as e:
            logger.error(f"Drawdown metric update error: {str(e)}")
    
    async def _update_volatility_metric(self, portfolio_data: Dict[str, Any]):
        """Update portfolio volatility metric."""
        try:
            # Calculate 30-day rolling volatility
            portfolio_history = portfolio_data.get('history', pd.Series())
            
            if len(portfolio_history) > 30:
                returns = portfolio_history.pct_change().dropna()
                volatility = returns.rolling(30).std().iloc[-1] * np.sqrt(252)  # Annualized
                
                await self._update_risk_metric(
                    RiskMetricType.VOLATILITY,
                    volatility,
                    threshold_warning=0.20,  # 20% annualized
                    threshold_critical=0.30  # 30% annualized
                )
            
        except Exception as e:
            logger.error(f"Volatility metric update error: {str(e)}")
    
    async def _update_concentration_metric(self, positions: List[Dict[str, Any]]):
        """Update concentration risk metric."""
        try:
            # Calculate portfolio concentration (Herfindahl index)
            if not positions:
                concentration = 0.0
            else:
                total_value = sum(abs(pos.get('market_value', 0)) for pos in positions)
                
                if total_value > 0:
                    weights = [
                        abs(pos.get('market_value', 0)) / total_value 
                        for pos in positions
                    ]
                    concentration = sum(w ** 2 for w in weights)  # Herfindahl index
                else:
                    concentration = 0.0
            
            await self._update_risk_metric(
                RiskMetricType.CONCENTRATION_RISK,
                concentration,
                threshold_warning=0.25,  # 25% concentration
                threshold_critical=0.40  # 40% concentration
            )
            
        except Exception as e:
            logger.error(f"Concentration metric update error: {str(e)}")
    
    async def _update_beta_metric(self, portfolio_data: Dict[str, Any]):
        """Update portfolio beta metric."""
        try:
            # Simplified beta calculation
            portfolio_history = portfolio_data.get('history', pd.Series())
            market_history = portfolio_data.get('market_history', pd.Series())
            
            if len(portfolio_history) > 30 and len(market_history) > 30:
                portfolio_returns = portfolio_history.pct_change().dropna()
                market_returns = market_history.pct_change().dropna()
                
                # Align series
                common_index = portfolio_returns.index.intersection(market_returns.index)
                if len(common_index) > 20:
                    portfolio_aligned = portfolio_returns.reindex(common_index)
                    market_aligned = market_returns.reindex(common_index)
                    
                    # Calculate beta
                    covariance = np.cov(portfolio_aligned, market_aligned)[0, 1]
                    market_variance = np.var(market_aligned)
                    beta = covariance / market_variance if market_variance > 0 else 1.0
                else:
                    beta = 1.0
            else:
                beta = 1.0
            
            await self._update_risk_metric(
                RiskMetricType.BETA,
                beta,
                threshold_warning=1.5,
                threshold_critical=2.0
            )
            
        except Exception as e:
            logger.error(f"Beta metric update error: {str(e)}")
    
    async def _update_sharpe_metric(self, portfolio_data: Dict[str, Any]):
        """Update Sharpe ratio metric."""
        try:
            # Calculate Sharpe ratio
            portfolio_history = portfolio_data.get('history', pd.Series())
            
            if len(portfolio_history) > 30:
                returns = portfolio_history.pct_change().dropna()
                excess_return = returns.mean() * 252 - 0.02  # Annualized excess return
                volatility = returns.std() * np.sqrt(252)  # Annualized volatility
                sharpe_ratio = excess_return / volatility if volatility > 0 else 0
            else:
                sharpe_ratio = 0.0
            
            await self._update_risk_metric(
                RiskMetricType.SHARPE_RATIO,
                sharpe_ratio,
                threshold_warning=0.5,  # Minimum acceptable Sharpe
                threshold_critical=0.2   # Critical Sharpe level
            )
            
        except Exception as e:
            logger.error(f"Sharpe ratio metric update error: {str(e)}")
    
    async def _update_risk_metric(self, metric_type: RiskMetricType, current_value: float,
                                threshold_warning: float, threshold_critical: float):
        """Update individual risk metric."""
        try:
            metric_key = metric_type.value
            
            # Get previous value
            previous_value = self.risk_metrics.get(metric_key).current_value if metric_key in self.risk_metrics else None
            
            # Calculate change
            if previous_value is not None and previous_value != 0:
                change_pct = ((current_value - previous_value) / abs(previous_value)) * 100
            else:
                change_pct = 0.0
            
            # Determine trend
            if previous_value is None:
                trend = 'stable'
            elif abs(change_pct) < 1.0:
                trend = 'stable'
            elif change_pct > 0:
                trend = 'increasing'
            else:
                trend = 'decreasing'
            
            # Update metric
            self.risk_metrics[metric_key] = RiskMetric(
                metric_type=metric_type,
                current_value=current_value,
                previous_value=previous_value,
                change_pct=change_pct,
                trend=trend,
                threshold_warning=threshold_warning,
                threshold_critical=threshold_critical,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Risk metric update error for {metric_type}: {str(e)}")
    
    async def _check_alerts(self):
        """Check for alert conditions."""
        try:
            for metric_key, metric in self.risk_metrics.items():
                await self._check_metric_alerts(metric)
            
        except Exception as e:
            logger.error(f"Alert checking error: {str(e)}")
    
    async def _check_metric_alerts(self, metric: RiskMetric):
        """Check alerts for specific metric."""
        try:
            # Check critical threshold
            if metric.current_value > metric.threshold_critical:
                await self._create_alert(
                    alert_type=f"{metric.metric_type.value}_critical",
                    severity=AlertSeverity.CRITICAL,
                    title=f"Critical {metric.metric_type.value} Alert",
                    message=f"{metric.metric_type.value} exceeded critical threshold",
                    metric_value=metric.current_value,
                    threshold_value=metric.threshold_critical
                )
            
            # Check warning threshold
            elif metric.current_value > metric.threshold_warning:
                await self._create_alert(
                    alert_type=f"{metric.metric_type.value}_warning",
                    severity=AlertSeverity.WARNING,
                    title=f"{metric.metric_type.value} Warning",
                    message=f"{metric.metric_type.value} approaching critical level",
                    metric_value=metric.current_value,
                    threshold_value=metric.threshold_warning
                )
            
            # Check trend-based alerts
            if metric.trend == 'increasing' and metric.change_pct > 10:
                await self._create_alert(
                    alert_type=f"{metric.metric_type.value}_trend_increase",
                    severity=AlertSeverity.WARNING,
                    title=f"{metric.metric_type.value} Increasing Rapidly",
                    message=f"{metric.metric_type.value} increased {metric.change_pct:.1f}%",
                    metric_value=metric.current_value,
                    threshold_value=metric.current_value * 0.9
                )
            
        except Exception as e:
            logger.error(f"Metric alert checking error for {metric.metric_type}: {str(e)}")
    
    async def _create_alert(self, alert_type: str, severity: AlertSeverity, title: str,
                          message: str, metric_value: float, threshold_value: float):
        """Create new risk alert."""
        try:
            # Check if similar alert already exists (avoid spam)
            recent_alerts = [
                alert for alert in self.alerts
                if (alert.alert_type == alert_type and 
                    (datetime.now() - alert.timestamp).seconds < 300)  # 5 minutes
            ]
            
            if recent_alerts:
                return  # Don't create duplicate alert
            
            # Create alert
            alert = RiskAlert(
                alert_id=f"{self.user_id}_{datetime.now().timestamp()}_{alert_type}",
                alert_type=alert_type,
                severity=severity,
                title=title,
                message=message,
                metric_value=metric_value,
                threshold_value=threshold_value,
                timestamp=datetime.now(),
                user_id=self.user_id
            )
            
            # Add to alerts list
            self.alerts.append(alert)
            
            # Notify callbacks
            for callback in self.alert_callbacks:
                try:
                    await callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback error: {str(e)}")
            
            logger.warning(f"Risk alert created for user {self.user_id}: {title}")
            
        except Exception as e:
            logger.error(f"Alert creation error: {str(e)}")
    
    async def _generate_recommendations(self):
        """Generate risk management recommendations."""
        try:
            recommendations = []
            
            # Analyze current risk metrics
            for metric_key, metric in self.risk_metrics.items():
                metric_recommendations = self._get_metric_recommendations(metric)
                recommendations.extend(metric_recommendations)
            
            # Store recommendations for dashboard
            self._current_recommendations = recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation error: {str(e)}")
    
    def _get_metric_recommendations(self, metric: RiskMetric) -> List[str]:
        """Get recommendations based on metric status."""
        try:
            recommendations = []
            
            if metric.metric_type == RiskMetricType.VALUE_AT_RISK:
                if metric.current_value > metric.threshold_critical:
                    recommendations.append("Reduce position sizes to lower portfolio VaR")
                    recommendations.append("Consider hedging strategies to reduce downside risk")
                elif metric.current_value > metric.threshold_warning:
                    recommendations.append("Monitor VaR levels closely and consider risk reduction")
            
            elif metric.metric_type == RiskMetricType.MAX_DRAWDOWN:
                if metric.current_value > metric.threshold_critical:
                    recommendations.append("URGENT: Implement emergency risk controls")
                    recommendations.append("Consider reducing overall portfolio exposure")
                elif metric.current_value > metric.threshold_warning:
                    recommendations.append("Review portfolio for stress testing")
            
            elif metric.metric_type == RiskMetricType.CONCENTRATION_RISK:
                if metric.current_value > metric.threshold_warning:
                    recommendations.append("Reduce concentration in single positions")
                    recommendations.append("Diversify across more securities")
            
            elif metric.metric_type == RiskMetricType.VOLATILITY:
                if metric.current_value > metric.threshold_warning:
                    recommendations.append("Consider reducing volatile positions")
                    recommendations.append("Increase allocation to low-volatility assets")
            
            elif metric.metric_type == RiskMetricType.SHARPE_RATIO:
                if metric.current_value < metric.threshold_critical:
                    recommendations.append("Review portfolio composition for better risk-adjusted returns")
                    recommendations.append("Consider switching to higher Sharpe ratio strategies")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Metric recommendations error: {str(e)}")
            return []
    
    async def _update_dashboard(self):
        """Update risk dashboard data."""
        try:
            # Get current portfolio data
            portfolio_data = await self.data_provider.get_portfolio_data(self.user_id)
            
            # Create dashboard
            dashboard = RiskDashboard(
                timestamp=datetime.now(),
                user_id=self.user_id,
                portfolio_value=portfolio_data.get('total_value', 0),
                daily_pnl=portfolio_data.get('daily_pnl', 0),
                daily_pnl_pct=portfolio_data.get('daily_pnl_pct', 0),
                risk_metrics=self.risk_metrics,
                active_alerts=[alert for alert in self.alerts if not alert.is_resolved],
                position_summary=portfolio_data.get('position_summary', {}),
                limit_status=portfolio_data.get('limit_status', {}),
                performance_metrics=portfolio_data.get('performance_metrics', {}),
                market_conditions=portfolio_data.get('market_conditions', {}),
                recommendations=getattr(self, '_current_recommendations', [])
            )
            
            # Store dashboard for access
            self.current_dashboard = dashboard
            
        except Exception as e:
            logger.error(f"Dashboard update error: {str(e)}")
    
    async def get_dashboard_data(self) -> Optional[RiskDashboard]:
        """Get current dashboard data."""
        return getattr(self, 'current_dashboard', None)
    
    async def get_active_alerts(self) -> List[RiskAlert]:
        """Get all active (unresolved) alerts."""
        return [alert for alert in self.alerts if not alert.is_resolved]
    
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        try:
            for alert in self.alerts:
                if alert.alert_id == alert_id:
                    alert.is_acknowledged = True
                    logger.info(f"Alert {alert_id} acknowledged by user {self.user_id}")
                    return True
            return False
        except Exception as e:
            logger.error(f"Alert acknowledgment error: {str(e)}")
            return False
    
    async def resolve_alert(self, alert_id: str, resolution_notes: str = "") -> bool:
        """Resolve an alert."""
        try:
            for alert in self.alerts:
                if alert.alert_id == alert_id:
                    alert.is_resolved = True
                    alert.resolution_notes = resolution_notes
                    logger.info(f"Alert {alert_id} resolved by user {self.user_id}")
                    return True
            return False
        except Exception as e:
            logger.error(f"Alert resolution error: {str(e)}")
            return False
    
    def add_alert_callback(self, callback: Callable[[RiskAlert], None]):
        """Add callback function for alert notifications."""
        self.alert_callbacks.append(callback)
    
    def set_metric_thresholds(self, thresholds: Dict[str, float]):
        """Set custom thresholds for risk metrics."""
        self.metric_thresholds.update(thresholds)


# Placeholder classes for risk engines (would be implemented separately)
class VaREngine:
    """Placeholder VaR calculation engine."""
    async def calculate_var(self, portfolio_data: Dict[str, Any]) -> float:
        return portfolio_data.get('total_value', 0) * 0.05


class CVaREngine:
    """Placeholder CVaR calculation engine."""
    async def calculate_cvar(self, portfolio_data: Dict[str, Any]) -> float:
        return portfolio_data.get('total_value', 0) * 0.08


class CorrelationEngine:
    """Placeholder correlation calculation engine."""
    async def calculate_correlations(self, positions: List[Dict[str, Any]]) -> Dict[str, float]:
        return {}


class WebSocketRiskMonitor:
    """
    WebSocket-based real-time risk monitoring.
    
    Provides real-time risk data streaming via WebSocket connections.
    """
    
    def __init__(self, user_id: int, data_provider: RiskDataProvider):
        """
        Initialize WebSocket risk monitor.
        
        Args:
            user_id: User identifier
            data_provider: Data provider
        """
        self.user_id = user_id
        self.data_provider = data_provider
        self.monitor = RealTimeRiskMonitor(user_id, data_provider)
        self.websocket_connections = set()
        
    async def start_websocket_server(self, host: str = "localhost", port: int = 8765):
        """Start WebSocket server for real-time risk streaming."""
        try:
            # Start monitoring
            await self.monitor.start_monitoring()
            
            # Add alert callback to forward alerts via WebSocket
            self.monitor.add_alert_callback(self._broadcast_alert)
            
            # Start WebSocket server
            async with websockets.serve(self._handle_connection, host, port):
                logger.info(f"WebSocket risk monitor started on {host}:{port}")
                await asyncio.Future()  # Run forever
                
        except Exception as e:
            logger.error(f"WebSocket server start error: {str(e)}")
            raise
    
    async def _handle_connection(self, websocket, path):
        """Handle WebSocket connection."""
        try:
            self.websocket_connections.add(websocket)
            logger.info(f"WebSocket connection established for user {self.user_id}")
            
            # Send initial dashboard data
            dashboard = await self.monitor.get_dashboard_data()
            if dashboard:
                await websocket.send(json.dumps({
                    'type': 'dashboard_update',
                    'data': self._serialize_dashboard(dashboard)
                }))
            
            # Keep connection alive and handle messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_message(websocket, data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON message received: {message}")
                except Exception as e:
                    logger.error(f"Message handling error: {str(e)}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
        except Exception as e:
            logger.error(f"WebSocket connection error: {str(e)}")
        finally:
            self.websocket_connections.discard(websocket)
    
    async def _handle_message(self, websocket, data: Dict[str, Any]):
        """Handle incoming WebSocket messages."""
        try:
            message_type = data.get('type')
            
            if message_type == 'get_dashboard':
                dashboard = await self.monitor.get_dashboard_data()
                if dashboard:
                    await websocket.send(json.dumps({
                        'type': 'dashboard_update',
                        'data': self._serialize_dashboard(dashboard)
                    }))
            
            elif message_type == 'acknowledge_alert':
                alert_id = data.get('alert_id')
                success = await self.monitor.acknowledge_alert(alert_id)
                await websocket.send(json.dumps({
                    'type': 'alert_ack_response',
                    'alert_id': alert_id,
                    'success': success
                }))
            
            elif message_type == 'resolve_alert':
                alert_id = data.get('alert_id')
                resolution_notes = data.get('resolution_notes', '')
                success = await self.monitor.resolve_alert(alert_id, resolution_notes)
                await websocket.send(json.dumps({
                    'type': 'alert_resolve_response',
                    'alert_id': alert_id,
                    'success': success
                }))
            
        except Exception as e:
            logger.error(f"WebSocket message handling error: {str(e)}")
    
    async def _broadcast_alert(self, alert: RiskAlert):
        """Broadcast alert to all connected clients."""
        if self.websocket_connections:
            message = json.dumps({
                'type': 'risk_alert',
                'data': self._serialize_alert(alert)
            })
            
            # Send to all connections
            disconnected = set()
            for websocket in self.websocket_connections:
                try:
                    await websocket.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected.add(websocket)
                except Exception as e:
                    logger.error(f"Alert broadcast error: {str(e)}")
                    disconnected.add(websocket)
            
            # Remove disconnected connections
            self.websocket_connections -= disconnected
    
    def _serialize_alert(self, alert: RiskAlert) -> Dict[str, Any]:
        """Serialize alert for JSON transmission."""
        return {
            'alert_id': alert.alert_id,
            'alert_type': alert.alert_type,
            'severity': alert.severity.value,
            'title': alert.title,
            'message': alert.message,
            'metric_value': alert.metric_value,
            'threshold_value': alert.threshold_value,
            'timestamp': alert.timestamp.isoformat(),
            'is_acknowledged': alert.is_acknowledged,
            'is_resolved': alert.is_resolved
        }
    
    def _serialize_dashboard(self, dashboard: RiskDashboard) -> Dict[str, Any]:
        """serialize dashboard for JSON transmission."""
        return {
            'timestamp': dashboard.timestamp.isoformat(),
            'portfolio_value': dashboard.portfolio_value,
            'daily_pnl': dashboard.daily_pnl,
            'daily_pnl_pct': dashboard.daily_pnl_pct,
            'risk_metrics': {
                key: {
                    'metric_type': metric.metric_type.value,
                    'current_value': metric.current_value,
                    'change_pct': metric.change_pct,
                    'trend': metric.trend,
                    'last_updated': metric.last_updated.isoformat()
                }
                for key, metric in dashboard.risk_metrics.items()
            },
            'active_alerts': [
                self._serialize_alert(alert) for alert in dashboard.active_alerts
            ],
            'recommendations': dashboard.recommendations
        }


class RiskMonitorFactory:
    """Factory class for creating risk monitoring components."""
    
    @staticmethod
    def create_monitor(monitor_type: str, user_id: int, data_provider: RiskDataProvider) -> Any:
        """
        Create risk monitor instance.
        
        Args:
            monitor_type: Type of monitor
            user_id: User identifier
            data_provider: Data provider
            
        Returns:
            Risk monitor instance
        """
        monitor_types = {
            'realtime': RealTimeRiskMonitor,
            'websocket': WebSocketRiskMonitor
        }
        
        if monitor_type.lower() not in monitor_types:
            raise ValueError(f"Unsupported monitor type: {monitor_type}")
        
        return monitor_types[monitor_type.lower()](user_id, data_provider)
