"""
Demo Mode Dashboard - Real-time monitoring and analytics for demo trading
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from pathlib import Path

from loguru import logger

from .demo_mode_manager import DemoModeManager, DemoModeState
from .virtual_portfolio import VirtualPortfolio, PortfolioPosition, RiskMetrics
from .demo_logging import DemoLogger, LogLevel, LogCategory
from .demo_backtesting import BacktestResults, DemoStrategy


class DashboardMetric(Enum):
    """Dashboard metric types"""
    PORTFOLIO_VALUE = "portfolio_value"
    DAILY_RETURN = "daily_return"
    TOTAL_RETURN = "total_return"
    CASH_BALANCE = "cash_balance"
    UNREALIZED_PNL = "unrealized_pnl"
    REALIZED_PNL = "realized_pnl"
    POSITIONS_COUNT = "positions_count"
    TRADES_COUNT = "trades_count"
    WIN_RATE = "win_rate"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    VAR = "var"
    EXECUTION_TIME = "execution_time"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class DashboardWidget:
    """Individual dashboard widget configuration"""
    widget_id: str
    title: str
    widget_type: str  # chart, number, table, etc.
    metric: DashboardMetric
    position: Tuple[int, int]  # x, y coordinates
    size: Tuple[int, int]  # width, height
    refresh_interval: int = 5  # seconds
    config: Dict[str, Any] = None


@dataclass
class Alert:
    """System alert"""
    alert_id: str
    timestamp: datetime
    level: AlertLevel
    title: str
    message: str
    metric: Optional[str] = None
    current_value: Optional[float] = None
    threshold: Optional[float] = None
    acknowledged: bool = False


@dataclass
class DashboardData:
    """Complete dashboard data snapshot"""
    timestamp: datetime
    portfolio: Dict[str, Any]
    performance: Dict[str, Any]
    positions: List[Dict[str, Any]]
    trades: List[Dict[str, Any]]
    risk_metrics: Dict[str, Any]
    alerts: List[Alert]
    system_status: Dict[str, Any]
    demo_mode_status: Dict[str, Any]


class DemoDashboard:
    """
    Real-time dashboard for demo mode monitoring
    
    Provides:
    - Real-time portfolio metrics
    - Performance visualization
    - Risk monitoring and alerts
    - Trade execution tracking
    - System status monitoring
    """
    
    def __init__(self, demo_manager: DemoModeManager):
        self.demo_manager = demo_manager
        self.portfolio = None
        self.logger = None
        
        # Dashboard configuration
        self.widgets: List[DashboardWidget] = []
        self.alerts: List[Alert] = []
        self.alert_thresholds = self._initialize_alert_thresholds()
        
        # Real-time data
        self.latest_data: Optional[DashboardData] = None
        self.data_history: List[DashboardData] = []
        self.max_history_size = 1000
        
        # Background tasks
        self.update_task = None
        self.alert_task = None
        self.is_running = False
        
        # Statistics
        self.update_count = 0
        self.alert_count = 0
        
        # Initialize default widgets
        self._initialize_default_widgets()
    
    async def start_dashboard(self):
        """Start the dashboard monitoring system"""
        try:
            if self.is_running:
                return
            
            self.is_running = True
            
            # Initialize components
            from .virtual_portfolio import get_virtual_portfolio
            self.portfolio = await get_virtual_portfolio()
            
            from .demo_logging import get_demo_logger
            self.logger = await get_demo_logger()
            
            # Start background tasks
            self.update_task = asyncio.create_task(self._data_update_loop())
            self.alert_task = asyncio.create_task(self._alert_monitoring_loop())
            
            # Log startup
            await self.logger.log_system_event(
                "dashboard_started", "demo_dashboard", "info", 0.0,
                {"widgets_count": len(self.widgets)}
            )
            
            logger.info("Demo dashboard started")
            
        except Exception as e:
            logger.error(f"Failed to start demo dashboard: {e}")
            self.is_running = False
    
    async def stop_dashboard(self):
        """Stop the dashboard monitoring system"""
        try:
            if not self.is_running:
                return
            
            self.is_running = False
            
            # Cancel background tasks
            if self.update_task:
                self.update_task.cancel()
            if self.alert_task:
                self.alert_task.cancel()
            
            # Log shutdown
            if self.logger:
                await self.logger.log_system_event(
                    "dashboard_stopped", "demo_dashboard", "info", 0.0,
                    {
                        "update_count": self.update_count,
                        "alert_count": self.alert_count,
                        "data_points": len(self.data_history)
                    }
                )
            
            logger.info("Demo dashboard stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop demo dashboard: {e}")
    
    async def get_dashboard_data(self) -> DashboardData:
        """Get current dashboard data"""
        if self.latest_data is None:
            await self._update_dashboard_data()
        return self.latest_data
    
    async def get_historical_data(self, hours: int = 24) -> List[DashboardData]:
        """Get historical dashboard data"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [data for data in self.data_history if data.timestamp >= cutoff_time]
    
    async def add_widget(self, widget: DashboardWidget):
        """Add a new dashboard widget"""
        self.widgets.append(widget)
        logger.info(f"Added widget: {widget.title}")
    
    async def remove_widget(self, widget_id: str):
        """Remove a dashboard widget"""
        self.widgets = [w for w in self.widgets if w.widget_id != widget_id]
        logger.info(f"Removed widget: {widget_id}")
    
    async def update_widget(self, widget_id: str, updates: Dict[str, Any]):
        """Update widget configuration"""
        for widget in self.widgets:
            if widget.widget_id == widget_id:
                for key, value in updates.items():
                    if hasattr(widget, key):
                        setattr(widget, key, value)
                break
    
    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary for dashboard"""
        try:
            if not self.portfolio:
                return {}
            
            current_value = await self.portfolio.get_portfolio_value()
            metrics = await self.portfolio.get_portfolio_metrics()
            positions = await self.portfolio.get_positions_summary()
            
            # Calculate additional metrics
            daily_change = 0
            if len(self.data_history) > 0:
                prev_value = self.data_history[-1].portfolio.get("total_value", current_value)
                daily_change = ((current_value - prev_value) / prev_value) * 100 if prev_value > 0 else 0
            
            return {
                "total_value": current_value,
                "daily_change_pct": daily_change,
                "cash_balance": metrics.get("cash_balance", 0),
                "invested_value": metrics.get("invested_value", 0),
                "unrealized_pnl": metrics.get("absolute_return", 0) - metrics.get("realized_pnl", 0),
                "realized_pnl": metrics.get("realized_pnl", 0),
                "positions_count": len([p for p in positions if p.quantity != 0]),
                "total_return_pct": metrics.get("total_return_pct", 0),
                "benchmark_return": 0,  # Would calculate vs benchmark
                "outperformance": 0  # Would calculate vs benchmark
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return {}
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for dashboard"""
        try:
            if not self.portfolio:
                return {}
            
            metrics = await self.portfolio.get_portfolio_metrics()
            risk_metrics = await self.portfolio.get_risk_metrics()
            
            # Get recent trades
            trade_history = await self.portfolio.get_trade_history(10)
            
            # Calculate recent performance
            win_rate = 0
            avg_trade_size = 0
            if trade_history:
                completed_trades = [t for t in trade_history if t.get('side') in ['buy', 'sell']]
                if completed_trades:
                    winning_trades = len([t for t in completed_trades if float(t.get('pnl', 0)) > 0])
                    win_rate = (winning_trades / len(completed_trades)) * 100
                    avg_trade_size = np.mean([abs(float(t.get('quantity', 0)) * float(t.get('price', 0))) for t in completed_trades])
            
            return {
                "total_return_pct": metrics.get("total_return_pct", 0),
                "annualized_return": metrics.get("annualized_return", 0),
                "volatility": metrics.get("volatility", 0),
                "sharpe_ratio": metrics.get("sharpe_ratio", 0),
                "max_drawdown_pct": metrics.get("max_drawdown_pct", 0),
                "win_rate": win_rate,
                "avg_trade_size": avg_trade_size,
                "total_trades": len(trade_history),
                "profit_factor": metrics.get("profit_factor", 0),
                "calmar_ratio": metrics.get("calmar_ratio", 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {}
    
    async def get_positions_summary(self) -> List[Dict[str, Any]]:
        """Get positions summary for dashboard"""
        try:
            if not self.portfolio:
                return []
            
            positions = await self.portfolio.get_positions_summary()
            
            # Format positions for dashboard
            position_list = []
            for pos in positions:
                if pos.quantity != 0:
                    position_dict = {
                        "symbol": pos.symbol,
                        "side": "Long" if pos.quantity > 0 else "Short",
                        "quantity": abs(pos.quantity),
                        "avg_price": pos.avg_entry_price,
                        "current_price": pos.current_price,
                        "market_value": pos.market_value,
                        "unrealized_pnl": pos.unrealized_pnl,
                        "pnl_percentage": pos.pnl_percentage,
                        "weight": pos.weight,
                        "commission_paid": pos.commission_paid
                    }
                    position_list.append(position_dict)
            
            # Sort by market value (largest first)
            position_list.sort(key=lambda x: abs(x["market_value"]), reverse=True)
            
            return position_list
            
        except Exception as e:
            logger.error(f"Error getting positions summary: {e}")
            return []
    
    async def get_recent_trades(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent trades for dashboard"""
        try:
            if not self.portfolio:
                return []
            
            trade_history = await self.portfolio.get_trade_history(limit)
            
            # Format trades for dashboard
            formatted_trades = []
            for trade in reversed(trade_history[-limit:]):  # Most recent first
                trade_dict = {
                    "timestamp": trade.get('timestamp', datetime.now()),
                    "symbol": trade.get('symbol', ''),
                    "side": trade.get('side', ''),
                    "quantity": trade.get('quantity', 0),
                    "price": trade.get('price', 0),
                    "commission": trade.get('commission', 0),
                    "total_value": trade.get('quantity', 0) * trade.get('price', 0),
                    "execution_time": trade.get('timestamp', datetime.now())
                }
                formatted_trades.append(trade_dict)
            
            return formatted_trades
            
        except Exception as e:
            logger.error(f"Error getting recent trades: {e}")
            return []
    
    async def get_risk_summary(self) -> Dict[str, Any]:
        """Get risk summary for dashboard"""
        try:
            if not self.portfolio:
                return {}
            
            risk_metrics = await self.portfolio.get_risk_metrics()
            
            # Convert to dashboard format
            return {
                "var_95_pct": risk_metrics.var_95 * 100,
                "var_99_pct": risk_metrics.var_99 * 100,
                "expected_shortfall_pct": risk_metrics.expected_shortfall * 100,
                "volatility": risk_metrics.volatility * 100,
                "sharpe_ratio": risk_metrics.sharpe_ratio,
                "sortino_ratio": risk_metrics.sortino_ratio,
                "max_drawdown": risk_metrics.max_drawdown,
                "calmar_ratio": risk_metrics.calmar_ratio,
                "current_alerts": len([a for a in self.alerts if not a.acknowledged])
            }
            
        except Exception as e:
            logger.error(f"Error getting risk summary: {e}")
            return {}
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status for dashboard"""
        try:
            # Demo mode status
            demo_status = await self.demo_manager.get_demo_status()
            
            # Logging system status
            logging_stats = {}
            if self.logger:
                logging_stats = await self.logger.get_statistics()
            
            return {
                "demo_mode": demo_status,
                "is_demo_active": demo_status.get("is_active", False),
                "session_id": demo_status.get("session", {}).get("session_id"),
                "logging_stats": logging_stats,
                "dashboard_stats": {
                    "update_count": self.update_count,
                    "alert_count": self.alert_count,
                    "is_running": self.is_running,
                    "widgets_count": len(self.widgets),
                    "data_points": len(self.data_history)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {}
    
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        try:
            for alert in self.alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    logger.info(f"Alert acknowledged: {alert_id}")
                    return True
            return False
        except Exception as e:
            logger.error(f"Error acknowledging alert: {e}")
            return False
    
    async def clear_alerts(self, level: Optional[AlertLevel] = None):
        """Clear alerts, optionally by level"""
        try:
            if level:
                self.alerts = [a for a in self.alerts if a.level != level]
            else:
                self.alerts.clear()
            
            logger.info(f"Cleared alerts (level: {level})")
            
        except Exception as e:
            logger.error(f"Error clearing alerts: {e}")
    
    async def export_dashboard_data(self, filepath: str, format: str = "json"):
        """Export dashboard data"""
        try:
            data = await self.get_dashboard_data()
            historical = self.data_history
            
            if format.lower() == "json":
                export_data = {
                    "current_data": asdict(data),
                    "historical_data": [asdict(h) for h in historical],
                    "export_time": datetime.now().isoformat(),
                    "summary": {
                        "total_data_points": len(historical),
                        "time_span_hours": (historical[-1].timestamp - historical[0].timestamp).total_seconds() / 3600 if historical else 0,
                        "total_alerts": len(self.alerts),
                        "active_alerts": len([a for a in self.alerts if not a.acknowledged])
                    }
                }
                
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Dashboard data exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting dashboard data: {e}")
    
    # Private methods
    
    def _initialize_default_widgets(self):
        """Initialize default dashboard widgets"""
        self.widgets = [
            DashboardWidget(
                widget_id="portfolio_value",
                title="Portfolio Value",
                widget_type="number",
                metric=DashboardMetric.PORTFOLIO_VALUE,
                position=(0, 0),
                size=(3, 2)
            ),
            DashboardWidget(
                widget_id="daily_return",
                title="Daily Return",
                widget_type="chart",
                metric=DashboardMetric.DAILY_RETURN,
                position=(3, 0),
                size=(6, 3),
                config={"chart_type": "line", "timeframe": "1d"}
            ),
            DashboardWidget(
                widget_id="positions_table",
                title="Positions",
                widget_type="table",
                metric=DashboardMetric.POSITIONS_COUNT,
                position=(0, 2),
                size=(9, 4)
            ),
            DashboardWidget(
                widget_id="risk_metrics",
                title="Risk Metrics",
                widget_type="number",
                metric=DashboardMetric.VAR,
                position=(0, 6),
                size=(4, 2)
            ),
            DashboardWidget(
                widget_id="trades_feed",
                title="Recent Trades",
                widget_type="table",
                metric=DashboardMetric.TRADES_COUNT,
                position=(4, 6),
                size=(5, 3)
            )
        ]
    
    def _initialize_alert_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize alert thresholds"""
        return {
            "portfolio_value": {"min": 50000, "max": 200000},
            "daily_return": {"min": -5.0, "max": 5.0},
            "max_drawdown": {"max": 15.0},
            "var_95": {"max": -3.0},
            "sharpe_ratio": {"min": 0.5},
            "win_rate": {"min": 40.0},
            "positions_count": {"max": 20},
            "trades_count": {"max": 100}
        }
    
    async def _data_update_loop(self):
        """Background task for dashboard data updates"""
        while self.is_running:
            try:
                await self._update_dashboard_data()
                await asyncio.sleep(5)  # Update every 5 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Dashboard update error: {e}")
                await asyncio.sleep(5)
    
    async def _alert_monitoring_loop(self):
        """Background task for alert monitoring"""
        while self.is_running:
            try:
                await self._check_alerts()
                await asyncio.sleep(10)  # Check alerts every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Alert monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _update_dashboard_data(self):
        """Update dashboard data"""
        try:
            # Collect all data
            portfolio_data = await self.get_portfolio_summary()
            performance_data = await self.get_performance_summary()
            positions_data = await self.get_positions_summary()
            trades_data = await self.get_recent_trades()
            risk_data = await self.get_risk_summary()
            system_data = await self.get_system_status()
            
            # Create dashboard data
            dashboard_data = DashboardData(
                timestamp=datetime.now(),
                portfolio=portfolio_data,
                performance=performance_data,
                positions=positions_data,
                trades=trades_data,
                risk_metrics=risk_data,
                alerts=[a for a in self.alerts if not a.acknowledged],
                system_status=system_data,
                demo_mode_status=await self.demo_manager.get_demo_status()
            )
            
            # Store data
            self.latest_data = dashboard_data
            self.data_history.append(dashboard_data)
            
            # Manage history size
            if len(self.data_history) > self.max_history_size:
                self.data_history = self.data_history[-self.max_history_size//2:]
            
            self.update_count += 1
            
        except Exception as e:
            logger.error(f"Error updating dashboard data: {e}")
    
    async def _check_alerts(self):
        """Check for alert conditions"""
        try:
            if not self.latest_data:
                return
            
            # Check portfolio alerts
            await self._check_portfolio_alerts()
            
            # Check risk alerts
            await self._check_risk_alerts()
            
            # Check system alerts
            await self._check_system_alerts()
            
            # Clean up old alerts
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.alerts = [a for a in self.alerts if a.timestamp >= cutoff_time]
            
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
    
    async def _check_portfolio_alerts(self):
        """Check portfolio-related alerts"""
        try:
            portfolio = self.latest_data.portfolio
            risk_metrics = self.latest_data.risk_metrics
            
            # Portfolio value alerts
            portfolio_value = portfolio.get("total_value", 0)
            await self._check_threshold_alert(
                "Portfolio Value",
                portfolio_value,
                self.alert_thresholds.get("portfolio_value", {}),
                "portfolio_value"
            )
            
            # Daily return alerts
            daily_return = portfolio.get("daily_change_pct", 0)
            await self._check_threshold_alert(
                "Daily Return",
                daily_return,
                self.alert_thresholds.get("daily_return", {}),
                "daily_return"
            )
            
            # Win rate alerts
            win_rate = self.latest_data.performance.get("win_rate", 0)
            await self._check_threshold_alert(
                "Win Rate",
                win_rate,
                self.alert_thresholds.get("win_rate", {}),
                "win_rate"
            )
            
            # Position count alerts
            position_count = len(self.latest_data.positions)
            await self._check_threshold_alert(
                "Position Count",
                position_count,
                self.alert_thresholds.get("positions_count", {}),
                "positions_count"
            )
            
        except Exception as e:
            logger.error(f"Error checking portfolio alerts: {e}")
    
    async def _check_risk_alerts(self):
        """Check risk-related alerts"""
        try:
            risk_metrics = self.latest_data.risk_metrics
            
            # VaR alerts
            var_95 = risk_metrics.get("var_95_pct", 0)
            await self._check_threshold_alert(
                "95% VaR",
                var_95,
                self.alert_thresholds.get("var_95", {}),
                "var_95"
            )
            
            # Max drawdown alerts
            max_dd = risk_metrics.get("max_drawdown", 0)
            await self._check_threshold_alert(
                "Max Drawdown",
                max_dd,
                self.alert_thresholds.get("max_drawdown", {}),
                "max_drawdown"
            )
            
            # Sharpe ratio alerts
            sharpe = risk_metrics.get("sharpe_ratio", 0)
            await self._check_threshold_alert(
                "Sharpe Ratio",
                sharpe,
                self.alert_thresholds.get("sharpe_ratio", {}),
                "sharpe_ratio"
            )
            
        except Exception as e:
            logger.error(f"Error checking risk alerts: {e}")
    
    async def _check_system_alerts(self):
        """Check system-related alerts"""
        try:
            system_status = self.latest_data.system_status
            
            # Demo mode status
            is_demo_active = system_status.get("is_demo_active", False)
            if not is_demo_active:
                await self._create_alert(
                    AlertLevel.WARNING,
                    "Demo Mode Inactive",
                    "Demo mode is not currently active"
                )
            
            # Trading activity
            trades_today = len([t for t in self.latest_data.trades if 
                             (datetime.now() - t.get('timestamp', datetime.now())).days == 0])
            await self._check_threshold_alert(
                "Daily Trades",
                trades_today,
                self.alert_thresholds.get("trades_count", {}),
                "trades_count"
            )
            
        except Exception as e:
            logger.error(f"Error checking system alerts: {e}")
    
    async def _check_threshold_alert(
        self,
        metric_name: str,
        current_value: float,
        thresholds: Dict[str, float],
        metric_key: str
    ):
        """Check if a metric violates thresholds"""
        try:
            # Check minimum threshold
            if "min" in thresholds and current_value < thresholds["min"]:
                await self._create_alert(
                    AlertLevel.WARNING,
                    f"{metric_name} Below Threshold",
                    f"{metric_name} ({current_value:.2f}) is below minimum threshold ({thresholds['min']:.2f})",
                    metric_key,
                    current_value,
                    thresholds["min"]
                )
            
            # Check maximum threshold
            if "max" in thresholds and current_value > thresholds["max"]:
                await self._create_alert(
                    AlertLevel.WARNING,
                    f"{metric_name} Above Threshold",
                    f"{metric_name} ({current_value:.2f}) is above maximum threshold ({thresholds['max']:.2f})",
                    metric_key,
                    current_value,
                    thresholds["max"]
                )
            
        except Exception as e:
            logger.error(f"Error checking threshold alert for {metric_name}: {e}")
    
    async def _create_alert(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        metric: Optional[str] = None,
        current_value: Optional[float] = None,
        threshold: Optional[float] = None
    ):
        """Create a new alert"""
        try:
            # Check if similar alert already exists (avoid spam)
            recent_alerts = [a for a in self.alerts if 
                           (datetime.now() - a.timestamp).seconds < 300]  # 5 minutes
            
            for existing_alert in recent_alerts:
                if (existing_alert.title == title and 
                    existing_alert.level == level and
                    not existing_alert.acknowledged):
                    return  # Don't create duplicate
            
            # Create new alert
            alert = Alert(
                alert_id=f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                timestamp=datetime.now(),
                level=level,
                title=title,
                message=message,
                metric=metric,
                current_value=current_value,
                threshold=threshold
            )
            
            self.alerts.append(alert)
            self.alert_count += 1
            
            # Log alert
            await self.logger.log(
                LogLevel.WARNING if level == AlertLevel.WARNING else LogLevel.ERROR,
                LogCategory.ALERT,
                f"Alert: {title} - {message}",
                {
                    "alert_id": alert.alert_id,
                    "metric": metric,
                    "current_value": current_value,
                    "threshold": threshold
                }
            )
            
        except Exception as e:
            logger.error(f"Error creating alert: {e}")


# Global dashboard instance
demo_dashboard = None


async def get_demo_dashboard() -> DemoDashboard:
    """Get global demo dashboard instance"""
    global demo_dashboard
    if demo_dashboard is None:
        manager = await get_demo_manager()
        demo_dashboard = DemoDashboard(manager)
    return demo_dashboard


# WebSocket-like interface for real-time updates
class DashboardWebSocket:
    """WebSocket-like interface for dashboard updates"""
    
    def __init__(self, dashboard: DemoDashboard):
        self.dashboard = dashboard
        self.subscribers: List[Any] = []
    
    async def subscribe(self, callback: callable):
        """Subscribe to dashboard updates"""
        self.subscribers.append(callback)
    
    async def unsubscribe(self, callback: callable):
        """Unsubscribe from dashboard updates"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    async def broadcast_update(self, data: DashboardData):
        """Broadcast update to all subscribers"""
        for callback in self.subscribers:
            try:
                await callback(data)
            except Exception as e:
                logger.error(f"Error broadcasting to subscriber: {e}")


if __name__ == "__main__":
    # Example usage
    async def main():
        # Get demo manager and enable demo mode
        manager = await get_demo_manager()
        await manager.initialize()
        await manager.enable_demo_mode()
        
        # Start dashboard
        dashboard = await get_demo_dashboard()
        await dashboard.start_dashboard()
        
        # Simulate some trading activity
        portfolio = await get_virtual_portfolio()
        await portfolio.update_position("AAPL", 10, 150.0, 1.50)
        await portfolio.update_position("MSFT", 5, 300.0, 1.50)
        
        # Get dashboard data
        data = await dashboard.get_dashboard_data()
        print(f"Dashboard data: {data}")
        
        # Export dashboard data
        await dashboard.export_dashboard_data("dashboard_export.json")
        
        await dashboard.stop_dashboard()
    
    asyncio.run(main())
