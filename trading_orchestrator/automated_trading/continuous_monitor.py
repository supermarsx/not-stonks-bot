"""
Continuous Monitoring System

Real-time monitoring of trading operations, positions, risk, performance, and system health
with automated alerts and health management.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import psutil
import json

from loguru import logger

from .config import AutomatedTradingConfig


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"


class AlertLevel(Enum):
    """Alert levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Metric types"""
    PERFORMANCE = "performance"
    RISK = "risk"
    SYSTEM = "system"
    TRADING = "trading"
    POSITION = "position"


@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    process_count: int
    uptime: float
    health_status: HealthStatus


@dataclass
class TradingMetrics:
    """Trading performance metrics"""
    timestamp: datetime
    total_pnl: float
    daily_pnl: float
    unrealized_pnl: float
    realized_pnl: float
    active_positions: int
    total_orders: int
    successful_orders: int
    failed_orders: int
    win_rate: float
    average_trade_duration: float
    sharpe_ratio: float
    max_drawdown: float
    current_drawdown: float
    volatility: float
    beta: float


@dataclass
class RiskMetrics:
    """Risk management metrics"""
    timestamp: datetime
    portfolio_var: float
    max_position_exposure: float
    sector_exposure: Dict[str, float]
    concentration_risk: float
    leverage_ratio: float
    liquidity_score: float
    correlation_risk: float
    tail_risk: float
    daily_var: float
    risk_on_score: float


@dataclass
class PositionMetrics:
    """Position monitoring metrics"""
    timestamp: datetime
    total_positions: int
    long_positions: int
    short_positions: int
    average_position_size: float
    position_concentration: float
    unrealized_pnl_by_symbol: Dict[str, float]
    position_duration: Dict[str, float]
    stop_loss_hits: int
    take_profit_hits: int
    trailing_stop_active: int


@dataclass
class HealthAlert:
    """Health alert data"""
    alert_id: str
    timestamp: datetime
    level: AlertLevel
    category: MetricType
    title: str
    message: str
    value: float
    threshold: float
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    actions_taken: List[str] = field(default_factory=list)


class ContinuousMonitoringSystem:
    """
    Continuous Monitoring System
    
    Features:
    - Real-time system health monitoring
    - Trading performance tracking
    - Risk metrics monitoring
    - Position tracking and analysis
    - Automated alerting system
    - Performance reporting
    - Health-based automation controls
    """
    
    def __init__(self, config: AutomatedTradingConfig):
        self.config = config
        
        # Metrics storage
        self.system_metrics_history: deque = deque(maxlen=1000)
        self.trading_metrics_history: deque = deque(maxlen=1000)
        self.risk_metrics_history: deque = deque(maxlen=1000)
        self.position_metrics_history: deque = deque(maxlen=1000)
        
        # Current metrics
        self.current_system_metrics: Optional[SystemMetrics] = None
        self.current_trading_metrics: Optional[TradingMetrics] = None
        self.current_risk_metrics: Optional[RiskMetrics] = None
        self.current_position_metrics: Optional[PositionMetrics] = None
        
        # Alert management
        self.active_alerts: Dict[str, HealthAlert] = {}
        self.alert_history: List[HealthAlert] = []
        self.alert_callbacks: List[Callable] = []
        
        # Monitoring configuration
        self.monitoring_interval = 30  # seconds
        self.alert_cooldown = 300  # 5 minutes
        self.metrics_retention_days = 7
        
        # Performance tracking
        self.start_time = datetime.utcnow()
        self.baseline_metrics: Dict[str, float] = {}
        
        # Component status
        self.monitoring_active = False
        self.last_health_check: Optional[datetime] = None
        
        logger.info("Continuous Monitoring System initialized")
    
    async def start(self):
        """Start the monitoring system"""
        try:
            logger.info("🚀 Starting Continuous Monitoring System...")
            
            # Start monitoring tasks
            self.system_monitor_task = asyncio.create_task(self._system_monitoring_loop())
            self.trading_monitor_task = asyncio.create_task(self._trading_monitoring_loop())
            self.risk_monitor_task = asyncio.create_task(self._risk_monitoring_loop())
            self.position_monitor_task = asyncio.create_task(self._position_monitoring_loop())
            self.alert_processor_task = asyncio.create_task(self._alert_processing_loop())
            
            # Establish baselines
            await self._establish_baselines()
            
            self.monitoring_active = True
            logger.success("✅ Continuous Monitoring System started")
            
        except Exception as e:
            logger.error(f"Error starting monitoring system: {e}")
            raise
    
    async def stop(self):
        """Stop the monitoring system"""
        logger.info("🛑 Stopping Continuous Monitoring System...")
        
        self.monitoring_active = False
        
        # Cancel all monitoring tasks
        for task_name in ['system_monitor_task', 'trading_monitor_task', 
                         'risk_monitor_task', 'position_monitor_task', 
                         'alert_processor_task']:
            task = getattr(self, task_name, None)
            if task and not task.done():
                task.cancel()
        
        # Generate final report
        await self._generate_final_report()
        
        logger.success("✅ Continuous Monitoring System stopped")
    
    async def _establish_baselines(self):
        """Establish baseline metrics for comparison"""
        try:
            # Run baseline measurements
            for _ in range(10):  # Take 10 samples
                await self._collect_system_metrics()
                await asyncio.sleep(1)
            
            # Calculate baseline averages
            if self.system_metrics_history:
                recent_metrics = list(self.system_metrics_history)[-10:]
                self.baseline_metrics = {
                    'cpu_usage': sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics),
                    'memory_usage': sum(m.memory_usage for m in recent_metrics) / len(recent_metrics),
                    'disk_usage': sum(m.disk_usage for m in recent_metrics) / len(recent_metrics)
                }
                
                logger.info(f"Baseline metrics established: CPU {self.baseline_metrics['cpu_usage']:.1f}%, "
                          f"Memory {self.baseline_metrics['memory_usage']:.1f}%")
        
        except Exception as e:
            logger.error(f"Error establishing baselines: {e}")
    
    async def _system_monitoring_loop(self):
        """Monitor system resources and health"""
        while self.monitoring_active:
            try:
                metrics = await self._collect_system_metrics()
                self.current_system_metrics = metrics
                self.system_metrics_history.append(metrics)
                
                # Check for health alerts
                await self._check_system_health(metrics)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _trading_monitoring_loop(self):
        """Monitor trading performance"""
        while self.monitoring_active:
            try:
                metrics = await self._collect_trading_metrics()
                self.current_trading_metrics = metrics
                self.trading_metrics_history.append(metrics)
                
                # Check trading performance alerts
                await self._check_trading_health(metrics)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in trading monitoring: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _risk_monitoring_loop(self):
        """Monitor risk metrics"""
        while self.monitoring_active:
            try:
                metrics = await self._collect_risk_metrics()
                self.current_risk_metrics = metrics
                self.risk_metrics_history.append(metrics)
                
                # Check risk alerts
                await self._check_risk_health(metrics)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in risk monitoring: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _position_monitoring_loop(self):
        """Monitor positions"""
        while self.monitoring_active:
            try:
                metrics = await self._collect_position_metrics()
                self.current_position_metrics = metrics
                self.position_metrics_history.append(metrics)
                
                # Check position alerts
                await self._check_position_health(metrics)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in position monitoring: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _alert_processing_loop(self):
        """Process and manage alerts"""
        while self.monitoring_active:
            try:
                # Process active alerts
                await self._process_alerts()
                
                # Clean up old data
                await self._cleanup_old_metrics()
                
                await asyncio.sleep(60)  # Process every minute
                
            except Exception as e:
                logger.error(f"Error in alert processing: {e}")
                await asyncio.sleep(60)
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system performance metrics"""
        try:
            # Get system stats
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            process_count = len(psutil.pids())
            
            # Calculate uptime
            uptime = (datetime.utcnow() - self.start_time).total_seconds()
            
            # Determine health status
            health_status = HealthStatus.HEALTHY
            if cpu_percent > 80 or memory.percent > 85:
                health_status = HealthStatus.WARNING
            if cpu_percent > 95 or memory.percent > 95:
                health_status = HealthStatus.CRITICAL
            
            return SystemMetrics(
                timestamp=datetime.utcnow(),
                cpu_usage=cpu_percent,
                memory_usage=memory.percent,
                disk_usage=(disk.used / disk.total) * 100,
                network_io={
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                },
                process_count=process_count,
                uptime=uptime,
                health_status=health_status
            )
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return SystemMetrics(
                timestamp=datetime.utcnow(),
                cpu_usage=0,
                memory_usage=0,
                disk_usage=0,
                network_io={},
                process_count=0,
                uptime=0,
                health_status=HealthStatus.FAILED
            )
    
    async def _collect_trading_metrics(self) -> TradingMetrics:
        """Collect trading performance metrics"""
        try:
            # Mock trading metrics (in real implementation, get from trading engine)
            current_time = datetime.utcnow()
            
            # Calculate metrics from history
            total_trades = len(self.trading_metrics_history)
            if total_trades > 0:
                recent_metrics = list(self.trading_metrics_history)[-10:]
                avg_win_rate = sum(m.win_rate for m in recent_metrics) / len(recent_metrics)
                avg_drawdown = sum(m.current_drawdown for m in recent_metrics) / len(recent_metrics)
            else:
                avg_win_rate = 0.5
                avg_drawdown = 0.0
            
            return TradingMetrics(
                timestamp=current_time,
                total_pnl=1000.0,  # Mock P&L
                daily_pnl=50.0,   # Mock daily P&L
                unrealized_pnl=200.0,
                realized_pnl=800.0,
                active_positions=5,
                total_orders=100,
                successful_orders=95,
                failed_orders=5,
                win_rate=avg_win_rate,
                average_trade_duration=1800.0,  # 30 minutes
                sharpe_ratio=1.2,
                max_drawdown=max(avg_drawdown, 0.05),
                current_drawdown=avg_drawdown,
                volatility=0.15,
                beta=1.0
            )
            
        except Exception as e:
            logger.error(f"Error collecting trading metrics: {e}")
            return TradingMetrics(
                timestamp=datetime.utcnow(),
                total_pnl=0,
                daily_pnl=0,
                unrealized_pnl=0,
                realized_pnl=0,
                active_positions=0,
                total_orders=0,
                successful_orders=0,
                failed_orders=0,
                win_rate=0,
                average_trade_duration=0,
                sharpe_ratio=0,
                max_drawdown=0,
                current_drawdown=0,
                volatility=0,
                beta=0
            )
    
    async def _collect_risk_metrics(self) -> RiskMetrics:
        """Collect risk management metrics"""
        try:
            # Mock risk metrics (in real implementation, get from risk manager)
            return RiskMetrics(
                timestamp=datetime.utcnow(),
                portfolio_var=5000.0,
                max_position_exposure=0.15,  # 15%
                sector_exposure={
                    "technology": 0.25,
                    "healthcare": 0.15,
                    "financial": 0.20,
                    "energy": 0.10
                },
                concentration_risk=0.3,
                leverage_ratio=1.2,
                liquidity_score=0.8,
                correlation_risk=0.4,
                tail_risk=0.02,
                daily_var=2000.0,
                risk_on_score=0.6
            )
            
        except Exception as e:
            logger.error(f"Error collecting risk metrics: {e}")
            return RiskMetrics(
                timestamp=datetime.utcnow(),
                portfolio_var=0,
                max_position_exposure=0,
                sector_exposure={},
                concentration_risk=0,
                leverage_ratio=0,
                liquidity_score=0,
                correlation_risk=0,
                tail_risk=0,
                daily_var=0,
                risk_on_score=0
            )
    
    async def _collect_position_metrics(self) -> PositionMetrics:
        """Collect position monitoring metrics"""
        try:
            # Mock position metrics (in real implementation, get from position tracker)
            return PositionMetrics(
                timestamp=datetime.utcnow(),
                total_positions=5,
                long_positions=3,
                short_positions=2,
                average_position_size=10000.0,
                position_concentration=0.25,
                unrealized_pnl_by_symbol={
                    "AAPL": 150.0,
                    "GOOGL": -50.0,
                    "MSFT": 100.0,
                    "TSLA": -25.0,
                    "AMZN": 75.0
                },
                position_duration={
                    "AAPL": 1800.0,
                    "GOOGL": 3600.0,
                    "MSFT": 900.0,
                    "TSLA": 2700.0,
                    "AMZN": 1200.0
                },
                stop_loss_hits=1,
                take_profit_hits=2,
                trailing_stop_active=3
            )
            
        except Exception as e:
            logger.error(f"Error collecting position metrics: {e}")
            return PositionMetrics(
                timestamp=datetime.utcnow(),
                total_positions=0,
                long_positions=0,
                short_positions=0,
                average_position_size=0,
                position_concentration=0,
                unrealized_pnl_by_symbol={},
                position_duration={},
                stop_loss_hits=0,
                take_profit_hits=0,
                trailing_stop_active=0
            )
    
    async def _check_system_health(self, metrics: SystemMetrics):
        """Check for system health alerts"""
        # CPU usage alert
        if metrics.cpu_usage > 80:
            await self._create_alert(
                AlertLevel.WARNING,
                MetricType.SYSTEM,
                "High CPU Usage",
                f"CPU usage at {metrics.cpu_usage:.1f}%",
                metrics.cpu_usage,
                80.0
            )
        
        # Memory usage alert
        if metrics.memory_usage > 85:
            await self._create_alert(
                AlertLevel.WARNING,
                MetricType.SYSTEM,
                "High Memory Usage",
                f"Memory usage at {metrics.memory_usage:.1f}%",
                metrics.memory_usage,
                85.0
            )
        
        # Critical system alert
        if metrics.cpu_usage > 95 or metrics.memory_usage > 95:
            await self._create_alert(
                AlertLevel.CRITICAL,
                MetricType.SYSTEM,
                "Critical System Resource Usage",
                f"Critical resource usage: CPU {metrics.cpu_usage:.1f}%, Memory {metrics.memory_usage:.1f}%",
                max(metrics.cpu_usage, metrics.memory_usage),
                95.0
            )
    
    async def _check_trading_health(self, metrics: TradingMetrics):
        """Check for trading performance alerts"""
        # High failure rate alert
        if metrics.total_orders > 10:  # Only alert if meaningful number of orders
            failure_rate = metrics.failed_orders / metrics.total_orders
            if failure_rate > 0.1:  # 10% failure rate
                await self._create_alert(
                    AlertLevel.WARNING,
                    MetricType.TRADING,
                    "High Order Failure Rate",
                    f"Order failure rate: {failure_rate:.1%}",
                    failure_rate,
                    0.1
                )
        
        # High drawdown alert
        if metrics.current_drawdown > 0.05:  # 5% drawdown
            await self._create_alert(
                AlertLevel.WARNING,
                MetricType.PERFORMANCE,
                "High Drawdown",
                f"Current drawdown: {metrics.current_drawdown:.1%}",
                metrics.current_drawdown,
                0.05
            )
        
        # Extreme drawdown alert
        if metrics.current_drawdown > 0.10:  # 10% drawdown
            await self._create_alert(
                AlertLevel.CRITICAL,
                MetricType.PERFORMANCE,
                "Extreme Drawdown",
                f"Extreme drawdown: {metrics.current_drawdown:.1%}",
                metrics.current_drawdown,
                0.10
            )
    
    async def _check_risk_health(self, metrics: RiskMetrics):
        """Check for risk alerts"""
        # High VaR alert
        if metrics.portfolio_var > 10000:  # $10k VaR
            await self._create_alert(
                AlertLevel.WARNING,
                MetricType.RISK,
                "High Portfolio VaR",
                f"Portfolio VaR: ${metrics.portfolio_var:,.0f}",
                metrics.portfolio_var,
                10000
            )
        
        # High concentration risk
        if metrics.concentration_risk > 0.5:  # 50% concentration
            await self._create_alert(
                AlertLevel.WARNING,
                MetricType.RISK,
                "High Concentration Risk",
                f"Position concentration: {metrics.concentration_risk:.1%}",
                metrics.concentration_risk,
                0.5
            )
        
        # High leverage alert
        if metrics.leverage_ratio > 2.0:  # 2x leverage
            await self._create_alert(
                AlertLevel.WARNING,
                MetricType.RISK,
                "High Leverage",
                f"Leverage ratio: {metrics.leverage_ratio:.1f}x",
                metrics.leverage_ratio,
                2.0
            )
    
    async def _check_position_health(self, metrics: PositionMetrics):
        """Check for position alerts"""
        # High position count
        if metrics.total_positions > 20:
            await self._create_alert(
                AlertLevel.WARNING,
                MetricType.POSITION,
                "High Position Count",
                f"Total positions: {metrics.total_positions}",
                metrics.total_positions,
                20
            )
        
        # High concentration
        if metrics.position_concentration > 0.3:  # 30% in single position
            await self._create_alert(
                AlertLevel.WARNING,
                MetricType.POSITION,
                "High Position Concentration",
                f"Max position concentration: {metrics.position_concentration:.1%}",
                metrics.position_concentration,
                0.3
            )
        
        # Frequent stop loss hits
        if metrics.stop_loss_hits > 5:
            await self._create_alert(
                AlertLevel.WARNING,
                MetricType.POSITION,
                "Frequent Stop Loss Hits",
                f"Stop loss hits in session: {metrics.stop_loss_hits}",
                metrics.stop_loss_hits,
                5
            )
    
    async def _create_alert(self, level: AlertLevel, category: MetricType, 
                          title: str, message: str, value: float, threshold: float):
        """Create a new alert"""
        alert_id = f"{category.value}_{title.replace(' ', '_').lower()}"
        
        # Check if alert already exists and not in cooldown
        existing_alert = self.active_alerts.get(alert_id)
        if existing_alert and not self._is_alert_ready_for_update(existing_alert):
            return
        
        # Create new or update existing alert
        alert = HealthAlert(
            alert_id=alert_id,
            timestamp=datetime.utcnow(),
            level=level,
            category=category,
            title=title,
            message=message,
            value=value,
            threshold=threshold,
            resolved=False
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Notify callbacks
        await self._notify_alert_callbacks(alert)
        
        # Log alert
        log_level = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.ERROR: logger.error,
            AlertLevel.CRITICAL: logger.critical
        }
        
        log_level[level](f"🚨 ALERT [{level.value.upper()}] {title}: {message}")
    
    def _is_alert_ready_for_update(self, alert: HealthAlert) -> bool:
        """Check if alert is ready for update (not in cooldown)"""
        time_since_alert = (datetime.utcnow() - alert.timestamp).total_seconds()
        return time_since_alert > self.alert_cooldown
    
    async def _notify_alert_callbacks(self, alert: HealthAlert):
        """Notify all alert callbacks"""
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    async def _process_alerts(self):
        """Process active alerts"""
        try:
            current_time = datetime.utcnow()
            
            for alert_id, alert in list(self.active_alerts.items()):
                # Check if alert should be resolved
                if not alert.resolved:
                    # Simple resolution logic - can be enhanced
                    time_elapsed = (current_time - alert.timestamp).total_seconds()
                    
                    # Auto-resolve certain alerts after period
                    if (alert.level == AlertLevel.WARNING and time_elapsed > 1800) or \
                       (alert.level == AlertLevel.CRITICAL and time_elapsed > 600):
                        
                        alert.resolved = True
                        alert.resolution_time = current_time
                        
                        logger.info(f"✅ Alert resolved: {alert.title}")
                        
                        # Remove from active alerts after resolution period
                        if time_elapsed > 3600:  # 1 hour
                            del self.active_alerts[alert_id]
            
        except Exception as e:
            logger.error(f"Error processing alerts: {e}")
    
    async def _cleanup_old_metrics(self):
        """Clean up old metrics data"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=self.metrics_retention_days)
            
            # Clean each metrics history
            for history_attr in ['system_metrics_history', 'trading_metrics_history', 
                               'risk_metrics_history', 'position_metrics_history']:
                history = getattr(self, history_attr)
                while history and history[0].timestamp < cutoff_time:
                    history.popleft()
            
            # Clean old alerts
            self.alert_history = [
                alert for alert in self.alert_history
                if alert.timestamp > cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"Error cleaning up old metrics: {e}")
    
    async def _generate_final_report(self):
        """Generate final monitoring report"""
        try:
            report = {
                "monitoring_duration": (datetime.utcnow() - self.start_time).total_seconds(),
                "total_alerts_generated": len(self.alert_history),
                "active_alerts_at_shutdown": len(self.active_alerts),
                "average_system_health": self._calculate_average_system_health(),
                "final_metrics": {
                    "system": self.current_system_metrics.__dict__ if self.current_system_metrics else None,
                    "trading": self.current_trading_metrics.__dict__ if self.current_trading_metrics else None,
                    "risk": self.current_risk_metrics.__dict__ if self.current_risk_metrics else None,
                    "position": self.current_position_metrics.__dict__ if self.current_position_metrics else None
                }
            }
            
            # Save report
            report_file = f"monitoring_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"📊 Final monitoring report saved: {report_file}")
            
        except Exception as e:
            logger.error(f"Error generating final report: {e}")
    
    def _calculate_average_system_health(self) -> float:
        """Calculate average system health score"""
        if not self.system_metrics_history:
            return 100.0
        
        recent_metrics = list(self.system_metrics_history)[-10:]
        health_scores = []
        
        for metrics in recent_metrics:
            score = 100.0
            score -= max(0, metrics.cpu_usage - 50) * 2  # Penalty for high CPU
            score -= max(0, metrics.memory_usage - 50) * 2  # Penalty for high memory
            score = max(0, score)
            health_scores.append(score)
        
        return sum(health_scores) / len(health_scores)
    
    def register_alert_callback(self, callback: Callable[[HealthAlert], None]):
        """Register alert callback"""
        self.alert_callbacks.append(callback)
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            "total_pnl": self.current_trading_metrics.total_pnl if self.current_trading_metrics else 0,
            "daily_pnl": self.current_trading_metrics.daily_pnl if self.current_trading_metrics else 0,
            "active_positions": self.current_position_metrics.total_positions if self.current_position_metrics else 0,
            "win_rate": self.current_trading_metrics.win_rate if self.current_trading_metrics else 0,
            "current_drawdown": self.current_trading_metrics.current_drawdown if self.current_trading_metrics else 0,
            "system_health": self._calculate_average_system_health(),
            "active_alerts": len(self.active_alerts),
            "monitoring_uptime": (datetime.utcnow() - self.start_time).total_seconds()
        }
    
    async def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        return {
            "monitoring_active": self.monitoring_active,
            "start_time": self.start_time.isoformat(),
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "current_metrics": {
                "system": self.current_system_metrics.__dict__ if self.current_system_metrics else None,
                "trading": self.current_trading_metrics.__dict__ if self.current_trading_metrics else None,
                "risk": self.current_risk_metrics.__dict__ if self.current_risk_metrics else None,
                "position": self.current_position_metrics.__dict__ if self.current_position_metrics else None
            },
            "active_alerts": {
                alert_id: alert.__dict__ 
                for alert_id, alert in self.active_alerts.items()
            },
            "metrics_counts": {
                "system_metrics": len(self.system_metrics_history),
                "trading_metrics": len(self.trading_metrics_history),
                "risk_metrics": len(self.risk_metrics_history),
                "position_metrics": len(self.position_metrics_history),
                "total_alerts": len(self.alert_history)
            }
        }