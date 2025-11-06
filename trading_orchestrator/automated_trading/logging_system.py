"""
Automated Trading Logging System

Comprehensive logging for automated trading operations including:
- Automated trading logs
- Performance reports  
- Risk monitoring alerts
- Decision audit trails
- Error reporting and handling
"""

import os
import json
import gzip
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import asyncio

from loguru import logger


@dataclass
class TradingLogEntry:
    """Individual trading log entry"""
    timestamp: datetime
    level: str
    category: str  # trading, risk, system, decision, performance
    message: str
    data: Dict[str, Any]
    user_id: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class PerformanceReport:
    """Performance report data"""
    report_id: str
    timestamp: datetime
    report_type: str  # daily, weekly, monthly, session
    period_start: datetime
    period_end: datetime
    
    # Trading metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    
    # Risk metrics
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    total_var: float
    
    # System metrics
    uptime_seconds: float
    system_availability: float
    error_count: int
    avg_response_time: float
    
    # Strategy breakdown
    strategy_performance: Dict[str, Dict[str, Any]]
    
    # Market conditions
    market_regime: str
    average_volatility: float
    total_market_hours: float


@dataclass
class RiskAlert:
    """Risk alert log entry"""
    alert_id: str
    timestamp: datetime
    alert_level: str  # info, warning, error, critical
    category: str  # position, portfolio, system, compliance
    title: str
    description: str
    metric_value: float
    threshold_value: float
    actions_taken: List[str]
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class DecisionAudit:
    """Decision audit trail"""
    decision_id: str
    timestamp: datetime
    decision_type: str  # trade, risk, strategy, exit
    automation_level: str
    confidence: float
    reasoning: str
    inputs: Dict[str, Any]
    output: Dict[str, Any]
    execution_result: Optional[Dict[str, Any]] = None
    pnl: Optional[float] = None


class TradingLogger:
    """
    Comprehensive logging system for automated trading
    
    Features:
    - Multi-level logging (INFO, WARNING, ERROR, CRITICAL)
    - Category-based filtering (trading, risk, system, decision)
    - Automatic log rotation and compression
    - Performance report generation
    - Risk alert logging
    - Decision audit trails
    - Real-time log streaming
    """
    
    def __init__(self, config):
        self.config = config
        
        # Log directories
        self.log_dir = Path("logs/automated_trading")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Specific log files
        self.trading_log_file = self.log_dir / "trading.log"
        self.risk_log_file = self.log_dir / "risk.log"
        self.system_log_file = self.log_dir / "system.log"
        self.decision_log_file = self.log_dir / "decisions.log"
        self.performance_log_file = self.log_dir / "performance.log"
        
        # Performance reports directory
        self.reports_dir = self.log_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        # In-memory storage for recent logs (for real-time queries)
        self.recent_trading_logs: List[TradingLogEntry] = []
        self.recent_risk_alerts: List[RiskAlert] = []
        self.recent_decisions: List[DecisionAudit] = []
        
        # Log retention settings
        self.max_log_files = 30  # Keep 30 daily log files
        self.compression_enabled = True
        
        # Setup logging
        self._setup_loguru_logging()
        
        logger.info("Trading Logger initialized")
    
    def _setup_loguru_logging(self):
        """Configure loguru logging for automated trading"""
        
        # Remove default handler
        logger.remove()
        
        # Trading logs - All automated trading activity
        logger.add(
            self.trading_log_file,
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            rotation="1 day",
            retention=f"{self.max_log_files} days",
            compression="gz" if self.compression_enabled else None,
            enqueue=True
        )
        
        # Risk logs - Risk management and alerts
        logger.add(
            self.risk_log_file,
            level="INFO", 
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            rotation="1 day",
            retention=f"{self.max_log_files} days",
            compression="gz" if self.compression_enabled else None,
            enqueue=True
        )
        
        # System logs - System health and performance
        logger.add(
            self.system_log_file,
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            rotation="1 day", 
            retention=f"{self.max_log_files} days",
            compression="gz" if self.compression_enabled else None,
            enqueue=True
        )
        
        # Decision logs - All trading decisions and reasoning
        logger.add(
            self.decision_log_file,
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            rotation="1 day",
            retention=f"{self.max_log_files} days", 
            compression="gz" if self.compression_enabled else None,
            enqueue=True
        )
        
        # Performance logs - Performance metrics and reports
        logger.add(
            self.performance_log_file,
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            rotation="1 day",
            retention=f"{self.max_log_files} days",
            compression="gz" if self.compression_enabled else None,
            enqueue=True
        )
    
    async def start(self):
        """Start the logging system"""
        try:
            logger.info("🚀 Starting Trading Logger...")
            
            # Create session start log
            await self.log_system_event("INFO", "System started", {
                "timestamp": datetime.utcnow().isoformat(),
                "version": "1.0.0"
            })
            
            logger.success("✅ Trading Logger started")
            
        except Exception as e:
            logger.error(f"Error starting Trading Logger: {e}")
            raise
    
    async def stop(self):
        """Stop the logging system"""
        try:
            logger.info("🛑 Stopping Trading Logger...")
            
            # Create session end log
            await self.log_system_event("INFO", "System stopped", {
                "timestamp": datetime.utcnow().isoformat()
            })
            
            logger.success("✅ Trading Logger stopped")
            
        except Exception as e:
            logger.error(f"Error stopping Trading Logger: {e}")
    
    async def log_trading_event(self, level: str, message: str, data: Dict[str, Any]):
        """Log trading event"""
        try:
            # Create log entry
            entry = TradingLogEntry(
                timestamp=datetime.utcnow(),
                level=level,
                category="trading",
                message=message,
                data=data
            )
            
            # Add to recent logs
            self.recent_trading_logs.append(entry)
            
            # Keep only recent logs in memory
            if len(self.recent_trading_logs) > 1000:
                self.recent_trading_logs = self.recent_trading_logs[-500:]
            
            # Log to appropriate file
            log_data = {
                "timestamp": entry.timestamp.isoformat(),
                "message": message,
                "data": data
            }
            
            logger.log(level.upper(), f"TRADING: {message} | {json.dumps(log_data)}")
            
        except Exception as e:
            logger.error(f"Error logging trading event: {e}")
    
    async def log_risk_event(self, level: str, message: str, data: Dict[str, Any]):
        """Log risk management event"""
        try:
            entry = TradingLogEntry(
                timestamp=datetime.utcnow(),
                level=level,
                category="risk",
                message=message,
                data=data
            )
            
            logger.log(level.upper(), f"RISK: {message} | {json.dumps(data)}")
            
        except Exception as e:
            logger.error(f"Error logging risk event: {e}")
    
    async def log_system_event(self, level: str, message: str, data: Dict[str, Any]):
        """Log system event"""
        try:
            logger.log(level.upper(), f"SYSTEM: {message} | {json.dumps(data)}")
            
        except Exception as e:
            logger.error(f"Error logging system event: {e}")
    
    async def log_decision(self, decision: DecisionAudit):
        """Log trading decision"""
        try:
            # Add to recent decisions
            self.recent_decisions.append(decision)
            
            # Keep only recent decisions in memory
            if len(self.recent_decisions) > 1000:
                self.recent_decisions = self.recent_decisions[-500:]
            
            # Log decision
            decision_data = {
                "decision_id": decision.decision_id,
                "timestamp": decision.timestamp.isoformat(),
                "decision_type": decision.decision_type,
                "automation_level": decision.automation_level,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning,
                "inputs": decision.inputs,
                "output": decision.output,
                "execution_result": decision.execution_result,
                "pnl": decision.pnl
            }
            
            logger.info(f"DECISION: {decision.decision_type} | {json.dumps(decision_data)}")
            
        except Exception as e:
            logger.error(f"Error logging decision: {e}")
    
    async def log_risk_alert(self, alert: RiskAlert):
        """Log risk alert"""
        try:
            # Add to recent alerts
            self.recent_risk_alerts.append(alert)
            
            # Keep only recent alerts in memory
            if len(self.recent_risk_alerts) > 500:
                self.recent_risk_alerts = self.recent_risk_alerts[-250:]
            
            # Log alert
            alert_data = {
                "alert_id": alert.alert_id,
                "timestamp": alert.timestamp.isoformat(),
                "level": alert.alert_level,
                "category": alert.category,
                "title": alert.title,
                "description": alert.description,
                "metric_value": alert.metric_value,
                "threshold_value": alert.threshold_value,
                "actions_taken": alert.actions_taken,
                "resolved": alert.resolved,
                "resolution_time": alert.resolution_time.isoformat() if alert.resolution_time else None
            }
            
            # Choose log level based on alert level
            log_level = {
                "info": "INFO",
                "warning": "WARNING", 
                "error": "ERROR",
                "critical": "CRITICAL"
            }.get(alert.alert_level, "INFO")
            
            logger.log(log_level, f"ALERT: {alert.title} | {json.dumps(alert_data)}")
            
        except Exception as e:
            logger.error(f"Error logging risk alert: {e}")
    
    async def generate_performance_report(self, report_type: str = "daily") -> PerformanceReport:
        """Generate comprehensive performance report"""
        try:
            now = datetime.utcnow()
            
            # Calculate report period
            if report_type == "daily":
                period_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
                period_end = now
            elif report_type == "weekly":
                days_back = now.weekday()
                period_start = (now - timedelta(days=days_back)).replace(hour=0, minute=0, second=0, microsecond=0)
                period_end = now
            elif report_type == "monthly":
                period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                period_end = now
            else:  # session
                # Get recent session start (simplified)
                period_start = now - timedelta(hours=8)  # Assume 8-hour session
                period_end = now
            
            # Generate report ID
            report_id = f"report_{report_type}_{now.strftime('%Y%m%d_%H%M%S')}"
            
            # Calculate metrics from recent logs and data
            total_trades = len([log for log in self.recent_trading_logs 
                              if log.timestamp >= period_start and log.data.get('trade_executed')])
            
            winning_trades = len([log for log in self.recent_trading_logs 
                                if log.timestamp >= period_start and log.data.get('pnl', 0) > 0])
            
            losing_trades = total_trades - winning_trades
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Mock additional metrics (in real implementation, calculate from actual data)
            total_pnl = sum(log.data.get('pnl', 0) for log in self.recent_trading_logs 
                          if log.timestamp >= period_start)
            realized_pnl = total_pnl * 0.8  # Mock 80% realized
            unrealized_pnl = total_pnl * 0.2  # Mock 20% unrealized
            
            max_drawdown = 0.05  # Mock 5%
            current_drawdown = 0.02  # Mock 2%
            sharpe_ratio = 1.2  # Mock
            total_var = 5000.0  # Mock
            
            uptime_seconds = (now - period_start).total_seconds()
            system_availability = 0.995  # Mock 99.5%
            error_count = len([log for log in self.recent_trading_logs 
                             if log.level == "ERROR" and log.timestamp >= period_start])
            avg_response_time = 0.05  # Mock 50ms
            
            # Strategy performance breakdown
            strategy_performance = {
                "trend_following": {"trades": total_trades // 2, "pnl": total_pnl * 0.4, "win_rate": 0.6},
                "mean_reversion": {"trades": total_trades // 3, "pnl": total_pnl * 0.3, "win_rate": 0.7},
                "momentum": {"trades": total_trades // 6, "pnl": total_pnl * 0.3, "win_rate": 0.5}
            }
            
            # Market conditions
            market_regime = "bull_trending"  # Mock
            average_volatility = 0.15  # Mock 15%
            total_market_hours = 6.5  # Mock 6.5 hours
            
            # Create performance report
            report = PerformanceReport(
                report_id=report_id,
                timestamp=now,
                report_type=report_type,
                period_start=period_start,
                period_end=period_end,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                total_pnl=total_pnl,
                realized_pnl=realized_pnl,
                unrealized_pnl=unrealized_pnl,
                max_drawdown=max_drawdown,
                current_drawdown=current_drawdown,
                sharpe_ratio=sharpe_ratio,
                total_var=total_var,
                uptime_seconds=uptime_seconds,
                system_availability=system_availability,
                error_count=error_count,
                avg_response_time=avg_response_time,
                strategy_performance=strategy_performance,
                market_regime=market_regime,
                average_volatility=average_volatility,
                total_market_hours=total_market_hours
            )
            
            # Save report to file
            await self._save_performance_report(report)
            
            # Log report generation
            await self.log_performance_event("INFO", f"Performance report generated: {report_type}", {
                "report_id": report_id,
                "total_trades": total_trades,
                "total_pnl": total_pnl,
                "win_rate": win_rate,
                "period": f"{period_start.isoformat()} to {period_end.isoformat()}"
            })
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            raise
    
    async def log_performance_event(self, level: str, message: str, data: Dict[str, Any]):
        """Log performance event"""
        try:
            logger.log(level.upper(), f"PERFORMANCE: {message} | {json.dumps(data)}")
            
        except Exception as e:
            logger.error(f"Error logging performance event: {e}")
    
    async def _save_performance_report(self, report: PerformanceReport):
        """Save performance report to file"""
        try:
            # Save as JSON
            report_file = self.reports_dir / f"{report.report_id}.json"
            with open(report_file, 'w') as f:
                json.dump(asdict(report), f, indent=2, default=str)
            
            # Also save as compressed text summary
            summary_file = self.reports_dir / f"{report.report_id}_summary.txt.gz"
            summary_text = self._generate_report_summary(report)
            
            with gzip.open(summary_file, 'wt') as f:
                f.write(summary_text)
            
            logger.debug(f"Performance report saved: {report.report_id}")
            
        except Exception as e:
            logger.error(f"Error saving performance report: {e}")
    
    def _generate_report_summary(self, report: PerformanceReport) -> str:
        """Generate human-readable report summary"""
        summary = f"""
AUTOMATED TRADING PERFORMANCE REPORT
====================================
Report ID: {report.report_id}
Generated: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
Period: {report.period_start.strftime('%Y-%m-%d %H:%M')} to {report.period_end.strftime('%Y-%m-%d %H:%M')}

TRADING PERFORMANCE
------------------
Total Trades: {report.total_trades}
Winning Trades: {report.winning_trades}
Losing Trades: {report.losing_trades}
Win Rate: {report.win_rate:.1%}
Total P&L: ${report.total_pnl:,.2f}
Realized P&L: ${report.realized_pnl:,.2f}
Unrealized P&L: ${report.unrealized_pnl:,.2f}

RISK METRICS
-----------
Max Drawdown: {report.max_drawdown:.1%}
Current Drawdown: {report.current_drawdown:.1%}
Sharpe Ratio: {report.sharpe_ratio:.2f}
Total VaR: ${report.total_var:,.2f}

SYSTEM METRICS
-------------
Uptime: {report.uptime_seconds / 3600:.1f} hours
System Availability: {report.system_availability:.1%}
Total Errors: {report.error_count}
Avg Response Time: {report.avg_response_time * 1000:.1f}ms

STRATEGY BREAKDOWN
----------------
"""
        
        for strategy, metrics in report.strategy_performance.items():
            summary += f"{strategy.title()}: {metrics['trades']} trades, ${metrics['pnl']:,.2f} P&L, {metrics['win_rate']:.1%} win rate\n"
        
        summary += f"""
MARKET CONDITIONS
---------------
Market Regime: {report.market_regime.replace('_', ' ').title()}
Average Volatility: {report.average_volatility:.1%}
Market Hours: {report.total_market_hours:.1f} hours
"""
        
        return summary
    
    async def get_recent_logs(self, category: str = "all", limit: int = 100) -> List[TradingLogEntry]:
        """Get recent logs by category"""
        try:
            if category == "all":
                return self.recent_trading_logs[-limit:]
            elif category == "trading":
                return [log for log in self.recent_trading_logs if log.category == "trading"][-limit:]
            elif category == "risk":
                return [log for log in self.recent_trading_logs if log.category == "risk"][-limit:]
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error getting recent logs: {e}")
            return []
    
    async def get_recent_alerts(self, limit: int = 50) -> List[RiskAlert]:
        """Get recent risk alerts"""
        try:
            return self.recent_risk_alerts[-limit:]
        except Exception as e:
            logger.error(f"Error getting recent alerts: {e}")
            return []
    
    async def get_recent_decisions(self, limit: int = 100) -> List[DecisionAudit]:
        """Get recent decisions"""
        try:
            return self.recent_decisions[-limit:]
        except Exception as e:
            logger.error(f"Error getting recent decisions: {e}")
            return []
    
    async def search_logs(self, query: str, category: str = "all", 
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> List[TradingLogEntry]:
        """Search logs by query and filters"""
        try:
            # This would implement full-text search on logs
            # For now, return recent logs as placeholder
            return await self.get_recent_logs(category, 50)
            
        except Exception as e:
            logger.error(f"Error searching logs: {e}")
            return []
    
    async def export_logs(self, start_date: datetime, end_date: datetime, 
                         output_file: str, format: str = "json"):
        """Export logs for date range"""
        try:
            # This would export logs in the specified format
            # For now, create a placeholder export
            export_data = {
                "export_date": datetime.utcnow().isoformat(),
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "summary": {
                    "total_logs": len(self.recent_trading_logs),
                    "trades": len([log for log in self.recent_trading_logs if log.category == "trading"]),
                    "alerts": len(self.recent_risk_alerts),
                    "decisions": len(self.recent_decisions)
                }
            }
            
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Logs exported to {output_file}")
            
        except Exception as e:
            logger.error(f"Error exporting logs: {e}")
    
    async def log_session_start(self, session_info: Dict[str, Any]):
        """Log session start"""
        await self.log_system_event("INFO", "Trading session started", {
            **session_info,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def log_session_end(self, session_info: Dict[str, Any]):
        """Log session end"""
        await self.log_system_event("INFO", "Trading session ended", {
            **session_info,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def get_logging_statistics(self) -> Dict[str, Any]:
        """Get logging system statistics"""
        try:
            return {
                "log_files": {
                    "trading": str(self.trading_log_file),
                    "risk": str(self.risk_log_file), 
                    "system": str(self.system_log_file),
                    "decisions": str(self.decision_log_file),
                    "performance": str(self.performance_log_file)
                },
                "recent_activity": {
                    "trading_logs": len(self.recent_trading_logs),
                    "risk_alerts": len(self.recent_risk_alerts),
                    "decisions": len(self.recent_decisions)
                },
                "reports": {
                    "directory": str(self.reports_dir),
                    "total_reports": len(list(self.reports_dir.glob("*.json")))
                },
                "configuration": {
                    "retention_days": self.max_log_files,
                    "compression_enabled": self.compression_enabled,
                    "log_directory": str(self.log_dir)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting logging statistics: {e}")
            return {}