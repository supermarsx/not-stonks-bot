"""
Demo Logging System - Comprehensive logging for virtual trading activities
"""

import asyncio
import json
import csv
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import pandas as pd
from collections import defaultdict

from loguru import logger

from .demo_mode_manager import DemoModeManager


class LogLevel(Enum):
    """Log levels for demo mode"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class LogCategory(Enum):
    """Categories for demo logging"""
    TRADING = "trading"
    EXECUTION = "execution"
    RISK = "risk"
    PERFORMANCE = "performance"
    MARKET_DATA = "market_data"
    SYSTEM = "system"
    ALERT = "alert"
    BACKTEST = "backtest"


class LogFormat(Enum):
    """Log output formats"""
    JSON = "json"
    CSV = "csv"
    TEXT = "text"
    DASHBOARD = "dashboard"


@dataclass
class LogEntry:
    """Individual log entry"""
    id: str
    timestamp: datetime
    level: LogLevel
    category: LogCategory
    message: str
    data: Dict[str, Any]
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    source: str = "demo_system"


@dataclass
class TradeLogEntry:
    """Specific log entry for trading activities"""
    trade_id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: float
    commission: float
    slippage: float
    execution_time: float
    timestamp: datetime
    broker: str
    algorithm: str
    success: bool
    error_message: Optional[str] = None


@dataclass
class PerformanceLogEntry:
    """Performance tracking log entry"""
    timestamp: datetime
    portfolio_value: float
    cash_balance: float
    unrealized_pnl: float
    realized_pnl: float
    total_return_pct: float
    daily_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    positions_count: int
    trades_count: int


@dataclass
class RiskLogEntry:
    """Risk management log entry"""
    timestamp: datetime
    risk_metric: str
    current_value: float
    threshold_value: float
    status: str  # normal, warning, critical
    message: str
    action_taken: Optional[str] = None


@dataclass
class SystemLogEntry:
    """System operation log entry"""
    timestamp: datetime
    operation: str
    component: str
    status: str
    duration: float
    details: Dict[str, Any]


class DemoLogger:
    """
    Comprehensive demo mode logging system
    
    Handles logging for all demo mode activities including:
    - Trade execution logs
    - Performance tracking
    - Risk monitoring
    - System operations
    """
    
    def __init__(self, demo_manager: DemoModeManager):
        self.demo_manager = demo_manager
        self.log_entries: List[LogEntry] = []
        self.trade_logs: List[TradeLogEntry] = []
        self.performance_logs: List[PerformanceLogEntry] = []
        self.risk_logs: List[RiskLogEntry] = []
        self.system_logs: List[SystemLogEntry] = []
        
        # Logging configuration
        self.max_entries = 100000
        self.auto_flush = True
        self.flush_interval = 300  # 5 minutes
        self.log_to_file = True
        self.log_dir = Path("demo_logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Background tasks
        self._flush_task = None
        self._compression_task = None
        
        # Statistics
        self.stats = {
            "total_logs": 0,
            "trade_logs": 0,
            "performance_logs": 0,
            "risk_logs": 0,
            "system_logs": 0,
            "errors": 0,
            "warnings": 0
        }
    
    async def start_logging(self):
        """Start the logging system"""
        try:
            # Start background tasks
            self._flush_task = asyncio.create_task(self._auto_flush_loop())
            self._compression_task = asyncio.create_task(self._compression_loop())
            
            await self.log_system_event("demo_logger_started", "demo_logger", "info", 0.0, {})
            
            logger.info("Demo logging system started")
            
        except Exception as e:
            logger.error(f"Failed to start demo logging system: {e}")
    
    async def stop_logging(self):
        """Stop the logging system"""
        try:
            # Cancel background tasks
            if self._flush_task:
                self._flush_task.cancel()
            if self._compression_task:
                self._compression_task.cancel()
            
            # Flush all logs
            await self.flush_logs()
            
            await self.log_system_event("demo_logger_stopped", "demo_logger", "info", 0.0, {})
            
            logger.info("Demo logging system stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop demo logging system: {e}")
    
    async def log_trade(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float,
        commission: float,
        slippage: float,
        execution_time: float,
        broker: str,
        algorithm: str,
        success: bool,
        error_message: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ):
        """Log trade execution"""
        try:
            trade_entry = TradeLogEntry(
                trade_id=trade_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                commission=commission,
                slippage=slippage,
                execution_time=execution_time,
                timestamp=datetime.now(),
                broker=broker,
                algorithm=algorithm,
                success=success,
                error_message=error_message
            )
            
            self.trade_logs.append(trade_entry)
            self.stats["trade_logs"] += 1
            
            # Add to general log
            await self.log(
                LogLevel.INFO,
                LogCategory.TRADING,
                f"Trade executed: {symbol} {side} {quantity} @ {price}",
                {
                    "trade_id": trade_id,
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "price": price,
                    "commission": commission,
                    "slippage": slippage,
                    "execution_time": execution_time,
                    "success": success,
                    **(additional_data or {})
                }
            )
            
            if not success and error_message:
                self.stats["errors"] += 1
            
            # Auto flush if enabled
            if self.auto_flush and len(self.trade_logs) % 100 == 0:
                await self.flush_logs()
                
        except Exception as e:
            logger.error(f"Error logging trade: {e}")
    
    async def log_performance(
        self,
        portfolio_value: float,
        cash_balance: float,
        unrealized_pnl: float,
        realized_pnl: float,
        total_return_pct: float,
        daily_return_pct: float,
        max_drawdown_pct: float,
        sharpe_ratio: float,
        positions_count: int,
        trades_count: int
    ):
        """Log portfolio performance"""
        try:
            performance_entry = PerformanceLogEntry(
                timestamp=datetime.now(),
                portfolio_value=portfolio_value,
                cash_balance=cash_balance,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl,
                total_return_pct=total_return_pct,
                daily_return_pct=daily_return_pct,
                max_drawdown_pct=max_drawdown_pct,
                sharpe_ratio=sharpe_ratio,
                positions_count=positions_count,
                trades_count=trades_count
            )
            
            self.performance_logs.append(performance_entry)
            self.stats["performance_logs"] += 1
            
            # Auto flush if enabled
            if self.auto_flush and len(self.performance_logs) % 50 == 0:
                await self.flush_logs()
                
        except Exception as e:
            logger.error(f"Error logging performance: {e}")
    
    async def log_risk_event(
        self,
        risk_metric: str,
        current_value: float,
        threshold_value: float,
        status: str,
        message: str,
        action_taken: Optional[str] = None
    ):
        """Log risk management event"""
        try:
            risk_entry = RiskLogEntry(
                timestamp=datetime.now(),
                risk_metric=risk_metric,
                current_value=current_value,
                threshold_value=threshold_value,
                status=status,
                message=message,
                action_taken=action_taken
            )
            
            self.risk_logs.append(risk_entry)
            self.stats["risk_logs"] += 1
            
            # Log appropriate level based on status
            level = LogLevel.INFO
            if status == "warning":
                level = LogLevel.WARNING
                self.stats["warnings"] += 1
            elif status == "critical":
                level = LogLevel.ERROR
                self.stats["errors"] += 1
            
            await self.log(
                level,
                LogCategory.RISK,
                f"Risk event: {message}",
                {
                    "risk_metric": risk_metric,
                    "current_value": current_value,
                    "threshold_value": threshold_value,
                    "status": status,
                    "action_taken": action_taken
                }
            )
            
        except Exception as e:
            logger.error(f"Error logging risk event: {e}")
    
    async def log_system_event(
        self,
        operation: str,
        component: str,
        status: str,
        duration: float,
        details: Dict[str, Any]
    ):
        """Log system operation"""
        try:
            system_entry = SystemLogEntry(
                timestamp=datetime.now(),
                operation=operation,
                component=component,
                status=status,
                duration=duration,
                details=details
            )
            
            self.system_logs.append(system_entry)
            self.stats["system_logs"] += 1
            
            level = LogLevel.INFO
            if status == "error":
                level = LogLevel.ERROR
                self.stats["errors"] += 1
            elif status == "warning":
                level = LogLevel.WARNING
                self.stats["warnings"] += 1
            
            await self.log(
                level,
                LogCategory.SYSTEM,
                f"System event: {operation} - {status}",
                {
                    "operation": operation,
                    "component": component,
                    "status": status,
                    "duration": duration,
                    **details
                }
            )
            
        except Exception as e:
            logger.error(f"Error logging system event: {e}")
    
    async def log(
        self,
        level: LogLevel,
        category: LogCategory,
        message: str,
        data: Dict[str, Any] = None
    ):
        """General purpose logging"""
        try:
            log_entry = LogEntry(
                id=f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                timestamp=datetime.now(),
                level=level,
                category=category,
                message=message,
                data=data or {},
                session_id=self.demo_manager.get_session_id()
            )
            
            self.log_entries.append(log_entry)
            self.stats["total_logs"] += 1
            
            # Manage memory by limiting entries
            if len(self.log_entries) > self.max_entries:
                self.log_entries = self.log_entries[-self.max_entries//2:]
            
            # Log to console/file using loguru
            log_message = f"[{level.value.upper()}] {category.value}: {message}"
            if data:
                log_message += f" | Data: {json.dumps(data, default=str)}"
            
            if level == LogLevel.DEBUG:
                logger.debug(log_message)
            elif level == LogLevel.INFO:
                logger.info(log_message)
            elif level == LogLevel.WARNING:
                logger.warning(log_message)
            elif level == LogLevel.ERROR:
                logger.error(log_message)
            elif level == LogLevel.CRITICAL:
                logger.critical(log_message)
                
        except Exception as e:
            logger.error(f"Error creating log entry: {e}")
    
    async def flush_logs(self):
        """Flush all logs to files"""
        try:
            if not self.log_to_file:
                return
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Flush general logs
            if self.log_entries:
                await self._write_logs_to_file(
                    self.log_entries,
                    self.log_dir / f"demo_logs_{timestamp}.json"
                )
            
            # Flush trade logs
            if self.trade_logs:
                await self._write_trades_to_file(
                    self.trade_logs,
                    self.log_dir / f"trade_logs_{timestamp}.json"
                )
            
            # Flush performance logs
            if self.performance_logs:
                await self._write_performance_to_file(
                    self.performance_logs,
                    self.log_dir / f"performance_logs_{timestamp}.json"
                )
            
            # Flush risk logs
            if self.risk_logs:
                await self._write_risk_to_file(
                    self.risk_logs,
                    self.log_dir / f"risk_logs_{timestamp}.json"
                )
            
            # Flush system logs
            if self.system_logs:
                await self._write_system_to_file(
                    self.system_logs,
                    self.log_dir / f"system_logs_{timestamp}.json"
                )
            
            # Clear in-memory logs after successful write
            await self._clear_logs_after_flush()
            
        except Exception as e:
            logger.error(f"Error flushing logs: {e}")
    
    async def get_logs(
        self,
        category: Optional[LogCategory] = None,
        level: Optional[LogLevel] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[LogEntry]:
        """Retrieve logs with filtering"""
        try:
            filtered_logs = self.log_entries
            
            # Apply filters
            if category:
                filtered_logs = [log for log in filtered_logs if log.category == category]
            
            if level:
                filtered_logs = [log for log in filtered_logs if log.level == level]
            
            if start_time:
                filtered_logs = [log for log in filtered_logs if log.timestamp >= start_time]
            
            if end_time:
                filtered_logs = [log for log in filtered_logs if log.timestamp <= end_time]
            
            # Apply limit and sort by timestamp (newest first)
            filtered_logs = sorted(filtered_logs, key=lambda x: x.timestamp, reverse=True)
            return filtered_logs[:limit]
            
        except Exception as e:
            logger.error(f"Error retrieving logs: {e}")
            return []
    
    async def get_trade_logs(self, symbol: Optional[str] = None, limit: int = 100) -> List[TradeLogEntry]:
        """Retrieve trade logs with optional symbol filter"""
        try:
            filtered_trades = self.trade_logs
            
            if symbol:
                filtered_trades = [trade for trade in filtered_trades if trade.symbol == symbol]
            
            # Sort by timestamp (newest first)
            filtered_trades = sorted(filtered_trades, key=lambda x: x.timestamp, reverse=True)
            return filtered_trades[:limit]
            
        except Exception as e:
            logger.error(f"Error retrieving trade logs: {e}")
            return []
    
    async def get_performance_logs(self, days: int = 30) -> List[PerformanceLogEntry]:
        """Retrieve performance logs for specified period"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            filtered_logs = [
                log for log in self.performance_logs 
                if log.timestamp >= cutoff_date
            ]
            
            return sorted(filtered_logs, key=lambda x: x.timestamp)
            
        except Exception as e:
            logger.error(f"Error retrieving performance logs: {e}")
            return []
    
    async def get_risk_logs(self, status: Optional[str] = None, hours: int = 24) -> List[RiskLogEntry]:
        """Retrieve risk logs with optional status filter"""
        try:
            cutoff_date = datetime.now() - timedelta(hours=hours)
            filtered_logs = [
                log for log in self.risk_logs 
                if log.timestamp >= cutoff_date
            ]
            
            if status:
                filtered_logs = [log for log in filtered_logs if log.status == status]
            
            return sorted(filtered_logs, key=lambda x: x.timestamp, reverse=True)
            
        except Exception as e:
            logger.error(f"Error retrieving risk logs: {e}")
            return []
    
    async def export_logs(self, output_dir: str, format: LogFormat = LogFormat.JSON):
        """Export all logs to specified directory"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if format == LogFormat.JSON:
                # Export as JSON
                await self._export_as_json(output_path, timestamp)
            elif format == LogFormat.CSV:
                # Export as CSV
                await self._export_as_csv(output_path, timestamp)
            
            await self.log_system_event(
                "logs_exported", "demo_logger", "info", 0.0,
                {"output_dir": str(output_path), "format": format.value}
            )
            
        except Exception as e:
            logger.error(f"Error exporting logs: {e}")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get logging statistics"""
        try:
            stats = self.stats.copy()
            
            # Add additional statistics
            stats.update({
                "memory_usage": {
                    "log_entries": len(self.log_entries),
                    "trade_logs": len(self.trade_logs),
                    "performance_logs": len(self.performance_logs),
                    "risk_logs": len(self.risk_logs),
                    "system_logs": len(self.system_logs)
                },
                "time_span": {
                    "first_log": min([log.timestamp for log in self.log_entries], default=None),
                    "last_log": max([log.timestamp for log in self.log_entries], default=None)
                }
            })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    # Private methods
    
    async def _write_logs_to_file(self, logs: List[LogEntry], filepath: Path):
        """Write general logs to JSON file"""
        try:
            log_data = [asdict(log) for log in logs]
            with open(filepath, 'w') as f:
                json.dump(log_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error writing logs to file: {e}")
    
    async def _write_trades_to_file(self, trades: List[TradeLogEntry], filepath: Path):
        """Write trade logs to JSON file"""
        try:
            trade_data = [asdict(trade) for trade in trades]
            with open(filepath, 'w') as f:
                json.dump(trade_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error writing trades to file: {e}")
    
    async def _write_performance_to_file(self, performance: List[PerformanceLogEntry], filepath: Path):
        """Write performance logs to JSON file"""
        try:
            performance_data = [asdict(perf) for perf in performance]
            with open(filepath, 'w') as f:
                json.dump(performance_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error writing performance to file: {e}")
    
    async def _write_risk_to_file(self, risk: List[RiskLogEntry], filepath: Path):
        """Write risk logs to JSON file"""
        try:
            risk_data = [asdict(risk_log) for risk_log in risk]
            with open(filepath, 'w') as f:
                json.dump(risk_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error writing risk to file: {e}")
    
    async def _write_system_to_file(self, system: List[SystemLogEntry], filepath: Path):
        """Write system logs to JSON file"""
        try:
            system_data = [asdict(sys_log) for sys_log in system]
            with open(filepath, 'w') as f:
                json.dump(system_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error writing system to file: {e}")
    
    async def _export_as_json(self, output_path: Path, timestamp: str):
        """Export logs as JSON format"""
        try:
            # Export individual log types
            if self.log_entries:
                with open(output_path / f"logs_{timestamp}.json", 'w') as f:
                    json.dump([asdict(log) for log in self.log_entries], f, indent=2, default=str)
            
            if self.trade_logs:
                with open(output_path / f"trades_{timestamp}.json", 'w') as f:
                    json.dump([asdict(trade) for trade in self.trade_logs], f, indent=2, default=str)
            
            # Export other log types...
            
        except Exception as e:
            logger.error(f"Error exporting as JSON: {e}")
    
    async def _export_as_csv(self, output_path: Path, timestamp: str):
        """Export logs as CSV format"""
        try:
            # Export trade logs as CSV
            if self.trade_logs:
                with open(output_path / f"trades_{timestamp}.csv", 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=asdict(self.trade_logs[0]).keys())
                    writer.writeheader()
                    for trade in self.trade_logs:
                        writer.writerow(asdict(trade))
            
            # Export performance logs as CSV
            if self.performance_logs:
                with open(output_path / f"performance_{timestamp}.csv", 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=asdict(self.performance_logs[0]).keys())
                    writer.writeheader()
                    for perf in self.performance_logs:
                        writer.writerow(asdict(perf))
            
        except Exception as e:
            logger.error(f"Error exporting as CSV: {e}")
    
    async def _auto_flush_loop(self):
        """Background task for automatic log flushing"""
        while True:
            try:
                await asyncio.sleep(self.flush_interval)
                await self.flush_logs()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto flush error: {e}")
    
    async def _compression_loop(self):
        """Background task for log compression"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self._compress_old_logs()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Log compression error: {e}")
    
    async def _compress_old_logs(self):
        """Compress old log files"""
        try:
            # This would implement log compression/archiving
            # For now, just log the intent
            await self.log_system_event(
                "log_compression_cycle", "demo_logger", "info", 0.0,
                {"files_processed": 0}
            )
        except Exception as e:
            logger.error(f"Error compressing logs: {e}")
    
    async def _clear_logs_after_flush(self):
        """Clear logs after successful flush"""
        try:
            # Clear only old logs, keep recent ones in memory
            if len(self.log_entries) > self.max_entries // 2:
                self.log_entries = self.log_entries[-self.max_entries//2:]
            
            if len(self.trade_logs) > 10000:
                self.trade_logs = self.trade_logs[-5000:]
            
            if len(self.performance_logs) > 5000:
                self.performance_logs = self.performance_logs[-2500:]
            
            # Update stats
            self.stats["total_logs"] = len(self.log_entries)
            self.stats["trade_logs"] = len(self.trade_logs)
            self.stats["performance_logs"] = len(self.performance_logs)
            self.stats["risk_logs"] = len(self.risk_logs)
            self.stats["system_logs"] = len(self.system_logs)
            
        except Exception as e:
            logger.error(f"Error clearing logs after flush: {e}")


# Global demo logger instance
demo_logger = None


async def get_demo_logger() -> DemoLogger:
    """Get global demo logger instance"""
    global demo_logger
    if demo_logger is None:
        manager = await get_demo_manager()
        demo_logger = DemoLogger(manager)
    return demo_logger


if __name__ == "__main__":
    # Example usage
    async def main():
        # Get demo manager and enable demo mode
        manager = await get_demo_manager()
        await manager.initialize()
        await manager.enable_demo_mode()
        
        # Get demo logger
        logger_system = await get_demo_logger()
        await logger_system.start_logging()
        
        # Log some demo activities
        await logger_system.log_trade(
            trade_id="trade_001",
            symbol="AAPL",
            side="buy",
            order_type="market",
            quantity=10,
            price=150.0,
            commission=1.50,
            slippage=0.75,
            execution_time=0.05,
            broker="demo_alpaca",
            algorithm="aggressive",
            success=True
        )
        
        await logger_system.log_performance(
            portfolio_value=101500.0,
            cash_balance=98500.0,
            unrealized_pnl=1000.0,
            realized_pnl=2000.0,
            total_return_pct=1.5,
            daily_return_pct=0.1,
            max_drawdown_pct=2.0,
            sharpe_ratio=1.2,
            positions_count=3,
            trades_count=25
        )
        
        await logger_system.log_risk_event(
            risk_metric="portfolio_var",
            current_value=0.05,
            threshold_value=0.10,
            status="normal",
            message="Portfolio VaR within limits"
        )
        
        await logger_system.log_system_event(
            operation="demo_update",
            component="demo_system",
            status="info",
            duration=0.1,
            details={"update_type": "performance"}
        )
        
        # Get statistics
        stats = await logger_system.get_statistics()
        print(f"Demo logger statistics: {stats}")
        
        # Export logs
        await logger_system.export_logs("demo_export")
        
        await logger_system.stop_logging()
    
    asyncio.run(main())
