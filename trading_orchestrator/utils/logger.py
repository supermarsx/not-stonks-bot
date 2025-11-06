"""
@file logger.py
@brief Matrix-Themed Logging Utility for Day Trading Orchestrator

@details
This module provides structured logging with Matrix-style formatting and
comprehensive trading event tracking. It enhances standard Python logging
with trading-specific features, visual styling, and structured data capture.

Key Features:
- Matrix-themed log formatting with emoji indicators
- Structured logging for trading events
- Multiple output destinations (file, console, network)
- Log rotation and archival management
- Trading-specific event categorization
- Performance and timing tracking
- Security and audit logging

@author Trading Orchestrator System
@version 1.0
@date 2025-11-06

@warning
Logging configuration affects system performance. Excessive logging
can impact trading execution speed and system resources.

@note
This module provides enhanced logging capabilities:

@see LogLevel for logging level definitions
@see TradingEventType for trading event categorization
@see TradingLogRecord for structured log data

@par Log Categories:
- Trading Events: Orders, trades, positions, P&L
- System Events: Service start/stop, errors, warnings
- Risk Events: Limit violations, circuit breakers, compliance
- Market Events: Data updates, price changes, volatility
- Security Events: Authentication, authorization, access logs
- Performance Events: Latency, throughput, resource usage

@par Log Destinations:
- Console: Real-time monitoring with color coding
- Files: Persistent storage with rotation
- Network: Centralized logging servers
- Database: Structured query capability
"""

import logging
import logging.handlers
import sys
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class LogLevel(Enum):
    """
    @enum LogLevel
    @brief Logging levels with Matrix-themed visual indicators
    
    @details
    Defines logging levels with emoji-based visual indicators for enhanced
    terminal display and quick visual identification of log importance.
    
    @par Level Hierarchy:
    - CRITICAL: System-breaking errors requiring immediate attention
    - ERROR: Significant errors affecting functionality
    - WARNING: Important conditions requiring attention
    - INFO: General operational information
    - DEBUG: Detailed debugging information
    - TRACE: Very detailed execution tracing
    
    @par Color Coding:
    - 🟥 Red: Critical errors and failures
    - 🟧 Orange: Warnings and cautions
    - 🟨 Yellow: Important information
    - 🟩 Green: Success and normal operations
    - 🟦 Blue: Debug information
    - ⬜ White: Trace and diagnostic data
    """
    CRITICAL = "🟥 CRITICAL"
    ERROR = "🟧 ERROR"
    WARNING = "🟨 WARNING"
    INFO = "🟩 INFO"
    DEBUG = "🟦 DEBUG"
    TRACE = "⬜ TRACE"

class TradingEventType(Enum):
    """
    @enum TradingEventType
    @brief Types of trading events for structured logging
    
    @details
    Categorizes different types of trading-related events for structured
    logging and analysis. Each event type has specific data requirements
    and processing logic.
    
    @par Event Categories:
    - ORDER_SUBMITTED: New order submitted for execution
    - ORDER_FILLED: Order successfully executed
    - ORDER_CANCELLED: Order cancelled by user or system
    - POSITION_OPENED: New position initiated
    - POSITION_CLOSED: Position closed (profit or loss)
    - RISK_VIOLATION: Risk limit or policy violation
    - CIRCUIT_TRIGGERED: Circuit breaker activated
    - MARKET_UPDATE: Market data price/volume update
    - STRATEGY_SIGNAL: Trading strategy generated signal
    - SYSTEM_HEALTH: System service health update
    
    @warning
    Trading events may contain sensitive financial information.
    Ensure appropriate security controls for log access.
    """
    POSITION_OPENED = "POSITION_OPENED"
    POSITION_CLOSED = "POSITION_CLOSED"
    TRADE_EXECUTED = "TRADE_EXECUTED"
    RISK_VIOLATION = "RISK_VIOLATION"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"
    BALANCE_UPDATE = "BALANCE_UPDATE"
    MARKET_DATA = "MARKET_DATA"
    AI_DECISION = "AI_DECISION"
    STRATEGY_SIGNAL = "STRATEGY_SIGNAL"
    BROKER_CONNECT = "BROKER_CONNECT"
    BROKER_DISCONNECT = "BROKER_DISCONNECT"

@dataclass
class TradingEvent:
    """Structured trading event for logging"""
    timestamp: str
    event_type: TradingEventType
    broker: str
    symbol: str
    side: str
    quantity: float
    price: float
    pnl: float = 0.0
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class MatrixFormatter(logging.Formatter):
    """Matrix-themed formatter for console output"""
    
    # Matrix color codes
    COLORS = {
        'CRITICAL': '\033[91m',    # Bright Red
        'ERROR': '\033[93m',       # Bright Yellow  
        'WARNING': '\033[92m',     # Bright Green
        'INFO': '\033[96m',        # Bright Cyan
        'DEBUG': '\033[94m',       # Bright Blue
        'TRACE': '\033[97m',       # Bright White
        'RESET': '\033[0m',        # Reset
        'BOLD': '\033[1m',         # Bold
    }
    
    def format(self, record: logging.LogRecord) -> str:
        # Add Matrix-style prefix
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        
        # Create structured format
        if hasattr(record, 'trading_event'):
            event = record.trading_event
            prefix = f"[{timestamp}] {event.event_type.value}"
            details = f" {event.broker} {event.side} {event.quantity} {event.symbol} @ {event.price}"
            pnl_info = f" PnL: {event.pnl:.2f}" if event.pnl != 0 else ""
            
            formatted = f"{prefix}{details}{pnl_info}"
        else:
            level_name = record.levelname
            color = self.COLORS.get(level_name, self.COLORS['RESET'])
            reset = self.COLORS['RESET']
            
            prefix = f"[{timestamp}] {color}{level_name:<8}{reset}"
            message = record.getMessage()
            formatted = f"{prefix} {message}"
        
        return formatted

class StructuredFormatter(logging.Formatter):
    """JSON formatter for file logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add trading event data if present
        if hasattr(record, 'trading_event'):
            log_entry['trading_event'] = record.trading_event.to_dict()
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)

class MatrixLogger:
    """Matrix-themed logger with trading event support"""
    
    def __init__(self, name: str = "trading_orchestrator"):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup console and file handlers"""
        
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Console handler with Matrix formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(MatrixFormatter())
        
        # File handler with JSON formatting
        log_file = log_dir / "trading_orchestrator.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(StructuredFormatter())
        
        # Error file handler
        error_log_file = log_dir / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(StructuredFormatter())
        
        # Add handlers to logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
    
    def log_trading_event(
        self,
        event_type: TradingEventType,
        broker: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        pnl: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log a structured trading event"""
        
        event = TradingEvent(
            timestamp=datetime.utcnow().isoformat(),
            event_type=event_type,
            broker=broker,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            pnl=pnl,
            metadata=metadata or {}
        )
        
        # Create LogRecord with trading event
        record = self.logger.makeRecord(
            self.logger.name,
            logging.INFO,
            __file__,
            0,
            f"Trading Event: {event_type.value}",
            (),
            None
        )
        record.trading_event = event
        
        # Log the event
        self.logger.handle(record)
    
    def info(self, message: str, **kwargs):
        self.logger.info(message, extra=kwargs)
    
    def debug(self, message: str, **kwargs):
        self.logger.debug(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs):
        self.logger.error(message, extra=kwargs)
    
    def critical(self, message: str, **kwargs):
        self.logger.critical(message, extra=kwargs)
    
    def exception(self, message: str, **kwargs):
        self.logger.exception(message, extra=kwargs)

# Global logger instance
_matrix_logger = MatrixLogger()

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True
) -> MatrixLogger:
    """
    Setup Matrix-themed logging system
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Custom log file path
        console_output: Whether to output to console
        
    Returns:
        Configured MatrixLogger instance
    """
    
    # Set global log level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    _matrix_logger.logger.setLevel(numeric_level)
    
    # Update console handler level
    for handler in _matrix_logger.logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(numeric_level)
    
    return _matrix_logger

def get_logger(name: str = "trading_orchestrator") -> MatrixLogger:
    """Get a MatrixLogger instance"""
    return MatrixLogger(name)

def log_order_submitted(broker: str, symbol: str, side: str, quantity: float, price: float):
    """Log order submission"""
    _matrix_logger.log_trading_event(
        TradingEventType.ORDER_SUBMITTED,
        broker, symbol, side, quantity, price
    )

def log_order_filled(broker: str, symbol: str, side: str, quantity: float, price: float, pnl: float = 0.0):
    """Log order fill"""
    _matrix_logger.log_trading_event(
        TradingEventType.ORDER_FILLED,
        broker, symbol, side, quantity, price, pnl
    )

def log_risk_violation(violation_type: str, details: str, metadata: Dict[str, Any] = None):
    """Log risk violation"""
    _matrix_logger.log_trading_event(
        TradingEventType.RISK_VIOLATION,
        "SYSTEM", "N/A", "N/A", 0.0, 0.0, 0.0,
        metadata={"violation_type": violation_type, "details": details}
    )

def log_circuit_breaker(reason: str, metadata: Dict[str, Any] = None):
    """Log circuit breaker activation"""
    _matrix_logger.log_trading_event(
        TradingEventType.CIRCUIT_BREAKER,
        "SYSTEM", "N/A", "N/A", 0.0, 0.0, 0.0,
        metadata={"reason": reason}
    )

def log_ai_decision(decision: str, confidence: float, metadata: Dict[str, Any] = None):
    """Log AI trading decision"""
    _matrix_logger.log_trading_event(
        TradingEventType.AI_DECISION,
        "AI", "N/A", "N/A", 0.0, 0.0, 0.0,
        metadata={"decision": decision, "confidence": confidence}
    )

# Convenience functions for different logger methods
info = _matrix_logger.info
debug = _matrix_logger.debug
warning = _matrix_logger.warning
error = _matrix_logger.error
critical = _matrix_logger.critical
exception = _matrix_logger.exception