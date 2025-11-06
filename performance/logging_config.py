"""
Logging Configuration

Comprehensive logging system with structured JSON format, multiple levels,
and async logging implementation for the trading system.
"""

import json
import logging
import logging.handlers
import sys
import os
import asyncio
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import queue
import threading
from contextlib import contextmanager


class LogLevel(Enum):
    """Log level enumeration"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    WARN = "WARN"  # Alias for WARNING
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    FATAL = "FATAL"  # Alias for CRITICAL


class LogFormat(Enum):
    """Log format enumeration"""
    JSON = "json"
    CONSOLE = "console"
    TEXT = "text"


class LogCategory(Enum):
    """Log category enumeration for structured logging"""
    TRADING = "trading"
    MARKET_DATA = "market_data"
    RISK_MANAGEMENT = "risk_management"
    API = "api"
    DATABASE = "database"
    PERFORMANCE = "performance"
    SECURITY = "security"
    ERROR = "error"
    SYSTEM = "system"
    AUDIT = "audit"


@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: str
    level: str
    category: str
    message: str
    module: str = ""
    function: str = ""
    line_number: int = 0
    thread_id: Optional[int] = None
    process_id: Optional[int] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    trade_id: Optional[str] = None
    symbol: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.additional_data is None:
            self.additional_data = {}


@dataclass
class LoggingConfig:
    """Logging configuration"""
    log_level: str = "INFO"
    log_format: str = "json"
    log_directory: str = "logs"
    log_file_prefix: str = "trading_system"
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    backup_count: int = 10
    async_logging: bool = True
    enable_console_logging: bool = True
    enable_file_logging: bool = True
    enable_structured_logging: bool = True
    enable_performance_logging: bool = True
    enable_security_logging: bool = True
    enable_audit_logging: bool = True
    timezone: str = "UTC"
    json_indent: Optional[int] = None
    include_traceback: bool = True
    include_system_info: bool = True
    custom_formatters: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.custom_formatters is None:
            self.custom_formatters = {}


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def __init__(self, config: LoggingConfig):
        self.config = config
        self.json_indent = config.json_indent
        self.include_traceback = config.include_traceback
        self.include_system_info = config.include_system_info
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        # Create base log entry
        log_entry = LogEntry(
            timestamp=datetime.utcnow().isoformat() + "Z",
            level=record.levelname,
            category=getattr(record, 'category', LogCategory.SYSTEM.value),
            message=record.getMessage(),
            module=record.module,
            function=record.funcName,
            line_number=record.lineno,
            thread_id=record.thread,
            process_id=record.process,
        )
        
        # Add custom fields if present
        for attr_name in ['user_id', 'session_id', 'trade_id', 'symbol']:
            if hasattr(record, attr_name):
                setattr(log_entry, attr_name, getattr(record, attr_name))
        
        # Add additional data
        additional_data = {}
        
        # Exception information
        if record.exc_info and self.include_traceback:
            additional_data['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Custom log record attributes
        if hasattr(record, 'extra_data') and record.extra_data:
            additional_data.update(record.extra_data)
        
        # System information if enabled
        if self.include_system_info and not hasattr(record, 'system_info_included'):
            try:
                import psutil
                process = psutil.Process()
                additional_data['system_info'] = {
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_usage': process.memory_info()._asdict(),
                    'disk_usage': psutil.disk_usage('/')._asdict()
                }
                record.system_info_included = True
            except ImportError:
                pass  # psutil not available
        
        log_entry.additional_data = additional_data
        
        # Convert to JSON
        try:
            return json.dumps(asdict(log_entry), indent=self.json_indent)
        except Exception as e:
            # Fallback to simple format if JSON serialization fails
            fallback_format = f"{log_entry.timestamp} {log_entry.level} {log_entry.module}:{log_entry.line_number} - {log_entry.message}"
            if record.exc_info:
                fallback_format += f"\nException: {record.exc_info[1]}"
            return fallback_format


class ColoredConsoleFormatter(logging.Formatter):
    """Console formatter with colors"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    
    def __init__(self):
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        # Add color to log level
        if hasattr(record, 'category'):
            category = record.category.upper()
        else:
            category = record.name
        
        # Format timestamp
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        
        # Get color for level
        color = self.COLORS.get(record.levelname, '')
        reset = self.RESET if color else ''
        
        # Format the message
        formatted = f"{color}[{timestamp}] {record.levelname:<8} {category:<15} {record.getMessage()}{reset}"
        
        # Add exception info if present
        if record.exc_info:
            formatted += f"\n{traceback.format_exception(*record.exc_info)}"
        
        return formatted


class AsyncLogHandler(logging.Handler):
    """Async log handler for non-blocking logging"""
    
    def __init__(self, target_handler: logging.Handler, max_queue_size: int = 10000):
        super().__init__()
        self.target_handler = target_handler
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.thread = threading.Thread(target=self._process_log_queue, daemon=True)
        self.thread.start()
        self.is_running = True
    
    def emit(self, record: logging.LogRecord):
        """Emit log record asynchronously"""
        try:
            # Serialize the record for queue
            serialized = self._serialize_record(record)
            self.queue.put_nowait(serialized)
        except queue.Full:
            # Queue is full, drop the record to prevent blocking
            pass
        except Exception:
            self.handleError(record)
    
    def _serialize_record(self, record: logging.LogRecord) -> Dict[str, Any]:
        """Serialize log record for queue"""
        return {
            'created': record.created,
            'msecs': record.msecs,
            'relativeCreated': record.relativeCreated,
            'levelno': record.levelno,
            'levelname': record.levelname,
            'getMessage': record.getMessage(),
            'module': record.module,
            'funcName': record.funcName,
            'lineno': record.lineno,
            'thread': record.thread,
            'process': record.process,
            'exc_info': record.exc_info,
            'exc_text': record.exc_text,
            'pathname': record.pathname,
            'name': record.name,
        }
    
    def _process_log_queue(self):
        """Process log queue in background thread"""
        while self.is_running:
            try:
                record_data = self.queue.get(timeout=1)
                record = self._deserialize_record(record_data)
                self.target_handler.emit(record)
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                pass
    
    def _deserialize_record(self, data: Dict[str, Any]) -> logging.LogRecord:
        """Deserialize log record from queue"""
        record = logging.LogRecord(
            name=data['name'],
            level=data['levelno'],
            pathname=data['pathname'],
            lineno=data['lineno'],
            msg=data['getMessage'],
            args=(),
            exc_info=data['exc_info']
        )
        
        # Set additional attributes
        for attr in ['created', 'msecs', 'relativeCreated', 'module', 'funcName', 
                     'thread', 'process', 'exc_text']:
            if attr in data:
                setattr(record, attr, data[attr])
        
        return record
    
    def close(self):
        """Close async handler"""
        self.is_running = False
        if self.thread.is_alive():
            self.thread.join(timeout=5)
        self.target_handler.close()
        super().close()


class LogRotationHandler(logging.handlers.RotatingFileHandler):
    """Enhanced log rotation handler with compression and cleanup"""
    
    def __init__(self, filename: str, mode: str = 'a', maxBytes: int = 0, 
                 backupCount: int = 0, encoding: Optional[str] = None, 
                 delay: bool = False, errors: Optional[str] = None,
                 compress_on_rotation: bool = True, 
                 cleanup_old_files: bool = True,
                 max_age_days: int = 30):
        super().__init__(filename, mode, maxBytes, backupCount, 
                        encoding, delay, errors)
        self.compress_on_rotation = compress_on_rotation
        self.cleanup_old_files = cleanup_old_files
        self.max_age_days = max_age_days
    
    def doRollover(self):
        """Rollover with compression and cleanup"""
        # Get current file path
        dfn = self.baseFilename + "." + time.strftime("%Y%m%d-%H%M%S")
        
        # Close current file
        self.stream.close()
        
        # Rename current file
        if os.path.exists(self.baseFilename):
            os.rename(self.baseFilename, dfn)
        
        # Compress if enabled
        if self.compress_on_rotation:
            self._compress_log_file(dfn)
        
        # Reopen log file
        self.stream = self._open()
        
        # Cleanup old files
        if self.cleanup_old_files:
            self._cleanup_old_files()
    
    def _compress_log_file(self, log_path: str):
        """Compress log file using gzip"""
        try:
            import gzip
            with open(log_path, 'rb') as f_in:
                with gzip.open(log_path + '.gz', 'wb') as f_out:
                    f_out.writelines(f_in)
            os.remove(log_path)
        except Exception:
            pass  # Compress failed, keep original file
    
    def _cleanup_old_files(self):
        """Cleanup old log files based on age"""
        try:
            import time
            cutoff_time = time.time() - (self.max_age_days * 24 * 60 * 60)
            
            # Pattern to match rotated files
            pattern = f"{os.path.dirname(self.baseFilename)}/{os.path.basename(self.baseFilename)}.*"
            
            for log_file in glob.glob(pattern):
                if os.path.getmtime(log_file) < cutoff_time:
                    try:
                        os.remove(log_file)
                    except Exception:
                        pass  # File might be in use or deleted
        except Exception:
            pass


class LoggingManager:
    """Comprehensive logging manager"""
    
    def __init__(self, config: Optional[LoggingConfig] = None):
        self.config = config or LoggingConfig()
        self._loggers = {}
        self._handlers = {}
        self._setup_complete = False
        
        # Ensure log directory exists
        if self.config.enable_file_logging:
            Path(self.config.log_directory).mkdir(parents=True, exist_ok=True)
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        # Clear existing handlers
        logging.root.handlers.clear()
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Remove all existing handlers from root
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Setup console logging
        if self.config.enable_console_logging:
            console_handler = self._create_console_handler()
            root_logger.addHandler(console_handler)
            self._handlers['console'] = console_handler
        
        # Setup file logging
        if self.config.enable_file_logging:
            file_handlers = self._create_file_handlers()
            for category, handler in file_handlers.items():
                root_logger.addHandler(handler)
                self._handlers[f'file_{category}'] = handler
        
        self._setup_complete = True
        logging.info("Logging system initialized", extra={'category': 'system'})
    
    def _create_console_handler(self) -> logging.Handler:
        """Create console handler"""
        console_handler = logging.StreamHandler(sys.stdout)
        
        if self.config.log_format == LogFormat.JSON.value:
            console_handler.setFormatter(JSONFormatter(self.config))
        else:
            console_handler.setFormatter(ColoredConsoleFormatter())
        
        return console_handler
    
    def _create_file_handlers(self) -> Dict[str, logging.Handler]:
        """Create file handlers for different log categories"""
        handlers = {}
        
        # Main application log
        main_log_path = os.path.join(self.config.log_directory, f"{self.config.log_file_prefix}.log")
        main_handler = LogRotationHandler(
            main_log_path,
            maxBytes=self.config.max_file_size,
            backupCount=self.config.backup_count,
            compress_on_rotation=True,
            cleanup_old_files=True,
            max_age_days=30
        )
        
        if self.config.log_format == LogFormat.JSON.value:
            main_handler.setFormatter(JSONFormatter(self.config))
        else:
            main_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)s %(name)s %(message)s'
            ))
        
        handlers['main'] = main_handler
        
        # Category-specific logs
        for category in LogCategory:
            if category == LogCategory.ERROR:
                continue  # Already included in main log
            
            category_log_path = os.path.join(
                self.config.log_directory,
                f"{self.config.log_file_prefix}_{category.value}.log"
            )
            
            category_handler = LogRotationHandler(
                category_log_path,
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count,
                compress_on_rotation=True,
                cleanup_old_files=True,
                max_age_days=7  # Keep category logs longer
            )
            
            if self.config.log_format == LogFormat.JSON.value:
                category_handler.setFormatter(JSONFormatter(self.config))
            else:
                category_handler.setFormatter(logging.Formatter(
                    '%(asctime)s %(levelname)s %(message)s'
                ))
            
            handlers[category.value] = category_handler
        
        return handlers
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get or create logger"""
        if name not in self._loggers:
            logger = logging.getLogger(name)
            
            # Add category to logger if it's a standard trading category
            for category in LogCategory:
                if category.value in name.lower():
                    logger.extra_data = {'category': category.value}
                    break
            
            self._loggers[name] = logger
        
        return self._loggers[name]
    
    @contextmanager
    def log_context(self, 
                   category: LogCategory = LogCategory.SYSTEM,
                   user_id: Optional[str] = None,
                   session_id: Optional[str] = None,
                   trade_id: Optional[str] = None,
                   symbol: Optional[str] = None,
                   **extra_data):
        """Context manager for structured logging"""
        # Set context data
        context_data = {
            'category': category.value,
            'user_id': user_id,
            'session_id': session_id,
            'trade_id': trade_id,
            'symbol': symbol,
            'extra_data': extra_data
        }
        
        # Add context to all loggers
        for logger in self._loggers.values():
            if hasattr(logger, 'extra_data'):
                logger.extra_data.update(context_data)
        
        try:
            yield context_data
        finally:
            # Clear context data
            for logger in self._loggers.values():
                if hasattr(logger, 'extra_data'):
                    logger.extra_data = {}


# Global logging manager instance
_logging_manager = None

def get_logging_manager(config: Optional[LoggingConfig] = None) -> LoggingManager:
    """Get global logging manager instance"""
    global _logging_manager
    if _logging_manager is None:
        _logging_manager = LoggingManager(config)
    return _logging_manager


def get_logger(name: str, 
              category: Optional[LogCategory] = None,
              **extra_context) -> logging.Logger:
    """Get logger with optional category and context"""
    manager = get_logging_manager()
    logger = manager.get_logger(name)
    
    # Set category if provided
    if category:
        logger.extra_data = {'category': category.value}
    
    return logger


# Context managers for specific log categories
def log_trading_context(**context):
    """Context manager for trading logs"""
    return get_logging_manager().log_context(LogCategory.TRADING, **context)


def log_market_data_context(**context):
    """Context manager for market data logs"""
    return get_logging_manager().log_context(LogCategory.MARKET_DATA, **context)


def log_risk_management_context(**context):
    """Context manager for risk management logs"""
    return get_logging_manager().log_context(LogCategory.RISK_MANAGEMENT, **context)


def log_api_context(**context):
    """Context manager for API logs"""
    return get_logging_manager().log_context(LogCategory.API, **context)


def log_database_context(**context):
    """Context manager for database logs"""
    return get_logging_manager().log_context(LogCategory.DATABASE, **context)


def log_performance_context(**context):
    """Context manager for performance logs"""
    return get_logging_manager().log_context(LogCategory.PERFORMANCE, **context)


def log_security_context(**context):
    """Context manager for security logs"""
    return get_logging_manager().log_context(LogCategory.SECURITY, **context)


def log_audit_context(**context):
    """Context manager for audit logs"""
    return get_logging_manager().log_context(LogCategory.AUDIT, **context)


# Async logging utilities
async def async_log_info(logger: logging.Logger, message: str, **kwargs):
    """Async logging helper for info level"""
    await asyncio.get_event_loop().run_in_executor(
        None, logger.info, message, **kwargs
    )


async def async_log_error(logger: logging.Logger, message: str, **kwargs):
    """Async logging helper for error level"""
    await asyncio.get_event_loop().run_in_executor(
        None, logger.error, message, **kwargs
    )


async def async_log_warning(logger: logging.Logger, message: str, **kwargs):
    """Async logging helper for warning level"""
    await asyncio.get_event_loop().run_in_executor(
        None, logger.warning, message, **kwargs
    )


async def async_log_debug(logger: logging.Logger, message: str, **kwargs):
    """Async logging helper for debug level"""
    await asyncio.get_event_loop().run_in_executor(
        None, logger.debug, message, **kwargs
    )


# Log search and filtering utilities
class LogSearch:
    """Log search and filtering utility"""
    
    def __init__(self, log_directory: str):
        self.log_directory = Path(log_directory)
    
    def search_logs(self, 
                   query: str,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   level: Optional[str] = None,
                   category: Optional[str] = None,
                   **filters) -> List[Dict[str, Any]]:
        """Search log files for matching entries"""
        results = []
        
        # Find all log files
        log_files = list(self.log_directory.glob("*.log*"))
        
        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, 1):
                        log_entry = self._parse_log_line(line)
                        
                        if log_entry and self._matches_criteria(
                            log_entry, query, start_time, end_time, 
                            level, category, **filters
                        ):
                            results.append({
                                'file': str(log_file),
                                'line_number': line_num,
                                'entry': log_entry
                            })
            except Exception:
                continue  # Skip corrupted log files
        
        return results
    
    def _parse_log_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse log line into structured data"""
        try:
            # Try JSON first
            if line.strip().startswith('{'):
                return json.loads(line.strip())
            
            # Fallback to text parsing
            # This is a simplified parser - in practice, you'd want more sophisticated parsing
            import re
            pattern = r'\[([^\]]+)\] (\w+) ([^\s]+) ([^\s]+) - (.+)'
            match = re.match(pattern, line.strip())
            
            if match:
                return {
                    'timestamp': match.group(1),
                    'level': match.group(2),
                    'category': match.group(3),
                    'module': match.group(4),
                    'message': match.group(5)
                }
        except Exception:
            pass
        
        return None
    
    def _matches_criteria(self, 
                         log_entry: Dict[str, Any],
                         query: str,
                         start_time: Optional[datetime],
                         end_time: Optional[datetime],
                         level: Optional[str],
                         category: Optional[str],
                         **filters) -> bool:
        """Check if log entry matches search criteria"""
        # Query search (case-insensitive)
        if query:
            message = log_entry.get('message', '').lower()
            if query.lower() not in message:
                return False
        
        # Time range
        if start_time or end_time:
            timestamp_str = log_entry.get('timestamp', '')
            try:
                # Parse timestamp - this would depend on your timestamp format
                entry_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                
                if start_time and entry_time < start_time:
                    return False
                if end_time and entry_time > end_time:
                    return False
            except Exception:
                return False  # Invalid timestamp format
        
        # Level filter
        if level and log_entry.get('level', '').upper() != level.upper():
            return False
        
        # Category filter
        if category and log_entry.get('category', '') != category:
            return False
        
        # Additional filters
        for key, value in filters.items():
            if log_entry.get(key) != value:
                return False
        
        return True