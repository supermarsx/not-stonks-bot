"""
Crawler Configuration and Error Handling System
Manages configurations, error handling, and retry logic
"""

import asyncio
import logging
import json
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict, field
from pathlib import Path
import aiofiles
import traceback
from enum import Enum
import hashlib

from ..base.base_crawler import CrawlerConfig, CrawlerStatus


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"         # Warning, can continue
    MEDIUM = "medium"   # Retry with delay
    HIGH = "high"       # Stop crawler temporarily
    CRITICAL = "critical"  # Stop crawler, require manual intervention


class RetryStrategy(Enum):
    """Retry strategies"""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"
    IMMEDIATE = "immediate"


@dataclass
class ErrorRecord:
    """Error record for tracking"""
    timestamp: datetime
    crawler_name: str
    error_type: str
    error_message: str
    severity: ErrorSeverity
    stack_trace: str
    retry_count: int = 0
    resolved: bool = False
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetryConfig:
    """Retry configuration"""
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 300.0  # 5 minutes
    backoff_multiplier: float = 2.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    retry_on_errors: List[str] = field(default_factory=lambda: ["TimeoutError", "ConnectionError", "HTTPError"])
    stop_on_errors: List[str] = field(default_factory=lambda: ["KeyboardInterrupt", "MemoryError"])


@dataclass
class CrawlerConfigExtended:
    """Extended crawler configuration"""
    base_config: CrawlerConfig
    retry_config: RetryConfig
    error_thresholds: Dict[str, int] = field(default_factory=lambda: {
        'max_consecutive_errors': 5,
        'max_error_rate': 0.2,  # 20%
        'max_execution_time': 300  # 5 minutes
    })
    monitoring_config: Dict[str, Any] = field(default_factory=lambda: {
        'enable_detailed_logging': True,
        'enable_performance_tracking': True,
        'enable_error_alerts': True,
        'log_level': 'INFO'
    })
    rate_limiting_config: Dict[str, Any] = field(default_factory=lambda: {
        'requests_per_minute': 100,
        'burst_limit': 10,
        'cooldown_period': 60
    })
    data_validation_config: Dict[str, Any] = field(default_factory=lambda: {
        'enable_validation': True,
        'required_fields': [],
        'data_quality_checks': True
    })


class ConfigManager:
    """Manages crawler configurations"""
    
    def __init__(self, config_dir: str = "./configs"):
        self.config_dir = Path(config_dir)
        self.logger = logging.getLogger(__name__)
        
        # Configuration cache
        self._configs: Dict[str, CrawlerConfigExtended] = {}
        
        # Configuration templates
        self._templates = {}
        
        # Initialize configuration directory
        self.config_dir.mkdir(exist_ok=True)
        
        # Load default configurations
        self._setup_default_configs()
        self._load_configurations()
    
    def _setup_default_configs(self):
        """Setup default configurations"""
        default_configs = {
            'market_data': {
                'name': 'market_data',
                'data_type': 'market_data',
                'interval': 30,
                'max_retries': 3,
                'timeout': 30,
                'rate_limit': 100,
                'enable_cache': True,
                'enable_storage': True,
                'monitoring_config': {
                    'enable_detailed_logging': True,
                    'log_level': 'INFO'
                },
                'retry_config': {
                    'max_retries': 3,
                    'initial_delay': 2.0,
                    'max_delay': 60.0,
                    'strategy': 'exponential'
                }
            },
            'news': {
                'name': 'news',
                'data_type': 'news',
                'interval': 3600,
                'max_retries': 3,
                'timeout': 45,
                'rate_limit': 50,
                'retry_config': {
                    'max_retries': 3,
                    'initial_delay': 5.0,
                    'max_delay': 120.0,
                    'strategy': 'exponential'
                }
            },
            'social_media': {
                'name': 'social_media',
                'data_type': 'social_media',
                'interval': 300,
                'max_retries': 3,
                'timeout': 30,
                'rate_limit': 30,
                'retry_config': {
                    'max_retries': 2,
                    'initial_delay': 3.0,
                    'max_delay': 90.0,
                    'strategy': 'exponential'
                }
            },
            'economic': {
                'name': 'economic',
                'data_type': 'economic',
                'interval': 86400,
                'max_retries': 2,
                'timeout': 60,
                'rate_limit': 20,
                'retry_config': {
                    'max_retries': 2,
                    'initial_delay': 10.0,
                    'max_delay': 300.0,
                    'strategy': 'exponential'
                }
            },
            'patterns': {
                'name': 'patterns',
                'data_type': 'technical',
                'interval': 3600,
                'max_retries': 2,
                'timeout': 120,
                'rate_limit': 10,
                'retry_config': {
                    'max_retries': 2,
                    'initial_delay': 5.0,
                    'max_delay': 180.0,
                    'strategy': 'exponential'
                }
            }
        }
        
        # Save default configs to files
        for name, config in default_configs.items():
            config_file = self.config_dir / f"{name}.json"
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
    
    def _load_configurations(self):
        """Load all configuration files"""
        try:
            for config_file in self.config_dir.glob("*.json"):
                try:
                    with open(config_file, 'r') as f:
                        config_data = json.load(f)
                    
                    crawler_name = config_file.stem
                    self._configs[crawler_name] = self._create_extended_config(config_data)
                    
                except Exception as e:
                    self.logger.error(f"Error loading config {config_file}: {e}")
            
            self.logger.info(f"Loaded {len(self._configs)} configurations")
        
        except Exception as e:
            self.logger.error(f"Error loading configurations: {e}")
    
    def _create_extended_config(self, config_data: Dict[str, Any]) -> CrawlerConfigExtended:
        """Create extended configuration from data"""
        # Create base config
        base_config = CrawlerConfig(
            name=config_data['name'],
            data_type=getattr(DataType, config_data['data_type'].upper()),
            interval=config_data.get('interval', 60),
            max_retries=config_data.get('max_retries', 3),
            timeout=config_data.get('timeout', 30),
            rate_limit=config_data.get('rate_limit', 100),
            enable_cache=config_data.get('enable_cache', True),
            cache_duration=config_data.get('cache_duration', 300),
            enable_storage=config_data.get('enable_storage', True),
            storage_path=config_data.get('storage_path', './data'),
            enable_monitoring=config_data.get('enable_monitoring', True),
            error_threshold=config_data.get('error_threshold', 5),
            recovery_timeout=config_data.get('recovery_timeout', 60)
        )
        
        # Create retry config
        retry_data = config_data.get('retry_config', {})
        retry_config = RetryConfig(
            max_retries=retry_data.get('max_retries', 3),
            initial_delay=retry_data.get('initial_delay', 1.0),
            max_delay=retry_data.get('max_delay', 300.0),
            backoff_multiplier=retry_data.get('backoff_multiplier', 2.0),
            strategy=RetryStrategy(retry_data.get('strategy', 'exponential')),
            retry_on_errors=retry_data.get('retry_on_errors', []),
            stop_on_errors=retry_data.get('stop_on_errors', [])
        )
        
        return CrawlerConfigExtended(
            base_config=base_config,
            retry_config=retry_config,
            error_thresholds=config_data.get('error_thresholds', {}),
            monitoring_config=config_data.get('monitoring_config', {}),
            rate_limiting_config=config_data.get('rate_limiting_config', {}),
            data_validation_config=config_data.get('data_validation_config', {})
        )
    
    def get_config(self, crawler_name: str) -> Optional[CrawlerConfigExtended]:
        """Get configuration for crawler"""
        return self._configs.get(crawler_name)
    
    def update_config(self, crawler_name: str, config: CrawlerConfigExtended):
        """Update crawler configuration"""
        self._configs[crawler_name] = config
        
        # Save to file
        config_file = self.config_dir / f"{crawler_name}.json"
        config_data = self._config_to_dict(config)
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        self.logger.info(f"Updated configuration for {crawler_name}")
    
    def _config_to_dict(self, config: CrawlerConfigExtended) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        data = asdict(config.base_config)
        data.update({
            'retry_config': asdict(config.retry_config),
            'error_thresholds': config.error_thresholds,
            'monitoring_config': config.monitoring_config,
            'rate_limiting_config': config.rate_limiting_config,
            'data_validation_config': config.data_validation_config
        })
        return data
    
    def create_template(self, name: str, template_data: Dict[str, Any]):
        """Create configuration template"""
        self._templates[name] = template_data
        self.logger.info(f"Created template: {name}")
    
    def apply_template(self, crawler_name: str, template_name: str):
        """Apply template to crawler configuration"""
        if template_name not in self._templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        if crawler_name not in self._configs:
            raise ValueError(f"Crawler '{crawler_name}' configuration not found")
        
        template = self._templates[template_name]
        current_config = self._configs[crawler_name]
        
        # Apply template values
        for key, value in template.items():
            if hasattr(current_config, key):
                setattr(current_config, key, value)
        
        self.logger.info(f"Applied template '{template_name}' to {crawler_name}")
    
    def export_configs(self, export_path: str):
        """Export all configurations to file"""
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'configs': {name: self._config_to_dict(config) for name, config in self._configs.items()},
            'templates': self._templates
        }
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Exported configurations to {export_path}")
    
    def import_configs(self, import_path: str):
        """Import configurations from file"""
        try:
            with open(import_path, 'r') as f:
                import_data = json.load(f)
            
            # Import configs
            for name, config_data in import_data.get('configs', {}).items():
                self._configs[name] = self._create_extended_config(config_data)
            
            # Import templates
            self._templates.update(import_data.get('templates', {}))
            
            self.logger.info(f"Imported configurations from {import_path}")
        
        except Exception as e:
            self.logger.error(f"Error importing configurations: {e}")
            raise
    
    def validate_config(self, crawler_name: str) -> Dict[str, Any]:
        """Validate crawler configuration"""
        config = self._configs.get(crawler_name)
        if not config:
            return {'valid': False, 'errors': [f"Configuration for '{crawler_name}' not found"]}
        
        errors = []
        warnings = []
        
        # Validate base config
        base_config = config.base_config
        
        if base_config.interval <= 0:
            errors.append("Interval must be positive")
        
        if base_config.timeout <= 0:
            errors.append("Timeout must be positive")
        
        if base_config.rate_limit <= 0:
            errors.append("Rate limit must be positive")
        
        if base_config.max_retries < 0:
            errors.append("Max retries cannot be negative")
        
        # Validate retry config
        retry_config = config.retry_config
        
        if retry_config.max_retries < 0:
            errors.append("Max retries cannot be negative")
        
        if retry_config.initial_delay <= 0:
            errors.append("Initial delay must be positive")
        
        if retry_config.max_delay <= 0:
            errors.append("Max delay must be positive")
        
        if retry_config.initial_delay > retry_config.max_delay:
            errors.append("Initial delay cannot be greater than max delay")
        
        # Warnings
        if base_config.interval < 10:
            warnings.append("Very short interval may cause high resource usage")
        
        if base_config.rate_limit > 1000:
            warnings.append("Very high rate limit may trigger API restrictions")
        
        if retry_config.max_retries > 10:
            warnings.append("Very high retry count may cause long delays")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }


class ErrorHandler:
    """Comprehensive error handling and retry logic"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Error tracking
        self.error_records: List[ErrorRecord] = []
        self.crawler_error_counts: Dict[str, int] = {}
        self.last_error_times: Dict[str, datetime] = {}
        
        # Circuit breaker state
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        # Error statistics
        self.error_stats: Dict[str, Dict[str, Any]] = {}
    
    async def handle_error(self, crawler_name: str, error: Exception, 
                          context: Dict[str, Any] = None) -> bool:
        """Handle error with retry logic"""
        try:
            error_type = type(error).__name__
            error_message = str(error)
            
            # Record error
            await self._record_error(crawler_name, error, context)
            
            # Get configuration
            config = self.config_manager.get_config(crawler_name)
            if not config:
                self.logger.error(f"No configuration found for {crawler_name}")
                return False
            
            # Check if error should stop execution
            if error_type in config.retry_config.stop_on_errors:
                self.logger.critical(f"Stopping execution due to critical error: {error_message}")
                await self._trip_circuit_breaker(crawler_name, error)
                return False
            
            # Check if error should be retried
            if error_type not in config.retry_config.retry_on_errors:
                self.logger.warning(f"Error {error_type} not in retry list, not retrying")
                return False
            
            # Check circuit breaker
            if await self._is_circuit_open(crawler_name):
                self.logger.warning(f"Circuit breaker open for {crawler_name}, not retrying")
                return False
            
            # Get retry count
            retry_count = self.crawler_error_counts.get(crawler_name, 0)
            
            if retry_count >= config.retry_config.max_retries:
                self.logger.error(f"Max retries reached for {crawler_name}: {retry_count}")
                await self._trip_circuit_breaker(crawler_name, error)
                return False
            
            # Calculate retry delay
            delay = self._calculate_retry_delay(retry_count, config.retry_config)
            
            self.logger.info(f"Retrying {crawler_name} in {delay:.2f}s (attempt {retry_count + 1}/{config.retry_config.max_retries})")
            
            # Wait before retry
            await asyncio.sleep(delay)
            
            # Update retry count
            self.crawler_error_counts[crawler_name] = retry_count + 1
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error in error handler: {e}")
            return False
    
    async def _record_error(self, crawler_name: str, error: Exception, context: Dict[str, Any]):
        """Record error details"""
        error_record = ErrorRecord(
            timestamp=datetime.now(),
            crawler_name=crawler_name,
            error_type=type(error).__name__,
            error_message=str(error),
            severity=self._classify_error_severity(error),
            stack_trace=traceback.format_exc(),
            context=context or {}
        )
        
        self.error_records.append(error_record)
        
        # Keep only recent errors
        cutoff_time = datetime.now() - timedelta(days=30)
        self.error_records = [r for r in self.error_records if r.timestamp >= cutoff_time]
        
        # Update last error time
        self.last_error_times[crawler_name] = datetime.now()
        
        # Update error statistics
        if crawler_name not in self.error_stats:
            self.error_stats[crawler_name] = {
                'total_errors': 0,
                'error_types': {},
                'last_error': None,
                'consecutive_errors': 0
            }
        
        stats = self.error_stats[crawler_name]
        stats['total_errors'] += 1
        stats['last_error'] = error_record
        stats['consecutive_errors'] += 1
        
        # Track error types
        error_type = error_record.error_type
        if error_type not in stats['error_types']:
            stats['error_types'][error_type] = 0
        stats['error_types'][error_type] += 1
    
    def _classify_error_severity(self, error: Exception) -> ErrorSeverity:
        """Classify error severity"""
        error_type = type(error).__name__
        
        # Critical errors
        if error_type in ['MemoryError', 'KeyboardInterrupt', 'SystemExit', 'GeneratorExit']:
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if error_type in ['ConnectionError', 'TimeoutError', 'HTTPError', 'FileNotFoundError']:
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if error_type in ['ValueError', 'TypeError', 'KeyError', 'IndexError']:
            return ErrorSeverity.MEDIUM
        
        # Default to medium
        return ErrorSeverity.MEDIUM
    
    def _calculate_retry_delay(self, retry_count: int, retry_config: RetryConfig) -> float:
        """Calculate retry delay based on strategy"""
        if retry_config.strategy == RetryStrategy.IMMEDIATE:
            return 0
        
        elif retry_config.strategy == RetryStrategy.LINEAR:
            delay = retry_config.initial_delay + (retry_count * retry_config.initial_delay)
        
        elif retry_config.strategy == RetryStrategy.EXPONENTIAL:
            delay = retry_config.initial_delay * (retry_config.backoff_multiplier ** retry_count)
        
        else:  # FIXED
            delay = retry_config.initial_delay
        
        # Cap at max delay
        return min(delay, retry_config.max_delay)
    
    async def _is_circuit_open(self, crawler_name: str) -> bool:
        """Check if circuit breaker is open"""
        if crawler_name not in self.circuit_breakers:
            return False
        
        circuit = self.circuit_breakers[crawler_name]
        
        # Check if enough time has passed to attempt reset
        if datetime.now() >= circuit['next_attempt_time']:
            # Attempt to reset circuit breaker
            circuit['failure_count'] = 0
            circuit['state'] = 'closed'
            return False
        
        return circuit['state'] == 'open'
    
    async def _trip_circuit_breaker(self, crawler_name: str, error: Exception):
        """Trip circuit breaker due to repeated failures"""
        if crawler_name not in self.circuit_breakers:
            self.circuit_breakers[crawler_name] = {
                'state': 'closed',
                'failure_count': 0,
                'failure_threshold': 5,
                'recovery_timeout': 300,  # 5 minutes
                'next_attempt_time': None
            }
        
        circuit = self.circuit_breakers[crawler_name]
        circuit['failure_count'] += 1
        
        if circuit['failure_count'] >= circuit['failure_threshold']:
            circuit['state'] = 'open'
            circuit['next_attempt_time'] = datetime.now() + timedelta(seconds=circuit['recovery_timeout'])
            self.logger.warning(f"Circuit breaker opened for {crawler_name} after {circuit['failure_count']} failures")
    
    def reset_circuit_breaker(self, crawler_name: str):
        """Manually reset circuit breaker"""
        if crawler_name in self.circuit_breakers:
            self.circuit_breakers[crawler_name]['state'] = 'closed'
            self.circuit_breakers[crawler_name]['failure_count'] = 0
            self.circuit_breakers[crawler_name]['next_attempt_time'] = None
            self.logger.info(f"Circuit breaker reset for {crawler_name}")
    
    def mark_success(self, crawler_name: str):
        """Mark successful execution"""
        # Reset error count
        self.crawler_error_counts[crawler_name] = 0
        
        # Update error statistics
        if crawler_name in self.error_stats:
            self.error_stats[crawler_name]['consecutive_errors'] = 0
        
        # Reset circuit breaker if closed
        if crawler_name in self.circuit_breakers:
            self.circuit_breakers[crawler_name]['failure_count'] = 0
            self.circuit_breakers[crawler_name]['state'] = 'closed'
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary for specified time period"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_errors = [r for r in self.error_records if r.timestamp >= cutoff_time]
            
            # Group by crawler
            crawler_errors = {}
            for error in recent_errors:
                crawler_name = error.crawler_name
                if crawler_name not in crawler_errors:
                    crawler_errors[crawler_name] = []
                crawler_errors[crawler_name].append(error)
            
            summary = {
                'period_hours': hours,
                'timestamp': datetime.now().isoformat(),
                'total_errors': len(recent_errors),
                'crawlers_affected': len(crawler_errors),
                'error_breakdown': {},
                'circuit_breakers': {}
            }
            
            # Error breakdown by crawler
            for crawler_name, errors in crawler_errors.items():
                error_types = {}
                severity_count = {level.value: 0 for level in ErrorSeverity}
                
                for error in errors:
                    # Count error types
                    if error.error_type not in error_types:
                        error_types[error.error_type] = 0
                    error_types[error.error_type] += 1
                    
                    # Count severities
                    severity_count[error.severity.value] += 1
                
                summary['error_breakdown'][crawler_name] = {
                    'total_errors': len(errors),
                    'error_types': error_types,
                    'severity_breakdown': severity_count,
                    'last_error': errors[-1].timestamp.isoformat() if errors else None
                }
            
            # Circuit breaker status
            for crawler_name, circuit in self.circuit_breakers.items():
                summary['circuit_breakers'][crawler_name] = {
                    'state': circuit['state'],
                    'failure_count': circuit['failure_count'],
                    'next_attempt_time': circuit['next_attempt_time'].isoformat() if circuit['next_attempt_time'] else None
                }
            
            return summary
        
        except Exception as e:
            self.logger.error(f"Error generating error summary: {e}")
            return {'error': str(e)}
    
    def get_crawler_health(self, crawler_name: str) -> Dict[str, Any]:
        """Get health status for specific crawler"""
        try:
            # Get recent errors
            recent_errors = [r for r in self.error_records if r.crawler_name == crawler_name]
            
            # Get error statistics
            stats = self.error_stats.get(crawler_name, {
                'total_errors': 0,
                'error_types': {},
                'last_error': None,
                'consecutive_errors': 0
            })
            
            # Get circuit breaker status
            circuit_breaker = self.circuit_breakers.get(crawler_name)
            
            # Calculate health metrics
            total_errors_24h = len([e for e in recent_errors if (datetime.now() - e.timestamp).total_seconds() < 86400])
            total_errors_1h = len([e for e in recent_errors if (datetime.now() - e.timestamp).total_seconds() < 3600])
            
            # Determine health status
            if circuit_breaker and circuit_breaker['state'] == 'open':
                health_status = 'critical'
            elif stats['consecutive_errors'] > 5:
                health_status = 'poor'
            elif total_errors_1h > 10:
                health_status = 'degraded'
            else:
                health_status = 'healthy'
            
            return {
                'crawler_name': crawler_name,
                'health_status': health_status,
                'total_errors': stats['total_errors'],
                'errors_24h': total_errors_24h,
                'errors_1h': total_errors_1h,
                'consecutive_errors': stats['consecutive_errors'],
                'error_types': stats['error_types'],
                'circuit_breaker': {
                    'active': circuit_breaker is not None,
                    'state': circuit_breaker['state'] if circuit_breaker else None,
                    'failure_count': circuit_breaker['failure_count'] if circuit_breaker else 0
                } if circuit_breaker else None,
                'last_error': stats['last_error'].timestamp.isoformat() if stats['last_error'] else None
            }
        
        except Exception as e:
            self.logger.error(f"Error getting crawler health for {crawler_name}: {e}")
            return {'error': str(e)}


class DataValidator:
    """Validates data quality and integrity"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Validation rules
        self.validation_rules = {
            'market_data': self._validate_market_data,
            'news': self._validate_news_data,
            'social_media': self._validate_social_media_data,
            'economic': self._validate_economic_data,
            'patterns': self._validate_pattern_data
        }
        
        # Data quality metrics
        self.quality_metrics: Dict[str, Dict[str, Any]] = {}
    
    def validate_data(self, crawler_name: str, data: Any) -> Dict[str, Any]:
        """Validate data according to crawler type"""
        try:
            if crawler_name in self.validation_rules:
                validation_func = self.validation_rules[crawler_name]
                return validation_func(data)
            else:
                return self._generic_validation(data)
        
        except Exception as e:
            self.logger.error(f"Error validating data for {crawler_name}: {e}")
            return {
                'valid': False,
                'errors': [f"Validation error: {str(e)}"],
                'quality_score': 0.0
            }
    
    def _validate_market_data(self, data: Any) -> Dict[str, Any]:
        """Validate market data"""
        errors = []
        warnings = []
        
        if not isinstance(data, dict):
            errors.append("Data must be a dictionary")
            return {'valid': False, 'errors': errors, 'quality_score': 0.0}
        
        # Check required sections
        required_sections = ['real_time']
        for section in required_sections:
            if section not in data:
                errors.append(f"Missing required section: {section}")
        
        # Validate real-time data
        if 'real_time' in data:
            real_time_data = data['real_time']
            if isinstance(real_time_data, dict):
                for symbol, quote in real_time_data.items():
                    if not isinstance(quote, dict):
                        errors.append(f"Quote data for {symbol} is not a dictionary")
                        continue
                    
                    required_fields = ['price', 'volume', 'timestamp']
                    for field in required_fields:
                        if field not in quote:
                            errors.append(f"Missing field '{field}' for symbol {symbol}")
                    
                    # Validate numeric fields
                    if 'price' in quote and quote['price'] is not None:
                        try:
                            price = float(quote['price'])
                            if price <= 0:
                                errors.append(f"Invalid price for {symbol}: {price}")
                        except (ValueError, TypeError):
                            errors.append(f"Invalid price format for {symbol}: {quote['price']}")
                    
                    if 'volume' in quote and quote['volume'] is not None:
                        try:
                            volume = float(quote['volume'])
                            if volume < 0:
                                errors.append(f"Invalid volume for {symbol}: {volume}")
                        except (ValueError, TypeError):
                            errors.append(f"Invalid volume format for {symbol}: {quote['volume']}")
        
        # Calculate quality score
        total_checks = 10  # Base number of checks
        passed_checks = total_checks - len(errors)
        quality_score = max(0.0, passed_checks / total_checks)
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'quality_score': quality_score,
            'data_points_validated': len(data.get('real_time', {}))
        }
    
    def _validate_news_data(self, data: Any) -> Dict[str, Any]:
        """Validate news data"""
        errors = []
        warnings = []
        
        if not isinstance(data, dict):
            errors.append("Data must be a dictionary")
            return {'valid': False, 'errors': errors, 'quality_score': 0.0}
        
        # Check required sections
        if 'news' not in data:
            errors.append("Missing 'news' section")
        
        if 'news' in data:
            news_data = data['news']
            if isinstance(news_data, list):
                for i, article in enumerate(news_data):
                    if not isinstance(article, dict):
                        errors.append(f"Article {i} is not a dictionary")
                        continue
                    
                    required_fields = ['title', 'url', 'source', 'published_date']
                    for field in required_fields:
                        if field not in article or not article[field]:
                            errors.append(f"Missing or empty field '{field}' in article {i}")
        
        # Calculate quality score
        total_checks = 5
        passed_checks = total_checks - len(errors)
        quality_score = max(0.0, passed_checks / total_checks)
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'quality_score': quality_score,
            'articles_validated': len(data.get('news', []))
        }
    
    def _validate_social_media_data(self, data: Any) -> Dict[str, Any]:
        """Validate social media data"""
        errors = []
        warnings = []
        
        if not isinstance(data, dict):
            errors.append("Data must be a dictionary")
            return {'valid': False, 'errors': errors, 'quality_score': 0.0}
        
        # Check required sections
        if 'platform_data' not in data:
            errors.append("Missing 'platform_data' section")
        
        if 'platform_data' in data:
            platform_data = data['platform_data']
            if isinstance(platform_data, dict):
                for platform, posts in platform_data.items():
                    if isinstance(posts, list):
                        for i, post in enumerate(posts):
                            if not isinstance(post, dict):
                                errors.append(f"Post {i} in {platform} is not a dictionary")
                                continue
                            
                            required_fields = ['content', 'timestamp']
                            for field in required_fields:
                                if field not in post or not post[field]:
                                    errors.append(f"Missing field '{field}' in {platform} post {i}")
        
        # Calculate quality score
        total_checks = 3
        passed_checks = total_checks - len(errors)
        quality_score = max(0.0, passed_checks / total_checks)
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'quality_score': quality_score,
            'posts_validated': sum(len(posts) for posts in data.get('platform_data', {}).values() if isinstance(posts, list))
        }
    
    def _validate_economic_data(self, data: Any) -> Dict[str, Any]:
        """Validate economic data"""
        errors = []
        warnings = []
        
        if not isinstance(data, dict):
            errors.append("Data must be a dictionary")
            return {'valid': False, 'errors': errors, 'quality_score': 0.0}
        
        # Check for any data sections
        data_sections = ['indicators', 'announcements', 'calendar_events', 'sentiment']
        found_sections = [section for section in data_sections if section in data]
        
        if not found_sections:
            errors.append("No valid data sections found")
        
        # Validate indicators if present
        if 'indicators' in data:
            indicators = data['indicators']
            if isinstance(indicators, list):
                for i, indicator in enumerate(indicators):
                    if not isinstance(indicator, dict):
                        errors.append(f"Indicator {i} is not a dictionary")
                        continue
                    
                    required_fields = ['indicator_name', 'value', 'timestamp']
                    for field in required_fields:
                        if field not in indicator:
                            errors.append(f"Missing field '{field}' in indicator {i}")
        
        # Calculate quality score
        total_checks = 4
        passed_checks = total_checks - len(errors)
        quality_score = max(0.0, passed_checks / total_checks)
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'quality_score': quality_score,
            'data_sections_found': found_sections
        }
    
    def _validate_pattern_data(self, data: Any) -> Dict[str, Any]:
        """Validate pattern data"""
        errors = []
        warnings = []
        
        if not isinstance(data, dict):
            errors.append("Data must be a dictionary")
            return {'valid': False, 'errors': errors, 'quality_score': 0.0}
        
        # Check for any data sections
        data_sections = ['patterns', 'indicators', 'microstructure']
        found_sections = [section for section in data_sections if section in data]
        
        if not found_sections:
            errors.append("No valid data sections found")
        
        # Calculate quality score
        total_checks = 3
        passed_checks = total_checks - len(errors)
        quality_score = max(0.0, passed_checks / total_checks)
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'quality_score': quality_score,
            'data_sections_found': found_sections
        }
    
    def _generic_validation(self, data: Any) -> Dict[str, Any]:
        """Generic validation for unknown data types"""
        errors = []
        warnings = []
        
        if data is None:
            errors.append("Data is None")
        
        if isinstance(data, (list, dict)) and len(data) == 0:
            warnings.append("Data collection is empty")
        
        quality_score = 0.8 if not errors else 0.2
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'quality_score': quality_score,
            'data_type': type(data).__name__
        }