"""Configuration Validator

Comprehensive configuration validation and testing system for the trading orchestrator.
Validates all configuration files, detects common issues, and provides detailed
feedback for configuration problems.

Features:
- Multi-format configuration support (JSON, YAML, environment variables)
- Comprehensive validation rules for all components
- Connection testing for external services
- Performance impact assessment
- Security validation
- Best practice recommendations
- Detailed error reporting and suggestions

Author: Trading System Development Team
Version: 1.0.0
Date: 2024-12-19
"""

import asyncio
import json
import logging
import os
import sys
import yaml
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import re
import urllib.parse

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from trading_orchestrator.config import TradingConfig, BrokerConfig, StrategyConfig
except ImportError:
    print("Warning: Trading configuration modules not available. Running in standalone validation mode.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ValidationSeverity(Enum):
    """Validation issue severity levels"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    SUGGESTION = "suggestion"

@dataclass
class ValidationIssue:
    """Validation issue structure"""
    severity: ValidationSeverity
    category: str
    message: str
    field_path: str
    suggestion: Optional[str] = None
    code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

@dataclass
class ValidationResult:
    """Validation result structure"""
    config_file: str
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings: int = 0
    errors: int = 0
    suggestions: int = 0
    validated_at: datetime = field(default_factory=datetime.utcnow)
    
    def add_issue(self, issue: ValidationIssue):
        """Add validation issue"""
        self.issues.append(issue)
        
        if issue.severity == ValidationSeverity.ERROR:
            self.errors += 1
            self.is_valid = False
        elif issue.severity == ValidationSeverity.WARNING:
            self.warnings += 1
        elif issue.severity == ValidationSeverity.SUGGESTION:
            self.suggestions += 1
    
    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get issues by severity"""
        return [issue for issue in self.issues if issue.severity == severity]
    
    def get_issues_by_category(self, category: str) -> List[ValidationIssue]:
        """Get issues by category"""
        return [issue for issue in self.issues if issue.category == category]

class ConfigValidator:
    """Main configuration validator"""
    
    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self.validation_results: List[ValidationResult] = []
        self.connection_tests_enabled = True
        self.performance_tests_enabled = True
        
    async def validate_config_file(self, config_path: Union[str, Path]) -> ValidationResult:
        """Validate a configuration file"""
        config_path = Path(config_path)
        result = ValidationResult(config_file=str(config_path))
        
        try:
            if not config_path.exists():
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="file_access",
                    message=f"Configuration file not found: {config_path}",
                    field_path="file",
                    code="FILE_NOT_FOUND"
                ))
                return result
            
            # Load configuration
            config_data = await self._load_config_file(config_path)
            if not config_data:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="file_parsing",
                    message="Failed to parse configuration file",
                    field_path="parsing",
                    code="PARSING_FAILED"
                ))
                return result
            
            # Validate configuration structure
            await self._validate_config_structure(config_data, result)
            
            # Validate database configuration
            await self._validate_database_config(config_data.get('database', {}), result)
            
            # Validate broker configurations
            await self._validate_broker_config(config_data.get('brokers', {}), result)
            
            # Validate strategy configurations
            await self._validate_strategy_config(config_data.get('strategies', {}), result)
            
            # Validate risk management
            await self._validate_risk_management(config_data.get('risk_management', {}), result)
            
            # Validate system settings
            await self._validate_system_settings(config_data.get('system', {}), result)
            
            # Validate security settings
            await self._validate_security_settings(config_data, result)
            
            # Test connections if enabled
            if self.connection_tests_enabled:
                await self._test_connections(config_data, result)
            
            # Performance validation
            if self.performance_tests_enabled:
                await self._validate_performance_config(config_data, result)
            
        except Exception as e:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="validation",
                message=f"Validation error: {str(e)}",
                field_path="validation",
                code="VALIDATION_ERROR",
                details={"exception": str(e)}
            ))
        
        self.validation_results.append(result)
        return result
    
    async def _load_config_file(self, config_path: Path) -> Optional[Dict[str, Any]]:
        """Load configuration file based on extension"""
        try:
            with open(config_path, 'r') as f:
                content = f.read()
            
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(content)
            elif config_path.suffix.lower() == '.json':
                return json.loads(content)
            else:
                # Try to detect format
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    try:
                        return yaml.safe_load(content)
                    except yaml.YAMLError:
                        return None
        except Exception as e:
            logger.error(f"Error loading config file {config_path}: {e}")
            return None
    
    async def _validate_config_structure(self, config_data: Dict[str, Any], result: ValidationResult):
        """Validate basic configuration structure"""
        required_sections = ['database', 'brokers', 'strategies']
        
        for section in required_sections:
            if section not in config_data:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="structure",
                    message=f"Missing required configuration section: {section}",
                    field_path=section,
                    code="MISSING_SECTION"
                ))
        
        # Check for unknown sections
        known_sections = required_sections + ['system', 'risk_management', 'logging', 'performance']
        for section in config_data.keys():
            if section not in known_sections:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="structure",
                    message=f"Unknown configuration section: {section}",
                    field_path=section,
                    code="UNKNOWN_SECTION"
                ))
    
    async def _validate_database_config(self, db_config: Dict[str, Any], result: ValidationResult):
        """Validate database configuration"""
        # Required fields
        if 'path' not in db_config:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="database",
                message="Database path is required",
                field_path="database.path",
                code="MISSING_PATH"
            ))
        
        # Database path validation
        if 'path' in db_config:
            db_path = db_config['path']
            if isinstance(db_path, str):
                # Check if path is writable
                db_dir = Path(db_path).parent
                if not db_dir.exists():
                    result.add_issue(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="database",
                        message=f"Database directory does not exist: {db_dir}",
                        field_path="database.path",
                        code="DIR_NOT_EXISTS",
                        suggestion="Create the directory or use a different path"
                    ))
        
        # Connection pool settings
        if 'connection_pool_size' in db_config:
            pool_size = db_config['connection_pool_size']
            if not isinstance(pool_size, int) or pool_size < 1:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="database",
                    message="Connection pool size must be a positive integer",
                    field_path="database.connection_pool_size",
                    code="INVALID_POOL_SIZE"
                ))
            elif pool_size > 100:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="database",
                    message=f"High connection pool size ({pool_size}) may impact performance",
                    field_path="database.connection_pool_size",
                    code="HIGH_POOL_SIZE",
                    suggestion="Consider reducing pool size for better resource management"
                ))
        
        # Timeout settings
        if 'timeout' in db_config:
            timeout = db_config['timeout']
            if not isinstance(timeout, (int, float)) or timeout <= 0:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="database",
                    message="Database timeout must be a positive number",
                    field_path="database.timeout",
                    code="INVALID_TIMEOUT"
                ))
        
        # Backup settings
        if 'backup_enabled' in db_config and db_config['backup_enabled']:
            if 'backup_interval' not in db_config:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="database",
                    message="Backup enabled but no backup interval specified",
                    field_path="database.backup_interval",
                    code="MISSING_BACKUP_INTERVAL"
                ))
    
    async def _validate_broker_config(self, broker_config: Dict[str, Any], result: ValidationResult):
        """Validate broker configurations"""
        # Check if brokers are configured
        if not broker_config or 'enabled' not in broker_config:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="broker",
                message="No broker configurations found",
                field_path="brokers.enabled",
                code="NO_BROKERS"
            ))
            return
        
        enabled_brokers = broker_config.get('enabled', [])
        
        if not enabled_brokers:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="broker",
                message="No brokers enabled for trading",
                field_path="brokers.enabled",
                code="NO_ENABLED_BROKERS"
            ))
        
        # Validate each enabled broker
        for broker_name in enabled_brokers:
            broker_specific_config = broker_config.get(broker_name, {})
            await self._validate_individual_broker(broker_name, broker_specific_config, result)
    
    async def _validate_individual_broker(self, broker_name: str, broker_config: Dict[str, Any], result: ValidationResult):
        """Validate individual broker configuration"""
        # Common validation for all brokers
        if not broker_config:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="broker",
                message=f"No configuration found for broker: {broker_name}",
                field_path=f"brokers.{broker_name}",
                code="BROKER_NO_CONFIG"
            ))
            return
        
        # API credentials validation
        required_fields = self._get_broker_required_fields(broker_name)
        for field in required_fields:
            if field not in broker_config:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="broker",
                    message=f"Missing required field for {broker_name}: {field}",
                    field_path=f"brokers.{broker_name}.{field}",
                    code="MISSING_BROKER_FIELD"
                ))
        
        # API endpoint validation
        if 'api_url' in broker_config:
            url = broker_config['api_url']
            if not self._is_valid_url(url):
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="broker",
                    message=f"Invalid API URL for {broker_name}: {url}",
                    field_path=f"brokers.{broker_name}.api_url",
                    code="INVALID_API_URL"
                ))
        
        # Rate limiting
        if 'rate_limit' in broker_config:
            rate_limit = broker_config['rate_limit']
            if not isinstance(rate_limit, dict):
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="broker",
                    message=f"Rate limit must be a dictionary for {broker_name}",
                    field_path=f"brokers.{broker_name}.rate_limit",
                    code="INVALID_RATE_LIMIT_FORMAT"
                ))
        
        # Sandbox/testnet validation
        if 'sandbox' in broker_config and broker_config['sandbox']:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                category="broker",
                message=f"Broker {broker_name} is configured for sandbox/testnet mode",
                field_path=f"brokers.{broker_name}.sandbox",
                code="SANDBOX_MODE"
            ))
    
    def _get_broker_required_fields(self, broker_name: str) -> List[str]:
        """Get required fields for specific broker type"""
        broker_fields = {
            'alpaca': ['api_key', 'secret_key', 'api_url'],
            'binance': ['api_key', 'secret_key'],
            'interactive_brokers': ['host', 'port', 'client_id'],
            'paper_trading': [],
            'mock': []
        }
        return broker_fields.get(broker_name.lower(), [])
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid"""
        try:
            result = urllib.parse.urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    async def _validate_strategy_config(self, strategy_config: Dict[str, Any], result: ValidationResult):
        """Validate strategy configurations"""
        if not strategy_config or 'enabled' not in strategy_config:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="strategy",
                message="No strategy configurations found",
                field_path="strategies.enabled",
                code="NO_STRATEGIES"
            ))
            return
        
        enabled_strategies = strategy_config.get('enabled', [])
        
        if not enabled_strategies:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="strategy",
                message="No trading strategies enabled",
                field_path="strategies.enabled",
                code="NO_ENABLED_STRATEGIES"
            ))
        
        # Validate each enabled strategy
        for strategy_name in enabled_strategies:
            strategy_specific_config = strategy_config.get(strategy_name, {})
            await self._validate_individual_strategy(strategy_name, strategy_specific_config, result)
    
    async def _validate_individual_strategy(self, strategy_name: str, strategy_config: Dict[str, Any], result: ValidationResult):
        """Validate individual strategy configuration"""
        if not strategy_config:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="strategy",
                message=f"No configuration found for strategy: {strategy_name}",
                field_path=f"strategies.{strategy_name}",
                code="STRATEGY_NO_CONFIG"
            ))
            return
        
        # Common strategy parameters
        if 'position_size' in strategy_config:
            position_size = strategy_config['position_size']
            if not isinstance(position_size, (int, float)) or position_size <= 0:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="strategy",
                    message=f"Position size must be positive for {strategy_name}",
                    field_path=f"strategies.{strategy_name}.position_size",
                    code="INVALID_POSITION_SIZE"
                ))
        
        # Stop loss validation
        if 'stop_loss' in strategy_config:
            stop_loss = strategy_config['stop_loss']
            if isinstance(stop_loss, dict):
                if 'percentage' in stop_loss:
                    percentage = stop_loss['percentage']
                    if not isinstance(percentage, (int, float)) or percentage <= 0 or percentage > 50:
                        result.add_issue(ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            category="strategy",
                            message=f"Stop loss percentage must be between 0 and 50 for {strategy_name}",
                            field_path=f"strategies.{strategy_name}.stop_loss.percentage",
                            code="INVALID_STOP_LOSS_PERCENTAGE"
                        ))
        
        # Take profit validation
        if 'take_profit' in strategy_config:
            take_profit = strategy_config['take_profit']
            if isinstance(take_profit, dict):
                if 'percentage' in take_profit:
                    percentage = take_profit['percentage']
                    if not isinstance(percentage, (int, float)) or percentage <= 0:
                        result.add_issue(ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            category="strategy",
                            message=f"Take profit percentage must be positive for {strategy_name}",
                            field_path=f"strategies.{strategy_name}.take_profit.percentage",
                            code="INVALID_TAKE_PROFIT_PERCENTAGE"
                        ))
        
        # Timeframe validation
        if 'timeframe' in strategy_config:
            timeframe = strategy_config['timeframe']
            valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M']
            if timeframe not in valid_timeframes:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="strategy",
                    message=f"Unknown timeframe for {strategy_name}: {timeframe}",
                    field_path=f"strategies.{strategy_name}.timeframe",
                    code="UNKNOWN_TIMEFRAME",
                    suggestion=f"Valid timeframes: {', '.join(valid_timeframes)}"
                ))
    
    async def _validate_risk_management(self, risk_config: Dict[str, Any], result: ValidationResult):
        """Validate risk management configuration"""
        # Maximum position size
        if 'max_position_size' in risk_config:
            max_size = risk_config['max_position_size']
            if not isinstance(max_size, (int, float)) or max_size <= 0:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="risk",
                    message="Maximum position size must be positive",
                    field_path="risk_management.max_position_size",
                    code="INVALID_MAX_POSITION_SIZE"
                ))
        
        # Maximum drawdown
        if 'max_drawdown' in risk_config:
            max_drawdown = risk_config['max_drawdown']
            if not isinstance(max_drawdown, (int, float)) or max_drawdown <= 0 or max_drawdown > 100:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="risk",
                    message="Maximum drawdown must be between 0 and 100",
                    field_path="risk_management.max_drawdown",
                    code="INVALID_MAX_DRAWDOWN"
                ))
        
        # Daily loss limit
        if 'daily_loss_limit' in risk_config:
            daily_loss = risk_config['daily_loss_limit']
            if not isinstance(daily_loss, (int, float)) or daily_loss < 0:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="risk",
                    message="Daily loss limit must be non-negative",
                    field_path="risk_management.daily_loss_limit",
                    code="INVALID_DAILY_LOSS_LIMIT"
                ))
        
        # Maximum concurrent positions
        if 'max_concurrent_positions' in risk_config:
            max_positions = risk_config['max_concurrent_positions']
            if not isinstance(max_positions, int) or max_positions < 1:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="risk",
                    message="Maximum concurrent positions should be a positive integer",
                    field_path="risk_management.max_concurrent_positions",
                    code="INVALID_MAX_POSITIONS"
                ))
    
    async def _validate_system_settings(self, system_config: Dict[str, Any], result: ValidationResult):
        """Validate system configuration"""
        # Log level
        if 'log_level' in system_config:
            log_level = system_config['log_level']
            valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            if log_level not in valid_levels:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="system",
                    message=f"Invalid log level: {log_level}",
                    field_path="system.log_level",
                    code="INVALID_LOG_LEVEL",
                    suggestion=f"Valid levels: {', '.join(valid_levels)}"
                ))
        
        # Thread pool size
        if 'thread_pool_size' in system_config:
            pool_size = system_config['thread_pool_size']
            if not isinstance(pool_size, int) or pool_size < 1 or pool_size > 100:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="system",
                    message="Thread pool size should be between 1 and 100",
                    field_path="system.thread_pool_size",
                    code="INVALID_THREAD_POOL_SIZE"
                ))
        
        # Max memory usage
        if 'max_memory_mb' in system_config:
            max_memory = system_config['max_memory_mb']
            if not isinstance(max_memory, (int, float)) or max_memory <= 0:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="system",
                    message="Max memory must be positive",
                    field_path="system.max_memory_mb",
                    code="INVALID_MAX_MEMORY"
                ))
    
    async def _validate_security_settings(self, config_data: Dict[str, Any], result: ValidationResult):
        """Validate security configuration"""
        # Check for hardcoded secrets in configuration
        config_str = json.dumps(config_data)
        
        # Check for common secret patterns
        secret_patterns = [
            r'api[_-]?key["\']?\s*[:=]\s*["\']([^"\']{20,})["\']',  # API keys
            r'secret[_-]?key["\']?\s*[:=]\s*["\']([^"\']{20,})["\']',  # Secret keys
            r'password["\']?\s*[:=]\s*["\'][^"\']{8,}["\']'  # Passwords
        ]
        
        for pattern in secret_patterns:
            matches = re.finditer(pattern, config_str, re.IGNORECASE)
            for match in matches:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="security",
                    message="Potential hardcoded secret detected in configuration",
                    field_path="security",
                    code="HARDCODED_SECRET",
                    suggestion="Use environment variables or secure secret management"
                ))
        
        # Check for proper file permissions
        # This would require file system access to check actual permissions
    
    async def _test_connections(self, config_data: Dict[str, Any], result: ValidationResult):
        """Test connections to external services"""
        # This would require actual network calls to test connections
        # For now, we'll just add informational messages
        
        if 'brokers' in config_data:
            enabled_brokers = config_data['brokers'].get('enabled', [])
            for broker_name in enabled_brokers:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    category="connection",
                    message=f"Connection test for {broker_name} would be performed here",
                    field_path=f"brokers.{broker_name}",
                    code="CONNECTION_TEST_PENDING"
                ))
    
    async def _validate_performance_config(self, config_data: Dict[str, Any], result: ValidationResult):
        """Validate performance-related configuration"""
        # Check for performance bottlenecks
        
        # High connection pool size
        if 'database' in config_data:
            db_config = config_data['database']
            if 'connection_pool_size' in db_config and db_config['connection_pool_size'] > 50:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="performance",
                    message="High database connection pool size may impact performance",
                    field_path="database.connection_pool_size",
                    code="HIGH_POOL_SIZE"
                ))
        
        # Many enabled strategies
        if 'strategies' in config_data:
            enabled_strategies = config_data['strategies'].get('enabled', [])
            if len(enabled_strategies) > 10:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="performance",
                    message=f"Large number of enabled strategies ({len(enabled_strategies)}) may impact performance",
                    field_path="strategies.enabled",
                    code="MANY_STRATEGIES"
                ))
    
    def generate_report(self, results: List[ValidationResult]) -> str:
        """Generate comprehensive validation report"""
        report = []
        report.append("=" * 60)
        report.append("    CONFIGURATION VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.utcnow().isoformat()}")
        report.append(f"Files validated: {len(results)}")
        
        total_errors = sum(r.errors for r in results)
        total_warnings = sum(r.warnings for r in results)
        total_suggestions = sum(r.suggestions for r in results)
        
        report.append("\n--- SUMMARY ---")
        report.append(f"Errors: {total_errors}")
        report.append(f"Warnings: {total_warnings}")
        report.append(f"Suggestions: {total_suggestions}")
        
        if total_errors == 0:
            report.append("\n✅ All configuration files are valid!")
        else:
            report.append(f"\n❌ Found {total_errors} critical issues that must be fixed.")
        
        # Detailed results for each file
        for result in results:
            report.append(f"\n--- {result.config_file} ---")
            
            if result.is_valid:
                report.append("✅ Valid configuration")
            else:
                report.append("❌ Invalid configuration")
            
            # Group issues by severity
            for severity in [ValidationSeverity.ERROR, ValidationSeverity.WARNING, ValidationSeverity.SUGGESTION, ValidationSeverity.INFO]:
                issues = result.get_issues_by_severity(severity)
                if issues:
                    severity_title = severity.value.upper()
                    report.append(f"\n{severity_title}S:")
                    
                    for issue in issues:
                        report.append(f"  • {issue.message}")
                        if issue.suggestion:
                            report.append(f"    💡 {issue.suggestion}")
                        if issue.field_path:
                            report.append(f"    📍 Field: {issue.field_path}")
        
        # Overall recommendations
        report.append("\n--- RECOMMENDATIONS ---")
        
        if total_errors > 0:
            report.append("1. Fix all ERROR-level issues before deploying to production")
        
        if total_warnings > 0:
            report.append("2. Review and address WARNING-level issues for better stability")
        
        report.append("3. Consider implementing the suggested improvements")
        report.append("4. Test configuration changes in a development environment first")
        report.append("5. Keep configuration files secure and avoid hardcoding secrets")
        
        return "\n".join(report)

class ConfigValidatorCLI:
    """Command-line interface for configuration validation"""
    
    def __init__(self):
        self.validator = ConfigValidator()
    
    async def validate_file(self, config_path: str, output_file: Optional[str] = None) -> bool:
        """Validate a single configuration file"""
        print(f"Validating configuration file: {config_path}")
        
        result = await self.validator.validate_config_file(config_path)
        
        # Print immediate results
        if result.is_valid:
            print("✅ Configuration is valid!")
        else:
            print("❌ Configuration has issues:")
        
        if result.errors > 0:
            print(f"   - {result.errors} errors")
        if result.warnings > 0:
            print(f"   - {result.warnings} warnings")
        if result.suggestions > 0:
            print(f"   - {result.suggestions} suggestions")
        
        # Generate and display report
        report = self.validator.generate_report([result])
        print("\n" + report)
        
        # Save report if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"\nReport saved to: {output_file}")
        
        return result.is_valid
    
    async def validate_directory(self, config_dir: str, output_file: Optional[str] = None) -> bool:
        """Validate all configuration files in a directory"""
        config_dir = Path(config_dir)
        
        if not config_dir.exists() or not config_dir.is_dir():
            print(f"Error: Directory not found: {config_dir}")
            return False
        
        print(f"Validating configuration files in: {config_dir}")
        
        # Find all config files
        config_extensions = ['.json', '.yaml', '.yml']
        config_files = []
        
        for ext in config_extensions:
            config_files.extend(config_dir.glob(f'*{ext}'))
            config_files.extend(config_dir.glob(f'config*{ext}'))
        
        if not config_files:
            print("No configuration files found.")
            return True
        
        print(f"Found {len(config_files)} configuration files.")
        
        # Validate each file
        results = []
        for config_file in config_files:
            print(f"\nValidating: {config_file.name}")
            result = await self.validator.validate_config_file(config_file)
            results.append(result)
        
        # Generate and display combined report
        report = self.validator.generate_report(results)
        print("\n" + "="*60)
        print(report)
        
        # Save report if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"\nReport saved to: {output_file}")
        
        # Return overall success
        all_valid = all(r.is_valid for r in results)
        return all_valid

async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Configuration Validator for Trading System')
    parser.add_argument('config_path', help='Path to configuration file or directory')
    parser.add_argument('--output', '-o', help='Output file for validation report')
    parser.add_argument('--strict', action='store_true', help='Enable strict validation mode')
    parser.add_argument('--no-connections', action='store_true', help='Disable connection testing')
    parser.add_argument('--no-performance', action='store_true', help='Disable performance validation')
    
    args = parser.parse_args()
    
    # Initialize validator
    validator_cli = ConfigValidatorCLI()
    validator_cli.validator.strict_mode = args.strict
    validator_cli.validator.connection_tests_enabled = not args.no_connections
    validator_cli.validator.performance_tests_enabled = not args.no_performance
    
    try:
        # Determine if path is file or directory
        config_path = Path(args.config_path)
        
        if config_path.is_file():
            success = await validator_cli.validate_file(args.config_path, args.output)
        elif config_path.is_dir():
            success = await validator_cli.validate_directory(args.config_path, args.output)
        else:
            print(f"Error: Path does not exist: {args.config_path}")
            return 1
        
        return 0 if success else 1
    
    except KeyboardInterrupt:
        print("\nValidation interrupted by user.")
        return 1
    except Exception as e:
        print(f"Validation error: {e}")
        logger.error(f"Validation error: {e}", exc_info=True)
        return 1

if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
