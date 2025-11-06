"""
Configuration Validation System
Validates configuration data for type, format, and business rules
"""

import re
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse


class ConfigValidationError(Exception):
    """Configuration validation error"""
    pass


class ConfigValidator:
    """
    Configuration Validator
    
    Validates configuration data for:
    - Type validation (string, number, boolean, etc.)
    - Format validation (URL, email, port, etc.)
    - Range validation (min/max values)
    - Required fields validation
    - Business rule validation
    - Cross-reference validation
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validation_rules: Dict[str, Dict[str, Any]] = {}
        self.custom_validators: Dict[str, Callable] = {}
        self._load_builtin_rules()
    
    def validate(self, config_data: Dict[str, Any], config_name: str) -> List[str]:
        """
        Validate configuration data
        
        Args:
            config_data: Configuration data to validate
            config_name: Name of the configuration
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        try:
            # Get validation rules for config type
            rules = self._get_validation_rules(config_name)
            
            # Validate each field
            for field_path, value in self._flatten_config(config_data).items():
                field_errors = self._validate_field(field_path, value, rules.get(field_path, {}))
                errors.extend(field_errors)
            
            # Validate cross-references
            cross_errors = self._validate_cross_references(config_data, config_name)
            errors.extend(cross_errors)
            
            # Apply custom validators
            custom_errors = self._apply_custom_validators(config_data, config_name)
            errors.extend(custom_errors)
            
            return errors
            
        except Exception as e:
            self.logger.error(f"Configuration validation error: {e}")
            return [f"Validation system error: {e}"]
    
    def add_validation_rule(self, field_path: str, rule_type: str, rule_config: Dict[str, Any]):
        """Add custom validation rule"""
        if field_path not in self.validation_rules:
            self.validation_rules[field_path] = {}
        
        self.validation_rules[field_path][rule_type] = rule_config
    
    def add_custom_validator(self, name: str, validator_func: Callable):
        """Add custom validation function"""
        self.custom_validators[name] = validator_func
    
    def validate_required_fields(self, config_data: Dict[str, Any], required_fields: List[str]) -> List[str]:
        """Validate that required fields are present and not None"""
        errors = []
        
        for field_path in required_fields:
            value = self._get_nested_value(config_data, field_path)
            if value is None or value == "":
                errors.append(f"Required field missing or empty: {field_path}")
        
        return errors
    
    def validate_field_types(self, config_data: Dict[str, Any], field_types: Dict[str, type]) -> List[str]:
        """Validate field types"""
        errors = []
        
        for field_path, expected_type in field_types.items():
            value = self._get_nested_value(config_data, field_path)
            if value is not None and not isinstance(value, expected_type):
                actual_type = type(value).__name__
                errors.append(f"Field type mismatch: {field_path} expected {expected_type.__name__}, got {actual_type}")
        
        return errors
    
    def validate_url_format(self, url: str, allowed_schemes: List[str] = None) -> bool:
        """Validate URL format"""
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False
            
            if allowed_schemes and parsed.scheme not in allowed_schemes:
                return False
            
            return True
        except:
            return False
    
    def validate_email_format(self, email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    def validate_port_range(self, port: Union[str, int]) -> bool:
        """Validate port number"""
        try:
            port_num = int(port)
            return 1 <= port_num <= 65535
        except (ValueError, TypeError):
            return False
    
    def validate_percentage(self, value: Union[str, float, int]) -> bool:
        """Validate percentage value (0-100)"""
        try:
            num = float(value)
            return 0 <= num <= 100
        except (ValueError, TypeError):
            return False
    
    def validate_positive_number(self, value: Union[str, float, int]) -> bool:
        """Validate positive number"""
        try:
            num = float(value)
            return num > 0
        except (ValueError, TypeError):
            return False
    
    def validate_non_negative_number(self, value: Union[str, float, int]) -> bool:
        """Validate non-negative number"""
        try:
            num = float(value)
            return num >= 0
        except (ValueError, TypeError):
            return False
    
    def validate_integer_range(self, value: Union[str, int], min_val: int, max_val: int) -> bool:
        """Validate integer within range"""
        try:
            num = int(value)
            return min_val <= num <= max_val
        except (ValueError, TypeError):
            return False
    
    def validate_string_length(self, value: str, min_len: int = 0, max_len: int = None) -> bool:
        """Validate string length"""
        if not isinstance(value, str):
            return False
        
        if len(value) < min_len:
            return False
        
        if max_len is not None and len(value) > max_len:
            return False
        
        return True
    
    def validate_choice(self, value: str, choices: List[str], case_sensitive: bool = True) -> bool:
        """Validate value is in allowed choices"""
        if not case_sensitive:
            return value.lower() in [choice.lower() for choice in choices]
        return value in choices
    
    def validate_regex_pattern(self, value: str, pattern: str) -> bool:
        """Validate value matches regex pattern"""
        try:
            return bool(re.match(pattern, value))
        except:
            return False
    
    def validate_conditional(self, config_data: Dict[str, Any], 
                           condition_field: str, value: Any,
                           dependency_field: str, required_value: Any) -> bool:
        """Conditional validation based on other fields"""
        dependency_value = self._get_nested_value(config_data, dependency_field)
        
        if dependency_value == required_value:
            return value is not None and value != ""
        
        return True
    
    def _validate_field(self, field_path: str, value: Any, rules: Dict[str, Any]) -> List[str]:
        """Validate individual field"""
        errors = []
        
        # Skip validation if field is empty and not required
        if value is None or value == "":
            if rules.get('required', False):
                errors.append(f"Required field is empty: {field_path}")
            return errors
        
        # Type validation
        if 'type' in rules:
            expected_type = rules['type']
            if not isinstance(value, expected_type):
                errors.append(f"Field type error: {field_path} expected {expected_type.__name__}")
        
        # Format validation
        if 'format' in rules:
            format_validator = self._get_format_validator(rules['format'])
            if not format_validator(value):
                errors.append(f"Field format error: {field_path} format {rules['format']}")
        
        # Range validation
        if 'min' in rules or 'max' in rules:
            if not isinstance(value, (int, float)):
                errors.append(f"Range validation requires numeric field: {field_path}")
            else:
                if 'min' in rules and value < rules['min']:
                    errors.append(f"Field value too small: {field_path} minimum {rules['min']}")
                if 'max' in rules and value > rules['max']:
                    errors.append(f"Field value too large: {field_path} maximum {rules['max']}")
        
        # Length validation
        if 'min_length' in rules or 'max_length' in rules:
            if not isinstance(value, str):
                errors.append(f"Length validation requires string field: {field_path}")
            else:
                if 'min_length' in rules and len(value) < rules['min_length']:
                    errors.append(f"Field too short: {field_path} minimum length {rules['min_length']}")
                if 'max_length' in rules and len(value) > rules['max_length']:
                    errors.append(f"Field too long: {field_path} maximum length {rules['max_length']}")
        
        # Choice validation
        if 'choices' in rules:
            choices = rules['choices']
            case_sensitive = rules.get('case_sensitive', True)
            if not self.validate_choice(str(value), choices, case_sensitive):
                errors.append(f"Field value not in allowed choices: {field_path} {choices}")
        
        # Regex validation
        if 'pattern' in rules:
            pattern = rules['pattern']
            if not self.validate_regex_pattern(str(value), pattern):
                errors.append(f"Field pattern mismatch: {field_path} pattern {pattern}")
        
        # Custom validation
        if 'validator' in rules:
            validator_name = rules['validator']
            if validator_name in self.custom_validators:
                try:
                    if not self.custom_validators[validator_name](value, rules.get('validator_config', {})):
                        errors.append(f"Custom validation failed: {field_path} validator {validator_name}")
                except Exception as e:
                    errors.append(f"Custom validation error: {field_path} {e}")
        
        return errors
    
    def _validate_cross_references(self, config_data: Dict[str, Any], config_name: str) -> List[str]:
        """Validate cross-references between configuration fields"""
        errors = []
        
        # Database URL consistency
        if 'database' in config_data:
            db_config = config_data['database']
            if 'url' in db_config and 'type' in db_config:
                url = db_config['url']
                db_type = db_config['type']
                
                # Check URL matches database type
                if db_type == 'sqlite' and not url.startswith('sqlite'):
                    errors.append("Database URL doesn't match type 'sqlite'")
                elif db_type == 'postgresql' and not url.startswith('postgresql'):
                    errors.append("Database URL doesn't match type 'postgresql'")
        
        # Broker API key validation
        if 'brokers' in config_data:
            for broker_name, broker_config in config_data['brokers'].items():
                if broker_config.get('enabled', False):
                    if not broker_config.get('api_key'):
                        errors.append(f"Missing API key for enabled broker: {broker_name}")
        
        # Risk management consistency
        if 'risk' in config_data:
            risk_config = config_data['risk']
            if 'max_position_size' in risk_config and 'max_daily_loss' in risk_config:
                max_pos = risk_config['max_position_size']
                max_loss = risk_config['max_daily_loss']
                if max_pos <= max_loss:
                    errors.append("Max daily loss should be less than max position size")
        
        return errors
    
    def _apply_custom_validators(self, config_data: Dict[str, Any], config_name: str) -> List[str]:
        """Apply custom validators"""
        errors = []
        
        for validator_name, validator_func in self.custom_validators.items():
            try:
                validator_errors = validator_func(config_data)
                if isinstance(validator_errors, list):
                    errors.extend(validator_errors)
                elif validator_errors:
                    errors.append(validator_errors)
            except Exception as e:
                errors.append(f"Custom validator {validator_name} error: {e}")
        
        return errors
    
    def _get_validation_rules(self, config_name: str) -> Dict[str, Dict[str, Any]]:
        """Get validation rules for configuration"""
        # Load built-in validation rules based on config type
        if config_name == 'main':
            return self._get_main_config_rules()
        elif config_name.startswith('broker'):
            return self._get_broker_config_rules()
        elif config_name.startswith('risk'):
            return self._get_risk_config_rules()
        else:
            return {}
    
    def _get_format_validator(self, format_type: str) -> Callable:
        """Get format validator function"""
        format_validators = {
            'url': lambda x: self.validate_url_format(str(x)),
            'email': lambda x: self.validate_email_format(str(x)),
            'port': lambda x: self.validate_port_range(x),
            'percentage': lambda x: self.validate_percentage(x),
            'positive_number': lambda x: self.validate_positive_number(x),
            'non_negative_number': lambda x: self.validate_non_negative_number(x),
        }
        
        return format_validators.get(format_type, lambda x: True)
    
    def _flatten_config(self, config: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten nested configuration dictionary"""
        items = []
        for k, v in config.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_config(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _get_nested_value(self, config: Dict[str, Any], field_path: str) -> Any:
        """Get nested configuration value using dot notation"""
        keys = field_path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    def _load_builtin_rules(self):
        """Load built-in validation rules"""
        # Main configuration rules
        self._load_main_config_rules()
        
        # Broker configuration rules
        self._load_broker_config_rules()
        
        # Risk configuration rules
        self._load_risk_config_rules()
    
    def _get_main_config_rules(self) -> Dict[str, Dict[str, Any]]:
        """Get main configuration validation rules"""
        return {
            'environment': {
                'type': str,
                'choices': ['development', 'staging', 'production', 'testing'],
                'required': True
            },
            'debug': {'type': bool},
            'api_host': {
                'type': str,
                'min_length': 1,
                'required': True
            },
            'api_port': {
                'type': int,
                'min': 1024,
                'max': 65535,
                'required': True
            },
            'db_type': {
                'type': str,
                'choices': ['sqlite', 'postgresql'],
                'required': True
            },
            'db_path': {
                'type': str,
                'min_length': 1,
                'required': True
            },
            'max_position_size': {
                'type': (int, float),
                'min': 0,
                'required': True
            },
            'max_daily_loss': {
                'type': (int, float),
                'min': 0,
                'required': True
            },
            'rate_limit_requests': {
                'type': int,
                'min': 1,
                'required': True
            },
            'rate_limit_period': {
                'type': int,
                'min': 1,
                'required': True
            },
            'ai_provider': {
                'type': str,
                'choices': ['openai', 'anthropic', 'local'],
                'required': True
            },
            'log_level': {
                'type': str,
                'choices': ['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                'required': True
            }
        }
    
    def _load_main_config_rules(self):
        """Load main configuration validation rules"""
        rules = self._get_main_config_rules()
        for field_path, rule_config in rules.items():
            self.add_validation_rule(field_path, 'builtin', rule_config)
    
    def _get_broker_config_rules(self) -> Dict[str, Dict[str, Any]]:
        """Get broker configuration validation rules"""
        return {
            'enabled': {'type': bool},
            'api_key': {'type': str, 'required_if_enabled': True},
            'secret_key': {'type': str, 'required_if_enabled': True},
            'testnet': {'type': bool},
            'host': {
                'type': str,
                'min_length': 1,
                'required_if_enabled': True
            },
            'port': {
                'type': int,
                'min': 1024,
                'max': 65535,
                'required_if_enabled': True
            },
            'client_id': {
                'type': int,
                'min': 1,
                'max': 32,
                'required_if_enabled': True
            },
            'rate_limit': {
                'type': int,
                'min': 1,
                'max': 10000
            },
            'paper': {'type': bool}
        }
    
    def _load_broker_config_rules(self):
        """Load broker configuration validation rules"""
        rules = self._get_broker_config_rules()
        for field_path, rule_config in rules.items():
            self.add_validation_rule(f"brokers.*.{field_path}", 'builtin', rule_config)
    
    def _get_risk_config_rules(self) -> Dict[str, Dict[str, Any]]:
        """Get risk configuration validation rules"""
        return {
            'max_position_size': {
                'type': (int, float),
                'min': 0,
                'required': True
            },
            'max_daily_loss': {
                'type': (int, float),
                'min': 0,
                'required': True
            },
            'max_open_orders': {
                'type': int,
                'min': 1,
                'required': True
            },
            'risk_per_trade': {
                'type': (int, float),
                'min': 0,
                'max': 1,
                'required': True
            },
            'stop_loss_percentage': {
                'type': (int, float),
                'min': 0,
                'max': 100,
                'required': True
            },
            'take_profit_percentage': {
                'type': (int, float),
                'min': 0,
                'max': 100
            },
            'correlation_limit': {
                'type': (int, float),
                'min': 0,
                'max': 1,
                'required': True
            }
        }
    
    def _load_risk_config_rules(self):
        """Load risk configuration validation rules"""
        rules = self._get_risk_config_rules()
        for field_path, rule_config in rules.items():
            self.add_validation_rule(f"risk.{field_path}", 'builtin', rule_config)