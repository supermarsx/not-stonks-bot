#!/usr/bin/env python3
"""
Day Trading Orchestrator - Configuration Validator
Validates configuration files and detects common issues
"""

import sys
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ValidationError:
    """Configuration validation error"""
    section: str
    field: str
    message: str
    severity: str  # "error", "warning", "info"

@dataclass
class ValidationResult:
    """Validation result container"""
    is_valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationError]
    infos: List[ValidationError]

class ConfigurationValidator:
    """Validates Day Trading Orchestrator configuration files"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.config = {}
        self.result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            infos=[]
        )
    
    def validate(self) -> ValidationResult:
        """Run comprehensive configuration validation"""
        print(f"🔍 Validating configuration: {self.config_path}")
        
        # Load configuration file
        if not self.load_config():
            return self.result
        
        # Run validation checks
        self.check_file_structure()
        self.check_database_config()
        self.check_broker_configs()
        self.check_ai_config()
        self.check_risk_config()
        self.check_logging_config()
        self.check_ui_config()
        self.check_security_config()
        
        # Generate summary
        self.generate_summary()
        
        return self.result
    
    def load_config(self) -> bool:
        """Load configuration file"""
        try:
            if not self.config_path.exists():
                self.add_error("global", "config_file", "Configuration file not found", "error")
                return False
            
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            
            self.add_info("global", "config_file", "Configuration file loaded successfully", "info")
            return True
            
        except json.JSONDecodeError as e:
            self.add_error("global", "json_syntax", f"Invalid JSON syntax: {e}", "error")
            return False
        except Exception as e:
            self.add_error("global", "file_load", f"Failed to load configuration: {e}", "error")
            return False
    
    def check_file_structure(self):
        """Check configuration file structure"""
        required_sections = ["database", "brokers", "risk"]
        
        for section in required_sections:
            if section not in self.config:
                self.add_error("global", f"missing_{section}", f"Required section '{section}' not found", "error")
            else:
                self.add_info("global", f"section_{section}", f"Section '{section}' found", "info")
        
        # Check for deprecated sections
        deprecated_sections = ["old_config", "legacy_settings"]
        for section in deprecated_sections:
            if section in self.config:
                self.add_warning("global", f"deprecated_{section}", f"Section '{section}' is deprecated", "warning")
    
    def check_database_config(self):
        """Validate database configuration"""
        db_config = self.config.get("database", {})
        
        # Check required fields
        if "url" not in db_config:
            self.add_error("database", "url", "Database URL not specified", "error")
        else:
            url = db_config["url"]
            
            # Validate URL format
            if not url.startswith(("sqlite:", "postgresql:", "mysql:")):
                self.add_error("database", "url_format", 
                             f"Invalid database URL format: {url}", "error")
            else:
                self.add_info("database", "url_format", "Database URL format is valid", "info")
            
            # Check SQLite specific settings
            if url.startswith("sqlite:"):
                # Check if database file path is writable
                db_path = url.replace("sqlite:///", "")
                if db_path:
                    db_file = Path(db_path)
                    if db_file.parent.exists() and not os.access(db_file.parent, os.W_OK):
                        self.add_warning("database", "writable", 
                                       f"Database directory not writable: {db_file.parent}", "warning")
        
        # Check optional settings
        if "echo" in db_config and not isinstance(db_config["echo"], bool):
            self.add_error("database", "echo_type", "Database 'echo' must be boolean", "error")
        
        if "pool_size" in db_config:
            pool_size = db_config["pool_size"]
            if not isinstance(pool_size, int) or pool_size < 1:
                self.add_error("database", "pool_size", "Database pool_size must be positive integer", "error")
            elif pool_size > 100:
                self.add_warning("database", "pool_size_large", 
                               f"Large pool_size ({pool_size}) may impact performance", "warning")
    
    def check_broker_configs(self):
        """Validate broker configurations"""
        brokers_config = self.config.get("brokers", {})
        
        if not brokers_config:
            self.add_warning("brokers", "empty", "No brokers configured", "warning")
            return
        
        supported_brokers = ["alpaca", "binance", "ibkr", "trading212", "degiro", "xtb", "trade_republic"]
        
        for broker_name, broker_config in brokers_config.items():
            self.validate_single_broker(broker_name, broker_config, supported_brokers)
    
    def validate_single_broker(self, name: str, config: Dict[str, Any], supported_brokers: List[str]):
        """Validate individual broker configuration"""
        # Check if broker is supported
        if name.lower() not in supported_brokers:
            self.add_warning("brokers", f"unsupported_{name}", 
                           f"Broker '{name}' not in supported list", "warning")
        
        # Check if enabled
        if not config.get("enabled", False):
            self.add_info("brokers", f"disabled_{name}", f"Broker '{name}' is disabled", "info")
            return
        
        self.add_info("brokers", f"enabled_{name}", f"Broker '{name}' is enabled", "info")
        
        # Validate common broker fields
        if name.lower() in ["alpaca", "binance"]:
            self.validate_api_key_broker(name, config)
        elif name.lower() == "ibkr":
            self.validate_ibkr_broker(name, config)
    
    def validate_api_key_broker(self, name: str, config: Dict[str, Any]):
        """Validate broker with API key authentication"""
        api_key = config.get("api_key")
        secret_key = config.get("secret_key")
        
        # Check API key presence
        if not api_key:
            self.add_error("brokers", f"{name}_api_key", 
                         f"{name} API key not configured", "error")
        elif api_key in ["YOUR_ALPACA_API_KEY", "YOUR_BINANCE_API_KEY"]:
            self.add_warning("brokers", f"{name}_api_key_placeholder", 
                           f"{name} API key is placeholder value", "warning")
        else:
            self.validate_api_key_format(name, api_key, "api_key")
        
        # Check secret key
        if not secret_key:
            self.add_error("brokers", f"{name}_secret_key", 
                         f"{name} secret key not configured", "error")
        elif secret_key in ["YOUR_ALPACA_SECRET_KEY", "YOUR_BINANCE_SECRET_KEY"]:
            self.add_warning("brokers", f"{name}_secret_key_placeholder", 
                           f"{name} secret key is placeholder value", "warning")
        else:
            self.validate_api_key_format(name, secret_key, "secret_key")
        
        # Check environment setting
        if name.lower() == "alpaca":
            paper_setting = config.get("paper")
            if paper_setting is None:
                self.add_warning("brokers", f"{name}_paper_mode", 
                               f"{name} paper mode not explicitly set", "warning")
            elif not isinstance(paper_setting, bool):
                self.add_error("brokers", f"{name}_paper_type", 
                             f"{name} paper setting must be boolean", "error")
        
        elif name.lower() == "binance":
            testnet_setting = config.get("testnet")
            if testnet_setting is None:
                self.add_warning("brokers", f"{name}_testnet_mode", 
                               f"{name} testnet mode not explicitly set", "warning")
            elif not isinstance(testnet_setting, bool):
                self.add_error("brokers", f"{name}_testnet_type", 
                             f"{name} testnet setting must be boolean", "error")
    
    def validate_api_key_format(self, broker: str, api_key: str, key_type: str):
        """Validate API key format"""
        if not isinstance(api_key, str):
            self.add_error("brokers", f"{broker}_{key_type}_type", 
                         f"{broker} {key_type} must be string", "error")
            return
        
        # Broker-specific validation
        if broker.lower() == "alpaca":
            # Alpaca API keys are typically alphanumeric
            if not re.match(r'^[A-Za-z0-9]+$', api_key):
                self.add_warning("brokers", f"{broker}_{key_type}_format", 
                               f"{broker} {key_type} may have invalid format", "warning")
        
        elif broker.lower() == "binance":
            # Binance API keys are 64 characters
            if len(api_key) != 64:
                self.add_warning("brokers", f"{broker}_{key_type}_length", 
                               f"{broker} {key_type} should be 64 characters", "warning")
    
    def validate_ibkr_broker(self, name: str, config: Dict[str, Any]):
        """Validate Interactive Brokers configuration"""
        required_fields = ["host", "port"]
        
        for field in required_fields:
            if field not in config:
                self.add_error("brokers", f"{name}_{field}", 
                             f"{name} {field} not configured", "error")
        
        # Validate host
        if "host" in config:
            host = config["host"]
            if not isinstance(host, str) or not host:
                self.add_error("brokers", f"{name}_host_type", 
                             f"{name} host must be non-empty string", "error")
        
        # Validate port
        if "port" in config:
            port = config["port"]
            if not isinstance(port, int) or port < 1 or port > 65535:
                self.add_error("brokers", f"{name}_port_invalid", 
                             f"{name} port must be between 1 and 65535", "error")
    
    def check_ai_config(self):
        """Validate AI configuration"""
        ai_config = self.config.get("ai", {})
        
        if not ai_config:
            self.add_info("ai", "empty", "AI configuration not present", "info")
            return
        
        # Check trading mode
        trading_mode = ai_config.get("trading_mode", "PAPER")
        valid_modes = ["PAPER", "LIVE", "DEMO"]
        
        if trading_mode not in valid_modes:
            self.add_error("ai", "trading_mode", 
                         f"Invalid trading mode: {trading_mode}. Valid modes: {valid_modes}", "error")
        else:
            if trading_mode == "LIVE":
                self.add_warning("ai", "live_mode", 
                               "LIVE trading mode enabled - be careful with real money!", "warning")
            self.add_info("ai", "trading_mode", f"Trading mode set to {trading_mode}", "info")
        
        # Check API keys
        openai_key = ai_config.get("openai_api_key")
        if openai_key:
            if openai_key == "YOUR_OPENAI_API_KEY":
                self.add_warning("ai", "openai_key_placeholder", 
                               "OpenAI API key is placeholder value", "warning")
            elif not openai_key.startswith("sk-"):
                self.add_warning("ai", "openai_key_format", 
                               "OpenAI API key should start with 'sk-'", "warning")
        
        anthropic_key = ai_config.get("anthropic_api_key")
        if anthropic_key:
            if anthropic_key == "YOUR_ANTHROPIC_API_KEY":
                self.add_warning("ai", "anthropic_key_placeholder", 
                               "Anthropic API key is placeholder value", "warning")
        
        # Check local models configuration
        local_models = ai_config.get("local_models", {})
        if local_models.get("enabled", False):
            if not local_models.get("model_path"):
                self.add_warning("ai", "local_model_path", 
                               "Local models enabled but model path not specified", "warning")
    
    def check_risk_config(self):
        """Validate risk management configuration"""
        risk_config = self.config.get("risk", {})
        
        if not risk_config:
            self.add_error("risk", "empty", "Risk management configuration missing", "error")
            return
        
        # Check required risk parameters
        required_params = ["max_position_size", "max_daily_loss"]
        
        for param in required_params:
            if param not in risk_config:
                self.add_error("risk", f"missing_{param}", 
                             f"Required risk parameter '{param}' not configured", "error")
            else:
                value = risk_config[param]
                if not isinstance(value, (int, float)) or value <= 0:
                    self.add_error("risk", f"invalid_{param}", 
                                 f"Risk parameter '{param}' must be positive number", "error")
        
        # Validate risk parameters
        if "max_position_size" in risk_config:
            max_pos = risk_config["max_position_size"]
            if max_pos > 1000000:  # $1M
                self.add_warning("risk", "max_position_large", 
                               f"Large maximum position size: ${max_pos:,}", "warning")
            elif max_pos < 100:  # $100
                self.add_warning("risk", "max_position_small", 
                               f"Small maximum position size: ${max_pos}", "warning")
        
        if "max_daily_loss" in risk_config:
            max_loss = risk_config["max_daily_loss"]
            if max_loss > 50000:  # $50K
                self.add_warning("risk", "max_daily_loss_large", 
                               f"Large daily loss limit: ${max_loss:,}", "warning")
        
        # Check circuit breakers
        circuit_breakers = risk_config.get("circuit_breakers", {})
        if circuit_breakers.get("enabled", False):
            required_cb_params = ["daily_loss_limit", "consecutive_loss_limit"]
            for param in required_cb_params:
                if param not in circuit_breakers:
                    self.add_warning("risk", f"missing_cb_{param}", 
                                   f"Circuit breaker parameter '{param}' not configured", "warning")
    
    def check_logging_config(self):
        """Validate logging configuration"""
        logging_config = self.config.get("logging", {})
        
        if not logging_config:
            self.add_info("logging", "empty", "Logging configuration not present", "info")
            return
        
        # Check log level
        log_level = logging_config.get("level", "INFO").upper()
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        
        if log_level not in valid_levels:
            self.add_error("logging", "log_level", 
                         f"Invalid log level: {log_level}. Valid levels: {valid_levels}", "error")
        else:
            self.add_info("logging", "log_level", f"Log level set to {log_level}", "info")
        
        # Check log file
        log_file = logging_config.get("file")
        if log_file:
            log_path = Path(log_file)
            log_dir = log_path.parent
            
            if not log_dir.exists():
                self.add_warning("logging", "log_dir_missing", 
                               f"Log directory does not exist: {log_dir}", "warning")
            elif not os.access(log_dir, os.W_OK):
                self.add_error("logging", "log_dir_not_writable", 
                             f"Log directory not writable: {log_dir}", "error")
        
        # Check log file size limits
        if "max_file_size" in logging_config:
            max_size = logging_config["max_file_size"]
            if not isinstance(max_size, (int, float)) or max_size <= 0:
                self.add_error("logging", "max_file_size_invalid", 
                             "Log file size must be positive number", "error")
    
    def check_ui_config(self):
        """Validate UI configuration"""
        ui_config = self.config.get("ui", {})
        
        if not ui_config:
            self.add_info("ui", "empty", "UI configuration not present", "info")
            return
        
        # Check theme
        theme = ui_config.get("theme", "matrix")
        valid_themes = ["matrix", "dark", "light"]
        
        if theme not in valid_themes:
            self.add_warning("ui", "theme", 
                           f"Unknown theme: {theme}. Valid themes: {valid_themes}", "warning")
        
        # Check terminal size
        terminal_size = ui_config.get("terminal_size", {})
        if "width" in terminal_size:
            width = terminal_size["width"]
            if not isinstance(width, int) or width < 80:
                self.add_warning("ui", "width_small", 
                               f"Terminal width {width} may be too small (minimum 80)", "warning")
        
        if "height" in terminal_size:
            height = terminal_size["height"]
            if not isinstance(height, int) or height < 24:
                self.add_warning("ui", "height_small", 
                               f"Terminal height {height} may be too small (minimum 24)", "warning")
    
    def check_security_config(self):
        """Validate security configuration"""
        security_config = self.config.get("security", {})
        
        if not security_config:
            self.add_info("security", "empty", "Security configuration not present", "info")
            return
        
        # Check encryption settings
        encryption = security_config.get("encryption", {})
        if encryption.get("enabled", False):
            algorithm = encryption.get("algorithm", "AES-256")
            valid_algorithms = ["AES-256", "AES-128", "ChaCha20"]
            
            if algorithm not in valid_algorithms:
                self.add_warning("security", "encryption_algorithm", 
                               f"Unknown encryption algorithm: {algorithm}", "warning")
        
        # Check authentication settings
        auth = security_config.get("authentication", {})
        if auth.get("api_keys_required", False):
            self.add_info("security", "api_keys_required", "API key authentication required", "info")
        
        if "session_timeout" in auth:
            timeout = auth["session_timeout"]
            if not isinstance(timeout, int) or timeout < 300:  # 5 minutes
                self.add_warning("security", "session_timeout_low", 
                               f"Session timeout {timeout}s may be too low (minimum 300s)", "warning")
    
    def add_error(self, section: str, field: str, message: str, severity: str = "error"):
        """Add validation error"""
        error = ValidationError(section, field, message, severity)
        
        if severity == "error":
            self.result.errors.append(error)
            self.result.is_valid = False
        elif severity == "warning":
            self.result.warnings.append(error)
        else:
            self.result.infos.append(error)
    
    def add_warning(self, section: str, field: str, message: str, severity: str = "warning"):
        """Add validation warning"""
        self.add_error(section, field, message, severity)
    
    def add_info(self, section: str, field: str, message: str, severity: str = "info"):
        """Add validation info"""
        self.add_error(section, field, message, severity)
    
    def generate_summary(self):
        """Generate validation summary"""
        print(f"\n{'='*60}")
        print("📋 VALIDATION SUMMARY")
        print(f"{'='*60}")
        
        if self.result.is_valid and not self.result.warnings:
            print("✅ Configuration is valid!")
        elif self.result.is_valid:
            print("⚠️  Configuration is valid with warnings")
        else:
            print("❌ Configuration has errors")
        
        print(f"\n📊 Results:")
        print(f"  Errors: {len(self.result.errors)}")
        print(f"  Warnings: {len(self.result.warnings)}")
        print(f"  Info: {len(self.result.infos)}")
        
        # Print errors
        if self.result.errors:
            print(f"\n❌ ERRORS ({len(self.result.errors)}):")
            for error in self.result.errors:
                print(f"  • {error.section}.{error.field}: {error.message}")
        
        # Print warnings
        if self.result.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.result.warnings)}):")
            for warning in self.result.warnings:
                print(f"  • {warning.section}.{warning.field}: {warning.message}")
        
        # Print infos
        if self.result.infos and len(self.result.infos) <= 10:
            print(f"\nℹ️  INFO ({len(self.result.infos)}):")
            for info in self.result.infos:
                print(f"  • {info.section}.{info.field}: {info.message}")
        elif len(self.result.infos) > 10:
            print(f"\nℹ️  INFO ({len(self.result.infos)} items - showing first 5):")
            for info in self.result.infos[:5]:
                print(f"  • {info.section}.{info.field}: {info.message}")
        
        # Recommendations
        if not self.result.is_valid:
            print(f"\n💡 RECOMMENDATIONS:")
            print("  1. Fix all errors before running the application")
            print("  2. Review warnings and apply if appropriate")
            print("  3. Use config.example.json as a reference")
            print("  4. Test configuration with --create-config first")
    
    def save_report(self, filename: str = None):
        """Save validation report to file"""
        if not filename:
            timestamp = Path(self.config_path).stem
            filename = f"validation_report_{timestamp}.json"
        
        report = {
            "config_file": str(self.config_path),
            "is_valid": self.result.is_valid,
            "summary": {
                "errors": len(self.result.errors),
                "warnings": len(self.result.warnings),
                "infos": len(self.result.infos)
            },
            "errors": [self.error_to_dict(error) for error in self.result.errors],
            "warnings": [self.error_to_dict(error) for error in self.result.warnings],
            "infos": [self.error_to_dict(error) for error in self.result.infos]
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n📄 Validation report saved to: {filename}")
        except Exception as e:
            print(f"\n❌ Failed to save validation report: {e}")
    
    def error_to_dict(self, error: ValidationError) -> Dict[str, str]:
        """Convert error to dictionary"""
        return {
            "section": error.section,
            "field": error.field,
            "message": error.message,
            "severity": error.severity
        }

def main():
    """Main validation runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Day Trading Orchestrator Configuration Validator")
    parser.add_argument("--config", default="config.json", help="Configuration file path")
    parser.add_argument("--output", help="Output file for validation report")
    parser.add_argument("--errors-only", action="store_true", help="Show only errors")
    parser.add_argument("--warnings-only", action="store_true", help="show only warnings")
    
    args = parser.parse_args()
    
    # Validate configuration
    validator = ConfigurationValidator(args.config)
    result = validator.validate()
    
    # Save report if requested
    if args.output:
        validator.save_report(args.output)
    
    # Exit with appropriate code
    exit_code = 0 if result.is_valid else 1
    sys.exit(exit_code)

if __name__ == "__main__":
    main()