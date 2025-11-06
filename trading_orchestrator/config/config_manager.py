"""
Comprehensive Configuration Management System
Advanced centralized configuration manager with hot-reloading, validation, and encryption
"""

import os
import json
import yaml
import asyncio
import hashlib
import logging
from typing import Dict, Any, List, Optional, Union, Callable
from pathlib import Path
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import threading
from contextlib import contextmanager

from .validator import ConfigValidator
from .version_manager import ConfigVersionManager
from .encryption import ConfigEncryption
from .audit_logger import ConfigAuditLogger


class ConfigType(str, Enum):
    """Configuration file types"""
    JSON = "json"
    YAML = "yaml"
    ENV = "env"


class DeploymentEnvironment(str, Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    LOCAL = "local"
    CLOUD = "cloud"
    HFT = "hft"  # High-frequency trading
    RISK_FOCUSED = "risk_focused"


class ConfigStatus(str, Enum):
    """Configuration status"""
    VALID = "valid"
    INVALID = "invalid"
    CHANGED = "changed"
    LOADING = "loading"
    ERROR = "error"


@dataclass
class ConfigEntry:
    """Individual configuration entry"""
    path: str
    value: Any
    value_type: str
    encrypted: bool = False
    last_modified: datetime = field(default_factory=datetime.utcnow)
    version: str = "1.0.0"
    description: str = ""
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    sensitive: bool = False


@dataclass
class ConfigTemplate:
    """Configuration template"""
    name: str
    environment: DeploymentEnvironment
    description: str
    config_data: Dict[str, Any]
    variables: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    version: str = "1.0.0"


class ConfigChange:
    """Configuration change tracking"""
    def __init__(self, key_path: str, old_value: Any, new_value: Any, 
                 timestamp: datetime, user_id: str = "system"):
        self.key_path = key_path
        self.old_value = old_value
        self.new_value = new_value
        self.timestamp = timestamp
        self.user_id = user_id
        self.id = hashlib.md5(f"{key_path}{timestamp.isoformat()}".encode()).hexdigest()


class ConfigWatchHandler(FileSystemEventHandler):
    """File system event handler for configuration changes"""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        
    def on_modified(self, event):
        if event.is_directory:
            return
            
        if not self._is_config_file(event.src_path):
            return
            
        self.config_manager._reload_config_file(event.src_path)
    
    def _is_config_file(self, file_path: str) -> bool:
        """Check if file is a configuration file"""
        config_extensions = {'.json', '.yaml', '.yml', '.env'}
        return any(file_path.endswith(ext) for ext in config_extensions)


class ComprehensiveConfigManager:
    """
    Advanced Configuration Manager
    
    Features:
    - Centralized configuration management
    - Hot-reloading without restart
    - Environment variable interpolation
    - Configuration validation
    - Sensitive data encryption
    - Version control and rollback
    - Change audit logging
    - Multiple configuration formats
    - Template-based configurations
    - Real-time change notifications
    """
    
    def __init__(self, base_config_dir: str = "./config"):
        self.base_config_dir = Path(base_config_dir)
        self.base_config_dir.mkdir(parents=True, exist_ok=True)
        
        # Core components
        self.validator = ConfigValidator()
        self.version_manager = ConfigVersionManager(self.base_config_dir)
        self.encryption = ConfigEncryption()
        self.audit_logger = ConfigAuditLogger(self.base_config_dir / "logs")
        
        # Configuration storage
        self.config_cache: Dict[str, Any] = {}
        self.config_metadata: Dict[str, ConfigEntry] = {}
        self.config_status: Dict[str, ConfigStatus] = {}
        
        # Templates and profiles
        self.templates: Dict[str, ConfigTemplate] = {}
        self.active_template: Optional[str] = None
        
        # Change tracking
        self.change_history: List[ConfigChange] = []
        self.change_callbacks: List[Callable] = []
        
        # File watching
        self.observer = Observer()
        self.observer.schedule(
            ConfigWatchHandler(self), 
            str(self.base_config_dir), 
            recursive=True
        )
        self.observer.start()
        
        # Thread safety
        self.lock = threading.RLock()
        
        self.logger = logging.getLogger(__name__)
    
    def __del__(self):
        """Cleanup resources"""
        try:
            self.observer.stop()
            self.observer.join()
        except:
            pass
    
    def initialize(self) -> bool:
        """Initialize the configuration manager"""
        try:
            self.logger.info("Initializing comprehensive configuration manager...")
            
            # Load built-in templates
            self._load_builtin_templates()
            
            # Load environment configurations
            self._load_environment_configs()
            
            # Validate all configurations
            validation_results = self.validate_all_configs()
            
            # Start monitoring
            self._start_monitoring()
            
            self.logger.info("Configuration manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize configuration manager: {e}")
            return False
    
    def load_config(self, config_name: str, config_type: ConfigType = ConfigType.JSON) -> bool:
        """
        Load configuration from file
        
        Args:
            config_name: Configuration name (without extension)
            config_type: Configuration file type
            
        Returns:
            bool: True if loaded successfully
        """
        with self.lock:
            try:
                self.config_status[config_name] = ConfigStatus.LOADING
                
                config_file = self.base_config_dir / f"{config_name}.{config_type.value}"
                
                if not config_file.exists():
                    raise FileNotFoundError(f"Configuration file not found: {config_file}")
                
                # Load configuration data
                if config_type == ConfigType.JSON:
                    with open(config_file, 'r') as f:
                        config_data = json.load(f)
                elif config_type == ConfigType.YAML:
                    with open(config_file, 'r') as f:
                        config_data = yaml.safe_load(f)
                else:  # ENV
                    config_data = self._load_env_config(config_file)
                
                # Apply environment variable interpolation
                config_data = self._interpolate_env_vars(config_data)
                
                # Decrypt sensitive values
                config_data = self._decrypt_sensitive_values(config_data)
                
                # Validate configuration
                validation_errors = self.validator.validate(config_data, config_name)
                if validation_errors:
                    self.logger.error(f"Configuration validation failed for {config_name}: {validation_errors}")
                    self.config_status[config_name] = ConfigStatus.INVALID
                    return False
                
                # Store configuration
                self.config_cache[config_name] = config_data
                
                # Update metadata
                self._update_metadata(config_name, config_data)
                
                # Log audit event
                self.audit_logger.log_config_loaded(config_name, config_type, len(config_data))
                
                self.config_status[config_name] = ConfigStatus.VALID
                self.logger.info(f"Configuration loaded: {config_name}")
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to load configuration {config_name}: {e}")
                self.config_status[config_name] = ConfigStatus.ERROR
                return False
    
    def save_config(self, config_name: str, config_data: Dict[str, Any], 
                   config_type: ConfigType = ConfigType.JSON) -> bool:
        """
        Save configuration to file
        
        Args:
            config_name: Configuration name
            config_data: Configuration data
            config_type: Configuration file type
            
        Returns:
            bool: True if saved successfully
        """
        with self.lock:
            try:
                # Encrypt sensitive values before saving
                config_data = self._encrypt_sensitive_values(config_data)
                
                config_file = self.base_config_dir / f"{config_name}.{config_type.value}"
                
                # Create backup
                if config_file.exists():
                    backup_file = config_file.with_suffix(f".{config_type.value}.backup")
                    config_file.rename(backup_file)
                
                # Save configuration
                if config_type == ConfigType.JSON:
                    with open(config_file, 'w') as f:
                        json.dump(config_data, f, indent=2, default=str)
                elif config_type == ConfigType.YAML:
                    with open(config_file, 'w') as f:
                        yaml.dump(config_data, f, default_flow_style=False, indent=2)
                
                # Update cache
                self.config_cache[config_name] = config_data
                
                # Create version
                version_id = self.version_manager.create_version(config_name, config_data)
                
                # Log audit event
                self.audit_logger.log_config_saved(config_name, config_type, version_id)
                
                self.config_status[config_name] = ConfigStatus.VALID
                self.logger.info(f"Configuration saved: {config_name}")
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to save configuration {config_name}: {e}")
                self.config_status[config_name] = ConfigStatus.ERROR
                return False
    
    def get_config(self, config_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration from cache"""
        return self.config_cache.get(config_name)
    
    def get_config_value(self, config_name: str, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            config_name: Configuration name
            key_path: Dot-separated key path (e.g., "database.url")
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        config = self.config_cache.get(config_name)
        if not config:
            return default
        
        keys = key_path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set_config_value(self, config_name: str, key_path: str, value: Any) -> bool:
        """
        Set configuration value using dot notation
        
        Args:
            config_name: Configuration name
            key_path: Dot-separated key path
            value: New value
            
        Returns:
            bool: True if set successfully
        """
        with self.lock:
            try:
                config = self.config_cache.get(config_name)
                if not config:
                    config = {}
                    self.config_cache[config_name] = config
                
                # Get old value for change tracking
                old_value = self.get_config_value(config_name, key_path, None)
                
                # Set new value
                keys = key_path.split('.')
                current = config
                
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                
                current[keys[-1]] = value
                
                # Track change
                change = ConfigChange(key_path, old_value, value, datetime.utcnow())
                self.change_history.append(change)
                
                # Trigger callbacks
                asyncio.create_task(self._trigger_change_callbacks(config_name, change))
                
                # Log audit event
                self.audit_logger.log_config_changed(config_name, key_path, old_value, value)
                
                self.logger.info(f"Configuration value set: {config_name}.{key_path} = {value}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to set configuration value: {e}")
                return False
    
    def validate_config(self, config_name: str) -> List[str]:
        """Validate configuration and return error list"""
        config = self.config_cache.get(config_name)
        if not config:
            return [f"Configuration not loaded: {config_name}"]
        
        return self.validator.validate(config, config_name)
    
    def validate_all_configs(self) -> Dict[str, List[str]]:
        """Validate all loaded configurations"""
        results = {}
        for config_name in self.config_cache:
            results[config_name] = self.validate_config(config_name)
        return results
    
    def reload_config(self, config_name: str) -> bool:
        """Reload configuration from file"""
        return self.load_config(config_name)
    
    def reload_all_configs(self) -> Dict[str, bool]:
        """Reload all configurations"""
        results = {}
        for config_name in list(self.config_cache.keys()):
            results[config_name] = self.reload_config(config_name)
        return results
    
    def create_template(self, template_name: str, environment: DeploymentEnvironment,
                       config_data: Dict[str, Any], description: str = "",
                       variables: Dict[str, Any] = None) -> bool:
        """Create configuration template"""
        template = ConfigTemplate(
            name=template_name,
            environment=environment,
            description=description,
            config_data=config_data,
            variables=variables or {}
        )
        
        self.templates[template_name] = template
        
        # Save template to file
        template_file = self.base_config_dir / "templates" / f"{template_name}.json"
        template_file.parent.mkdir(exist_ok=True)
        
        with open(template_file, 'w') as f:
            json.dump({
                'name': template.name,
                'environment': template.environment.value,
                'description': template.description,
                'config_data': template.config_data,
                'variables': template.variables,
                'version': template.version
            }, f, indent=2)
        
        self.logger.info(f"Configuration template created: {template_name}")
        return True
    
    def apply_template(self, template_name: str, variables: Dict[str, Any] = None) -> bool:
        """Apply configuration template"""
        template = self.templates.get(template_name)
        if not template:
            self.logger.error(f"Template not found: {template_name}")
            return False
        
        try:
            # Substitute variables
            config_data = self._substitute_template_variables(template.config_data, variables or template.variables)
            
            # Apply configuration
            config_name = f"{template_name}_{template.environment.value}"
            return self.save_config(config_name, config_data)
            
        except Exception as e:
            self.logger.error(f"Failed to apply template {template_name}: {e}")
            return False
    
    def add_change_callback(self, callback: Callable[[ConfigChange], None]):
        """Add configuration change callback"""
        self.change_callbacks.append(callback)
    
    async def _trigger_change_callbacks(self, config_name: str, change: ConfigChange):
        """Trigger configuration change callbacks"""
        for callback in self.change_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(change)
                else:
                    callback(change)
            except Exception as e:
                self.logger.error(f"Configuration change callback error: {e}")
    
    def get_change_history(self, config_name: str = None, 
                          start_time: datetime = None, 
                          end_time: datetime = None) -> List[ConfigChange]:
        """Get configuration change history"""
        filtered_changes = self.change_history
        
        if config_name:
            filtered_changes = [c for c in filtered_changes if c.key_path.startswith(config_name)]
        
        if start_time:
            filtered_changes = [c for c in filtered_changes if c.timestamp >= start_time]
        
        if end_time:
            filtered_changes = [c for c in filtered_changes if c.timestamp <= end_time]
        
        return filtered_changes
    
    def export_config(self, config_name: str, export_path: str, 
                     include_sensitive: bool = False) -> bool:
        """Export configuration to file"""
        try:
            config_data = self.config_cache.get(config_name)
            if not config_data:
                return False
            
            # Remove sensitive data if not requested
            if not include_sensitive:
                config_data = self._sanitize_sensitive_data(config_data)
            
            # Save export
            with open(export_path, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
            
            self.logger.info(f"Configuration exported: {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export configuration: {e}")
            return False
    
    def import_config(self, import_path: str, config_name: str, 
                     overwrite: bool = False) -> bool:
        """Import configuration from file"""
        try:
            with open(import_path, 'r') as f:
                config_data = json.load(f)
            
            if not overwrite and config_name in self.config_cache:
                self.logger.error(f"Configuration already exists: {config_name}")
                return False
            
            return self.save_config(config_name, config_data)
            
        except Exception as e:
            self.logger.error(f"Failed to import configuration: {e}")
            return False
    
    def get_config_status(self, config_name: str = None) -> Dict[str, Any]:
        """Get configuration status"""
        if config_name:
            return {
                config_name: {
                    'status': self.config_status.get(config_name, ConfigStatus.ERROR).value,
                    'last_modified': self.config_metadata.get(config_name, {}).get('last_modified'),
                    'has_changes': any(c.key_path.startswith(config_name) for c in self.change_history[-10:])
                }
            }
        
        return {
            name: {
                'status': status.value,
                'metadata': {
                    'last_modified': self.config_metadata.get(name, {}).get('last_modified'),
                    'sensitive_count': len([m for m in self.config_metadata.values() if m.sensitive]),
                    'total_keys': len(self._flatten_config(self.config_cache.get(name, {})))
                },
                'recent_changes': len([c for c in self.change_history if c.key_path.startswith(name)]) > 0
            }
            for name, status in self.config_status.items()
        }
    
    def _interpolate_env_vars(self, config_data: Any) -> Any:
        """Interpolate environment variables in configuration"""
        if isinstance(config_data, dict):
            return {k: self._interpolate_env_vars(v) for k, v in config_data.items()}
        elif isinstance(config_data, list):
            return [self._interpolate_env_vars(item) for item in config_data]
        elif isinstance(config_data, str) and config_data.startswith("${") and config_data.endswith("}"):
            env_var = config_data[2:-1]
            default_value = None
            if "=" in env_var:
                env_var, default_value = env_var.split("=", 1)
            return os.getenv(env_var, default_value)
        return config_data
    
    def _encrypt_sensitive_values(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive configuration values"""
        sensitive_keys = ['api_key', 'secret', 'password', 'token', 'key', 'credential']
        
        def encrypt_value(value: Any) -> Any:
            if isinstance(value, dict):
                return {k: encrypt_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [encrypt_value(item) for item in value]
            elif isinstance(value, str) and any(key in value.lower() for key in sensitive_keys):
                if not value.startswith("encrypted:"):
                    encrypted = self.encryption.encrypt(value)
                    return f"encrypted:{encrypted}"
            return value
        
        return encrypt_value(config_data)
    
    def _decrypt_sensitive_values(self, config_data: Any) -> Any:
        """Decrypt sensitive configuration values"""
        def decrypt_value(value: Any) -> Any:
            if isinstance(value, dict):
                return {k: decrypt_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [decrypt_value(item) for item in value]
            elif isinstance(value, str) and value.startswith("encrypted:"):
                try:
                    encrypted_data = value[10:]  # Remove "encrypted:" prefix
                    return self.encryption.decrypt(encrypted_data)
                except:
                    return value  # Return original if decryption fails
            return value
        
        return decrypt_value(config_data)
    
    def _update_metadata(self, config_name: str, config_data: Dict[str, Any]):
        """Update configuration metadata"""
        def update_recursive(data: Any, path: str = ""):
            if isinstance(data, dict):
                for key, value in data.items():
                    full_path = f"{path}.{key}" if path else key
                    
                    if isinstance(value, (dict, list)):
                        update_recursive(value, full_path)
                    else:
                        # Check if value might be sensitive
                        sensitive_keys = ['api_key', 'secret', 'password', 'token']
                        is_sensitive = any(key in key.lower() for key in sensitive_keys)
                        
                        entry = ConfigEntry(
                            path=full_path,
                            value=value,
                            value_type=type(value).__name__,
                            encrypted=value.startswith("encrypted:") if isinstance(value, str) else False,
                            last_modified=datetime.utcnow(),
                            sensitive=is_sensitive
                        )
                        
                        self.config_metadata[f"{config_name}.{full_path}"] = entry
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    full_path = f"{path}[{i}]"
                    if isinstance(item, (dict, list)):
                        update_recursive(item, full_path)
                    else:
                        entry = ConfigEntry(
                            path=full_path,
                            value=item,
                            value_type=type(item).__name__,
                            last_modified=datetime.utcnow()
                        )
                        self.config_metadata[f"{config_name}.{full_path}"] = entry
        
        update_recursive(config_data)
    
    def _load_env_config(self, env_file: Path) -> Dict[str, str]:
        """Load environment variables from .env file"""
        config = {}
        if not env_file.exists():
            return config
        
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    config[key] = value.strip('"\'')
        
        return config
    
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
    
    def _sanitize_sensitive_data(self, config_data: Any) -> Any:
        """Remove sensitive data from configuration"""
        if isinstance(config_data, dict):
            sensitive_keys = {'api_key', 'secret', 'password', 'token', 'key', 'credential'}
            return {
                k: self._sanitize_sensitive_data(v) 
                for k, v in config_data.items()
                if k.lower() not in sensitive_keys
            }
        elif isinstance(config_data, list):
            return [self._sanitize_sensitive_data(item) for item in config_data]
        else:
            return config_data
    
    def _substitute_template_variables(self, config_data: Any, variables: Dict[str, Any]) -> Any:
        """Substitute template variables in configuration"""
        def substitute_value(value: Any) -> Any:
            if isinstance(value, dict):
                return {k: substitute_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [substitute_value(item) for item in value]
            elif isinstance(value, str) and value.startswith("{{") and value.endswith("}}"):
                var_name = value[2:-2].strip()
                return variables.get(var_name, value)
            return value
        
        return substitute_value(config_data)
    
    def _load_builtin_templates(self):
        """Load built-in configuration templates"""
        # This will be implemented in the templates section
        pass
    
    def _load_environment_configs(self):
        """Load environment-specific configurations"""
        env = os.getenv('TRADING_ENV', 'development').lower()
        
        # Load main config
        self.load_config('main', ConfigType.JSON)
        
        # Load environment-specific config
        env_config_files = [
            f'{env}.json',
            f'{env}_broker.json',
            f'{env}_risk.json'
        ]
        
        for config_file in env_config_files:
            config_name = config_file.replace('.json', '')
            if (self.base_config_dir / config_file).exists():
                self.load_config(config_name, ConfigType.JSON)
    
    def _start_monitoring(self):
        """Start configuration monitoring"""
        self.logger.info("Started configuration monitoring")
    
    def _reload_config_file(self, file_path: str):
        """Reload configuration file when changed"""
        config_name = Path(file_path).stem
        if config_name in self.config_cache:
            self.logger.info(f"Configuration file changed: {config_name}")
            self.reload_config(config_name)
    
    @contextmanager
    def config_transaction(self, config_name: str):
        """Context manager for configuration transactions"""
        original_config = self.config_cache.get(config_name, {}).copy()
        try:
            yield self
            # Transaction successful
            self.save_config(config_name, self.config_cache[config_name])
        except Exception as e:
            # Rollback on error
            self.config_cache[config_name] = original_config
            self.logger.error(f"Configuration transaction failed for {config_name}: {e}")
            raise
    
    def shutdown(self):
        """Shutdown configuration manager"""
        try:
            self.observer.stop()
            self.observer.join()
            self.logger.info("Configuration manager shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during configuration manager shutdown: {e}")


# Global configuration manager instance
_config_manager = None


def get_config_manager(base_config_dir: str = "./config") -> ComprehensiveConfigManager:
    """Get or create global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ComprehensiveConfigManager(base_config_dir)
    return _config_manager


async def get_config() -> Dict[str, Any]:
    """Get all configurations as async function"""
    manager = get_config_manager()
    return manager.config_cache