# Comprehensive Configuration System Documentation

## Overview

The Trading Orchestrator Configuration System provides a complete, enterprise-grade solution for managing trading system configurations. It supports multiple environments, real-time updates, security, and comprehensive audit logging.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Installation and Setup](#installation-and-setup)
3. [Core Components](#core-components)
4. [Configuration Management](#configuration-management)
5. [Templates and Environments](#templates-and-environments)
6. [Security and Encryption](#security-and-encryption)
7. [Version Management](#version-management)
8. [Migration System](#migration-system)
9. [Admin Interface](#admin-interface)
10. [API Reference](#api-reference)
11. [Examples](#examples)
12. [Troubleshooting](#troubleshooting)

## System Architecture

### Core Components

```
Trading Orchestrator Configuration System
├── Configuration Manager (config_manager.py)
├── Validator (validator.py)
├── Version Manager (version_manager.py)
├── Encryption (encryption.py)
├── Audit Logger (audit_logger.py)
├── Migration System (migration.py)
├── Admin Interface (admin_interface.py)
└── Configuration Templates
    ├── development.json
    ├── production.json
    ├── hft.json
    ├── risk_focused.json
    └── config_templates.yaml
```

### Key Features

- **Centralized Configuration**: All configurations managed in one place
- **Hot-reloading**: Changes applied without system restart
- **Multi-format Support**: JSON, YAML, and environment variables
- **Environment-specific Templates**: Pre-configured settings for different deploys
- **Security**: Encryption of sensitive data, audit logging
- **Version Control**: Full version history with rollback capability
- **Real-time Monitoring**: Live configuration status and changes
- **Admin Interface**: Web-based configuration management

## Installation and Setup

### Requirements

```python
# Core dependencies
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0
pyyaml>=6.0
cryptography>=3.4.8
watchdog>=3.0.0

# Optional dependencies for specific features
redis>=4.0.0  # For caching
prometheus-client>=0.17.0  # For metrics
jinja2>=3.1.0  # For templates
```

### Basic Setup

```python
from trading_orchestrator.config.config_manager import get_config_manager
from trading_orchestrator.config.admin_interface import create_admin_interface

# Initialize configuration manager
config_manager = get_config_manager("./config")
success = config_manager.initialize()

if success:
    print("Configuration system initialized successfully")
else:
    print("Failed to initialize configuration system")
```

### Environment Setup

1. **Development Environment**
```bash
export TRADING_ENV=development
export BINANCE_API_KEY=your_dev_key
export ALPACA_API_KEY=your_dev_key
export OPENAI_API_KEY=your_openai_key
```

2. **Production Environment**
```bash
export TRADING_ENV=production
export DB_PASSWORD=secure_password
export ENCRYPTION_KEY_FILE=/secure/path/.encryption_key
export ALLOWED_IPS=192.168.1.0/24
```

## Core Components

### 1. Configuration Manager

The central hub for all configuration operations:

```python
from trading_orchestrator.config.config_manager import get_config_manager

# Get configuration manager instance
config_manager = get_config_manager("./config")

# Load configuration
success = config_manager.load_config("main", ConfigType.JSON)

# Get configuration value
api_port = config_manager.get_config_value("main", "application.api_port")

# Set configuration value
config_manager.set_config_value("main", "application.api_port", 8000)

# Save configuration
config_manager.save_config("main", config_data)
```

### 2. Configuration Validator

Ensures configurations meet requirements:

```python
from trading_orchestrator.config.validator import ConfigValidator

validator = ConfigValidator()

# Validate configuration
errors = validator.validate(config_data, "main")

if errors:
    print("Validation errors:", errors)
else:
    print("Configuration is valid")
```

### 3. Version Manager

Handles configuration versioning and rollback:

```python
from trading_orchestrator.config.version_manager import ConfigVersionManager

version_manager = ConfigVersionManager(config_dir)

# Create version
version_id = version_manager.create_version(
    "main", 
    config_data, 
    author="admin",
    description="Updated API settings"
)

# Get version history
versions = version_manager.get_versions("main")

# Rollback to previous version
success = version_manager.rollback("main", version_id)
```

### 4. Encryption System

Protects sensitive configuration data:

```python
from trading_orchestrator.config.encryption import ConfigEncryption

encryption = ConfigEncryption()

# Encrypt sensitive data
encrypted_value = encryption.encrypt("sensitive_api_key")

# Decrypt data
decrypted_value = encryption.decrypt(encrypted_value)

# Encrypt configuration section
secure_config = encryption.encrypt_config_section(
    config_data, 
    sensitive_keys=["api_key", "secret"]
)
```

### 5. Audit Logger

Tracks all configuration changes:

```python
from trading_orchestrator.config.audit_logger import get_audit_logger

audit_logger = get_audit_logger("./config/logs")

# Log configuration change
audit_logger.log_config_changed(
    "main", 
    "api_port", 
    old_value=8000, 
    new_value=8080
)

# Get audit events
events = audit_logger.get_events(
    config_name="main",
    start_time=datetime.now() - timedelta(days=1)
)
```

## Configuration Management

### Basic Configuration Structure

```json
{
  "application": {
    "name": "Trading Orchestrator",
    "version": "1.0.0",
    "environment": "development",
    "debug": true,
    "api_port": 8000
  },
  "database": {
    "type": "postgresql",
    "url": "postgresql://user:pass@localhost/db",
    "pool_size": 10
  },
  "brokers": {
    "binance": {
      "enabled": true,
      "api_key": "${BINANCE_API_KEY}",
      "testnet": true
    }
  },
  "risk": {
    "max_position_size": 10000.0,
    "max_daily_loss": 1000.0,
    "circuit_breakers": {
      "enabled": true,
      "daily_loss_limit": 1500.0
    }
  }
}
```

### Configuration Operations

```python
# Load configuration from file
config_manager.load_config("production", ConfigType.JSON)

# Create new configuration
new_config = {
    "version": "1.0.0",
    "environment": "development",
    "settings": {
        "api_port": 8000,
        "debug": True
    }
}
config_manager.save_config("development", new_config)

# Update specific value
config_manager.set_config_value("development", "settings.api_port", 8080)

# Get current value
api_port = config_manager.get_config_value("development", "settings.api_port")

# Validate configuration
errors = config_manager.validate_config("development")
if errors:
    print("Validation errors:", errors)

# Export configuration
config_manager.export_config("development", "./exports/dev_config.json")

# Import configuration
config_manager.import_config("./imports/prod_config.json", "production")
```

### Environment Variable Interpolation

Configuration files support environment variable substitution:

```json
{
  "database": {
    "url": "postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}"
  },
  "brokers": {
    "binance": {
      "api_key": "${BINANCE_API_KEY}",
      "api_secret": "${BINANCE_API_SECRET}"
    }
  }
}
```

## Templates and Environments

### Available Templates

1. **Development Template** (`development.json`)
   - Debug mode enabled
   - Testnet/paper trading
   - Detailed logging
   - Local database

2. **Production Template** (`production.json`)
   - Security hardened
   - Live trading enabled
   - Audit logging
   - Production database

3. **High-Frequency Trading Template** (`hft.json`)
   - Ultra-low latency settings
   - Performance optimized
   - Real-time monitoring
   - Co-location ready

4. **Risk-Focused Template** (`risk_focused.json`)
   - Conservative settings
   - Enhanced risk management
   - Stress testing enabled
   - Compliance reporting

### Using Templates

```python
# Apply template to create configuration
success = config_manager.apply_template("production")
if success:
    print("Production template applied")

# Apply template with custom variables
variables = {
    "api_port": 8443,
    "database_url": "postgresql://custom-db:5432/trading"
}
success = config_manager.apply_template("development", variables)
```

### Creating Custom Templates

```python
# Create custom template
custom_config = {
    "application": {
        "name": "Custom Trading System",
        "environment": "staging"
    },
    "settings": {
        "api_port": 8080,
        "features": ["real_time", "backtesting"]
    }
}

config_manager.create_template(
    "staging",
    environment=DeploymentEnvironment.STAGING,
    config_data=custom_config,
    description="Staging environment configuration",
    variables={"api_port": 8080}
)
```

## Security and Encryption

### Encryption Features

- **Fernet Symmetric Encryption**: For API keys and secrets
- **AES-256-CBC**: For high-security requirements
- **Password-Based Encryption**: For additional security layers
- **Key Rotation**: Automatic and manual key rotation
- **Secure Key Storage**: Encrypted key files with proper permissions

### Security Configuration

```python
# Enable encryption for configuration
config_manager.encryption.set_master_key(generated_key)

# Encrypt sensitive fields automatically
sensitive_config = {
    "api_key": "your_api_key",
    "secret": "your_secret",
    "database_password": "your_db_password"
}

encrypted_config = config_manager.encryption.encrypt_config_section(
    sensitive_config,
    sensitive_keys=["api_key", "secret", "password"]
)

# Result: sensitive fields are encrypted
# "api_key": "encrypted:..." 
```

### Security Best Practices

1. **Separate Sensitive Data**
   ```python
   # Store API keys separately
   config_manager.set_config_value("secrets", "binance_api_key", "your_key")
   
   # Reference in main config
   config_manager.set_config_value("main", "brokers.binance.api_key", "${binance_api_key}")
   ```

2. **Environment-Specific Secrets**
   ```python
   # Development: use test keys
   dev_secrets = {
       "binance_testnet_key": "test_key_123",
       "alpaca_paper_key": "paper_key_456"
   }
   
   # Production: use live keys
   prod_secrets = {
       "binance_live_key": "live_key_789",
       "alpaca_live_key": "production_key_012"
   }
   ```

3. **Key Rotation Schedule**
   ```python
   # Rotate encryption keys monthly
   if datetime.now().day == 1:
       config_manager.encryption.rotate_key()
   ```

## Version Management

### Version Control Features

- **Automatic Versioning**: Every save creates a new version
- **Manual Versioning**: Create versions with custom descriptions
- **Version Comparison**: See differences between versions
- **Rollback**: Restore to any previous version
- **Backup and Restore**: Full configuration backup/restore
- **Version Tagging**: Tag important versions

### Version Operations

```python
# Create manual version with description
version_id = version_manager.create_version(
    config_name="production",
    config_data=config_data,
    author="admin",
    description="Updated risk management settings",
    tags=["risk_update", "important"]
)

# Compare two versions
diff = version_manager.compare_versions(
    "production", 
    "version1_id", 
    "version2_id"
)

print("Changes:", diff["changes"])

# Export specific version
version_manager.export_version(
    config_name="production",
    version_id="version_id",
    export_path="./exports/prod_v1.2.0.json"
)

# Create backup
version_manager.create_backup(
    config_name="production",
    backup_name="pre_deployment_backup"
)
```

## Migration System

### Migration Types

1. **Version Upgrades**: Automatic migration between config versions
2. **Environment Changes**: Migrate between dev/staging/production
3. **Broker Migrations**: Change broker configurations
4. **Security Updates**: Apply security enhancements
5. **Template Applications**: Apply new templates

### Migration Operations

```python
from trading_orchestrator.config.migration import ConfigMigrationManager

migration_manager = ConfigMigrationManager(config_manager)

# Migrate to new version
result = migration_manager.migrate_config(
    config_name="main",
    target_version="1.2.0",
    dry_run=True  # Test without applying
)

if result.success:
    print("Changes to be applied:", result.changes_applied)
else:
    print("Migration errors:", result.errors)

# Apply migration
result = migration_manager.migrate_config(
    config_name="main",
    target_version="1.2.0"
)

# Migrate between brokers
result = migration_manager.migrate_broker_config(
    config_name="main",
    from_broker="binance",
    to_broker="alpaca"
)

# Apply security updates
result = migration_manager.apply_security_updates(
    config_name="production",
    security_level="high"
)
```

## Admin Interface

### Web-Based Management

The admin interface provides a comprehensive web-based GUI for configuration management:

```python
from trading_orchestrator.config.admin_interface import create_admin_interface

# Create admin interface
admin = create_admin_interface(
    config_dir="./config",
    port=8080,
    host="0.0.0.0",
    username="admin",
    password="secure_password"
)

# Run the interface
admin.run()
```

### Admin Interface Features

1. **Dashboard**: System overview and status
2. **Configuration Management**: View, edit, create, delete configs
3. **Template Management**: Apply and manage templates
4. **Migration Tools**: Execute configuration migrations
5. **Security Management**: Apply security updates
6. **Audit Logs**: View configuration change history
7. **Version Control**: Browse and rollback versions
8. **Real-time Monitoring**: Live configuration status

### Admin Interface Access

```bash
# Access the admin interface
http://localhost:8080

# Default credentials
Username: admin
Password: admin

# Change credentials
export ADMIN_USERNAME=new_username
export ADMIN_PASSWORD=new_password
```

## API Reference

### Configuration Manager API

```python
class ComprehensiveConfigManager:
    def initialize(self) -> bool
    def load_config(self, config_name: str, config_type: ConfigType = ConfigType.JSON) -> bool
    def save_config(self, config_name: str, config_data: Dict[str, Any], config_type: ConfigType = ConfigType.JSON) -> bool
    def get_config(self, config_name: str) -> Optional[Dict[str, Any]]
    def get_config_value(self, config_name: str, key_path: str, default: Any = None) -> Any
    def set_config_value(self, config_name: str, key_path: str, value: Any) -> bool
    def validate_config(self, config_name: str) -> List[str]
    def validate_all_configs(self) -> Dict[str, List[str]]
    def reload_config(self, config_name: str) -> bool
    def create_template(self, template_name: str, environment: DeploymentEnvironment, config_data: Dict[str, Any]) -> bool
    def apply_template(self, template_name: str, variables: Dict[str, Any] = None) -> bool
    def export_config(self, config_name: str, export_path: str, include_sensitive: bool = False) -> bool
    def import_config(self, import_path: str, config_name: str, overwrite: bool = False) -> bool
```

### REST API Endpoints

```
GET    /api/status              # System status
GET    /api/configs             # List configurations
GET    /api/config/{name}       # Get configuration
POST   /api/config              # Create configuration
PUT    /api/config/{name}       # Update configuration
DELETE /api/config/{name}       # Delete configuration
GET    /api/templates           # List templates
POST   /api/templates/apply     # Apply template
POST   /api/migrate             # Migrate configuration
POST   /api/security-update     # Apply security updates
POST   /api/export              # Export configuration
POST   /api/import              # Import configuration
POST   /api/validate/{name}     # Validate configuration
GET    /api/audit               # Get audit logs
GET    /api/versions/{name}     # Get version history
POST   /api/rollback/{name}/{version_id}  # Rollback to version
GET    /health                  # Health check
WS     /ws/monitor              # Real-time monitoring
```

## Examples

### Example 1: Basic Configuration Setup

```python
from trading_orchestrator.config.config_manager import get_config_manager, ConfigType
from trading_orchestrator.config.validator import ConfigValidator

# Initialize configuration system
config_manager = get_config_manager("./config")
if not config_manager.initialize():
    raise RuntimeError("Failed to initialize configuration system")

# Create development configuration
dev_config = {
    "version": "1.0.0",
    "environment": "development",
    "application": {
        "name": "Trading Orchestrator",
        "debug": True,
        "api_port": 8000
    },
    "database": {
        "type": "sqlite",
        "url": "sqlite:///./dev_trading.db"
    },
    "brokers": {
        "binance": {
            "enabled": True,
            "testnet": True,
            "api_key": "${BINANCE_API_KEY}",
            "api_secret": "${BINANCE_API_SECRET}"
        }
    },
    "risk": {
        "max_position_size": 1000.0,
        "max_daily_loss": 100.0,
        "circuit_breakers": {
            "enabled": True,
            "daily_loss_limit": 200.0
        }
    }
}

# Save configuration
success = config_manager.save_config("development", dev_config)
if not success:
    raise RuntimeError("Failed to save development configuration")

# Validate configuration
errors = config_manager.validate_config("development")
if errors:
    print("Configuration validation errors:", errors)

print("Development configuration setup completed")
```

### Example 2: Production Deployment

```python
# Apply production template
success = config_manager.apply_template("production")
if not success:
    raise RuntimeError("Failed to apply production template")

# Set environment-specific values
env_vars = {
    "DB_PASSWORD": os.getenv("PROD_DB_PASSWORD"),
    "BINANCE_API_KEY": os.getenv("PROD_BINANCE_KEY"),
    "ALPACA_API_KEY": os.getenv("PROD_ALPACA_KEY"),
    "ALLOWED_IPS": os.getenv("ALLOWED_IPS", "192.168.1.0/24")
}

# Apply environment variables
for key, value in env_vars.items():
    if value:
        config_manager.set_config_value("production", f"variables.{key}", value)

# Enable encryption for sensitive data
config_manager.encryption.rotate_key()

# Apply high security settings
result = config_manager.migration_manager.apply_security_updates(
    config_name="production",
    security_level="ultra"
)

if result.success:
    print("Security updates applied successfully")
    print("Changes:", result.changes_applied)
else:
    print("Security update errors:", result.errors)

# Create backup
backup_name = f"production_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
config_manager.version_manager.create_backup("production", backup_name)

print(f"Production configuration deployed with backup: {backup_name}")
```

### Example 3: High-Frequency Trading Setup

```python
# Apply HFT template
success = config_manager.apply_template("hft")
if not success:
    raise RuntimeError("Failed to apply HFT template")

# Configure for ultra-low latency
hft_optimizations = {
    "performance.async.max_workers": 100,
    "performance.database.pool_size": 100,
    "performance.caching.redis_host": "redis-cluster-hft",
    "broker.binance.timeout": 3,
    "broker.binance.low_latency_mode": True,
    "monitoring.real_time_monitoring.enabled": True
}

# Apply optimizations
for key_path, value in hft_optimizations.items():
    config_manager.set_config_value("hft", key_path, value)

# Configure HFT strategies
hft_strategies = {
    "strategies.enabled_strategies": [
        "market_making",
        "arbitrage",
        "momentum",
        "latency_arbitrage"
    ],
    "strategies.market_making.spread_target_bps": 2,
    "strategies.arbitrage.min_profit_threshold_bps": 1,
    "strategies.latency_arbitrage.latency_threshold_ms": 10
}

for key_path, value in hft_strategies.items():
    config_manager.set_config_value("hft", key_path, value)

# Enable real-time monitoring
result = config_manager.migration_manager.apply_security_updates(
    config_name="hft",
    security_level="high"
)

print("HFT configuration optimized for ultra-low latency trading")
```

### Example 4: Risk Management Configuration

```python
# Apply risk-focused template
success = config_manager.apply_template("risk_focused")
if not success:
    raise RuntimeError("Failed to apply risk-focused template")

# Configure conservative risk settings
risk_settings = {
    "risk.max_position_size": 10000.0,
    "risk.max_daily_loss": 1000.0,
    "risk.max_open_orders": 5,
    "risk.risk_per_trade": 0.01,
    "risk.circuit_breakers.daily_loss_limit": 1500.0,
    "risk.circuit_breakers.consecutive_loss_limit": 2,
    "risk.circuit_breakers.drawdown_limit": 0.05,
    "risk.stress_testing.enabled": True,
    "risk.stress_testing.monte_carlo_simulations": 10000
}

for key_path, value in risk_settings.items():
    config_manager.set_config_value("risk_focused", key_path, value)

# Configure conservative strategies
strategy_settings = {
    "strategies.enabled_strategies": [
        "low_risk_etf_trading",
        "dividend_arbitrage",
        "put_call_ratio",
        "vix_trading"
    ],
    "strategies.low_risk_etf_trading.etf_universe": ["SPY", "QQQ", "IWM", "VTI"],
    "strategies.dividend_arbitrage.min_dividend_yield": 0.03
}

for key_path, value in strategy_settings.items():
    config_manager.set_config_value("risk_focused", key_path, value)

# Enable compliance reporting
compliance_settings = {
    "compliance.enabled": True,
    "compliance.regulatory_framework": "SEC_FINRA_BASEL_III",
    "compliance.record_retention_days": 2555,
    "compliance.daily_reports": True,
    "compliance.monthly_reports": True
}

for key_path, value in compliance_settings.items():
    config_manager.set_config_value("risk_focused", key_path, value)

print("Risk-focused configuration optimized for conservative trading")
```

### Example 5: Configuration Migration

```python
# Test migration from development to production
result = config_manager.migration_manager.migrate_config(
    config_name="development",
    target_version="1.2.0",
    environment="production",
    dry_run=True
)

if result.success:
    print("Migration preview:")
    print("Changes to be applied:", result.changes_applied)
    print("Warnings:", result.warnings)
    
    # Apply migration if no errors
    if not result.errors:
        result = config_manager.migration_manager.migrate_config(
            config_name="development",
            target_version="1.2.0",
            environment="production"
        )
        
        if result.success:
            print("Migration completed successfully")
            print("Changes applied:", result.changes_applied)
        else:
            print("Migration failed:", result.errors)
else:
    print("Migration validation failed:", result.errors)

# Migrate broker configuration
result = config_manager.migration_manager.migrate_broker_config(
    config_name="development",
    from_broker="binance",
    to_broker="alpaca"
)

if result.success:
    print("Broker migration completed")
    print("Changes:", result.changes_applied)
else:
    print("Broker migration failed:", result.errors)
```

## Troubleshooting

### Common Issues

1. **Configuration Loading Errors**
   ```
   Error: Configuration file not found
   Solution: Check file path and permissions
   ```

2. **Validation Failures**
   ```
   Error: Field type mismatch
   Solution: Check data types in configuration
   ```

3. **Encryption Errors**
   ```
   Error: Decryption failed
   Solution: Verify encryption key and password
   ```

4. **Permission Errors**
   ```
   Error: Permission denied
   Solution: Check file/directory permissions
   ```

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Initialize with debug output
config_manager = get_config_manager("./config")
success = config_manager.initialize()

# Check configuration status
status = config_manager.get_config_status()
print("Configuration status:", status)
```

### Health Checks

```python
# Run system health check
health_data = {
    "config_manager": "running" if config_manager else "not initialized",
    "encryption": "available" if config_manager.encryption else "not available",
    "audit_logger": "running" if config_manager.audit_logger else "not running",
    "templates_loaded": len(config_manager.templates),
    "configs_loaded": len(config_manager.config_cache)
}

print("System health:", health_data)
```

### Log Analysis

```python
# Get recent audit events
recent_events = config_manager.audit_logger.get_events(
    start_time=datetime.now() - timedelta(hours=1)
)

# Analyze configuration changes
changes = config_manager.get_change_history(limit=10)
for change in changes:
    print(f"{change.timestamp}: {change.key_path} changed")
```

### Reset Configuration System

```python
# Reset to default state
config_manager.config_cache.clear()
config_manager.config_metadata.clear()
config_manager.change_history.clear()

# Reinitialize
success = config_manager.initialize()

# Load default configurations
for template_name in ["development", "production"]:
    config_manager.apply_template(template_name)
```

## Best Practices

### Security Best Practices

1. **Use Strong Passwords**
   ```python
   # Generate secure passwords
   import secrets
   secure_password = secrets.token_urlsafe(32)
   ```

2. **Enable Encryption**
   ```python
   # Always encrypt sensitive data
   config_manager.encryption.set_master_key(generated_key)
   ```

3. **Regular Key Rotation**
   ```python
   # Rotate keys monthly
   config_manager.encryption.rotate_key()
   ```

4. **Audit Logging**
   ```python
   # Enable comprehensive audit logging
   config_manager.audit_logger.log_system_error("test", "System error test")
   ```

### Performance Best Practices

1. **Use Appropriate Templates**
   ```python
   # Use HFT template for high-frequency trading
   config_manager.apply_template("hft")
   
   # Use risk-focused for conservative trading
   config_manager.apply_template("risk_focused")
   ```

2. **Enable Caching**
   ```python
   # Enable Redis caching for production
   config_manager.set_config_value("production", "performance.caching.enabled", True)
   ```

3. **Optimize Database Settings**
   ```python
   # Adjust connection pool for workload
   config_manager.set_config_value("production", "database.pool_size", 20)
   ```

### Operational Best Practices

1. **Regular Backups**
   ```python
   # Create backups before major changes
   backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
   config_manager.version_manager.create_backup("production", backup_name)
   ```

2. **Version Control**
   ```python
   # Tag important versions
   config_manager.version_manager.tag_version(
       "production", 
       "version_id", 
       ["v1.2.0", "production_ready"]
   )
   ```

3. **Change Management**
   ```python
   # Use dry-run for major changes
   result = migration_manager.migrate_config(
       config_name="production",
       target_version="1.3.0",
       dry_run=True
   )
   ```

4. **Monitoring**
   ```python
   # Monitor configuration changes
   config_manager.add_change_callback(
       lambda change: print(f"Config changed: {change.key_path}")
   )
   ```

This comprehensive configuration system provides enterprise-grade features for managing trading system configurations with security, reliability, and ease of use.