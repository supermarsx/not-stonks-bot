#!/usr/bin/env python3
"""
Example Usage of Comprehensive Configuration System
Demonstrates key features of the trading orchestrator configuration system
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the trading orchestrator to Python path
sys.path.insert(0, str(Path(__file__).parent))

from config import ConfigurationSystem
from config.config_manager import ConfigType, DeploymentEnvironment


async def example_basic_setup():
    """Example: Basic configuration system setup"""
    print("🔧 Example 1: Basic Configuration Setup")
    print("=" * 50)
    
    # Initialize configuration system
    config_system = ConfigurationSystem("./examples/config")
    
    if not config_system.initialize():
        print("❌ Failed to initialize configuration system")
        return False
    
    print("✅ Configuration system initialized")
    
    # Setup development environment
    print("\n📦 Setting up development environment...")
    success = config_system.setup_environment("development")
    if success:
        print("✅ Development environment setup completed")
    else:
        print("❌ Failed to setup development environment")
        return False
    
    # Check system health
    print("\n🏥 Checking system health...")
    health = config_system.validate_system_health()
    print(f"System status: {health['status']}")
    
    return True


async def example_template_usage():
    """Example: Using configuration templates"""
    print("\n\n🏗️ Example 2: Template Usage")
    print("=" * 50)
    
    config_system = ConfigurationSystem("./examples/config")
    config_system.initialize()
    
    # Create configuration from template
    print("📋 Creating configuration from production template...")
    success = config_system.create_config_from_template(
        template_name="production",
        config_name="my_production_config",
        variables={
            "api_port": 8443,
            "database_url": "postgresql://prod:5432/trading",
            "max_position_size": 50000.0
        }
    )
    
    if success:
        print("✅ Configuration created from template")
    else:
        print("❌ Failed to create configuration from template")
        return False
    
    # Create HFT configuration
    print("\n⚡ Creating HFT configuration...")
    success = config_system.create_config_from_template(
        template_name="hft",
        config_name="hft_trading",
        variables={
            "max_workers": 100,
            "database_pool_size": 50
        }
    )
    
    if success:
        print("✅ HFT configuration created")
    else:
        print("❌ Failed to create HFT configuration")
        return False
    
    return True


async def example_manual_configuration():
    """Example: Manual configuration management"""
    print("\n\n✏️ Example 3: Manual Configuration Management")
    print("=" * 50)
    
    config_system = ConfigurationSystem("./examples/config")
    config_system.initialize()
    
    config_manager = config_system.config_manager
    
    # Create custom configuration manually
    custom_config = {
        "version": "1.0.0",
        "environment": "custom",
        "application": {
            "name": "My Custom Trading System",
            "version": "2.0.0",
            "debug": False,
            "api_port": 9000
        },
        "database": {
            "type": "postgresql",
            "url": "postgresql://user:pass@localhost:5432/custom_db",
            "pool_size": 15
        },
        "brokers": {
            "binance": {
                "enabled": True,
                "testnet": False,
                "api_key": "${BINANCE_API_KEY}",
                "api_secret": "${BINANCE_API_SECRET}"
            },
            "alpaca": {
                "enabled": False
            }
        },
        "risk": {
            "max_position_size": 25000.0,
            "max_daily_loss": 2500.0,
            "max_open_orders": 25,
            "circuit_breakers": {
                "enabled": True,
                "daily_loss_limit": 3000.0,
                "consecutive_loss_limit": 3
            }
        },
        "custom_settings": {
            "theme": "dark",
            "notifications": True,
            "backup_frequency": "daily"
        }
    }
    
    # Save configuration
    print("💾 Saving custom configuration...")
    success = config_manager.save_config("custom_trading", custom_config)
    if success:
        print("✅ Custom configuration saved")
    else:
        print("❌ Failed to save custom configuration")
        return False
    
    # Update configuration values
    print("🔄 Updating configuration values...")
    
    config_manager.set_config_value("custom_trading", "application.api_port", 9001)
    config_manager.set_config_value("custom_trading", "application.debug", True)
    config_manager.set_config_value("custom_trading", "risk.max_position_size", 30000.0)
    
    print("✅ Configuration values updated")
    
    # Retrieve and display values
    print("\n📊 Current configuration values:")
    api_port = config_manager.get_config_value("custom_trading", "application.api_port")
    debug = config_manager.get_config_value("custom_trading", "application.debug")
    max_position = config_manager.get_config_value("custom_trading", "risk.max_position_size")
    
    print(f"  API Port: {api_port}")
    print(f"  Debug Mode: {debug}")
    print(f"  Max Position Size: ${max_position:,.2f}")
    
    # Validate configuration
    print("\n🔍 Validating configuration...")
    errors = config_manager.validate_config("custom_trading")
    if errors:
        print(f"❌ Validation errors found: {errors}")
        return False
    else:
        print("✅ Configuration is valid")
    
    return True


async def example_security_features():
    """Example: Security and encryption features"""
    print("\n\n🔒 Example 4: Security Features")
    print("=" * 50)
    
    config_system = ConfigurationSystem("./examples/config")
    config_system.initialize()
    
    config_manager = config_system.config_manager
    
    # Create configuration with sensitive data
    secure_config = {
        "version": "1.0.0",
        "database": {
            "url": "postgresql://user:password123@localhost:5432/trading",
            "password": "database_secret_password"
        },
        "brokers": {
            "binance": {
                "api_key": "sensitive_binance_api_key",
                "api_secret": "sensitive_binance_secret"
            },
            "alpaca": {
                "api_key": "sensitive_alpaca_key",
                "api_secret": "sensitive_alpaca_secret"
            }
        },
        "secrets": {
            "jwt_secret": "super_secret_jwt_key",
            "encryption_key": "very_secure_encryption_key"
        }
    }
    
    # Save configuration with encryption
    print("🔐 Encrypting sensitive configuration data...")
    
    # Manually encrypt sensitive fields
    encryption = config_manager.encryption
    
    encrypted_config = secure_config.copy()
    
    # Encrypt specific fields
    for path in [
        "database.password",
        "brokers.binance.api_key",
        "brokers.binance.api_secret",
        "brokers.alpaca.api_key",
        "brokers.alpaca.api_secret",
        "secrets.jwt_secret",
        "secrets.encryption_key"
    ]:
        value = config_manager.get_config_value("secure_config", path)
        if value:
            encrypted_value = f"encrypted:{encryption.encrypt(str(value))}"
            config_manager.set_config_value("secure_config", path, encrypted_value)
    
    # Save encrypted configuration
    config_manager.save_config("secure_config", encrypted_config)
    print("✅ Sensitive data encrypted and configuration saved")
    
    # Demonstrate decryption
    print("\n🔓 Decrypting configuration...")
    config = config_manager.get_config("secure_config")
    
    # Decrypt the configuration
    decrypted_config = encryption.decrypt_config_section(config)
    
    # Verify sensitive data is encrypted in storage but decrypted when accessed
    print("✅ Configuration can be securely accessed with proper decryption")
    
    # Show encryption info
    encryption_info = encryption.get_encryption_info()
    print(f"\n🛡️ Encryption Status:")
    print(f"  Method: {encryption_info['encryption_method']}")
    print(f"  Key Available: {encryption_info['key_available']}")
    
    return True


async def example_version_control():
    """Example: Version control and rollback"""
    print("\n\n📚 Example 5: Version Control")
    print("=" * 50)
    
    config_system = ConfigurationSystem("./examples/config")
    config_system.initialize()
    
    config_manager = config_system.config_manager
    version_manager = config_manager.version_manager
    
    # Create initial configuration
    initial_config = {
        "version": "1.0.0",
        "application": {
            "name": "Trading System",
            "api_port": 8000
        },
        "settings": {
            "debug": True,
            "logging_level": "DEBUG"
        }
    }
    
    print("📝 Creating initial configuration...")
    config_manager.save_config("versioned_config", initial_config)
    
    # Create first version
    print("🏷️ Creating first version...")
    version1_id = version_manager.create_version(
        "versioned_config",
        initial_config,
        author="admin",
        description="Initial configuration",
        tags=["initial", "v1.0.0"]
    )
    print(f"Version 1 ID: {version1_id}")
    
    # Make changes
    print("\n🔄 Making configuration changes...")
    config_manager.set_config_value("versioned_config", "application.api_port", 8080)
    config_manager.set_config_value("versioned_config", "application.debug", False)
    config_manager.set_config_value("versioned_config", "settings.logging_level", "INFO")
    
    updated_config = config_manager.get_config("versioned_config")
    
    # Create second version
    print("🏷️ Creating second version...")
    version2_id = version_manager.create_version(
        "versioned_config",
        updated_config,
        author="admin",
        description="Updated API settings and logging",
        tags=["update", "v1.1.0"]
    )
    print(f"Version 2 ID: {version2_id}")
    
    # Get version history
    print("\n📋 Version history:")
    versions = version_manager.get_versions("versioned_config")
    for i, version in enumerate(versions, 1):
        print(f"  {i}. {version.version_id} - {version.timestamp}")
        print(f"     Author: {version.author}")
        print(f"     Description: {version.description}")
        print(f"     Tags: {version.tags}")
    
    # Compare versions
    print("\n🔍 Comparing versions...")
    diff = version_manager.compare_versions("versioned_config", version1_id, version2_id)
    print(f"Changes between versions:")
    print(f"  Added: {len(diff['changes']['added'])} fields")
    print(f"  Removed: {len(diff['changes']['removed'])} fields")
    print(f"  Modified: {len(diff['changes']['modified'])} fields")
    
    for change in diff['changes']['modified']:
        print(f"    {change['path']}: {change['old_value']} → {change['new_value']}")
    
    return True


async def example_migration():
    """Example: Configuration migration"""
    print("\n\n🔄 Example 6: Configuration Migration")
    print("=" * 50)
    
    config_system = ConfigurationSystem("./examples/config")
    config_system.initialize()
    
    migration_manager = config_system.migration_manager
    
    # Test migration from development to production
    print("🧪 Testing migration (dry run)...")
    result = migration_manager.migrate_config(
        config_name="development",
        target_version="1.2.0",
        environment="production",
        dry_run=True
    )
    
    print(f"Migration result: {'✅ SUCCESS' if result.success else '❌ FAILED'}")
    if result.success:
        print("Changes to be applied:")
        for change in result.changes_applied:
            print(f"  • {change}")
        if result.warnings:
            print("Warnings:")
            for warning in result.warnings:
                print(f"  ⚠️  {warning}")
    else:
        print("Errors:")
        for error in result.errors:
            print(f"  ❌ {error}")
    
    # Apply security updates
    print("\n🛡️ Applying security updates...")
    result = migration_manager.apply_security_updates(
        config_name="production",
        security_level="high",
        dry_run=True
    )
    
    print(f"Security update result: {'✅ SUCCESS' if result.success else '❌ FAILED'}")
    if result.success:
        print("Security changes:")
        for change in result.changes_applied:
            print(f"  • {change}")
    
    return True


async def example_audit_logging():
    """Example: Audit logging and monitoring"""
    print("\n\n📊 Example 7: Audit Logging")
    print("=" * 50)
    
    config_system = ConfigurationSystem("./examples/config")
    config_system.initialize()
    
    config_manager = config_system.config_manager
    audit_logger = config_manager.audit_logger
    
    # Simulate some configuration changes
    print("📝 Simulating configuration changes...")
    
    config_manager.set_config_value("audit_test", "application.name", "Audit Test System")
    config_manager.set_config_value("audit_test", "application.api_port", 7777)
    config_manager.set_config_value("audit_test", "database.url", "sqlite:///audit_test.db")
    
    # Load and export configuration (to generate audit events)
    config_manager.load_config("audit_test")
    config_manager.export_config("audit_test", "./examples/audit_export.json")
    
    # Get audit events
    print("\n📋 Recent audit events:")
    events = audit_logger.get_events(limit=10)
    
    for event in events[-5:]:  # Show last 5 events
        print(f"  {event.timestamp}: {event.event_type.value}")
        print(f"    User: {event.user_id}")
        print(f"    Config: {event.config_name}")
        print(f"    Description: {event.description}")
    
    # Get audit statistics
    print("\n📈 Audit statistics:")
    stats = audit_logger.get_statistics(days=1)
    print(f"  Total events: {stats['totals']['total_events']}")
    print(f"  By severity:")
    for severity, count in stats['totals']['by_severity'].items():
        print(f"    {severity}: {count}")
    
    return True


async def example_health_check():
    """Example: System health monitoring"""
    print("\n\n💓 Example 8: Health Check")
    print("=" * 50)
    
    config_system = ConfigurationSystem("./examples/config")
    config_system.initialize()
    
    # Run comprehensive health check
    print("🏥 Running comprehensive health check...")
    health = config_system.validate_system_health()
    
    print(f"System Health Status: {health['status'].upper()}")
    print(f"Timestamp: {health['timestamp']}")
    
    print("\n🔧 Component Status:")
    for component, info in health['components'].items():
        status = info.get('status', 'unknown')
        status_icon = "✅" if status in ["initialized", "available"] else "❌"
        print(f"  {status_icon} {component}: {status}")
        
        # Show additional info
        for key, value in info.items():
            if key != 'status':
                print(f"    {key}: {value}")
    
    if health['issues']:
        print("\n⚠️ Issues Found:")
        for issue in health['issues']:
            print(f"  • {issue}")
    else:
        print("\n✅ No issues found")
    
    # Export health report
    health_report_path = "./examples/health_report.json"
    with open(health_report_path, 'w') as f:
        import json
        json.dump(health, f, indent=2, default=str)
    
    print(f"\n📄 Health report exported to: {health_report_path}")
    
    return True


async def main():
    """Run all configuration system examples"""
    print("🚀 Trading Orchestrator Configuration System Examples")
    print("=" * 60)
    
    # Create examples directory
    examples_dir = Path("./examples")
    examples_dir.mkdir(exist_ok=True)
    
    examples = [
        example_basic_setup,
        example_template_usage,
        example_manual_configuration,
        example_security_features,
        example_version_control,
        example_migration,
        example_audit_logging,
        example_health_check
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            success = await example_func()
            if not success:
                print(f"\n❌ Example {i} failed")
                break
        except Exception as e:
            print(f"\n❌ Example {i} encountered an error: {e}")
            break
        
        if i < len(examples):
            print(f"\n{'='*60}")
    
    print("\n🎉 All examples completed!")
    print(f"\n📁 Check the './examples/' directory for generated files")
    print("\nTo start the admin interface, run:")
    print("python -m config admin --config-dir ./examples/config")


if __name__ == "__main__":
    asyncio.run(main())