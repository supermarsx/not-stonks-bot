"""
Configuration Migration System
Tools for migrating between configuration versions and environments
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum


class MigrationType(str, Enum):
    """Types of configuration migrations"""
    VERSION_UPGRADE = "version_upgrade"
    ENVIRONMENT_CHANGE = "environment_change"
    BROKER_MIGRATION = "broker_migration"
    SECURITY_UPDATE = "security_update"
    TEMPLATE_APPLICATION = "template_application"


@dataclass
class MigrationRule:
    """Individual migration rule"""
    from_version: str
    to_version: str
    migration_type: MigrationType
    description: str
    apply_function: Callable
    validation_function: Callable = None
    rollback_function: Callable = None
    required_backup: bool = True


@dataclass
class MigrationResult:
    """Migration execution result"""
    success: bool
    migration_type: MigrationType
    from_version: str
    to_version: str
    timestamp: datetime
    changes_applied: List[str]
    errors: List[str]
    warnings: List[str]
    backup_created: bool = False


class ConfigMigrationManager:
    """
    Configuration Migration Manager
    
    Provides tools for migrating configurations between versions and environments:
    - Version upgrades
    - Environment-specific migrations
    - Broker configuration migrations
    - Security updates
    - Template applications
    - Rollback capabilities
    """
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Migration rules registry
        self.migration_rules: List[MigrationRule] = []
        self._register_default_migrations()
    
    def migrate_config(self, config_name: str, target_version: str, 
                      environment: str = None, dry_run: bool = False) -> MigrationResult:
        """
        Migrate configuration to target version
        
        Args:
            config_name: Configuration to migrate
            target_version: Target version
            environment: Target environment
            dry_run: If True, only validate migration without applying changes
            
        Returns:
            MigrationResult with execution details
        """
        try:
            current_config = self.config_manager.get_config(config_name)
            if not current_config:
                return MigrationResult(
                    success=False,
                    migration_type=MigrationType.VERSION_UPGRADE,
                    from_version="unknown",
                    to_version=target_version,
                    timestamp=datetime.utcnow(),
                    changes_applied=[],
                    errors=[f"Configuration not found: {config_name}"]
                )
            
            # Get current version
            current_version = current_config.get("version", "1.0.0")
            
            # Find applicable migration rules
            applicable_rules = self._find_migration_rules(current_version, target_version, environment)
            
            if not applicable_rules:
                return MigrationResult(
                    success=False,
                    migration_type=MigrationType.VERSION_UPGRADE,
                    from_version=current_version,
                    to_version=target_version,
                    timestamp=datetime.utcnow(),
                    changes_applied=[],
                    errors=[f"No migration path found from {current_version} to {target_version}"]
                )
            
            # Execute migrations
            changes_applied = []
            errors = []
            warnings = []
            backup_created = False
            
            if not dry_run:
                # Create backup if required
                for rule in applicable_rules:
                    if rule.required_backup and not backup_created:
                        backup_created = self.config_manager.version_manager.create_backup(
                            config_name, f"pre_migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        )
            
            # Apply migration rules
            for rule in applicable_rules:
                try:
                    if dry_run:
                        # Validate migration without applying
                        if rule.validation_function:
                            validation_result = rule.validation_function(current_config)
                            if not validation_result[0]:
                                errors.extend(validation_result[1])
                            else:
                                warnings.extend(validation_result[1] if len(validation_result) > 1 else [])
                    else:
                        # Apply migration
                        result = rule.apply_function(current_config, environment)
                        if result[0]:
                            changes_applied.append(f"{rule.description}")
                            if len(result) > 1 and result[1]:
                                warnings.extend(result[1])
                        else:
                            errors.extend(result[1])
                            
                except Exception as e:
                    errors.append(f"Migration rule failed: {rule.description} - {str(e)}")
            
            # Update version if migration successful
            if not errors and not dry_run:
                current_config["version"] = target_version
                current_config["migration_date"] = datetime.utcnow().isoformat()
                self.config_manager.save_config(config_name, current_config)
            
            return MigrationResult(
                success=len(errors) == 0,
                migration_type=MigrationType.VERSION_UPGRADE,
                from_version=current_version,
                to_version=target_version,
                timestamp=datetime.utcnow(),
                changes_applied=changes_applied,
                errors=errors,
                warnings=warnings,
                backup_created=backup_created
            )
            
        except Exception as e:
            return MigrationResult(
                success=False,
                migration_type=MigrationType.VERSION_UPGRADE,
                from_version="unknown",
                to_version=target_version,
                timestamp=datetime.utcnow(),
                changes_applied=[],
                errors=[f"Migration failed: {str(e)}"]
            )
    
    def apply_environment_template(self, config_name: str, environment: str, 
                                  variables: Dict[str, Any] = None,
                                  dry_run: bool = False) -> MigrationResult:
        """
        Apply environment-specific template to configuration
        
        Args:
            config_name: Configuration name
            environment: Target environment
            variables: Template variables
            dry_run: If True, only validate without applying
            
        Returns:
            MigrationResult with execution details
        """
        try:
            # Get template for environment
            template = self.config_manager.templates.get(environment)
            if not template:
                return MigrationResult(
                    success=False,
                    migration_type=MigrationType.TEMPLATE_APPLICATION,
                    from_version="unknown",
                    to_version=environment,
                    timestamp=datetime.utcnow(),
                    changes_applied=[],
                    errors=[f"Template not found for environment: {environment}"]
                )
            
            current_config = self.config_manager.get_config(config_name)
            original_config = current_config.copy() if current_config else {}
            
            # Apply template variables
            template_config = self.config_manager._substitute_template_variables(
                template.config_data, variables or template.variables
            )
            
            # Merge configurations
            merged_config = self._merge_configurations(original_config, template_config)
            
            # Validate merged configuration
            validation_errors = self.config_manager.validator.validate(merged_config, config_name)
            
            changes_applied = []
            if not validation_errors:
                changes_applied.append(f"Applied {environment} template")
                
                # Add metadata
                merged_config["template_applied"] = environment
                merged_config["template_variables"] = variables or {}
                merged_config["template_date"] = datetime.utcnow().isoformat()
                
                if not dry_run:
                    self.config_manager.save_config(config_name, merged_config)
            else:
                validation_errors = [f"Validation error: {error}" for error in validation_errors]
            
            return MigrationResult(
                success=len(validation_errors) == 0,
                migration_type=MigrationType.TEMPLATE_APPLICATION,
                from_version=original_config.get("version", "unknown"),
                to_version=environment,
                timestamp=datetime.utcnow(),
                changes_applied=changes_applied,
                errors=validation_errors,
                warnings=[],
                backup_created=False
            )
            
        except Exception as e:
            return MigrationResult(
                success=False,
                migration_type=MigrationType.TEMPLATE_APPLICATION,
                from_version="unknown",
                to_version=environment,
                timestamp=datetime.utcnow(),
                changes_applied=[],
                errors=[f"Template application failed: {str(e)}"]
            )
    
    def migrate_broker_config(self, config_name: str, from_broker: str, 
                             to_broker: str, dry_run: bool = False) -> MigrationResult:
        """
        Migrate broker configuration between brokers
        
        Args:
            config_name: Configuration name
            from_broker: Source broker
            to_broker: Target broker
            dry_run: If True, only validate without applying
            
        Returns:
            MigrationResult with execution details
        """
        try:
            current_config = self.config_manager.get_config(config_name)
            if not current_config:
                return MigrationResult(
                    success=False,
                    migration_type=MigrationType.BROKER_MIGRATION,
                    from_version="unknown",
                    to_version="unknown",
                    timestamp=datetime.utcnow(),
                    changes_applied=[],
                    errors=[f"Configuration not found: {config_name}"]
                )
            
            # Check if source broker exists
            if f"brokers.{from_broker}" not in self.config_manager.config_metadata:
                return MigrationResult(
                    success=False,
                    migration_type=MigrationType.BROKER_MIGRATION,
                    from_version=from_broker,
                    to_version=to_broker,
                    timestamp=datetime.utcnow(),
                    changes_applied=[],
                    errors=[f"Source broker configuration not found: {from_broker}"]
                )
            
            # Get broker migration rules
            broker_rules = self._get_broker_migration_rules(from_broker, to_broker)
            
            # Apply migration
            changes_applied = []
            new_config = current_config.copy()
            
            for field_mapping in broker_rules:
                old_path = field_mapping["from"]
                new_path = field_mapping["to"]
                
                # Get old value
                old_value = self.config_manager.get_config_value(config_name, old_path)
                if old_value is not None:
                    # Set new value
                    self.config_manager.set_config_value(config_name, new_path, old_value)
                    changes_applied.append(f"Migrated {old_path} -> {new_path}")
            
            # Update broker name
            if "brokers" in new_config and from_broker in new_config["brokers"]:
                new_config["brokers"][to_broker] = new_config["brokers"].pop(from_broker)
                changes_applied.append(f"Renamed broker: {from_broker} -> {to_broker}")
            
            if not dry_run:
                self.config_manager.save_config(config_name, new_config)
            
            return MigrationResult(
                success=True,
                migration_type=MigrationType.BROKER_MIGRATION,
                from_version=from_broker,
                to_version=to_broker,
                timestamp=datetime.utcnow(),
                changes_applied=changes_applied,
                errors=[],
                warnings=[]
            )
            
        except Exception as e:
            return MigrationResult(
                success=False,
                migration_type=MigrationType.BROKER_MIGRATION,
                from_version=from_broker,
                to_version=to_broker,
                timestamp=datetime.utcnow(),
                changes_applied=[],
                errors=[f"Broker migration failed: {str(e)}"]
            )
    
    def apply_security_updates(self, config_name: str, security_level: str = "standard",
                              dry_run: bool = False) -> MigrationResult:
        """
        Apply security updates to configuration
        
        Args:
            config_name: Configuration name
            security_level: Security level (basic, standard, high, ultra)
            dry_run: If True, only validate without applying
            
        Returns:
            MigrationResult with execution details
        """
        try:
            current_config = self.config_manager.get_config(config_name)
            if not current_config:
                return MigrationResult(
                    success=False,
                    migration_type=MigrationType.SECURITY_UPDATE,
                    from_version="unknown",
                    to_version="unknown",
                    timestamp=datetime.utcnow(),
                    changes_applied=[],
                    errors=[f"Configuration not found: {config_name}"]
                )
            
            # Get security updates for level
            security_updates = self._get_security_updates(security_level)
            
            changes_applied = []
            new_config = current_config.copy()
            
            for update in security_updates:
                path = update["path"]
                new_value = update["value"]
                description = update["description"]
                
                # Apply update
                old_value = self.config_manager.get_config_value(config_name, path)
                self.config_manager.set_config_value(config_name, path, new_value)
                changes_applied.append(f"{description}: {old_value} -> {new_value}")
            
            if not dry_run:
                # Encrypt sensitive data if needed
                if security_level in ["high", "ultra"]:
                    sensitive_keys = self.config_manager.encryption._detect_sensitive_keys(new_config)
                    if sensitive_keys:
                        new_config = self.config_manager.encryption.encrypt_config_section(
                            new_config, sensitive_keys
                        )
                        changes_applied.append(f"Encrypted {len(sensitive_keys)} sensitive fields")
                
                self.config_manager.save_config(config_name, new_config)
            
            return MigrationResult(
                success=True,
                migration_type=MigrationType.SECURITY_UPDATE,
                from_version=current_config.get("version", "unknown"),
                to_version=security_level,
                timestamp=datetime.utcnow(),
                changes_applied=changes_applied,
                errors=[],
                warnings=[]
            )
            
        except Exception as e:
            return MigrationResult(
                success=False,
                migration_type=MigrationType.SECURITY_UPDATE,
                from_version="unknown",
                to_version=security_level,
                timestamp=datetime.utcnow(),
                changes_applied=[],
                errors=[f"Security update failed: {str(e)}"]
            )
    
    def rollback_migration(self, config_name: str, migration_id: str) -> bool:
        """Rollback a previous migration"""
        try:
            # This would need to be implemented with proper migration tracking
            # For now, use the version manager rollback functionality
            versions = self.config_manager.version_manager.get_versions(config_name)
            
            # Find version with migration_id in metadata
            target_version = None
            for version in versions:
                if migration_id in version.metadata.get("migration_ids", []):
                    target_version = version
                    break
            
            if target_version:
                return self.config_manager.version_manager.rollback(
                    config_name, target_version.version_id
                )
            
            return False
            
        except Exception as e:
            self.logger.error(f"Migration rollback failed: {e}")
            return False
    
    def get_migration_history(self, config_name: str = None) -> List[Dict[str, Any]]:
        """Get migration history for configurations"""
        history = []
        
        # Get version history with migration info
        versions = self.config_manager.version_manager.get_versions(config_name)
        
        for version in versions:
            if "migration" in version.metadata:
                history.append({
                    "config_name": config_name,
                    "version": version.version_id,
                    "timestamp": version.timestamp.isoformat(),
                    "migration_type": version.metadata["migration"]["type"],
                    "description": version.metadata["migration"]["description"],
                    "changes": version.metadata["migration"]["changes"]
                })
        
        return sorted(history, key=lambda x: x["timestamp"], reverse=True)
    
    def validate_migration(self, config_name: str, target_version: str) -> Tuple[bool, List[str]]:
        """Validate migration without applying changes"""
        result = self.migrate_config(config_name, target_version, dry_run=True)
        return result.success, result.errors + result.warnings
    
    def _merge_configurations(self, base_config: Dict[str, Any], 
                            template_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge template configuration with base configuration"""
        merged = base_config.copy()
        
        def merge_recursive(base, template):
            for key, value in template.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge_recursive(base[key], value)
                else:
                    base[key] = value
        
        merge_recursive(merged, template_config)
        return merged
    
    def _find_migration_rules(self, from_version: str, to_version: str, 
                             environment: str = None) -> List[MigrationRule]:
        """Find applicable migration rules"""
        applicable_rules = []
        
        for rule in self.migration_rules:
            # Check version compatibility
            if rule.from_version == from_version and rule.to_version == to_version:
                # Check environment if specified
                if environment is None or rule.migration_type == MigrationType.TEMPLATE_APPLICATION:
                    applicable_rules.append(rule)
        
        # Sort by migration type priority
        type_priority = {
            MigrationType.VERSION_UPGRADE: 1,
            MigrationType.ENVIRONMENT_CHANGE: 2,
            MigrationType.BROKER_MIGRATION: 3,
            MigrationType.SECURITY_UPDATE: 4,
            MigrationType.TEMPLATE_APPLICATION: 5
        }
        
        applicable_rules.sort(key=lambda r: type_priority.get(r.migration_type, 999))
        return applicable_rules
    
    def _get_broker_migration_rules(self, from_broker: str, to_broker: str) -> List[Dict[str, str]]:
        """Get field mapping rules for broker migration"""
        # Common broker field mappings
        mappings = {
            ("binance", "alpaca"): [
                {"from": "brokers.binance.api_key", "to": "brokers.alpaca.api_key"},
                {"from": "brokers.binance.api_secret", "to": "brokers.alpaca.api_secret"},
                {"from": "brokers.binance.testnet", "to": "brokers.alpaca.paper"},
            ],
            ("alpaca", "binance"): [
                {"from": "brokers.alpaca.api_key", "to": "brokers.binance.api_key"},
                {"from": "brokers.alpaca.api_secret", "to": "brokers.binance.api_secret"},
                {"from": "brokers.alpaca.paper", "to": "brokers.binance.testnet"},
            ],
            ("binance", "trading212"): [
                {"from": "brokers.binance.api_key", "to": "brokers.trading212.api_key"},
                {"from": "brokers.binance.testnet", "to": "brokers.trading212.practice"},
            ],
        }
        
        key = (from_broker.lower(), to_broker.lower())
        return mappings.get(key, [])
    
    def _get_security_updates(self, security_level: str) -> List[Dict[str, Any]]:
        """Get security updates for specified level"""
        updates = {
            "basic": [
                {
                    "path": "security.audit.enabled",
                    "value": True,
                    "description": "Enable audit logging"
                }
            ],
            "standard": [
                {
                    "path": "security.encryption.enabled",
                    "value": True,
                    "description": "Enable encryption"
                },
                {
                    "path": "security.audit.enabled",
                    "value": True,
                    "description": "Enable audit logging"
                },
                {
                    "path": "security.authentication.api_keys_required",
                    "value": True,
                    "description": "Require API keys"
                }
            ],
            "high": [
                {
                    "path": "security.encryption.enabled",
                    "value": True,
                    "description": "Enable encryption"
                },
                {
                    "path": "security.audit.enabled",
                    "value": True,
                    "description": "Enable audit logging"
                },
                {
                    "path": "security.authentication.two_factor_auth",
                    "value": True,
                    "description": "Enable 2FA"
                },
                {
                    "path": "security.network.ssl_required",
                    "value": True,
                    "description": "Require SSL"
                }
            ],
            "ultra": [
                {
                    "path": "security.encryption.enabled",
                    "value": True,
                    "description": "Enable encryption"
                },
                {
                    "path": "security.audit.enabled",
                    "value": True,
                    "description": "Enable audit logging"
                },
                {
                    "path": "security.authentication.two_factor_auth",
                    "value": True,
                    "description": "Enable 2FA"
                },
                {
                    "path": "security.network.ssl_required",
                    "value": True,
                    "description": "Require SSL"
                },
                {
                    "path": "security.authentication.ip_whitelist",
                    "value": True,
                    "description": "Enable IP whitelist"
                },
                {
                    "path": "security.audit.immutable_logs",
                    "value": True,
                    "description": "Immutable audit logs"
                }
            ]
        }
        
        return updates.get(security_level, [])
    
    def _register_default_migrations(self):
        """Register default migration rules"""
        # Version upgrade migrations
        self.migration_rules.append(
            MigrationRule(
                from_version="1.0.0",
                to_version="1.1.0",
                migration_type=MigrationType.VERSION_UPGRADE,
                description="Add new monitoring configuration",
                apply_function=self._upgrade_to_1_1_0,
                validation_function=self._validate_1_1_0
            )
        )
        
        self.migration_rules.append(
            MigrationRule(
                from_version="1.1.0",
                to_version="1.2.0",
                migration_type=MigrationType.VERSION_UPGRADE,
                description="Add advanced risk management",
                apply_function=self._upgrade_to_1_2_0,
                validation_function=self._validate_1_2_0
            )
        )
        
        # Environment change migrations
        self.migration_rules.append(
            MigrationRule(
                from_version="development",
                to_version="production",
                migration_type=MigrationType.ENVIRONMENT_CHANGE,
                description="Upgrade to production settings",
                apply_function=self._env_development_to_production,
                validation_function=self._validate_production_config
            )
        )
    
    # Migration implementation functions
    def _upgrade_to_1_1_0(self, config: Dict[str, Any], environment: str = None) -> Tuple[bool, List[str]]:
        """Upgrade configuration to version 1.1.0"""
        warnings = []
        
        # Add monitoring section if not present
        if "monitoring" not in config:
            config["monitoring"] = {
                "health_checks": {"enabled": True, "interval": 30},
                "metrics": {"enabled": True, "prometheus_port": 9090}
            }
            warnings.append("Added monitoring configuration")
        
        return True, warnings
    
    def _validate_1_1_0(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate configuration for version 1.1.0"""
        errors = []
        
        # Check for required monitoring fields
        if "monitoring" in config:
            if "health_checks" not in config["monitoring"]:
                errors.append("Missing monitoring.health_checks configuration")
            if "metrics" not in config["monitoring"]:
                errors.append("Missing monitoring.metrics configuration")
        
        return len(errors) == 0, errors
    
    def _upgrade_to_1_2_0(self, config: Dict[str, Any], environment: str = None) -> Tuple[bool, List[str]]:
        """Upgrade configuration to version 1.2.0"""
        warnings = []
        
        # Enhance risk management
        if "risk" in config:
            if "circuit_breakers" not in config["risk"]:
                config["risk"]["circuit_breakers"] = {
                    "enabled": True,
                    "daily_loss_limit": config["risk"].get("max_daily_loss", 1000) * 1.5,
                    "consecutive_loss_limit": 3,
                    "drawdown_limit": 0.15
                }
                warnings.append("Added circuit breaker configuration")
        
        return True, warnings
    
    def _validate_1_2_0(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate configuration for version 1.2.0"""
        errors = []
        
        # Check risk management requirements
        if "risk" in config:
            if "max_position_size" not in config["risk"]:
                errors.append("Missing risk.max_position_size")
            if "max_daily_loss" not in config["risk"]:
                errors.append("Missing risk.max_daily_loss")
        
        return len(errors) == 0, errors
    
    def _env_development_to_production(self, config: Dict[str, Any], environment: str = None) -> Tuple[bool, List[str]]:
        """Migrate development environment to production"""
        warnings = []
        
        # Production-specific changes
        changes = [
            ("debug", False),
            ("application.log_level", "WARNING"),
            ("database.echo", False),
            ("security.audit.enabled", True),
            ("monitoring.alerts.enabled", True)
        ]
        
        for change in changes:
            path = change[0]
            new_value = change[1]
            keys = path.split('.')
            
            # Navigate to parent
            current = config
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # Set value
            old_value = current.get(keys[-1])
            current[keys[-1]] = new_value
            warnings.append(f"Changed {path}: {old_value} -> {new_value}")
        
        return True, warnings
    
    def _validate_production_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate configuration for production environment"""
        errors = []
        
        # Production requirements
        if config.get("debug", False):
            errors.append("Debug mode should be disabled in production")
        
        if config.get("database", {}).get("echo", False):
            errors.append("Database echo should be disabled in production")
        
        # Check required security settings
        security = config.get("security", {})
        if not security.get("audit", {}).get("enabled", False):
            errors.append("Audit logging should be enabled in production")
        
        return len(errors) == 0, errors


# Migration utilities
def create_migration_script(source_config: str, target_config: str, 
                          output_path: str) -> bool:
    """Create a migration script between two configurations"""
    try:
        source = Path(source_config)
        target = Path(target_config)
        
        if not source.exists() or not target.exists():
            return False
        
        # Load configurations
        with open(source, 'r') as f:
            source_config_data = json.load(f)
        
        with open(target, 'r') as f:
            target_config_data = json.load(f)
        
        # Create migration script
        migration_script = f"""#!/usr/bin/env python3
\"\"\"
Configuration Migration Script
Migrates from {source.name} to {target.name}
Generated on {datetime.now().isoformat()}
\"\"\"

import json
from pathlib import Path

def migrate_config():
    \"\"\"Apply migration from source to target configuration\"\"\"
    
    # Load source configuration
    with open('{source}', 'r') as f:
        source_data = json.load(f)
    
    # Apply migration changes
    # TODO: Add specific migration logic here
    
    # Save migrated configuration
    with open('{target_config}', 'w') as f:
        json.dump(source_data, f, indent=2)
    
    print("Configuration migration completed")

if __name__ == "__main__":
    migrate_config()
"""
        
        # Write migration script
        with open(output_path, 'w') as f:
            f.write(migration_script)
        
        # Make executable
        os.chmod(output_path, 0o755)
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to create migration script: {e}")
        return False