"""
Configuration System Initializer
Main entry point for the comprehensive configuration system
"""

import os
import sys
import logging
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
import argparse
import json

from .config_manager import ComprehensiveConfigManager, ConfigType, DeploymentEnvironment
from .validator import ConfigValidator
from .version_manager import ConfigVersionManager
from .migration import ConfigMigrationManager
from .admin_interface import ConfigAdminInterface, create_admin_interface
from .audit_logger import get_audit_logger


class ConfigurationSystem:
    """
    Main Configuration System Interface
    
    Provides a unified interface to all configuration system components:
    - Configuration management
    - Template application
    - Migration management
    - Admin interface
    - System initialization
    """
    
    def __init__(self, config_dir: str = "./config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.config_manager: Optional[ComprehensiveConfigManager] = None
        self.migration_manager: Optional[ConfigMigrationManager] = None
        self.admin_interface: Optional[ConfigAdminInterface] = None
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Load built-in templates
        self._load_builtin_templates()
    
    def initialize(self) -> bool:
        """Initialize the entire configuration system"""
        try:
            self.logger.info("Initializing configuration system...")
            
            # Initialize main configuration manager
            self.config_manager = ComprehensiveConfigManager(str(self.config_dir))
            
            if not self.config_manager.initialize():
                self.logger.error("Failed to initialize configuration manager")
                return False
            
            # Initialize migration manager
            self.migration_manager = ConfigMigrationManager(self.config_manager)
            
            # Load templates
            self._load_templates_from_files()
            
            self.logger.info("Configuration system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize configuration system: {e}")
            return False
    
    def setup_environment(self, environment: str, variables: Dict[str, Any] = None) -> bool:
        """Setup configuration for specific environment"""
        try:
            if not self.config_manager:
                self.logger.error("Configuration system not initialized")
                return False
            
            # Convert environment string to enum
            env_enum = DeploymentEnvironment(environment.lower())
            
            # Check if template exists
            template_name = environment
            if template_name not in self.config_manager.templates:
                # Try to load from template file
                template_file = self.config_dir / "templates" / f"{environment}.json"
                if template_file.exists():
                    self._load_template_from_file(template_file, environment)
            
            # Apply template
            if template_name in self.config_manager.templates:
                success = self.config_manager.apply_template(template_name, variables)
                if success:
                    self.logger.info(f"Applied {environment} template")
                    return True
                else:
                    self.logger.error(f"Failed to apply {environment} template")
                    return False
            else:
                self.logger.error(f"Template not found for environment: {environment}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to setup environment {environment}: {e}")
            return False
    
    def run_admin_interface(self, port: int = 8080, host: str = "127.0.0.1", 
                           username: str = "admin", password: str = "admin") -> None:
        """Run the web-based admin interface"""
        try:
            if not self.config_manager:
                self.logger.error("Configuration system not initialized")
                return
            
            self.admin_interface = create_admin_interface(
                config_dir=str(self.config_dir),
                port=port,
                host=host,
                username=username,
                password=password
            )
            
            self.logger.info(f"Starting admin interface on {host}:{port}")
            self.admin_interface.run()
            
        except Exception as e:
            self.logger.error(f"Failed to start admin interface: {e}")
            raise
    
    def create_config_from_template(self, template_name: str, config_name: str,
                                  variables: Dict[str, Any] = None) -> bool:
        """Create configuration from template"""
        try:
            if not self.config_manager:
                self.logger.error("Configuration system not initialized")
                return False
            
            # Apply template to create configuration
            success = self.config_manager.apply_template(template_name, variables)
            
            if success:
                # Rename config if needed
                if config_name != template_name:
                    template_config = self.config_manager.get_config(template_name)
                    if template_config:
                        self.config_manager.save_config(config_name, template_config)
                        
                        # Remove original template config
                        config_file = self.config_dir / f"{template_name}.json"
                        if config_file.exists():
                            config_file.unlink()
                
                self.logger.info(f"Created configuration '{config_name}' from template '{template_name}'")
                return True
            else:
                self.logger.error(f"Failed to create configuration from template '{template_name}'")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to create configuration from template: {e}")
            return False
    
    def validate_system_health(self) -> Dict[str, Any]:
        """Validate configuration system health"""
        health_status = {
            "status": "healthy",
            "timestamp": None,
            "components": {},
            "issues": []
        }
        
        try:
            from datetime import datetime
            health_status["timestamp"] = datetime.utcnow().isoformat()
            
            # Check configuration manager
            if self.config_manager:
                health_status["components"]["config_manager"] = {
                    "status": "initialized",
                    "configs_loaded": len(self.config_manager.config_cache),
                    "templates_loaded": len(self.config_manager.templates)
                }
            else:
                health_status["components"]["config_manager"] = {
                    "status": "not_initialized"
                }
                health_status["issues"].append("Configuration manager not initialized")
            
            # Check migration manager
            if self.migration_manager:
                health_status["components"]["migration_manager"] = {
                    "status": "initialized",
                    "rules_registered": len(self.migration_manager.migration_rules)
                }
            else:
                health_status["components"]["migration_manager"] = {
                    "status": "not_initialized"
                }
                health_status["issues"].append("Migration manager not initialized")
            
            # Check template files
            template_dir = self.config_dir / "templates"
            if template_dir.exists():
                template_files = list(template_dir.glob("*.json"))
                health_status["components"]["templates"] = {
                    "status": "available",
                    "template_files": len(template_files)
                }
            else:
                health_status["components"]["templates"] = {
                    "status": "missing"
                }
                health_status["issues"].append("Templates directory missing")
            
            # Check log directory
            log_dir = self.config_dir / "logs"
            if log_dir.exists():
                health_status["components"]["logging"] = {
                    "status": "available",
                    "log_dir": str(log_dir)
                }
            else:
                health_status["components"]["logging"] = {
                    "status": "missing"
                }
                health_status["issues"].append("Log directory missing")
            
            # Determine overall status
            if health_status["issues"]:
                health_status["status"] = "degraded" if len(health_status["issues"]) < 3 else "unhealthy"
            
            return health_status
            
        except Exception as e:
            health_status["status"] = "error"
            health_status["issues"].append(f"Health check error: {e}")
            return health_status
    
    def export_system_config(self, export_path: str) -> bool:
        """Export entire configuration system setup"""
        try:
            export_data = {
                "export_info": {
                    "timestamp": asyncio.get_event_loop().time(),
                    "config_system_version": "1.0.0",
                    "config_dir": str(self.config_dir)
                },
                "templates": {},
                "configurations": {},
                "system_health": self.validate_system_health()
            }
            
            # Export templates
            if self.config_manager:
                for name, template in self.config_manager.templates.items():
                    export_data["templates"][name] = {
                        "name": template.name,
                        "environment": template.environment.value,
                        "description": template.description,
                        "config_data": template.config_data,
                        "variables": template.variables,
                        "version": template.version
                    }
                
                # Export configurations
                for config_name, config_data in self.config_manager.config_cache.items():
                    export_data["configurations"][config_name] = config_data
            
            # Save export
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"System configuration exported to {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export system configuration: {e}")
            return False
    
    def _setup_logging(self):
        """Setup configuration system logging"""
        log_dir = self.config_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "config_system.log"),
                logging.StreamHandler()
            ]
        )
    
    def _load_builtin_templates(self):
        """Load built-in templates"""
        template_dir = self.config_dir / "templates"
        
        # Create template directory if it doesn't exist
        template_dir.mkdir(exist_ok=True)
        
        # Define built-in templates
        builtin_templates = [
            "development.json",
            "production.json",
            "hft.json",
            "risk_focused.json"
        ]
        
        # Check if templates exist, if not create them
        for template_file in builtin_templates:
            template_path = template_dir / template_file
            if not template_path.exists():
                self.logger.warning(f"Template file not found: {template_path}")
    
    def _load_templates_from_files(self):
        """Load templates from template files"""
        template_dir = self.config_dir / "templates"
        
        if not template_dir.exists():
            return
        
        # Load YAML templates config
        yaml_config_file = template_dir / "config_templates.yaml"
        if yaml_config_file.exists():
            try:
                import yaml
                with open(yaml_config_file, 'r') as f:
                    yaml_data = yaml.safe_load(f)
                
                # Process templates from YAML
                if "templates" in yaml_data:
                    for template_name, template_data in yaml_data["templates"].items():
                        self._create_template_from_yaml(template_name, template_data)
                        
            except Exception as e:
                self.logger.error(f"Failed to load YAML templates: {e}")
        
        # Load JSON template files
        for template_file in template_dir.glob("*.json"):
            if template_file.name != "config_templates.yaml":
                self._load_template_from_file(template_file, template_file.stem)
    
    def _load_template_from_file(self, template_file: Path, template_name: str):
        """Load individual template from JSON file"""
        try:
            with open(template_file, 'r') as f:
                template_data = json.load(f)
            
            # Create template in config manager
            environment = template_data.get("environment", "development")
            description = template_data.get("description", "")
            config_data = template_data.get("config", {})
            variables = template_data.get("variables", {})
            
            if self.config_manager:
                self.config_manager.templates[template_name] = type('Template', (), {
                    'name': template_name,
                    'environment': DeploymentEnvironment(environment),
                    'description': description,
                    'config_data': config_data,
                    'variables': variables,
                    'version': template_data.get("version", "1.0.0")
                })()
            
            self.logger.debug(f"Loaded template: {template_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to load template from {template_file}: {e}")
    
    def _create_template_from_yaml(self, template_name: str, template_data: Dict[str, Any]):
        """Create template from YAML data"""
        try:
            environment = template_data.get("environment", "development")
            description = template_data.get("description", "")
            config_data = template_data.get("config", {})
            variables = template_data.get("variables", {})
            
            if self.config_manager:
                self.config_manager.templates[template_name] = type('Template', (), {
                    'name': template_name,
                    'environment': DeploymentEnvironment(environment),
                    'description': description,
                    'config_data': config_data,
                    'variables': variables,
                    'version': template_data.get("version", "1.0.0")
                })()
            
            self.logger.debug(f"Created template from YAML: {template_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to create template from YAML {template_name}: {e}")


def main():
    """Main entry point for configuration system CLI"""
    parser = argparse.ArgumentParser(description="Trading Orchestrator Configuration System")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize configuration system")
    init_parser.add_argument("--config-dir", default="./config", help="Configuration directory")
    
    # Setup environment command
    setup_parser = subparsers.add_parser("setup", help="Setup environment configuration")
    setup_parser.add_argument("--config-dir", default="./config", help="Configuration directory")
    setup_parser.add_argument("--environment", required=True, choices=["development", "production", "staging", "local", "cloud", "hft", "risk_focused"])
    setup_parser.add_argument("--config-name", help="Configuration name (defaults to environment)")
    
    # Admin interface command
    admin_parser = subparsers.add_parser("admin", help="Run admin interface")
    admin_parser.add_argument("--config-dir", default="./config", help="Configuration directory")
    admin_parser.add_argument("--port", type=int, default=8080, help="Admin interface port")
    admin_parser.add_argument("--host", default="127.0.0.1", help="Admin interface host")
    admin_parser.add_argument("--username", default="admin", help="Admin username")
    admin_parser.add_argument("--password", default="admin", help="Admin password")
    
    # Health check command
    health_parser = subparsers.add_parser("health", help="Check system health")
    health_parser.add_argument("--config-dir", default="./config", help="Configuration directory")
    health_parser.add_argument("--export", help="Export health report to file")
    
    # Export system command
    export_parser = subparsers.add_parser("export", help="Export configuration system")
    export_parser.add_argument("--config-dir", default="./config", help="Configuration directory")
    export_parser.add_argument("--output", required=True, help="Output file path")
    
    # Create from template command
    create_parser = subparsers.add_parser("create", help="Create configuration from template")
    create_parser.add_argument("--config-dir", default="./config", help="Configuration directory")
    create_parser.add_argument("--template", required=True, help="Template name")
    create_parser.add_argument("--config-name", required=True, help="Configuration name")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        config_system = ConfigurationSystem(args.config_dir)
        
        if args.command == "init":
            success = config_system.initialize()
            if success:
                print("✅ Configuration system initialized successfully")
            else:
                print("❌ Failed to initialize configuration system")
                sys.exit(1)
        
        elif args.command == "setup":
            config_name = args.config_name or args.environment
            success = config_system.setup_environment(args.environment)
            if success:
                print(f"✅ Environment '{args.environment}' setup completed")
                print(f"Configuration saved as: {config_name}")
            else:
                print(f"❌ Failed to setup environment '{args.environment}'")
                sys.exit(1)
        
        elif args.command == "admin":
            config_system.initialize()
            config_system.run_admin_interface(
                port=args.port,
                host=args.host,
                username=args.username,
                password=args.password
            )
        
        elif args.command == "health":
            config_system.initialize()
            health = config_system.validate_system_health()
            
            print(f"System Health: {health['status'].upper()}")
            print(f"Timestamp: {health['timestamp']}")
            
            if health['components']:
                print("\nComponents:")
                for name, info in health['components'].items():
                    status = info.get('status', 'unknown')
                    print(f"  {name}: {status}")
            
            if health['issues']:
                print("\nIssues:")
                for issue in health['issues']:
                    print(f"  ⚠️  {issue}")
            
            if args.export:
                with open(args.export, 'w') as f:
                    json.dump(health, f, indent=2)
                print(f"\nHealth report exported to: {args.export}")
        
        elif args.command == "export":
            config_system.initialize()
            success = config_system.export_system_config(args.output)
            if success:
                print(f"✅ Configuration system exported to {args.output}")
            else:
                print(f"❌ Failed to export configuration system")
                sys.exit(1)
        
        elif args.command == "create":
            success = config_system.create_config_from_template(args.template, args.config_name)
            if success:
                print(f"✅ Configuration '{args.config_name}' created from template '{args.template}'")
            else:
                print(f"❌ Failed to create configuration")
                sys.exit(1)
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()