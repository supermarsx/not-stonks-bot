#!/usr/bin/env python3
"""
Quick Setup Script for Trading Orchestrator Configuration System
Sets up the configuration system with default templates and examples
"""

import os
import sys
import json
import shutil
from pathlib import Path
import argparse


def create_directory_structure(base_dir: Path):
    """Create configuration directory structure"""
    directories = [
        base_dir,
        base_dir / "templates",
        base_dir / "exports",
        base_dir / "backups",
        base_dir / "logs",
        base_dir / "versions",
        base_dir / "examples"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    return True


def copy_template_files(base_dir: Path):
    """Copy template files to the configuration directory"""
    templates_dir = Path(__file__).parent / "templates"
    
    if not templates_dir.exists():
        print("⚠️ Templates directory not found, skipping template file copy")
        return False
    
    target_templates_dir = base_dir / "templates"
    
    # Copy template files
    template_files = list(templates_dir.glob("*.json")) + list(templates_dir.glob("*.yaml"))
    
    for template_file in template_files:
        target_file = target_templates_dir / template_file.name
        shutil.copy2(template_file, target_file)
        print(f"📄 Copied template: {template_file.name}")
    
    return True


def create_initial_config(base_dir: Path):
    """Create initial configuration files"""
    main_config = {
        "version": "1.0.0",
        "environment": "development",
        "created_by": "setup_script",
        "created_at": "2025-11-06T05:37:51Z",
        "application": {
            "name": "Trading Orchestrator",
            "version": "1.0.0",
            "environment": "development",
            "debug": True,
            "api_host": "0.0.0.0",
            "api_port": 8000
        },
        "database": {
            "type": "sqlite",
            "url": "sqlite:///./trading_orchestrator.db",
            "echo": True,
            "pool_size": 5
        },
        "brokers": {
            "binance": {
                "enabled": False,
                "testnet": True,
                "api_key": "${BINANCE_API_KEY}",
                "api_secret": "${BINANCE_API_SECRET}"
            },
            "alpaca": {
                "enabled": False,
                "paper": True,
                "api_key": "${ALPACA_API_KEY}",
                "api_secret": "${ALPACA_API_SECRET}"
            }
        },
        "risk": {
            "max_position_size": 1000.0,
            "max_daily_loss": 100.0,
            "max_open_orders": 20,
            "circuit_breakers": {
                "enabled": True,
                "daily_loss_limit": 200.0
            }
        },
        "logging": {
            "level": "DEBUG",
            "file": "logs/trading_orchestrator.log"
        }
    }
    
    # Save main configuration
    main_config_file = base_dir / "main.json"
    with open(main_config_file, 'w') as f:
        json.dump(main_config, f, indent=2)
    
    print(f"✅ Created main configuration: {main_config_file}")
    
    # Create environment variables template
    env_template = """# Environment Variables Template
# Copy this file to .env and update with your actual values

# Database Configuration
DB_PASSWORD=your_database_password

# Broker API Keys
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret

ALPACA_API_KEY=your_alpaca_api_key
ALPACA_API_SECRET=your_alpaca_api_secret

TRADING212_API_KEY=your_trading212_api_key

# AI Configuration
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key

# Security Configuration
SECRET_KEY=your_secret_key_for_jwt
ENCRYPTION_KEY_FILE=./.encryption_key

# Network Configuration
ALLOWED_IPS=127.0.0.1,192.168.1.0/24

# Email Configuration (for alerts)
SMTP_SERVER=smtp.gmail.com
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_email_password

# Monitoring Configuration
PROMETHEUS_PORT=9090
"""
    
    env_file = base_dir / ".env.template"
    with open(env_file, 'w') as f:
        f.write(env_template)
    
    print(f"📄 Created environment template: {env_file}")
    
    return True


def create_readme(base_dir: Path):
    """Create README file for the configuration system"""
    readme_content = """# Trading Orchestrator Configuration System

This directory contains the comprehensive configuration system for the Trading Orchestrator.

## Quick Start

### 1. Set Environment Variables

Copy the environment template:
```bash
cp .env.template .env
```

Edit `.env` with your actual values:
```bash
# Required: Set your broker API keys
BINANCE_API_KEY=your_actual_binance_key
ALPACA_API_KEY=your_actual_alpaca_key

# Database password
DB_PASSWORD=your_database_password

# AI API keys
OPENAI_API_KEY=your_openai_key
```

### 2. Initialize Configuration System

```bash
python -m config init
```

### 3. Setup Environment

```bash
# For development
python -m config setup --environment development

# For production
python -m config setup --environment production

# For high-frequency trading
python -m config setup --environment hft

# For risk-focused trading
python -m config setup --environment risk_focused
```

### 4. Start Admin Interface

```bash
python -m config admin --host 0.0.0.0 --port 8080
```

Then access: http://localhost:8080 (username: admin, password: admin)

## Available Commands

### CLI Commands

```bash
# Initialize configuration system
python -m config init --config-dir ./config

# Setup environment
python -m config setup --environment development --config-name my_config

# Run admin interface
python -m config admin --config-dir ./config --port 8080

# Check system health
python -m config health --config-dir ./config

# Export configuration system
python -m config export --config-dir ./config --output system_export.json

# Create configuration from template
python -m config create --config-dir ./config --template production --config-name my_production
```

### Python API

```python
from config import ConfigurationSystem

# Initialize system
config_system = ConfigurationSystem("./config")
config_system.initialize()

# Setup environment
config_system.setup_environment("development")

# Create configuration from template
config_system.create_config_from_template("production", "my_prod_config")

# Check health
health = config_system.validate_system_health()
print(f"System health: {health['status']}")
```

## Configuration Templates

- **development.json**: Development environment with debug enabled
- **production.json**: Production environment with security hardening
- **hft.json**: High-frequency trading with ultra-low latency
- **risk_focused.json**: Conservative risk management
- **cloud.json**: Cloud deployment configuration

## Directory Structure

```
config/
├── main.json                 # Main configuration
├── .env.template            # Environment variables template
├── templates/               # Configuration templates
│   ├── development.json
│   ├── production.json
│   ├── hft.json
│   ├── risk_focused.json
│   └── config_templates.yaml
├── logs/                    # System logs
├── versions/                # Configuration version history
├── backups/                 # Configuration backups
├── exports/                 # Exported configurations
└── examples/                # Example configurations
```

## Features

### 🔧 Configuration Management
- Centralized configuration storage
- Hot-reloading without restart
- Multiple format support (JSON, YAML, ENV)
- Environment variable interpolation

### 🔒 Security
- Encryption of sensitive data
- Secure key storage and rotation
- Audit logging of all changes
- Access control and permissions

### 📚 Version Control
- Automatic version creation
- Manual version tagging
- Configuration rollback
- Version comparison and diff

### 🔄 Migration System
- Version upgrade migrations
- Environment migrations
- Broker configuration migrations
- Security update migrations

### 🖥️ Admin Interface
- Web-based configuration management
- Real-time monitoring
- Template management
- Audit log viewing

### 📊 Monitoring & Logging
- Comprehensive audit logging
- Configuration change tracking
- System health monitoring
- Error reporting and alerts

## Environment Variables

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `BINANCE_API_KEY` | Binance API key | `abcd1234...` |
| `ALPACA_API_KEY` | Alpaca API key | `AK...` |
| `DB_PASSWORD` | Database password | `secure_password` |
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` |

### Optional Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `TRADING212_API_KEY` | Trading 212 API key | `t212_...` |
| `ANTHROPIC_API_KEY` | Anthropic API key | `sk-ant-...` |
| `SECRET_KEY` | JWT secret key | `your_secret` |
| `SMTP_USERNAME` | Email for alerts | `user@domain.com` |

## Security Best Practices

1. **Never commit `.env` files to version control**
2. **Use strong, unique passwords for all accounts**
3. **Enable encryption for all sensitive data**
4. **Regularly rotate API keys and passwords**
5. **Use environment-specific configurations**
6. **Enable audit logging in production**
7. **Restrict admin interface access by IP**
8. **Regularly backup configurations**

## Troubleshooting

### Common Issues

1. **Configuration not loading**
   - Check file permissions
   - Verify JSON/YAML syntax
   - Check environment variable values

2. **Encryption errors**
   - Regenerate encryption key
   - Check key file permissions
   - Verify password compatibility

3. **Admin interface not accessible**
   - Check firewall settings
   - Verify port availability
   - Check authentication credentials

### Debug Mode

Enable debug logging:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
export TRADING_ENV=development
python -m config admin --config-dir ./config
```

### Health Check

```bash
python -m config health --config-dir ./config --export health_report.json
```

## Examples

See the `examples.py` file for comprehensive usage examples:

```bash
python config/examples.py
```

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the comprehensive documentation
3. Run health checks for system status
4. Check logs in the `logs/` directory

## Next Steps

1. Configure your broker API keys
2. Set up database connection
3. Configure AI services
4. Customize risk management settings
5. Test configuration with development environment
6. Deploy to production with security hardening
"""

    readme_file = base_dir / "README.md"
    with open(readme_file, 'w') as f:
        f.write(readme_content)
    
    print(f"📄 Created README: {readme_file}")
    
    return True


def create_gitignore(base_dir: Path):
    """Create .gitignore file"""
    gitignore_content = """# Environment files (contains sensitive data)
.env
*.env

# Encryption keys
.encryption_key
*.key

# Log files
logs/*.log
logs/*.log.*
audit.log
audit.log.*

# Backup files
backups/*
versions/*.backup

# Database files
*.db
*.sqlite
*.sqlite3

# Python cache
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# IDE files
.vscode/
.idea/
*.swp
*.swo
*~

# OS files
.DS_Store
Thumbs.db

# Configuration exports (may contain sensitive data)
exports/*
*.export

# Temporary files
tmp/
temp/
*.tmp
*.temp
"""

    gitignore_file = base_dir / ".gitignore"
    with open(gitignore_file, 'w') as f:
        f.write(gitignore_content)
    
    print(f"📄 Created .gitignore: {gitignore_file}")
    
    return True


def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Quick setup for Trading Orchestrator Configuration System")
    parser.add_argument("--config-dir", default="./config", help="Configuration directory path")
    parser.add_argument("--no-examples", action="store_true", help="Skip creating examples")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    
    args = parser.parse_args()
    
    base_dir = Path(args.config_dir)
    
    print("🚀 Trading Orchestrator Configuration System Setup")
    print("=" * 60)
    print(f"📁 Configuration directory: {base_dir.absolute()}")
    print()
    
    # Check if directory exists and has content
    if base_dir.exists() and any(base_dir.iterdir()):
        if not args.force:
            response = input("Directory exists and has content. Continue? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                print("Setup cancelled")
                return
        else:
            print("⚠️ Overwriting existing files")
    
    # Create directory structure
    print("📁 Creating directory structure...")
    create_directory_structure(base_dir)
    
    # Copy template files
    print("\n📋 Setting up templates...")
    copy_template_files(base_dir)
    
    # Create initial configuration
    print("\n⚙️ Creating initial configuration...")
    create_initial_config(base_dir)
    
    # Create README
    print("\n📚 Creating documentation...")
    create_readme(base_dir)
    
    # Create .gitignore
    print("\n🛡️ Creating security files...")
    create_gitignore(base_dir)
    
    # Copy examples if not disabled
    if not args.no_examples:
        print("\n📝 Setting up examples...")
        examples_src = Path(__file__).parent / "examples.py"
        examples_dst = base_dir / "examples.py"
        if examples_src.exists():
            shutil.copy2(examples_src, examples_dst)
            print(f"📄 Copied examples: {examples_dst}")
        else:
            print("⚠️ Examples file not found")
    
    print("\n" + "=" * 60)
    print("✅ Configuration system setup completed!")
    print("\n📋 Next Steps:")
    print(f"1. cd {base_dir}")
    print("2. cp .env.template .env")
    print("3. Edit .env with your API keys and settings")
    print("4. python -m config init")
    print("5. python -m config setup --environment development")
    print("6. python -m config admin")
    print("\n🌐 Then access the admin interface at http://localhost:8080")
    print("   Username: admin")
    print("   Password: admin")
    print("\n📖 See README.md for detailed documentation")


if __name__ == "__main__":
    main()