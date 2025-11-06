# Scripts Directory for not-stonks-bot

This directory contains utility scripts for development, deployment, and system maintenance.

## 📁 Directory Structure

```
scripts/
├── health_check.py             # System health validation
├── setup_dev.py               # Development environment setup
├── test_integration.py        # Integration testing
├── validate_config.py         # Configuration validation
├── analysis/                  # Repository analysis tools
├── development/              # Development utilities
├── testing/                  # Testing and validation
├── deployment/               # Deployment scripts
└── maintenance/              # System maintenance
```

## 🚀 Quick Scripts

### Health Check
```bash
# Full system health check
python scripts/health_check.py --full

# Create default configuration
python scripts/health_check.py --create-config

# Check specific components
python scripts/health_check.py --config config.json
```

### Development Setup
```bash
# Set up development environment
python scripts/setup_dev.py

# Install development dependencies
python scripts/setup_dev.py --dev-deps

# Configure pre-commit hooks
python scripts/setup_dev.py --pre-commit
```

### Testing
```bash
# Run integration tests
python scripts/test_integration.py

# Run with specific broker
python scripts/test_integration.py --broker alpaca

# Load testing
python scripts/test_integration.py --load-test
```

### Configuration
```bash
# Validate configuration
python scripts/validate_config.py

# Validate specific file
python scripts/validate_config.py --config config.json

# Check environment variables
python scripts/validate_config.py --env-check
```

## 📝 Script Categories

### Development (`development/`)
- **Environment setup** and dependency management
- **Code quality** tools and linting
- **Database** migration and setup
- **Docker** containerization scripts

### Testing (`testing/`)
- **Unit tests** and test runners
- **Integration tests** for broker APIs
- **Performance tests** and benchmarks
- **Load testing** for system capacity

### Analysis (`analysis/`)
- **Repository analysis** and code metrics
- **Performance profiling** and optimization
- **Security scanning** and vulnerability assessment
- **Documentation** generation and validation

### Deployment (`deployment/`)
- **Application deployment** to various platforms
- **Database** deployment and migration
- **Environment** configuration and setup
- **Monitoring** and alerting setup

### Maintenance (`maintenance/`)
- **Log rotation** and cleanup
- **Data backup** and recovery
- **System updates** and patching
- **Performance monitoring** and alerts

## 🔧 Usage Examples

### Daily Development Workflow
```bash
# Morning setup
python scripts/development/setup_environment.py

# Run tests before committing
python scripts/testing/run_all_tests.py

# Code quality check
python scripts/development/code_quality.py
```

### Deployment Workflow
```bash
# Pre-deployment checks
python scripts/deployment/pre_deploy_check.py

# Deploy to staging
python scripts/deployment/deploy.py --environment staging

# Run smoke tests
python scripts/testing/smoke_tests.py
```

### System Maintenance
```bash
# Weekly maintenance
python scripts/maintenance/weekly_cleanup.py

# Backup database
python scripts/maintenance/backup_database.py

# Update dependencies
python scripts/maintenance/update_dependencies.py
```

## 🛠️ Adding New Scripts

When adding new scripts to this directory:

1. **Follow naming conventions**:
   - Use snake_case for Python scripts
   - Include descriptive names
   - Add version suffix if needed

2. **Add documentation**:
   - Include docstring at the top
   - Add usage examples
   - Document command-line arguments

3. **Make executable**:
   - Add shebang line: `#!/usr/bin/env python3`
   - Set execute permissions: `chmod +x script_name.py`

4. **Include error handling**:
   - Handle common errors gracefully
   - Provide helpful error messages
   - Use appropriate exit codes

5. **Add to README**:
   - Update this README with new script
   - Include usage examples
   - Document dependencies

## 📊 Script Status

| Script | Status | Purpose |
|--------|--------|----------|
| `health_check.py` | ✅ Complete | System health validation |
| `setup_dev.py` | ✅ Complete | Development environment setup |
| `test_integration.py` | ✅ Complete | Integration testing |
| `validate_config.py` | ✅ Complete | Configuration validation |
| `analysis/*` | 📝 Placeholder | Repository analysis tools |
| `development/*` | 📝 Placeholder | Development utilities |
| `testing/*` | 📝 Placeholder | Testing and validation |
| `deployment/*` | 📝 Placeholder | Deployment scripts |
| `maintenance/*` | 📝 Placeholder | System maintenance |

## 🔍 Troubleshooting

### Common Issues

**Script not found**
```bash
# Make script executable
chmod +x scripts/script_name.py

# Add to PATH
export PATH=$PATH:/path/to/not-stonks-bot/scripts
```

**Permission denied**
```bash
# Run with Python directly
python scripts/script_name.py

# Or fix permissions
sudo chmod +x scripts/script_name.py
```

**Missing dependencies**
```bash
# Install script dependencies
pip install -r scripts/requirements.txt

# Or use virtual environment
source venv/bin/activate
python scripts/script_name.py
```

### Getting Help

- Check script documentation with `--help` flag
- Review log files in `logs/` directory
- Run health check to identify issues
- Check GitHub issues for known problems

---

**Note**: All scripts should be self-contained and not depend on external services being available. They should provide clear error messages and helpful suggestions for common issues.