"""
Perpetual Trading Operations Package
====================================

This package provides comprehensive 24/7 operation capabilities for the trading orchestrator system.

Main Components:
- perpetual_manager: Core perpetual operations management
- dashboard: Real-time monitoring dashboard and API
- runbooks: Operational procedures and documentation
- __init__: Integration and setup utilities

Features:
- Automatic system recovery and fault tolerance
- Comprehensive health monitoring and alerting
- Zero-downtime maintenance mode
- Performance optimization and resource management
- Backup and recovery procedures
- Real-time monitoring dashboard
- Operational runbooks and procedures
- SLA compliance monitoring

Author: Trading Orchestrator System
Version: 2.0.0
Date: 2025-11-06
"""

# Import main components for easy access
from .perpetual_manager import (
    PerpetualOperationsManager,
    SystemMonitor,
    HealthChecker,
    AlertManager,
    AutoRecoveryManager,
    MaintenanceManager,
    SystemMetrics,
    HealthCheckResult,
    AlertInfo,
    MaintenanceTask
)

from .dashboard import (
    integrate_perpetual_dashboard,
    DashboardWebSocketManager,
    DashboardMetrics,
    WebSocketMessage
)

from .runbooks import (
    RunbookManager,
    OperationalProcedures,
    ServiceLevelAgreement,
    OperationalRunbook,
    Procedure
)

from .__init__ import (
    PerpetualIntegration,
    PerpetualApplicationEnhancer,
    enable_perpetual_operations,
    disable_perpetual_operations,
    get_perpetual_status,
    setup_perpetual_fastapi_integration
)

# Package metadata
__version__ = "2.0.0"
__author__ = "Trading Orchestrator System"
__description__ = "Comprehensive 24/7 operation capabilities for trading orchestrator"

# Export main classes and functions
__all__ = [
    # Core management
    "PerpetualOperationsManager",
    "SystemMonitor", 
    "HealthChecker",
    "AlertManager",
    "AutoRecoveryManager",
    "MaintenanceManager",
    
    # Data models
    "SystemMetrics",
    "HealthCheckResult", 
    "AlertInfo",
    "MaintenanceTask",
    
    # Dashboard
    "integrate_perpetual_dashboard",
    "DashboardWebSocketManager",
    "DashboardMetrics",
    "WebSocketMessage",
    
    # Runbooks and procedures
    "RunbookManager",
    "OperationalProcedures",
    "ServiceLevelAgreement",
    "OperationalRunbook",
    "Procedure",
    
    # Integration
    "PerpetualIntegration",
    "PerpetualApplicationEnhancer",
    "enable_perpetual_operations",
    "disable_perpetual_operations", 
    "get_perpetual_status",
    "setup_perpetual_fastapi_integration"
]


# Package initialization
def initialize_package():
    """Initialize the perpetual operations package"""
    import os
    import logging
    from pathlib import Path
    
    # Create necessary directories
    directories = [
        "logs",
        "logs/metrics", 
        "backups",
        "cache",
        "tmp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Setup logging for perpetual operations
    logger = logging.getLogger("perpetual_operations")
    logger.setLevel(logging.INFO)
    
    # File handler for perpetual operations logs
    handler = logging.FileHandler("logs/perpetual_operations.log")
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


# Initialize on import
_logger = initialize_package()


# Quick start function
def quick_start():
    """
    Quick start guide for perpetual operations
    
    This function provides a simple interface to get started with perpetual operations.
    """
    guide = """
    
🎯 Perpetual Trading Operations - Quick Start Guide
=================================================

Welcome to Perpetual Trading Operations! This system provides 24/7 operation capabilities
for your trading orchestrator.

📋 Quick Start Steps:

1. Basic Integration:
   ```python
   from trading_orchestrator.perpetual_operations import enable_perpetual_operations
   
   # Enable perpetual operations with your app config
   success = await enable_perpetual_operations(app_config=your_app_config)
   ```

2. FastAPI Integration:
   ```python
   from fastapi import FastAPI
   from trading_orchestrator.perpetual_operations import setup_perpetual_fastapi_integration
   
   app = FastAPI()
   integration = setup_perpetual_fastapi_integration(app, app_config=your_app_config)
   ```

3. Access Dashboard:
   - Open http://localhost:8000/dashboard in your browser
   - Monitor system health, alerts, and performance
   - Manage maintenance and backup operations

4. API Endpoints:
   - Health check: GET /api/perpetual/health
   - System status: GET /api/perpetual/status
   - Maintenance: POST /api/perpetual/maintenance/start
   - Runbooks: GET /api/perpetual/runbooks

🔧 Key Features Enabled:
- ✅ Real-time system monitoring
- ✅ Automatic recovery from failures
- ✅ Health monitoring and alerts
- ✅ Maintenance mode (zero-downtime)
- ✅ Performance optimization
- ✅ Backup and recovery
- ✅ Resource management
- ✅ Memory leak detection
- ✅ Connection pooling
- ✅ SLA compliance monitoring

📊 Dashboard Features:
- System metrics visualization
- Health status monitoring
- Alert management
- Maintenance scheduling
- Performance analytics
- Backup management

⚡ Auto-Recovery Handles:
- Database connection issues
- Broker connectivity problems
- Memory pressure situations
- High CPU usage
- Disk space constraints
- Network connectivity issues

🔧 Maintenance Operations:
- Automatic log rotation
- Database optimization
- Cache cleanup
- Connection pool maintenance
- Memory optimization
- Health audit

📋 Operational Runbooks:
- System startup procedures
- Emergency response
- Maintenance operations
- Troubleshooting guides
- Backup and recovery
- SLA compliance

For more information, visit the dashboard or check the operational runbooks.

Happy Trading! 🚀
    """
    
    print(guide)
    return guide


# Version information
def get_version_info():
    """Get detailed version and capability information"""
    return {
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'capabilities': [
            '24/7 System Operation',
            'Automatic Failure Recovery',
            'Real-time Health Monitoring',
            'Zero-downtime Maintenance',
            'Performance Optimization',
            'Comprehensive Alerting',
            'Backup and Recovery',
            'Resource Management',
            'Memory Leak Detection',
            'Connection Pooling',
            'SLA Compliance Monitoring',
            'Operational Runbooks',
            'Real-time Dashboard',
            'RESTful API Integration'
        ],
        'integrations': [
            'Trading Orchestrator',
            'FastAPI Applications',
            'Database Systems',
            'Broker APIs',
            'AI/ML Systems',
            'Monitoring Systems'
        ],
        'requirements': [
            'Python 3.8+',
            'FastAPI (optional)',
            'WebSocket support',
            'SQLite/PostgreSQL',
            'System monitoring tools'
        ]
    }


# Convenience function to check system readiness
def check_system_readiness():
    """Check if system is ready for perpetual operations"""
    import sys
    import platform
    
    readiness = {
        'ready': True,
        'checks': [],
        'warnings': [],
        'errors': []
    }
    
    # Python version check
    if sys.version_info < (3, 8):
        readiness['errors'].append("Python 3.8+ required")
        readiness['ready'] = False
    else:
        readiness['checks'].append(f"Python {sys.version.split()[0]} ✓")
    
    # Platform check
    readiness['checks'].append(f"Platform: {platform.system()} {platform.release()} ✓")
    
    # Directory checks
    import os
    required_dirs = ['logs', 'backups', 'cache']
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            try:
                os.makedirs(dir_name, exist_ok=True)
                readiness['checks'].append(f"Created directory: {dir_name} ✓")
            except Exception as e:
                readiness['errors'].append(f"Cannot create directory {dir_name}: {e}")
                readiness['ready'] = False
        else:
            readiness['checks'].append(f"Directory exists: {dir_name} ✓")
    
    # Permission checks
    try:
        test_file = Path("logs/.perpetual_test")
        test_file.write_text("test")
        test_file.unlink()
        readiness['checks'].append("File system write access ✓")
    except Exception as e:
        readiness['errors'].append(f"File system write access denied: {e}")
        readiness['ready'] = False
    
    # Memory check (basic)
    try:
        import psutil
        memory = psutil.virtual_memory()
        if memory.total < 2 * 1024 * 1024 * 1024:  # Less than 2GB
            readiness['warnings'].append("Low memory: Consider 4GB+ for optimal performance")
        else:
            readiness['checks'].append(f"Memory: {memory.total // (1024**3)}GB ✓")
    except ImportError:
        readiness['warnings'].append("psutil not installed: memory monitoring limited")
    except Exception as e:
        readiness['warnings'].append(f"Memory check failed: {e}")
    
    return readiness


# Print readiness report
def print_readiness_report():
    """Print a formatted system readiness report"""
    readiness = check_system_readiness()
    
    print("\n🔍 System Readiness Report")
    print("=" * 40)
    
    if readiness['ready']:
        print("✅ System is READY for Perpetual Operations")
    else:
        print("❌ System is NOT READY - Fix errors before proceeding")
    
    print("\n📋 Checks:")
    for check in readiness['checks']:
        print(f"  ✓ {check}")
    
    if readiness['warnings']:
        print("\n⚠️  Warnings:")
        for warning in readiness['warnings']:
            print(f"  ⚠ {warning}")
    
    if readiness['errors']:
        print("\n❌ Errors:")
        for error in readiness['errors']:
            print(f"  ❌ {error}")
    
    print(f"\n🎯 Status: {'READY' if readiness['ready'] else 'NOT READY'}")
    print("=" * 40)
    
    return readiness


# Auto-execute readiness check on import if in development
if __name__ != "__main__":
    # Only run readiness check in development environment
    import os
    if os.getenv('PERPETUAL_CHECK_READINESS', 'true').lower() == 'true':
        try:
            print_readiness_report()
        except Exception:
            pass  # Silently ignore readiness check errors


if __name__ == "__main__":
    # When run as main module
    print_readiness_report()
    print("\n" + "=" * 50)
    quick_start()