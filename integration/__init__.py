"""
Integration Framework Package
System integration, deployment, monitoring, and documentation.
"""

from .integration_manager import IntegrationManager, integration_manager, ComponentStatus
from .doc_generator import DocumentationGenerator, doc_generator
from .deployment_manager import DeploymentManager, deployment_manager, DeploymentStatus
from .system_monitor import SystemMonitor, system_monitor, AlertSeverity
from .health_checks import HealthCheckFramework, health_check_framework, HealthCheckStatus

__all__ = [
    'IntegrationManager',
    'integration_manager',
    'ComponentStatus',
    'DocumentationGenerator',
    'doc_generator',
    'DeploymentManager',
    'deployment_manager',
    'DeploymentStatus',
    'SystemMonitor',
    'system_monitor',
    'AlertSeverity',
    'HealthCheckFramework',
    'health_check_framework',
    'HealthCheckStatus'
]

__version__ = "1.0.0"
__description__ = "System integration and deployment framework"