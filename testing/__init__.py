"""
Testing Framework Package
Comprehensive testing suite for the trading system.
"""

from .test_suite_manager import TestSuiteManager, test_manager
from .integration_tests import IntegrationTestFramework, integration_test_framework
from .performance_tests import PerformanceTestSuite, performance_test_suite
from .security_tests import SecurityTestSuite, security_test_suite
from .uat_scenarios import UATFramework, uat_framework, create_trading_system_uat_scenarios
from .final_validation import FinalIntegrationValidator, final_validator

__all__ = [
    'TestSuiteManager',
    'test_manager',
    'IntegrationTestFramework',
    'integration_test_framework',
    'PerformanceTestSuite',
    'performance_test_suite',
    'SecurityTestSuite',
    'security_test_suite',
    'UATFramework',
    'uat_framework',
    'create_trading_system_uat_scenarios',
    'FinalIntegrationValidator',
    'final_validator'
]

__version__ = "1.0.0"
__description__ = "Comprehensive testing framework for trading system"
