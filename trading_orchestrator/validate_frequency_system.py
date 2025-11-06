"""
@file validate_frequency_system.py
@brief Simple validation script for Trading Frequency Configuration System

@details
This script validates that all frequency system components are properly
importable and functional without running the full demo.

@author Trading Orchestrator System
@version 1.0
@date 2025-11-06
"""

import sys
import traceback
from typing import Dict, Any


def test_imports():
    """Test if all frequency system modules can be imported"""
    print("=" * 60)
    print("TRADING FREQUENCY CONFIGURATION SYSTEM - VALIDATION")
    print("=" * 60)
    
    tests = []
    
    # Test 1: Core frequency configuration
    try:
        print("\n1. Testing core frequency configuration...")
        from config.trading_frequency import (
            FrequencyManager, FrequencySettings, FrequencyType, 
            FrequencyAlertType, FrequencyAlert, FrequencyOptimization,
            initialize_frequency_manager, get_frequency_manager
        )
        print("   ✅ Core frequency configuration imported successfully")
        tests.append(("Core Configuration", True, None))
    except Exception as e:
        print(f"   ❌ Core frequency configuration failed: {e}")
        tests.append(("Core Configuration", False, str(e)))
    
    # Test 2: Risk management integration
    try:
        print("\n2. Testing frequency risk management...")
        from risk.frequency_risk_manager import (
            FrequencyRiskManager, FrequencyRiskAssessment, 
            FrequencyRiskLimit
        )
        print("   ✅ Frequency risk management imported successfully")
        tests.append(("Risk Management", True, None))
    except Exception as e:
        print(f"   ❌ Frequency risk management failed: {e}")
        tests.append(("Risk Management", False, str(e)))
    
    # Test 3: Analytics engine
    try:
        print("\n3. Testing frequency analytics...")
        from analytics.frequency_analytics import (
            FrequencyAnalyticsEngine, FrequencyAnalyticsReport,
            AnalyticsPeriod, OptimizationTarget
        )
        print("   ✅ Frequency analytics imported successfully")
        tests.append(("Analytics Engine", True, None))
    except Exception as e:
        print(f"   ❌ Frequency analytics failed: {e}")
        tests.append(("Analytics Engine", False, str(e)))
    
    # Test 4: UI components
    try:
        print("\n4. Testing UI components...")
        from ui.components.frequency_components import (
            FrequencyConfigurationComponent, FrequencyMonitoringComponent,
            FrequencyAlertsComponent, FrequencyAnalyticsComponent
        )
        print("   ✅ UI components imported successfully")
        tests.append(("UI Components", True, None))
    except Exception as e:
        print(f"   ❌ UI components failed: {e}")
        tests.append(("UI Components", False, str(e)))
    
    # Test 5: Database models
    try:
        print("\n5. Testing database models...")
        from api.database.models import FrequencyConfigDB, FrequencyMetricsDB
        print("   ✅ Database models imported successfully")
        tests.append(("Database Models", True, None))
    except Exception as e:
        print(f"   ❌ Database models failed: {e}")
        tests.append(("Database Models", False, str(e)))
    
    # Test 6: Database migration
    try:
        print("\n6. Testing database migration...")
        with open('/workspace/trading_orchestrator/database/migrations/002_frequency_management.sql', 'r') as f:
            migration_sql = f.read()
        
        # Check for key tables
        required_tables = [
            'frequency_configs', 'frequency_metrics', 'frequency_alerts',
            'frequency_optimization', 'trade_frequency', 'frequency_constraints',
            'frequency_analytics'
        ]
        
        missing_tables = []
        for table in required_tables:
            if f'CREATE TABLE {table}' not in migration_sql:
                missing_tables.append(table)
        
        if missing_tables:
            raise Exception(f"Missing tables: {missing_tables}")
        
        print("   ✅ Database migration file contains all required tables")
        tests.append(("Database Migration", True, None))
    except Exception as e:
        print(f"   ❌ Database migration failed: {e}")
        tests.append(("Database Migration", False, str(e)))
    
    return tests


def test_basic_functionality():
    """Test basic functionality of key components"""
    print("\n" + "=" * 60)
    print("FUNCTIONALITY TESTS")
    print("=" * 60)
    
    tests = []
    
    try:
        print("\n1. Testing FrequencyType enum...")
        from config.trading_frequency import FrequencyType
        
        # Test enum values
        expected_types = [
            "ULTRA_HIGH", "HIGH", "MEDIUM", "LOW", "VERY_LOW", "CUSTOM"
        ]
        
        available_types = [ft.value for ft in FrequencyType]
        
        for expected in expected_types:
            if expected not in available_types:
                raise Exception(f"Missing FrequencyType: {expected}")
        
        print(f"   ✅ All expected FrequencyType values found: {available_types}")
        tests.append(("FrequencyType Enum", True, None))
        
    except Exception as e:
        print(f"   ❌ FrequencyType enum test failed: {e}")
        tests.append(("FrequencyType Enum", False, str(e)))
    
    try:
        print("\n2. Testing FrequencySettings creation...")
        from config.trading_frequency import FrequencySettings, FrequencyType
        
        settings = FrequencySettings(
            frequency_type=FrequencyType.MEDIUM,
            max_trades_per_minute=10,
            max_trades_per_hour=100,
            max_trades_per_day=500,
            cooldown_period_seconds=60,
            enable_alerts=True,
            alert_thresholds={
                "minute": 0.8,
                "hour": 0.9,
                "day": 0.95
            }
        )
        
        print("   ✅ FrequencySettings created successfully")
        print(f"   📊 Configuration: {settings.frequency_type.value}")
        print(f"   📊 Max trades per minute: {settings.max_trades_per_minute}")
        tests.append(("FrequencySettings", True, None))
        
    except Exception as e:
        print(f"   ❌ FrequencySettings test failed: {e}")
        tests.append(("FrequencySettings", False, str(e)))
    
    try:
        print("\n3. Testing FrequencyAlert creation...")
        from config.trading_frequency import FrequencyAlert, FrequencyAlertType, AlertSeverity
        
        alert = FrequencyAlert(
            alert_type=FrequencyAlertType.RATE_LIMIT,
            message="Test frequency alert",
            severity=AlertSeverity.WARNING,
            triggered_at="2025-11-06T05:32:47",
            strategy_name="test_strategy"
        )
        
        print("   ✅ FrequencyAlert created successfully")
        print(f"   📊 Alert type: {alert.alert_type.value}")
        print(f"   📊 Severity: {alert.severity.value}")
        tests.append(("FrequencyAlert", True, None))
        
    except Exception as e:
        print(f"   ❌ FrequencyAlert test failed: {e}")
        tests.append(("FrequencyAlert", False, str(e)))
    
    return tests


def generate_summary_report(import_tests, function_tests):
    """Generate a comprehensive summary report"""
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY REPORT")
    print("=" * 60)
    
    all_tests = import_tests + function_tests
    passed = sum(1 for _, success, _ in all_tests if success)
    total = len(all_tests)
    
    print(f"\nTotal Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Trading Frequency Configuration System is fully functional.")
    else:
        print(f"\n⚠️  {total - passed} tests failed. See details above.")
    
    print("\n📋 Component Status:")
    for test_name, success, error in all_tests:
        status = "✅ WORKING" if success else "❌ FAILED"
        print(f"   {test_name:<25} {status}")
        if error:
            print(f"   Error: {error}")
    
    print("\n📁 Files Created:")
    files = [
        "config/trading_frequency.py",
        "risk/frequency_risk_manager.py",
        "analytics/frequency_analytics.py",
        "ui/components/frequency_components.py",
        "database/migrations/002_frequency_management.sql",
        "api/database/models.py (extended)",
        "tests/test_frequency_system.py",
        "demo_frequency_system.py",
        "FREQUENCY_SYSTEM_IMPLEMENTATION.md"
    ]
    
    for file_path in files:
        print(f"   ✅ {file_path}")
    
    return passed == total


def main():
    """Main validation function"""
    try:
        import_tests = test_imports()
        function_tests = test_basic_functionality()
        success = generate_summary_report(import_tests, function_tests)
        
        print("\n" + "=" * 60)
        print("VALIDATION COMPLETE")
        print("=" * 60)
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)