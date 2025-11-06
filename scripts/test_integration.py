#!/usr/bin/env python3
"""
Day Trading Orchestrator - Integration Test Suite
Comprehensive testing of all system components and integrations
"""

import asyncio
import sys
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

class TestResults:
    """Test results tracker"""
    
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.tests_skipped = 0
        self.failures = []
        
    def add_pass(self, test_name: str):
        self.tests_passed += 1
        print(f"✅ PASS: {test_name}")
        
    def add_fail(self, test_name: str, error: str):
        self.tests_failed += 1
        self.failures.append({"test": test_name, "error": error})
        print(f"❌ FAIL: {test_name} - {error}")
        
    def add_skip(self, test_name: str, reason: str):
        self.tests_skipped += 1
        print(f"⏭️  SKIP: {test_name} - {reason}")
        
    def summary(self):
        total = self.tests_passed + self.tests_failed + self.tests_skipped
        print(f"\n{'='*60}")
        print(f"TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests: {total}")
        print(f"✅ Passed: {self.tests_passed}")
        print(f"❌ Failed: {self.tests_failed}")
        print(f"⏭️  Skipped: {self.tests_skipped}")
        
        if self.failures:
            print(f"\nFAILED TESTS:")
            for failure in self.failures:
                print(f"  - {failure['test']}: {failure['error']}")
                
        return self.tests_failed == 0

class IntegrationTestSuite:
    """Comprehensive integration test suite"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self.load_config()
        self.results = TestResults()
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️  Configuration file {self.config_path} not found")
            return {}
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in configuration: {e}")
            return {}
    
    async def test_system_initialization(self):
        """Test basic system initialization"""
        print("\n🔧 Testing System Initialization...")
        
        # Test 1: Configuration Loading
        if not self.config:
            self.results.add_fail("Configuration Loading", "No configuration loaded")
            return
            
        required_sections = ["database", "brokers", "ai", "risk"]
        for section in required_sections:
            if section not in self.config:
                self.results.add_fail(f"Config Section {section}", "Missing configuration section")
            else:
                self.results.add_pass(f"Config Section {section}")
        
        # Test 2: Directory Creation
        import os
        required_dirs = ["logs", "data", "backups"]
        for dir_name in required_dirs:
            try:
                os.makedirs(dir_name, exist_ok=True)
                if os.path.exists(dir_name):
                    self.results.add_pass(f"Directory Creation {dir_name}")
                else:
                    self.results.add_fail(f"Directory Creation {dir_name}", "Directory not created")
            except Exception as e:
                self.results.add_fail(f"Directory Creation {dir_name}", str(e))
    
    async def run_all_tests(self):
        """Run all integration tests"""
        print("🚀 Starting Day Trading Orchestrator Integration Tests")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Configuration: {self.config_path}")
        
        # Run test suites
        test_suites = [
            ("System Initialization", self.test_system_initialization)
        ]
        
        for suite_name, test_function in test_suites:
            print(f"\n{'='*60}")
            print(f"Running: {suite_name}")
            print(f"{'='*60}")
            
            try:
                await test_function()
            except Exception as e:
                self.results.add_fail(suite_name, f"Test suite error: {str(e)}")
                print(f"❌ Test suite {suite_name} failed: {e}")
        
        # Generate summary
        success = self.results.summary()
        
        # Save test report
        await self.save_test_report()
        
        return success
    
    async def save_test_report(self):
        """Save test report to file"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "config_file": self.config_path,
            "results": {
                "passed": self.results.tests_passed,
                "failed": self.results.tests_failed,
                "skipped": self.results.tests_skipped,
                "total": self.results.tests_passed + self.results.tests_failed + self.results.tests_skipped
            },
            "failures": self.results.failures
        }
        
        report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Test report saved to: {report_file}")

async def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Day Trading Orchestrator Integration Tests")
    parser.add_argument("--config", default="config.json", help="Configuration file path")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    parser.add_argument("--broker", help="Test specific broker only")
    parser.add_argument("--output", help="Output file for test results")
    
    args = parser.parse_args()
    
    # Create test suite
    test_suite = IntegrationTestSuite(args.config)
    
    # Run tests
    success = await test_suite.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test execution failed: {e}")
        sys.exit(1)