"""
Test Suite Manager
Manages automated test discovery, execution, and coverage reporting.
"""

import unittest
import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import subprocess
import coverage
import HTMLTestRunner
import io
from datetime import datetime
import importlib.util
import sys
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """Test status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class TestType(Enum):
    """Test type enumeration."""
    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"
    PERFORMANCE = "performance"
    SECURITY = "security"
    UAT = "uat"


@dataclass
class TestResult:
    """Test result container."""
    test_name: str
    test_type: TestType
    status: TestStatus
    duration: float
    error_message: Optional[str] = None
    traceback: Optional[str] = None
    assertions: int = 0
    timestamp: datetime = None


@dataclass
class TestSuite:
    """Test suite configuration."""
    name: str
    test_type: TestType
    test_files: List[str]
    dependencies: List[str] = None
    parallel: bool = True
    timeout: int = 300  # 5 minutes default
    priority: int = 1


class TestSuiteManager:
    """Manages test suite execution and reporting."""
    
    def __init__(self, test_root: str = "tests"):
        self.test_root = Path(test_root)
        self.test_suites: Dict[str, TestSuite] = {}
        self.test_results: List[TestResult] = []
        self.coverage_data: Dict[str, Any] = {}
        self.test_config = self._load_test_config()
        self.is_running = False
        
        # Initialize coverage
        self.coverage = coverage.Coverage(
            source=[str(self.test_root)],
            omit=["*/tests/*", "*/venv/*", "*/node_modules/*"]
        )
        
    def _load_test_config(self) -> Dict[str, Any]:
        """Load test configuration."""
        try:
            config_path = self.test_root / "test_config.json"
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Test config not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default test configuration."""
        return {
            "parallel_execution": True,
            "max_workers": 4,
            "test_timeout": 300,
            "coverage_threshold": 80,
            "report_formats": ["html", "json", "xml"],
            "test_patterns": ["test_*.py", "*_test.py"],
            "excluded_patterns": ["__pycache__", "*.pyc"]
        }
    
    def discover_tests(self) -> Dict[str, TestSuite]:
        """Discover available test files and organize into suites."""
        logger.info("Discovering test files...")
        
        discovered_suites = {}
        
        # Scan test directories
        for test_dir in self.test_root.iterdir():
            if test_dir.is_dir() and not test_dir.name.startswith('.'):
                suite_name = test_dir.name
                test_files = []
                
                # Find test files
                for pattern in self.test_config["test_patterns"]:
                    test_files.extend(test_dir.glob(pattern))
                
                if test_files:
                    # Determine test type from directory name
                    test_type = self._determine_test_type(suite_name)
                    
                    discovered_suites[suite_name] = TestSuite(
                        name=suite_name,
                        test_type=test_type,
                        test_files=[str(f) for f in test_files],
                        priority=self._get_test_priority(suite_name)
                    )
                    
                    logger.info(f"Discovered test suite: {suite_name} "
                              f"({len(test_files)} test files, type: {test_type.value})")
        
        self.test_suites = discovered_suites
        return discovered_suites
    
    def _determine_test_type(self, suite_name: str) -> TestType:
        """Determine test type from suite name."""
        name_lower = suite_name.lower()
        
        if "integration" in name_lower:
            return TestType.INTEGRATION
        elif "e2e" in name_lower or "end-to-end" in name_lower:
            return TestType.E2E
        elif "performance" in name_lower or "perf" in name_lower:
            return TestType.PERFORMANCE
        elif "security" in name_lower:
            return TestType.SECURITY
        elif "uat" in name_lower:
            return TestType.UAT
        else:
            return TestType.UNIT
    
    def _get_test_priority(self, suite_name: str) -> int:
        """Get test suite priority."""
        test_type = self._determine_test_type(suite_name)
        
        priority_map = {
            TestType.UNIT: 1,
            TestType.INTEGRATION: 2,
            TestType.E2E: 3,
            TestType.PERFORMANCE: 4,
            TestType.SECURITY: 5,
            TestType.UAT: 6
        }
        
        return priority_map.get(test_type, 1)
    
    async def run_test_suite(self, suite_name: str) -> List[TestResult]:
        """Run a specific test suite."""
        if suite_name not in self.test_suites:
            logger.error(f"Test suite {suite_name} not found")
            return []
        
        suite = self.test_suites[suite_name]
        logger.info(f"Running test suite: {suite_name}")
        
        start_time = time.time()
        
        try:
            # Start coverage collection
            if self._should_collect_coverage(suite.test_type):
                self.coverage.start()
            
            # Run tests based on suite configuration
            if suite.parallel and self.test_config["parallel_execution"]:
                results = await self._run_tests_parallel(suite)
            else:
                results = await self._run_tests_sequential(suite)
            
            # Stop coverage collection
            if self._should_collect_coverage(suite.test_type):
                self.coverage.stop()
                self.coverage.save()
            
            # Update results with timing
            duration = time.time() - start_time
            for result in results:
                result.duration = duration / len(results) if results else 0
            
            self.test_results.extend(results)
            logger.info(f"Test suite {suite_name} completed: "
                       f"{len(results)} tests, {len([r for r in results if r.status == TestStatus.PASSED])} passed")
            
            return results
            
        except Exception as e:
            logger.error(f"Error running test suite {suite_name}: {str(e)}")
            return []
    
    def _should_collect_coverage(self, test_type: TestType) -> bool:
        """Determine if coverage should be collected for test type."""
        return test_type in [TestType.UNIT, TestType.INTEGRATION]
    
    async def _run_tests_parallel(self, suite: TestSuite) -> List[TestResult]:
        """Run tests in parallel."""
        results = []
        tasks = []
        
        max_workers = min(len(suite.test_files), self.test_config["max_workers"])
        semaphore = asyncio.Semaphore(max_workers)
        
        async def run_test_file(test_file: str):
            async with semaphore:
                return await self._run_test_file(test_file, suite.test_type)
        
        for test_file in suite.test_files:
            task = asyncio.create_task(run_test_file(test_file))
            tasks.append(task)
        
        file_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results
        for file_result in file_results:
            if isinstance(file_result, list):
                results.extend(file_result)
            elif isinstance(file_result, Exception):
                logger.error(f"Test file execution error: {str(file_result)}")
        
        return results
    
    async def _run_tests_sequential(self, suite: TestSuite) -> List[TestResult]:
        """Run tests sequentially."""
        results = []
        
        for test_file in suite.test_files:
            file_results = await self._run_test_file(test_file, suite.test_type)
            results.extend(file_results)
        
        return results
    
    async def _run_test_file(self, test_file: str, test_type: TestType) -> List[TestResult]:
        """Run tests from a single file."""
        results = []
        
        try:
            # Load test file as module
            spec = importlib.util.spec_from_file_location("test_module", test_file)
            test_module = importlib.util.module_from_spec(spec)
            sys.modules["test_module"] = test_module
            spec.loader.exec_module(test_module)
            
            # Discover test classes and methods
            test_classes = self._discover_test_classes(test_module)
            
            for test_class in test_classes:
                class_results = await self._run_test_class(test_class, test_file, test_type)
                results.extend(class_results)
        
        except Exception as e:
            logger.error(f"Error running test file {test_file}: {str(e)}")
            results.append(TestResult(
                test_name=test_file,
                test_type=test_type,
                status=TestStatus.ERROR,
                duration=0,
                error_message=str(e),
                traceback=traceback.format_exc()
            ))
        
        return results
    
    def _discover_test_classes(self, module) -> List[type]:
        """Discover test classes in a module."""
        test_classes = []
        
        for name in dir(module):
            obj = getattr(module, name)
            if (isinstance(obj, type) and 
                name.startswith('Test') and 
                issubclass(obj, unittest.TestCase)):
                test_classes.append(obj)
        
        return test_classes
    
    async def _run_test_class(self, test_class: type, test_file: str, test_type: TestType) -> List[TestResult]:
        """Run all tests in a test class."""
        results = []
        test_instance = test_class()
        
        # Get all test methods
        test_methods = [method for method in dir(test_instance) 
                       if method.startswith('test_')]
        
        for method_name in test_methods:
            result = await self._run_single_test(test_instance, method_name, test_file, test_type)
            results.append(result)
        
        return results
    
    async def _run_single_test(self, test_instance: unittest.TestCase, 
                              method_name: str, test_file: str, test_type: TestType) -> TestResult:
        """Run a single test method."""
        start_time = time.time()
        
        try:
            # Set up test
            if hasattr(test_instance, 'setUp'):
                test_instance.setUp()
            
            # Run test
            method = getattr(test_instance, method_name)
            method()
            
            # Count assertions
            runner = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0)
            result = runner.run(unittest.TestSuite())
            
            duration = time.time() - start_time
            
            return TestResult(
                test_name=f"{test_instance.__class__.__name__}.{method_name}",
                test_type=test_type,
                status=TestStatus.PASSED,
                duration=duration,
                assertions=result.testsRun if hasattr(result, 'testsRun') else 1
            )
            
        except unittest.SkipTest as e:
            duration = time.time() - start_time
            return TestResult(
                test_name=f"{test_instance.__class__.__name__}.{method_name}",
                test_type=test_type,
                status=TestStatus.SKIPPED,
                duration=duration,
                error_message=str(e)
            )
        
        except AssertionError as e:
            duration = time.time() - start_time
            return TestResult(
                test_name=f"{test_instance.__class__.__name__}.{method_name}",
                test_type=test_type,
                status=TestStatus.FAILED,
                duration=duration,
                error_message=str(e),
                traceback=traceback.format_exc()
            )
        
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name=f"{test_instance.__class__.__name__}.{method_name}",
                test_type=test_type,
                status=TestStatus.ERROR,
                duration=duration,
                error_message=str(e),
                traceback=traceback.format_exc()
            )
        
        finally:
            # Tear down test
            if hasattr(test_instance, 'tearDown'):
                try:
                    test_instance.tearDown()
                except Exception as e:
                    logger.warning(f"Error in test teardown: {str(e)}")
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all discovered test suites."""
        logger.info("Starting comprehensive test execution...")
        
        self.is_running = True
        self.test_results = []
        
        # Sort suites by priority
        sorted_suites = sorted(self.test_suites.values(), 
                             key=lambda s: s.priority)
        
        overall_start_time = time.time()
        
        for suite in sorted_suites:
            if not self.is_running:
                break
                
            logger.info(f"Executing suite: {suite.name}")
            await self.run_test_suite(suite.name)
        
        overall_duration = time.time() - overall_start_time
        
        # Generate comprehensive report
        report = self._generate_test_report(overall_duration)
        
        self.is_running = False
        logger.info(f"Test execution completed in {overall_duration:.2f} seconds")
        
        return report
    
    def _generate_test_report(self, total_duration: float) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_duration": total_duration,
            "total_tests": len(self.test_results),
            "test_summary": {
                "passed": len([r for r in self.test_results if r.status == TestStatus.PASSED]),
                "failed": len([r for r in self.test_results if r.status == TestStatus.FAILED]),
                "error": len([r for r in self.test_results if r.status == TestStatus.ERROR]),
                "skipped": len([r for r in self.test_results if r.status == TestStatus.SKIPPED])
            },
            "test_results": [self._result_to_dict(r) for r in self.test_results],
            "coverage": self._get_coverage_report()
        }
        
        # Generate reports in configured formats
        self._generate_reports(report)
        
        return report
    
    def _result_to_dict(self, result: TestResult) -> Dict[str, Any]:
        """Convert TestResult to dictionary."""
        return {
            "test_name": result.test_name,
            "test_type": result.test_type.value,
            "status": result.status.value,
            "duration": result.duration,
            "error_message": result.error_message,
            "assertions": result.assertions,
            "timestamp": result.timestamp.isoformat() if result.timestamp else None
        }
    
    def _get_coverage_report(self) -> Dict[str, Any]:
        """Get coverage report."""
        try:
            # Get coverage data
            coverage_data = self.coverage.report(show_missing=True)
            
            # Calculate line coverage percentage
            total_lines = self.coverage.report(file=io.StringIO())
            
            return {
                "line_coverage": coverage_data,
                "html_report": "htmlcov/index.html",
                "xml_report": "coverage.xml"
            }
        except Exception as e:
            logger.error(f"Error generating coverage report: {str(e)}")
            return {}
    
    def _generate_reports(self, report: Dict[str, Any]):
        """Generate reports in configured formats."""
        for format_type in self.test_config["report_formats"]:
            try:
                if format_type == "json":
                    with open(f"test_report_{int(time.time())}.json", 'w') as f:
                        json.dump(report, f, indent=2)
                
                elif format_type == "html":
                    self._generate_html_report(report)
                
                elif format_type == "xml":
                    self._generate_xml_report(report)
                    
            except Exception as e:
                logger.error(f"Error generating {format_type} report: {str(e)}")
    
    def _generate_html_report(self, report: Dict[str, Any]):
        """Generate HTML test report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background: #f0f0f0; padding: 15px; margin-bottom: 20px; }}
                .passed {{ color: green; }}
                .failed {{ color: red; }}
                .error {{ color: orange; }}
                .skipped {{ color: gray; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Test Execution Report</h1>
            <div class="summary">
                <h2>Summary</h2>
                <p>Total Tests: {report['total_tests']}</p>
                <p class="passed">Passed: {report['test_summary']['passed']}</p>
                <p class="failed">Failed: {report['test_summary']['failed']}</p>
                <p class="error">Errors: {report['test_summary']['error']}</p>
                <p class="skipped">Skipped: {report['test_summary']['skipped']}</p>
                <p>Duration: {report['total_duration']:.2f} seconds</p>
            </div>
            <h2>Test Results</h2>
            <table>
                <tr>
                    <th>Test Name</th>
                    <th>Type</th>
                    <th>Status</th>
                    <th>Duration</th>
                    <th>Assertions</th>
                </tr>
        """
        
        for result in report['test_results']:
            status_class = result['status']
            html_content += f"""
                <tr>
                    <td>{result['test_name']}</td>
                    <td>{result['test_type']}</td>
                    <td class="{status_class}">{result['status']}</td>
                    <td>{result['duration']:.3f}s</td>
                    <td>{result['assertions']}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        with open(f"test_report_{int(time.time())}.html", 'w') as f:
            f.write(html_content)
    
    def _generate_xml_report(self, report: Dict[str, Any]):
        """Generate XML test report."""
        xml_content = '<?xml version="1.0" encoding="UTF-8"?>\n'
        xml_content += f'<testsuite name="All Tests" tests="{report["total_tests"]}" '
        xml_content += f'failures="{report["test_summary"]["failed"]}" '
        xml_content += f'errors="{report["test_summary"]["error"]}">\n'
        
        for result in report['test_results']:
            status = "passed" if result['status'] == 'passed' else 'failed'
            xml_content += f'    <testcase name="{result["test_name"]}" '
            xml_content += f'type="{result["test_type"]}" '
            xml_content += f'status="{status}" '
            xml_content += f'time="{result["duration"]:.3f}">\n'
            
            if result['status'] in ['failed', 'error']:
                xml_content += f'        <failure message="{result["error_message"]}"/>\n'
            
            xml_content += '    </testcase>\n'
        
        xml_content += '</testsuite>'
        
        with open(f"test_report_{int(time.time())}.xml", 'w') as f:
            f.write(xml_content)
    
    def stop_tests(self):
        """Stop test execution."""
        self.is_running = False
        logger.info("Test execution stopped")
    
    def get_test_status(self) -> Dict[str, Any]:
        """Get current test execution status."""
        return {
            "is_running": self.is_running,
            "total_discovered_suites": len(self.test_suites),
            "total_completed_tests": len(self.test_results),
            "discovered_suites": list(self.test_suites.keys())
        }


# Global test suite manager instance
test_manager = TestSuiteManager()