#!/usr/bin/env python3
"""
System Integration and Testing Runner
Executes comprehensive system integration and testing workflows.
"""

import asyncio
import logging
import argparse
import json
from datetime import datetime
from pathlib import Path

# Import all system components
from integration.integration_manager import integration_manager
from integration.system_monitor import system_monitor
from integration.health_checks import health_check_framework
from testing.test_suite_manager import test_manager
from testing.integration_tests import integration_test_framework
from testing.performance_tests import performance_test_suite
from testing.security_tests import security_test_suite
from testing.uat_scenarios import uat_framework, create_trading_system_uat_scenarios
from testing.final_validation import final_validator
from integration.doc_generator import doc_generator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SystemIntegrationTester:
    """Main system integration and testing coordinator."""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.results = {}
    
    async def run_complete_integration_test(self):
        """Run complete system integration and testing suite."""
        logger.info("🚀 Starting Complete System Integration and Testing")
        logger.info(f"Start time: {self.start_time}")
        
        try:
            # Step 1: Initialize system components
            await self._initialize_system()
            
            # Step 2: Run health checks
            await self._run_health_checks()
            
            # Step 3: Run unit and integration tests
            await self._run_test_suites()
            
            # Step 4: Run performance tests
            await self._run_performance_tests()
            
            # Step 5: Run security tests
            await self._run_security_tests()
            
            # Step 6: Run UAT scenarios
            await self._run_uat_tests()
            
            # Step 7: Run final integration validation
            await self._run_final_validation()
            
            # Step 8: Generate documentation
            await self._generate_documentation()
            
            # Step 9: Generate final report
            await self._generate_final_report()
            
            logger.info("✅ Complete system integration and testing finished successfully")
            
        except Exception as e:
            logger.error(f"❌ System integration testing failed: {str(e)}")
            raise
        finally:
            end_time = datetime.now()
            duration = (end_time - self.start_time).total_seconds()
            logger.info(f"Total execution time: {duration:.2f} seconds")
    
    async def _initialize_system(self):
        """Initialize system components."""
        logger.info("🔧 Initializing system components...")
        
        # Register components in integration manager
        components = [
            ("crawler_manager", "1.0.0"),
            ("trading_orchestrator", "1.0.0"),
            ("risk_manager", "1.0.0"),
            ("ui_components", "1.0.0"),
            ("api_server", "1.0.0"),
            ("database", "1.0.0")
        ]
        
        for name, version in components:
            success = await integration_manager.register_component(name, version)
            if success:
                await integration_manager.initialize_component(name)
                logger.info(f"  ✓ {name} v{version} initialized")
            else:
                logger.warning(f"  ⚠ {name} v{version} failed to initialize")
        
        # Initialize monitoring
        await system_monitor.start_monitoring()
        logger.info("  ✓ System monitoring started")
        
        logger.info("System initialization completed")
    
    async def _run_health_checks(self):
        """Run comprehensive health checks."""
        logger.info("🏥 Running comprehensive health checks...")
        
        # Run health check framework
        diagnostic_report = await health_check_framework.run_comprehensive_diagnostics()
        
        logger.info(f"  Overall health status: {diagnostic_report.overall_status.value}")
        logger.info(f"  Total checks: {len(diagnostic_report.health_checks)}")
        logger.info(f"  Critical issues: {len(diagnostic_report.critical_issues)}")
        
        if diagnostic_report.critical_issues:
            logger.warning("  ⚠ Critical issues found:")
            for issue in diagnostic_report.critical_issues:
                logger.warning(f"    - {issue}")
        
        self.results["health_checks"] = {
            "overall_status": diagnostic_report.overall_status.value,
            "total_checks": len(diagnostic_report.health_checks),
            "critical_issues": len(diagnostic_report.critical_issues),
            "recommendations": diagnostic_report.recommendations
        }
        
        # Export diagnostic report
        report_file = health_check_framework.export_diagnostic_report(diagnostic_report)
        logger.info(f"  📄 Health report exported: {report_file}")
    
    async def _run_test_suites(self):
        """Run unit and integration test suites."""
        logger.info("🧪 Running test suites...")
        
        # Discover tests
        test_manager.discover_tests()
        logger.info(f"  Discovered {len(test_manager.test_suites)} test suites")
        
        # Run all tests
        test_report = await test_manager.run_all_tests()
        
        logger.info(f"  Total tests: {test_report['total_tests']}")
        logger.info(f"  Passed: {test_report['test_summary']['passed']}")
        logger.info(f"  Failed: {test_report['test_summary']['failed']}")
        logger.info(f"  Execution time: {test_report['total_duration']:.2f} seconds")
        
        self.results["test_suites"] = test_report
    
    async def _run_performance_tests(self):
        """Run performance tests."""
        logger.info("⚡ Running performance tests...")
        
        # Configure performance tests
        from testing.performance_tests import PerformanceTestConfig, LoadTestType
        
        # Add performance test configurations
        performance_configs = [
            PerformanceTestConfig(
                test_name="api_response_time",
                test_type=LoadTestType.STEADY_STATE,
                target_url="http://localhost:8000/health",
                concurrent_users=10,
                test_duration=30,
                max_response_time=2.0
            ),
            PerformanceTestConfig(
                test_name="load_test",
                test_type=LoadTestType.RAMP_UP,
                target_url="http://localhost:8000/api/market-data",
                concurrent_users=50,
                test_duration=60,
                ramp_up_time=30
            )
        ]
        
        for config in performance_configs:
            performance_test_suite.add_test_config(config)
        
        # Run performance tests
        perf_report = await performance_test_suite.run_all_performance_tests()
        
        logger.info(f"  Performance tests completed: {perf_report['total_tests']}")
        logger.info(f"  Average throughput: {perf_report['performance_summary']['average_throughput']:.2f} req/s")
        logger.info(f"  Performance score: {perf_report['performance_summary']['performance_score']:.1f}/100")
        
        self.results["performance_tests"] = perf_report
    
    async def _run_security_tests(self):
        """Run security tests."""
        logger.info("🔒 Running security tests...")
        
        # Run comprehensive security scan
        security_report = await security_test_suite.run_comprehensive_security_scan()
        
        logger.info(f"  Security tests completed: {security_report['total_tests']}")
        logger.info(f"  Critical vulnerabilities: {security_report['findings_by_severity']['critical']}")
        logger.info(f"  High vulnerabilities: {security_report['findings_by_severity']['high']}")
        logger.info(f"  Security score: {security_report['security_score']:.1f}/100")
        
        self.results["security_tests"] = security_report
    
    async def _run_uat_tests(self):
        """Run user acceptance tests."""
        logger.info("👥 Running user acceptance tests...")
        
        # Load sample UAT scenarios
        uat_scenarios = create_trading_system_uat_scenarios()
        for scenario in uat_scenarios:
            uat_framework.register_uat_scenario(scenario)
        
        # Run all UAT tests
        uat_report = await uat_framework.run_all_uat_tests()
        
        logger.info(f"  UAT scenarios completed: {uat_report['total_scenarios']}")
        logger.info(f"  Success rate: {uat_report['success_rate']:.1f}%")
        logger.info(f"  Overall status: {uat_report['overall_status']}")
        
        self.results["uat_tests"] = uat_report
    
    async def _run_final_validation(self):
        """Run final integration validation."""
        logger.info("🎯 Running final integration validation...")
        
        # Run complete validation
        readiness_assessment = await final_validator.run_complete_validation()
        
        logger.info(f"  Production readiness level: {readiness_assessment.overall_level.value}")
        logger.info(f"  Readiness score: {readiness_assessment.readiness_score:.1f}%")
        logger.info(f"  Critical issues: {len(readiness_assessment.critical_issues)}")
        
        if readiness_assessment.critical_issues:
            logger.warning("  ⚠ Critical issues found:")
            for issue in readiness_assessment.critical_issues[:5]:  # Show first 5
                logger.warning(f"    - {issue}")
        
        self.results["final_validation"] = {
            "readiness_level": readiness_assessment.overall_level.value,
            "readiness_score": readiness_assessment.readiness_score,
            "critical_issues": readiness_assessment.critical_issues,
            "recommendations": readiness_assessment.recommendations
        }
        
        # Export readiness report
        report_file = final_validator.export_readiness_report(readiness_assessment)
        logger.info(f"  📄 Readiness report exported: {report_file}")
    
    async def _generate_documentation(self):
        """Generate comprehensive documentation."""
        logger.info("📚 Generating documentation...")
        
        # Generate complete documentation package
        doc_files = doc_generator.generate_complete_documentation()
        
        logger.info(f"  Generated {len(doc_files)} documentation files:")
        for doc_type, file_path in doc_files.items():
            logger.info(f"    - {doc_type}: {file_path}")
        
        self.results["documentation"] = doc_files
    
    async def _generate_final_report(self):
        """Generate comprehensive final report."""
        logger.info("📊 Generating comprehensive final report...")
        
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        final_report = {
            "timestamp": end_time.isoformat(),
            "start_time": self.start_time.isoformat(),
            "total_duration_seconds": total_duration,
            "execution_summary": {
                "health_checks": self.results.get("health_checks", {}),
                "test_suites": {
                    "total_tests": self.results.get("test_suites", {}).get("total_tests", 0),
                    "passed": self.results.get("test_suites", {}).get("test_summary", {}).get("passed", 0),
                    "failed": self.results.get("test_suites", {}).get("test_summary", {}).get("failed", 0)
                },
                "performance_tests": {
                    "tests_run": self.results.get("performance_tests", {}).get("total_tests", 0),
                    "performance_score": self.results.get("performance_tests", {}).get("performance_summary", {}).get("performance_score", 0)
                },
                "security_tests": {
                    "tests_run": self.results.get("security_tests", {}).get("total_tests", 0),
                    "security_score": self.results.get("security_tests", {}).get("security_score", 0),
                    "critical_vulnerabilities": self.results.get("security_tests", {}).get("findings_by_severity", {}).get("critical", 0)
                },
                "uat_tests": {
                    "scenarios_run": self.results.get("uat_tests", {}).get("total_scenarios", 0),
                    "success_rate": self.results.get("uat_tests", {}).get("success_rate", 0),
                    "overall_status": self.results.get("uat_tests", {}).get("overall_status", "unknown")
                },
                "final_validation": {
                    "readiness_level": self.results.get("final_validation", {}).get("readiness_level", "unknown"),
                    "readiness_score": self.results.get("final_validation", {}).get("readiness_score", 0)
                }
            },
            "overall_assessment": self._calculate_overall_assessment(),
            "recommendations": self._generate_overall_recommendations()
        }
        
        # Save final report
        report_file = f"system_integration_report_{end_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(final_report, f, indent=2)
        
        logger.info(f"  📄 Final report saved: {report_file}")
        
        # Print summary
        self._print_summary(final_report)
    
    def _calculate_overall_assessment(self) -> dict:
        """Calculate overall system assessment."""
        scores = []
        
        # Health check score
        if "health_checks" in self.results:
            health_score = 100 - (self.results["health_checks"].get("critical_issues", 0) * 20)
            scores.append(max(0, health_score))
        
        # Test suite score
        if "test_suites" in self.results:
            test_summary = self.results["test_suites"].get("test_summary", {})
            total_tests = test_summary.get("passed", 0) + test_summary.get("failed", 0)
            if total_tests > 0:
                test_score = (test_summary.get("passed", 0) / total_tests) * 100
                scores.append(test_score)
        
        # Performance score
        if "performance_tests" in self.results:
            perf_score = self.results["performance_tests"].get("performance_summary", {}).get("performance_score", 0)
            scores.append(perf_score)
        
        # Security score
        if "security_tests" in self.results:
            security_score = self.results["security_tests"].get("security_score", 0)
            scores.append(security_score)
        
        # UAT score
        if "uat_tests" in self.results:
            uat_score = self.results["uat_tests"].get("success_rate", 0)
            scores.append(uat_score)
        
        # Final validation score
        if "final_validation" in self.results:
            validation_score = self.results["final_validation"].get("readiness_score", 0)
            scores.append(validation_score)
        
        if scores:
            overall_score = sum(scores) / len(scores)
            if overall_score >= 90:
                status = "excellent"
            elif overall_score >= 80:
                status = "good"
            elif overall_score >= 70:
                status = "acceptable"
            else:
                status = "needs_improvement"
        else:
            overall_score = 0
            status = "unknown"
        
        return {
            "overall_score": overall_score,
            "status": status,
            "component_scores": {
                "health": scores[0] if len(scores) > 0 else 0,
                "tests": scores[1] if len(scores) > 1 else 0,
                "performance": scores[2] if len(scores) > 2 else 0,
                "security": scores[3] if len(scores) > 3 else 0,
                "uat": scores[4] if len(scores) > 4 else 0,
                "validation": scores[5] if len(scores) > 5 else 0
            }
        }
    
    def _generate_overall_recommendations(self) -> list:
        """Generate overall system recommendations."""
        recommendations = []
        
        # Health-based recommendations
        if "health_checks" in self.results:
            critical_issues = self.results["health_checks"].get("critical_issues", 0)
            if critical_issues > 0:
                recommendations.append(f"Address {critical_issues} critical health issues before production deployment")
        
        # Security recommendations
        if "security_tests" in self.results:
            critical_vulns = self.results["security_tests"].get("findings_by_severity", {}).get("critical", 0)
            if critical_vulns > 0:
                recommendations.append(f"Fix {critical_vulns} critical security vulnerabilities immediately")
        
        # Test-based recommendations
        if "test_suites" in self.results:
            failed_tests = self.results["test_suites"].get("test_summary", {}).get("failed", 0)
            if failed_tests > 0:
                recommendations.append(f"Address {failed_tests} failed tests to ensure system reliability")
        
        # Performance recommendations
        if "performance_tests" in self.results:
            perf_score = self.results["performance_tests"].get("performance_summary", {}).get("performance_score", 0)
            if perf_score < 80:
                recommendations.append("Optimize system performance to meet production requirements")
        
        # Final validation recommendations
        if "final_validation" in self.results:
            readiness_score = self.results["final_validation"].get("readiness_score", 0)
            if readiness_score < 85:
                recommendations.append("Improve system readiness score before production deployment")
        
        if not recommendations:
            recommendations.append("System appears ready for production deployment")
        
        return recommendations
    
    def _print_summary(self, report: dict):
        """Print execution summary."""
        print("\n" + "="*80)
        print("🚀 SYSTEM INTEGRATION AND TESTING SUMMARY")
        print("="*80)
        
        print(f"Execution Time: {report['total_duration_seconds']:.2f} seconds")
        print(f"Overall Assessment: {report['overall_assessment']['status'].upper()}")
        print(f"Overall Score: {report['overall_assessment']['overall_score']:.1f}/100")
        
        print("\n📊 Component Scores:")
        for component, score in report['overall_assessment']['component_scores'].items():
            print(f"  {component.capitalize()}: {score:.1f}/100")
        
        print("\n🏥 Health Checks:")
        health = report['execution_summary']['health_checks']
        print(f"  Status: {health.get('overall_status', 'unknown')}")
        print(f"  Total Checks: {health.get('total_checks', 0)}")
        print(f"  Critical Issues: {health.get('critical_issues', 0)}")
        
        print("\n🧪 Test Suites:")
        tests = report['execution_summary']['test_suites']
        print(f"  Total Tests: {tests['total_tests']}")
        print(f"  Passed: {tests['passed']}")
        print(f"  Failed: {tests['failed']}")
        
        print("\n⚡ Performance Tests:")
        perf = report['execution_summary']['performance_tests']
        print(f"  Tests Run: {perf['tests_run']}")
        print(f"  Performance Score: {perf['performance_score']:.1f}/100")
        
        print("\n🔒 Security Tests:")
        security = report['execution_summary']['security_tests']
        print(f"  Tests Run: {security['tests_run']}")
        print(f"  Security Score: {security['security_score']:.1f}/100")
        print(f"  Critical Vulnerabilities: {security['critical_vulnerabilities']}")
        
        print("\n👥 UAT Tests:")
        uat = report['execution_summary']['uat_tests']
        print(f"  Scenarios Run: {uat['scenarios_run']}")
        print(f"  Success Rate: {uat['success_rate']:.1f}%")
        print(f"  Overall Status: {uat['overall_status']}")
        
        print("\n🎯 Final Validation:")
        validation = report['execution_summary']['final_validation']
        print(f"  Readiness Level: {validation['readiness_level']}")
        print(f"  Readiness Score: {validation['readiness_score']:.1f}/100")
        
        print("\n💡 Recommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print("\n" + "="*80)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="System Integration and Testing Runner")
    parser.add_argument("--mode", choices=["full", "health", "tests", "performance", "security", "uat", "validation"], 
                       default="full", help="Test mode to run")
    parser.add_argument("--output-dir", default="test_results", help="Output directory for results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Change to output directory
    import os
    os.chdir(args.output_dir)
    
    # Initialize and run tester
    tester = SystemIntegrationTester()
    
    if args.mode == "full":
        await tester.run_complete_integration_test()
    elif args.mode == "health":
        await tester._initialize_system()
        await tester._run_health_checks()
    elif args.mode == "tests":
        await tester._initialize_system()
        await tester._run_test_suites()
    elif args.mode == "performance":
        await tester._initialize_system()
        await tester._run_performance_tests()
    elif args.mode == "security":
        await tester._initialize_system()
        await tester._run_security_tests()
    elif args.mode == "uat":
        await tester._initialize_system()
        await tester._run_uat_tests()
    elif args.mode == "validation":
        await tester._initialize_system()
        await tester._run_final_validation()
    
    print(f"\n✅ {args.mode.title()} testing completed! Results saved in: {args.output_dir}")


if __name__ == "__main__":
    asyncio.run(main())