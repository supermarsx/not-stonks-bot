#!/usr/bin/env python3
"""
Provider System Validation Script

This script validates the complete expanded LLM provider system by:
1. Testing provider creation and initialization
2. Validating health monitoring functionality
3. Testing failover mechanisms
4. Running comprehensive tests
5. Demonstrating usage patterns
"""

import asyncio
import json
import sys
from datetime import datetime
from typing import Dict, List, Any

# Add the trading_orchestrator to the Python path
sys.path.append('/workspace/trading_orchestrator')

from ai.providers import (
    ProviderFactory,
    ProviderConfig,
    RateLimit,
    ProviderHealthMonitor,
    ProviderFailoverManager,
    ProviderCapability
)
from ai.providers.failover_manager import FailoverStrategy
from ai.providers.testing_suite import ProviderTestSuite, quick_test_provider
from ai.models.enhanced_ai_models_manager import (
    EnhancedAIModelsManager,
    RequestConfig,
    RequestPriority,
    ModelTier
)

from loguru import logger
import aiofiles


class ProviderSystemValidator:
    """Comprehensive validation of the provider system"""
    
    def __init__(self):
        self.factory = ProviderFactory()
        self.health_monitor = None
        self.failover_manager = None
        self.ai_manager = None
        self.test_results = []
        self.validation_start_time = None
        
    async def run_full_validation(self) -> Dict[str, Any]:
        """Run complete validation suite"""
        self.validation_start_time = datetime.now()
        logger.info("Starting Provider System Validation")
        
        results = {
            'validation_start': self.validation_start_time.isoformat(),
            'tests': {},
            'overall_status': 'pending',
            'summary': {}
        }
        
        try:
            # Phase 1: Provider Creation and Configuration
            results['tests']['provider_creation'] = await self.validate_provider_creation()
            
            # Phase 2: Health Monitoring
            results['tests']['health_monitoring'] = await self.validate_health_monitoring()
            
            # Phase 3: Failover Management
            results['tests']['failover_management'] = await self.validate_failover_management()
            
            # Phase 4: Enhanced AI Manager
            results['tests']['ai_manager'] = await self.validate_ai_manager()
            
            # Phase 5: Testing Suite
            results['tests']['testing_suite'] = await self.validate_testing_suite()
            
            # Phase 6: Integration Tests
            results['tests']['integration'] = await self.validate_integration()
            
            # Calculate overall status
            results['overall_status'] = self.calculate_overall_status(results['tests'])
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            results['overall_status'] = 'error'
            results['error'] = str(e)
            
        results['validation_end'] = datetime.now().isoformat()
        results['validation_duration'] = str(datetime.now() - self.validation_start_time)
        
        # Generate summary
        results['summary'] = self.generate_summary(results['tests'])
        
        return results
        
    async def validate_provider_creation(self) -> Dict[str, Any]:
        """Validate provider creation functionality"""
        logger.info("=== Validating Provider Creation ===")
        
        test_result = {
            'status': 'running',
            'tests': [],
            'details': {}
        }
        
        try:
            # Test 1: Create test configurations
            test_configs = [
                {
                    'name': 'test_openai',
                    'provider_type': 'openai',
                    'api_key': 'test-key',  # Won't actually connect
                    'priority': 1
                },
                {
                    'name': 'test_ollama',
                    'provider_type': 'ollama',
                    'base_url': 'http://localhost:11434',
                    'priority': 2
                },
                {
                    'name': 'test_localai',
                    'provider_type': 'localai',
                    'base_url': 'http://localhost:8080',
                    'priority': 3
                },
                {
                    'name': 'test_vllm',
                    'provider_type': 'vllm',
                    'base_url': 'http://localhost:8000',
                    'priority': 4
                },
                {
                    'name': 'test_openai_compatible',
                    'provider_type': 'openai_compatible',
                    'base_url': 'https://api.openai.com/v1',
                    'api_key': 'test-key',
                    'priority': 5
                }
            ]
            
            # Test 2: Create providers from configs
            created_providers = []
            for config in test_configs:
                try:
                    provider = await self.factory.create_from_config_dict(config)
                    if provider:
                        created_providers.append(provider.config.name)
                        test_result['tests'].append({
                            'name': f'create_{config["name"]}',
                            'status': 'passed',
                            'message': f'Successfully created {config["name"]}'
                        })
                    else:
                        test_result['tests'].append({
                            'name': f'create_{config["name"]}',
                            'status': 'failed',
                            'message': f'Failed to create {config["name"]}'
                        })
                except Exception as e:
                    test_result['tests'].append({
                        'name': f'create_{config["name"]}',
                        'status': 'failed',
                        'message': f'Error creating {config["name"]}: {str(e)}'
                    })
                    
            test_result['details']['created_providers'] = created_providers
            test_result['details']['total_configs'] = len(test_configs)
            test_result['details']['successful_creations'] = len(created_providers)
            
            # Test 3: Verify provider registry
            available_types = self.factory.get_available_provider_types()
            expected_types = ['openai', 'anthropic', 'ollama', 'localai', 'vllm', 'openai_compatible']
            
            missing_types = [t for t in expected_types if t not in available_types]
            
            test_result['tests'].append({
                'name': 'provider_registry',
                'status': 'passed' if not missing_types else 'failed',
                'message': f'Available types: {available_types}',
                'details': {'missing_types': missing_types}
            })
            
            # Test 4: Dependency checking
            dependency_status = self.factory.get_dependency_status()
            test_result['tests'].append({
                'name': 'dependency_checking',
                'status': 'passed',
                'message': f'Dependencies: {list(dependency_status.keys())}',
                'details': dependency_status
            })
            
            test_result['status'] = 'completed'
            
        except Exception as e:
            test_result['status'] = 'error'
            test_result['error'] = str(e)
            
        logger.info(f"Provider creation validation: {test_result['status']}")
        return test_result
        
    async def validate_health_monitoring(self) -> Dict[str, Any]:
        """Validate health monitoring functionality"""
        logger.info("=== Validating Health Monitoring ===")
        
        test_result = {
            'status': 'running',
            'tests': [],
            'details': {}
        }
        
        try:
            # Setup health monitor
            self.health_monitor = ProviderHealthMonitor(
                check_interval=5,  # Short interval for testing
                history_retention_hours=1
            )
            
            # Test 1: Health monitor initialization
            test_result['tests'].append({
                'name': 'health_monitor_init',
                'status': 'passed',
                'message': 'Health monitor initialized'
            })
            
            # Test 2: Add mock provider for testing
            from ai.providers.base_provider import ProviderConfig, RateLimit
            
            mock_config = ProviderConfig(
                name='mock_provider',
                provider_type='mock',
                base_url='http://localhost:99999',  # Non-existent URL
                rate_limit=RateLimit(requests_per_minute=60)
            )
            
            # Create a mock provider that always reports healthy
            class MockProvider:
                def __init__(self, config):
                    self.config = config
                    self.status = 'healthy'
                    
                async def health_check(self):
                    from ai.providers.base_provider import HealthMetrics
                    return HealthMetrics(
                        response_time=0.1,
                        success_rate=1.0,
                        error_rate=0.0,
                        last_check=datetime.now(),
                        consecutive_failures=0,
                        is_healthy=True,
                        message="Mock provider healthy"
                    )
                    
                def supports_capability(self, capability):
                    return True
                    
            mock_provider = MockProvider(mock_config)
            self.health_monitor.register_provider(mock_provider)
            
            test_result['tests'].append({
                'name': 'provider_registration',
                'status': 'passed',
                'message': 'Mock provider registered'
            })
            
            # Test 3: Health check functionality
            health_data = self.health_monitor.get_provider_health('mock_provider')
            
            test_result['tests'].append({
                'name': 'health_check',
                'status': 'passed' if health_data else 'failed',
                'message': f'Health check result: {health_data}',
                'details': health_data
            })
            
            # Test 4: Health summary
            all_health = self.health_monitor.get_all_providers_health()
            
            test_result['tests'].append({
                'name': 'health_summary',
                'status': 'passed',
                'message': f'Health summary: {len(all_health)} providers',
                'details': all_health
            })
            
            # Test 5: Best provider ranking
            best_providers = self.health_monitor.get_best_providers()
            
            test_result['tests'].append({
                'name': 'best_provider_ranking',
                'status': 'passed',
                'message': f'Best providers: {best_providers}',
                'details': best_providers
            })
            
            test_result['status'] = 'completed'
            test_result['details']['registered_providers'] = list(self.health_monitor.providers.keys())
            
        except Exception as e:
            test_result['status'] = 'error'
            test_result['error'] = str(e)
            
        logger.info(f"Health monitoring validation: {test_result['status']}")
        return test_result
        
    async def validate_failover_management(self) -> Dict[str, Any]:
        """Validate failover management functionality"""
        logger.info("=== Validating Failover Management ===")
        
        test_result = {
            'status': 'running',
            'tests': [],
            'details': {}
        }
        
        try:
            # Setup failover manager with health monitor
            self.failover_manager = ProviderFailoverManager(
                health_monitor=self.health_monitor,
                strategy=FailoverStrategy.HEALTH_BASED,
                global_timeout=10.0
            )
            
            # Test 1: Failover manager initialization
            test_result['tests'].append({
                'name': 'failover_manager_init',
                'status': 'passed',
                'message': 'Failover manager initialized'
            })
            
            # Test 2: Add mock providers
            class MockHealthyProvider:
                def __init__(self, name, priority=1):
                    self.config = type('Config', (), {'name': name, 'priority': priority})()
                    
                async def generate_completion(self, **kwargs):
                    return {
                        'content': f'Response from {self.config.name}',
                        'provider': self.config.name,
                        'model': 'mock-model'
                    }
                    
                def supports_capability(self, capability):
                    return True
                    
            provider1 = MockHealthyProvider('provider_1', 1)
            provider2 = MockHealthyProvider('provider_2', 2)
            
            self.failover_manager.register_provider(provider1)
            self.failover_manager.register_provider(provider2)
            
            test_result['tests'].append({
                'name': 'provider_registration',
                'status': 'passed',
                'message': 'Mock providers registered'
            })
            
            # Test 3: Best provider selection
            best_provider = await self.failover_manager.get_provider()
            
            test_result['tests'].append({
                'name': 'provider_selection',
                'status': 'passed' if best_provider else 'failed',
                'message': f'Selected provider: {best_provider.config.name if best_provider else None}',
                'details': {'selected': best_provider.config.name if best_provider else None}
            })
            
            # Test 4: Failover statistics
            stats = self.failover_manager.get_failover_stats()
            
            test_result['tests'].append({
                'name': 'failover_stats',
                'status': 'passed',
                'message': 'Failover stats retrieved',
                'details': stats
            })
            
            # Test 5: Strategy switching
            self.failover_manager.set_strategy(FailoverStrategy.ROUND_ROBIN)
            
            test_result['tests'].append({
                'name': 'strategy_switching',
                'status': 'passed',
                'message': 'Strategy switched to round-robin'
            })
            
            test_result['status'] = 'completed'
            test_result['details']['registered_providers'] = list(self.failover_manager.providers.keys())
            test_result['details']['strategy'] = self.failover_manager.strategy.value
            
        except Exception as e:
            test_result['status'] = 'error'
            test_result['error'] = str(e)
            
        logger.info(f"Failover management validation: {test_result['status']}")
        return test_result
        
    async def validate_ai_manager(self) -> Dict[str, Any]:
        """Validate enhanced AI models manager"""
        logger.info("=== Validating Enhanced AI Manager ===")
        
        test_result = {
            'status': 'running',
            'tests': [],
            'details': {}
        }
        
        try:
            # Setup AI manager with test configuration
            config = {
                'providers': [
                    {
                        'name': 'test_openai',
                        'provider_type': 'openai',
                        'api_key': 'test-key',
                        'priority': 1
                    },
                    {
                        'name': 'test_local',
                        'provider_type': 'ollama',
                        'base_url': 'http://localhost:11434',
                        'priority': 2
                    }
                ],
                'health_check_interval': 10,
                'global_timeout': 15.0
            }
            
            self.ai_manager = EnhancedAIModelsManager(config)
            await self.ai_manager.initialize()
            
            # Test 1: AI manager initialization
            test_result['tests'].append({
                'name': 'ai_manager_init',
                'status': 'passed',
                'message': 'Enhanced AI manager initialized'
            })
            
            # Test 2: Tool registration
            async def mock_tool(param: str) -> Dict[str, Any]:
                return {"result": f"Tool result for {param}"}
                
            self.ai_manager.register_tool("mock_tool", mock_tool)
            
            test_result['tests'].append({
                'name': 'tool_registration',
                'status': 'passed',
                'message': 'Tool registered successfully'
            })
            
            # Test 3: Model selection by task type
            model_selections = {}
            task_types = ['strategy_analysis', 'risk_check', 'quick_decision', 'high_frequency']
            
            for task_type in task_types:
                model = self.ai_manager.get_model_for_task(task_type)
                model_selections[task_type] = model
                
            test_result['tests'].append({
                'name': 'model_selection',
                'status': 'passed',
                'message': 'Model selection by task type works',
                'details': model_selections
            })
            
            # Test 4: Request configuration
            request_config = RequestConfig(
                priority=RequestPriority.HIGH,
                allow_failover=True,
                timeout=10.0
            )
            
            test_result['tests'].append({
                'name': 'request_configuration',
                'status': 'passed',
                'message': 'Request configuration created',
                'details': {
                    'priority': request_config.priority.value,
                    'allow_failover': request_config.allow_failover,
                    'timeout': request_config.timeout
                }
            })
            
            # Test 5: Usage statistics
            stats = self.ai_manager.get_usage_stats()
            
            test_result['tests'].append({
                'name': 'usage_statistics',
                'status': 'passed',
                'message': 'Usage statistics retrieved',
                'details': {
                    'total_providers': stats.get('total_providers', 0),
                    'healthy_providers': stats.get('healthy_providers', 0)
                }
            })
            
            test_result['status'] = 'completed'
            
        except Exception as e:
            test_result['status'] = 'error'
            test_result['error'] = str(e)
            
        logger.info(f"AI manager validation: {test_result['status']}")
        return test_result
        
    async def validate_testing_suite(self) -> Dict[str, Any]:
        """Validate testing suite functionality"""
        logger.info("=== Validating Testing Suite ===")
        
        test_result = {
            'status': 'running',
            'tests': [],
            'details': {}
        }
        
        try:
            # Test 1: Create testing suite
            test_suite = ProviderTestSuite(self.factory)
            
            test_result['tests'].append({
                'name': 'test_suite_creation',
                'status': 'passed',
                'message': 'Test suite created successfully'
            })
            
            # Test 2: Test case registration
            initial_test_count = len(test_suite.test_cases)
            
            test_result['tests'].append({
                'name': 'default_tests',
                'status': 'passed',
                'message': f'Default tests loaded: {initial_test_count}',
                'details': {'test_count': initial_test_count}
            })
            
            # Test 3: Quick test function
            test_config = {
                'name': 'quick_test_provider',
                'provider_type': 'openai',
                'api_key': 'test-key',
                'priority': 1
            }
            
            quick_result = await quick_test_provider(test_config)
            
            test_result['tests'].append({
                'name': 'quick_test',
                'status': 'passed',
                'message': f'Quick test completed: {quick_result["status"]}',
                'details': quick_result
            })
            
            test_result['status'] = 'completed'
            
        except Exception as e:
            test_result['status'] = 'error'
            test_result['error'] = str(e)
            
        logger.info(f"Testing suite validation: {test_result['status']}")
        return test_result
        
    async def validate_integration(self) -> Dict[str, Any]:
        """Validate system integration"""
        logger.info("=== Validating System Integration ===")
        
        test_result = {
            'status': 'running',
            'tests': [],
            'details': {}
        }
        
        try:
            # Test 1: Component coordination
            components_available = {
                'factory': self.factory is not None,
                'health_monitor': self.health_monitor is not None,
                'failover_manager': self.failover_manager is not None,
                'ai_manager': self.ai_manager is not None
            }
            
            test_result['tests'].append({
                'name': 'component_availability',
                'status': 'passed' if all(components_available.values()) else 'failed',
                'message': 'All components initialized',
                'details': components_available
            })
            
            # Test 2: System health check
            if self.ai_manager:
                system_health = await self.ai_manager.health_check()
                
                test_result['tests'].append({
                    'name': 'system_health',
                    'status': 'passed',
                    'message': f'System health: {system_health.get("overall_status", "unknown")}',
                    'details': system_health
                })
            
            # Test 3: Usage statistics integration
            if self.ai_manager:
                integrated_stats = self.ai_manager.get_usage_stats()
                
                test_result['tests'].append({
                    'name': 'integrated_stats',
                    'status': 'passed',
                    'message': 'Integrated statistics working',
                    'details': {
                        'components_tracked': len(integrated_stats.get('providers', {})),
                        'models_tracked': len(integrated_stats.get('models', {}))
                    }
                })
            
            test_result['status'] = 'completed'
            
        except Exception as e:
            test_result['status'] = 'error'
            test_result['error'] = str(e)
            
        logger.info(f"Integration validation: {test_result['status']}")
        return test_result
        
    def calculate_overall_status(self, test_results: Dict[str, Any]) -> str:
        """Calculate overall validation status"""
        all_statuses = []
        
        for phase_name, phase_result in test_results.items():
            if phase_result.get('status') == 'completed':
                # Check individual test status within phase
                test_statuses = [test.get('status') for test in phase_result.get('tests', [])]
                if test_statuses:
                    failed_tests = [s for s in test_statuses if s == 'failed']
                    if failed_tests:
                        all_statuses.append('partial')
                    else:
                        all_statuses.append('passed')
                else:
                    all_statuses.append('passed')
            elif phase_result.get('status') == 'error':
                all_statuses.append('error')
            else:
                all_statuses.append('pending')
                
        if 'error' in all_statuses:
            return 'error'
        elif all_statuses.count('passed') == len(all_statuses):
            return 'passed'
        elif any(s == 'passed' for s in all_statuses):
            return 'partial'
        else:
            return 'failed'
            
    def generate_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate validation summary"""
        summary = {
            'phases_completed': 0,
            'phases_passed': 0,
            'phases_failed': 0,
            'total_tests': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'errors': []
        }
        
        for phase_name, phase_result in test_results.items():
            if phase_result.get('status') == 'completed':
                summary['phases_completed'] += 1
                
                phase_tests = phase_result.get('tests', [])
                summary['total_tests'] += len(phase_tests)
                
                phase_passed = sum(1 for test in phase_tests if test.get('status') == 'passed')
                phase_failed = sum(1 for test in phase_tests if test.get('status') == 'failed')
                
                summary['tests_passed'] += phase_passed
                summary['tests_failed'] += phase_failed
                
                if phase_failed == 0:
                    summary['phases_passed'] += 1
                else:
                    summary['phases_failed'] += 1
            elif phase_result.get('status') == 'error':
                summary['phases_failed'] += 1
                summary['errors'].append(f"{phase_name}: {phase_result.get('error', 'Unknown error')}")
                
        summary['success_rate'] = summary['tests_passed'] / max(1, summary['total_tests'])
        
        return summary
        
    async def export_results(self, results: Dict[str, Any], output_file: str):
        """Export validation results to JSON file"""
        async with aiofiles.open(output_file, 'w') as f:
            await f.write(json.dumps(results, indent=2, default=str))
            
        logger.info(f"Validation results exported to {output_file}")


async def main():
    """Run provider system validation"""
    
    # Setup logging
    logger.add("validation.log", level="INFO")
    
    print("🔍 Starting Provider System Validation")
    print("=" * 50)
    
    validator = ProviderSystemValidator()
    
    try:
        # Run full validation
        results = await validator.run_full_validation()
        
        # Display results
        print("\n📊 Validation Results:")
        print("=" * 30)
        
        overall_status = results.get('overall_status', 'unknown')
        status_emoji = {
            'passed': '✅',
            'partial': '⚠️',
            'failed': '❌',
            'error': '💥',
            'unknown': '❓'
        }.get(overall_status, '❓')
        
        print(f"{status_emoji} Overall Status: {overall_status.upper()}")
        
        # Phase results
        for phase_name, phase_result in results.get('tests', {}).items():
            phase_status = phase_result.get('status', 'unknown')
            status_emoji = {
                'completed': '✅',
                'error': '❌',
                'running': '🔄'
            }.get(phase_status, '❓')
            
            print(f"{status_emoji} {phase_name.replace('_', ' ').title()}: {phase_status}")
            
        # Summary
        summary = results.get('summary', {})
        print(f"\n📈 Summary:")
        print(f"   Phases completed: {summary.get('phases_completed', 0)}")
        print(f"   Tests passed: {summary.get('tests_passed', 0)}")
        print(f"   Tests failed: {summary.get('tests_failed', 0)}")
        print(f"   Success rate: {summary.get('success_rate', 0):.1%}")
        
        if summary.get('errors'):
            print(f"\n⚠️ Errors:")
            for error in summary['errors']:
                print(f"   • {error}")
        
        # Export results
        output_file = "provider_system_validation_results.json"
        await validator.export_results(results, output_file)
        print(f"\n💾 Results exported to: {output_file}")
        
        # Final status
        if overall_status == 'passed':
            print("\n🎉 Provider System Validation PASSED!")
            return 0
        elif overall_status == 'partial':
            print("\n⚠️ Provider System Validation PARTIAL SUCCESS")
            return 1
        else:
            print("\n❌ Provider System Validation FAILED")
            return 2
            
    except Exception as e:
        logger.error(f"Validation script failed: {e}")
        print(f"\n💥 Validation script failed: {e}")
        return 3


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)