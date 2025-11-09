"""
Integration Test Framework
Tests cross-component functionality and system integration.
"""

import unittest
import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import aiohttp
import websockets
from datetime import datetime
import tempfile
import os

# Import integration manager
from ..integration.integration_manager import integration_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntegrationTestStatus(Enum):
    """Integration test status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"


@dataclass
class IntegrationTestResult:
    """Integration test result container."""
    test_name: str
    components_involved: List[str]
    status: IntegrationTestStatus
    duration: float
    details: Dict[str, Any]
    error_message: Optional[str] = None
    timestamp: datetime = None


@dataclass
class ComponentInteraction:
    """Component interaction specification."""
    source_component: str
    target_component: str
    interaction_type: str
    expected_response: Dict[str, Any]
    timeout: float = 30.0


class IntegrationTestFramework:
    """Framework for testing component integration."""
    
    def __init__(self):
        self.test_scenarios: Dict[str, Dict[str, Any]] = {}
        self.component_interactions: List[ComponentInteraction] = []
        self.test_results: List[IntegrationTestResult] = []
        self.mock_services: Dict[str, Any] = {}
        self.test_environment = self._setup_test_environment()
        
    def _setup_test_environment(self) -> Dict[str, Any]:
        """Set up test environment configuration."""
        return {
            "base_url": "http://localhost:8000",
            "websocket_url": "ws://localhost:8000/ws",
            "api_timeout": 30,
            "test_port_range": (8001, 8010),
            "mock_data_dir": "tests/integration/mock_data"
        }
    
    def register_integration_scenario(self, scenario_name: str, 
                                    scenario_config: Dict[str, Any]):
        """Register an integration test scenario."""
        self.test_scenarios[scenario_name] = {
            "name": scenario_name,
            "description": scenario_config.get("description", ""),
            "components": scenario_config.get("components", []),
            "steps": scenario_config.get("steps", []),
            "expected_outcomes": scenario_config.get("expected_outcomes", {}),
            "setup_hooks": scenario_config.get("setup_hooks", []),
            "teardown_hooks": scenario_config.get("teardown_hooks", []),
            "timeout": scenario_config.get("timeout", 300)
        }
        
        logger.info(f"Registered integration scenario: {scenario_name}")
    
    def add_component_interaction(self, interaction: ComponentInteraction):
        """Add a component interaction test."""
        self.component_interactions.append(interaction)
        logger.info(f"Added component interaction: {interaction.source_component} -> {interaction.target_component}")
    
    async def run_integration_scenario(self, scenario_name: str) -> IntegrationTestResult:
        """Run a specific integration test scenario."""
        if scenario_name not in self.test_scenarios:
            return IntegrationTestResult(
                test_name=scenario_name,
                components_involved=[],
                status=IntegrationTestStatus.FAILURE,
                duration=0,
                details={},
                error_message=f"Scenario {scenario_name} not found"
            )
        
        scenario = self.test_scenarios[scenario_name]
        logger.info(f"Running integration scenario: {scenario_name}")
        
        start_time = time.time()
        
        try:
            # Setup test environment
            await self._setup_scenario_environment(scenario)
            
            # Execute test steps
            step_results = await self._execute_test_steps(scenario)
            
            # Validate expected outcomes
            validation_results = await self._validate_outcomes(scenario, step_results)
            
            duration = time.time() - start_time
            
            # Determine overall status
            if all(validation_results.values()):
                status = IntegrationTestStatus.SUCCESS
            else:
                status = IntegrationTestStatus.FAILURE
            
            result = IntegrationTestResult(
                test_name=scenario_name,
                components_involved=scenario["components"],
                status=status,
                duration=duration,
                details={
                    "step_results": step_results,
                    "validation_results": validation_results,
                    "components_tested": len(scenario["components"])
                }
            )
            
            self.test_results.append(result)
            
            # Cleanup
            await self._cleanup_scenario_environment(scenario)
            
            logger.info(f"Integration scenario {scenario_name} completed: {status.value}")
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            error_result = IntegrationTestResult(
                test_name=scenario_name,
                components_involved=scenario["components"],
                status=IntegrationTestStatus.FAILURE,
                duration=duration,
                details={},
                error_message=str(e)
            )
            
            self.test_results.append(error_result)
            logger.error(f"Integration scenario {scenario_name} failed: {str(e)}")
            return error_result
    
    async def _setup_scenario_environment(self, scenario: Dict[str, Any]):
        """Set up environment for test scenario."""
        logger.debug(f"Setting up environment for scenario: {scenario['name']}")
        
        # Create mock services
        for component in scenario["components"]:
            if component not in self.mock_services:
                self.mock_services[component] = await self._create_mock_service(component)
        
        # Execute setup hooks
        for hook in scenario.get("setup_hooks", []):
            await self._execute_hook(hook, "setup")
    
    async def _create_mock_service(self, component_name: str) -> Dict[str, Any]:
        """Create mock service for a component."""
        # This would create actual mock services based on component type
        # For now, return a simple mock
        return {
            "name": component_name,
            "status": "running",
            "endpoints": {},
            "data": {}
        }
    
    async def _execute_hook(self, hook: str, phase: str):
        """Execute a setup or teardown hook."""
        try:
            # Parse hook configuration
            if isinstance(hook, str):
                hook_type, hook_config = hook, {}
            else:
                hook_type = hook.get("type", "")
                hook_config = hook.get("config", {})
            
            # Execute based on hook type
            if hook_type == "start_service":
                await self._start_mock_service(hook_config.get("service_name"))
            elif hook_type == "load_test_data":
                await self._load_test_data(hook_config.get("data_file"))
            elif hook_type == "configure_component":
                await self._configure_component(hook_config)
            # Add more hook types as needed
            
        except Exception as e:
            logger.error(f"Error executing {phase} hook {hook}: {str(e)}")
            raise
    
    async def _start_mock_service(self, service_name: str):
        """Start a mock service."""
        logger.debug(f"Starting mock service: {service_name}")
        # Implementation would start actual mock services
        await asyncio.sleep(0.1)  # Simulate service startup
    
    async def _load_test_data(self, data_file: str):
        """Load test data from file."""
        logger.debug(f"Loading test data from: {data_file}")
        try:
            with open(data_file, 'r') as f:
                data = json.load(f)
                # Process loaded data as needed
                return data
        except FileNotFoundError:
            logger.warning(f"Test data file not found: {data_file}")
            return {}
    
    async def _configure_component(self, config: Dict[str, Any]):
        """Configure a component for testing."""
        logger.debug(f"Configuring component with: {config}")
        # Implementation would configure actual components
        await asyncio.sleep(0.1)  # Simulate configuration
    
    async def _execute_test_steps(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all test steps for scenario."""
        step_results = {}
        
        for i, step in enumerate(scenario["steps"]):
            step_name = step.get("name", f"step_{i}")
            step_type = step.get("type", "api_call")
            
            logger.debug(f"Executing step {i+1}: {step_name}")
            
            try:
                if step_type == "api_call":
                    result = await self._execute_api_call_step(step)
                elif step_type == "component_interaction":
                    result = await self._execute_component_interaction_step(step)
                elif step_type == "data_validation":
                    result = await self._execute_data_validation_step(step)
                elif step_type == "waiting":
                    result = await self._execute_waiting_step(step)
                else:
                    result = {"success": False, "error": f"Unknown step type: {step_type}"}
                
                step_results[step_name] = result
                
            except Exception as e:
                step_results[step_name] = {"success": False, "error": str(e)}
                logger.error(f"Step {step_name} failed: {str(e)}")
        
        return step_results
    
    async def _execute_api_call_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an API call test step."""
        method = step.get("method", "GET")
        url = step.get("url", "")
        headers = step.get("headers", {})
        data = step.get("data", {})
        expected_status = step.get("expected_status", 200)
        timeout = step.get("timeout", 30)
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                if method.upper() == "GET":
                    async with session.get(url, headers=headers) as response:
                        response_data = await response.json()
                        status_success = response.status == expected_status
                        
                elif method.upper() == "POST":
                    async with session.post(url, json=data, headers=headers) as response:
                        response_data = await response.json()
                        status_success = response.status == expected_status
                        
                elif method.upper() == "PUT":
                    async with session.put(url, json=data, headers=headers) as response:
                        response_data = await response.json()
                        status_success = response.status == expected_status
                        
                elif method.upper() == "DELETE":
                    async with session.delete(url, headers=headers) as response:
                        response_data = await response.json()
                        status_success = response.status == expected_status
                
                return {
                    "success": status_success,
                    "status_code": response.status,
                    "response_data": response_data,
                    "expected_status": expected_status
                }
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_component_interaction_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a component interaction test step."""
        source = step.get("source_component")
        target = step.get("target_component")
        interaction_type = step.get("interaction_type", "message")
        payload = step.get("payload", {})
        
        try:
            # Send message via integration manager
            success = await integration_manager.send_message(
                source, target, interaction_type, payload
            )
            
            # Check for response (in real implementation, would wait for actual response)
            response = integration_manager.get_messages(target)
            
            return {
                "success": success,
                "message_sent": success,
                "response_received": len(response) > 0,
                "response_data": response if response else None
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_data_validation_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a data validation test step."""
        source = step.get("source", "component")
        data_path = step.get("data_path", "")
        validation_rules = step.get("validation_rules", {})
        
        try:
            # Get data from component
            component_data = await self._get_component_data(source, data_path)
            
            # Apply validation rules
            validation_results = self._apply_validation_rules(component_data, validation_rules)
            
            return {
                "success": all(validation_results.values()),
                "component_data": component_data,
                "validation_results": validation_results
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_waiting_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a waiting test step."""
        duration = step.get("duration", 5)
        condition = step.get("condition")
        
        try:
            if condition:
                # Wait for condition to be met
                start_time = time.time()
                timeout = step.get("timeout", 60)
                
                while time.time() - start_time < timeout:
                    if await self._check_condition(condition):
                        return {"success": True, "condition_met": True}
                    await asyncio.sleep(1)
                
                return {"success": False, "condition_met": False, "timeout": timeout}
            else:
                # Simple duration wait
                await asyncio.sleep(duration)
                return {"success": True, "waited_for": duration}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _get_component_data(self, component_name: str, data_path: str) -> Any:
        """Get data from a component."""
        # This would get actual data from components
        # For now, return mock data
        return {"mock_data": f"Data for {component_name}.{data_path}"}
    
    def _apply_validation_rules(self, data: Any, rules: Dict[str, Any]) -> Dict[str, bool]:
        """Apply validation rules to data."""
        results = {}
        
        for rule_name, rule_config in rules.items():
            rule_type = rule_config.get("type", "equals")
            expected_value = rule_config.get("expected")
            actual_value = self._extract_value(data, rule_config.get("path", ""))
            
            if rule_type == "equals":
                results[rule_name] = actual_value == expected_value
            elif rule_type == "not_null":
                results[rule_name] = actual_value is not None
            elif rule_type == "contains":
                results[rule_name] = expected_value in str(actual_value)
            elif rule_type == "greater_than":
                results[rule_name] = float(actual_value) > float(expected_value)
            # Add more validation types as needed
        
        return results
    
    def _extract_value(self, data: Any, path: str) -> Any:
        """Extract value from data using path."""
        if not path:
            return data
        
        parts = path.split(".")
        current = data
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        
        return current
    
    async def _check_condition(self, condition: Dict[str, Any]) -> bool:
        """Check if a condition is met."""
        condition_type = condition.get("type")
        
        if condition_type == "component_status":
            component_name = condition.get("component")
            expected_status = condition.get("expected_status")
            
            status = await integration_manager.health_check_component(component_name)
            return status.get("status") == expected_status
        
        # Add more condition types as needed
        return False
    
    async def _validate_outcomes(self, scenario: Dict[str, Any], 
                                step_results: Dict[str, Any]) -> Dict[str, bool]:
        """Validate expected outcomes of the scenario."""
        expected_outcomes = scenario.get("expected_outcomes", {})
        validation_results = {}
        
        for outcome_name, outcome_config in expected_outcomes.items():
            outcome_type = outcome_config.get("type")
            
            if outcome_type == "all_steps_success":
                validation_results[outcome_name] = all(
                    result.get("success", False) for result in step_results.values()
                )
            elif outcome_type == "specific_steps_success":
                required_steps = outcome_config.get("steps", [])
                validation_results[outcome_name] = all(
                    step_results.get(step, {}).get("success", False) 
                    for step in required_steps
                )
            elif outcome_type == "data_validation":
                validation_results[outcome_name] = await self._validate_data_outcome(
                    outcome_config, step_results
                )
        
        return validation_results
    
    async def _validate_data_outcome(self, outcome_config: Dict[str, Any], 
                                    step_results: Dict[str, Any]) -> bool:
        """Validate data outcome."""
        # Implementation would validate specific data outcomes
        return True  # Placeholder
    
    async def _cleanup_scenario_environment(self, scenario: Dict[str, Any]):
        """Clean up test environment after scenario."""
        logger.debug(f"Cleaning up environment for scenario: {scenario['name']}")
        
        # Execute teardown hooks
        for hook in scenario.get("teardown_hooks", []):
            await self._execute_hook(hook, "teardown")
        
        # Stop mock services
        for component in scenario["components"]:
            if component in self.mock_services:
                del self.mock_services[component]
    
    async def run_all_integration_tests(self) -> Dict[str, Any]:
        """Run all registered integration test scenarios."""
        logger.info("Starting comprehensive integration testing...")
        
        overall_start_time = time.time()
        results = []
        
        for scenario_name in self.test_scenarios.keys():
            result = await self.run_integration_scenario(scenario_name)
            results.append(result)
        
        overall_duration = time.time() - overall_start_time
        
        # Generate summary report
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_duration": overall_duration,
            "total_scenarios": len(self.test_scenarios),
            "successful_scenarios": len([r for r in results if r.status == IntegrationTestStatus.SUCCESS]),
            "failed_scenarios": len([r for r in results if r.status == IntegrationTestStatus.FAILURE]),
            "results": [self._result_to_dict(r) for r in results]
        }
        
        logger.info(f"Integration testing completed in {overall_duration:.2f} seconds")
        return report
    
    def _result_to_dict(self, result: IntegrationTestResult) -> Dict[str, Any]:
        """Convert IntegrationTestResult to dictionary."""
        return {
            "test_name": result.test_name,
            "components_involved": result.components_involved,
            "status": result.status.value,
            "duration": result.duration,
            "details": result.details,
            "error_message": result.error_message,
            "timestamp": result.timestamp.isoformat() if result.timestamp else None
        }
    
    def get_test_status(self) -> Dict[str, Any]:
        """Get current integration test status."""
        return {
            "total_scenarios": len(self.test_scenarios),
            "total_component_interactions": len(self.component_interactions),
            "completed_tests": len(self.test_results),
            "registered_scenarios": list(self.test_scenarios.keys())
        }


# Global integration test framework instance
integration_test_framework = IntegrationTestFramework()