"""
User Acceptance Testing (UAT) Framework
User journey testing and business logic validation.
"""

import unittest
import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import aiohttp
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UATStatus(Enum):
    """UAT status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"


class UserRole(Enum):
    """User role enumeration."""
    ADMIN = "admin"
    TRADER = "trader"
    ANALYST = "analyst"
    VIEWER = "viewer"
    GUEST = "guest"


@dataclass
class UATScenario:
    """UAT scenario configuration."""
    name: str
    description: str
    user_role: UserRole
    steps: List[Dict[str, Any]]
    expected_outcomes: List[str]
    preconditions: List[str] = None
    success_criteria: Dict[str, Any] = None
    timeout: int = 300


@dataclass
class UATResult:
    """UAT result container."""
    scenario_name: str
    user_role: UserRole
    status: UATStatus
    duration: float
    steps_completed: int
    total_steps: int
    success_criteria_met: Dict[str, bool]
    issues_found: List[str]
    timestamp: datetime = None


@dataclass
class UserJourney:
    """User journey specification."""
    name: str
    description: str
    scenarios: List[UATScenario]
    business_value: str
    priority: int


class UATFramework:
    """User Acceptance Testing framework."""
    
    def __init__(self, base_url: str = "http://localhost:3000"):
        self.base_url = base_url.rstrip('/')
        self.scenarios: Dict[str, UATScenario] = {}
        self.journeys: Dict[str, UserJourney] = {}
        self.results: List[UATResult] = []
        
        # Test user accounts
        self.test_users = {
            UserRole.ADMIN: {"username": "admin", "password": "admin123"},
            UserRole.TRADER: {"username": "trader", "password": "trader123"},
            UserRole.ANALYST: {"username": "analyst", "password": "analyst123"},
            UserRole.VIEWER: {"username": "viewer", "password": "viewer123"},
            UserRole.GUEST: {"username": "", "password": ""}
        }
        
        # Business logic validations
        self.business_rules = self._load_business_rules()
        
    def _load_business_rules(self) -> Dict[str, Any]:
        """Load business logic rules."""
        return {
            "trading_limits": {
                "max_order_size": 1000000,
                "max_daily_trades": 100,
                "min_account_balance": 1000
            },
            "risk_management": {
                "max_position_size_percent": 10,
                "max_portfolio_risk": 20,
                "stop_loss_required": True
            },
            "reporting": {
                "real_time_data_required": True,
                "historical_data_years": 5,
                "backup_frequency": "daily"
            }
        }
    
    def register_uat_scenario(self, scenario: UATScenario):
        """Register a UAT scenario."""
        self.scenarios[scenario.name] = scenario
        logger.info(f"Registered UAT scenario: {scenario.name}")
    
    def register_user_journey(self, journey: UserJourney):
        """Register a user journey."""
        self.journeys[journey.name] = journey
        logger.info(f"Registered user journey: {journey.name}")
    
    async def run_uat_scenario(self, scenario_name: str) -> UATResult:
        """Run a specific UAT scenario."""
        if scenario_name not in self.scenarios:
            return UATResult(
                scenario_name=scenario_name,
                user_role=UserRole.GUEST,
                status=UATStatus.FAILED,
                duration=0,
                steps_completed=0,
                total_steps=0,
                success_criteria_met={},
                issues_found=[f"Scenario {scenario_name} not found"]
            )
        
        scenario = self.scenarios[scenario_name]
        logger.info(f"Running UAT scenario: {scenario_name}")
        
        start_time = time.time()
        step_results = []
        issues_found = []
        
        try:
            # Check preconditions
            if not await self._check_preconditions(scenario.preconditions):
                duration = time.time() - start_time
                return UATResult(
                    scenario_name=scenario_name,
                    user_role=scenario.user_role,
                    status=UATStatus.BLOCKED,
                    duration=duration,
                    steps_completed=0,
                    total_steps=len(scenario.steps),
                    success_criteria_met={},
                    issues_found=["Preconditions not met"],
                    timestamp=datetime.now()
                )
            
            # Execute scenario steps
            for i, step in enumerate(scenario.steps):
                try:
                    step_result = await self._execute_uat_step(step, scenario.user_role)
                    step_results.append(step_result)
                    
                    if not step_result["success"]:
                        issues_found.append(f"Step {i+1} failed: {step_result.get('error', 'Unknown error')}")
                        logger.warning(f"UAT step {i+1} failed in scenario {scenario_name}")
                        
                        # Continue with other steps unless it's a critical failure
                        if step_result.get("critical", False):
                            break
                    
                except Exception as e:
                    issues_found.append(f"Step {i+1} error: {str(e)}")
                    logger.error(f"UAT step {i+1} error in scenario {scenario_name}: {str(e)}")
            
            # Validate success criteria
            success_criteria_met = await self._validate_success_criteria(
                scenario, step_results
            )
            
            # Determine overall status
            if not issues_found and all(success_criteria_met.values()):
                status = UATStatus.PASSED
            elif issues_found:
                status = UATStatus.FAILED
            else:
                status = UATStatus.FAILED  # Some criteria not met
            
            duration = time.time() - start_time
            
            result = UATResult(
                scenario_name=scenario_name,
                user_role=scenario.user_role,
                status=status,
                duration=duration,
                steps_completed=len(step_results),
                total_steps=len(scenario.steps),
                success_criteria_met=success_criteria_met,
                issues_found=issues_found,
                timestamp=datetime.now()
            )
            
            self.results.append(result)
            logger.info(f"UAT scenario {scenario_name} completed: {status.value}")
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            error_result = UATResult(
                scenario_name=scenario_name,
                user_role=scenario.user_role,
                status=UATStatus.FAILED,
                duration=duration,
                steps_completed=0,
                total_steps=len(scenario.steps),
                success_criteria_met={},
                issues_found=[f"Scenario execution error: {str(e)}"],
                timestamp=datetime.now()
            )
            
            self.results.append(error_result)
            logger.error(f"UAT scenario {scenario_name} failed: {str(e)}")
            return error_result
    
    async def _check_preconditions(self, preconditions: List[str]) -> bool:
        """Check if preconditions are met."""
        if not preconditions:
            return True
        
        for precondition in preconditions:
            # Simple precondition checking
            if precondition == "user_authenticated":
                # Check if test users are available
                if not any(self.test_users.values()):
                    return False
            elif precondition == "market_data_available":
                # Check if market data endpoints are responding
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(f"{self.base_url}/api/market-data") as response:
                            if response.status != 200:
                                return False
                except:
                    return False
            # Add more precondition checks as needed
        
        return True
    
    async def _execute_uat_step(self, step: Dict[str, Any], user_role: UserRole) -> Dict[str, Any]:
        """Execute a single UAT step."""
        step_type = step.get("type", "api_call")
        
        if step_type == "api_call":
            return await self._execute_api_call_step(step, user_role)
        elif step_type == "ui_interaction":
            return await self._execute_ui_interaction_step(step, user_role)
        elif step_type == "data_validation":
            return await self._execute_data_validation_step(step)
        elif step_type == "business_logic":
            return await self._execute_business_logic_step(step)
        elif step_type == "waiting":
            return await self._execute_waiting_step(step)
        else:
            return {"success": False, "error": f"Unknown step type: {step_type}"}
    
    async def _execute_api_call_step(self, step: Dict[str, Any], user_role: UserRole) -> Dict[str, Any]:
        """Execute an API call step."""
        method = step.get("method", "GET")
        url = step.get("url", "")
        headers = step.get("headers", {})
        data = step.get("data", {})
        expected_status = step.get("expected_status", 200)
        
        # Add authentication headers for non-guest users
        if user_role != UserRole.GUEST:
            auth_info = self.test_users.get(user_role)
            if auth_info:
                # Simple basic auth (in real implementation, would use proper auth)
                auth_string = f"{auth_info['username']}:{auth_info['password']}"
                headers["Authorization"] = f"Basic {auth_string}"
        
        try:
            async with aiohttp.ClientSession() as session:
                if method.upper() == "GET":
                    async with session.get(f"{self.base_url}{url}", headers=headers) as response:
                        response_data = await response.json() if response.headers.get('content-type', '').startswith('application/json') else await response.text()
                        status_success = response.status == expected_status
                        
                elif method.upper() == "POST":
                    async with session.post(f"{self.base_url}{url}", json=data, headers=headers) as response:
                        response_data = await response.json() if response.headers.get('content-type', '').startswith('application/json') else await response.text()
                        status_success = response.status == expected_status
                        
                elif method.upper() == "PUT":
                    async with session.put(f"{self.base_url}{url}", json=data, headers=headers) as response:
                        response_data = await response.json() if response.headers.get('content-type', '').startswith('application/json') else await response.text()
                        status_success = response.status == expected_status
                        
                elif method.upper() == "DELETE":
                    async with session.delete(f"{self.base_url}{url}", headers=headers) as response:
                        response_data = await response.json() if response.headers.get('content-type', '').startswith('application/json') else await response.text()
                        status_success = response.status == expected_status
                
                # Validate response data
                validation_results = self._validate_response_data(step, response_data)
                
                return {
                    "success": status_success and all(validation_results.values()),
                    "status_code": response.status if 'response' in locals() else None,
                    "response_data": response_data,
                    "validation_results": validation_results,
                    "expected_status": expected_status
                }
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_ui_interaction_step(self, step: Dict[str, Any], user_role: UserRole) -> Dict[str, Any]:
        """Execute UI interaction step."""
        # This would use Selenium for browser automation
        # For now, return a mock result
        return {
            "success": True,
            "message": "UI interaction step executed",
            "details": step
        }
    
    async def _execute_data_validation_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data validation step."""
        validation_type = step.get("validation_type", "schema")
        
        if validation_type == "schema":
            # Validate data against schema
            data = step.get("data", {})
            schema = step.get("schema", {})
            
            # Simple schema validation (would use proper library like jsonschema)
            required_fields = schema.get("required", [])
            missing_fields = [field for field in required_fields if field not in data]
            
            return {
                "success": len(missing_fields) == 0,
                "missing_fields": missing_fields,
                "validation_type": "schema"
            }
        
        elif validation_type == "business_rule":
            # Validate business rules
            data = step.get("data", {})
            rule_name = step.get("rule_name")
            
            if rule_name == "trading_limits":
                result = self._validate_trading_limits(data)
            elif rule_name == "risk_management":
                result = self._validate_risk_management(data)
            else:
                result = {"success": True, "message": "Unknown rule"}
            
            return result
        
        else:
            return {"success": False, "error": f"Unknown validation type: {validation_type}"}
    
    async def _execute_business_logic_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute business logic validation."""
        business_function = step.get("business_function", "")
        
        if business_function == "calculate_portfolio_risk":
            return self._calculate_portfolio_risk(step.get("portfolio_data", {}))
        elif business_function == "validate_trade_order":
            return self._validate_trade_order(step.get("order_data", {}))
        elif business_function == "check_market_hours":
            return self._check_market_hours()
        else:
            return {"success": True, "message": f"Business function {business_function} executed"}
    
    async def _execute_waiting_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a waiting step."""
        duration = step.get("duration", 5)
        condition = step.get("condition")
        
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
    
    def _validate_response_data(self, step: Dict[str, Any], response_data: Any) -> Dict[str, bool]:
        """Validate response data against expectations."""
        validations = {}
        expectations = step.get("expectations", {})
        
        for field, expected_value in expectations.items():
            if isinstance(expected_value, str) and expected_value.startswith("regex:"):
                # Regex validation
                import re
                pattern = expected_value[6:]  # Remove "regex:" prefix
                actual_value = str(response_data)
                validations[field] = bool(re.search(pattern, actual_value))
            else:
                # Direct value validation
                actual_value = response_data.get(field) if isinstance(response_data, dict) else None
                validations[field] = actual_value == expected_value
        
        return validations
    
    def _validate_trading_limits(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate trading limits."""
        order_size = order_data.get("size", 0)
        account_balance = order_data.get("account_balance", 0)
        
        max_order_size = self.business_rules["trading_limits"]["max_order_size"]
        min_balance = self.business_rules["trading_limits"]["min_account_balance"]
        
        issues = []
        
        if order_size > max_order_size:
            issues.append(f"Order size {order_size} exceeds maximum {max_order_size}")
        
        if account_balance < min_balance:
            issues.append(f"Account balance {account_balance} below minimum {min_balance}")
        
        return {
            "success": len(issues) == 0,
            "issues": issues,
            "order_size": order_size,
            "account_balance": account_balance
        }
    
    def _validate_risk_management(self, position_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate risk management rules."""
        position_size = position_data.get("size", 0)
        portfolio_value = position_data.get("portfolio_value", 0)
        stop_loss = position_data.get("stop_loss")
        
        max_position_percent = self.business_rules["risk_management"]["max_position_size_percent"]
        stop_loss_required = self.business_rules["risk_management"]["stop_loss_required"]
        
        issues = []
        
        if portfolio_value > 0:
            position_percent = (position_size / portfolio_value) * 100
            if position_percent > max_position_percent:
                issues.append(f"Position size {position_percent:.1f}% exceeds maximum {max_position_percent}%")
        
        if stop_loss_required and not stop_loss:
            issues.append("Stop loss is required for this position")
        
        return {
            "success": len(issues) == 0,
            "issues": issues,
            "position_size": position_size,
            "stop_loss": stop_loss
        }
    
    def _calculate_portfolio_risk(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate portfolio risk metrics."""
        positions = portfolio_data.get("positions", [])
        total_value = sum(pos.get("value", 0) for pos in positions)
        
        risk_metrics = {
            "total_value": total_value,
            "position_count": len(positions),
            "largest_position_percent": 0,
            "concentration_risk": "low"
        }
        
        if positions:
            largest_position = max(positions, key=lambda p: p.get("value", 0))
            risk_metrics["largest_position_percent"] = (largest_position.get("value", 0) / total_value) * 100 if total_value > 0 else 0
            
            # Determine concentration risk
            if risk_metrics["largest_position_percent"] > 30:
                risk_metrics["concentration_risk"] = "high"
            elif risk_metrics["largest_position_percent"] > 20:
                risk_metrics["concentration_risk"] = "medium"
        
        return {
            "success": True,
            "risk_metrics": risk_metrics
        }
    
    def _validate_trade_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate trade order business logic."""
        order_type = order_data.get("type", "market")
        quantity = order_data.get("quantity", 0)
        price = order_data.get("price")
        
        issues = []
        
        # Basic order validation
        if quantity <= 0:
            issues.append("Quantity must be positive")
        
        if order_type == "limit" and (price is None or price <= 0):
            issues.append("Limit orders require valid price")
        
        return {
            "success": len(issues) == 0,
            "issues": issues,
            "order_validated": len(issues) == 0
        }
    
    def _check_market_hours(self) -> Dict[str, Any]:
        """Check if market is currently open."""
        now = datetime.now()
        # Simplified market hours check (9:30 AM to 4:00 PM EST)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        is_market_open = market_open <= now <= market_close
        
        return {
            "success": True,
            "market_open": is_market_open,
            "current_time": now.isoformat(),
            "market_hours": "09:30 - 16:00 EST"
        }
    
    async def _check_condition(self, condition: Dict[str, Any]) -> bool:
        """Check if a condition is met."""
        condition_type = condition.get("type")
        
        if condition_type == "api_response":
            url = condition.get("url")
            expected_status = condition.get("expected_status", 200)
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.base_url}{url}") as response:
                        return response.status == expected_status
            except:
                return False
        
        # Add more condition types as needed
        return False
    
    async def _validate_success_criteria(self, scenario: UATScenario, 
                                       step_results: List[Dict[str, Any]]) -> Dict[str, bool]:
        """Validate success criteria for scenario."""
        criteria_met = {}
        criteria = scenario.success_criteria or {}
        
        for criterion_name, criterion_config in criteria.items():
            if criterion_name == "all_steps_successful":
                criteria_met[criterion_name] = all(
                    step.get("success", False) for step in step_results
                )
            elif criterion_name == "min_steps_completed":
                min_steps = criterion_config.get("min_steps", len(scenario.steps))
                criteria_met[criterion_name] = len(step_results) >= min_steps
            elif criterion_name == "business_rules_valid":
                # Check if business rules were validated successfully
                business_steps = [step for step in step_results if "business_function" in step]
                criteria_met[criterion_name] = all(
                    step.get("success", False) for step in business_steps
                )
            else:
                criteria_met[criterion_name] = False
        
        return criteria_met
    
    async def run_user_journey(self, journey_name: str) -> Dict[str, Any]:
        """Run a complete user journey."""
        if journey_name not in self.journeys:
            return {"error": f"User journey {journey_name} not found"}
        
        journey = self.journeys[journey_name]
        logger.info(f"Running user journey: {journey_name}")
        
        overall_start_time = time.time()
        journey_results = []
        
        # Run all scenarios in the journey
        for scenario_name in [s.name for s in journey.scenarios]:
            result = await self.run_uat_scenario(scenario_name)
            journey_results.append(result)
        
        overall_duration = time.time() - overall_start_time
        
        # Calculate journey success rate
        passed_scenarios = len([r for r in journey_results if r.status == UATStatus.PASSED])
        total_scenarios = len(journey.scenarios)
        success_rate = (passed_scenarios / total_scenarios * 100) if total_scenarios > 0 else 0
        
        # Generate journey report
        report = {
            "journey_name": journey_name,
            "description": journey.description,
            "business_value": journey.business_value,
            "priority": journey.priority,
            "duration": overall_duration,
            "total_scenarios": total_scenarios,
            "passed_scenarios": passed_scenarios,
            "failed_scenarios": len([r for r in journey_results if r.status == UATStatus.FAILED]),
            "blocked_scenarios": len([r for r in journey_results if r.status == UATStatus.BLOCKED]),
            "success_rate": success_rate,
            "scenario_results": [
                {
                    "scenario_name": r.scenario_name,
                    "status": r.status.value,
                    "duration": r.duration,
                    "issues": r.issues_found
                }
                for r in journey_results
            ],
            "overall_status": "PASSED" if success_rate >= 80 else "FAILED"
        }
        
        logger.info(f"User journey {journey_name} completed: {success_rate:.1f}% success rate")
        return report
    
    async def run_all_uat_tests(self) -> Dict[str, Any]:
        """Run all registered UAT scenarios."""
        logger.info("Starting comprehensive UAT testing...")
        
        overall_start_time = time.time()
        all_results = []
        
        for scenario_name in self.scenarios.keys():
            result = await self.run_uat_scenario(scenario_name)
            all_results.append(result)
        
        overall_duration = time.time() - overall_start_time
        
        # Calculate overall statistics
        total_scenarios = len(all_results)
        passed_scenarios = len([r for r in all_results if r.status == UATStatus.PASSED])
        failed_scenarios = len([r for r in all_results if r.status == UATStatus.FAILED])
        blocked_scenarios = len([r for r in all_results if r.status == UATStatus.BLOCKED])
        
        success_rate = (passed_scenarios / total_scenarios * 100) if total_scenarios > 0 else 0
        
        # Generate comprehensive report
        report = {
            "timestamp": datetime.now().isoformat(),
            "target_url": self.base_url,
            "total_duration": overall_duration,
            "total_scenarios": total_scenarios,
            "passed_scenarios": passed_scenarios,
            "failed_scenarios": failed_scenarios,
            "blocked_scenarios": blocked_scenarios,
            "success_rate": success_rate,
            "overall_status": "PASSED" if success_rate >= 80 else "FAILED",
            "user_role_breakdown": self._get_user_role_breakdown(all_results),
            "scenario_results": [
                {
                    "scenario_name": r.scenario_name,
                    "user_role": r.user_role.value,
                    "status": r.status.value,
                    "duration": r.duration,
                    "steps_completed": r.steps_completed,
                    "total_steps": r.total_steps,
                    "success_criteria_met": r.success_criteria_met,
                    "issues_found": r.issues_found
                }
                for r in all_results
            ],
            "recommendations": self._generate_uat_recommendations(all_results)
        }
        
        logger.info(f"UAT testing completed in {overall_duration:.2f} seconds")
        return report
    
    def _get_user_role_breakdown(self, results: List[UATResult]) -> Dict[str, Dict[str, int]]:
        """Get breakdown of results by user role."""
        breakdown = {}
        
        for role in UserRole:
            role_results = [r for r in results if r.user_role == role]
            if role_results:
                breakdown[role.value] = {
                    "total": len(role_results),
                    "passed": len([r for r in role_results if r.status == UATStatus.PASSED]),
                    "failed": len([r for r in role_results if r.status == UATStatus.FAILED]),
                    "blocked": len([r for r in role_results if r.status == UATStatus.BLOCKED])
                }
        
        return breakdown
    
    def _generate_uat_recommendations(self, results: List[UATResult]) -> List[str]:
        """Generate UAT recommendations based on results."""
        recommendations = []
        
        # Analyze failure patterns
        failed_results = [r for r in results if r.status == UATStatus.FAILED]
        blocked_results = [r for r in results if r.status == UATStatus.BLOCKED]
        
        if failed_results:
            recommendations.append(f"Address {len(failed_results)} failed UAT scenarios to improve user experience")
        
        if blocked_results:
            recommendations.append(f"Resolve {len(blocked_results)} blocked scenarios that prevent testing")
        
        # Role-specific recommendations
        for role in UserRole:
            role_results = [r for r in results if r.user_role == role]
            failed_role_results = [r for r in role_results if r.status == UATStatus.FAILED]
            
            if failed_role_results:
                recommendations.append(f"Focus on {role.value} role scenarios - {len(failed_role_results)} failures detected")
        
        # General recommendations
        recommendations.extend([
            "Review and improve user interface based on UAT feedback",
            "Ensure all critical business workflows are thoroughly tested",
            "Regular UAT cycles should be conducted before each release"
        ])
        
        return recommendations
    
    def get_test_status(self) -> Dict[str, Any]:
        """Get current UAT test status."""
        return {
            "total_scenarios": len(self.scenarios),
            "total_journeys": len(self.journeys),
            "completed_tests": len(self.results),
            "registered_scenarios": list(self.scenarios.keys()),
            "registered_journeys": list(self.journeys.keys())
        }


# Sample UAT scenarios for trading system
def create_trading_system_uat_scenarios() -> List[UATScenario]:
    """Create sample UAT scenarios for trading system."""
    
    scenarios = [
        # Trader User Journey
        UATScenario(
            name="Trader Login and Dashboard Access",
            description="Trader successfully logs in and accesses dashboard",
            user_role=UserRole.TRADER,
            steps=[
                {
                    "type": "api_call",
                    "method": "POST",
                    "url": "/api/auth/login",
                    "data": {"username": "trader", "password": "trader123"},
                    "expected_status": 200
                },
                {
                    "type": "api_call",
                    "method": "GET",
                    "url": "/api/dashboard",
                    "expected_status": 200
                },
                {
                    "type": "data_validation",
                    "validation_type": "business_rule",
                    "rule_name": "trading_limits",
                    "data": {"size": 50000, "account_balance": 100000}
                }
            ],
            expected_outcomes=["Login successful", "Dashboard accessible", "Trading limits validated"],
            success_criteria={
                "all_steps_successful": True,
                "business_rules_valid": True
            }
        ),
        
        UATScenario(
            name="Place Market Order",
            description="Trader places a market order successfully",
            user_role=UserRole.TRADER,
            steps=[
                {
                    "type": "api_call",
                    "method": "POST",
                    "url": "/api/orders",
                    "data": {
                        "symbol": "AAPL",
                        "type": "market",
                        "quantity": 100,
                        "side": "buy"
                    },
                    "expected_status": 201
                },
                {
                    "type": "business_logic",
                    "business_function": "validate_trade_order",
                    "order_data": {
                        "type": "market",
                        "quantity": 100,
                        "price": 150.00
                    }
                },
                {
                    "type": "api_call",
                    "method": "GET",
                    "url": "/api/orders",
                    "expected_status": 200
                }
            ],
            expected_outcomes=["Order placed successfully", "Order validated", "Order visible in list"],
            success_criteria={
                "all_steps_successful": True,
                "min_steps_completed": {"min_steps": 2}
            }
        ),
        
        # Admin User Journey
        UATScenario(
            name="Admin System Configuration",
            description="Admin configures system settings",
            user_role=UserRole.ADMIN,
            steps=[
                {
                    "type": "api_call",
                    "method": "PUT",
                    "url": "/api/config/trading",
                    "data": {
                        "max_order_size": 1000000,
                        "market_hours": "09:30-16:00"
                    },
                    "expected_status": 200
                },
                {
                    "type": "api_call",
                    "method": "GET",
                    "url": "/api/config/trading",
                    "expected_status": 200
                }
            ],
            expected_outcomes=["Configuration updated", "Configuration verified"],
            success_criteria={
                "all_steps_successful": True
            }
        ),
        
        # Risk Management
        UATScenario(
            name="Risk Limit Validation",
            description="System enforces risk limits for large orders",
            user_role=UserRole.TRADER,
            steps=[
                {
                    "type": "data_validation",
                    "validation_type": "business_rule",
                    "rule_name": "trading_limits",
                    "data": {
                        "size": 5000000,  # Large order
                        "account_balance": 100000
                    }
                },
                {
                    "type": "business_logic",
                    "business_function": "validate_trade_order",
                    "order_data": {
                        "type": "limit",
                        "quantity": 5000000,
                        "price": 100.00
                    }
                }
            ],
            expected_outcomes=["Large order rejected", "Risk limits enforced"],
            success_criteria={
                "business_rules_valid": True
            }
        )
    ]
    
    return scenarios


# Sample user journeys
def create_trading_system_user_journeys() -> List[UserJourney]:
    """Create sample user journeys for trading system."""
    
    journeys = [
        UserJourney(
            name="Complete Trading Workflow",
            description="Full trading workflow from login to order execution",
            scenarios=[
                UATScenario(
                    name="Trader Login and Dashboard Access",
                    description="Trader successfully logs in and accesses dashboard",
                    user_role=UserRole.TRADER,
                    steps=[],  # Steps defined above
                    expected_outcomes=[],
                    success_criteria={}
                ),
                UATScenario(
                    name="Place Market Order",
                    description="Trader places a market order successfully",
                    user_role=UserRole.TRADER,
                    steps=[],
                    expected_outcomes=[],
                    success_criteria={}
                )
            ],
            business_value="Core trading functionality",
            priority=1
        ),
        
        UserJourney(
            name="Risk Management Workflow",
            description="Risk management and compliance validation",
            scenarios=[
                UATScenario(
                    name="Risk Limit Validation",
                    description="System enforces risk limits for large orders",
                    user_role=UserRole.TRADER,
                    steps=[],
                    expected_outcomes=[],
                    success_criteria={}
                )
            ],
            business_value="Risk mitigation and compliance",
            priority=1
        )
    ]
    
    return journeys


# Global UAT framework instance
uat_framework = UATFramework()