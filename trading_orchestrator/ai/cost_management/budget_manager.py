"""
Budget Manager - Advanced budget configuration and control system

Manages budget tiers, auto-scaling, and budget optimization strategies.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json

from loguru import logger


class BudgetAction(Enum):
    """Actions that can be triggered by budget limits"""
    SWITCH_TO_CHEAPER_MODELS = "switch_to_cheaper_models"
    REDUCE_TOKEN_LIMITS = "reduce_token_limits"
    THROTTLE_REQUESTS = "throttle_requests"
    SWITCH_PROVIDERS = "switch_providers"
    STOP_NON_CRITICAL_TASKS = "stop_non_critical_tasks"
    SEND_ALERT = "send_alert"
    ENABLE_EMERGENCY_MODE = "enable_emergency_mode"


class BudgetTier(Enum):
    """Budget tiers with different priority levels"""
    LOW = "low"      # < $100/month
    MEDIUM = "medium" # $100-$1000/month  
    HIGH = "high"    # $1000-$10000/month
    ENTERPRISE = "enterprise" # > $10000/month


@dataclass
class BudgetRule:
    """Individual budget rule configuration"""
    name: str
    condition: str  # JSON string defining the condition
    actions: List[BudgetAction]
    priority: int  # Higher priority rules execute first
    is_active: bool = True
    cooldown_period: timedelta = field(default_factory=lambda: timedelta(minutes=30))
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    
    def should_trigger(self) -> bool:
        """Check if rule should trigger based on cooldown"""
        if not self.is_active or self.last_triggered is None:
            return True
        
        return datetime.now() - self.last_triggered > self.cooldown_period
    
    def trigger(self):
        """Mark rule as triggered"""
        self.last_triggered = datetime.now()
        self.trigger_count += 1


@dataclass 
class BudgetProfile:
    """Budget profile with predefined configurations"""
    name: str
    tier: BudgetTier
    monthly_limit: float
    daily_limit: float
    rules: List[BudgetRule] = field(default_factory=list)
    auto_optimization: bool = True
    emergency_mode_threshold: float = 0.95
    
    def add_rule(self, rule: BudgetRule):
        """Add a rule to this profile"""
        self.rules.append(rule)
        # Sort rules by priority
        self.rules.sort(key=lambda r: r.priority, reverse=True)
    
    def get_active_rules(self) -> List[BudgetRule]:
        """Get active rules sorted by priority"""
        return [rule for rule in self.rules if rule.is_active]


class BudgetManager:
    """
    Advanced budget management system
    
    Features:
    - Tier-based budget profiles
    - Dynamic rule engine
    - Auto-scaling budget controls
    - Emergency budget management
    - Budget optimization recommendations
    """
    
    def __init__(self, cost_manager):
        """
        Initialize budget manager
        
        Args:
            cost_manager: LLMCostManager instance
        """
        self.cost_manager = cost_manager
        self.profiles: Dict[str, BudgetProfile] = {}
        self.active_profile: Optional[str] = None
        
        # Initialize default profiles
        self._setup_default_profiles()
        
        # Current budget state
        self.emergency_mode = False
        self.auto_scaling_enabled = True
        self.current_daily_budget = 0.0
        
        logger.info("Budget Manager initialized")
    
    def _setup_default_profiles(self):
        """Setup default budget profiles"""
        
        # Low tier profile
        low_profile = BudgetProfile(
            name="Development",
            tier=BudgetTier.LOW,
            monthly_limit=100.0,
            daily_limit=5.0
        )
        low_profile.add_rule(BudgetRule(
            name="Switch to fast models",
            condition=json.dumps({
                "type": "budget_percentage",
                "threshold": 0.7
            }),
            actions=[BudgetAction.SWITCH_TO_CHEAPER_MODELS],
            priority=1
        ))
        
        # Medium tier profile  
        medium_profile = BudgetProfile(
            name="Standard",
            tier=BudgetTier.MEDIUM,
            monthly_limit=1000.0,
            daily_limit=50.0,
            emergency_mode_threshold=0.90
        )
        medium_profile.add_rule(BudgetRule(
            name="Reduce token limits",
            condition=json.dumps({
                "type": "budget_percentage", 
                "threshold": 0.8
            }),
            actions=[BudgetAction.REDUCE_TOKEN_LIMITS],
            priority=2
        ))
        medium_profile.add_rule(BudgetRule(
            name="Enable throttling",
            condition=json.dumps({
                "type": "budget_percentage",
                "threshold": 0.9
            }),
            actions=[BudgetAction.THROTTLE_REQUESTS],
            priority=1
        ))
        
        # High tier profile
        high_profile = BudgetProfile(
            name="Production",
            tier=BudgetTier.HIGH,
            monthly_limit=10000.0,
            daily_limit=500.0,
            emergency_mode_threshold=0.95
        )
        high_profile.add_rule(BudgetRule(
            name="Optimize model selection",
            condition=json.dumps({
                "type": "budget_percentage",
                "threshold": 0.85
            }),
            actions=[BudgetAction.SWITCH_PROVIDERS, BudgetAction.SEND_ALERT],
            priority=3
        ))
        high_profile.add_rule(BudgetRule(
            name="Emergency mode",
            condition=json.dumps({
                "type": "budget_percentage",
                "threshold": 0.95
            }),
            actions=[BudgetAction.ENABLE_EMERGENCY_MODE],
            priority=1
        ))
        
        # Enterprise profile
        enterprise_profile = BudgetProfile(
            name="Enterprise",
            tier=BudgetTier.ENTERPRISE,
            monthly_limit=50000.0,
            daily_limit=2500.0
        )
        enterprise_profile.add_rule(BudgetRule(
            name="Monitor and alert",
            condition=json.dumps({
                "type": "budget_percentage",
                "threshold": 0.8
            }),
            actions=[BudgetAction.SEND_ALERT],
            priority=5
        ))
        
        self.add_profile(low_profile)
        self.add_profile(medium_profile)
        self.add_profile(high_profile)
        self.add_profile(enterprise_profile)
        
        # Set default active profile
        self.set_active_profile("Standard")
    
    def add_profile(self, profile: BudgetProfile):
        """Add a budget profile"""
        self.profiles[profile.name] = profile
        logger.info(f"Added budget profile: {profile.name}")
    
    def remove_profile(self, profile_name: str):
        """Remove a budget profile"""
        if profile_name in self.profiles:
            del self.profiles[profile_name]
            logger.info(f"Removed budget profile: {profile_name}")
    
    def set_active_profile(self, profile_name: str):
        """Set active budget profile"""
        if profile_name in self.profiles:
            self.active_profile = profile_name
            profile = self.profiles[profile_name]
            
            # Import BudgetLimit here to avoid circular import
            from .cost_manager import BudgetLimit
            
            # Update cost manager with profile budgets
            self.cost_manager.add_budget(
                BudgetLimit(
                    name=f"Profile: {profile_name} Daily",
                    limit_amount=profile.daily_limit,
                    period=timedelta(days=1),
                    alert_thresholds=[0.5, 0.7, 0.9, 0.95],
                    auto_actions=self._profile_to_auto_actions(profile)
                )
            )
            
            self.cost_manager.add_budget(
                BudgetLimit(
                    name=f"Profile: {profile_name} Monthly",
                    limit_amount=profile.monthly_limit,
                    period=timedelta(days=30),
                    alert_thresholds=[0.6, 0.8, 0.9, 0.95],
                    auto_actions=self._profile_to_auto_actions(profile)
                )
            )
            
            logger.info(f"Active budget profile set to: {profile_name}")
        else:
            raise ValueError(f"Profile not found: {profile_name}")
    
    def _profile_to_auto_actions(self, profile: BudgetProfile) -> Dict[str, Any]:
        """Convert profile rules to auto actions for cost manager"""
        actions = {}
        
        for rule in profile.get_active_rules():
            if BudgetAction.SWITCH_TO_CHEAPER_MODELS in rule.actions:
                actions["switch_to_faster_models"] = True
            if BudgetAction.REDUCE_TOKEN_LIMITS in rule.actions:
                actions["reduce_token_limits"] = True
            if BudgetAction.THROTTLE_REQUESTS in rule.actions:
                actions["throttle_requests"] = True
            if BudgetAction.ENABLE_EMERGENCY_MODE in rule.actions:
                actions["send_emergency_alert"] = True
        
        return actions
    
    async def evaluate_budget_rules(self) -> List[Dict[str, Any]]:
        """
        Evaluate budget rules against current usage
        
        Returns:
            List of triggered rules and their actions
        """
        if not self.active_profile:
            return []
        
        profile = self.profiles[self.active_profile]
        triggered_rules = []
        
        current_metrics = self.cost_manager.get_current_metrics()
        daily_usage = current_metrics.total_cost
        
        # Check if we should evaluate emergency mode
        emergency_threshold = profile.emergency_mode_threshold
        daily_percentage = daily_usage / profile.daily_limit if profile.daily_limit > 0 else 0
        
        if daily_percentage >= emergency_threshold:
            await self._enable_emergency_mode()
        
        # Evaluate each rule
        for rule in profile.get_active_rules():
            if not rule.should_trigger():
                continue
            
            if self._evaluate_rule_condition(rule, current_metrics):
                triggered_rules.append(await self._execute_rule(rule))
        
        return triggered_rules
    
    def _evaluate_rule_condition(self, rule: BudgetRule, metrics) -> bool:
        """Evaluate if a rule condition is met"""
        try:
            condition = json.loads(rule.condition)
            condition_type = condition.get("type")
            
            if condition_type == "budget_percentage":
                threshold = condition.get("threshold", 0.8)
                
                # Get current usage as percentage of daily budget
                if not self.active_profile:
                    return False
                
                profile = self.profiles[self.active_profile]
                daily_percentage = metrics.total_cost / profile.daily_limit
                
                return daily_percentage >= threshold
            
            elif condition_type == "daily_spending":
                limit = condition.get("limit", profile.daily_limit)
                return metrics.total_cost >= limit
            
            elif condition_type == "request_rate":
                rate = condition.get("rate", 100)  # requests per hour
                return metrics.request_count >= rate
            
        except Exception as e:
            logger.error(f"Error evaluating rule condition: {e}")
            return False
        
        return False
    
    async def _execute_rule(self, rule: BudgetRule) -> Dict[str, Any]:
        """Execute a budget rule"""
        rule.trigger()
        
        executed_actions = []
        
        for action in rule.actions:
            try:
                if action == BudgetAction.SWITCH_TO_CHEAPER_MODELS:
                    await self._switch_to_cheaper_models()
                    executed_actions.append("Switched to cheaper models")
                
                elif action == BudgetAction.REDUCE_TOKEN_LIMITS:
                    await self._reduce_token_limits()
                    executed_actions.append("Reduced token limits")
                
                elif action == BudgetAction.THROTTLE_REQUESTS:
                    await self._enable_throttling()
                    executed_actions.append("Enabled request throttling")
                
                elif action == BudgetAction.SWITCH_PROVIDERS:
                    await self._switch_providers()
                    executed_actions.append("Switched providers")
                
                elif action == BudgetAction.STOP_NON_CRITICAL_TASKS:
                    await self._stop_non_critical_tasks()
                    executed_actions.append("Stopped non-critical tasks")
                
                elif action == BudgetAction.SEND_ALERT:
                    await self._send_budget_alert(rule)
                    executed_actions.append("Sent budget alert")
                
                elif action == BudgetAction.ENABLE_EMERGENCY_MODE:
                    await self._enable_emergency_mode()
                    executed_actions.append("Enabled emergency mode")
                    
            except Exception as e:
                logger.error(f"Failed to execute budget action {action}: {e}")
        
        result = {
            'rule_name': rule.name,
            'executed_actions': executed_actions,
            'triggered_at': datetime.now().isoformat(),
            'trigger_count': rule.trigger_count
        }
        
        logger.info(f"Budget rule executed: {rule.name} - Actions: {executed_actions}")
        return result
    
    async def _switch_to_cheaper_models(self):
        """Switch to more cost-effective models"""
        logger.info("Budget action: Switching to cheaper models")
        # Implementation would integrate with AI models manager
    
    async def _reduce_token_limits(self):
        """Reduce maximum token limits"""
        logger.info("Budget action: Reducing token limits")
        # Implementation would reduce max_tokens parameter
    
    async def _enable_throttling(self):
        """Enable request throttling"""
        logger.info("Budget action: Enabling request throttling")
        # Implementation would add delays between requests
    
    async def _switch_providers(self):
        """Switch to different providers if available"""
        logger.info("Budget action: Switching providers")
        # Implementation would route requests to cheaper providers
    
    async def _stop_non_critical_tasks(self):
        """Stop non-critical background tasks"""
        logger.info("Budget action: Stopping non-critical tasks")
        # Implementation would cancel non-essential operations
    
    async def _send_budget_alert(self, rule: BudgetRule):
        """Send budget alert notification"""
        logger.warning(f"Budget rule alert: {rule.name}")
        # Implementation would send notifications
    
    async def _enable_emergency_mode(self):
        """Enable emergency budget mode"""
        if not self.emergency_mode:
            self.emergency_mode = True
            logger.critical("EMERGENCY MODE ENABLED - Budget limits critical")
            
            # Execute emergency actions
            await self._execute_emergency_actions()
    
    async def _execute_emergency_actions(self):
        """Execute emergency budget actions"""
        logger.info("Executing emergency budget actions")
        
        # Stop all non-essential operations
        await self._stop_non_critical_tasks()
        
        # Switch to cheapest models
        await self._switch_to_cheaper_models()
        
        # Enable maximum throttling
        await self._enable_throttling()
        
        # Send emergency alerts
        # (This would trigger notifications)
    
    async def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status"""
        if not self.active_profile:
            return {}
        
        profile = self.profiles[self.active_profile]
        metrics = self.cost_manager.get_current_metrics()
        
        daily_percentage = (metrics.total_cost / profile.daily_limit) * 100 if profile.daily_limit > 0 else 0
        
        return {
            'active_profile': self.active_profile,
            'tier': profile.tier.value,
            'daily_spent': f"${metrics.total_cost:.2f}",
            'daily_limit': f"${profile.daily_limit:.2f}",
            'daily_percentage': f"{daily_percentage:.1f}%",
            'monthly_spent': f"${metrics.total_cost * 30:.2f}",  # Simplified calculation
            'monthly_limit': f"${profile.monthly_limit:.2f}",
            'emergency_mode': self.emergency_mode,
            'auto_scaling': self.auto_scaling_enabled,
            'triggered_rules': [rule.name for rule in profile.rules if rule.last_triggered]
        }
    
    def get_recommended_budget_tier(self, monthly_spending: float) -> BudgetTier:
        """Recommend appropriate budget tier based on spending"""
        if monthly_spending < 100:
            return BudgetTier.LOW
        elif monthly_spending < 1000:
            return BudgetTier.MEDIUM
        elif monthly_spending < 10000:
            return BudgetTier.HIGH
        else:
            return BudgetTier.ENTERPRISE
    
    async def optimize_budget_allocation(self) -> Dict[str, Any]:
        """Optimize budget allocation based on usage patterns"""
        metrics = self.cost_manager.get_current_metrics()
        
        # Analyze usage patterns
        provider_usage = metrics.provider_breakdown
        model_usage = metrics.model_breakdown
        
        recommendations = []
        
        # Provider recommendations
        if len(provider_usage) > 1:
            costs = {provider: data['total_cost'] for provider, data in provider_usage.items()}
            most_expensive = max(costs, key=costs.get)
            recommendations.append({
                'type': 'provider_switch',
                'current_provider': most_expensive,
                'recommendation': f"Consider reducing usage of {most_expensive} - highest cost provider",
                'potential_savings': costs[most_expensive] * 0.3  # 30% potential savings
            })
        
        # Model recommendations
        for model, data in model_usage.items():
            if data['total_cost'] > 50:  # High cost models
                recommendations.append({
                    'type': 'model_optimization',
                    'model': model,
                    'recommendation': f"Optimize usage of {model} - high cost model",
                    'potential_savings': data['total_cost'] * 0.2  # 20% potential savings
                })
        
        # Budget tier recommendation
        monthly_estimate = metrics.total_cost * 30
        recommended_tier = self.get_recommended_budget_tier(monthly_estimate)
        current_tier = self.profiles[self.active_profile].tier
        
        if recommended_tier != current_tier:
            recommendations.append({
                'type': 'tier_adjustment',
                'current_tier': current_tier.value,
                'recommended_tier': recommended_tier.value,
                'reason': f"Based on ${monthly_estimate:.2f}/month estimated usage"
            })
        
        return {
            'recommendations': recommendations,
            'current_profile': self.active_profile,
            'estimated_monthly_cost': monthly_estimate,
            'potential_total_savings': sum(r.get('potential_savings', 0) for r in recommendations)
        }
    
    def reset_emergency_mode(self):
        """Reset emergency mode"""
        self.emergency_mode = False
        logger.info("Emergency mode reset")
    
    def get_budget_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Get all budget profiles"""
        return {
            name: {
                'name': profile.name,
                'tier': profile.tier.value,
                'monthly_limit': profile.monthly_limit,
                'daily_limit': profile.daily_limit,
                'rule_count': len(profile.rules),
                'auto_optimization': profile.auto_optimization
            }
            for name, profile in self.profiles.items()
        }