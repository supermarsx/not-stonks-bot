"""
Policy Engine for Trade Validation

Manages business rules and trading policies:
- Order validation rules
- Strategy-specific policies
- Market condition policies
- Time-based restrictions
- Asset class restrictions
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, time
from enum import Enum

from config.database import get_db
from database.models.risk import ComplianceRule

logger = logging.getLogger(__name__)


class PolicyType(str, Enum):
    """Types of trading policies."""
    TRADING_HOURS = "trading_hours"
    MARKET_CONDITIONS = "market_conditions"
    ASSET_CLASS = "asset_class"
    STRATEGY_RULES = "strategy_rules"
    ORDER_SIZE = "order_size"
    CONCENTRATION = "concentration"
    LEVERAGE = "leverage"
    TIME_BASED = "time_based"


class PolicyEngine:
    """
    Manages and enforces trading policies and business rules.
    """
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.db = get_db()
        
        # Policy cache
        self._policies_cache = {}
        self._cache_ttl = 300  # 5 minutes
        
        # Built-in policy functions
        self.builtin_policies = {
            "trading_hours_check": self._check_trading_hours,
            "market_hours_check": self._check_market_hours,
            "volatility_check": self._check_volatility,
            "news_blackout_check": self._check_news_blackout,
            "earnings_blackout_check": self._check_earnings_blackout,
            "concentration_check": self._check_concentration,
            "leverage_check": self._check_leverage,
            "order_size_check": self._check_order_size,
            "strategy_validation": self._check_strategy_rules
        }
        
        logger.info(f"PolicyEngine initialized for user {self.user_id}")
    
    async def validate_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate order against all applicable policies.
        
        Args:
            order_data: Order information to validate
            
        Returns:
            Validation result with policy status
        """
        result = {
            "approved": True,
            "rejection_reason": None,
            "warnings": [],
            "policy_checks": [],
            "actions_taken": []
        }
        
        try:
            # Get applicable policies
            policies = await self._get_applicable_policies(order_data)
            
            for policy in policies:
                policy_result = await self._evaluate_policy(policy, order_data)
                result["policy_checks"].append({
                    "policy_id": policy.id,
                    "policy_name": policy.rule_name,
                    "policy_code": policy.rule_code,
                    "result": policy_result
                })
                
                if not policy_result["approved"]:
                    result["approved"] = False
                    
                    if policy.enforcement_level == "strict":
                        result["rejection_reason"] = policy_result["rejection_reason"]
                        break  # Stop on first strict violation
                    else:
                        result["warnings"].append(policy_result["warning_message"])
                
                # Track actions taken
                if policy_result.get("actions_taken"):
                    result["actions_taken"].extend(policy_result["actions_taken"])
            
            # Additional built-in policy checks
            builtin_results = await self._run_builtin_policies(order_data)
            result["policy_checks"].extend(builtin_results)
            
            # Check if any builtin policy rejected the order
            for check in builtin_results:
                if not check["result"]["approved"] and check["result"].get("enforcement_level") == "strict":
                    result["approved"] = False
                    result["rejection_reason"] = check["result"]["rejection_reason"]
                    break
                elif not check["result"]["approved"]:
                    result["warnings"].append(check["result"]["warning_message"])
            
        except Exception as e:
            logger.error(f"Order policy validation error: {str(e)}")
            result.update({
                "approved": False,
                "rejection_reason": f"Policy validation error: {str(e)}"
            })
        
        return result
    
    async def add_custom_policy(self, rule_code: str, rule_name: str, 
                              policy_config: Dict[str, Any], 
                              category: str = "custom",
                              enforcement_level: str = "strict") -> int:
        """
        Add a custom trading policy.
        
        Args:
            rule_code: Unique policy code
            rule_name: Human-readable policy name
            policy_config: Policy configuration and rules
            category: Policy category
            enforcement_level: Enforcement level (strict, warn, audit)
            
        Returns:
            Created policy ID
        """
        try:
            policy = ComplianceRule(
                rule_code=rule_code,
                rule_name=rule_name,
                description=policy_config.get("description"),
                category=category,
                applies_to=[self.user_id],  # Apply to this user
                rule_config=policy_config,
                enforcement_level=enforcement_level,
                is_active=True,
                effective_from=datetime.now()
            )
            
            self.db.add(policy)
            self.db.commit()
            
            # Invalidate cache
            self._policies_cache.clear()
            
            logger.info(f"Custom policy added: {rule_code}")
            return policy.id
            
        except Exception as e:
            logger.error(f"Policy creation error: {str(e)}")
            raise
    
    async def get_policy_status(self) -> Dict[str, Any]:
        """
        Get status of all policies for the user.
        
        Returns:
            Policy status information
        """
        status = {
            "user_id": self.user_id,
            "policies": [],
            "summary": {
                "total_policies": 0,
                "active_policies": 0,
                "categories": {}
            }
        }
        
        try:
            policies = self.db.query(ComplianceRule).filter(
                ComplianceRule.applies_to.op("?")(self.user_id)  # JSON array contains user_id
            ).all()
            
            status["summary"]["total_policies"] = len(policies)
            
            for policy in policies:
                policy_info = {
                    "id": policy.id,
                    "code": policy.rule_code,
                    "name": policy.rule_name,
                    "category": policy.category,
                    "is_active": policy.is_active,
                    "enforcement_level": policy.enforcement_level,
                    "effective_from": policy.effective_from,
                    "config_summary": {
                        key: str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                        for key, value in policy.rule_config.items()
                    }
                }
                
                status["policies"].append(policy_info)
                
                if policy.is_active:
                    status["summary"]["active_policies"] += 1
                    
                    # Count by category
                    if policy.category not in status["summary"]["categories"]:
                        status["summary"]["categories"][policy.category] = 0
                    status["summary"]["categories"][policy.category] += 1
            
        except Exception as e:
            logger.error(f"Policy status error: {str(e)}")
            status["error"] = str(e)
        
        return status
    
    async def _get_applicable_policies(self, order_data: Dict[str, Any]) -> List[ComplianceRule]:
        """Get policies applicable to this order."""
        try:
            symbol = order_data.get("symbol", "")
            asset_class = order_data.get("asset_class", "")
            strategy_id = order_data.get("strategy_id")
            
            # Get all applicable policies
            policies = self.db.query(ComplianceRule).filter(
                ComplianceRule.is_active == True,
                ComplianceRule.effective_from <= datetime.now(),
                or_(
                    ComplianceRule.applies_to == ["all"],  # Global policies
                    ComplianceRule.applies_to.op("?")(self.user_id),  # User-specific policies
                    and_(
                        ComplianceRule.applies_to.op("?")("symbols"),
                        symbol in ComplianceRule.applies_to["symbols"]
                    ),  # Symbol-specific policies
                    and_(
                        ComplianceRule.applies_to.op("?")("asset_classes"),
                        asset_class in ComplianceRule.applies_to["asset_classes"]
                    ),  # Asset class policies
                    and_(
                        ComplianceRule.applies_to.op("?")("strategies"),
                        str(strategy_id) in ComplianceRule.applies_to["strategies"]
                    )  # Strategy policies
                )
            ).all()
            
            return policies
            
        except Exception as e:
            logger.error(f"Policy retrieval error: {str(e)}")
            return []
    
    async def _evaluate_policy(self, policy: ComplianceRule, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single policy against the order."""
        try:
            policy_config = policy.rule_config
            policy_type = policy_config.get("type")
            
            if policy_type in self.builtin_policies:
                # Use built-in policy function
                policy_func = self.builtin_policies[policy_type]
                result = await policy_func(order_data, policy_config)
            else:
                # Custom policy evaluation
                result = await self._evaluate_custom_policy(policy, order_data)
            
            # Add policy metadata
            result["policy_id"] = policy.id
            result["policy_code"] = policy.rule_code
            result["enforcement_level"] = policy.enforcement_level
            
            return result
            
        except Exception as e:
            logger.error(f"Policy evaluation error for {policy.rule_code}: {str(e)}")
            return {
                "approved": False,
                "rejection_reason": f"Policy evaluation error: {str(e)}",
                "warning_message": f"Error evaluating policy {policy.rule_name}",
                "enforcement_level": policy.enforcement_level
            }
    
    async def _evaluate_custom_policy(self, policy: ComplianceRule, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate custom policy based on configuration."""
        try:
            config = policy.rule_config
            policy_type = config.get("type")
            
            if policy_type == "time_window":
                return await self._check_time_window(order_data, config)
            elif policy_type == "order_value_range":
                return await self._check_order_value_range(order_data, config)
            elif policy_type == "symbol_restriction":
                return await self._check_symbol_restriction(order_data, config)
            elif policy_type == "strategy_limit":
                return await self._check_strategy_limit(order_data, config)
            else:
                # Generic rule evaluation
                return await self._evaluate_generic_rule(order_data, config)
                
        except Exception as e:
            return {
                "approved": True,  # Default to allowing order if custom policy fails
                "rejection_reason": None,
                "warning_message": f"Custom policy evaluation error: {str(e)}"
            }
    
    async def _run_builtin_policies(self, order_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run all built-in policy checks."""
        results = []
        
        try:
            # Core policies to always check
            core_policies = [
                "trading_hours_check",
                "market_hours_check", 
                "volatility_check",
                "order_size_check"
            ]
            
            for policy_name in core_policies:
                if policy_name in self.builtin_policies:
                    policy_func = self.builtin_policies[policy_name]
                    result = await policy_func(order_data, {})
                    results.append({
                        "policy_name": policy_name,
                        "result": result
                    })
            
        except Exception as e:
            logger.error(f"Built-in policy check error: {str(e)}")
        
        return results
    
    # Built-in policy implementations
    
    async def _check_trading_hours(self, order_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Check if order is within allowed trading hours."""
        try:
            current_time = datetime.now().time()
            
            # Default trading hours (9:30 AM - 4:00 PM EST)
            start_time = config.get("start_time", time(9, 30))
            end_time = config.get("end_time", time(16, 0))
            
            # Check if current time is within trading hours
            if not (start_time <= current_time <= end_time):
                return {
                    "approved": False,
                    "rejection_reason": f"Order outside trading hours: {start_time} - {end_time}",
                    "warning_message": f"Current time {current_time} is outside trading hours"
                }
            
            return {
                "approved": True,
                "rejection_reason": None,
                "warning_message": None
            }
            
        except Exception as e:
            return {
                "approved": True,  # Default to allowing if check fails
                "rejection_reason": None,
                "warning_message": f"Trading hours check error: {str(e)}"
            }
    
    async def _check_market_hours(self, order_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Check if market is open for the symbol's exchange."""
        try:
            # Simplified market hours check
            # In practice, this would check exchange calendars, holidays, etc.
            symbol = order_data.get("symbol", "").upper()
            
            # Skip check for crypto (24/7 markets)
            if any(crypto in symbol for crypto in ["USDT", "BTC", "ETH", "BNB"]):
                return {
                    "approved": True,
                    "rejection_reason": None,
                    "warning_message": None
                }
            
            # Simplified stock market hours check
            current_time = datetime.now()
            weekday = current_time.weekday()
            
            # Weekend check
            if weekday >= 5:  # Saturday = 5, Sunday = 6
                return {
                    "approved": False,
                    "rejection_reason": "Market closed on weekends",
                    "warning_message": "Order placed during market close (weekend)"
                }
            
            # Hours check (simplified)
            market_open = time(9, 30)  # 9:30 AM EST
            market_close = time(16, 0)  # 4:00 PM EST
            
            if not (market_open <= current_time.time() <= market_close):
                return {
                    "approved": False,
                    "rejection_reason": "Market closed",
                    "warning_message": "Order placed outside market hours"
                }
            
            return {
                "approved": True,
                "rejection_reason": None,
                "warning_message": None
            }
            
        except Exception as e:
            return {
                "approved": True,
                "rejection_reason": None,
                "warning_message": f"Market hours check error: {str(e)}"
            }
    
    async def _check_volatility(self, order_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Check if market volatility is within acceptable limits."""
        try:
            # Simplified volatility check
            # In practice, this would calculate real volatility metrics
            symbol = order_data.get("symbol", "")
            
            # High volatility symbols that might need special handling
            high_vol_symbols = config.get("high_volatility_symbols", [])
            
            if symbol in high_vol_symbols:
                return {
                    "approved": True,  # Allow but warn
                    "rejection_reason": None,
                    "warning_message": f"High volatility symbol detected: {symbol}"
                }
            
            return {
                "approved": True,
                "rejection_reason": None,
                "warning_message": None
            }
            
        except Exception as e:
            return {
                "approved": True,
                "rejection_reason": None,
                "warning_message": f"Volatility check error: {str(e)}"
            }
    
    async def _check_news_blackout(self, order_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Check for news blackout periods."""
        try:
            # Simplified news blackout check
            # In practice, this would check news calendars, earnings dates, etc.
            symbol = order_data.get("symbol", "")
            
            # Check if symbol has upcoming earnings (simplified)
            earnings_blackout_symbols = config.get("earnings_blackout_symbols", [])
            
            if symbol in earnings_blackout_symbols:
                return {
                    "approved": True,  # Allow but warn
                    "rejection_reason": None,
                    "warning_message": f"Earnings blackout period for {symbol}"
                }
            
            return {
                "approved": True,
                "rejection_reason": None,
                "warning_message": None
            }
            
        except Exception as e:
            return {
                "approved": True,
                "rejection_reason": None,
                "warning_message": f"News blackout check error: {str(e)}"
            }
    
    async def _check_earnings_blackout(self, order_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Check for earnings-related trading blackout."""
        # This would be implemented similar to news blackout
        return {
            "approved": True,
            "rejection_reason": None,
            "warning_message": None
        }
    
    async def _check_concentration(self, order_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Check portfolio concentration limits."""
        try:
            symbol = order_data.get("symbol", "")
            quantity = order_data.get("quantity", 0)
            estimated_price = order_data.get("limit_price") or order_data.get("estimated_price", 0)
            
            # Calculate new position value
            new_position_value = quantity * estimated_price
            
            # Get total portfolio value
            db = get_db()
            positions = db.query(Position).filter(Position.user_id == self.user_id).all()
            total_portfolio = sum(abs(pos.market_value or 0) for pos in positions)
            
            if total_portfolio > 0:
                # Add new position value for estimation
                new_total_portfolio = total_portfolio + new_position_value
                new_concentration = new_position_value / new_total_portfolio
                
                max_concentration = config.get("max_concentration", 0.20)  # 20% default
                
                if new_concentration > max_concentration:
                    return {
                        "approved": False,
                        "rejection_reason": f"Concentration limit exceeded: {new_concentration:.1%} > {max_concentration:.1%}",
                        "warning_message": f"High concentration in {symbol}: {new_concentration:.1%}"
                    }
            
            return {
                "approved": True,
                "rejection_reason": None,
                "warning_message": None
            }
            
        except Exception as e:
            return {
                "approved": True,
                "rejection_reason": None,
                "warning_message": f"Concentration check error: {str(e)}"
            }
    
    async def _check_leverage(self, order_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Check leverage limits."""
        try:
            # Simplified leverage check
            # In practice, this would check actual margin requirements
            max_leverage = config.get("max_leverage", 2.0)  # 2:1 default
            
            return {
                "approved": True,
                "rejection_reason": None,
                "warning_message": None
            }
            
        except Exception as e:
            return {
                "approved": True,
                "rejection_reason": None,
                "warning_message": f"Leverage check error: {str(e)}"
            }
    
    async def _check_order_size(self, order_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Check order size limits."""
        try:
            quantity = order_data.get("quantity", 0)
            max_order_size = config.get("max_order_size", 10000)
            
            if quantity > max_order_size:
                return {
                    "approved": False,
                    "rejection_reason": f"Order size exceeds limit: {quantity} > {max_order_size}",
                    "warning_message": f"Large order size: {quantity} shares"
                }
            
            return {
                "approved": True,
                "rejection_reason": None,
                "warning_message": None
            }
            
        except Exception as e:
            return {
                "approved": True,
                "rejection_reason": None,
                "warning_message": f"Order size check error: {str(e)}"
            }
    
    async def _check_strategy_rules(self, order_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Check strategy-specific rules."""
        try:
            strategy_id = order_data.get("strategy_id")
            
            if not strategy_id:
                return {
                    "approved": True,
                    "rejection_reason": None,
                    "warning_message": None
                }
            
            # Strategy-specific validation would go here
            # This is a placeholder implementation
            
            return {
                "approved": True,
                "rejection_reason": None,
                "warning_message": None
            }
            
        except Exception as e:
            return {
                "approved": True,
                "rejection_reason": None,
                "warning_message": f"Strategy validation error: {str(e)}"
            }
    
    # Custom policy implementations
    
    async def _check_time_window(self, order_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Check if order is within specified time window."""
        try:
            start_hour = config.get("start_hour", 9)
            end_hour = config.get("end_hour", 16)
            current_hour = datetime.now().hour
            
            if not (start_hour <= current_hour < end_hour):
                return {
                    "approved": False,
                    "rejection_reason": f"Order outside allowed time window: {start_hour}:00 - {end_hour}:00",
                    "warning_message": f"Current hour {current_hour} outside allowed window"
                }
            
            return {
                "approved": True,
                "rejection_reason": None,
                "warning_message": None
            }
            
        except Exception as e:
            return {
                "approved": True,
                "rejection_reason": None,
                "warning_message": f"Time window check error: {str(e)}"
            }
    
    async def _check_order_value_range(self, order_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Check if order value is within specified range."""
        try:
            quantity = order_data.get("quantity", 0)
            price = order_data.get("limit_price") or order_data.get("estimated_price", 0)
            order_value = quantity * price
            
            min_value = config.get("min_value", 0)
            max_value = config.get("max_value", float('inf'))
            
            if order_value < min_value or order_value > max_value:
                return {
                    "approved": False,
                    "rejection_reason": f"Order value ${order_value:.2f} outside range ${min_value:.2f} - ${max_value:.2f}",
                    "warning_message": f"Order value ${order_value:.2f} outside allowed range"
                }
            
            return {
                "approved": True,
                "rejection_reason": None,
                "warning_message": None
            }
            
        except Exception as e:
            return {
                "approved": True,
                "rejection_reason": None,
                "warning_message": f"Order value range check error: {str(e)}"
            }
    
    async def _check_symbol_restriction(self, order_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Check if symbol is in allowed/restricted list."""
        try:
            symbol = order_data.get("symbol", "").upper()
            allowed_symbols = config.get("allowed_symbols", [])
            restricted_symbols = config.get("restricted_symbols", [])
            
            if restricted_symbols and symbol in restricted_symbols:
                return {
                    "approved": False,
                    "rejection_reason": f"Symbol {symbol} is restricted",
                    "warning_message": f"Restricted symbol: {symbol}"
                }
            
            if allowed_symbols and symbol not in allowed_symbols:
                return {
                    "approved": False,
                    "rejection_reason": f"Symbol {symbol} not in allowed list",
                    "warning_message": f"Symbol not in allowed list: {symbol}"
                }
            
            return {
                "approved": True,
                "rejection_reason": None,
                "warning_message": None
            }
            
        except Exception as e:
            return {
                "approved": True,
                "rejection_reason": None,
                "warning_message": f"Symbol restriction check error: {str(e)}"
            }
    
    async def _check_strategy_limit(self, order_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Check strategy-specific limits."""
        try:
            strategy_id = order_data.get("strategy_id")
            max_trades_per_day = config.get("max_trades_per_day", 100)
            
            if strategy_id:
                # Count trades for this strategy today
                today = datetime.now().date()
                db = get_db()
                strategy_trades_today = db.query(Trade).filter(
                    and_(
                        Trade.user_id == self.user_id,
                        Trade.executed_at >= datetime.combine(today, datetime.min.time()),
                        # Strategy ID would need to be tracked in trades
                    )
                ).count()
                
                if strategy_trades_today >= max_trades_per_day:
                    return {
                        "approved": False,
                        "rejection_reason": f"Strategy {strategy_id} daily limit exceeded",
                        "warning_message": f"Strategy daily limit reached: {strategy_trades_today}/{max_trades_per_day}"
                    }
            
            return {
                "approved": True,
                "rejection_reason": None,
                "warning_message": None
            }
            
        except Exception as e:
            return {
                "approved": True,
                "rejection_reason": None,
                "warning_message": f"Strategy limit check error: {str(e)}"
            }
    
    async def _evaluate_generic_rule(self, order_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate generic policy rules."""
        try:
            # This is a placeholder for custom rule evaluation logic
            # Implement specific rule evaluation based on config
            
            return {
                "approved": True,
                "rejection_reason": None,
                "warning_message": None
            }
            
        except Exception as e:
            return {
                "approved": True,
                "rejection_reason": None,
                "warning_message": f"Generic rule evaluation error: {str(e)}"
            }
    
    def close(self):
        """Cleanup resources."""
        self.db.close()
