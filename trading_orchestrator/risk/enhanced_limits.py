"""
Enhanced Risk Limits System

Advanced user-configurable risk limits including:
- Position-level limits (max size, concentration)
- Portfolio-level limits (VaR, drawdown, leverage)
- Strategy-level limits (max loss, win rate, Sharpe ratio)
- Time-based limits (daily/weekly/monthly)
- Market-based limits (sector, country, currency)
- Dynamic limits that adjust with market conditions
- Emergency stop mechanisms and circuit breakers
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import asyncio
from abc import ABC, abstractmethod

from database.models.risk import RiskLimit, RiskLevel
from database.models.trading import Position, Order, Trade
from database.models.user import User

logger = logging.getLogger(__name__)


class LimitType(Enum):
    """Types of risk limits."""
    POSITION_SIZE = "position_size"
    POSITION_VALUE = "position_value"
    DAILY_LOSS = "daily_loss"
    DAILY_PNL = "daily_pnl"
    PORTFOLIO_VAR = "portfolio_var"
    PORTFOLIO_CVAR = "portfolio_cvar"
    MAX_DRAWDOWN = "max_drawdown"
    LEVERAGE_RATIO = "leverage_ratio"
    CONCENTRATION = "concentration"
    CORRELATION_BREAK = "correlation_break"
    VOLATILITY_SPIKE = "volatility_spike"
    SECTOR_EXPOSURE = "sector_exposure"
    COUNTRY_EXPOSURE = "country_exposure"
    CURRENCY_EXPOSURE = "currency_exposure"
    STRATEGY_LOSS = "strategy_loss"
    STRATEGY_DRAWDOWN = "strategy_drawdown"
    STRATEGY_WIN_RATE = "strategy_win_rate"
    STRATEGY_SHARPE = "strategy_sharpe"
    ORDER_SIZE = "order_size"
    ORDER_FREQUENCY = "order_frequency"
    EXECUTION_TIME = "execution_time"


class LimitScope(Enum):
    """Scope of risk limits."""
    GLOBAL = "global"
    PORTFOLIO = "portfolio"
    STRATEGY = "strategy"
    SYMBOL = "symbol"
    SECTOR = "sector"
    COUNTRY = "country"
    CURRENCY = "currency"
    BROKER = "broker"


class LimitAction(Enum):
    """Actions when limits are breached."""
    WARN = "warn"
    BLOCK = "block"
    LIQUIDATE = "liquidate"
    HALT_TRADING = "halt_trading"
    REDUCE_POSITION = "reduce_position"
    CANCEL_ORDERS = "cancel_orders"
    INCREASE_MARGIN = "increase_margin"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class RiskLimitConfig:
    """Enhanced risk limit configuration."""
    limit_id: Optional[int] = None
    limit_name: str = ""
    limit_type: LimitType = LimitType.POSITION_SIZE
    limit_scope: LimitScope = LimitScope.PORTFOLIO
    scope_target: Optional[str] = None
    
    # Limit values
    soft_limit: float = 0.0  # Warning threshold
    hard_limit: float = 0.0  # Breach threshold
    
    # Limit behavior
    action_on_breach: LimitAction = LimitAction.WARN
    auto_reset: bool = True
    reset_threshold: float = 0.8  # Reset when back below 80% of limit
    reset_time_hours: Optional[int] = None
    
    # Dynamic limits
    is_dynamic: bool = False
    dynamic_formula: Optional[str] = None
    market_condition_dependent: bool = False
    volatility_adjustment: bool = False
    correlation_adjustment: bool = False
    
    # Time-based limits
    time_window: Optional[str] = None  # "daily", "weekly", "monthly"
    rolling_window: Optional[int] = None  # Number of periods
    
    # Emergency controls
    emergency_trigger: bool = False
    emergency_action: Optional[LimitAction] = None
    circuit_breaker: bool = False
    
    # Metadata
    is_active: bool = True
    created_by: int = 0
    last_modified: datetime = field(default_factory=datetime.now)
    description: str = ""
    tags: List[str] = field(default_factory=list)


class BaseLimitCalculator(ABC):
    """Abstract base class for limit calculators."""
    
    @abstractmethod
    async def calculate_current_value(self, limit_config: RiskLimitConfig, 
                                    context: Dict[str, Any]) -> float:
        """Calculate current value for the limit."""
        pass
    
    @abstractmethod
    def is_breached(self, current_value: float, limit_config: RiskLimitConfig) -> bool:
        """Check if limit is breached."""
        pass
    
    @abstractmethod
    def is_warning(self, current_value: float, limit_config: RiskLimitConfig) -> bool:
        """Check if limit is in warning zone."""
        pass


class PositionSizeLimitCalculator(BaseLimitCalculator):
    """Calculator for position size limits."""
    
    async def calculate_current_value(self, limit_config: RiskLimitConfig, 
                                    context: Dict[str, Any]) -> float:
        """Calculate current position size."""
        try:
            symbol = limit_config.scope_target
            if not symbol or symbol not in context.get('positions', {}):
                return 0.0
            
            position = context['positions'][symbol]
            quantity = abs(position.get('quantity', 0))
            
            return quantity
            
        except Exception as e:
            logger.error(f"Position size calculation error: {str(e)}")
            return 0.0
    
    def is_breached(self, current_value: float, limit_config: RiskLimitConfig) -> bool:
        """Check if position size limit is breached."""
        return current_value > limit_config.hard_limit
    
    def is_warning(self, current_value: float, limit_config: RiskLimitConfig) -> bool:
        """Check if position size limit is in warning zone."""
        return current_value > limit_config.soft_limit


class PortfolioVaRLimitCalculator(BaseLimitCalculator):
    """Calculator for portfolio VaR limits."""
    
    async def calculate_current_value(self, limit_config: RiskLimitConfig, 
                                    context: Dict[str, Any]) -> float:
        """Calculate current portfolio VaR."""
        try:
            var_data = context.get('portfolio_var', {})
            
            if limit_config.limit_type == LimitType.PORTFOLIO_VAR:
                return var_data.get('var_amount', 0.0)
            elif limit_config.limit_type == LimitType.PORTFOLIO_CVAR:
                return var_data.get('cvar_amount', 0.0)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Portfolio VaR calculation error: {str(e)}")
            return 0.0
    
    def is_breached(self, current_value: float, limit_config: RiskLimitConfig) -> bool:
        """Check if portfolio VaR limit is breached."""
        return current_value > limit_config.hard_limit
    
    def is_warning(self, current_value: float, limit_config: RiskLimitConfig) -> bool:
        """Check if portfolio VaR limit is in warning zone."""
        return current_value > limit_config.soft_limit


class DailyLossLimitCalculator(BaseLimitCalculator):
    """Calculator for daily loss limits."""
    
    async def calculate_current_value(self, limit_config: RiskLimitConfig, 
                                    context: Dict[str, Any]) -> float:
        """Calculate current daily loss."""
        try:
            today = datetime.now().date()
            trades_today = context.get('daily_trades', [])
            
            daily_pnl = sum(trade.get('realized_pnl', 0) for trade in trades_today)
            daily_loss = abs(min(0, daily_pnl))  # Positive value for losses
            
            return daily_loss
            
        except Exception as e:
            logger.error(f"Daily loss calculation error: {str(e)}")
            return 0.0
    
    def is_breached(self, current_value: float, limit_config: RiskLimitConfig) -> bool:
        """Check if daily loss limit is breached."""
        return current_value > limit_config.hard_limit
    
    def is_warning(self, current_value: float, limit_config: RiskLimitConfig) -> bool:
        """Check if daily loss limit is in warning zone."""
        return current_value > limit_config.soft_limit


class ConcentrationLimitCalculator(BaseLimitCalculator):
    """Calculator for concentration limits."""
    
    async def calculate_current_value(self, limit_config: RiskLimitConfig, 
                                    context: Dict[str, Any]) -> float:
        """Calculate current concentration."""
        try:
            symbol = limit_config.scope_target
            positions = context.get('positions', {})
            
            if not symbol or symbol not in positions:
                return 0.0
            
            # Calculate symbol exposure
            symbol_position = positions[symbol]
            symbol_value = abs(symbol_position.get('market_value', 0))
            
            # Calculate total portfolio value
            total_value = sum(
                abs(pos.get('market_value', 0)) for pos in positions.values()
            )
            
            if total_value > 0:
                concentration = (symbol_value / total_value) * 100
            else:
                concentration = 0.0
            
            return concentration
            
        except Exception as e:
            logger.error(f"Concentration calculation error: {str(e)}")
            return 0.0
    
    def is_breached(self, current_value: float, limit_config: RiskLimitConfig) -> bool:
        """Check if concentration limit is breached."""
        return current_value > limit_config.hard_limit
    
    def is_warning(self, current_value: float, limit_config: RiskLimitConfig) -> bool:
        """Check if concentration limit is in warning zone."""
        return current_value > limit_config.soft_limit


class EnhancedRiskLimitManager:
    """
    Enhanced Risk Limit Manager.
    
    Manages comprehensive risk limits with advanced features including
    dynamic limits, emergency controls, and real-time monitoring.
    """
    
    def __init__(self, user_id: int):
        """
        Initialize enhanced risk limit manager.
        
        Args:
            user_id: User identifier
        """
        self.user_id = user_id
        self.limit_configs: Dict[int, RiskLimitConfig] = {}
        self.limit_calculators: Dict[LimitType, BaseLimitCalculator] = self._initialize_calculators()
        self.monitoring_active = False
        self.limit_violations = []
        self.limit_warnings = []
        self.emergency_triggers = []
        
    def _initialize_calculators(self) -> Dict[LimitType, BaseLimitCalculator]:
        """Initialize limit calculators."""
        return {
            LimitType.POSITION_SIZE: PositionSizeLimitCalculator(),
            LimitType.POSITION_VALUE: PositionSizeLimitCalculator(),  # Reuse with different units
            LimitType.PORTFOLIO_VAR: PortfolioVaRLimitCalculator(),
            LimitType.PORTFOLIO_CVAR: PortfolioVaRLimitCalculator(),
            LimitType.DAILY_LOSS: DailyLossLimitCalculator(),
            LimitType.CONCENTRATION: ConcentrationLimitCalculator()
        }
    
    async def create_limit(self, config: RiskLimitConfig) -> Dict[str, Any]:
        """
        Create new risk limit configuration.
        
        Args:
            config: Risk limit configuration
            
        Returns:
            Creation result with limit ID
        """
        try:
            # Validate configuration
            validation_result = await self._validate_limit_config(config)
            if not validation_result['valid']:
                return {
                    'success': False,
                    'error': validation_result['error'],
                    'validation_warnings': validation_result['warnings']
                }
            
            # Generate limit ID (in practice, would be database-generated)
            config.limit_id = len(self.limit_configs) + 1
            config.created_by = self.user_id
            config.last_modified = datetime.now()
            
            # Store configuration
            self.limit_configs[config.limit_id] = config
            
            # Log limit creation
            logger.info(f"Created risk limit {config.limit_name} for user {self.user_id}")
            
            return {
                'success': True,
                'limit_id': config.limit_id,
                'limit_name': config.limit_name,
                'limit_type': config.limit_type.value,
                'soft_limit': config.soft_limit,
                'hard_limit': config.hard_limit,
                'action_on_breach': config.action_on_breach.value,
                'is_dynamic': config.is_dynamic,
                'is_active': config.is_active
            }
            
        except Exception as e:
            logger.error(f"Risk limit creation error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def _validate_limit_config(self, config: RiskLimitConfig) -> Dict[str, Any]:
        """Validate risk limit configuration."""
        try:
            warnings = []
            
            # Basic validation
            if not config.limit_name:
                return {'valid': False, 'error': 'Limit name is required'}
            
            if config.soft_limit <= 0 and config.hard_limit <= 0:
                return {'valid': False, 'error': 'At least one limit value must be positive'}
            
            if config.soft_limit >= config.hard_limit:
                warnings.append('Soft limit should be lower than hard limit')
            
            # Scope validation
            if config.limit_scope == LimitType.SYMBOL and not config.scope_target:
                return {'valid': False, 'error': 'Symbol scope requires target symbol'}
            
            # Dynamic limit validation
            if config.is_dynamic and not config.dynamic_formula:
                warnings.append('Dynamic limit without formula will use default calculation')
            
            # Emergency controls validation
            if config.emergency_trigger and not config.emergency_action:
                warnings.append('Emergency trigger without action may not work as expected')
            
            # Time-based limits validation
            if config.time_window and config.time_window not in ['daily', 'weekly', 'monthly']:
                return {'valid': False, 'error': 'Invalid time window. Must be daily, weekly, or monthly'}
            
            return {
                'valid': True,
                'warnings': warnings
            }
            
        except Exception as e:
            logger.error(f"Limit config validation error: {str(e)}")
            return {'valid': False, 'error': str(e)}
    
    async def check_all_limits(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check all active risk limits.
        
        Args:
            context: Risk context data
            
        Returns:
            Comprehensive limit check results
        """
        try:
            results = {
                'checked_limits': 0,
                'breached_limits': [],
                'warning_limits': [],
                'ok_limits': [],
                'failed_limits': [],
                'emergency_triggers': [],
                'total_violations': 0,
                'overall_status': 'ok'
            }
            
            # Clear previous violations
            self.limit_violations.clear()
            self.limit_warnings.clear()
            self.emergency_triggers.clear()
            
            for limit_id, config in self.limit_configs.items():
                if not config.is_active:
                    continue
                
                try:
                    # Calculate current value
                    current_value = await self._calculate_limit_value(config, context)
                    
                    # Check limit status
                    is_breached = config.is_dynamic or self.limit_calculators[config.limit_type].is_breached(current_value, config)
                    is_warning = self.limit_calculators[config.limit_type].is_warning(current_value, config)
                    
                    limit_result = {
                        'limit_id': limit_id,
                        'limit_name': config.limit_name,
                        'limit_type': config.limit_type.value,
                        'current_value': current_value,
                        'soft_limit': config.soft_limit,
                        'hard_limit': config.hard_limit,
                        'percentage_used': self._calculate_limit_usage(current_value, config),
                        'is_breached': is_breached,
                        'is_warning': is_warning,
                        'action_required': config.action_on_breach.value,
                        'scope': config.limit_scope.value,
                        'scope_target': config.scope_target
                    }
                    
                    results['checked_limits'] += 1
                    
                    if is_breached:
                        results['breached_limits'].append(limit_result)
                        self.limit_violations.append(limit_result)
                        
                        # Check for emergency triggers
                        if config.emergency_trigger:
                            emergency_trigger = await self._trigger_emergency_limit(config, limit_result)
                            if emergency_trigger:
                                results['emergency_triggers'].append(emergency_trigger)
                                self.emergency_triggers.append(emergency_trigger)
                    elif is_warning:
                        results['warning_limits'].append(limit_result)
                        self.limit_warnings.append(limit_result)
                    else:
                        results['ok_limits'].append(limit_result)
                    
                except Exception as e:
                    logger.error(f"Limit check error for {config.limit_name}: {str(e)}")
                    results['failed_limits'].append({
                        'limit_id': limit_id,
                        'limit_name': config.limit_name,
                        'error': str(e)
                    })
            
            # Determine overall status
            results['total_violations'] = len(results['breached_limits'])
            results['total_warnings'] = len(results['warning_limits'])
            
            if results['emergency_triggers']:
                results['overall_status'] = 'emergency'
            elif results['breached_limits']:
                results['overall_status'] = 'breached'
            elif results['warning_limits']:
                results['overall_status'] = 'warning'
            else:
                results['overall_status'] = 'ok'
            
            return results
            
        except Exception as e:
            logger.error(f"Risk limits check error: {str(e)}")
            return {
                'checked_limits': 0,
                'breached_limits': [],
                'warning_limits': [],
                'ok_limits': [],
                'failed_limits': [{'error': str(e)}],
                'overall_status': 'error'
            }
    
    async def _calculate_limit_value(self, config: RiskLimitConfig, 
                                   context: Dict[str, Any]) -> float:
        """Calculate current value for specific limit."""
        try:
            # Use calculator if available
            if config.limit_type in self.limit_calculators:
                calculator = self.limit_calculators[config.limit_type]
                return await calculator.calculate_current_value(config, context)
            
            # Fallback calculations for unhandled limit types
            return await self._calculate_fallback_limit(config, context)
            
        except Exception as e:
            logger.error(f"Limit value calculation error: {str(e)}")
            return 0.0
    
    async def _calculate_fallback_limit(self, config: RiskLimitConfig, 
                                      context: Dict[str, Any]) -> float:
        """Fallback calculation for unhandled limit types."""
        try:
            limit_type = config.limit_type
            
            if limit_type == LimitType.MAX_DRAWDOWN:
                # Calculate current drawdown
                portfolio_values = context.get('portfolio_history', pd.Series())
                if len(portfolio_values) > 0:
                    running_max = portfolio_values.expanding().max()
                    drawdown = (portfolio_values - running_max) / running_max
                    return abs(drawdown.min()) * 100  # Percentage
                return 0.0
            
            elif limit_type == LimitType.LEVERAGE_RATIO:
                # Calculate leverage ratio
                portfolio_value = context.get('portfolio_value', 0)
                margin_used = context.get('margin_used', 0)
                if portfolio_value > 0:
                    return margin_used / portfolio_value
                return 0.0
            
            elif limit_type == LimitType.ORDER_FREQUENCY:
                # Count orders in time window
                orders = context.get('recent_orders', [])
                if config.time_window == 'daily':
                    cutoff_time = datetime.now() - timedelta(days=1)
                elif config.time_window == 'weekly':
                    cutoff_time = datetime.now() - timedelta(weeks=1)
                else:
                    cutoff_time = datetime.now() - timedelta(days=1)  # Default to daily
                
                recent_orders = [
                    order for order in orders 
                    if order.get('timestamp', datetime.min) > cutoff_time
                ]
                return len(recent_orders)
            
            # Default fallback
            return 0.0
            
        except Exception as e:
            logger.error(f"Fallback limit calculation error: {str(e)}")
            return 0.0
    
    def _calculate_limit_usage(self, current_value: float, config: RiskLimitConfig) -> float:
        """Calculate limit usage percentage."""
        try:
            if config.hard_limit > 0:
                return (current_value / config.hard_limit) * 100
            return 0.0
        except Exception:
            return 0.0
    
    async def _trigger_emergency_limit(self, config: RiskLimitConfig, 
                                     limit_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Trigger emergency limit response."""
        try:
            if not config.emergency_trigger:
                return None
            
            emergency_response = {
                'limit_id': config.limit_id,
                'limit_name': config.limit_name,
                'triggered_at': datetime.now(),
                'trigger_type': 'emergency_limit_breach',
                'current_value': limit_result['current_value'],
                'hard_limit': config.hard_limit,
                'emergency_action': config.emergency_action.value if config.emergency_action else None,
                'circuit_breaker': config.circuit_breaker,
                'severity': 'critical'
            }
            
            # Execute emergency action
            if config.emergency_action == LimitAction.EMERGENCY_STOP:
                emergency_response['action_taken'] = 'emergency_trading_halt'
                logger.critical(f"EMERGENCY STOP triggered for user {self.user_id}: {config.limit_name}")
            elif config.emergency_action == LimitAction.LIQUIDATE:
                emergency_response['action_taken'] = 'initiate_liquidation'
                logger.critical(f"Emergency liquidation triggered for user {self.user_id}: {config.limit_name}")
            elif config.emergency_action == LimitAction.HALT_TRADING:
                emergency_response['action_taken'] = 'halt_new_orders'
                logger.warning(f"Trading halt triggered for user {self.user_id}: {config.limit_name}")
            
            return emergency_response
            
        except Exception as e:
            logger.error(f"Emergency limit trigger error: {str(e)}")
            return None
    
    async def update_limit(self, limit_id: int, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update existing risk limit configuration.
        
        Args:
            limit_id: Limit identifier
            updates: Configuration updates
            
        Returns:
            Update result
        """
        try:
            if limit_id not in self.limit_configs:
                return {'success': False, 'error': f'Limit {limit_id} not found'}
            
            config = self.limit_configs[limit_id]
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            config.last_modified = datetime.now()
            
            logger.info(f"Updated risk limit {limit_id} for user {self.user_id}")
            
            return {
                'success': True,
                'limit_id': limit_id,
                'updated_fields': list(updates.keys()),
                'last_modified': config.last_modified
            }
            
        except Exception as e:
            logger.error(f"Limit update error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def delete_limit(self, limit_id: int) -> Dict[str, Any]:
        """
        Delete risk limit configuration.
        
        Args:
            limit_id: Limit identifier
            
        Returns:
            Deletion result
        """
        try:
            if limit_id not in self.limit_configs:
                return {'success': False, 'error': f'Limit {limit_id} not found'}
            
            config = self.limit_configs[limit_id]
            del self.limit_configs[limit_id]
            
            logger.info(f"Deleted risk limit {limit_id} for user {self.user_id}")
            
            return {
                'success': True,
                'deleted_limit': {
                    'limit_id': limit_id,
                    'limit_name': config.limit_name,
                    'limit_type': config.limit_type.value
                }
            }
            
        except Exception as e:
            logger.error(f"Limit deletion error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def get_limit_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of all limits.
        
        Returns:
            Limit status report
        """
        try:
            total_limits = len(self.limit_configs)
            active_limits = sum(1 for config in self.limit_configs.values() if config.is_active)
            
            # Count by type
            limits_by_type = {}
            limits_by_scope = {}
            
            for config in self.limit_configs.values():
                limit_type = config.limit_type.value
                limit_scope = config.limit_scope.value
                
                limits_by_type[limit_type] = limits_by_type.get(limit_type, 0) + 1
                limits_by_scope[limit_scope] = limits_by_scope.get(limit_scope, 0) + 1
            
            # Recent violations
            recent_violations = [
                violation for violation in self.limit_violations
                if (datetime.now() - violation.get('checked_at', datetime.now())).seconds < 3600  # Last hour
            ]
            
            # Emergency triggers
            emergency_triggers = [
                trigger for trigger in self.emergency_triggers
                if (datetime.now() - trigger.get('triggered_at', datetime.now())).seconds < 86400  # Last day
            ]
            
            return {
                'summary': {
                    'total_limits': total_limits,
                    'active_limits': active_limits,
                    'inactive_limits': total_limits - active_limits,
                    'recent_violations': len(recent_violations),
                    'recent_warnings': len(self.limit_warnings),
                    'emergency_triggers': len(emergency_triggers)
                },
                'limits_by_type': limits_by_type,
                'limits_by_scope': limits_by_scope,
                'recent_violations': recent_violations[-10:],  # Last 10 violations
                'recent_warnings': self.limit_warnings[-10:],  # Last 10 warnings
                'emergency_triggers': emergency_triggers[-5:],  # Last 5 emergency triggers
                'monitoring_status': 'active' if self.monitoring_active else 'inactive',
                'last_check': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Limit status error: {str(e)}")
            return {'error': str(e)}
    
    async def create_dynamic_limit(self, name: str, base_limit: float, 
                                 market_conditions: Dict[str, float]) -> RiskLimitConfig:
        """
        Create a dynamic limit that adjusts based on market conditions.
        
        Args:
            name: Limit name
            base_limit: Base limit value
            market_conditions: Market condition multipliers
            
        Returns:
            Dynamic limit configuration
        """
        try:
            config = RiskLimitConfig(
                limit_name=name,
                limit_type=LimitType.PORTFOLIO_VAR,
                limit_scope=LimitScope.PORTFOLIO,
                soft_limit=base_limit * 0.8,
                hard_limit=base_limit,
                is_dynamic=True,
                market_condition_dependent=True,
                volatility_adjustment=True,
                correlation_adjustment=True,
                dynamic_formula="base_limit * volatility_multiplier * correlation_multiplier"
            )
            
            logger.info(f"Created dynamic limit {name} for user {self.user_id}")
            return config
            
        except Exception as e:
            logger.error(f"Dynamic limit creation error: {str(e)}")
            raise
    
    async def enable_monitoring(self, context: Dict[str, Any] = None):
        """Enable real-time limit monitoring."""
        self.monitoring_active = True
        logger.info(f"Risk limit monitoring enabled for user {self.user_id}")
        
        if context:
            # Initial check
            await self.check_all_limits(context)
    
    async def disable_monitoring(self):
        """Disable real-time limit monitoring."""
        self.monitoring_active = False
        logger.info(f"Risk limit monitoring disabled for user {self.user_id}")


class RiskLimitFactory:
    """Factory class for creating risk limits."""
    
    @staticmethod
    def create_preset_limit(preset_type: str, **kwargs) -> RiskLimitConfig:
        """
        Create preset risk limit configuration.
        
        Args:
            preset_type: Type of preset limit
            **kwargs: Additional parameters
            
        Returns:
            Risk limit configuration
        """
        presets = {
            'daily_loss_5_percent': RiskLimitConfig(
                limit_name="Daily Loss Limit (5%)",
                limit_type=LimitType.DAILY_LOSS,
                limit_scope=LimitScope.PORTFOLIO,
                soft_limit=kwargs.get('portfolio_value', 100000) * 0.03,
                hard_limit=kwargs.get('portfolio_value', 100000) * 0.05,
                action_on_breach=LimitAction.HALT_TRADING,
                time_window='daily'
            ),
            'position_concentration_10_percent': RiskLimitConfig(
                limit_name="Position Concentration (10%)",
                limit_type=LimitType.CONCENTRATION,
                limit_scope=LimitScope.SYMBOL,
                scope_target=kwargs.get('symbol'),
                soft_limit=7.0,
                hard_limit=10.0,
                action_on_breach=LimitAction.REDUCE_POSITION
            ),
            'portfolio_var_2_percent': RiskLimitConfig(
                limit_name="Portfolio VaR (2%)",
                limit_type=LimitType.PORTFOLIO_VAR,
                limit_scope=LimitScope.PORTFOLIO,
                soft_limit=kwargs.get('portfolio_value', 100000) * 0.015,
                hard_limit=kwargs.get('portfolio_value', 100000) * 0.02,
                action_on_breach=LimitAction.WARN
            ),
            'max_drawdown_15_percent': RiskLimitConfig(
                limit_name="Maximum Drawdown (15%)",
                limit_type=LimitType.MAX_DRAWDOWN,
                limit_scope=LimitScope.PORTFOLIO,
                soft_limit=10.0,
                hard_limit=15.0,
                action_on_breach=LimitAction.LIQUIDATE,
                emergency_trigger=True,
                emergency_action=LimitAction.EMERGENCY_STOP
            ),
            'leverage_ratio_3x': RiskLimitConfig(
                limit_name="Leverage Ratio (3x)",
                limit_type=LimitType.LEVERAGE_RATIO,
                limit_scope=LimitScope.PORTFOLIO,
                soft_limit=2.5,
                hard_limit=3.0,
                action_on_breach=LimitAction.INCREASE_MARGIN
            )
        }
        
        if preset_type not in presets:
            raise ValueError(f"Unknown preset limit type: {preset_type}")
        
        return presets[preset_type]
    
    @staticmethod
    def create_custom_limit(limit_type: str, name: str, **kwargs) -> RiskLimitConfig:
        """
        Create custom risk limit configuration.
        
        Args:
            limit_type: Type of limit
            name: Limit name
            **kwargs: Limit configuration parameters
            
        Returns:
            Risk limit configuration
        """
        return RiskLimitConfig(
            limit_name=name,
            limit_type=LimitType(limit_type),
            limit_scope=LimitScope(kwargs.get('scope', 'portfolio')),
            scope_target=kwargs.get('target'),
            soft_limit=kwargs.get('soft_limit', 0),
            hard_limit=kwargs.get('hard_limit', 0),
            action_on_breach=LimitAction(kwargs.get('action', 'warn')),
            is_dynamic=kwargs.get('is_dynamic', False),
            time_window=kwargs.get('time_window'),
            emergency_trigger=kwargs.get('emergency_trigger', False),
            description=kwargs.get('description', ''),
            tags=kwargs.get('tags', [])
        )
