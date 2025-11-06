"""
Drawdown Protection and Risk Controls

Comprehensive drawdown protection mechanisms:
- Maximum portfolio drawdown monitoring
- Strategy-specific drawdown limits
- Automatic position reduction triggers
- Dynamic risk adjustment based on drawdown
- Recovery tracking and analysis

@author Trading Orchestrator System
@version 1.0
@date 2025-11-06
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import pandas as pd
import numpy as np
from collections import deque

from database.models.risk import RiskLevel
from database.models.trading import Position, Order, Trade, Account
from database.models.user import User
from config.database import get_db

logger = logging.getLogger(__name__)


class DrawdownPhase(Enum):
    """Drawdown phases for tracking."""
    RECOVERY = "recovery"
    NEUTRAL = "neutral"
    WARNING = "warning"
    DANGER = "danger"
    CRITICAL = "critical"


class ProtectionAction(Enum):
    """Actions when drawdown limits are breached."""
    WARN = "warn"
    REDUCE_POSITION_SIZE = "reduce_position_size"
    HALT_NEW_TRADES = "halt_new_trades"
    REDUCE_LEVERAGE = "reduce_leverage"
    CLOSE_POSITIONS = "close_positions"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class DrawdownMetric:
    """Drawdown measurement and tracking."""
    peak_value: float
    current_value: float
    drawdown_amount: float
    drawdown_percentage: float
    duration_days: int
    phase: DrawdownPhase
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DrawdownLimit:
    """Drawdown protection limit configuration."""
    limit_id: Optional[int] = None
    limit_name: str = ""
    max_drawdown_percentage: float = 0.0
    warning_threshold: float = 0.0
    action_on_breach: ProtectionAction = ProtectionAction.WARN
    recovery_target: float = 0.0
    auto_recovery: bool = True
    is_active: bool = True
    scope: str = "portfolio"  # portfolio, strategy, symbol, sector
    scope_target: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class RecoveryPlan:
    """Recovery plan for drawdown situations."""
    target_return: float
    time_horizon_days: int
    max_risk_per_trade: float
    max_positions: int
    focus_areas: List[str]
    restrictions: List[str]
    success_metrics: Dict[str, float]


class DrawdownProtector:
    """
    Drawdown Protection and Risk Management System
    
    Provides comprehensive drawdown protection with automatic
    risk adjustment and recovery planning.
    """
    
    def __init__(self, user_id: int):
        """
        Initialize Drawdown Protection System.
        
        Args:
            user_id: User identifier for drawdown tracking
        """
        self.user_id = user_id
        self.db = get_db()
        
        # Default drawdown limits
        self.default_limits = {
            "portfolio": DrawdownLimit(
                limit_name="Portfolio Max Drawdown",
                max_drawdown_percentage=15.0,
                warning_threshold=10.0,
                action_on_breach=ProtectionAction.REDUCE_POSITION_SIZE,
                recovery_target=5.0,
                auto_recovery=True
            ),
            "strategy_conservative": DrawdownLimit(
                limit_name="Conservative Strategy DD",
                max_drawdown_percentage=8.0,
                warning_threshold=5.0,
                action_on_breach=ProtectionAction.HALT_NEW_TRADES,
                recovery_target=3.0,
                scope="strategy"
            ),
            "daily": DrawdownLimit(
                limit_name="Daily Max Drawdown",
                max_drawdown_percentage=5.0,
                warning_threshold=3.0,
                action_on_breach=ProtectionAction.WARN,
                recovery_target=1.0,
                scope="portfolio"
            )
        }
        
        # Active limits
        self.active_limits: Dict[str, DrawdownLimit] = {}
        self.current_metrics: Dict[str, DrawdownMetric] = {}
        
        # Portfolio value history for drawdown calculation
        self.portfolio_history = deque(maxlen=252)  # 1 year of daily data
        self.peak_value = 0.0
        self.peak_date = None
        self.in_drawdown = False
        self.drawdown_start = None
        
        # Recovery tracking
        self.recovery_plans: Dict[str, RecoveryPlan] = {}
        self.recovery_progress = {}
        
        # Current protection status
        self.protection_level = "normal"
        self.restrictions_active = []
        self.last_calculation = None
        
        # Load active limits
        asyncio.create_task(self._load_active_limits())
        asyncio.create_task(self._initialize_portfolio_history())
        
        logger.info(f"DrawdownProtector initialized for user {self.user_id}")
    
    async def calculate_drawdown_metrics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive drawdown metrics for portfolio.
        
        Returns:
            Detailed drawdown analysis
        """
        try:
            # Get current portfolio value
            current_value = await self._get_current_portfolio_value()
            
            # Update portfolio history
            await self._update_portfolio_history(current_value)
            
            # Calculate drawdown metrics
            if len(self.portfolio_history) < 2:
                return {
                    "current_value": current_value,
                    "peak_value": current_value,
                    "drawdown_amount": 0.0,
                    "drawdown_percentage": 0.0,
                    "in_drawdown": False,
                    "drawdown_duration": 0,
                    "phase": DrawdownPhase.NEUTRAL
                }
            
            # Find current peak
            peak_value = max(self.portfolio_history)
            peak_index = list(self.portfolio_history).index(peak_value)
            
            # Calculate current drawdown
            drawdown_amount = peak_value - current_value
            drawdown_percentage = (drawdown_amount / peak_value * 100) if peak_value > 0 else 0
            
            # Calculate drawdown duration
            if current_value < peak_value:
                if not self.in_drawdown:
                    self.in_drawdown = True
                    self.drawdown_start = datetime.now() - timedelta(days=len(self.portfolio_history) - 1 - peak_index)
                drawdown_duration = (datetime.now() - self.drawdown_start).days
            else:
                drawdown_duration = 0
                self.in_drawdown = False
                self.drawdown_start = None
            
            # Determine phase
            phase = self._determine_drawdown_phase(drawdown_percentage)
            
            # Calculate additional metrics
            max_drawdown_1m = await self._calculate_max_drawdown_period(30)
            max_drawdown_3m = await self._calculate_max_drawdown_period(90)
            max_drawdown_6m = await self._calculate_max_drawdown_period(180)
            max_drawdown_1y = await self._calculate_max_drawdown_period(252)
            
            # Recovery metrics
            if self.in_drawdown and drawdown_percentage > 0:
                recovery_needed = current_value * (1 + drawdown_percentage / 100) - current_value
                recovery_percentage_needed = drawdown_percentage
            else:
                recovery_needed = 0
                recovery_percentage_needed = 0
            
            metrics = {
                "current_value": current_value,
                "peak_value": peak_value,
                "drawdown_amount": drawdown_amount,
                "drawdown_percentage": drawdown_percentage,
                "in_drawdown": self.in_drawdown,
                "drawdown_duration": drawdown_duration,
                "phase": phase.value,
                "max_drawdowns": {
                    "1_month": max_drawdown_1m,
                    "3_months": max_drawdown_3m,
                    "6_months": max_drawdown_6m,
                    "1_year": max_drawdown_1y
                },
                "recovery_metrics": {
                    "recovery_needed": recovery_needed,
                    "recovery_percentage": recovery_percentage_needed,
                    "days_to_recover_estimate": await self._estimate_recovery_time(recovery_percentage_needed)
                },
                "timestamp": datetime.now()
            }
            
            # Update current metrics
            self.current_metrics["portfolio"] = DrawdownMetric(
                peak_value=peak_value,
                current_value=current_value,
                drawdown_amount=drawdown_amount,
                drawdown_percentage=drawdown_percentage,
                duration_days=drawdown_duration,
                phase=phase
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Drawdown calculation error: {str(e)}")
            return {"error": f"Drawdown calculation failed: {str(e)}"}
    
    async def check_drawdown_limits(self) -> Dict[str, Any]:
        """
        Check all active drawdown limits against current metrics.
        
        Returns:
            Drawdown limits check results
        """
        try:
            # Calculate current metrics
            metrics = await self.calculate_drawdown_metrics()
            if "error" in metrics:
                return {"error": metrics["error"]}
            
            limit_check_results = {
                "overall_status": "ok",
                "current_metrics": metrics,
                "limits_checked": [],
                "breached_limits": [],
                "warning_limits": [],
                "actions_taken": [],
                "protection_level": "normal",
                "timestamp": datetime.now()
            }
            
            # Check each active limit
            for limit_key, limit_config in self.active_limits.items():
                if not limit_config.is_active:
                    continue
                
                # Get metrics for this scope
                scope_metrics = await self._get_scope_metrics(limit_config.scope, limit_config.scope_target)
                if not scope_metrics:
                    continue
                
                current_drawdown = scope_metrics["drawdown_percentage"]
                
                # Check against limits
                limit_status = {
                    "limit_name": limit_config.limit_name,
                    "scope": limit_config.scope,
                    "scope_target": limit_config.scope_target,
                    "current_drawdown": current_drawdown,
                    "max_drawdown": limit_config.max_drawdown_percentage,
                    "warning_threshold": limit_config.warning_threshold,
                    "action_required": limit_config.action_on_breach.value,
                    "status": "ok"
                }
                
                # Check for critical breach
                if current_drawdown >= limit_config.max_drawdown_percentage:
                    limit_status["status"] = "breached"
                    limit_status["severity"] = "critical"
                    limit_check_results["breached_limits"].append(limit_status)
                    
                    # Trigger protection action
                    action_result = await self._trigger_protection_action(limit_config, scope_metrics)
                    limit_check_results["actions_taken"].append(action_result)
                    
                # Check for warning
                elif current_drawdown >= limit_config.warning_threshold:
                    limit_status["status"] = "warning"
                    limit_status["severity"] = "warning"
                    limit_check_results["warning_limits"].append(limit_status)
                
                # Check for recovery
                elif current_drawdown <= limit_config.recovery_target and limit_config.auto_recovery:
                    limit_status["status"] = "recovery"
                    limit_status["severity"] = "info"
                    recovery_action = await self._trigger_recovery_action(limit_config, scope_metrics)
                    if recovery_action:
                        limit_check_results["actions_taken"].append(recovery_action)
                
                limit_check_results["limits_checked"].append(limit_status)
            
            # Determine overall protection level
            if limit_check_results["breached_limits"]:
                limit_check_results["protection_level"] = "critical"
                limit_check_results["overall_status"] = "breached"
            elif limit_check_results["warning_limits"]:
                limit_check_results["protection_level"] = "elevated"
                limit_check_results["overall_status"] = "warning"
            else:
                limit_check_results["protection_level"] = "normal"
                limit_check_results["overall_status"] = "ok"
            
            return limit_check_results
            
        except Exception as e:
            logger.error(f"Drawdown limits check error: {str(e)}")
            return {"error": f"Limits check failed: {str(e)}"}
    
    async def create_drawdown_limit(self, limit_name: str, max_drawdown: float,
                                  warning_threshold: float = None,
                                  action: ProtectionAction = ProtectionAction.WARN,
                                  scope: str = "portfolio",
                                  scope_target: str = None) -> Dict[str, Any]:
        """
        Create new drawdown protection limit.
        
        Args:
            limit_name: Name of the limit
            max_drawdown: Maximum allowed drawdown percentage
            warning_threshold: Warning threshold (defaults to 70% of max)
            action: Action to take when breached
            scope: Scope of protection (portfolio, strategy, symbol)
            scope_target: Target for scope-based limits
            
        Returns:
            Creation result
        """
        try:
            if warning_threshold is None:
                warning_threshold = max_drawdown * 0.7
            
            # Validate parameters
            if max_drawdown <= 0 or max_drawdown > 50:
                return {"success": False, "error": "Max drawdown must be between 0 and 50%"}
            
            if warning_threshold >= max_drawdown:
                return {"success": False, "error": "Warning threshold must be less than max drawdown"}
            
            # Create limit
            limit_config = DrawdownLimit(
                limit_name=limit_name,
                max_drawdown_percentage=max_drawdown,
                warning_threshold=warning_threshold,
                action_on_breach=action,
                recovery_target=max_drawdown * 0.5,
                scope=scope,
                scope_target=scope_target
            )
            
            # Generate ID and add to active limits
            limit_id = f"{scope}_{scope_target or 'general'}_{len(self.active_limits)}"
            self.active_limits[limit_id] = limit_config
            
            logger.info(f"Drawdown limit created for user {self.user_id}: {limit_name} ({max_drawdown}%)")
            
            return {
                "success": True,
                "limit_id": limit_id,
                "limit_name": limit_name,
                "max_drawdown": max_drawdown,
                "warning_threshold": warning_threshold,
                "action": action.value,
                "scope": scope,
                "scope_target": scope_target,
                "created_at": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Drawdown limit creation error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def generate_recovery_plan(self, target_return: float, 
                                   time_horizon_days: int = 90) -> Dict[str, Any]:
        """
        Generate recovery plan for drawdown situation.
        
        Args:
            target_return: Target return to achieve
            time_horizon_days: Time horizon for recovery
            
        Returns:
            Detailed recovery plan
        """
        try:
            # Get current metrics
            metrics = await self.calculate_drawdown_metrics()
            if "error" in metrics:
                return {"error": metrics["error"]}
            
            current_drawdown = metrics["drawdown_percentage"]
            current_value = metrics["current_value"]
            target_value = current_value * (1 + target_return / 100)
            
            if current_drawdown <= 0:
                return {"error": "No drawdown to recover from"}
            
            # Calculate required daily return
            daily_return_required = ((target_value / current_value) ** (1 / time_horizon_days) - 1) * 100
            
            # Risk assessment
            risk_level = self._assess_recovery_risk(current_drawdown, daily_return_required)
            
            # Strategy recommendations
            strategies = await self._recommend_recovery_strategies(current_drawdown, risk_level)
            
            # Position sizing recommendations
            max_risk_per_trade = min(2.0, 10.0 - current_drawdown)  # Reduce risk as drawdown increases
            
            # Create recovery plan
            recovery_plan = RecoveryPlan(
                target_return=target_return,
                time_horizon_days=time_horizon_days,
                max_risk_per_trade=max_risk_per_trade,
                max_positions=max(3, int(10 - current_drawdown / 2)),  # Fewer positions with higher drawdown
                focus_areas=strategies["focus_areas"],
                restrictions=strategies["restrictions"],
                success_metrics={
                    "daily_return_target": daily_return_required,
                    "max_drawdown_acceptable": max(3, current_drawdown * 0.5),
                    "min_win_rate_needed": max(60, 50 + current_drawdown),
                    "profit_factor_target": max(1.5, 1.0 + (current_drawdown / 20))
                }
            )
            
            # Store recovery plan
            plan_id = f"recovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.recovery_plans[plan_id] = recovery_plan
            self.recovery_progress[plan_id] = {
                "start_date": datetime.now(),
                "target_return": target_return,
                "current_progress": 0.0,
                "days_elapsed": 0,
                "success_probability": strategies["success_probability"]
            }
            
            return {
                "plan_id": plan_id,
                "current_drawdown": current_drawdown,
                "target_return": target_return,
                "target_value": target_value,
                "required_daily_return": daily_return_required,
                "risk_level": risk_level,
                "recovery_strategies": strategies,
                "plan_details": {
                    "max_risk_per_trade": recovery_plan.max_risk_per_trade,
                    "max_positions": recovery_plan.max_positions,
                    "focus_areas": recovery_plan.focus_areas,
                    "restrictions": recovery_plan.restrictions,
                    "success_metrics": recovery_plan.success_metrics
                },
                "created_at": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Recovery plan generation error: {str(e)}")
            return {"error": f"Recovery plan generation failed: {str(e)}"}
    
    async def track_recovery_progress(self, plan_id: str) -> Dict[str, Any]:
        """
        Track progress against recovery plan.
        
        Args:
            plan_id: Recovery plan identifier
            
        Returns:
            Recovery progress update
        """
        try:
            if plan_id not in self.recovery_plans:
                return {"error": f"Recovery plan {plan_id} not found"}
            
            plan = self.recovery_plans[plan_id]
            progress = self.recovery_progress[plan_id]
            
            # Get current portfolio value
            current_value = await self._get_current_portfolio_value()
            
            # Calculate progress
            start_value = current_value / (1 + progress["target_return"] / 100)
            progress_value = (current_value - start_value) / start_value * 100
            progress_percentage = min(100, (progress_value / progress["target_return"]) * 100)
            
            days_elapsed = (datetime.now() - progress["start_date"]).days
            days_remaining = max(0, plan.time_horizon_days - days_elapsed)
            
            # Check success probability update
            success_probability = await self._update_success_probability(plan_id, progress_percentage, days_elapsed)
            
            # Update progress
            progress.update({
                "current_progress": progress_percentage,
                "days_elapsed": days_elapsed,
                "days_remaining": days_remaining,
                "current_value": current_value,
                "progress_value": progress_value,
                "success_probability": success_probability,
                "last_updated": datetime.now()
            })
            
            # Generate recommendations
            recommendations = await self._generate_recovery_recommendations(plan_id, progress)
            
            return {
                "plan_id": plan_id,
                "progress_percentage": progress_percentage,
                "days_elapsed": days_elapsed,
                "days_remaining": days_remaining,
                "success_probability": success_probability,
                "recommendations": recommendations,
                "current_metrics": {
                    "portfolio_value": current_value,
                    "progress_value": progress_value,
                    "target_return": progress["target_return"]
                },
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Recovery progress tracking error: {str(e)}")
            return {"error": f"Progress tracking failed: {str(e)}"}
    
    async def get_drawdown_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive drawdown report.
        
        Returns:
            Detailed drawdown analysis and protection report
        """
        try:
            # Calculate current metrics
            metrics = await self.calculate_drawdown_metrics()
            
            # Check limits
            limit_status = await self.check_drawdown_limits()
            
            # Get active recovery plans
            active_plans = []
            for plan_id, plan in self.recovery_plans.items():
                progress = self.recovery_progress.get(plan_id, {})
                if progress.get("current_progress", 0) < 100:
                    active_plans.append({
                        "plan_id": plan_id,
                        "target_return": plan.target_return,
                        "progress": progress.get("current_progress", 0),
                        "days_remaining": progress.get("days_remaining", 0),
                        "success_probability": progress.get("success_probability", 0)
                    })
            
            # Calculate historical statistics
            historical_stats = await self._calculate_historical_drawdown_stats()
            
            # Generate insights and recommendations
            insights = await self._generate_drawdown_insights(metrics, limit_status)
            
            return {
                "current_status": {
                    "in_drawdown": metrics["in_drawdown"],
                    "drawdown_percentage": metrics["drawdown_percentage"],
                    "drawdown_duration": metrics["drawdown_duration"],
                    "phase": metrics["phase"],
                    "protection_level": limit_status.get("protection_level", "normal")
                },
                "limits_status": {
                    "active_limits": len(self.active_limits),
                    "breached_limits": len(limit_status.get("breached_limits", [])),
                    "warning_limits": len(limit_status.get("warning_limits", [])),
                    "overall_status": limit_status.get("overall_status", "ok")
                },
                "recovery_plans": {
                    "active_plans": active_plans,
                    "total_plans": len(self.recovery_plans)
                },
                "historical_analysis": historical_stats,
                "insights": insights,
                "recommendations": await self._generate_recommendations(metrics, limit_status),
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Drawdown report generation error: {str(e)}")
            return {"error": f"Report generation failed: {str(e)}"}
    
    async def _get_current_portfolio_value(self) -> float:
        """Get current portfolio value."""
        try:
            # Get account and positions
            account = self.db.query(Account).filter(
                Account.user_id == self.user_id
            ).first()
            
            if not account:
                return 0.0
            
            positions = self.db.query(Position).filter(
                and_(
                    Position.user_id == self.user_id,
                    Position.quantity != 0
                )
            ).all()
            
            total_position_value = sum(abs(pos.market_value or 0) for pos in positions)
            total_value = account.cash_balance + total_position_value
            
            return total_value
            
        except Exception as e:
            logger.error(f"Portfolio value calculation error: {str(e)}")
            return 0.0
    
    async def _update_portfolio_history(self, current_value: float):
        """Update portfolio value history."""
        try:
            self.portfolio_history.append(current_value)
            
            # Update peak tracking
            if current_value > self.peak_value:
                self.peak_value = current_value
                self.peak_date = datetime.now()
            
            self.last_calculation = datetime.now()
            
        except Exception as e:
            logger.error(f"Portfolio history update error: {str(e)}")
    
    def _determine_drawdown_phase(self, drawdown_percentage: float) -> DrawdownPhase:
        """Determine current drawdown phase."""
        if drawdown_percentage >= 20:
            return DrawdownPhase.CRITICAL
        elif drawdown_percentage >= 15:
            return DrawdownPhase.DANGER
        elif drawdown_percentage >= 10:
            return DrawdownPhase.WARNING
        elif drawdown_percentage >= 5:
            return DrawdownPhase.NEUTRAL
        else:
            return DrawdownPhase.RECOVERY
    
    async def _calculate_max_drawdown_period(self, days: int) -> float:
        """Calculate maximum drawdown over specified period."""
        try:
            if len(self.portfolio_history) < days:
                return 0.0
            
            # Get data for the period
            period_data = list(self.portfolio_history)[-days:]
            
            if len(period_data) < 2:
                return 0.0
            
            max_drawdown = 0.0
            peak = period_data[0]
            
            for value in period_data:
                if value > peak:
                    peak = value
                else:
                    drawdown = (peak - value) / peak * 100 if peak > 0 else 0
                    max_drawdown = max(max_drawdown, drawdown)
            
            return max_drawdown
            
        except Exception as e:
            logger.error(f"Max drawdown calculation error: {str(e)}")
            return 0.0
    
    async def _estimate_recovery_time(self, recovery_percentage: float) -> Optional[int]:
        """Estimate days to recover from drawdown."""
        try:
            if recovery_percentage <= 0:
                return None
            
            # Use recent performance to estimate recovery
            if len(self.portfolio_history) < 30:
                return None
            
            # Calculate average daily return over last 30 days
            recent_values = list(self.portfolio_history)[-30:]
            daily_returns = []
            
            for i in range(1, len(recent_values)):
                if recent_values[i-1] > 0:
                    daily_return = (recent_values[i] - recent_values[i-1]) / recent_values[i-1]
                    daily_returns.append(daily_return)
            
            if not daily_returns:
                return None
            
            avg_daily_return = np.mean(daily_returns) * 100
            
            if avg_daily_return <= 0:
                return None  # No positive returns to recover
            
            # Estimate recovery time
            recovery_days = recovery_percentage / avg_daily_return
            
            return max(1, int(recovery_days))
            
        except Exception as e:
            logger.error(f"Recovery time estimation error: {str(e)}")
            return None
    
    async def _get_scope_metrics(self, scope: str, scope_target: Optional[str]) -> Optional[Dict[str, float]]:
        """Get metrics for specific scope (strategy, symbol, etc.)."""
        try:
            if scope == "portfolio":
                return await self.calculate_drawdown_metrics()
            elif scope == "strategy" and scope_target:
                # Would calculate strategy-specific drawdown
                # For now, return portfolio metrics as placeholder
                return await self.calculate_drawdown_metrics()
            elif scope == "symbol" and scope_target:
                # Would calculate symbol-specific drawdown
                return None
            else:
                return None
                
        except Exception as e:
            logger.error(f"Scope metrics calculation error: {str(e)}")
            return None
    
    async def _trigger_protection_action(self, limit_config: DrawdownLimit, 
                                       metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger protection action when limit is breached."""
        try:
            action_taken = None
            details = {}
            
            if limit_config.action_on_breach == ProtectionAction.REDUCE_POSITION_SIZE:
                # Would reduce position sizes
                action_taken = "position_size_reduced"
                details["reduction_percentage"] = 25  # 25% reduction
                
            elif limit_config.action_on_breach == ProtectionAction.HALT_NEW_TRADES:
                # Would halt new trades
                action_taken = "new_trades_halted"
                details["halt_duration"] = "until_recovery"
                
            elif limit_config.action_on_breach == ProtectionAction.REDUCE_LEVERAGE:
                # Would reduce leverage
                action_taken = "leverage_reduced"
                details["leverage_reduction"] = "50%"
                
            elif limit_config.action_on_breach == ProtectionAction.CLOSE_POSITIONS:
                # Would close positions
                action_taken = "positions_closed"
                details["positions_affected"] = "all_non_essential"
                
            elif limit_config.action_on_breach == ProtectionAction.EMERGENCY_STOP:
                # Emergency stop
                action_taken = "emergency_stop_triggered"
                details["stop_type"] = "complete_trading_halt"
            
            logger.warning(f"Drawdown protection action for user {self.user_id}: {action_taken}")
            
            return {
                "action_taken": action_taken,
                "limit_name": limit_config.limit_name,
                "current_drawdown": metrics["drawdown_percentage"],
                "details": details,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Protection action trigger error: {str(e)}")
            return {"error": str(e)}
    
    async def _trigger_recovery_action(self, limit_config: DrawdownLimit, 
                                     metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Trigger recovery action when drawdown improves."""
        try:
            if not limit_config.auto_recovery:
                return None
            
            # Recovery actions could include:
            # - Resuming normal trading
            # - Increasing position sizes
            # - Removing restrictions
            
            action_taken = "recovery_resumed"
            
            return {
                "action_taken": action_taken,
                "limit_name": limit_config.limit_name,
                "current_drawdown": metrics["drawdown_percentage"],
                "recovery_threshold": limit_config.recovery_target,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Recovery action error: {str(e)}")
            return None
    
    def _assess_recovery_risk(self, drawdown_percentage: float, 
                            daily_return_required: float) -> str:
        """Assess risk level of recovery attempt."""
        try:
            risk_score = 0
            
            # Drawdown risk
            if drawdown_percentage > 20:
                risk_score += 3
            elif drawdown_percentage > 15:
                risk_score += 2
            elif drawdown_percentage > 10:
                risk_score += 1
            
            # Required return risk
            if daily_return_required > 2:
                risk_score += 3
            elif daily_return_required > 1:
                risk_score += 2
            elif daily_return_required > 0.5:
                risk_score += 1
            
            # Determine risk level
            if risk_score >= 5:
                return "very_high"
            elif risk_score >= 3:
                return "high"
            elif risk_score >= 2:
                return "medium"
            else:
                return "low"
                
        except Exception as e:
            logger.error(f"Recovery risk assessment error: {str(e)}")
            return "medium"
    
    async def _recommend_recovery_strategies(self, drawdown_percentage: float, 
                                          risk_level: str) -> Dict[str, Any]:
        """Recommend recovery strategies based on situation."""
        try:
            strategies = {
                "focus_areas": [],
                "restrictions": [],
                "success_probability": 0.0
            }
            
            # Base strategies on drawdown level
            if drawdown_percentage >= 20:
                strategies["focus_areas"] = [
                    "capital_preservation",
                    "low_risk_trades",
                    "diversification"
                ]
                strategies["restrictions"] = [
                    "no_high_leverage",
                    "max_2_positions",
                    "strict_stop_losses"
                ]
                strategies["success_probability"] = 30.0
                
            elif drawdown_percentage >= 15:
                strategies["focus_areas"] = [
                    "moderate_risk_trades",
                    "trend_following",
                    "quality_stocks"
                ]
                strategies["restrictions"] = [
                    "max_3_positions",
                    "tight_risk_management"
                ]
                strategies["success_probability"] = 50.0
                
            elif drawdown_percentage >= 10:
                strategies["focus_areas"] = [
                    "balanced_approach",
                    "sector_rotation",
                    "value_opportunities"
                ]
                strategies["restrictions"] = [
                    "max_5_positions"
                ]
                strategies["success_probability"] = 65.0
                
            else:
                strategies["focus_areas"] = [
                    "normal_trading",
                    "opportunity_focus"
                ]
                strategies["restrictions"] = []
                strategies["success_probability"] = 80.0
            
            # Adjust for risk level
            if risk_level == "very_high":
                strategies["success_probability"] *= 0.7
                strategies["restrictions"].append("very_conservative_approach")
            elif risk_level == "high":
                strategies["success_probability"] *= 0.85
                strategies["restrictions"].append("conservative_approach")
            
            return strategies
            
        except Exception as e:
            logger.error(f"Recovery strategies recommendation error: {str(e)}")
            return {
                "focus_areas": ["capital_preservation"],
                "restrictions": ["very_conservative"],
                "success_probability": 50.0
            }
    
    async def _update_success_probability(self, plan_id: str, progress_percentage: float, 
                                        days_elapsed: int) -> float:
        """Update success probability based on progress."""
        try:
            progress = self.recovery_progress.get(plan_id)
            if not progress:
                return 0.0
            
            base_probability = progress["success_probability"]
            plan = self.recovery_plans[plan_id]
            
            # Adjust based on progress rate
            expected_progress = (days_elapsed / plan.time_horizon_days) * 100
            progress_rate = progress_percentage / expected_progress if expected_progress > 0 else 1
            
            # Update probability
            if progress_rate > 1.1:  # Ahead of schedule
                adjusted_probability = min(95, base_probability * 1.2)
            elif progress_rate < 0.9:  # Behind schedule
                adjusted_probability = max(5, base_probability * 0.8)
            else:  # On track
                adjusted_probability = base_probability
            
            return adjusted_probability
            
        except Exception as e:
            logger.error(f"Success probability update error: {str(e)}")
            return 50.0
    
    async def _generate_recovery_recommendations(self, plan_id: str, progress: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on recovery progress."""
        try:
            recommendations = []
            
            progress_percentage = progress.get("current_progress", 0)
            days_elapsed = progress.get("days_elapsed", 0)
            success_probability = progress.get("success_probability", 50)
            
            if progress_percentage < 25 and days_elapsed > 30:
                recommendations.append("Consider revising recovery plan - progress is slow")
            
            if success_probability < 30:
                recommendations.append("High risk recovery - consider more conservative approach")
            elif success_probability > 80:
                recommendations.append("Recovery is on track - maintain current strategy")
            
            if progress_percentage > 75:
                recommendations.append("Nearly recovered - prepare for normal trading resumption")
            
            return recommendations if recommendations else ["Continue with current recovery strategy"]
            
        except Exception as e:
            logger.error(f"Recovery recommendations error: {str(e)}")
            return ["Monitor recovery progress closely"]
    
    async def _calculate_historical_drawdown_stats(self) -> Dict[str, Any]:
        """Calculate historical drawdown statistics."""
        try:
            if len(self.portfolio_history) < 60:
                return {"error": "Insufficient historical data"}
            
            # Calculate all drawdowns
            drawdowns = []
            peak = self.portfolio_history[0]
            
            for value in self.portfolio_history:
                if value > peak:
                    peak = value
                else:
                    drawdown = (peak - value) / peak * 100 if peak > 0 else 0
                    drawdowns.append(drawdown)
            
            if not drawdowns:
                return {"error": "No drawdowns found"}
            
            return {
                "max_drawdown": max(drawdowns),
                "average_drawdown": np.mean(drawdowns),
                "drawdown_frequency": len(drawdowns) / len(self.portfolio_history) * 100,
                "median_drawdown": np.median(drawdowns),
                "drawdown_std": np.std(drawdowns),
                "total_drawdown_periods": len(drawdowns)
            }
            
        except Exception as e:
            logger.error(f"Historical stats calculation error: {str(e)}")
            return {"error": "Calculation failed"}
    
    async def _generate_drawdown_insights(self, metrics: Dict[str, Any], 
                                        limit_status: Dict[str, Any]) -> List[str]:
        """Generate insights from drawdown analysis."""
        try:
            insights = []
            
            # Current state insights
            if metrics["in_drawdown"]:
                insights.append(f"Currently in {metrics['drawdown_duration']}-day drawdown period")
            
            # Phase insights
            phase = metrics["phase"]
            if phase == DrawdownPhase.CRITICAL.value:
                insights.append("Drawdown level is critical - immediate action required")
            elif phase == DrawdownPhase.DANGER.value:
                insights.append("Drawdown level is dangerous - consider risk reduction")
            elif phase == DrawdownPhase.WARNING.value:
                insights.append("Drawdown level requires attention - monitor closely")
            
            # Recovery insights
            recovery_metrics = metrics.get("recovery_metrics", {})
            if recovery_metrics.get("recovery_percentage", 0) > 10:
                days_estimate = recovery_metrics.get("days_to_recover_estimate")
                if days_estimate:
                    insights.append(f"Estimated recovery time: {days_estimate} days")
            
            # Limit status insights
            breached = len(limit_status.get("breached_limits", []))
            warnings = len(limit_status.get("warning_limits", []))
            
            if breached > 0:
                insights.append(f"{breached} drawdown limit(s) breached - protection activated")
            if warnings > 0:
                insights.append(f"{warnings} drawdown limit(s) approaching threshold")
            
            return insights if insights else ["No significant drawdown insights at this time"]
            
        except Exception as e:
            logger.error(f"Drawdown insights generation error: {str(e)}")
            return ["Unable to generate insights"]
    
    async def _generate_recommendations(self, metrics: Dict[str, Any], 
                                      limit_status: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations."""
        try:
            recommendations = []
            
            # Based on current drawdown
            drawdown = metrics["drawdown_percentage"]
            if drawdown >= 15:
                recommendations.append("Consider implementing emergency stop measures")
            elif drawdown >= 10:
                recommendations.append("Review and reduce position sizes")
            elif drawdown >= 5:
                recommendations.append("Monitor positions more closely")
            
            # Based on limit status
            protection_level = limit_status.get("protection_level", "normal")
            if protection_level == "critical":
                recommendations.append("All trading should be halted immediately")
            elif protection_level == "elevated":
                recommendations.append("Reduce risk exposure and limit new positions")
            
            # Based on duration
            duration = metrics.get("drawdown_duration", 0)
            if duration > 30:
                recommendations.append("Extended drawdown - consider strategy review")
            elif duration > 60:
                recommendations.append("Long-term drawdown - fundamental analysis needed")
            
            return recommendations if recommendations else ["Continue current risk management approach"]
            
        except Exception as e:
            logger.error(f"Recommendations generation error: {str(e)}")
            return ["Monitor drawdown levels closely"]
    
    async def _load_active_limits(self):
        """Load active limits from database."""
        try:
            # In practice, this would load from database
            # For now, use defaults
            self.active_limits = self.default_limits.copy()
            logger.info(f"Loaded {len(self.active_limits)} drawdown limits for user {self.user_id}")
            
        except Exception as e:
            logger.error(f"Active limits loading error: {str(e)}")
            self.active_limits = self.default_limits.copy()
    
    async def _initialize_portfolio_history(self):
        """Initialize portfolio history with historical data."""
        try:
            # In practice, this would load historical portfolio values
            # For now, start with current value
            current_value = await self._get_current_portfolio_value()
            self.portfolio_history.append(current_value)
            
        except Exception as e:
            logger.error(f"Portfolio history initialization error: {str(e)}")
    
    def close(self):
        """Cleanup resources."""
        self.db.close()