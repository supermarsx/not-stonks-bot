"""
Risk Limits Checking System

Monitors and enforces various risk limits:
- Position size limits
- Daily loss limits
- Order limits
- Exposure limits
- Concentration limits
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func

from config.database import get_db
from database.models.risk import RiskLimit, RiskLevel
from database.models.trading import Position, Order, Trade

logger = logging.getLogger(__name__)


class RiskLimitChecker:
    """
    Manages and enforces all types of risk limits.
    """
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.db = get_db()
        
        # Cache for limit calculations
        self._limits_cache = {}
        self._cache_ttl = 300  # 5 minutes
        
        logger.info(f"RiskLimitChecker initialized for user {self.user_id}")
    
    async def check_order_limits(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if order complies with all risk limits.
        
        Args:
            order_data: Order information to validate
            
        Returns:
            Validation result with status and details
        """
        result = {
            "approved": True,
            "rejection_reason": None,
            "warnings": [],
            "limit_checks": [],
            "actions_taken": []
        }
        
        try:
            # Get all active risk limits
            active_limits = self.db.query(RiskLimit).filter(
                and_(
                    RiskLimit.user_id == self.user_id,
                    RiskLimit.is_active == True
                )
            ).all()
            
            for limit in active_limits:
                check_result = await self._check_single_limit(limit, order_data)
                result["limit_checks"].append({
                    "limit_id": limit.id,
                    "limit_name": limit.limit_name,
                    "limit_type": limit.limit_type,
                    "check_result": check_result
                })
                
                if not check_result["approved"]:
                    result["approved"] = False
                    result["rejection_reason"] = check_result["rejection_reason"]
                    
                    # Apply breach action
                    if limit.breach_action == "block":
                        break  # Stop checking if order should be blocked
                    elif limit.breach_action == "warn":
                        result["warnings"].append(check_result["warning_message"])
                
                # Update current values for monitoring
                await self._update_limit_current_value(limit, order_data)
            
        except Exception as e:
            logger.error(f"Order limit check error: {str(e)}")
            result.update({
                "approved": False,
                "rejection_reason": f"Limit check error: {str(e)}"
            })
        
        return result
    
    async def check_position_limits(self, position: Position) -> List[Dict[str, Any]]:
        """
        Check risk limits for an existing position.
        
        Args:
            position: Position to check
            
        Returns:
            List of limit violations
        """
        violations = []
        
        try:
            # Get position-related limits
            position_limits = self.db.query(RiskLimit).filter(
                and_(
                    RiskLimit.user_id == self.user_id,
                    RiskLimit.is_active == True,
                    or_(
                        RiskLimit.scope == "global",
                        and_(
                            RiskLimit.scope == "per_symbol",
                            RiskLimit.scope_target == position.symbol
                        )
                    )
                )
            ).all()
            
            for limit in position_limits:
                if limit.limit_type in ["position_size", "exposure", "concentration"]:
                    violation = await self._check_position_limit(limit, position)
                    if violation:
                        violations.append(violation)
                        
                        # Mark limit as breached
                        limit.is_breached = True
                        limit.last_breached_at = datetime.now()
                        
                        logger.warning(f"Position limit breach for user {self.user_id}: {limit.limit_name}")
            
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Position limit check error: {str(e)}")
        
        return violations
    
    async def check_daily_limits(self) -> Dict[str, Any]:
        """
        Check daily risk limits (loss, order count, etc.).
        
        Returns:
            Daily limit check results
        """
        results = {
            "date": datetime.now().date(),
            "limits_checked": 0,
            "violations": [],
            "warnings": []
        }
        
        try:
            today = datetime.now().date()
            
            # Get daily-related limits
            daily_limits = self.db.query(RiskLimit).filter(
                and_(
                    RiskLimit.user_id == self.user_id,
                    RiskLimit.is_active == True,
                    RiskLimit.limit_type.in_(["daily_loss", "daily_orders", "daily_exposure"])
                )
            ).all()
            
            results["limits_checked"] = len(daily_limits)
            
            for limit in daily_limits:
                violation = await self._check_daily_limit(limit, today)
                if violation:
                    if violation.get("severity") == "critical":
                        results["violations"].append(violation)
                    else:
                        results["warnings"].append(violation)
            
        except Exception as e:
            logger.error(f"Daily limit check error: {str(e)}")
            results["error"] = str(e)
        
        return results
    
    async def get_limit_status(self) -> Dict[str, Any]:
        """
        Get current status of all risk limits.
        
        Returns:
            Status of all limits with current usage
        """
        status = {
            "user_id": self.user_id,
            "limits": {},
            "summary": {
                "total_limits": 0,
                "active_limits": 0,
                "breached_limits": 0,
                "warning_limits": 0
            }
        }
        
        try:
            limits = self.db.query(RiskLimit).filter(
                RiskLimit.user_id == self.user_id
            ).all()
            
            status["summary"]["total_limits"] = len(limits)
            
            for limit in limits:
                usage_percent = (limit.current_value / limit.limit_value * 100) if limit.limit_value > 0 else 0
                
                status["limits"][limit.id] = {
                    "name": limit.limit_name,
                    "type": limit.limit_type,
                    "limit_value": limit.limit_value,
                    "current_value": limit.current_value,
                    "usage_percent": usage_percent,
                    "is_active": limit.is_active,
                    "is_breached": limit.is_breached,
                    "scope": limit.scope,
                    "scope_target": limit.scope_target,
                    "breach_action": limit.breach_action,
                    "warning_threshold": limit.warning_threshold,
                    "last_checked": limit.last_checked_at
                }
                
                if limit.is_active:
                    status["summary"]["active_limits"] += 1
                    
                    if limit.is_breached:
                        status["summary"]["breached_limits"] += 1
                    elif usage_percent >= (limit.warning_threshold * 100 if limit.warning_threshold else 80):
                        status["summary"]["warning_limits"] += 1
            
        except Exception as e:
            logger.error(f"Limit status error: {str(e)}")
            status["error"] = str(e)
        
        return status
    
    async def _check_single_limit(self, limit: RiskLimit, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check a single risk limit against an order."""
        try:
            current_value = await self._get_current_limit_value(limit, order_data)
            
            # Update the limit's current value
            limit.current_value = current_value
            limit.last_checked_at = datetime.now()
            
            # Check if limit is breached
            is_breached = current_value > limit.limit_value
            
            if is_breached:
                return {
                    "approved": False,
                    "rejection_reason": f"{limit.limit_name} exceeded: {current_value:.2f} > {limit.limit_value:.2f}",
                    "warning_message": f"Limit {limit.limit_name} approaching threshold: {current_value:.2f}/{limit.limit_value:.2f}",
                    "breach_severity": "critical" if current_value > limit.limit_value * 1.1 else "warning"
                }
            
            # Check warning threshold
            if (limit.warning_threshold and 
                current_value >= limit.limit_value * limit.warning_threshold):
                return {
                    "approved": True,
                    "rejection_reason": None,
                    "warning_message": f"Limit {limit.limit_name} at warning threshold: {current_value:.2f}/{limit.limit_value:.2f}",
                    "breach_severity": "warning"
                }
            
            return {
                "approved": True,
                "rejection_reason": None,
                "warning_message": None,
                "breach_severity": None
            }
            
        except Exception as e:
            logger.error(f"Single limit check error: {str(e)}")
            return {
                "approved": False,
                "rejection_reason": f"Limit check error: {str(e)}"
            }
    
    async def _check_position_limit(self, limit: RiskLimit, position: Position) -> Optional[Dict[str, Any]]:
        """Check position against a specific limit."""
        try:
            if limit.limit_type == "position_size":
                current_size = abs(position.quantity)
                if current_size > limit.limit_value:
                    return {
                        "type": "position_size",
                        "current_value": current_size,
                        "limit_value": limit.limit_value,
                        "severity": "critical" if current_size > limit.limit_value * 1.2 else "warning",
                        "action_required": "liquidate" if current_size > limit.limit_value * 1.5 else "reduce",
                        "limit_id": limit.id,
                        "symbol": position.symbol
                    }
            
            elif limit.limit_type == "exposure":
                current_exposure = abs(position.market_value or 0)
                if current_exposure > limit.limit_value:
                    return {
                        "type": "exposure",
                        "current_value": current_exposure,
                        "limit_value": limit.limit_value,
                        "severity": "critical" if current_exposure > limit.limit_value * 1.2 else "warning",
                        "action_required": "liquidate" if current_exposure > limit.limit_value * 1.5 else "reduce",
                        "limit_id": limit.id,
                        "symbol": position.symbol
                    }
            
            elif limit.limit_type == "concentration":
                # Calculate position as percentage of total portfolio
                total_portfolio = sum(abs(pos.market_value or 0) for pos in 
                                    self.db.query(Position).filter(
                                        Position.user_id == self.user_id
                                    ).all())
                
                if total_portfolio > 0:
                    concentration = abs(position.market_value or 0) / total_portfolio * 100
                    if concentration > limit.limit_value:
                        return {
                            "type": "concentration",
                            "current_value": concentration,
                            "limit_value": limit.limit_value,
                            "severity": "critical" if concentration > limit.limit_value * 1.3 else "warning",
                            "action_required": "reduce",
                            "limit_id": limit.id,
                            "symbol": position.symbol
                        }
            
        except Exception as e:
            logger.error(f"Position limit check error: {str(e)}")
        
        return None
    
    async def _check_daily_limit(self, limit: RiskLimit, check_date: datetime.date) -> Optional[Dict[str, Any]]:
        """Check daily limits."""
        try:
            if limit.limit_type == "daily_loss":
                # Calculate today's realized P&L
                trades_today = self.db.query(Trade).filter(
                    and_(
                        Trade.user_id == self.user_id,
                        Trade.executed_at >= datetime.combine(check_date, datetime.min.time()),
                        Trade.executed_at < datetime.combine(check_date + timedelta(days=1), datetime.min.time()),
                        Trade.realized_pnl.isnot(None)
                    )
                ).all()
                
                daily_loss = sum(trade.realized_pnl or 0 for trade in trades_today)
                
                if daily_loss < -limit.limit_value:
                    return {
                        "type": "daily_loss",
                        "current_value": abs(daily_loss),
                        "limit_value": limit.limit_value,
                        "severity": "critical",
                        "action_required": "halt_trading",
                        "limit_id": limit.id,
                        "daily_pnl": daily_loss
                    }
            
            elif limit.limit_type == "daily_orders":
                # Count orders today
                orders_today = self.db.query(Order).filter(
                    and_(
                        Order.user_id == self.user_id,
                        Order.submitted_at >= datetime.combine(check_date, datetime.min.time()),
                        Order.submitted_at < datetime.combine(check_date + timedelta(days=1), datetime.min.time())
                    )
                ).count()
                
                if orders_today > limit.limit_value:
                    return {
                        "type": "daily_orders",
                        "current_value": orders_today,
                        "limit_value": limit.limit_value,
                        "severity": "warning",
                        "action_required": "warn",
                        "limit_id": limit.id
                    }
            
            elif limit.limit_type == "daily_exposure":
                # Calculate today's exposure change
                positions = self.db.query(Position).filter(
                    Position.user_id == self.user_id
                ).all()
                
                current_exposure = sum(abs(pos.market_value or 0) for pos in positions)
                
                if current_exposure > limit.limit_value:
                    return {
                        "type": "daily_exposure",
                        "current_value": current_exposure,
                        "limit_value": limit.limit_value,
                        "severity": "warning",
                        "action_required": "warn",
                        "limit_id": limit.id
                    }
            
        except Exception as e:
            logger.error(f"Daily limit check error: {str(e)}")
        
        return None
    
    async def _get_current_limit_value(self, limit: RiskLimit, context: Dict[str, Any]) -> float:
        """Get current value for a specific limit type."""
        try:
            if limit.limit_type == "position_size":
                # Check if order would exceed position size for this symbol
                symbol = context.get("symbol")
                existing_position = self.db.query(Position).filter(
                    and_(
                        Position.user_id == self.user_id,
                        Position.symbol == symbol,
                        Position.side == context.get("side")
                    )
                ).first()
                
                current_quantity = existing_position.quantity if existing_position else 0
                new_quantity = current_quantity + context.get("quantity", 0)
                
                return abs(new_quantity)
            
            elif limit.limit_type == "order_value":
                # Calculate total order value
                quantity = context.get("quantity", 0)
                price = context.get("limit_price") or context.get("estimated_price", 0)
                return quantity * price
            
            elif limit.limit_type == "daily_loss":
                # Get today's realized P&L
                today = datetime.now().date()
                trades_today = self.db.query(Trade).filter(
                    and_(
                        Trade.user_id == self.user_id,
                        Trade.executed_at >= datetime.combine(today, datetime.min.time()),
                        Trade.realized_pnl.isnot(None)
                    )
                ).all()
                
                return abs(sum(trade.realized_pnl or 0 for trade in trades_today))
            
            elif limit.limit_type == "daily_orders":
                # Count orders today
                today = datetime.now().date()
                orders_today = self.db.query(Order).filter(
                    and_(
                        Order.user_id == self.user_id,
                        Order.submitted_at >= datetime.combine(today, datetime.min.time())
                    )
                ).count()
                
                return orders_today
            
            else:
                # Default calculation - sum of current exposures
                positions = self.db.query(Position).filter(
                    Position.user_id == self.user_id
                ).all()
                
                return sum(abs(pos.market_value or 0) for pos in positions)
                
        except Exception as e:
            logger.error(f"Current limit value calculation error: {str(e)}")
            return 0.0
    
    async def _update_limit_current_value(self, limit: RiskLimit, order_data: Dict[str, Any]) -> None:
        """Update current value for a limit after order processing."""
        try:
            new_value = await self._get_current_limit_value(limit, order_data)
            limit.current_value = new_value
            limit.last_checked_at = datetime.now()
            
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Limit value update error: {str(e)}")
    
    def close(self):
        """Cleanup resources."""
        self.db.close()
