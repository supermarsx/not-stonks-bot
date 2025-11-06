"""
Margin Trading Controls and Risk Management

Comprehensive margin trading safeguards including:
- Margin trading toggle/controls
- Margin requirement validation
- Leverage limit enforcement
- Maintenance margin monitoring
- Margin call prevention
- Regulatory margin compliance

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
import math

from database.models.risk import RiskLevel
from database.models.trading import Position, Order, Trade, Account
from database.models.user import User
from config.database import get_db

logger = logging.getLogger(__name__)


class MarginAccountType(Enum):
    """Types of margin accounts."""
    REGULAR = "regular"           # Standard margin
    PORTFOLIO_MARGIN = "portfolio" # Portfolio margin
    MINIMUM = "minimum"           # Minimum margin
    RESTRICTED = "restricted"     # Restricted margin


class MarginCallAction(Enum):
    """Actions when margin requirements are not met."""
    WARNING = "warning"
    REDUCE_POSITIONS = "reduce_positions"
    ADD_COLLATERAL = "add_collateral"
    FORCE_LIQUIDATION = "force_liquidation"
    HALT_TRADING = "halt_trading"


class LeverageLevel(Enum):
    """Leverage levels for different risk profiles."""
    CONSERVATIVE = 1.5    # 1.5x leverage
    MODERATE = 2.0        # 2x leverage
    AGGRESSIVE = 3.0      # 3x leverage
    HIGH_RISK = 5.0       # 5x leverage (day trading)


@dataclass
class MarginRequirement:
    """Margin requirement for different instruments."""
    instrument_type: str
    initial_margin_rate: float
    maintenance_margin_rate: float
    pattern_day_trader: bool
    regulatory_limit: Optional[float] = None
    custom_limits: Dict[str, float] = field(default_factory=dict)


@dataclass
class MarginStatus:
    """Current margin status and health metrics."""
    account_value: float
    margin_used: float
    margin_available: float
    maintenance_margin_required: float
    leverage_ratio: float
    margin_call_level: float
    liquidation_level: float
    margin_health_score: float  # 0-100
    days_to_margin_call: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MarginAlert:
    """Margin-related alert."""
    alert_type: str
    severity: str  # info, warning, critical, emergency
    current_value: float
    threshold: float
    message: str
    recommended_action: str
    timestamp: datetime = field(default_factory=datetime.now)


class MarginController:
    """
    Margin Trading Controls and Risk Management System
    
    Provides comprehensive margin trading safeguards including
    real-time monitoring, automatic controls, and regulatory compliance.
    """
    
    def __init__(self, user_id: int):
        """
        Initialize Margin Controller.
        
        Args:
            user_id: User identifier for margin tracking
        """
        self.user_id = user_id
        self.db = get_db()
        
        # Margin account settings
        self.account_type = MarginAccountType.REGULAR
        self.maximum_leverage = LeverageLevel.MODERATE
        self.margin_trading_enabled = False
        
        # Margin requirements by instrument (simplified)
        self.margin_requirements = {
            "equity": MarginRequirement("equity", 0.50, 0.25, False, 4.0),
            "etf": MarginRequirement("etf", 0.50, 0.25, False, 4.0),
            "option": MarginRequirement("option", 0.20, 0.15, False, None),
            "future": MarginRequirement("future", 0.05, 0.03, True, None),
            "forex": MarginRequirement("forex", 0.01, 0.005, False, 50.0),
            "crypto": MarginRequirement("crypto", 0.50, 0.30, False, 2.0)
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            "margin_call_warning": 0.80,  # 80% of margin call level
            "maintenance_warning": 0.90,   # 90% of maintenance margin
            "leverage_warning": 0.95       # 95% of maximum leverage
        }
        
        # Current status
        self.current_status = None
        self.margin_alerts = []
        self.last_calculated = None
        
        logger.info(f"MarginController initialized for user {self.user_id}")
    
    async def enable_margin_trading(self, leverage_limit: LeverageLevel = LeverageLevel.MODERATE,
                                  account_type: MarginAccountType = MarginAccountType.REGULAR) -> Dict[str, Any]:
        """
        Enable margin trading with specified limits.
        
        Args:
            leverage_limit: Maximum leverage allowed
            account_type: Type of margin account
            
        Returns:
            Margin trading enablement result
        """
        try:
            # Validate account can support margin trading
            account = self.db.query(Account).filter(
                Account.user_id == self.user_id
            ).first()
            
            if not account:
                return {"success": False, "error": "Account not found"}
            
            if account.cash_balance < 2000:  # Minimum for margin
                return {
                    "success": False, 
                    "error": "Insufficient cash balance for margin trading (minimum $2,000 required)"
                }
            
            # Check regulatory compliance
            compliance_result = await self._check_regulatory_compliance(account_type)
            if not compliance_result["compliant"]:
                return {
                    "success": False,
                    "error": f"Regulatory compliance issue: {compliance_result['issue']}"
                }
            
            # Set margin settings
            self.margin_trading_enabled = True
            self.maximum_leverage = leverage_limit
            self.account_type = account_type
            
            # Update account margin status
            account.margin_enabled = True
            account.maximum_leverage = leverage_limit.value
            self.db.commit()
            
            logger.info(f"Margin trading enabled for user {self.user_id}: {leverage_limit.value}x leverage")
            
            return {
                "success": True,
                "leverage_limit": leverage_limit.value,
                "account_type": account_type.value,
                "minimum_maintenance_margin": account.cash_balance * 0.25,
                "regulatory_compliance": compliance_result,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Margin trading enablement error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def disable_margin_trading(self, liquidate_positions: bool = False) -> Dict[str, Any]:
        """
        Disable margin trading and optionally liquidate positions.
        
        Args:
            liquidate_positions: Whether to liquidate all margin positions
            
        Returns:
            Margin trading disablement result
        """
        try:
            if not self.margin_trading_enabled:
                return {"success": True, "message": "Margin trading already disabled"}
            
            # Check current margin status
            status = await self.calculate_margin_status()
            if "error" in status:
                return {"success": False, "error": status["error"]}
            
            actions_taken = []
            
            if liquidate_positions:
                # Liquidate all margin positions
                liquidation_result = await self._liquidate_all_margin_positions()
                actions_taken.append(("position_liquidation", liquidation_result))
            
            # Disable margin trading
            self.margin_trading_enabled = False
            self.maximum_leverage = LeverageLevel.CONSERVATIVE  # Reset to lowest
            
            # Update account
            account = self.db.query(Account).filter(
                Account.user_id == self.user_id
            ).first()
            if account:
                account.margin_enabled = False
                account.maximum_leverage = 1.0
                self.db.commit()
            
            logger.info(f"Margin trading disabled for user {self.user_id}")
            
            return {
                "success": True,
                "margin_disabled": True,
                "positions_liquidated": liquidate_positions,
                "actions_taken": actions_taken,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Margin trading disablement error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def calculate_margin_requirement(self, symbol: str, quantity: float, 
                                         side: str, price: float) -> Dict[str, Any]:
        """
        Calculate margin requirement for proposed trade.
        
        Args:
            symbol: Trading symbol
            quantity: Number of shares/units
            side: Buy/sell side
            price: Trade price
            
        Returns:
            Margin requirement calculation
        """
        try:
            # Determine instrument type (simplified classification)
            instrument_type = self._classify_instrument_type(symbol)
            
            if instrument_type not in self.margin_requirements:
                return {"error": f"No margin requirements defined for instrument type: {instrument_type}"}
            
            requirement = self.margin_requirements[instrument_type]
            
            # Calculate position value
            position_value = abs(quantity * price)
            
            # Calculate required margin
            initial_margin = position_value * requirement.initial_margin_rate
            maintenance_margin = position_value * requirement.maintenance_margin_rate
            
            # Apply regulatory limits
            if requirement.regulatory_limit:
                max_leverage = requirement.regulatory_limit
                regulatory_margin = position_value / max_leverage
                initial_margin = max(initial_margin, regulatory_margin)
            
            # Calculate effective leverage
            effective_leverage = position_value / initial_margin if initial_margin > 0 else 1.0
            
            # Check against account leverage limit
            account_leverage_limit = self.maximum_leverage.value
            if effective_leverage > account_leverage_limit:
                # Calculate adjusted margin to meet leverage limit
                adjusted_initial_margin = position_value / account_leverage_limit
                excess_leverage = effective_leverage - account_leverage_limit
                
                return {
                    "position_value": position_value,
                    "instrument_type": instrument_type,
                    "initial_margin": adjusted_initial_margin,
                    "maintenance_margin": maintenance_margin,
                    "effective_leverage": effective_leverage,
                    "requested_leverage": account_leverage_limit,
                    "excess_leverage": excess_leverage,
                    "margin_call_risk": "high",
                    "warning": f"Requested leverage {effective_leverage:.1f}x exceeds account limit {account_leverage_limit:.1f}x"
                }
            
            return {
                "position_value": position_value,
                "instrument_type": instrument_type,
                "initial_margin": initial_margin,
                "maintenance_margin": maintenance_margin,
                "effective_leverage": effective_leverage,
                "margin_call_risk": "low" if effective_leverage <= 2.0 else "medium",
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Margin requirement calculation error: {str(e)}")
            return {"error": f"Margin calculation failed: {str(e)}"}
    
    async def validate_margin_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate order against margin requirements and limits.
        
        Args:
            order_data: Order information to validate
            
        Returns:
            Validation result with margin checks
        """
        try:
            # Check if margin trading is enabled
            if not self.margin_trading_enabled:
                return {
                    "approved": False,
                    "reason": "Margin trading is disabled",
                    "margin_check": "skipped"
                }
            
            # Calculate margin requirement
            requirement = await self.calculate_margin_requirement(
                symbol=order_data.get("symbol"),
                quantity=order_data.get("quantity", 0),
                side=order_data.get("side"),
                price=order_data.get("limit_price", 0) or order_data.get("estimated_price", 0)
            )
            
            if "error" in requirement:
                return {
                    "approved": False,
                    "reason": requirement["error"],
                    "margin_check": "failed"
                }
            
            # Get current margin status
            status = await self.calculate_margin_status()
            if "error" in status:
                return {
                    "approved": False,
                    "reason": status["error"],
                    "margin_check": "failed"
                }
            
            # Validate against available margin
            margin_available = status["margin_available"]
            margin_required = requirement["initial_margin"]
            
            validation_result = {
                "approved": True,
                "margin_check": "passed",
                "requirement": requirement,
                "current_status": status,
                "warnings": []
            }
            
            if margin_required > margin_available:
                validation_result.update({
                    "approved": False,
                    "reason": f"Insufficient margin: Need ${margin_required:.2f}, have ${margin_available:.2f}",
                    "shortfall": margin_required - margin_available,
                    "margin_check": "insufficient_margin"
                })
            
            # Check leverage limits
            effective_leverage = requirement["effective_leverage"]
            if effective_leverage > self.maximum_leverage.value:
                validation_result.update({
                    "approved": False,
                    "reason": f"Leverage limit exceeded: {effective_leverage:.1f}x > {self.maximum_leverage.value:.1f}x",
                    "margin_check": "leverage_exceeded"
                })
            
            # Check margin call risk
            if requirement.get("margin_call_risk") == "high":
                validation_result["warnings"].append(
                    "High margin call risk - consider reducing position size"
                )
            
            # Check if order would trigger margin call
            projected_status = await self._project_margin_status_after_order(status, requirement)
            if projected_status["margin_health_score"] < 50:
                validation_result["warnings"].append(
                    "Order would significantly impact margin health"
                )
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Margin order validation error: {str(e)}")
            return {
                "approved": False,
                "reason": f"Validation error: {str(e)}",
                "margin_check": "error"
            }
    
    async def calculate_margin_status(self) -> Dict[str, Any]:
        """
        Calculate current margin status and health metrics.
        
        Returns:
            Comprehensive margin status report
        """
        try:
            # Get account information
            account = self.db.query(Account).filter(
                Account.user_id == self.user_id
            ).first()
            
            if not account:
                return {"error": "Account not found"}
            
            # Get all margin positions
            positions = self.db.query(Position).filter(
                and_(
                    Position.user_id == self.user_id,
                    Position.quantity != 0
                )
            ).all()
            
            # Calculate current margin usage
            total_exposure = sum(abs(pos.market_value or 0) for pos in positions)
            total_unrealized_pnl = sum(pos.unrealized_pnl or 0 for pos in positions)
            
            # Account values
            account_value = account.cash_balance + total_unrealized_pnl
            margin_used = total_exposure - account.cash_balance if total_exposure > account.cash_balance else 0
            
            # Calculate leverage
            leverage_ratio = total_exposure / account_value if account_value > 0 else 1.0
            
            # Calculate maintenance requirements
            maintenance_margin_required = await self._calculate_total_maintenance_margin()
            
            # Calculate margin call and liquidation levels
            margin_call_level = maintenance_margin_required * 1.4  # 140% of maintenance
            liquidation_level = maintenance_margin_required * 1.25  # 125% of maintenance
            
            # Calculate available margin
            margin_available = account_value - margin_used if margin_used < account_value else 0
            
            # Calculate margin health score (0-100)
            margin_health_score = self._calculate_margin_health_score(
                account_value, margin_used, maintenance_margin_required, leverage_ratio
            )
            
            # Estimate days to margin call (simplified)
            days_to_margin_call = await self._estimate_days_to_margin_call(positions)
            
            status = MarginStatus(
                account_value=account_value,
                margin_used=margin_used,
                margin_available=margin_available,
                maintenance_margin_required=maintenance_margin_required,
                leverage_ratio=leverage_ratio,
                margin_call_level=margin_call_level,
                liquidation_level=liquidation_level,
                margin_health_score=margin_health_score,
                days_to_margin_call=days_to_margin_call
            )
            
            self.current_status = status
            
            return {
                "account_value": status.account_value,
                "margin_used": status.margin_used,
                "margin_available": status.margin_available,
                "maintenance_margin_required": status.maintenance_margin_required,
                "leverage_ratio": status.leverage_ratio,
                "margin_call_level": status.margin_call_level,
                "liquidation_level": status.liquidation_level,
                "margin_health_score": status.margin_health_score,
                "margin_utilization": (status.margin_used / status.account_value * 100) if status.account_value > 0 else 0,
                "days_to_margin_call": status.days_to_margin_call,
                "timestamp": status.timestamp
            }
            
        except Exception as e:
            logger.error(f"Margin status calculation error: {str(e)}")
            return {"error": f"Status calculation failed: {str(e)}"}
    
    async def monitor_margin_alerts(self) -> List[Dict[str, Any]]:
        """
        Monitor margin levels and generate alerts.
        
        Returns:
            List of margin alerts
        """
        alerts = []
        
        try:
            # Get current status
            status = await self.calculate_margin_status()
            if "error" in status:
                return [{"error": status["error"]}]
            
            margin_used = status["margin_used"]
            account_value = status["account_value"]
            maintenance_required = status["maintenance_margin_required"]
            leverage_ratio = status["leverage_ratio"]
            
            # Margin call warning
            if margin_used >= status["margin_call_level"] * self.alert_thresholds["margin_call_warning"]:
                alert = MarginAlert(
                    alert_type="margin_call_warning",
                    severity="critical",
                    current_value=margin_used,
                    threshold=status["margin_call_level"],
                    message=f"Margin usage approaching call level: ${margin_used:.2f}/${status['margin_call_level']:.2f}",
                    recommended_action="Reduce positions or add collateral immediately"
                )
                alerts.append(self._alert_to_dict(alert))
            
            # Maintenance margin warning
            if margin_used >= maintenance_required * self.alert_thresholds["maintenance_warning"]:
                alert = MarginAlert(
                    alert_type="maintenance_warning",
                    severity="warning",
                    current_value=margin_used,
                    threshold=maintenance_required,
                    message=f"Margin usage approaching maintenance level: ${margin_used:.2f}/${maintenance_required:.2f}",
                    recommended_action="Monitor positions closely"
                )
                alerts.append(self._alert_to_dict(alert))
            
            # Leverage warning
            max_leverage = self.maximum_leverage.value
            if leverage_ratio >= max_leverage * self.alert_thresholds["leverage_warning"]:
                alert = MarginAlert(
                    alert_type="leverage_warning",
                    severity="warning",
                    current_value=leverage_ratio,
                    threshold=max_leverage,
                    message=f"Leverage approaching limit: {leverage_ratio:.1f}x/{max_leverage:.1f}x",
                    recommended_action="Consider reducing position sizes"
                )
                alerts.append(self._alert_to_dict(alert))
            
            # Health score warning
            if status["margin_health_score"] < 60:
                severity = "critical" if status["margin_health_score"] < 40 else "warning"
                alert = MarginAlert(
                    alert_type="margin_health",
                    severity=severity,
                    current_value=status["margin_health_score"],
                    threshold=60,
                    message=f"Margin health score is low: {status['margin_health_score']:.1f}/100",
                    recommended_action="Review and optimize positions"
                )
                alerts.append(self._alert_to_dict(alert))
            
            self.margin_alerts.extend(alerts)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Margin alert monitoring error: {str(e)}")
            return [{"error": str(e)}]
    
    async def handle_margin_call(self, call_type: str = "warning") -> Dict[str, Any]:
        """
        Handle margin call situations with appropriate actions.
        
        Args:
            call_type: Type of margin call (warning, call, liquidation)
            
        Returns:
            Margin call handling result
        """
        try:
            status = await self.calculate_margin_status()
            if "error" in status:
                return {"error": status["error"]}
            
            actions_taken = []
            severity = "info"
            
            if call_type == "warning":
                # Send warning, no automatic action
                actions_taken.append("warning_sent")
                severity = "info"
                
            elif call_type == "call":
                # Reduce positions automatically
                reduction_result = await self._automatic_position_reduction(0.3)  # Reduce by 30%
                actions_taken.append(("position_reduction", reduction_result))
                severity = "warning"
                
            elif call_type == "liquidation":
                # Aggressive liquidation
                liquidation_result = await self._automatic_position_reduction(0.7)  # Reduce by 70%
                actions_taken.append(("position_liquidation", liquidation_result))
                severity = "critical"
                
                # Also halt new trading
                self.margin_trading_enabled = False
                actions_taken.append("margin_trading_halted")
            
            # Log the action
            logger.warning(f"Margin call handled for user {self.user_id}: {call_type} - Actions: {actions_taken}")
            
            return {
                "call_type": call_type,
                "severity": severity,
                "actions_taken": actions_taken,
                "status_after_action": await self.calculate_margin_status(),
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Margin call handling error: {str(e)}")
            return {"error": f"Margin call handling failed: {str(e)}"}
    
    async def get_margin_compliance_report(self) -> Dict[str, Any]:
        """
        Generate margin compliance report for regulatory requirements.
        
        Returns:
            Comprehensive margin compliance report
        """
        try:
            status = await self.calculate_margin_status()
            if "error" in status:
                return {"error": status["error"]}
            
            # Get all margin positions for compliance check
            positions = self.db.query(Position).filter(
                and_(
                    Position.user_id == self.user_id,
                    Position.quantity != 0
                )
            ).all()
            
            compliance_checks = []
            
            # Pattern Day Trader compliance
            if await self._is_pattern_day_trader():
                compliance_checks.append({
                    "check": "Pattern Day Trader",
                    "status": "compliant" if status["account_value"] >= 25000 else "non_compliant",
                    "requirement": "Minimum $25,000 for PDT accounts",
                    "current_value": status["account_value"]
                })
            
            # Maintenance margin compliance
            compliance_checks.append({
                "check": "Maintenance Margin",
                "status": "compliant" if status["margin_used"] <= status["maintenance_margin_required"] else "non_compliant",
                "requirement": f"${status['maintenance_margin_required']:.2f}",
                "current_value": status["margin_used"]
            })
            
            # Leverage limit compliance
            max_leverage = self.maximum_leverage.value
            compliance_checks.append({
                "check": "Leverage Limit",
                "status": "compliant" if status["leverage_ratio"] <= max_leverage else "non_compliant",
                "requirement": f"Maximum {max_leverage:.1f}x leverage",
                "current_value": status["leverage_ratio"]
            })
            
            # Calculate overall compliance
            compliant_checks = sum(1 for check in compliance_checks if check["status"] == "compliant")
            compliance_percentage = (compliant_checks / len(compliance_checks)) * 100
            
            return {
                "overall_compliance": "compliant" if compliance_percentage >= 100 else "non_compliant",
                "compliance_percentage": compliance_percentage,
                "compliance_checks": compliance_checks,
                "account_type": self.account_type.value,
                "leverage_limit": max_leverage,
                "margin_health_score": status["margin_health_score"],
                "risk_level": self._calculate_risk_level(status),
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Margin compliance report error: {str(e)}")
            return {"error": f"Compliance report failed: {str(e)}"}
    
    def set_leverage_limit(self, leverage: LeverageLevel):
        """Set maximum leverage limit."""
        self.maximum_leverage = leverage
        logger.info(f"Leverage limit set to {leverage.value}x for user {self.user_id}")
    
    def _classify_instrument_type(self, symbol: str) -> str:
        """Classify instrument type for margin requirement lookup."""
        symbol_upper = symbol.upper()
        
        # Simple classification logic (would be more sophisticated in practice)
        if symbol_upper.endswith(("USD", "USDT")) or "BTC" in symbol_upper or "ETH" in symbol_upper:
            return "crypto"
        elif any(curr in symbol_upper for curr in ["EUR", "USD", "JPY", "GBP"]):
            return "forex"
        elif symbol_upper.startswith("ES") or symbol_upper.startswith("NQ"):
            return "future"
        elif symbol_upper.endswith(("P", "C")):  # Options
            return "option"
        elif symbol_upper.startswith(("SPY", "QQQ", "IWM")):  # Popular ETFs
            return "etf"
        else:
            return "equity"  # Default to equity
    
    async def _calculate_total_maintenance_margin(self) -> float:
        """Calculate total maintenance margin requirement."""
        try:
            positions = self.db.query(Position).filter(
                and_(
                    Position.user_id == self.user_id,
                    Position.quantity != 0
                )
            ).all()
            
            total_maintenance_margin = 0.0
            
            for position in positions:
                instrument_type = self._classify_instrument_type(position.symbol)
                requirement = self.margin_requirements.get(instrument_type, self.margin_requirements["equity"])
                
                position_value = abs(position.market_value or 0)
                maintenance_margin = position_value * requirement.maintenance_margin_rate
                total_maintenance_margin += maintenance_margin
            
            return total_maintenance_margin
            
        except Exception as e:
            logger.error(f"Maintenance margin calculation error: {str(e)}")
            return 0.0
    
    def _calculate_margin_health_score(self, account_value: float, margin_used: float,
                                     maintenance_required: float, leverage_ratio: float) -> float:
        """Calculate margin health score (0-100)."""
        try:
            if account_value <= 0:
                return 0.0
            
            score = 100.0
            
            # Deduct for high leverage
            if leverage_ratio > 3.0:
                score -= 30
            elif leverage_ratio > 2.0:
                score -= 15
            
            # Deduct for margin utilization
            utilization = margin_used / account_value
            if utilization > 0.8:
                score -= 40
            elif utilization > 0.6:
                score -= 20
            elif utilization > 0.4:
                score -= 10
            
            # Deduct for maintenance margin ratio
            if margin_used > maintenance_required:
                score -= 50
            
            return max(0.0, min(100.0, score))
            
        except Exception as e:
            logger.error(f"Health score calculation error: {str(e)}")
            return 50.0
    
    async def _estimate_days_to_margin_call(self, positions: List[Position]) -> Optional[int]:
        """Estimate days until potential margin call based on position drift."""
        # Simplified estimation - would use more sophisticated modeling
        try:
            if not positions:
                return None
            
            # Calculate daily P&L volatility
            total_unrealized_pnl = sum(pos.unrealized_pnl or 0 for pos in positions)
            if abs(total_unrealized_pnl) < 100:  # Very low exposure
                return None
            
            # Simple heuristic: assume 2% daily volatility
            daily_volatility = abs(total_unrealized_pnl) * 0.02
            status = self.current_status
            
            if not status or daily_volatility <= 0:
                return None
            
            # Estimate based on distance to margin call
            distance_to_call = status.margin_call_level - status.margin_used
            if distance_to_call <= 0:
                return 0
            
            return max(1, int(distance_to_call / daily_volatility))
            
        except Exception as e:
            logger.error(f"Days to margin call estimation error: {str(e)}")
            return None
    
    def _alert_to_dict(self, alert: MarginAlert) -> Dict[str, Any]:
        """Convert MarginAlert to dictionary."""
        return {
            "alert_type": alert.alert_type,
            "severity": alert.severity,
            "current_value": alert.current_value,
            "threshold": alert.threshold,
            "message": alert.message,
            "recommended_action": alert.recommended_action,
            "timestamp": alert.timestamp
        }
    
    async def _project_margin_status_after_order(self, current_status: Dict, 
                                               margin_requirement: Dict) -> Dict[str, Any]:
        """Project margin status after executing proposed order."""
        try:
            # Simplified projection
            additional_margin = margin_requirement["initial_margin"]
            
            projected_margin_used = current_status["margin_used"] + additional_margin
            projected_account_value = current_status["account_value"]
            
            projected_leverage = (current_status["margin_used"] + additional_margin) / projected_account_value
            projected_health_score = self._calculate_margin_health_score(
                projected_account_value,
                projected_margin_used,
                current_status["maintenance_margin_required"],
                projected_leverage
            )
            
            return {
                "margin_health_score": projected_health_score,
                "projected_leverage": projected_leverage,
                "projected_margin_used": projected_margin_used
            }
            
        except Exception as e:
            logger.error(f"Status projection error: {str(e)}")
            return {"margin_health_score": current_status.get("margin_health_score", 50)}
    
    async def _check_regulatory_compliance(self, account_type: MarginAccountType) -> Dict[str, Any]:
        """Check regulatory compliance for margin trading."""
        try:
            account = self.db.query(Account).filter(
                Account.user_id == self.user_id
            ).first()
            
            compliance_checks = {
                "compliant": True,
                "checks": []
            }
            
            # Minimum account value for margin
            if account.cash_balance < 2000:
                compliance_checks["compliant"] = False
                compliance_checks["checks"].append({
                    "check": "Minimum Account Value",
                    "status": "failed",
                    "requirement": "$2,000 minimum",
                    "current": account.cash_balance
                })
            else:
                compliance_checks["checks"].append({
                    "check": "Minimum Account Value",
                    "status": "passed",
                    "requirement": "$2,000 minimum",
                    "current": account.cash_balance
                })
            
            # Pattern Day Trader requirement
            if account_type in [MarginAccountType.PORTFOLIO_MARGIN, MarginAccountType.REGULAR]:
                if account.cash_balance < 25000:
                    compliance_checks["checks"].append({
                        "check": "PDT Requirement",
                        "status": "warning",
                        "requirement": "$25,000 for PDT accounts",
                        "current": account.cash_balance
                    })
                else:
                    compliance_checks["checks"].append({
                        "check": "PDT Requirement",
                        "status": "passed",
                        "requirement": "$25,000 for PDT accounts",
                        "current": account.cash_balance
                    })
            
            return compliance_checks
            
        except Exception as e:
            logger.error(f"Regulatory compliance check error: {str(e)}")
            return {"compliant": False, "error": str(e)}
    
    async def _is_pattern_day_trader(self) -> bool:
        """Check if account qualifies as Pattern Day Trader."""
        # Simplified check - would count day trades in practice
        try:
            # Count day trades in last 5 business days
            five_days_ago = datetime.now() - timedelta(days=5)
            day_trades = self.db.query(Trade).filter(
                and_(
                    Trade.user_id == self.user_id,
                    Trade.executed_at >= five_days_ago
                )
            ).count()
            
            return day_trades >= 3  # PDT definition: 3+ day trades in 5 days
            
        except Exception:
            return False
    
    def _calculate_risk_level(self, status: Dict) -> str:
        """Calculate overall margin risk level."""
        try:
            health_score = status["margin_health_score"]
            leverage_ratio = status["leverage_ratio"]
            
            if health_score < 40 or leverage_ratio > 3.5:
                return "high"
            elif health_score < 60 or leverage_ratio > 2.5:
                return "medium"
            else:
                return "low"
                
        except Exception:
            return "medium"
    
    async def _liquidate_all_margin_positions(self) -> Dict[str, Any]:
        """Liquidate all margin positions."""
        # Simplified implementation
        return {
            "positions_liquidated": 0,
            "reason": "Manual margin account closure",
            "timestamp": datetime.now()
        }
    
    async def _automatic_position_reduction(self, reduction_percentage: float) -> Dict[str, Any]:
        """Automatically reduce positions to meet margin requirements."""
        # Simplified implementation
        return {
            "reduction_percentage": reduction_percentage,
            "positions_affected": 0,
            "margin_freed": 0.0,
            "timestamp": datetime.now()
        }
    
    def close(self):
        """Cleanup resources."""
        self.db.close()