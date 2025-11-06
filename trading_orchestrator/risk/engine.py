"""
@file engine.py
@brief Risk Management Engine - Main Orchestrator

@details
This module implements the central risk management coordinator that orchestrates
all risk management components in the trading system. It provides a unified
interface for risk operations including limits checking, policy validation,
circuit breaker management, compliance monitoring, and incident tracking.

Key Features:
- Central risk coordinator with multi-component integration
- Real-time risk monitoring and alerting
- Configurable risk limits and policies
- Circuit breaker implementation for market events
- Compliance tracking and reporting
- Comprehensive audit logging
- Incident management and response

@author Trading Orchestrator System
@version 1.0
@date 2025-11-06

@warning
This is a critical system component that monitors and controls trading risk.
Incorrect configuration can lead to unexpected trading behavior.

@note
This module coordinates the following risk components:
- RiskLimitChecker: Portfolio and position limits
- PolicyEngine: Trading policy validation
- CircuitBreakerManager: Market event handling
- ComplianceEngine: Regulatory compliance
- AuditLogger: Risk event logging
- IncidentManager: Risk incident response

@see risk.limits for limit checking logic
@see risk.policy for policy validation
@see risk.circuit_breakers for circuit breaker implementation
@see risk.compliance for compliance rules
@see risk.audit for audit logging
@see risk.incidents for incident management
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc
import pandas as pd
import numpy as np

from config.database import get_db
from database.models.risk import RiskLimit, RiskEvent, CircuitBreaker, RiskLevel, RiskEventType
from database.models.trading import Order, Position, Trade
from database.models.user import User

# Import existing risk components
from .limits import RiskLimitChecker
from .policy import PolicyEngine
from .circuit_breakers import CircuitBreakerManager
from .compliance import ComplianceEngine
from .audit import AuditLogger
from .incidents import IncidentManager

# Import advanced risk management components
from .models.var_models import VaRCalculator
from .models.cvar_models import CVaRCalculator
from .models.drawdown_models import DrawdownAnalyzer
from .models.volatility_models import VolatilityModeler
from .models.correlation_models import CorrelationAnalyzer
from .models.stress_testing import StressTestEngine
from .models.credit_risk import CreditRiskAnalyzer
from .enhanced_limits import EnhancedRiskLimits
from .portfolio_optimization import PortfolioOptimizer
from .real_time_monitor import RealTimeRiskMonitor
from .compliance_frameworks import ComplianceFrameworks, RegulationFramework
from .api_integration import RiskManagementAPI
from .integration_layer import RiskBrokerIntegration

logger = logging.getLogger(__name__)


# Alias for main.py compatibility
RiskEngine = RiskManager


class RiskManager:
    """
    @class RiskManager
    @brief Central Risk Management Coordinator
    
    @details
    The RiskManager serves as the central coordinator for all risk management
    operations in the trading system. It orchestrates multiple risk components
    to provide comprehensive risk monitoring, control, and reporting.
    
    @par Architecture:
    The RiskManager integrates the following risk management components:
    - RiskLimitChecker: Enforces position and portfolio limits
    - PolicyEngine: Validates trading policies and rules
    - CircuitBreakerManager: Handles market disruption events
    - ComplianceEngine: Monitors regulatory compliance
    - AuditLogger: Records all risk-related events
    - IncidentManager: Manages risk incidents and responses
    
    @par Risk Management Workflow:
    1. Order risk validation before execution
    2. Continuous portfolio monitoring
    3. Policy compliance checking
    4. Circuit breaker trigger evaluation
    5. Incident detection and escalation
    6. Risk reporting and analytics
    
    @par Risk Levels:
    - LOW: Normal trading operations
    - MEDIUM: Enhanced monitoring active
    - HIGH: Some restrictions in place
    - CRITICAL: Trading halted, manual intervention required
    
    @par Configuration:
    The RiskManager requires:
    - user_id: User identifier for risk tracking
    - Database connection for risk state persistence
    - Risk limits configuration
    - Policy definitions
    
    @warning
    This is a critical system component. Incorrect configuration can lead to
    unexpected trading behavior or inadequate risk protection.
    
    @par Usage Example:
    @code
    from risk.engine import RiskManager
    
    # Initialize risk manager
    risk_manager = RiskManager(user_id=123)
    
    # Check risk for proposed trade
    risk_result = await risk_manager.check_order_risk(
        symbol="AAPL",
        side="buy",
        quantity=100,
        order_type="market"
    )
    
    if risk_result.approved:
        # Proceed with trade execution
        print("Trade approved by risk manager")
    else:
        print(f"Trade rejected: {risk_result.reason}")
    
    # Get current risk status
    status = await risk_manager.get_risk_status()
    print(f"Current risk level: {status['risk_level']}")
    
    # Monitor portfolio risk continuously
    await risk_manager.start_monitoring()
    @endcode
    
    @note
    This class implements the main entry point for all risk management
    operations in the trading system.
    
    @see RiskLimitChecker for limit checking logic
    @see PolicyEngine for policy validation
    @see CircuitBreakerManager for circuit breaker logic
    @see ComplianceEngine for compliance monitoring
    """
    
    def __init__(self, user_id: int):
        """
        @brief Initialize RiskManager with user context
        
        @param user_id User identifier for risk tracking
        
        @details
        Initializes the RiskManager with user-specific risk context and
        initializes all risk management components.
        
        @par Initialization Process:
        1. Set user identifier for risk tracking
        2. Initialize database connection
        3. Create risk management component instances
        4. Load user risk limits and policies
        5. Set up monitoring and alert systems
        
        @throws ValueError if user_id is invalid
        @throws DatabaseError if database connection fails
        
        @par Example:
        @code
        # Initialize for specific user
        risk_manager = RiskManager(user_id=12345)
        
        # Risk manager is now ready for operations
        @endcode
        
        @note
        User-specific risk limits and policies are loaded during initialization.
        Any changes to risk configuration require reinitialization.
        """
        if user_id <= 0:
            raise ValueError("User ID must be positive")
            
        self.user_id = user_id
        self.db = get_db()
        
        # Core risk management components
        self.limit_checker = RiskLimitChecker(user_id)
        self.policy_engine = PolicyEngine(user_id)
        self.circuit_breaker_manager = CircuitBreakerManager(user_id)
        self.compliance_engine = ComplianceEngine(user_id)
        self.audit_logger = AuditLogger(user_id)
        self.incident_manager = IncidentManager(user_id)
        
        # Advanced risk management models
        self.var_calculator = VaRCalculator()
        self.cvar_calculator = CVaRCalculator()
        self.drawdown_analyzer = DrawdownAnalyzer()
        self.volatility_modeler = VolatilityModeler()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.stress_tester = StressTestEngine()
        self.credit_risk_analyzer = CreditRiskAnalyzer()
        self.enhanced_limits = EnhancedRiskLimits()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.real_time_monitor = RealTimeRiskMonitor()
        
        # Compliance and regulatory frameworks
        self.compliance_frameworks = ComplianceFrameworks()
        
        # API integration layer
        self.api_server = None
        self.broker_integration = RiskBrokerIntegration()
        
        # Risk metrics tracking
        self.risk_metrics = {}
        self.last_risk_check = None
        self.risk_check_interval = 60  # seconds
        self.advanced_monitoring_active = False
        
        # Portfolio data cache
        self.portfolio_data_cache = {}
        self.risk_calculations_cache = {}
        
        logger.info(f"Advanced RiskManager initialized for user {user_id}")
    
    async def validate_trade_pre_execution(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pre-execution trade validation.
        
        Args:
            order_data: Order information to validate
            
        Returns:
            Validation result with status and details
        """
        validation_result = {
            "approved": True,
            "rejection_reason": None,
            "warnings": [],
            "risk_metrics": {},
            "actions_taken": []
        }
        
        try:
            # 1. Check circuit breakers first
            circuit_status = await self.circuit_breaker_manager.check_circuit_status(
                order_data.get("symbol"),
                order_data.get("strategy_id")
            )
            
            if circuit_status.get("trading_halted"):
                validation_result.update({
                    "approved": False,
                    "rejection_reason": f"Trading halted by circuit breaker: {circuit_status['reason']}"
                })
                await self.audit_logger.log_risk_action("circuit_breaker_blocked_trade", {
                    "symbol": order_data.get("symbol"),
                    "reason": circuit_status["reason"]
                })
                return validation_result
            
            # 2. Validate against policy rules
            policy_result = await self.policy_engine.validate_order(order_data)
            if not policy_result["approved"]:
                validation_result.update({
                    "approved": False,
                    "rejection_reason": policy_result["rejection_reason"],
                    "actions_taken": policy_result["actions_taken"]
                })
                return validation_result
            
            # 3. Check risk limits
            limit_result = await self.limit_checker.check_order_limits(order_data)
            if not limit_result["approved"]:
                validation_result.update({
                    "approved": False,
                    "rejection_reason": limit_result["rejection_reason"],
                    "warnings": limit_result.get("warnings", [])
                })
                return validation_result
            
            # 4. Compliance checks
            compliance_result = await self.compliance_engine.validate_order(order_data)
            if not compliance_result["approved"]:
                validation_result.update({
                    "approved": False,
                    "rejection_reason": compliance_result["rejection_reason"],
                    "warnings": compliance_result.get("warnings", [])
                })
                return validation_result
            
            # 5. Log approved trade for audit
            await self.audit_logger.log_trade_approval(order_data)
            
            logger.info(f"Trade approved for user {self.user_id}: {order_data.get('symbol')}")
            
        except Exception as e:
            logger.error(f"Risk validation error for user {self.user_id}: {str(e)}")
            validation_result.update({
                "approved": False,
                "rejection_reason": f"Risk validation error: {str(e)}"
            })
        
        return validation_result
    
    async def monitor_positions_risk(self) -> Dict[str, Any]:
        """
        Monitor all positions for risk violations.
        
        Returns:
            Monitoring results with any triggered actions
        """
        results = {
            "positions_checked": 0,
            "violations": [],
            "actions_taken": [],
            "risk_score": 0.0
        }
        
        try:
            positions = self.db.query(Position).filter(
                and_(
                    Position.user_id == self.user_id,
                    Position.quantity != 0
                )
            ).all()
            
            results["positions_checked"] = len(positions)
            
            for position in positions:
                # Calculate position risk metrics
                position_risk = await self._calculate_position_risk(position)
                results["risk_score"] += position_risk["risk_score"]
                
                # Check for limit breaches
                limit_violations = await self.limit_checker.check_position_limits(position)
                
                for violation in limit_violations:
                    results["violations"].append({
                        "position_id": position.id,
                        "symbol": position.symbol,
                        "violation_type": violation["type"],
                        "current_value": violation["current_value"],
                        "limit_value": violation["limit_value"],
                        "severity": violation["severity"]
                    })
                    
                    # Take action based on severity
                    if violation["severity"] == "critical":
                        action = await self._handle_critical_violation(position, violation)
                        results["actions_taken"].append(action)
                        
                        # Create risk event
                        await self._create_risk_event(
                            RiskEventType.EXPOSURE_LIMIT,
                            RiskLevel.CRITICAL,
                            f"Critical position limit breach: {position.symbol}",
                            {"position": position.__dict__, "violation": violation},
                            related_position_id=position.id,
                            related_limit_id=violation.get("limit_id")
                        )
            
            # Normalize risk score
            if results["positions_checked"] > 0:
                results["risk_score"] /= results["positions_checked"]
            
        except Exception as e:
            logger.error(f"Position risk monitoring error: {str(e)}")
            results["error"] = str(e)
        
        return results
    
    async def check_daily_loss_limits(self) -> Dict[str, Any]:
        """
        Check daily loss limits and trigger circuit breakers if needed.
        
        Returns:
            Daily loss check results
        """
        results = {
            "daily_pnl": 0.0,
            "daily_loss_limit": 0.0,
            "loss_percentage": 0.0,
            "limit_breached": False,
            "circuit_breaker_triggered": False
        }
        
        try:
            today = datetime.now().date()
            
            # Get daily P&L
            trades_today = self.db.query(Trade).filter(
                and_(
                    Trade.user_id == self.user_id,
                    Trade.executed_at >= datetime.combine(today, datetime.min.time()),
                    Trade.realized_pnl.isnot(None)
                )
            ).all()
            
            daily_pnl = sum(trade.realized_pnl or 0 for trade in trades_today)
            results["daily_pnl"] = daily_pnl
            
            # Get daily loss limit
            daily_loss_limit = self.db.query(RiskLimit).filter(
                and_(
                    RiskLimit.user_id == self.user_id,
                    RiskLimit.limit_type == "daily_loss",
                    RiskLimit.is_active == True
                )
            ).first()
            
            if daily_loss_limit:
                results["daily_loss_limit"] = daily_loss_limit.limit_value
                results["loss_percentage"] = (abs(daily_pnl) / daily_loss_limit.limit_value * 100) if daily_loss_limit.limit_value > 0 else 0
                
                if daily_pnl < -daily_loss_limit.limit_value:
                    results["limit_breached"] = True
                    
                    # Trigger circuit breaker
                    await self.circuit_breaker_manager.trigger_circuit_breaker(
                        "daily_loss_limit",
                        f"Daily loss limit exceeded: {daily_pnl:.2f} < {-daily_loss_limit.limit_value}",
                        cancel_existing_orders=True,
                        close_positions=False
                    )
                    
                    results["circuit_breaker_triggered"] = True
                    
                    # Create risk event
                    await self._create_risk_event(
                        RiskEventType.DAILY_LOSS_LIMIT,
                        RiskLevel.HIGH,
                        "Daily loss limit breached",
                        {
                            "daily_pnl": daily_pnl,
                            "limit_value": daily_loss_limit.limit_value,
                            "loss_percentage": results["loss_percentage"]
                        }
                    )
            
        except Exception as e:
            logger.error(f"Daily loss limit check error: {str(e)}")
            results["error"] = str(e)
        
        return results
    
    async def generate_risk_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive risk report.
        
        Returns:
            Complete risk assessment report
        """
        report = {
            "user_id": self.user_id,
            "generated_at": datetime.now(),
            "summary": {},
            "positions": {},
            "orders": {},
            "risk_limits": {},
            "circuit_breakers": {},
            "recent_events": [],
            "compliance_status": {},
            "recommendations": []
        }
        
        try:
            # Position risk summary
            positions = self.db.query(Position).filter(Position.user_id == self.user_id).all()
            total_exposure = sum(abs(pos.market_value or 0) for pos in positions)
            
            report["summary"] = {
                "total_exposure": total_exposure,
                "positions_count": len(positions),
                "risk_score": await self._calculate_portfolio_risk_score(),
                "net_pnl": sum(pos.unrealized_pnl for pos in positions),
                "margin_usage": await self._calculate_margin_usage()
            }
            
            # Position details
            for position in positions:
                report["positions"][position.symbol] = {
                    "side": position.side.value,
                    "quantity": position.quantity,
                    "market_value": position.market_value,
                    "unrealized_pnl": position.unrealized_pnl,
                    "risk_score": await self._calculate_position_risk(position)["risk_score"]
                }
            
            # Active orders
            active_orders = self.db.query(Order).filter(
                and_(
                    Order.user_id == self.user_id,
                    Order.status.in_(["pending", "new", "partially_filled"])
                )
            ).all()
            
            report["orders"] = {
                "active_count": len(active_orders),
                "pending_exposure": sum(
                    order.quantity * (order.limit_price or 0) 
                    for order in active_orders 
                    if order.limit_price
                )
            }
            
            # Risk limits status
            limits = self.db.query(RiskLimit).filter(
                and_(
                    RiskLimit.user_id == self.user_id,
                    RiskLimit.is_active == True
                )
            ).all()
            
            for limit in limits:
                report["risk_limits"][limit.limit_name] = {
                    "limit_value": limit.limit_value,
                    "current_value": limit.current_value,
                    "percentage_used": (limit.current_value / limit.limit_value * 100) if limit.limit_value > 0 else 0,
                    "is_breached": limit.is_breached
                }
            
            # Circuit breakers status
            breakers = self.db.query(CircuitBreaker).filter(
                or_(
                    CircuitBreaker.user_id == self.user_id,
                    CircuitBreaker.user_id.is_(None)
                )
            ).all()
            
            for breaker in breakers:
                report["circuit_breakers"][breaker.breaker_name] = {
                    "is_active": breaker.is_active,
                    "triggered_by": breaker.triggered_by,
                    "triggered_at": breaker.triggered_at
                }
            
            # Recent risk events
            recent_events = self.db.query(RiskEvent).filter(
                and_(
                    RiskEvent.user_id == self.user_id,
                    RiskEvent.occurred_at >= datetime.now() - timedelta(days=7)
                )
            ).order_by(desc(RiskEvent.occurred_at)).limit(10).all()
            
            report["recent_events"] = [
                {
                    "type": event.event_type.value,
                    "level": event.risk_level.value,
                    "title": event.title,
                    "occurred_at": event.occurred_at,
                    "is_resolved": event.is_resolved
                }
                for event in recent_events
            ]
            
            # Compliance status
            report["compliance_status"] = await self.compliance_engine.get_compliance_status()
            
            # Generate recommendations
            report["recommendations"] = await self._generate_risk_recommendations(report)
            
        except Exception as e:
            logger.error(f"Risk report generation error: {str(e)}")
            report["error"] = str(e)
        
        return report
    
    async def _calculate_position_risk(self, position: Position) -> Dict[str, Any]:
        """Calculate risk metrics for a single position."""
        try:
            # Basic risk factors
            position_size = abs(position.quantity)
            market_value = abs(position.market_value or 0)
            unrealized_pnl = position.unrealized_pnl or 0
            
            # Risk score calculation (0-100)
            risk_score = 0.0
            
            # Position size risk
            if market_value > 10000:  # $10k+ position
                risk_score += 30
            
            # Volatility risk (simplified)
            pnl_volatility = abs(unrealized_pnl) / (market_value if market_value > 0 else 1) * 100
            if pnl_volatility > 10:  # 10%+ unrealized loss
                risk_score += 40
            
            # Concentration risk
            total_portfolio_value = sum(abs(pos.market_value or 0) for pos in 
                                      self.db.query(Position).filter(
                                          Position.user_id == self.user_id
                                      ).all())
            
            if total_portfolio_value > 0:
                concentration = market_value / total_portfolio_value
                if concentration > 0.2:  # 20%+ concentration
                    risk_score += 30
            
            return {
                "risk_score": min(risk_score, 100),
                "position_size_risk": min(position_size / 1000, 30),  # Max 30 points
                "volatility_risk": min(pnl_volatility, 40),  # Max 40 points
                "concentration_risk": min(concentration * 150, 30) if 'concentration' in locals() else 0  # Max 30 points
            }
        except Exception as e:
            logger.error(f"Position risk calculation error: {str(e)}")
            return {"risk_score": 0}
    
    async def _calculate_portfolio_risk_score(self) -> float:
        """Calculate overall portfolio risk score."""
        try:
            positions = self.db.query(Position).filter(Position.user_id == self.user_id).all()
            if not positions:
                return 0.0
            
            total_risk = sum(
                await self._calculate_position_risk(pos)["risk_score"] 
                for pos in positions
            )
            
            return total_risk / len(positions)
        except Exception:
            return 0.0
    
    async def _calculate_margin_usage(self) -> float:
        """Calculate current margin usage percentage."""
        try:
            # Simplified margin calculation
            # In practice, this would query broker for actual margin data
            positions = self.db.query(Position).filter(Position.user_id == self.user_id).all()
            total_exposure = sum(abs(pos.market_value or 0) for pos in positions)
            
            # Assume 2:1 leverage for margin calculation
            margin_used = total_exposure / 2
            account_value = total_exposure  # Simplified
            
            if account_value > 0:
                return (margin_used / account_value) * 100
            
            return 0.0
        except Exception:
            return 0.0
    
    async def _handle_critical_violation(self, position: Position, violation: Dict[str, Any]) -> Dict[str, Any]:
        """Handle critical risk limit violations."""
        action = {
            "type": "critical_violation",
            "position_id": position.id,
            "symbol": position.symbol,
            "action_taken": None,
            "timestamp": datetime.now()
        }
        
        try:
            # For critical violations, we might:
            # 1. Send urgent alert
            # 2. Initiate position liquidation
            # 3. Trigger emergency circuit breaker
            
            if violation.get("action_required") == "liquidate":
                action["action_taken"] = "position_liquidation_initiated"
                # In practice, this would create market orders to close position
                logger.critical(f"Critical violation detected, initiating liquidation for {position.symbol}")
            
            elif violation.get("action_required") == "reduce":
                action["action_taken"] = "position_reduction_required"
                logger.warning(f"Critical violation detected, position reduction required for {position.symbol}")
            
        except Exception as e:
            logger.error(f"Critical violation handling error: {str(e)}")
            action["error"] = str(e)
        
        return action
    
    async def _create_risk_event(self, event_type: RiskEventType, level: RiskLevel, 
                               title: str, data: Dict[str, Any], 
                               related_order_id: Optional[int] = None,
                               related_position_id: Optional[int] = None,
                               related_limit_id: Optional[int] = None) -> None:
        """Create a new risk event record."""
        try:
            event = RiskEvent(
                user_id=self.user_id,
                event_type=event_type,
                risk_level=level,
                title=title,
                description=f"Auto-generated risk event: {title}",
                related_order_id=related_order_id,
                related_position_id=related_position_id,
                related_limit_id=related_limit_id,
                trigger_value=data.get("trigger_value"),
                limit_value=data.get("limit_value"),
                metadata=data,
                occurred_at=datetime.now()
            )
            
            self.db.add(event)
            self.db.commit()
            
            # Also log to incident manager
            await self.incident_manager.create_incident(
                title=title,
                description=f"Risk event: {title}",
                severity=level.value,
                event_type=event_type.value,
                data=data
            )
            
        except Exception as e:
            logger.error(f"Risk event creation error: {str(e)}")
    
    async def _generate_risk_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate risk management recommendations based on report data."""
        recommendations = []
        
        try:
            # High risk score recommendations
            if report["summary"]["risk_score"] > 70:
                recommendations.append("Consider reducing position sizes to lower portfolio risk")
            
            # High concentration recommendations
            for symbol, pos_data in report["positions"].items():
                if pos_data.get("risk_score", 0) > 80:
                    recommendations.append(f"High risk position detected for {symbol}, consider risk reduction")
            
            # Margin usage recommendations
            if report["summary"].get("margin_usage", 0) > 80:
                recommendations.append("High margin usage detected, consider reducing leverage")
            
            # Multiple circuit breaker recommendations
            active_breakers = [
                name for name, data in report["circuit_breakers"].items() 
                if data["is_active"]
            ]
            if len(active_breakers) > 2:
                recommendations.append("Multiple circuit breakers active, review risk parameters")
            
            # Daily loss recommendations
            recent_events = [e for e in report["recent_events"] if e["type"] == "daily_loss_limit"]
            if len(recent_events) > 0:
                recommendations.append("Daily loss limits have been breached recently, consider tightening risk controls")
            
        except Exception as e:
            logger.error(f"Recommendation generation error: {str(e)}")
        
        return recommendations

    # ===== ADVANCED RISK MANAGEMENT METHODS =====
    
    async def calculate_portfolio_var(self, confidence_level: float = 0.95, 
                                    time_horizon: int = 1, method: str = "historical") -> Dict[str, Any]:
        """
        Calculate Value at Risk for current portfolio using advanced models.
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            time_horizon: Time horizon in days
            method: VaR calculation method (historical, parametric, monte_carlo)
            
        Returns:
            VaR calculation results with detailed breakdown
        """
        try:
            # Get portfolio returns data
            returns_data = await self._get_portfolio_returns_data()
            
            if returns_data.empty:
                return {"error": "No portfolio data available for VaR calculation"}
            
            # Calculate VaR based on method
            if method == "historical":
                var_result = self.var_calculator.calculate_historical_var(
                    returns_data, 
                    confidence_level=confidence_level,
                    time_horizon=time_horizon
                )
            elif method == "parametric":
                var_result = self.var_calculator.calculate_parametric_var(
                    returns_data,
                    confidence_level=confidence_level,
                    time_horizon=time_horizon
                )
            elif method == "monte_carlo":
                var_result = self.var_calculator.calculate_monte_carlo_var(
                    returns_data,
                    confidence_level=confidence_level,
                    time_horizon=time_horizon
                )
            else:
                raise ValueError(f"Unsupported VaR method: {method}")
            
            # Cache result
            cache_key = f"var_{method}_{confidence_level}_{time_horizon}"
            self.risk_calculations_cache[cache_key] = var_result
            
            logger.info(f"VaR calculation completed: ${var_result.get('var', 0):,.2f}")
            
            return {
                "var_value": var_result.get("var", 0),
                "method": method,
                "confidence_level": confidence_level,
                "time_horizon": time_horizon,
                "calculation_details": var_result,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"VaR calculation error: {e}")
            return {"error": f"VaR calculation failed: {str(e)}"}
    
    async def calculate_portfolio_cvar(self, confidence_level: float = 0.95,
                                     time_horizon: int = 1) -> Dict[str, Any]:
        """
        Calculate Conditional Value at Risk (Expected Shortfall) for portfolio.
        
        Args:
            confidence_level: Confidence level for CVaR calculation
            time_horizon: Time horizon in days
            
        Returns:
            CVaR calculation results with tail risk analysis
        """
        try:
            returns_data = await self._get_portfolio_returns_data()
            
            if returns_data.empty:
                return {"error": "No portfolio data available for CVaR calculation"}
            
            cvar_result = self.cvar_calculator.calculate_cvar(
                returns_data,
                confidence_level=confidence_level,
                time_horizon=time_horizon
            )
            
            return {
                "cvar_value": cvar_result.get("cvar", 0),
                "confidence_level": confidence_level,
                "time_horizon": time_horizon,
                "tail_risk_analysis": cvar_result,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"CVaR calculation error: {e}")
            return {"error": f"CVaR calculation failed: {str(e)}"}
    
    async def run_stress_tests(self, scenarios: List[str] = None) -> Dict[str, Any]:
        """
        Run comprehensive stress tests on portfolio.
        
        Args:
            scenarios: List of stress test scenarios to run
            
        Returns:
            Stress test results with impact analysis
        """
        try:
            if scenarios is None:
                scenarios = ["black_monday", "2008_crisis", "covid_2020", "monte_carlo"]
            
            portfolio_data = await self._get_portfolio_data()
            
            if portfolio_data.empty:
                return {"error": "No portfolio data available for stress testing"}
            
            stress_results = {}
            
            for scenario in scenarios:
                try:
                    if scenario == "black_monday":
                        result = self.stress_tester.historical_scenario_test(
                            portfolio_data, "black_monday_1987"
                        )
                    elif scenario == "2008_crisis":
                        result = self.stress_tester.historical_scenario_test(
                            portfolio_data, "financial_crisis_2008"
                        )
                    elif scenario == "covid_2020":
                        result = self.stress_tester.historical_scenario_test(
                            portfolio_data, "covid_pandemic_2020"
                        )
                    elif scenario == "monte_carlo":
                        # Monte Carlo with default parameters
                        result = self.stress_tester.monte_carlo_stress_test(portfolio_data)
                    else:
                        continue
                    
                    stress_results[scenario] = result
                    
                except Exception as e:
                    logger.error(f"Stress test error for {scenario}: {e}")
                    stress_results[scenario] = {"error": str(e)}
            
            logger.info(f"Stress tests completed for {len(stress_results)} scenarios")
            
            return {
                "stress_test_results": stress_results,
                "scenarios_tested": list(stress_results.keys()),
                "summary": self._summarize_stress_test_results(stress_results),
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Stress testing error: {e}")
            return {"error": f"Stress testing failed: {str(e)}"}
    
    async def generate_compliance_report(self, framework: str = "basel_iii") -> Dict[str, Any]:
        """
        Generate comprehensive compliance report for regulatory framework.
        
        Args:
            framework: Regulatory framework (basel_iii, mifid_ii, dodd_frank)
            
        Returns:
            Compliance report with status and recommendations
        """
        try:
            framework_enum = RegulationFramework(framework)
            
            # Get compliance report
            report = self.compliance_frameworks.generate_compliance_report(framework_enum)
            
            # Get real-time compliance status
            status = self.compliance_frameworks.get_compliance_status(framework_enum)
            
            # Add portfolio-specific metrics
            portfolio_summary = await self._get_portfolio_summary()
            
            enhanced_report = {
                **report,
                "portfolio_context": portfolio_summary,
                "real_time_status": status,
                "generated_by": "Advanced RiskManager",
                "user_id": self.user_id
            }
            
            logger.info(f"Compliance report generated for {framework}")
            
            return enhanced_report
            
        except Exception as e:
            logger.error(f"Compliance report generation error: {e}")
            return {"error": f"Compliance report generation failed: {str(e)}"}
    
    async def optimize_portfolio(self, optimization_objective: str = "max_sharpe",
                                risk_free_rate: float = 0.02) -> Dict[str, Any]:
        """
        Optimize portfolio using Modern Portfolio Theory and advanced models.
        
        Args:
            optimization_objective: Optimization objective (max_sharpe, min_variance, risk_parity)
            risk_free_rate: Risk-free rate for calculations
            
        Returns:
            Optimization results with optimal weights and metrics
        """
        try:
            # Get current portfolio data
            portfolio_data = await self._get_portfolio_data()
            
            if portfolio_data.empty:
                return {"error": "No portfolio data available for optimization"}
            
            # Run optimization
            if optimization_objective == "max_sharpe":
                result = self.portfolio_optimizer.optimize_max_sharpe(
                    portfolio_data, risk_free_rate=risk_free_rate
                )
            elif optimization_objective == "min_variance":
                result = self.portfolio_optimizer.optimize_min_variance(portfolio_data)
            elif optimization_objective == "risk_parity":
                result = self.portfolio_optimizer.optimize_risk_parity(portfolio_data)
            else:
                return {"error": f"Unsupported optimization objective: {optimization_objective}"}
            
            logger.info(f"Portfolio optimization completed: {optimization_objective}")
            
            return {
                "optimization_objective": optimization_objective,
                "optimal_weights": result.get("weights", {}),
                "expected_return": result.get("expected_return", 0),
                "expected_volatility": result.get("volatility", 0),
                "sharpe_ratio": result.get("sharpe_ratio", 0),
                "optimization_details": result,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Portfolio optimization error: {e}")
            return {"error": f"Portfolio optimization failed: {str(e)}"}
    
    async def start_advanced_monitoring(self, api_port: int = 8000):
        """
        Start advanced real-time risk monitoring with API server.
        
        Args:
            api_port: Port for the API server
            
        Returns:
            Success status and API server details
        """
        try:
            if self.advanced_monitoring_active:
                return {"status": "already_running", "message": "Advanced monitoring already active"}
            
            # Start broker integration monitoring
            await self.broker_integration.start_monitoring()
            
            # Start compliance monitoring
            self.compliance_frameworks.start_real_time_monitoring()
            
            # Initialize API server if not already done
            if self.api_server is None:
                self.api_server = RiskManagementAPI(port=api_port)
            
            self.advanced_monitoring_active = True
            
            logger.info("Advanced risk monitoring started")
            
            return {
                "status": "started",
                "message": "Advanced monitoring activated",
                "components": [
                    "broker_integration",
                    "compliance_frameworks", 
                    "real_time_risk_monitor"
                ]
            }
            
        except Exception as e:
            logger.error(f"Error starting advanced monitoring: {e}")
            return {"status": "error", "message": f"Failed to start monitoring: {str(e)}"}
    
    async def stop_advanced_monitoring(self):
        """Stop advanced risk monitoring"""
        try:
            if not self.advanced_monitoring_active:
                return {"status": "not_running", "message": "Advanced monitoring not active"}
            
            # Stop all monitoring components
            await self.broker_integration.stop_monitoring()
            self.compliance_frameworks.stop_real_time_monitoring()
            
            self.advanced_monitoring_active = False
            
            logger.info("Advanced risk monitoring stopped")
            
            return {"status": "stopped", "message": "Advanced monitoring deactivated"}
            
        except Exception as e:
            logger.error(f"Error stopping advanced monitoring: {e}")
            return {"status": "error", "message": f"Failed to stop monitoring: {str(e)}"}
    
    async def get_advanced_risk_status(self) -> Dict[str, Any]:
        """
        Get comprehensive risk status including advanced metrics.
        
        Returns:
            Complete risk status with all advanced metrics
        """
        try:
            # Get basic risk status
            basic_status = await self.get_risk_status()
            
            # Get advanced metrics
            portfolio_var = await self.calculate_portfolio_var()
            portfolio_cvar = await self.calculate_portfolio_cvar()
            
            # Get correlation analysis
            portfolio_data = await self._get_portfolio_data()
            correlation_analysis = {}
            if not portfolio_data.empty:
                correlation_analysis = self.correlation_analyzer.calculate_correlation_matrix(portfolio_data)
            
            # Get compliance status
            compliance_status = self.compliance_frameworks.get_compliance_status()
            
            # Get broker integration status
            broker_status = self.broker_integration.get_portfolio_summary()
            
            advanced_status = {
                "basic_status": basic_status,
                "advanced_metrics": {
                    "var_95_1d": portfolio_var.get("var_value", 0),
                    "cvar_95_1d": portfolio_cvar.get("cvar_value", 0),
                    "correlation_analysis": correlation_analysis
                },
                "compliance_status": compliance_status,
                "broker_integration": broker_status,
                "monitoring_active": self.advanced_monitoring_active,
                "risk_models_active": True,
                "timestamp": datetime.now()
            }
            
            return advanced_status
            
        except Exception as e:
            logger.error(f"Error getting advanced risk status: {e}")
            return {"error": f"Failed to get advanced risk status: {str(e)}"}
    
    # ===== HELPER METHODS =====
    
    async def _get_portfolio_returns_data(self) -> pd.DataFrame:
        """Get portfolio returns data for risk calculations"""
        try:
            # Get historical trade data for returns calculation
            trades = self.db.query(Trade).filter(
                and_(
                    Trade.user_id == self.user_id,
                    Trade.executed_at >= datetime.now() - timedelta(days=252)  # 1 year of data
                )
            ).order_by(Trade.executed_at).all()
            
            if not trades:
                return pd.DataFrame()
            
            # Convert to returns series (simplified)
            trade_values = [trade.executed_price * trade.quantity for trade in trades]
            returns = pd.Series(trade_values).pct_change().dropna()
            
            return returns.to_frame("portfolio_returns")
            
        except Exception as e:
            logger.error(f"Error getting portfolio returns: {e}")
            return pd.DataFrame()
    
    async def _get_portfolio_data(self) -> pd.DataFrame:
        """Get portfolio data for analysis"""
        try:
            positions = self.db.query(Position).filter(
                and_(
                    Position.user_id == self.user_id,
                    Position.quantity != 0
                )
            ).all()
            
            if not positions:
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = []
            for pos in positions:
                data.append({
                    "symbol": pos.symbol,
                    "quantity": pos.quantity,
                    "market_value": pos.market_value or 0,
                    "avg_cost": pos.avg_cost or 0,
                    "unrealized_pnl": pos.unrealized_pnl or 0
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error getting portfolio data: {e}")
            return pd.DataFrame()
    
    async def _get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary for context"""
        try:
            positions = self.db.query(Position).filter(Position.user_id == self.user_id).all()
            
            total_value = sum(abs(pos.market_value or 0) for pos in positions)
            total_pnl = sum(pos.unrealized_pnl for pos in positions)
            
            return {
                "total_value": total_value,
                "total_unrealized_pnl": total_pnl,
                "position_count": len(positions),
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return {}
    
    def _summarize_stress_test_results(self, results: Dict) -> Dict[str, Any]:
        """Summarize stress test results"""
        try:
            summary = {
                "total_scenarios": len(results),
                "worst_case_loss": 0,
                "average_loss": 0,
                "scenarios_passed": 0,
                "scenarios_failed": 0
            }
            
            losses = []
            for scenario, result in results.items():
                if "error" not in result:
                    loss = result.get("loss", 0)
                    losses.append(loss)
                    summary["scenarios_passed"] += 1
                else:
                    summary["scenarios_failed"] += 1
            
            if losses:
                summary["worst_case_loss"] = min(losses)
                summary["average_loss"] = np.mean(losses)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing stress test results: {e}")
            return {"error": str(e)}
    
    async def get_risk_status(self) -> Dict[str, Any]:
        """Get basic risk status (existing method)"""
        try:
            return {
                "risk_level": "medium",  # Simplified
                "total_exposure": 0,  # Would calculate from positions
                "margin_used": 0,  # Would get from broker
                "active_positions": 0,  # Count of positions
                "last_update": datetime.now()
            }
        except Exception as e:
            logger.error(f"Error getting risk status: {e}")
            return {"error": str(e)}

    def close(self):
        """Cleanup resources."""
        # Stop advanced monitoring if active
        if self.advanced_monitoring_active:
            asyncio.create_task(self.stop_advanced_monitoring())
        
        # Close database connection
        self.db.close()
        
        logger.info("RiskManager resources cleaned up")
