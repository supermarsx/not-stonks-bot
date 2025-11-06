"""
Trading Safeguards Integration Layer

Integrates all trading safeguards and risk controls:
- Capital preservation safeguards
- Margin trading controls  
- Daily trading limits
- Drawdown protection
- Emergency stop procedures
- Circuit breakers integration
- Unified risk monitoring

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

from .capital_management import CapitalManager, CapitalAllocationStrategy
from .margin_control import MarginController, LeverageLevel, MarginAccountType
from .daily_limits import DailyLimitManager, LimitType
from .drawdown_protection import DrawdownProtector, ProtectionAction
from .emergency_stop import EmergencyStopManager, EmergencyStopType, EmergencyLevel
from .circuit_breakers import CircuitBreakerManager
from .market_circuit_breakers import MarketCircuitBreakerManager, CircuitBreakerLevel
from .regulatory_compliance import RegulatoryComplianceManager, ComplianceLevel, RegulatoryRegion
from .alerting_system import RiskAlertingManager, AlertSeverity, AlertCategory
from .backup_mechanisms import BackupTradingManager, BrokerStatus
from .liquidity_validation import LiquidityValidationManager, LiquidityLevel
from .audit_trail import AuditTrailManager, AuditCategory, AuditSeverity, AuditContext, RiskDecision
from .limits import RiskLimitChecker
from .audit import AuditLogger
from .incidents import IncidentManager
from ..database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)


class SafeguardStatus(Enum):
    """Overall safeguard status."""
    OPTIMAL = "optimal"       # All systems optimal
    MONITORING = "monitoring"  # Normal monitoring active
    WARNING = "warning"       # Some warnings active
    RESTRICTED = "restricted" # Some restrictions active
    CRITICAL = "critical"     # Critical restrictions active
    EMERGENCY = "emergency"   # Emergency procedures active


@dataclass
class SafeguardConfiguration:
    """Unified safeguard configuration."""
    capital_preservation: bool = True
    margin_controls: bool = True
    daily_limits: bool = True
    drawdown_protection: bool = True
    emergency_procedures: bool = True
    circuit_breakers: bool = True
    market_circuit_breakers: bool = True
    regulatory_compliance: bool = True
    alerting_system: bool = True
    backup_mechanisms: bool = True
    liquidity_validation: bool = True
    audit_trail: bool = True
    auto_monitoring: bool = True
    
    # Default settings
    max_daily_loss: float = 1000.0
    max_drawdown: float = 15.0
    max_leverage: float = 2.0
    max_position_size: float = 0.10  # 10% of capital
    
    # Alert settings
    alert_enabled: bool = True
    alert_channels: List[str] = field(default_factory=lambda: ["console", "email"])
    
    # Database and external integrations
    db_manager: Optional[DatabaseManager] = None
    alerting_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SafeguardMetrics:
    """Comprehensive safeguard metrics."""
    status: SafeguardStatus
    capital_preservation: Dict[str, Any]
    margin_controls: Dict[str, Any]
    daily_limits: Dict[str, Any]
    drawdown_protection: Dict[str, Any]
    emergency_status: Dict[str, Any]
    overall_risk_score: float
    active_restrictions: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class TradingSafeguardsManager:
    """
    Unified Trading Safeguards and Risk Controls Manager
    
    Integrates all trading safeguards into a comprehensive risk management
    system with unified monitoring, controls, and emergency procedures.
    
    Features:
    - Capital preservation safeguards
    - Margin trading controls
    - Daily trading limits
    - Drawdown protection
    - Emergency stop procedures
    - Circuit breakers (internal and market)
    - Regulatory compliance checks
    - Risk alerting and escalation
    - Backup trading mechanisms
    - Liquidity validation
    - Comprehensive audit trails
    """
    
    def __init__(self, user_id: int, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize Trading Safeguards Manager.
        
        Args:
            user_id: User identifier for safeguard tracking
            db_manager: Database manager for async operations
        """
        self.user_id = user_id
        self.db_manager = db_manager
        
        # Core safeguard components
        self.capital_manager = None
        self.margin_controller = None
        self.daily_limit_manager = None
        self.drawdown_protector = None
        self.emergency_stop_manager = None
        
        # Advanced risk management components
        self.market_circuit_breaker_manager = None
        self.regulatory_compliance_manager = None
        self.alerting_manager = None
        self.backup_trading_manager = None
        self.liquidity_validation_manager = None
        self.audit_trail_manager = None
        
        # Existing risk management components
        self.circuit_breaker_manager = None
        self.risk_limit_checker = None
        self.audit_logger = None
        self.incident_manager = None
        
        # Configuration and state
        self.config = SafeguardConfiguration()
        self.metrics = None
        self.monitoring_active = False
        
        # Integration settings
        self.cross_component_correlation = True
        self.auto_response_enabled = True
        self.escalation_threshold = 0.8  # 80% of limits
        
        # Tracking
        self.safeguard_violations = []
        self.system_alerts = []
        self.last_integration_check = None
        self.active_orders: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"TradingSafeguardsManager initialized for user {self.user_id}")
    
    async def initialize_safeguards(self, config: SafeguardConfiguration = None) -> Dict[str, Any]:
        """
        Initialize all trading safeguards with configuration.
        
        Args:
            config: Safeguard configuration
            
        Returns:
            Initialization result
        """
        try:
            if config:
                self.config = config
            
            # Initialize each component
            init_results = []
            
            # Capital preservation safeguards
            if self.config.capital_preservation:
                capital_result = await self._initialize_capital_preservation()
                init_results.append(("capital_preservation", capital_result))
            
            # Margin trading controls
            if self.config.margin_controls:
                margin_result = await self._initialize_margin_controls()
                init_results.append(("margin_controls", margin_result))
            
            # Daily trading limits
            if self.config.daily_limits:
                daily_result = await self._initialize_daily_limits()
                init_results.append(("daily_limits", daily_result))
            
            # Drawdown protection
            if self.config.drawdown_protection:
                drawdown_result = await self._initialize_drawdown_protection()
                init_results.append(("drawdown_protection", drawdown_result))
            
            # Emergency procedures
            if self.config.emergency_procedures:
                emergency_result = await self._initialize_emergency_procedures()
                init_results.append(("emergency_procedures", emergency_result))
            
            # Market circuit breakers
            if self.config.market_circuit_breakers:
                market_cb_result = await self._initialize_market_circuit_breakers()
                init_results.append(("market_circuit_breakers", market_cb_result))
            
            # Regulatory compliance
            if self.config.regulatory_compliance:
                compliance_result = await self._initialize_regulatory_compliance()
                init_results.append(("regulatory_compliance", compliance_result))
            
            # Alerting system
            if self.config.alerting_system:
                alerting_result = await self._initialize_alerting_system()
                init_results.append(("alerting_system", alerting_result))
            
            # Backup mechanisms
            if self.config.backup_mechanisms:
                backup_result = await self._initialize_backup_mechanisms()
                init_results.append(("backup_mechanisms", backup_result))
            
            # Liquidity validation
            if self.config.liquidity_validation:
                liquidity_result = await self._initialize_liquidity_validation()
                init_results.append(("liquidity_validation", liquidity_result))
            
            # Audit trail
            if self.config.audit_trail:
                audit_result = await self._initialize_audit_trail()
                init_results.append(("audit_trail", audit_result))
            
            # Enable auto monitoring
            if self.config.auto_monitoring:
                await self.enable_unified_monitoring()
            
            logger.info(f"All safeguards initialized for user {self.user_id}")
            
            return {
                "success": True,
                "initialized_components": len(init_results),
                "components": dict(init_results),
                "configuration": self.config.__dict__,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Safeguards initialization error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def comprehensive_risk_assessment(self) -> Dict[str, Any]:
        """
        Perform comprehensive risk assessment across all safeguards.
        
        Returns:
            Complete risk assessment report
        """
        try:
            assessment = {
                "user_id": self.user_id,
                "assessment_time": datetime.now(),
                "overall_status": SafeguardStatus.MONITORING.value,
                "risk_score": 0.0,
                "components": {},
                "violations": [],
                "warnings": [],
                "recommendations": [],
                "immediate_actions": []
            }
            
            # Assess each safeguard component
            component_scores = []
            
            # Capital preservation assessment
            if self.config.capital_preservation:
                capital_assessment = await self._assess_capital_preservation()
                assessment["components"]["capital_preservation"] = capital_assessment
                component_scores.append(capital_assessment.get("risk_score", 50))
            
            # Margin controls assessment
            if self.config.margin_controls:
                margin_assessment = await self._assess_margin_controls()
                assessment["components"]["margin_controls"] = margin_assessment
                component_scores.append(margin_assessment.get("risk_score", 50))
            
            # Daily limits assessment
            if self.config.daily_limits:
                daily_assessment = await self._assess_daily_limits()
                assessment["components"]["daily_limits"] = daily_assessment
                component_scores.append(daily_assessment.get("risk_score", 50))
            
            # Drawdown protection assessment
            if self.config.drawdown_protection:
                drawdown_assessment = await self._assess_drawdown_protection()
                assessment["components"]["drawdown_protection"] = drawdown_assessment
                component_scores.append(drawdown_assessment.get("risk_score", 50))
            
            # Emergency status assessment
            emergency_assessment = await self._assess_emergency_status()
            assessment["components"]["emergency_status"] = emergency_assessment
            component_scores.append(emergency_assessment.get("risk_score", 50))
            
            # Calculate overall risk score
            if component_scores:
                assessment["risk_score"] = sum(component_scores) / len(component_scores)
            
            # Determine overall status
            assessment["overall_status"] = self._determine_overall_status(assessment)
            
            # Collect violations and warnings
            for component_name, component_data in assessment["components"].items():
                violations = component_data.get("violations", [])
                warnings = component_data.get("warnings", [])
                
                for violation in violations:
                    assessment["violations"].append({
                        "component": component_name,
                        "violation": violation
                    })
                
                for warning in warnings:
                    assessment["warnings"].append({
                        "component": component_name,
                        "warning": warning
                    })
            
            # Generate recommendations
            assessment["recommendations"] = await self._generate_comprehensive_recommendations(assessment)
            
            # Determine immediate actions
            assessment["immediate_actions"] = await self._determine_immediate_actions(assessment)
            
            # Update stored metrics
            self.metrics = SafeguardMetrics(
                status=SafeguardStatus(assessment["overall_status"]),
                capital_preservation=assessment["components"].get("capital_preservation", {}),
                margin_controls=assessment["components"].get("margin_controls", {}),
                daily_limits=assessment["components"].get("daily_limits", {}),
                drawdown_protection=assessment["components"].get("drawdown_protection", {}),
                emergency_status=assessment["components"].get("emergency_status", {}),
                overall_risk_score=assessment["risk_score"],
                active_restrictions=assessment["immediate_actions"]
            )
            
            return assessment
            
        except Exception as e:
            logger.error(f"Comprehensive risk assessment error: {str(e)}")
            return {
                "error": f"Risk assessment failed: {str(e)}",
                "overall_status": SafeguardStatus.CRITICAL.value,
                "timestamp": datetime.now()
            }
    
    async def validate_trade_comprehensive(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive trade validation across all safeguards.
        
        Args:
            order_data: Trade order information
            
        Returns:
            Comprehensive validation result
        """
        try:
            validation_result = {
                "approved": True,
                "validation_passed": True,
                "component_validations": {},
                "warnings": [],
                "restrictions": [],
                "recommendations": [],
                "risk_metrics": {},
                "timestamp": datetime.now()
            }
            
            # Validate capital preservation
            if self.config.capital_preservation:
                capital_validation = await self._validate_capital_preservation(order_data)
                validation_result["component_validations"]["capital_preservation"] = capital_validation
                
                if not capital_validation["approved"]:
                    validation_result["approved"] = False
                    validation_result["validation_passed"] = False
            
            # Validate margin controls
            if self.config.margin_controls:
                margin_validation = await self._validate_margin_controls(order_data)
                validation_result["component_validations"]["margin_controls"] = margin_validation
                
                if not margin_validation["approved"]:
                    validation_result["approved"] = False
                    validation_result["validation_passed"] = False
            
            # Validate daily limits
            if self.config.daily_limits:
                daily_validation = await self._validate_daily_limits(order_data)
                validation_result["component_validations"]["daily_limits"] = daily_validation
                
                if not daily_validation["approved"]:
                    validation_result["approved"] = False
                    validation_result["validation_passed"] = False
            
            # Validate drawdown protection
            if self.config.drawdown_protection:
                drawdown_validation = await self._validate_drawdown_protection(order_data)
                validation_result["component_validations"]["drawdown_protection"] = drawdown_validation
                
                if not drawdown_validation["approved"]:
                    validation_result["approved"] = False
                    validation_result["validation_passed"] = False
            
            # Check emergency status
            emergency_status = await self.emergency_stop_manager.get_emergency_status()
            validation_result["component_validations"]["emergency_status"] = emergency_status
            
            if emergency_status.get("emergency_status", {}).get("global_halt"):
                validation_result.update({
                    "approved": False,
                    "validation_passed": False,
                    "reason": "Global trading halt is active"
                })
            
            # Aggregate warnings and recommendations
            for component, component_validation in validation_result["component_validations"].items():
                if "warnings" in component_validation:
                    validation_result["warnings"].extend(component_validation["warnings"])
                if "recommendations" in component_validation:
                    validation_result["recommendations"].extend(component_validation["recommendations"])
            
            # Calculate risk metrics
            validation_result["risk_metrics"] = await self._calculate_integrated_risk_metrics(order_data)
            
            # Check if auto-response should be triggered
            if validation_result["risk_metrics"].get("overall_risk_score", 0) > 80:
                auto_response_result = await self._trigger_auto_response(validation_result)
                validation_result["auto_response"] = auto_response_result
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Comprehensive trade validation error: {str(e)}")
            return {
                "approved": False,
                "validation_passed": False,
                "error": str(e),
                "timestamp": datetime.now()
            }
    
    async def enable_unified_monitoring(self, monitoring_interval: int = 60):
        """
        Enable unified monitoring across all safeguard components.
        
        Args:
            monitoring_interval: Monitoring frequency in seconds
        """
        try:
            if self.monitoring_active:
                logger.info("Unified monitoring already active")
                return
            
            self.monitoring_active = True
            
            # Enable component-specific monitoring
            await self.margin_controller._enable_monitoring() if hasattr(self.margin_controller, '_enable_monitoring') else None
            await self.drawdown_protector.enable_auto_monitoring() if hasattr(self.drawdown_protector, 'enable_auto_monitoring') else None
            await self.emergency_stop_manager.enable_auto_monitoring() if hasattr(self.emergency_stop_manager, 'enable_auto_monitoring') else None
            
            # Start unified monitoring loop
            asyncio.create_task(self._unified_monitoring_loop(monitoring_interval))
            
            logger.info(f"Unified monitoring enabled for user {self.user_id}")
            
        except Exception as e:
            logger.error(f"Unified monitoring enable error: {str(e)}")
    
    async def disable_unified_monitoring(self):
        """Disable unified monitoring across all components."""
        try:
            self.monitoring_active = False
            
            # Disable component-specific monitoring
            await self.margin_controller.disable_auto_monitoring() if hasattr(self.margin_controller, 'disable_auto_monitoring') else None
            await self.drawdown_protector.disable_auto_monitoring() if hasattr(self.drawdown_protector, 'disable_auto_monitoring') else None
            await self.emergency_stop_manager.disable_auto_monitoring() if hasattr(self.emergency_stop_manager, 'disable_auto_monitoring') else None
            
            logger.info(f"Unified monitoring disabled for user {self.user_id}")
            
        except Exception as e:
            logger.error(f"Unified monitoring disable error: {str(e)}")
    
    async def get_safeguards_dashboard(self) -> Dict[str, Any]:
        """
        Get comprehensive safeguards dashboard data.
        
        Returns:
            Dashboard data for UI display
        """
        try:
            # Get current metrics
            current_metrics = await self.comprehensive_risk_assessment()
            
            # Get individual component summaries
            component_summaries = {}
            
            if self.config.capital_preservation:
                capital_summary = await self._get_capital_summary()
                component_summaries["capital_preservation"] = capital_summary
            
            if self.config.margin_controls:
                margin_summary = await self._get_margin_summary()
                component_summaries["margin_controls"] = margin_summary
            
            if self.config.daily_limits:
                daily_summary = await self._get_daily_limits_summary()
                component_summaries["daily_limits"] = daily_summary
            
            if self.config.drawdown_protection:
                drawdown_summary = await self._get_drawdown_summary()
                component_summaries["drawdown_protection"] = drawdown_summary
            
            if self.config.emergency_procedures:
                emergency_summary = await self._get_emergency_summary()
                component_summaries["emergency_procedures"] = emergency_summary
            
            # Create dashboard data
            dashboard = {
                "overview": {
                    "status": current_metrics["overall_status"],
                    "risk_score": current_metrics["risk_score"],
                    "violations_count": len(current_metrics["violations"]),
                    "warnings_count": len(current_metrics["warnings"]),
                    "last_assessment": current_metrics["assessment_time"]
                },
                "components": component_summaries,
                "active_restrictions": current_metrics["immediate_actions"],
                "recommendations": current_metrics["recommendations"],
                "system_health": {
                    "monitoring_active": self.monitoring_active,
                    "components_initialized": len([c for c in component_summaries.keys()]),
                    "auto_response_enabled": self.auto_response_enabled
                },
                "timestamp": datetime.now()
            }
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Safeguards dashboard error: {str(e)}")
            return {"error": str(e)}
    
    async def emergency_override_all_safeguards(self, reason: str = "Manual override",
                                              duration_minutes: int = 60) -> Dict[str, Any]:
        """
        Emergency override of all safeguards (use with extreme caution).
        
        Args:
            reason: Reason for override
            duration_minutes: Override duration
            
        Returns:
            Override result
        """
        try:
            logger.critical(f"EMERGENCY OVERRIDE activated for user {self.user_id}: {reason}")
            
            # Trigger emergency stop
            emergency_result = await self.emergency_stop_manager.trigger_emergency_stop(
                stop_type=EmergencyStopType.MANUAL,
                reason=f"Emergency Override: {reason}",
                triggered_by="user",
                level=EmergencyLevel.EMERGENCY,
                confirm_execution=True
            )
            
            # Set temporary override period
            override_until = datetime.now() + timedelta(minutes=duration_minutes)
            
            # Log override event
            await self.audit_logger.log_risk_action("emergency_override", {
                "reason": reason,
                "duration_minutes": duration_minutes,
                "override_until": override_until,
                "triggered_by": "user"
            })
            
            return {
                "success": True,
                "override_activated": True,
                "emergency_stop": emergency_result,
                "override_until": override_until,
                "duration_minutes": duration_minutes,
                "reason": reason,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Emergency override error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _initialize_capital_preservation(self) -> Dict[str, Any]:
        """Initialize capital preservation safeguards."""
        try:
            # Set conservative allocation strategy
            self.capital_manager.set_allocation_strategy(CapitalAllocationStrategy.MODERATE)
            
            # Configure position sizing
            from .capital_management import PositionSizingConfig
            sizing_config = PositionSizingConfig(
                max_position_percent=self.config.max_position_size,
                max_single_trade_percent=self.config.max_position_size * 0.5
            )
            self.capital_manager.set_position_sizing_config(sizing_config)
            
            return {"success": True, "allocation_strategy": "moderate"}
            
        except Exception as e:
            logger.error(f"Capital preservation initialization error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _initialize_margin_controls(self) -> Dict[str, Any]:
        """Initialize margin control safeguards."""
        try:
            # Set conservative leverage limits
            self.margin_controller.set_leverage_limit(LeverageLevel.MODERATE)
            
            return {"success": True, "leverage_limit": "moderate"}
            
        except Exception as e:
            logger.error(f"Margin controls initialization error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _initialize_daily_limits(self) -> Dict[str, Any]:
        """Initialize daily trading limits."""
        try:
            # Update default daily loss limit
            await self.daily_limit_manager.update_daily_limit(
                LimitType.DAILY_LOSS,
                self.config.max_daily_loss,
                self.config.max_daily_loss * 0.8
            )
            
            return {"success": True, "daily_loss_limit": self.config.max_daily_loss}
            
        except Exception as e:
            logger.error(f"Daily limits initialization error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _initialize_drawdown_protection(self) -> Dict[str, Any]:
        """Initialize drawdown protection."""
        try:
            # Create portfolio drawdown limit
            await self.drawdown_protector.create_drawdown_limit(
                limit_name="Portfolio Max Drawdown",
                max_drawdown=self.config.max_drawdown,
                warning_threshold=self.config.max_drawdown * 0.7,
                action=ProtectionAction.REDUCE_POSITION_SIZE
            )
            
            return {"success": True, "max_drawdown": self.config.max_drawdown}
            
        except Exception as e:
            logger.error(f"Drawdown protection initialization error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _initialize_emergency_procedures(self) -> Dict[str, Any]:
        """Initialize emergency procedures."""
        try:
            # Emergency procedures are auto-initialized in constructor
            return {"success": True, "procedures_configured": len(self.emergency_stop_manager.emergency_procedures)}
            
        except Exception as e:
            logger.error(f"Emergency procedures initialization error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _initialize_market_circuit_breakers(self) -> Dict[str, Any]:
        """Initialize market circuit breakers."""
        try:
            if not self.db_manager:
                return {"success": False, "error": "Database manager required for market circuit breakers"}
            
            from .market_circuit_breakers import create_circuit_breaker_manager
            self.market_circuit_breaker_manager = await create_circuit_breaker_manager(self.db_manager)
            
            return {"success": True, "circuit_breakers_active": True}
            
        except Exception as e:
            logger.error(f"Market circuit breakers initialization error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _initialize_regulatory_compliance(self) -> Dict[str, Any]:
        """Initialize regulatory compliance checks."""
        try:
            if not self.db_manager:
                return {"success": False, "error": "Database manager required for regulatory compliance"}
            
            from .regulatory_compliance import create_compliance_manager
            self.regulatory_compliance_manager = await create_compliance_manager(self.db_manager)
            
            return {"success": True, "compliance_checks_active": True}
            
        except Exception as e:
            logger.error(f"Regulatory compliance initialization error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _initialize_alerting_system(self) -> Dict[str, Any]:
        """Initialize risk alerting system."""
        try:
            if not self.db_manager:
                return {"success": False, "error": "Database manager required for alerting system"}
            
            from .alerting_system import create_alerting_manager
            self.alerting_manager = await create_alerting_manager(self.db_manager, self.config.alerting_config)
            
            return {"success": True, "alerting_active": True}
            
        except Exception as e:
            logger.error(f"Alerting system initialization error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _initialize_backup_mechanisms(self) -> Dict[str, Any]:
        """Initialize backup trading mechanisms."""
        try:
            if not self.db_manager:
                return {"success": False, "error": "Database manager required for backup mechanisms"}
            
            from .backup_mechanisms import create_backup_trading_manager
            self.backup_trading_manager = await create_backup_trading_manager(self.db_manager)
            
            return {"success": True, "backup_mechanisms_active": True}
            
        except Exception as e:
            logger.error(f"Backup mechanisms initialization error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _initialize_liquidity_validation(self) -> Dict[str, Any]:
        """Initialize liquidity validation system."""
        try:
            if not self.db_manager:
                return {"success": False, "error": "Database manager required for liquidity validation"}
            
            from .liquidity_validation import create_liquidity_manager
            self.liquidity_validation_manager = await create_liquidity_manager(self.db_manager)
            
            return {"success": True, "liquidity_validation_active": True}
            
        except Exception as e:
            logger.error(f"Liquidity validation initialization error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _initialize_audit_trail(self) -> Dict[str, Any]:
        """Initialize comprehensive audit trail system."""
        try:
            if not self.db_manager:
                return {"success": False, "error": "Database manager required for audit trail"}
            
            from .audit_trail import create_audit_trail_manager
            self.audit_trail_manager = await create_audit_trail_manager(self.db_manager)
            
            return {"success": True, "audit_trail_active": True}
            
        except Exception as e:
            logger.error(f"Audit trail initialization error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _assess_capital_preservation(self) -> Dict[str, Any]:
        """Assess capital preservation safeguards."""
        try:
            capital_metrics = await self.capital_manager.calculate_available_capital()
            capital_alerts = await self.capital_manager.monitor_capital_alerts()
            
            risk_score = 50  # Base score
            
            # Adjust based on capital utilization
            utilization = capital_metrics.get("capital_utilization", 0)
            if utilization > 0.8:
                risk_score += 30
            elif utilization > 0.6:
                risk_score += 15
            
            # Adjust based on alerts
            critical_alerts = [alert for alert in capital_alerts if alert.get("level") == "critical"]
            if critical_alerts:
                risk_score += 40
            
            return {
                "risk_score": min(100, risk_score),
                "status": "warning" if critical_alerts else "ok",
                "capital_metrics": capital_metrics,
                "alerts": capital_alerts,
                "violations": [alert for alert in capital_alerts if alert.get("level") == "critical"],
                "warnings": [alert for alert in capital_alerts if alert.get("level") == "warning"]
            }
            
        except Exception as e:
            logger.error(f"Capital preservation assessment error: {str(e)}")
            return {"risk_score": 100, "error": str(e)}
    
    async def _assess_margin_controls(self) -> Dict[str, Any]:
        """Assess margin control safeguards."""
        try:
            margin_status = await self.margin_controller.calculate_margin_status()
            margin_alerts = await self.margin_controller.monitor_margin_alerts()
            
            risk_score = 50
            
            # Adjust based on leverage
            leverage = margin_status.get("leverage_ratio", 1)
            if leverage > 3:
                risk_score += 40
            elif leverage > 2:
                risk_score += 20
            
            # Adjust based on health score
            health_score = margin_status.get("margin_health_score", 100)
            if health_score < 40:
                risk_score += 30
            elif health_score < 60:
                risk_score += 15
            
            critical_alerts = [alert for alert in margin_alerts if alert.get("severity") == "critical"]
            if critical_alerts:
                risk_score += 40
            
            return {
                "risk_score": min(100, risk_score),
                "status": "warning" if critical_alerts else "ok",
                "margin_status": margin_status,
                "alerts": margin_alerts,
                "violations": [alert for alert in margin_alerts if alert.get("severity") == "critical"],
                "warnings": [alert for alert in margin_alerts if alert.get("severity") == "warning"]
            }
            
        except Exception as e:
            logger.error(f"Margin controls assessment error: {str(e)}")
            return {"risk_score": 100, "error": str(e)}
    
    async def _assess_daily_limits(self) -> Dict[str, Any]:
        """Assess daily trading limits."""
        try:
            limit_status = await self.daily_limit_manager.check_daily_limits()
            
            risk_score = 50
            
            if limit_status.get("halted"):
                risk_score += 50
            elif limit_status.get("cooldown_active"):
                risk_score += 30
            
            breached_limits = len(limit_status.get("limits_breached", []))
            if breached_limits > 0:
                risk_score += breached_limits * 20
            
            return {
                "risk_score": min(100, risk_score),
                "status": "critical" if limit_status.get("halted") else "warning" if breached_limits > 0 else "ok",
                "limit_status": limit_status,
                "violations": limit_status.get("limits_breached", []),
                "warnings": limit_status.get("warnings", [])
            }
            
        except Exception as e:
            logger.error(f"Daily limits assessment error: {str(e)}")
            return {"risk_score": 100, "error": str(e)}
    
    async def _assess_drawdown_protection(self) -> Dict[str, Any]:
        """Assess drawdown protection."""
        try:
            drawdown_status = await self.drawdown_protector.check_drawdown_limits()
            drawdown_metrics = await self.drawdown_protector.calculate_drawdown_metrics()
            
            risk_score = 50
            
            # Adjust based on current drawdown
            current_dd = drawdown_metrics.get("drawdown_percentage", 0)
            if current_dd > 20:
                risk_score += 40
            elif current_dd > 15:
                risk_score += 30
            elif current_dd > 10:
                risk_score += 20
            elif current_dd > 5:
                risk_score += 10
            
            # Adjust based on protection level
            protection_level = drawdown_status.get("protection_level", "normal")
            if protection_level == "critical":
                risk_score += 40
            elif protection_level == "elevated":
                risk_score += 20
            
            breached_limits = len(drawdown_status.get("breached_limits", []))
            if breached_limits > 0:
                risk_score += breached_limits * 25
            
            return {
                "risk_score": min(100, risk_score),
                "status": protection_level,
                "drawdown_status": drawdown_status,
                "drawdown_metrics": drawdown_metrics,
                "violations": drawdown_status.get("breached_limits", []),
                "warnings": drawdown_status.get("warning_limits", [])
            }
            
        except Exception as e:
            logger.error(f"Drawdown protection assessment error: {str(e)}")
            return {"risk_score": 100, "error": str(e)}
    
    async def _assess_emergency_status(self) -> Dict[str, Any]:
        """Assess emergency procedures status."""
        try:
            emergency_status = await self.emergency_stop_manager.get_emergency_status()
            
            risk_score = 10  # Low base score for emergency status
            
            if emergency_status.get("emergency_status", {}).get("global_halt"):
                risk_score = 100
            elif emergency_status.get("active_emergencies", 0) > 0:
                risk_score = 80
            
            return {
                "risk_score": risk_score,
                "status": "emergency" if risk_score >= 80 else "normal",
                "emergency_status": emergency_status,
                "violations": [],
                "warnings": []
            }
            
        except Exception as e:
            logger.error(f"Emergency status assessment error: {str(e)}")
            return {"risk_score": 100, "error": str(e)}
    
    def _determine_overall_status(self, assessment: Dict[str, Any]) -> str:
        """Determine overall safeguard status."""
        try:
            risk_score = assessment["risk_score"]
            
            if risk_score >= 90:
                return SafeguardStatus.EMERGENCY.value
            elif risk_score >= 75:
                return SafeguardStatus.CRITICAL.value
            elif risk_score >= 60:
                return SafeguardStatus.RESTRICTED.value
            elif risk_score >= 40:
                return SafeguardStatus.WARNING.value
            else:
                return SafeguardStatus.OPTIMAL.value
                
        except Exception as e:
            logger.error(f"Overall status determination error: {str(e)}")
            return SafeguardStatus.CRITICAL.value
    
    async def _generate_comprehensive_recommendations(self, assessment: Dict[str, Any]) -> List[str]:
        """Generate comprehensive recommendations."""
        try:
            recommendations = []
            
            # Capital recommendations
            capital_component = assessment["components"].get("capital_preservation", {})
            if capital_component.get("risk_score", 50) > 70:
                recommendations.append("Consider reducing position sizes due to high capital utilization")
            
            # Margin recommendations
            margin_component = assessment["components"].get("margin_controls", {})
            if margin_component.get("risk_score", 50) > 70:
                recommendations.append("Monitor margin levels closely - consider reducing leverage")
            
            # Daily limits recommendations
            daily_component = assessment["components"].get("daily_limits", {})
            if daily_component.get("status") == "critical":
                recommendations.append("Daily limits breached - review trading strategy immediately")
            
            # Drawdown recommendations
            drawdown_component = assessment["components"].get("drawdown_protection", {})
            current_dd = drawdown_component.get("drawdown_metrics", {}).get("drawdown_percentage", 0)
            if current_dd > 10:
                recommendations.append("High drawdown detected - consider risk reduction measures")
            
            return recommendations if recommendations else ["All safeguards operating within normal parameters"]
            
        except Exception as e:
            logger.error(f"Recommendations generation error: {str(e)}")
            return ["Monitor all safeguard systems closely"]
    
    async def _determine_immediate_actions(self, assessment: Dict[str, Any]) -> List[str]:
        """Determine immediate actions required."""
        try:
            actions = []
            
            # Check for critical violations
            violations = assessment.get("violations", [])
            for violation in violations:
                if violation.get("component") == "daily_limits":
                    actions.append("halt_trading")
                elif violation.get("component") == "drawdown_protection":
                    actions.append("reduce_positions")
                elif violation.get("component") == "margin_controls":
                    actions.append("increase_margin")
            
            # Check overall risk score
            if assessment["risk_score"] > 80:
                actions.append("emergency_review")
            
            return actions
            
        except Exception as e:
            logger.error(f"Immediate actions determination error: {str(e)}")
            return ["immediate_review"]
    
    async def _validate_capital_preservation(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate order against capital preservation safeguards."""
        try:
            validation = await self.capital_manager.validate_position_size(
                symbol=order_data.get("symbol"),
                quantity=order_data.get("quantity", 0),
                price=order_data.get("limit_price", 0) or order_data.get("estimated_price", 0),
                order_type=order_data.get("order_type", "market")
            )
            
            return validation
            
        except Exception as e:
            logger.error(f"Capital preservation validation error: {str(e)}")
            return {"approved": False, "error": str(e)}
    
    async def _validate_margin_controls(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate order against margin controls."""
        try:
            validation = await self.margin_controller.validate_margin_order(order_data)
            return validation
            
        except Exception as e:
            logger.error(f"Margin controls validation error: {str(e)}")
            return {"approved": False, "error": str(e)}
    
    async def _validate_daily_limits(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate order against daily limits."""
        try:
            validation = await self.daily_limit_manager.validate_trade_against_daily_limits(order_data)
            return validation
            
        except Exception as e:
            logger.error(f"Daily limits validation error: {str(e)}")
            return {"approved": False, "error": str(e)}
    
    async def _validate_drawdown_protection(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate order against drawdown protection."""
        try:
            # Drawdown protection is more about monitoring than order validation
            # Return a simplified validation
            drawdown_status = await self.drawdown_protector.check_drawdown_limits()
            
            if drawdown_status.get("overall_status") == "breached":
                return {
                    "approved": False,
                    "reason": "Drawdown limits breached - trading restricted"
                }
            
            return {"approved": True, "warnings": drawdown_status.get("warnings", [])}
            
        except Exception as e:
            logger.error(f"Drawdown protection validation error: {str(e)}")
            return {"approved": False, "error": str(e)}
    
    async def _calculate_integrated_risk_metrics(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate integrated risk metrics for the order."""
        try:
            # Get current comprehensive assessment
            assessment = await self.comprehensive_risk_assessment()
            
            # Project order impact on various metrics
            projected_metrics = {
                "overall_risk_score": assessment["risk_score"],
                "capital_impact": "low",  # Would calculate actual impact
                "margin_impact": "low",
                "drawdown_impact": "low",
                "daily_limits_impact": "low"
            }
            
            # Calculate projected risk score
            base_score = assessment["risk_score"]
            
            # Add risk based on order size relative to portfolio
            order_value = (order_data.get("quantity", 0) * 
                          (order_data.get("limit_price", 0) or order_data.get("estimated_price", 0)))
            
            # This would need actual portfolio value to calculate properly
            projected_metrics["order_size_risk"] = "medium" if order_value > 10000 else "low"
            
            return projected_metrics
            
        except Exception as e:
            logger.error(f"Integrated risk metrics calculation error: {str(e)}")
            return {"overall_risk_score": 50}
    
    async def _trigger_auto_response(self, validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger automatic response based on risk level."""
        try:
            if not self.auto_response_enabled:
                return {"triggered": False, "reason": "Auto response disabled"}
            
            risk_score = validation_result.get("risk_metrics", {}).get("overall_risk_score", 0)
            
            if risk_score > 90:
                # Trigger emergency stop
                result = await self.emergency_stop_manager.trigger_emergency_stop(
                    stop_type=EmergencyStopType.SYSTEM_ERROR,
                    reason=f"Auto response: High risk score ({risk_score})",
                    triggered_by="auto_system"
                )
                return {"triggered": True, "action": "emergency_stop", "result": result}
            
            elif risk_score > 80:
                # Apply trading restrictions
                result = await self.daily_limit_manager.force_trading_halt(
                    reason=f"Auto response: Elevated risk ({risk_score})",
                    duration_hours=2
                )
                return {"triggered": True, "action": "trading_halt", "result": result}
            
            return {"triggered": False, "reason": "Risk score below threshold"}
            
        except Exception as e:
            logger.error(f"Auto response trigger error: {str(e)}")
            return {"triggered": False, "error": str(e)}
    
    async def _unified_monitoring_loop(self, interval: int):
        """Unified monitoring loop."""
        while self.monitoring_active:
            try:
                # Run comprehensive assessment
                await self.comprehensive_risk_assessment()
                
                # Check for cross-component correlations
                if self.cross_component_correlation:
                    await self._check_cross_component_correlations()
                
                # Log monitoring cycle
                if self.last_integration_check:
                    cycle_time = (datetime.now() - self.last_integration_check).total_seconds()
                    logger.debug(f"Unified monitoring cycle completed in {cycle_time:.2f} seconds")
                
                self.last_integration_check = datetime.now()
                
                # Wait for next cycle
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Unified monitoring loop error: {str(e)}")
                await asyncio.sleep(interval)  # Continue despite errors
    
    async def _check_cross_component_correlations(self):
        """Check for correlations between different safeguard components."""
        try:
            correlations = []
            
            # Example: High capital utilization + high margin usage
            # This would check actual component states
            
            # Log correlations if significant
            if correlations:
                logger.info(f"Cross-component correlations detected: {correlations}")
            
        except Exception as e:
            logger.error(f"Cross-component correlation check error: {str(e)}")
    
    async def _get_capital_summary(self) -> Dict[str, Any]:
        """Get capital preservation summary."""
        try:
            metrics = await self.capital_manager.calculate_available_capital()
            alerts = await self.capital_manager.monitor_capital_alerts()
            
            return {
                "available_capital": metrics.get("available_for_trading", 0),
                "capital_utilization": metrics.get("capital_utilization", 0),
                "active_alerts": len(alerts),
                "status": "warning" if alerts else "ok"
            }
            
        except Exception as e:
            logger.error(f"Capital summary error: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def _get_margin_summary(self) -> Dict[str, Any]:
        """Get margin controls summary."""
        try:
            status = await self.margin_controller.calculate_margin_status()
            alerts = await self.margin_controller.monitor_margin_alerts()
            
            return {
                "margin_health_score": status.get("margin_health_score", 0),
                "leverage_ratio": status.get("leverage_ratio", 0),
                "margin_available": status.get("margin_available", 0),
                "active_alerts": len(alerts),
                "status": "warning" if alerts else "ok"
            }
            
        except Exception as e:
            logger.error(f"Margin summary error: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def _get_daily_limits_summary(self) -> Dict[str, Any]:
        """Get daily limits summary."""
        try:
            status = await self.daily_limit_manager.check_daily_limits()
            cooldown_status = self.daily_limit_manager.get_cooldown_status()
            
            return {
                "limits_breached": len(status.get("limits_breached", [])),
                "warnings": len(status.get("warnings", [])),
                "trading_halted": status.get("halted", False),
                "in_cooldown": cooldown_status.get("in_cooldown", False),
                "status": "critical" if status.get("halted") else "warning" if status.get("warnings") else "ok"
            }
            
        except Exception as e:
            logger.error(f"Daily limits summary error: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def _get_drawdown_summary(self) -> Dict[str, Any]:
        """Get drawdown protection summary."""
        try:
            metrics = await self.drawdown_protector.calculate_drawdown_metrics()
            status = await self.drawdown_protector.check_drawdown_limits()
            
            return {
                "current_drawdown": metrics.get("drawdown_percentage", 0),
                "max_drawdown_1y": metrics.get("max_drawdowns", {}).get("1_year", 0),
                "protection_level": status.get("protection_level", "normal"),
                "limits_breached": len(status.get("breached_limits", [])),
                "status": status.get("overall_status", "ok")
            }
            
        except Exception as e:
            logger.error(f"Drawdown summary error: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def _get_emergency_summary(self) -> Dict[str, Any]:
        """Get emergency procedures summary."""
        try:
            status = await self.emergency_stop_manager.get_emergency_status()
            
            return {
                "active_emergencies": len(status.get("active_events", [])),
                "global_halt": status.get("system_status", {}).get("global_halt_active", False),
                "procedures_configured": status.get("procedures", {}).get("total_configured", 0),
                "status": "emergency" if status.get("system_status", {}).get("global_halt_active") else "normal"
            }
            
        except Exception as e:
            logger.error(f"Emergency summary error: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def close(self):
        """Cleanup resources."""
        if self.monitoring_active:
            asyncio.create_task(self.disable_unified_monitoring())
        
        # Close component managers
        self.capital_manager.close()
        self.margin_controller.close()
        self.daily_limit_manager.close()
        self.drawdown_protector.close()
        self.emergency_stop_manager.close()
        
        logger.info(f"TradingSafeguardsManager closed for user {self.user_id}")