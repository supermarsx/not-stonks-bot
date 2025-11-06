"""
@file frequency_risk_manager.py
@brief Frequency-based Risk Management Integration

@details
This module provides frequency-aware risk management integration that extends
the core risk management system with trading frequency constraints and controls.
It ensures that trading frequency is properly managed alongside traditional
risk parameters for comprehensive risk control.

Key Features:
- Frequency-based position sizing limits
- Trading frequency risk scoring
- Frequency-aware violation detection
- Integration with existing risk management framework
- Real-time frequency risk monitoring
- Cross-strategy frequency risk management
- Frequency risk reporting and analytics

@author Trading Orchestrator System
@version 1.0
@date 2025-11-06

@warning
Frequency-based risk management should be integrated with the main risk
management system to ensure comprehensive risk control across all dimensions.

@note
This module provides frequency-aware extensions to the core risk management
system and should be used in conjunction with traditional risk controls.
"""

from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
from loguru import logger

from config.trading_frequency import (
    FrequencyManager, FrequencySettings, FrequencyType, FrequencyAlertType,
    FrequencyAlert, get_frequency_manager
)
from .manager import RiskManager, RiskLevel, ViolationType, RiskViolation


class FrequencyViolationType(str, Enum):
    """
    @enum FrequencyViolationType
    @brief Types of frequency-based risk violations
    
    @details
    Enumerates the different types of risk violations that can occur
    due to trading frequency constraints being exceeded or violated.
    """
    FREQUENCY_LIMIT_EXCEEDED = "frequency_limit_exceeded"
    FREQUENCY_COOLDOWN_VIOLATION = "frequency_cooldown_violation"
    FREQUENCY_RISK_THRESHOLD = "frequency_risk_threshold"
    FREQUENCY_POSITION_SIZE_VIOLATION = "frequency_position_size_violation"
    CROSS_STRATEGY_FREQUENCY_RISK = "cross_strategy_frequency_risk"
    TIME_WINDOW_FREQUENCY_VIOLATION = "time_window_frequency_violation"
    MARKET_HOURS_FREQUENCY_VIOLATION = "market_hours_frequency_violation"
    FREQUENCY_VOLATILITY_RISK = "frequency_volatility_risk"


@dataclass
class FrequencyRiskAssessment:
    """
    @class FrequencyRiskAssessment
    @brief Comprehensive frequency risk assessment
    
    @details
    Provides detailed frequency risk assessment including current frequency
    metrics, risk scoring, violation tracking, and recommendations.
    
    @par Risk Components:
    - current_frequency_rate: Current trades per minute
    - frequency_risk_score: Calculated risk score (0.0-1.0)
    - risk_level: Overall risk level classification
    - active_violations: List of active risk violations
    - recommendations: Risk mitigation recommendations
    """
    
    strategy_id: str
    current_frequency_rate: float = 0.0
    frequency_risk_score: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    max_frequency_rate: float = 0.0
    frequency_efficiency: float = 0.0
    
    # Violations
    active_violations: List[FrequencyViolationType] = field(default_factory=list)
    violation_count_today: int = 0
    last_violation_time: Optional[datetime] = None
    
    # Risk metrics
    concentration_risk: float = 0.0
    volatility_risk: float = 0.0
    correlation_risk: float = 0.0
    drawdown_risk: float = 0.0
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    position_size_adjustment: float = 1.0
    frequency_adjustment_factor: float = 1.0
    
    # Assessment details
    assessment_time: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FrequencyRiskLimit:
    """
    @class FrequencyRiskLimit
    @brief Frequency-based risk limits configuration
    
    @details
    Defines risk limits specifically for trading frequency management,
    including hard limits, soft limits, and recommended ranges.
    """
    
    limit_id: str
    strategy_id: str
    limit_type: str  # hard, soft, recommended
    
    # Frequency limits
    max_trades_per_minute: Optional[float] = None
    max_trades_per_hour: Optional[float] = None
    max_trades_per_day: Optional[float] = None
    max_frequency_rate: Optional[float] = None
    
    # Risk limits
    max_frequency_risk_score: float = 1.0
    max_position_size_multiplier: float = 1.0
    max_daily_frequency_loss: Optional[float] = None
    
    # Time constraints
    cooldown_enforcement: bool = True
    market_hours_only: bool = False
    time_window_restrictions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Volatility adjustments
    volatility_threshold: float = 0.0
    volatility_adjustment_factor: float = 0.5
    
    # Status
    is_active: bool = True
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    effective_from: datetime = field(default_factory=datetime.utcnow)
    effective_until: Optional[datetime] = None


class FrequencyRiskManager:
    """
    @class FrequencyRiskManager
    @brief Frequency-based risk management integration
    
    @details
    Integrates frequency-aware risk management with the existing risk
    management framework. Provides comprehensive frequency risk control,
    monitoring, and violation detection.
    
    @par Integration Features:
    - Extends existing RiskManager with frequency controls
    - Frequency-based position sizing and risk limits
    - Real-time frequency risk monitoring and alerts
    - Cross-strategy frequency risk management
    - Frequency risk scoring and assessment
    - Integration with frequency optimization recommendations
    
    @par Risk Management Process:
    1. Evaluate current frequency metrics and risk exposure
    2. Check frequency limits and constraints compliance
    3. Calculate frequency-adjusted risk scores
    4. Generate risk alerts and recommendations
    5. Apply frequency-based position sizing adjustments
    6. Monitor cross-strategy frequency risk correlation
    """
    
    def __init__(self, base_risk_manager: RiskManager):
        """
        Initialize frequency risk manager
        
        Args:
            base_risk_manager: Core risk manager instance
        """
        self.base_manager = base_risk_manager
        self.frequency_manager = get_frequency_manager()
        
        # Risk tracking
        self.frequency_risk_assessments: Dict[str, FrequencyRiskAssessment] = {}
        self.frequency_risk_limits: Dict[str, FrequencyRiskLimit] = {}
        self.frequency_violations: List[RiskViolation] = []
        
        # Risk scoring
        self.frequency_risk_weights = {
            "frequency_rate": 0.3,
            "violation_history": 0.2,
            "concentration": 0.2,
            "volatility": 0.15,
            "correlation": 0.15
        }
        
        # Monitoring
        self.risk_monitoring_enabled = True
        self.risk_assessment_interval = 60  # seconds
        
        logger.info("FrequencyRiskManager initialized with base risk manager integration")
    
    async def assess_frequency_risk(
        self,
        strategy_id: str,
        current_frequency_rate: float = 0.0,
        position_size: Optional[Decimal] = None,
        market_volatility: float = 0.0,
        portfolio_context: Optional[Dict[str, Any]] = None
    ) -> FrequencyRiskAssessment:
        """
        Assess frequency-based risk for a strategy
        
        Args:
            strategy_id: Strategy identifier
            current_frequency_rate: Current trading frequency rate
            position_size: Current position size
            market_volatility: Market volatility measure
            portfolio_context: Portfolio context for correlation analysis
            
        Returns:
            Comprehensive frequency risk assessment
        """
        try:
            # Get current frequency metrics
            frequency_metrics = None
            if self.frequency_manager:
                frequency_metrics = self.frequency_manager.get_frequency_metrics(strategy_id)
            
            # Initialize assessment
            assessment = FrequencyRiskAssessment(strategy_id=strategy_id)
            
            if frequency_metrics:
                assessment.current_frequency_rate = frequency_metrics.current_frequency_rate
                assessment.frequency_efficiency = frequency_metrics.frequency_efficiency
            
            # Get risk limits for strategy
            risk_limits = self._get_strategy_risk_limits(strategy_id)
            
            # Calculate frequency risk score
            assessment.frequency_risk_score = await self._calculate_frequency_risk_score(
                strategy_id, current_frequency_rate, risk_limits, market_volatility
            )
            
            # Determine risk level
            assessment.risk_level = self._determine_risk_level(assessment.frequency_risk_score)
            
            # Check for active violations
            violations = await self._check_frequency_violations(
                strategy_id, current_frequency_rate, risk_limits
            )
            assessment.active_violations = violations
            assessment.violation_count_today = self._get_daily_violation_count(strategy_id)
            
            # Calculate risk metrics
            assessment.concentration_risk = await self._calculate_concentration_risk(
                strategy_id, position_size
            )
            assessment.volatility_risk = await self._calculate_volatility_risk(
                strategy_id, market_volatility
            )
            assessment.correlation_risk = await self._calculate_correlation_risk(
                strategy_id, portfolio_context
            )
            
            # Generate recommendations
            assessment.recommendations = await self._generate_risk_recommendations(
                assessment, risk_limits
            )
            
            # Calculate adjustments
            assessment.position_size_adjustment = await self._calculate_position_size_adjustment(
                assessment, risk_limits
            )
            assessment.frequency_adjustment_factor = await self._calculate_frequency_adjustment(
                assessment, risk_limits
            )
            
            # Store assessment
            self.frequency_risk_assessments[strategy_id] = assessment
            
            logger.debug(f"Frequency risk assessment completed for {strategy_id}: "
                        f"score={assessment.frequency_risk_score:.3f}, "
                        f"level={assessment.risk_level.value}")
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error assessing frequency risk for {strategy_id}: {e}")
            # Return default low-risk assessment
            return FrequencyRiskAssessment(strategy_id=strategy_id)
    
    async def check_frequency_risk_compliance(
        self,
        strategy_id: str,
        proposed_trade: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Check if proposed trade complies with frequency risk limits
        
        Args:
            strategy_id: Strategy identifier
            proposed_trade: Proposed trade details
            
        Returns:
            Tuple of (compliance_status, violation_messages)
        """
        violations = []
        
        try:
            # Get current frequency metrics
            frequency_metrics = None
            if self.frequency_manager:
                frequency_metrics = self.frequency_manager.get_frequency_metrics(strategy_id)
            
            current_rate = frequency_metrics.current_frequency_rate if frequency_metrics else 0.0
            
            # Get risk limits
            risk_limits = self._get_strategy_risk_limits(strategy_id)
            
            # Check frequency rate limits
            if risk_limits.max_frequency_rate and current_rate >= risk_limits.max_frequency_rate:
                violations.append(
                    f"Frequency rate {current_rate:.2f} exceeds maximum "
                    f"{risk_limits.max_frequency_rate:.2f}"
                )
            
            # Check cooldown compliance
            if risk_limits.cooldown_enforcement and frequency_metrics and frequency_metrics.in_cooldown:
                violations.append("Trade blocked due to active cooldown period")
            
            # Check time window restrictions
            if risk_limits.time_window_restrictions:
                window_violations = await self._check_time_window_compliance(
                    strategy_id, risk_limits.time_window_restrictions
                )
                violations.extend(window_violations)
            
            # Check market hours restriction
            if risk_limits.market_hours_only and not await self._is_market_hours():
                violations.append("Trading restricted to market hours only")
            
            # Check volatility-based adjustments
            if risk_limits.volatility_threshold > 0 and proposed_trade.get('market_volatility', 0) > risk_limits.volatility_threshold:
                violations.append(f"Market volatility {proposed_trade.get('market_volatility', 0):.3f} exceeds threshold {risk_limits.volatility_threshold:.3f}")
            
            compliance_status = len(violations) == 0
            
            if not compliance_status:
                logger.warning(f"Frequency risk violations for {strategy_id}: {violations}")
                
                # Generate risk violation
                violation = RiskViolation(
                    timestamp=datetime.utcnow(),
                    violation_type=frequency_violation_type_to_risk_violation(violations[0]),
                    description="; ".join(violations),
                    current_value=current_rate,
                    limit_value=risk_limits.max_frequency_rate or float('inf'),
                    severity=self._get_violation_severity(violations),
                    position_details={
                        "strategy_id": strategy_id,
                        "trade_details": proposed_trade,
                        "violation_count": len(violations)
                    }
                )
                
                self.frequency_violations.append(violation)
                
                # Alert if enabled
                if self.frequency_manager:
                    await self.frequency_manager.generate_alert(
                        strategy_id=strategy_id,
                        alert_type=FrequencyAlertType.RISK_LIMIT_REACHED,
                        message="Frequency risk limit violations detected",
                        severity="high" if len(violations) > 1 else "medium",
                        current_value=current_rate,
                        threshold_value=risk_limits.max_frequency_rate
                    )
            
            return compliance_status, violations
            
        except Exception as e:
            logger.error(f"Error checking frequency risk compliance for {strategy_id}: {e}")
            return False, [f"Error checking compliance: {str(e)}"]
    
    async def calculate_frequency_adjusted_position_size(
        self,
        strategy_id: str,
        base_position_size: Decimal,
        current_frequency_rate: float = 0.0,
        market_volatility: float = 0.0
    ) -> Decimal:
        """
        Calculate frequency-adjusted position size
        
        Args:
            strategy_id: Strategy identifier
            base_position_size: Base position size
            current_frequency_rate: Current frequency rate
            market_volatility: Market volatility
            
        Returns:
            Frequency-adjusted position size
        """
        try:
            # Get frequency risk assessment
            assessment = await self.assess_frequency_risk(
                strategy_id, current_frequency_rate, base_position_size, market_volatility
            )
            
            # Apply position size adjustments
            adjusted_size = base_position_size * Decimal(str(assessment.position_size_adjustment))
            
            # Get frequency manager position size if available
            if self.frequency_manager:
                freq_adjusted_size = await self.frequency_manager.calculate_position_size(
                    strategy_id, base_position_size, current_frequency_rate, market_volatility
                )
                # Take the more conservative (smaller) of the two adjustments
                adjusted_size = min(adjusted_size, freq_adjusted_size)
            
            # Apply risk limits
            risk_limits = self._get_strategy_risk_limits(strategy_id)
            if risk_limits.max_position_size_multiplier < 1.0:
                max_size = base_position_size * Decimal(str(risk_limits.max_position_size_multiplier))
                adjusted_size = min(adjusted_size, max_size)
            
            # Ensure minimum size
            min_size = base_position_size * Decimal("0.01")  # 1% minimum
            adjusted_size = max(adjusted_size, min_size)
            
            # Round to appropriate precision
            adjusted_size = adjusted_size.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            
            logger.debug(
                f"Frequency-adjusted position size for {strategy_id}: "
                f"base={base_position_size}, adjusted={adjusted_size}, "
                f"adjustment_factor={assessment.position_size_adjustment:.3f}"
            )
            
            return adjusted_size
            
        except Exception as e:
            logger.error(f"Error calculating frequency-adjusted position size for {strategy_id}: {e}")
            return base_position_size
    
    async def monitor_cross_strategy_frequency_risk(self) -> Dict[str, Any]:
        """
        Monitor frequency risk across all strategies
        
        Returns:
            Cross-strategy frequency risk analysis
        """
        try:
            # Get all strategy assessments
            all_assessments = list(self.frequency_risk_assessments.values())
            
            if not all_assessments:
                return {"status": "no_data", "message": "No frequency risk assessments available"}
            
            # Calculate portfolio-level metrics
            portfolio_frequency_rate = sum(a.current_frequency_rate for a in all_assessments) / len(all_assessments)
            portfolio_risk_score = sum(a.frequency_risk_score for a in all_assessments) / len(all_assessments)
            high_risk_strategies = len([a for a in all_assessments if a.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]])
            
            # Check for concentration risk
            concentration_risk = await self._assess_portfolio_concentration_risk(all_assessments)
            
            # Correlation analysis
            correlation_risk = await self._assess_frequency_correlation_risk(all_assessments)
            
            # Generate portfolio-level recommendations
            recommendations = await self._generate_portfolio_recommendations(
                all_assessments, concentration_risk, correlation_risk
            )
            
            # Risk summary
            risk_summary = {
                "portfolio_frequency_rate": portfolio_frequency_rate,
                "portfolio_risk_score": portfolio_risk_score,
                "high_risk_strategies_count": high_risk_strategies,
                "total_strategies": len(all_assessments),
                "concentration_risk": concentration_risk,
                "correlation_risk": correlation_risk,
                "recommendations": recommendations,
                "assessment_time": datetime.utcnow().isoformat()
            }
            
            # Strategy details
            strategy_details = {
                strategy_id: {
                    "frequency_rate": assessment.current_frequency_rate,
                    "risk_score": assessment.frequency_risk_score,
                    "risk_level": assessment.risk_level.value,
                    "active_violations": [v.value for v in assessment.active_violations],
                    "position_size_adjustment": assessment.position_size_adjustment,
                    "recommendations": assessment.recommendations
                }
                for strategy_id, assessment in self.frequency_risk_assessments.items()
            }
            
            return {
                "risk_summary": risk_summary,
                "strategy_details": strategy_details,
                "monitoring_status": "active"
            }
            
        except Exception as e:
            logger.error(f"Error monitoring cross-strategy frequency risk: {e}")
            return {"status": "error", "message": str(e)}
    
    def _get_strategy_risk_limits(self, strategy_id: str) -> FrequencyRiskLimit:
        """Get frequency risk limits for a strategy"""
        return self.frequency_risk_limits.get(strategy_id) or FrequencyRiskLimit(
            limit_id=f"default_{strategy_id}",
            strategy_id=strategy_id,
            limit_type="recommended",
            max_frequency_rate=10.0,  # 10 trades per minute default
            max_frequency_risk_score=0.7,
            is_active=True
        )
    
    async def _calculate_frequency_risk_score(
        self,
        strategy_id: str,
        current_rate: float,
        risk_limits: FrequencyRiskLimit,
        market_volatility: float
    ) -> float:
        """Calculate frequency risk score (0.0-1.0)"""
        try:
            score = 0.0
            
            # Frequency rate component
            if risk_limits.max_frequency_rate:
                frequency_component = min(current_rate / risk_limits.max_frequency_rate, 1.0)
                score += self.frequency_risk_weights["frequency_rate"] * frequency_component
            
            # Violation history component
            violation_count = self._get_daily_violation_count(strategy_id)
            violation_component = min(violation_count / 10.0, 1.0)  # Normalize to 10 violations
            score += self.frequency_risk_weights["violation_history"] * violation_component
            
            # Volatility component
            if risk_limits.volatility_threshold > 0 and market_volatility > risk_limits.volatility_threshold:
                volatility_component = min(market_volatility / (risk_limits.volatility_threshold * 2), 1.0)
                score += self.frequency_risk_weights["volatility"] * volatility_component
            
            # Cooldown component (if in cooldown, high risk)
            if self.frequency_manager:
                metrics = self.frequency_manager.get_frequency_metrics(strategy_id)
                if metrics and metrics.in_cooldown:
                    score += self.frequency_risk_weights["frequency_rate"] * 0.5
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating frequency risk score: {e}")
            return 0.5  # Default medium risk
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from risk score"""
        if risk_score >= 0.8:
            return RiskLevel.CRITICAL
        elif risk_score >= 0.6:
            return RiskLevel.HIGH
        elif risk_score >= 0.4:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    async def _check_frequency_violations(
        self,
        strategy_id: str,
        current_rate: float,
        risk_limits: FrequencyRiskLimit
    ) -> List[FrequencyViolationType]:
        """Check for frequency violations"""
        violations = []
        
        # Check frequency rate violations
        if risk_limits.max_frequency_rate and current_rate >= risk_limits.max_frequency_rate:
            violations.append(FrequencyViolationType.FREQUENCY_LIMIT_EXCEEDED)
        
        # Check risk score violations
        assessment = self.frequency_risk_assessments.get(strategy_id)
        if assessment and assessment.frequency_risk_score >= risk_limits.max_frequency_risk_score:
            violations.append(FrequencyViolationType.FREQUENCY_RISK_THRESHOLD)
        
        return violations
    
    def _get_daily_violation_count(self, strategy_id: str) -> int:
        """Get count of violations for strategy today"""
        today = datetime.utcnow().date()
        return len([
            v for v in self.frequency_violations
            if v.position_details and 
            v.position_details.get("strategy_id") == strategy_id and
            v.timestamp.date() == today
        ])
    
    async def _calculate_concentration_risk(
        self,
        strategy_id: str,
        position_size: Optional[Decimal]
    ) -> float:
        """Calculate concentration risk (0.0-1.0)"""
        # This would calculate based on portfolio allocation
        # For now, return a simple calculation
        return 0.1 if position_size and position_size > 10000 else 0.05
    
    async def _calculate_volatility_risk(
        self,
        strategy_id: str,
        market_volatility: float
    ) -> float:
        """Calculate volatility risk (0.0-1.0)"""
        return min(market_volatility / 0.5, 1.0)  # Normalize to 50% volatility
    
    async def _calculate_correlation_risk(
        self,
        strategy_id: str,
        portfolio_context: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate correlation risk (0.0-1.0)"""
        # This would calculate based on strategy correlations
        # For now, return default value
        return 0.2
    
    async def _generate_risk_recommendations(
        self,
        assessment: FrequencyRiskAssessment,
        risk_limits: FrequencyRiskLimit
    ) -> List[str]:
        """Generate risk mitigation recommendations"""
        recommendations = []
        
        if assessment.frequency_risk_score > 0.7:
            recommendations.append("Reduce trading frequency to lower risk exposure")
            recommendations.append("Consider increasing cooldown periods")
        
        if assessment.concentration_risk > 0.3:
            recommendations.append("Reduce position size to lower concentration risk")
        
        if assessment.volatility_risk > 0.5:
            recommendations.append("Adjust position sizing for high volatility environment")
        
        if assessment.active_violations:
            recommendations.append("Address active frequency violations immediately")
        
        return recommendations
    
    async def _calculate_position_size_adjustment(
        self,
        assessment: FrequencyRiskAssessment,
        risk_limits: FrequencyRiskLimit
    ) -> float:
        """Calculate position size adjustment factor"""
        # Base adjustment from risk score
        adjustment = 1.0 - (assessment.frequency_risk_score * 0.5)  # Reduce up to 50%
        
        # Apply volatility adjustment
        if assessment.volatility_risk > 0.3:
            adjustment *= (1.0 - assessment.volatility_risk * 0.3)
        
        return max(adjustment, 0.1)  # Minimum 10% of original size
    
    async def _calculate_frequency_adjustment(
        self,
        assessment: FrequencyRiskAssessment,
        risk_limits: FrequencyRiskLimit
    ) -> float:
        """Calculate frequency adjustment factor"""
        if assessment.frequency_risk_score > 0.8:
            return 0.5  # Reduce frequency by 50%
        elif assessment.frequency_risk_score > 0.6:
            return 0.7  # Reduce frequency by 30%
        elif assessment.frequency_risk_score > 0.4:
            return 0.85  # Reduce frequency by 15%
        else:
            return 1.0  # No adjustment needed
    
    async def _check_time_window_compliance(
        self,
        strategy_id: str,
        time_windows: List[Dict[str, Any]]
    ) -> List[str]:
        """Check time window compliance"""
        # This would check against configured time windows
        # For now, return empty list (compliant)
        return []
    
    async def _is_market_hours(self) -> bool:
        """Check if current time is within market hours"""
        # Simplified - assume 24/7 markets for now
        return True
    
    def _get_violation_severity(self, violations: List[str]) -> RiskLevel:
        """Get violation severity based on violations"""
        if len(violations) > 2:
            return RiskLevel.CRITICAL
        elif len(violations) > 1:
            return RiskLevel.HIGH
        else:
            return RiskLevel.MEDIUM
    
    async def _assess_portfolio_concentration_risk(self, assessments: List[FrequencyRiskAssessment]) -> float:
        """Assess portfolio concentration risk"""
        # This would calculate portfolio concentration based on all strategies
        return sum(a.concentration_risk for a in assessments) / len(assessments) if assessments else 0.0
    
    async def _assess_frequency_correlation_risk(self, assessments: List[FrequencyRiskAssessment]) -> float:
        """Assess frequency correlation risk across strategies"""
        # This would calculate correlation between strategy frequencies
        return 0.15  # Placeholder value
    
    async def _generate_portfolio_recommendations(
        self,
        assessments: List[FrequencyRiskAssessment],
        concentration_risk: float,
        correlation_risk: float
    ) -> List[str]:
        """Generate portfolio-level recommendations"""
        recommendations = []
        
        high_risk_count = len([a for a in assessments if a.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]])
        
        if high_risk_count > len(assessments) * 0.5:
            recommendations.append("Consider reducing frequency across multiple strategies")
        
        if concentration_risk > 0.3:
            recommendations.append("Rebalance portfolio to reduce concentration")
        
        if correlation_risk > 0.5:
            recommendations.append("Reduce correlation between strategy frequencies")
        
        return recommendations
    
    def add_risk_limit(self, risk_limit: FrequencyRiskLimit):
        """Add frequency risk limit configuration"""
        self.frequency_risk_limits[risk_limit.strategy_id] = risk_limit
        logger.info(f"Frequency risk limit added for strategy {risk_limit.strategy_id}")
    
    def get_risk_assessment(self, strategy_id: str) -> Optional[FrequencyRiskAssessment]:
        """Get frequency risk assessment for a strategy"""
        return self.frequency_risk_assessments.get(strategy_id)
    
    def get_all_risk_assessments(self) -> Dict[str, FrequencyRiskAssessment]:
        """Get all frequency risk assessments"""
        return self.frequency_risk_assessments.copy()
    
    def get_frequency_violations(
        self,
        strategy_id: Optional[str] = None,
        since: Optional[datetime] = None
    ) -> List[RiskViolation]:
        """Get frequency violations"""
        violations = self.frequency_violations
        
        if strategy_id:
            violations = [
                v for v in violations
                if v.position_details and v.position_details.get("strategy_id") == strategy_id
            ]
        
        if since:
            violations = [v for v in violations if v.timestamp >= since]
        
        return violations
    
    async def get_frequency_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive frequency risk summary"""
        try:
            assessments = list(self.frequency_risk_assessments.values())
            
            if not assessments:
                return {"status": "no_data", "message": "No frequency risk assessments available"}
            
            # Summary statistics
            total_strategies = len(assessments)
            high_risk_strategies = len([a for a in assessments if a.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]])
            average_risk_score = sum(a.frequency_risk_score for a in assessments) / total_strategies
            total_violations = len(self.frequency_violations)
            
            # Risk distribution
            risk_distribution = {
                "low": len([a for a in assessments if a.risk_level == RiskLevel.LOW]),
                "medium": len([a for a in assessments if a.risk_level == RiskLevel.MEDIUM]),
                "high": len([a for a in assessments if a.risk_level == RiskLevel.HIGH]),
                "critical": len([a for a in assessments if a.risk_level == RiskLevel.CRITICAL])
            }
            
            # Active issues
            active_violations = len([v for v in self.frequency_violations if not hasattr(v, 'resolved') or not v.resolved])
            strategies_with_violations = len(set(
                v.position_details.get("strategy_id") for v in self.frequency_violations
                if v.position_details and v.position_details.get("strategy_id")
            ))
            
            return {
                "summary": {
                    "total_strategies": total_strategies,
                    "high_risk_strategies": high_risk_strategies,
                    "average_risk_score": average_risk_score,
                    "total_violations": total_violations,
                    "active_violations": active_violations,
                    "strategies_with_violations": strategies_with_violations
                },
                "risk_distribution": risk_distribution,
                "risk_limits_configured": len(self.frequency_risk_limits),
                "monitoring_status": "active" if self.risk_monitoring_enabled else "inactive",
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating frequency risk summary: {e}")
            return {"status": "error", "message": str(e)}


def frequency_violation_type_to_risk_violation(violation_msg: str) -> ViolationType:
    """Convert frequency violation message to standard violation type"""
    if "frequency" in violation_msg.lower() and "limit" in violation_msg.lower():
        return ViolationType.POSITION_SIZE_EXCEEDED  # Use as fallback
    return ViolationType.POSITION_SIZE_EXCEEDED


# Global frequency risk manager instance
_frequency_risk_manager: Optional[FrequencyRiskManager] = None


def get_frequency_risk_manager() -> Optional[FrequencyRiskManager]:
    """Get global frequency risk manager instance"""
    return _frequency_risk_manager


def initialize_frequency_risk_manager(base_risk_manager: RiskManager) -> FrequencyRiskManager:
    """
    Initialize global frequency risk manager
    
    Args:
        base_risk_manager: Core risk manager instance
        
    Returns:
        Initialized frequency risk manager
    """
    global _frequency_risk_manager
    _frequency_risk_manager = FrequencyRiskManager(base_risk_manager)
    return _frequency_risk_manager


async def assess_frequency_risk(
    strategy_id: str,
    current_frequency_rate: float = 0.0,
    position_size: Optional[Decimal] = None,
    market_volatility: float = 0.0
) -> FrequencyRiskAssessment:
    """
    Convenience function to assess frequency risk
    
    Args:
        strategy_id: Strategy identifier
        current_frequency_rate: Current frequency rate
        position_size: Current position size
        market_volatility: Market volatility
        
    Returns:
        Frequency risk assessment
    """
    manager = get_frequency_risk_manager()
    if manager:
        return await manager.assess_frequency_risk(
            strategy_id, current_frequency_rate, position_size, market_volatility
        )
    return FrequencyRiskAssessment(strategy_id=strategy_id)


async def check_frequency_risk_compliance(
    strategy_id: str,
    proposed_trade: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """
    Convenience function to check frequency risk compliance
    
    Args:
        strategy_id: Strategy identifier
        proposed_trade: Proposed trade details
        
    Returns:
        Compliance status and violation messages
    """
    manager = get_frequency_risk_manager()
    if manager:
        return await manager.check_frequency_risk_compliance(strategy_id, proposed_trade)
    return True, []


# Example usage and testing
if __name__ == "__main__":
    async def test_frequency_risk_manager():
        """Test frequency risk manager functionality"""
        
        # This would typically use the actual RiskManager
        # For testing, we'll create a mock
        class MockRiskManager:
            def __init__(self):
                pass
        
        base_manager = MockRiskManager()
        
        # Initialize frequency risk manager
        freq_risk_manager = initialize_frequency_risk_manager(base_manager)
        
        # Test risk assessment
        assessment = await freq_risk_manager.assess_frequency_risk(
            strategy_id="test_strategy",
            current_frequency_rate=2.5,
            position_size=Decimal("5000"),
            market_volatility=0.2
        )
        print(f"Risk assessment: {assessment.frequency_risk_score:.3f}, level: {assessment.risk_level.value}")
        
        # Test compliance check
        compliance, violations = await freq_risk_manager.check_frequency_risk_compliance(
            strategy_id="test_strategy",
            proposed_trade={
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 100,
                "market_volatility": 0.15
            }
        )
        print(f"Compliance: {compliance}, violations: {violations}")
        
        # Test position size calculation
        adjusted_size = await freq_risk_manager.calculate_frequency_adjusted_position_size(
            strategy_id="test_strategy",
            base_position_size=Decimal("10000"),
            current_frequency_rate=1.8,
            market_volatility=0.25
        )
        print(f"Adjusted position size: {adjusted_size}")
        
        # Test monitoring
        monitoring_result = await freq_risk_manager.monitor_cross_strategy_frequency_risk()
        print(f"Monitoring result: {monitoring_result.get('status')}")
        
        # Test risk summary
        summary = await freq_risk_manager.get_frequency_risk_summary()
        print(f"Risk summary: {summary}")
    
    # Run tests
    asyncio.run(test_frequency_risk_manager())