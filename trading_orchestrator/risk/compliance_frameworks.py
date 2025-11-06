"""
Compliance and Regulation Framework Module

Provides institutional-grade compliance monitoring and reporting for:
- Basel III banking regulations
- MiFID II investment firm regulations  
- Dodd-Frank financial reforms
- Regional regulation support (US, EU, Asia)

Integrates with the advanced risk management system for real-time
regulatory limit monitoring and automated compliance reporting.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import json
import warnings
from collections import defaultdict
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegulationFramework(Enum):
    """Supported regulatory frameworks"""
    BASEL_III = "basel_iii"
    MIFID_II = "mifid_ii"
    DODD_FRANK = "dodd_frank"
    SEC = "sec"
    ESMA = "esma"
    FCA = "fca"


class ComplianceStatus(Enum):
    """Compliance status indicators"""
    COMPLIANT = "compliant"
    WARNING = "warning"
    BREACH = "breach"
    CRITICAL = "critical"


@dataclass
class RegulatoryLimit:
    """Regulatory limit definition"""
    framework: RegulationFramework
    regulation: str
    metric_name: str
    threshold_value: float
    threshold_type: str  # "maximum", "minimum", "percentage"
    calculation_method: str
    reporting_frequency: str
    jurisdiction: str
    effective_date: datetime
    expiry_date: Optional[datetime] = None


@dataclass
class ComplianceEvent:
    """Compliance event record"""
    timestamp: datetime
    framework: RegulationFramework
    regulation: str
    event_type: str
    status: ComplianceStatus
    metric_value: float
    threshold_value: float
    breach_details: str
    action_required: str
    resolved: bool = False


class ComplianceFrameworks:
    """
    Comprehensive compliance monitoring and reporting system
    
    Provides real-time monitoring of regulatory limits and automated
    compliance reporting for major financial regulations.
    """
    
    def __init__(self):
        """Initialize compliance frameworks"""
        self.logger = logging.getLogger(__name__)
        
        # Regulatory limits registry
        self.regulatory_limits: Dict[str, RegulatoryLimit] = {}
        
        # Compliance events log
        self.compliance_events: List[ComplianceEvent] = []
        
        # Real-time monitoring flags
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Framework-specific calculators
        self.basel_calculator = BaselIIICalculator()
        self.mifid_calculator = MiFIDIICalculator()
        self.dodd_frank_calculator = DoddFrankCalculator()
        
        # Initialize regulatory limits
        self._initialize_regulatory_limits()
        
        self.logger.info("Compliance Frameworks initialized")
    
    def _initialize_regulatory_limits(self):
        """Initialize all regulatory limits"""
        
        # Basel III Capital Requirements
        basel_iii_limits = [
            RegulatoryLimit(
                framework=RegulationFramework.BASEL_III,
                regulation="CRR_4_1",
                metric_name="Tier_1_Capital_Ratio",
                threshold_value=0.06,
                threshold_type="minimum",
                calculation_method="tier_1_capital / risk_weighted_assets",
                reporting_frequency="daily",
                jurisdiction="EU",
                effective_date=datetime(2014, 1, 1)
            ),
            RegulatoryLimit(
                framework=RegulationFramework.BASEL_III,
                regulation="CRR_4_1",
                metric_name="Total_Capital_Ratio",
                threshold_value=0.08,
                threshold_type="minimum",
                calculation_method="total_capital / risk_weighted_assets",
                reporting_frequency="daily",
                jurisdiction="EU",
                effective_date=datetime(2014, 1, 1)
            ),
            RegulatoryLimit(
                framework=RegulationFramework.BASEL_III,
                regulation="CRR_4_1",
                metric_name="Leverage_Ratio",
                threshold_value=0.03,
                threshold_type="minimum",
                calculation_method="tier_1_capital / total_exposures",
                reporting_frequency="monthly",
                jurisdiction="EU",
                effective_date=datetime(2018, 1, 1)
            ),
            RegulatoryLimit(
                framework=RegulationFramework.BASEL_III,
                regulation="CRR_4_2",
                metric_name="Liquidity_Coverage_Ratio",
                threshold_value=1.0,
                threshold_type="minimum",
                calculation_method="high_quality_liquid_assets / net_cash_outflows",
                reporting_frequency="daily",
                jurisdiction="EU",
                effective_date=datetime(2018, 1, 1)
            ),
            RegulatoryLimit(
                framework=RegulationFramework.BASEL_III,
                regulation="CRR_4_2",
                metric_name="Net_Stable_Funding_Ratio",
                threshold_value=1.0,
                threshold_type="minimum",
                calculation_method="available_stable_funding / required_stable_funding",
                reporting_frequency="monthly",
                jurisdiction="EU",
                effective_date=datetime(2018, 1, 1)
            )
        ]
        
        # MiFID II Requirements
        mifid_ii_limits = [
            RegulatoryLimit(
                framework=RegulationFramework.MIFID_II,
                regulation="Article_25",
                metric_name="Best_Execution_Price_Impact",
                threshold_value=0.02,
                threshold_type="maximum",
                calculation_method="execution_price_deviation / market_price",
                reporting_frequency="real-time",
                jurisdiction="EU",
                effective_date=datetime(2018, 1, 3)
            ),
            RegulatoryLimit(
                framework=RegulationFramework.MIFID_II,
                regulation="Article_23",
                metric_name="Order_Processing_Delay",
                threshold_value=100,  # milliseconds
                threshold_type="maximum",
                calculation_method="order_processing_time_ms",
                reporting_frequency="real-time",
                jurisdiction="EU",
                effective_date=datetime(2018, 1, 3)
            ),
            RegulatoryLimit(
                framework=RegulationFramework.MIFID_II,
                regulation="Article_50",
                metric_name="Transaction_Reporting_Delay",
                threshold_value=300,  # seconds
                threshold_type="maximum",
                calculation_method="time_to_report_transaction",
                reporting_frequency="real-time",
                jurisdiction="EU",
                effective_date=datetime(2018, 1, 3)
            ),
            RegulatoryLimit(
                framework=RegulationFramework.MIFID_II,
                regulation="Article_16",
                metric_name="Investment_Advice_Conflict_Check",
                threshold_value=1.0,  # No conflicts allowed
                threshold_type="maximum",
                calculation_method="conflict_of_interest_score",
                reporting_frequency="real-time",
                jurisdiction="EU",
                effective_date=datetime(2018, 1, 3)
            )
        ]
        
        # Dodd-Frank Requirements
        dodd_frank_limits = [
            RegulatoryLimit(
                framework=RegulationFramework.DODD_FRANK,
                regulation="Section_716",
                metric_name="Swap_Dealer_Capital_Ratio",
                threshold_value=0.08,
                threshold_type="minimum",
                calculation_method="net_capital / net_aggregate_swap_positions",
                reporting_frequency="daily",
                jurisdiction="US",
                effective_date=datetime(2013, 7, 22)
            ),
            RegulatoryLimit(
                framework=RegulationFramework.DODD_FRANK,
                regulation="Section_763",
                metric_name="Position_Limit_Reporting_Threshold",
                threshold_value=25,  # Notional amount in billions
                threshold_type="maximum",
                calculation_method="aggregate_position_notional_usd",
                reporting_frequency="daily",
                jurisdiction="US",
                effective_date=datetime(2014, 4, 17)
            ),
            RegulatoryLimit(
                framework=RegulationFramework.DODD_FRANK,
                regulation="Section_730",
                metric_name="Large_Trader_Reporting_Threshold",
                threshold_value=0.05,  # 5% of trading volume
                threshold_type="maximum",
                calculation_method="trading_volume / market_volume",
                reporting_frequency="daily",
                jurisdiction="US",
                effective_date=datetime(2011, 7, 21)
            ),
            RegulatoryLimit(
                framework=RegulationFramework.DODD_FRANK,
                regulation="Section_756",
                metric_name="Systemically_Important_Financial_Market_Utility_Limit",
                threshold_value=0.10,  # 10% market share limit
                threshold_type="maximum",
                calculation_method="daily_transaction_volume / total_market_volume",
                reporting_frequency="daily",
                jurisdiction="US",
                effective_date=datetime(2012, 7, 18)
            )
        ]
        
        # Store all limits
        all_limits = basel_iii_limits + mifid_ii_limits + dodd_frank_limits
        for limit in all_limits:
            key = f"{limit.framework.value}_{limit.regulation}_{limit.metric_name}"
            self.regulatory_limits[key] = limit
        
        self.logger.info(f"Initialized {len(all_limits)} regulatory limits")
    
    def start_real_time_monitoring(self):
        """Start real-time compliance monitoring"""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Real-time compliance monitoring started")
    
    def stop_real_time_monitoring(self):
        """Stop real-time compliance monitoring"""
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join()
        
        self.logger.info("Real-time compliance monitoring stopped")
    
    def _monitoring_loop(self):
        """Real-time monitoring loop"""
        while self.monitoring_active:
            try:
                # Monitor each framework
                self._check_basel_iii_compliance()
                self._check_mifid_ii_compliance()
                self._check_dodd_frank_compliance()
                
                # Sleep for monitoring interval
                time.sleep(1)  # 1 second monitoring interval
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Wait before retrying
    
    def _check_basel_iii_compliance(self):
        """Check Basel III compliance"""
        try:
            # Get portfolio metrics
            metrics = self.basel_calculator.get_portfolio_metrics()
            
            # Check each Basel III limit
            for key, limit in self.regulatory_limits.items():
                if limit.framework == RegulationFramework.BASEL_III:
                    current_value = self._get_current_metric_value(limit, metrics)
                    if current_value is not None:
                        self._evaluate_limit(limit, current_value)
                        
        except Exception as e:
            self.logger.error(f"Error checking Basel III compliance: {e}")
    
    def _check_mifid_ii_compliance(self):
        """Check MiFID II compliance"""
        try:
            # Get execution metrics
            metrics = self.mifid_calculator.get_execution_metrics()
            
            # Check each MiFID II limit
            for key, limit in self.regulatory_limits.items():
                if limit.framework == RegulationFramework.MIFID_II:
                    current_value = self._get_current_metric_value(limit, metrics)
                    if current_value is not None:
                        self._evaluate_limit(limit, current_value)
                        
        except Exception as e:
            self.logger.error(f"Error checking MiFID II compliance: {e}")
    
    def _check_dodd_frank_compliance(self):
        """Check Dodd-Frank compliance"""
        try:
            # Get derivatives metrics
            metrics = self.dodd_frank_calculator.get_derivatives_metrics()
            
            # Check each Dodd-Frank limit
            for key, limit in self.regulatory_limits.items():
                if limit.framework == RegulationFramework.DODD_FRANK:
                    current_value = self._get_current_metric_value(limit, metrics)
                    if current_value is not None:
                        self._evaluate_limit(limit, current_value)
                        
        except Exception as e:
            self.logger.error(f"Error checking Dodd-Frank compliance: {e}")
    
    def _get_current_metric_value(self, limit: RegulatoryLimit, metrics: Dict) -> Optional[float]:
        """Get current value for a specific metric"""
        # Map metric names to actual metric values
        metric_mapping = {
            "Tier_1_Capital_Ratio": "tier_1_ratio",
            "Total_Capital_Ratio": "total_capital_ratio",
            "Leverage_Ratio": "leverage_ratio",
            "Liquidity_Coverage_Ratio": "lcr",
            "Net_Stable_Funding_Ratio": "nsfr",
            "Best_Execution_Price_Impact": "price_impact",
            "Order_Processing_Delay": "processing_delay_ms",
            "Transaction_Reporting_Delay": "reporting_delay_seconds",
            "Investment_Advice_Conflict_Check": "conflict_score",
            "Swap_Dealer_Capital_Ratio": "swap_dealer_ratio",
            "Position_Limit_Reporting_Threshold": "position_notional",
            "Large_Trader_Reporting_Threshold": "trader_volume_ratio",
            "Systemically_Important_Financial_Market_Utility_Limit": "market_share"
        }
        
        actual_name = metric_mapping.get(limit.metric_name)
        if actual_name and actual_name in metrics:
            return metrics[actual_name]
        
        return None
    
    def _evaluate_limit(self, limit: RegulatoryLimit, current_value: float) -> ComplianceEvent:
        """Evaluate if current value breaches regulatory limit"""
        try:
            # Determine compliance status
            status = ComplianceStatus.COMPLIANT
            breach_details = ""
            action_required = "No action required"
            
            if limit.threshold_type == "maximum":
                if current_value > limit.threshold_value:
                    status = ComplianceStatus.BREACH
                    breach_details = f"Current value {current_value:.4f} exceeds maximum {limit.threshold_value:.4f}"
                    action_required = "Immediate action required to reduce exposure"
                elif current_value > 0.8 * limit.threshold_value:  # 80% warning threshold
                    status = ComplianceStatus.WARNING
                    breach_details = f"Current value {current_value:.4f} approaches maximum {limit.threshold_value:.4f}"
                    action_required = "Monitor closely and prepare mitigation strategies"
            
            elif limit.threshold_type == "minimum":
                if current_value < limit.threshold_value:
                    status = ComplianceStatus.BREACH
                    breach_details = f"Current value {current_value:.4f} below minimum {limit.threshold_value:.4f}"
                    action_required = "Immediate action required to increase exposure/capital"
                elif current_value < 1.2 * limit.threshold_value:  # 120% warning threshold
                    status = ComplianceStatus.WARNING
                    breach_details = f"Current value {current_value:.4f} approaches minimum {limit.threshold_value:.4f}"
                    action_required = "Monitor closely and prepare to increase exposure"
            
            # Create compliance event
            event = ComplianceEvent(
                timestamp=datetime.now(),
                framework=limit.framework,
                regulation=limit.regulation,
                event_type="limit_check",
                status=status,
                metric_value=current_value,
                threshold_value=limit.threshold_value,
                breach_details=breach_details,
                action_required=action_required
            )
            
            # Log the event
            self.compliance_events.append(event)
            
            # Log according to status
            if status == ComplianceStatus.BREACH:
                self.logger.critical(f"COMPLIANCE BREACH: {limit.framework.value} {limit.regulation} - {breach_details}")
            elif status == ComplianceStatus.WARNING:
                self.logger.warning(f"COMPLIANCE WARNING: {limit.framework.value} {limit.regulation} - {breach_details}")
            
            return event
            
        except Exception as e:
            self.logger.error(f"Error evaluating limit {limit.metric_name}: {e}")
            return None
    
    def get_compliance_status(self, framework: RegulationFramework = None) -> Dict:
        """Get current compliance status"""
        try:
            # Filter events by framework if specified
            if framework:
                events = [e for e in self.compliance_events if e.framework == framework]
            else:
                events = self.compliance_events
            
            # Get recent events (last 24 hours)
            recent_cutoff = datetime.now() - timedelta(hours=24)
            recent_events = [e for e in events if e.timestamp > recent_cutoff]
            
            # Calculate status counts
            status_counts = {
                ComplianceStatus.COMPLIANT.value: len([e for e in recent_events if e.status == ComplianceStatus.COMPLIANT]),
                ComplianceStatus.WARNING.value: len([e for e in recent_events if e.status == ComplianceStatus.WARNING]),
                ComplianceStatus.BREACH.value: len([e for e in recent_events if e.status == ComplianceStatus.BREACH]),
                ComplianceStatus.CRITICAL.value: len([e for e in recent_events if e.status == ComplianceStatus.CRITICAL])
            }
            
            # Get active breaches
            active_breaches = [e for e in recent_events if e.status in [ComplianceStatus.BREACH, ComplianceStatus.CRITICAL]]
            
            return {
                "status_summary": status_counts,
                "active_breaches": len(active_breaches),
                "breach_events": [
                    {
                        "timestamp": e.timestamp.isoformat(),
                        "framework": e.framework.value,
                        "regulation": e.regulation,
                        "metric_value": e.metric_value,
                        "threshold_value": e.threshold_value,
                        "breach_details": e.breach_details,
                        "action_required": e.action_required
                    } for e in active_breaches
                ],
                "recent_events": len(recent_events),
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting compliance status: {e}")
            return {}
    
    def generate_compliance_report(self, framework: RegulationFramework = None) -> Dict:
        """Generate comprehensive compliance report"""
        try:
            # Get framework metrics
            if framework == RegulationFramework.BASEL_III:
                metrics = self.basel_calculator.calculate_capital_ratios()
                report_title = "Basel III Capital Requirements Report"
                
            elif framework == RegulationFramework.MIFID_II:
                metrics = self.mifid_calculator.calculate_execution_quality()
                report_title = "MiFID II Execution Quality Report"
                
            elif framework == RegulationFramework.DODD_FRANK:
                metrics = self.dodd_frank_calculate_swaps_compliance()
                report_title = "Dodd-Frank Swaps Compliance Report"
                
            else:
                # Generate combined report
                metrics = {
                    "basel_iii": self.basel_calculator.calculate_capital_ratios(),
                    "mifid_ii": self.mifid_calculator.calculate_execution_quality(),
                    "dodd_frank": self.dodd_frank_calculate_swaps_compliance()
                }
                report_title = "Comprehensive Regulatory Compliance Report"
            
            # Get compliance status
            compliance_status = self.get_compliance_status(framework)
            
            # Generate report
            report = {
                "report_title": report_title,
                "generation_timestamp": datetime.now().isoformat(),
                "reporting_period": {
                    "start_date": (datetime.now() - timedelta(days=30)).isoformat(),
                    "end_date": datetime.now().isoformat()
                },
                "executive_summary": {
                    "overall_status": "COMPLIANT" if compliance_status["active_breaches"] == 0 else "NON_COMPLIANT",
                    "total_events": compliance_status["recent_events"],
                    "active_breaches": compliance_status["active_breaches"],
                    "critical_issues": compliance_status["status_summary"][ComplianceStatus.CRITICAL.value]
                },
                "compliance_status": compliance_status,
                "regulatory_metrics": metrics,
                "recommendations": self._generate_recommendations(compliance_status),
                "next_review_date": (datetime.now() + timedelta(days=30)).isoformat()
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating compliance report: {e}")
            return {}
    
    def _generate_recommendations(self, compliance_status: Dict) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        try:
            # Analyze breach patterns
            breaches = compliance_status.get("breach_events", [])
            
            if not breaches:
                recommendations.append("Continue current compliance practices. All regulatory limits are within acceptable ranges.")
                return recommendations
            
            # Analyze breach frequency by framework
            framework_breaches = defaultdict(int)
            for breach in breaches:
                framework_breaches[breach["framework"]] += 1
            
            # Generate targeted recommendations
            if framework_breaches["basel_iii"]:
                recommendations.append("Implement enhanced capital management procedures to meet Basel III requirements")
                recommendations.append("Review and optimize risk-weighted asset calculations")
            
            if framework_breaches["mifid_ii"]:
                recommendations.append("Upgrade order processing systems to meet MiFID II execution quality standards")
                recommendations.append("Implement real-time transaction reporting automation")
            
            if framework_breaches["dodd_frank"]:
                recommendations.append("Enhance swap position monitoring for Dodd-Frank compliance")
                recommendations.append("Review large trader reporting procedures")
            
            # Add general recommendations
            recommendations.extend([
                "Schedule quarterly compliance reviews with regulatory experts",
                "Implement automated early warning systems for regulatory limits",
                "Maintain updated regulatory documentation and procedures",
                "Conduct regular compliance training for all trading personnel"
            ])
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Error generating specific recommendations - manual review required")
        
        return recommendations
    
    def trade_surveillance(self, trade_data: Dict) -> Dict:
        """Perform trade surveillance for compliance"""
        try:
            surveillance_results = {
                "trade_id": trade_data.get("trade_id"),
                "surveillance_timestamp": datetime.now().isoformat(),
                "risk_flags": [],
                "compliance_flags": [],
                "surveillance_score": 0
            }
            
            # Check for manipulative trading patterns
            surveillance_results["risk_flags"].extend(self._check_manipulative_patterns(trade_data))
            
            # Check regulatory reporting requirements
            surveillance_results["compliance_flags"].extend(self._check_regulatory_reporting(trade_data))
            
            # Check best execution requirements
            surveillance_results["compliance_flags"].extend(self._check_best_execution(trade_data))
            
            # Calculate surveillance score (0-100, lower is better)
            surveillance_results["surveillance_score"] = len(surveillance_results["risk_flags"]) * 10 + len(surveillance_results["compliance_flags"]) * 5
            
            return surveillance_results
            
        except Exception as e:
            self.logger.error(f"Error in trade surveillance: {e}")
            return {"error": str(e)}
    
    def _check_manipulative_patterns(self, trade_data: Dict) -> List[str]:
        """Check for manipulative trading patterns"""
        flags = []
        
        try:
            # Check for wash trading
            if self._detect_wash_trading(trade_data):
                flags.append("Potential wash trading detected")
            
            # Check for spoofing
            if self._detect_spoofing_patterns(trade_data):
                flags.append("Potential spoofing patterns detected")
            
            # Check for layering
            if self._detect_layering_patterns(trade_data):
                flags.append("Potential layering detected")
                
        except Exception as e:
            self.logger.error(f"Error checking manipulative patterns: {e}")
        
        return flags
    
    def _check_regulatory_reporting(self, trade_data: Dict) -> List[str]:
        """Check regulatory reporting requirements"""
        flags = []
        
        try:
            # Check for large trader reporting requirements
            notional_amount = trade_data.get("notional_amount", 0)
            if notional_amount > 25000000:  # $25M threshold
                flags.append("Large trader reporting required under Dodd-Frank Section 730")
            
            # Check for swap reporting requirements
            if trade_data.get("instrument_type") in ["swap", "swaption"]:
                flags.append("Swap transaction reporting required under Dodd-Frank Section 763")
            
            # Check for short sale reporting
            if trade_data.get("short_sale"):
                flags.append("Short sale position reporting may be required")
                
        except Exception as e:
            self.logger.error(f"Error checking regulatory reporting: {e}")
        
        return flags
    
    def _check_best_execution(self, trade_data: Dict) -> List[str]:
        """Check best execution requirements"""
        flags = []
        
        try:
            # Check execution price deviation
            execution_price = trade_data.get("execution_price")
            market_price = trade_data.get("market_price")
            
            if execution_price and market_price:
                price_impact = abs(execution_price - market_price) / market_price
                if price_impact > 0.02:  # 2% threshold
                    flags.append(f"Price impact {price_impact:.2%} exceeds MiFID II best execution threshold")
            
            # Check for potential front running
            if self._detect_front_running(trade_data):
                flags.append("Potential front running detected")
                
        except Exception as e:
            self.logger.error(f"Error checking best execution: {e}")
        
        return flags
    
    def _detect_wash_trading(self, trade_data: Dict) -> bool:
        """Detect potential wash trading"""
        # Simplified detection - real implementation would be more sophisticated
        trades_per_day = trade_data.get("client_trades_per_day", 0)
        return trades_per_day > 100  # Arbitrary threshold
    
    def _detect_spoofing_patterns(self, trade_data: Dict) -> bool:
        """Detect potential spoofing patterns"""
        # Simplified detection
        large_orders_cancelled = trade_data.get("large_orders_cancelled", 0)
        return large_orders_cancelled > 5  # Arbitrary threshold
    
    def _detect_layering_patterns(self, trade_data: Dict) -> bool:
        """Detect potential layering patterns"""
        # Simplified detection
        layered_orders = trade_data.get("layered_order_count", 0)
        return layered_orders > 3  # Arbitrary threshold
    
    def _detect_front_running(self, trade_data: Dict) -> bool:
        """Detect potential front running"""
        # Simplified detection
        time_diff = trade_data.get("time_between_large_orders_ms", 0)
        return time_diff < 1000  # Less than 1 second
    
    def get_regulatory_limits(self, framework: RegulationFramework = None) -> List[Dict]:
        """Get all regulatory limits"""
        try:
            limits = []
            
            for key, limit in self.regulatory_limits.items():
                if framework is None or limit.framework == framework:
                    limits.append({
                        "framework": limit.framework.value,
                        "regulation": limit.regulation,
                        "metric_name": limit.metric_name,
                        "threshold_value": limit.threshold_value,
                        "threshold_type": limit.threshold_type,
                        "calculation_method": limit.calculation_method,
                        "reporting_frequency": limit.reporting_frequency,
                        "jurisdiction": limit.jurisdiction,
                        "effective_date": limit.effective_date.isoformat()
                    })
            
            return limits
            
        except Exception as e:
            self.logger.error(f"Error getting regulatory limits: {e}")
            return []


class BaselIIICalculator:
    """Basel III capital requirements calculator"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_portfolio_metrics(self) -> Dict:
        """Get portfolio metrics for Basel III calculations"""
        # In real implementation, this would fetch actual portfolio data
        return {
            "tier_1_ratio": 0.08,  # 8%
            "total_capital_ratio": 0.10,  # 10%
            "leverage_ratio": 0.035,  # 3.5%
            "lcr": 1.15,  # 115%
            "nsfr": 1.08  # 108%
        }
    
    def calculate_capital_ratios(self) -> Dict:
        """Calculate Basel III capital ratios"""
        try:
            metrics = self.get_portfolio_metrics()
            
            ratios = {
                "Tier_1_Capital_Ratio": {
                    "current_value": metrics["tier_1_ratio"],
                    "regulatory_minimum": 0.06,
                    "status": "COMPLIANT" if metrics["tier_1_ratio"] >= 0.06 else "NON_COMPLIANT"
                },
                "Total_Capital_Ratio": {
                    "current_value": metrics["total_capital_ratio"],
                    "regulatory_minimum": 0.08,
                    "status": "COMPLIANT" if metrics["total_capital_ratio"] >= 0.08 else "NON_COMPLIANT"
                },
                "Leverage_Ratio": {
                    "current_value": metrics["leverage_ratio"],
                    "regulatory_minimum": 0.03,
                    "status": "COMPLIANT" if metrics["leverage_ratio"] >= 0.03 else "NON_COMPLIANT"
                },
                "Liquidity_Coverage_Ratio": {
                    "current_value": metrics["lcr"],
                    "regulatory_minimum": 1.0,
                    "status": "COMPLIANT" if metrics["lcr"] >= 1.0 else "NON_COMPLIANT"
                },
                "Net_Stable_Funding_Ratio": {
                    "current_value": metrics["nsfr"],
                    "regulatory_minimum": 1.0,
                    "status": "COMPLIANT" if metrics["nsfr"] >= 1.0 else "NON_COMPLIANT"
                }
            }
            
            return ratios
            
        except Exception as e:
            self.logger.error(f"Error calculating Basel III ratios: {e}")
            return {}


class MiFIDIICalculator:
    """MiFID II execution quality calculator"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_execution_metrics(self) -> Dict:
        """Get execution metrics for MiFID II calculations"""
        # In real implementation, this would fetch actual execution data
        return {
            "price_impact": 0.015,  # 1.5%
            "processing_delay_ms": 85,  # 85 milliseconds
            "reporting_delay_seconds": 45,  # 45 seconds
            "conflict_score": 0.1  # Low conflict score
        }
    
    def calculate_execution_quality(self) -> Dict:
        """Calculate MiFID II execution quality metrics"""
        try:
            metrics = self.get_execution_metrics()
            
            execution_quality = {
                "Best_Execution_Price_Impact": {
                    "current_value": metrics["price_impact"],
                    "regulatory_maximum": 0.02,
                    "status": "COMPLIANT" if metrics["price_impact"] <= 0.02 else "NON_COMPLIANT"
                },
                "Order_Processing_Delay": {
                    "current_value": metrics["processing_delay_ms"],
                    "regulatory_maximum": 100,
                    "status": "COMPLIANT" if metrics["processing_delay_ms"] <= 100 else "NON_COMPLIANT"
                },
                "Transaction_Reporting_Delay": {
                    "current_value": metrics["reporting_delay_seconds"],
                    "regulatory_maximum": 300,
                    "status": "COMPLIANT" if metrics["reporting_delay_seconds"] <= 300 else "NON_COMPLIANT"
                },
                "Investment_Advice_Conflict_Check": {
                    "current_value": metrics["conflict_score"],
                    "regulatory_maximum": 1.0,
                    "status": "COMPLIANT" if metrics["conflict_score"] <= 1.0 else "NON_COMPLIANT"
                }
            }
            
            return execution_quality
            
        except Exception as e:
            self.logger.error(f"Error calculating MiFID II execution quality: {e}")
            return {}


class DoddFrankCalculator:
    """Dodd-Frank swaps compliance calculator"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_derivatives_metrics(self) -> Dict:
        """Get derivatives metrics for Dodd-Frank calculations"""
        # In real implementation, this would fetch actual derivatives data
        return {
            "swap_dealer_ratio": 0.12,  # 12%
            "position_notional": 18.5,  # $18.5B
            "trader_volume_ratio": 0.03,  # 3%
            "market_share": 0.06  # 6%
        }
    
    def dodd_frank_calculate_swaps_compliance(self) -> Dict:
        """Calculate Dodd-Frank swaps compliance"""
        try:
            metrics = self.get_derivatives_metrics()
            
            swaps_compliance = {
                "Swap_Dealer_Capital_Ratio": {
                    "current_value": metrics["swap_dealer_ratio"],
                    "regulatory_minimum": 0.08,
                    "status": "COMPLIANT" if metrics["swap_dealer_ratio"] >= 0.08 else "NON_COMPLIANT"
                },
                "Position_Limit_Reporting_Threshold": {
                    "current_value": metrics["position_notional"],
                    "regulatory_maximum": 25,
                    "status": "COMPLIANT" if metrics["position_notional"] <= 25 else "NON_COMPLIANT"
                },
                "Large_Trader_Reporting_Threshold": {
                    "current_value": metrics["trader_volume_ratio"],
                    "regulatory_maximum": 0.05,
                    "status": "COMPLIANT" if metrics["trader_volume_ratio"] <= 0.05 else "NON_COMPLIANT"
                },
                "Systemically_Important_Financial_Market_Utility_Limit": {
                    "current_value": metrics["market_share"],
                    "regulatory_maximum": 0.10,
                    "status": "COMPLIANT" if metrics["market_share"] <= 0.10 else "NON_COMPLIANT"
                }
            }
            
            return swaps_compliance
            
        except Exception as e:
            self.logger.error(f"Error calculating Dodd-Frank compliance: {e}")
            return {}


# Example usage and testing
if __name__ == "__main__":
    # Initialize compliance frameworks
    compliance = ComplianceFrameworks()
    
    # Start real-time monitoring
    compliance.start_real_time_monitoring()
    
    # Generate compliance report
    report = compliance.generate_compliance_report()
    print("Compliance Report Generated")
    print(f"Active Breaches: {report.get('executive_summary', {}).get('active_breaches', 'Unknown')}")
    
    # Example trade surveillance
    sample_trade = {
        "trade_id": "TRD_001",
        "notional_amount": 50000000,
        "instrument_type": "equity",
        "execution_price": 102.5,
        "market_price": 102.0,
        "short_sale": True,
        "client_trades_per_day": 50,
        "large_orders_cancelled": 2,
        "layered_order_count": 1,
        "time_between_large_orders_ms": 2000
    }
    
    surveillance_result = compliance.trade_surveillance(sample_trade)
    print("Trade Surveillance Result:")
    print(f"Surveillance Score: {surveillance_result.get('surveillance_score', 'Unknown')}")
    print(f"Risk Flags: {surveillance_result.get('risk_flags', [])}")
    
    # Stop monitoring
    compliance.stop_real_time_monitoring()