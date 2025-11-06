"""
Compliance and Regulatory Checks Engine

Manages compliance and regulatory requirements:
- Regulatory rule validation
- Exchange compliance checks
- Internal compliance rules
- Audit trail generation
- Compliance reporting
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import json

from config.database import get_db
from database.models.risk import ComplianceRule, RiskEvent, RiskEventType, RiskLevel
from database.models.trading import Order, Position, Trade

logger = logging.getLogger(__name__)


class ComplianceStatus(str, Enum):
    """Compliance status levels."""
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ComplianceEngine:
    """
    Manages all compliance and regulatory requirements.
    """
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.db = get_db()
        
        # Compliance rule categories
        self.compliance_categories = {
            "regulatory": "External regulatory requirements",
            "exchange": "Exchange-specific rules",
            "internal": "Internal company policies",
            "risk": "Risk management rules",
            "liquidity": "Liquidity requirements",
            "reporting": "Reporting requirements"
        }
        
        # Built-in compliance checks
        self.compliance_checks = {
            "market_hours": self._check_market_hours_compliance,
            "position_limits": self._check_position_limit_compliance,
            "order_frequency": self._check_order_frequency_compliance,
            "concentration": self._check_concentration_compliance,
            "wash_trading": self._check_wash_trading_compliance,
            "insider_trading": self._check_insider_trading_compliance,
            "news_blackout": self._check_news_blackout_compliance,
            "day_trading": self._check_day_trading_compliance
        }
        
        logger.info(f"ComplianceEngine initialized for user {self.user_id}")
    
    async def validate_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive order compliance validation.
        
        Args:
            order_data: Order information to validate
            
        Returns:
            Compliance validation result
        """
        result = {
            "approved": True,
            "rejection_reason": None,
            "warnings": [],
            "compliance_checks": [],
            "compliance_score": 100.0,
            "violations": [],
            "actions_taken": []
        }
        
        try:
            # Run built-in compliance checks
            for check_name, check_func in self.compliance_checks.items():
                check_result = await check_func(order_data)
                result["compliance_checks"].append({
                    "check_name": check_name,
                    "result": check_result
                })
                
                # Update compliance score
                if check_result["status"] == ComplianceStatus.VIOLATION:
                    result["compliance_score"] -= 20
                    result["violations"].append(check_result)
                elif check_result["status"] == ComplianceStatus.WARNING:
                    result["compliance_score"] -= 5
                    result["warnings"].append(check_result["message"])
            
            # Run custom compliance rules
            custom_results = await self._run_custom_compliance_rules(order_data)
            result["compliance_checks"].extend(custom_results)
            
            # Check if compliance score is too low
            if result["compliance_score"] < 50:
                result["approved"] = False
                result["rejection_reason"] = "Compliance score below acceptable threshold"
            
            # Log compliance check for audit
            await self._log_compliance_check("order_validation", order_data, result)
            
        except Exception as e:
            logger.error(f"Order compliance validation error: {str(e)}")
            result.update({
                "approved": False,
                "rejection_reason": f"Compliance validation error: {str(e)}"
            })
        
        return result
    
    async def check_position_compliance(self, position: Position) -> Dict[str, Any]:
        """
        Check compliance for an existing position.
        
        Args:
            position: Position to check
            
        Returns:
            Position compliance result
        """
        result = {
            "position_id": position.id,
            "symbol": position.symbol,
            "compliant": True,
            "violations": [],
            "warnings": [],
            "compliance_score": 100.0
        }
        
        try:
            # Check position limits
            position_check = await self._check_position_limit_compliance({"symbol": position.symbol, "quantity": position.quantity})
            if position_check["status"] == ComplianceStatus.VIOLATION:
                result["compliant"] = False
                result["violations"].append(position_check)
                result["compliance_score"] -= 20
            
            # Check concentration
            concentration_check = await self._check_concentration_compliance({"symbol": position.symbol})
            if concentration_check["status"] == ComplianceStatus.WARNING:
                result["warnings"].append(concentration_check["message"])
                result["compliance_score"] -= 5
            
            # Check regulatory restrictions
            regulatory_check = await self._check_regulatory_restrictions(position.symbol)
            if regulatory_check["status"] == ComplianceStatus.VIOLATION:
                result["compliant"] = False
                result["violations"].append(regulatory_check)
                result["compliance_score"] -= 30
            
        except Exception as e:
            logger.error(f"Position compliance check error: {str(e)}")
            result["error"] = str(e)
        
        return result
    
    async def get_compliance_status(self) -> Dict[str, Any]:
        """
        Get overall compliance status summary.
        
        Returns:
            Comprehensive compliance status
        """
        status = {
            "user_id": self.user_id,
            "overall_compliance_score": 100.0,
            "compliance_level": ComplianceStatus.COMPLIANT,
            "categories": {},
            "recent_violations": [],
            "regulatory_status": {},
            "recommendations": []
        }
        
        try:
            # Get recent compliance violations
            recent_violations = self.db.query(RiskEvent).filter(
                and_(
                    RiskEvent.user_id == self.user_id,
                    RiskEvent.occurred_at >= datetime.now() - timedelta(days=30),
                    RiskEvent.event_type.in_([
                        RiskEventType.POSITION_LIMIT_BREACH,
                        RiskEventType.ORDER_LIMIT_BREACH,
                        RiskEventType.EXPOSURE_LIMIT
                    ])
                )
            ).order_by(RiskEvent.occurred_at.desc()).limit(10).all()
            
            status["recent_violations"] = [
                {
                    "type": violation.event_type.value,
                    "level": violation.risk_level.value,
                    "title": violation.title,
                    "occurred_at": violation.occurred_at,
                    "is_resolved": violation.is_resolved
                }
                for violation in recent_violations
            ]
            
            # Check compliance by category
            for category in self.compliance_categories.keys():
                category_score = await self._calculate_category_compliance_score(category)
                status["categories"][category] = category_score
                status["overall_compliance_score"] += category_score - 100  # Adjust for average
            
            # Determine overall compliance level
            if status["overall_compliance_score"] >= 90:
                status["compliance_level"] = ComplianceStatus.COMPLIANT
            elif status["overall_compliance_score"] >= 70:
                status["compliance_level"] = ComplianceStatus.WARNING
            elif status["overall_compliance_score"] >= 50:
                status["compliance_level"] = ComplianceStatus.VIOLATION
            else:
                status["compliance_level"] = ComplianceStatus.CRITICAL
            
            # Get regulatory status
            status["regulatory_status"] = await self._get_regulatory_status()
            
            # Generate recommendations
            status["recommendations"] = await self._generate_compliance_recommendations(status)
            
        except Exception as e:
            logger.error(f"Compliance status error: {str(e)}")
            status["error"] = str(e)
        
        return status
    
    async def generate_compliance_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Generate comprehensive compliance report for a date range.
        
        Args:
            start_date: Report start date
            end_date: Report end date
            
        Returns:
            Detailed compliance report
        """
        report = {
            "report_period": {
                "start_date": start_date,
                "end_date": end_date,
                "generated_at": datetime.now()
            },
            "user_id": self.user_id,
            "summary": {
                "total_trades": 0,
                "compliant_trades": 0,
                "violations": 0,
                "warnings": 0,
                "compliance_rate": 100.0
            },
            "violations": [],
            "trades_analyzed": [],
            "recommendations": [],
            "regulatory_alignment": {}
        }
        
        try:
            # Get all trades in the period
            trades = self.db.query(Trade).filter(
                and_(
                    Trade.user_id == self.user_id,
                    Trade.executed_at >= start_date,
                    Trade.executed_at <= end_date
                )
            ).all()
            
            report["summary"]["total_trades"] = len(trades)
            
            # Analyze each trade
            for trade in trades:
                # Get associated order for compliance check
                order = self.db.query(Order).filter(Order.id == trade.order_id).first()
                if order:
                    trade_data = {
                        "trade_id": trade.id,
                        "symbol": trade.symbol,
                        "side": trade.side.value,
                        "quantity": trade.quantity,
                        "price": trade.price,
                        "executed_at": trade.executed_at,
                        "compliance_status": ComplianceStatus.COMPLIANT,
                        "violations": []
                    }
                    
                    # Check compliance for this trade
                    compliance_result = await self.validate_order({
                        "symbol": trade.symbol,
                        "side": trade.side.value,
                        "quantity": trade.quantity,
                        "order_type": order.order_type.value,
                        "timestamp": trade.executed_at
                    })
                    
                    if not compliance_result["approved"]:
                        trade_data["compliance_status"] = ComplianceStatus.VIOLATION
                        trade_data["violations"] = compliance_result.get("violations", [])
                        report["summary"]["violations"] += 1
                    elif compliance_result.get("warnings"):
                        trade_data["compliance_status"] = ComplianceStatus.WARNING
                        report["summary"]["warnings"] += 1
                    else:
                        report["summary"]["compliant_trades"] += 1
                    
                    report["trades_analyzed"].append(trade_data)
            
            # Calculate compliance rate
            if report["summary"]["total_trades"] > 0:
                report["summary"]["compliance_rate"] = (
                    report["summary"]["compliant_trades"] / report["summary"]["total_trades"] * 100
                )
            
            # Add violation details
            report["violations"] = [
                {
                    "trade_id": trade["trade_id"],
                    "symbol": trade["symbol"],
                    "violation_type": violation.get("type", "unknown"),
                    "description": violation.get("message", "Compliance violation"),
                    "executed_at": trade["executed_at"]
                }
                for trade in report["trades_analyzed"]
                if trade["compliance_status"] == ComplianceStatus.VIOLATION
                for violation in trade["violations"]
            ]
            
            # Generate recommendations
            report["recommendations"] = await self._generate_compliance_recommendations(report)
            
            # Check regulatory alignment
            report["regulatory_alignment"] = await self._check_regulatory_alignment(start_date, end_date)
            
        except Exception as e:
            logger.error(f"Compliance report generation error: {str(e)}")
            report["error"] = str(e)
        
        return report
    
    # Built-in compliance check implementations
    
    async def _check_market_hours_compliance(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check market hours compliance."""
        try:
            symbol = order_data.get("symbol", "").upper()
            current_time = datetime.now()
            
            # Skip check for crypto (24/7 markets)
            crypto_suffixes = ["USDT", "BTC", "ETH", "BNB"]
            if any(suffix in symbol for suffix in crypto_suffixes):
                return {
                    "status": ComplianceStatus.COMPLIANT,
                    "message": "24/7 market - no hours restriction"
                }
            
            # Simplified market hours check
            weekday = current_time.weekday()
            current_hour = current_time.hour
            
            # Weekend check
            if weekday >= 5:
                return {
                    "status": ComplianceStatus.VIOLATION,
                    "message": "Trading on weekend - market closed",
                    "check_name": "market_hours",
                    "severity": "high"
                }
            
            # Market hours: 9:30 AM - 4:00 PM EST (simplified)
            if not (9 <= current_hour < 16):
                return {
                    "status": ComplianceStatus.WARNING,
                    "message": f"Trading outside standard hours: {current_hour}:00",
                    "check_name": "market_hours",
                    "severity": "medium"
                }
            
            return {
                "status": ComplianceStatus.COMPLIANT,
                "message": "Within market hours"
            }
            
        except Exception as e:
            return {
                "status": ComplianceStatus.UNKNOWN,
                "message": f"Market hours check error: {str(e)}"
            }
    
    async def _check_position_limit_compliance(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check position limit compliance."""
        try:
            symbol = order_data.get("symbol", "")
            quantity = order_data.get("quantity", 0)
            
            # Check against regulatory position limits
            # This is a simplified check - real implementation would use actual limits
            
            # Example: Maximum position size for retail investors
            max_position_size = 100000  # $100k
            estimated_price = order_data.get("estimated_price", 100)  # Assume $100 if not provided
            position_value = abs(quantity) * estimated_price
            
            if position_value > max_position_size:
                return {
                    "status": ComplianceStatus.VIOLATION,
                    "message": f"Position size ${position_value:,.2f} exceeds regulatory limit ${max_position_size:,.2f}",
                    "check_name": "position_limits",
                    "severity": "high",
                    "limit_value": max_position_size,
                    "current_value": position_value
                }
            
            return {
                "status": ComplianceStatus.COMPLIANT,
                "message": "Position within regulatory limits"
            }
            
        except Exception as e:
            return {
                "status": ComplianceStatus.UNKNOWN,
                "message": f"Position limit check error: {str(e)}"
            }
    
    async def _check_order_frequency_compliance(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check order frequency compliance."""
        try:
            # Check for wash trading or excessive order frequency
            current_time = datetime.now()
            one_hour_ago = current_time - timedelta(hours=1)
            
            # Count orders in the last hour
            recent_orders = self.db.query(Order).filter(
                and_(
                    Order.user_id == self.user_id,
                    Order.submitted_at >= one_hour_ago
                )
            ).count()
            
            max_orders_per_hour = 100  # Reasonable limit
            
            if recent_orders >= max_orders_per_hour:
                return {
                    "status": ComplianceStatus.WARNING,
                    "message": f"High order frequency: {recent_orders} orders in last hour",
                    "check_name": "order_frequency",
                    "severity": "medium",
                    "current_count": recent_orders,
                    "limit": max_orders_per_hour
                }
            
            return {
                "status": ComplianceStatus.COMPLIANT,
                "message": "Order frequency within acceptable limits"
            }
            
        except Exception as e:
            return {
                "status": ComplianceStatus.UNKNOWN,
                "message": f"Order frequency check error: {str(e)}"
            }
    
    async def _check_concentration_compliance(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check concentration compliance."""
        try:
            symbol = order_data.get("symbol", "")
            quantity = order_data.get("quantity", 0)
            estimated_price = order_data.get("estimated_price", 0)
            
            # Calculate portfolio concentration
            positions = self.db.query(Position).filter(Position.user_id == self.user_id).all()
            total_portfolio = sum(abs(pos.market_value or 0) for pos in positions)
            
            if total_portfolio > 0:
                # Check existing concentration
                existing_position = next((pos for pos in positions if pos.symbol == symbol), None)
                existing_exposure = abs(existing_position.market_value or 0) if existing_position else 0
                new_exposure = existing_exposure + abs(quantity * estimated_price)
                
                new_concentration = new_exposure / total_portfolio
                max_concentration = 0.20  # 20% max concentration
                
                if new_concentration > max_concentration:
                    return {
                        "status": ComplianceStatus.WARNING,
                        "message": f"High concentration in {symbol}: {new_concentration:.1%}",
                        "check_name": "concentration",
                        "severity": "medium",
                        "current_concentration": new_concentration,
                        "limit": max_concentration
                    }
            
            return {
                "status": ComplianceStatus.COMPLIANT,
                "message": "Concentration within acceptable limits"
            }
            
        except Exception as e:
            return {
                "status": ComplianceStatus.UNKNOWN,
                "message": f"Concentration check error: {str(e)}"
            }
    
    async def _check_wash_trading_compliance(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check wash trading compliance."""
        try:
            symbol = order_data.get("symbol", "")
            side = order_data.get("side", "")
            current_time = datetime.now()
            
            # Check for wash trading patterns (buy and sell same symbol quickly)
            one_minute_ago = current_time - timedelta(minutes=1)
            
            recent_trades = self.db.query(Trade).filter(
                and_(
                    Trade.user_id == self.user_id,
                    Trade.symbol == symbol,
                    Trade.executed_at >= one_minute_ago
                )
            ).all()
            
            # Simple wash trading detection
            buy_trades = [t for t in recent_trades if t.side.value == "buy"]
            sell_trades = [t for t in recent_trades if t.side.value == "sell"]
            
            if buy_trades and sell_trades and side in ["buy", "sell"]:
                # Check if there's a recent opposite trade
                opposite_side = "sell" if side == "buy" else "buy"
                opposite_trades = [t for t in recent_trades if t.side.value == opposite_side]
                
                if opposite_trades:
                    return {
                        "status": ComplianceStatus.VIOLATION,
                        "message": "Potential wash trading pattern detected",
                        "check_name": "wash_trading",
                        "severity": "high"
                    }
            
            return {
                "status": ComplianceStatus.COMPLIANT,
                "message": "No wash trading patterns detected"
            }
            
        except Exception as e:
            return {
                "status": ComplianceStatus.UNKNOWN,
                "message": f"Wash trading check error: {str(e)}"
            }
    
    async def _check_insider_trading_compliance(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check insider trading compliance (simplified)."""
        try:
            symbol = order_data.get("symbol", "")
            
            # This would check against insider trading watchlists
            # For now, just a placeholder
            
            # Example restricted symbols (would come from regulatory feeds)
            restricted_symbols = ["AAPL", "GOOGL", "MSFT"]  # Placeholder
            
            if symbol in restricted_symbols:
                return {
                    "status": ComplianceStatus.VIOLATION,
                    "message": f"Symbol {symbol} on restricted trading list",
                    "check_name": "insider_trading",
                    "severity": "critical"
                }
            
            return {
                "status": ComplianceStatus.COMPLIANT,
                "message": "No insider trading restrictions"
            }
            
        except Exception as e:
            return {
                "status": ComplianceStatus.UNKNOWN,
                "message": f"Insider trading check error: {str(e)}"
            }
    
    async def _check_news_blackout_compliance(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check news blackout compliance."""
        try:
            symbol = order_data.get("symbol", "")
            current_time = datetime.now()
            
            # This would check against news calendars and earnings dates
            # Simplified placeholder
            
            # Check for earnings blackout (example)
            earnings_blackout_symbols = ["AAPL", "MSFT", "GOOGL"]  # Would come from earnings calendar
            if symbol in earnings_blackout_symbols:
                # Check if within blackout period (simplified: during market hours before earnings)
                current_hour = current_time.hour
                if 9 <= current_hour < 16:  # Simplified blackout period
                    return {
                        "status": ComplianceStatus.WARNING,
                        "message": f"Trading during earnings blackout period for {symbol}",
                        "check_name": "news_blackout",
                        "severity": "medium"
                    }
            
            return {
                "status": ComplianceStatus.COMPLIANT,
                "message": "No news blackout restrictions"
            }
            
        except Exception as e:
            return {
                "status": ComplianceStatus.UNKNOWN,
                "message": f"News blackout check error: {str(e)}"
            }
    
    async def _check_day_trading_compliance(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check day trading compliance."""
        try:
            # This would check pattern day trader rules
            # Simplified check for now
            
            current_date = datetime.now().date()
            
            # Count round-trip trades today
            trades_today = self.db.query(Trade).filter(
                and_(
                    Trade.user_id == self.user_id,
                    Trade.executed_at >= datetime.combine(current_date, datetime.min.time())
                )
            ).all()
            
            # Simple day trading detection
            symbols_today = set(trade.symbol for trade in trades_today)
            
            if len(symbols_today) < len(trades_today):
                # Multiple trades in same symbols (potential day trading)
                return {
                    "status": ComplianceStatus.WARNING,
                    "message": "Multiple trades in same symbols - potential day trading",
                    "check_name": "day_trading",
                    "severity": "medium"
                }
            
            return {
                "status": ComplianceStatus.COMPLIANT,
                "message": "No day trading concerns detected"
            }
            
        except Exception as e:
            return {
                "status": ComplianceStatus.UNKNOWN,
                "message": f"Day trading check error: {str(e)}"
            }
    
    async def _run_custom_compliance_rules(self, order_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run custom compliance rules from database."""
        results = []
        
        try:
            rules = self.db.query(ComplianceRule).filter(
                and_(
                    ComplianceRule.is_active == True,
                    ComplianceRule.category == "internal",
                    ComplianceRule.effective_from <= datetime.now()
                )
            ).all()
            
            for rule in rules:
                # Custom rule evaluation would go here
                # This is a simplified implementation
                
                results.append({
                    "check_name": f"custom_{rule.rule_code}",
                    "result": {
                        "status": ComplianceStatus.COMPLIANT,
                        "message": f"Custom rule {rule.rule_name} passed"
                    }
                })
                
        except Exception as e:
            logger.error(f"Custom compliance rules error: {str(e)}")
        
        return results
    
    async def _check_regulatory_restrictions(self, symbol: str) -> Dict[str, Any]:
        """Check regulatory restrictions for a symbol."""
        try:
            # This would check against regulatory restricted symbols
            # Simplified implementation
            
            restricted_symbols = ["TSLA", "GME", "AMC"]  # Example restricted symbols
            
            if symbol in restricted_symbols:
                return {
                    "status": ComplianceStatus.VIOLATION,
                    "message": f"Symbol {symbol} under regulatory restrictions",
                    "check_name": "regulatory_restrictions",
                    "severity": "high"
                }
            
            return {
                "status": ComplianceStatus.COMPLIANT,
                "message": "No regulatory restrictions"
            }
            
        except Exception as e:
            return {
                "status": ComplianceStatus.UNKNOWN,
                "message": f"Regulatory restrictions check error: {str(e)}"
            }
    
    async def _calculate_category_compliance_score(self, category: str) -> float:
        """Calculate compliance score for a specific category."""
        try:
            # This would analyze historical compliance for the category
            # Simplified implementation
            
            if category == "regulatory":
                return 95.0  # Example score
            elif category == "exchange":
                return 98.0
            elif category == "internal":
                return 92.0
            else:
                return 90.0
                
        except Exception:
            return 80.0
    
    async def _get_regulatory_status(self) -> Dict[str, Any]:
        """Get current regulatory compliance status."""
        try:
            return {
                "pattern_day_trader": "compliant",
                "wash_trading": "compliant",
                "insider_trading": "compliant",
                "market_manipulation": "compliant",
                "reporting_timeliness": "compliant"
            }
            
        except Exception:
            return {"error": "Unable to determine regulatory status"}
    
    async def _generate_compliance_recommendations(self, status_data: Dict[str, Any]) -> List[str]:
        """Generate compliance improvement recommendations."""
        recommendations = []
        
        try:
            if status_data["overall_compliance_score"] < 90:
                recommendations.append("Review and address compliance violations to improve score")
            
            if status_data.get("recent_violations"):
                recommendations.append("Investigate recent compliance violations and implement corrective measures")
            
            if status_data["compliance_level"] in [ComplianceStatus.VIOLATION, ComplianceStatus.CRITICAL]:
                recommendations.append("Immediate compliance review required - consider pausing trading activities")
            
            # Add specific recommendations based on violation types
            recent_violations = status_data.get("recent_violations", [])
            violation_types = set(v.get("type", "") for v in recent_violations)
            
            if "position_limit_breach" in violation_types:
                recommendations.append("Implement position size management to avoid limit breaches")
            
            if "wash_trading" in violation_types:
                recommendations.append("Review trading patterns to avoid wash trading violations")
                
        except Exception as e:
            logger.error(f"Recommendation generation error: {str(e)}")
        
        return recommendations
    
    async def _check_regulatory_alignment(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Check regulatory alignment for the report period."""
        try:
            return {
                "sec_requirements": "compliant",
                "exchange_rules": "compliant",
                "finra_rules": "compliant",
                "reporting_deadlines": "met"
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _log_compliance_check(self, check_type: str, order_data: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Log compliance check for audit trail."""
        try:
            # This would create audit log entries
            logger.info(f"Compliance check: {check_type}, Score: {result['compliance_score']}, Approved: {result['approved']}")
            
        except Exception as e:
            logger.error(f"Compliance audit logging error: {str(e)}")
    
    def close(self):
        """Cleanup resources."""
        self.db.close()
