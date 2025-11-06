"""
Regulatory Compliance Module

This module implements comprehensive regulatory compliance checks for trading operations,
including pattern day trader rules, regulatory position limits, and compliance monitoring.

Features:
- Pattern day trader rule validation
- Regulatory position limit enforcement
- Compliance monitoring and reporting
- Multi-jurisdictional regulation support
- Compliance breach detection and escalation
- Audit trail for regulatory purposes
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Union
from enum import Enum
from dataclasses import dataclass, field
from decimal import Decimal
import json
from sqlalchemy import (
    Column, Integer, String, DateTime, Boolean, Float, Text, 
    Index, ForeignKey, DECIMAL
)
from sqlalchemy.orm import declarative_base, relationship, Session
from sqlalchemy.sql import func

from trading_orchestrator.database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

Base = declarative_base()


class RegulatoryRegion(Enum):
    """Regulatory regions with different rules."""
    US_SEC = "us_sec"
    EU_ESMA = "eu_esma"
    UK_FCA = "uk_fca"
    JP_FSA = "jp_fsa"
    HK_SFC = "hk_sfc"
    SINGAPORE_MAS = "singapore_mas"
    AUSTRALIA_ASIC = "australia_asic"
    CANADA_CSA = "canada_csa"


class ComplianceLevel(Enum):
    """Compliance check levels."""
    INFO = "info"
    WARNING = "warning"
    VIOLATION = "violation"
    CRITICAL = "critical"


class OrderType(Enum):
    """Order types for compliance checking."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    BRACKET = "bracket"


@dataclass
class RegulatoryRule:
    """Base class for regulatory rules."""
    rule_id: str
    description: str
    region: RegulatoryRegion
    applies_to: Set[str] = field(default_factory=set)  # asset types
    effective_date: datetime = field(default_factory=datetime.utcnow)
    expiry_date: Optional[datetime] = None


@dataclass
class PositionLimitRule(RegulatoryRule):
    """Position limit regulatory rule."""
    max_position_value: Decimal
    max_position_percentage: Optional[Decimal] = None
    asset_class: str = "equity"
    concentration_limit: Optional[Decimal] = None


@dataclass
class PatternDayTraderRule(RegulatoryRule):
    """Pattern day trader rule configuration."""
    minimum_equity: Decimal = Decimal("25000")  # $25,000 minimum
    day_trade_count_threshold: int = 4  # 4+ day trades in 5 days
    rolling_days: int = 5  # Rolling 5-day window


@dataclass
class TradingSession:
    """Trading session information."""
    account_id: str
    user_id: str
    session_start: datetime
    total_trades: int = 0
    day_trades: int = 0
    total_volume: Decimal = Decimal("0")
    equity_balance: Decimal = Decimal("0")
    margin_equity: Decimal = Decimal("0")


class ComplianceCheck(Base):
    """Database model for compliance checks."""
    __tablename__ = "compliance_checks"
    
    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(String(100), nullable=False, index=True)
    user_id = Column(String(100), nullable=False)
    rule_id = Column(String(100), nullable=False)
    region = Column(String(50), nullable=False)
    check_type = Column(String(50), nullable=False)
    asset_symbol = Column(String(20), nullable=True)
    order_size = Column(DECIMAL, nullable=True)
    current_position = Column(DECIMAL, nullable=True)
    compliance_level = Column(String(20), nullable=False)
    violation_details = Column(Text, nullable=True)
    checked_at = Column(DateTime, server_default=func.now())
    resolved = Column(Boolean, default=False, nullable=False)
    resolved_at = Column(DateTime, nullable=True)


class RegulatoryViolation(Base):
    """Database model for regulatory violations."""
    __tablename__ = "regulatory_violations"
    
    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(String(100), nullable=False, index=True)
    user_id = Column(String(100), nullable=False)
    violation_type = Column(String(100), nullable=False)
    severity = Column(String(20), nullable=False)
    rule_id = Column(String(100), nullable=False)
    description = Column(Text, nullable=False)
    asset_symbol = Column(String(20), nullable=True)
    violation_value = Column(DECIMAL, nullable=True)
    limit_value = Column(DECIMAL, nullable=True)
    detected_at = Column(DateTime, server_default=func.now())
    acknowledged = Column(Boolean, default=False, nullable=False)
    resolved = Column(Boolean, default=False, nullable=False)
    resolution_notes = Column(Text, nullable=True)
    escalated = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())


class ComplianceReport(Base):
    """Database model for compliance reports."""
    __tablename__ = "compliance_reports"
    
    id = Column(Integer, primary_key=True, index=True)
    report_type = Column(String(50), nullable=False)
    report_period = Column(String(50), nullable=False)
    region = Column(String(50), nullable=False)
    generated_by = Column(String(100), nullable=False)
    report_data = Column(Text, nullable=False)  # JSON data
    generated_at = Column(DateTime, server_default=func.now())
    status = Column(String(20), default="generated", nullable=False)


class RegulatoryComplianceManager:
    """Manager for regulatory compliance checks."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self._compliance_rules: Dict[str, RegulatoryRule] = {}
        self._trading_sessions: Dict[str, TradingSession] = {}
        self._position_limits: Dict[str, Decimal] = {}
        self._regional_rules: Dict[RegulatoryRegion, List[RegulatoryRule]] = {}
        
    async def initialize(self) -> None:
        """Initialize the compliance manager."""
        logger.info("Initializing regulatory compliance manager")
        try:
            # Create database tables
            await self._create_tables()
            
            # Load compliance rules
            await self._load_compliance_rules()
            
            # Initialize regional rule sets
            await self._initialize_regional_rules()
            
            logger.info("Regulatory compliance manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize compliance manager: {e}")
            raise
    
    async def _create_tables(self) -> None:
        """Create database tables for compliance."""
        async with self.db_manager.get_session() as session:
            Base.metadata.create_all(bind=session.get_bind())
    
    async def _load_compliance_rules(self) -> None:
        """Load compliance rules from configuration."""
        # This would load from a configuration file or database
        # For now, we'll initialize with default rules
        await self._initialize_default_rules()
    
    async def _initialize_default_rules(self) -> None:
        """Initialize default compliance rules."""
        # US Pattern Day Trader Rule
        pdt_rule = PatternDayTraderRule(
            rule_id="US_PDT_RULE",
            description="Pattern Day Trader Rule - Minimum $25,000 equity for day trading",
            region=RegulatoryRegion.US_SEC,
            minimum_equity=Decimal("25000"),
            day_trade_count_threshold=4,
            rolling_days=5
        )
        self._compliance_rules[pdt_rule.rule_id] = pdt_rule
        
        # US Equity Position Limit (Regulation T)
        us_position_limit = PositionLimitRule(
            rule_id="US_POSITION_LIMIT",
            description="US Equity Position Limit - 50% initial margin requirement",
            region=RegulatoryRegion.US_SEC,
            max_position_percentage=Decimal("0.5")
        )
        self._compliance_rules[us_position_limit.rule_id] = us_position_limit
        
        # EU ESMA Position Limits
        esma_position_limit = PositionLimitRule(
            rule_id="ESMA_POSITION_LIMIT",
            description="ESMA Position Limits - Binary options and CFDs",
            region=RegulatoryRegion.EU_ESMA,
            max_position_value=Decimal("1000000")  # €1M
        )
        self._compliance_rules[esma_position_limit.rule_id] = esma_position_limit
        
        logger.info("Loaded default compliance rules")
    
    async def _initialize_regional_rules(self) -> None:
        """Initialize rules by region."""
        for rule in self._compliance_rules.values():
            if rule.region not in self._regional_rules:
                self._regional_rules[rule.region] = []
            self._regional_rules[rule.region].append(rule)
    
    async def check_pattern_day_trader_compliance(self, account_id: str, user_id: str,
                                                order_size: Decimal, asset_symbol: str,
                                                region: RegulatoryRegion = RegulatoryRegion.US_SEC) -> ComplianceLevel:
        """Check pattern day trader rule compliance."""
        try:
            pdt_rule = self._compliance_rules.get("US_PDT_RULE")
            if not pdt_rule or region != RegulatoryRegion.US_SEC:
                return ComplianceLevel.INFO
            
            # Get or create trading session
            session_key = f"{account_id}_{user_id}"
            session = await self._get_trading_session(session_key, account_id, user_id)
            
            # Check if this would be a day trade
            current_time = datetime.utcnow()
            is_day_trade = await self._is_day_trade(account_id, user_id, asset_symbol, current_time)
            
            if is_day_trade:
                session.day_trades += 1
                session.total_trades += 1
                
                # Check day trade count threshold
                if session.day_trades >= pdt_rule.day_trade_count_threshold:
                    # Check minimum equity requirement
                    if session.equity_balance < pdt_rule.minimum_equity:
                        violation_level = ComplianceLevel.VIOLATION
                        details = f"Pattern Day Trader violation: {session.day_trades} day trades, equity ${session.equity_balance} < minimum ${pdt_rule.minimum_equity}"
                        
                        await self._record_compliance_violation(
                            account_id, user_id, "PATTERN_DAY_TRADER", 
                            violation_level, "US_PDT_RULE", details, 
                            asset_symbol, session.equity_balance, pdt_rule.minimum_equity
                        )
                        return violation_level
                    else:
                        return ComplianceLevel.WARNING
            else:
                session.total_trades += 1
            
            return ComplianceLevel.INFO
            
        except Exception as e:
            logger.error(f"Error checking PDT compliance for {account_id}: {e}")
            return ComplianceLevel.CRITICAL
    
    async def _is_day_trade(self, account_id: str, user_id: str, 
                          asset_symbol: str, current_time: datetime) -> bool:
        """Check if a trade would constitute a day trade."""
        # This is a simplified implementation
        # In practice, you'd query the database for trades in the last 5 business days
        # and check if there's an existing position in the same asset
        return False  # Placeholder
    
    async def _get_trading_session(self, session_key: str, account_id: str, 
                                 user_id: str) -> TradingSession:
        """Get or create trading session."""
        if session_key not in self._trading_sessions:
            self._trading_sessions[session_key] = TradingSession(
                account_id=account_id,
                user_id=user_id,
                session_start=datetime.utcnow()
            )
        return self._trading_sessions[session_key]
    
    async def check_position_limit_compliance(self, account_id: str, user_id: str,
                                            asset_symbol: str, order_size: Decimal,
                                            current_position: Decimal,
                                            region: RegulatoryRegion) -> ComplianceLevel:
        """Check position limit compliance."""
        try:
            # Get applicable position limit rules for the region
            rules = self._regional_rules.get(region, [])
            position_rules = [r for r in rules if isinstance(r, PositionLimitRule)]
            
            highest_level = ComplianceLevel.INFO
            
            for rule in position_rules:
                level = await self._check_single_position_limit(
                    account_id, user_id, asset_symbol, order_size, 
                    current_position, rule
                )
                if level.value > highest_level.value:
                    highest_level = level
            
            return highest_level
            
        except Exception as e:
            logger.error(f"Error checking position limit compliance for {account_id}: {e}")
            return ComplianceLevel.CRITICAL
    
    async def _check_single_position_limit(self, account_id: str, user_id: str,
                                         asset_symbol: str, order_size: Decimal,
                                         current_position: Decimal, 
                                         rule: PositionLimitRule) -> ComplianceLevel:
        """Check a single position limit rule."""
        try:
            new_position = current_position + order_size
            
            # Check absolute position value limit
            if rule.max_position_value and new_position > rule.max_position_value:
                violation_level = ComplianceLevel.VIOLATION
                details = f"Position limit exceeded: ${new_position} > ${rule.max_position_value}"
                
                await self._record_compliance_violation(
                    account_id, user_id, "POSITION_LIMIT", violation_level,
                    rule.rule_id, details, asset_symbol, new_position, rule.max_position_value
                )
                return violation_level
            
            # Check percentage limit (if we had portfolio value)
            if rule.max_position_percentage:
                # This would require portfolio value calculation
                pass
            
            # Check concentration limit
            if rule.concentration_limit:
                # This would require sector/industry analysis
                pass
            
            return ComplianceLevel.INFO
            
        except Exception as e:
            logger.error(f"Error checking position limit rule {rule.rule_id}: {e}")
            return ComplianceLevel.CRITICAL
    
    async def check_order_compliance(self, account_id: str, user_id: str,
                                   asset_symbol: str, order_size: Decimal,
                                   order_type: OrderType, region: RegulatoryRegion) -> ComplianceLevel:
        """Comprehensive order compliance check."""
        try:
            # Check pattern day trader compliance
            pdt_level = await self.check_pattern_day_trader_compliance(
                account_id, user_id, order_size, asset_symbol, region
            )
            
            # Get current position (simplified - would query database)
            current_position = Decimal("0")
            
            # Check position limits
            position_level = await self.check_position_limit_compliance(
                account_id, user_id, asset_symbol, order_size, 
                current_position, region
            )
            
            # Return the highest severity level
            if pdt_level.value > position_level.value:
                return pdt_level
            else:
                return position_level
                
        except Exception as e:
            logger.error(f"Error checking order compliance for {account_id}: {e}")
            return ComplianceLevel.CRITICAL
    
    async def _record_compliance_violation(self, account_id: str, user_id: str,
                                         violation_type: str, level: ComplianceLevel,
                                         rule_id: str, description: str,
                                         asset_symbol: Optional[str] = None,
                                         violation_value: Optional[Decimal] = None,
                                         limit_value: Optional[Decimal] = None) -> None:
        """Record a compliance violation."""
        try:
            # Save compliance check
            async with self.db_manager.get_session() as session:
                check = ComplianceCheck(
                    account_id=account_id,
                    user_id=user_id,
                    rule_id=rule_id,
                    region="US_SEC",  # This would be dynamic
                    check_type=violation_type,
                    asset_symbol=asset_symbol,
                    order_size=violation_value,
                    compliance_level=level.value,
                    violation_details=description
                )
                session.add(check)
                
                # Save violation record if it's a real violation
                if level in [ComplianceLevel.VIOLATION, ComplianceLevel.CRITICAL]:
                    violation = RegulatoryViolation(
                        account_id=account_id,
                        user_id=user_id,
                        violation_type=violation_type,
                        severity=level.value,
                        rule_id=rule_id,
                        description=description,
                        asset_symbol=asset_symbol,
                        violation_value=violation_value,
                        limit_value=limit_value
                    )
                    session.add(violation)
                
                await session.commit()
                
                logger.warning(f"Compliance violation recorded for {account_id}: {description}")
                
        except Exception as e:
            logger.error(f"Error recording compliance violation for {account_id}: {e}")
    
    async def generate_compliance_report(self, report_type: str, 
                                       region: RegulatoryRegion,
                                       period_start: datetime,
                                       period_end: datetime) -> Dict[str, Any]:
        """Generate compliance report for a period and region."""
        try:
            # Get violations for the period
            async with self.db_manager.get_session() as session:
                violations = await session.execute(
                    RegulatoryViolation.__table__.select()
                    .where(
                        RegulatoryViolation.detected_at >= period_start,
                        RegulatoryViolation.detected_at <= period_end
                    )
                )
                
                violations_list = violations.fetchall()
                
                # Get compliance checks for the period
                checks = await session.execute(
                    ComplianceCheck.__table__.select()
                    .where(
                        ComplianceCheck.checked_at >= period_start,
                        ComplianceCheck.checked_at <= period_end
                    )
                )
                
                checks_list = checks.fetchall()
            
            # Generate report data
            report_data = {
                "report_type": report_type,
                "region": region.value,
                "period": {
                    "start": period_start.isoformat(),
                    "end": period_end.isoformat()
                },
                "summary": {
                    "total_violations": len(violations_list),
                    "total_checks": len(checks_list),
                    "violation_rate": len(violations_list) / max(len(checks_list), 1)
                },
                "violations_by_type": {},
                "violations_by_severity": {},
                "top_violators": [],
                "generated_at": datetime.utcnow().isoformat()
            }
            
            # Analyze violations
            for violation in violations_list:
                # By type
                vtype = violation.violation_type
                report_data["violations_by_type"][vtype] = \
                    report_data["violations_by_type"].get(vtype, 0) + 1
                
                # By severity
                severity = violation.severity
                report_data["violations_by_severity"][severity] = \
                    report_data["violations_by_severity"].get(severity, 0) + 1
            
            # Save report to database
            async with self.db_manager.get_session() as session:
                report_record = ComplianceReport(
                    report_type=report_type,
                    report_period=f"{period_start.date()}_{period_end.date()}",
                    region=region.value,
                    report_data=json.dumps(report_data),
                    generated_by="compliance_manager"
                )
                session.add(report_record)
                await session.commit()
            
            logger.info(f"Compliance report generated: {report_type} for {region.value}")
            return report_data
            
        except Exception as e:
            logger.error(f"Error generating compliance report: {e}")
            return {}
    
    async def acknowledge_violation(self, violation_id: int, notes: str) -> bool:
        """Acknowledge a regulatory violation."""
        try:
            async with self.db_manager.get_session() as session:
                await session.execute(
                    RegulatoryViolation.__table__.update()
                    .where(RegulatoryViolation.id == violation_id)
                    .values(acknowledged=True, resolution_notes=notes)
                )
                await session.commit()
                
            logger.info(f"Violation {violation_id} acknowledged")
            return True
            
        except Exception as e:
            logger.error(f"Error acknowledging violation {violation_id}: {e}")
            return False
    
    async def resolve_violation(self, violation_id: int, resolution_notes: str) -> bool:
        """Resolve a regulatory violation."""
        try:
            async with self.db_manager.get_session() as session:
                await session.execute(
                    RegulatoryViolation.__table__.update()
                    .where(RegulatoryViolation.id == violation_id)
                    .values(
                        resolved=True, 
                        resolution_notes=resolution_notes,
                        resolved_at=datetime.utcnow()
                    )
                )
                await session.commit()
                
            logger.info(f"Violation {violation_id} resolved")
            return True
            
        except Exception as e:
            logger.error(f"Error resolving violation {violation_id}: {e}")
            return False
    
    async def get_violations_summary(self, account_id: Optional[str] = None,
                                   days_back: int = 30) -> Dict[str, Any]:
        """Get violations summary."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            
            async with self.db_manager.get_session() as session:
                query = RegulatoryViolation.__table__.select().where(
                    RegulatoryViolation.detected_at >= cutoff_date
                )
                
                if account_id:
                    query = query.where(RegulatoryViolation.account_id == account_id)
                
                violations = await session.execute(query)
                violations_list = violations.fetchall()
            
            # Group by type and severity
            by_type = {}
            by_severity = {}
            unresolved_count = 0
            
            for violation in violations_list:
                vtype = violation.violation_type
                severity = violation.severity
                
                by_type[vtype] = by_type.get(vtype, 0) + 1
                by_severity[severity] = by_severity.get(severity, 0) + 1
                
                if not violation.resolved:
                    unresolved_count += 1
            
            return {
                "total_violations": len(violations_list),
                "unresolved_violations": unresolved_count,
                "by_type": by_type,
                "by_severity": by_severity,
                "period_days": days_back
            }
            
        except Exception as e:
            logger.error(f"Error getting violations summary: {e}")
            return {}
    
    async def add_custom_rule(self, rule: RegulatoryRule) -> None:
        """Add a custom regulatory rule."""
        try:
            self._compliance_rules[rule.rule_id] = rule
            
            if rule.region not in self._regional_rules:
                self._regional_rules[rule.region] = []
            self._regional_rules[rule.region].append(rule)
            
            logger.info(f"Custom rule added: {rule.rule_id}")
            
        except Exception as e:
            logger.error(f"Error adding custom rule {rule.rule_id}: {e}")
    
    async def get_compliance_status(self) -> Dict[str, Any]:
        """Get overall compliance status."""
        try:
            # Get recent violations
            recent_violations = await self.get_violations_summary(days_back=7)
            
            return {
                "active_rules": len(self._compliance_rules),
                "regions_monitored": len(self._regional_rules),
                "recent_violations": recent_violations,
                "active_trading_sessions": len(self._trading_sessions),
                "last_update": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting compliance status: {e}")
            return {}


# Factory function for creating compliance manager
async def create_compliance_manager(db_manager: DatabaseManager) -> RegulatoryComplianceManager:
    """Create a configured compliance manager."""
    manager = RegulatoryComplianceManager(db_manager)
    await manager.initialize()
    
    return manager


# Common compliance rule templates
US_PATTERN_DAY_TRADER_RULES = [
    PatternDayTraderRule(
        rule_id="US_PDT_STANDARD",
        description="Standard US PDT Rule",
        region=RegulatoryRegion.US_SEC,
        minimum_equity=Decimal("25000"),
        day_trade_count_threshold=4,
        rolling_days=5
    )
]

US_POSITION_LIMIT_RULES = [
    PositionLimitRule(
        rule_id="US_EQ_MARGIN",
        description="US Equity Margin Requirements",
        region=RegulatoryRegion.US_SEC,
        max_position_percentage=Decimal("0.5")  # 50% initial margin
    ),
    PositionLimitRule(
        rule_id="US_PDT_MARGIN",
        description="US PDT Margin Requirements",
        region=RegulatoryRegion.US_SEC,
        min_equity=Decimal("25000")
    )
]

EU_ESMA_RULES = [
    PositionLimitRule(
        rule_id="ESMA_CFD_LIMIT",
        description="ESMA CFD Position Limits",
        region=RegulatoryRegion.EU_ESMA,
        max_position_value=Decimal("1000000"),
        concentration_limit=Decimal("0.1")  # 10% concentration limit
    )
]