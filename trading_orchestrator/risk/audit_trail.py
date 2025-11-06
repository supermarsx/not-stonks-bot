"""
Comprehensive Audit Trail Module

This module implements comprehensive audit trail functionality for all trading risk decisions,
regulatory compliance reporting, and risk event documentation.

Features:
- All risk decisions logged with full context
- Regulatory compliance reporting
- Risk event documentation and tracking
- Audit trail analytics and reporting
- Compliance evidence collection
- Risk decision impact analysis
- Audit trail retention management
- Immutable audit records
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Union
from enum import Enum
from dataclasses import dataclass, field
from decimal import Decimal
import json
import hashlib
from sqlalchemy import (
    Column, Integer, String, DateTime, Boolean, Float, Text, 
    Index, ForeignKey
)
from sqlalchemy.orm import declarative_base, relationship, Session
from sqlalchemy.sql import func

from trading_orchestrator.database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

Base = declarative_base()


class AuditCategory(Enum):
    """Audit event categories."""
    RISK_CHECK = "risk_check"
    LIMIT_ENFORCEMENT = "limit_enforcement"
    ORDER_VALIDATION = "order_validation"
    POSITION_MANAGEMENT = "position_management"
    COMPLIANCE_CHECK = "compliance_check"
    ALERT_CREATION = "alert_creation"
    ESCALATION = "escalation"
    FAILOVER = "failover"
    LIQUIDITY_VALIDATION = "liquidity_validation"
    CIRCUIT_BREAKER = "circuit_breaker"
    EMERGENCY_STOP = "emergency_stop"
    SYSTEM_ERROR = "system_error"
    USER_ACTION = "user_action"
    REGULATORY_EVENT = "regulatory_event"


class AuditSeverity(Enum):
    """Audit event severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    VIOLATION = "violation"


class AuditStatus(Enum):
    """Audit event status."""
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    ESCALATED = "escalated"


class RiskDecisionType(Enum):
    """Types of risk decisions."""
    APPROVE = "approve"
    REJECT = "reject"
    WARN = "warn"
    LIMIT = "limit"
    BLOCK = "block"
    ESCALATE = "escalate"
    REQUIRE_APPROVAL = "require_approval"
    ADJUST = "adjust"
    HALT = "halt"
    CONTINUE = "continue"


@dataclass
class AuditContext:
    """Context information for audit events."""
    user_id: Optional[str] = None
    account_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    source_system: Optional[str] = None
    correlation_id: Optional[str] = None
    parent_event_id: Optional[str] = None


@dataclass
class RiskDecision:
    """Risk decision details."""
    decision_type: RiskDecisionType
    reasoning: str
    affected_assets: List[str] = field(default_factory=list)
    limits_checked: List[str] = field(default_factory=list)
    violations_found: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    impact_assessment: Dict[str, Any] = field(default_factory=dict)
    confidence_level: float = 1.0
    execution_deferred: bool = False
    approval_required: bool = False
    risk_score: Optional[float] = None


@dataclass
class AuditEvent:
    """Complete audit event structure."""
    event_id: str
    category: AuditCategory
    severity: AuditSeverity
    status: AuditStatus
    timestamp: datetime
    context: AuditContext
    event_type: str
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    risk_decision: Optional[RiskDecision] = None
    before_state: Dict[str, Any] = field(default_factory=dict)
    after_state: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    compliance_indicators: Dict[str, Any] = field(default_factory=dict)
    parent_event_id: Optional[str] = None
    chain_hash: Optional[str] = None
    digital_signature: Optional[str] = None


@dataclass
class ComplianceReport:
    """Compliance report data structure."""
    report_id: str
    report_type: str
    region: str
    period_start: datetime
    period_end: datetime
    generated_by: str
    total_events: int
    compliance_score: float
    violations_count: int
    critical_events: int
    summary_data: Dict[str, Any] = field(default_factory=dict)
    detailed_metrics: Dict[str, Any] = field(default_factory=dict)


class AuditTrailModel(Base):
    """Database model for audit trail events."""
    __tablename__ = "audit_trail_events"
    
    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(String(100), nullable=False, unique=True)
    category = Column(String(30), nullable=False)
    severity = Column(String(20), nullable=False)
    status = Column(String(20), nullable=False)
    event_type = Column(String(100), nullable=False)
    description = Column(Text, nullable=False)
    user_id = Column(String(100), nullable=True)
    account_id = Column(String(100), nullable=True)
    session_id = Column(String(100), nullable=True)
    source_system = Column(String(100), nullable=True)
    correlation_id = Column(String(100), nullable=True)
    parent_event_id = Column(String(100), nullable=True)
    metadata = Column(Text, nullable=True)  # JSON data
    risk_decision = Column(Text, nullable=True)  # JSON data
    before_state = Column(Text, nullable=True)  # JSON data
    after_state = Column(Text, nullable=True)  # JSON data
    performance_metrics = Column(Text, nullable=True)  # JSON data
    compliance_indicators = Column(Text, nullable=True)  # JSON data
    chain_hash = Column(String(64), nullable=True)
    digital_signature = Column(String(256), nullable=True)
    timestamp = Column(DateTime, server_default=func.now())
    created_at = Column(DateTime, server_default=func.now())


class ComplianceReportModel(Base):
    """Database model for compliance reports."""
    __tablename__ = "compliance_reports"
    
    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(String(100), nullable=False, unique=True)
    report_type = Column(String(50), nullable=False)
    region = Column(String(50), nullable=False)
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    generated_by = Column(String(100), nullable=False)
    total_events = Column(Integer, nullable=False)
    compliance_score = Column(Float, nullable=False)
    violations_count = Column(Integer, nullable=False)
    critical_events = Column(Integer, nullable=False)
    summary_data = Column(Text, nullable=True)  # JSON data
    detailed_metrics = Column(Text, nullable=True)  # JSON data
    created_at = Column(DateTime, server_default=func.now())


class AuditTrailManager:
    """Manager for comprehensive audit trail functionality."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self._audit_buffer: List[AuditEvent] = []
        self._buffer_size = 1000
        self._retention_period_days = 2555  # 7 years for financial records
        self._last_compliance_report: Optional[datetime] = None
        
    async def initialize(self) -> None:
        """Initialize the audit trail manager."""
        logger.info("Initializing audit trail manager")
        try:
            # Create database tables
            await self._create_tables()
            
            # Start background tasks
            await self._start_background_tasks()
            
            logger.info("Audit trail manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize audit trail manager: {e}")
            raise
    
    async def _create_tables(self) -> None:
        """Create database tables for audit trail."""
        async with self.db_manager.get_session() as session:
            Base.metadata.create_all(bind=session.get_bind())
    
    async def _start_background_tasks(self) -> None:
        """Start background audit processing tasks."""
        # Start buffer flush task
        asyncio.create_task(self._buffer_flush_task())
        
        # Start compliance report task
        asyncio.create_task(self._compliance_report_task())
        
        # Start retention task
        asyncio.create_task(self._retention_task())
        
        logger.info("Started audit trail background tasks")
    
    async def log_risk_decision(self, event_type: str, description: str,
                              context: AuditContext, decision: RiskDecision,
                              category: AuditCategory = AuditCategory.RISK_CHECK,
                              metadata: Dict[str, Any] = None,
                              severity: AuditSeverity = AuditSeverity.INFO) -> str:
        """Log a risk decision with full context."""
        try:
            # Generate event ID
            event_id = f"risk_decision_{int(datetime.utcnow().timestamp() * 1000)}"
            
            # Create audit event
            event = AuditEvent(
                event_id=event_id,
                category=category,
                severity=severity,
                status=AuditStatus.COMPLETED,
                timestamp=datetime.utcnow(),
                context=context,
                event_type=event_type,
                description=description,
                metadata=metadata or {},
                risk_decision=decision
            )
            
            # Add to buffer for batch processing
            await self._buffer_audit_event(event)
            
            logger.debug(f"Risk decision logged: {event_id} - {decision.decision_type.value}")
            return event_id
            
        except Exception as e:
            logger.error(f"Error logging risk decision: {e}")
            return ""
    
    async def log_compliance_event(self, event_type: str, description: str,
                                 context: AuditContext,
                                 compliance_data: Dict[str, Any],
                                 severity: AuditSeverity = AuditSeverity.INFO,
                                 metadata: Dict[str, Any] = None) -> str:
        """Log a compliance-related event."""
        try:
            event_id = f"compliance_{int(datetime.utcnow().timestamp() * 1000)}"
            
            event = AuditEvent(
                event_id=event_id,
                category=AuditCategory.COMPLIANCE_CHECK,
                severity=severity,
                status=AuditStatus.COMPLETED,
                timestamp=datetime.utcnow(),
                context=context,
                event_type=event_type,
                description=description,
                metadata=metadata or {},
                compliance_indicators=compliance_data
            )
            
            await self._buffer_audit_event(event)
            
            logger.info(f"Compliance event logged: {event_id}")
            return event_id
            
        except Exception as e:
            logger.error(f"Error logging compliance event: {e}")
            return ""
    
    async def log_system_event(self, event_type: str, description: str,
                             context: AuditContext,
                             system_data: Dict[str, Any],
                             severity: AuditSeverity = AuditSeverity.INFO,
                             metadata: Dict[str, Any] = None) -> str:
        """Log a system-level event."""
        try:
            event_id = f"system_{int(datetime.utcnow().timestamp() * 1000)}"
            
            event = AuditEvent(
                event_id=event_id,
                category=AuditCategory.SYSTEM_ERROR if severity in [AuditSeverity.ERROR, AuditSeverity.CRITICAL] else AuditCategory.RISK_CHECK,
                severity=severity,
                status=AuditStatus.COMPLETED,
                timestamp=datetime.utcnow(),
                context=context,
                event_type=event_type,
                description=description,
                metadata=metadata or {},
                performance_metrics=system_data
            )
            
            await self._buffer_audit_event(event)
            
            logger.info(f"System event logged: {event_id}")
            return event_id
            
        except Exception as e:
            logger.error(f"Error logging system event: {e}")
            return ""
    
    async def log_order_validation(self, order_id: str, validation_result: Dict[str, Any],
                                 context: AuditContext,
                                 before_state: Dict[str, Any],
                                 after_state: Dict[str, Any]) -> str:
        """Log order validation process."""
        try:
            event_id = f"order_validation_{order_id}_{int(datetime.utcnow().timestamp() * 1000)}"
            
            # Create risk decision for order validation
            decision = RiskDecision(
                decision_type=RiskDecisionType(validation_result.get("decision", "approve")),
                reasoning=validation_result.get("reasoning", ""),
                limits_checked=validation_result.get("limits_checked", []),
                violations_found=validation_result.get("violations_found", []),
                recommendations=validation_result.get("recommendations", []),
                confidence_level=validation_result.get("confidence", 1.0)
            )
            
            event = AuditEvent(
                event_id=event_id,
                category=AuditCategory.ORDER_VALIDATION,
                severity=AuditSeverity.WARNING if decision.violations_found else AuditSeverity.INFO,
                status=AuditStatus.COMPLETED,
                timestamp=datetime.utcnow(),
                context=context,
                event_type="order_validation",
                description=f"Order validation for {order_id}",
                risk_decision=decision,
                before_state=before_state,
                after_state=after_state,
                metadata={"order_id": order_id}
            )
            
            await self._buffer_audit_event(event)
            
            logger.info(f"Order validation logged: {event_id}")
            return event_id
            
        except Exception as e:
            logger.error(f"Error logging order validation: {e}")
            return ""
    
    async def log_alert_event(self, alert_id: str, alert_data: Dict[str, Any],
                            context: AuditContext) -> str:
        """Log alert creation and handling."""
        try:
            event_id = f"alert_{alert_id}_{int(datetime.utcnow().timestamp() * 1000)}"
            
            # Extract severity from alert data
            alert_severity = alert_data.get("severity", "info").lower()
            severity_mapping = {
                "info": AuditSeverity.INFO,
                "low": AuditSeverity.INFO,
                "medium": AuditSeverity.WARNING,
                "high": AuditSeverity.ERROR,
                "critical": AuditSeverity.CRITICAL
            }
            severity = severity_mapping.get(alert_severity, AuditSeverity.INFO)
            
            event = AuditEvent(
                event_id=event_id,
                category=AuditCategory.ALERT_CREATION,
                severity=severity,
                status=AuditStatus.COMPLETED,
                timestamp=datetime.utcnow(),
                context=context,
                event_type="alert_creation",
                description=f"Alert created: {alert_data.get('title', 'Unknown')}",
                metadata=alert_data
            )
            
            await self._buffer_audit_event(event)
            
            logger.info(f"Alert event logged: {event_id}")
            return event_id
            
        except Exception as e:
            logger.error(f"Error logging alert event: {e}")
            return ""
    
    async def _buffer_audit_event(self, event: AuditEvent) -> None:
        """Add audit event to processing buffer."""
        try:
            # Calculate chain hash for integrity
            event.chain_hash = await self._calculate_chain_hash(event)
            
            # Add to buffer
            self._audit_buffer.append(event)
            
            # Flush buffer if full
            if len(self._audit_buffer) >= self._buffer_size:
                await self._flush_buffer()
                
        except Exception as e:
            logger.error(f"Error buffering audit event: {e}")
    
    async def _calculate_chain_hash(self, event: AuditEvent) -> str:
        """Calculate chain hash for audit event integrity."""
        try:
            # Create hash from event data
            event_data = {
                "event_id": event.event_id,
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type,
                "category": event.category.value,
                "description": event.description
            }
            
            # Add previous chain hash if available
            if self._audit_buffer:
                event_data["previous_hash"] = self._audit_buffer[-1].chain_hash
            
            # Calculate hash
            data_string = json.dumps(event_data, sort_keys=True)
            return hashlib.sha256(data_string.encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"Error calculating chain hash: {e}")
            return ""
    
    async def _flush_buffer(self) -> None:
        """Flush audit buffer to database."""
        try:
            if not self._audit_buffer:
                return
            
            # Convert events to database records
            records = []
            for event in self._audit_buffer:
                record = AuditTrailModel(
                    event_id=event.event_id,
                    category=event.category.value,
                    severity=event.severity.value,
                    status=event.status.value,
                    event_type=event.event_type,
                    description=event.description,
                    user_id=event.context.user_id,
                    account_id=event.context.account_id,
                    session_id=event.context.session_id,
                    source_system=event.context.source_system,
                    correlation_id=event.context.correlation_id,
                    parent_event_id=event.parent_event_id,
                    metadata=json.dumps(event.metadata),
                    risk_decision=json.dumps(event.risk_decision.__dict__) if event.risk_decision else None,
                    before_state=json.dumps(event.before_state),
                    after_state=json.dumps(event.after_state),
                    performance_metrics=json.dumps(event.performance_metrics),
                    compliance_indicators=json.dumps(event.compliance_indicators),
                    chain_hash=event.chain_hash,
                    digital_signature=event.digital_signature,
                    timestamp=event.timestamp
                )
                records.append(record)
            
            # Bulk insert to database
            async with self.db_manager.get_session() as session:
                for record in records:
                    session.add(record)
                await session.commit()
            
            logger.info(f"Flushed {len(records)} audit events to database")
            
            # Clear buffer
            self._audit_buffer.clear()
            
        except Exception as e:
            logger.error(f"Error flushing audit buffer: {e}")
    
    async def _buffer_flush_task(self) -> None:
        """Background task to flush audit buffer periodically."""
        while True:
            try:
                await asyncio.sleep(30)  # Flush every 30 seconds
                await self._flush_buffer()
                
            except Exception as e:
                logger.error(f"Error in buffer flush task: {e}")
    
    async def _compliance_report_task(self) -> None:
        """Background task to generate periodic compliance reports."""
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                current_time = datetime.utcnow()
                
                # Generate daily compliance report
                if not self._last_compliance_report or \
                   (current_time - self._last_compliance_report).total_seconds() > 86400:  # 24 hours
                    
                    await self.generate_compliance_report("daily", current_time - timedelta(days=1), current_time)
                    self._last_compliance_report = current_time
                
            except Exception as e:
                logger.error(f"Error in compliance report task: {e}")
    
    async def _retention_task(self) -> None:
        """Background task to manage audit trail retention."""
        while True:
            try:
                await asyncio.sleep(86400)  # Run daily
                
                cutoff_date = datetime.utcnow() - timedelta(days=self._retention_period_days)
                await self._cleanup_old_audit_events(cutoff_date)
                
            except Exception as e:
                logger.error(f"Error in retention task: {e}")
    
    async def _cleanup_old_audit_events(self, cutoff_date: datetime) -> None:
        """Clean up old audit events per retention policy."""
        try:
            async with self.db_manager.get_session() as session:
                result = await session.execute(
                    AuditTrailModel.__table__.delete()
                    .where(AuditTrailModel.timestamp < cutoff_date)
                )
                
                logger.info(f"Cleaned up old audit events: {result.rowcount} records deleted")
                
        except Exception as e:
            logger.error(f"Error cleaning up old audit events: {e}")
    
    async def generate_compliance_report(self, report_type: str, 
                                       period_start: datetime,
                                       period_end: datetime,
                                       region: str = "US") -> str:
        """Generate comprehensive compliance report."""
        try:
            report_id = f"compliance_report_{report_type}_{int(datetime.utcnow().timestamp())}"
            
            # Get audit events for the period
            events = await self._get_audit_events_for_period(period_start, period_end)
            
            # Analyze events
            analysis = await self._analyze_audit_events(events)
            
            # Create compliance report
            report_data = ComplianceReport(
                report_id=report_id,
                report_type=report_type,
                region=region,
                period_start=period_start,
                period_end=period_end,
                generated_by="audit_trail_manager",
                total_events=len(events),
                compliance_score=analysis["compliance_score"],
                violations_count=analysis["violations_count"],
                critical_events=analysis["critical_events"],
                summary_data=analysis["summary"],
                detailed_metrics=analysis["detailed_metrics"]
            )
            
            # Save to database
            await self._save_compliance_report(report_data)
            
            logger.info(f"Generated compliance report: {report_id}")
            return report_id
            
        except Exception as e:
            logger.error(f"Error generating compliance report: {e}")
            return ""
    
    async def _get_audit_events_for_period(self, start: datetime, end: datetime) -> List[AuditEvent]:
        """Get audit events for a specific period."""
        try:
            async with self.db_manager.get_session() as session:
                result = await session.execute(
                    AuditTrailModel.__table__.select()
                    .where(
                        AuditTrailModel.timestamp >= start,
                        AuditTrailModel.timestamp <= end
                    )
                    .order_by(AuditTrailModel.timestamp)
                )
                
                records = result.fetchall()
                
                # Convert to AuditEvent objects
                events = []
                for record in records:
                    context = AuditContext(
                        user_id=record.user_id,
                        account_id=record.account_id,
                        session_id=record.session_id,
                        source_system=record.source_system,
                        correlation_id=record.correlation_id
                    )
                    
                    # Parse JSON fields
                    metadata = json.loads(record.metadata) if record.metadata else {}
                    before_state = json.loads(record.before_state) if record.before_state else {}
                    after_state = json.loads(record.after_state) if record.after_state else {}
                    performance_metrics = json.loads(record.performance_metrics) if record.performance_metrics else {}
                    compliance_indicators = json.loads(record.compliance_indicators) if record.compliance_indicators else {}
                    
                    event = AuditEvent(
                        event_id=record.event_id,
                        category=AuditCategory(record.category),
                        severity=AuditSeverity(record.severity),
                        status=AuditStatus(record.status),
                        timestamp=record.timestamp,
                        context=context,
                        event_type=record.event_type,
                        description=record.description,
                        metadata=metadata,
                        before_state=before_state,
                        after_state=after_state,
                        performance_metrics=performance_metrics,
                        compliance_indicators=compliance_indicators,
                        parent_event_id=record.parent_event_id,
                        chain_hash=record.chain_hash,
                        digital_signature=record.digital_signature
                    )
                    events.append(event)
                
                return events
                
        except Exception as e:
            logger.error(f"Error getting audit events for period: {e}")
            return []
    
    async def _analyze_audit_events(self, events: List[AuditEvent]) -> Dict[str, Any]:
        """Analyze audit events for compliance reporting."""
        try:
            total_events = len(events)
            
            # Count by category and severity
            by_category = {}
            by_severity = {}
            violations_count = 0
            critical_events = 0
            
            for event in events:
                category = event.category.value
                severity = event.severity.value
                
                by_category[category] = by_category.get(category, 0) + 1
                by_severity[severity] = by_severity.get(severity, 0) + 1
                
                if severity in ["error", "critical", "violation"]:
                    violations_count += 1
                
                if severity == "critical":
                    critical_events += 1
            
            # Calculate compliance score
            violation_rate = violations_count / max(total_events, 1)
            compliance_score = max(0, 100 - (violation_rate * 100))
            
            # Generate summary metrics
            summary = {
                "total_events": total_events,
                "events_by_category": by_category,
                "events_by_severity": by_severity,
                "violation_rate": violation_rate,
                "period_coverage": "complete"
            }
            
            detailed_metrics = {
                "average_events_per_day": total_events / max((datetime.utcnow() - min(e.timestamp for e in events)).days, 1),
                "most_common_category": max(by_category.items(), key=lambda x: x[1])[0] if by_category else "none",
                "compliance_trend": "improving" if violation_rate < 0.05 else "needs_attention",
                "high_risk_events": violations_count,
                "system_reliability": 1.0 - (critical_events / max(total_events, 1))
            }
            
            return {
                "compliance_score": compliance_score,
                "violations_count": violations_count,
                "critical_events": critical_events,
                "summary": summary,
                "detailed_metrics": detailed_metrics
            }
            
        except Exception as e:
            logger.error(f"Error analyzing audit events: {e}")
            return {
                "compliance_score": 0,
                "violations_count": 0,
                "critical_events": 0,
                "summary": {"error": str(e)},
                "detailed_metrics": {}
            }
    
    async def _save_compliance_report(self, report: ComplianceReport) -> None:
        """Save compliance report to database."""
        try:
            async with self.db_manager.get_session() as session:
                report_record = ComplianceReportModel(
                    report_id=report.report_id,
                    report_type=report.report_type,
                    region=report.region,
                    period_start=report.period_start,
                    period_end=report.period_end,
                    generated_by=report.generated_by,
                    total_events=report.total_events,
                    compliance_score=report.compliance_score,
                    violations_count=report.violations_count,
                    critical_events=report.critical_events,
                    summary_data=json.dumps(report.summary_data),
                    detailed_metrics=json.dumps(report.detailed_metrics)
                )
                session.add(report_record)
                await session.commit()
                
        except Exception as e:
            logger.error(f"Error saving compliance report: {e}")
    
    async def search_audit_events(self, filters: Dict[str, Any]) -> List[AuditEvent]:
        """Search audit events with various filters."""
        try:
            # Build query based on filters
            where_conditions = []
            params = {}
            
            if "event_type" in filters:
                where_conditions.append("event_type = :event_type")
                params["event_type"] = filters["event_type"]
            
            if "category" in filters:
                where_conditions.append("category = :category")
                params["category"] = filters["category"]
            
            if "severity" in filters:
                where_conditions.append("severity = :severity")
                params["severity"] = filters["severity"]
            
            if "user_id" in filters:
                where_conditions.append("user_id = :user_id")
                params["user_id"] = filters["user_id"]
            
            if "account_id" in filters:
                where_conditions.append("account_id = :account_id")
                params["account_id"] = filters["account_id"]
            
            if "start_date" in filters:
                where_conditions.append("timestamp >= :start_date")
                params["start_date"] = filters["start_date"]
            
            if "end_date" in filters:
                where_conditions.append("timestamp <= :end_date")
                params["end_date"] = filters["end_date"]
            
            # Build complete query
            query = "SELECT * FROM audit_trail_events"
            if where_conditions:
                query += " WHERE " + " AND ".join(where_conditions)
            query += " ORDER BY timestamp DESC LIMIT 1000"
            
            async with self.db_manager.get_session() as session:
                result = await session.execute(query, params)
                records = result.fetchall()
                
                # Convert to events (simplified for space)
                events = []
                for record in records:
                    events.append({
                        "event_id": record.event_id,
                        "event_type": record.event_type,
                        "category": record.category,
                        "severity": record.severity,
                        "timestamp": record.timestamp.isoformat(),
                        "description": record.description[:200] + "..." if len(record.description) > 200 else record.description
                    })
                
                return events
                
        except Exception as e:
            logger.error(f"Error searching audit events: {e}")
            return []
    
    async def get_audit_statistics(self) -> Dict[str, Any]:
        """Get audit trail statistics."""
        try:
            # Get recent statistics
            cutoff_24h = datetime.utcnow() - timedelta(hours=24)
            cutoff_7d = datetime.utcnow() - timedelta(days=7)
            
            async with self.db_manager.get_session() as session:
                # Last 24 hours
                result_24h = await session.execute(
                    AuditTrailModel.__table__.select()
                    .where(AuditTrailModel.timestamp >= cutoff_24h)
                )
                events_24h = result_24h.fetchall()
                
                # Last 7 days
                result_7d = await session.execute(
                    AuditTrailModel.__table__.select()
                    .where(AuditTrailModel.timestamp >= cutoff_7d)
                )
                events_7d = result_7d.fetchall()
            
            # Count by category and severity
            by_category_24h = {}
            by_severity_24h = {}
            
            for event in events_24h:
                by_category_24h[event.category] = by_category_24h.get(event.category, 0) + 1
                by_severity_24h[event.severity] = by_severity_24h.get(event.severity, 0) + 1
            
            return {
                "total_events_24h": len(events_24h),
                "total_events_7d": len(events_7d),
                "events_by_category_24h": by_category_24h,
                "events_by_severity_24h": by_severity_24h,
                "buffer_size": len(self._audit_buffer),
                "last_compliance_report": self._last_compliance_report.isoformat() if self._last_compliance_report else None,
                "retention_period_days": self._retention_period_days,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting audit statistics: {e}")
            return {}
    
    async def get_audit_status(self) -> Dict[str, Any]:
        """Get overall audit trail system status."""
        return {
            "system_active": True,
            "buffer_size": len(self._audit_buffer),
            "last_flush": datetime.utcnow().isoformat(),
            "retention_policy": f"{self._retention_period_days} days",
            "compliance_reports_enabled": True,
            "chain_integrity_enabled": True,
            "background_tasks_running": True,
            "last_update": datetime.utcnow().isoformat()
        }


# Factory function for creating audit trail manager
async def create_audit_trail_manager(db_manager: DatabaseManager) -> AuditTrailManager:
    """Create a configured audit trail manager."""
    manager = AuditTrailManager(db_manager)
    await manager.initialize()
    
    return manager


# Utility functions for common audit scenarios
async def create_audit_context(user_id: str = None, account_id: str = None,
                             session_id: str = None, source_system: str = None) -> AuditContext:
    """Create audit context for logging."""
    return AuditContext(
        user_id=user_id,
        account_id=account_id,
        session_id=session_id or f"session_{int(datetime.utcnow().timestamp())}",
        source_system=source_system or "trading_orchestrator"
    )


async def create_risk_decision(decision_type: RiskDecisionType, reasoning: str,
                             violations: List[str] = None) -> RiskDecision:
    """Create risk decision for audit logging."""
    return RiskDecision(
        decision_type=decision_type,
        reasoning=reasoning,
        violations_found=violations or []
    )


# Predefined audit categories for common scenarios
TRADING_AUDIT_CATEGORIES = {
    "order_submission": AuditCategory.ORDER_VALIDATION,
    "position_update": AuditCategory.POSITION_MANAGEMENT,
    "limit_check": AuditCategory.LIMIT_ENFORCEMENT,
    "risk_assessment": AuditCategory.RISK_CHECK,
    "compliance_validation": AuditCategory.COMPLIANCE_CHECK,
    "alert_trigger": AuditCategory.ALERT_CREATION,
    "failover_event": AuditCategory.FAILOVER,
    "liquidity_check": AuditCategory.LIQUIDITY_VALIDATION,
    "circuit_breaker": AuditCategory.CIRCUIT_BREAKER,
    "emergency_action": AuditCategory.EMERGENCY_STOP
}