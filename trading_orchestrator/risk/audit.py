"""
Audit Logging System

Provides comprehensive audit trail for all trading activities:
- Trade execution logs
- Risk management actions
- Policy violations
- System access logs
- Compliance events
"""

import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc
import traceback

from config.database import get_db

logger = logging.getLogger(__name__)


class AuditLogger:
    """
    Manages comprehensive audit logging for all trading activities.
    """
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.db = get_db()
        
        # Audit log types
        self.audit_types = {
            "trade_submission": "Order submitted to broker",
            "trade_execution": "Trade executed",
            "trade_cancellation": "Order cancelled",
            "risk_action": "Risk management action taken",
            "policy_violation": "Policy violation detected",
            "compliance_check": "Compliance check performed",
            "circuit_breaker": "Circuit breaker triggered",
            "system_access": "System access event",
            "configuration_change": "Configuration parameter changed",
            "user_action": "User-initiated action",
            "error_event": "System error or exception"
        }
        
        logger.info(f"AuditLogger initialized for user {self.user_id}")
    
    async def log_trade_submission(self, order_data: Dict[str, Any], 
                                 broker_response: Dict[str, Any]) -> None:
        """
        Log order submission to broker.
        
        Args:
            order_data: Order information
            broker_response: Broker API response
        """
        try:
            await self._create_audit_log(
                event_type="trade_submission",
                event_data={
                    "order": order_data,
                    "broker_response": broker_response,
                    "timestamp": datetime.now().isoformat()
                },
                severity="info"
            )
            
            logger.info(f"Trade submission logged for user {self.user_id}: {order_data.get('symbol')}")
            
        except Exception as e:
            logger.error(f"Trade submission audit log error: {str(e)}")
    
    async def log_trade_execution(self, trade_data: Dict[str, Any], 
                                execution_details: Dict[str, Any]) -> None:
        """
        Log trade execution.
        
        Args:
            trade_data: Trade information
            execution_details: Execution details from broker
        """
        try:
            await self._create_audit_log(
                event_type="trade_execution",
                event_data={
                    "trade": trade_data,
                    "execution": execution_details,
                    "timestamp": datetime.now().isoformat()
                },
                severity="info"
            )
            
            logger.info(f"Trade execution logged for user {self.user_id}: {trade_data.get('symbol')}")
            
        except Exception as e:
            logger.error(f"Trade execution audit log error: {str(e)}")
    
    async def log_trade_cancellation(self, order_id: int, cancellation_reason: str,
                                   cancellation_details: Dict[str, Any] = None) -> None:
        """
        Log order cancellation.
        
        Args:
            order_id: ID of cancelled order
            cancellation_reason: Reason for cancellation
            cancellation_details: Additional cancellation details
        """
        try:
            await self._create_audit_log(
                event_type="trade_cancellation",
                event_data={
                    "order_id": order_id,
                    "cancellation_reason": cancellation_reason,
                    "cancellation_details": cancellation_details or {},
                    "timestamp": datetime.now().isoformat()
                },
                severity="info"
            )
            
            logger.info(f"Trade cancellation logged for user {self.user_id}: Order {order_id}")
            
        except Exception as e:
            logger.error(f"Trade cancellation audit log error: {str(e)}")
    
    async def log_risk_action(self, action_type: str, action_details: Dict[str, Any],
                            severity: str = "warning") -> None:
        """
        Log risk management action.
        
        Args:
            action_type: Type of risk action
            action_details: Details of the action taken
            severity: Log severity level
        """
        try:
            await self._create_audit_log(
                event_type="risk_action",
                event_data={
                    "action_type": action_type,
                    "action_details": action_details,
                    "timestamp": datetime.now().isoformat()
                },
                severity=severity
            )
            
            logger.log(getattr(logging, severity.upper()), 
                      f"Risk action logged for user {self.user_id}: {action_type}")
            
        except Exception as e:
            logger.error(f"Risk action audit log error: {str(e)}")
    
    async def log_policy_violation(self, policy_name: str, violation_details: Dict[str, Any],
                                 severity: str = "warning") -> None:
        """
        Log policy violation.
        
        Args:
            policy_name: Name of violated policy
            violation_details: Details of the violation
            severity: Log severity level
        """
        try:
            await self._create_audit_log(
                event_type="policy_violation",
                event_data={
                    "policy_name": policy_name,
                    "violation_details": violation_details,
                    "timestamp": datetime.now().isoformat()
                },
                severity=severity
            )
            
            logger.log(getattr(logging, severity.upper()), 
                      f"Policy violation logged for user {self.user_id}: {policy_name}")
            
        except Exception as e:
            logger.error(f"Policy violation audit log error: {str(e)}")
    
    async def log_compliance_check(self, check_type: str, check_result: Dict[str, Any],
                                 order_data: Dict[str, Any] = None) -> None:
        """
        Log compliance check result.
        
        Args:
            check_type: Type of compliance check
            check_result: Result of the compliance check
            order_data: Associated order data (if applicable)
        """
        try:
            event_data = {
                "check_type": check_type,
                "check_result": check_result,
                "timestamp": datetime.now().isoformat()
            }
            
            if order_data:
                event_data["order_data"] = order_data
            
            await self._create_audit_log(
                event_type="compliance_check",
                event_data=event_data,
                severity="info" if check_result.get("approved", True) else "warning"
            )
            
        except Exception as e:
            logger.error(f"Compliance check audit log error: {str(e)}")
    
    async def log_circuit_breaker(self, breaker_name: str, trigger_reason: str,
                                trigger_details: Dict[str, Any],
                                severity: str = "critical") -> None:
        """
        Log circuit breaker activation.
        
        Args:
            breaker_name: Name of circuit breaker
            trigger_reason: Reason for trigger
            trigger_details: Additional trigger details
            severity: Log severity level
        """
        try:
            await self._create_audit_log(
                event_type="circuit_breaker",
                event_data={
                    "breaker_name": breaker_name,
                    "trigger_reason": trigger_reason,
                    "trigger_details": trigger_details,
                    "timestamp": datetime.now().isoformat()
                },
                severity=severity
            )
            
            logger.critical(f"Circuit breaker logged for user {self.user_id}: {breaker_name}")
            
        except Exception as e:
            logger.error(f"Circuit breaker audit log error: {str(e)}")
    
    async def log_system_access(self, access_type: str, details: Dict[str, Any],
                              ip_address: str = None, user_agent: str = None) -> None:
        """
        Log system access event.
        
        Args:
            access_type: Type of access (login, logout, api_call, etc.)
            details: Access details
            ip_address: Source IP address
            user_agent: User agent string
        """
        try:
            event_data = {
                "access_type": access_type,
                "details": details,
                "timestamp": datetime.now().isoformat()
            }
            
            if ip_address:
                event_data["ip_address"] = ip_address
            if user_agent:
                event_data["user_agent"] = user_agent
            
            await self._create_audit_log(
                event_type="system_access",
                event_data=event_data,
                severity="info"
            )
            
        except Exception as e:
            logger.error(f"System access audit log error: {str(e)}")
    
    async def log_configuration_change(self, config_key: str, old_value: Any, 
                                     new_value: Any, change_reason: str = None) -> None:
        """
        Log configuration parameter change.
        
        Args:
            config_key: Configuration key that was changed
            old_value: Previous value
            new_value: New value
            change_reason: Reason for the change
        """
        try:
            await self._create_audit_log(
                event_type="configuration_change",
                event_data={
                    "config_key": config_key,
                    "old_value": old_value,
                    "new_value": new_value,
                    "change_reason": change_reason,
                    "timestamp": datetime.now().isoformat()
                },
                severity="info"
            )
            
            logger.info(f"Configuration change logged for user {self.user_id}: {config_key}")
            
        except Exception as e:
            logger.error(f"Configuration change audit log error: {str(e)}")
    
    async def log_user_action(self, action: str, action_details: Dict[str, Any],
                            severity: str = "info") -> None:
        """
        Log user-initiated action.
        
        Args:
            action: Description of the action
            action_details: Details of the action
            severity: Log severity level
        """
        try:
            await self._create_audit_log(
                event_type="user_action",
                event_data={
                    "action": action,
                    "action_details": action_details,
                    "timestamp": datetime.now().isoformat()
                },
                severity=severity
            )
            
        except Exception as e:
            logger.error(f"User action audit log error: {str(e)}")
    
    async def log_error_event(self, error_type: str, error_message: str,
                            error_details: Dict[str, Any] = None,
                            traceback_info: str = None) -> None:
        """
        Log system error or exception.
        
        Args:
            error_type: Type of error
            error_message: Error message
            error_details: Additional error details
            traceback_info: Stack trace information
        """
        try:
            event_data = {
                "error_type": error_type,
                "error_message": error_message,
                "timestamp": datetime.now().isoformat()
            }
            
            if error_details:
                event_data["error_details"] = error_details
            
            if traceback_info:
                event_data["traceback"] = traceback_info
            
            await self._create_audit_log(
                event_type="error_event",
                event_data=event_data,
                severity="error"
            )
            
            logger.error(f"Error event logged for user {self.user_id}: {error_type}")
            
        except Exception as e:
            logger.error(f"Error event audit log error: {str(e)}")
    
    async def log_trade_approval(self, order_data: Dict[str, Any]) -> None:
        """
        Log trade approval by risk management.
        
        Args:
            order_data: Approved order data
        """
        try:
            await self._create_audit_log(
                event_type="trade_approval",
                event_data={
                    "order_data": order_data,
                    "approval_timestamp": datetime.now().isoformat(),
                    "approved_by": "risk_management_system"
                },
                severity="info"
            )
            
            logger.info(f"Trade approval logged for user {self.user_id}: {order_data.get('symbol')}")
            
        except Exception as e:
            logger.error(f"Trade approval audit log error: {str(e)}")
    
    async def get_audit_trail(self, start_date: datetime, end_date: datetime,
                            event_types: List[str] = None, 
                            limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Retrieve audit trail for a date range.
        
        Args:
            start_date: Start date for audit trail
            end_date: End date for audit trail
            event_types: Filter by event types (optional)
            limit: Maximum number of records to return
            
        Returns:
            List of audit log entries
        """
        try:
            # Build query
            query = self.db.query(AuditLog).filter(
                and_(
                    AuditLog.user_id == self.user_id,
                    AuditLog.timestamp >= start_date,
                    AuditLog.timestamp <= end_date
                )
            )
            
            if event_types:
                query = query.filter(AuditLog.event_type.in_(event_types))
            
            # Get results
            logs = query.order_by(desc(AuditLog.timestamp)).limit(limit).all()
            
            audit_trail = []
            for log in logs:
                audit_trail.append({
                    "id": log.id,
                    "event_type": log.event_type,
                    "event_description": log.event_description,
                    "severity": log.severity,
                    "timestamp": log.timestamp,
                    "event_data": log.event_data or {},
                    "ip_address": log.ip_address,
                    "user_agent": log.user_agent
                })
            
            return audit_trail
            
        except Exception as e:
            logger.error(f"Audit trail retrieval error: {str(e)}")
            return []
    
    async def export_audit_log(self, start_date: datetime, end_date: datetime,
                             format_type: str = "json") -> str:
        """
        Export audit log in specified format.
        
        Args:
            start_date: Start date for export
            end_date: End date for export
            format_type: Export format (json, csv)
            
        Returns:
            Exported audit log data
        """
        try:
            audit_trail = await self.get_audit_trail(start_date, end_date)
            
            if format_type.lower() == "json":
                return json.dumps(audit_trail, indent=2, default=str)
            elif format_type.lower() == "csv":
                return self._format_audit_as_csv(audit_trail)
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
            
        except Exception as e:
            logger.error(f"Audit log export error: {str(e)}")
            raise
    
    async def generate_audit_summary(self, start_date: datetime, 
                                   end_date: datetime) -> Dict[str, Any]:
        """
        Generate audit summary for a date range.
        
        Args:
            start_date: Start date for summary
            end_date: End date for summary
            
        Returns:
            Audit summary statistics
        """
        try:
            audit_trail = await self.get_audit_trail(start_date, end_date)
            
            summary = {
                "period": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "total_days": (end_date - start_date).days + 1
                },
                "totals": {
                    "total_events": len(audit_trail),
                    "trade_events": 0,
                    "risk_events": 0,
                    "compliance_events": 0,
                    "error_events": 0,
                    "system_events": 0
                },
                "by_severity": {
                    "info": 0,
                    "warning": 0,
                    "error": 0,
                    "critical": 0
                },
                "by_event_type": {},
                "daily_activity": {}
            }
            
            # Analyze events
            for event in audit_trail:
                event_type = event["event_type"]
                severity = event["severity"]
                
                # Count by category
                if event_type in ["trade_submission", "trade_execution", "trade_cancellation", "trade_approval"]:
                    summary["totals"]["trade_events"] += 1
                elif event_type in ["risk_action", "circuit_breaker"]:
                    summary["totals"]["risk_events"] += 1
                elif event_type in ["compliance_check", "policy_violation"]:
                    summary["totals"]["compliance_events"] += 1
                elif event_type == "error_event":
                    summary["totals"]["error_events"] += 1
                else:
                    summary["totals"]["system_events"] += 1
                
                # Count by severity
                summary["by_severity"][severity] = summary["by_severity"].get(severity, 0) + 1
                
                # Count by event type
                summary["by_event_type"][event_type] = summary["by_event_type"].get(event_type, 0) + 1
                
                # Daily activity
                event_date = event["timestamp"].date()
                if event_date not in summary["daily_activity"]:
                    summary["daily_activity"][event_date] = 0
                summary["daily_activity"][event_date] += 1
            
            return summary
            
        except Exception as e:
            logger.error(f"Audit summary generation error: {str(e)}")
            return {"error": str(e)}
    
    async def _create_audit_log(self, event_type: str, event_data: Dict[str, Any],
                              severity: str = "info", ip_address: str = None,
                              user_agent: str = None) -> None:
        """
        Create audit log entry in database.
        
        Args:
            event_type: Type of event
            event_data: Event data
            severity: Log severity
            ip_address: Source IP (optional)
            user_agent: User agent (optional)
        """
        try:
            audit_log = AuditLog(
                user_id=self.user_id,
                event_type=event_type,
                event_description=self.audit_types.get(event_type, "Unknown event type"),
                event_data=event_data,
                severity=severity,
                ip_address=ip_address,
                user_agent=user_agent,
                timestamp=datetime.now()
            )
            
            self.db.add(audit_log)
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Audit log creation error: {str(e)}")
            # Don't raise - audit logging should not break main functionality
    
    def _format_audit_as_csv(self, audit_trail: List[Dict[str, Any]]) -> str:
        """Format audit trail as CSV."""
        try:
            import csv
            from io import StringIO
            
            output = StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow([
                "ID", "Event Type", "Description", "Severity", 
                "Timestamp", "Event Data", "IP Address", "User Agent"
            ])
            
            # Write data
            for event in audit_trail:
                writer.writerow([
                    event["id"],
                    event["event_type"],
                    event["event_description"],
                    event["severity"],
                    event["timestamp"],
                    json.dumps(event["event_data"]),
                    event.get("ip_address", ""),
                    event.get("user_agent", "")
                ])
            
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"CSV formatting error: {str(e)}")
            return "Error formatting CSV"
    
    def close(self):
        """Cleanup resources."""
        self.db.close()


# Audit Log model (would normally be in database/models/, but included here for completeness)
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.sql import func
from config.database import Base


class AuditLog(Base):
    """Audit log table for tracking all system events."""
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    
    # Event details
    event_type = Column(String(50), nullable=False, index=True)
    event_description = Column(Text, nullable=True)
    event_data = Column(JSON, default=dict)
    
    # Metadata
    severity = Column(String(20), default="info", nullable=False)
    ip_address = Column(String(45), nullable=True)  # IPv6 support
    user_agent = Column(Text, nullable=True)
    
    # Timestamp
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    def __repr__(self):
        return f"<AuditLog(id={self.id}, type='{self.event_type}', severity='{self.severity}')>"
