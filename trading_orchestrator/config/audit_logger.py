"""
Configuration Audit Logging System
Logs all configuration changes, accesses, and operations for security and compliance
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum
import hashlib
import threading
from collections import defaultdict, deque


class AuditEventType(str, Enum):
    """Types of audit events"""
    CONFIG_LOADED = "config_loaded"
    CONFIG_SAVED = "config_saved"
    CONFIG_CHANGED = "config_changed"
    CONFIG_VIEWED = "config_viewed"
    CONFIG_EXPORTED = "config_exported"
    CONFIG_IMPORTED = "config_imported"
    CONFIG_DELETED = "config_deleted"
    CONFIG_ROLLED_BACK = "config_rolled_back"
    KEY_GENERATED = "key_generated"
    KEY_ROTATED = "key_rotated"
    ENCRYPTION_APPLIED = "encryption_applied"
    DECRYPTION_APPLIED = "decryption_applied"
    VALIDATION_ERROR = "validation_error"
    ACCESS_DENIED = "access_denied"
    SYSTEM_ERROR = "system_error"


class AuditSeverity(str, Enum):
    """Audit event severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Individual audit event record"""
    event_id: str
    event_type: AuditEventType
    severity: AuditSeverity
    timestamp: datetime
    user_id: str
    config_name: str
    description: str
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: str = ""
    user_agent: str = ""
    session_id: str = ""
    correlation_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['event_type'] = self.event_type.value
        data['severity'] = self.severity.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEvent':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['event_type'] = AuditEventType(data['event_type'])
        data['severity'] = AuditSeverity(data['severity'])
        return cls(**data)


class ConfigAuditLogger:
    """
    Configuration Audit Logger
    
    Provides comprehensive audit logging for configuration management:
    - All configuration changes
    - User access tracking
    - Security events
    - Error tracking
    - Compliance reporting
    - Real-time monitoring
    """
    
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Log files
        self.main_log_file = self.log_dir / "audit.log"
        self.json_log_file = self.log_dir / "audit.json"
        self.summary_file = self.log_dir / "daily_summary.json"
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Event buffer for real-time processing
        self.event_buffer = deque(maxlen=10000)
        
        # Statistics
        self.event_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        
        # Logging setup
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
    
    def log_config_loaded(self, config_name: str, config_type: str, 
                         size_bytes: int, user_id: str = "system") -> str:
        """Log configuration file loaded event"""
        event = self._create_event(
            event_type=AuditEventType.CONFIG_LOADED,
            severity=AuditSeverity.INFO,
            user_id=user_id,
            config_name=config_name,
            description=f"Configuration loaded: {config_name}",
            details={
                "config_type": config_type,
                "size_bytes": size_bytes,
                "operation": "load"
            }
        )
        
        return self._log_event(event)
    
    def log_config_saved(self, config_name: str, config_type: str, 
                        version_id: str, user_id: str = "system") -> str:
        """Log configuration file saved event"""
        event = self._create_event(
            event_type=AuditEventType.CONFIG_SAVED,
            severity=AuditSeverity.INFO,
            user_id=user_id,
            config_name=config_name,
            description=f"Configuration saved: {config_name}",
            details={
                "config_type": config_type,
                "version_id": version_id,
                "operation": "save"
            }
        )
        
        return self._log_event(event)
    
    def log_config_changed(self, config_name: str, key_path: str, 
                          old_value: Any, new_value: Any, 
                          user_id: str = "system") -> str:
        """Log configuration value change event"""
        # Sanitize values for logging (hide sensitive data)
        safe_old_value = self._sanitize_for_logging(old_value)
        safe_new_value = self._sanitize_for_logging(new_value)
        
        event = self._create_event(
            event_type=AuditEventType.CONFIG_CHANGED,
            severity=AuditSeverity.INFO,
            user_id=user_id,
            config_name=config_name,
            description=f"Configuration changed: {config_name}.{key_path}",
            details={
                "key_path": key_path,
                "old_value": safe_old_value,
                "new_value": safe_new_value,
                "operation": "change"
            }
        )
        
        return self._log_event(event)
    
    def log_config_viewed(self, config_name: str, access_type: str = "read",
                         user_id: str = "system") -> str:
        """Log configuration access event"""
        event = self._create_event(
            event_type=AuditEventType.CONFIG_VIEWED,
            severity=AuditSeverity.INFO,
            user_id=user_id,
            config_name=config_name,
            description=f"Configuration accessed: {config_name}",
            details={
                "access_type": access_type,
                "operation": "view"
            }
        )
        
        return self._log_event(event)
    
    def log_config_exported(self, config_name: str, export_path: str, 
                           user_id: str = "system") -> str:
        """Log configuration export event"""
        event = self._create_event(
            event_type=AuditEventType.CONFIG_EXPORTED,
            severity=AuditSeverity.WARNING,  # Export is potentially risky
            user_id=user_id,
            config_name=config_name,
            description=f"Configuration exported: {config_name}",
            details={
                "export_path": export_path,
                "operation": "export"
            }
        )
        
        return self._log_event(event)
    
    def log_config_imported(self, config_name: str, import_path: str, 
                           user_id: str = "system") -> str:
        """Log configuration import event"""
        event = self._create_event(
            event_type=AuditEventType.CONFIG_IMPORTED,
            severity=AuditSeverity.WARNING,  # Import can be risky
            user_id=user_id,
            config_name=config_name,
            description=f"Configuration imported: {config_name}",
            details={
                "import_path": import_path,
                "operation": "import"
            }
        )
        
        return self._log_event(event)
    
    def log_config_deleted(self, config_name: str, user_id: str = "system") -> str:
        """Log configuration deletion event"""
        event = self._create_event(
            event_type=AuditEventType.CONFIG_DELETED,
            severity=AuditSeverity.ERROR,
            user_id=user_id,
            config_name=config_name,
            description=f"Configuration deleted: {config_name}",
            details={
                "operation": "delete"
            }
        )
        
        return self._log_event(event)
    
    def log_config_rolled_back(self, config_name: str, from_version: str, 
                              to_version: str, user_id: str = "system") -> str:
        """Log configuration rollback event"""
        event = self._create_event(
            event_type=AuditEventType.CONFIG_ROLLED_BACK,
            severity=AuditSeverity.WARNING,
            user_id=user_id,
            config_name=config_name,
            description=f"Configuration rolled back: {config_name}",
            details={
                "from_version": from_version,
                "to_version": to_version,
                "operation": "rollback"
            }
        )
        
        return self._log_event(event)
    
    def log_key_generated(self, key_type: str, method: str, 
                         user_id: str = "system") -> str:
        """Log encryption key generation event"""
        event = self._create_event(
            event_type=AuditEventType.KEY_GENERATED,
            severity=AuditSeverity.WARNING,
            user_id=user_id,
            config_name="encryption",
            description=f"Encryption key generated",
            details={
                "key_type": key_type,
                "method": method,
                "operation": "key_generation"
            }
        )
        
        return self._log_event(event)
    
    def log_key_rotated(self, method: str, user_id: str = "system") -> str:
        """Log encryption key rotation event"""
        event = self._create_event(
            event_type=AuditEventType.KEY_ROTATED,
            severity=AuditSeverity.WARNING,
            user_id=user_id,
            config_name="encryption",
            description=f"Encryption key rotated",
            details={
                "method": method,
                "operation": "key_rotation"
            }
        )
        
        return self._log_event(event)
    
    def log_encryption_applied(self, config_name: str, field_count: int, 
                              method: str, user_id: str = "system") -> str:
        """Log encryption application event"""
        event = self._create_event(
            event_type=AuditEventType.ENCRYPTION_APPLIED,
            severity=AuditSeverity.INFO,
            user_id=user_id,
            config_name=config_name,
            description=f"Encryption applied to {field_count} fields",
            details={
                "field_count": field_count,
                "method": method,
                "operation": "encryption"
            }
        )
        
        return self._log_event(event)
    
    def log_validation_error(self, config_name: str, errors: List[str], 
                           user_id: str = "system") -> str:
        """Log configuration validation errors"""
        event = self._create_event(
            event_type=AuditEventType.VALIDATION_ERROR,
            severity=AuditSeverity.ERROR,
            user_id=user_id,
            config_name=config_name,
            description=f"Configuration validation failed: {len(errors)} errors",
            details={
                "errors": errors,
                "error_count": len(errors),
                "operation": "validation"
            }
        )
        
        return self._log_event(event)
    
    def log_access_denied(self, config_name: str, reason: str, 
                         user_id: str = "unknown") -> str:
        """Log access denied event"""
        event = self._create_event(
            event_type=AuditEventType.ACCESS_DENIED,
            severity=AuditSeverity.WARNING,
            user_id=user_id,
            config_name=config_name,
            description=f"Access denied to configuration",
            details={
                "reason": reason,
                "operation": "access_denied"
            }
        )
        
        return self._log_event(event)
    
    def log_system_error(self, operation: str, error_message: str, 
                        user_id: str = "system") -> str:
        """Log system error event"""
        event = self._create_event(
            event_type=AuditEventType.SYSTEM_ERROR,
            severity=AuditSeverity.ERROR,
            user_id=user_id,
            config_name="system",
            description=f"System error during {operation}",
            details={
                "operation": operation,
                "error_message": error_message,
                "operation_type": "error"
            }
        )
        
        return self._log_event(event)
    
    def get_events(self, config_name: str = None, event_type: AuditEventType = None,
                  start_time: datetime = None, end_time: datetime = None,
                  limit: int = None) -> List[AuditEvent]:
        """Get audit events with optional filtering"""
        events = []
        
        # Load events from JSON log file
        if self.json_log_file.exists():
            try:
                with open(self.json_log_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            event_data = json.loads(line)
                            event = AuditEvent.from_dict(event_data)
                            
                            # Apply filters
                            if config_name and event.config_name != config_name:
                                continue
                            if event_type and event.event_type != event_type:
                                continue
                            if start_time and event.timestamp < start_time:
                                continue
                            if end_time and event.timestamp > end_time:
                                continue
                            
                            events.append(event)
            except Exception as e:
                self.logger.error(f"Failed to load audit events: {e}")
        
        # Add events from buffer
        for event in self.event_buffer:
            if config_name and event.config_name != config_name:
                continue
            if event_type and event.event_type != event_type:
                continue
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue
            
            events.append(event)
        
        # Sort by timestamp (newest first)
        events.sort(key=lambda e: e.timestamp, reverse=True)
        
        # Apply limit
        if limit:
            events = events[:limit]
        
        return events
    
    def get_statistics(self, days: int = 7) -> Dict[str, Any]:
        """Get audit statistics for specified period"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        
        events = self.get_events(start_time=start_time, end_time=end_time)
        
        # Calculate statistics
        stats = {
            "period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "days": days
            },
            "totals": {
                "total_events": len(events),
                "by_type": {},
                "by_severity": {},
                "by_config": {},
                "errors": 0,
                "warnings": 0
            }
        }
        
        for event in events:
            # By type
            event_type = event.event_type.value
            stats["totals"]["by_type"][event_type] = stats["totals"]["by_type"].get(event_type, 0) + 1
            
            # By severity
            severity = event.severity.value
            stats["totals"]["by_severity"][severity] = stats["totals"]["by_severity"].get(severity, 0) + 1
            
            # By config
            config_name = event.config_name
            stats["totals"]["by_config"][config_name] = stats["totals"]["by_config"].get(config_name, 0) + 1
            
            # Errors and warnings
            if severity == "error":
                stats["totals"]["errors"] += 1
            elif severity == "warning":
                stats["totals"]["warnings"] += 1
        
        return stats
    
    def generate_compliance_report(self, start_date: datetime, 
                                  end_date: datetime) -> Dict[str, Any]:
        """Generate compliance report for security audit"""
        events = self.get_events(start_time=start_date, end_time=end_date)
        
        report = {
            "report_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "duration_days": (end_date - start_date).days
            },
            "executive_summary": {},
            "security_events": {
                "encryption_events": 0,
                "key_operations": 0,
                "access_denied": 0,
                "system_errors": 0
            },
            "compliance_checks": {
                "config_changes_logged": True,
                "access_controls_enforced": True,
                "encryption_usage": "compliant",
                "retention_policy_followed": True
            },
            "recommendations": [],
            "detailed_events": []
        }
        
        # Analyze events for compliance
        for event in events:
            # Security events
            if event.event_type in [
                AuditEventType.ENCRYPTION_APPLIED, 
                AuditEventType.DECRYPTION_APPLIED
            ]:
                report["security_events"]["encryption_events"] += 1
            
            if event.event_type in [
                AuditEventType.KEY_GENERATED, 
                AuditEventType.KEY_ROTATED
            ]:
                report["security_events"]["key_operations"] += 1
            
            if event.event_type == AuditEventType.ACCESS_DENIED:
                report["security_events"]["access_denied"] += 1
            
            if event.event_type == AuditEventType.SYSTEM_ERROR:
                report["security_events"]["system_errors"] += 1
            
            # Add critical events to detailed section
            if event.severity == AuditSeverity.CRITICAL:
                report["detailed_events"].append(event.to_dict())
        
        # Generate recommendations
        if report["security_events"]["system_errors"] > 10:
            report["recommendations"].append(
                "High number of system errors detected. Review system stability."
            )
        
        if report["security_events"]["access_denied"] > 50:
            report["recommendations"].append(
                "High number of access denied events. Review user permissions."
            )
        
        return report
    
    def export_audit_log(self, export_path: str, format: str = "json",
                        start_date: datetime = None, end_date: datetime = None) -> bool:
        """Export audit log to file"""
        try:
            events = self.get_events(start_time=start_date, end_time=end_date)
            
            with open(export_path, 'w') as f:
                if format.lower() == "json":
                    for event in events:
                        f.write(json.dumps(event.to_dict()) + '\n')
                elif format.lower() == "csv":
                    # Simple CSV export
                    f.write("timestamp,event_type,severity,user_id,config_name,description\n")
                    for event in events:
                        f.write(f"{event.timestamp.isoformat()},{event.event_type.value},"
                               f"{event.severity.value},{event.user_id},{event.config_name},"
                               f'"{event.description}"\n')
            
            self.logger.info(f"Audit log exported: {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export audit log: {e}")
            return False
    
    def cleanup_old_logs(self, retention_days: int = 90) -> int:
        """Clean up old audit logs"""
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        deleted_count = 0
        
        try:
            # Clean up main log file
            if self.main_log_file.exists():
                with open(self.main_log_file, 'r') as f:
                    lines = f.readlines()
                
                # Keep only recent lines
                recent_lines = []
                for line in reversed(lines):
                    if cutoff_date <= datetime.fromtimestamp(os.path.getmtime(self.main_log_file)):
                        recent_lines.append(line)
                    else:
                        deleted_count += 1
                        break
                
                if deleted_count > 0:
                    recent_lines.reverse()
                    with open(self.main_log_file, 'w') as f:
                        f.writelines(recent_lines)
            
            # Clean up JSON log file (create new file with recent events)
            if self.json_log_file.exists():
                recent_events = self.get_events(start_time=cutoff_date)
                with open(self.json_log_file, 'w') as f:
                    for event in recent_events:
                        f.write(json.dumps(event.to_dict()) + '\n')
            
            self.logger.info(f"Cleaned up {deleted_count} old audit log entries")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old logs: {e}")
            return 0
    
    def _create_event(self, event_type: AuditEventType, severity: AuditSeverity,
                     user_id: str, config_name: str, description: str, 
                     details: Dict[str, Any] = None) -> AuditEvent:
        """Create audit event"""
        event_id = hashlib.md5(
            f"{event_type.value}{config_name}{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]
        
        return AuditEvent(
            event_id=event_id,
            event_type=event_type,
            severity=severity,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            config_name=config_name,
            description=description,
            details=details or {}
        )
    
    def _log_event(self, event: AuditEvent) -> str:
        """Log audit event to all outputs"""
        with self.lock:
            try:
                # Add to buffer
                self.event_buffer.append(event)
                
                # Update statistics
                self.event_counts[event.event_type.value] += 1
                self.error_counts[event.severity.value] += 1
                
                # Write to JSON log file
                with open(self.json_log_file, 'a') as f:
                    f.write(json.dumps(event.to_dict()) + '\n')
                
                # Write to main log file
                self._write_to_main_log(event)
                
                # Write to daily summary
                self._update_daily_summary(event)
                
                return event.event_id
                
            except Exception as e:
                self.logger.error(f"Failed to log audit event: {e}")
                return ""
    
    def _write_to_main_log(self, event: AuditEvent):
        """Write event to main log file"""
        log_entry = (
            f"[{event.timestamp.isoformat()}] {event.severity.value.upper()} "
            f"[{event.event_type.value}] {event.user_id} - {event.config_name} - "
            f"{event.description}"
        )
        
        with open(self.main_log_file, 'a') as f:
            f.write(log_entry + '\n')
    
    def _update_daily_summary(self, event: AuditEvent):
        """Update daily summary statistics"""
        today = datetime.utcnow().date().isoformat()
        
        summary_file = self.summary_file
        summary_data = {}
        
        # Load existing summary
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    summary_data = json.load(f)
            except:
                pass
        
        # Initialize day entry
        if today not in summary_data:
            summary_data[today] = {
                "total_events": 0,
                "by_type": {},
                "by_severity": {},
                "by_config": {}
            }
        
        # Update statistics
        day_data = summary_data[today]
        day_data["total_events"] += 1
        
        event_type = event.event_type.value
        day_data["by_type"][event_type] = day_data["by_type"].get(event_type, 0) + 1
        
        severity = event.severity.value
        day_data["by_severity"][severity] = day_data["by_severity"].get(severity, 0) + 1
        
        config_name = event.config_name
        day_data["by_config"][config_name] = day_data["by_config"].get(config_name, 0) + 1
        
        # Save updated summary
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
    
    def _setup_logging(self):
        """Setup audit logging configuration"""
        # Create separate logger for audit events
        audit_logger = logging.getLogger('config_audit')
        audit_logger.setLevel(logging.INFO)
        
        # Create file handler for audit log
        audit_handler = logging.FileHandler(self.main_log_file)
        audit_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        audit_handler.setFormatter(formatter)
        
        # Add handler to logger
        audit_logger.addHandler(audit_handler)
    
    def _sanitize_for_logging(self, value: Any) -> Any:
        """Sanitize values for safe logging (hide sensitive data)"""
        if isinstance(value, str):
            # Check for common sensitive patterns
            sensitive_patterns = [
                'key', 'secret', 'password', 'token', 'private', 'credential'
            ]
            
            if any(pattern in value.lower() for pattern in sensitive_patterns):
                if len(value) > 8:
                    return value[:4] + '*' * (len(value) - 8) + value[-4:]
                else:
                    return '*' * len(value)
            return value
        
        return value


# Global audit logger instance
_audit_logger = None


def get_audit_logger(log_dir: str = "./config/logs") -> ConfigAuditLogger:
    """Get or create global audit logger instance"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = ConfigAuditLogger(Path(log_dir))
    return _audit_logger