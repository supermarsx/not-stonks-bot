"""
Compliance Engine for API Rate Limiting

Provides compliance monitoring, audit logging, and regulatory features
for API rate limiting across all broker integrations.
"""

import asyncio
import hashlib
import json
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import threading
from pathlib import Path

from ..core.rate_limiter import RateLimiterManager, RequestType
from ..core.request_manager import RequestManager
from ..brokers.rate_limit_configs import RateLimitConfig
from ..monitoring.monitor import RateLimitMonitor, Metric


class ComplianceStatus(Enum):
    """Compliance status levels"""
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"
    CRITICAL = "critical"


class LogLevel(Enum):
    """Logging levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditEventType(Enum):
    """Types of audit events"""
    REQUEST_SUBMITTED = "request_submitted"
    REQUEST_APPROVED = "request_approved"
    REQUEST_REJECTED = "request_rejected"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    CIRCUIT_BREAKER_TRIPPED = "circuit_breaker_tripped"
    AUTHENTICATION_FAILED = "authentication_failed"
    API_KEY_ROTATION = "api_key_rotation"
    COMPLIANCE_VIOLATION = "compliance_violation"
    COST_THRESHOLD_EXCEEDED = "cost_threshold_exceeded"
    MANUAL_INTERVENTION = "manual_intervention"


@dataclass
class AuditEvent:
    """Audit event record"""
    id: str
    timestamp: datetime
    broker: str
    event_type: AuditEventType
    level: LogLevel
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "broker": self.broker,
            "event_type": self.event_type.value,
            "level": self.level.value,
            "user_id": self.user_id,
            "request_id": self.request_id,
            "details": self.details,
            "metadata": self.metadata,
            "hash": self._calculate_hash()
        }
    
    def _calculate_hash(self) -> str:
        """Calculate hash for event integrity"""
        event_data = json.dumps({
            "timestamp": self.timestamp.isoformat(),
            "broker": self.broker,
            "event_type": self.event_type.value,
            "details": self.details
        }, sort_keys=True)
        
        return hashlib.sha256(event_data.encode()).hexdigest()


@dataclass
class ComplianceRule:
    """Compliance rule definition"""
    id: str
    name: str
    description: str
    broker_filter: Optional[str] = None  # None = applies to all brokers
    request_types: Optional[List[RequestType]] = None
    conditions: Dict[str, Any] = field(default_factory=dict)
    action: str = "log"  # log, block, alert, escalate
    severity: LogLevel = LogLevel.WARNING
    
    def applies_to(self, broker: str, request_type: RequestType) -> bool:
        """Check if rule applies to given broker and request type"""
        if self.broker_filter and self.broker_filter != broker:
            return False
        
        if self.request_types and request_type not in self.request_types:
            return False
        
        return True
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate rule against context"""
        for key, expected_value in self.conditions.items():
            if key not in context:
                return False
            
            actual_value = context[key]
            
            # Handle different comparison types
            if isinstance(expected_value, dict):
                operator = expected_value.get("operator", "equals")
                expected = expected_value.get("value")
                
                if operator == "equals":
                    if actual_value != expected:
                        return False
                elif operator == "greater_than":
                    if not (actual_value > expected):
                        return False
                elif operator == "less_than":
                    if not (actual_value < expected):
                        return False
                elif operator == "in":
                    if actual_value not in expected:
                        return False
                elif operator == "not_in":
                    if actual_value in expected:
                        return False
            else:
                # Simple equality check
                if actual_value != expected_value:
                    return False
        
        return True


@dataclass
class CostBreakdown:
    """Cost breakdown by category"""
    broker: str
    period_start: datetime
    period_end: datetime
    
    # Request costs
    market_data_cost: float = 0.0
    trading_cost: float = 0.0
    account_info_cost: float = 0.0
    other_costs: float = 0.0
    
    # Request counts
    market_data_requests: int = 0
    trading_requests: int = 0
    account_info_requests: int = 0
    other_requests: int = 0
    
    # Rate limit cost (estimated impact)
    rate_limit_penalty_cost: float = 0.0
    retry_cost: float = 0.0
    
    def total_cost(self) -> float:
        """Calculate total cost"""
        return (self.market_data_cost + self.trading_cost + 
                self.account_info_cost + self.other_costs +
                self.rate_limit_penalty_cost + self.retry_cost)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "broker": self.broker,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "market_data_cost": self.market_data_cost,
            "trading_cost": self.trading_cost,
            "account_info_cost": self.account_info_cost,
            "other_costs": self.other_costs,
            "rate_limit_penalty_cost": self.rate_limit_penalty_cost,
            "retry_cost": self.retry_cost,
            "total_cost": self.total_cost(),
            "market_data_requests": self.market_data_requests,
            "trading_requests": self.trading_requests,
            "account_info_requests": self.account_info_requests,
            "other_requests": self.other_requests,
            "cost_per_request": self.total_cost() / max(1, self.total_requests())
        }
    
    def total_requests(self) -> int:
        """Get total request count"""
        return (self.market_data_requests + self.trading_requests + 
                self.account_info_requests + self.other_requests)


class APILogger:
    """API request logging and audit system"""
    
    def __init__(self, storage_path: Optional[str] = None, retention_days: int = 90):
        self.storage_path = Path(storage_path) if storage_path else Path("./api_audit_logs")
        self.retention_days = retention_days
        
        # Setup storage
        self.storage_path.mkdir(exist_ok=True)
        
        # Event storage
        self._audit_events: deque = deque(maxlen=100000)
        self._event_lock = threading.RLock()
        
        # Event listeners
        self._listeners: List[Callable[[AuditEvent], None]] = []
        
        # Current user context
        self._current_user: Optional[str] = None
        
        # Setup default compliance rules
        self._compliance_rules = self._setup_default_compliance_rules()
    
    def _setup_default_compliance_rules(self) -> List[ComplianceRule]:
        """Setup default compliance rules"""
        return [
            # Rate limit compliance
            ComplianceRule(
                id="rate_limit_basic",
                name="Basic Rate Limit Compliance",
                description="Ensure requests respect broker rate limits",
                conditions={
                    "rate_limit_exceeded": {"operator": "equals", "value": False}
                },
                action="log"
            ),
            
            # Trading hours compliance
            ComplianceRule(
                id="trading_hours",
                name="Trading Hours Compliance",
                description="Ensure trading occurs only during allowed hours",
                conditions={
                    "is_trading_hours": {"operator": "equals", "value": True}
                },
                action="block"
            ),
            
            # High-frequency trading detection
            ComplianceRule(
                id="hft_detection",
                name="High Frequency Trading Detection",
                description="Detect potential high-frequency trading patterns",
                conditions={
                    "requests_per_second": {"operator": "greater_than", "value": 10}
                },
                action="alert"
            ),
            
            # Cost control
            ComplianceRule(
                id="daily_cost_limit",
                name="Daily Cost Limit",
                description="Ensure daily API costs don't exceed limits",
                conditions={
                    "daily_cost": {"operator": "less_than", "value": 100.0}
                },
                action="alert"
            ),
            
            # Authentication security
            ComplianceRule(
                id="api_key_age",
                name="API Key Age Check",
                description="Ensure API keys are not too old",
                conditions={
                    "api_key_age_days": {"operator": "less_than", "value": 90}
                },
                action="alert"
            )
        ]
    
    def set_current_user(self, user_id: str):
        """Set current user context"""
        self._current_user = user_id
    
    def log_event(
        self,
        broker: str,
        event_type: AuditEventType,
        level: LogLevel,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        **metadata
    ) -> str:
        """Log audit event"""
        event = AuditEvent(
            id=f"{int(time.time() * 1000000)}",
            timestamp=datetime.utcnow(),
            broker=broker,
            event_type=event_type,
            level=level,
            user_id=self._current_user,
            request_id=request_id,
            details=details or {},
            metadata=metadata
        )
        
        with self._event_lock:
            self._audit_events.append(event)
        
        # Notify listeners
        for listener in self._listeners:
            try:
                listener(event)
            except Exception as e:
                print(f"Event listener error: {e}")
        
        # Write to file
        self._write_event_to_file(event)
        
        return event.id
    
    def log_request_submitted(self, broker: str, request_type: RequestType, **details):
        """Log request submission"""
        return self.log_event(
            broker=broker,
            event_type=AuditEventType.REQUEST_SUBMITTED,
            level=LogLevel.INFO,
            request_type=request_type.value,
            **details
        )
    
    def log_request_approved(self, broker: str, request_id: str, **details):
        """Log request approval"""
        return self.log_event(
            broker=broker,
            event_type=AuditEventType.REQUEST_APPROVED,
            level=LogLevel.INFO,
            request_id=request_id,
            **details
        )
    
    def log_request_rejected(self, broker: str, request_id: str, reason: str, **details):
        """Log request rejection"""
        return self.log_event(
            broker=broker,
            event_type=AuditEventType.REQUEST_REJECTED,
            level=LogLevel.WARNING,
            request_id=request_id,
            reason=reason,
            **details
        )
    
    def log_rate_limit_exceeded(self, broker: str, limit_type: str, current_usage: float, limit: float, **details):
        """Log rate limit exceeded"""
        return self.log_event(
            broker=broker,
            event_type=AuditEventType.RATE_LIMIT_EXCEEDED,
            level=LogLevel.WARNING,
            limit_type=limit_type,
            current_usage=current_usage,
            limit=limit,
            **details
        )
    
    def log_circuit_breaker_tripped(self, broker: str, reason: str, **details):
        """Log circuit breaker trip"""
        return self.log_event(
            broker=broker,
            event_type=AuditEventType.CIRCUIT_BREAKER_TRIPPED,
            level=LogLevel.CRITICAL,
            reason=reason,
            **details
        )
    
    def log_authentication_failure(self, broker: str, failure_reason: str, **details):
        """Log authentication failure"""
        return self.log_event(
            broker=broker,
            event_type=AuditEventType.AUTHENTICATION_FAILED,
            level=LogLevel.ERROR,
            failure_reason=failure_reason,
            **details
        )
    
    def log_compliance_violation(self, broker: str, rule_id: str, violation_details: Dict[str, Any]):
        """Log compliance violation"""
        return self.log_event(
            broker=broker,
            event_type=AuditEventType.COMPLIANCE_VIOLATION,
            level=LogLevel.ERROR,
            rule_id=rule_id,
            violation_details=violation_details
        )
    
    def _write_event_to_file(self, event: AuditEvent):
        """Write event to daily log file"""
        try:
            date_str = event.timestamp.strftime("%Y-%m-%d")
            file_path = self.storage_path / f"audit_{date_str}.jsonl"
            
            with open(file_path, 'a') as f:
                f.write(json.dumps(event.to_dict()) + '\n')
        except Exception as e:
            print(f"Error writing audit event to file: {e}")
    
    def add_listener(self, listener: Callable[[AuditEvent], None]):
        """Add event listener"""
        self._listeners.append(listener)
    
    def get_events(
        self,
        broker: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        level: Optional[LogLevel] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[AuditEvent]:
        """Get audit events with filters"""
        with self._event_lock:
            events = list(self._audit_events)
        
        # Apply filters
        if broker:
            events = [e for e in events if e.broker == broker]
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if level:
            events = [e for e in events if e.level == level]
        
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        
        # Sort by timestamp (most recent first) and limit
        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:limit]
    
    def get_compliance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate compliance report"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        events = self.get_events(start_time=cutoff_time)
        
        # Count events by type and severity
        violations = [e for e in events if e.event_type == AuditEventType.COMPLIANCE_VIOLATION]
        rate_limits = [e for e in events if e.event_type == AuditEventType.RATE_LIMIT_EXCEEDED]
        
        report = {
            "period_hours": hours,
            "total_events": len(events),
            "compliance_violations": len(violations),
            "rate_limit_exceeded": len(rate_limits),
            "events_by_type": {},
            "events_by_level": {},
            "events_by_broker": {},
            "top_violations": [],
            "recommendations": []
        }
        
        # Event type distribution
        for event_type in AuditEventType:
            count = len([e for e in events if e.event_type == event_type])
            if count > 0:
                report["events_by_type"][event_type.value] = count
        
        # Level distribution
        for level in LogLevel:
            count = len([e for e in events if e.level == level])
            if count > 0:
                report["events_by_level"][level.value] = count
        
        # Broker distribution
        brokers = set(e.broker for e in events)
        for broker in brokers:
            broker_events = [e for e in events if e.broker == broker]
            report["events_by_broker"][broker] = len(broker_events)
        
        # Top violations
        report["top_violations"] = [
            {
                "type": v.event_type.value,
                "level": v.level.value,
                "details": v.details,
                "timestamp": v.timestamp.isoformat()
            }
            for v in violations[:10]
        ]
        
        # Recommendations
        if len(violations) > 10:
            report["recommendations"].append("High number of compliance violations - review rate limiting configuration")
        
        if len(rate_limits) > 100:
            report["recommendations"].append("Frequent rate limit exceeded - consider adjusting limits or request patterns")
        
        if report["events_by_level"].get("critical", 0) > 0:
            report["recommendations"].append("Critical events detected - immediate attention required")
        
        return report


class CostOptimizer:
    """Cost optimization for API usage"""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._cost_thresholds = {
            "daily_warning": 50.0,
            "daily_critical": 100.0,
            "monthly_warning": 1000.0,
            "monthly_critical": 2000.0
        }
    
    def calculate_cost_impact(self, request_count: int, request_type: RequestType) -> float:
        """Calculate cost impact of requests"""
        if not self.config.market_data_fee_per_request:
            return 0.0
        
        base_cost = request_count * self.config.market_data_fee_per_request
        
        # Apply multiplier based on request type
        multipliers = {
            RequestType.MARKET_DATA: 1.0,
            RequestType.HISTORICAL_DATA: 1.5,
            RequestType.REAL_TIME_DATA: 2.0,
            RequestType.ORDER_PLACE: 0.1,  # Lower cost for trading ops
            RequestType.ACCOUNT_INFO: 0.05,
            RequestType.POSITION_QUERY: 0.05
        }
        
        multiplier = multipliers.get(request_type, 1.0)
        return base_cost * multiplier
    
    def optimize_request_pattern(
        self,
        current_requests: Dict[RequestType, int],
        time_window_hours: float
    ) -> Dict[str, Any]:
        """Provide optimization recommendations for request patterns"""
        total_requests = sum(current_requests.values())
        
        # Calculate current cost
        current_cost = sum(
            self.calculate_cost_impact(count, req_type)
            for req_type, count in current_requests.items()
        )
        
        recommendations = []
        optimizations = {}
        
        # High-cost request analysis
        high_cost_requests = []
        for req_type, count in current_requests.items():
            cost_per_request = self.calculate_cost_impact(1, req_type)
            if cost_per_request > 0.01:  # Significant cost threshold
                high_cost_requests.append((req_type, count, cost_per_request))
        
        if high_cost_requests:
            recommendations.append(
                "Consider batching high-cost requests or reducing frequency"
            )
        
        # Market data optimization
        market_data_count = current_requests.get(RequestType.MARKET_DATA, 0)
        if market_data_count > 100:
            optimizations["reduce_market_data_frequency"] = {
                "current_count": market_data_count,
                "recommended": max(10, market_data_count // 10),
                "savings_estimate": self.calculate_cost_impact(market_data_count - max(10, market_data_count // 10), RequestType.MARKET_DATA)
            }
        
        # Historical data optimization
        historical_count = current_requests.get(RequestType.HISTORICAL_DATA, 0)
        if historical_count > 50:
            recommendations.append(
                "Historical data requests are expensive - consider caching or requesting less frequently"
            )
        
        # Rate limit efficiency
        requests_per_minute = total_requests / (time_window_hours * 60)
        if requests_per_minute > self.config.global_rate_limit * 0.8:
            recommendations.append(
                "High request rate detected - consider implementing request queuing or batching"
            )
        
        # Calculate potential savings
        estimated_savings = sum(opt.get("savings_estimate", 0) for opt in optimizations.values())
        
        return {
            "current_cost": current_cost,
            "total_requests": total_requests,
            "requests_per_minute": requests_per_minute,
            "recommendations": recommendations,
            "optimizations": optimizations,
            "estimated_savings": estimated_savings,
            "cost_efficiency_score": self._calculate_cost_efficiency(current_requests, current_cost)
        }
    
    def _calculate_cost_efficiency(
        self,
        requests: Dict[RequestType, int],
        total_cost: float
    ) -> float:
        """Calculate cost efficiency score (0-100)"""
        if total_cost == 0:
            return 100.0
        
        # Base score
        score = 100.0
        
        # Penalize expensive request types
        expensive_requests = [
            RequestType.HISTORICAL_DATA,
            RequestType.REAL_TIME_DATA,
            RequestType.MARKET_DATA
        ]
        
        expensive_count = sum(requests.get(req_type, 0) for req_type in expensive_requests)
        total_count = sum(requests.values())
        
        if total_count > 0:
            expensive_ratio = expensive_count / total_count
            score -= expensive_ratio * 30  # Penalty for high ratio of expensive requests
        
        # Penalty for high cost per request
        cost_per_request = total_cost / max(1, total_count)
        if cost_per_request > 0.1:
            score -= min(20, cost_per_request * 100)
        
        return max(0.0, min(100.0, score))
    
    def generate_cost_alert(self, current_cost: float, period_hours: float) -> Optional[Dict[str, Any]]:
        """Generate cost alert based on thresholds"""
        # Calculate period cost
        if period_hours >= 24:
            daily_cost = current_cost / (period_hours / 24)
        else:
            daily_cost = current_cost * (24 / period_hours)
        
        monthly_cost = daily_cost * 30
        
        alerts = []
        
        if daily_cost >= self._cost_thresholds["daily_critical"]:
            alerts.append({
                "severity": "critical",
                "type": "daily_cost_critical",
                "current_cost": daily_cost,
                "threshold": self._cost_thresholds["daily_critical"],
                "message": f"Daily API costs (${daily_cost:.2f}) exceed critical threshold (${self._cost_thresholds['daily_critical']:.2f})"
            })
        elif daily_cost >= self._cost_thresholds["daily_warning"]:
            alerts.append({
                "severity": "warning",
                "type": "daily_cost_warning",
                "current_cost": daily_cost,
                "threshold": self._cost_thresholds["daily_warning"],
                "message": f"Daily API costs (${daily_cost:.2f}) exceed warning threshold (${self._cost_thresholds['daily_warning']:.2f})"
            })
        
        if monthly_cost >= self._cost_thresholds["monthly_critical"]:
            alerts.append({
                "severity": "critical",
                "type": "monthly_cost_critical",
                "current_cost": monthly_cost,
                "threshold": self._cost_thresholds["monthly_critical"],
                "message": f"Monthly API costs (${monthly_cost:.2f}) exceed critical threshold (${self._cost_thresholds['monthly_critical']:.2f})"
            })
        elif monthly_cost >= self._cost_thresholds["monthly_warning"]:
            alerts.append({
                "severity": "warning",
                "type": "monthly_cost_warning",
                "current_cost": monthly_cost,
                "threshold": self._cost_thresholds["monthly_warning"],
                "message": f"Monthly API costs (${monthly_cost:.2f}) exceed warning threshold (${self._cost_thresholds['monthly_warning']:.2f})"
            })
        
        return alerts[0] if alerts else None


class ComplianceEngine:
    """
    Central compliance engine
    
    Coordinates compliance monitoring, audit logging, and cost optimization
    across all broker integrations.
    """
    
    def __init__(self):
        self._api_logger = APILogger()
        self._cost_optimizers: Dict[str, CostOptimizer] = {}
        self._broker_configs: Dict[str, RateLimitConfig] = {}
        
        # Compliance status
        self._status_by_broker: Dict[str, ComplianceStatus] = {}
        
        # Event handlers
        self._handlers: Dict[AuditEventType, List[Callable]] = defaultdict(list)
        
        # Setup default handlers
        self._setup_default_handlers()
    
    def _setup_default_handlers(self):
        """Setup default event handlers"""
        # Rate limit exceeded handler
        async def handle_rate_limit_exceeded(event: AuditEvent):
            if event.broker in self._broker_configs:
                config = self._broker_configs[event.broker]
                if hasattr(config, 'warning_threshold'):
                    # Could implement automatic rate limiting adjustment here
                    pass
        
        self._handlers[AuditEventType.RATE_LIMIT_EXCEEDED].append(handle_rate_limit_exceeded)
        
        # Circuit breaker handler
        async def handle_circuit_breaker(event: AuditEvent):
            # Log circuit breaker trip and potentially notify
            print(f"CIRCUIT BREAKER TRIPPED: {event.broker} - {event.details}")
        
        self._handlers[AuditEventType.CIRCUIT_BREAKER_TRIPPED].append(handle_circuit_breaker)
        
        # Compliance violation handler
        async def handle_compliance_violation(event: AuditEvent):
            # Update compliance status
            self._status_by_broker[event.broker] = ComplianceStatus.VIOLATION
        
        self._handlers[AuditEventType.COMPLIANCE_VIOLATION].append(handle_compliance_violation)
    
    def add_broker_config(self, broker_name: str, config: RateLimitConfig):
        """Add broker configuration"""
        self._broker_configs[broker_name] = config
        self._cost_optimizers[broker_name] = CostOptimizer(config)
    
    def check_compliance(
        self,
        broker: str,
        request_type: RequestType,
        context: Dict[str, Any]
    ) -> ComplianceStatus:
        """Check compliance for request"""
        config = self._broker_configs.get(broker)
        if not config:
            return ComplianceStatus.COMPLIANT
        
        # Check API logger compliance rules
        violations = 0
        for rule in self._api_logger._compliance_rules:
            if rule.applies_to(broker, request_type):
                if not rule.evaluate(context):
                    violations += 1
                    # Log violation
                    self._api_logger.log_compliance_violation(
                        broker=broker,
                        rule_id=rule.id,
                        violation_details={
                            "rule_name": rule.name,
                            "context": context,
                            "conditions": rule.conditions
                        }
                    )
        
        # Check rate limit specific compliance
        if context.get("rate_limit_exceeded", False):
            violations += 1
            self._api_logger.log_rate_limit_exceeded(
                broker=broker,
                limit_type="broker_specific",
                current_usage=context.get("current_usage", 0),
                limit=context.get("limit", 0)
            )
        
        # Determine status
        if violations == 0:
            status = ComplianceStatus.COMPLIANT
        elif violations <= 2:
            status = ComplianceStatus.WARNING
        else:
            status = ComplianceStatus.VIOLATION
        
        self._status_by_broker[broker] = status
        return status
    
    def get_cost_optimization(
        self,
        broker: str,
        requests: Dict[RequestType, int],
        time_window_hours: float
    ) -> Optional[Dict[str, Any]]:
        """Get cost optimization recommendations"""
        optimizer = self._cost_optimizers.get(broker)
        if not optimizer:
            return None
        
        return optimizer.optimize_request_pattern(requests, time_window_hours)
    
    def get_compliance_status(self, broker: Optional[str] = None) -> Dict[str, Any]:
        """Get current compliance status"""
        if broker:
            return {
                "broker": broker,
                "status": self._status_by_broker.get(broker, ComplianceStatus.COMPLIANT).value,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        return {
            "brokers": {
                broker: status.value
                for broker, status in self._status_by_broker.items()
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_audit_report(self, hours: int = 24) -> Dict[str, Any]:
        """Get audit report"""
        return self._api_logger.get_compliance_report(hours)
    
    async def handle_event(self, event: AuditEvent):
        """Handle audit event"""
        # Call registered handlers
        for handler in self._handlers[event.event_type]:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                print(f"Event handler error: {e}")
    
    def export_compliance_data(self, file_path: str, hours: int = 24):
        """Export compliance data"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        export_data = {
            "export_time": datetime.utcnow().isoformat(),
            "period_hours": hours,
            "compliance_status": self.get_compliance_status(),
            "audit_events": [
                event.to_dict()
                for event in self._api_logger.get_events(start_time=cutoff_time)
            ]
        }
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2)