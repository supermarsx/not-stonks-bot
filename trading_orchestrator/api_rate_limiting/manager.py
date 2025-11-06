"""
Main API Rate Limiting Manager

Central coordinator for all rate limiting operations across broker integrations.
Provides unified interface for rate limiting, request management, monitoring, and compliance.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
from enum import Enum
import threading

from .core.rate_limiter import RateLimiterManager, RequestType, RequestPriority
from .core.request_manager import RequestManager, Request, RetryConfig
from .brokers.rate_limit_configs import get_broker_config, RateLimitConfig
from .monitoring.monitor import (
    RateLimitMonitor, UsageAnalytics, Alert, AlertSeverity
)
from .compliance.compliance_engine import (
    ComplianceEngine, ComplianceStatus, APILogger
)
from .core.exceptions import RateLimitExceededException


class SystemStatus(Enum):
    """Overall system status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    MAINTENANCE = "maintenance"


@dataclass
class SystemHealth:
    """System health summary"""
    status: SystemStatus
    timestamp: str
    uptime_seconds: float
    active_brokers: int
    total_requests: int
    successful_requests: int
    active_alerts: int
    compliance_score: float
    issues: List[str]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "timestamp": self.timestamp,
            "uptime_seconds": self.uptime_seconds,
            "active_brokers": self.active_brokers,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "success_rate": self.successful_requests / max(1, self.total_requests),
            "active_alerts": self.active_alerts,
            "compliance_score": self.compliance_score,
            "issues": self.issues,
            "recommendations": self.recommendations
        }


class APITransaction:
    """
    Context manager for API transactions with rate limiting and compliance
    """
    
    def __init__(
        self,
        manager: 'APIRateLimitManager',
        broker: str,
        request_type: RequestType,
        priority: RequestPriority = RequestPriority.NORMAL,
        force_synchronous: bool = False
    ):
        self.manager = manager
        self.broker = broker
        self.request_type = request_type
        self.priority = priority
        self.force_synchronous = force_synchronous
        
        self.request_id: Optional[str] = None
        self.start_time: Optional[float] = None
        self.result: Optional[Any] = None
        self.error: Optional[Exception] = None
    
    async def __aenter__(self) -> 'APITransaction':
        self.start_time = time.time()
        
        # Submit request to manager
        self.request_id = await self.manager.submit_request(
            broker=self.broker,
            request_type=self.request_type,
            priority=self.priority
        )
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            self.error = exc_val
            # Log error for monitoring and compliance
            await self.manager.log_error(
                broker=self.broker,
                request_type=self.request_type,
                error=exc_val
            )
    
    def get_status(self) -> Dict[str, Any]:
        """Get transaction status"""
        return {
            "request_id": self.request_id,
            "broker": self.broker,
            "request_type": self.request_type.value,
            "priority": self.priority.name,
            "start_time": self.start_time,
            "duration_seconds": time.time() - self.start_time if self.start_time else 0,
            "completed": self.result is not None,
            "error": str(self.error) if self.error else None
        }


class APIRateLimitManager:
    """
    Central API Rate Limiting Manager
    
    Coordinates all rate limiting, request management, monitoring, and compliance
    across multiple broker integrations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._lock = threading.RLock()
        
        # Component instances
        self._brokers: Dict[str, Dict[str, Any]] = {}
        self._rate_limiters: Dict[str, RateLimiterManager] = {}
        self._request_managers: Dict[str, RequestManager] = {}
        self._configs: Dict[str, RateLimitConfig] = {}
        
        # Core components
        self._monitor = RateLimitMonitor(
            storage_path=self.config.get("monitoring_path"),
            collection_interval=self.config.get("monitoring_interval", 10.0),
            retention_days=self.config.get("monitoring_retention_days", 30)
        )
        
        self._compliance = ComplianceEngine()
        self._api_logger = APILogger(
            storage_path=self.config.get("audit_path"),
            retention_days=self.config.get("audit_retention_days", 90)
        )
        
        # System status
        self._start_time = time.time()
        self._is_running = False
        
        # Setup monitoring callbacks
        self._setup_monitoring_callbacks()
        
        print("API Rate Limit Manager initialized")
    
    def _setup_monitoring_callbacks(self):
        """Setup monitoring callbacks for events"""
        
        # Alert callback
        async def alert_callback(alert: Alert):
            # Log alert for audit
            await self._compliance.handle_event(
                type(alert).__new__(type(alert), "alert_created", (), {})(
                    id=f"alert_{alert.id}",
                    timestamp=alert.timestamp,
                    broker=alert.broker,
                    event_type="alert_created",
                    level=alert.severity.value,
                    details={"alert": alert.to_dict()}
                )
            )
            
            # Handle critical alerts
            if alert.severity == AlertSeverity.CRITICAL:
                await self._handle_critical_alert(alert)
        
        self._monitor.add_alert_callback(alert_callback)
    
    async def _handle_critical_alert(self, alert: Alert):
        """Handle critical alert"""
        # Could implement automatic scaling, circuit breaker activation, etc.
        print(f"CRITICAL ALERT: {alert.title} - {alert.message}")
        
        # Auto-resolve if it's a rate limit alert that should resolve itself
        if "rate_limit" in alert.title.lower():
            asyncio.create_task(self._auto_resolve_rate_limit_alert(alert))
    
    async def _auto_resolve_rate_limit_alert(self, alert: Alert):
        """Auto-resolve rate limit alerts after delay"""
        await asyncio.sleep(60)  # Wait 1 minute
        self._monitor.resolve_alert(alert.id)
    
    def add_broker(
        self,
        broker_name: str,
        is_futures: bool = False,
        is_paper: bool = True,
        custom_config: Optional[RateLimitConfig] = None
    ):
        """Add broker to rate limiting system"""
        with self._lock:
            # Get or create configuration
            if custom_config:
                config = custom_config
            else:
                config = get_broker_config(broker_name, is_futures, is_paper)
            
            # Create rate limiter manager
            rate_limiter = RateLimiterManager(broker_name, config.__dict__)
            
            # Create request manager
            request_manager = RequestManager(rate_limiter)
            
            # Store components
            self._brokers[broker_name] = {
                "config": config,
                "rate_limiter": rate_limiter,
                "request_manager": request_manager,
                "is_futures": is_futures,
                "is_paper": is_paper
            }
            
            self._rate_limiters[broker_name] = rate_limiter
            self._request_managers[broker_name] = request_manager
            self._configs[broker_name] = config
            
            # Add to monitoring
            self._monitor.add_rate_limiter(broker_name, rate_limiter)
            self._monitor.add_request_manager(broker_name, request_manager)
            self._monitor.add_config(broker_name, config)
            
            # Add to compliance
            self._compliance.add_broker_config(broker_name, config)
            
            print(f"Added broker: {broker_name} (futures={is_futures}, paper={is_paper})")
    
    async def start(self):
        """Start the rate limiting system"""
        if self._is_running:
            return
        
        self._is_running = True
        
        # Start monitoring
        await self._monitor.start_monitoring()
        
        # Start request processing for all brokers
        for request_manager in self._request_managers.values():
            await request_manager.start_processing()
        
        print("API Rate Limit Manager started")
    
    async def stop(self):
        """Stop the rate limiting system"""
        if not self._is_running:
            return
        
        self._is_running = False
        
        # Stop request processing
        for request_manager in self._request_managers.values():
            await request_manager.stop_processing()
        
        # Stop monitoring
        await self._monitor.stop_monitoring()
        
        print("API Rate Limit Manager stopped")
    
    async def submit_request(
        self,
        broker: str,
        request_type: RequestType,
        callback: Callable,
        priority: RequestPriority = RequestPriority.NORMAL,
        symbol: Optional[str] = None,
        timeout: Optional[float] = None,
        retry_config: Optional[RetryConfig] = None,
        **kwargs
    ) -> str:
        """Submit API request with rate limiting"""
        
        if broker not in self._request_managers:
            raise ValueError(f"Broker '{broker}' not found")
        
        request_manager = self._request_managers[broker]
        
        # Check compliance before submitting
        context = {
            "request_type": request_type,
            "symbol": symbol,
            "priority": priority,
            "timestamp": time.time()
        }
        
        compliance_status = self._compliance.check_compliance(broker, request_type, context)
        
        if compliance_status == ComplianceStatus.VIOLATION:
            raise ValueError(f"Compliance violation prevents request submission")
        
        # Submit request
        request_id = await request_manager.submit_request(
            request_type=request_type,
            callback=callback,
            priority=priority,
            symbol=symbol,
            timeout=timeout,
            retry_config=retry_config,
            **kwargs
        )
        
        # Log submission
        await self._compliance.handle_event(
            type("AuditEvent", (), {})(
                id=f"submit_{int(time.time() * 1000)}",
                timestamp=type("datetime", (), {"utcnow": lambda: type("datetime", (), {"isoformat": lambda: "2023-01-01T00:00:00"})()})(),
                broker=broker,
                event_type="request_submitted",
                level="info",
                details={
                    "request_id": request_id,
                    "request_type": request_type.value,
                    "compliance_status": compliance_status.value,
                    "symbol": symbol,
                    "priority": priority.name
                }
            )
        )
        
        return request_id
    
    def get_rate_limit_status(self, broker: str) -> Dict[str, Any]:
        """Get rate limit status for broker"""
        if broker not in self._rate_limiters:
            raise ValueError(f"Broker '{broker}' not found")
        
        return self._rate_limiters[broker].get_global_status()
    
    def get_request_status(self, broker: str, request_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific request"""
        if broker not in self._request_managers:
            raise ValueError(f"Broker '{broker}' not found")
        
        return self._request_managers[broker].get_request_status(request_id)
    
    def get_active_alerts(self, broker: Optional[str] = None) -> List[Alert]:
        """Get active alerts"""
        return self._monitor.get_active_alerts(broker=broker)
    
    def get_analytics(self, broker: str, hours: int = 24) -> UsageAnalytics:
        """Get usage analytics for broker"""
        return self._monitor.get_analytics(broker, hours)
    
    def get_compliance_status(self) -> Dict[str, Any]:
        """Get compliance status for all brokers"""
        return self._compliance.get_compliance_status()
    
    async def log_error(
        self,
        broker: str,
        request_type: RequestType,
        error: Exception
    ):
        """Log error for monitoring and compliance"""
        await self._compliance.handle_event(
            type("AuditEvent", (), {})(
                id=f"error_{int(time.time() * 1000)}",
                timestamp=type("datetime", (), {"utcnow": lambda: type("datetime", (), {"isoformat": lambda: "2023-01-01T00:00:00"})()})(),
                broker=broker,
                event_type="request_error",
                level="error",
                details={
                    "request_type": request_type.value,
                    "error_type": type(error).__name__,
                    "error_message": str(error)
                }
            )
        )
    
    def get_cost_optimization(
        self,
        broker: str,
        requests: Dict[RequestType, int],
        time_window_hours: float
    ) -> Dict[str, Any]:
        """Get cost optimization recommendations"""
        return self._compliance.get_cost_optimization(broker, requests, time_window_hours)
    
    def get_system_health(self) -> SystemHealth:
        """Get overall system health"""
        
        # Calculate metrics
        uptime = time.time() - self._start_time
        active_brokers = len(self._brokers)
        
        # Get request statistics
        total_requests = 0
        successful_requests = 0
        
        for request_manager in self._request_managers.values():
            stats = request_manager.get_stats()
            total_requests += stats.get("total_requests", 0)
            successful_requests += stats.get("completed_requests", 0)
        
        # Get alerts
        active_alerts = len(self.get_active_alerts())
        critical_alerts = len([a for a in self.get_active_alerts() if a.severity == AlertSeverity.CRITICAL])
        
        # Get compliance score
        compliance_data = self.get_compliance_status()
        compliance_score = 100.0 - (active_alerts * 10)
        
        # Determine system status
        if critical_alerts > 0:
            status = SystemStatus.CRITICAL
            issues = [f"{critical_alerts} critical alerts active"]
        elif active_alerts > 5:
            status = SystemStatus.DEGRADED
            issues = [f"{active_alerts} active alerts"]
        elif compliance_score < 70:
            status = SystemStatus.DEGRADED
            issues = ["Low compliance score"]
        else:
            status = SystemStatus.HEALTHY
            issues = []
        
        # Generate recommendations
        recommendations = []
        if active_alerts > 0:
            recommendations.append("Review active alerts and address underlying issues")
        
        if successful_requests / max(1, total_requests) < 0.95:
            recommendations.append("Low success rate - check rate limiting configuration")
        
        if active_brokers < 2:
            recommendations.append("Consider adding more brokers for redundancy")
        
        return SystemHealth(
            status=status,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            uptime_seconds=uptime,
            active_brokers=active_brokers,
            total_requests=total_requests,
            successful_requests=successful_requests,
            active_alerts=active_alerts,
            compliance_score=compliance_score,
            issues=issues,
            recommendations=recommendations
        )
    
    def export_data(
        self,
        output_path: str,
        broker: Optional[str] = None,
        hours: int = 24,
        include_compliance: bool = True,
        include_metrics: bool = True
    ):
        """Export system data"""
        import json
        from pathlib import Path
        
        output_file = Path(output_path)
        output_file.parent.mkdir(exist_ok=True)
        
        export_data = {
            "export_time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "system_health": self.get_system_health().to_dict(),
            "export_options": {
                "broker": broker,
                "hours": hours,
                "include_compliance": include_compliance,
                "include_metrics": include_metrics
            }
        }
        
        # Export broker-specific data
        if broker:
            if broker in self._brokers:
                export_data["broker_data"] = {
                    "config": self._configs[broker].__dict__,
                    "rate_limit_status": self.get_rate_limit_status(broker),
                    "analytics": self.get_analytics(broker, hours).to_dict()
                }
        else:
            # Export all brokers
            export_data["all_brokers"] = {}
            for broker_name in self._brokers:
                export_data["all_brokers"][broker_name] = {
                    "config": self._configs[broker_name].__dict__,
                    "rate_limit_status": self.get_rate_limit_status(broker_name),
                    "analytics": self.get_analytics(broker_name, hours).to_dict()
                }
        
        # Export compliance data
        if include_compliance:
            export_data["compliance"] = self.get_compliance_status()
            export_data["audit_report"] = self._compliance.get_audit_report(hours)
        
        # Export metrics
        if include_metrics:
            metrics_file = output_file.with_suffix(".metrics.json")
            self._monitor.export_metrics(str(metrics_file), broker, hours)
            export_data["metrics_file"] = str(metrics_file)
        
        # Write main export file
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"Exported system data to {output_file}")
    
    # Context manager for API transactions
    def transaction(
        self,
        broker: str,
        request_type: RequestType,
        priority: RequestPriority = RequestPriority.NORMAL
    ) -> APITransaction:
        """Create API transaction context manager"""
        return APITransaction(
            manager=self,
            broker=broker,
            request_type=request_type,
            priority=priority
        )
    
    # Convenience methods for common operations
    async def wait_for_rate_limit(self, broker: str, request_type: RequestType, tokens: int = 1, timeout: float = None) -> bool:
        """Wait for rate limit tokens to become available"""
        if broker not in self._rate_limiters:
            raise ValueError(f"Broker '{broker}' not found")
        
        return await self._rate_limiters[broker].wait_for_tokens(request_type, tokens, timeout)
    
    def get_broker_list(self) -> List[str]:
        """Get list of configured brokers"""
        return list(self._brokers.keys())
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        return {
            "brokers": {
                name: {
                    "is_futures": broker_data["is_futures"],
                    "is_paper": broker_data["is_paper"],
                    "global_rate_limit": broker_data["config"].global_rate_limit,
                    "global_window_seconds": broker_data["config"].global_window_seconds,
                    "endpoint_limits": len(broker_data["config"].endpoint_limits)
                }
                for name, broker_data in self._brokers.items()
            },
            "monitoring": {
                "collection_interval": self._monitor.collection_interval,
                "retention_days": self._monitor.retention_days
            },
            "system": {
                "uptime_seconds": time.time() - self._start_time,
                "is_running": self._is_running
            }
        }