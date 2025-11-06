"""
Health Check System for API Rate Limiting

Provides comprehensive health monitoring and status endpoints
for the API rate limiting system across all broker integrations.
"""

import asyncio
import time
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import threading
from pathlib import Path

from ..manager import APIRateLimitManager, SystemStatus
from ..monitoring.monitor import RateLimitMonitor, AlertSeverity
from ..compliance.compliance_engine import ComplianceStatus


class HealthStatus(Enum):
    """Individual component health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


class CheckType(Enum):
    """Types of health checks"""
    LIVENESS = "liveness"     # Is the component alive?
    READINESS = "readiness"   # Is the component ready to serve requests?
    PERFORMANCE = "performance"  # Is the component performing well?


@dataclass
class HealthCheck:
    """Individual health check definition"""
    id: str
    name: str
    description: str
    check_type: CheckType
    component: str  # Which component this check applies to
    check_function: Callable
    timeout_seconds: float = 30.0
    critical_threshold: float = 0.0
    warning_threshold: float = 0.0
    enabled: bool = True
    
    async def run(self) -> Dict[str, Any]:
        """Run the health check"""
        try:
            # Run check with timeout
            result = await asyncio.wait_for(
                self.check_function(),
                timeout=self.timeout_seconds
            )
            
            return {
                "id": self.id,
                "name": self.name,
                "status": HealthStatus.HEALTHY.value,
                "check_type": self.check_type.value,
                "component": self.component,
                "result": result,
                "timestamp": datetime.utcnow().isoformat(),
                "execution_time": time.time()
            }
        except asyncio.TimeoutError:
            return {
                "id": self.id,
                "name": self.name,
                "status": HealthStatus.CRITICAL.value,
                "check_type": self.check_type.value,
                "component": self.component,
                "error": "Health check timeout",
                "timestamp": datetime.utcnow().isoformat(),
                "execution_time": time.time()
            }
        except Exception as e:
            return {
                "id": self.id,
                "name": self.name,
                "status": HealthStatus.CRITICAL.value,
                "check_type": self.check_type.value,
                "component": self.component,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "execution_time": time.time()
            }


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    check_id: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_id": self.check_id,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "execution_time": self.execution_time
        }


class HealthCheckRegistry:
    """Registry for health checks"""
    
    def __init__(self):
        self._checks: Dict[str, HealthCheck] = {}
        self._check_history: deque = deque(maxlen=1000)
    
    def register_check(self, check: HealthCheck):
        """Register a health check"""
        self._checks[check.id] = check
    
    def get_check(self, check_id: str) -> Optional[HealthCheck]:
        """Get registered health check"""
        return self._checks.get(check_id)
    
    def get_all_checks(self) -> Dict[str, HealthCheck]:
        """Get all registered checks"""
        return self._checks.copy()
    
    def get_checks_by_component(self, component: str) -> Dict[str, HealthCheck]:
        """Get checks for specific component"""
        return {
            check_id: check
            for check_id, check in self._checks.items()
            if check.component == component
        }
    
    def get_checks_by_type(self, check_type: CheckType) -> Dict[str, HealthCheck]:
        """Get checks by type"""
        return {
            check_id: check
            for check_id, check in self._checks.items()
            if check.check_type == check_type
        }
    
    def add_result(self, result: HealthCheckResult):
        """Add health check result to history"""
        self._check_history.append(result)
    
    def get_history(self, hours: int = 24) -> List[HealthCheckResult]:
        """Get check history"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [
            result for result in self._check_history
            if result.timestamp >= cutoff_time
        ]


class HealthChecker:
    """
    Central health checker for the rate limiting system
    """
    
    def __init__(self, rate_limit_manager: Optional[APIRateLimitManager] = None):
        self.rate_limit_manager = rate_limit_manager
        self.registry = HealthCheckRegistry()
        self._is_running = False
        self._check_task: Optional[asyncio.Task] = None
        self._check_interval = 30.0  # Run checks every 30 seconds
        
        # Results storage
        self._latest_results: Dict[str, HealthCheckResult] = {}
        self._results_lock = threading.RLock()
        
        # Setup default checks
        self._setup_default_checks()
    
    def _setup_default_checks(self):
        """Setup default health checks"""
        
        # System-level checks
        self.registry.register_check(HealthCheck(
            id="system_liveness",
            name="System Liveness",
            description="Check if the system is responding",
            check_type=CheckType.LIVENESS,
            component="system",
            check_function=self._check_system_liveness
        ))
        
        self.registry.register_check(HealthCheck(
            id="system_readiness",
            name="System Readiness",
            description="Check if the system is ready to serve requests",
            check_type=CheckType.READINESS,
            component="system",
            check_function=self._check_system_readiness
        ))
        
        self.registry.register_check(HealthCheck(
            id="system_performance",
            name="System Performance",
            description="Check if the system is performing well",
            check_type=CheckType.PERFORMANCE,
            component="system",
            check_function=self._check_system_performance
        ))
        
        # Rate limiter checks
        self.registry.register_check(HealthCheck(
            id="rate_limiters_health",
            name="Rate Limiters Health",
            description="Check health of all rate limiters",
            check_type=CheckType.LIVENESS,
            component="rate_limiters",
            check_function=self._check_rate_limiters
        ))
        
        self.registry.register_check(HealthCheck(
            id="rate_limiters_performance",
            name="Rate Limiters Performance",
            description="Check performance of rate limiters",
            check_type=CheckType.PERFORMANCE,
            component="rate_limiters",
            check_function=self._check_rate_limiter_performance
        ))
        
        # Request manager checks
        self.registry.register_check(HealthCheck(
            id="request_managers_health",
            name="Request Managers Health",
            description="Check health of request managers",
            check_type=CheckType.LIVENESS,
            component="request_managers",
            check_function=self._check_request_managers
        ))
        
        self.registry.register_check(HealthCheck(
            id="request_queue_performance",
            name="Request Queue Performance",
            description="Check request queue performance",
            check_type=CheckType.PERFORMANCE,
            component="request_managers",
            check_function=self._check_request_queue_performance
        ))
        
        # Monitoring checks
        self.registry.register_check(HealthCheck(
            id="monitoring_system",
            name="Monitoring System",
            description="Check monitoring system health",
            check_type=CheckType.LIVENESS,
            component="monitoring",
            check_function=self._check_monitoring_system
        ))
        
        # Compliance checks
        self.registry.register_check(HealthCheck(
            id="compliance_status",
            name="Compliance Status",
            description="Check compliance across all brokers",
            check_type=CheckType.LIVENESS,
            component="compliance",
            check_function=self._check_compliance_status
        ))
        
        # Broker-specific checks
        self.registry.register_check(HealthCheck(
            id="brokers_connectivity",
            name="Broker Connectivity",
            description="Check connectivity to all configured brokers",
            check_type=CheckType.READINESS,
            component="brokers",
            check_function=self._check_broker_connectivity
        ))
        
        self.registry.register_check(HealthCheck(
            id="broker_rate_limits",
            name="Broker Rate Limits",
            description="Check rate limit utilization across brokers",
            check_type=CheckType.PERFORMANCE,
            component="brokers",
            check_function=self._check_broker_rate_limits
        ))
    
    async def _check_system_liveness(self) -> Dict[str, Any]:
        """Check system liveness"""
        return {
            "status": "alive",
            "uptime_seconds": time.time() - getattr(self, "_start_time", time.time()),
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0"
        }
    
    async def _check_system_readiness(self) -> Dict[str, Any]:
        """Check system readiness"""
        if not self.rate_limit_manager:
            return {
                "status": "not_configured",
                "message": "Rate limit manager not configured"
            }
        
        # Check if system is running
        if not self.rate_limit_manager._is_running:
            return {
                "status": "not_ready",
                "message": "Rate limiting system is not running"
            }
        
        # Check if brokers are configured
        brokers = self.rate_limit_manager.get_broker_list()
        if not brokers:
            return {
                "status": "not_ready",
                "message": "No brokers configured"
            }
        
        return {
            "status": "ready",
            "configured_brokers": len(brokers),
            "brokers": brokers,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _check_system_performance(self) -> Dict[str, Any]:
        """Check system performance"""
        if not self.rate_limit_manager:
            return {"error": "Rate limit manager not configured"}
        
        system_health = self.rate_limit_manager.get_system_health()
        
        return {
            "overall_status": system_health.status.value,
            "active_brokers": system_health.active_brokers,
            "total_requests": system_health.total_requests,
            "success_rate": system_health.successful_requests / max(1, system_health.total_requests),
            "active_alerts": system_health.active_alerts,
            "compliance_score": system_health.compliance_score,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _check_rate_limiters(self) -> Dict[str, Any]:
        """Check rate limiter health"""
        if not self.rate_limit_manager:
            return {"error": "Rate limit manager not configured"}
        
        rate_limiters_health = {}
        
        for broker_name in self.rate_limit_manager.get_broker_list():
            try:
                status = self.rate_limit_manager.get_rate_limit_status(broker_name)
                rate_limiters_health[broker_name] = {
                    "status": "healthy" if status.get("total_requests", 0) > 0 else "new",
                    "total_requests": status.get("total_requests", 0),
                    "success_rate": status.get("success_rate", 0),
                    "current_load": status.get("current_load", 0)
                }
            except Exception as e:
                rate_limiters_health[broker_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        # Determine overall status
        error_count = sum(1 for health in rate_limiters_health.values() 
                         if health.get("status") == "error")
        
        if error_count > 0:
            overall_status = "degraded" if error_count < len(rate_limiters_health) else "critical"
        else:
            overall_status = "healthy"
        
        return {
            "overall_status": overall_status,
            "total_rate_limiters": len(rate_limiters_health),
            "healthy_rate_limiters": len(rate_limiters_health) - error_count,
            "broker_health": rate_limiters_health,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _check_rate_limiter_performance(self) -> Dict[str, Any]:
        """Check rate limiter performance"""
        if not self.rate_limit_manager:
            return {"error": "Rate limit manager not configured"}
        
        performance_data = {}
        total_requests = 0
        total_allowed = 0
        total_rejected = 0
        
        for broker_name in self.rate_limit_manager.get_broker_list():
            try:
                analytics = self.rate_limit_manager.get_analytics(broker_name, hours=1)
                
                performance_data[broker_name] = {
                    "requests_last_hour": analytics.total_requests,
                    "successful_requests": analytics.successful_requests,
                    "rate_limited_requests": analytics.rate_limited_requests,
                    "success_rate": analytics.successful_requests / max(1, analytics.total_requests),
                    "rate_limit_hit_rate": analytics.rate_limited_requests / max(1, analytics.total_requests),
                    "average_response_time": analytics.average_response_time
                }
                
                total_requests += analytics.total_requests
                total_allowed += analytics.successful_requests
                total_rejected += analytics.rate_limited_requests
                
            except Exception as e:
                performance_data[broker_name] = {"error": str(e)}
        
        # Overall performance assessment
        if total_requests > 0:
            overall_success_rate = total_allowed / total_requests
            overall_rate_limit_rate = total_rejected / total_requests
            
            if overall_success_rate < 0.9:
                performance_status = "degraded"
            elif overall_rate_limit_rate > 0.1:
                performance_status = "degraded"
            else:
                performance_status = "healthy"
        else:
            performance_status = "new"
            overall_success_rate = 0
            overall_rate_limit_rate = 0
        
        return {
            "performance_status": performance_status,
            "overall_success_rate": overall_success_rate,
            "overall_rate_limit_rate": overall_rate_limit_rate,
            "total_requests": total_requests,
            "broker_performance": performance_data,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _check_request_managers(self) -> Dict[str, Any]:
        """Check request manager health"""
        if not self.rate_limit_manager:
            return {"error": "Rate limit manager not configured"}
        
        managers_health = {}
        
        for broker_name in self.rate_limit_manager.get_broker_list():
            try:
                if broker_name in self.rate_limit_manager._request_managers:
                    manager = self.rate_limit_manager._request_managers[broker_name]
                    stats = manager.get_stats()
                    
                    managers_health[broker_name] = {
                        "status": "healthy" if stats.get("total_requests", 0) > 0 else "new",
                        "total_requests": stats.get("total_requests", 0),
                        "pending_requests": stats.get("pending_requests", 0),
                        "processing_requests": stats.get("processing_requests", 0),
                        "success_rate": stats.get("success_rate", 0),
                        "average_processing_time": stats.get("average_processing_time", 0)
                    }
                else:
                    managers_health[broker_name] = {"status": "not_found"}
                    
            except Exception as e:
                managers_health[broker_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        # Overall status
        error_count = sum(1 for health in managers_health.values() 
                         if health.get("status") == "error")
        
        overall_status = "healthy" if error_count == 0 else "degraded"
        
        return {
            "overall_status": overall_status,
            "total_managers": len(managers_health),
            "healthy_managers": len(managers_health) - error_count,
            "manager_health": managers_health,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _check_request_queue_performance(self) -> Dict[str, Any]:
        """Check request queue performance"""
        if not self.rate_limit_manager:
            return {"error": "Rate limit manager not configured"}
        
        queue_data = {}
        total_pending = 0
        total_processing = 0
        
        for broker_name in self.rate_limit_manager.get_broker_list():
            try:
                if broker_name in self.rate_limit_manager._request_managers:
                    manager = self.rate_limit_manager._request_managers[broker_name]
                    stats = manager.get_stats()
                    
                    queue_data[broker_name] = {
                        "pending_requests": stats.get("pending_requests", 0),
                        "processing_requests": stats.get("processing_requests", 0),
                        "average_queue_depth": stats.get("average_queue_depth", 0),
                        "max_queue_depth": stats.get("max_queue_depth", 0),
                        "retry_rate": stats.get("retry_rate", 0)
                    }
                    
                    total_pending += stats.get("pending_requests", 0)
                    total_processing += stats.get("processing_requests", 0)
                    
            except Exception as e:
                queue_data[broker_name] = {"error": str(e)}
        
        # Queue health assessment
        if total_pending > 100:
            queue_status = "degraded"
            message = f"High queue depth: {total_pending} pending requests"
        elif total_processing > 50:
            queue_status = "degraded"
            message = f"High processing load: {total_processing} processing requests"
        else:
            queue_status = "healthy"
            message = "Queue performance normal"
        
        return {
            "queue_status": queue_status,
            "message": message,
            "total_pending": total_pending,
            "total_processing": total_processing,
            "broker_queue_data": queue_data,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _check_monitoring_system(self) -> Dict[str, Any]:
        """Check monitoring system health"""
        if not self.rate_limit_manager:
            return {"error": "Rate limit manager not configured"}
        
        monitor = self.rate_limit_manager._monitor
        
        # Get monitor health
        monitor_health = monitor.get_health_status()
        
        return {
            "monitoring_status": "healthy" if monitor_health["is_monitoring"] else "offline",
            "collection_interval": monitor_health["collection_interval"],
            "retention_days": monitor_health["retention_days"],
            "total_metrics": monitor_health["total_metrics"],
            "active_alerts": monitor_health["active_alerts"],
            "critical_alerts": monitor_health["critical_alerts"],
            "last_collection": monitor_health["last_collection"],
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _check_compliance_status(self) -> Dict[str, Any]:
        """Check compliance status"""
        if not self.rate_limit_manager:
            return {"error": "Rate limit manager not configured"}
        
        compliance_status = self.rate_limit_manager.get_compliance_status()
        
        # Count compliance violations
        violation_count = 0
        warning_count = 0
        
        for broker_status in compliance_status.get("brokers", {}).values():
            if broker_status == ComplianceStatus.VIOLATION.value:
                violation_count += 1
            elif broker_status == ComplianceStatus.WARNING.value:
                warning_count += 1
        
        # Overall compliance assessment
        if violation_count > 0:
            overall_compliance = "violation"
        elif warning_count > 0:
            overall_compliance = "warning"
        else:
            overall_compliance = "compliant"
        
        return {
            "overall_compliance": overall_compliance,
            "violation_count": violation_count,
            "warning_count": warning_count,
            "broker_compliance": compliance_status.get("brokers", {}),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _check_broker_connectivity(self) -> Dict[str, Any]:
        """Check broker connectivity"""
        if not self.rate_limit_manager:
            return {"error": "Rate limit manager not configured"}
        
        connectivity_data = {}
        
        for broker_name in self.rate_limit_manager.get_broker_list():
            try:
                # Check if rate limiter is accessible
                status = self.rate_limit_manager.get_rate_limit_status(broker_name)
                
                connectivity_data[broker_name] = {
                    "status": "connected",
                    "last_activity": datetime.fromtimestamp(status.get("timestamp", time.time())).isoformat(),
                    "total_requests": status.get("total_requests", 0)
                }
                
            except Exception as e:
                connectivity_data[broker_name] = {
                    "status": "disconnected",
                    "error": str(e)
                }
        
        # Overall connectivity status
        disconnected_count = sum(1 for data in connectivity_data.values() 
                                if data.get("status") == "disconnected")
        
        if disconnected_count == 0:
            connectivity_status = "healthy"
        elif disconnected_count < len(connectivity_data):
            connectivity_status = "degraded"
        else:
            connectivity_status = "critical"
        
        return {
            "connectivity_status": connectivity_status,
            "total_brokers": len(connectivity_data),
            "connected_brokers": len(connectivity_data) - disconnected_count,
            "broker_connectivity": connectivity_data,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _check_broker_rate_limits(self) -> Dict[str, Any]:
        """Check broker rate limit utilization"""
        if not self.rate_limit_manager:
            return {"error": "Rate limit manager not configured"}
        
        rate_limit_data = {}
        overall_utilization = {}
        
        for broker_name in self.rate_limit_manager.get_broker_list():
            try:
                status = self.rate_limit_manager.get_rate_limit_status(broker_name)
                
                # Get broker-specific data
                broker_data = self.rate_limit_manager._brokers.get(broker_name, {})
                config = broker_data.get("config")
                
                rate_limit_data[broker_name] = {
                    "current_requests": status.get("current_requests", 0),
                    "total_requests": status.get("total_requests", 0),
                    "success_rate": status.get("success_rate", 0),
                    "current_load": status.get("current_load", 0),
                    "average_wait_time": status.get("average_wait_time", 0)
                }
                
                # Calculate utilization if config available
                if config:
                    utilization = status.get("current_load", 0) / config.global_rate_limit
                    overall_utilization[broker_name] = utilization
                
            except Exception as e:
                rate_limit_data[broker_name] = {"error": str(e)}
        
        # Overall utilization assessment
        if overall_utilization:
            avg_utilization = sum(overall_utilization.values()) / len(overall_utilization)
            max_utilization = max(overall_utilization.values())
            
            if max_utilization > 0.95:
                utilization_status = "critical"
            elif max_utilization > 0.8:
                utilization_status = "warning"
            else:
                utilization_status = "healthy"
        else:
            avg_utilization = 0
            max_utilization = 0
            utilization_status = "unknown"
        
        return {
            "utilization_status": utilization_status,
            "average_utilization": avg_utilization,
            "max_utilization": max_utilization,
            "broker_utilization": overall_utilization,
            "broker_rate_limit_data": rate_limit_data,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def run_all_checks(self, check_types: Optional[List[CheckType]] = None) -> Dict[str, Any]:
        """Run all health checks"""
        checks_to_run = self.registry.get_all_checks()
        
        # Filter by check type if specified
        if check_types:
            checks_to_run = {
                check_id: check
                for check_id, check in checks_to_run.items()
                if check.check_type in check_types
            }
        
        # Run checks
        results = {}
        for check_id, check in checks_to_run.items():
            if check.enabled:
                result = await check.run()
                results[check_id] = HealthCheckResult(
                    check_id=result["id"],
                    status=HealthStatus(result["status"]),
                    message=result.get("message", ""),
                    details=result.get("result", {}),
                    execution_time=result.get("execution_time", 0) - time.time()
                )
                
                # Store result
                with self._results_lock:
                    self._latest_results[check_id] = results[check_id]
                
                # Add to registry history
                self.registry.add_result(results[check_id])
        
        # Generate overall status
        overall_status = self._calculate_overall_status(results.values())
        
        return {
            "overall_status": overall_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "checks_run": len(results),
            "results": {check_id: result.to_dict() for check_id, result in results.items()}
        }
    
    def _calculate_overall_status(self, results: List[HealthCheckResult]) -> HealthStatus:
        """Calculate overall system status from check results"""
        status_counts = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.DEGRADED: 0,
            HealthStatus.CRITICAL: 0,
            HealthStatus.OFFLINE: 0,
            HealthStatus.UNKNOWN: 0
        }
        
        for result in results:
            status_counts[result.status] += 1
        
        total_checks = len(results)
        if total_checks == 0:
            return HealthStatus.UNKNOWN
        
        # Determine overall status
        if status_counts[HealthStatus.CRITICAL] > 0:
            return HealthStatus.CRITICAL
        elif status_counts[HealthStatus.OFFLINE] > 0:
            return HealthStatus.OFFLINE
        elif status_counts[HealthStatus.DEGRADED] > 0:
            return HealthStatus.DEGRADED
        elif status_counts[HealthStatus.HEALTHY] == total_checks:
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.DEGRADED
    
    def get_latest_results(self) -> Dict[str, HealthCheckResult]:
        """Get latest health check results"""
        with self._results_lock:
            return self._latest_results.copy()
    
    def get_check_history(self, hours: int = 24) -> List[HealthCheckResult]:
        """Get health check history"""
        return self.registry.get_history(hours)
    
    def start_monitoring(self):
        """Start continuous health monitoring"""
        if self._is_running:
            return
        
        self._is_running = True
        self._check_task = asyncio.create_task(self._monitoring_loop())
    
    def stop_monitoring(self):
        """Stop continuous health monitoring"""
        if not self._is_running:
            return
        
        self._is_running = False
        
        if self._check_task:
            self._check_task.cancel()
    
    async def _monitoring_loop(self):
        """Continuous monitoring loop"""
        while self._is_running:
            try:
                await self.run_all_checks()
                await asyncio.sleep(self._check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Health check monitoring error: {e}")
                await asyncio.sleep(self._check_interval)
    
    def add_custom_check(self, check: HealthCheck):
        """Add custom health check"""
        self.registry.register_check(check)
    
    def get_summary_report(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary health report"""
        history = self.get_check_history(hours)
        
        if not history:
            return {
                "status": "no_data",
                "message": "No health check history available",
                "hours": hours
            }
        
        # Calculate summary statistics
        status_summary = defaultdict(int)
        component_summary = defaultdict(int)
        check_type_summary = defaultdict(int)
        
        for result in history:
            status_summary[result.status.value] += 1
            
            # Find check details
            check = self.registry.get_check(result.check_id)
            if check:
                component_summary[check.component] += 1
                check_type_summary[check.check_type.value] += 1
        
        # Recent trends
        recent_results = history[-10:]  # Last 10 results
        recent_healthy = sum(1 for r in recent_results if r.status == HealthStatus.HEALTHY)
        recent_success_rate = recent_healthy / len(recent_results) if recent_results else 0
        
        return {
            "report_period_hours": hours,
            "total_checks": len(history),
            "status_summary": dict(status_summary),
            "component_summary": dict(component_summary),
            "check_type_summary": dict(check_type_summary),
            "recent_success_rate": recent_success_rate,
            "latest_result": history[-1].to_dict() if history else None,
            "generated_at": datetime.utcnow().isoformat()
        }


# FastAPI health check endpoints
def create_health_check_endpoints(health_checker: HealthChecker):
    """
    Create FastAPI health check endpoints
    
    Args:
        health_checker: HealthChecker instance
        
    Returns:
        Dict of endpoint handlers
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import JSONResponse
    except ImportError:
        return {}
    
    endpoints = {}
    
    @endpoints.get("/health")
    async def basic_health():
        """Basic health check endpoint"""
        try:
            results = await health_checker.run_all_checks([CheckType.LIVENESS])
            status = results["overall_status"]
            
            if status == "healthy":
                return JSONResponse(content={"status": "healthy"})
            elif status == "degraded":
                return JSONResponse(
                    status_code=200,
                    content={"status": "degraded"}
                )
            else:
                raise HTTPException(
                    status_code=503,
                    content={"status": "critical"}
                )
        except Exception as e:
            raise HTTPException(
                status_code=503,
                content={"status": "error", "message": str(e)}
            )
    
    @endpoints.get("/health/detailed")
    async def detailed_health():
        """Detailed health check endpoint"""
        try:
            results = await health_checker.run_all_checks()
            return JSONResponse(content=results)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                content={"error": str(e)}
            )
    
    @endpoints.get("/health/ready")
    async def readiness_check():
        """Readiness check endpoint"""
        try:
            results = await health_checker.run_all_checks([CheckType.READINESS])
            status = results["overall_status"]
            
            if status in ["healthy", "degraded"]:
                return JSONResponse(content={"status": "ready"})
            else:
                raise HTTPException(
                    status_code=503,
                    content={"status": "not_ready"}
                )
        except Exception as e:
            raise HTTPException(
                status_code=503,
                content={"status": "error", "message": str(e)}
            )
    
    @endpoints.get("/health/live")
    async def liveness_check():
        """Liveness check endpoint"""
        try:
            results = await health_checker.run_all_checks([CheckType.LIVENESS])
            status = results["overall_status"]
            
            if status == "healthy":
                return JSONResponse(content={"status": "alive"})
            else:
                raise HTTPException(
                    status_code=503,
                    content={"status": "not_alive"}
                )
        except Exception as e:
            raise HTTPException(
                status_code=503,
                content={"status": "error", "message": str(e)}
            )
    
    @endpoints.get("/metrics")
    async def metrics_endpoint():
        """Metrics endpoint"""
        try:
            summary = health_checker.get_summary_report()
            return JSONResponse(content=summary)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                content={"error": str(e)}
            )
    
    return endpoints