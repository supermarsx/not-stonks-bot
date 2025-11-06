"""
Provider Health Monitor - Monitors and tracks provider health

Provides continuous monitoring of all LLM providers to track:
- Response times and performance metrics
- Error rates and availability
- Automatic health checks
- Health-based provider selection
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import aiofiles
import statistics

from .base_provider import BaseLLMProvider, HealthMetrics, ProviderStatus
from loguru import logger


@dataclass
class HealthAlert:
    """Health alert configuration and data"""
    provider_name: str
    alert_type: str
    message: str
    severity: str  # 'critical', 'warning', 'info'
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class ProviderHealthHistory:
    """Historical health data for a provider"""
    provider_name: str
    health_metrics: List[HealthMetrics] = field(default_factory=list)
    alerts: List[HealthAlert] = field(default_factory=list)
    last_check: Optional[datetime] = None
    consecutive_failures: int = 0
    total_checks: int = 0
    uptime_percentage: float = 100.0


class ProviderHealthMonitor:
    """
    Monitors health and performance of all LLM providers
    
    Features:
    - Periodic health checks
    - Performance tracking
    - Alert generation
    - Health-based failover decisions
    - Historical data storage
    """
    
    def __init__(
        self,
        check_interval: int = 60,  # seconds
        history_retention_hours: int = 24,
        alert_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize health monitor
        
        Args:
            check_interval: How often to check provider health (seconds)
            history_retention_hours: How long to keep health history
            alert_thresholds: Alert thresholds configuration
        """
        self.check_interval = check_interval
        self.history_retention_hours = history_retention_hours
        self.alert_thresholds = alert_thresholds or {
            'max_response_time': 10.0,  # seconds
            'min_success_rate': 0.95,   # 95%
            'max_error_rate': 0.05,     # 5%
            'max_consecutive_failures': 5
        }
        
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.health_history: Dict[str, ProviderHealthHistory] = {}
        self.alerts: List[HealthAlert] = []
        self.alert_callbacks: List[Callable[[HealthAlert], None]] = []
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        
    def register_provider(self, provider: BaseLLMProvider):
        """Register a provider for monitoring"""
        provider_name = provider.config.name
        
        self.providers[provider_name] = provider
        self.health_history[provider_name] = ProviderHealthHistory(
            provider_name=provider_name
        )
        
        logger.info(f"Registered provider for health monitoring: {provider_name}")
        
    def unregister_provider(self, provider_name: str):
        """Unregister a provider from monitoring"""
        if provider_name in self.providers:
            del self.providers[provider_name]
            
        if provider_name in self.health_history:
            del self.health_history[provider_name]
            
        logger.info(f"Unregistered provider from health monitoring: {provider_name}")
        
    def add_alert_callback(self, callback: Callable[[HealthAlert], None]):
        """Add a callback function to be called when alerts are triggered"""
        self.alert_callbacks.append(callback)
        
    async def start_monitoring(self):
        """Start the health monitoring loop"""
        if self._running:
            logger.warning("Health monitor is already running")
            return
            
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started provider health monitoring")
        
    async def stop_monitoring(self):
        """Stop the health monitoring loop"""
        self._running = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
                
        logger.info("Stopped provider health monitoring")
        
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                await self._check_all_providers()
                await self._cleanup_old_history()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(5)  # Short delay before retry
                
    async def _check_all_providers(self):
        """Check health of all registered providers"""
        for provider_name, provider in self.providers.items():
            try:
                await self._check_provider_health(provider_name, provider)
            except Exception as e:
                logger.error(f"Error checking health of provider {provider_name}: {e}")
                
    async def _check_provider_health(self, provider_name: str, provider: BaseLLMProvider):
        """Check health of a specific provider"""
        try:
            # Perform health check
            health_metrics = await provider.health_check()
            health_metrics.last_check = datetime.now()
            
            # Update provider health
            provider.update_health(health_metrics)
            
            # Update history
            history = self.health_history[provider_name]
            history.health_metrics.append(health_metrics)
            history.last_check = datetime.now()
            history.total_checks += 1
            
            # Update consecutive failures
            if not health_metrics.is_healthy:
                history.consecutive_failures += 1
            else:
                history.consecutive_failures = 0
                
            # Calculate uptime percentage
            healthy_checks = sum(
                1 for m in history.health_metrics[-100:]  # Last 100 checks
                if m.is_healthy
            )
            history.uptime_percentage = (healthy_checks / min(len(history.health_metrics), 100)) * 100
            
            # Check for alerts
            await self._check_alert_conditions(provider_name, health_metrics)
            
        except Exception as e:
            logger.error(f"Failed to check health of provider {provider_name}: {e}")
            
            # Create error health metrics
            error_metrics = HealthMetrics(
                response_time=0.0,
                success_rate=0.0,
                error_rate=1.0,
                last_check=datetime.now(),
                consecutive_failures=999,  # Force error state
                is_healthy=False,
                message=f"Health check failed: {str(e)}"
            )
            
            provider.update_health(error_metrics)
            
    async def _check_alert_conditions(self, provider_name: str, health_metrics: HealthMetrics):
        """Check if any alert conditions are met"""
        history = self.health_history[provider_name]
        alerts_to_create = []
        
        # Response time alert
        if health_metrics.response_time > self.alert_thresholds['max_response_time']:
            alerts_to_create.append(HealthAlert(
                provider_name=provider_name,
                alert_type='high_response_time',
                message=f"High response time: {health_metrics.response_time:.2f}s (threshold: {self.alert_thresholds['max_response_time']}s)",
                severity='warning',
                timestamp=datetime.now()
            ))
            
        # Success rate alert
        if health_metrics.success_rate < self.alert_thresholds['min_success_rate']:
            alerts_to_create.append(HealthAlert(
                provider_name=provider_name,
                alert_type='low_success_rate',
                message=f"Low success rate: {health_metrics.success_rate:.2%} (threshold: {self.alert_thresholds['min_success_rate']:.2%})",
                severity='warning',
                timestamp=datetime.now()
            ))
            
        # Error rate alert
        if health_metrics.error_rate > self.alert_thresholds['max_error_rate']:
            alerts_to_create.append(HealthAlert(
                provider_name=provider_name,
                alert_type='high_error_rate',
                message=f"High error rate: {health_metrics.error_rate:.2%} (threshold: {self.alert_thresholds['max_error_rate']:.2%})",
                severity='critical',
                timestamp=datetime.now()
            ))
            
        # Consecutive failures alert
        if history.consecutive_failures >= self.alert_thresholds['max_consecutive_failures']:
            alerts_to_create.append(HealthAlert(
                provider_name=provider_name,
                alert_type='consecutive_failures',
                message=f"Consecutive failures: {history.consecutive_failures} (threshold: {self.alert_thresholds['max_consecutive_failures']})",
                severity='critical',
                timestamp=datetime.now()
            ))
            
        # Provider unavailable alert
        if history.consecutive_failures >= 10:
            alerts_to_create.append(HealthAlert(
                provider_name=provider_name,
                alert_type='provider_unavailable',
                message=f"Provider appears to be unavailable after {history.consecutive_failures} consecutive failures",
                severity='critical',
                timestamp=datetime.now()
            ))
            
        # Create and handle alerts
        for alert in alerts_to_create:
            self.alerts.append(alert)
            history.alerts.append(alert)
            
            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    await callback(alert) if asyncio.iscoroutinefunction(callback) else callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
                    
            logger.warning(f"Health alert for {provider_name}: {alert.message}")
            
        # Check for resolved alerts
        await self._check_resolved_alerts(provider_name, health_metrics)
        
    async def _check_resolved_alerts(self, provider_name: str, health_metrics: HealthMetrics):
        """Check if any previous alerts should be marked as resolved"""
        if not health_metrics.is_healthy:
            return  # Can't resolve alerts if provider is still unhealthy
            
        for alert in self.alerts:
            if (alert.provider_name == provider_name and 
                not alert.resolved and
                alert.alert_type in ['high_response_time', 'low_success_rate', 'high_error_rate']):
                
                # Alert is resolved if provider is now healthy
                alert.resolved = True
                alert.resolved_at = datetime.now()
                
                logger.info(f"Resolved health alert for {provider_name}: {alert.alert_type}")
                
    async def _cleanup_old_history(self):
        """Clean up old health history data"""
        cutoff_time = datetime.now() - timedelta(hours=self.history_retention_hours)
        
        for history in self.health_history.values():
            # Clean old health metrics
            history.health_metrics = [
                metrics for metrics in history.health_metrics
                if metrics.last_check > cutoff_time
            ]
            
            # Clean old alerts (keep resolved alerts for a shorter period)
            alert_cutoff = cutoff_time - timedelta(hours=1)  # Keep resolved alerts for 1 hour
            history.alerts = [
                alert for alert in history.alerts
                if alert.resolved_at is None or alert.resolved_at > alert_cutoff
            ]
            
    def get_provider_health(self, provider_name: str) -> Optional[Dict[str, Any]]:
        """Get current health status of a provider"""
        if provider_name not in self.providers:
            return None
            
        provider = self.providers[provider_name]
        history = self.health_history[provider_name]
        
        # Get recent metrics
        recent_metrics = history.health_metrics[-10:] if history.health_metrics else []
        avg_response_time = statistics.mean([m.response_time for m in recent_metrics]) if recent_metrics else 0
        avg_success_rate = statistics.mean([m.success_rate for m in recent_metrics]) if recent_metrics else 0
        
        return {
            'provider_name': provider_name,
            'status': provider.get_status().value,
            'is_healthy': provider.get_health_metrics().is_healthy,
            'response_time': provider.get_health_metrics().response_time,
            'avg_response_time': avg_response_time,
            'success_rate': provider.get_health_metrics().success_rate,
            'avg_success_rate': avg_success_rate,
            'error_rate': provider.get_health_metrics().error_rate,
            'consecutive_failures': history.consecutive_failures,
            'total_checks': history.total_checks,
            'uptime_percentage': history.uptime_percentage,
            'last_check': provider.get_health_metrics().last_check.isoformat() if provider.get_health_metrics().last_check else None,
            'message': provider.get_health_metrics().message
        }
        
    def get_all_providers_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of all providers"""
        return {
            name: self.get_provider_health(name)
            for name in self.providers.keys()
        }
        
    def get_active_alerts(self, provider_name: Optional[str] = None) -> List[HealthAlert]:
        """Get active (unresolved) alerts"""
        alerts = [alert for alert in self.alerts if not alert.resolved]
        
        if provider_name:
            alerts = [alert for alert in alerts if alert.provider_name == provider_name]
            
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
        
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of all alerts"""
        total_alerts = len(self.alerts)
        active_alerts = len(self.get_active_alerts())
        critical_alerts = len([a for a in self.alerts if a.severity == 'critical' and not a.resolved])
        warning_alerts = len([a for a in self.alerts if a.severity == 'warning' and not a.resolved])
        
        # Group alerts by provider
        alerts_by_provider = {}
        for alert in self.alerts:
            if alert.provider_name not in alerts_by_provider:
                alerts_by_provider[alert.provider_name] = {
                    'total': 0,
                    'active': 0,
                    'critical': 0,
                    'warning': 0
                }
                
            alerts_by_provider[alert.provider_name]['total'] += 1
            
            if not alert.resolved:
                alerts_by_provider[alert.provider_name]['active'] += 1
                
            if alert.severity == 'critical':
                if not alert.resolved:
                    alerts_by_provider[alert.provider_name]['critical'] += 1
            elif alert.severity == 'warning':
                if not alert.resolved:
                    alerts_by_provider[alert.provider_name]['warning'] += 1
                    
        return {
            'total_alerts': total_alerts,
            'active_alerts': active_alerts,
            'critical_alerts': critical_alerts,
            'warning_alerts': warning_alerts,
            'alerts_by_provider': alerts_by_provider
        }
        
    def get_best_providers(self, capability_filter: Optional[str] = None) -> List[str]:
        """Get list of providers ranked by health and performance"""
        provider_scores = []
        
        for provider_name in self.providers.keys():
            health_data = self.get_provider_health(provider_name)
            if not health_data:
                continue
                
            # Calculate health score
            score = 0
            
            # Uptime contributes most to score
            score += health_data['uptime_percentage'] * 0.4
            
            # Success rate contributes
            score += health_data['avg_success_rate'] * 100 * 0.3
            
            # Low response time contributes
            if health_data['avg_response_time'] > 0:
                response_score = max(0, 100 - (health_data['avg_response_time'] * 10))
                score += response_score * 0.2
                
            # Penalize for consecutive failures
            score -= health_data['consecutive_failures'] * 5
            
            # Provider is healthy
            if health_data['is_healthy']:
                score += 50
                
            provider_scores.append((provider_name, score))
            
        # Sort by score (highest first)
        provider_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [name for name, _ in provider_scores]
        
    async def export_health_data(self, file_path: str):
        """Export health data to JSON file"""
        data = {
            'export_timestamp': datetime.now().isoformat(),
            'providers_health': self.get_all_providers_health(),
            'alerts': [
                {
                    'provider_name': alert.provider_name,
                    'alert_type': alert.alert_type,
                    'message': alert.message,
                    'severity': alert.severity,
                    'timestamp': alert.timestamp.isoformat(),
                    'resolved': alert.resolved,
                    'resolved_at': alert.resolved_at.isoformat() if alert.resolved_at else None
                }
                for alert in self.alerts
            ],
            'alert_summary': self.get_alert_summary()
        }
        
        async with aiofiles.open(file_path, 'w') as f:
            await f.write(json.dumps(data, indent=2, default=str))
            
        logger.info(f"Exported health data to {file_path}")
        
    async def force_health_check(self, provider_name: str) -> bool:
        """Force a health check for a specific provider"""
        if provider_name not in self.providers:
            return False
            
        try:
            await self._check_provider_health(provider_name, self.providers[provider_name])
            return True
        except Exception as e:
            logger.error(f"Failed to force health check for {provider_name}: {e}")
            return False