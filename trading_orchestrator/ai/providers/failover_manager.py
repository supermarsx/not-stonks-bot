"""
Provider Failover Manager - Automatic provider switching and failover

Provides intelligent failover between LLM providers when primary providers fail.
Features:
- Automatic failover based on health monitoring
- Configurable failover strategies (round-robin, priority-based, etc.)
- Circuit breaker pattern for failed providers
- Graceful degradation when fewer providers are available
"""

import asyncio
import random
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import statistics

from .base_provider import BaseLLMProvider, ProviderCapability, ProviderStatus
from .health_monitor import ProviderHealthMonitor, HealthAlert
from loguru import logger


class FailoverStrategy(Enum):
    """Failover strategy selection"""
    ROUND_ROBIN = "round_robin"
    PRIORITY_BASED = "priority_based"
    HEALTH_BASED = "health_based"
    RANDOM = "random"
    LEAST_RECENTLY_USED = "lru"


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"    # Normal operation
    OPEN = "open"        # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if provider recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5  # Number of failures to open circuit
    recovery_timeout: int = 60  # Seconds to wait before trying half-open
    success_threshold: int = 3  # Successes needed to close circuit from half-open
    timeout_duration: int = 30  # Request timeout


@dataclass
class CircuitBreaker:
    """Circuit breaker for a provider"""
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    
    def __post_init__(self):
        self.config = CircuitBreakerConfig()


@dataclass
class FailoverStats:
    """Failover statistics tracking"""
    total_requests: int = 0
    failed_requests: int = 0
    failover_events: int = 0
    provider_usage: Dict[str, int] = field(default_factory=dict)
    avg_failover_time: float = 0.0
    last_failover: Optional[datetime] = None


class ProviderFailoverManager:
    """
    Manages automatic failover between LLM providers
    
    Features:
    - Automatic provider switching on failures
    - Circuit breaker pattern
    - Multiple failover strategies
    - Performance tracking
    - Graceful degradation
    """
    
    def __init__(
        self,
        health_monitor: ProviderHealthMonitor,
        strategy: FailoverStrategy = FailoverStrategy.HEALTH_BASED,
        enable_circuit_breaker: bool = True,
        global_timeout: float = 30.0
    ):
        """
        Initialize failover manager
        
        Args:
            health_monitor: Health monitor instance
            strategy: Failover strategy to use
            enable_circuit_breaker: Whether to use circuit breaker pattern
            global_timeout: Global timeout for requests
        """
        self.health_monitor = health_monitor
        self.strategy = strategy
        self.enable_circuit_breaker = enable_circuit_breaker
        self.global_timeout = global_timeout
        
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.failover_stats = FailoverStats()
        self.provider_queue: List[str] = []  # For round-robin
        self.current_provider_index = 0
        self.provider_last_used: Dict[str, datetime] = {}
        self.failed_providers: Set[str] = set()
        
        # Callback for failover events
        self.failover_callbacks: List[callable] = []
        
        # Register with health monitor
        health_monitor.add_alert_callback(self._on_health_alert)
        
    def register_provider(self, provider: BaseLLMProvider):
        """Register a provider for failover management"""
        provider_name = provider.config.name
        
        self.providers[provider_name] = provider
        self.circuit_breakers[provider_name] = CircuitBreaker()
        self.provider_queue.append(provider_name)
        self.provider_last_used[provider_name] = datetime.now()
        
        # Register with health monitor
        self.health_monitor.register_provider(provider)
        
        logger.info(f"Registered provider for failover management: {provider_name}")
        
    def unregister_provider(self, provider_name: str):
        """Unregister a provider from failover management"""
        if provider_name in self.providers:
            del self.providers[provider_name]
            
        if provider_name in self.circuit_breakers:
            del self.circuit_breakers[provider_name]
            
        if provider_name in self.provider_queue:
            self.provider_queue.remove(provider_name)
            
        if provider_name in self.provider_last_used:
            del self.provider_last_used[provider_name]
            
        self.failed_providers.discard(provider_name)
        
        # Unregister from health monitor
        self.health_monitor.unregister_provider(provider_name)
        
        logger.info(f"Unregistered provider from failover management: {provider_name}")
        
    def add_failover_callback(self, callback: callable):
        """Add callback for failover events"""
        self.failover_callbacks.append(callback)
        
    async def get_provider(
        self,
        capability: Optional[ProviderCapability] = None,
        exclude_providers: Optional[List[str]] = None,
        preferred_provider: Optional[str] = None
    ) -> Optional[BaseLLMProvider]:
        """
        Get the best available provider based on failover strategy
        
        Args:
            capability: Required capability filter
            exclude_providers: Provider names to exclude
            preferred_provider: Prefer this provider if available
            
        Returns:
            Best available provider or None
        """
        available_providers = self._get_available_providers(
            capability, exclude_providers, preferred_provider
        )
        
        if not available_providers:
            logger.warning("No available providers found")
            return None
            
        # Select provider based on strategy
        if self.strategy == FailoverStrategy.PRIORITY_BASED:
            return self._select_priority_provider(available_providers)
        elif self.strategy == FailoverStrategy.HEALTH_BASED:
            return self._select_health_based_provider(available_providers)
        elif self.strategy == FailoverStrategy.ROUND_ROBIN:
            return self._select_round_robin_provider(available_providers)
        elif self.strategy == FailoverStrategy.RANDOM:
            return random.choice(available_providers)
        elif self.strategy == FailoverStrategy.LEAST_RECENTLY_USED:
            return self._select_lru_provider(available_providers)
        else:
            return available_providers[0]
            
    def _get_available_providers(
        self,
        capability: Optional[ProviderCapability] = None,
        exclude_providers: Optional[List[str]] = None,
        preferred_provider: Optional[str] = None
    ) -> List[BaseLLMProvider]:
        """Get list of available providers"""
        available = []
        
        for provider_name, provider in self.providers.items():
            # Skip excluded providers
            if exclude_providers and provider_name in exclude_providers:
                continue
                
            # Skip failed providers (if circuit breaker is enabled)
            if self.enable_circuit_breaker and self._is_provider_failed(provider_name):
                continue
                
            # Check capability requirements
            if capability and not provider.supports_capability(capability):
                continue
                
            # Check provider health
            if not self._is_provider_healthy(provider_name):
                continue
                
            available.append(provider)
            
        # Add preferred provider to the beginning if it's available
        if preferred_provider and preferred_provider in [p.config.name for p in available]:
            preferred_idx = next(i for i, p in enumerate(available) if p.config.name == preferred_provider)
            preferred_provider_obj = available.pop(preferred_idx)
            available.insert(0, preferred_provider_obj)
            
        return available
        
    def _is_provider_healthy(self, provider_name: str) -> bool:
        """Check if provider is considered healthy"""
        health_data = self.health_monitor.get_provider_health(provider_name)
        return health_data and health_data.get('is_healthy', False)
        
    def _is_provider_failed(self, provider_name: str) -> bool:
        """Check if provider is in failed state (circuit breaker open)"""
        if not self.enable_circuit_breaker:
            return False
            
        circuit_breaker = self.circuit_breakers.get(provider_name)
        if not circuit_breaker:
            return False
            
        return circuit_breaker.state == CircuitBreakerState.OPEN
        
    def _select_priority_provider(self, providers: List[BaseLLMProvider]) -> BaseLLMProvider:
        """Select provider based on priority configuration"""
        # Sort by priority (lower number = higher priority)
        providers.sort(key=lambda p: p.config.priority)
        return providers[0]
        
    def _select_health_based_provider(self, providers: List[BaseLLMProvider]) -> BaseLLMProvider:
        """Select provider based on health metrics"""
        best_provider = None
        best_score = -1
        
        for provider in providers:
            health_data = self.health_monitor.get_provider_health(provider.config.name)
            if not health_data:
                continue
                
            # Calculate health score
            score = 0
            
            # Uptime contributes most
            score += health_data['uptime_percentage'] * 0.4
            
            # Success rate contributes
            score += health_data['avg_success_rate'] * 100 * 0.3
            
            # Low response time contributes
            if health_data['avg_response_time'] > 0:
                response_score = max(0, 100 - (health_data['avg_response_time'] * 10))
                score += response_score * 0.2
                
            # Prioritize lower priority numbers
            score += (10 - provider.config.priority) * 10
            
            if score > best_score:
                best_score = score
                best_provider = provider
                
        return best_provider or providers[0]
        
    def _select_round_robin_provider(self, providers: List[BaseLLMProvider]) -> BaseLLMProvider:
        """Select provider using round-robin strategy"""
        if not self.provider_queue:
            return providers[0]
            
        # Filter queue to only include available providers
        available_names = [p.config.name for p in providers]
        self.provider_queue = [name for name in self.provider_queue if name in available_names]
        
        if not self.provider_queue:
            return providers[0]
            
        # Select next provider in queue
        selected_name = self.provider_queue[self.current_provider_index % len(self.provider_queue)]
        self.current_provider_index += 1
        
        return self.providers[selected_name]
        
    def _select_lru_provider(self, providers: List[BaseLLMProvider]) -> BaseLLMProvider:
        """Select least recently used provider"""
        provider_with_oldest_time = None
        oldest_time = datetime.now()
        
        for provider in providers:
            provider_name = provider.config.name
            last_used = self.provider_last_used.get(provider_name, datetime.min)
            
            if last_used < oldest_time:
                oldest_time = last_used
                provider_with_oldest_time = provider
                
        return provider_with_oldest_time or providers[0]
        
    async def execute_with_failover(
        self,
        provider: BaseLLMProvider,
        operation: callable,
        *args,
        capability: Optional[ProviderCapability] = None,
        exclude_providers: Optional[List[str]] = None,
        **kwargs
    ) -> Any:
        """
        Execute operation with automatic failover
        
        Args:
            provider: Preferred provider (can be overridden by failover)
            operation: Async function to execute
            *args: Arguments for the operation
            capability: Required capability filter for failover
            exclude_providers: Provider names to exclude from failover
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Operation result or None if all providers fail
        """
        providers_to_try = [provider]
        
        # Add fallback providers
        fallback_provider = await self.get_provider(
            capability=capability,
            exclude_providers=exclude_providers + [provider.config.name]
        )
        
        if fallback_provider:
            providers_to_try.append(fallback_provider)
            
        # Try providers until one succeeds
        last_error = None
        for try_provider in providers_to_try:
            try:
                result = await asyncio.wait_for(
                    operation(try_provider, *args, **kwargs),
                    timeout=self.global_timeout
                )
                
                # Update success metrics
                self._record_success(try_provider.config.name)
                
                # Update last used time
                self.provider_last_used[try_provider.config.name] = datetime.now()
                
                return result
                
            except Exception as e:
                last_error = e
                
                # Record failure and potentially open circuit breaker
                self._record_failure(try_provider.config.name, str(e))
                
                # Continue to next provider
                logger.warning(f"Provider {try_provider.config.name} failed: {e}")
                continue
                
        # All providers failed
        self.failover_stats.failed_requests += 1
        logger.error("All providers failed during failover execution")
        
        # Call failover callbacks
        await self._notify_failover_callbacks(providers_to_try, last_error)
        
        raise Exception(f"All providers failed. Last error: {last_error}")
        
    def _record_success(self, provider_name: str):
        """Record successful request"""
        self.failover_stats.total_requests += 1
        self.failover_stats.provider_usage[provider_name] = self.failover_stats.provider_usage.get(provider_name, 0) + 1
        
        # Update circuit breaker
        if self.enable_circuit_breaker and provider_name in self.circuit_breakers:
            circuit_breaker = self.circuit_breakers[provider_name]
            
            if circuit_breaker.state == CircuitBreakerState.HALF_OPEN:
                circuit_breaker.success_count += 1
                
                if circuit_breaker.success_count >= circuit_breaker.config.success_threshold:
                    circuit_breaker.state = CircuitBreakerState.CLOSED
                    circuit_breaker.failure_count = 0
                    circuit_breaker.success_count = 0
                    logger.info(f"Circuit breaker closed for provider {provider_name}")
                    
            circuit_breaker.last_success_time = datetime.now()
            
        # Remove from failed providers
        self.failed_providers.discard(provider_name)
        
    def _record_failure(self, provider_name: str, error_message: str):
        """Record failed request"""
        if provider_name in self.failed_providers:
            return  # Already marked as failed
            
        self.failed_providers.add(provider_name)
        
        # Update circuit breaker
        if self.enable_circuit_breaker and provider_name in self.circuit_breakers:
            circuit_breaker = self.circuit_breakers[provider_name]
            circuit_breaker.failure_count += 1
            circuit_breaker.last_failure_time = datetime.now()
            
            if circuit_breaker.state == CircuitBreakerState.CLOSED:
                if circuit_breaker.failure_count >= circuit_breaker.config.failure_threshold:
                    circuit_breaker.state = CircuitBreakerState.OPEN
                    logger.warning(f"Circuit breaker opened for provider {provider_name}")
                    
            elif circuit_breaker.state == CircuitBreakerState.HALF_OPEN:
                circuit_breaker.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker reopened for provider {provider_name}")
                
    async def _check_circuit_breakers(self):
        """Check if any circuit breakers should transition state"""
        if not self.enable_circuit_breaker:
            return
            
        for provider_name, circuit_breaker in self.circuit_breakers.items():
            if circuit_breaker.state == CircuitBreakerState.OPEN:
                # Check if recovery timeout has passed
                if circuit_breaker.last_failure_time:
                    time_since_failure = datetime.now() - circuit_breaker.last_failure_time
                    if time_since_failure.total_seconds() >= circuit_breaker.config.recovery_timeout:
                        circuit_breaker.state = CircuitBreakerState.HALF_OPEN
                        circuit_breaker.failure_count = 0
                        circuit_breaker.success_count = 0
                        logger.info(f"Circuit breaker half-opened for provider {provider_name}")
                        
    async def _on_health_alert(self, alert: HealthAlert):
        """Handle health monitor alerts"""
        if alert.severity == 'critical' and not alert.resolved:
            # Critical provider failure - record in circuit breaker
            self._record_failure(alert.provider_name, alert.message)
            
            # Trigger failover if this was the preferred provider
            await self._notify_failover_callbacks([self.providers.get(alert.provider_name)], None)
            
    async def _notify_failover_callbacks(self, failed_providers: List[BaseLLMProvider], error: Optional[Exception]):
        """Notify all failover callbacks"""
        for callback in self.failover_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(failed_providers, error)
                else:
                    callback(failed_providers, error)
            except Exception as e:
                logger.error(f"Error in failover callback: {e}")
                
    def get_failover_stats(self) -> Dict[str, Any]:
        """Get failover statistics"""
        return {
            'total_requests': self.failover_stats.total_requests,
            'failed_requests': self.failover_stats.failed_requests,
            'success_rate': (
                (self.failover_stats.total_requests - self.failover_stats.failed_requests) / 
                max(1, self.failover_stats.total_requests)
            ),
            'failover_events': self.failover_stats.failover_events,
            'provider_usage': self.failover_stats.provider_usage,
            'active_providers': len([p for p in self.providers.values() if self._is_provider_healthy(p.config.name)]),
            'failed_providers': list(self.failed_providers),
            'circuit_breaker_states': {
                name: cb.state.value
                for name, cb in self.circuit_breakers.items()
            }
        }
        
    def reset_circuit_breaker(self, provider_name: str):
        """Manually reset circuit breaker for a provider"""
        if provider_name in self.circuit_breakers:
            self.circuit_breakers[provider_name].state = CircuitBreakerState.CLOSED
            self.circuit_breakers[provider_name].failure_count = 0
            self.circuit_breakers[provider_name].success_count = 0
            self.failed_providers.discard(provider_name)
            logger.info(f"Manually reset circuit breaker for provider {provider_name}")
            
    async def start_circuit_breaker_monitoring(self):
        """Start monitoring circuit breaker states"""
        while True:
            await self._check_circuit_breakers()
            await asyncio.sleep(10)  # Check every 10 seconds
            
    def set_strategy(self, strategy: FailoverStrategy):
        """Change failover strategy"""
        self.strategy = strategy
        logger.info(f"Failover strategy changed to {strategy.value}")
        
    def get_available_providers_summary(self) -> Dict[str, Any]:
        """Get summary of available providers"""
        total_providers = len(self.providers)
        healthy_providers = sum(1 for provider in self.providers.values() if self._is_provider_healthy(provider.config.name))
        
        return {
            'total_providers': total_providers,
            'healthy_providers': healthy_providers,
            'failed_providers': len(self.failed_providers),
            'strategy': self.strategy.value,
            'circuit_breaker_enabled': self.enable_circuit_breaker
        }