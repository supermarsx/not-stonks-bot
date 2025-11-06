"""
Base Provider Interface - Abstract base class for all LLM providers

This module defines the common interface that all LLM providers must implement,
ensuring consistent functionality across different provider types.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Callable, AsyncGenerator
from enum import Enum, Flag
from dataclasses import dataclass, field
import time
import json
from datetime import datetime, timedelta


class ProviderCapability(Flag):
    """Capabilities that a provider may support"""
    FUNCTION_CALLING = 1 << 0      # Tool/function calling
    STREAMING = 1 << 1             # Streaming responses
    VISION = 1 << 2                # Image understanding
    LONG_CONTEXT = 1 << 3          # Extended context window
    BATCH_REQUESTS = 1 << 4        # Batch processing
    EMBEDDINGS = 1 << 5            # Text embeddings
    JSON_MODE = 1 << 6             # Structured JSON output
    SYSTEM_PROMPTS = 1 << 7        # System message support
    MULTIMODAL = 1 << 8            # Multiple input types


class ProviderStatus(Enum):
    """Provider operational status"""
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


class ProviderHealth(Enum):
    """Provider health levels"""
    CRITICAL = "critical"          # Provider down, immediate failover
    WARNING = "warning"            # Performance degradation
    HEALTHY = "healthy"            # Full functionality
    UNKNOWN = "unknown"            # Health not yet determined


@dataclass
class ModelInfo:
    """Information about a specific model"""
    name: str
    display_name: str
    context_length: int
    capabilities: ProviderCapability
    cost_per_token: float = 0.0
    tokens_per_second: Optional[float] = None
    description: str = ""
    version: str = ""
    deprecated: bool = False


@dataclass
class RateLimit:
    """Rate limiting configuration"""
    requests_per_minute: int = 60
    tokens_per_minute: int = 100000
    concurrent_requests: int = 10
    burst_limit: int = 100


@dataclass 
class ProviderStats:
    """Provider usage statistics"""
    total_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    avg_response_time: float = 0.0
    error_rate: float = 0.0
    last_used: Optional[datetime] = None
    uptime_percentage: float = 100.0
    requests_last_hour: int = 0


@dataclass
class HealthMetrics:
    """Health monitoring metrics"""
    response_time: float
    success_rate: float
    error_rate: float
    last_check: datetime
    consecutive_failures: int
    is_healthy: bool
    message: str = ""


@dataclass
class ProviderConfig:
    """Configuration for a provider instance"""
    name: str
    provider_type: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit: RateLimit = field(default_factory=RateLimit)
    health_check_interval: int = 60  # seconds
    enable_caching: bool = True
    cache_ttl: int = 3600  # seconds
    priority: int = 5      # Lower numbers = higher priority
    max_concurrent_requests: int = 10
    compression: bool = True
    extra_config: Dict[str, Any] = field(default_factory=dict)


class BaseLLMProvider(ABC):
    """
    Abstract base class for all LLM providers
    
    All providers must implement these methods to ensure consistent functionality
    across different LLM services and local models.
    """
    
    def __init__(self, config: ProviderConfig):
        """
        Initialize the provider with configuration
        
        Args:
            config: Provider configuration
        """
        self.config = config
        self.stats = ProviderStats()
        self.status = ProviderStatus.UNKNOWN
        self.capabilities: ProviderCapability = ProviderCapability(0)
        self.models: Dict[str, ModelInfo] = {}
        self.last_health_check: Optional[datetime] = None
        self._health_metrics = HealthMetrics(
            response_time=0.0,
            success_rate=0.0,
            error_rate=1.0,
            last_check=datetime.now(),
            consecutive_failures=0,
            is_healthy=False
        )
        
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the provider connection and validate configuration
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
        
    @abstractmethod
    async def health_check(self) -> HealthMetrics:
        """
        Perform health check on the provider
        
        Returns:
            HealthMetrics with current provider health status
        """
        pass
        
    @abstractmethod
    async def list_models(self) -> List[ModelInfo]:
        """
        List available models from this provider
        
        Returns:
            List of ModelInfo objects
        """
        pass
        
    @abstractmethod
    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """
        Generate completion using the provider
        
        Args:
            messages: Conversation messages
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Function calling tools
            stream: Whether to stream response
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Completion response or async generator for streaming
        """
        pass
        
    @abstractmethod
    def supports_capability(self, capability: ProviderCapability) -> bool:
        """
        Check if provider supports a specific capability
        
        Args:
            capability: Capability to check
            
        Returns:
            True if capability is supported
        """
        pass
        
    async def get_stats(self) -> ProviderStats:
        """
        Get provider usage statistics
        
        Returns:
            ProviderStats object
        """
        return self.stats
        
    async def reset_stats(self):
        """Reset provider statistics"""
        self.stats = ProviderStats()
        
    def get_status(self) -> ProviderStatus:
        """
        Get current provider status
        
        Returns:
            ProviderStatus enum
        """
        return self.status
        
    def get_health_metrics(self) -> HealthMetrics:
        """
        Get current health metrics
        
        Returns:
            HealthMetrics object
        """
        return self._health_metrics
        
    def update_health(self, metrics: HealthMetrics):
        """Update health metrics"""
        self._health_metrics = metrics
        self.last_health_check = datetime.now()
        
        # Update status based on health
        if metrics.consecutive_failures >= 5:
            self.status = ProviderStatus.ERROR
        elif not metrics.is_healthy or metrics.error_rate > 0.5:
            self.status = ProviderStatus.DEGRADED
        else:
            self.status = ProviderStatus.HEALTHY
            
    def increment_request_count(self, tokens_used: int, cost: float, response_time: float):
        """Update provider statistics"""
        now = datetime.now()
        
        self.stats.total_requests += 1
        self.stats.total_tokens += tokens_used
        self.stats.total_cost += cost
        self.stats.last_used = now
        
        # Update average response time (exponential moving average)
        if self.stats.avg_response_time == 0:
            self.stats.avg_response_time = response_time
        else:
            alpha = 0.1  # Smoothing factor
            self.stats.avg_response_time = (
                alpha * response_time + (1 - alpha) * self.stats.avg_response_time
            )
            
        # Calculate requests in last hour
        if now - timedelta(minutes=60) < (self.last_request_time or now):
            self.stats.requests_last_hour += 1
        else:
            self.stats.requests_last_hour = 1
            
        self.last_request_time = now
        
    async def validate_api_key(self) -> bool:
        """
        Validate the API key or configuration
        
        Returns:
            True if API key is valid
        """
        # Default implementation - override in specific providers
        try:
            await self.health_check()
            return self._health_metrics.is_healthy
        except Exception:
            return False
            
    async def get_model_capabilities(self, model_name: str) -> ProviderCapability:
        """
        Get capabilities for a specific model
        
        Args:
            model_name: Name of the model
            
        Returns:
            ProviderCapability flags for the model
        """
        model_info = self.models.get(model_name)
        return model_info.capabilities if model_info else ProviderCapability(0)
        
    async def estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Estimate cost for a completion
        
        Args:
            model: Model name
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            
        Returns:
            Estimated cost in USD
        """
        model_info = self.models.get(model)
        if not model_info or model_info.cost_per_token == 0:
            return 0.0
            
        total_tokens = prompt_tokens + completion_tokens
        return total_tokens * model_info.cost_per_token
        
    def __str__(self) -> str:
        """String representation"""
        return f"{self.__class__.__name__}(name={self.config.name}, status={self.status.value})"
        
    def __repr__(self) -> str:
        """Detailed string representation"""
        return (f"{self.__class__.__name__}("
                f"config={self.config}, "
                f"stats={self.stats}, "
                f"status={self.status.value})")