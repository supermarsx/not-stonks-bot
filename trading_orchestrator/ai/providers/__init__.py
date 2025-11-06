"""
LLM Provider System - Extended Support for Local Models and APIs

This module provides comprehensive support for multiple LLM providers including:
- OpenAI and Anthropic (existing)
- Ollama (local models)
- LocalAI (OpenAI-compatible)
- vLLM (high-performance local serving)
- OpenAI-compatible API endpoints

Features:
- Provider abstraction and factory patterns
- Automatic failover and health monitoring
- Provider-specific optimizations
- Rate limiting per provider
- Capability detection
- Configuration management
"""

from .base_provider import (
    BaseLLMProvider,
    ProviderCapability,
    ProviderHealth,
    ProviderConfig,
    RateLimit,
    ProviderStats,
    ModelInfo,
    HealthMetrics
)

from .factory import ProviderFactory

from .ollama_provider import OllamaProvider
from .localai_provider import LocalAIProvider
from .vllm_provider import VLLMProvider
from .openai_compatible_provider import OpenAICompatibleProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider

from .health_monitor import ProviderHealthMonitor
from .failover_manager import ProviderFailoverManager
from .rate_limiter import ProviderRateLimiter

from .enhanced_ai_models_manager import EnhancedAIModelsManager, RequestConfig, ModelTier, RequestPriority

__all__ = [
    'BaseLLMProvider',
    'ProviderCapability', 
    'ProviderHealth',
    'ProviderConfig',
    'RateLimit',
    'ProviderStats',
    'ModelInfo',
    'HealthMetrics',
    'ProviderFactory',
    'OllamaProvider',
    'LocalAIProvider', 
    'VLLMProvider',
    'OpenAICompatibleProvider',
    'OpenAIProvider',
    'AnthropicProvider',
    'ProviderHealthMonitor',
    'ProviderFailoverManager',
    'ProviderRateLimiter',
    'EnhancedAIModelsManager',
    'RequestConfig',
    'ModelTier',
    'RequestPriority'
]
