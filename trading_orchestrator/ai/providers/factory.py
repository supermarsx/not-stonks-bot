"""
Provider Factory - Creates and manages LLM providers

Factory pattern implementation for creating and configuring different types of LLM providers
with automatic dependency management and configuration validation.
"""

from typing import Dict, List, Any, Optional, Type, Union
import importlib
import asyncio
import aiohttp
from loguru import logger

from .base_provider import (
    BaseLLMProvider,
    ProviderConfig,
    ProviderCapability,
    HealthMetrics,
    RateLimit
)
from .ollama_provider import OllamaProvider
from .localai_provider import LocalAIProvider
from .vllm_provider import VLLMProvider


class ProviderFactory:
    """
    Factory for creating and managing LLM providers
    
    Supports dependency checking, automatic configuration, and provider lifecycle management.
    """
    
    # Registry of available provider types
    PROVIDER_REGISTRY = {
        'openai': 'openai',
        'anthropic': 'anthropic', 
        'ollama': OllamaProvider,
        'localai': LocalAIProvider,
        'vllm': VLLMProvider,
        'openai_compatible': 'openai_compatible',
        'custom': 'custom'
    }
    
    def __init__(self):
        """Initialize provider factory"""
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.available_provider_types = set()
        self.dependency_status = {}
        
    async def create_provider(
        self,
        config: ProviderConfig,
        auto_dependencies: bool = True
    ) -> BaseLLMProvider:
        """
        Create and initialize a provider
        
        Args:
            config: Provider configuration
            auto_dependencies: Automatically check/install dependencies
            
        Returns:
            Initialized provider instance
        """
        provider_type = config.provider_type.lower()
        
        # Check if we have a custom implementation
        provider_class = self.PROVIDER_REGISTRY.get(provider_type)
        
        if isinstance(provider_class, str):
            # Handle string-based providers (OpenAI, Anthropic, etc.)
            return await self._create_string_provider(config, provider_class)
        else:
            # Handle class-based providers (Ollama, LocalAI, vLLM)
            return await self._create_class_provider(config, provider_class)
            
    async def _create_string_provider(
        self,
        config: ProviderConfig,
        provider_type: str
    ) -> BaseLLMProvider:
        """Create a string-based provider (OpenAI, Anthropic, etc.)"""
        
        if provider_type == 'openai':
            # Check OpenAI dependency
            if not await self._check_dependency('openai', 'openai'):
                raise ImportError("OpenAI package not installed. Install with: pip install openai")
                
            from .openai_provider import OpenAIProvider
            provider = OpenAIProvider(config)
            
        elif provider_type == 'anthropic':
            # Check Anthropic dependency  
            if not await self._check_dependency('anthropic', 'anthropic'):
                raise ImportError("Anthropic package not installed. Install with: pip install anthropic")
                
            from .anthropic_provider import AnthropicProvider
            provider = AnthropicProvider(config)
            
        elif provider_type == 'openai_compatible':
            # OpenAI-compatible API provider
            from .openai_compatible_provider import OpenAICompatibleProvider
            provider = OpenAICompatibleProvider(config)
            
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")
            
        # Initialize the provider
        await provider.initialize()
        
        # Store in registry
        self.providers[config.name] = provider
        
        logger.info(f"Created {provider_type} provider: {config.name}")
        return provider
        
    async def _create_class_provider(
        self,
        config: ProviderConfig,
        provider_class: Type[BaseLLMProvider]
    ) -> BaseLLMProvider:
        """Create a class-based provider"""
        
        if not provider_class:
            raise ValueError(f"No provider class found for type: {config.provider_type}")
            
        provider = provider_class(config)
        
        # Initialize the provider
        await provider.initialize()
        
        # Store in registry
        self.providers[config.name] = provider
        
        logger.info(f"Created {provider_class.__name__} provider: {config.name}")
        return provider
        
    async def _check_dependency(self, package_name: str, import_name: str) -> bool:
        """
        Check if a dependency is available
        
        Args:
            package_name: Package name for pip install
            import_name: Import name for Python
            
        Returns:
            True if dependency is available
        """
        if package_name in self.dependency_status:
            return self.dependency_status[package_name]
            
        try:
            importlib.import_module(import_name)
            self.dependency_status[package_name] = True
            return True
        except ImportError:
            self.dependency_status[package_name] = False
            logger.warning(f"Missing dependency: {package_name}")
            return False
            
    async def create_from_config_dict(
        self,
        config_dict: Dict[str, Any]
    ) -> BaseLLMProvider:
        """
        Create provider from configuration dictionary
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Initialized provider
        """
        # Convert dictionary to ProviderConfig
        rate_limit_dict = config_dict.get('rate_limit', {})
        rate_limit = RateLimit(**rate_limit_dict) if rate_limit_dict else RateLimit()
        
        config = ProviderConfig(
            name=config_dict['name'],
            provider_type=config_dict['provider_type'],
            base_url=config_dict.get('base_url'),
            api_key=config_dict.get('api_key'),
            timeout=config_dict.get('timeout', 30.0),
            max_retries=config_dict.get('max_retries', 3),
            retry_delay=config_dict.get('retry_delay', 1.0),
            rate_limit=rate_limit,
            health_check_interval=config_dict.get('health_check_interval', 60),
            enable_caching=config_dict.get('enable_caching', True),
            cache_ttl=config_dict.get('cache_ttl', 3600),
            priority=config_dict.get('priority', 5),
            max_concurrent_requests=config_dict.get('max_concurrent_requests', 10),
            compression=config_dict.get('compression', True),
            extra_config=config_dict.get('extra_config', {})
        )
        
        return await self.create_provider(config)
        
    async def batch_create_providers(
        self,
        configs: List[Union[ProviderConfig, Dict[str, Any]]]
    ) -> List[BaseLLMProvider]:
        """
        Create multiple providers in parallel
        
        Args:
            configs: List of provider configurations
            
        Returns:
            List of initialized providers
        """
        tasks = []
        
        for config in configs:
            if isinstance(config, dict):
                task = self.create_from_config_dict(config)
            else:
                task = self.create_provider(config)
                
            tasks.append(task)
            
        providers = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log errors
        successful_providers = []
        for i, provider in enumerate(providers):
            if isinstance(provider, Exception):
                logger.error(f"Failed to create provider {i}: {provider}")
            else:
                successful_providers.append(provider)
                
        return successful_providers
        
    def get_provider(self, name: str) -> Optional[BaseLLMProvider]:
        """
        Get a provider by name
        
        Args:
            name: Provider name
            
        Returns:
            Provider instance or None if not found
        """
        return self.providers.get(name)
        
    def list_providers(self) -> Dict[str, BaseLLMProvider]:
        """
        Get all registered providers
        
        Returns:
            Dictionary of name -> provider mapping
        """
        return self.providers.copy()
        
    def remove_provider(self, name: str) -> bool:
        """
        Remove a provider from registry
        
        Args:
            name: Provider name to remove
            
        Returns:
            True if provider was removed
        """
        if name in self.providers:
            provider = self.providers[name]
            # Cleanup if provider has cleanup method
            if hasattr(provider, 'cleanup'):
                asyncio.create_task(provider.cleanup())
                
            del self.providers[name]
            logger.info(f"Removed provider: {name}")
            return True
        return False
        
    async def get_healthy_providers(
        self,
        capability: Optional[ProviderCapability] = None
    ) -> List[BaseLLMProvider]:
        """
        Get all healthy providers that support a capability
        
        Args:
            capability: Required capability filter (None for all healthy)
            
        Returns:
            List of healthy providers
        """
        healthy_providers = []
        
        for provider in self.providers.values():
            if provider.get_status().value in ['healthy', 'degraded']:
                if capability is None or provider.supports_capability(capability):
                    healthy_providers.append(provider)
                    
        return healthy_providers
        
    async def get_best_provider(
        self,
        capability: ProviderCapability,
        exclude_names: Optional[List[str]] = None
    ) -> Optional[BaseLLMProvider]:
        """
        Get the best available provider for a capability
        
        Args:
            capability: Required capability
            exclude_names: Provider names to exclude
            
        Returns:
            Best provider or None if no healthy providers
        """
        healthy_providers = await self.get_healthy_providers(capability)
        
        if exclude_names:
            healthy_providers = [
                p for p in healthy_providers 
                if p.config.name not in exclude_names
            ]
            
        if not healthy_providers:
            return None
            
        # Sort by priority (lower number = higher priority)
        healthy_providers.sort(key=lambda p: p.config.priority)
        
        return healthy_providers[0]
        
    def get_available_provider_types(self) -> List[str]:
        """
        Get list of available provider types
        
        Returns:
            List of provider type strings
        """
        return list(self.PROVIDER_REGISTRY.keys())
        
    async def validate_all_providers(self) -> Dict[str, bool]:
        """
        Validate all registered providers
        
        Returns:
            Dictionary mapping provider name to validation status
        """
        results = {}
        
        for name, provider in self.providers.items():
            try:
                is_valid = await provider.validate_api_key()
                results[name] = is_valid
                
                if not is_valid:
                    logger.warning(f"Provider {name} validation failed")
                    
            except Exception as e:
                logger.error(f"Error validating provider {name}: {e}")
                results[name] = False
                
        return results
        
    def get_dependency_status(self) -> Dict[str, bool]:
        """Get status of all dependencies"""
        return self.dependency_status.copy()
        
    async def check_all_dependencies(self) -> Dict[str, bool]:
        """Check all provider dependencies"""
        # Common dependencies
        dependencies = [
            ('openai', 'openai'),
            ('anthropic', 'anthropic'),
            ('requests', 'requests'),
            ('aiohttp', 'aiohttp')
        ]
        
        status = {}
        for package, import_name in dependencies:
            status[package] = await self._check_dependency(package, import_name)
            
        return status