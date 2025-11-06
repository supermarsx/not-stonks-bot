"""
Enhanced AI Models Manager - Multi-provider LLM orchestration

Enhanced version that integrates with the new provider system for:
- Multiple provider support (OpenAI, Anthropic, Ollama, LocalAI, vLLM, etc.)
- Automatic failover between providers
- Health monitoring and circuit breaker patterns
- Provider-specific optimizations
- Capability-based model selection
- Cost and performance tracking across providers
"""

from typing import Dict, List, Any, Optional, Callable, Union, AsyncGenerator
from enum import Enum
from dataclasses import dataclass, field
import json
import asyncio
import time
from datetime import datetime

from loguru import logger

from ..providers import (
    BaseLLMProvider,
    ProviderCapability,
    ProviderFactory,
    ProviderHealthMonitor,
    ProviderFailoverManager,
    ProviderConfig,
    RateLimit,
    FailoverStrategy
)


class ModelTier(Enum):
    """LLM tier classification with enhanced capabilities"""
    REASONING = "reasoning"     # High-quality, slower, expensive (GPT-4, Claude 3.5)
    FAST = "fast"               # Balanced quality/speed/cost (GPT-3.5, Claude Haiku)
    LOCAL = "local"             # Local models, fastest, cheapest (Ollama, vLLM)
    EMBEDDING = "embedding"     # Specialized embedding models
    VISION = "vision"           # Vision-capable models
    CODING = "coding"           # Code-optimized models


class RequestPriority(Enum):
    """Request priority levels"""
    CRITICAL = 1    # Risk checks, stop losses
    HIGH = 2        # Market analysis, strategy generation
    MEDIUM = 3      # Regular trading decisions
    LOW = 4         # Research, backtesting


@dataclass
class RequestConfig:
    """Configuration for individual requests"""
    priority: RequestPriority = RequestPriority.MEDIUM
    max_retries: int = 3
    timeout: float = 30.0
    prefer_cache: bool = True
    allow_failover: bool = True
    cost_limit: Optional[float] = None


@dataclass
class ModelPreference:
    """Preference configuration for a model tier"""
    tier: ModelTier
    preferred_models: List[str] = field(default_factory=list)
    fallback_models: List[str] = field(default_factory=list)
    capability_requirements: List[ProviderCapability] = field(default_factory=list)
    max_cost_per_request: Optional[float] = None
    max_response_time: Optional[float] = None


class EnhancedAIModelsManager:
    """
    Enhanced AI Models Manager with comprehensive provider support
    
    Features:
    - Multi-provider orchestration with automatic failover
    - Health monitoring and circuit breaker patterns
    - Request priority and cost-aware routing
    - Provider-specific optimizations
    - Comprehensive usage tracking and analytics
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize enhanced AI models manager
        
        Args:
            config: Configuration dictionary with provider settings
        """
        self.config = config or {}
        self.provider_factory = ProviderFactory()
        self.health_monitor = ProviderHealthMonitor(
            check_interval=self.config.get('health_check_interval', 60)
        )
        self.failover_manager = ProviderFailoverManager(
            health_monitor=self.health_monitor,
            strategy=FailoverStrategy.HEALTH_BASED,
            global_timeout=self.config.get('global_timeout', 30.0)
        )
        
        # State tracking
        self.conversation_history: Dict[str, List[Dict]] = {}
        self.tool_functions: Dict[str, Callable] = {}
        self.usage_stats: Dict[str, Dict] = {}
        self.cached_responses: Dict[str, Any] = {}
        self.request_counters: Dict[str, int] = {}
        
        # Model preferences by tier
        self.model_preferences: Dict[ModelTier, ModelPreference] = {
            ModelTier.REASONING: ModelPreference(
                tier=ModelTier.REASONING,
                preferred_models=['claude-3-5-sonnet', 'gpt-4-turbo'],
                capability_requirements=[ProviderCapability.FUNCTION_CALLING],
                max_cost_per_request=0.10,
                max_response_time=30.0
            ),
            ModelTier.FAST: ModelPreference(
                tier=ModelTier.FAST,
                preferred_models=['gpt-3.5-turbo', 'claude-3-haiku'],
                capability_requirements=[ProviderCapability.FUNCTION_CALLING],
                max_cost_per_request=0.01,
                max_response_time=10.0
            ),
            ModelTier.LOCAL: ModelPreference(
                tier=ModelTier.LOCAL,
                preferred_models=['llama3', 'mistral', 'codellama'],
                fallback_models=['llama2', 'mixtral'],
                max_cost_per_request=0.0,
                max_response_time=5.0
            ),
            ModelTier.EMBEDDING: ModelPreference(
                tier=ModelTier.EMBEDDING,
                preferred_models=['text-embedding-ada-002'],
                capability_requirements=[ProviderCapability.EMBEDDINGS],
                max_cost_per_request=0.001,
                max_response_time=5.0
            )
        }
        
        # Register failover event handlers
        self.failover_manager.add_failover_callback(self._on_failover_event)
        
        logger.info("Enhanced AI Models Manager initialized")
        
    async def initialize(self):
        """Initialize the manager with configured providers"""
        if not self.config:
            # Create default providers if no config provided
            await self._create_default_providers()
        else:
            # Initialize providers from configuration
            await self._initialize_providers_from_config()
            
        # Start health monitoring
        await self.health_monitor.start_monitoring()
        
        logger.info(f"Enhanced AI Models Manager initialized with {len(self.failover_manager.providers)} providers")
        
    async def _create_default_providers(self):
        """Create a set of default providers for testing"""
        default_configs = [
            # OpenAI (if API key available)
            ProviderConfig(
                name="openai_default",
                provider_type="openai",
                api_key=None,  # Will use environment variable
                priority=1,
                rate_limit=RateLimit(requests_per_minute=5000, tokens_per_minute=2000000)
            ),
            # Anthropic (if API key available)
            ProviderConfig(
                name="anthropic_default", 
                provider_type="anthropic",
                api_key=None,  # Will use environment variable
                priority=2,
                rate_limit=RateLimit(requests_per_minute=1000, tokens_per_minute=400000)
            ),
            # Ollama (local)
            ProviderConfig(
                name="ollama_local",
                provider_type="ollama",
                base_url="http://localhost:11434",
                priority=3,
                rate_limit=RateLimit(requests_per_minute=60, tokens_per_minute=100000)
            ),
            # LocalAI (local)
            ProviderConfig(
                name="localai_local",
                provider_type="localai", 
                base_url="http://localhost:8080",
                priority=4,
                rate_limit=RateLimit(requests_per_minute=60, tokens_per_minute=100000)
            )
        ]
        
        # Try to initialize providers
        for config in default_configs:
            try:
                provider = await self.provider_factory.create_provider(config)
                if provider:
                    self.failover_manager.register_provider(provider)
                    logger.info(f"Initialized provider: {config.name}")
            except Exception as e:
                logger.warning(f"Failed to initialize provider {config.name}: {e}")
                
    async def _initialize_providers_from_config(self):
        """Initialize providers from configuration"""
        provider_configs = self.config.get('providers', [])
        
        for provider_config in provider_configs:
            try:
                provider = await self.provider_factory.create_from_config_dict(provider_config)
                if provider:
                    self.failover_manager.register_provider(provider)
                    logger.info(f"Initialized provider: {provider_config['name']}")
            except Exception as e:
                logger.error(f"Failed to initialize provider {provider_config['name']}: {e}")
                
    def register_tool(self, name: str, function: Callable):
        """Register a tool function that models can call"""
        self.tool_functions[name] = function
        logger.info(f"Registered tool: {name}")
        
    def get_model_for_task(
        self,
        task_type: str,
        tier: Optional[ModelTier] = None,
        capability_requirements: Optional[List[ProviderCapability]] = None
    ) -> str:
        """
        Get appropriate model for a specific task
        
        Args:
            task_type: Type of task (e.g., 'strategy_analysis', 'risk_check')
            tier: Preferred model tier
            capability_requirements: Required capabilities
            
        Returns:
            Model name to use
        """
        # Determine appropriate tier from task type
        if tier is None:
            tier = self._get_tier_for_task(task_type)
            
        # Get preferences for this tier
        preference = self.model_preferences.get(tier)
        if not preference:
            return 'gpt-3.5-turbo'  # Default fallback
            
        # Filter capabilities
        if capability_requirements:
            preference.capability_requirements.extend(capability_requirements)
            
        # Get available providers with capabilities
        available_providers = []
        for provider in self.failover_manager.providers.values():
            has_capabilities = all(
                provider.supports_capability(cap) 
                for cap in preference.capability_requirements
            )
            if has_capabilities:
                available_providers.append(provider)
                
        # Select best provider
        if available_providers:
            # Sort by preference
            for preferred_model in preference.preferred_models:
                for provider in available_providers:
                    if provider.config.name == preferred_model or preferred_model in provider.models:
                        return preferred_model
                        
            # Use first available if no preference matches
            return list(available_providers[0].models.keys())[0]
            
        # Fallback to default
        return preference.preferred_models[0] if preference.preferred_models else 'gpt-3.5-turbo'
        
    def _get_tier_for_task(self, task_type: str) -> ModelTier:
        """Determine appropriate tier for task type"""
        task_tier_mapping = {
            'strategy_analysis': ModelTier.REASONING,
            'market_analysis': ModelTier.REASONING,
            'backtest_review': ModelTier.REASONING,
            'risk_check': ModelTier.FAST,
            'quick_decision': ModelTier.FAST,
            'order_routing': ModelTier.FAST,
            'position_sizing': ModelTier.FAST,
            'high_frequency': ModelTier.LOCAL,
            'embedding_generation': ModelTier.EMBEDDING,
            'vision_analysis': ModelTier.VISION,
            'code_generation': ModelTier.CODING
        }
        
        return task_tier_mapping.get(task_type, ModelTier.FAST)
        
    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        task_type: str = 'general',
        tier: Optional[ModelTier] = None,
        request_config: Optional[RequestConfig] = None,
        model: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        stream: bool = False,
        session_id: Optional[str] = None,
        **kwargs
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """
        Generate completion with enhanced provider support and failover
        
        Args:
            messages: Conversation messages
            task_type: Type of task for model selection
            tier: Preferred model tier
            request_config: Request configuration
            model: Specific model to use (overrides auto-selection)
            tools: Function calling tools
            stream: Whether to stream response
            session_id: Session ID for conversation history
            **kwargs: Additional parameters
            
        Returns:
            Completion response or async generator for streaming
        """
        request_config = request_config or RequestConfig()
        start_time = time.time()
        
        try:
            # Add to conversation history if session provided
            if session_id:
                if session_id not in self.conversation_history:
                    self.conversation_history[session_id] = []
                self.conversation_history[session_id].extend(messages)
                messages = self.conversation_history[session_id]
                
            # Determine model to use
            if model is None:
                model = self.get_model_for_task(task_type, tier)
                
            # Get provider capabilities needed
            capability_requirements = []
            if tools:
                capability_requirements.append(ProviderCapability.FUNCTION_CALLING)
            if stream:
                capability_requirements.append(ProviderCapability.STREAMING)
                
            # Get best provider
            if request_config.allow_failover:
                provider = await self.failover_manager.get_provider(
                    capability=capability_requirements[0] if capability_requirements else None,
                    preferred_provider=None  # Let failover manager decide
                )
            else:
                # Get specific provider for this model
                provider = None
                for p in self.failover_manager.providers.values():
                    if model in p.models:
                        provider = p
                        break
                        
            if not provider:
                raise Exception(f"No provider available for model {model}")
                
            # Execute with failover if enabled
            if request_config.allow_failover:
                result = await self.failover_manager.execute_with_failover(
                    provider=provider,
                    operation=self._generate_completion_on_provider,
                    messages=messages,
                    model=model,
                    tools=tools,
                    stream=stream,
                    request_config=request_config,
                    **kwargs
                )
            else:
                result = await self._generate_completion_on_provider(
                    provider, messages, model, tools, stream, request_config, **kwargs
                )
                
            # Store response in history if successful
            if session_id and result.get('content'):
                self.conversation_history[session_id].append({
                    'role': 'assistant',
                    'content': result['content'],
                    'model': model,
                    'provider': result.get('provider', 'unknown')
                })
                
            # Update usage stats
            self._update_usage_stats(result, model, provider.config.name)
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            return {
                'error': str(e),
                'model': model,
                'provider': 'unknown',
                'response_time': time.time() - start_time
            }
            
    async def _generate_completion_on_provider(
        self,
        provider: BaseLLMProvider,
        messages: List[Dict[str, str]],
        model: str,
        tools: Optional[List[Dict]],
        stream: bool,
        request_config: RequestConfig,
        **kwargs
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Generate completion on a specific provider"""
        return await provider.generate_completion(
            messages=messages,
            model=model,
            temperature=kwargs.get('temperature', 0.7),
            max_tokens=kwargs.get('max_tokens'),
            tools=tools,
            stream=stream,
            **kwargs
        )
        
    def _update_usage_stats(
        self,
        result: Dict[str, Any],
        model: str,
        provider_name: str
    ):
        """Update comprehensive usage statistics"""
        usage = result.get('usage', {})
        total_tokens = usage.get('total_tokens', 0)
        cost = result.get('cost_estimate', 0.0)
        
        # Update per-model stats
        if model not in self.usage_stats:
            self.usage_stats[model] = {
                'total_tokens': 0,
                'total_cost': 0.0,
                'request_count': 0,
                'providers_used': set(),
                'avg_response_time': 0.0,
                'last_used': None
            }
            
        model_stats = self.usage_stats[model]
        model_stats['total_tokens'] += total_tokens
        model_stats['total_cost'] += cost
        model_stats['request_count'] += 1
        model_stats['providers_used'].add(provider_name)
        model_stats['last_used'] = datetime.now()
        
        # Update average response time
        response_time = result.get('response_time', 0)
        if model_stats['avg_response_time'] == 0:
            model_stats['avg_response_time'] = response_time
        else:
            alpha = 0.1
            model_stats['avg_response_time'] = (
                alpha * response_time + (1 - alpha) * model_stats['avg_response_time']
            )
            
        # Update per-provider stats
        provider_key = f"{provider_name}_{model}"
        if provider_key not in self.usage_stats:
            self.usage_stats[provider_key] = {
                'total_tokens': 0,
                'total_cost': 0.0,
                'request_count': 0,
                'success_rate': 1.0,
                'avg_response_time': 0.0
            }
            
        provider_stats = self.usage_stats[provider_key]
        provider_stats['total_tokens'] += total_tokens
        provider_stats['total_cost'] += cost
        provider_stats['request_count'] += 1
        
        # Calculate success rate
        if result.get('error'):
            provider_stats['success_rate'] = max(0, provider_stats['success_rate'] - 0.1)
        else:
            provider_stats['success_rate'] = min(1, provider_stats['success_rate'] + 0.05)
            
    async def _on_failover_event(self, failed_providers: List[BaseLLMProvider], error: Optional[Exception]):
        """Handle failover events"""
        if error:
            logger.warning(f"Failover occurred. Failed providers: {[p.config.name for p in failed_providers]}. Error: {error}")
        else:
            logger.info(f"Provider failover event. Affected providers: {[p.config.name for p in failed_providers]}")
            
    async def get_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate embeddings using best available embedding model"""
        if model is None:
            model = self.get_model_for_task('embedding_generation')
            
        # Get provider that supports embeddings
        target_provider = None
        if provider:
            target_provider = self.failover_manager.providers.get(provider)
        else:
            # Find any provider that supports embeddings
            for prov in self.failover_manager.providers.values():
                if prov.supports_capability(ProviderCapability.EMBEDDINGS):
                    target_provider = prov
                    break
                    
        if not target_provider:
            raise Exception("No provider available that supports embeddings")
            
        # Generate embeddings
        if hasattr(target_provider, 'embeddings'):
            return await target_provider.embeddings(texts=texts, model=model)
        else:
            # Try as regular completion if no embeddings method
            results = []
            for text in texts:
                response = await target_provider.generate_completion(
                    messages=[{"role": "user", "content": f"Generate a compact vector representation for: {text}"}],
                    model=model
                )
                results.append(response.get('content', ''))
                
            return {'data': results, 'model': model, 'provider': target_provider.config.name}
            
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics"""
        # Convert sets to lists for JSON serialization
        serializable_stats = {}
        for key, stats in self.usage_stats.items():
            serializable_stats[key] = {k: v for k, v in stats.items() if k != 'providers_used' or isinstance(v, set)}
            if 'providers_used' in stats:
                serializable_stats[key]['providers_used'] = list(stats['providers_used'])
                
        return {
            'models': serializable_stats,
            'providers': self.health_monitor.get_all_providers_health(),
            'failover_stats': self.failover_manager.get_failover_stats(),
            'total_providers': len(self.failover_manager.providers),
            'healthy_providers': len([p for p in self.failover_manager.providers.values() 
                                   if self.health_monitor.get_provider_health(p.config.name)?.get('is_healthy', False)])
        }
        
    def get_provider_health_summary(self) -> Dict[str, Any]:
        """Get provider health summary"""
        return self.health_monitor.get_all_providers_health()
        
    def get_recommended_model(self, task_type: str, max_cost: Optional[float] = None) -> str:
        """Get recommended model for task within cost constraints"""
        model = self.get_model_for_task(task_type)
        
        # Check cost constraint
        if max_cost:
            tier = self._get_tier_for_task(task_type)
            preference = self.model_preferences.get(tier)
            if preference and preference.max_cost_per_request:
                if preference.max_cost_per_request > max_cost:
                    # Suggest cheaper alternative
                    return self.get_model_for_task(task_type, tier=ModelTier.LOCAL)
                    
        return model
        
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health_data = {
            'overall_status': 'healthy',
            'providers': {},
            'summary': {}
        }
        
        # Check each provider
        healthy_count = 0
        for provider_name in self.failover_manager.providers.keys():
            provider_health = self.health_monitor.get_provider_health(provider_name)
            health_data['providers'][provider_name] = provider_health
            
            if provider_health and provider_health.get('is_healthy'):
                healthy_count += 1
                
        total_providers = len(self.failover_manager.providers)
        health_data['summary'] = {
            'total_providers': total_providers,
            'healthy_providers': healthy_count,
            'health_percentage': (healthy_count / max(1, total_providers)) * 100
        }
        
        if healthy_count == 0:
            health_data['overall_status'] = 'critical'
        elif healthy_count < total_providers:
            health_data['overall_status'] = 'degraded'
            
        return health_data
        
    def clear_conversation(self, session_id: str):
        """Clear conversation history for a session"""
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]
            logger.info(f"Cleared conversation history for session {session_id}")
            
    async def cleanup(self):
        """Cleanup resources"""
        await self.health_monitor.stop_monitoring()
        logger.info("Enhanced AI Models Manager cleaned up")
        
    def __del__(self):
        """Cleanup when object is garbage collected"""
        try:
            # Try to cleanup in background (not awaited)
            asyncio.create_task(self.cleanup())
        except:
            pass