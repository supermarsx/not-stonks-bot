"""
OpenAI Provider - OpenAI API integration

Extends the base provider to provide OpenAI-specific functionality with
proper authentication and feature detection.
"""

from typing import Dict, List, Any, Optional, Union, AsyncGenerator
import asyncio
import time
from datetime import datetime

from .base_provider import (
    BaseLLMProvider,
    ProviderCapability,
    ProviderHealth,
    ProviderConfig,
    ModelInfo,
    HealthMetrics
)
from loguru import logger


class OpenAIProvider(BaseLLMProvider):
    """
    Provider for OpenAI API
    
    Provides access to OpenAI's language models including GPT-4, GPT-3.5, etc.
    with full feature support including function calling, streaming, and embeddings.
    """
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.client = None
        self._initialized = False
        
        # OpenAI capabilities
        self.capabilities = (
            ProviderCapability.FUNCTION_CALLING |
            ProviderCapability.STREAMING |
            ProviderCapability.SYSTEM_PROMPTS |
            ProviderCapability.JSON_MODE |
            ProviderCapability.VISION |
            ProviderCapability.LONG_CONTEXT
        )
        
        # OpenAI model capabilities mapping
        self.model_capabilities = {
            'gpt-4': ProviderCapability.VISION | ProviderCapability.LONG_CONTEXT,
            'gpt-4-turbo': ProviderCapability.VISION | ProviderCapability.LONG_CONTEXT,
            'gpt-4-turbo-preview': ProviderCapability.VISION | ProviderCapability.LONG_CONTEXT,
            'gpt-4-vision-preview': ProviderCapability.VISION,
            'gpt-3.5-turbo': ProviderCapability.FUNCTION_CALLING | ProviderCapability.STREAMING,
            'gpt-3.5-turbo-16k': ProviderCapability.LONG_CONTEXT,
        }
        
    async def initialize(self) -> bool:
        """
        Initialize OpenAI provider and validate connection
        
        Returns:
            True if initialization successful
        """
        try:
            from openai import AsyncOpenAI
            
            # Initialize OpenAI client
            self.client = AsyncOpenAI(
                api_key=self.config.api_key,
                timeout=self.config.timeout
            )
            
            # Test connection and discover models
            await self._test_connection()
            await self._discover_models()
            
            self._initialized = True
            logger.info("OpenAI provider initialized")
            return True
            
        except ImportError:
            logger.error("OpenAI package not installed. Install with: pip install openai")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI provider: {e}")
            self._initialized = False
            return False
            
    async def _test_connection(self):
        """Test connection to OpenAI API"""
        try:
            models = await self.client.models.list()
            model_count = len(models.data)
            logger.info(f"OpenAI connection successful. {model_count} models available.")
        except Exception as e:
            raise Exception(f"OpenAI API connection failed: {e}")
            
    async def _discover_models(self):
        """Discover and register available OpenAI models"""
        try:
            models = await self.client.models.list()
            
            for model in models.data:
                model_id = model.id
                
                # Skip deprecated or inactive models
                if hasattr(model, 'deprecated') and model.deprecated:
                    continue
                    
                # Determine model capabilities
                capabilities = self.model_capabilities.get(
                    model_id,
                    self._determine_model_capabilities(model_id)
                )
                
                model_info = ModelInfo(
                    name=model_id,
                    display_name=model_id,
                    context_length=model.context_length if hasattr(model, 'context_length') else 4096,
                    capabilities=capabilities,
                    cost_per_token=self._get_model_cost(model_id),
                    description=getattr(model, 'description', '') if hasattr(model, 'description') else '',
                    version=getattr(model, 'version', '') if hasattr(model, 'version') else ''
                )
                
                self.models[model_id] = model_info
                
            logger.info(f"Discovered {len(self.models)} OpenAI models")
            
        except Exception as e:
            logger.error(f"Failed to discover OpenAI models: {e}")
            
    def _determine_model_capabilities(self, model_name: str) -> ProviderCapability:
        """Determine capabilities based on model name"""
        name_lower = model_name.lower()
        
        capabilities = self.capabilities
        
        # Remove capabilities not supported by specific models
        if 'gpt-3.5' in name_lower:
            capabilities &= ~ProviderCapability.LONG_CONTEXT
            if '16k' not in name_lower:
                capabilities &= ~ProviderCapability.VISION
                
        return capabilities
        
    def _get_model_cost(self, model_name: str) -> float:
        """Get cost per token for OpenAI models"""
        # OpenAI pricing (approximate, as of 2024)
        pricing = {
            'gpt-4': 0.03,
            'gpt-4-turbo': 0.01,
            'gpt-4-turbo-preview': 0.01,
            'gpt-4-vision-preview': 0.01,
            'gpt-3.5-turbo': 0.0015,
            'gpt-3.5-turbo-16k': 0.004,
        }
        
        return pricing.get(model_name, 0.001)  # Default cost
        
    async def health_check(self) -> HealthMetrics:
        """Perform health check on OpenAI provider"""
        start_time = time.time()
        
        try:
            await self.client.models.list()
            response_time = time.time() - start_time
            
            return HealthMetrics(
                response_time=response_time,
                success_rate=1.0,
                error_rate=0.0,
                last_check=datetime.now(),
                consecutive_failures=0,
                is_healthy=True,
                message="OpenAI API is healthy"
            )
        except Exception as e:
            response_time = time.time() - start_time
            
            return HealthMetrics(
                response_time=response_time,
                success_rate=0.0,
                error_rate=1.0,
                last_check=datetime.now(),
                consecutive_failures=1,
                is_healthy=False,
                message=f"OpenAI API error: {str(e)}"
            )
            
    async def list_models(self) -> List[ModelInfo]:
        """List all available OpenAI models"""
        return list(self.models.values())
        
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
        """Generate completion using OpenAI API"""
        if not self._initialized:
            raise RuntimeError("OpenAI provider not initialized")
            
        start_time = time.time()
        
        try:
            if stream:
                return self._stream_completion(messages, model, temperature, max_tokens, tools, start_time)
            else:
                return await self._single_completion(messages, model, temperature, max_tokens, tools, start_time)
                
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"OpenAI completion failed: {e}")
            
            self.increment_request_count(0, 0.0, response_time)
            
            return {
                'content': f"Error generating completion: {str(e)}",
                'model': model,
                'provider': 'openai',
                'error': str(e),
                'response_time': response_time
            }
            
    async def _single_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        tools: Optional[List[Dict]],
        start_time: float
    ) -> Dict[str, Any]:
        """Handle single completion"""
        kwargs = {
            'model': model,
            'messages': messages,
            'temperature': temperature
        }
        
        if max_tokens:
            kwargs['max_tokens'] = max_tokens
            
        if tools:
            kwargs['tools'] = tools
            kwargs['tool_choice'] = 'auto'
            
        response = await self.client.chat.completions.create(**kwargs)
        response_time = time.time() - start_time
        
        # Extract content
        message = response.choices[0].message
        content = message.content or ''
        
        # Update stats
        usage = response.usage
        total_tokens = usage.total_tokens if usage else 0
        cost = (total_tokens / 1000) * self._get_model_cost(model)
        self.increment_request_count(total_tokens, cost, response_time)
        
        return {
            'content': content,
            'model': model,
            'provider': 'openai',
            'response_time': response_time,
            'usage': {
                'prompt_tokens': usage.prompt_tokens if usage else 0,
                'completion_tokens': usage.completion_tokens if usage else 0,
                'total_tokens': total_tokens
            },
            'cost_estimate': cost
        }
        
    async def _stream_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        tools: Optional[List[Dict]],
        start_time: float
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle streaming completion"""
        kwargs = {
            'model': model,
            'messages': messages,
            'temperature': temperature,
            'stream': True
        }
        
        if max_tokens:
            kwargs['max_tokens'] = max_tokens
            
        if tools:
            kwargs['tools'] = tools
            kwargs['tool_choice'] = 'auto'
            
        stream = await self.client.chat.completions.create(**kwargs)
        
        total_tokens = 0
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content_chunk = chunk.choices[0].delta.content
                total_tokens += 1  # Approximate token count
                
                chunk_data = {
                    'content': content_chunk,
                    'model': model,
                    'provider': 'openai',
                    'done': chunk.choices[0].finish_reason is not None
                }
                
                yield chunk_data
                
        # Update stats after streaming completes
        response_time = time.time() - start_time
        cost = (total_tokens / 1000) * self._get_model_cost(model)
        self.increment_request_count(total_tokens, cost, response_time)
        
    async def embeddings(
        self,
        texts: List[str],
        model: str = "text-embedding-ada-002",
        **kwargs
    ) -> Dict[str, Any]:
        """Generate embeddings using OpenAI"""
        try:
            response = await self.client.embeddings.create(
                input=texts,
                model=model,
                **kwargs
            )
            
            return {
                'data': [item.embedding for item in response.data],
                'model': response.model,
                'usage': {
                    'total_tokens': response.usage.total_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"OpenAI embeddings failed: {e}")
            raise
            
    def supports_capability(self, capability: ProviderCapability) -> bool:
        """Check if OpenAI provider supports capability"""
        return bool(self.capabilities & capability)
        
    async def cleanup(self):
        """Cleanup resources"""
        if self.client:
            # AsyncOpenAI doesn't have an explicit close method in older versions
            pass
            
        logger.info("OpenAI provider cleaned up")