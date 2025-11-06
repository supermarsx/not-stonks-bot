"""
Anthropic Provider - Anthropic API integration

Extends the base provider to provide Anthropic-specific functionality with
proper authentication and Claude model support.
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


class AnthropicProvider(BaseLLMProvider):
    """
    Provider for Anthropic API
    
    Provides access to Claude models including Claude 3.5 Sonnet, Claude 3 Haiku, etc.
    with full feature support including function calling and streaming.
    """
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.client = None
        self._initialized = False
        
        # Anthropic capabilities
        self.capabilities = (
            ProviderCapability.FUNCTION_CALLING |
            ProviderCapability.STREAMING |
            ProviderCapability.SYSTEM_PROMPTS |
            ProviderCapability.VISION |
            ProviderCapability.LONG_CONTEXT
        )
        
        # Claude model capabilities mapping
        self.model_capabilities = {
            'claude-3-5-sonnet': ProviderCapability.VISION | ProviderCapability.LONG_CONTEXT,
            'claude-3-sonnet': ProviderCapability.VISION | ProviderCapability.LONG_CONTEXT,
            'claude-3-haiku': ProviderCapability.FUNCTION_CALLING | ProviderCapability.STREAMING,
            'claude-3-opus': ProviderCapability.VISION | ProviderCapability.LONG_CONTEXT,
            'claude-3': ProviderCapability.VISION | ProviderCapability.LONG_CONTEXT,
            'claude-2.1': ProviderCapability.LONG_CONTEXT,
            'claude-2': ProviderCapability.LONG_CONTEXT,
            'claude-instant': ProviderCapability.FUNCTION_CALLING,
        }
        
    async def initialize(self) -> bool:
        """
        Initialize Anthropic provider and validate connection
        
        Returns:
            True if initialization successful
        """
        try:
            from anthropic import AsyncAnthropic
            
            # Initialize Anthropic client
            self.client = AsyncAnthropic(
                api_key=self.config.api_key,
                timeout=self.config.timeout
            )
            
            # Test connection by making a simple API call
            await self._test_connection()
            
            # Discover models (Anthropic has a fixed set of models)
            await self._discover_models()
            
            self._initialized = True
            logger.info("Anthropic provider initialized")
            return True
            
        except ImportError:
            logger.error("Anthropic package not installed. Install with: pip install anthropic")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic provider: {e}")
            self._initialized = False
            return False
            
    async def _test_connection(self):
        """Test connection to Anthropic API"""
        try:
            # Anthropic doesn't have a models endpoint, so we test with a minimal message
            message = await self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}]
            )
            logger.info("Anthropic connection successful")
        except Exception as e:
            raise Exception(f"Anthropic API connection failed: {e}")
            
    async def _discover_models(self):
        """Register known Anthropic Claude models"""
        # Anthropic has a fixed set of models, so we register them directly
        claude_models = [
            {
                'name': 'claude-3-5-sonnet-20241022',
                'context_length': 200000,
                'capabilities': ProviderCapability.VISION | ProviderCapability.LONG_CONTEXT,
                'cost_per_token': 0.015,
                'description': 'Most intelligent Claude 3.5 model'
            },
            {
                'name': 'claude-3-sonnet-20240229',
                'context_length': 200000,
                'capabilities': ProviderCapability.VISION | ProviderCapability.LONG_CONTEXT,
                'cost_per_token': 0.003,
                'description': 'Balanced Claude 3 model with vision'
            },
            {
                'name': 'claude-3-haiku-20240307',
                'context_length': 200000,
                'capabilities': ProviderCapability.FUNCTION_CALLING | ProviderCapability.STREAMING,
                'cost_per_token': 0.00025,
                'description': 'Fastest Claude 3 model'
            },
            {
                'name': 'claude-3-opus-20240229',
                'context_length': 200000,
                'capabilities': ProviderCapability.VISION | ProviderCapability.LONG_CONTEXT,
                'cost_per_token': 0.075,
                'description': 'Most powerful Claude 3 model'
            },
            {
                'name': 'claude-2.1',
                'context_length': 200000,
                'capabilities': ProviderCapability.LONG_CONTEXT,
                'cost_per_token': 0.008,
                'description': 'Claude 2.1 with improved reasoning'
            },
            {
                'name': 'claude-2',
                'context_length': 100000,
                'capabilities': ProviderCapability.LONG_CONTEXT,
                'cost_per_token': 0.01102,
                'description': 'Claude 2 model'
            },
            {
                'name': 'claude-instant-1.2',
                'context_length': 100000,
                'capabilities': ProviderCapability.FUNCTION_CALLING,
                'cost_per_token': 0.00163,
                'description': 'Fast Claude Instant model'
            }
        ]
        
        for model_data in claude_models:
            model_info = ModelInfo(
                name=model_data['name'],
                display_name=model_data['name'],
                context_length=model_data['context_length'],
                capabilities=model_data['capabilities'],
                cost_per_token=model_data['cost_per_token'],
                description=model_data['description']
            )
            
            self.models[model_data['name']] = model_info
            
        logger.info(f"Registered {len(self.models)} Claude models")
        
    async def health_check(self) -> HealthMetrics:
        """Perform health check on Anthropic provider"""
        start_time = time.time()
        
        try:
            # Test with a minimal API call
            message = await self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}]
            )
            
            response_time = time.time() - start_time
            
            return HealthMetrics(
                response_time=response_time,
                success_rate=1.0,
                error_rate=0.0,
                last_check=datetime.now(),
                consecutive_failures=0,
                is_healthy=True,
                message="Anthropic API is healthy"
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
                message=f"Anthropic API error: {str(e)}"
            )
            
    async def list_models(self) -> List[ModelInfo]:
        """List all available Claude models"""
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
        """Generate completion using Anthropic API"""
        if not self._initialized:
            raise RuntimeError("Anthropic provider not initialized")
            
        start_time = time.time()
        
        try:
            if stream:
                return self._stream_completion(messages, model, temperature, max_tokens, tools, start_time)
            else:
                return await self._single_completion(messages, model, temperature, max_tokens, tools, start_time)
                
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Anthropic completion failed: {e}")
            
            self.increment_request_count(0, 0.0, response_time)
            
            return {
                'content': f"Error generating completion: {str(e)}",
                'model': model,
                'provider': 'anthropic',
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
        # Convert messages to Anthropic format
        system_messages = [m['content'] for m in messages if m['role'] == 'system']
        system = system_messages[0] if system_messages else None
        
        anthropic_messages = [
            {'role': m['role'], 'content': m['content']}
            for m in messages
            if m['role'] in ['user', 'assistant']
        ]
        
        kwargs = {
            'model': model,
            'messages': anthropic_messages,
            'temperature': temperature
        }
        
        if system:
            kwargs['system'] = system
            
        if max_tokens:
            kwargs['max_tokens'] = max_tokens
            
        if tools:
            kwargs['tools'] = tools
            
        response = await self.client.messages.create(**kwargs)
        response_time = time.time() - start_time
        
        # Extract content
        text_content = ''.join([
            block.text for block in response.content
            if hasattr(block, 'text')
        ])
        
        # Update stats
        usage = response.usage
        total_tokens = usage.input_tokens + usage.output_tokens if usage else 0
        cost = (total_tokens / 1000) * self._get_model_cost(model)
        self.increment_request_count(total_tokens, cost, response_time)
        
        return {
            'content': text_content,
            'model': model,
            'provider': 'anthropic',
            'response_time': response_time,
            'usage': {
                'prompt_tokens': usage.input_tokens if usage else 0,
                'completion_tokens': usage.output_tokens if usage else 0,
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
        # Convert messages to Anthropic format
        system_messages = [m['content'] for m in messages if m['role'] == 'system']
        system = system_messages[0] if system_messages else None
        
        anthropic_messages = [
            {'role': m['role'], 'content': m['content']}
            for m in messages
            if m['role'] in ['user', 'assistant']
        ]
        
        kwargs = {
            'model': model,
            'messages': anthropic_messages,
            'temperature': temperature,
            'stream': True
        }
        
        if system:
            kwargs['system'] = system
            
        if max_tokens:
            kwargs['max_tokens'] = max_tokens
            
        if tools:
            kwargs['tools'] = tools
            
        stream = await self.client.messages.create(**kwargs)
        
        total_tokens = 0
        async for chunk in stream:
            if chunk.type == 'content_block_delta':
                if chunk.delta.type == 'text_delta':
                    content_chunk = chunk.delta.text
                    total_tokens += 1  # Approximate
                    
                    chunk_data = {
                        'content': content_chunk,
                        'model': model,
                        'provider': 'anthropic',
                        'done': False
                    }
                    
                    yield chunk_data
            elif chunk.type == 'message_delta':
                if chunk.delta.stop_reason:
                    chunk_data = {
                        'content': '',
                        'model': model,
                        'provider': 'anthropic',
                        'done': True
                    }
                    
                    yield chunk_data
                    break
                    
        # Update stats after streaming completes
        response_time = time.time() - start_time
        cost = (total_tokens / 1000) * self._get_model_cost(model)
        self.increment_request_count(total_tokens, cost, response_time)
        
    def _get_model_cost(self, model_name: str) -> float:
        """Get cost per token for Anthropic models"""
        model_info = self.models.get(model_name)
        return model_info.cost_per_token if model_info else 0.001
        
    def supports_capability(self, capability: ProviderCapability) -> bool:
        """Check if Anthropic provider supports capability"""
        return bool(self.capabilities & capability)
        
    async def validate_api_key(self) -> bool:
        """Validate Anthropic API key"""
        try:
            message = await self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}]
            )
            return True
        except Exception:
            return False
            
    async def cleanup(self):
        """Cleanup resources"""
        if self.client:
            # AsyncAnthropic doesn't have an explicit close method
            pass
            
        logger.info("Anthropic provider cleaned up")