"""
OpenAI-Compatible Provider - Generic OpenAI-compatible API wrapper

This provider can work with any API that follows the OpenAI format, including:
- OpenAI API
- OpenRouter
- Azure OpenAI
- Custom OpenAI-compatible endpoints
- Local endpoints using compatible frameworks

Features:
- OpenAI-compatible interface
- Support for streaming and function calling
- Automatic endpoint validation
- Custom authentication methods
"""

from typing import Dict, List, Any, Optional, Union, AsyncGenerator
import aiohttp
import asyncio
import json
import time
from datetime import datetime
from urllib.parse import urljoin

from .base_provider import (
    BaseLLMProvider,
    ProviderCapability,
    ProviderHealth,
    ProviderConfig,
    ModelInfo,
    HealthMetrics
)
from loguru import logger


class OpenAICompatibleProvider(BaseLLMProvider):
    """
    Provider for OpenAI-compatible APIs
    
    This provider can work with any service that implements the OpenAI Chat Completions API,
    including custom endpoints, local servers, and commercial alternatives.
    """
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        
        # Validate base URL
        if not config.base_url:
            raise ValueError("base_url is required for OpenAI-compatible provider")
            
        self.base_url = config.base_url.rstrip('/')
        self.session: Optional[aiohttp.ClientSession] = None
        self._initialized = False
        
        # Default OpenAI-compatible capabilities
        self.capabilities = (
            ProviderCapability.FUNCTION_CALLING |
            ProviderCapability.STREAMING |
            ProviderCapability.SYSTEM_PROMPTS |
            ProviderCapability.JSON_MODE |
            ProviderCapability.BATCH_REQUESTS
        )
        
        # Determine API type based on URL patterns
        self.api_type = self._determine_api_type()
        
    def _determine_api_type(self) -> str:
        """Determine API type based on URL"""
        url_lower = self.base_url.lower()
        
        if 'openai.com' in url_lower:
            return 'openai'
        elif 'openrouter.ai' in url_lower:
            return 'openrouter'
        elif 'azure' in url_lower:
            return 'azure'
        elif 'localhost' in url_lower or '127.0.0.1' in url_lower:
            return 'local'
        else:
            return 'compatible'
        
    async def initialize(self) -> bool:
        """
        Initialize OpenAI-compatible provider and validate connection
        
        Returns:
            True if initialization successful
        """
        try:
            # Create HTTP session
            connector = aiohttp.TCPConnector(
                limit=self.config.max_concurrent_requests,
                limit_per_host=5
            )
            
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=self._build_headers()
            )
            
            # Test connection and discover models
            await self._test_connection()
            await self._discover_models()
            
            self._initialized = True
            logger.info(f"OpenAI-compatible provider initialized at {self.base_url} (type: {self.api_type})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI-compatible provider: {e}")
            self._initialized = False
            return False
            
    def _build_headers(self) -> Dict[str, str]:
        """Build request headers based on configuration"""
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        # Add API key if provided
        if self.config.api_key:
            if self.api_type == 'azure':
                # Azure uses different authentication
                headers['api-key'] = self.config.api_key
            else:
                headers['Authorization'] = f"Bearer {self.config.api_key}"
                
        # Add user agent
        headers['User-Agent'] = 'Trading-Orchestrator/1.0'
        
        return headers
        
    async def _test_connection(self) -> bool:
        """Test connection to the API"""
        try:
            # Try different endpoints based on API type
            endpoints_to_try = [
                f"{self.base_url}/v1/models",
                f"{self.base_url}/models",
                f"{self.base_url}/api/models",
            ]
            
            for endpoint in endpoints_to_try:
                try:
                    async with self.session.get(endpoint) as response:
                        if response.status == 200:
                            data = await response.json()
                            model_count = len(data.get('data', []))
                            logger.info(f"OpenAI-compatible API connection successful. {model_count} models available.")
                            return True
                        elif response.status == 401:
                            # Authentication error - API is reachable but key is invalid
                            logger.warning(f"API authentication failed at {endpoint}")
                            continue
                        elif response.status == 404:
                            # Endpoint doesn't exist, try next one
                            continue
                        else:
                            logger.warning(f"API returned status {response.status} at {endpoint}")
                            continue
                            
                except Exception as e:
                    logger.debug(f"Failed to connect to {endpoint}: {e}")
                    continue
                    
            # If we get here, none of the endpoints worked
            raise Exception(f"Could not find valid models endpoint at {self.base_url}")
            
        except Exception as e:
            raise Exception(f"Cannot connect to OpenAI-compatible API at {self.base_url}: {e}")
            
    async def _discover_models(self):
        """Discover and register available models"""
        try:
            # Try different endpoints
            endpoints = [
                f"{self.base_url}/v1/models",
                f"{self.base_url}/models",
                f"{self.base_url}/api/models",
            ]
            
            for endpoint in endpoints:
                try:
                    async with self.session.get(endpoint) as response:
                        if response.status == 200:
                            data = await response.json()
                            await self._process_models_response(data.get('data', []))
                            return
                except Exception as e:
                    logger.debug(f"Failed to fetch models from {endpoint}: {e}")
                    continue
                    
            # If discovery fails, register some common model placeholders
            logger.warning("Could not discover models, registering defaults")
            await self._register_default_models()
            
        except Exception as e:
            logger.error(f"Failed to discover models: {e}")
            
    async def _process_models_response(self, models_data: List[Dict]):
        """Process models from API response"""
        for model_data in models_data:
            model_id = model_data.get('id', '')
            context_length = model_data.get('context_length', 4096)
            
            model_info = ModelInfo(
                name=model_id,
                display_name=model_id,
                context_length=context_length,
                capabilities=self.capabilities,
                cost_per_token=0.0,  # Cost detection not supported by all APIs
                description=model_data.get('description', ''),
                version=model_data.get('version', '')
            )
            
            self.models[model_id] = model_info
            
        logger.info(f"Discovered {len(self.models)} models")
        
    async def _register_default_models(self):
        """Register default model configurations"""
        common_models = [
            'gpt-3.5-turbo',
            'gpt-4',
            'gpt-4-turbo',
            'claude-3-sonnet',
            'claude-3-haiku',
            'llama-2-70b-chat',
            'llama-2-7b-chat',
            'mistral-7b-instruct',
            'mixtral-8x7b-instruct',
        ]
        
        for model_name in common_models:
            model_info = ModelInfo(
                name=model_name,
                display_name=model_name,
                context_length=4096,
                capabilities=self.capabilities,
                cost_per_token=0.0,
                description=f"OpenAI-compatible model: {model_name}"
            )
            
            self.models[model_name] = model_info
            
        logger.info(f"Registered {len(self.models)} default models")
        
    async def health_check(self) -> HealthMetrics:
        """
        Perform health check on the OpenAI-compatible provider
        
        Returns:
            HealthMetrics with current health status
        """
        start_time = time.time()
        
        try:
            # Test with models endpoint
            async with self.session.get(f"{self.base_url}/v1/models") as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    is_healthy = True
                    message = f"{self.api_type.title()} API is running"
                elif response.status == 401:
                    is_healthy = False
                    message = "Authentication failed"
                else:
                    is_healthy = False
                    message = f"API returned status {response.status}"
                    
        except Exception as e:
            response_time = time.time() - start_time
            is_healthy = False
            message = f"Connection failed: {str(e)}"
            
        return HealthMetrics(
            response_time=response_time,
            success_rate=1.0 if is_healthy else 0.0,
            error_rate=0.0 if is_healthy else 1.0,
            last_check=datetime.now(),
            consecutive_failures=0 if is_healthy else 1,
            is_healthy=is_healthy,
            message=message
        )
        
    async def list_models(self) -> List[ModelInfo]:
        """List all available models"""
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
        """
        Generate completion using OpenAI-compatible API
        
        Args:
            messages: Conversation messages
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Function calling tools
            stream: Whether to stream response
            **kwargs: Additional parameters
            
        Returns:
            Completion response or async generator for streaming
        """
        if not self._initialized:
            raise RuntimeError("OpenAI-compatible provider not initialized")
            
        start_time = time.time()
        
        try:
            # Prepare request payload
            payload = {
                "model": model,
                "messages": messages,
                "stream": stream,
                "temperature": temperature
            }
            
            if max_tokens:
                payload["max_tokens"] = max_tokens
                
            # Add tools if provided
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"
                
            # Add additional parameters
            for key, value in kwargs.items():
                if key not in ['model', 'messages', 'stream', 'temperature', 'max_tokens', 'tools', 'tool_choice']:
                    payload[key] = value
                    
            if stream:
                return self._stream_completion(payload, start_time)
            else:
                return await self._single_completion(payload, start_time)
                
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"OpenAI-compatible completion failed: {e}")
            
            self.increment_request_count(0, 0.0, response_time)
            
            return {
                'content': f"Error generating completion: {str(e)}",
                'model': model,
                'provider': self.api_type,
                'error': str(e),
                'response_time': response_time
            }
            
    async def _single_completion(
        self,
        payload: Dict[str, Any],
        start_time: float
    ) -> Dict[str, Any]:
        """Handle single completion (non-streaming)"""
        endpoint = f"{self.base_url}/v1/chat/completions"
        
        async with self.session.post(endpoint, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"API returned status {response.status}: {error_text}")
                
            data = await response.json()
            response_time = time.time() - start_time
            
            # Extract content
            choice = data.get('choices', [{}])[0]
            message = choice.get('message', {})
            content = message.get('content', '')
            
            # Extract usage information
            usage = data.get('usage', {})
            
            # Update stats
            total_tokens = usage.get('total_tokens', 0)
            cost = self._estimate_cost(payload['model'], total_tokens)
            self.increment_request_count(total_tokens, cost, response_time)
            
            return {
                'content': content,
                'model': payload['model'],
                'provider': self.api_type,
                'response_time': response_time,
                'usage': {
                    'prompt_tokens': usage.get('prompt_tokens', 0),
                    'completion_tokens': usage.get('completion_tokens', 0),
                    'total_tokens': total_tokens
                },
                'cost_estimate': cost
            }
            
    async def _stream_completion(
        self,
        payload: Dict[str, Any],
        start_time: float
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle streaming completion"""
        endpoint = f"{self.base_url}/v1/chat/completions"
        headers = {
            'Accept': 'text/event-stream'
        }
        
        async with self.session.post(endpoint, json=payload, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Streaming API returned status {response.status}: {error_text}")
                
            async for line in response.content:
                if line:
                    line = line.decode('utf-8').strip()
                    
                    if line.startswith('data: '):
                        data_str = line[6:]  # Remove 'data: ' prefix
                        
                        if data_str == '[DONE]':
                            break
                            
                        try:
                            data = json.loads(data_str)
                            
                            # Extract content from streaming response
                            choice = data.get('choices', [{}])[0]
                            delta = choice.get('delta', {})
                            content = delta.get('content', '')
                            
                            if content:
                                chunk = {
                                    'content': content,
                                    'model': payload['model'],
                                    'provider': self.api_type,
                                    'done': choice.get('finish_reason') is not None
                                }
                                
                                yield chunk
                                
                        except json.JSONDecodeError:
                            continue
                            
    async def embeddings(
        self,
        texts: List[str],
        model: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate embeddings using OpenAI-compatible API
        
        Args:
            texts: List of texts to embed
            model: Model name to use for embeddings
            **kwargs: Additional parameters
            
        Returns:
            Embeddings response
        """
        try:
            payload = {
                "model": model,
                "input": texts
            }
            
            # Add additional parameters
            for key, value in kwargs.items():
                if key not in ['model', 'input']:
                    payload[key] = value
                    
            endpoint = f"{self.base_url}/v1/embeddings"
            
            async with self.session.post(endpoint, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Embeddings API returned status {response.status}: {error_text}")
                    
        except Exception as e:
            logger.error(f"OpenAI-compatible embeddings failed: {e}")
            raise
            
    def _estimate_cost(self, model: str, tokens_used: int) -> float:
        """Estimate cost based on model name"""
        # Very rough estimates for common models
        model_lower = model.lower()
        
        cost_rates = {
            'gpt-4': 0.03,  # per 1K tokens
            'gpt-4-turbo': 0.01,
            'gpt-3.5-turbo': 0.0015,
            'claude-3-5-sonnet': 0.015,
            'claude-3-haiku': 0.00025,
            'default': 0.001  # Default estimate
        }
        
        # Find matching rate
        rate = cost_rates.get('default')
        for model_pattern, pattern_rate in cost_rates.items():
            if model_pattern in model_lower:
                rate = pattern_rate
                break
                
        return (tokens_used / 1000) * rate
        
    def supports_capability(self, capability: ProviderCapability) -> bool:
        """Check if provider supports capability"""
        return bool(self.capabilities & capability)
        
    async def validate_api_key(self) -> bool:
        """Validate API key by testing authentication"""
        try:
            async with self.session.get(f"{self.base_url}/v1/models") as response:
                return response.status == 200
        except Exception:
            return False
            
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
            self.session = None
            
        logger.info("OpenAI-compatible provider cleaned up")
        
    def __del__(self):
        """Cleanup when object is garbage collected"""
        if self.session and not self.session.closed:
            try:
                # Try to cleanup in background (not awaited)
                asyncio.create_task(self.cleanup())
            except:
                pass