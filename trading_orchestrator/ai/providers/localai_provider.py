"""
LocalAI Provider - OpenAI-compatible local LLM serving

LocalAI is an OpenAI-compatible API that can serve open-source models locally.
This provider provides a drop-in replacement for OpenAI with local models.

Features:
- OpenAI-compatible interface
- Support for various open-source models
- Function calling support
- Streaming responses
- Local serving with zero cost
"""

from typing import Dict, List, Any, Optional, Union, AsyncGenerator
import aiohttp
import asyncio
import json
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


class LocalAIProvider(BaseLLMProvider):
    """
    Provider for LocalAI - OpenAI-compatible local LLM serving
    
    LocalAI provides an OpenAI-compatible API interface for local models,
    supporting models like Llama, Mistral, Alpaca, Vicuna, and more.
    """
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.base_url = config.base_url or "http://localhost:8080"
        self.session: Optional[aiohttp.ClientSession] = None
        self._initialized = False
        
        # Default capabilities for LocalAI (varies by backend model)
        self.capabilities = (
            ProviderCapability.FUNCTION_CALLING | 
            ProviderCapability.STREAMING | 
            ProviderCapability.SYSTEM_PROMPTS |
            ProviderCapability.JSON_MODE
        )
        
    async def initialize(self) -> bool:
        """
        Initialize LocalAI provider and validate connection
        
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
                timeout=timeout
            )
            
            # Test connection to LocalAI
            await self._test_connection()
            
            # Discover available models
            await self._discover_models()
            
            self._initialized = True
            logger.info(f"LocalAI provider initialized at {self.base_url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize LocalAI provider: {e}")
            self._initialized = False
            return False
            
    async def _test_connection(self) -> bool:
        """Test connection to LocalAI server"""
        try:
            async with self.session.get(f"{self.base_url}/v1/models") as response:
                if response.status == 200:
                    data = await response.json()
                    model_count = len(data.get('data', []))
                    logger.info(f"LocalAI connection successful. {model_count} models available.")
                    return True
                else:
                    raise Exception(f"LocalAI API returned status {response.status}")
        except Exception as e:
            raise Exception(f"Cannot connect to LocalAI at {self.base_url}: {e}")
            
    async def _discover_models(self):
        """Discover and register available LocalAI models"""
        try:
            async with self.session.get(f"{self.base_url}/v1/models") as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for model_data in data.get('data', []):
                        model_id = model_data.get('id', '')
                        
                        # Extract model information
                        model_info = ModelInfo(
                            name=model_id,
                            display_name=model_id,
                            context_length=model_data.get('context_length', 4096),
                            capabilities=self.capabilities,
                            cost_per_token=0.0,  # Local inference is free
                            description=model_data.get('description', ''),
                            version=model_data.get('version', '')
                        )
                        
                        self.models[model_id] = model_info
                        
                    logger.info(f"Discovered {len(self.models)} LocalAI models")
                    
        except Exception as e:
            logger.error(f"Failed to discover LocalAI models: {e}")
            
    async def health_check(self) -> HealthMetrics:
        """
        Perform health check on LocalAI provider
        
        Returns:
            HealthMetrics with current health status
        """
        start_time = time.time()
        
        try:
            # Test API endpoint
            async with self.session.get(f"{self.base_url}/v1/models") as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    is_healthy = True
                    message = "LocalAI is running"
                else:
                    is_healthy = False
                    message = f"LocalAI API returned status {response.status}"
                    
        except Exception as e:
            response_time = time.time() - start_time
            is_healthy = False
            message = f"LocalAI connection failed: {str(e)}"
            
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
        """List all available LocalAI models"""
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
        Generate completion using LocalAI (OpenAI-compatible)
        
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
            raise RuntimeError("LocalAI provider not initialized")
            
        # Validate model exists
        if model not in self.models:
            # Refresh model list if model not found
            await self._discover_models()
            if model not in self.models:
                raise ValueError(f"Model {model} not found in LocalAI")
                
        start_time = time.time()
        
        try:
            # Prepare request payload (OpenAI-compatible format)
            payload = {
                "model": model,
                "messages": messages,
                "stream": stream,
                "temperature": temperature
            }
            
            if max_tokens:
                payload["max_tokens"] = max_tokens
                
            # Add tools if supported
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
            logger.error(f"LocalAI completion failed: {e}")
            
            # Update stats
            self.increment_request_count(0, 0.0, response_time)
            
            return {
                'content': f"Error generating completion: {str(e)}",
                'model': model,
                'provider': 'localai',
                'error': str(e),
                'response_time': response_time
            }
            
    async def _single_completion(
        self,
        payload: Dict[str, Any],
        start_time: float
    ) -> Dict[str, Any]:
        """Handle single completion (non-streaming)"""
        async with self.session.post(f"{self.base_url}/v1/chat/completions", json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"LocalAI API returned status {response.status}: {error_text}")
                
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
            self.increment_request_count(total_tokens, 0.0, response_time)
            
            return {
                'content': content,
                'model': payload['model'],
                'provider': 'localai',
                'response_time': response_time,
                'usage': {
                    'prompt_tokens': usage.get('prompt_tokens', 0),
                    'completion_tokens': usage.get('completion_tokens', 0),
                    'total_tokens': total_tokens
                }
            }
            
    async def _stream_completion(
        self,
        payload: Dict[str, Any],
        start_time: float
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle streaming completion"""
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'text/event-stream'
        }
        
        async with self.session.post(
            f"{self.base_url}/v1/chat/completions", 
            json=payload, 
            headers=headers
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"LocalAI streaming API returned status {response.status}: {error_text}")
                
            async for line in response.content:
                if line:
                    line = line.decode('utf-8').strip()
                    
                    if line.startswith('data: '):
                        data_str = line[6:]  # Remove 'data: ' prefix
                        
                        if data_str == '[DONE]':
                            break
                            
                        try:
                            data = json.loads(data_str)
                            
                            # Extract content from the streaming response
                            choice = data.get('choices', [{}])[0]
                            delta = choice.get('delta', {})
                            content = delta.get('content', '')
                            
                            if content:
                                chunk = {
                                    'content': content,
                                    'model': payload['model'],
                                    'provider': 'localai',
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
        Generate embeddings using LocalAI
        
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
                payload[key] = value
                
            async with self.session.post(f"{self.base_url}/v1/embeddings", json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Embeddings API returned status {response.status}: {error_text}")
                    
        except Exception as e:
            logger.error(f"LocalAI embeddings failed: {e}")
            raise
            
    async def list_embeddings_models(self) -> List[str]:
        """
        List available embedding models
        
        Returns:
            List of embedding model names
        """
        # LocalAI typically uses the same models for embeddings and chat
        # This is a simplified implementation
        return [
            model for model in self.models.keys() 
            if any(embedding_name in model.lower() for embedding_name in ['embedding', 'embed', 'e5', 'all-minilm'])
        ]
        
    async def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get detailed information about a specific model"""
        return self.models.get(model_name)
        
    def supports_capability(self, capability: ProviderCapability) -> bool:
        """Check if LocalAI provider supports capability"""
        return bool(self.capabilities & capability)
        
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
            self.session = None
            
        logger.info("LocalAI provider cleaned up")
        
    def __del__(self):
        """Cleanup when object is garbage collected"""
        if self.session and not self.session.closed:
            try:
                # Try to cleanup in background (not awaited)
                asyncio.create_task(self.cleanup())
            except:
                pass