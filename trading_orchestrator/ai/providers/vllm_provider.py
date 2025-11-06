"""
vLLM Provider - High-performance local LLM serving

vLLM is a high-performance LLM serving library that provides fast inference
for open-source models. This provider provides integration with vLLM's API.

Features:
- High-performance local inference
- Streaming responses
- Batch processing support
- OpenAI-compatible API
- Ray-based distributed serving
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


class VLLMProvider(BaseLLMProvider):
    """
    Provider for vLLM - High-performance local LLM serving
    
    vLLM provides high-throughput inference for open-source models with
    support for various backends and hardware configurations.
    """
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.base_url = config.base_url or "http://localhost:8000"
        self.session: Optional[aiohttp.ClientSession] = None
        self._initialized = False
        
        # vLLM capabilities (depends on model and configuration)
        self.capabilities = (
            ProviderCapability.FUNCTION_CALLING |
            ProviderCapability.STREAMING |
            ProviderCapability.SYSTEM_PROMPTS |
            ProviderCapability.BATCH_REQUESTS |
            ProviderCapability.JSON_MODE
        )
        
        # Default vLLM models and their properties
        self.model_defaults = {
            'llama-2-7b': {'context_length': 4096, 'capabilities': self.capabilities},
            'llama-2-13b': {'context_length': 4096, 'capabilities': self.capabilities},
            'llama-2-70b': {'context_length': 4096, 'capabilities': self.capabilities},
            'llama-3-8b': {'context_length': 8192, 'capabilities': self.capabilities},
            'llama-3-70b': {'context_length': 8192, 'capabilities': self.capabilities},
            'mistral-7b': {'context_length': 32768, 'capabilities': self.capabilities},
            'mixtral-8x7b': {'context_length': 32768, 'capabilities': self.capabilities},
            'qwen-7b': {'context_length': 32768, 'capabilities': self.capabilities},
            'qwen-14b': {'context_length': 32768, 'capabilities': self.capabilities},
        }
        
    async def initialize(self) -> bool:
        """
        Initialize vLLM provider and validate connection
        
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
            
            # Test connection to vLLM
            await self._test_connection()
            
            # Discover available models
            await self._discover_models()
            
            self._initialized = True
            logger.info(f"vLLM provider initialized at {self.base_url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize vLLM provider: {e}")
            self._initialized = False
            return False
            
    async def _test_connection(self) -> bool:
        """Test connection to vLLM server"""
        try:
            # vLLM typically exposes /v1/models endpoint
            async with self.session.get(f"{self.base_url}/v1/models") as response:
                if response.status == 200:
                    data = await response.json()
                    model_count = len(data.get('data', []))
                    logger.info(f"vLLM connection successful. {model_count} models available.")
                    return True
                else:
                    # Try alternative endpoint
                    async with self.session.get(f"{self.base_url}/models") as response:
                        if response.status == 200:
                            data = await response.json()
                            model_count = len(data.get('data', []))
                            logger.info(f"vLLM connection successful (alt endpoint). {model_count} models available.")
                            return True
                        else:
                            raise Exception(f"vLLM API returned status {response.status}")
        except Exception as e:
            raise Exception(f"Cannot connect to vLLM at {self.base_url}: {e}")
            
    async def _discover_models(self):
        """Discover and register available vLLM models"""
        try:
            # Try OpenAI-compatible endpoint first
            try:
                async with self.session.get(f"{self.base_url}/v1/models") as response:
                    if response.status == 200:
                        data = await response.json()
                        await self._process_models_response(data.get('data', []))
                        return
            except Exception:
                pass
                
            # Try alternative endpoint
            try:
                async with self.session.get(f"{self.base_url}/models") as response:
                    if response.status == 200:
                        data = await response.json()
                        await self._process_models_response(data.get('data', []))
                        return
            except Exception:
                pass
                
            # If discovery fails, register default models
            logger.warning("Could not discover vLLM models, using defaults")
            await self._register_default_models()
            
        except Exception as e:
            logger.error(f"Failed to discover vLLM models: {e}")
            
    async def _process_models_response(self, models_data: List[Dict]):
        """Process models from API response"""
        for model_data in models_data:
            model_id = model_data.get('id', '')
            
            # Get context length (may not be provided by vLLM)
            context_length = model_data.get('context_length', 4096)
            
            # Determine capabilities based on model name
            capabilities = self._determine_model_capabilities(model_id)
            
            model_info = ModelInfo(
                name=model_id,
                display_name=model_id,
                context_length=context_length,
                capabilities=capabilities,
                cost_per_token=0.0,  # Local inference is free
                description=model_data.get('description', ''),
                version=model_data.get('version', '')
            )
            
            self.models[model_id] = model_info
            
        logger.info(f"Discovered {len(self.models)} vLLM models")
        
    async def _register_default_models(self):
        """Register default model configurations"""
        for model_name, config in self.model_defaults.items():
            model_info = ModelInfo(
                name=model_name,
                display_name=model_name,
                context_length=config['context_length'],
                capabilities=config['capabilities'],
                cost_per_token=0.0,
                description=f"vLLM served {model_name} model"
            )
            
            self.models[model_name] = model_info
            
        logger.info(f"Registered {len(self.models)} default vLLM models")
        
    def _determine_model_capabilities(self, model_name: str) -> ProviderCapability:
        """Determine capabilities based on model name"""
        name_lower = model_name.lower()
        
        # Most vLLM models support function calling and streaming
        capabilities = (
            ProviderCapability.FUNCTION_CALLING |
            ProviderCapability.STREAMING |
            ProviderCapability.SYSTEM_PROMPTS |
            ProviderCapability.BATCH_REQUESTS |
            ProviderCapability.JSON_MODE
        )
        
        # Some models have longer context windows
        if any(llama_version in name_lower for llama_version in ['llama-3', 'llama3']):
            capabilities |= ProviderCapability.LONG_CONTEXT
        elif 'mixtral' in name_lower or 'mistral' in name_lower:
            capabilities |= ProviderCapability.LONG_CONTEXT
            
        return capabilities
        
    async def health_check(self) -> HealthMetrics:
        """
        Perform health check on vLLM provider
        
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
                    message = "vLLM is running"
                else:
                    is_healthy = False
                    message = f"vLLM API returned status {response.status}"
                    
        except Exception as e:
            response_time = time.time() - start_time
            is_healthy = False
            message = f"vLLM connection failed: {str(e)}"
            
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
        """List all available vLLM models"""
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
        Generate completion using vLLM (OpenAI-compatible)
        
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
            raise RuntimeError("vLLM provider not initialized")
            
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
                
            # Add vLLM-specific parameters
            if 'top_p' in kwargs:
                payload["top_p"] = kwargs['top_p']
            if 'top_k' in kwargs:
                payload["top_k"] = kwargs['top_k']
            if 'presence_penalty' in kwargs:
                payload["presence_penalty"] = kwargs['presence_penalty']
            if 'frequency_penalty' in kwargs:
                payload["frequency_penalty"] = kwargs['frequency_penalty']
                
            if stream:
                return self._stream_completion(payload, start_time)
            else:
                return await self._single_completion(payload, start_time)
                
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"vLLM completion failed: {e}")
            
            # Update stats
            self.increment_request_count(0, 0.0, response_time)
            
            return {
                'content': f"Error generating completion: {str(e)}",
                'model': model,
                'provider': 'vllm',
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
                raise Exception(f"vLLM API returned status {response.status}: {error_text}")
                
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
                'provider': 'vllm',
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
                raise Exception(f"vLLM streaming API returned status {response.status}: {error_text}")
                
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
                                    'provider': 'vllm',
                                    'done': choice.get('finish_reason') is not None
                                }
                                
                                yield chunk
                                
                        except json.JSONDecodeError:
                            continue
                            
    async def batch_completions(
        self,
        requests: List[Dict[str, Any]],
        model: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Process multiple completions in a batch
        
        Args:
            requests: List of completion requests
            model: Model name to use
            **kwargs: Additional parameters
            
        Returns:
            List of completion responses
        """
        try:
            # Prepare batch payload
            payload = {
                "model": model,
                "requests": requests,
                **kwargs
            }
            
            async with self.session.post(f"{self.base_url}/v1/completions", json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('choices', [])
                else:
                    error_text = await response.text()
                    raise Exception(f"vLLM batch API returned status {response.status}: {error_text}")
                    
        except Exception as e:
            logger.error(f"vLLM batch completion failed: {e}")
            raise
            
    def supports_capability(self, capability: ProviderCapability) -> bool:
        """Check if vLLM provider supports capability"""
        return bool(self.capabilities & capability)
        
    async def get_server_info(self) -> Dict[str, Any]:
        """
        Get vLLM server information
        
        Returns:
            Server information dictionary
        """
        try:
            async with self.session.get(f"{self.base_url}/v1/models") as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Extract server info from models response
                    # vLLM may not provide detailed server info in this endpoint
                    info = {
                        'server_type': 'vllm',
                        'models_available': len(data.get('data', [])),
                        'api_version': 'v1'
                    }
                    
                    return info
                    
        except Exception as e:
            logger.error(f"Failed to get vLLM server info: {e}")
            return {}
            
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
            self.session = None
            
        logger.info("vLLM provider cleaned up")
        
    def __del__(self):
        """Cleanup when object is garbage collected"""
        if self.session and not self.session.closed:
            try:
                # Try to cleanup in background (not awaited)
                asyncio.create_task(self.cleanup())
            except:
                pass