"""
Ollama Provider - Local LLM serving via Ollama

Ollama is a local LLM serving framework that supports various open-source models.
This provider provides a high-level interface for interacting with Ollama's REST API.

Features:
- Model management and listing
- Streaming responses
- Function calling support (for compatible models)
- Local model serving
- Zero-cost local inference
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


class OllamaProvider(BaseLLMProvider):
    """
    Provider for local LLM serving via Ollama
    
    Ollama serves various open-source models locally, providing:
    - Zero inference cost
    - Fast response times (local serving)
    - Privacy (no data leaves machine)
    - Support for models like Llama, Mistral, CodeLlama, etc.
    """
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.base_url = config.base_url or "http://localhost:11434"
        self.session: Optional[aiohttp.ClientSession] = None
        self._initialized = False
        
        # Ollama model capabilities mapping
        self.model_capabilities = {
            # Llama 2/3 models
            'llama2': ProviderCapability.SYSTEM_PROMPTS,
            'llama3': ProviderCapability.FUNCTION_CALLING | ProviderCapability.SYSTEM_PROMPTS,
            'llama2-coder': ProviderCapability.FUNCTION_CALLING | ProviderCapability.SYSTEM_PROMPTS,
            
            # Mistral models
            'mistral': ProviderCapability.FUNCTION_CALLING | ProviderCapability.SYSTEM_PROMPTS,
            'mixtral': ProviderCapability.FUNCTION_CALLING | ProviderCapability.SYSTEM_PROMPTS,
            
            # Code models
            'codellama': ProviderCapability.FUNCTION_CALLING | ProviderCapability.SYSTEM_PROMPTS,
            'codellama:code': ProviderCapability.FUNCTION_CALLING | ProviderCapability.SYSTEM_PROMPTS,
            
            # Other models
            'phi': ProviderCapability.SYSTEM_PROMPTS,
            'gemma': ProviderCapability.SYSTEM_PROMPTS,
            'qwen': ProviderCapability.FUNCTION_CALLING | ProviderCapability.SYSTEM_PROMPTS,
        }
        
    async def initialize(self) -> bool:
        """
        Initialize Ollama provider and validate connection
        
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
            
            # Test connection to Ollama
            await self._test_connection()
            
            # Discover available models
            await self._discover_models()
            
            self._initialized = True
            logger.info(f"Ollama provider initialized at {self.base_url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Ollama provider: {e}")
            self._initialized = False
            return False
            
    async def _test_connection(self) -> bool:
        """Test connection to Ollama server"""
        try:
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    model_count = len(data.get('models', []))
                    logger.info(f"Ollama connection successful. {model_count} models available.")
                    return True
                else:
                    raise Exception(f"Ollama API returned status {response.status}")
        except Exception as e:
            raise Exception(f"Cannot connect to Ollama at {self.base_url}: {e}")
            
    async def _discover_models(self):
        """Discover and register available Ollama models"""
        try:
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for model_data in data.get('models', []):
                        model_name = model_data.get('name', '')
                        
                        # Extract base model name for capability detection
                        base_name = self._extract_base_model_name(model_name)
                        capabilities = self.model_capabilities.get(
                            base_name, 
                            ProviderCapability.SYSTEM_PROMPTS
                        )
                        
                        # Get model info from Ollama
                        model_info = await self._get_model_info(model_name)
                        
                        model_info_obj = ModelInfo(
                            name=model_name,
                            display_name=model_name,
                            context_length=model_info.get('context_length', 4096),
                            capabilities=capabilities,
                            cost_per_token=0.0,  # Local inference is free
                            description=model_info.get('description', ''),
                            version=model_info.get('version', '')
                        )
                        
                        self.models[model_name] = model_info_obj
                        
                    logger.info(f"Discovered {len(self.models)} Ollama models")
                    
        except Exception as e:
            logger.error(f"Failed to discover Ollama models: {e}")
            
    def _extract_base_model_name(self, model_name: str) -> str:
        """Extract base model name from full Ollama model name"""
        # Handle variants like "llama3:8b", "llama3:70b", etc.
        name = model_name.lower()
        
        for base_name in self.model_capabilities.keys():
            if base_name in name:
                return base_name
                
        # Try common patterns
        if 'llama3' in name:
            return 'llama3'
        elif 'llama2' in name:
            return 'llama2'
        elif 'codellama' in name:
            return 'codellama'
        elif 'mistral' in name:
            return 'mistral'
        elif 'mixtral' in name:
            return 'mixtral'
        elif 'gemma' in name:
            return 'gemma'
        elif 'phi' in name:
            return 'phi'
        elif 'qwen' in name:
            return 'qwen'
        else:
            return 'llama3'  # Default assumption
            
    async def _get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed model information from Ollama"""
        try:
            payload = {
                "name": model_name
            }
            
            async with self.session.post(f"{self.base_url}/api/show", json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {}
        except Exception:
            return {}
            
    async def health_check(self) -> HealthMetrics:
        """
        Perform health check on Ollama provider
        
        Returns:
            HealthMetrics with current health status
        """
        start_time = time.time()
        
        try:
            # Test API endpoint
            async with self.session.get(f"{self.base_url}/api/version") as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    version_data = await response.json()
                    is_healthy = True
                    message = f"Ollama running (v{version_data.get('version', 'unknown')})"
                else:
                    is_healthy = False
                    message = f"Ollama API returned status {response.status}"
                    
        except Exception as e:
            response_time = time.time() - start_time
            is_healthy = False
            message = f"Ollama connection failed: {str(e)}"
            
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
        """List all available Ollama models"""
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
        Generate completion using Ollama
        
        Args:
            messages: Conversation messages
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Function calling tools (if supported by model)
            stream: Whether to stream response
            **kwargs: Additional parameters
            
        Returns:
            Completion response or async generator for streaming
        """
        if not self._initialized:
            raise RuntimeError("Ollama provider not initialized")
            
        # Validate model exists
        if model not in self.models:
            # Auto-discover if model not found
            await self._discover_models()
            if model not in self.models:
                raise ValueError(f"Model {model} not found in Ollama")
                
        start_time = time.time()
        
        try:
            # Format messages for Ollama
            ollama_messages = self._format_messages(messages)
            
            # Prepare request payload
            payload = {
                "model": model,
                "messages": ollama_messages,
                "stream": stream,
                "options": {
                    "temperature": temperature
                }
            }
            
            if max_tokens:
                payload["options"]["num_predict"] = max_tokens
                
            # Handle tools/function calling for supported models
            if tools and self.supports_capability(ProviderCapability.FUNCTION_CALLING):
                payload["tools"] = tools
                
            if stream:
                return self._stream_completion(payload)
            else:
                return await self._single_completion(payload, start_time)
                
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Ollama completion failed: {e}")
            
            # Update stats
            self.increment_request_count(0, 0.0, response_time)
            
            return {
                'content': f"Error generating completion: {str(e)}",
                'model': model,
                'provider': 'ollama',
                'error': str(e),
                'response_time': response_time
            }
            
    def _format_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Format messages for Ollama API"""
        formatted = []
        
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            # Ollama uses different roles
            if role == 'system':
                # System messages go in the options
                formatted.append({'role': 'user', 'content': f"System: {content}"})
            else:
                formatted.append({'role': role, 'content': content})
                
        return formatted
        
    async def _single_completion(
        self,
        payload: Dict[str, Any],
        start_time: float
    ) -> Dict[str, Any]:
        """Handle single completion (non-streaming)"""
        async with self.session.post(f"{self.base_url}/api/chat", json=payload) as response:
            if response.status != 200:
                raise Exception(f"Ollama API returned status {response.status}")
                
            data = await response.json()
            response_time = time.time() - start_time
            
            # Extract content
            content = data.get('message', {}).get('content', '')
            
            # Estimate tokens used (Ollama doesn't return this directly)
            tokens_used = len(content.split()) * 1.3  # Rough estimation
            
            # Update stats
            self.increment_request_count(tokens_used, 0.0, response_time)
            
            return {
                'content': content,
                'model': payload['model'],
                'provider': 'ollama',
                'response_time': response_time,
                'usage': {
                    'prompt_tokens': 0,  # Not provided by Ollama
                    'completion_tokens': tokens_used,
                    'total_tokens': tokens_used
                }
            }
            
    async def _stream_completion(
        self,
        payload: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle streaming completion"""
        async with self.session.post(f"{self.base_url}/api/chat", json=payload) as response:
            if response.status != 200:
                raise Exception(f"Ollama API returned status {response.status}")
                
            async for line in response.content:
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        
                        if data.get('done'):
                            break
                            
                        chunk = {
                            'content': data.get('message', {}).get('content', ''),
                            'model': payload['model'],
                            'provider': 'ollama',
                            'done': data.get('done', False)
                        }
                        
                        yield chunk
                        
                    except json.JSONDecodeError:
                        continue
                        
    async def pull_model(self, model_name: str) -> bool:
        """
        Pull a new model from Ollama registry
        
        Args:
            model_name: Name of model to pull
            
        Returns:
            True if pull successful
        """
        try:
            payload = {"name": model_name, "stream": True}
            
            async with self.session.post(f"{self.base_url}/api/pull", json=payload) as response:
                if response.status == 200:
                    # Monitor pull progress
                    async for line in response.content:
                        if line:
                            data = json.loads(line.decode('utf-8'))
                            if data.get('status') == 'success':
                                # Model downloaded, refresh model list
                                await self._discover_models()
                                return True
                            elif data.get('status') == 'error':
                                raise Exception(data.get('error', 'Unknown pull error'))
                                
                return False
                
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False
            
    async def delete_model(self, model_name: str) -> bool:
        """
        Delete a model from Ollama
        
        Args:
            model_name: Name of model to delete
            
        Returns:
            True if deletion successful
        """
        try:
            payload = {"name": model_name}
            
            async with self.session.delete(f"{self.base_url}/api/delete", json=payload) as response:
                success = response.status == 200
                
                if success:
                    # Remove from our model list
                    if model_name in self.models:
                        del self.models[model_name]
                        
                return success
                
        except Exception as e:
            logger.error(f"Failed to delete model {model_name}: {e}")
            return False
            
    def supports_capability(self, capability: ProviderCapability) -> bool:
        """Check if Ollama provider supports capability"""
        # This is a general check - specific models may have different capabilities
        return bool(self.capabilities & capability)
        
    def get_supported_models(self, capability: ProviderCapability) -> List[str]:
        """
        Get models that support a specific capability
        
        Args:
            capability: Required capability
            
        Returns:
            List of model names supporting the capability
        """
        supported = []
        
        for model_name, model_info in self.models.items():
            if model_info.capabilities & capability:
                supported.append(model_name)
                
        return supported
        
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
            self.session = None
            
        logger.info("Ollama provider cleaned up")
        
    def __del__(self):
        """Cleanup when object is garbage collected"""
        if self.session and not self.session.closed:
            try:
                # Try to cleanup in background (not awaited)
                asyncio.create_task(self.cleanup())
            except:
                pass