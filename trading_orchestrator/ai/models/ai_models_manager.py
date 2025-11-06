"""
AI Models Manager - Multi-tier LLM orchestration for trading decisions

Supports multiple LLM providers and implements a tiered approach:
- Tier 1 (Reasoning): Claude 3.5 Sonnet, GPT-4 for complex analysis
- Tier 2 (Fast): GPT-3.5 Turbo, Claude Haiku for quick decisions  
- Tier 3 (Local): Local SLMs for high-frequency operations

Each tier is optimized for different latency/cost/quality tradeoffs.
"""

from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass
import json
import asyncio

from loguru import logger

# Optional imports for LLM providers
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    AsyncOpenAI = None
    OPENAI_AVAILABLE = False

try:
    from anthropic import AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    AsyncAnthropic = None
    ANTHROPIC_AVAILABLE = False


class ModelTier(Enum):
    """LLM tier classification"""
    REASONING = "reasoning"  # High-quality, slower, expensive
    FAST = "fast"            # Balanced quality/speed/cost
    LOCAL = "local"          # Local models, fastest, cheapest


class ModelProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


@dataclass
class ModelConfig:
    """Configuration for an LLM model"""
    name: str
    provider: ModelProvider
    tier: ModelTier
    max_tokens: int = 4096
    temperature: float = 0.7
    supports_function_calling: bool = True
    cost_per_1k_tokens: float = 0.01
    
    
# Model registry with predefined configurations
MODEL_REGISTRY = {
    # Tier 1: Reasoning models
    "claude-3-5-sonnet": ModelConfig(
        name="claude-3-5-sonnet-20241022",
        provider=ModelProvider.ANTHROPIC,
        tier=ModelTier.REASONING,
        max_tokens=8192,
        temperature=0.7,
        supports_function_calling=True,
        cost_per_1k_tokens=0.015
    ),
    "gpt-4-turbo": ModelConfig(
        name="gpt-4-turbo-preview",
        provider=ModelProvider.OPENAI,
        tier=ModelTier.REASONING,
        max_tokens=4096,
        temperature=0.7,
        supports_function_calling=True,
        cost_per_1k_tokens=0.01
    ),
    
    # Tier 2: Fast models
    "gpt-3.5-turbo": ModelConfig(
        name="gpt-3.5-turbo",
        provider=ModelProvider.OPENAI,
        tier=ModelTier.FAST,
        max_tokens=4096,
        temperature=0.7,
        supports_function_calling=True,
        cost_per_1k_tokens=0.0015
    ),
    "claude-haiku": ModelConfig(
        name="claude-3-haiku-20240307",
        provider=ModelProvider.ANTHROPIC,
        tier=ModelTier.FAST,
        max_tokens=4096,
        temperature=0.7,
        supports_function_calling=True,
        cost_per_1k_tokens=0.00025
    ),
    
    # Tier 3: Local models (placeholder)
    "local-slm": ModelConfig(
        name="local-slm",
        provider=ModelProvider.LOCAL,
        tier=ModelTier.LOCAL,
        max_tokens=2048,
        temperature=0.7,
        supports_function_calling=False,
        cost_per_1k_tokens=0.0
    )
}


class AIModelsManager:
    """
    Manages multiple LLM providers and handles model selection, caching, and orchestration
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        default_tier: ModelTier = ModelTier.FAST
    ):
        """
        Initialize AI models manager
        
        Args:
            openai_api_key: OpenAI API key
            anthropic_api_key: Anthropic API key
            default_tier: Default model tier to use
        """
        self.default_tier = default_tier
        
        # Initialize clients
        self.openai_client = AsyncOpenAI(api_key=openai_api_key) if openai_api_key else None
        self.anthropic_client = AsyncAnthropic(api_key=anthropic_api_key) if anthropic_api_key else None
        
        # Cache and state
        self.conversation_history: Dict[str, List[Dict]] = {}
        self.tool_functions: Dict[str, Callable] = {}
        self.usage_stats: Dict[str, Dict] = {}
        
        logger.info(f"AI Models Manager initialized (default_tier={default_tier.value})")
        
    def register_tool(self, name: str, function: Callable):
        """Register a tool function that models can call"""
        self.tool_functions[name] = function
        logger.info(f"Registered tool: {name}")
        
    def get_model_for_task(
        self,
        task_type: str,
        preferred_tier: Optional[ModelTier] = None
    ) -> ModelConfig:
        """
        Select appropriate model based on task type and tier
        
        Args:
            task_type: Type of task (e.g., 'strategy_analysis', 'risk_check', 'quick_decision')
            preferred_tier: Preferred model tier (uses default if not specified)
            
        Returns:
            ModelConfig for selected model
        """
        tier = preferred_tier or self.default_tier
        
        # Task-specific model selection logic
        task_tier_map = {
            'strategy_analysis': ModelTier.REASONING,
            'market_analysis': ModelTier.REASONING,
            'backtest_review': ModelTier.REASONING,
            'risk_check': ModelTier.FAST,
            'quick_decision': ModelTier.FAST,
            'order_routing': ModelTier.FAST,
            'position_sizing': ModelTier.FAST,
            'high_frequency': ModelTier.LOCAL
        }
        
        # Use task-specific tier if available
        if task_type in task_tier_map:
            tier = task_tier_map[task_type]
            
        # Select model from registry based on tier
        if tier == ModelTier.REASONING:
            model_key = "claude-3-5-sonnet" if self.anthropic_client else "gpt-4-turbo"
        elif tier == ModelTier.FAST:
            model_key = "gpt-3.5-turbo" if self.openai_client else "claude-haiku"
        else:  # LOCAL
            model_key = "local-slm"
            
        return MODEL_REGISTRY[model_key]
        
    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        model_key: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        max_tool_calls: int = 5,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate completion with automatic tool calling support
        
        Args:
            messages: Conversation messages
            model_key: Specific model to use (None = auto-select)
            tools: Tool definitions for function calling
            max_tool_calls: Maximum number of tool calls to make
            session_id: Session ID for conversation history
            
        Returns:
            Response dictionary with content, tool calls, and metadata
        """
        try:
            # Get model config
            if model_key:
                model_config = MODEL_REGISTRY.get(model_key)
                if not model_config:
                    raise ValueError(f"Unknown model: {model_key}")
            else:
                model_config = self.get_model_for_task('general')
                
            # Add to conversation history if session provided
            if session_id:
                if session_id not in self.conversation_history:
                    self.conversation_history[session_id] = []
                self.conversation_history[session_id].extend(messages)
                messages = self.conversation_history[session_id]
                
            # Route to appropriate provider
            if model_config.provider == ModelProvider.OPENAI:
                response = await self._generate_openai(
                    messages, model_config, tools, max_tool_calls
                )
            elif model_config.provider == ModelProvider.ANTHROPIC:
                response = await self._generate_anthropic(
                    messages, model_config, tools, max_tool_calls
                )
            else:
                response = await self._generate_local(
                    messages, model_config
                )
                
            # Update usage stats
            self._update_usage_stats(model_config, response.get('usage', {}))
            
            # Store response in history
            if session_id and response.get('content'):
                self.conversation_history[session_id].append({
                    'role': 'assistant',
                    'content': response['content']
                })
                
            return response
            
        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            return {'error': str(e)}
            
    async def _generate_openai(
        self,
        messages: List[Dict],
        model_config: ModelConfig,
        tools: Optional[List[Dict]],
        max_tool_calls: int
    ) -> Dict[str, Any]:
        """Generate completion using OpenAI API"""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
            
        tool_calls_made = 0
        current_messages = messages.copy()
        
        while tool_calls_made < max_tool_calls:
            # Make API call
            kwargs = {
                'model': model_config.name,
                'messages': current_messages,
                'max_tokens': model_config.max_tokens,
                'temperature': model_config.temperature
            }
            
            if tools and model_config.supports_function_calling:
                kwargs['tools'] = tools
                kwargs['tool_choice'] = 'auto'
                
            response = await self.openai_client.chat.completions.create(**kwargs)
            
            message = response.choices[0].message
            
            # Check if model wants to call a tool
            if message.tool_calls:
                tool_calls_made += 1
                
                # Execute tool calls
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    # Call the tool
                    if function_name in self.tool_functions:
                        result = await self.tool_functions[function_name](**function_args)
                    else:
                        result = {'error': f'Tool {function_name} not registered'}
                        
                    # Add tool response to messages
                    current_messages.append({
                        'role': 'assistant',
                        'content': None,
                        'tool_calls': [tool_call]
                    })
                    current_messages.append({
                        'role': 'tool',
                        'tool_call_id': tool_call.id,
                        'content': json.dumps(result)
                    })
                    
                # Continue conversation with tool results
                continue
            else:
                # No more tool calls, return final response
                return {
                    'content': message.content,
                    'model': model_config.name,
                    'provider': model_config.provider.value,
                    'tier': model_config.tier.value,
                    'tool_calls_made': tool_calls_made,
                    'usage': {
                        'prompt_tokens': response.usage.prompt_tokens,
                        'completion_tokens': response.usage.completion_tokens,
                        'total_tokens': response.usage.total_tokens
                    }
                }
                
        # Max tool calls reached
        return {
            'content': 'Maximum tool calls reached',
            'model': model_config.name,
            'provider': model_config.provider.value,
            'tier': model_config.tier.value,
            'tool_calls_made': tool_calls_made,
            'error': 'max_tool_calls_exceeded'
        }
        
    async def _generate_anthropic(
        self,
        messages: List[Dict],
        model_config: ModelConfig,
        tools: Optional[List[Dict]],
        max_tool_calls: int
    ) -> Dict[str, Any]:
        """Generate completion using Anthropic API"""
        if not self.anthropic_client:
            raise ValueError("Anthropic client not initialized")
            
        # Convert messages to Anthropic format
        system_messages = [m['content'] for m in messages if m['role'] == 'system']
        system = system_messages[0] if system_messages else None
        
        anthropic_messages = [
            {'role': m['role'], 'content': m['content']}
            for m in messages
            if m['role'] in ['user', 'assistant']
        ]
        
        tool_calls_made = 0
        current_messages = anthropic_messages.copy()
        
        while tool_calls_made < max_tool_calls:
            # Make API call
            kwargs = {
                'model': model_config.name,
                'messages': current_messages,
                'max_tokens': model_config.max_tokens,
                'temperature': model_config.temperature
            }
            
            if system:
                kwargs['system'] = system
                
            if tools and model_config.supports_function_calling:
                kwargs['tools'] = tools
                
            response = await self.anthropic_client.messages.create(**kwargs)
            
            # Check for tool use
            tool_use_blocks = [
                block for block in response.content
                if block.type == 'tool_use'
            ]
            
            if tool_use_blocks:
                tool_calls_made += 1
                
                # Execute tool calls
                tool_results = []
                for tool_block in tool_use_blocks:
                    function_name = tool_block.name
                    function_args = tool_block.input
                    
                    # Call the tool
                    if function_name in self.tool_functions:
                        result = await self.tool_functions[function_name](**function_args)
                    else:
                        result = {'error': f'Tool {function_name} not registered'}
                        
                    tool_results.append({
                        'type': 'tool_result',
                        'tool_use_id': tool_block.id,
                        'content': json.dumps(result)
                    })
                    
                # Add assistant response and tool results to messages
                current_messages.append({
                    'role': 'assistant',
                    'content': response.content
                })
                current_messages.append({
                    'role': 'user',
                    'content': tool_results
                })
                
                # Continue conversation with tool results
                continue
            else:
                # No more tool calls, extract text content
                text_content = ''.join([
                    block.text for block in response.content
                    if hasattr(block, 'text')
                ])
                
                return {
                    'content': text_content,
                    'model': model_config.name,
                    'provider': model_config.provider.value,
                    'tier': model_config.tier.value,
                    'tool_calls_made': tool_calls_made,
                    'usage': {
                        'prompt_tokens': response.usage.input_tokens,
                        'completion_tokens': response.usage.output_tokens,
                        'total_tokens': response.usage.input_tokens + response.usage.output_tokens
                    }
                }
                
        # Max tool calls reached
        return {
            'content': 'Maximum tool calls reached',
            'model': model_config.name,
            'provider': model_config.provider.value,
            'tier': model_config.tier.value,
            'tool_calls_made': tool_calls_made,
            'error': 'max_tool_calls_exceeded'
        }
        
    async def _generate_local(
        self,
        messages: List[Dict],
        model_config: ModelConfig
    ) -> Dict[str, Any]:
        """Generate completion using local model"""
        # Placeholder for local model integration
        # In production, integrate with local LLM server (Ollama, vLLM, etc.)
        
        return {
            'content': 'Local model integration pending',
            'model': model_config.name,
            'provider': model_config.provider.value,
            'tier': model_config.tier.value,
            'usage': {'total_tokens': 0}
        }
        
    def _update_usage_stats(self, model_config: ModelConfig, usage: Dict):
        """Track model usage and costs"""
        model_name = model_config.name
        
        if model_name not in self.usage_stats:
            self.usage_stats[model_name] = {
                'total_tokens': 0,
                'total_cost': 0.0,
                'request_count': 0
            }
            
        total_tokens = usage.get('total_tokens', 0)
        cost = (total_tokens / 1000) * model_config.cost_per_1k_tokens
        
        self.usage_stats[model_name]['total_tokens'] += total_tokens
        self.usage_stats[model_name]['total_cost'] += cost
        self.usage_stats[model_name]['request_count'] += 1
        
    def get_usage_stats(self) -> Dict[str, Dict]:
        """Get usage statistics for all models"""
        return self.usage_stats.copy()
        
    def clear_conversation(self, session_id: str):
        """Clear conversation history for a session"""
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]
            logger.info(f"Cleared conversation history for session {session_id}")
