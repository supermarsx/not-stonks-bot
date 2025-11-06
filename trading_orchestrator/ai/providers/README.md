# LLM Provider System Documentation

## Overview

The expanded LLM Provider System provides comprehensive support for multiple Language Model providers, enabling intelligent routing, automatic failover, health monitoring, and performance optimization across different AI services.

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                  AI Models Manager                           │
│        (Enhanced with Multi-Provider Support)               │
└──────────────────────┬──────────────────────────────────────┘
                       │
    ┌──────────────────┼──────────────────┐
    ↓                  ↓                   ↓
┌─────────────┐ ┌─────────────────┐ ┌──────────────┐
│  Provider   │ │   Health        │ │   Failover   │
│  Factory    │ │   Monitor        │ │   Manager    │
└─────────────┘ └─────────────────┘ └──────────────┘
    │                  │                   │
    └──────────────────┼───────────────────┘
                       │
    ┌──────────────────┼──────────────────┐
    ↓                  ↓                   ↓
┌──────────┐    ┌─────────────┐    ┌─────────────┐
│OpenAI    │    │  Ollama     │    │  LocalAI    │
│Provider  │    │  Provider   │    │  Provider   │
└──────────┘    └─────────────┘    └─────────────┘
       │               │                   │
    ┌──┴──┐      ┌────┴────┐         ┌────┴────┐
    ↓     ↓      ↓         ↓         ↓         ↓
Ollama  vLLM  Anthropic  OpenAI     Custom   Custom
Local  Local   Cloud     Compatible  Local    Cloud
```

## Provider Types

### 1. OpenAI Provider
- **Models**: GPT-4, GPT-4 Turbo, GPT-3.5 Turbo
- **Capabilities**: Function calling, streaming, vision, JSON mode
- **Rate Limits**: 5000 req/min, 2M tokens/min
- **Cost**: Tiered pricing per 1K tokens

### 2. Anthropic Provider  
- **Models**: Claude 3.5 Sonnet, Claude 3 Haiku, Claude 3 Opus
- **Capabilities**: Function calling, streaming, vision, long context
- **Rate Limits**: 1000 req/min, 400K tokens/min
- **Cost**: Competitive with OpenAI

### 3. Ollama Provider
- **Models**: Llama 2/3, Mistral, Mixtral, CodeLlama, Gemma, Phi
- **Capabilities**: Function calling (model dependent), streaming, local inference
- **Rate Limits**: 60 req/min, 100K tokens/min (local hardware dependent)
- **Cost**: Free (local inference)

### 4. LocalAI Provider
- **Models**: Various open-source models via OpenAI-compatible API
- **Capabilities**: Function calling, streaming, embeddings
- **Rate Limits**: 60 req/min, 100K tokens/min (local hardware dependent)  
- **Cost**: Free (local inference)

### 5. vLLM Provider
- **Models**: Llama, Mistral, Mixtral, Qwen (high-performance serving)
- **Capabilities**: Function calling, streaming, batch processing
- **Rate Limits**: 120 req/min, 200K tokens/min (local hardware dependent)
- **Cost**: Free (local inference)

### 6. OpenAI-Compatible Provider
- **Models**: Any model served via OpenAI-compatible API
- **Endpoints**: OpenRouter, Azure OpenAI, custom endpoints
- **Capabilities**: Depends on underlying provider
- **Rate Limits**: Configurable per endpoint

## Key Features

### 1. Provider Abstraction
All providers implement a common interface:
```python
from ai.providers import BaseLLMProvider

class CustomProvider(BaseLLMProvider):
    async def generate_completion(self, messages, model, **kwargs):
        # Provider-specific implementation
        pass
        
    async def health_check(self):
        # Health check implementation
        pass
```

### 2. Automatic Failover
- **Health-based routing**: Choose healthy providers automatically
- **Circuit breaker pattern**: Prevent cascading failures
- **Multiple strategies**: Priority-based, round-robin, health-based, LRU
- **Graceful degradation**: Fall back to less optimal providers when primary fails

### 3. Health Monitoring
- **Continuous monitoring**: Automatic health checks
- **Performance metrics**: Response times, success rates, error rates
- **Alert system**: Configurable alerts for performance degradation
- **Historical data**: Track provider performance over time

### 4. Rate Limiting
- **Per-provider limits**: Respect API rate limits
- **Multiple strategies**: Fixed window, sliding window, token bucket
- **Priority-based**: Higher priority requests get preference
- **Adaptive backoff**: Increase delays on consecutive failures

### 5. Capability Detection
- **Feature support**: Detect function calling, streaming, vision support
- **Model capabilities**: Track which models support which features
- **Automatic routing**: Route requests to providers with required capabilities

## Usage Examples

### Basic Setup

```python
import asyncio
from ai.providers import ProviderFactory, ProviderConfig, RateLimit

async def setup_providers():
    factory = ProviderFactory()
    
    # Configure OpenAI provider
    openai_config = ProviderConfig(
        name="openai_primary",
        provider_type="openai", 
        api_key="your-api-key",
        priority=1,
        rate_limit=RateLimit(requests_per_minute=5000)
    )
    
    # Configure Ollama provider (local)
    ollama_config = ProviderConfig(
        name="ollama_local",
        provider_type="ollama",
        base_url="http://localhost:11434",
        priority=2,
        rate_limit=RateLimit(requests_per_minute=60)
    )
    
    # Create providers
    openai_provider = await factory.create_provider(openai_config)
    ollama_provider = await factory.create_provider(ollama_config)
    
    return openai_provider, ollama_provider
```

### Enhanced AI Models Manager

```python
from ai.models.enhanced_ai_models_manager import (
    EnhancedAIModelsManager,
    RequestConfig,
    RequestPriority,
    ModelTier
)

async def setup_ai_manager():
    # Configuration for multiple providers
    config = {
        'providers': [
            {
                'name': 'openai_primary',
                'provider_type': 'openai',
                'api_key': 'your-openai-key',
                'priority': 1
            },
            {
                'name': 'ollama_local', 
                'provider_type': 'ollama',
                'base_url': 'http://localhost:11434',
                'priority': 2
            }
        ],
        'health_check_interval': 30,
        'global_timeout': 30.0
    }
    
    # Initialize enhanced AI models manager
    ai_manager = EnhancedAIModelsManager(config)
    await ai_manager.initialize()
    
    return ai_manager
```

### Request with Automatic Failover

```python
async def make_request_with_failover(ai_manager):
    # Register tools
    async def get_weather(location: str) -> dict:
        return {"location": location, "temperature": "20°C", "condition": "Sunny"}
    
    ai_manager.register_tool("get_weather", get_weather)
    
    # Configure request
    request_config = RequestConfig(
        priority=RequestPriority.HIGH,
        allow_failover=True,
        max_retries=3
    )
    
    # Make request with automatic failover
    result = await ai_manager.generate_completion(
        messages=[{"role": "user", "content": "What's the weather in New York?"}],
        task_type="quick_decision",
        request_config=request_config,
        tools=[
            {
                "type": "function", 
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information"
                }
            }
        ]
    )
    
    print(f"Response: {result['content']}")
    print(f"Provider used: {result['provider']}")
    print(f"Model: {result['model']}")
```

### Health Monitoring

```python
async def monitor_provider_health(ai_manager):
    # Get comprehensive health data
    health_data = await ai_manager.health_check()
    
    print(f"Overall status: {health_data['overall_status']}")
    
    for provider_name, provider_health in health_data['providers'].items():
        if provider_health:
            status = "✓" if provider_health['is_healthy'] else "✗"
            print(f"{status} {provider_name}")
            print(f"  Response time: {provider_health['response_time']:.2f}s")
            print(f"  Success rate: {provider_health['success_rate']:.1%}")
            print(f"  Uptime: {provider_health['uptime_percentage']:.1%}")
```

### Performance Benchmarking

```python
async def benchmark_providers(ai_manager):
    # Get usage statistics
    stats = ai_manager.get_usage_stats()
    
    print("Provider Usage Statistics:")
    for model_name, model_stats in stats['models'].items():
        print(f"\n{model_name}:")
        print(f"  Total requests: {model_stats['request_count']}")
        print(f"  Total tokens: {model_stats['total_tokens']}")
        print(f"  Total cost: ${model_stats['total_cost']:.4f}")
        print(f"  Avg response time: {model_stats['avg_response_time']:.2f}s")
        print(f"  Success rate: {model_stats.get('success_rate', 0):.1%}")
```

### Custom Provider Testing

```python
from ai.providers.testing_suite import ProviderTestSuite, quick_test_provider

async def test_provider_config(config):
    # Quick test of provider configuration
    result = await quick_test_provider(config)
    
    print(f"Provider: {result['provider_name']}")
    print(f"Status: {result['status']}")
    print(f"Success rate: {result['success_rate']:.1%}")
    
    if result['status'] == 'error':
        print(f"Error: {result['error']}")
    
    return result

async def run_comprehensive_tests(factory):
    # Create test suite
    test_suite = ProviderTestSuite(factory)
    
    # Get all providers
    providers = list(factory.providers.values())
    
    # Run comprehensive tests
    results = await test_suite.run_all_tests(providers, parallel=True)
    
    # Export results
    await test_suite.export_results(results, "provider_test_results.json")
    
    # Display summary
    for provider_name, provider_results in results.items():
        passed = len([r for r in provider_results if r.status == "passed"])
        total = len(provider_results)
        print(f"{provider_name}: {passed}/{total} tests passed")
```

## Configuration Guide

### Environment Variables
```bash
# OpenAI
OPENAI_API_KEY=sk-your-openai-key

# Anthropic
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key

# OpenRouter
OPENROUTER_API_KEY=sk-or-your-openrouter-key

# Local endpoints (optional)
OLLAMA_URL=http://localhost:11434
LOCALAI_URL=http://localhost:8080
VLLM_URL=http://localhost:8000
```

### Configuration Files

#### Development Configuration
```json
{
  "providers": [
    {
      "name": "ollama_local",
      "provider_type": "ollama",
      "base_url": "http://localhost:11434",
      "priority": 1,
      "rate_limit": {
        "requests_per_minute": 60,
        "tokens_per_minute": 100000,
        "burst_limit": 10
      }
    },
    {
      "name": "openai_backup",
      "provider_type": "openai",
      "api_key": "env:OPENAI_API_KEY",
      "priority": 2,
      "rate_limit": {
        "requests_per_minute": 1000,
        "tokens_per_minute": 100000,
        "burst_limit": 50
      }
    }
  ],
  "health_check_interval": 60,
  "global_timeout": 30.0
}
```

#### Production Configuration  
```json
{
  "providers": [
    {
      "name": "openai_primary",
      "provider_type": "openai",
      "api_key": "env:OPENAI_API_KEY",
      "priority": 1,
      "rate_limit": {
        "requests_per_minute": 5000,
        "tokens_per_minute": 2000000,
        "burst_limit": 100
      },
      "health_check_interval": 30,
      "enable_caching": true
    },
    {
      "name": "anthropic_primary",
      "provider_type": "anthropic",
      "api_key": "env:ANTHROPIC_API_KEY",
      "priority": 2,
      "rate_limit": {
        "requests_per_minute": 1000,
        "tokens_per_minute": 400000,
        "burst_limit": 20
      },
      "health_check_interval": 30,
      "enable_caching": true
    },
    {
      "name": "ollama_local",
      "provider_type": "ollama",
      "base_url": "http://localhost:11434",
      "priority": 3,
      "rate_limit": {
        "requests_per_minute": 60,
        "tokens_per_minute": 100000,
        "burst_limit": 10
      },
      "health_check_interval": 120
    }
  ],
  "health_check_interval": 30,
  "global_timeout": 30.0,
  "failover_strategy": "health_based"
}
```

## Best Practices

### 1. Provider Selection
- **Primary/Secondary**: Configure primary providers for quality, backup for reliability
- **Local vs Cloud**: Use local providers for cost-sensitive, high-volume tasks
- **Capability Matching**: Route requests based on required capabilities (vision, function calling)

### 2. Rate Limiting
- **Conservative Limits**: Set rate limits below provider maximums
- **Priority Queuing**: Use higher limits for critical requests
- **Monitor Usage**: Track rate limit utilization to avoid throttling

### 3. Health Monitoring
- **Regular Checks**: Enable periodic health monitoring
- **Alert Thresholds**: Configure appropriate alert thresholds
- **Recovery Testing**: Test provider recovery from failure states

### 4. Failover Strategy
- **Health-Based**: Use health-based routing for dynamic selection
- **Priority-Based**: Use priority-based for predictable routing
- **Circuit Breaker**: Enable circuit breaker to prevent cascading failures

### 5. Error Handling
- **Graceful Degradation**: Plan for provider unavailability
- **Retry Logic**: Implement appropriate retry strategies
- **Cost Monitoring**: Track costs across providers

## Troubleshooting

### Common Issues

#### Provider Not Responding
```python
# Check provider health
health = await provider.health_check()
print(f"Status: {health.is_healthy}")
print(f"Message: {health.message}")

# Check connection
try:
    models = await provider.list_models()
    print(f"Models found: {len(models)}")
except Exception as e:
    print(f"Connection error: {e}")
```

#### High Error Rates
```python
# Check provider statistics
stats = await provider.get_stats()
print(f"Error rate: {stats.error_rate:.2%}")
print(f"Total requests: {stats.total_requests}")
print(f"Failed requests: {stats.failed_requests}")

# Check for circuit breaker activation
if hasattr(provider, 'circuit_breaker'):
    state = provider.circuit_breaker.state
    print(f"Circuit breaker state: {state}")
```

#### Rate Limiting Issues
```python
# Check rate limiting status
if hasattr(provider, 'rate_limiter'):
    usage = provider.rate_limiter.get_current_usage()
    print(f"Usage: {usage['requests_last_minute']}/{usage['requests_per_minute_limit']}")
    print(f"Wait time: {provider.rate_limiter.get_wait_time()}s")
```

### Performance Optimization

#### Response Time Issues
1. **Provider Selection**: Choose providers closer to your location
2. **Model Selection**: Use faster models for time-sensitive tasks
3. **Caching**: Enable response caching for repeated requests
4. **Connection Pooling**: Use persistent connections

#### Cost Optimization
1. **Model Selection**: Use appropriate models for task complexity
2. **Token Optimization**: Minimize prompt length and max tokens
3. **Local Models**: Use local models for cost-sensitive tasks
4. **Batch Processing**: Group requests when possible

## Integration Examples

### With Trading Orchestrator

```python
from ai.orchestrator import AITradingOrchestrator
from ai.models.enhanced_ai_models_manager import EnhancedAIModelsManager

async def setup_trading_ai():
    # Setup enhanced AI manager with multiple providers
    ai_config = {
        'providers': [
            {
                'name': 'openai_primary',
                'provider_type': 'openai',
                'api_key': 'env:OPENAI_API_KEY',
                'priority': 1
            },
            {
                'name': 'ollama_local',
                'provider_type': 'ollama',
                'base_url': 'http://localhost:11434',
                'priority': 2
            }
        ]
    }
    
    ai_models = EnhancedAIModelsManager(ai_config)
    await ai_models.initialize()
    
    # Setup trading orchestrator
    orchestrator = AITradingOrchestrator(
        ai_models_manager=ai_models,
        trading_tools=trading_tools,
        trading_mode=TradingMode.PAPER
    )
    
    return orchestrator
```

### With FastAPI Integration

```python
from fastapi import FastAPI
from ai.providers import ProviderFactory

app = FastAPI()
factory = ProviderFactory()

@app.on_event("startup")
async def startup_event():
    # Initialize providers on startup
    config = load_provider_config()  # Your config loading function
    providers = await factory.batch_create_providers(config)
    
    for provider in providers:
        if provider:
            logger.info(f"Started provider: {provider.config.name}")

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # Get appropriate provider
    provider = await factory.get_best_provider(
        capability=ProviderCapability.FUNCTION_CALLING
    )
    
    if not provider:
        raise HTTPException(status_code=503, detail="No providers available")
    
    # Process request
    response = await provider.generate_completion(
        messages=request.messages,
        model=request.model or "gpt-3.5-turbo"
    )
    
    return {"response": response}
```

## API Reference

### Core Classes

#### ProviderFactory
- `create_provider(config)`: Create a single provider
- `batch_create_providers(configs)`: Create multiple providers
- `get_best_provider(capability)`: Get best available provider
- `list_providers()`: List all registered providers

#### BaseLLMProvider (Abstract)
- `initialize()`: Initialize provider connection
- `health_check()`: Perform health check
- `generate_completion()`: Generate text completion
- `list_models()`: List available models
- `supports_capability()`: Check capability support

#### EnhancedAIModelsManager
- `generate_completion()`: Generate with failover support
- `get_model_for_task()`: Get best model for task type
- `register_tool()`: Register function calling tools
- `get_usage_stats()`: Get comprehensive usage statistics

#### ProviderHealthMonitor
- `start_monitoring()`: Begin health monitoring
- `get_provider_health()`: Get provider health data
- `get_best_providers()`: Get providers ranked by health
- `export_health_data()`: Export health data

#### ProviderFailoverManager
- `get_provider()`: Get best available provider
- `execute_with_failover()`: Execute with automatic failover
- `get_failover_stats()`: Get failover statistics
- `set_strategy()`: Change failover strategy

## Contributing

### Adding New Providers

1. **Create Provider Class**: Extend `BaseLLMProvider`
2. **Implement Required Methods**: `initialize`, `health_check`, `generate_completion`, `list_models`
3. **Register in Factory**: Add to `ProviderFactory.PROVIDER_REGISTRY`
4. **Add Tests**: Create test cases in `testing_suite.py`
5. **Update Documentation**: Add provider-specific documentation

### Example Custom Provider

```python
from ai.providers import BaseLLMProvider, ProviderCapability, ProviderConfig

class CustomProvider(BaseLLMProvider):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.capabilities = ProviderCapability.FUNCTION_CALLING | ProviderCapability.STREAMING
        
    async def initialize(self) -> bool:
        # Initialize connection to your service
        try:
            # Your initialization code
            return True
        except Exception:
            return False
            
    async def health_check(self) -> HealthMetrics:
        # Check service health
        start_time = time.time()
        try:
            # Your health check code
            return HealthMetrics(...)
        except Exception as e:
            return HealthMetrics(...)
            
    async def generate_completion(self, messages, model, **kwargs):
        # Implement completion generation
        pass
        
    async def list_models(self) -> List[ModelInfo]:
        # Return available models
        pass
        
    def supports_capability(self, capability: ProviderCapability) -> bool:
        return bool(self.capabilities & capability)
```

## Support

For issues, questions, or contributions:

1. **Check Logs**: Review provider logs for error details
2. **Test Configuration**: Use `quick_test_provider()` to validate configs
3. **Health Checks**: Use health monitoring to identify issues
4. **Community**: Refer to provider-specific documentation

## License

This provider system is part of the Trading Orchestrator project and follows the same licensing terms.