"""
Provider System Examples and Usage Documentation

This module demonstrates comprehensive usage of the expanded LLM provider system
including setup, configuration, testing, and best practices.
"""

import asyncio
import json
from typing import Dict, List, Any

from .providers import (
    ProviderFactory,
    ProviderConfig,
    RateLimit,
    ProviderHealthMonitor,
    ProviderFailoverManager,
    FailoverStrategy,
    ProviderCapability
)
from .providers.testing_suite import ProviderTestSuite, quick_test_provider
from .enhanced_ai_models_manager import EnhancedAIModelsManager, RequestConfig, RequestPriority, ModelTier

from loguru import logger


class ProviderSystemDemo:
    """Comprehensive demonstration of the provider system"""
    
    def __init__(self):
        self.factory = ProviderFactory()
        self.health_monitor = None
        self.failover_manager = None
        self.ai_models_manager = None
        
    async def setup_providers(self):
        """Setup various providers for demonstration"""
        
        # Example 1: OpenAI Provider
        openai_config = {
            "name": "openai_primary",
            "provider_type": "openai",
            "api_key": "your-openai-api-key",  # Use environment variable in production
            "priority": 1,
            "rate_limit": {
                "requests_per_minute": 5000,
                "tokens_per_minute": 2000000,
                "burst_limit": 100
            },
            "health_check_interval": 30,
            "enable_caching": True
        }
        
        # Example 2: Anthropic Provider
        anthropic_config = {
            "name": "anthropic_primary",
            "provider_type": "anthropic", 
            "api_key": "your-anthropic-api-key",
            "priority": 2,
            "rate_limit": {
                "requests_per_minute": 1000,
                "tokens_per_minute": 400000,
                "burst_limit": 20
            }
        }
        
        # Example 3: Ollama (Local) Provider
        ollama_config = {
            "name": "ollama_local",
            "provider_type": "ollama",
            "base_url": "http://localhost:11434",
            "priority": 3,
            "rate_limit": {
                "requests_per_minute": 60,
                "tokens_per_minute": 100000,
                "burst_limit": 10
            },
            "timeout": 60.0
        }
        
        # Example 4: LocalAI Provider
        localai_config = {
            "name": "localai_local",
            "provider_type": "localai",
            "base_url": "http://localhost:8080", 
            "priority": 4,
            "rate_limit": {
                "requests_per_minute": 60,
                "tokens_per_minute": 100000,
                "burst_limit": 10
            }
        }
        
        # Example 5: vLLM Provider
        vllm_config = {
            "name": "vllm_local",
            "provider_type": "vllm",
            "base_url": "http://localhost:8000",
            "priority": 5,
            "rate_limit": {
                "requests_per_minute": 120,
                "tokens_per_minute": 200000,
                "burst_limit": 20
            }
        }
        
        # Example 6: OpenAI-Compatible Provider (OpenRouter)
        openrouter_config = {
            "name": "openrouter_backup",
            "provider_type": "openai_compatible",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "your-openrouter-key",
            "priority": 6,
            "rate_limit": {
                "requests_per_minute": 100,
                "tokens_per_minute": 100000,
                "burst_limit": 10
            }
        }
        
        configs = [
            openai_config,
            anthropic_config, 
            ollama_config,
            localai_config,
            vllm_config,
            openrouter_config
        ]
        
        # Create providers
        providers = await self.factory.batch_create_providers(configs)
        
        # Setup health monitoring and failover
        self.health_monitor = ProviderHealthMonitor(
            check_interval=30,
            history_retention_hours=24
        )
        
        self.failover_manager = ProviderFailoverManager(
            health_monitor=self.health_monitor,
            strategy=FailoverStrategy.HEALTH_BASED,
            global_timeout=30.0
        )
        
        # Register providers with failover manager
        for provider in providers:
            if provider:  # Only register successfully created providers
                self.failover_manager.register_provider(provider)
                
        # Setup enhanced AI models manager
        self.ai_models_manager = EnhancedAIModelsManager({
            'providers': configs,
            'health_check_interval': 30,
            'global_timeout': 30.0
        })
        await self.ai_models_manager.initialize()
        
        return providers
        
    async def demonstrate_basic_usage(self):
        """Demonstrate basic provider usage"""
        
        print("\n=== Basic Provider Usage ===")
        
        # Get available providers
        providers = self.factory.list_providers()
        print(f"Available providers: {list(providers.keys())}")
        
        # Test provider health
        print("\n--- Provider Health Check ---")
        for name, provider in providers.items():
            health = await provider.health_check()
            print(f"{name}: {'✓' if health.is_healthy else '✗'} ({health.message})")
            
        # List available models
        print("\n--- Available Models ---")
        for name, provider in providers.items():
            models = await provider.list_models()
            print(f"{name}: {len(models)} models")
            if models:
                print(f"  Sample: {models[0].name}")
                
    async def demonstrate_failover(self):
        """Demonstrate failover capabilities"""
        
        print("\n=== Failover Demonstration ===")
        
        # Get best provider for a task
        provider = await self.failover_manager.get_provider(
            capability=ProviderCapability.FUNCTION_CALLING,
            preferred_provider="openai_primary"
        )
        
        if provider:
            print(f"Selected provider: {provider.config.name}")
            
            # Simulate a request with failover
            async def make_request(p):
                messages = [{"role": "user", "content": "What is the capital of France?"}]
                return await p.generate_completion(
                    messages=messages,
                    model="gpt-3.5-turbo" if "gpt-3.5-turbo" in p.models else list(p.models.keys())[0]
                )
                
            try:
                result = await self.failover_manager.execute_with_failover(
                    provider=provider,
                    operation=make_request
                )
                print(f"Response: {result.get('content', 'No content')[:100]}...")
                print(f"Provider used: {result.get('provider', 'Unknown')}")
            except Exception as e:
                print(f"Request failed: {e}")
                
    async def demonstrate_testing(self):
        """Demonstrate provider testing suite"""
        
        print("\n=== Provider Testing Suite ===")
        
        # Get list of providers
        providers = list(self.failover_manager.providers.values())
        
        if not providers:
            print("No providers available for testing")
            return
            
        # Create test suite
        test_suite = ProviderTestSuite(self.factory)
        
        # Run tests on first provider as example
        provider = providers[0]
        print(f"Running tests on {provider.config.name}...")
        
        try:
            results = await test_suite.run_provider_tests(provider)
            
            # Display results
            for result in results:
                status_icon = {
                    TestResultStatus.PASSED: "✓",
                    TestResultStatus.FAILED: "✗", 
                    TestResultStatus.WARNING: "⚠",
                    TestResultStatus.SKIPPED: "⊘",
                    TestResultStatus.ERROR: "⚡"
                }
                
                icon = status_icon.get(result.status, "?")
                print(f"{icon} {result.test_name}: {result.message}")
                
        except Exception as e:
            print(f"Testing failed: {e}")
            
    async def demonstrate_ai_manager(self):
        """Demonstrate enhanced AI models manager"""
        
        print("\n=== Enhanced AI Models Manager ===")
        
        # Register a tool
        async def get_weather(location: str) -> Dict[str, Any]:
            """Get weather information for a location"""
            return {
                "location": location,
                "temperature": "20°C",
                "condition": "Sunny"
            }
            
        self.ai_models_manager.register_tool("get_weather", get_weather)
        
        # Test different task types
        tasks = [
            ("strategy_analysis", ModelTier.REASONING),
            ("risk_check", ModelTier.FAST),
            ("quick_decision", ModelTier.FAST),
            ("high_frequency", ModelTier.LOCAL)
        ]
        
        print("\n--- Model Selection by Task Type ---")
        for task_type, expected_tier in tasks:
            model = self.ai_models_manager.get_model_for_task(task_type)
            print(f"{task_type}: {model} (tier: {expected_tier.value})")
            
        # Test completion with tools
        print("\n--- Completion with Function Calling ---")
        try:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather information for a location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"}
                            },
                            "required": ["location"]
                        }
                    }
                }
            ]
            
            result = await self.ai_models_manager.generate_completion(
                messages=[
                    {"role": "user", "content": "What's the weather in New York?"}
                ],
                task_type="risk_check",
                tools=tools,
                session_id="demo_session"
            )
            
            print(f"Response: {result.get('content', 'No content')[:200]}...")
            print(f"Model: {result.get('model', 'Unknown')}")
            print(f"Provider: {result.get('provider', 'Unknown')}")
            
        except Exception as e:
            print(f"Completion failed: {e}")
            
    async def demonstrate_performance_monitoring(self):
        """Demonstrate performance monitoring"""
        
        print("\n=== Performance Monitoring ===")
        
        # Get usage statistics
        stats = self.ai_models_manager.get_usage_stats()
        
        print("Provider Health Summary:")
        for provider_name, health_data in stats['providers'].items():
            if health_data:
                status = "✓" if health_data.get('is_healthy') else "✗"
                uptime = health_data.get('uptime_percentage', 0)
                print(f"  {status} {provider_name}: {uptime:.1f}% uptime")
                
        print("\nFailover Statistics:")
        failover_stats = stats['failover_stats']
        print(f"  Total requests: {failover_stats['total_requests']}")
        print(f"  Failed requests: {failover_stats['failed_requests']}")
        print(f"  Success rate: {(1 - failover_stats['failed_requests'] / max(1, failover_stats['total_requests'])):.1%}")
        
    async def demonstrate_rate_limiting(self):
        """Demonstrate rate limiting"""
        
        print("\n=== Rate Limiting ===")
        
        # Get rate limiting information from providers
        for name, provider in self.factory.providers.items():
            if hasattr(provider, 'config') and hasattr(provider.config, 'rate_limit'):
                rate_limit = provider.config.rate_limit
                print(f"{name}:")
                print(f"  Requests per minute: {rate_limit.requests_per_minute}")
                print(f"  Tokens per minute: {rate_limit.tokens_per_minute}")
                print(f"  Burst limit: {rate_limit.burst_limit}")
                
    async def run_comprehensive_demo(self):
        """Run comprehensive demonstration"""
        
        print("Starting Provider System Comprehensive Demo")
        print("=" * 50)
        
        try:
            # Setup providers
            await self.setup_providers()
            
            # Run demonstrations
            await self.demonstrate_basic_usage()
            await self.demonstrate_failover()
            await self.demonstrate_testing()
            await self.demonstrate_ai_manager()
            await self.demonstrate_performance_monitoring()
            await self.demonstrate_rate_limiting()
            
            print("\n" + "=" * 50)
            print("Demo completed successfully!")
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            print(f"Demo failed: {e}")


# Example configurations for different use cases
EXAMPLE_CONFIGURATIONS = {
    "development": {
        "providers": [
            {
                "name": "local_ollama",
                "provider_type": "ollama",
                "base_url": "http://localhost:11434",
                "priority": 1,
                "rate_limit": {"requests_per_minute": 60},
                "health_check_interval": 60
            },
            {
                "name": "backup_openai",
                "provider_type": "openai",
                "api_key": "env:OPENAI_API_KEY",
                "priority": 2,
                "rate_limit": {"requests_per_minute": 1000},
                "health_check_interval": 30
            }
        ],
        "health_check_interval": 60,
        "global_timeout": 30.0
    },
    
    "production": {
        "providers": [
            {
                "name": "openai_primary",
                "provider_type": "openai", 
                "api_key": "env:OPENAI_API_KEY",
                "priority": 1,
                "rate_limit": {"requests_per_minute": 5000, "tokens_per_minute": 2000000},
                "health_check_interval": 30,
                "enable_caching": True
            },
            {
                "name": "anthropic_primary",
                "provider_type": "anthropic",
                "api_key": "env:ANTHROPIC_API_KEY", 
                "priority": 2,
                "rate_limit": {"requests_per_minute": 1000, "tokens_per_minute": 400000},
                "health_check_interval": 30,
                "enable_caching": True
            },
            {
                "name": "openrouter_backup",
                "provider_type": "openai_compatible",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "env:OPENROUTER_API_KEY",
                "priority": 3,
                "rate_limit": {"requests_per_minute": 100, "tokens_per_minute": 100000},
                "health_check_interval": 60
            },
            {
                "name": "ollama_local",
                "provider_type": "ollama",
                "base_url": "http://localhost:11434",
                "priority": 4,
                "rate_limit": {"requests_per_minute": 60, "tokens_per_minute": 100000},
                "health_check_interval": 120
            }
        ],
        "health_check_interval": 30,
        "global_timeout": 30.0
    },
    
    "research": {
        "providers": [
            {
                "name": "local_vllm",
                "provider_type": "vllm",
                "base_url": "http://localhost:8000",
                "priority": 1,
                "rate_limit": {"requests_per_minute": 200, "tokens_per_minute": 500000},
                "health_check_interval": 60
            },
            {
                "name": "localai_local",
                "provider_type": "localai",
                "base_url": "http://localhost:8080", 
                "priority": 2,
                "rate_limit": {"requests_per_minute": 120, "tokens_per_minute": 200000},
                "health_check_interval": 60
            }
        ],
        "health_check_interval": 60,
        "global_timeout": 60.0
    }
}


async def quick_start_example():
    """Quick start example for new users"""
    
    print("Quick Start: Setting up your first provider")
    print("-" * 40)
    
    # Configuration for OpenAI as primary provider
    config = {
        "name": "my_openai_provider",
        "provider_type": "openai",
        "api_key": "your-api-key-here",  # Replace with actual key
        "priority": 1,
        "rate_limit": {
            "requests_per_minute": 1000,
            "tokens_per_minute": 100000,
            "burst_limit": 50
        },
        "health_check_interval": 30,
        "enable_caching": True
    }
    
    # Test the configuration
    print("Testing provider configuration...")
    result = await quick_test_provider(config)
    
    print(f"Provider: {result['provider_name']}")
    print(f"Status: {result['status']}")
    print(f"Success rate: {result['success_rate']:.1%}")
    
    if result['status'] == 'healthy':
        print("\n✓ Provider is working! You can now integrate it into your application.")
        
        # Example usage
        factory = ProviderFactory()
        provider = await factory.create_from_config_dict(config)
        
        if provider:
            # Make a test request
            response = await provider.generate_completion(
                messages=[{"role": "user", "content": "Hello, world!"}],
                model="gpt-3.5-turbo"
            )
            
            print(f"Test response: {response.get('content', 'No response')}")
            
    else:
        print(f"\n✗ Provider test failed: {result.get('error', 'Unknown error')}")
        print("Check your configuration and API key.")


async def main():
    """Main demonstration function"""
    
    # Run quick start for immediate feedback
    await quick_start_example()
    
    print("\n" + "="*60 + "\n")
    
    # Run comprehensive demo
    demo = ProviderSystemDemo()
    await demo.run_comprehensive_demo()


if __name__ == "__main__":
    # Configure logging
    logger.add("provider_demo.log", level="INFO")
    
    # Run the demonstration
    asyncio.run(main())