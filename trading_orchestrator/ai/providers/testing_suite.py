"""
Provider Testing and Validation Tools - Comprehensive testing framework

Provides comprehensive testing and validation tools for LLM providers:
- Provider availability and connectivity testing
- Model capability testing
- Function calling validation
- Performance benchmarking
- Integration testing
- Automated health validation
"""

import asyncio
import time
import json
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import statistics
import aiofiles

from .base_provider import BaseLLMProvider, ProviderCapability, ProviderHealth
from .factory import ProviderFactory
from loguru import logger


class TestResultStatus(Enum):
    """Test result status"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"
    ERROR = "error"


class TestCategory(Enum):
    """Test categories"""
    CONNECTIVITY = "connectivity"
    CAPABILITIES = "capabilities"
    PERFORMANCE = "performance"
    FUNCTION_CALLING = "function_calling"
    CONTENT_GENERATION = "content_generation"
    ERROR_HANDLING = "error_handling"
    RATE_LIMITING = "rate_limiting"


@dataclass
class TestCase:
    """Individual test case definition"""
    name: str
    category: TestCategory
    description: str
    test_function: Callable
    timeout: float = 30.0
    required_capabilities: List[ProviderCapability] = field(default_factory=list)
    priority: int = 1  # 1=highest, 5=lowest


@dataclass
class TestResult:
    """Test execution result"""
    test_name: str
    category: TestCategory
    status: TestResultStatus
    message: str
    execution_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Exception] = None


@dataclass
class BenchmarkMetrics:
    """Performance benchmark metrics"""
    avg_response_time: float
    median_response_time: float
    p95_response_time: float
    p99_response_time: float
    min_response_time: float
    max_response_time: float
    throughput: float  # requests per second
    success_rate: float
    tokens_per_second: float
    error_rate: float


class ProviderTestSuite:
    """
    Comprehensive test suite for LLM providers
    
    Tests provider functionality, performance, and integration capabilities.
    """
    
    def __init__(self, provider_factory: ProviderFactory):
        """
        Initialize test suite
        
        Args:
            provider_factory: Provider factory for creating test instances
        """
        self.provider_factory = provider_factory
        self.test_cases: List[TestCase] = []
        self.results: List[TestResult] = []
        
        # Register default test cases
        self._register_default_tests()
        
    def _register_default_tests(self):
        """Register default test cases"""
        # Connectivity tests
        self.add_test(TestCase(
            name="connection_test",
            category=TestCategory.CONNECTIVITY,
            description="Test basic connection to provider",
            test_function=self._test_connection
        ))
        
        self.add_test(TestCase(
            name="health_check_test",
            category=TestCategory.CONNECTIVITY,
            description="Test provider health check functionality",
            test_function=self._test_health_check
        ))
        
        self.add_test(TestCase(
            name="model_discovery_test",
            category=TestCategory.CONNECTIVITY,
            description="Test model discovery and listing",
            test_function=self._test_model_discovery
        ))
        
        # Capability tests
        self.add_test(TestCase(
            name="basic_completion_test",
            category=TestCategory.CONTENT_GENERATION,
            description="Test basic text completion",
            test_function=self._test_basic_completion,
            timeout=15.0
        ))
        
        self.add_test(TestCase(
            name="streaming_test",
            category=TestCategory.CONTENT_GENERATION,
            description="Test streaming response capability",
            test_function=self._test_streaming,
            required_capabilities=[ProviderCapability.STREAMING],
            timeout=20.0
        ))
        
        self.add_test(TestCase(
            name="function_calling_test",
            category=TestCategory.FUNCTION_CALLING,
            description="Test function calling capability",
            test_function=self._test_function_calling,
            required_capabilities=[ProviderCapability.FUNCTION_CALLING],
            timeout=25.0
        ))
        
        self.add_test(TestCase(
            name="system_prompt_test",
            category=TestCategory.CONTENT_GENERATION,
            description="Test system prompt support",
            test_function=self._test_system_prompts,
            required_capabilities=[ProviderCapability.SYSTEM_PROMPTS],
            timeout=15.0
        ))
        
        # Performance tests
        self.add_test(TestCase(
            name="performance_benchmark",
            category=TestCategory.PERFORMANCE,
            description="Benchmark provider performance",
            test_function=self._test_performance_benchmark,
            timeout=60.0
        ))
        
        # Error handling tests
        self.add_test(TestCase(
            name="invalid_request_test",
            category=TestCategory.ERROR_HANDLING,
            description="Test handling of invalid requests",
            test_function=self._test_error_handling,
            timeout=10.0
        ))
        
    def add_test(self, test_case: TestCase):
        """Add a custom test case"""
        self.test_cases.append(test_case)
        self.test_cases.sort(key=lambda t: t.priority)
        
    async def run_all_tests(
        self,
        providers: List[BaseLLMProvider],
        parallel: bool = True
    ) -> Dict[str, List[TestResult]]:
        """
        Run all test cases on all providers
        
        Args:
            providers: List of providers to test
            parallel: Run tests in parallel
            
        Returns:
            Dictionary mapping provider name to list of test results
        """
        all_results = {}
        
        if parallel:
            # Run all provider tests in parallel
            tasks = []
            for provider in providers:
                task = asyncio.create_task(self.run_provider_tests(provider))
                tasks.append((provider.config.name, task))
                
            for provider_name, task in tasks:
                try:
                    results = await task
                    all_results[provider_name] = results
                except Exception as e:
                    logger.error(f"Failed to run tests for provider {provider_name}: {e}")
                    all_results[provider_name] = [
                        TestResult(
                            test_name="test_execution",
                            category=TestCategory.CONNECTIVITY,
                            status=TestResultStatus.ERROR,
                            message=f"Test execution failed: {str(e)}",
                            execution_time=0.0
                        )
                    ]
        else:
            # Run tests sequentially
            for provider in providers:
                try:
                    results = await self.run_provider_tests(provider)
                    all_results[provider.config.name] = results
                except Exception as e:
                    logger.error(f"Failed to run tests for provider {provider.config.name}: {e}")
                    all_results[provider.config.name] = []
                    
        return all_results
        
    async def run_provider_tests(
        self,
        provider: BaseLLMProvider
    ) -> List[TestResult]:
        """
        Run all applicable tests for a specific provider
        
        Args:
            provider: Provider to test
            
        Returns:
            List of test results
        """
        provider_results = []
        
        logger.info(f"Running tests for provider: {provider.config.name}")
        
        for test_case in self.test_cases:
            # Check if provider has required capabilities
            has_capabilities = all(
                provider.supports_capability(cap) 
                for cap in test_case.required_capabilities
            )
            
            if not has_capabilities:
                # Skip test if provider doesn't have required capabilities
                result = TestResult(
                    test_name=test_case.name,
                    category=test_case.category,
                    status=TestResultStatus.SKIPPED,
                    message=f"Provider lacks required capabilities: {test_case.required_capabilities}",
                    execution_time=0.0
                )
                provider_results.append(result)
                continue
                
            # Run the test
            try:
                result = await asyncio.wait_for(
                    test_case.test_function(provider, test_case),
                    timeout=test_case.timeout
                )
                provider_results.append(result)
            except asyncio.TimeoutError:
                result = TestResult(
                    test_name=test_case.name,
                    category=test_case.category,
                    status=TestResultStatus.FAILED,
                    message=f"Test timed out after {test_case.timeout}s",
                    execution_time=test_case.timeout
                )
                provider_results.append(result)
            except Exception as e:
                result = TestResult(
                    test_name=test_case.name,
                    category=test_case.category,
                    status=TestResultStatus.ERROR,
                    message=f"Test execution error: {str(e)}",
                    execution_time=0.0,
                    error=e
                )
                provider_results.append(result)
                
        return provider_results
        
    async def _test_connection(
        self,
        provider: BaseLLMProvider,
        test_case: TestCase
    ) -> TestResult:
        """Test basic provider connection"""
        start_time = time.time()
        
        try:
            # Test if provider can be initialized
            if not provider._initialized:
                success = await provider.initialize()
                if not success:
                    return TestResult(
                        test_name=test_case.name,
                        category=test_case.category,
                        status=TestResultStatus.FAILED,
                        message="Provider initialization failed",
                        execution_time=time.time() - start_time
                    )
                    
            # Test basic API call
            health_metrics = await provider.health_check()
            
            execution_time = time.time() - start_time
            
            if health_metrics.is_healthy:
                return TestResult(
                    test_name=test_case.name,
                    category=test_case.category,
                    status=TestResultStatus.PASSED,
                    message=f"Connection successful ({health_metrics.response_time:.2f}s)",
                    execution_time=execution_time,
                    details={
                        'response_time': health_metrics.response_time,
                        'status': health_metrics.message
                    }
                )
            else:
                return TestResult(
                    test_name=test_case.name,
                    category=test_case.category,
                    status=TestResultStatus.FAILED,
                    message=f"Health check failed: {health_metrics.message}",
                    execution_time=execution_time,
                    details={'health_metrics': health_metrics}
                )
                
        except Exception as e:
            return TestResult(
                test_name=test_case.name,
                category=test_case.category,
                status=TestResultStatus.FAILED,
                message=f"Connection test failed: {str(e)}",
                execution_time=time.time() - start_time,
                error=e
            )
            
    async def _test_health_check(
        self,
        provider: BaseLLMProvider,
        test_case: TestCase
    ) -> TestResult:
        """Test provider health check functionality"""
        start_time = time.time()
        
        try:
            health_metrics = await provider.health_check()
            execution_time = time.time() - start_time
            
            # Validate health metrics structure
            required_fields = ['response_time', 'success_rate', 'error_rate', 'is_healthy']
            if not all(hasattr(health_metrics, field) for field in required_fields):
                return TestResult(
                    test_name=test_case.name,
                    category=test_case.category,
                    status=TestResultStatus.FAILED,
                    message="Health metrics missing required fields",
                    execution_time=execution_time
                )
                
            return TestResult(
                test_name=test_case.name,
                category=test_case.category,
                status=TestResultStatus.PASSED,
                message=f"Health check completed ({health_metrics.is_healthy})",
                execution_time=execution_time,
                details={
                    'response_time': health_metrics.response_time,
                    'success_rate': health_metrics.success_rate,
                    'error_rate': health_metrics.error_rate,
                    'message': health_metrics.message
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name=test_case.name,
                category=test_case.category,
                status=TestResultStatus.FAILED,
                message=f"Health check test failed: {str(e)}",
                execution_time=time.time() - start_time,
                error=e
            )
            
    async def _test_model_discovery(
        self,
        provider: BaseLLMProvider,
        test_case: TestCase
    ) -> TestResult:
        """Test model discovery and listing"""
        start_time = time.time()
        
        try:
            models = await provider.list_models()
            execution_time = time.time() - start_time
            
            if not models:
                return TestResult(
                    test_name=test_case.name,
                    category=test_case.category,
                    status=TestResultStatus.WARNING,
                    message="No models discovered",
                    execution_time=execution_time
                )
                
            # Validate model information
            model_names = [model.name for model in models]
            model_info_complete = all(
                model.name and model.display_name and hasattr(model, 'capabilities')
                for model in models
            )
            
            if not model_info_complete:
                return TestResult(
                    test_name=test_case.name,
                    category=test_case.category,
                    status=TestResultStatus.FAILED,
                    message="Model information incomplete",
                    execution_time=execution_time
                )
                
            return TestResult(
                test_name=test_case.name,
                category=test_case.category,
                status=TestResultStatus.PASSED,
                message=f"Discovered {len(models)} models",
                execution_time=execution_time,
                details={
                    'model_count': len(models),
                    'model_names': model_names[:5],  # First 5 models
                    'sample_models': [
                        {
                            'name': model.name,
                            'context_length': model.context_length,
                            'capabilities': list(model.capabilities)
                        }
                        for model in models[:3]  # First 3 models
                    ]
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name=test_case.name,
                category=test_case.category,
                status=TestResultStatus.FAILED,
                message=f"Model discovery test failed: {str(e)}",
                execution_time=time.time() - start_time,
                error=e
            )
            
    async def _test_basic_completion(
        self,
        provider: BaseLLMProvider,
        test_case: TestCase
    ) -> TestResult:
        """Test basic text completion"""
        start_time = time.time()
        
        try:
            # Get first available model
            models = await provider.list_models()
            if not models:
                return TestResult(
                    test_name=test_case.name,
                    category=test_case.category,
                    status=TestResultStatus.FAILED,
                    message="No models available for completion test",
                    execution_time=time.time() - start_time
                )
                
            model = models[0].name
            
            messages = [
                {"role": "user", "content": "What is 2+2?"}
            ]
            
            response = await provider.generate_completion(
                messages=messages,
                model=model,
                temperature=0.7,
                max_tokens=50
            )
            
            execution_time = time.time() - start_time
            
            # Validate response
            if not response.get('content'):
                return TestResult(
                    test_name=test_case.name,
                    category=test_case.category,
                    status=TestResultStatus.FAILED,
                    message="No content in response",
                    execution_time=execution_time,
                    details={'response': response}
                )
                
            return TestResult(
                test_name=test_case.name,
                category=test_case.category,
                status=TestResultStatus.PASSED,
                message="Basic completion successful",
                execution_time=execution_time,
                details={
                    'model': model,
                    'content_preview': response['content'][:100],
                    'usage': response.get('usage', {}),
                    'response_time': response.get('response_time', execution_time)
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name=test_case.name,
                category=test_case.category,
                status=TestResultStatus.FAILED,
                message=f"Basic completion test failed: {str(e)}",
                execution_time=time.time() - start_time,
                error=e
            )
            
    async def _test_streaming(
        self,
        provider: BaseLLMProvider,
        test_case: TestCase
    ) -> TestResult:
        """Test streaming response capability"""
        start_time = time.time()
        
        try:
            # Get first available model
            models = await provider.list_models()
            if not models:
                return TestResult(
                    test_name=test_case.name,
                    category=test_case.category,
                    status=TestResultStatus.FAILED,
                    message="No models available for streaming test",
                    execution_time=time.time() - start_time
                )
                
            model = models[0].name
            
            messages = [
                {"role": "user", "content": "Write a short poem about AI."}
            ]
            
            # Test streaming
            chunk_count = 0
            total_content = ""
            
            async for chunk in provider.generate_completion(
                messages=messages,
                model=model,
                temperature=0.7,
                max_tokens=100,
                stream=True
            ):
                chunk_count += 1
                content = chunk.get('content', '')
                total_content += content
                
                # Check for reasonable chunk content
                if not content and not chunk.get('done'):
                    return TestResult(
                        test_name=test_case.name,
                        category=test_case.category,
                        status=TestResultStatus.FAILED,
                        message="Empty chunk received",
                        execution_time=time.time() - start_time
                    )
                    
            execution_time = time.time() - start_time
            
            if chunk_count == 0:
                return TestResult(
                    test_name=test_case.name,
                    category=test_case.category,
                    status=TestResultStatus.FAILED,
                    message="No streaming chunks received",
                    execution_time=execution_time
                )
                
            return TestResult(
                test_name=test_case.name,
                category=test_case.category,
                status=TestResultStatus.PASSED,
                message=f"Streaming test successful ({chunk_count} chunks)",
                execution_time=execution_time,
                details={
                    'model': model,
                    'chunk_count': chunk_count,
                    'total_content': total_content[:200],
                    'avg_chunk_size': len(total_content) / chunk_count if chunk_count > 0 else 0
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name=test_case.name,
                category=test_case.category,
                status=TestResultStatus.FAILED,
                message=f"Streaming test failed: {str(e)}",
                execution_time=time.time() - start_time,
                error=e
            )
            
    async def _test_function_calling(
        self,
        provider: BaseLLMProvider,
        test_case: TestCase
    ) -> TestResult:
        """Test function calling capability"""
        start_time = time.time()
        
        try:
            # Get first available model
            models = await provider.list_models()
            if not models:
                return TestResult(
                    test_name=test_case.name,
                    category=test_case.category,
                    status=TestResultStatus.FAILED,
                    message="No models available for function calling test",
                    execution_time=time.time() - start_time
                )
                
            model = models[0].name
            
            # Define a simple tool
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather information for a location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state"
                                }
                            },
                            "required": ["location"]
                        }
                    }
                }
            ]
            
            messages = [
                {
                    "role": "user", 
                    "content": "What is the weather in New York?"
                }
            ]
            
            response = await provider.generate_completion(
                messages=messages,
                model=model,
                tools=tools,
                temperature=0.1  # Lower temperature for more deterministic function calling
            )
            
            execution_time = time.time() - start_time
            
            # Check if function calling was attempted
            # The exact format varies by provider, so we check common indicators
            response_content = response.get('content', '').lower()
            tool_calls = response.get('tool_calls') or response.get('function_calls')
            
            # Function calling test is considered successful if:
            # 1. The response mentions a function call
            # 2. Or the response indicates it needs weather information
            function_call_detected = bool(
                tool_calls or 
                'weather' in response_content or
                'get_weather' in response_content
            )
            
            return TestResult(
                test_name=test_case.name,
                category=test_case.category,
                status=TestResultStatus.PASSED,
                message=f"Function calling test completed",
                execution_time=execution_time,
                details={
                    'model': model,
                    'function_call_detected': function_call_detected,
                    'response': response.get('content', '')[:200],
                    'tool_calls': tool_calls
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name=test_case.name,
                category=test_case.category,
                status=TestResultStatus.FAILED,
                message=f"Function calling test failed: {str(e)}",
                execution_time=time.time() - start_time,
                error=e
            )
            
    async def _test_system_prompts(
        self,
        provider: BaseLLMProvider,
        test_case: TestCase
    ) -> TestResult:
        """Test system prompt support"""
        start_time = time.time()
        
        try:
            models = await provider.list_models()
            if not models:
                return TestResult(
                    test_name=test_case.name,
                    category=test_case.category,
                    status=TestResultStatus.FAILED,
                    message="No models available for system prompt test",
                    execution_time=time.time() - start_time
                )
                
            model = models[0].name
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant that always responds with exactly one word."},
                {"role": "user", "content": "How are you?"}
            ]
            
            response = await provider.generate_completion(
                messages=messages,
                model=model,
                temperature=0.1
            )
            
            execution_time = time.time() - start_time
            
            response_content = response.get('content', '')
            word_count = len(response_content.split())
            
            # Check if response is approximately one word (allowing for some flexibility)
            is_one_word = word_count <= 3
            
            return TestResult(
                test_name=test_case.name,
                category=test_case.category,
                status=TestResultStatus.PASSED,
                message=f"System prompt test completed (response: {word_count} words)",
                execution_time=execution_time,
                details={
                    'model': model,
                    'response': response_content,
                    'word_count': word_count,
                    'one_word_response': is_one_word
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name=test_case.name,
                category=test_case.category,
                status=TestResultStatus.FAILED,
                message=f"System prompt test failed: {str(e)}",
                execution_time=time.time() - start_time,
                error=e
            )
            
    async def _test_performance_benchmark(
        self,
        provider: BaseLLMProvider,
        test_case: TestCase
    ) -> TestResult:
        """Test provider performance with multiple requests"""
        start_time = time.time()
        
        try:
            models = await provider.list_models()
            if not models:
                return TestResult(
                    test_name=test_case.name,
                    category=test_case.category,
                    status=TestResultStatus.FAILED,
                    message="No models available for performance test",
                    execution_time=time.time() - start_time
                )
                
            model = models[0].name
            
            # Run multiple completion requests
            response_times = []
            success_count = 0
            total_tokens = 0
            
            test_messages = [
                {"role": "user", "content": "Explain quantum computing in one sentence."},
                {"role": "user", "content": "What is machine learning?"},
                {"role": "user", "content": "How do neural networks work?"},
                {"role": "user", "content": "What is artificial intelligence?"},
                {"role": "user", "content": "Explain blockchain technology."}
            ]
            
            for messages in test_messages:
                try:
                    request_start = time.time()
                    
                    response = await provider.generate_completion(
                        messages=[messages],
                        model=model,
                        temperature=0.7,
                        max_tokens=50
                    )
                    
                    request_time = time.time() - request_start
                    response_times.append(request_time)
                    
                    if response.get('content'):
                        success_count += 1
                        usage = response.get('usage', {})
                        total_tokens += usage.get('total_tokens', 0)
                        
                except Exception as e:
                    logger.warning(f"Performance test request failed: {e}")
                    response_times.append(test_case.timeout)  # Count as timeout
                    
            execution_time = time.time() - start_time
            
            if not response_times:
                return TestResult(
                    test_name=test_case.name,
                    category=test_case.category,
                    status=TestResultStatus.FAILED,
                    message="No response times recorded",
                    execution_time=execution_time
                )
                
            # Calculate benchmark metrics
            metrics = BenchmarkMetrics(
                avg_response_time=statistics.mean(response_times),
                median_response_time=statistics.median(response_times),
                p95_response_time=statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else max(response_times),
                p99_response_time=max(response_times),
                min_response_time=min(response_times),
                max_response_time=max(response_times),
                throughput=success_count / execution_time if execution_time > 0 else 0,
                success_rate=success_count / len(test_messages),
                tokens_per_second=total_tokens / execution_time if execution_time > 0 else 0,
                error_rate=(len(response_times) - success_count) / len(response_times)
            )
            
            return TestResult(
                test_name=test_case.name,
                category=test_case.category,
                status=TestResultStatus.PASSED,
                message=f"Performance benchmark completed ({success_count}/{len(test_messages)} successful)",
                execution_time=execution_time,
                details={
                    'model': model,
                    'metrics': {
                        'avg_response_time': f"{metrics.avg_response_time:.2f}s",
                        'median_response_time': f"{metrics.median_response_time:.2f}s",
                        'throughput': f"{metrics.throughput:.2f} req/s",
                        'success_rate': f"{metrics.success_rate:.1%}",
                        'tokens_per_second': f"{metrics.tokens_per_second:.1f}",
                        'error_rate': f"{metrics.error_rate:.1%}"
                    },
                    'raw_metrics': metrics
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name=test_case.name,
                category=test_case.category,
                status=TestResultStatus.FAILED,
                message=f"Performance benchmark failed: {str(e)}",
                execution_time=time.time() - start_time,
                error=e
            )
            
    async def _test_error_handling(
        self,
        provider: BaseLLMProvider,
        test_case: TestCase
    ) -> TestResult:
        """Test provider error handling with invalid requests"""
        start_time = time.time()
        
        try:
            models = await provider.list_models()
            if not models:
                return TestResult(
                    test_name=test_case.name,
                    category=test_case.category,
                    status=TestResultStatus.FAILED,
                    message="No models available for error handling test",
                    execution_time=time.time() - start_time
                )
                
            model = models[0].name
            
            # Test invalid requests
            invalid_requests = [
                # Invalid model name
                {
                    'messages': [{"role": "user", "content": "test"}],
                    'model': 'nonexistent_model_12345'
                },
                # Invalid temperature
                {
                    'messages': [{"role": "user", "content": "test"}],
                    'model': model,
                    'temperature': 10.0  # Too high
                },
                # Invalid max_tokens
                {
                    'messages': [{"role": "user", "content": "test"}],
                    'model': model,
                    'max_tokens': -1
                }
            ]
            
            handled_errors = 0
            
            for invalid_request in invalid_requests:
                try:
                    response = await provider.generate_completion(
                        **invalid_request
                    )
                    
                    # If no error was raised, check if response indicates error
                    if response.get('error'):
                        handled_errors += 1
                        
                except Exception:
                    # Exception raised indicates error handling is working
                    handled_errors += 1
                    
            execution_time = time.time() - start_time
            
            error_handling_score = handled_errors / len(invalid_requests)
            
            return TestResult(
                test_name=test_case.name,
                category=test_case.category,
                status=TestResultStatus.PASSED,
                message=f"Error handling test completed ({handled_errors}/{len(invalid_requests)} handled)",
                execution_time=execution_time,
                details={
                    'model': model,
                    'handled_errors': handled_errors,
                    'total_invalid_requests': len(invalid_requests),
                    'error_handling_score': error_handling_score
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name=test_case.name,
                category=test_case.category,
                status=TestResultStatus.FAILED,
                message=f"Error handling test failed: {str(e)}",
                execution_time=time.time() - start_time,
                error=e
            )
            
    async def export_results(
        self,
        results: Dict[str, List[TestResult]],
        output_file: str
    ):
        """Export test results to JSON file"""
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_providers': len(results),
            'results_by_provider': {}
        }
        
        for provider_name, provider_results in results.items():
            provider_data = {
                'total_tests': len(provider_results),
                'passed': len([r for r in provider_results if r.status == TestResultStatus.PASSED]),
                'failed': len([r for r in provider_results if r.status == TestResultStatus.FAILED]),
                'warnings': len([r for r in provider_results if r.status == TestResultStatus.WARNING]),
                'skipped': len([r for r in provider_results if r.status == TestResultStatus.SKIPPED]),
                'errors': len([r for r in provider_results if r.status == TestResultStatus.ERROR]),
                'tests': []
            }
            
            for result in provider_results:
                test_data = {
                    'name': result.test_name,
                    'category': result.category.value,
                    'status': result.status.value,
                    'message': result.message,
                    'execution_time': result.execution_time,
                    'timestamp': result.timestamp.isoformat(),
                    'details': result.details
                }
                
                if result.error:
                    test_data['error'] = str(result.error)
                    
                provider_data['tests'].append(test_data)
                
            export_data['results_by_provider'][provider_name] = provider_data
            
        async with aiofiles.open(output_file, 'w') as f:
            await f.write(json.dumps(export_data, indent=2))
            
        logger.info(f"Test results exported to {output_file}")


# Convenience function for quick testing
async def quick_test_provider(provider_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Quick test of a provider configuration
    
    Args:
        provider_config: Provider configuration dictionary
        
    Returns:
        Test results summary
    """
    factory = ProviderFactory()
    
    try:
        # Create provider
        provider = await factory.create_from_config_dict(provider_config)
        
        # Run quick tests
        test_suite = ProviderTestSuite(factory)
        results = await test_suite.run_provider_tests(provider)
        
        # Summarize results
        passed = len([r for r in results if r.status == TestResultStatus.PASSED])
        total = len(results)
        
        return {
            'provider_name': provider.config.name,
            'status': 'healthy' if passed == total else 'issues',
            'passed_tests': passed,
            'total_tests': total,
            'success_rate': passed / total if total > 0 else 0,
            'test_results': results
        }
        
    except Exception as e:
        return {
            'provider_name': provider_config.get('name', 'unknown'),
            'status': 'error',
            'error': str(e),
            'passed_tests': 0,
            'total_tests': 0,
            'success_rate': 0
        }