"""
Performance Test Suite
Load testing, stress testing, and performance optimization validation.
"""

import unittest
import asyncio
import logging
import json
import time
import statistics
import concurrent.futures
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import aiohttp
import websockets
from datetime import datetime, timedelta
import psutil
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoadTestType(Enum):
    """Load test type enumeration."""
    RAMP_UP = "ramp_up"
    STEADY_STATE = "steady_state"
    SPIKE = "spike"
    STRESS = "stress"
    VOLUME = "volume"


class PerformanceMetric(Enum):
    """Performance metric enumeration."""
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"


@dataclass
class PerformanceTestConfig:
    """Performance test configuration."""
    test_name: str
    test_type: LoadTestType
    target_url: str
    concurrent_users: int
    test_duration: int  # seconds
    ramp_up_time: int = 0
    think_time: float = 0  # seconds between requests
    max_response_time: float = 5.0  # seconds
    max_error_rate: float = 5.0  # percentage
    payload_size: Optional[int] = None
    http_method: str = "GET"
    headers: Dict[str, str] = None
    body: Any = None


@dataclass
class PerformanceTestResult:
    """Performance test result container."""
    test_name: str
    test_type: LoadTestType
    duration: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    response_times: List[float]
    throughput: float
    error_rate: float
    percentile_95: float
    percentile_99: float
    cpu_usage_avg: float
    memory_usage_avg: float
    timestamp: datetime = None


@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_percent: float
    memory_percent: float
    disk_io_read: float
    disk_io_write: float
    network_bytes_sent: float
    network_bytes_recv: float
    timestamp: datetime


class PerformanceTestSuite:
    """Performance testing suite for system optimization."""
    
    def __init__(self):
        self.test_configs: List[PerformanceTestConfig] = []
        self.test_results: List[PerformanceTestResult] = []
        self.monitoring_active = False
        self.system_metrics: List[SystemMetrics] = []
        self.monitoring_thread = None
        self.stop_monitoring = False
        
        # Performance thresholds
        self.performance_thresholds = {
            "response_time_p95": 2.0,  # seconds
            "response_time_p99": 5.0,  # seconds
            "throughput_min": 100,     # requests/second
            "error_rate_max": 5.0,     # percentage
            "cpu_usage_max": 80.0,     # percentage
            "memory_usage_max": 80.0   # percentage
        }
    
    def add_test_config(self, config: PerformanceTestConfig):
        """Add a performance test configuration."""
        self.test_configs.append(config)
        logger.info(f"Added performance test config: {config.test_name}")
    
    async def run_performance_test(self, config: PerformanceTestConfig) -> PerformanceTestResult:
        """Run a specific performance test."""
        logger.info(f"Starting performance test: {config.test_name}")
        
        start_time = time.time()
        response_times = []
        successful_requests = 0
        failed_requests = 0
        request_count = 0
        
        # Start system monitoring
        self._start_system_monitoring()
        
        try:
            if config.test_type == LoadTestType.RAMP_UP:
                result = await self._run_ramp_up_test(config)
            elif config.test_type == LoadTestType.STEADY_STATE:
                result = await self._run_steady_state_test(config)
            elif config.test_type == LoadTestType.SPIKE:
                result = await self._run_spike_test(config)
            elif config.test_type == LoadTestType.STRESS:
                result = await self._run_stress_test(config)
            elif config.test_type == LoadTestType.VOLUME:
                result = await self._run_volume_test(config)
            else:
                raise ValueError(f"Unknown test type: {config.test_type}")
            
            return result
            
        except Exception as e:
            logger.error(f"Performance test {config.test_name} failed: {str(e)}")
            return PerformanceTestResult(
                test_name=config.test_name,
                test_type=config.test_type,
                duration=0,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                response_times=[],
                throughput=0,
                error_rate=100,
                percentile_95=0,
                percentile_99=0,
                cpu_usage_avg=0,
                memory_usage_avg=0
            )
        
        finally:
            # Stop system monitoring
            self._stop_system_monitoring()
    
    async def _run_ramp_up_test(self, config: PerformanceTestConfig) -> PerformanceTestResult:
        """Run ramp-up load test."""
        start_time = time.time()
        response_times = []
        successful_requests = 0
        failed_requests = 0
        request_count = 0
        
        # Calculate ramp-up steps
        ramp_steps = config.concurrent_users
        step_duration = config.ramp_up_time / ramp_steps if config.ramp_up_time > 0 else 1
        
        active_users = 0
        user_tasks = []
        
        for step in range(ramp_steps):
            # Add users gradually
            active_users = min(step + 1, config.concurrent_users)
            logger.info(f"Ramp-up step {step + 1}: {active_users} users")
            
            # Start user sessions for new users
            for _ in range(min(1, config.concurrent_users - len(user_tasks))):
                task = asyncio.create_task(self._run_user_session(config))
                user_tasks.append(task)
            
            # Run for step duration
            step_end_time = time.time() + step_duration
            while time.time() < step_end_time:
                await asyncio.sleep(0.1)
                if user_tasks:
                    done_tasks = [t for t in user_tasks if t.done()]
                    for task in done_tasks:
                        try:
                            result = task.result()
                            if result:
                                response_times.append(result['response_time'])
                                if result['success']:
                                    successful_requests += 1
                                else:
                                    failed_requests += 1
                            request_count += 1
                        except Exception as e:
                            logger.error(f"User session error: {str(e)}")
                            failed_requests += 1
                        user_tasks.remove(task)
        
        # Wait for remaining tasks
        if user_tasks:
            done_tasks = await asyncio.gather(*user_tasks, return_exceptions=True)
            for task in done_tasks:
                try:
                    result = task.result()
                    if result:
                        response_times.append(result['response_time'])
                        if result['success']:
                            successful_requests += 1
                        else:
                            failed_requests += 1
                    request_count += 1
                except Exception as e:
                    logger.error(f"User session error: {str(e)}")
                    failed_requests += 1
        
        duration = time.time() - start_time
        
        return self._create_performance_result(
            config, duration, request_count, successful_requests, 
            failed_requests, response_times
        )
    
    async def _run_steady_state_test(self, config: PerformanceTestConfig) -> PerformanceTestResult:
        """Run steady-state load test."""
        start_time = time.time()
        response_times = []
        successful_requests = 0
        failed_requests = 0
        request_count = 0
        
        # Create user sessions
        user_sessions = []
        for _ in range(config.concurrent_users):
            session = asyncio.create_task(self._run_user_session(config, duration=config.test_duration))
            user_sessions.append(session)
        
        # Run all sessions
        results = await asyncio.gather(*user_sessions, return_exceptions=True)
        
        # Process results
        for result in results:
            try:
                if isinstance(result, dict):
                    if result['success']:
                        successful_requests += 1
                        response_times.append(result['response_time'])
                    else:
                        failed_requests += 1
                    request_count += 1
                elif isinstance(result, Exception):
                    failed_requests += 1
                    logger.error(f"User session error: {str(result)}")
            except Exception as e:
                logger.error(f"Error processing session result: {str(e)}")
        
        duration = time.time() - start_time
        
        return self._create_performance_result(
            config, duration, request_count, successful_requests, 
            failed_requests, response_times
        )
    
    async def _run_spike_test(self, config: PerformanceTestConfig) -> PerformanceTestResult:
        """Run spike load test."""
        start_time = time.time()
        response_times = []
        successful_requests = 0
        failed_requests = 0
        request_count = 0
        
        # Baseline period
        logger.info("Starting baseline period...")
        baseline_duration = config.test_duration * 0.2
        baseline_end = time.time() + baseline_duration
        
        # Spike period
        logger.info("Starting spike period...")
        spike_users = config.concurrent_users * 3  # Triple the load
        spike_duration = config.test_duration * 0.6
        spike_end = time.time() + spike_duration
        
        # Cool-down period
        logger.info("Starting cool-down period...")
        cooldown_end = time.time() + config.test_duration
        
        current_time = time.time()
        
        while current_time < cooldown_end:
            if current_time < baseline_end:
                # Baseline period
                current_users = config.concurrent_users // 2
            elif current_time < spike_end:
                # Spike period
                current_users = spike_users
            else:
                # Cool-down period
                current_users = config.concurrent_users // 3
            
            # Run requests with current user load
            tasks = []
            for _ in range(current_users):
                task = asyncio.create_task(self._run_single_request(config))
                tasks.append(task)
            
            # Wait for tasks with think time
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    try:
                        if isinstance(result, dict):
                            if result['success']:
                                successful_requests += 1
                                response_times.append(result['response_time'])
                            else:
                                failed_requests += 1
                            request_count += 1
                    except Exception as e:
                        logger.error(f"Request error: {str(e)}")
                        failed_requests += 1
            
            # Think time
            if config.think_time > 0:
                await asyncio.sleep(config.think_time)
            
            current_time = time.time()
        
        duration = time.time() - start_time
        
        return self._create_performance_result(
            config, duration, request_count, successful_requests, 
            failed_requests, response_times
        )
    
    async def _run_stress_test(self, config: PerformanceTestConfig) -> PerformanceTestResult:
        """Run stress test to find breaking point."""
        start_time = time.time()
        response_times = []
        successful_requests = 0
        failed_requests = 0
        request_count = 0
        
        # Start with moderate load and increase
        max_users = config.concurrent_users * 5
        step_increase = config.concurrent_users
        step_duration = config.test_duration // (max_users // step_increase)
        
        current_users = config.concurrent_users
        test_end = time.time() + config.test_duration
        
        while time.time() < test_end:
            logger.info(f"Stress test - current users: {current_users}")
            
            # Create user sessions
            tasks = []
            for _ in range(current_users):
                task = asyncio.create_task(self._run_single_request(config))
                tasks.append(task)
            
            # Run for step duration
            step_end = time.time() + step_duration
            batch_start = time.time()
            
            while time.time() < step_end and time.time() < test_end:
                if tasks:
                    results = await asyncio.gather(*tasks[:10], return_exceptions=True)  # Process in batches
                    
                    for result in results:
                        try:
                            if isinstance(result, dict):
                                if result['success']:
                                    successful_requests += 1
                                    response_times.append(result['response_time'])
                                else:
                                    failed_requests += 1
                                request_count += 1
                        except Exception as e:
                            logger.error(f"Request error: {str(e)}")
                            failed_requests += 1
                    
                    # Remove processed tasks
                    tasks = tasks[10:]
                
                # Small delay between batches
                await asyncio.sleep(0.1)
            
            # Increase load
            current_users = min(current_users + step_increase, max_users)
        
        duration = time.time() - start_time
        
        return self._create_performance_result(
            config, duration, request_count, successful_requests, 
            failed_requests, response_times
        )
    
    async def _run_volume_test(self, config: PerformanceTestConfig) -> PerformanceTestResult:
        """Run volume test with large payloads."""
        # Override payload size for volume test
        if config.payload_size is None:
            config.payload_size = 1024 * 1024  # 1MB default
        
        start_time = time.time()
        response_times = []
        successful_requests = 0
        failed_requests = 0
        request_count = 0
        
        # Generate large payload
        large_payload = "x" * config.payload_size
        
        test_end = time.time() + config.test_duration
        
        while time.time() < test_end:
            # Create POST request with large payload
            config.body = large_payload
            config.http_method = "POST"
            
            result = await self._run_single_request(config)
            
            if result:
                if result['success']:
                    successful_requests += 1
                    response_times.append(result['response_time'])
                else:
                    failed_requests += 1
                request_count += 1
            
            # Think time
            if config.think_time > 0:
                await asyncio.sleep(config.think_time)
        
        duration = time.time() - start_time
        
        return self._create_performance_result(
            config, duration, request_count, successful_requests, 
            failed_requests, response_times
        )
    
    async def _run_user_session(self, config: PerformanceTestConfig, 
                               duration: int = None) -> Optional[Dict[str, Any]]:
        """Run a single user session."""
        session_duration = duration or config.test_duration
        session_end = time.time() + session_duration
        
        session_results = []
        
        while time.time() < session_end:
            result = await self._run_single_request(config)
            
            if result:
                session_results.append(result)
            
            # Think time between requests
            if config.think_time > 0:
                await asyncio.sleep(config.think_time)
        
        # Return session summary
        total_requests = len(session_results)
        successful_requests = sum(1 for r in session_results if r['success'])
        
        return {
            'success': successful_requests > total_requests * 0.8,  # 80% success rate
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'response_times': [r['response_time'] for r in session_results if r['success']]
        }
    
    async def _run_single_request(self, config: PerformanceTestConfig) -> Optional[Dict[str, Any]]:
        """Run a single HTTP request."""
        try:
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                kwargs = {
                    'url': config.target_url,
                    'headers': config.headers or {},
                    'timeout': aiohttp.ClientTimeout(total=30)
                }
                
                if config.http_method.upper() == "GET":
                    async with session.get(**kwargs) as response:
                        await response.text()
                        response_time = time.time() - start_time
                        return {
                            'success': response.status < 400,
                            'status_code': response.status,
                            'response_time': response_time
                        }
                
                elif config.http_method.upper() == "POST":
                    kwargs['json'] = config.body
                    async with session.post(**kwargs) as response:
                        await response.text()
                        response_time = time.time() - start_time
                        return {
                            'success': response.status < 400,
                            'status_code': response.status,
                            'response_time': response_time
                        }
                
                elif config.http_method.upper() == "PUT":
                    kwargs['json'] = config.body
                    async with session.put(**kwargs) as response:
                        await response.text()
                        response_time = time.time() - start_time
                        return {
                            'success': response.status < 400,
                            'status_code': response.status,
                            'response_time': response_time
                        }
                
                elif config.http_method.upper() == "DELETE":
                    async with session.delete(**kwargs) as response:
                        await response.text()
                        response_time = time.time() - start_time
                        return {
                            'success': response.status < 400,
                            'status_code': response.status,
                            'response_time': response_time
                        }
        
        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            return {
                'success': False,
                'error': 'timeout',
                'response_time': response_time
            }
        
        except Exception as e:
            response_time = time.time() - start_time
            return {
                'success': False,
                'error': str(e),
                'response_time': response_time
            }
    
    def _create_performance_result(self, config: PerformanceTestConfig, duration: float,
                                 total_requests: int, successful_requests: int,
                                 failed_requests: int, response_times: List[float]) -> PerformanceTestResult:
        """Create performance test result."""
        error_rate = (failed_requests / total_requests * 100) if total_requests > 0 else 0
        throughput = total_requests / duration if duration > 0 else 0
        
        # Calculate percentiles
        if response_times:
            percentile_95 = statistics.quantile(response_times, 0.95)
            percentile_99 = statistics.quantile(response_times, 0.99)
        else:
            percentile_95 = percentile_99 = 0
        
        # Calculate average system metrics
        if self.system_metrics:
            cpu_avg = statistics.mean([m.cpu_percent for m in self.system_metrics])
            memory_avg = statistics.mean([m.memory_percent for m in self.system_metrics])
        else:
            cpu_avg = memory_avg = 0
        
        return PerformanceTestResult(
            test_name=config.test_name,
            test_type=config.test_type,
            duration=duration,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            response_times=response_times,
            throughput=throughput,
            error_rate=error_rate,
            percentile_95=percentile_95,
            percentile_99=percentile_99,
            cpu_usage_avg=cpu_avg,
            memory_usage_avg=memory_avg,
            timestamp=datetime.now()
        )
    
    def _start_system_monitoring(self):
        """Start system performance monitoring."""
        self.monitoring_active = True
        self.system_metrics = []
        self.stop_monitoring = False
        
        self.monitoring_thread = threading.Thread(target=self._monitor_system_resources)
        self.monitoring_thread.start()
        
        logger.info("System monitoring started")
    
    def _stop_system_monitoring(self):
        """Stop system performance monitoring."""
        self.monitoring_active = False
        self.stop_monitoring = True
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        logger.info("System monitoring stopped")
    
    def _monitor_system_resources(self):
        """Monitor system resources in background thread."""
        process = psutil.Process()
        
        while self.monitoring_active and not self.stop_monitoring:
            try:
                # Get system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk_io = psutil.disk_io_counters()
                network_io = psutil.net_io_counters()
                
                # Get process-specific metrics
                process_cpu = process.cpu_percent()
                process_memory = process.memory_percent()
                
                metrics = SystemMetrics(
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    disk_io_read=disk_io.read_bytes if disk_io else 0,
                    disk_io_write=disk_io.write_bytes if disk_io else 0,
                    network_bytes_sent=network_io.bytes_sent if network_io else 0,
                    network_bytes_recv=network_io.bytes_recv if network_io else 0,
                    timestamp=datetime.now()
                )
                
                self.system_metrics.append(metrics)
                
                # Keep only recent metrics (last 1000 entries)
                if len(self.system_metrics) > 1000:
                    self.system_metrics = self.system_metrics[-1000:]
                
                time.sleep(1)  # Monitor every second
                
            except Exception as e:
                logger.error(f"System monitoring error: {str(e)}")
                time.sleep(1)
    
    async def run_all_performance_tests(self) -> Dict[str, Any]:
        """Run all configured performance tests."""
        logger.info("Starting comprehensive performance testing...")
        
        overall_start_time = time.time()
        results = []
        
        for config in self.test_configs:
            logger.info(f"Running performance test: {config.test_name}")
            result = await self.run_performance_test(config)
            results.append(result)
            self.test_results.append(result)
            
            # Brief pause between tests
            await asyncio.sleep(2)
        
        overall_duration = time.time() - overall_start_time
        
        # Generate comprehensive report
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_duration": overall_duration,
            "total_tests": len(self.test_configs),
            "test_results": [self._result_to_dict(r) for r in results],
            "performance_summary": self._generate_performance_summary(results),
            "recommendations": self._generate_performance_recommendations(results)
        }
        
        logger.info(f"Performance testing completed in {overall_duration:.2f} seconds")
        return report
    
    def _result_to_dict(self, result: PerformanceTestResult) -> Dict[str, Any]:
        """Convert PerformanceTestResult to dictionary."""
        return {
            "test_name": result.test_name,
            "test_type": result.test_type.value,
            "duration": result.duration,
            "total_requests": result.total_requests,
            "successful_requests": result.successful_requests,
            "failed_requests": result.failed_requests,
            "throughput": result.throughput,
            "error_rate": result.error_rate,
            "response_time_p95": result.percentile_95,
            "response_time_p99": result.percentile_99,
            "cpu_usage_avg": result.cpu_usage_avg,
            "memory_usage_avg": result.memory_usage_avg,
            "timestamp": result.timestamp.isoformat() if result.timestamp else None
        }
    
    def _generate_performance_summary(self, results: List[PerformanceTestResult]) -> Dict[str, Any]:
        """Generate performance summary."""
        if not results:
            return {}
        
        avg_throughput = statistics.mean([r.throughput for r in results])
        avg_error_rate = statistics.mean([r.error_rate for r in results])
        avg_cpu = statistics.mean([r.cpu_usage_avg for r in results])
        avg_memory = statistics.mean([r.memory_usage_avg for r in results])
        
        # Check against thresholds
        threshold_violations = []
        
        for result in results:
            if result.percentile_95 > self.performance_thresholds["response_time_p95"]:
                threshold_violations.append(f"Response time P95 ({result.percentile_95:.2f}s) exceeds threshold for {result.test_name}")
            
            if result.error_rate > self.performance_thresholds["error_rate_max"]:
                threshold_violations.append(f"Error rate ({result.error_rate:.1f}%) exceeds threshold for {result.test_name}")
        
        return {
            "average_throughput": avg_throughput,
            "average_error_rate": avg_error_rate,
            "average_cpu_usage": avg_cpu,
            "average_memory_usage": avg_memory,
            "threshold_violations": threshold_violations,
            "performance_score": self._calculate_performance_score(results)
        }
    
    def _generate_performance_recommendations(self, results: List[PerformanceTestResult]) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Analyze results and generate recommendations
        for result in results:
            if result.error_rate > self.performance_thresholds["error_rate_max"]:
                recommendations.append(f"High error rate ({result.error_rate:.1f}%) in {result.test_name}. Check system stability and error handling.")
            
            if result.percentile_99 > self.performance_thresholds["response_time_p99"]:
                recommendations.append(f"High response time P99 ({result.percentile_99:.2f}s) in {result.test_name}. Consider performance optimization.")
            
            if result.cpu_usage_avg > self.performance_thresholds["cpu_usage_max"]:
                recommendations.append(f"High CPU usage ({result.cpu_usage_avg:.1f}%) in {result.test_name}. Consider scaling or optimization.")
            
            if result.memory_usage_avg > self.performance_thresholds["memory_usage_max"]:
                recommendations.append(f"High memory usage ({result.memory_usage_avg:.1f}%) in {result.test_name}. Check for memory leaks.")
        
        if not recommendations:
            recommendations.append("System performance is within acceptable limits.")
        
        return recommendations
    
    def _calculate_performance_score(self, results: List[PerformanceTestResult]) -> float:
        """Calculate overall performance score (0-100)."""
        if not results:
            return 0
        
        score = 100
        
        for result in results:
            # Deduct for high error rates
            if result.error_rate > 0:
                score -= min(result.error_rate * 2, 20)  # Max 20 points for errors
            
            # Deduct for slow response times
            if result.percentile_95 > self.performance_thresholds["response_time_p95"]:
                score -= min((result.percentile_95 - self.performance_thresholds["response_time_p95"]) * 10, 15)
            
            # Deduct for high resource usage
            if result.cpu_usage_avg > self.performance_thresholds["cpu_usage_max"]:
                score -= min((result.cpu_usage_avg - self.performance_thresholds["cpu_usage_max"]) * 0.5, 10)
        
        return max(0, score)
    
    def get_test_status(self) -> Dict[str, Any]:
        """Get current performance test status."""
        return {
            "total_configs": len(self.test_configs),
            "completed_tests": len(self.test_results),
            "monitoring_active": self.monitoring_active,
            "system_metrics_collected": len(self.system_metrics)
        }


# Global performance test suite instance
performance_test_suite = PerformanceTestSuite()