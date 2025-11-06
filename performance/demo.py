#!/usr/bin/env python3
"""
Performance System Demo Script

Simple demonstration of the performance optimization system
showing key features and capabilities.
"""

import asyncio
import time
import logging
import sys
from datetime import datetime
import random

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def demo_redis_caching():
    """Demonstrate Redis caching capabilities"""
    print("\n" + "="*60)
    print("REDIS CACHING DEMONSTRATION")
    print("="*60)
    
    try:
        from performance.redis_manager import get_redis_manager
        
        # Initialize Redis manager
        redis_manager = get_redis_manager({
            'redis': {
                'host': 'localhost',
                'port': 6379,
                'db': 0
            },
            'performance_monitoring': True
        })
        
        # Test connection
        if asyncio.run(redis_manager.connect()):
            print("✅ Redis connection established")
        else:
            print("❌ Redis connection failed - skipping caching demo")
            return
        
        # Simulate market data caching
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        
        print("\n📊 Caching market data...")
        for symbol in symbols:
            market_data = {
                'symbol': symbol,
                'price': round(random.uniform(100, 500), 2),
                'volume': random.randint(1000000, 10000000),
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache the data
            asyncio.run(redis_manager.cache_market_data(symbol, market_data, ttl=60))
            print(f"  ✅ Cached {symbol}: ${market_data['price']}")
        
        print("\n🔍 Retrieving cached data...")
        for symbol in symbols:
            cached_data = asyncio.run(redis_manager.get_cached_market_data(symbol))
            if cached_data:
                print(f"  📈 {symbol}: ${cached_data['price']} (cached)")
            else:
                print(f"  ❌ {symbol}: Not found in cache")
        
        # Get performance statistics
        stats = redis_manager.get_performance_stats()
        print(f"\n📊 Cache Statistics:")
        print(f"  Hit Ratio: {stats['hit_ratio']:.2%}")
        print(f"  Total Requests: {stats['query_count']}")
        print(f"  Memory Keys: {stats['cache_levels']['memory_keys']}")
        
    except Exception as e:
        print(f"❌ Redis demo error: {e}")


def demo_logging_system():
    """Demonstrate logging system"""
    print("\n" + "="*60)
    print("LOGGING SYSTEM DEMONSTRATION")
    print("="*60)
    
    try:
        from performance.logging_config import get_logger, LogCategory
        
        # Get different category loggers
        trading_logger = get_logger("trading_system", LogCategory.TRADING)
        api_logger = get_logger("api_server", LogCategory.API)
        performance_logger = get_logger("performance_monitor", LogCategory.PERFORMANCE)
        
        print("📝 Structured logging demonstration:")
        
        # Log trading events
        trading_logger.info(
            "Trade executed",
            extra={
                'symbol': 'AAPL',
                'quantity': 100,
                'price': 150.25,
                'side': 'BUY'
            }
        )
        print("  ✅ Trading event logged")
        
        # Log API events
        api_logger.info(
            "API request processed",
            extra={
                'endpoint': '/api/v1/market/quote',
                'response_time_ms': 45.2,
                'status_code': 200
            }
        )
        print("  ✅ API event logged")
        
        # Log performance events
        performance_logger.info(
            "Performance metric recorded",
            extra={
                'cpu_usage_percent': 65.3,
                'memory_usage_mb': 1024,
                'response_time_ms': 89.1
            }
        )
        print("  ✅ Performance event logged")
        
        print("\n📊 Log files created:")
        print("  📁 logs/trading_system.log")
        print("  📁 logs/trading_system_api.log")
        print("  📁 logs/trading_system_performance.log")
        
    except Exception as e:
        print(f"❌ Logging demo error: {e}")


def demo_metrics_collection():
    """Demonstrate metrics collection"""
    print("\n" + "="*60)
    print("METRICS COLLECTION DEMONSTRATION")
    print("="*60)
    
    try:
        from performance.metrics_collector import get_metrics_collector
        
        # Initialize metrics collector
        metrics = get_metrics_collector()
        
        print("📊 Recording custom metrics:")
        
        # Record various metrics
        metrics.record_custom_metric("trades_executed", 150, "counter")
        print("  ✅ Trades executed: 150")
        
        metrics.record_custom_metric("api_response_time", 0.085, "timer")
        print("  ✅ API response time: 85ms")
        
        metrics.record_custom_metric("market_data_cache_hits", 95, "counter")
        print("  ✅ Cache hits: 95%")
        
        # Get current metrics
        current_metrics = metrics.get_latest_metrics()
        print(f"\n📈 Current metrics count: {len(current_metrics)}")
        
        # Generate performance summary
        summary = metrics.get_performance_summary()
        print(f"🖥️  System uptime: {summary['uptime_seconds']:.1f} seconds")
        print(f"💾 Active alerts: {summary['application']['active_alerts']}")
        
    except Exception as e:
        print(f"❌ Metrics demo error: {e}")


def demo_memory_optimization():
    """Demonstrate memory optimization"""
    print("\n" + "="*60)
    print("MEMORY OPTIMIZATION DEMONSTRATION")
    print("="*60)
    
    try:
        from performance.memory_optimizer import get_memory_optimizer, memory_optimized
        
        # Get memory optimizer
        memory_opt = get_memory_optimizer()
        
        print("🧠 Memory optimization demonstration:")
        
        # Get current memory usage
        current_memory = memory_opt.profiler.get_current_memory()
        if current_memory:
            print(f"  📊 Current RSS: {current_memory.rss_mb:.2f} MB")
            print(f"  📊 Heap allocated: {current_memory.heap_allocated / (1024*1024):.2f} MB")
        
        # Optimize for trading
        memory_opt.optimize_for_high_frequency_trading()
        print("  ✅ Optimized for high-frequency trading")
        
        # Create object pool
        class TradeData:
            def __init__(self):
                self.symbol = ""
                self.quantity = 0
                self.price = 0.0
            
            def reset(self):
                self.symbol = ""
                self.quantity = 0
                self.price = 0.0
        
        pool = memory_opt.create_object_pool(TradeData, max_size=10)
        print("  ✅ Created object pool for TradeData")
        
        # Test object pooling
        obj1 = pool.get()
        obj1.symbol = "AAPL"
        obj1.quantity = 100
        obj1.price = 150.25
        
        pool.return_object(obj1)
        print("  ✅ Object pooling test completed")
        
        # Get optimization report
        report = memory_opt.get_optimization_report()
        print(f"  📊 Memory recommendations: {len(report.get('recommendations', []))}")
        
    except Exception as e:
        print(f"❌ Memory optimization demo error: {e}")


def demo_performance_profiling():
    """Demonstrate performance profiling"""
    print("\n" + "="*60)
    print("PERFORMANCE PROFILING DEMONSTRATION")
    print("="*60)
    
    try:
        from performance.profiler import get_profiler, profile_function
        
        # Get profiler
        profiler = get_profiler()
        
        print("🔍 Performance profiling demonstration:")
        
        # Profile some functions
        @profile_function()
        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
        
        @profile_function()
        def quick_sort(arr):
            if len(arr) <= 1:
                return arr
            pivot = arr[len(arr) // 2]
            left = [x for x in arr if x < pivot]
            middle = [x for x in arr if x == pivot]
            right = [x for x in arr if x > pivot]
            return quick_sort(left) + middle + quick_sort(right)
        
        # Run profiled functions
        print("  🧮 Calculating Fibonacci(10)...")
        result1 = fibonacci(10)
        print(f"    Result: {result1}")
        
        print("  📊 Sorting random array...")
        random_array = [random.randint(1, 1000) for _ in range(100)]
        result2 = quick_sort(random_array)
        print(f"    Sorted {len(random_array)} elements")
        
        # Generate performance report
        report = profiler.get_performance_report()
        print(f"\n📈 Profiling Results:")
        print(f"  🔢 Total functions profiled: {report['summary']['total_functions']}")
        print(f"  ⏱️  Total execution time: {report['summary']['total_time']:.4f}s")
        print(f"  📊 Bottlenecks detected: {report['summary']['bottlenecks_detected']}")
        
        # Show top functions
        if report['function_analysis']:
            print(f"\n🏆 Top 3 functions by execution time:")
            for i, func in enumerate(report['function_analysis'][:3], 1):
                print(f"  {i}. {func['function']}: {func['total_time']:.4f}s")
        
    except Exception as e:
        print(f"❌ Profiling demo error: {e}")


def demo_cache_strategies():
    """Demonstrate cache strategies"""
    print("\n" + "="*60)
    print("CACHE STRATEGIES DEMONSTRATION")
    print("="*60)
    
    try:
        from performance.cache_strategies import get_cache_manager, cached
        
        # Get cache manager
        cache = get_cache_manager()
        
        print("💾 Cache strategies demonstration:")
        
        # Demo cached function
        @cached(cache, key_prefix="fibonacci", ttl=300)
        def slow_fibonacci(n):
            time.sleep(0.01)  # Simulate slow calculation
            if n <= 1:
                return n
            return slow_fibonacci(n-1) + slow_fibonacci(n-2)
        
        print("  🧮 First Fibonacci(20) call (will cache result)...")
        start_time = time.time()
        result1 = slow_fibonacci(20)
        first_call_time = time.time() - start_time
        print(f"    Result: {result1}, Time: {first_call_time:.4f}s")
        
        print("  🔄 Second Fibonacci(20) call (from cache)...")
        start_time = time.time()
        result2 = slow_fibonacci(20)
        second_call_time = time.time() - start_time
        print(f"    Result: {result2}, Time: {second_call_time:.4f}s")
        
        speedup = first_call_time / max(second_call_time, 0.001)
        print(f"  🚀 Speedup from caching: {speedup:.1f}x")
        
        # Get cache performance report
        cache_report = cache.get_performance_report()
        print(f"\n📊 Cache Performance:")
        print(f"  🎯 Overall hit ratio: {cache_report['overall_performance']['overall_hit_ratio']:.2%}")
        print(f"  📈 Total operations: {cache_report['overall_performance']['total_operations']}")
        
    except Exception as e:
        print(f"❌ Cache demo error: {e}")


def main():
    """Main demonstration function"""
    print("🚀 PERFORMANCE OPTIMIZATION SYSTEM DEMO")
    print("=" * 60)
    print("Demonstrating key features of the performance system")
    print("=" * 60)
    
    # Run all demonstrations
    demo_logging_system()
    demo_metrics_collection()
    demo_memory_optimization()
    demo_performance_profiling()
    demo_cache_strategies()
    demo_redis_caching()
    
    print("\n" + "="*60)
    print("✅ DEMONSTRATION COMPLETE")
    print("="*60)
    print("\n🎯 Key Features Demonstrated:")
    print("  ✅ Structured JSON logging with categories")
    print("  ✅ Real-time metrics collection")
    print("  ✅ Memory optimization and object pooling")
    print("  ✅ Function performance profiling")
    print("  ✅ Multi-level caching strategies")
    print("  ✅ Redis integration for high-performance caching")
    
    print("\n📚 Next Steps:")
    print("  1. Review the comprehensive README.md")
    print("  2. Check the integration_example.py for full system integration")
    print("  3. Configure performance monitoring for your specific use case")
    print("  4. Set up alerts and monitoring dashboards")
    print("  5. Optimize for your specific trading requirements")
    
    print("\n💡 For production use:")
    print("  - Configure Redis connection in redis_manager.py")
    print("  - Set up proper database connections")
    print("  - Configure alert thresholds based on your SLAs")
    print("  - Enable performance profiling for critical paths")
    print("  - Set up log rotation and archival")
    
    print("\n🔧 Configuration Tips:")
    print("  - Start with conservative settings and optimize based on metrics")
    print("  - Monitor system performance in staging environment first")
    print("  - Use the performance profiler to identify bottlenecks")
    print("  - Implement caching strategies based on access patterns")
    print("  - Set up automated performance testing")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("\n💡 Make sure all dependencies are installed:")
        print("  pip install redis asyncio aioredis psutil")
        sys.exit(1)