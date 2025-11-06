"""
Performance System Integration Example

Demonstrates how to use all components of the performance optimization system
together for a complete trading system performance solution.
"""

import asyncio
import logging
import time
import json
from datetime import datetime
from typing import Dict, Any, List

# Import all performance system components
from performance.redis_manager import RedisManager, CacheLevel
from performance.logging_config import (
    LoggingConfig, get_logger, LogCategory,
    log_trading_context, log_performance_context
)
from performance.metrics_collector import get_metrics_collector, initialize_metrics_collection
from performance.apm_system import get_apm_client, initialize_apm
from performance.redis_manager import initialize_redis
from performance.connection_pool import create_database_pool, get_pool_manager
from performance.memory_optimizer import get_memory_optimizer
from performance.cache_strategies import get_cache_manager, initialize_cache_manager
from performance.lazy_loading import get_lazy_loading_framework, initialize_lazy_loading
from performance.profiler import get_profiler, initialize_profiler


class TradingSystemPerformanceDemo:
    """Demonstration of integrated performance system for trading"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize all performance components
        self._setup_logging()
        self._setup_redis()
        self._setup_metrics()
        self._setup_apm()
        self._setup_connection_pool()
        self._setup_cache_system()
        self._setup_memory_optimizer()
        self._setup_lazy_loading()
        self._setup_profiler()
        
        # Performance tracking
        self.trade_count = 0
        self.performance_data = {}
    
    def _setup_logging(self):
        """Setup comprehensive logging system"""
        logging_config = LoggingConfig(
            log_level="INFO",
            log_format="json",
            log_directory="logs",
            log_file_prefix="trading_system",
            enable_structured_logging=True,
            enable_performance_logging=True,
            enable_audit_logging=True
        )
        
        from performance.logging_config import get_logging_manager
        get_logging_manager(logging_config)
        
        self.logger = get_logger("trading_demo", LogCategory.SYSTEM)
        self.logger.info("Trading performance demo initialized - Logging system setup complete")
    
    def _setup_redis(self):
        """Setup Redis caching and session management"""
        redis_config = {
            'redis': {
                'host': 'localhost',
                'port': 6379,
                'db': 0,
                'max_connections': 50,
                'timeout': 5.0
            },
            'session_ttl': 3600,
            'performance_monitoring': True
        }
        
        self.redis_manager = initialize_redis(redis_config)
        
        # Test Redis connection
        if asyncio.run(self.redis_manager.connect()):
            self.logger.info("Redis connection established successfully")
        else:
            self.logger.error("Failed to establish Redis connection")
    
    def _setup_metrics(self):
        """Setup performance metrics collection"""
        metrics_config = {
            'collection_interval': 10
        }
        
        self.metrics_collector = initialize_metrics_collection(metrics_config)
        self.metrics_collector.start_collection()
        
        self.logger.info("Metrics collection started")
    
    def _setup_apm(self):
        """Setup Application Performance Monitoring"""
        apm_config = {
            'service_name': 'trading_system',
            'sampling_rate': 1.0,
            'max_transactions': 1000,
            'auto_instrument': True
        }
        
        self.apm_client = initialize_apm(apm_config)
        
        self.logger.info("APM system initialized")
    
    def _setup_connection_pool(self):
        """Setup database and broker connection pooling"""
        try:
            # Database connection pool
            db_pool = create_database_pool(
                pool_name="main_db",
                connection_string="sqlite:///trading.db",
                max_connections=20,
                min_connections=5
            )
            
            # Broker connection pool
            broker_config = {
                'type': 'alpaca',
                'api_key': 'demo_key',
                'secret_key': 'demo_secret'
            }
            
            broker_pool = create_broker_pool(
                pool_name="alpaca_broker",
                broker_config=broker_config,
                max_connections=10,
                min_connections=2
            )
            
            self.pool_manager = get_pool_manager()
            self.logger.info("Connection pools initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to setup connection pools: {e}")
    
    def _setup_cache_system(self):
        """Setup multi-level caching system"""
        cache_config = {
            'memory_cache': {
                'max_size': 1000,
                'max_memory_mb': 100
            },
            'strategy': 'adaptive'
        }
        
        self.cache_manager = initialize_cache_manager(cache_config)
        
        self.logger.info("Cache system initialized")
    
    def _setup_memory_optimizer(self):
        """Setup memory optimization and garbage collection"""
        self.memory_optimizer = get_memory_optimizer()
        
        # Optimize for trading performance
        self.memory_optimizer.optimize_for_high_frequency_trading()
        
        self.logger.info("Memory optimizer initialized and optimized for trading")
    
    def _setup_lazy_loading(self):
        """Setup lazy loading framework"""
        lazy_config = {
            'max_workers': 4
        }
        
        self.lazy_framework = initialize_lazy_loading(lazy_config)
        
        # Register mock data loaders for demo
        self._register_demo_loaders()
        
        self.logger.info("Lazy loading framework initialized")
    
    def _setup_profiler(self):
        """Setup performance profiler and analysis"""
        profiler_config = {
            'sample_rate': 1.0,
            'max_samples': 10000,
            'bottleneck_threshold_time': 0.1
        }
        
        self.profiler = initialize_profiler(profiler_config)
        self.profiler.start_profiling()
        
        self.logger.info("Performance profiler started")
    
    def _register_demo_loaders(self):
        """Register demo data loaders for lazy loading"""
        from performance.lazy_loading import DatabaseDataLoader, APIDataLoader
        
        # Create mock database connection
        mock_connection = {"type": "sqlite", "data": []}
        
        # Register database loader
        db_loader = DatabaseDataLoader(mock_connection)
        self.lazy_framework.register_loader("database", db_loader)
        
        # Register API loader
        mock_api_client = {"get": lambda url, params: {"data": []}}
        api_loader = APIDataLoader(mock_api_client)
        self.lazy_framework.register_loader("api", api_loader)
    
    async def simulate_trading_workload(self):
        """Simulate a trading workload to demonstrate performance monitoring"""
        
        with log_trading_context(
            user_id="demo_user",
            session_id="demo_session_001"
        ) as context:
            
            self.logger.info("Starting trading workload simulation")
            
            # Simulate multiple trades
            for i in range(10):
                await self._simulate_single_trade(i)
                
                # Periodic performance checks
                if i % 5 == 0:
                    await self._check_performance_metrics()
                
                await asyncio.sleep(0.1)  # Small delay between trades
    
    async def _simulate_single_trade(self, trade_id: int):
        """Simulate processing a single trade"""
        
        with log_performance_context() as perf_context:
            
            # Start APM transaction
            trace_id = self.apm_client.start_transaction(
                f"process_trade_{trade_id}",
                tags={'trade_id': trade_id, 'user_id': 'demo_user'}
            )
            
            try:
                # 1. Market data retrieval with caching
                market_data = await self._get_market_data("AAPL")
                
                # 2. Risk assessment
                risk_score = await self._assess_risk(market_data)
                
                # 3. Trade execution with connection pooling
                execution_result = await self._execute_trade("AAPL", "BUY", 100, market_data)
                
                # 4. Cache trade result
                await self._cache_trade_result(trade_id, execution_result)
                
                # 5. Record custom metrics
                self.metrics_collector.record_custom_metric(
                    f"trade_{trade_id}_completed",
                    1,
                    metric_type="counter",
                    category="business"
                )
                
                self.trade_count += 1
                
                self.logger.info(
                    f"Trade {trade_id} completed successfully",
                    extra={
                        'category': 'trading',
                        'trade_id': trade_id,
                        'risk_score': risk_score,
                        'execution_time': execution_result.get('execution_time', 0)
                    }
                )
                
            except Exception as e:
                self.logger.error(
                    f"Trade {trade_id} failed: {e}",
                    extra={
                        'category': 'error',
                        'trade_id': trade_id,
                        'error': str(e)
                    }
                )
                
                # Record error metrics
                self.metrics_collector.record_custom_metric(
                    f"trade_{trade_id}_errors",
                    1,
                    metric_type="counter",
                    category="error"
                )
            
            finally:
                # Finish APM transaction
                self.apm_client.finish_transaction(trace_id)
    
    async def _get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get market data with caching"""
        
        # Check cache first
        cached_data = await self.redis_manager.get_cached_market_data(symbol)
        if cached_data:
            return cached_data
        
        # Simulate API call
        await asyncio.sleep(0.01)  # Simulate network latency
        
        market_data = {
            'symbol': symbol,
            'price': 150.25,
            'volume': 1000000,
            'timestamp': datetime.now().isoformat()
        }
        
        # Cache the result
        await self.redis_manager.cache_market_data(symbol, market_data, ttl=60)
        
        return market_data
    
    async def _assess_risk(self, market_data: Dict[str, Any]) -> float:
        """Assess trading risk"""
        
        # Simulate risk calculation
        await asyncio.sleep(0.005)  # Simulate computation
        
        # Simple risk score based on volume
        risk_score = min(market_data.get('volume', 0) / 100000, 1.0)
        
        return risk_score
    
    async def _execute_trade(self, symbol: str, side: str, quantity: int, 
                           market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trade with connection pooling"""
        
        start_time = time.time()
        
        try:
            # Get database connection from pool
            with self.pool_manager.get_pool("main_db").get_connection(timeout=5) as conn:
                
                # Simulate database operations
                await asyncio.sleep(0.02)  # Simulate database latency
                
                execution_result = {
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'price': market_data['price'],
                    'execution_time': time.time() - start_time,
                    'status': 'filled',
                    'timestamp': datetime.now().isoformat()
                }
                
                return execution_result
                
        except Exception as e:
            self.logger.error(f"Trade execution failed: {e}")
            return {
                'symbol': symbol,
                'status': 'failed',
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    async def _cache_trade_result(self, trade_id: int, result: Dict[str, Any]):
        """Cache trade execution result"""
        await self.redis_manager.cache_trade_execution(
            str(trade_id), result, ttl=3600
        )
    
    async def _check_performance_metrics(self):
        """Check and log performance metrics"""
        
        # Get current performance snapshot
        performance_report = self.profiler.get_performance_report()
        metrics_report = self.metrics_collector.get_performance_summary()
        cache_report = self.cache_manager.get_performance_report()
        
        # Log performance summary
        self.logger.info(
            "Performance metrics check",
            extra={
                'category': 'performance',
                'trade_count': self.trade_count,
                'profiling_active': performance_report['profiling_active'],
                'memory_usage_mb': metrics_report.get('system', {}).get('memory_usage_percent', 0),
                'cache_hit_ratio': cache_report['overall_performance']['overall_hit_ratio']
            }
        )
    
    async def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        self.logger.info("Generating comprehensive performance report")
        
        # Collect all performance reports
        performance_report = self.profiler.get_performance_report()
        metrics_report = self.metrics_collector.get_performance_summary()
        cache_report = self.cache_manager.get_performance_report()
        apm_report = self.apm_client.get_performance_report()
        memory_report = self.memory_optimizer.get_optimization_report()
        pool_stats = self.pool_manager.get_pool_statistics()
        lazy_performance = self.lazy_framework.get_performance_stats()
        
        # Combine into comprehensive report
        comprehensive_report = {
            'timestamp': datetime.now().isoformat(),
            'system_summary': {
                'total_trades_processed': self.trade_count,
                'uptime_seconds': (datetime.now() - datetime.fromisoformat(
                    metrics_report['timestamp']
                )).total_seconds() if 'timestamp' in metrics_report else 0
            },
            'performance_analysis': {
                'profiler': performance_report,
                'metrics': metrics_report,
                'cache': cache_report,
                'apm': apm_report,
                'memory': memory_report
            },
            'infrastructure': {
                'connection_pools': pool_stats,
                'lazy_loading': lazy_performance
            },
            'system_health': {
                'cpu_usage': metrics_report.get('system', {}).get('cpu_usage_percent', 0),
                'memory_usage': metrics_report.get('system', {}).get('memory_usage_percent', 0),
                'cache_hit_ratio': cache_report['overall_performance']['overall_hit_ratio'],
                'active_alerts': metrics_report.get('alerts', {}).get('active', 0)
            },
            'bottlenecks_detected': performance_report.get('bottlenecks', []),
            'recommendations': performance_report.get('recommendations', [])
        }
        
        return comprehensive_report
    
    async def cleanup(self):
        """Cleanup all performance monitoring systems"""
        self.logger.info("Cleaning up performance monitoring systems")
        
        try:
            # Stop metrics collection
            await self.metrics_collector.stop_collection()
            
            # Stop profiler
            self.profiler.stop_profiling()
            
            # Close connection pools
            self.pool_manager.close_all_pools()
            
            # Close Redis connections
            self.redis_manager.close()
            
            # Shutdown memory optimizer
            self.memory_optimizer.shutdown()
            
            # Shutdown lazy loading framework
            self.lazy_framework.shutdown()
            
            self.logger.info("Performance monitoring cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


async def main():
    """Main demonstration function"""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Performance system configuration
    config = {
        'redis': {
            'host': 'localhost',
            'port': 6379,
            'db': 0
        },
        'metrics': {
            'collection_interval': 5
        },
        'apm': {
            'service_name': 'trading_demo',
            'sampling_rate': 1.0
        },
        'cache': {
            'memory_cache': {
                'max_size': 1000,
                'max_memory_mb': 100
            }
        },
        'profiler': {
            'sample_rate': 2.0,
            'bottleneck_threshold_time': 0.05
        }
    }
    
    # Initialize and run trading system performance demo
    trading_system = TradingSystemPerformanceDemo(config)
    
    try:
        # Run trading workload simulation
        await trading_system.simulate_trading_workload()
        
        # Generate comprehensive performance report
        report = await trading_system.generate_comprehensive_report()
        
        # Save report to file
        report_filename = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n{'='*60}")
        print("PERFORMANCE OPTIMIZATION SYSTEM DEMONSTRATION COMPLETE")
        print(f"{'='*60}")
        print(f"Comprehensive performance report saved to: {report_filename}")
        print(f"Total trades processed: {report['system_summary']['total_trades_processed']}")
        print(f"System uptime: {report['system_summary']['uptime_seconds']:.1f} seconds")
        print(f"Cache hit ratio: {report['system_health']['cache_hit_ratio']:.2%}")
        print(f"Bottlenecks detected: {len(report['bottlenecks_detected'])}")
        print(f"Recommendations: {len(report['recommendations'])}")
        print(f"{'='*60}")
        
        # Display key recommendations
        if report['recommendations']:
            print("\nKey Performance Recommendations:")
            for i, rec in enumerate(report['recommendations'][:5], 1):
                print(f"{i}. {rec}")
        
    except Exception as e:
        print(f"Error in performance demonstration: {e}")
        
    finally:
        # Cleanup
        await trading_system.cleanup()


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())