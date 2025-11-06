"""Debug Mode Launcher

Enhanced debugging capabilities for the trading orchestrator system.
Provides interactive debugging, component inspection, log analysis,
and performance profiling tools.

Features:
- Interactive component inspection
- Real-time log monitoring
- Performance profiling
- System state visualization
- Error diagnosis and resolution
- Configuration validation
- Database debugging tools

Usage:
    python debug.py [--component COMPONENT] [--level LEVEL] [--output FILE]
    python debug.py --interactive
    python debug.py --monitor-logs
    python debug.py --profile-performance

Author: Trading System Development Team
Version: 1.0.0
Date: 2024-12-19
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from contextlib import asynccontextmanager
import signal
import traceback
from dataclasses import dataclass, asdict
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import psutil
import numpy as np

# Add the parent directory to the path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from trading_orchestrator.config import TradingConfig
    from trading_orchestrator.database import DatabaseManager
    from trading_orchestrator.brokers import BrokerManager
    from trading_orchestrator.strategies import StrategyManager
except ImportError as e:
    print(f"Warning: Some modules not available for import: {e}")
    print("Running in limited debug mode")

# Configure logging for debug mode
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

@dataclass
class DebugMetrics:
    """Debug metrics data structure"""
    timestamp: datetime
    component: str
    operation: str
    duration: float
    success: bool
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
class Debugger:
    """Main debug interface and coordinator"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = None
        self.database_manager = None
        self.broker_manager = None
        self.strategy_manager = None
        self.debug_metrics: List[DebugMetrics] = []
        self.performance_profile: Dict[str, Any] = {}
        self.is_running = False
        self.log_monitors: Dict[str, Any] = {}
        
    async def initialize(self):
        """Initialize debug environment"""
        logger.info("Initializing debug environment...")
        
        try:
            if self.config_path and Path(self.config_path).exists():
                self.config = TradingConfig.from_file(self.config_path)
                logger.info(f"Loaded configuration from {self.config_path}")
            else:
                logger.warning("No valid config path provided, using default configuration")
                self.config = self._get_default_config()
                
            # Initialize components if available
            await self._initialize_components()
            
            # Set up performance monitoring
            await self._setup_performance_monitoring()
            
            # Initialize log monitoring
            await self._setup_log_monitoring()
            
            logger.info("Debug environment initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize debug environment: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def _get_default_config(self) -> TradingConfig:
        """Get default configuration for debugging"""
        # Create a minimal config for debugging
        config_data = {
            'database': {
                'path': 'debug_trading.db',
                'backup_enabled': False
            },
            'brokers': {
                'enabled': []
            },
            'strategies': {
                'enabled': []
            },
            'system': {
                'debug_mode': True,
                'log_level': 'DEBUG'
            }
        }
        return TradingConfig(**config_data)
    
    async def _initialize_components(self):
        """Initialize trading components"""
        try:
            # Initialize database manager
            if hasattr(self, 'DatabaseManager'):
                self.database_manager = DatabaseManager(self.config.database)
                await self.database_manager.initialize()
                logger.info("Database manager initialized")
        except Exception as e:
            logger.warning(f"Could not initialize components: {e}")
    
    async def _setup_performance_monitoring(self):
        """Set up performance monitoring"""
        self.performance_profile = {
            'start_time': datetime.utcnow(),
            'operations': {},
            'system_metrics': {},
            'memory_usage': [],
            'cpu_usage': []
        }
        
        # Start monitoring system resources
        asyncio.create_task(self._monitor_system_resources())
    
    async def _setup_log_monitoring(self):
        """Set up log monitoring"""
        # Set up log file watchers
        log_files = [
            'debug.log',
            'trading.log',
            'error.log'
        ]
        
        for log_file in log_files:
            if Path(log_file).exists():
                self.log_monitors[log_file] = {
                    'path': log_file,
                    'last_position': 0,
                    'errors': [],
                    'warnings': []
                }
    
    async def _monitor_system_resources(self):
        """Monitor system resources continuously"""
        while self.is_running:
            try:
                # Get current system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Store metrics
                timestamp = datetime.utcnow()
                self.performance_profile['system_metrics'][timestamp] = {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_used_mb': memory.used // (1024 * 1024),
                    'disk_percent': (disk.used / disk.total) * 100,
                    'disk_free_gb': disk.free // (1024 * 1024 * 1024)
                }
                
                # Keep only last 100 entries
                if len(self.performance_profile['system_metrics']) > 100:
                    # Remove oldest entries
                    keys = list(self.performance_profile['system_metrics'].keys())
                    for key in keys[:-100]:
                        del self.performance_profile['system_metrics'][key]
                
                await asyncio.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring system resources: {e}")
                await asyncio.sleep(10)
    
    @asynccontextmanager
    async def measure_operation(self, component: str, operation: str):
        """Context manager to measure operation performance"""
        start_time = time.time()
        start_timestamp = datetime.utcnow()
        
        try:
            yield
            success = True
            error_message = None
        except Exception as e:
            success = False
            error_message = str(e)
            raise
        finally:
            duration = time.time() - start_time
            end_timestamp = datetime.utcnow()
            
            # Record metrics
            metric = DebugMetrics(
                timestamp=end_timestamp,
                component=component,
                operation=operation,
                duration=duration,
                success=success,
                error_message=error_message
            )
            
            self.debug_metrics.append(metric)
            
            # Update performance profile
            if component not in self.performance_profile['operations']:
                self.performance_profile['operations'][component] = []
            
            self.performance_profile['operations'][component].append({
                'operation': operation,
                'start_time': start_timestamp,
                'end_time': end_timestamp,
                'duration': duration,
                'success': success,
                'error': error_message
            })
    
    async def inspect_component(self, component_name: str) -> Dict[str, Any]:
        """Inspect a specific component's state"""
        result = {
            'component': component_name,
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'unknown',
            'details': {}
        }
        
        try:
            async with self.measure_operation('debugger', 'inspect_component'):
                if component_name == 'database' and self.database_manager:
                    result = await self._inspect_database()
                elif component_name == 'broker' and self.broker_manager:
                    result = await self._inspect_broker()
                elif component_name == 'strategy' and self.strategy_manager:
                    result = await self._inspect_strategy()
                elif component_name == 'config':
                    result = await self._inspect_config()
                elif component_name == 'performance':
                    result = await self._inspect_performance()
                else:
                    result['status'] = 'not_available'
                    result['details'] = {'error': f'Component {component_name} not available'}
                    
        except Exception as e:
            result['status'] = 'error'
            result['details'] = {'error': str(e), 'traceback': traceback.format_exc()}
        
        return result
    
    async def _inspect_database(self) -> Dict[str, Any]:
        """Inspect database component"""
        if not self.database_manager:
            return {'status': 'not_initialized', 'details': {}}
        
        try:
            # Check database connectivity
            async with self.database_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get table info
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                # Get database size
                cursor.execute("PRAGMA page_count")
                page_count = cursor.fetchone()[0]
                cursor.execute("PRAGMA page_size")
                page_size = cursor.fetchone()[0]
                db_size = (page_count * page_size) / (1024 * 1024)  # MB
                
                return {
                    'status': 'healthy',
                    'details': {
                        'tables': tables,
                        'database_size_mb': round(db_size, 2),
                        'total_tables': len(tables)
                    }
                }
        except Exception as e:
            return {
                'status': 'error',
                'details': {'error': str(e)}
            }
    
    async def _inspect_broker(self) -> Dict[str, Any]:
        """Inspect broker component"""
        if not self.broker_manager:
            return {'status': 'not_initialized', 'details': {}}
        
        try:
            # Get broker status
            broker_status = {}
            for broker_name, broker in self.broker_manager.brokers.items():
                broker_status[broker_name] = {
                    'connected': broker.is_connected(),
                    'status': 'active' if broker.is_connected() else 'disconnected'
                }
            
            return {
                'status': 'healthy',
                'details': {
                    'brokers': broker_status,
                    'total_brokers': len(self.broker_manager.brokers),
                    'connected_brokers': sum(1 for b in broker_status.values() if b['connected'])
                }
            }
        except Exception as e:
            return {
                'status': 'error',
                'details': {'error': str(e)}
            }
    
    async def _inspect_strategy(self) -> Dict[str, Any]:
        """Inspect strategy component"""
        if not self.strategy_manager:
            return {'status': 'not_initialized', 'details': {}}
        
        try:
            # Get strategy status
            strategies_status = {}
            for strategy_id, strategy in self.strategy_manager.strategies.items():
                strategies_status[strategy_id] = {
                    'name': strategy.name,
                    'status': str(strategy.status),
                    'enabled': strategy.is_enabled()
                }
            
            return {
                'status': 'healthy',
                'details': {
                    'strategies': strategies_status,
                    'total_strategies': len(self.strategy_manager.strategies),
                    'enabled_strategies': sum(1 for s in strategies_status.values() if s['enabled'])
                }
            }
        except Exception as e:
            return {
                'status': 'error',
                'details': {'error': str(e)}
            }
    
    async def _inspect_config(self) -> Dict[str, Any]:
        """Inspect configuration"""
        try:
            config_dict = asdict(self.config) if self.config else {}
            return {
                'status': 'healthy',
                'details': {
                    'config_path': self.config_path,
                    'has_config': self.config is not None,
                    'config_keys': list(config_dict.keys()) if config_dict else []
                }
            }
        except Exception as e:
            return {
                'status': 'error',
                'details': {'error': str(e)}
            }
    
    async def _inspect_performance(self) -> Dict[str, Any]:
        """Inspect performance metrics"""
        try:
            # Calculate performance statistics
            total_operations = len(self.debug_metrics)
            successful_operations = sum(1 for m in self.debug_metrics if m.success)
            failed_operations = total_operations - successful_operations
            
            # Calculate average duration
            if self.debug_metrics:
                avg_duration = sum(m.duration for m in self.debug_metrics) / len(self.debug_metrics)
                max_duration = max(m.duration for m in self.debug_metrics)
                min_duration = min(m.duration for m in self.debug_metrics)
            else:
                avg_duration = max_duration = min_duration = 0
            
            # Get recent system metrics
            recent_metrics = list(self.performance_profile['system_metrics'].values())[-10:]
            
            return {
                'status': 'healthy',
                'details': {
                    'total_operations': total_operations,
                    'successful_operations': successful_operations,
                    'failed_operations': failed_operations,
                    'success_rate': (successful_operations / total_operations * 100) if total_operations > 0 else 0,
                    'avg_duration': round(avg_duration, 4),
                    'max_duration': round(max_duration, 4),
                    'min_duration': round(min_duration, 4),
                    'recent_system_metrics': recent_metrics
                }
            }
        except Exception as e:
            return {
                'status': 'error',
                'details': {'error': str(e)}
            }
    
    async def monitor_logs(self, log_files: Optional[List[str]] = None):
        """Monitor log files in real-time"""
        if not log_files:
            log_files = list(self.log_monitors.keys())
        
        logger.info(f"Starting log monitoring for files: {log_files}")
        
        for log_file in log_files:
            if log_file not in self.log_monitors:
                self.log_monitors[log_file] = {
                    'path': log_file,
                    'last_position': 0,
                    'errors': [],
                    'warnings': []
                }
        
        while self.is_running:
            try:
                for log_file, monitor in self.log_monitors.items():
                    if not Path(monitor['path']).exists():
                        continue
                    
                    # Read new lines
                    with open(monitor['path'], 'r') as f:
                        f.seek(monitor['last_position'])
                        new_lines = f.readlines()
                        monitor['last_position'] = f.tell()
                    
                    # Process new lines
                    for line in new_lines:
                        line = line.strip()
                        if 'ERROR' in line:
                            monitor['errors'].append({
                                'timestamp': datetime.utcnow().isoformat(),
                                'message': line
                            })
                        elif 'WARNING' in line:
                            monitor['warnings'].append({
                                'timestamp': datetime.utcnow().isoformat(),
                                'message': line
                            })
                    
                    # Keep only last 100 entries
                    monitor['errors'] = monitor['errors'][-100:]
                    monitor['warnings'] = monitor['warnings'][-100:]
                
                await asyncio.sleep(2)  # Check every 2 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring logs: {e}")
                await asyncio.sleep(5)
    
    async def run_performance_profile(self, duration: int = 60) -> Dict[str, Any]:
        """Run performance profiling for specified duration"""
        logger.info(f"Starting performance profiling for {duration} seconds...")
        
        self.is_running = True
        start_time = time.time()
        
        # Run profiling in background
        profile_task = asyncio.create_task(self._run_profile_operations())
        monitor_task = asyncio.create_task(self.monitor_logs())
        
        # Wait for duration
        await asyncio.sleep(duration)
        
        # Stop profiling
        self.is_running = False
        profile_task.cancel()
        monitor_task.cancel()
        
        # Generate report
        return await self._generate_performance_report()
    
    async def _run_profile_operations(self):
        """Run various operations for profiling"""
        operations = [
            ('database', 'query_positions'),
            ('database', 'query_orders'),
            ('database', 'insert_trade'),
            ('broker', 'get_account_info'),
            ('broker', 'get_positions'),
            ('strategy', 'evaluate_signals'),
            ('strategy', 'calculate_indicators')
        ]
        
        while self.is_running:
            for component, operation in operations:
                if not self.is_running:
                    break
                
                try:
                    async with self.measure_operation(component, operation):
                        await self._simulate_operation(component, operation)
                except Exception as e:
                    logger.debug(f"Operation {component}.{operation} failed: {e}")
                
                await asyncio.sleep(0.1)  # Small delay between operations
    
    async def _simulate_operation(self, component: str, operation: str):
        """Simulate various operations for profiling"""
        if component == 'database':
            # Simulate database operations
            if hasattr(self, 'database_manager') and self.database_manager:
                # Simulate a simple query
                await asyncio.sleep(0.01)  # Simulate query time
            else:
                await asyncio.sleep(0.005)  # Simulate lightweight operation
        
        elif component == 'broker':
            # Simulate broker operations
            await asyncio.sleep(0.02)  # Simulate API call time
        
        elif component == 'strategy':
            # Simulate strategy calculations
            await asyncio.sleep(0.005)  # Simulate calculation time
    
    async def _generate_performance_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        end_time = datetime.utcnow()
        
        # Calculate statistics
        component_stats = {}
        for metric in self.debug_metrics:
            component = metric.component
            if component not in component_stats:
                component_stats[component] = {
                    'operations': 0,
                    'total_duration': 0,
                    'success_count': 0,
                    'error_count': 0,
                    'durations': []
                }
            
            stats = component_stats[component]
            stats['operations'] += 1
            stats['total_duration'] += metric.duration
            stats['durations'].append(metric.duration)
            
            if metric.success:
                stats['success_count'] += 1
            else:
                stats['error_count'] += 1
        
        # Calculate averages and percentiles
        for component, stats in component_stats.items():
            if stats['operations'] > 0:
                stats['avg_duration'] = stats['total_duration'] / stats['operations']
                stats['success_rate'] = (stats['success_count'] / stats['operations']) * 100
                
                # Calculate percentiles
                durations = sorted(stats['durations'])
                n = len(durations)
                stats['p50'] = durations[int(n * 0.5)]
                stats['p90'] = durations[int(n * 0.9)]
                stats['p95'] = durations[int(n * 0.95)]
                stats['p99'] = durations[int(n * 0.99)]
        
        # System metrics summary
        system_metrics = list(self.performance_profile['system_metrics'].values())
        if system_metrics:
            cpu_values = [m['cpu_percent'] for m in system_metrics]
            memory_values = [m['memory_percent'] for m in system_metrics]
            
            system_summary = {
                'avg_cpu_percent': sum(cpu_values) / len(cpu_values),
                'max_cpu_percent': max(cpu_values),
                'avg_memory_percent': sum(memory_values) / len(memory_values),
                'max_memory_percent': max(memory_values),
                'samples': len(system_metrics)
            }
        else:
            system_summary = {}
        
        return {
            'profile_duration': (end_time - self.performance_profile['start_time']).total_seconds(),
            'end_time': end_time.isoformat(),
            'component_stats': component_stats,
            'system_summary': system_summary,
            'log_errors': {k: v['errors'][-5:] for k, v in self.log_monitors.items()},  # Last 5 errors per log
            'log_warnings': {k: v['warnings'][-5:] for k, v in self.log_monitors.items()}  # Last 5 warnings per log
        }
    
    def save_report(self, report: Dict[str, Any], output_file: str):
        """Save performance report to file"""
        try:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Performance report saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
    
    async def interactive_mode(self):
        """Start interactive debug mode"""
        print("\n=== Trading System Debug Interactive Mode ===")
        print("Available commands:")
        print("  inspect <component>  - Inspect component (database, broker, strategy, config, performance)")
        print("  monitor              - Start log monitoring")
        print("  profile <seconds>    - Run performance profiling")
        print("  report <file>        - Generate and save performance report")
        print("  status               - Show system status")
        print("  help                 - Show this help")
        print("  quit                 - Exit interactive mode")
        print("\n" + "="*50)
        
        while True:
            try:
                command = input("\ndebug> ").strip().split()
                
                if not command:
                    continue
                
                cmd = command[0].lower()
                
                if cmd == 'quit' or cmd == 'exit':
                    break
                elif cmd == 'help':
                    self._show_help()
                elif cmd == 'inspect' and len(command) > 1:
                    await self._cmd_inspect(command[1])
                elif cmd == 'monitor':
                    await self._cmd_monitor()
                elif cmd == 'profile' and len(command) > 1:
                    duration = int(command[1])
                    await self._cmd_profile(duration)
                elif cmd == 'report' and len(command) > 1:
                    output_file = command[1]
                    await self._cmd_report(output_file)
                elif cmd == 'status':
                    await self._cmd_status()
                else:
                    print(f"Unknown command: {' '.join(command)}")
                    print("Type 'help' for available commands.")
            
            except KeyboardInterrupt:
                print("\nUse 'quit' to exit.")
            except Exception as e:
                print(f"Error: {e}")
        
        print("\nExiting interactive debug mode.")
    
    def _show_help(self):
        """Show help text"""
        print("\n=== Help ===")
        print("Available commands:")
        print("  inspect <component>  - Inspect component state")
        print("  monitor              - Start log monitoring (runs until stopped)")
        print("  profile <seconds>    - Run performance profiling")
        print("  report <file>        - Generate performance report and save to file")
        print("  status               - Show overall system status")
        print("  help                 - Show this help")
        print("  quit                 - Exit interactive mode")
        print("\nComponents to inspect:")
        print("  database             - Database status and metrics")
        print("  broker               - Broker connections and status")
        print("  strategy             - Strategy status and performance")
        print("  config               - Configuration details")
        print("  performance          - Performance metrics")
    
    async def _cmd_inspect(self, component: str):
        """Handle inspect command"""
        print(f"\nInspecting {component}...")
        result = await self.inspect_component(component)
        print(json.dumps(result, indent=2, default=str))
    
    async def _cmd_monitor(self):
        """Handle monitor command"""
        print("\nStarting log monitoring... (Press Ctrl+C to stop)")
        try:
            self.is_running = True
            await self.monitor_logs()
        except KeyboardInterrupt:
            self.is_running = False
            print("\nLog monitoring stopped.")
    
    async def _cmd_profile(self, duration: int):
        """Handle profile command"""
        print(f"\nStarting performance profiling for {duration} seconds...")
        try:
            report = await self.run_performance_profile(duration)
            print("\n=== Performance Profile Results ===")
            print(json.dumps(report, indent=2, default=str))
        except Exception as e:
            print(f"Error during profiling: {e}")
    
    async def _cmd_report(self, output_file: str):
        """Handle report command"""
        print(f"\nGenerating performance report...")
        try:
            report = await self._generate_performance_report()
            self.save_report(report, output_file)
            print(f"Report saved to {output_file}")
        except Exception as e:
            print(f"Error generating report: {e}")
    
    async def _cmd_status(self):
        """Handle status command"""
        print("\n=== System Status ===")
        
        # Overall status
        print(f"Debugger running: {self.is_running}")
        print(f"Components initialized: {sum(1 for c in [self.database_manager, self.broker_manager, self.strategy_manager] if c is not None)}")
        print(f"Metrics collected: {len(self.debug_metrics)}")
        print(f"Log monitors: {len(self.log_monitors)}")
        
        # Recent performance
        if self.debug_metrics:
            recent_metrics = self.debug_metrics[-10:]
            avg_duration = sum(m.duration for m in recent_metrics) / len(recent_metrics)
            success_rate = sum(1 for m in recent_metrics if m.success) / len(recent_metrics) * 100
            
            print(f"Recent avg duration: {avg_duration:.4f}s")
            print(f"Recent success rate: {success_rate:.1f}%")
        
        # System resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        print(f"CPU usage: {cpu_percent:.1f}%")
        print(f"Memory usage: {memory.percent:.1f}% ({memory.used // (1024*1024)} MB)")


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Trading System Debug Tool')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--component', help='Component to inspect (database, broker, strategy, config, performance)')
    parser.add_argument('--level', default='INFO', help='Log level (DEBUG, INFO, WARNING, ERROR)')
    parser.add_argument('--output', help='Output file for reports')
    parser.add_argument('--interactive', action='store_true', help='Start interactive debug mode')
    parser.add_argument('--monitor-logs', action='store_true', help='Monitor log files')
    parser.add_argument('--profile-performance', type=int, metavar='SECONDS', help='Run performance profiling for specified seconds')
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.level.upper()))
    
    # Initialize debugger
    debugger = Debugger(args.config)
    
    # Handle cleanup on exit
    def signal_handler(signum, frame):
        print("\nShutting down debugger...")
        debugger.is_running = False
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize
        if not await debugger.initialize():
            print("Failed to initialize debug environment")
            return 1
        
        debugger.is_running = True
        
        # Handle different modes
        if args.interactive:
            await debugger.interactive_mode()
        elif args.component:
            result = await debugger.inspect_component(args.component)
            print(json.dumps(result, indent=2, default=str))
        elif args.monitor_logs:
            print("Starting log monitoring... (Press Ctrl+C to stop)")
            await debugger.monitor_logs()
        elif args.profile_performance:
            print(f"Running performance profiling for {args.profile_performance} seconds...")
            report = await debugger.run_performance_profile(args.profile_performance)
            
            if args.output:
                debugger.save_report(report, args.output)
                print(f"Report saved to {args.output}")
            else:
                print(json.dumps(report, indent=2, default=str))
        else:
            # Default: show system status
            await debugger._cmd_status()
        
        return 0
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Debug operation failed: {e}")
        logger.error(traceback.format_exc())
        return 1
    finally:
        debugger.is_running = False


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
