"""
Crawler Scheduling and Management System
Orchestrates and manages all crawlers with scheduling, monitoring, and coordination
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
import yaml
from pathlib import Path

from ..base.base_crawler import BaseCrawler, CrawlerConfig, CrawlerStatus
from .market_data.market_data_crawler import MarketDataCrawler
from .news.news_crawler import NewsCrawler
from .social_media.social_media_crawler import SocialMediaCrawler
from .economic.economic_crawler import EconomicCrawler
from .patterns.pattern_crawler import PatternCrawler


class ScheduleType(Enum):
    """Crawler schedule types"""
    REAL_TIME = "real_time"  # Every few seconds
    INTRADAY = "intraday"    # Every few minutes
    HOURLY = "hourly"        # Every hour
    DAILY = "daily"          # Once per day
    WEEKLY = "weekly"        # Once per week
    MANUAL = "manual"        # Triggered manually


@dataclass
class CrawlerSchedule:
    """Crawler schedule configuration"""
    crawler_name: str
    schedule_type: ScheduleType
    interval_seconds: int
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    enabled: bool = True
    priority: int = 1  # 1-10, higher = more important
    retry_delay: int = 60  # seconds
    max_retries: int = 3
    dependencies: List[str] = None  # Other crawlers that must complete first


@dataclass
class CrawlerManagerConfig:
    """Crawler manager configuration"""
    name: str
    max_concurrent_crawlers: int = 5
    default_retry_delay: int = 60
    health_check_interval: int = 30
    alert_thresholds: Dict[str, float] = None
    storage_config: Dict[str, Any] = None
    monitoring_config: Dict[str, Any] = None


class CrawlerManager:
    """Manages and coordinates all crawlers"""
    
    def __init__(self, config: CrawlerManagerConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Registry of crawlers
        self.crawlers: Dict[str, BaseCrawler] = {}
        self.schedules: Dict[str, CrawlerSchedule] = {}
        self.crawler_status: Dict[str, Dict[str, Any]] = {}
        
        # Execution tracking
        self.execution_queue: List[str] = []
        self.running_crawlers: Dict[str, asyncio.Task] = {}
        self.crawler_tasks: Dict[str, asyncio.Task] = {}
        
        # Performance metrics
        self.performance_metrics: Dict[str, Dict[str, Any]] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {
            'crawler_started': [],
            'crawler_completed': [],
            'crawler_failed': [],
            'crawler_error': [],
            'health_check': [],
            'schedule_triggered': []
        }
        
        # Background tasks
        self._health_check_task: Optional[asyncio.Task] = None
        self._schedule_monitor_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Initialize default configurations
        self._setup_default_schedules()
    
    def _setup_default_schedules(self):
        """Setup default schedules for all crawler types"""
        self.schedules = {
            'market_data': CrawlerSchedule(
                crawler_name='market_data',
                schedule_type=ScheduleType.INTRADAY,
                interval_seconds=30,  # 30 seconds
                priority=10,
                dependencies=[]
            ),
            'news': CrawlerSchedule(
                crawler_name='news',
                schedule_type=ScheduleType.HOURLY,
                interval_seconds=3600,  # 1 hour
                priority=8,
                dependencies=[]
            ),
            'social_media': CrawlerSchedule(
                crawler_name='social_media',
                schedule_type=ScheduleType.INTRADAY,
                interval_seconds=300,  # 5 minutes
                priority=7,
                dependencies=[]
            ),
            'economic': CrawlerSchedule(
                crawler_name='economic',
                schedule_type=ScheduleType.DAILY,
                interval_seconds=86400,  # 24 hours
                priority=9,
                dependencies=[]
            ),
            'patterns': CrawlerSchedule(
                crawler_name='patterns',
                schedule_type=ScheduleType.HOURLY,
                interval_seconds=3600,  # 1 hour
                priority=6,
                dependencies=['market_data']
            )
        }
    
    async def register_crawler(self, name: str, crawler: BaseCrawler, schedule: Optional[CrawlerSchedule] = None):
        """Register a crawler with the manager"""
        self.crawlers[name] = crawler
        self.crawler_status[name] = {
            'status': 'registered',
            'last_run': None,
            'last_success': None,
            'run_count': 0,
            'error_count': 0,
            'consecutive_errors': 0
        }
        
        if schedule:
            self.schedules[name] = schedule
        elif name in self.schedules:
            pass  # Use default schedule
        else:
            # Create default schedule
            self.schedules[name] = CrawlerSchedule(
                crawler_name=name,
                schedule_type=ScheduleType.MANUAL,
                interval_seconds=3600
            )
        
        self.logger.info(f"Registered crawler '{name}' with schedule '{self.schedules[name].schedule_type.value}'")
    
    async def create_and_register_crawlers(self, symbols: List[str]) -> Dict[str, BaseCrawler]:
        """Create all crawler instances and register them"""
        crawlers = {}
        
        try:
            # Market Data Crawler
            market_config = CrawlerConfig(
                name='market_data',
                data_type=DataType.MARKET_DATA,
                interval=30,
                max_retries=3,
                rate_limit=100
            )
            market_crawler = MarketDataCrawler(market_config, symbols)
            crawlers['market_data'] = market_crawler
            await self.register_crawler('market_data', market_crawler)
            
            # News Crawler
            news_config = CrawlerConfig(
                name='news',
                data_type=DataType.NEWS,
                interval=3600,
                max_retries=3,
                rate_limit=50
            )
            news_crawler = NewsCrawler(news_config, symbols)
            crawlers['news'] = news_crawler
            await self.register_crawler('news', news_crawler)
            
            # Social Media Crawler
            social_config = CrawlerConfig(
                name='social_media',
                data_type=DataType.SOCIAL_MEDIA,
                interval=300,
                max_retries=3,
                rate_limit=30
            )
            social_crawler = SocialMediaCrawler(social_config, symbols)
            crawlers['social_media'] = social_crawler
            await self.register_crawler('social_media', social_crawler)
            
            # Economic Crawler
            economic_config = CrawlerConfig(
                name='economic',
                data_type=DataType.ECONOMIC,
                interval=86400,
                max_retries=3,
                rate_limit=20
            )
            economic_crawler = EconomicCrawler(economic_config)
            crawlers['economic'] = economic_crawler
            await self.register_crawler('economic', economic_crawler)
            
            # Pattern Crawler
            pattern_config = CrawlerConfig(
                name='patterns',
                data_type=DataType.TECHNICAL,
                interval=3600,
                max_retries=3,
                rate_limit=10
            )
            pattern_crawler = PatternCrawler(pattern_config, symbols)
            crawlers['patterns'] = pattern_crawler
            await self.register_crawler('patterns', pattern_crawler)
            
            self.logger.info(f"Created and registered {len(crawlers)} crawlers")
            return crawlers
        
        except Exception as e:
            self.logger.error(f"Error creating crawlers: {e}")
            raise
    
    async def start_all_crawlers(self):
        """Start all registered crawlers"""
        try:
            # Start crawlers in dependency order
            sorted_crawlers = self._sort_crawlers_by_dependency()
            
            for crawler_name in sorted_crawlers:
                if self.schedules[crawler_name].enabled:
                    await self.start_crawler(crawler_name)
            
            # Start monitoring tasks
            await self._start_monitoring_tasks()
            
            self.logger.info(f"Started {len(sorted_crawlers)} crawlers")
        
        except Exception as e:
            self.logger.error(f"Error starting crawlers: {e}")
            raise
    
    def _sort_crawlers_by_dependency(self) -> List[str]:
        """Sort crawlers by dependency order"""
        visited = set()
        sorted_crawlers = []
        
        def visit(crawler_name: str):
            if crawler_name in visited:
                return
            
            visited.add(crawler_name)
            
            # Visit dependencies first
            schedule = self.schedules.get(crawler_name)
            if schedule and schedule.dependencies:
                for dep in schedule.dependencies:
                    if dep in self.crawlers:
                        visit(dep)
            
            sorted_crawlers.append(crawler_name)
        
        # Visit all registered crawlers
        for crawler_name in self.crawlers.keys():
            visit(crawler_name)
        
        return sorted_crawlers
    
    async def start_crawler(self, name: str):
        """Start a specific crawler"""
        if name not in self.crawlers:
            raise ValueError(f"Crawler '{name}' not registered")
        
        crawler = self.crawlers[name]
        
        if name in self.running_crawlers:
            self.logger.warning(f"Crawler '{name}' is already running")
            return
        
        try:
            # Update status
            self.crawler_status[name]['status'] = 'starting'
            
            # Start the crawler
            await crawler.start()
            
            # Create scheduled task
            schedule = self.schedules[name]
            task = asyncio.create_task(self._scheduled_crawler_run(name, schedule))
            self.crawler_tasks[name] = task
            self.running_crawlers[name] = task
            
            # Update status
            self.crawler_status[name]['status'] = 'running'
            
            self.logger.info(f"Started crawler '{name}'")
            await self._emit_event('crawler_started', {'crawler_name': name})
        
        except Exception as e:
            self.logger.error(f"Error starting crawler '{name}': {e}")
            self.crawler_status[name]['status'] = 'error'
            await self._emit_event('crawler_error', {'crawler_name': name, 'error': str(e)})
            raise
    
    async def stop_crawler(self, name: str):
        """Stop a specific crawler"""
        if name not in self.crawlers:
            raise ValueError(f"Crawler '{name}' not registered")
        
        try:
            # Cancel scheduled task
            if name in self.crawler_tasks:
                self.crawler_tasks[name].cancel()
                del self.crawler_tasks[name]
            
            # Stop the crawler
            crawler = self.crawlers[name]
            await crawler.stop()
            
            # Update status
            if name in self.running_crawlers:
                del self.running_crawlers[name]
            self.crawler_status[name]['status'] = 'stopped'
            
            self.logger.info(f"Stopped crawler '{name}'")
        
        except Exception as e:
            self.logger.error(f"Error stopping crawler '{name}': {e}")
            raise
    
    async def stop_all_crawlers(self):
        """Stop all running crawlers"""
        try:
            # Cancel all tasks
            for name, task in list(self.crawler_tasks.items()):
                task.cancel()
            
            # Stop all crawlers
            stop_tasks = [self.stop_crawler(name) for name in list(self.crawlers.keys())]
            await asyncio.gather(*stop_tasks, return_exceptions=True)
            
            # Stop monitoring tasks
            await self._stop_monitoring_tasks()
            
            self.logger.info("Stopped all crawlers")
        
        except Exception as e:
            self.logger.error(f"Error stopping all crawlers: {e}")
            raise
    
    async def _scheduled_crawler_run(self, name: str, schedule: CrawlerSchedule):
        """Run crawler according to its schedule"""
        try:
            while True:
                # Check if crawler is enabled
                if not schedule.enabled:
                    await asyncio.sleep(10)
                    continue
                
                # Check time constraints
                now = datetime.now()
                if schedule.start_time and now < schedule.start_time:
                    sleep_time = (schedule.start_time - now).total_seconds()
                    await asyncio.sleep(min(sleep_time, 3600))  # Sleep max 1 hour
                    continue
                
                if schedule.end_time and now > schedule.end_time:
                    self.logger.info(f"Crawler '{name}' schedule ended")
                    break
                
                # Check dependencies
                if schedule.dependencies:
                    await self._wait_for_dependencies(schedule.dependencies)
                
                # Execute crawler
                await self._execute_crawler(name, schedule)
                
                # Wait for next execution
                await asyncio.sleep(schedule.interval_seconds)
        
        except asyncio.CancelledError:
            self.logger.info(f"Crawler '{name}' schedule cancelled")
        except Exception as e:
            self.logger.error(f"Error in scheduled run for '{name}': {e}")
    
    async def _execute_crawler(self, name: str, schedule: CrawlerSchedule):
        """Execute a single crawler run"""
        try:
            start_time = datetime.now()
            self.crawler_status[name]['status'] = 'executing'
            
            self.logger.info(f"Executing crawler '{name}'")
            await self._emit_event('schedule_triggered', {'crawler_name': name})
            
            # Execute the crawler
            result = await self.crawlers[name].fetch_once()
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Update status
            self.crawler_status[name].update({
                'last_run': start_time.isoformat(),
                'run_count': self.crawler_status[name]['run_count'] + 1,
                'last_execution_time': execution_time,
                'status': 'running' if result.success else 'error'
            })
            
            if result.success:
                self.crawler_status[name]['last_success'] = start_time.isoformat()
                self.crawler_status[name]['consecutive_errors'] = 0
                self.logger.info(f"Crawler '{name}' completed successfully in {execution_time:.2f}s")
                await self._emit_event('crawler_completed', {
                    'crawler_name': name,
                    'execution_time': execution_time,
                    'success': True
                })
            else:
                self.crawler_status[name]['error_count'] += 1
                self.crawler_status[name]['consecutive_errors'] += 1
                self.logger.error(f"Crawler '{name}' failed: {result.error_message}")
                await self._emit_event('crawler_failed', {
                    'crawler_name': name,
                    'error': result.error_message,
                    'consecutive_errors': self.crawler_status[name]['consecutive_errors']
                })
            
            # Record execution history
            self.execution_history.append({
                'crawler_name': name,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'execution_time': execution_time,
                'success': result.success,
                'error_message': result.error_message
            })
            
            # Keep only last 1000 records
            if len(self.execution_history) > 1000:
                self.execution_history = self.execution_history[-1000:]
        
        except Exception as e:
            self.logger.error(f"Error executing crawler '{name}': {e}")
            self.crawler_status[name]['error_count'] += 1
            self.crawler_status[name]['consecutive_errors'] += 1
            self.crawler_status[name]['status'] = 'error'
            
            await self._emit_event('crawler_error', {'crawler_name': name, 'error': str(e)})
    
    async def _wait_for_dependencies(self, dependencies: List[str]):
        """Wait for dependency crawlers to complete"""
        for dep_name in dependencies:
            if dep_name in self.crawlers:
                # Wait for dependency to complete its last run
                max_wait = 300  # 5 minutes max wait
                start_wait = datetime.now()
                
                while (datetime.now() - start_wait).total_seconds() < max_wait:
                    if dep_name not in self.running_crawlers:
                        break
                    
                    dep_status = self.crawler_status.get(dep_name, {})
                    last_run = dep_status.get('last_run')
                    
                    if last_run:
                        last_run_time = datetime.fromisoformat(last_run)
                        if (datetime.now() - last_run_time).total_seconds() < 60:  # Completed within last minute
                            break
                    
                    await asyncio.sleep(5)  # Check every 5 seconds
    
    async def _start_monitoring_tasks(self):
        """Start background monitoring tasks"""
        # Health check task
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        # Schedule monitoring task
        self._schedule_monitor_task = asyncio.create_task(self._schedule_monitor_loop())
        
        # Cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _stop_monitoring_tasks(self):
        """Stop background monitoring tasks"""
        tasks = [self._health_check_task, self._schedule_monitor_task, self._cleanup_task]
        
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
    
    async def _health_check_loop(self):
        """Periodic health check for all crawlers"""
        while True:
            try:
                await self.perform_health_check()
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health check: {e}")
                await asyncio.sleep(self.config.health_check_interval)
    
    async def perform_health_check(self) -> Dict[str, Any]:
        """Perform health check on all crawlers"""
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'crawlers': {},
            'summary': {
                'total_crawlers': len(self.crawlers),
                'running_crawlers': len(self.running_crawlers),
                'healthy_crawlers': 0,
                'error_crawlers': 0
            }
        }
        
        try:
            for name, crawler in self.crawlers.items():
                try:
                    crawler_health = await crawler.health_check()
                    health_status['crawlers'][name] = crawler_health
                    
                    if crawler_health['healthy']:
                        health_status['summary']['healthy_crawlers'] += 1
                    else:
                        health_status['summary']['error_crawlers'] += 1
                        health_status['overall_status'] = 'degraded'
                
                except Exception as e:
                    health_status['crawlers'][name] = {
                        'healthy': False,
                        'error': str(e)
                    }
                    health_status['summary']['error_crawlers'] += 1
                    health_status['overall_status'] = 'degraded'
            
            self.logger.info(f"Health check: {health_status['summary']['healthy_crawlers']}/{health_status['summary']['total_crawlers']} crawlers healthy")
            await self._emit_event('health_check', health_status)
            
            return health_status
        
        except Exception as e:
            self.logger.error(f"Error performing health check: {e}")
            health_status['overall_status'] = 'error'
            return health_status
    
    async def _schedule_monitor_loop(self):
        """Monitor and adjust crawler schedules"""
        while True:
            try:
                # Check for any issues that might require schedule adjustment
                for name, status in self.crawler_status.items():
                    if status['consecutive_errors'] >= 3:
                        # Reduce frequency for problematic crawlers
                        schedule = self.schedules[name]
                        if schedule.interval_seconds < 3600:  # Don't make it too infrequent
                            schedule.interval_seconds = min(schedule.interval_seconds * 2, 3600)
                            self.logger.warning(f"Reduced frequency for crawler '{name}' due to errors")
                
                await asyncio.sleep(300)  # Check every 5 minutes
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in schedule monitor: {e}")
                await asyncio.sleep(300)
    
    async def _cleanup_loop(self):
        """Periodic cleanup of old data and metrics"""
        while True:
            try:
                # Clean up execution history (keep last 1000 records)
                if len(self.execution_history) > 1000:
                    self.execution_history = self.execution_history[-1000:]
                
                # Clean up performance metrics
                cutoff_time = datetime.now() - timedelta(days=7)
                for crawler_name in list(self.performance_metrics.keys()):
                    metrics = self.performance_metrics[crawler_name]
                    if 'last_updated' in metrics:
                        last_updated = datetime.fromisoformat(metrics['last_updated'])
                        if last_updated < cutoff_time:
                            del self.performance_metrics[crawler_name]
                
                await asyncio.sleep(3600)  # Clean up every hour
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup: {e}")
                await asyncio.sleep(3600)
    
    async def trigger_crawler(self, name: str, force: bool = False):
        """Manually trigger a crawler"""
        if name not in self.crawlers:
            raise ValueError(f"Crawler '{name}' not registered")
        
        if not force and name in self.running_crawlers:
            raise ValueError(f"Crawler '{name}' is already running")
        
        self.logger.info(f"Manually triggering crawler '{name}'")
        await self._execute_crawler(name, self.schedules[name])
    
    def update_schedule(self, name: str, schedule: CrawlerSchedule):
        """Update crawler schedule"""
        if name not in self.crawlers:
            raise ValueError(f"Crawler '{name}' not registered")
        
        self.schedules[name] = schedule
        self.logger.info(f"Updated schedule for crawler '{name}'")
    
    def get_status(self) -> Dict[str, Any]:
        """Get overall manager status"""
        return {
            'manager_config': asdict(self.config),
            'registered_crawlers': list(self.crawlers.keys()),
            'running_crawlers': list(self.running_crawlers.keys()),
            'crawler_status': self.crawler_status.copy(),
            'schedules': {name: asdict(schedule) for name, schedule in self.schedules.items()},
            'performance_metrics': self.performance_metrics.copy(),
            'execution_history_count': len(self.execution_history)
        }
    
    def get_crawler_status(self, name: str) -> Optional[Dict[str, Any]]:
        """Get status of specific crawler"""
        return self.crawler_status.get(name)
    
    async def save_config(self, filepath: str):
        """Save configuration to file"""
        config_data = {
            'manager_config': asdict(self.config),
            'schedules': {name: asdict(schedule) for name, schedule in self.schedules.items()},
            'crawler_status': self.crawler_status
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
        
        self.logger.info(f"Saved configuration to {filepath}")
    
    async def load_config(self, filepath: str):
        """Load configuration from file"""
        try:
            with open(filepath, 'r') as f:
                config_data = json.load(f)
            
            # Update manager config
            if 'manager_config' in config_data:
                manager_data = config_data['manager_config']
                for key, value in manager_data.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
            
            # Update schedules
            if 'schedules' in config_data:
                for name, schedule_data in config_data['schedules'].items():
                    schedule_data['schedule_type'] = ScheduleType(schedule_data['schedule_type'])
                    self.schedules[name] = CrawlerSchedule(**schedule_data)
            
            # Update crawler status
            if 'crawler_status' in config_data:
                self.crawler_status.update(config_data['crawler_status'])
            
            self.logger.info(f"Loaded configuration from {filepath}")
        
        except Exception as e:
            self.logger.error(f"Error loading configuration from {filepath}: {e}")
            raise
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """Add event handler"""
        if event_type in self.event_handlers:
            self.event_handlers[event_type].append(handler)
        else:
            self.event_handlers[event_type] = [handler]
    
    async def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit event to all handlers"""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    self.logger.error(f"Error in event handler for {event_type}: {e}")