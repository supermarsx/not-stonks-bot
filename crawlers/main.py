"""
Market Data Crawler System - Main Integration
Provides unified interface for all crawler functionality
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

from .base.base_crawler import BaseCrawler, CrawlerConfig, DataType
from .scheduling.crawler_manager import CrawlerManager, CrawlerManagerConfig
from .storage.data_storage import DataStorage, DataMonitor
from .monitoring.health_monitor import AlertManager, CrawlerMonitor
from .config.error_handler import ConfigManager, ErrorHandler, DataValidator
from .monitoring.performance_monitor import PerformanceMonitor

# Import all crawler types
from .market_data.market_data_crawler import MarketDataCrawler
from .news.news_crawler import NewsCrawler  
from .social_media.social_media_crawler import SocialMediaCrawler
from .economic.economic_crawler import EconomicCrawler
from .patterns.pattern_crawler import PatternCrawler


class MarketDataCrawlerSystem:
    """Main market data crawler system orchestrator"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.config_manager = ConfigManager(self.config.get('config_dir', './configs'))
        self.error_handler = ErrorHandler(self.config_manager)
        self.data_validator = DataValidator()
        
        # Initialize storage
        storage_path = self.config.get('storage_path', './data')
        self.data_storage = DataStorage(storage_path)
        self.data_monitor = DataMonitor(self.data_storage)
        
        # Initialize manager and monitoring
        manager_config = CrawlerManagerConfig(
            name="MarketCrawlerSystem",
            max_concurrent_crawlers=self.config.get('max_concurrent_crawlers', 5)
        )
        self.manager = CrawlerManager(manager_config)
        
        # Initialize alerting and monitoring
        alert_config = self.config.get('alerts', {})
        self.alert_manager = AlertManager(alert_config)
        self.crawler_monitor = CrawlerMonitor(self.manager, self.alert_manager)
        self.performance_monitor = PerformanceMonitor(self.manager)
        
        # System state
        self._initialized = False
        self._running = False
        
        # Event handlers
        self._setup_event_handlers()
    
    def _default_config(self) -> Dict[str, Any]:
        """Get default system configuration"""
        return {
            'storage_path': './data',
            'config_dir': './configs',
            'max_concurrent_crawlers': 5,
            'symbols': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'],
            'alerts': {
                'email': {
                    'enabled': False,
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'from_email': '',
                    'to_emails': [],
                    'username': '',
                    'password': ''
                },
                'webhook': {
                    'enabled': False,
                    'url': ''
                },
                'slack': {
                    'enabled': False,
                    'webhook_url': ''
                }
            },
            'monitoring': {
                'enable_health_checks': True,
                'enable_performance_monitoring': True,
                'enable_alerts': True,
                'health_check_interval': 60,
                'performance_check_interval': 300
            }
        }
    
    def _setup_event_handlers(self):
        """Setup event handlers for monitoring"""
        # Alert manager event handlers
        self.alert_manager.add_rule("crawler_down", lambda data: self.logger.warning(f"Crawler down: {data['crawler_name']}"))
        self.alert_manager.add_rule("high_error_rate", lambda data: self.logger.warning(f"High error rate: {data['crawler_name']}"))
        
        # Manager event handlers
        self.manager.add_event_handler('crawler_completed', self._on_crawler_completed)
        self.manager.add_event_handler('crawler_failed', self._on_crawler_failed)
        self.manager.add_event_handler('health_check', self._on_health_check)
    
    async def _on_crawler_completed(self, data: Dict[str, Any]):
        """Handle crawler completion event"""
        self.logger.info(f"Crawler {data['crawler_name']} completed successfully in {data['execution_time']:.2f}s")
        
        # Record performance metric
        await self.performance_monitor.record_execution_snapshot(
            crawler_name=data['crawler_name'],
            execution_time=data['execution_time'],
            data_points_processed=data.get('data_points', 0),
            success=True
        )
        
        # Mark success in error handler
        self.error_handler.mark_success(data['crawler_name'])
    
    async def _on_crawler_failed(self, data: Dict[str, Any]):
        """Handle crawler failure event"""
        self.logger.error(f"Crawler {data['crawler_name']} failed: {data['error']}")
        
        # Record performance metric
        await self.performance_monitor.record_execution_snapshot(
            crawler_name=data['crawler_name'],
            execution_time=0,
            data_points_processed=0,
            success=False,
            error_message=data['error']
        )
    
    async def _on_health_check(self, data: Dict[str, Any]):
        """Handle health check event"""
        if data.get('overall_status') == 'degraded':
            self.logger.warning("System health degraded")
        elif data.get('overall_status') == 'error':
            self.logger.error("System health critical")
    
    async def initialize(self, symbols: Optional[List[str]] = None) -> bool:
        """Initialize the crawler system"""
        try:
            if self._initialized:
                self.logger.warning("System already initialized")
                return True
            
            # Use provided symbols or default
            target_symbols = symbols or self.config.get('symbols', ['AAPL', 'GOOGL'])
            
            # Create and register all crawlers
            crawlers = await self.manager.create_and_register_crawlers(target_symbols)
            
            self._initialized = True
            self.logger.info(f"Market Data Crawler System initialized with {len(crawlers)} crawlers")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to initialize system: {e}")
            return False
    
    async def start(self) -> bool:
        """Start the crawler system"""
        try:
            if not self._initialized:
                await self.initialize()
            
            if self._running:
                self.logger.warning("System already running")
                return True
            
            # Start all components
            await self.manager.start_all_crawlers()
            await self.crawler_monitor.start_monitoring()
            await self.performance_monitor.start_monitoring()
            
            self._running = True
            self.logger.info("Market Data Crawler System started")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to start system: {e}")
            return False
    
    async def stop(self):
        """Stop the crawler system"""
        try:
            if not self._running:
                self.logger.warning("System not running")
                return
            
            # Stop all components
            await self.performance_monitor.stop_monitoring()
            await self.crawler_monitor.stop_monitoring()
            await self.manager.stop_all_crawlers()
            
            self._running = False
            self.logger.info("Market Data Crawler System stopped")
        
        except Exception as e:
            self.logger.error(f"Error stopping system: {e}")
    
    async def restart(self) -> bool:
        """Restart the crawler system"""
        try:
            await self.stop()
            await asyncio.sleep(5)  # Brief pause
            return await self.start()
        except Exception as e:
            self.logger.error(f"Failed to restart system: {e}")
            return False
    
    async def trigger_crawler(self, crawler_name: str, force: bool = False) -> bool:
        """Manually trigger a specific crawler"""
        try:
            await self.manager.trigger_crawler(crawler_name, force)
            return True
        except Exception as e:
            self.logger.error(f"Failed to trigger crawler {crawler_name}: {e}")
            return False
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            status = {
                'timestamp': datetime.now().isoformat(),
                'system': {
                    'initialized': self._initialized,
                    'running': self._running,
                    'uptime': self._get_uptime()
                },
                'manager': self.manager.get_status(),
                'performance': self.performance_monitor.get_performance_summary(),
                'health': self.crawler_monitor.get_health_summary(),
                'storage': await self.data_monitor.check_storage_health(),
                'alerts': {
                    'active_alerts': len(self.alert_manager.get_active_alerts()),
                    'recent_alerts': len(self.alert_manager.get_alert_history())
                },
                'errors': self.error_handler.get_error_summary()
            }
            
            return status
        
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}
    
    def _get_uptime(self) -> Optional[float]:
        """Calculate system uptime"""
        # This would track start time in a real implementation
        return None
    
    async def get_crawler_data(self, crawler_name: str, start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get data from a specific crawler"""
        try:
            return await self.data_storage.retrieve_data(
                crawler_name=crawler_name,
                start_date=start_date,
                end_date=end_date,
                limit=limit
            )
        except Exception as e:
            self.logger.error(f"Error getting data for {crawler_name}: {e}")
            return []
    
    async def get_latest_data(self, crawler_name: str) -> Optional[Dict[str, Any]]:
        """Get latest data from a crawler"""
        try:
            return await self.data_storage.get_latest_data(crawler_name)
        except Exception as e:
            self.logger.error(f"Error getting latest data for {crawler_name}: {e}")
            return None
    
    async def search_data(self, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search crawled data"""
        try:
            return await self.data_storage.search_data(query_params)
        except Exception as e:
            self.logger.error(f"Error searching data: {e}")
            return []
    
    async def export_data(self, crawler_name: str, start_date: datetime, 
                         end_date: datetime, format: str = "json") -> Optional[str]:
        """Export data in various formats"""
        try:
            return await self.data_storage.export_data(
                crawler_name=crawler_name,
                start_date=start_date,
                end_date=end_date,
                output_format=format
            )
        except Exception as e:
            self.logger.error(f"Error exporting data: {e}")
            return None
    
    async def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        try:
            return {
                'performance_summary': self.performance_monitor.get_performance_summary(hours),
                'bottleneck_report': self.performance_monitor.get_bottleneck_report(hours),
                'optimization_report': self.performance_monitor.get_optimization_report(hours),
                'error_summary': self.error_handler.get_error_summary(hours)
            }
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
            return {'error': str(e)}
    
    async def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report"""
        try:
            return {
                'system_health': await self.data_monitor.generate_quality_report(),
                'crawler_health': self.crawler_monitor.get_health_summary(),
                'alert_status': {
                    'active_alerts': [alert.__dict__ for alert in self.alert_manager.get_active_alerts()],
                    'alert_rules': {name: rule.__dict__ for name, rule in self.alert_manager.rules.items()}
                }
            }
        except Exception as e:
            self.logger.error(f"Error generating health report: {e}")
            return {'error': str(e)}
    
    def update_config(self, crawler_name: str, config_updates: Dict[str, Any]):
        """Update configuration for a crawler"""
        try:
            # Get current config
            current_config = self.config_manager.get_config(crawler_name)
            if not current_config:
                raise ValueError(f"Configuration for {crawler_name} not found")
            
            # Apply updates (simplified - would need more sophisticated merging)
            if 'interval' in config_updates:
                current_config.base_config.interval = config_updates['interval']
            if 'timeout' in config_updates:
                current_config.base_config.timeout = config_updates['timeout']
            if 'rate_limit' in config_updates:
                current_config.base_config.rate_limit = config_updates['rate_limit']
            
            # Save updated config
            self.config_manager.update_config(crawler_name, current_config)
            
            self.logger.info(f"Updated configuration for {crawler_name}")
        
        except Exception as e:
            self.logger.error(f"Error updating config for {crawler_name}: {e}")
            raise
    
    def add_alert_rule(self, rule_name: str, condition: str, threshold: float, 
                      level: str, channels: List[str]):
        """Add custom alert rule"""
        try:
            from .monitoring.health_monitor import AlertRule, AlertLevel, AlertChannel
            
            rule = AlertRule(
                name=rule_name,
                condition=condition,
                threshold=threshold,
                level=AlertLevel(level),
                channels=[AlertChannel(channel) for channel in channels]
            )
            
            self.alert_manager.add_rule(rule)
            self.logger.info(f"Added alert rule: {rule_name}")
        
        except Exception as e:
            self.logger.error(f"Error adding alert rule: {e}")
            raise
    
    async def validate_configuration(self, crawler_name: str) -> Dict[str, Any]:
        """Validate crawler configuration"""
        try:
            return self.config_manager.validate_config(crawler_name)
        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return {'valid': False, 'errors': [str(e)]}
    
    async def validate_data(self, crawler_name: str, data: Any) -> Dict[str, Any]:
        """Validate crawler data quality"""
        try:
            return self.data_validator.validate_data(crawler_name, data)
        except Exception as e:
            self.logger.error(f"Error validating data: {e}")
            return {'valid': False, 'errors': [str(e)]}
    
    async def cleanup_old_data(self, days_old: int = 90):
        """Cleanup old data and metrics"""
        try:
            await self.data_storage.cleanup_old_data(days_old)
            self.logger.info(f"Cleaned up data older than {days_old} days")
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
    
    async def run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check"""
        try:
            health_status = await self.manager.perform_health_check()
            storage_health = await self.data_monitor.check_storage_health()
            quality_report = await self.data_monitor.generate_quality_report()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'crawler_health': health_status,
                'storage_health': storage_health,
                'quality_report': quality_report,
                'overall_status': self._determine_overall_status(health_status, storage_health)
            }
        
        except Exception as e:
            self.logger.error(f"Error running health check: {e}")
            return {'error': str(e)}
    
    def _determine_overall_status(self, crawler_health: Dict[str, Any], 
                                storage_health: Dict[str, Any]) -> str:
        """Determine overall system status"""
        try:
            crawler_status = crawler_health.get('overall_status', 'unknown')
            storage_status = storage_health.get('status', 'unknown')
            
            if crawler_status == 'error' or storage_status == 'error':
                return 'error'
            elif crawler_status == 'degraded' or storage_status == 'degraded':
                return 'degraded'
            elif crawler_status == 'healthy' and storage_status == 'healthy':
                return 'healthy'
            else:
                return 'unknown'
        
        except Exception:
            return 'unknown'
    
    def get_supported_operations(self) -> Dict[str, List[str]]:
        """Get list of supported operations"""
        return {
            'crawlers': ['market_data', 'news', 'social_media', 'economic', 'patterns'],
            'data_formats': ['json', 'csv'],
            'export_formats': ['json', 'csv'],
            'alert_channels': ['email', 'webhook', 'slack', 'log'],
            'crawler_operations': ['start', 'stop', 'restart', 'trigger', 'status'],
            'monitoring_operations': ['health_check', 'performance_report', 'bottleneck_analysis'],
            'storage_operations': ['export', 'search', 'cleanup', 'validate']
        }


# Convenience functions for quick access
async def create_crawler_system(config: Optional[Dict[str, Any]] = None) -> MarketDataCrawlerSystem:
    """Create and initialize a crawler system"""
    system = MarketDataCrawlerSystem(config)
    await system.initialize()
    return system


async def quick_start(symbols: List[str] = None, config: Optional[Dict[str, Any]] = None) -> MarketDataCrawlerSystem:
    """Quick start a crawler system with default settings"""
    system = MarketDataCrawlerSystem(config)
    await system.initialize(symbols)
    await system.start()
    return system


# Example usage
async def main():
    """Example usage of the market data crawler system"""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create system configuration
    config = {
        'symbols': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'],
        'storage_path': './market_data_storage',
        'alerts': {
            'email': {
                'enabled': False  # Set to True and configure for email alerts
            }
        }
    }
    
    # Create and start system
    system = await create_crawler_system(config)
    await system.start()
    
    try:
        # Wait for some data collection
        print("Waiting for data collection...")
        await asyncio.sleep(60)
        
        # Get system status
        status = await system.get_system_status()
        print(f"System Status: {status['system']}")
        
        # Get latest market data
        latest_data = await system.get_latest_data('market_data')
        if latest_data:
            print(f"Latest market data: {len(latest_data.get('data', {}).get('real_time', {}))} symbols")
        
        # Get performance report
        performance = await system.get_performance_report(hours=1)
        print(f"Performance: {performance.get('performance_summary', {}).get('overall_metrics', {})}")
        
        # Get health report
        health = await system.get_health_report()
        print(f"Health status: {health.get('system_health', {}).get('overall_status', 'unknown')}")
        
        # Keep running
        print("System running. Press Ctrl+C to stop...")
        while True:
            await asyncio.sleep(60)
            
            # Check if any critical alerts
            active_alerts = system.alert_manager.get_active_alerts()
            critical_alerts = [alert for alert in active_alerts if alert.level.value == 'critical']
            
            if critical_alerts:
                print(f"Critical alerts: {len(critical_alerts)}")
                for alert in critical_alerts:
                    print(f"  - {alert.title}: {alert.message}")
    
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        await system.stop()
        print("System stopped")


if __name__ == "__main__":
    asyncio.run(main())