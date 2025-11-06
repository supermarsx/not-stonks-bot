"""
Crawler Trading Integrator

Main integration class that coordinates all crawler data flows with the
trading orchestrator system, providing unified access to real-time market data,
news, sentiment, patterns, and economic indicators.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, AsyncIterator
from dataclasses import dataclass
import json

from loguru import logger

from .data_bridge import CrawlerDataBridge, TradingDataPoint, DataPriority
from .event_handler import CrawlerEventHandler, MarketEvent, EventType

from ..scheduling.crawler_manager import CrawlerManager
from ..storage.data_storage import DataStorage
from ..monitoring.health_monitor import HealthMonitor


@dataclass
class TradingSystemConfig:
    """Configuration for crawler-trading system integration"""
    # Data requirements
    required_data_types: List[str] = None
    symbols_to_monitor: List[str] = None
    update_frequencies: Dict[str, int] = None
    
    # Event handling
    enable_real_time_events: bool = True
    event_buffer_size: int = 1000
    alert_thresholds: Dict[str, float] = None
    
    # Performance settings
    max_concurrent_requests: int = 50
    data_retention_hours: int = 168  # 1 week
    cache_size_limit: int = 10000
    
    def __post_init__(self):
        if self.required_data_types is None:
            self.required_data_types = ['market_data', 'news', 'sentiment', 'patterns']
        
        if self.symbols_to_monitor is None:
            self.symbols_to_monitor = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'SPY']
        
        if self.update_frequencies is None:
            self.update_frequencies = {
                'market_data': 1,      # 1 second
                'news': 5,             # 5 seconds
                'sentiment': 10,       # 10 seconds
                'patterns': 30,        # 30 seconds
                'economic': 300        # 5 minutes
            }
        
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                'sentiment_change': 0.3,
                'price_change': 0.05,
                'pattern_confidence': 0.8
            }


class CrawlerTradingIntegrator:
    """
    Main integration class that bridges crawler system with trading orchestrator.
    
    This class provides:
    - Unified data access to all crawler data sources
    - Real-time event streaming
    - Data transformation and filtering
    - Performance optimization and caching
    - Health monitoring and alerting
    """
    
    def __init__(self, config: TradingSystemConfig = None):
        self.config = config or TradingSystemConfig()
        
        # Core components
        self.data_storage = None
        self.health_monitor = None
        self.crawler_manager = None
        
        # Integration components
        self.data_bridge = None
        self.event_handler = None
        
        # State
        self.is_initialized = False
        self.is_running = False
        
        # Data caches
        self.latest_data_cache: Dict[str, TradingDataPoint] = {}
        self.data_subscribers: Dict[str, List[Callable]] = {}
        self.event_subscribers: Dict[EventType, List[Callable]] = {}
        
        # Performance metrics
        self.metrics = {
            'data_points_processed': 0,
            'events_emitted': 0,
            'subscriber_notifications': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    async def initialize(self):
        """Initialize the integrator with all components"""
        try:
            logger.info("Initializing Crawler Trading Integrator...")
            
            # Initialize core crawler components
            self.data_storage = DataStorage()
            self.health_monitor = HealthMonitor()
            self.crawler_manager = CrawlerManager()
            
            # Initialize integration components
            self.data_bridge = CrawlerDataBridge(self.data_storage, self.health_monitor)
            self.event_handler = CrawlerEventHandler()
            
            # Start components
            await self.data_storage.initialize()
            await self.health_monitor.initialize()
            await self.crawler_manager.initialize()
            await self.data_bridge.start()
            await self.event_handler.start()
            
            # Set up data subscriptions
            await self._setup_data_subscriptions()
            await self._setup_event_subscriptions()
            
            self.is_initialized = True
            logger.info("Crawler Trading Integrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Crawler Trading Integrator: {e}")
            raise
    
    async def start(self):
        """Start the integration system"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            logger.info("Starting Crawler Trading Integrator...")
            
            # Start crawler managers
            await self.crawler_manager.start_all_crawlers()
            
            self.is_running = True
            logger.info("Crawler Trading Integrator started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Crawler Trading Integrator: {e}")
            raise
    
    async def stop(self):
        """Stop the integration system"""
        try:
            logger.info("Stopping Crawler Trading Integrator...")
            
            self.is_running = False
            
            # Stop components
            if self.data_bridge:
                await self.data_bridge.stop()
            
            if self.event_handler:
                await self.event_handler.stop()
            
            if self.crawler_manager:
                await self.crawler_manager.stop_all_crawlers()
            
            logger.info("Crawler Trading Integrator stopped")
            
        except Exception as e:
            logger.error(f"Error stopping Crawler Trading Integrator: {e}")
    
    async def _setup_data_subscriptions(self):
        """Set up data subscriptions from crawlers"""
        for data_type in self.config.required_data_types:
            callback = lambda dt: asyncio.create_task(self._handle_data_update(dt))
            self.data_bridge.subscribe(data_type, callback)
    
    async def _setup_event_subscriptions(self):
        """Set up event subscriptions for critical events"""
        # Subscribe to all event types for internal processing
        for event_type in EventType:
            self.event_handler.subscribe(event_type, self._handle_event)
    
    async def _handle_data_update(self, trading_data: TradingDataPoint):
        """Handle data update from crawlers"""
        try:
            # Update cache
            cache_key = f"{trading_data.symbol}_{trading_data.data_type}"
            self.latest_data_cache[cache_key] = trading_data
            
            # Create appropriate events
            if trading_data.data_type == 'price':
                previous_data = self.latest_data_cache.get(f"{trading_data.symbol}_price")
                await self.event_handler.create_price_event(trading_data, previous_data)
            
            elif trading_data.data_type == 'news':
                await self.event_handler.create_news_event(trading_data)
            
            elif trading_data.data_type == 'sentiment':
                previous_sentiment = 0.0
                if f"{trading_data.symbol}_sentiment" in self.latest_data_cache:
                    previous_sentiment = self.latest_data_cache[
                        f"{trading_data.symbol}_sentiment"
                    ].value.get('overall_sentiment', 0.0)
                
                await self.event_handler.create_sentiment_event(trading_data, previous_sentiment)
            
            elif trading_data.data_type == 'pattern':
                await self.event_handler.create_pattern_event(trading_data)
            
            elif trading_data.data_type == 'economic':
                await self.event_handler.create_economic_event(trading_data)
            
            # Notify subscribers
            await self._notify_data_subscribers(trading_data)
            
            # Update metrics
            self.metrics['data_points_processed'] += 1
            
        except Exception as e:
            logger.error(f"Error handling data update: {e}")
    
    async def _handle_event(self, event: MarketEvent):
        """Handle events from the event handler"""
        try:
            # Update metrics
            self.metrics['events_emitted'] += 1
            
            # Log critical events
            if event.priority == DataPriority.CRITICAL or event.priority == DataPriority.HIGH:
                logger.info(f"High priority event: {event.event_type.value} for {event.symbol}")
            
            # Notify event subscribers
            await self._notify_event_subscribers(event)
            
        except Exception as e:
            logger.error(f"Error handling event: {e}")
    
    async def _notify_data_subscribers(self, trading_data: TradingDataPoint):
        """Notify data subscribers"""
        data_type = trading_data.data_type
        if data_type in self.data_subscribers:
            for callback in self.data_subscribers[data_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(trading_data)
                    else:
                        callback(trading_data)
                    self.metrics['subscriber_notifications'] += 1
                except Exception as e:
                    logger.error(f"Error notifying data subscriber: {e}")
    
    async def _notify_event_subscribers(self, event: MarketEvent):
        """Notify event subscribers"""
        event_type = event.event_type
        if event_type in self.event_subscribers:
            for callback in self.event_subscribers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                    self.metrics['subscriber_notifications'] += 1
                except Exception as e:
                    logger.error(f"Error notifying event subscriber: {e}")
    
    # Public API methods for trading system
    
    async def get_latest_market_data(self, symbol: str) -> Optional[TradingDataPoint]:
        """Get latest market data for a symbol"""
        cache_key = f"{symbol}_market_data"
        
        if cache_key in self.latest_data_cache:
            self.metrics['cache_hits'] += 1
            return self.latest_data_cache[cache_key]
        
        self.metrics['cache_misses'] += 1
        return await self.data_bridge.get_latest_data('market_data', symbol, 1)
    
    async def get_latest_news(self, symbol: str, limit: int = 10) -> List[TradingDataPoint]:
        """Get latest news for a symbol"""
        return await self.data_bridge.get_latest_data('news', symbol, limit)
    
    async def get_latest_sentiment(self, symbol: str) -> Optional[TradingDataPoint]:
        """Get latest sentiment data for a symbol"""
        cache_key = f"{symbol}_sentiment"
        
        if cache_key in self.latest_data_cache:
            return self.latest_data_cache[cache_key]
        
        return await self.data_bridge.get_latest_data('sentiment', symbol, 1)
    
    async def get_latest_patterns(self, symbol: str) -> Optional[TradingDataPoint]:
        """Get latest pattern analysis for a symbol"""
        cache_key = f"{symbol}_patterns"
        
        if cache_key in self.latest_data_cache:
            return self.latest_data_cache[cache_key]
        
        return await self.data_bridge.get_latest_data('patterns', symbol, 1)
    
    async def get_latest_economic_data(self) -> List[TradingDataPoint]:
        """Get latest economic indicators and events"""
        return await self.data_bridge.get_latest_data('economic', None, 50)
    
    async def subscribe_to_data(self, data_type: str, callback: Callable):
        """Subscribe to real-time data updates"""
        if data_type not in self.data_subscribers:
            self.data_subscribers[data_type] = []
        self.data_subscribers[data_type].append(callback)
        logger.info(f"Trading system subscribed to {data_type} data")
    
    async def subscribe_to_events(self, event_type: EventType, callback: Callable):
        """Subscribe to specific events"""
        if event_type not in self.event_subscribers:
            self.event_subscribers[event_type] = []
        self.event_subscribers[event_type].append(callback)
        logger.info(f"Trading system subscribed to {event_type.value} events")
    
    async def get_data_stream(self, data_types: List[str]) -> AsyncIterator[TradingDataPoint]:
        """Get real-time data stream for multiple data types"""
        async for data_point in self.data_bridge.get_data_stream(data_types):
            yield data_point
    
    async def get_events_stream(self, event_types: List[EventType]) -> AsyncIterator[MarketEvent]:
        """Get real-time events stream"""
        while self.is_running:
            try:
                # Get recent events for each type
                for event_type in event_types:
                    recent_events = await self.event_handler.get_recent_events(
                        event_type=event_type,
                        limit=5
                    )
                    
                    for event in recent_events:
                        yield event
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error getting events stream: {e}")
                await asyncio.sleep(5)
    
    async def get_market_summary(self, symbols: List[str]) -> Dict[str, Any]:
        """Get comprehensive market summary for symbols"""
        summary = {}
        
        for symbol in symbols:
            try:
                # Get latest data for each type
                market_data = await self.get_latest_market_data(symbol)
                sentiment_data = await self.get_latest_sentiment(symbol)
                patterns_data = await self.get_latest_patterns(symbol)
                
                # Get recent news
                recent_news = await self.get_latest_news(symbol, limit=3)
                
                # Get recent events
                recent_events = await self.event_handler.get_recent_events(
                    symbol=symbol,
                    limit=10
                )
                
                # Compile summary
                summary[symbol] = {
                    'market_data': market_data.value if market_data else None,
                    'sentiment': sentiment_data.value if sentiment_data else None,
                    'patterns': patterns_data.value if patterns_data else None,
                    'recent_news': [news.value for news in recent_news],
                    'recent_events': [
                        {
                            'type': event.event_type.value,
                            'timestamp': event.timestamp.isoformat(),
                            'priority': event.priority.value,
                            'data': event.data
                        }
                        for event in recent_events
                    ],
                    'last_update': datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error getting market summary for {symbol}: {e}")
                summary[symbol] = {'error': str(e)}
        
        return summary
    
    async def get_trading_signals(self, symbol: str) -> Dict[str, Any]:
        """Get trading signals derived from all data sources"""
        try:
            # Get all latest data
            market_data = await self.get_latest_market_data(symbol)
            sentiment_data = await self.get_latest_sentiment(symbol)
            patterns_data = await self.get_latest_patterns(symbol)
            
            # Get recent high-impact events
            recent_events = await self.event_handler.get_recent_events(
                symbol=symbol,
                since=datetime.utcnow() - timedelta(hours=1),
                limit=20
            )
            
            # Analyze and generate signals
            signals = {
                'symbol': symbol,
                'timestamp': datetime.utcnow().isoformat(),
                'signals': [],
                'confidence': 0.0,
                'data_quality': 0.0
            }
            
            # Price-based signals
            if market_data and market_data.value:
                price_data = market_data.value
                current_price = price_data.get('close', 0)
                
                # Volume spike signal
                volume = price_data.get('volume', 0)
                # This would need historical volume comparison
                
                signals['signals'].append({
                    'type': 'price_update',
                    'value': current_price,
                    'confidence': 0.9,
                    'data': price_data
                })
            
            # Sentiment signals
            if sentiment_data and sentiment_data.value:
                sentiment_score = sentiment_data.value.get('overall_sentiment', 0)
                confidence = sentiment_data.value.get('confidence', 0)
                
                if abs(sentiment_score) > 0.3:
                    direction = 'bullish' if sentiment_score > 0 else 'bearish'
                    signals['signals'].append({
                        'type': 'sentiment',
                        'direction': direction,
                        'score': sentiment_score,
                        'confidence': confidence,
                        'data': sentiment_data.value
                    })
            
            # Pattern signals
            if patterns_data and patterns_data.value:
                patterns = patterns_data.value.get('patterns', {})
                high_confidence_patterns = [
                    name for name, pattern in patterns.items()
                    if pattern.get('confidence', 0) > 0.7
                ]
                
                for pattern_name in high_confidence_patterns:
                    pattern = patterns[pattern_name]
                    signals['signals'].append({
                        'type': 'pattern',
                        'pattern': pattern_name,
                        'direction': pattern.get('direction', 'neutral'),
                        'confidence': pattern.get('confidence', 0),
                        'target_price': pattern.get('target_price'),
                        'stop_loss': pattern.get('stop_loss'),
                        'data': pattern
                    })
            
            # Event-based signals
            high_priority_events = [
                event for event in recent_events
                if event.priority in [DataPriority.HIGH, DataPriority.CRITICAL]
            ]
            
            for event in high_priority_events:
                signals['signals'].append({
                    'type': 'event',
                    'event_type': event.event_type.value,
                    'priority': event.priority.value,
                    'data': event.data
                })
            
            # Calculate overall confidence and data quality
            if signals['signals']:
                confidences = [signal['confidence'] for signal in signals['signals']]
                signals['confidence'] = sum(confidences) / len(confidences)
                signals['data_quality'] = sum(s.data_quality for s in [market_data, sentiment_data, patterns_data] if s) / 3.0
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating trading signals for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get system health status"""
        try:
            health_data = await self.health_monitor.get_system_health()
            crawler_health = await self.crawler_manager.get_overall_health()
            
            return {
                'integration_system': {
                    'is_running': self.is_running,
                    'is_initialized': self.is_initialized,
                    'uptime': datetime.utcnow().isoformat(),  # Would track actual uptime
                    'metrics': self.metrics,
                    'cache_stats': {
                        'cache_size': len(self.latest_data_cache),
                        'subscribers': {
                            'data_subscribers': len(self.data_subscribers),
                            'event_subscribers': len(self.event_subscribers)
                        }
                    }
                },
                'crawlers': crawler_health,
                'data_storage': health_data.get('data_storage', {}),
                'overall_status': 'healthy' if self.is_running else 'stopped'
            }
            
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {
                'overall_status': 'error',
                'error': str(e)
            }
    
    async def update_config(self, new_config: TradingSystemConfig):
        """Update configuration at runtime"""
        try:
            self.config = new_config
            logger.info("Trading integrator configuration updated")
            
            # Reinitialize subscriptions if needed
            await self._setup_data_subscriptions()
            await self._setup_event_subscriptions()
            
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            raise
    
    # Context manager support
    async def __aenter__(self):
        await self.initialize()
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()