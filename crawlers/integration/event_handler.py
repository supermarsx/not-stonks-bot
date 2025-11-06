"""
Crawler Event Handler

Handles real-time events from crawlers and notifies trading systems
about critical market events, data updates, and system status changes.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum

from loguru import logger

from .data_bridge import TradingDataPoint, DataPriority


class EventType(Enum):
    """Types of events that can occur"""
    PRICE_UPDATE = "price_update"
    NEWS_ALERT = "news_alert"
    SENTIMENT_CHANGE = "sentiment_change"
    PATTERN_DETECTED = "pattern_detected"
    ECONOMIC_EVENT = "economic_event"
    SYSTEM_STATUS = "system_status"
    HEALTH_ALERT = "health_alert"
    DATA_QUALITY = "data_quality"


@dataclass
class MarketEvent:
    """Market event for trading system"""
    event_type: EventType
    symbol: str
    timestamp: datetime
    priority: DataPriority
    data: Dict[str, Any]
    event_id: str = field(default_factory=lambda: f"evt_{datetime.utcnow().timestamp()}")
    metadata: Dict[str, Any] = field(default_factory=dict)


class CrawlerEventHandler:
    """Handles events from crawler system and forwards to trading system"""
    
    def __init__(self):
        self.subscribers: Dict[EventType, List[Callable]] = {}
        self.event_history: List[MarketEvent] = []
        self.max_history_size = 10000
        self.alert_thresholds = {
            'sentiment_change': 0.3,
            'price_change': 0.05,  # 5% price change
            'volume_spike': 2.0,   # 2x average volume
            'pattern_confidence': 0.8
        }
        self.is_running = False
        
        # Event aggregation for high-frequency events
        self.event_aggregator: Dict[str, List[MarketEvent]] = {}
        self.aggregation_window = timedelta(seconds=5)
    
    async def start(self):
        """Start the event handler"""
        self.is_running = True
        logger.info("Crawler event handler started")
        
        # Start event processing tasks
        asyncio.create_task(self._process_events())
        asyncio.create_task(self._cleanup_old_events())
    
    async def stop(self):
        """Stop the event handler"""
        self.is_running = False
        logger.info("Crawler event handler stopped")
    
    def subscribe(self, event_type: EventType, callback: Callable):
        """Subscribe to specific event types"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
        logger.info(f"Subscribed to {event_type.value} events")
    
    def unsubscribe(self, event_type: EventType, callback: Callable):
        """Unsubscribe from event type"""
        if event_type in self.subscribers and callback in self.subscribers[event_type]:
            self.subscribers[event_type].remove(callback)
            logger.info(f"Unsubscribed from {event_type.value} events")
    
    async def emit_event(self, event: MarketEvent):
        """Emit an event to subscribers"""
        try:
            # Add to history
            self.event_history.append(event)
            
            # Limit history size
            if len(self.event_history) > self.max_history_size:
                self.event_history = self.event_history[-self.max_history_size:]
            
            # Check alert thresholds
            if self._should_trigger_alert(event):
                event.priority = DataPriority.HIGH
            
            # Aggregate high-frequency events
            await self._aggregate_event(event)
            
            # Broadcast to subscribers
            await self._broadcast_event(event)
            
        except Exception as e:
            logger.error(f"Error emitting event: {e}")
    
    def _should_trigger_alert(self, event: MarketEvent) -> bool:
        """Determine if event should trigger alert"""
        try:
            if event.event_type == EventType.SENTIMENT_CHANGE:
                sentiment_change = abs(event.data.get('sentiment_change', 0))
                return sentiment_change > self.alert_thresholds['sentiment_change']
            
            elif event.event_type == EventType.PRICE_UPDATE:
                price_change = abs(event.data.get('change_percent', 0))
                return price_change > self.alert_thresholds['price_change']
            
            elif event.event_type == EventType.PATTERN_DETECTED:
                confidence = event.data.get('confidence', 0)
                return confidence > self.alert_thresholds['pattern_confidence']
            
            elif event.event_type == EventType.NEWS_ALERT:
                relevance = event.data.get('relevance_score', 0)
                sentiment = abs(event.data.get('sentiment_score', 0))
                return relevance > 0.8 or sentiment > 0.7
            
            elif event.event_type == EventType.ECONOMIC_EVENT:
                impact = event.data.get('impact', '').lower()
                return impact in ['high', 'critical']
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking alert thresholds: {e}")
            return False
    
    async def _aggregate_event(self, event: MarketEvent):
        """Aggregate high-frequency events"""
        aggregation_key = f"{event.event_type.value}_{event.symbol}"
        
        if aggregation_key not in self.event_aggregator:
            self.event_aggregator[aggregation_key] = []
        
        # Add event to aggregation
        self.event_aggregator[aggregation_key].append(event)
        
        # Remove old events outside aggregation window
        cutoff_time = datetime.utcnow() - self.aggregation_window
        self.event_aggregator[aggregation_key] = [
            evt for evt in self.event_aggregator[aggregation_key]
            if evt.timestamp > cutoff_time
        ]
        
        # Create aggregated event if too many events in window
        if len(self.event_aggregator[aggregation_key]) > 10:
            aggregated_event = self._create_aggregated_event(
                aggregation_key, self.event_aggregator[aggregation_key]
            )
            if aggregated_event:
                await self._broadcast_event(aggregated_event)
            
            # Clear aggregated events
            self.event_aggregator[aggregation_key] = []
    
    def _create_aggregated_event(self, key: str, events: List[MarketEvent]) -> Optional[MarketEvent]:
        """Create aggregated event from multiple events"""
        if not events:
            return None
        
        event_type_str, symbol = key.split('_', 1)
        event_type = EventType(event_type_str)
        
        # Calculate aggregated data
        aggregated_data = {
            'event_count': len(events),
            'time_range': {
                'start': min(evt.timestamp for evt in events).isoformat(),
                'end': max(evt.timestamp for evt in events).isoformat()
            },
            'original_events': [evt.data for evt in events]
        }
        
        # Add type-specific aggregation
        if event_type == EventType.PRICE_UPDATE:
            price_changes = [evt.data.get('change_percent', 0) for evt in events]
            aggregated_data.update({
                'avg_price_change': sum(price_changes) / len(price_changes),
                'max_price_change': max(price_changes),
                'min_price_change': min(price_changes)
            })
        
        elif event_type == EventType.SENTIMENT_CHANGE:
            sentiment_changes = [evt.data.get('sentiment_change', 0) for evt in events]
            aggregated_data.update({
                'avg_sentiment_change': sum(sentiment_changes) / len(sentiment_changes),
                'sentiment_range': {
                    'max': max(sentiment_changes),
                    'min': min(sentiment_changes)
                }
            })
        
        return MarketEvent(
            event_type=event_type,
            symbol=symbol,
            timestamp=datetime.utcnow(),
            priority=DataPriority.MEDIUM,  # Aggregated events are medium priority
            data=aggregated_data,
            event_id=f"agg_{datetime.utcnow().timestamp()}",
            metadata={'aggregated': True, 'original_count': len(events)}
        )
    
    async def _broadcast_event(self, event: MarketEvent):
        """Broadcast event to subscribers"""
        event_type = event.event_type
        
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                except Exception as e:
                    logger.error(f"Error broadcasting event: {e}")
    
    async def _process_events(self):
        """Process events (placeholder for future event processing logic)"""
        while self.is_running:
            try:
                # Here you could add complex event correlation logic
                # Event pattern recognition
                # Multi-event analysis
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing events: {e}")
                await asyncio.sleep(5)
    
    async def _cleanup_old_events(self):
        """Clean up old events from history"""
        while self.is_running:
            try:
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                
                # Clean event history
                self.event_history = [
                    evt for evt in self.event_history
                    if evt.timestamp > cutoff_time
                ]
                
                # Clean event aggregator
                for key in list(self.event_aggregator.keys()):
                    self.event_aggregator[key] = [
                        evt for evt in self.event_aggregator[key]
                        if evt.timestamp > cutoff_time
                    ]
                    
                    if not self.event_aggregator[key]:
                        del self.event_aggregator[key]
                
                await asyncio.minute(60)  # Cleanup every hour
                
            except Exception as e:
                logger.error(f"Error cleaning up events: {e}")
                await asyncio.minute(60)
    
    # Event creation methods for different data types
    
    async def create_price_event(self, trading_data: TradingDataPoint, 
                               previous_data: Optional[Dict] = None):
        """Create price update event"""
        try:
            current_price = trading_data.value.get('close', 0)
            previous_price = previous_data.get('close', current_price) if previous_data else current_price
            
            price_change = current_price - previous_price
            change_percent = (price_change / previous_price * 100) if previous_price != 0 else 0
            
            event = MarketEvent(
                event_type=EventType.PRICE_UPDATE,
                symbol=trading_data.symbol,
                timestamp=trading_data.timestamp,
                priority=DataPriority.CRITICAL,
                data={
                    'current_price': current_price,
                    'previous_price': previous_price,
                    'change': price_change,
                    'change_percent': change_percent,
                    'volume': trading_data.value.get('volume', 0),
                    'high': trading_data.value.get('high', 0),
                    'low': trading_data.value.get('low', 0),
                    'data_source': trading_data.metadata.get('source')
                }
            )
            
            await self.emit_event(event)
            
        except Exception as e:
            logger.error(f"Error creating price event: {e}")
    
    async def create_news_event(self, trading_data: TradingDataPoint):
        """Create news alert event"""
        try:
            event = MarketEvent(
                event_type=EventType.NEWS_ALERT,
                symbol=trading_data.symbol,
                timestamp=trading_data.timestamp,
                priority=trading_data.priority,
                data={
                    'headline': trading_data.value.get('headline', ''),
                    'summary': trading_data.value.get('summary', ''),
                    'sentiment_score': trading_data.value.get('sentiment_score', 0),
                    'relevance_score': trading_data.value.get('relevance_score', 0),
                    'source': trading_data.value.get('source', ''),
                    'category': trading_data.value.get('category', 'general'),
                    'url': trading_data.metadata.get('url')
                }
            )
            
            await self.emit_event(event)
            
        except Exception as e:
            logger.error(f"Error creating news event: {e}")
    
    async def create_sentiment_event(self, trading_data: TradingDataPoint, 
                                   previous_sentiment: float = 0.0):
        """Create sentiment change event"""
        try:
            current_sentiment = trading_data.value.get('overall_sentiment', 0)
            sentiment_change = current_sentiment - previous_sentiment
            
            event = MarketEvent(
                event_type=EventType.SENTIMENT_CHANGE,
                symbol=trading_data.symbol,
                timestamp=trading_data.timestamp,
                priority=trading_data.priority,
                data={
                    'current_sentiment': current_sentiment,
                    'previous_sentiment': previous_sentiment,
                    'sentiment_change': sentiment_change,
                    'confidence': trading_data.value.get('confidence', 0),
                    'volume': trading_data.value.get('volume', 0),
                    'sources': trading_data.metadata.get('sources', [])
                }
            )
            
            await self.emit_event(event)
            
        except Exception as e:
            logger.error(f"Error creating sentiment event: {e}")
    
    async def create_pattern_event(self, trading_data: TradingDataPoint):
        """Create pattern detection event"""
        try:
            patterns = trading_data.value.get('patterns', {})
            high_confidence_patterns = [
                name for name, pattern in patterns.items()
                if pattern.get('confidence', 0) > self.alert_thresholds['pattern_confidence']
            ]
            
            if high_confidence_patterns:
                event = MarketEvent(
                    event_type=EventType.PATTERN_DETECTED,
                    symbol=trading_data.symbol,
                    timestamp=trading_data.timestamp,
                    priority=DataPriority.MEDIUM,
                    data={
                        'patterns': {name: patterns[name] for name in high_confidence_patterns},
                        'pattern_count': len(high_confidence_patterns),
                        'timeframe': trading_data.metadata.get('timeframe', '1d'),
                        'indicators': trading_data.value.get('indicators', {}),
                        'signals': trading_data.value.get('pattern_signals', [])
                    }
                )
                
                await self.emit_event(event)
            
        except Exception as e:
            logger.error(f"Error creating pattern event: {e}")
    
    async def create_economic_event(self, trading_data: TradingDataPoint):
        """Create economic event"""
        try:
            events = trading_data.value.get('upcoming_events', [])
            high_impact_events = [
                event for event in events
                if event.get('impact', '').lower() in ['high', 'critical']
            ]
            
            if high_impact_events:
                event = MarketEvent(
                    event_type=EventType.ECONOMIC_EVENT,
                    symbol=trading_data.symbol,
                    timestamp=trading_data.timestamp,
                    priority=DataPriority.HIGH,
                    data={
                        'events': high_impact_events,
                        'event_count': len(high_impact_events),
                        'indicators': trading_data.value.get('indicators', {}),
                        'market_impact_score': trading_data.value.get('market_impact_score', 0),
                        'central_bank_signals': trading_data.value.get('central_bank_signals', {})
                    }
                )
                
                await self.emit_event(event)
            
        except Exception as e:
            logger.error(f"Error creating economic event: {e}")
    
    async def get_recent_events(self, event_type: Optional[EventType] = None, 
                              symbol: Optional[str] = None,
                              since: Optional[datetime] = None,
                              limit: int = 100) -> List[MarketEvent]:
        """Get recent events with optional filtering"""
        events = self.event_history
        
        # Apply filters
        if event_type:
            events = [evt for evt in events if evt.event_type == event_type]
        
        if symbol:
            events = [evt for evt in events if evt.symbol == symbol]
        
        if since:
            events = [evt for evt in events if evt.timestamp > since]
        
        # Sort by timestamp (most recent first) and limit
        events.sort(key=lambda x: x.timestamp, reverse=True)
        return events[:limit]
    
    async def get_event_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get event statistics for the last N hours"""
        since = datetime.utcnow() - timedelta(hours=hours)
        recent_events = [evt for evt in self.event_history if evt.timestamp > since]
        
        stats = {
            'total_events': len(recent_events),
            'events_by_type': {},
            'events_by_symbol': {},
            'events_by_priority': {},
            'time_range': {
                'start': recent_events[-1].timestamp.isoformat() if recent_events else None,
                'end': recent_events[0].timestamp.isoformat() if recent_events else None
            }
        }
        
        # Count by type
        for event in recent_events:
            event_type = event.event_type.value
            stats['events_by_type'][event_type] = stats['events_by_type'].get(event_type, 0) + 1
        
        # Count by symbol
        for event in recent_events:
            symbol = event.symbol
            stats['events_by_symbol'][symbol] = stats['events_by_symbol'].get(symbol, 0) + 1
        
        # Count by priority
        for event in recent_events:
            priority = event.priority.value
            stats['events_by_priority'][priority] = stats['events_by_priority'].get(priority, 0) + 1
        
        return stats