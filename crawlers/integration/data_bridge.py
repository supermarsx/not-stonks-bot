"""
Crawler Data Bridge

Provides data transformation and bridging between the crawler system and
the trading orchestrator's data models.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, AsyncIterator
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum

from loguru import logger

from ..storage.data_storage import DataStorage
from ..monitoring.health_monitor import HealthMonitor


class DataPriority(Enum):
    """Data priority levels for trading decisions"""
    CRITICAL = "critical"      # Real-time prices, order execution data
    HIGH = "high"             # News events, social sentiment
    MEDIUM = "medium"         # Technical indicators, patterns
    LOW = "low"               # Historical data, analytics


@dataclass
class TradingDataPoint:
    """Unified data point for trading system"""
    symbol: str
    timestamp: datetime
    data_type: str           # 'price', 'news', 'sentiment', 'pattern', 'economic'
    priority: DataPriority
    value: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "crawler"
    quality_score: float = 1.0  # 0.0 to 1.0


@dataclass
class MarketDataEvent:
    """Market data event for trading system"""
    event_type: str
    symbol: str
    timestamp: datetime
    data: Dict[str, Any]
    priority: DataPriority


class CrawlerDataBridge:
    """Bridges crawler data to trading system formats"""
    
    def __init__(self, data_storage: DataStorage, health_monitor: HealthMonitor):
        self.data_storage = data_storage
        self.health_monitor = health_monitor
        self.subscribers: Dict[str, List[Callable]] = {}
        self.data_queues: Dict[str, asyncio.Queue] = {}
        self.is_running = False
        
        # Data transformation functions
        self.transformers = {
            'market_data': self._transform_market_data,
            'news': self._transform_news_data,
            'social_sentiment': self._transform_sentiment_data,
            'technical_patterns': self._transform_pattern_data,
            'economic_indicators': self._transform_economic_data
        }
    
    async def start(self):
        """Start the data bridge"""
        self.is_running = True
        logger.info("Crawler data bridge started")
        
        # Start data processing tasks
        asyncio.create_task(self._process_market_data())
        asyncio.create_task(self._process_news_data())
        asyncio.create_task(self._process_social_data())
        asyncio.create_task(self._process_pattern_data())
        asyncio.create_task(self._process_economic_data())
    
    async def stop(self):
        """Stop the data bridge"""
        self.is_running = False
        logger.info("Crawler data bridge stopped")
    
    def subscribe(self, data_type: str, callback: Callable):
        """Subscribe to specific data types"""
        if data_type not in self.subscribers:
            self.subscribers[data_type] = []
        self.subscribers[data_type].append(callback)
        logger.info(f"Subscribed to {data_type} data")
    
    async def _process_market_data(self):
        """Process market data from crawlers"""
        while self.is_running:
            try:
                # Get latest market data from storage
                market_data = await self.data_storage.get_latest_data(
                    data_type='market_data',
                    limit=100
                )
                
                for data_point in market_data:
                    trading_data = await self._transform_market_data(data_point)
                    await self._broadcast_data(trading_data)
                
                await asyncio.sleep(1)  # Process every second
                
            except Exception as e:
                logger.error(f"Error processing market data: {e}")
                await asyncio.sleep(5)
    
    async def _process_news_data(self):
        """Process news data from crawlers"""
        while self.is_running:
            try:
                # Get latest news data from storage
                news_data = await self.data_storage.get_latest_data(
                    data_type='news',
                    limit=50
                )
                
                for data_point in news_data:
                    trading_data = await self._transform_news_data(data_point)
                    await self._broadcast_data(trading_data)
                
                await asyncio.sleep(5)  # Process every 5 seconds
                
            except Exception as e:
                logger.error(f"Error processing news data: {e}")
                await asyncio.sleep(10)
    
    async def _process_social_data(self):
        """Process social media sentiment data"""
        while self.is_running:
            try:
                # Get latest social sentiment data
                social_data = await self.data_storage.get_latest_data(
                    data_type='social_sentiment',
                    limit=50
                )
                
                for data_point in social_data:
                    trading_data = await self._transform_sentiment_data(data_point)
                    await self._broadcast_data(trading_data)
                
                await asyncio.sleep(10)  # Process every 10 seconds
                
            except Exception as e:
                logger.error(f"Error processing social data: {e}")
                await asyncio.sleep(15)
    
    async def _process_pattern_data(self):
        """Process technical pattern data"""
        while self.is_running:
            try:
                # Get latest pattern data
                pattern_data = await self.data_storage.get_latest_data(
                    data_type='patterns',
                    limit=25
                )
                
                for data_point in pattern_data:
                    trading_data = await self._transform_pattern_data(data_point)
                    await self._broadcast_data(trading_data)
                
                await asyncio.sleep(30)  # Process every 30 seconds
                
            except Exception as e:
                logger.error(f"Error processing pattern data: {e}")
                await asyncio.sleep(60)
    
    async def _process_economic_data(self):
        """Process economic indicator data"""
        while self.is_running:
            try:
                # Get latest economic data
                economic_data = await self.data_storage.get_latest_data(
                    data_type='economic',
                    limit=25
                )
                
                for data_point in economic_data:
                    trading_data = await self._transform_economic_data(data_point)
                    await self._broadcast_data(trading_data)
                
                await asyncio.sleep(300)  # Process every 5 minutes
                
            except Exception as e:
                logger.error(f"Error processing economic data: {e}")
                await asyncio.sleep(600)
    
    async def _transform_market_data(self, data_point: Dict[str, Any]) -> TradingDataPoint:
        """Transform market data to trading format"""
        try:
            symbol = data_point.get('symbol', 'UNKNOWN')
            price_data = data_point.get('price_data', {})
            
            # Create unified trading data point
            trading_data = TradingDataPoint(
                symbol=symbol,
                timestamp=datetime.fromisoformat(data_point.get('timestamp', datetime.utcnow().isoformat())),
                data_type='price',
                priority=DataPriority.CRITICAL,
                value={
                    'open': float(price_data.get('open', 0)),
                    'high': float(price_data.get('high', 0)),
                    'low': float(price_data.get('low', 0)),
                    'close': float(price_data.get('close', 0)),
                    'volume': float(price_data.get('volume', 0))
                },
                metadata={
                    'data_source': data_point.get('source', 'unknown'),
                    'interval': data_point.get('interval', '1m'),
                    'exchange': data_point.get('exchange', 'unknown')
                },
                quality_score=0.95  # High quality for market data
            )
            
            return trading_data
            
        except Exception as e:
            logger.error(f"Error transforming market data: {e}")
            # Return default trading data point
            return TradingDataPoint(
                symbol='UNKNOWN',
                timestamp=datetime.utcnow(),
                data_type='price',
                priority=DataPriority.MEDIUM,
                value={},
                quality_score=0.0
            )
    
    async def _transform_news_data(self, data_point: Dict[str, Any]) -> TradingDataPoint:
        """Transform news data to trading format"""
        try:
            symbol = data_point.get('symbol', 'GENERAL')
            content = data_point.get('content', {})
            
            # Extract sentiment and relevance
            sentiment_score = content.get('sentiment_score', 0.0)
            relevance_score = content.get('relevance_score', 0.0)
            
            # Determine priority based on sentiment and relevance
            if abs(sentiment_score) > 0.7 or relevance_score > 0.8:
                priority = DataPriority.HIGH
            else:
                priority = DataPriority.MEDIUM
            
            trading_data = TradingDataPoint(
                symbol=symbol,
                timestamp=datetime.fromisoformat(data_point.get('timestamp', datetime.utcnow().isoformat())),
                data_type='news',
                priority=priority,
                value={
                    'headline': content.get('headline', ''),
                    'summary': content.get('summary', ''),
                    'sentiment_score': sentiment_score,
                    'relevance_score': relevance_score,
                    'source': content.get('source', ''),
                    'category': content.get('category', 'general')
                },
                metadata={
                    'article_id': content.get('id'),
                    'url': content.get('url'),
                    'entities': content.get('entities', [])
                },
                quality_score=relevance_score
            )
            
            return trading_data
            
        except Exception as e:
            logger.error(f"Error transforming news data: {e}")
            return TradingDataPoint(
                symbol='GENERAL',
                timestamp=datetime.utcnow(),
                data_type='news',
                priority=DataPriority.LOW,
                value={},
                quality_score=0.0
            )
    
    async def _transform_sentiment_data(self, data_point: Dict[str, Any]) -> TradingDataPoint:
        """Transform social sentiment data to trading format"""
        try:
            symbol = data_point.get('symbol', 'GENERAL')
            sentiment_data = data_point.get('sentiment_data', {})
            
            # Calculate overall sentiment score
            social_sentiment = sentiment_data.get('social_sentiment', 0.0)
            news_sentiment = sentiment_data.get('news_sentiment', 0.0)
            
            # Weighted average sentiment
            overall_sentiment = (social_sentiment * 0.6 + news_sentiment * 0.4)
            
            # Determine priority based on sentiment strength
            if abs(overall_sentiment) > 0.5:
                priority = DataPriority.HIGH
            else:
                priority = DataPriority.MEDIUM
            
            trading_data = TradingDataPoint(
                symbol=symbol,
                timestamp=datetime.fromisoformat(data_point.get('timestamp', datetime.utcnow().isoformat())),
                data_type='sentiment',
                priority=priority,
                value={
                    'overall_sentiment': overall_sentiment,
                    'social_sentiment': social_sentiment,
                    'news_sentiment': news_sentiment,
                    'confidence': sentiment_data.get('confidence', 0.0),
                    'volume': sentiment_data.get('volume', 0),
                    'trending_keywords': sentiment_data.get('trending_keywords', [])
                },
                metadata={
                    'sources': sentiment_data.get('sources', []),
                    'sentiment_distribution': sentiment_data.get('distribution', {}),
                    'influencer_mentions': sentiment_data.get('influencer_mentions', 0)
                },
                quality_score=sentiment_data.get('confidence', 0.0)
            )
            
            return trading_data
            
        except Exception as e:
            logger.error(f"Error transforming sentiment data: {e}")
            return TradingDataPoint(
                symbol='GENERAL',
                timestamp=datetime.utcnow(),
                data_type='sentiment',
                priority=DataPriority.LOW,
                value={},
                quality_score=0.0
            )
    
    async def _transform_pattern_data(self, data_point: Dict[str, Any]) -> TradingDataPoint:
        """Transform technical pattern data to trading format"""
        try:
            symbol = data_point.get('symbol', 'UNKNOWN')
            pattern_data = data_point.get('pattern_data', {})
            
            # Extract pattern information
            patterns = pattern_data.get('patterns', {})
            indicators = pattern_data.get('indicators', {})
            
            # Determine priority based on pattern significance
            high_confidence_patterns = [p for p in patterns.values() if p.get('confidence', 0) > 0.7]
            
            if high_confidence_patterns:
                priority = DataPriority.MEDIUM
            else:
                priority = DataPriority.LOW
            
            trading_data = TradingDataPoint(
                symbol=symbol,
                timestamp=datetime.fromisoformat(data_point.get('timestamp', datetime.utcnow().isoformat())),
                data_type='pattern',
                priority=priority,
                value={
                    'patterns': patterns,
                    'indicators': indicators,
                    'pattern_signals': pattern_data.get('signals', []),
                    'support_resistance': pattern_data.get('support_resistance', {})
                },
                metadata={
                    'timeframe': pattern_data.get('timeframe', '1d'),
                    'pattern_count': len(patterns),
                    'bullish_patterns': len([p for p in patterns.values() if p.get('direction') == 'bullish']),
                    'bearish_patterns': len([p for p in patterns.values() if p.get('direction') == 'bearish'])
                },
                quality_score=sum(p.get('confidence', 0) for p in patterns.values()) / max(len(patterns), 1)
            )
            
            return trading_data
            
        except Exception as e:
            logger.error(f"Error transforming pattern data: {e}")
            return TradingDataPoint(
                symbol='UNKNOWN',
                timestamp=datetime.utcnow(),
                data_type='pattern',
                priority=DataPriority.LOW,
                value={},
                quality_score=0.0
            )
    
    async def _transform_economic_data(self, data_point: Dict[str, Any]) -> TradingDataPoint:
        """Transform economic indicator data to trading format"""
        try:
            symbol = data_point.get('symbol', 'MARKET')
            economic_data = data_point.get('economic_data', {})
            
            # Extract economic indicators
            indicators = economic_data.get('indicators', {})
            events = economic_data.get('events', [])
            
            # Determine priority based on event importance
            high_impact_events = [e for e in events if e.get('impact', '').lower() in ['high', 'critical']]
            
            if high_impact_events:
                priority = DataPriority.HIGH
            else:
                priority = DataPriority.LOW
            
            trading_data = TradingDataPoint(
                symbol=symbol,
                timestamp=datetime.fromisoformat(data_point.get('timestamp', datetime.utcnow().isoformat())),
                data_type='economic',
                priority=priority,
                value={
                    'indicators': indicators,
                    'upcoming_events': events,
                    'central_bank_signals': economic_data.get('central_bank_signals', {}),
                    'market_impact_score': economic_data.get('market_impact_score', 0.0)
                },
                metadata={
                    'data_source': economic_data.get('source', 'unknown'),
                    'event_count': len(events),
                    'high_impact_count': len(high_impact_events),
                    'last_updated': economic_data.get('last_updated')
                },
                quality_score=0.9  # Economic data is generally reliable
            )
            
            return trading_data
            
        except Exception as e:
            logger.error(f"Error transforming economic data: {e}")
            return TradingDataPoint(
                symbol='MARKET',
                timestamp=datetime.utcnow(),
                data_type='economic',
                priority=DataPriority.LOW,
                value={},
                quality_score=0.0
            )
    
    async def _broadcast_data(self, trading_data: TradingDataPoint):
        """Broadcast data to subscribers"""
        data_type = trading_data.data_type
        
        if data_type in self.subscribers:
            for callback in self.subscribers[data_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(trading_data)
                    else:
                        callback(trading_data)
                except Exception as e:
                    logger.error(f"Error broadcasting data to subscriber: {e}")
    
    async def get_latest_data(self, data_type: str, symbol: Optional[str] = None, 
                            limit: int = 100) -> List[TradingDataPoint]:
        """Get latest trading data"""
        try:
            # Get data from storage
            raw_data = await self.data_storage.get_latest_data(
                data_type=data_type,
                symbol=symbol,
                limit=limit
            )
            
            # Transform to trading format
            trading_data = []
            for data_point in raw_data:
                if data_type in self.transformers:
                    transformed = await self.transformers[data_type](data_point)
                    trading_data.append(transformed)
            
            return trading_data
            
        except Exception as e:
            logger.error(f"Error getting latest data: {e}")
            return []
    
    async def get_data_stream(self, data_types: List[str]) -> AsyncIterator[TradingDataPoint]:
        """Get real-time data stream for specific data types"""
        while self.is_running:
            for data_type in data_types:
                try:
                    latest_data = await self.get_latest_data(data_type, limit=1)
                    if latest_data:
                        yield latest_data[0]
                except Exception as e:
                    logger.error(f"Error getting data stream for {data_type}: {e}")
            
            await asyncio.sleep(1)  # Check every second