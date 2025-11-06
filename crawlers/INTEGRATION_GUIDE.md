# Crawler Trading System Integration Guide

## Overview

The **Crawler Trading System Integration** provides seamless connectivity between the comprehensive market data crawler system and trading orchestrator platforms. This integration enables real-time data flow from crawlers to trading strategies, risk management, and decision-making systems.

## 🎯 Key Features

### Real-Time Data Integration
- **Market Data Stream**: Real-time prices, OHLCV data, volume analysis
- **News Feed Integration**: Financial news with sentiment analysis
- **Social Sentiment**: Twitter, Reddit, StockTwits sentiment tracking
- **Technical Patterns**: Chart pattern recognition and signal generation
- **Economic Indicators**: Central bank announcements, economic calendar

### Event-Driven Architecture
- **Critical Events**: Price updates, high-impact news, system alerts
- **Event Aggregation**: High-frequency event processing and correlation
- **Alert System**: Configurable thresholds and priority handling
- **Event History**: Comprehensive event logging and analysis

### Trading System Compatibility
- **Unified Data Model**: Standardized data format for all sources
- **Performance Optimization**: Caching, prioritization, and resource management
- **Health Monitoring**: System status, performance metrics, quality scoring
- **Error Recovery**: Robust error handling and fallback mechanisms

## 🏗️ Architecture

### Core Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Crawlers      │───▶│  Data Bridge     │───▶│ Trading System  │
│   System        │    │  & Event Handler │    │  Orchestrator   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Data Storage    │    │ Event Processing │    │  Trading        │
│ & Management    │    │ & Correlation    │    │  Strategies     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Integration Flow

1. **Data Collection**: Crawlers gather data from multiple sources
2. **Data Transformation**: Raw data transformed to trading-compatible format
3. **Event Processing**: Critical events identified and prioritized
4. **Real-time Streaming**: Data and events streamed to trading system
5. **Signal Generation**: Trading signals generated from integrated data
6. **Execution**: Signals executed through broker APIs

## 🚀 Quick Start

### Basic Integration

```python
import asyncio
from crawlers.integration import CrawlerTradingIntegrator, TradingSystemConfig

async def basic_integration():
    # Configure the integration
    config = TradingSystemConfig(
        symbols_to_monitor=['AAPL', 'GOOGL', 'MSFT'],
        required_data_types=['market_data', 'news', 'sentiment'],
        enable_real_time_events=True
    )
    
    # Create and start integrator
    async with CrawlerTradingIntegrator(config) as integrator:
        # Get latest market data
        market_data = await integrator.get_latest_market_data('AAPL')
        print(f"AAPL Price: ${market_data.value['close']}")
        
        # Subscribe to price updates
        async def handle_price_update(event):
            print(f"Price update: {event.symbol} = ${event.data['current_price']}")
        
        await integrator.subscribe_to_events(EventType.PRICE_UPDATE, handle_price_update)
        
        # Run for 60 seconds
        await asyncio.sleep(60)

# Run the integration
asyncio.run(basic_integration())
```

### Real-Time Data Streaming

```python
async def realtime_streaming():
    async with CrawlerTradingIntegrator() as integrator:
        # Subscribe to data streams
        await integrator.subscribe_to_data('market_data', handle_market_data)
        await integrator.subscribe_to_data('sentiment', handle_sentiment)
        
        # Process events
        async for data_point in integrator.get_data_stream(['market_data', 'sentiment']):
            print(f"Data: {data_point.symbol} - {data_point.data_type}")

async def handle_market_data(trading_data):
    if trading_data.data_type == 'market_data':
        price = trading_data.value.get('close', 0)
        volume = trading_data.value.get('volume', 0)
        print(f"Market: {trading_data.symbol} ${price} (vol: {volume:,})")

async def handle_sentiment(trading_data):
    sentiment = trading_data.value.get('overall_sentiment', 0)
    confidence = trading_data.value.get('confidence', 0)
    print(f"Sentiment: {trading_data.symbol} {sentiment:+.2f} (conf: {confidence:.2f})")
```

### Trading Signal Generation

```python
async def generate_trading_signals():
    async with CrawlerTradingIntegrator() as integrator:
        symbols = ['AAPL', 'GOOGL', 'TSLA']
        
        # Get comprehensive market summary
        summary = await integrator.get_market_summary(symbols)
        
        # Generate trading signals for each symbol
        for symbol in symbols:
            signals = await integrator.get_trading_signals(symbol)
            
            print(f"\n🎯 Signals for {symbol}:")
            print(f"Confidence: {signals['confidence']:.2f}")
            
            for signal in signals['signals']:
                signal_type = signal['type']
                confidence = signal['confidence']
                print(f"  - {signal_type}: {confidence:.2f}")
                
                # Execute high-confidence signals
                if confidence > 0.8:
                    await execute_trade_signal(symbol, signal)

async def execute_trade_signal(symbol, signal):
    # Integration with broker API would go here
    print(f"🚀 Executing {signal['type']} signal for {symbol}")
```

## 📊 Data Types

### Market Data
```python
{
    "symbol": "AAPL",
    "timestamp": "2025-11-06T06:46:43Z",
    "data_type": "price",
    "priority": "critical",
    "value": {
        "open": 150.25,
        "high": 152.10,
        "low": 149.80,
        "close": 151.75,
        "volume": 45678923
    },
    "metadata": {
        "source": "alpha_vantage",
        "interval": "1m",
        "exchange": "NASDAQ"
    },
    "quality_score": 0.95
}
```

### News Data
```python
{
    "symbol": "AAPL",
    "timestamp": "2025-11-06T06:46:43Z",
    "data_type": "news",
    "priority": "high",
    "value": {
        "headline": "Apple Reports Strong Q4 Earnings",
        "summary": "Apple beats earnings expectations...",
        "sentiment_score": 0.75,
        "relevance_score": 0.92,
        "source": "Reuters",
        "category": "earnings"
    },
    "metadata": {
        "article_id": "abc123",
        "url": "https://...",
        "entities": ["AAPL", "Tim Cook", "iPhone"]
    },
    "quality_score": 0.92
}
```

### Sentiment Data
```python
{
    "symbol": "AAPL",
    "timestamp": "2025-11-06T06:46:43Z",
    "data_type": "sentiment",
    "priority": "medium",
    "value": {
        "overall_sentiment": 0.65,
        "social_sentiment": 0.72,
        "news_sentiment": 0.58,
        "confidence": 0.85,
        "volume": 1247,
        "trending_keywords": ["earnings", "iPhone", "growth"]
    },
    "metadata": {
        "sources": ["twitter", "reddit", "stocktwits"],
        "sentiment_distribution": {
            "positive": 0.65,
            "neutral": 0.25,
            "negative": 0.10
        },
        "influencer_mentions": 12
    },
    "quality_score": 0.85
}
```

### Pattern Data
```python
{
    "symbol": "AAPL",
    "timestamp": "2025-11-06T06:46:43Z",
    "data_type": "pattern",
    "priority": "medium",
    "value": {
        "patterns": {
            "head_and_shoulders": {
                "confidence": 0.85,
                "direction": "bearish",
                "target_price": 145.50,
                "stop_loss": 155.00
            },
            "ascending_triangle": {
                "confidence": 0.78,
                "direction": "bullish",
                "target_price": 160.00,
                "stop_loss": 148.00
            }
        },
        "indicators": {
            "rsi": 72.5,
            "macd": 0.45,
            "bb_position": "upper"
        },
        "pattern_signals": ["breakout_possible", "volume_increase"]
    },
    "metadata": {
        "timeframe": "1d",
        "pattern_count": 2,
        "bullish_patterns": 1,
        "bearish_patterns": 1
    },
    "quality_score": 0.82
}
```

## 🎛️ Configuration

### TradingSystemConfig Options

```python
config = TradingSystemConfig(
    # Data Requirements
    required_data_types=['market_data', 'news', 'sentiment', 'patterns', 'economic'],
    symbols_to_monitor=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'SPY'],
    
    # Update Frequencies (seconds)
    update_frequencies={
        'market_data': 1,      # Real-time prices
        'news': 5,             # News updates
        'sentiment': 10,       # Social sentiment
        'patterns': 30,        # Technical patterns
        'economic': 300        # Economic indicators
    },
    
    # Event Handling
    enable_real_time_events=True,
    event_buffer_size=1000,
    
    # Alert Thresholds
    alert_thresholds={
        'sentiment_change': 0.3,      # 30% sentiment change
        'price_change': 0.05,         # 5% price change
        'pattern_confidence': 0.8     # 80% pattern confidence
    },
    
    # Performance Settings
    max_concurrent_requests=50,
    data_retention_hours=168,        # 1 week
    cache_size_limit=10000
)
```

## 🔧 Event System

### Event Types

```python
from crawlers.integration.event_handler import EventType

# Price Events
EventType.PRICE_UPDATE           # Real-time price changes
EventType.SENTIMENT_CHANGE       # Significant sentiment shifts
EventType.PATTERN_DETECTED       # Technical pattern recognition
EventType.ECONOMIC_EVENT         # Economic announcements

# News Events
EventType.NEWS_ALERT             # High-impact news
EventType.SYSTEM_STATUS          # System health changes
EventType.HEALTH_ALERT           # Health monitoring alerts
EventType.DATA_QUALITY           # Data quality issues
```

### Event Subscription

```python
async def setup_event_handlers():
    async with CrawlerTradingIntegrator() as integrator:
        # Subscribe to specific events
        await integrator.subscribe_to_events(EventType.PRICE_UPDATE, handle_price)
        await integrator.subscribe_to_events(EventType.NEWS_ALERT, handle_news)
        await integrator.subscribe_to_events(EventType.SENTIMENT_CHANGE, handle_sentiment)
        
        # Subscribe to all data types
        await integrator.subscribe_to_data('market_data', handle_market)
        await integrator.subscribe_to_data('news', handle_news_data)
        await integrator.subscribe_to_data('sentiment', handle_sentiment_data)

async def handle_price(event):
    symbol = event.symbol
    price = event.data['current_price']
    change = event.data['change_percent']
    
    print(f"💹 {symbol}: ${price} ({change:+.2f}%)")
    
    # Execute trading logic
    if abs(change) > 2.0:
        await analyze_opportunity(symbol, price, change)

async def handle_news(event):
    headline = event.data['headline']
    sentiment = event.data['sentiment_score']
    relevance = event.data['relevance_score']
    
    print(f"📰 {event.symbol}: {headline}")
    print(f"   Sentiment: {sentiment:+.2f}, Relevance: {relevance:.2f}")
    
    # Process news impact
    if relevance > 0.8:
        await process_high_impact_news(event.symbol, event.data)
```

## 📈 Trading Strategies Integration

### Simple Momentum Strategy

```python
class MomentumStrategy:
    def __init__(self, integrator):
        self.integrator = integrator
        self.threshold = 0.03  # 3% price change
    
    async def check_signals(self, symbol):
        # Get latest data
        market_data = await self.integrator.get_latest_market_data(symbol)
        sentiment_data = await self.integrator.get_latest_sentiment(symbol)
        
        if not market_data or not sentiment_data:
            return None
        
        price_change = self._calculate_price_change(market_data)
        sentiment_score = sentiment_data.value.get('overall_sentiment', 0)
        
        # Generate signal
        if price_change > self.threshold and sentiment_score > 0.5:
            return {
                'action': 'BUY',
                'symbol': symbol,
                'confidence': min(price_change / self.threshold, 1.0),
                'reason': f"Price up {price_change:.1%}, positive sentiment {sentiment_score:.2f}"
            }
        elif price_change < -self.threshold and sentiment_score < -0.5:
            return {
                'action': 'SELL',
                'symbol': symbol,
                'confidence': min(abs(price_change) / self.threshold, 1.0),
                'reason': f"Price down {price_change:.1%}, negative sentiment {sentiment_score:.2f}"
            }
        
        return None
    
    def _calculate_price_change(self, market_data):
        # Calculate recent price change
        current_price = market_data.value.get('close', 0)
        previous_price = market_data.metadata.get('previous_close', current_price)
        return (current_price - previous_price) / previous_price
```

### Pattern-Based Strategy

```python
class PatternStrategy:
    def __init__(self, integrator):
        self.integrator = integrator
        self.min_confidence = 0.8
    
    async def check_patterns(self, symbol):
        patterns_data = await self.integrator.get_latest_patterns(symbol)
        
        if not patterns_data:
            return []
        
        signals = []
        patterns = patterns_data.value.get('patterns', {})
        
        for pattern_name, pattern_data in patterns.items():
            confidence = pattern_data.get('confidence', 0)
            
            if confidence >= self.min_confidence:
                direction = pattern_data.get('direction', 'neutral')
                target = pattern_data.get('target_price')
                stop_loss = pattern_data.get('stop_loss')
                
                signal = {
                    'strategy': 'pattern',
                    'pattern': pattern_name,
                    'action': 'BUY' if direction == 'bullish' else 'SELL',
                    'symbol': symbol,
                    'confidence': confidence,
                    'target': target,
                    'stop_loss': stop_loss
                }
                
                signals.append(signal)
        
        return signals
```

## 🔍 Monitoring & Health

### System Health Check

```python
async def monitor_system():
    async with CrawlerTradingIntegrator() as integrator:
        while True:
            # Get system health
            health = await integrator.get_system_health()
            
            print(f"System Status: {health['overall_status']}")
            print(f"Running: {health['integration_system']['is_running']}")
            print(f"Data Points: {health['integration_system']['metrics']['data_points_processed']}")
            print(f"Events: {health['integration_system']['metrics']['events_emitted']}")
            
            # Check for issues
            if health['overall_status'] != 'healthy':
                print(f"⚠️ System issues detected: {health}")
            
            await asyncio.minute(60)  # Check every minute
```

### Performance Metrics

```python
async def get_performance_stats():
    async with CrawlerTradingIntegrator() as integrator:
        health = await integrator.get_system_health()
        metrics = health['integration_system']['metrics']
        
        return {
            'throughput': {
                'data_points_per_minute': metrics['data_points_processed'] / 60,
                'events_per_minute': metrics['events_emitted'] / 60
            },
            'cache_performance': {
                'hit_rate': metrics['cache_hits'] / (metrics['cache_hits'] + metrics['cache_misses']),
                'cache_size': health['integration_system']['cache_stats']['cache_size']
            },
            'subscriber_metrics': {
                'active_subscribers': metrics['subscriber_notifications'],
                'data_subscribers': health['integration_system']['cache_stats']['subscribers']['data_subscribers'],
                'event_subscribers': health['integration_system']['cache_stats']['subscribers']['event_subscribers']
            }
        }
```

## 🛠️ Advanced Usage

### Custom Data Transformers

```python
class CustomDataTransformer:
    def __init__(self):
        self.transformers = {
            'custom_indicator': self._transform_custom_indicator
        }
    
    async def _transform_custom_indicator(self, raw_data):
        # Transform raw data to trading format
        return TradingDataPoint(
            symbol=raw_data.get('symbol'),
            timestamp=datetime.utcnow(),
            data_type='custom_indicator',
            priority=DataPriority.MEDIUM,
            value={
                'my_indicator': self.calculate_my_indicator(raw_data),
                'signal_strength': self.calculate_signal_strength(raw_data)
            }
        )
    
    def calculate_my_indicator(self, data):
        # Custom indicator calculation logic
        return data.get('value', 0) * 1.5
    
    def calculate_signal_strength(self, data):
        # Custom signal strength calculation
        return min(abs(data.get('value', 0)) / 100, 1.0)
```

### Event Correlation

```python
async def correlate_events():
    """Example of correlating multiple events for trading decisions"""
    
    async def handle_correlated_events(event):
        # Get recent events for correlation
        recent_events = await event_handler.get_recent_events(
            symbol=event.symbol,
            since=datetime.utcnow() - timedelta(minutes=15),
            limit=20
        )
        
        # Analyze correlation
        price_events = [e for e in recent_events if e.event_type == EventType.PRICE_UPDATE]
        sentiment_events = [e for e in recent_events if e.event_type == EventType.SENTIMENT_CHANGE]
        news_events = [e for e in recent_events if e.event_type == EventType.NEWS_ALERT]
        
        # Generate correlated signal
        if price_events and sentiment_events and news_events:
            correlation_score = calculate_correlation_score(
                price_events, sentiment_events, news_events
            )
            
            if correlation_score > 0.7:
                print(f"🎯 Strong correlation detected for {event.symbol}: {correlation_score:.2f}")
                await execute_correlated_trade(event.symbol, correlation_score)
    
    # Subscribe to correlation handler
    await integrator.subscribe_to_events(EventType.PRICE_UPDATE, handle_correlated_events)
```

### Custom Risk Management

```python
class IntegrationRiskManager:
    def __init__(self, integrator):
        self.integrator = integrator
        self.risk_limits = {
            'max_position': 10000,
            'max_correlation': 0.8,
            'max_daily_loss': 1000
        }
    
    async def validate_trade_signal(self, signal):
        """Validate trading signal against risk limits"""
        
        # Get current portfolio exposure
        portfolio = await self.get_portfolio_exposure()
        
        # Check position limits
        symbol = signal['symbol']
        if symbol in portfolio:
            current_exposure = portfolio[symbol]['market_value']
            new_exposure = current_exposure + signal.get('size', 0)
            
            if new_exposure > self.risk_limits['max_position']:
                signal['rejected'] = True
                signal['rejection_reason'] = 'Position limit exceeded'
                return signal
        
        # Check correlation limits
        correlation = await self.calculate_correlation_risk(symbol, portfolio)
        if correlation > self.risk_limits['max_correlation']:
            signal['rejected'] = True
            signal['rejection_reason'] = f'Correlation risk too high: {correlation:.2f}'
            return signal
        
        return signal
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Connection Timeouts
```python
# Increase timeout settings
config = TradingSystemConfig(
    update_frequencies={
        'market_data': 5,  # Increase from 1 to 5 seconds
        'news': 10         # Increase from 5 to 10 seconds
    },
    max_concurrent_requests=20  # Reduce from 50 to 20
)
```

#### 2. Memory Usage
```python
# Reduce cache sizes
config = TradingSystemConfig(
    cache_size_limit=5000,        # Reduce from 10000 to 5000
    data_retention_hours=24,      # Reduce from 168 to 24 hours
    event_buffer_size=500         # Reduce from 1000 to 500
)
```

#### 3. Event Processing Backlog
```python
# Monitor and adjust event processing
async def monitor_event_backlog():
    while True:
        recent_events = await event_handler.get_recent_events(limit=1)
        if recent_events:
            latest_event = recent_events[0]
            event_age = datetime.utcnow() - latest_event.timestamp
            
            if event_age > timedelta(seconds=30):
                print(f"⚠️ Event processing backlog: {event_age}")
        
        await asyncio.sleep(10)
```

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Add custom debug handler
async def debug_handler(data_point):
    print(f"DEBUG: {data_point.data_type} - {data_point.symbol}")

# Subscribe to debug handler
await integrator.subscribe_to_data('market_data', debug_handler)
```

## 📚 API Reference

### CrawlerTradingIntegrator

#### Methods

- `initialize()` - Initialize all components
- `start()` - Start the integration system
- `stop()` - Stop the integration system
- `get_latest_market_data(symbol)` - Get latest price data
- `get_latest_news(symbol, limit)` - Get latest news
- `get_latest_sentiment(symbol)` - Get sentiment analysis
- `get_latest_patterns(symbol)` - Get technical patterns
- `get_market_summary(symbols)` - Get comprehensive summary
- `get_trading_signals(symbol)` - Generate trading signals
- `subscribe_to_data(data_type, callback)` - Subscribe to data
- `subscribe_to_events(event_type, callback)` - Subscribe to events
- `get_data_stream(data_types)` - Get real-time data stream
- `get_events_stream(event_types)` - Get real-time events stream
- `get_system_health()` - Get system health status

### TradingDataPoint

#### Properties

- `symbol` - Stock symbol
- `timestamp` - Data timestamp
- `data_type` - Type of data (price, news, sentiment, etc.)
- `priority` - Data priority (critical, high, medium, low)
- `value` - Actual data content
- `metadata` - Additional metadata
- `source` - Data source
- `quality_score` - Data quality score (0.0 to 1.0)

### MarketEvent

#### Properties

- `event_type` - Type of event
- `symbol` - Related symbol
- `timestamp` - Event timestamp
- `priority` - Event priority
- `data` - Event data
- `event_id` - Unique event identifier
- `metadata` - Additional metadata

## 🤝 Integration Examples

### TradingView Integration

```python
# Send signals to TradingView webhook
async def send_to_tradingview(signal):
    webhook_url = "https://hooks.tradingview.com/hooks/your-webhook"
    
    payload = {
        "symbol": signal['symbol'],
        "action": signal['action'],
        "price": signal.get('price'),
        "confidence": signal['confidence'],
        "timestamp": datetime.utcnow().isoformat()
    }
    
    async with aiohttp.ClientSession() as session:
        await session.post(webhook_url, json=payload)
```

### Discord Bot Integration

```python
# Send alerts to Discord
async def send_discord_alert(event):
    webhook_url = "https://discord.com/api/webhooks/your-webhook"
    
    embed = {
        "title": f"Trading Alert: {event.symbol}",
        "description": f"Event: {event.event_type.value}",
        "color": 3447003 if event.priority == DataPriority.HIGH else 9817811,
        "fields": [
            {"name": "Priority", "value": event.priority.value, "inline": True},
            {"name": "Timestamp", "value": event.timestamp.isoformat(), "inline": True}
        ]
    }
    
    async with aiohttp.ClientSession() as session:
        await session.post(webhook_url, json={"embeds": [embed]})
```

### Database Integration

```python
# Store signals in database
async def store_signal_database(signal):
    # This would integrate with your trading database
    query = """
        INSERT INTO trading_signals (symbol, action, confidence, timestamp, metadata)
        VALUES (?, ?, ?, ?, ?)
    """
    
    await database.execute(
        query,
        signal['symbol'],
        signal['action'],
        signal['confidence'],
        datetime.utcnow(),
        json.dumps(signal)
    )
```

## 📞 Support

For issues, questions, or contributions:

1. Check the troubleshooting section above
2. Review the example integration scripts
3. Examine the system health metrics
4. Check the logs for detailed error information

## 🔄 Changelog

### Version 1.0.0
- Initial release
- Real-time data integration
- Event system
- Trading signal generation
- Health monitoring
- Comprehensive documentation

---

This integration system provides a robust foundation for connecting crawler data with trading systems, enabling sophisticated algorithmic trading strategies based on real-time market intelligence.