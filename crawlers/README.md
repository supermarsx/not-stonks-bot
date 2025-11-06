# Market Data Crawler System

A comprehensive, production-ready market data crawler system for collecting, processing, and analyzing financial data from multiple sources including real-time market data, news, social media sentiment, economic indicators, and technical analysis patterns.

## Features

### 🚀 Core Components

- **Market Data Crawler**: Real-time prices, historical data, and intraday data collection
- **News Crawler**: Financial news aggregation, earnings announcements, and regulatory filings
- **Social Media Crawler**: Twitter, Reddit, and StockTwits sentiment analysis
- **Economic Crawler**: Economic indicators, central bank announcements, and market sentiment
- **Pattern Crawler**: Chart pattern recognition, technical indicators, and market microstructure

### 📊 Data Collection & Processing

- **Multi-source data aggregation** from various financial data providers
- **Real-time and historical data collection** with configurable intervals
- **Data validation and quality checks** to ensure data integrity
- **Intelligent caching** to optimize performance and reduce API calls
- **Rate limiting and request throttling** to respect API limits

### 🎯 Advanced Features

- **Chart Pattern Recognition**: Head and shoulders, double tops/bottoms, triangles, flags, etc.
- **Technical Analysis**: RSI, MACD, Bollinger Bands, Stochastic, Williams %R, CCI, ADX, ATR, OBV
- **Social Sentiment Analysis**: Bullish/bearish sentiment scoring from social media
- **Economic Calendar Integration**: Central bank announcements, economic indicators, market events
- **Market Microstructure**: Order book analysis, volume profiles, bid-ask spreads

### 🔧 System Management

- **Intelligent Scheduling**: Dependency-aware crawler scheduling with priority management
- **Health Monitoring**: Comprehensive health checks, performance monitoring, and alerting
- **Error Handling**: Advanced retry logic, circuit breakers, and graceful error recovery
- **Configuration Management**: Flexible configuration system with templates and validation
- **Performance Optimization**: Bottleneck detection and optimization recommendations

### 📈 Monitoring & Alerts

- **Real-time Health Monitoring**: Continuous monitoring of crawler health and performance
- **Multi-channel Alerting**: Email, webhook, Slack, and log-based alert delivery
- **Performance Analytics**: Detailed performance reports, bottleneck analysis, and optimization suggestions
- **Data Quality Monitoring**: Data validation, completeness checks, and quality scoring
- **Historical Analysis**: Trend analysis, performance history, and predictive insights

### 🔗 Trading System Integration

- **Unified Data Bridge**: Seamlessly connect crawler data to trading orchestrators
- **Real-time Event System**: Stream critical market events to trading systems
- **Trading Signal Generation**: Automated signal generation from multi-source data
- **Risk Management Integration**: Built-in risk checking and position management
- **Performance Optimization**: High-throughput data processing for algorithmic trading

## Installation

1. **Clone or download the crawler system**
```bash
git clone <repository-url>
cd crawlers
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Install additional system dependencies** (Linux/Ubuntu):
```bash
sudo apt-get update
sudo apt-get install -y python3-dev libxml2-dev libxslt-dev libjpeg-dev zlib1g-dev
```

## Quick Start

### Basic Usage

```python
import asyncio
from crawlers.main import quick_start, create_crawler_system

# Quick start with default settings
async def main():
    # Start system with default symbols
    system = await quick_start(['AAPL', 'GOOGL', 'MSFT'])
    
    # The system is now running and collecting data
    # Keep the system running to collect data
    
    # Get latest market data
    latest_data = await system.get_latest_data('market_data')
    print(f"Collected data for {len(latest_data.get('data', {}).get('real_time', {}))} symbols")
    
    # Check system status
    status = await system.get_system_status()
    print(f"System status: {status['system']['running']}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Advanced Configuration

```python
import asyncio
from crawlers.main import create_crawler_system

async def advanced_example():
    # Custom configuration
    config = {
        'symbols': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA'],
        'storage_path': './custom_data_storage',
        'max_concurrent_crawlers': 10,
        'alerts': {
            'email': {
                'enabled': True,
                'smtp_server': 'smtp.gmail.com',
                'from_email': 'your-email@gmail.com',
                'to_emails': ['alerts@yourdomain.com'],
                'username': 'your-email@gmail.com',
                'password': 'your-app-password'
            },
            'webhook': {
                'enabled': True,
                'url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
            }
        }
    }
    
    # Create and initialize system
    system = await create_crawler_system(config)
    await system.initialize()
    
    # Configure crawlers
    system.update_config('market_data', {
        'interval': 30,  # 30 seconds
        'timeout': 45,
        'rate_limit': 200
    })
    
    # Add custom alert rule
    system.add_alert_rule(
        rule_name="high_memory_usage",
        condition=">",
        threshold=500,  # MB
        level="warning",
        channels=["email", "webhook"]
    )
    
    # Start the system
    await system.start()
    
    try:
        # Run for 1 hour collecting data
        await asyncio.sleep(3600)
        
        # Get comprehensive reports
        performance_report = await system.get_performance_report(hours=1)
        health_report = await system.get_health_report()
        
        print("Performance Report:")
        print(f"  Average execution time: {performance_report['performance_summary']['overall_metrics'].get('avg_execution_time', 0):.2f}s")
        print(f"  Success rate: {performance_report['performance_summary']['overall_metrics'].get('avg_success_rate', 0):.1%}")
        
        print("\\nHealth Report:")
        print(f"  Overall status: {health_report['system_health']['overall_status']}")
        print(f"  Active alerts: {len(health_report['alert_status']['active_alerts'])}")
        
    finally:
        await system.stop()

asyncio.run(advanced_example())
```

## Trading System Integration

### Overview

The crawler system includes a powerful **integration layer** that seamlessly connects all crawler data sources to trading orchestrators and algorithmic trading systems. This enables real-time data flow from market crawlers directly to trading decisions.

### Key Integration Features

- **Unified Data Bridge**: Transforms crawler data into trading-compatible formats
- **Real-time Event Streaming**: Critical market events streamed to trading systems
- **Trading Signal Generation**: Automated signal generation from multi-source analysis
- **Risk Management Integration**: Built-in position and risk checking
- **Performance Optimization**: High-throughput processing for algorithmic trading

### Quick Integration Example

```python
import asyncio
from crawlers.integration import CrawlerTradingIntegrator, TradingSystemConfig

async def trading_integration_example():
    # Configure integration for trading system
    config = TradingSystemConfig(
        symbols_to_monitor=['AAPL', 'GOOGL', 'MSFT', 'TSLA'],
        required_data_types=['market_data', 'news', 'sentiment', 'patterns'],
        enable_real_time_events=True,
        alert_thresholds={
            'sentiment_change': 0.3,
            'price_change': 0.05,
            'pattern_confidence': 0.8
        }
    )
    
    # Create and start integrator
    async with CrawlerTradingIntegrator(config) as integrator:
        # Subscribe to trading-relevant events
        async def handle_price_update(event):
            print(f"Price Update: {event.symbol} = ${event.data['current_price']}")
            
            # Execute trading logic
            if abs(event.data['change_percent']) > 2.0:
                print(f"Significant move detected: {event.symbol}")
        
        async def handle_news_alert(event):
            sentiment = event.data.get('sentiment_score', 0)
            relevance = event.data.get('relevance_score', 0)
            if relevance > 0.8:
                print(f"High-impact news: {event.symbol} ({sentiment:+.2f})")
        
        # Subscribe to events
        await integrator.subscribe_to_events('price_update', handle_price_update)
        await integrator.subscribe_to_events('news_alert', handle_news_alert)
        
        # Get trading signals
        signals = await integrator.get_trading_signals('AAPL')
        print(f"AAPL signals: {signals['confidence']:.2f} confidence")
        
        # Run for 60 seconds
        await asyncio.sleep(60)

asyncio.run(trading_integration_example())
```

### Real-time Data Streaming

```python
async def realtime_data_stream():
    async with CrawlerTradingIntegrator() as integrator:
        # Process data streams
        async for data_point in integrator.get_data_stream(['market_data', 'sentiment']):
            if data_point.data_type == 'market_data':
                price = data_point.value['close']
                print(f"Live Price: {data_point.symbol} ${price}")
                
            elif data_point.data_type == 'sentiment':
                sentiment = data_point.value['overall_sentiment']
                print(f"Sentiment: {data_point.symbol} {sentiment:+.2f}")

asyncio.run(realtime_data_stream())
```

### Trading Signal Generation

```python
async def generate_signals():
    async with CrawlerTradingIntegrator() as integrator:
        symbols = ['AAPL', 'GOOGL', 'TSLA']
        
        for symbol in symbols:
            # Get comprehensive market summary
            summary = await integrator.get_market_summary([symbol])
            
            # Generate trading signals
            signals = await integrator.get_trading_signals(symbol)
            
            print(f"\n{symbol} Analysis:")
            print(f"  Confidence: {signals['confidence']:.2f}")
            
            for signal in signals['signals']:
                signal_type = signal['type']
                confidence = signal['confidence']
                print(f"  - {signal_type}: {confidence:.2f}")
                
                # Execute high-confidence signals
                if confidence > 0.8:
                    print(f"    🚀 Execute {signal_type} signal")

asyncio.run(generate_signals())
```

### Event-Driven Trading

```python
async def event_driven_trading():
    async with CrawlerTradingIntegrator() as integrator:
        # Comprehensive event subscriptions
        event_handlers = {
            'price_update': handle_price_event,
            'news_alert': handle_news_event,
            'sentiment_change': handle_sentiment_event,
            'pattern_detected': handle_pattern_event
        }
        
        # Subscribe to all event types
        for event_type, handler in event_handlers.items():
            await integrator.subscribe_to_events(event_type, handler)
        
        # Run event loop
        await asyncio.sleep(3600)  # Run for 1 hour

async def handle_price_event(event):
    symbol = event.symbol
    price = event.data['current_price']
    change = event.data['change_percent']
    
    print(f"💹 {symbol}: ${price} ({change:+.2f}%)")
    
    # Execute momentum strategy
    if abs(change) > 3.0:
        await execute_momentum_trade(symbol, change)

async def handle_news_event(event):
    headline = event.data['headline']
    sentiment = event.data['sentiment_score']
    
    print(f"📰 {event.symbol}: {headline}")
    print(f"   Sentiment: {sentiment:+.2f}")
    
    # Execute news-based strategy
    if abs(sentiment) > 0.6:
        await execute_news_trade(event.symbol, sentiment)

asyncio.run(event_driven_trading())
```

### Integration Architecture

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

### Integration Guide

For detailed integration documentation, examples, and API reference, see:
- **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)** - Comprehensive integration guide
- **[integration_example.py](integration_example.py)** - Complete working example

## System Architecture

### Core Components

```
crawlers/
├── main.py                    # Main system integration
├── __init__.py               # Package initialization
├── base/                     # Base crawler framework
│   ├── base_crawler.py      # Abstract base crawler class
│   └── __init__.py
├── market_data/             # Market data collection
│   ├── market_data_crawler.py
│   └── __init__.py
├── news/                    # News and financial information
│   ├── news_crawler.py
│   └── __init__.py
├── social_media/           # Social media sentiment
│   ├── social_media_crawler.py
│   └── __init__.py
├── economic/               # Economic indicators
│   ├── economic_crawler.py
│   └── __init__.py
├── patterns/               # Technical analysis patterns
│   ├── pattern_crawler.py
│   └── __init__.py
├── scheduling/             # Crawler management
│   ├── crawler_manager.py
│   └── __init__.py
├── storage/                # Data storage and retrieval
│   ├── data_storage.py
│   └── __init__.py
├── monitoring/             # Health and performance monitoring
│   ├── health_monitor.py
│   ├── performance_monitor.py
│   └── __init__.py
├── config/                 # Configuration and error handling
│   ├── error_handler.py
│   └── __init__.py
└── requirements.txt        # Python dependencies
```

### Data Flow

1. **Crawler Execution**: Each crawler type (market data, news, social media, etc.) runs according to its schedule
2. **Data Collection**: Crawlers fetch data from various APIs and sources
3. **Data Processing**: Raw data is processed, validated, and enriched
4. **Storage**: Processed data is stored in organized file structure with SQLite metadata
5. **Monitoring**: System continuously monitors health, performance, and data quality
6. **Alerting**: Alerts are sent when thresholds are exceeded or issues are detected

## Configuration

### System Configuration

```python
config = {
    'symbols': ['AAPL', 'GOOGL', 'MSFT', 'TSLA'],  # Stock symbols to track
    'storage_path': './data',                        # Data storage directory
    'config_dir': './configs',                      # Configuration directory
    'max_concurrent_crawlers': 5,                   # Maximum concurrent crawlers
    
    # Alert configuration
    'alerts': {
        'email': {
            'enabled': True,
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'from_email': 'alerts@yourdomain.com',
            'to_emails': ['admin@yourdomain.com'],
            'username': 'alerts@yourdomain.com',
            'password': 'your-email-password'
        },
        'webhook': {
            'enabled': True,
            'url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
        }
    }
}
```

### Individual Crawler Configuration

Each crawler can be configured independently:

```python
# Market Data Crawler Configuration
market_config = {
    'name': 'market_data',
    'data_type': 'market_data',
    'interval': 30,           # Run every 30 seconds
    'timeout': 30,            # 30 second timeout
    'rate_limit': 100,        # 100 requests per minute
    'enable_cache': True,     # Enable caching
    'cache_duration': 300,    # 5 minute cache
    'retry_config': {
        'max_retries': 3,
        'initial_delay': 2.0,
        'max_delay': 60.0,
        'strategy': 'exponential'
    }
}

# Apply configuration
system.update_config('market_data', market_config)
```

## API Reference

### Main System Class

#### `MarketDataCrawlerSystem`

The main orchestrator class for the entire crawler system.

##### Methods

- `initialize(symbols=None)`: Initialize the crawler system
- `start()`: Start all crawlers and monitoring
- `stop()`: Stop all crawlers and monitoring
- `restart()`: Restart the entire system
- `trigger_crawler(name, force=False)`: Manually trigger a specific crawler
- `get_system_status()`: Get comprehensive system status
- `get_crawler_data(name, start_date, end_date, limit)`: Retrieve crawler data
- `get_latest_data(name)`: Get latest data from a crawler
- `search_data(query_params)`: Search across all crawled data
- `export_data(name, start_date, end_date, format)`: Export data in various formats
- `get_performance_report(hours)`: Get performance analytics
- `get_health_report()`: Get comprehensive health report
- `update_config(name, updates)`: Update crawler configuration
- `add_alert_rule(name, condition, threshold, level, channels)`: Add custom alerts

### Data Retrieval

#### Market Data

```python
# Get latest market data
latest_market_data = await system.get_latest_data('market_data')

# Access real-time quotes
real_time_quotes = latest_market_data['data']['real_time']
for symbol, quote in real_time_quotes.items():
    print(f"{symbol}: ${quote['price']:.2f} ({quote['change_percent']:+.2f}%)")
    print(f"  Volume: {quote['volume']:,}")
    print(f"  High/Low: ${quote['high']:.2f}/${quote['low']:.2f}")

# Get historical data
historical_data = await system.get_crawler_data(
    'market_data',
    start_date=datetime.now() - timedelta(days=7),
    end_date=datetime.now(),
    limit=10
)
```

#### News Data

```python
# Get latest news
latest_news = await system.get_latest_data('news')
news_articles = latest_news['data']['news']

for article in news_articles:
    print(f"Title: {article['title']}")
    print(f"Source: {article['source']}")
    print(f"Published: {article['published_date']}")
    print(f"Sentiment: {article.get('sentiment_score', 'N/A')}")
    print(f"Symbols: {article.get('symbols', [])}")
    print()
```

#### Social Media Sentiment

```python
# Get social media sentiment
social_data = await system.get_latest_data('social_media')
platform_data = social_data['data']['platform_data']

# Twitter sentiment
twitter_data = platform_data.get('twitter', [])
for tweet in twitter_data:
    print(f"@{tweet['author']}: {tweet['content']}")
    print(f"Sentiment: {tweet.get('sentiment_label', 'neutral')} "
          f"({tweet.get('sentiment_score', 0):.2f})")
    print(f"Engagement: {tweet['engagement']}")
```

### Performance Monitoring

#### Get Performance Report

```python
performance_report = await system.get_performance_report(hours=24)

print("Performance Summary:")
metrics = performance_report['performance_summary']['overall_metrics']
print(f"  Average execution time: {metrics.get('avg_execution_time', 0):.2f}s")
print(f"  Success rate: {metrics.get('avg_success_rate', 0):.1%}")
print(f"  Total throughput: {metrics.get('total_throughput', 0):.1f} points/second")

print("\\nBottlenecks:")
bottlenecks = performance_report['bottleneck_report']['by_severity']
for severity, bottleneck_list in bottlenecks.items():
    if bottleneck_list:
        print(f"  {severity}: {len(bottleneck_list)} issues")
        for bottleneck in bottleneck_list[:3]:  # Show first 3
            print(f"    - {bottleneck['description']}")

print("\\nOptimization Suggestions:")
suggestions = performance_report['optimization_report']['by_category']
for category, suggestion_list in suggestions.items():
    if suggestion_list:
        print(f"  {category}: {len(suggestion_list)} opportunities")
```

#### Get Health Report

```python
health_report = await system.get_health_report()

print("System Health:")
system_health = health_report['system_health']
print(f"  Overall status: {system_health['overall_status']}")
print(f"  Crawlers monitored: {system_health['summary']['total_crawlers']}")
print(f"  Healthy crawlers: {system_health['summary']['healthy_crawlers']}")

print("\\nActive Alerts:")
active_alerts = health_report['alert_status']['active_alerts']
for alert in active_alerts:
    print(f"  {alert['level']}: {alert['title']}")
    print(f"    {alert['message']}")
```

### Custom Alerts

#### Add Alert Rules

```python
# High execution time alert
system.add_alert_rule(
    rule_name="slow_execution",
    condition=">",
    threshold=300,  # 5 minutes
    level="warning",
    channels=["email", "slack"]
)

# High error rate alert
system.add_alert_rule(
    rule_name="high_error_rate",
    condition=">",
    threshold=0.2,  # 20% error rate
    level="error",
    channels=["email", "webhook"]
)

# No data received alert
system.add_alert_rule(
    rule_name="no_data",
    condition="<",
    threshold=1,  # No data points
    level="critical",
    channels=["email", "webhook", "slack"]
)
```

### Data Export

#### Export Data

```python
# Export last 7 days of market data as JSON
start_date = datetime.now() - timedelta(days=7)
end_date = datetime.now()

json_file = await system.export_data(
    crawler_name='market_data',
    start_date=start_date,
    end_date=end_date,
    format='json'
)
print(f"Exported JSON data to: {json_file}")

# Export as CSV for analysis
csv_file = await system.export_data(
    crawler_name='market_data',
    start_date=start_date,
    end_date=end_date,
    format='csv'
)
print(f"Exported CSV data to: {csv_file}")
```

## Advanced Usage

### Custom Crawler Implementation

```python
from crawlers.base.base_crawler import BaseCrawler, CrawlerConfig, DataType

class CustomDataCrawler(BaseCrawler):
    def __init__(self, config: CrawlerConfig):
        super().__init__(config)
    
    async def _fetch_data(self):
        # Implement your data fetching logic here
        data = await self._fetch_from_your_api()
        return data
    
    async def _process_data(self, data):
        # Implement data processing logic
        processed_data = self._enhance_data(data)
        await self._store_data(processed_data)
    
    def get_supported_symbols(self):
        return self.config.symbols if hasattr(self.config, 'symbols') else []
    
    def get_data_schema(self):
        return {
            'custom_field': 'str',
            'numeric_field': 'float',
            'timestamp': 'datetime'
        }

# Register custom crawler
custom_config = CrawlerConfig(
    name='custom_data',
    data_type=DataType.MARKET_DATA,
    interval=60,
    symbols=['YOUR_SYMBOLS']
)

custom_crawler = CustomDataCrawler(custom_config)
await system.manager.register_crawler('custom_data', custom_crawler)
```

### Data Validation

```python
# Validate data quality
validation_result = await system.validate_data('market_data', market_data)
print(f"Data valid: {validation_result['valid']}")
print(f"Quality score: {validation_result['quality_score']:.2f}")
print(f"Errors: {validation_result['errors']}")
print(f"Warnings: {validation_result['warnings']}")
```

### Configuration Management

```python
# Validate configuration
validation_result = system.validate_configuration('market_data')
print(f"Config valid: {validation_result['valid']}")
if not validation_result['valid']:
    print("Configuration errors:")
    for error in validation_result['errors']:
        print(f"  - {error}")

# Update configuration
system.update_config('market_data', {
    'interval': 60,  # Change to 1-minute intervals
    'timeout': 60,   # Increase timeout
    'rate_limit': 50 # Reduce rate limit
})
```

## Monitoring and Alerting

### Health Monitoring

The system continuously monitors:

- **Crawler Status**: Running, stopped, error states
- **Execution Performance**: Execution times, throughput, success rates
- **Data Quality**: Completeness, validity, freshness
- **Resource Usage**: Memory, CPU, disk usage
- **Error Rates**: Success/failure rates, error types, trends

### Alert Channels

#### Email Alerts
```python
# Configure email alerts in system config
config['alerts']['email'] = {
    'enabled': True,
    'smtp_server': 'smtp.gmail.com',
    'from_email': 'alerts@yourdomain.com',
    'to_emails': ['admin@yourdomain.com'],
    'username': 'alerts@yourdomain.com',
    'password': 'your-app-password'
}
```

#### Slack Integration
```python
# Configure Slack webhook
config['alerts']['slack'] = {
    'enabled': True,
    'webhook_url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
}
```

#### Custom Webhooks
```python
# Configure custom webhook
config['alerts']['webhook'] = {
    'enabled': True,
    'url': 'https://your-webhook-endpoint.com/alerts',
    'headers': {
        'Authorization': 'Bearer your-token',
        'Content-Type': 'application/json'
    }
}
```

### Built-in Alert Rules

The system includes several built-in alert rules:

1. **High Error Rate**: Triggered when error rate exceeds 20%
2. **Slow Execution**: Triggered when execution time exceeds 5 minutes
3. **Crawler Down**: Triggered when crawler stops responding
4. **No Data**: Triggered when no data is received
5. **Data Quality**: Triggered when data quality drops below threshold

### Custom Alert Rules

Create custom alert rules based on any metric:

```python
system.add_alert_rule(
    rule_name="memory_usage_high",
    condition=">",
    threshold=500,  # MB
    level="warning",
    channels=["email", "webhook"]
)

system.add_alert_rule(
    rule_name="throughput_low",
    condition="<",
    threshold=10,  # points per second
    level="error",
    channels=["email", "slack"]
)
```

## Performance Optimization

### Bottleneck Detection

The system automatically detects performance bottlenecks:

- **Execution Time**: Slow data fetching or processing
- **Memory Usage**: High memory consumption
- **Error Rates**: Frequent failures or timeouts
- **Throughput**: Low data processing rates
- **Resource Contention**: CPU or I/O bottlenecks

### Optimization Recommendations

The system provides actionable optimization recommendations:

```python
performance_report = await system.get_performance_report()

print("Optimization Opportunities:")
optimization_report = performance_report['optimization_report']

for category, suggestions in optimization_report['by_category'].items():
    print(f"\\n{category.upper()}:")
    for suggestion in suggestions[:3]:  # Show top 3
        print(f"  - {suggestion['title']}")
        print(f"    {suggestion['description']}")
        print(f"    Priority: {suggestion['priority']}/10")
        print(f"    Expected: {suggestion['expected_improvement']}")
```

### Performance Tuning

#### Adjust Crawler Intervals

```python
# More frequent updates for market data
system.update_config('market_data', {'interval': 15})  # 15 seconds

# Less frequent updates for economic data
system.update_config('economic', {'interval': 86400})  # 24 hours

# Custom intervals based on data importance
system.update_config('patterns', {
    'interval': 1800,  # 30 minutes for pattern detection
    'timeout': 120     # 2 minutes timeout
})
```

#### Optimize Rate Limiting

```python
# Balance between data freshness and API limits
system.update_config('market_data', {
    'rate_limit': 100,    # 100 requests per minute
    'timeout': 30,        # 30 second timeout
    'retry_config': {
        'max_retries': 3,
        'initial_delay': 2.0,
        'max_delay': 60.0
    }
})
```

#### Memory Optimization

```python
# Enable aggressive caching for frequently accessed data
system.update_config('market_data', {
    'enable_cache': True,
    'cache_duration': 600,  # 10 minutes
    'storage_config': {
        'compress_old_data': True,
        'cleanup_days': 30
    }
})
```

## Troubleshooting

### Common Issues

#### 1. Crawler Not Starting

```python
# Check system status
status = await system.get_system_status()
print(f"System running: {status['system']['running']}")
print(f"Registered crawlers: {status['manager']['registered_crawlers']}")

# Check specific crawler status
crawler_status = system.manager.get_crawler_status('market_data')
print(f"Crawler status: {crawler_status}")
```

#### 2. High Error Rates

```python
# Check error summary
error_summary = system.error_handler.get_error_summary()
print("Error Summary:")
for crawler_name, crawler_errors in error_summary['error_breakdown'].items():
    print(f"  {crawler_name}: {crawler_errors['total_errors']} errors")
    for error_type, count in crawler_errors['error_types'].items():
        print(f"    {error_type}: {count}")
```

#### 3. Performance Issues

```python
# Check performance bottlenecks
performance_report = await system.get_performance_report()
bottlenecks = performance_report['bottleneck_report']

print("Performance Bottlenecks:")
for severity, bottleneck_list in bottlenecks['by_severity'].items():
    if bottleneck_list:
        print(f"  {severity}: {len(bottleneck_list)} issues")
```

#### 4. Data Quality Issues

```python
# Check data quality
quality_report = await system.get_health_report()
data_quality = quality_report['system_health']

print(f"Data Quality Status: {data_quality['overall_status']}")
for crawler_name, crawler_quality in data_quality.get('crawler_reports', {}).items():
    print(f"  {crawler_name}: {crawler_quality['status']}")
    if crawler_quality.get('issues'):
        for issue in crawler_quality['issues']:
            print(f"    - {issue}")
```

### Debugging Tools

#### Enable Detailed Logging

```python
import logging

# Enable debug logging for crawler system
logging.getLogger('crawlers').setLevel(logging.DEBUG)
logging.getLogger('crawlers.base').setLevel(logging.DEBUG)
logging.getLogger('crawlers.scheduling').setLevel(logging.DEBUG)
```

#### Check Individual Crawler Health

```python
# Get detailed health for specific crawler
crawler_health = system.error_handler.get_crawler_health('market_data')
print(f"Crawler Health: {crawler_health['health_status']}")
print(f"Consecutive Errors: {crawler_health['consecutive_errors']}")
print(f"Circuit Breaker: {crawler_health['circuit_breaker']}")
```

#### Manual Crawler Trigger

```python
# Manually trigger a crawler for testing
try:
    await system.trigger_crawler('market_data', force=True)
    print("Crawler triggered successfully")
except Exception as e:
    print(f"Crawler trigger failed: {e}")
```

### Recovery Procedures

#### Reset Circuit Breakers

```python
# Reset circuit breaker for problematic crawler
system.error_handler.reset_circuit_breaker('market_data')
```

#### Clear Error Counts

```python
# Mark successful execution to reset error tracking
system.error_handler.mark_success('market_data')
```

#### Emergency Stop and Restart

```python
# Emergency stop all crawlers
await system.stop()

# Clear any stuck processes
await asyncio.sleep(5)

# Restart system
await system.restart()
```

## Data Structure Reference

### Market Data Format

```json
{
  "real_time": {
    "AAPL": {
      "symbol": "AAPL",
      "timestamp": "2024-01-15T10:30:00Z",
      "price": 185.25,
      "volume": 1250000,
      "high": 186.10,
      "low": 184.80,
      "open": 185.00,
      "close": 185.25,
      "change": 1.25,
      "change_percent": 0.68,
      "market_cap": 2890000000000,
      "source": "yahoo_finance"
    }
  },
  "intraday": {
    "AAPL": [
      {
        "symbol": "AAPL",
        "timestamp": "2024-01-15T10:30:00Z",
        "price": 185.25,
        "volume": 12500,
        "high": 185.30,
        "low": 185.20,
        "open": 185.25,
        "close": 185.25,
        "source": "yahoo_finance_intraday"
      }
    ]
  }
}
```

### News Data Format

```json
{
  "news": [
    {
      "title": "Apple Reports Strong Q4 Earnings",
      "content": "Apple Inc. reported better than expected...",
      "url": "https://example.com/news/article",
      "published_date": "2024-01-15T09:00:00Z",
      "source": "Reuters",
      "author": "John Doe",
      "symbols": ["AAPL"],
      "sentiment_score": 0.8,
      "article_type": "earnings",
      "language": "en"
    }
  ],
  "earnings": [
    {
      "symbol": "AAPL",
      "company_name": "Apple Inc.",
      "announcement_date": "2024-01-15T21:30:00Z",
      "fiscal_quarter": "Q4 2023",
      "fiscal_year": 2023,
      "revenue": 119000000000,
      "eps": 2.18,
      "surprise": 0.05,
      "source": "Alpha Vantage"
    }
  ]
}
```

### Social Media Sentiment Format

```json
{
  "platform_data": {
    "twitter": [
      {
        "platform": "twitter",
        "post_id": "1234567890",
        "content": "AAPL looking strong after earnings! 📈 #Apple",
        "author": "trader123",
        "timestamp": "2024-01-15T10:30:00Z",
        "engagement": {"likes": 45, "retweets": 12, "replies": 8},
        "symbols": ["AAPL"],
        "sentiment_score": 0.8,
        "sentiment_label": "positive",
        "hashtags": ["Apple", "Stocks"],
        "verified": false,
        "influence_score": 0.6
      }
    ],
    "reddit": [
      {
        "post_id": "abc123",
        "subreddit": "wallstreetbets",
        "title": "AAPL DD - Strong fundamentals",
        "content": "Detailed analysis of Apple's...",
        "author": "user456",
        "timestamp": "2024-01-15T10:25:00Z",
        "score": 234,
        "comments": 67,
        "symbols": ["AAPL"],
        "sentiment_score": 0.6,
        "sentiment_label": "positive"
      }
    ]
  },
  "symbol_sentiment": {
    "AAPL": {
      "avg_sentiment": 0.72,
      "sentiment_label": "positive",
      "mention_count": 156,
      "platforms": ["twitter", "reddit"],
      "confidence": 0.85
    }
  }
}
```

## Contributing

To contribute to the market data crawler system:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes** following the existing code style
4. **Add tests** for new functionality
5. **Update documentation** for any API changes
6. **Submit a pull request**

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-asyncio black flake8 mypy

# Run tests
pytest

# Code formatting
black crawlers/

# Linting
flake8 crawlers/

# Type checking
mypy crawlers/
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all function signatures
- Add docstrings for all public functions and classes
- Use async/await for all I/O operations
- Implement proper error handling and logging
- Add comprehensive error messages

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Support

For support and questions:

1. **Check the documentation** above for common solutions
2. **Review the troubleshooting section** for known issues
3. **Check the logs** for detailed error messages
4. **Create an issue** on the repository for bugs or feature requests

## Acknowledgments

- Yahoo Finance API for market data
- Reddit API for social media data
- Various news APIs for financial news
- Technical analysis libraries and pattern recognition algorithms
- The Python async ecosystem for high-performance crawling