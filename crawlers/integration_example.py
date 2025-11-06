"""
Crawler Trading System Integration Example

Demonstrates how to use the integrated crawler system with the trading orchestrator.
This example shows real-time data flow from crawlers to trading decisions.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any

from crawlers.integration.trading_integration import CrawlerTradingIntegrator, TradingSystemConfig
from crawlers.integration.event_handler import EventType
from crawlers.integration.data_bridge import TradingDataPoint, DataPriority


class TradingSystemExample:
    """Example trading system that uses crawler data"""
    
    def __init__(self):
        self.integrator = None
        self.portfolio = {}
        self.risk_limits = {
            'max_position_size': 10000,
            'max_daily_loss': 1000,
            'max_correlation': 0.7
        }
    
    async def initialize(self):
        """Initialize the trading system with crawler integration"""
        # Configure the integration
        config = TradingSystemConfig(
            required_data_types=['market_data', 'news', 'sentiment', 'patterns', 'economic'],
            symbols_to_monitor=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'SPY'],
            update_frequencies={
                'market_data': 1,
                'news': 5,
                'sentiment': 10,
                'patterns': 30,
                'economic': 300
            },
            enable_real_time_events=True
        )
        
        # Create and initialize integrator
        self.integrator = CrawlerTradingIntegrator(config)
        await self.integrator.initialize()
        await self.integrator.start()
        
        # Subscribe to critical events
        await self.integrator.subscribe_to_events(EventType.PRICE_UPDATE, self.handle_price_update)
        await self.integrator.subscribe_to_events(EventType.NEWS_ALERT, self.handle_news_alert)
        await self.integrator.subscribe_to_events(EventType.SENTIMENT_CHANGE, self.handle_sentiment_change)
        await self.integrator.subscribe_to_events(EventType.PATTERN_DETECTED, self.handle_pattern_detected)
        
        # Subscribe to data streams
        await self.integrator.subscribe_to_data('market_data', self.handle_market_data)
        await self.integrator.subscribe_to_data('sentiment', self.handle_sentiment_data)
        
        print("🚀 Trading System initialized with crawler integration")
    
    async def handle_price_update(self, event):
        """Handle real-time price updates"""
        try:
            symbol = event.symbol
            data = event.data
            
            current_price = data['current_price']
            change_percent = data['change_percent']
            volume = data['volume']
            
            print(f"💹 Price Update: {symbol} = ${current_price:.2f} ({change_percent:+.2f}%) "
                  f"Volume: {volume:,.0f}")
            
            # Check for significant price movements
            if abs(change_percent) > 2.0:
                print(f"🚨 Significant price movement detected for {symbol}")
                await self.analyze_trade_opportunity(symbol, current_price, change_percent)
            
            # Update portfolio
            if symbol in self.portfolio:
                self.portfolio[symbol]['last_price'] = current_price
                self.portfolio[symbol]['last_update'] = datetime.utcnow()
            
        except Exception as e:
            print(f"Error handling price update: {e}")
    
    async def handle_news_alert(self, event):
        """Handle news alerts"""
        try:
            symbol = event.symbol
            data = event.data
            
            headline = data.get('headline', 'No headline')
            sentiment_score = data.get('sentiment_score', 0)
            relevance_score = data.get('relevance_score', 0)
            source = data.get('source', 'Unknown')
            
            print(f"📰 News Alert [{symbol}]: {headline}")
            print(f"   Sentiment: {sentiment_score:+.2f}, Relevance: {relevance_score:.2f}, Source: {source}")
            
            # Act on high-impact news
            if abs(sentiment_score) > 0.6 and relevance_score > 0.8:
                print(f"🔥 High-impact news detected for {symbol}")
                await self.process_news_impact(symbol, sentiment_score, data)
            
        except Exception as e:
            print(f"Error handling news alert: {e}")
    
    async def handle_sentiment_change(self, event):
        """Handle sentiment changes"""
        try:
            symbol = event.symbol
            data = event.data
            
            current_sentiment = data['current_sentiment']
            sentiment_change = data['sentiment_change']
            confidence = data['confidence']
            
            print(f"💭 Sentiment Change [{symbol}]: {current_sentiment:+.2f} "
                  f"(Δ{sentiment_change:+.2f}, confidence: {confidence:.2f})")
            
            # Strong sentiment changes might indicate trading opportunities
            if abs(sentiment_change) > 0.4 and confidence > 0.7:
                print(f"⚡ Strong sentiment shift detected for {symbol}")
                await self.process_sentiment_signal(symbol, current_sentiment, confidence)
            
        except Exception as e:
            print(f"Error handling sentiment change: {e}")
    
    async def handle_pattern_detected(self, event):
        """Handle technical pattern detection"""
        try:
            symbol = event.symbol
            data = event.data
            
            patterns = data.get('patterns', {})
            timeframe = data.get('timeframe', '1d')
            
            print(f"📊 Patterns Detected [{symbol}] ({timeframe}):")
            for pattern_name, pattern_data in patterns.items():
                confidence = pattern_data.get('confidence', 0)
                direction = pattern_data.get('direction', 'neutral')
                target = pattern_data.get('target_price', 'N/A')
                
                print(f"   - {pattern_name}: {direction} (confidence: {confidence:.2f}, target: {target})")
            
            # Evaluate high-confidence patterns for trading signals
            high_confidence = {name: data for name, data in patterns.items() 
                             if data.get('confidence', 0) > 0.8}
            
            if high_confidence:
                print(f"🎯 High-confidence patterns for {symbol}: {list(high_confidence.keys())}")
                await self.process_pattern_signals(symbol, high_confidence)
            
        except Exception as e:
            print(f"Error handling pattern detection: {e}")
    
    async def handle_market_data(self, trading_data: TradingDataPoint):
        """Handle general market data updates"""
        try:
            symbol = trading_data.symbol
            data_type = trading_data.data_type
            
            if data_type == 'market_data':
                # Update cache for quick access
                await self.update_market_cache(symbol, trading_data)
            
        except Exception as e:
            print(f"Error handling market data: {e}")
    
    async def handle_sentiment_data(self, trading_data: TradingDataPoint):
        """Handle sentiment data updates"""
        try:
            symbol = trading_data.symbol
            sentiment = trading_data.value.get('overall_sentiment', 0)
            confidence = trading_data.value.get('confidence', 0)
            
            # Update sentiment tracking
            if symbol not in self.portfolio:
                self.portfolio[symbol] = {}
            
            self.portfolio[symbol]['sentiment'] = sentiment
            self.portfolio[symbol]['sentiment_confidence'] = confidence
            
        except Exception as e:
            print(f"Error handling sentiment data: {e}")
    
    async def analyze_trade_opportunity(self, symbol: str, price: float, change_percent: float):
        """Analyze trading opportunities from price movements"""
        try:
            # Get additional context
            sentiment_data = await self.integrator.get_latest_sentiment(symbol)
            patterns_data = await self.integrator.get_latest_patterns(symbol)
            
            # Simple trading logic
            opportunity = {
                'symbol': symbol,
                'action': None,
                'reason': [],
                'confidence': 0.0
            }
            
            # Price momentum analysis
            if change_percent > 3.0:
                opportunity['reason'].append('Strong positive momentum')
                opportunity['confidence'] += 0.3
            elif change_percent < -3.0:
                opportunity['reason'].append('Strong negative momentum')
                opportunity['confidence'] += 0.3
            
            # Sentiment confirmation
            if sentiment_data:
                sentiment_score = sentiment_data.value.get('overall_sentiment', 0)
                if sentiment_score * change_percent > 0:  # Same direction
                    opportunity['reason'].append('Sentiment confirms momentum')
                    opportunity['confidence'] += 0.4
            
            # Pattern confirmation
            if patterns_data:
                patterns = patterns_data.value.get('patterns', {})
                for pattern_name, pattern_data in patterns.items():
                    if pattern_data.get('confidence', 0) > 0.7:
                        direction = pattern_data.get('direction', 'neutral')
                        if (direction == 'bullish' and change_percent > 0) or \
                           (direction == 'bearish' and change_percent < 0):
                            opportunity['reason'].append(f'Pattern confirms: {pattern_name}')
                            opportunity['confidence'] += 0.3
            
            # Generate signal
            if opportunity['confidence'] > 0.6:
                opportunity['action'] = 'BUY' if change_percent > 0 else 'SELL'
                print(f"🎯 Trade Opportunity: {symbol} {opportunity['action']} "
                      f"(confidence: {opportunity['confidence']:.2f})")
                print(f"   Reasons: {', '.join(opportunity['reason'])}")
                
                await self.execute_trade_signal(opportunity)
            
        except Exception as e:
            print(f"Error analyzing trade opportunity: {e}")
    
    async def process_news_impact(self, symbol: str, sentiment_score: float, news_data: Dict):
        """Process news impact on trading decisions"""
        try:
            print(f"📈 Processing news impact for {symbol}")
            print(f"   Sentiment: {sentiment_score:+.2f}")
            print(f"   Category: {news_data.get('category', 'unknown')}")
            
            # Update news impact tracking
            if symbol not in self.portfolio:
                self.portfolio[symbol] = {}
            
            if 'news_impact' not in self.portfolio[symbol]:
                self.portfolio[symbol]['news_impact'] = []
            
            self.portfolio[symbol]['news_impact'].append({
                'timestamp': datetime.utcnow().isoformat(),
                'sentiment': sentiment_score,
                'category': news_data.get('category'),
                'headline': news_data.get('headline', '')[:100]
            })
            
        except Exception as e:
            print(f"Error processing news impact: {e}")
    
    async def process_sentiment_signal(self, symbol: str, sentiment: float, confidence: float):
        """Process sentiment-based trading signals"""
        try:
            if symbol not in self.portfolio:
                self.portfolio[symbol] = {}
            
            # Simple sentiment-based positioning
            position_size = 0
            
            if sentiment > 0.5 and confidence > 0.8:
                position_size = min(5000, self.risk_limits['max_position_size'] * 0.5)
                print(f"🟢 Bullish sentiment for {symbol}: Suggest long position of ${position_size}")
            
            elif sentiment < -0.5 and confidence > 0.8:
                position_size = min(3000, self.risk_limits['max_position_size'] * 0.3)
                print(f"🔴 Bearish sentiment for {symbol}: Suggest short position of ${position_size}")
            
            if position_size > 0:
                # This would integrate with actual broker API
                print(f"💡 Consider adjusting {symbol} position based on sentiment")
            
        except Exception as e:
            print(f"Error processing sentiment signal: {e}")
    
    async def process_pattern_signals(self, symbol: str, patterns: Dict):
        """Process technical pattern trading signals"""
        try:
            print(f"📊 Processing pattern signals for {symbol}")
            
            for pattern_name, pattern_data in patterns.items():
                confidence = pattern_data.get('confidence', 0)
                direction = pattern_data.get('direction', 'neutral')
                target_price = pattern_data.get('target_price')
                stop_loss = pattern_data.get('stop_loss')
                
                print(f"   Pattern: {pattern_name}")
                print(f"   Direction: {direction} (confidence: {confidence:.2f})")
                print(f"   Target: {target_price}, Stop Loss: {stop_loss}")
                
                # Generate trade recommendation
                if target_price and stop_loss and confidence > 0.85:
                    print(f"🎯 HIGH CONFIDENCE PATTERN: Consider {direction} position")
                    print(f"   Entry: Current price, Target: {target_price}, Stop: {stop_loss}")
            
        except Exception as e:
            print(f"Error processing pattern signals: {e}")
    
    async def execute_trade_signal(self, signal: Dict):
        """Execute trading signal (simulation)"""
        try:
            symbol = signal['symbol']
            action = signal['action']
            confidence = signal['confidence']
            
            print(f"🚀 Executing Signal: {action} {symbol} (confidence: {confidence:.2f})")
            
            # In a real system, this would:
            # 1. Check risk limits
            # 2. Calculate position size
            # 3. Execute through broker API
            # 4. Update portfolio
            # 5. Set stop losses and targets
            
            # Simulate execution
            if symbol not in self.portfolio:
                self.portfolio[symbol] = {}
            
            self.portfolio[symbol].update({
                'last_signal': signal,
                'last_signal_time': datetime.utcnow().isoformat()
            })
            
            print(f"✅ Signal executed (simulated): {action} {symbol}")
            
        except Exception as e:
            print(f"Error executing trade signal: {e}")
    
    async def update_market_cache(self, symbol: str, trading_data: TradingDataPoint):
        """Update internal market data cache"""
        try:
            if symbol not in self.portfolio:
                self.portfolio[symbol] = {}
            
            self.portfolio[symbol].update({
                'last_market_data': trading_data.value,
                'last_update': datetime.utcnow().isoformat(),
                'data_quality': trading_data.quality_score
            })
            
        except Exception as e:
            print(f"Error updating market cache: {e}")
    
    async def run_analysis_cycle(self):
        """Run a comprehensive analysis cycle"""
        try:
            print("\n" + "="*60)
            print(f"🔍 Running Analysis Cycle at {datetime.utcnow()}")
            print("="*60)
            
            # Get market summary for all symbols
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
            summary = await self.integrator.get_market_summary(symbols)
            
            print("\n📊 Market Summary:")
            for symbol, data in summary.items():
                if 'error' not in data:
                    market_data = data.get('market_data', {})
                    sentiment = data.get('sentiment', {})
                    patterns = data.get('patterns', {})
                    
                    current_price = market_data.get('close', 'N/A')
                    sentiment_score = sentiment.get('overall_sentiment', 0)
                    pattern_count = len(patterns.get('patterns', {}))
                    
                    print(f"   {symbol}: ${current_price} | "
                          f"Sentiment: {sentiment_score:+.2f} | "
                          f"Patterns: {pattern_count}")
            
            # Generate trading signals for each symbol
            print("\n🎯 Trading Signals:")
            for symbol in symbols:
                signals = await self.integrator.get_trading_signals(symbol)
                
                if 'error' not in signals and signals.get('signals'):
                    print(f"   {symbol}: {len(signals['signals'])} signals, "
                          f"confidence: {signals['confidence']:.2f}")
                    
                    for signal in signals['signals'][:3]:  # Show top 3
                        signal_type = signal['type']
                        confidence = signal['confidence']
                        print(f"     - {signal_type}: {confidence:.2f}")
            
            # System health check
            health = await self.integrator.get_system_health()
            print(f"\n💚 System Health: {health['overall_status']}")
            print(f"   Data Points Processed: {health['integration_system']['metrics']['data_points_processed']}")
            print(f"   Events Emitted: {health['integration_system']['metrics']['events_emitted']}")
            
        except Exception as e:
            print(f"Error in analysis cycle: {e}")
    
    async def run_realtime_demo(self, duration_seconds: int = 60):
        """Run real-time demonstration"""
        try:
            print(f"\n🚀 Starting Real-time Demo for {duration_seconds} seconds")
            print("Press Ctrl+C to stop early\n")
            
            start_time = datetime.utcnow()
            end_time = start_time.timestamp() + duration_seconds
            
            while datetime.utcnow().timestamp() < end_time:
                try:
                    # Run analysis every 10 seconds
                    await asyncio.sleep(10)
                    await self.run_analysis_cycle()
                    
                except KeyboardInterrupt:
                    print("\n⏹️  Demo interrupted by user")
                    break
                except Exception as e:
                    print(f"Error in demo cycle: {e}")
                    await asyncio.sleep(5)
            
            print(f"\n✅ Demo completed after {duration_seconds} seconds")
            
        except Exception as e:
            print(f"Error in real-time demo: {e}")
    
    async def shutdown(self):
        """Shutdown the trading system"""
        try:
            print("\n🔄 Shutting down trading system...")
            
            if self.integrator:
                await self.integrator.stop()
            
            print("✅ Trading system shutdown complete")
            
        except Exception as e:
            print(f"Error during shutdown: {e}")


async def main():
    """Main demonstration function"""
    print("🚀 Crawler Trading System Integration Demo")
    print("="*50)
    
    # Create trading system
    trading_system = TradingSystemExample()
    
    try:
        # Initialize the system
        await trading_system.initialize()
        
        # Run a short analysis cycle
        await trading_system.run_analysis_cycle()
        
        # Run real-time demo for 30 seconds
        await trading_system.run_realtime_demo(duration_seconds=30)
        
        # Show final portfolio status
        print(f"\n📊 Final Portfolio Status:")
        print(f"   Symbols tracked: {list(trading_system.portfolio.keys())}")
        
        # Show system health
        health = await trading_system.integrator.get_system_health()
        print(f"\n💚 Final System Health: {health['overall_status']}")
        
    except Exception as e:
        print(f"Demo error: {e}")
    
    finally:
        # Always shutdown cleanly
        await trading_system.shutdown()


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())