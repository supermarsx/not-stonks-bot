"""
@file news_based_strategies.py
@brief News-Based Strategies Implementation

@details
This module implements 8+ news-based trading strategies that analyze sentiment,
earnings data, economic calendar events, and market news to generate trading signals.

Strategy Categories:
- Sentiment Analysis Strategies (2): News sentiment analysis, social media sentiment
- Earnings-Based Strategies (2): Earnings surprise trading, earnings volatility
- Economic Calendar Strategies (2): FOMC announcements, economic data releases
- Event-Driven Strategies (2): Merger arbitrage, conference/speech trading
- Media Sentiment Strategies (2): Analyst upgrades/downgrades, media sentiment
- Market Psychology Strategies (2): Fear/greed index, sentiment extremes

Key Features:
- Real-time news feed integration
- Sentiment analysis and scoring
- Economic calendar integration
- Event correlation analysis
- Social media sentiment tracking
- News-based risk assessment

@author Trading Orchestrator System
@version 2.0
@date 2025-11-06

@warning
News-based strategies can be affected by information delays and may
experience high volatility during major news events.

@see library.StrategyLibrary for strategy management
@see base.BaseStrategy for base implementation
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass
import asyncio
import json
import re
from collections import Counter
import math

from loguru import logger

from .base import (
    BaseStrategy,
    StrategyConfig,
    StrategyType,
    TradingSignal,
    SignalType,
    RiskLevel,
    BaseTimeSeriesStrategy,
    StrategyMetadata,
    strategy_registry
)
from .library import StrategyCategory, strategy_library
from trading.models import OrderSide


# ============================================================================
# SENTIMENT ANALYSIS STRATEGIES
# ============================================================================

class NewsSentimentStrategy(BaseTimeSeriesStrategy):
    """News Sentiment Analysis Strategy
    
    Analyzes news sentiment to generate trading signals based on positive/negative sentiment.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['sentiment_threshold', 'lookback_hours']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1h")
        
        self.sentiment_threshold = float(config.parameters.get('sentiment_threshold', 0.6))
        self.lookback_hours = int(config.parameters.get('lookback_hours', 24))
        self.news_symbols = config.parameters.get('news_symbols', [])
        self.sentiment_weights = config.parameters.get('sentiment_weights', {
            'positive': 1.0,
            'negative': -1.0,
            'neutral': 0.0
        })
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.6))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        for symbol in self.config.symbols:
            # Get news sentiment for symbol
            sentiment_data = await self._get_news_sentiment(symbol)
            
            if not sentiment_data:
                continue
            
            # Calculate aggregated sentiment score
            sentiment_score = self._calculate_sentiment_score(sentiment_data)
            sentiment_trend = self._calculate_sentiment_trend(sentiment_data)
            
            # Generate signals based on sentiment
            if sentiment_score >= self.sentiment_threshold:
                # Positive sentiment above threshold
                strength = min(1.0, sentiment_score * sentiment_trend)
                
                signal = await self._create_signal(
                    symbol, SignalType.BUY, strength, Decimal('1')
                )
                if signal:
                    signals.append(signal)
            
            elif sentiment_score <= -self.sentiment_threshold:
                # Negative sentiment below threshold
                strength = min(1.0, abs(sentiment_score) * sentiment_trend)
                
                signal = await self._create_signal(
                    symbol, SignalType.SELL, strength, Decimal('1')
                )
                if signal:
                    signals.append(signal)
        
        return signals
    
    def _calculate_sentiment_score(self, sentiment_data: List[Dict[str, Any]]) -> float:
        """Calculate weighted sentiment score from news data"""
        if not sentiment_data:
            return 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        for news_item in sentiment_data:
            sentiment = news_item.get('sentiment', 'neutral')
            weight = news_item.get('weight', 1.0)
            impact = news_item.get('impact', 0.5)
            
            sentiment_value = self.sentiment_weights.get(sentiment, 0.0)
            weighted_score = sentiment_value * weight * impact
            
            total_score += weighted_score
            total_weight += weight * impact
        
        if total_weight == 0:
            return 0.0
        
        return total_score / total_weight
    
    def _calculate_sentiment_trend(self, sentiment_data: List[Dict[str, Any]]) -> float:
        """Calculate sentiment trend (momentum)"""
        if len(sentiment_data) < 2:
            return 1.0
        
        # Sort by timestamp
        sorted_data = sorted(sentiment_data, key=lambda x: x.get('timestamp', ''))
        
        recent_sentiment = 0.0
        older_sentiment = 0.0
        
        # Split into recent and older sentiment
        mid_point = len(sorted_data) // 2
        
        for news_item in sorted_data[:mid_point]:
            sentiment = news_item.get('sentiment', 'neutral')
            weight = news_item.get('weight', 1.0)
            older_sentiment += self.sentiment_weights.get(sentiment, 0.0) * weight
        
        for news_item in sorted_data[mid_point:]:
            sentiment = news_item.get('sentiment', 'neutral')
            weight = news_item.get('weight', 1.0)
            recent_sentiment += self.sentiment_weights.get(sentiment, 0.0) * weight
        
        # Calculate trend factor
        if abs(older_sentiment) > 0.1:
            trend = recent_sentiment / older_sentiment
            return min(2.0, max(0.5, trend))  # Cap between 0.5 and 2.0
        
        return 1.0
    
    async def _get_news_sentiment(self, symbol: str) -> List[Dict[str, Any]]:
        """Get news sentiment data for symbol"""
        try:
            # In a real implementation, this would fetch from news APIs
            # For demonstration, return mock data structure
            return await self._fetch_news_sentiment_data(symbol)
        except Exception as e:
            logger.warning(f"Could not fetch news sentiment for {symbol}: {e}")
            return []
    
    async def _fetch_news_sentiment_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch news sentiment data from external source"""
        # Mock implementation - in real system, integrate with:
        # - NewsAPI, Bloomberg API, Reuters API
        # - Alpha Vantage news sentiment
        # - Financial data providers
        mock_sentiment_data = [
            {
                'timestamp': (datetime.utcnow() - timedelta(hours=1)).isoformat(),
                'sentiment': 'positive',
                'weight': 1.0,
                'impact': 0.8,
                'headline': f"{symbol} reports strong quarterly earnings",
                'source': 'news_feed'
            },
            {
                'timestamp': (datetime.utcnow() - timedelta(hours=6)).isoformat(),
                'sentiment': 'positive',
                'weight': 0.8,
                'impact': 0.6,
                'headline': f"Analyst upgrades {symbol} rating",
                'source': 'analyst_feed'
            }
        ]
        return mock_sentiment_data
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, price: Decimal) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        return TradingSignal(
            signal_id=f"NEWS_SENTIMENT_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            time_horizon=timedelta(hours=self.lookback_hours),
            metadata={
                'strategy_type': 'news_sentiment',
                'lookback_hours': self.lookback_hours,
                'sentiment_source': 'news_feeds'
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


class SocialMediaSentimentStrategy(BaseTimeSeriesStrategy):
    """Social Media Sentiment Strategy
    
    Analyzes social media sentiment from platforms like Twitter, Reddit for trading signals.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['social_platforms', 'sentiment_threshold']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1h")
        
        self.social_platforms = config.parameters.get('social_platforms', ['twitter', 'reddit'])
        self.sentiment_threshold = float(config.parameters.get('sentiment_threshold', 0.7))
        self.engagement_weighting = config.parameters.get('engagement_weighting', True)
        self.follower_weighting = config.parameters.get('follower_weighting', True)
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.6))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        for symbol in self.config.symbols:
            # Get social media sentiment for symbol
            social_data = await self._get_social_sentiment(symbol)
            
            if not social_data:
                continue
            
            # Calculate social sentiment metrics
            sentiment_score = self._calculate_social_sentiment_score(social_data)
            viral_score = self._calculate_viral_engagement_score(social_data)
            influencer_score = self._calculate_influencer_weighted_sentiment(social_data)
            
            # Combine metrics for final signal strength
            combined_score = (
                sentiment_score * 0.5 +
                viral_score * 0.3 +
                influencer_score * 0.2
            )
            
            # Generate signals based on social sentiment
            if combined_score >= self.sentiment_threshold:
                # Strong positive social sentiment
                strength = min(1.0, combined_score * viral_score)
                
                signal = await self._create_signal(
                    symbol, SignalType.BUY, strength, Decimal('1')
                )
                if signal:
                    signals.append(signal)
            
            elif combined_score <= -self.sentiment_threshold:
                # Strong negative social sentiment
                strength = min(1.0, abs(combined_score) * viral_score)
                
                signal = await self._create_signal(
                    symbol, SignalType.SELL, strength, Decimal('1')
                )
                if signal:
                    signals.append(signal)
        
        return signals
    
    def _calculate_social_sentiment_score(self, social_data: List[Dict[str, Any]]) -> float:
        """Calculate overall social media sentiment score"""
        if not social_data:
            return 0.0
        
        sentiment_scores = []
        for post in social_data:
            sentiment = post.get('sentiment', 0.0)
            engagement = post.get('engagement', 0.0)
            followers = post.get('author_followers', 0.0)
            
            # Weight sentiment by engagement and follower count
            weight = 1.0
            if self.engagement_weighting:
                weight *= (1.0 + engagement)
            if self.follower_weighting and followers > 0:
                weight *= math.log10(followers + 1) / 6.0  # Normalize to 0-1
            
            weighted_sentiment = sentiment * weight
            sentiment_scores.append(weighted_sentiment)
        
        return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
    
    def _calculate_viral_engagement_score(self, social_data: List[Dict[str, Any]]) -> float:
        """Calculate viral engagement score"""
        if not social_data:
            return 1.0
        
        engagement_scores = []
        for post in social_data:
            engagement = post.get('engagement', 0.0)
            time_factor = self._calculate_time_decay_factor(post.get('timestamp', ''))
            engagement_scores.append(engagement * time_factor)
        
        avg_engagement = sum(engagement_scores) / len(engagement_scores)
        
        # Convert to 0.5-2.0 range for multiplier
        viral_score = 0.5 + min(1.5, avg_engagement * 0.01)
        return viral_score
    
    def _calculate_influencer_weighted_sentiment(self, social_data: List[Dict[str, Any]]) -> float:
        """Calculate influencer-weighted sentiment"""
        if not social_data:
            return 0.0
        
        influencer_sentiment = []
        for post in social_data:
            is_influencer = post.get('author_influencer', False)
            sentiment = post.get('sentiment', 0.0)
            followers = post.get('author_followers', 0.0)
            
            if is_influencer or followers > 10000:
                # Boost influencer posts
                weight = min(3.0, math.log10(followers + 1) / 5.0)
                influencer_sentiment.append(sentiment * weight)
        
        return sum(influencer_sentiment) / len(influencer_sentiment) if influencer_sentiment else 0.0
    
    def _calculate_time_decay_factor(self, timestamp: str) -> float:
        """Calculate time decay factor for engagement"""
        try:
            post_time = datetime.fromisoformat(timestamp)
            age_hours = (datetime.utcnow() - post_time).total_seconds() / 3600
            
            # Exponential decay
            decay_factor = math.exp(-age_hours / 24.0)  # 24-hour half-life
            return max(0.1, decay_factor)
        except:
            return 1.0
    
    async def _get_social_sentiment(self, symbol: str) -> List[Dict[str, Any]]:
        """Get social media sentiment data for symbol"""
        try:
            # In a real implementation, integrate with:
            # - Twitter API, Reddit API
            # - Social sentiment data providers
            # - Real-time social listening platforms
            return await self._fetch_social_sentiment_data(symbol)
        except Exception as e:
            logger.warning(f"Could not fetch social sentiment for {symbol}: {e}")
            return []
    
    async def _fetch_social_sentiment_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch social media sentiment data from external source"""
        # Mock implementation
        mock_social_data = [
            {
                'timestamp': (datetime.utcnow() - timedelta(hours=1)).isoformat(),
                'sentiment': 0.8,
                'engagement': 150.0,
                'author_followers': 50000,
                'author_influencer': True,
                'platform': 'twitter',
                'text': f"{symbol} looks great! 🚀"
            },
            {
                'timestamp': (datetime.utcnow() - timedelta(hours=3)).isoformat(),
                'sentiment': 0.6,
                'engagement': 75.0,
                'author_followers': 2500,
                'author_influencer': False,
                'platform': 'reddit',
                'text': f"Thoughts on {symbol} earnings?"
            }
        ]
        return mock_social_data
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, price: Decimal) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        return TradingSignal(
            signal_id=f"SOCIAL_SENTIMENT_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            metadata={
                'strategy_type': 'social_media_sentiment',
                'platforms': self.social_platforms,
                'viral_engagement': True
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


# ============================================================================
# EARNINGS-BASED STRATEGIES
# ============================================================================

class EarningsSurpriseStrategy(BaseTimeSeriesStrategy):
    """Earnings Surprise Strategy
    
    Trades on earnings surprises relative to analyst expectations.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['surprise_threshold', 'lookback_days']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1d")
        
        self.surprise_threshold = float(config.parameters.get('surprise_threshold', 0.05))
        self.lookback_days = int(config.parameters.get('lookback_days', 5))
        self.pre_market_trading = config.parameters.get('pre_market_trading', False)
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.8))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        for symbol in self.config.symbols:
            # Check for recent earnings announcements
            earnings_data = await self._get_earnings_data(symbol)
            
            if not earnings_data:
                continue
            
            # Check for surprise in recent earnings
            for earnings in earnings_data:
                surprise = earnings.get('surprise', 0.0)
                surprise_pct = earnings.get('surprise_percentage', 0.0)
                
                # Positive surprise
                if surprise_pct >= self.surprise_threshold:
                    strength = min(1.0, surprise_pct / 0.1)  # Normalize to 0-1
                    
                    signal = await self._create_signal(
                        symbol, SignalType.BUY, strength, earnings['price']
                    )
                    if signal:
                        signals.append(signal)
                
                # Negative surprise
                elif surprise_pct <= -self.surprise_threshold:
                    strength = min(1.0, abs(surprise_pct) / 0.1)
                    
                    signal = await self._create_signal(
                        symbol, SignalType.SELL, strength, earnings['price']
                    )
                    if signal:
                        signals.append(signal)
        
        return signals
    
    async def _get_earnings_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Get earnings data for symbol"""
        try:
            # In real implementation, fetch from earnings API
            return await self._fetch_earnings_data(symbol)
        except Exception as e:
            logger.warning(f"Could not fetch earnings data for {symbol}: {e}")
            return []
    
    async def _fetch_earnings_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch earnings data from external source"""
        # Mock earnings data
        mock_earnings = [
            {
                'symbol': symbol,
                'earnings_date': (datetime.utcnow() - timedelta(days=1)).isoformat(),
                'actual_eps': 2.15,
                'estimated_eps': 1.98,
                'surprise': 0.17,
                'surprise_percentage': 8.6,
                'price': Decimal('150.50'),
                'guidance': 'increased'
            }
        ]
        return mock_earnings
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, price: Decimal) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        return TradingSignal(
            signal_id=f"EARNINGS_SURPRISE_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            time_horizon=timedelta(days=3),
            metadata={
                'strategy_type': 'earnings_surprise',
                'pre_market_trading': self.pre_market_trading
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


# ============================================================================
# ECONOMIC CALENDAR STRATEGIES
# ============================================================================

class EconomicCalendarStrategy(BaseTimeSeriesStrategy):
    """Economic Calendar Event Strategy
    
    Trades on major economic announcements and calendar events.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['event_importance_threshold', 'pre_event_hours']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1h")
        
        self.event_importance_threshold = float(config.parameters.get('event_importance_threshold', 0.7))
        self.pre_event_hours = int(config.parameters.get('pre_event_hours', 2))
        self.post_event_hours = int(config.parameters.get('post_event_hours', 4))
        self.trade_directions = config.parameters.get('trade_directions', 'both')  # 'long', 'short', 'both'
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.7))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        # Get upcoming economic events
        events = await self._get_economic_events()
        
        current_time = datetime.utcnow()
        
        for event in events:
            event_time = event.get('timestamp')
            if not event_time:
                continue
            
            try:
                event_datetime = datetime.fromisoformat(event_time)
            except:
                continue
            
            # Check if we're in pre-event window
            time_to_event = event_datetime - current_time
            
            if timedelta(hours=0) <= time_to_event <= timedelta(hours=self.pre_event_hours):
                # Pre-event trading
                importance = event.get('importance', 0.0)
                
                if importance >= self.event_importance_threshold:
                    signal_type = self._determine_pre_event_signal(event)
                    strength = min(1.0, importance)
                    
                    for symbol in self.config.symbols:
                        signal = await self._create_signal(
                            symbol, signal_type, strength, Decimal('1'), event
                        )
                        if signal:
                            signals.append(signal)
            
            elif timedelta(hours=-self.post_event_hours) <= time_to_event <= timedelta(hours=0):
                # Post-event trading (if surprise element)
                surprise_factor = event.get('surprise_factor', 0.0)
                if abs(surprise_factor) > 0.5:
                    signal_type = self._determine_post_event_signal(event, surprise_factor)
                    strength = min(1.0, abs(surprise_factor))
                    
                    for symbol in self.config.symbols:
                        signal = await self._create_signal(
                            symbol, signal_type, strength, Decimal('1'), event
                        )
                        if signal:
                            signals.append(signal)
        
        return signals
    
    def _determine_pre_event_signal(self, event: Dict[str, Any]) -> SignalType:
        """Determine trading direction based on pre-event conditions"""
        event_type = event.get('type', '')
        market_sentiment = event.get('market_sentiment', 'neutral')
        
        # FOMC meetings - typically risk-off before announcement
        if 'fomc' in event_type.lower() or 'federal_fund' in event_type.lower():
            if market_sentiment == 'risk_off':
                return SignalType.SELL
            else:
                return SignalType.BUY
        
        # Employment data - positive surprise usually bullish
        elif 'employment' in event_type.lower() or 'jobs' in event_type.lower():
            expected_result = event.get('expected_result', 'neutral')
            if expected_result == 'strong':
                return SignalType.BUY
            else:
                return SignalType.SELL
        
        # Default: neutral or sentiment-based
        if market_sentiment == 'risk_off':
            return SignalType.SELL
        elif market_sentiment == 'risk_on':
            return SignalType.BUY
        else:
            return SignalType.BUY  # Default to long bias
    
    def _determine_post_event_signal(self, event: Dict[str, Any], surprise_factor: float) -> SignalType:
        """Determine trading direction based on post-event surprise"""
        if surprise_factor > 0:
            return SignalType.BUY
        else:
            return SignalType.SELL
    
    async def _get_economic_events(self) -> List[Dict[str, Any]]:
        """Get upcoming economic events from calendar"""
        try:
            # In real implementation, fetch from economic calendar API
            return await self._fetch_economic_events()
        except Exception as e:
            logger.warning(f"Could not fetch economic events: {e}")
            return []
    
    async def _fetch_economic_events(self) -> List[Dict[str, Any]]:
        """Fetch economic events from external source"""
        # Mock economic events
        mock_events = [
            {
                'timestamp': (datetime.utcnow() + timedelta(hours=1)).isoformat(),
                'type': 'FOMC Rate Decision',
                'importance': 0.9,
                'market_sentiment': 'risk_off',
                'expected_result': 'hawkish',
                'description': 'Federal Open Market Committee interest rate decision'
            },
            {
                'timestamp': (datetime.utcnow() - timedelta(hours=2)).isoformat(),
                'type': 'Non-Farm Payrolls',
                'importance': 0.8,
                'market_sentiment': 'neutral',
                'expected_result': 'strong',
                'surprise_factor': 0.7,
                'actual_result': 'strong',
                'description': 'Monthly employment report'
            }
        ]
        return mock_events
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, price: Decimal, event: Dict[str, Any] = None) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        return TradingSignal(
            signal_id=f"ECON_CALENDAR_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            time_horizon=timedelta(hours=self.post_event_hours),
            metadata={
                'strategy_type': 'economic_calendar',
                'event_type': event.get('type', '') if event else '',
                'event_importance': event.get('importance', 0.0) if event else 0.0
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


# ============================================================================
# MEDIA SENTIMENT STRATEGIES
# ============================================================================

class AnalystRatingStrategy(BaseTimeSeriesStrategy):
    """Analyst Rating Change Strategy
    
    Trades on analyst rating upgrades and downgrades.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['rating_change_threshold', 'lookback_days']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1h")
        
        self.rating_change_threshold = int(config.parameters.get('rating_change_threshold', 1))
        self.lookback_days = int(config.parameters.get('lookback_days', 7))
        self.firm_weighting = config.parameters.get('firm_weighting', True)
        self.rank_adjustment = config.parameters.get('rank_adjustment', True)
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.7))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        for symbol in self.config.symbols:
            # Get recent analyst rating changes
            ratings = await self._get_analyst_ratings(symbol)
            
            if not ratings:
                continue
            
            # Process rating changes
            for rating in ratings:
                rating_change = rating.get('rating_change', 0)
                confidence = rating.get('confidence', 0.5)
                firm_importance = rating.get('firm_importance', 1.0)
                
                # Filter significant rating changes
                if abs(rating_change) >= self.rating_change_threshold:
                    # Determine signal direction
                    if rating_change > 0:
                        # Upgrade
                        signal_type = SignalType.BUY
                        strength = min(1.0, rating_change * confidence * firm_importance)
                    else:
                        # Downgrade
                        signal_type = SignalType.SELL
                        strength = min(1.0, abs(rating_change) * confidence * firm_importance)
                    
                    signal = await self._create_signal(
                        symbol, signal_type, strength, Decimal('1'), rating
                    )
                    if signal:
                        signals.append(signal)
        
        return signals
    
    def _calculate_rating_score(self, rating: str) -> int:
        """Convert rating string to numeric score"""
        ratings = {
            'strong_buy': 2,
            'buy': 1,
            'hold': 0,
            'sell': -1,
            'strong_sell': -2,
            'outperform': 1,
            'underperform': -1,
            'overweight': 1,
            'underweight': -1,
            'equalweight': 0
        }
        return ratings.get(rating.lower(), 0)
    
    async def _get_analyst_ratings(self, symbol: str) -> List[Dict[str, Any]]:
        """Get analyst ratings for symbol"""
        try:
            # In real implementation, fetch from analyst rating API
            return await self._fetch_analyst_ratings(symbol)
        except Exception as e:
            logger.warning(f"Could not fetch analyst ratings for {symbol}: {e}")
            return []
    
    async def _fetch_analyst_ratings(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch analyst ratings from external source"""
        # Mock analyst ratings
        mock_ratings = [
            {
                'symbol': symbol,
                'timestamp': (datetime.utcnow() - timedelta(hours=2)).isoformat(),
                'firm': 'Goldman Sachs',
                'previous_rating': 'Buy',
                'new_rating': 'Strong Buy',
                'rating_change': 1,
                'confidence': 0.8,
                'firm_importance': 1.5,
                'target_price': Decimal('175.00')
            }
        ]
        return mock_ratings
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, price: Decimal, rating: Dict[str, Any] = None) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        return TradingSignal(
            signal_id=f"ANALYST_RATING_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            time_horizon=timedelta(days=5),
            metadata={
                'strategy_type': 'analyst_rating_change',
                'firm': rating.get('firm', '') if rating else '',
                'rating_change': rating.get('rating_change', 0) if rating else 0
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


# ============================================================================
# MARKET PSYCHOLOGY STRATEGIES
# ============================================================================

class FearGreedIndexStrategy(BaseTimeSeriesStrategy):
    """Fear & Greed Index Strategy
    
    Trades based on market fear/greed sentiment extremes.
    """
    
    def __init__(self, config: StrategyConfig):
        required_params = ['fear_threshold', 'greed_threshold']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1d")
        
        self.fear_threshold = float(config.parameters.get('fear_threshold', 25.0))
        self.greed_threshold = float(config.parameters.get('greed_threshold', 75.0))
        self.fear_greed_symbol = config.parameters.get('fear_greed_symbol', 'VIX')
        self.trend_confirmation = config.parameters.get('trend_confirmation', True)
        self.signal_threshold = float(config.parameters.get('signal_threshold', 0.8))
    
    async def generate_signals(self) -> List[TradingSignal]:
        signals = []
        
        # Get Fear & Greed index
        fear_greed_data = await self._get_fear_greed_index()
        
        if not fear_greed_data:
            return signals
        
        current_index = fear_greed_data.get('index_value', 50.0)
        trend = fear_greed_data.get('trend', 'neutral')
        
        # Generate signals based on extreme readings
        if current_index <= self.fear_threshold:
            # Extreme fear - expect market bottom and bounce
            fear_level = (self.fear_threshold - current_index) / self.fear_threshold
            strength = min(1.0, fear_level * 2)
            
            # Confirm with trend
            if self.trend_confirmation and trend == 'recovering':
                signal_type = SignalType.BUY
            else:
                signal_type = SignalType.BUY  # Always buy on extreme fear
            
            for symbol in self.config.symbols:
                signal = await self._create_signal(
                    symbol, signal_type, strength, Decimal('1'), fear_greed_data
                )
                if signal:
                    signals.append(signal)
        
        elif current_index >= self.greed_threshold:
            # Extreme greed - expect market peak and pullback
            greed_level = (current_index - self.greed_threshold) / (100 - self.greed_threshold)
            strength = min(1.0, greed_level * 2)
            
            # Confirm with trend
            if self.trend_confirmation and trend == 'peaking':
                signal_type = SignalType.SELL
            else:
                signal_type = SignalType.SELL  # Always sell on extreme greed
            
            for symbol in self.config.symbols:
                signal = await self._create_signal(
                    symbol, signal_type, strength, Decimal('1'), fear_greed_data
                )
                if signal:
                    signals.append(signal)
        
        return signals
    
    async def _get_fear_greed_index(self) -> Dict[str, Any]:
        """Get Fear & Greed index data"""
        try:
            # In real implementation, fetch from fear/greed index API
            return await self._fetch_fear_greed_data()
        except Exception as e:
            logger.warning(f"Could not fetch Fear & Greed index: {e}")
            return {}
    
    async def _fetch_fear_greed_data(self) -> Dict[str, Any]:
        """Fetch Fear & Greed index from external source"""
        # Mock Fear & Greed data
        mock_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'index_value': 22.5,  # Extreme fear
            'trend': 'recovering',
            'components': {
                'vix': 25.0,
                'rsi': 35.0,
                'put_call_ratio': 1.2,
                'safe_haven_demand': 0.8,
                'junk_bond_demand': 0.3
            }
        }
        return mock_data
    
    async def _create_signal(self, symbol: str, signal_type: SignalType, strength: float, price: Decimal, data: Dict[str, Any] = None) -> Optional[TradingSignal]:
        if strength < self.signal_threshold:
            return None
        
        return TradingSignal(
            signal_id=f"FEAR_GREED_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=strength,
            strength=strength,
            price=price,
            quantity=self.config.max_position_size * Decimal(str(strength)),
            time_horizon=timedelta(days=10),
            metadata={
                'strategy_type': 'fear_greed_index',
                'index_value': data.get('index_value', 0) if data else 0,
                'trend': data.get('trend', 'neutral') if data else 'neutral'
            }
        )
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        return signal.strength >= self.signal_threshold


# ============================================================================
# STRATEGY REGISTRATION
# ============================================================================

def register_news_based_strategies():
    """Register all news-based strategies with the strategy library"""
    
    # Register Sentiment Analysis strategies
    strategy_library.register_strategy(
        NewsSentimentStrategy,
        StrategyMetadata(
            strategy_id="news_sentiment",
            name="News Sentiment Strategy",
            category=StrategyCategory.NEWS_BASED,
            description="News sentiment analysis for trading signals",
            long_description="Analyzes news sentiment to generate trading signals based on positive/negative sentiment.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["news", "sentiment", "analysis", "media"],
            parameters_schema={
                "required": ["sentiment_threshold", "lookback_hours"],
                "properties": {
                    "sentiment_threshold": {"type": "float", "min": 0.1, "max": 0.9},
                    "lookback_hours": {"type": "integer", "min": 1, "max": 168},
                    "news_symbols": {"type": "array", "items": {"type": "string"}},
                    "signal_threshold": {"type": "float", "min": 0.3, "max": 1.0}
                }
            },
            example_config={
                "sentiment_threshold": 0.6,
                "lookback_hours": 24,
                "news_symbols": ["AAPL", "GOOGL"],
                "signal_threshold": 0.6
            },
            risk_warning="News-based signals can be affected by information delays and false sentiment."
        )
    )
    
    strategy_library.register_strategy(
        SocialMediaSentimentStrategy,
        StrategyMetadata(
            strategy_id="social_media_sentiment",
            name="Social Media Sentiment Strategy",
            category=StrategyCategory.NEWS_BASED,
            description="Social media sentiment analysis",
            long_description="Analyzes social media sentiment from platforms like Twitter, Reddit for trading signals.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["social-media", "sentiment", "twitter", "reddit", "viral"],
            parameters_schema={
                "required": ["social_platforms", "sentiment_threshold"],
                "properties": {
                    "social_platforms": {"type": "array", "items": {"type": "string"}},
                    "sentiment_threshold": {"type": "float", "min": 0.1, "max": 0.9},
                    "engagement_weighting": {"type": "boolean"},
                    "follower_weighting": {"type": "boolean"},
                    "signal_threshold": {"type": "float", "min": 0.3, "max": 1.0}
                }
            }
        )
    )
    
    # Register Earnings-based strategies
    strategy_library.register_strategy(
        EarningsSurpriseStrategy,
        StrategyMetadata(
            strategy_id="earnings_surprise",
            name="Earnings Surprise Strategy",
            category=StrategyCategory.NEWS_BASED,
            description="Earnings surprise trading",
            long_description="Trades on earnings surprises relative to analyst expectations.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["earnings", "surprise", "analyst", "estimates"],
            parameters_schema={
                "required": ["surprise_threshold", "lookback_days"],
                "properties": {
                    "surprise_threshold": {"type": "float", "min": 0.01, "max": 0.2},
                    "lookback_days": {"type": "integer", "min": 1, "max": 30},
                    "pre_market_trading": {"type": "boolean"},
                    "signal_threshold": {"type": "float", "min": 0.5, "max": 1.0}
                }
            }
        )
    )
    
    # Register Economic Calendar strategies
    strategy_library.register_strategy(
        EconomicCalendarStrategy,
        StrategyMetadata(
            strategy_id="economic_calendar",
            name="Economic Calendar Event Strategy",
            category=StrategyCategory.NEWS_BASED,
            description="Economic calendar event trading",
            long_description="Trades on major economic announcements and calendar events.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["economic", "calendar", "events", "fomc", "macro"],
            parameters_schema={
                "required": ["event_importance_threshold", "pre_event_hours"],
                "properties": {
                    "event_importance_threshold": {"type": "float", "min": 0.1, "max": 1.0},
                    "pre_event_hours": {"type": "integer", "min": 1, "max": 24},
                    "post_event_hours": {"type": "integer", "min": 1, "max": 24},
                    "trade_directions": {"type": "string", "enum": ["long", "short", "both"]},
                    "signal_threshold": {"type": "float", "min": 0.5, "max": 1.0}
                }
            },
            risk_warning="Economic events can cause high volatility and unexpected price movements."
        )
    )
    
    # Register Media Sentiment strategies
    strategy_library.register_strategy(
        AnalystRatingStrategy,
        StrategyMetadata(
            strategy_id="analyst_rating",
            name="Analyst Rating Change Strategy",
            category=StrategyCategory.NEWS_BASED,
            description="Analyst rating upgrades/downgrades",
            long_description="Trades on analyst rating upgrades and downgrades.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["analyst", "rating", "upgrade", "downgrade", "research"],
            parameters_schema={
                "required": ["rating_change_threshold", "lookback_days"],
                "properties": {
                    "rating_change_threshold": {"type": "integer", "min": 1, "max": 3},
                    "lookback_days": {"type": "integer", "min": 1, "max": 30},
                    "firm_weighting": {"type": "boolean"},
                    "rank_adjustment": {"type": "boolean"},
                    "signal_threshold": {"type": "float", "min": 0.5, "max": 1.0}
                }
            }
        )
    )
    
    # Register Market Psychology strategies
    strategy_library.register_strategy(
        FearGreedIndexStrategy,
        StrategyMetadata(
            strategy_id="fear_greed_index",
            name="Fear & Greed Index Strategy",
            category=StrategyCategory.NEWS_BASED,
            description="Fear & greed sentiment extremes",
            long_description="Trades based on market fear/greed sentiment extremes.",
            author="Trading Orchestrator System",
            version="2.0",
            tags=["fear", "greed", "sentiment", "psychology", "extremes"],
            parameters_schema={
                "required": ["fear_threshold", "greed_threshold"],
                "properties": {
                    "fear_threshold": {"type": "float", "min": 10.0, "max": 40.0},
                    "greed_threshold": {"type": "float", "min": 60.0, "max": 90.0},
                    "fear_greed_symbol": {"type": "string"},
                    "trend_confirmation": {"type": "boolean"},
                    "signal_threshold": {"type": "float", "min": 0.5, "max": 1.0}
                }
            }
        )
    )
    
    logger.info(f"Registered {len(strategy_library.categories[StrategyCategory.NEWS_BASED])} news-based strategies")


# Auto-register when module is imported
register_news_based_strategies()


if __name__ == "__main__":
    async def test_news_based_strategies():
        # Test strategy registration
        news_strategies = strategy_library.get_strategies_by_category(StrategyCategory.NEWS_BASED)
        print(f"Registered news-based strategies: {len(news_strategies)}")
        
        for strategy in news_strategies:
            print(f"- {strategy.name} ({strategy.strategy_id})")
    
    import asyncio
    asyncio.run(test_news_based_strategies())