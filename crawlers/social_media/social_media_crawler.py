"""
Social Media Sentiment Crawler
Twitter, Reddit, and StockTwits sentiment analysis
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import json
import re
import hashlib
from urllib.parse import urljoin, urlparse
import aiohttp
from bs4 import BeautifulSoup

from ..base.base_crawler import BaseCrawler, CrawlerConfig, DataType, CrawlResult


@dataclass
class SocialMediaPost:
    """Social media post data"""
    platform: str
    post_id: str
    content: str
    author: str
    author_id: str
    timestamp: datetime
    url: str
    engagement: Dict[str, int]  # likes, shares, comments, etc.
    symbols: List[str] = None
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None  # positive, negative, neutral
    hashtags: List[str] = None
    mentions: List[str] = None
    language: str = "en"
    verified: bool = False
    follower_count: Optional[int] = None
    influence_score: Optional[float] = None


@dataclass
class RedditPost:
    """Reddit post data"""
    post_id: str
    subreddit: str
    title: str
    content: str
    author: str
    timestamp: datetime
    url: str
    score: int
    comments: int
    upvotes: int
    downvotes: int
    symbols: List[str] = None
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    is_self: bool
    external_url: Optional[str] = None


@dataclass
class StockTwitsPost:
    """StockTwits post data"""
    post_id: str
    user: str
    user_id: str
    content: str
    timestamp: datetime
    symbols: List[str] = None
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    watchlist_count: int
    sentiment_count: Dict[str, int] = {'bullish': 0, 'bearish': 0, 'neutral': 0}
    chart_url: Optional[str] = None


class SocialMediaCrawler(BaseCrawler):
    """Crawler for social media sentiment data"""
    
    def __init__(self, config: CrawlerConfig, symbols: List[str]):
        super().__init__(config)
        self.symbols = symbols
        self.logger = logging.getLogger(__name__)
        
        # Platform URLs
        self.endpoints = {
            'twitter': 'https://api.twitter.com/2/tweets/search/recent',
            'reddit': 'https://www.reddit.com/search.json',
            'stocktwits': 'https://api.stocktwits.com/api/2/streams/symbol.json',
            'twitter_v1': 'https://api.twitter.com/1.1/search/tweets.json'
        }
        
        # Sentiment keywords
        self.sentiment_keywords = {
            'bullish': [
                'bullish', 'buy', 'long', 'moon', 'pump', 'rally', 'breakout',
                'support', 'oversold', 'bounce', 'recovery', 'strong', 'growth',
                'beat', 'strong earnings', 'guidance raise'
            ],
            'bearish': [
                'bearish', 'sell', 'short', 'dump', 'crash', 'decline', 'resistance',
                'overbought', 'correction', 'weak', 'miss', 'guidance cut',
                'sell off', 'panic', 'fear'
            ],
            'neutral': [
                'hold', 'wait', 'see', 'monitor', 'stable', 'mixed', 'neutral',
                'sideways', 'consolidation', 'range'
            ]
        }
        
        # Symbol patterns for detection
        self.symbol_patterns = {
            'stock': r'\$[A-Z]{1,5}(?:\.[A-Z]{1,2})?',  # $AAPL, $TSLA.B
            'cashtag': r'\$[A-Z]{1,5}',  # $AAPL
            'ticker': r'\b[A-Z]{1,5}(?:\.[A-Z]{1,2})?\b',  # AAPL, TSLA.B
        }
        
        # Hashtag patterns
        self.hashtag_pattern = r'#(\w+)'
        self.mention_pattern = r'@(\w+)'
    
    async def fetch_twitter_sentiment(self, symbols: List[str] = None) -> List[SocialMediaPost]:
        """Fetch Twitter posts for sentiment analysis"""
        if not symbols:
            symbols = self.symbols
        
        posts = []
        
        try:
            # Build search query for symbols
            search_query = " OR ".join([f"${symbol}" for symbol in symbols[:5]])
            search_query += " (stock OR trading OR market OR financial)"
            
            # Twitter API v2 parameters (would need API key in production)
            params = {
                'query': search_query,
                'tweet.fields': 'created_at,public_metrics,author_id,lang',
                'user.fields': 'verified,public_metrics',
                'expansions': 'author_id',
                'max_results': 50
            }
            
            # For demo purposes, simulate Twitter data
            posts = await self._simulate_twitter_data(symbols[:5])
            
            self.logger.info(f"Fetched {len(posts)} Twitter posts")
            return posts
        
        except Exception as e:
            self.logger.error(f"Error fetching Twitter sentiment: {e}")
            return []
    
    async def _simulate_twitter_data(self, symbols: List[str]) -> List[SocialMediaPost]:
        """Simulate Twitter data for demo purposes"""
        posts = []
        
        # Simulate some sample tweets
        sample_tweets = [
            {
                'content': f'$AAPL looking strong after earnings beat! 📈 #Apple #Stocks',
                'sentiment': 0.8,
                'symbols': ['AAPL'],
                'engagement': {'likes': 45, 'retweets': 12, 'replies': 8}
            },
            {
                'content': f'$TSLA bear case getting stronger. Production concerns mounting. 🐻',
                'sentiment': -0.7,
                'symbols': ['TSLA'],
                'engagement': {'likes': 23, 'retweets': 15, 'replies': 5}
            },
            {
                'content': f'Market looking mixed today. Waiting to see how $MSFT performs.',
                'sentiment': 0.1,
                'symbols': ['MSFT'],
                'engagement': {'likes': 12, 'retweets': 3, 'replies': 2}
            }
        ]
        
        for i, tweet in enumerate(sample_tweets):
            posts.append(SocialMediaPost(
                platform='twitter',
                post_id=f"demo_tweet_{i}",
                content=tweet['content'],
                author=f"user_{i}",
                author_id=f"user_id_{i}",
                timestamp=datetime.now() - timedelta(minutes=i*30),
                url=f"https://twitter.com/user_{i}/status/demo_tweet_{i}",
                engagement=tweet['engagement'],
                symbols=tweet['symbols'],
                sentiment_score=tweet['sentiment'],
                sentiment_label='positive' if tweet['sentiment'] > 0.1 else 'negative' if tweet['sentiment'] < -0.1 else 'neutral',
                hashtags=['Stocks', 'Apple'] if 'Apple' in tweet['content'] else ['Stocks'],
                mentions=[],
                language='en',
                verified=False,
                influence_score=0.5
            ))
        
        return posts
    
    async def fetch_reddit_sentiment(self, symbols: List[str] = None, subreddits: List[str] = None) -> List[RedditPost]:
        """Fetch Reddit posts from trading subreddits"""
        if not symbols:
            symbols = self.symbols
        
        if not subreddits:
            subreddits = ['wallstreetbets', 'stocks', 'investing', 'SecurityAnalysis']
        
        posts = []
        
        try:
            for subreddit in subreddits[:2]:  # Limit to avoid rate limits
                subreddit_posts = await self._fetch_reddit_subreddit(subreddit, symbols)
                posts.extend(subreddit_posts)
            
            self.logger.info(f"Fetched {len(posts)} Reddit posts")
            return posts
        
        except Exception as e:
            self.logger.error(f"Error fetching Reddit sentiment: {e}")
            return []
    
    async def _fetch_reddit_subreddit(self, subreddit: str, symbols: List[str]) -> List[RedditPost]:
        """Fetch posts from a specific subreddit"""
        posts = []
        
        try:
            # Build search query
            symbol_query = " OR ".join(symbols[:3])
            url = f"https://www.reddit.com/r/{subreddit}/search.json"
            params = {
                'q': symbol_query,
                'sort': 'new',
                'limit': 20,
                'restrict_sr': True,
                'syntax': 'lucene'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for post_data in data.get('data', {}).get('children', []):
                            post_info = post_data.get('data', {})
                            
                            # Extract mentioned symbols
                            mentioned_symbols = self._extract_symbols_from_text(
                                post_info.get('title', '') + ' ' + post_info.get('selftext', '')
                            )
                            
                            if mentioned_symbols:
                                posts.append(RedditPost(
                                    post_id=post_info.get('id', ''),
                                    subreddit=subreddit,
                                    title=post_info.get('title', ''),
                                    content=post_info.get('selftext', ''),
                                    author=post_info.get('author', ''),
                                    timestamp=datetime.fromtimestamp(post_info.get('created_utc', 0)),
                                    url=f"https://reddit.com{post_info.get('permalink', '')}",
                                    score=post_info.get('score', 0),
                                    comments=post_info.get('num_comments', 0),
                                    upvotes=post_info.get('ups', 0),
                                    downvotes=post_info.get('downs', 0),
                                    symbols=mentioned_symbols,
                                    is_self=post_info.get('is_self', False),
                                    external_url=post_info.get('url')
                                ))
        
        except Exception as e:
            self.logger.error(f"Error fetching subreddit {subreddit}: {e}")
        
        return posts
    
    async def fetch_stocktwits_sentiment(self, symbols: List[str] = None) -> List[StockTwitsPost]:
        """Fetch StockTwits posts for sentiment analysis"""
        if not symbols:
            symbols = self.symbols
        
        posts = []
        
        try:
            # StockTwits API simulation (would need API key in production)
            posts = await self._simulate_stocktwits_data(symbols[:5])
            
            self.logger.info(f"Fetched {len(posts)} StockTwits posts")
            return posts
        
        except Exception as e:
            self.logger.error(f"Error fetching StockTwits sentiment: {e}")
            return []
    
    async def _simulate_stocktwits_data(self, symbols: List[str]) -> List[StockTwitsPost]:
        """Simulate StockTwits data for demo purposes"""
        posts = []
        
        sample_posts = [
            {
                'content': 'Bullish on $AAPL. Technical breakout confirmed! 📈',
                'sentiment': 0.9,
                'symbols': ['AAPL'],
                'watchlist_count': 1250,
                'sentiment_count': {'bullish': 45, 'bearish': 5, 'neutral': 12}
            },
            {
                'content': 'Bearish divergence on $TSLA. Time to take profits.',
                'sentiment': -0.6,
                'symbols': ['TSLA'],
                'watchlist_count': 890,
                'sentiment_count': {'bullish': 8, 'bearish': 32, 'neutral': 15}
            }
        ]
        
        for i, post in enumerate(sample_posts):
            posts.append(StockTwitsPost(
                post_id=f"st_post_{i}",
                user=f"trader_{i}",
                user_id=f"st_user_{i}",
                content=post['content'],
                timestamp=datetime.now() - timedelta(hours=i),
                symbols=post['symbols'],
                sentiment_score=post['sentiment'],
                sentiment_label='bullish' if post['sentiment'] > 0.5 else 'bearish' if post['sentiment'] < -0.5 else 'neutral',
                watchlist_count=post['watchlist_count'],
                sentiment_count=post['sentiment_count']
            ))
        
        return posts
    
    async def fetch_aggregated_sentiment(self) -> Dict[str, Dict[str, Any]]:
        """Fetch and aggregate sentiment from all platforms"""
        all_data = {
            'twitter': await self.fetch_twitter_sentiment(),
            'reddit': await self.fetch_reddit_sentiment(),
            'stocktwits': await self.fetch_stocktwits_sentiment()
        }
        
        # Calculate aggregated sentiment by symbol
        symbol_sentiment = {}
        
        for platform, posts in all_data.items():
            for post in posts:
                if not post.symbols:
                    continue
                
                for symbol in post.symbols:
                    if symbol not in symbol_sentiment:
                        symbol_sentiment[symbol] = {
                            'sentiment_scores': [],
                            'mention_count': 0,
                            'platforms': set(),
                            'posts': []
                        }
                    
                    symbol_sentiment[symbol]['sentiment_scores'].append(
                        getattr(post, 'sentiment_score', 0) or 0
                    )
                    symbol_sentiment[symbol]['mention_count'] += 1
                    symbol_sentiment[symbol]['platforms'].add(platform)
                    symbol_sentiment[symbol]['posts'].append(post)
        
        # Calculate final sentiment metrics
        for symbol, data in symbol_sentiment.items():
            scores = data['sentiment_scores']
            data['avg_sentiment'] = sum(scores) / len(scores) if scores else 0
            data['sentiment_label'] = (
                'positive' if data['avg_sentiment'] > 0.1 else
                'negative' if data['avg_sentiment'] < -0.1 else
                'neutral'
            )
            data['platforms'] = list(data['platforms'])
            data['confidence'] = min(len(scores) / 10, 1.0)  # Confidence based on mention count
        
        return {
            'platform_data': all_data,
            'symbol_sentiment': symbol_sentiment,
            'timestamp': datetime.now().isoformat()
        }
    
    def _extract_symbols_from_text(self, text: str) -> List[str]:
        """Extract stock symbols from text using patterns"""
        mentioned_symbols = set()
        text_upper = text.upper()
        
        # Find cashtags ($AAPL)
        cashtags = re.findall(self.symbol_patterns['cashtag'], text_upper)
        mentioned_symbols.update(cashtag[1:] for cashtag in cashtags)
        
        # Find standalone tickers (avoid false positives with common words)
        standalone_tickers = re.findall(r'\b[A-Z]{2,5}\b', text_upper)
        common_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE'}
        standalone_tickers = [ticker for ticker in standalone_tickers if ticker not in common_words]
        mentioned_symbols.update(standalone_tickers)
        
        # Filter to only known symbols if available
        if self.symbols:
            mentioned_symbols = [s for s in mentioned_symbols if s in self.symbols]
        else:
            mentioned_symbols = list(mentioned_symbols)
        
        return mentioned_symbols
    
    def _analyze_sentiment(self, content: str, platform: str = 'unknown') -> float:
        """Analyze sentiment of social media content"""
        if not content:
            return 0.0
        
        content_lower = content.lower()
        
        bullish_score = 0
        bearish_score = 0
        total_keywords = 0
        
        # Count sentiment keywords
        for keyword in self.sentiment_keywords['bullish']:
            if keyword in content_lower:
                bullish_score += 1
                total_keywords += 1
        
        for keyword in self.sentiment_keywords['bearish']:
            if keyword in content_lower:
                bearish_score += 1
                total_keywords += 1
        
        # Calculate sentiment score (-1 to 1)
        if total_keywords == 0:
            return 0.0
        
        sentiment_score = (bullish_score - bearish_score) / total_keywords
        return max(-1.0, min(1.0, sentiment_score))
    
    def _calculate_influence_score(self, post: SocialMediaPost) -> float:
        """Calculate influence score based on engagement and author metrics"""
        base_score = 0.5
        
        # Adjust based on engagement
        engagement_total = sum(post.engagement.values())
        if engagement_total > 1000:
            base_score += 0.2
        elif engagement_total > 100:
            base_score += 0.1
        elif engagement_total < 10:
            base_score -= 0.1
        
        # Adjust based on follower count
        if post.follower_count:
            if post.follower_count > 100000:
                base_score += 0.2
            elif post.follower_count > 10000:
                base_score += 0.1
        
        # Adjust based on verification
        if post.verified:
            base_score += 0.1
        
        return max(0.0, min(1.0, base_score))
    
    async def _fetch_data(self) -> Dict[str, Any]:
        """Fetch all social media data"""
        return await self.fetch_aggregated_sentiment()
    
    async def _process_data(self, data: Dict[str, Any], source: str = "live"):
        """Process social media sentiment data"""
        try:
            platform_data = data.get('platform_data', {})
            symbol_sentiment = data.get('symbol_sentiment', {})
            
            # Log sentiment summary
            total_posts = sum(len(platforms) for platforms in platform_data.values())
            self.logger.info(f"Social Media Summary: {total_posts} posts processed")
            
            # Log symbol sentiment
            for symbol, sentiment_data in symbol_sentiment.items():
                label = sentiment_data['sentiment_label']
                score = sentiment_data['avg_sentiment']
                count = sentiment_data['mention_count']
                platforms = ', '.join(sentiment_data['platforms'])
                
                self.logger.info(
                    f"{symbol}: {label.upper()} sentiment "
                    f"(score: {score:.2f}, mentions: {count}, platforms: {platforms})"
                )
            
            # Store data if configured
            if self.config.enable_storage:
                await self._store_social_media_data(data)
        
        except Exception as e:
            self.logger.error(f"Error processing social media data: {e}")
            raise
    
    async def _store_social_media_data(self, data: Dict[str, Any]):
        """Store social media sentiment data"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"social_sentiment_{timestamp}.json"
            filepath = Path(self.config.storage_path) / filename
            
            # Convert data to JSON-serializable format
            json_data = {
                'timestamp': data['timestamp'],
                'platform_data': {},
                'symbol_sentiment': {}
            }
            
            # Process platform data
            for platform, posts in data['platform_data'].items():
                if platform == 'twitter':
                    json_data['platform_data'][platform] = [{
                        'platform': post.platform,
                        'post_id': post.post_id,
                        'content': post.content,
                        'author': post.author,
                        'timestamp': post.timestamp.isoformat(),
                        'url': post.url,
                        'engagement': post.engagement,
                        'symbols': post.symbols or [],
                        'sentiment_score': post.sentiment_score,
                        'sentiment_label': post.sentiment_label,
                        'hashtags': post.hashtags or [],
                        'mentions': post.mentions or [],
                        'verified': post.verified,
                        'influence_score': post.influence_score
                    } for post in posts]
                
                elif platform == 'reddit':
                    json_data['platform_data'][platform] = [{
                        'post_id': post.post_id,
                        'subreddit': post.subreddit,
                        'title': post.title,
                        'content': post.content,
                        'author': post.author,
                        'timestamp': post.timestamp.isoformat(),
                        'url': post.url,
                        'score': post.score,
                        'comments': post.comments,
                        'symbols': post.symbols or [],
                        'sentiment_score': post.sentiment_score,
                        'sentiment_label': post.sentiment_label,
                        'is_self': post.is_self
                    } for post in posts]
                
                elif platform == 'stocktwits':
                    json_data['platform_data'][platform] = [{
                        'post_id': post.post_id,
                        'user': post.user,
                        'content': post.content,
                        'timestamp': post.timestamp.isoformat(),
                        'symbols': post.symbols or [],
                        'sentiment_score': post.sentiment_score,
                        'sentiment_label': post.sentiment_label,
                        'watchlist_count': post.watchlist_count,
                        'sentiment_count': post.sentiment_count
                    } for post in posts]
            
            # Process symbol sentiment
            for symbol, sentiment_data in data['symbol_sentiment'].items():
                json_data['symbol_sentiment'][symbol] = {
                    'avg_sentiment': sentiment_data['avg_sentiment'],
                    'sentiment_label': sentiment_data['sentiment_label'],
                    'mention_count': sentiment_data['mention_count'],
                    'platforms': sentiment_data['platforms'],
                    'confidence': sentiment_data['confidence']
                }
            
            await aiofiles.makedirs(filepath.parent, exist_ok=True)
            async with aiofiles.open(filepath, 'w') as f:
                await f.write(json.dumps(json_data, indent=2, default=str))
            
            self.logger.debug(f"Social media data stored to {filepath}")
        
        except Exception as e:
            self.logger.error(f"Failed to store social media data: {e}")
    
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols"""
        return self.symbols
    
    def get_data_schema(self) -> Dict[str, Any]:
        """Get expected data schema"""
        return {
            'twitter': {
                'platform': 'str',
                'post_id': 'str',
                'content': 'str',
                'author': 'str',
                'timestamp': 'datetime',
                'url': 'str',
                'engagement': 'dict',
                'symbols': 'list[str]',
                'sentiment_score': 'float',
                'sentiment_label': 'str',
                'hashtags': 'list[str]',
                'mentions': 'list[str]',
                'verified': 'bool',
                'influence_score': 'float'
            },
            'reddit': {
                'post_id': 'str',
                'subreddit': 'str',
                'title': 'str',
                'content': 'str',
                'author': 'str',
                'timestamp': 'datetime',
                'url': 'str',
                'score': 'int',
                'comments': 'int',
                'symbols': 'list[str]',
                'sentiment_score': 'float',
                'sentiment_label': 'str',
                'is_self': 'bool'
            },
            'stocktwits': {
                'post_id': 'str',
                'user': 'str',
                'content': 'str',
                'timestamp': 'datetime',
                'symbols': 'list[str]',
                'sentiment_score': 'float',
                'sentiment_label': 'str',
                'watchlist_count': 'int',
                'sentiment_count': 'dict'
            }
        }