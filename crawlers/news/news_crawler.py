"""
News Crawler
Financial news aggregation, earnings announcements, and regulatory filings
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import re
from urllib.parse import urljoin, urlparse
import aiohttp
from bs4 import BeautifulSoup

from ..base.base_crawler import BaseCrawler, CrawlerConfig, DataType, CrawlResult


@dataclass
class NewsArticle:
    """Financial news article"""
    title: str
    content: str
    url: str
    published_date: datetime
    source: str
    summary: Optional[str] = None
    author: Optional[str] = None
    symbols: List[str] = None
    sentiment_score: Optional[float] = None
    relevance_score: Optional[float] = None
    article_type: str = "news"
    language: str = "en"
    image_url: Optional[str] = None


@dataclass
class EarningsAnnouncement:
    """Earnings announcement data"""
    symbol: str
    company_name: str
    announcement_date: datetime
    fiscal_quarter: str
    fiscal_year: int
    revenue: Optional[float] = None
    eps: Optional[float] = None
    revenue_estimate: Optional[float] = None
    eps_estimate: Optional[float] = None
    surprise: Optional[float] = None
    surprise_percent: Optional[float] = None
    guidance: Optional[str] = None
    pre_market_reaction: Optional[float] = None
    source: str = "unknown"


@dataclass
class RegulatoryFiling:
    """SEC regulatory filing"""
    symbol: str
    company_name: str
    filing_type: str
    filing_date: datetime
    period_end_date: Optional[datetime]
    url: str
    summary: Optional[str] = None
    importance_score: Optional[float] = None
    filing_size: Optional[int] = None
    source: str = "SEC"


class NewsCrawler(BaseCrawler):
    """Crawler for financial news and regulatory information"""
    
    def __init__(self, config: CrawlerConfig, symbols: List[str]):
        super().__init__(config)
        self.symbols = symbols
        self.logger = logging.getLogger(__name__)
        
        # News API endpoints
        self.endpoints = {
            'newsapi': 'https://newsapi.org/v2/everything',
            'alpha_vantage': 'https://www.alphavantage.co/query',
            'market_watch': 'https://www.marketwatch.com/investing/stock',
            'yahoo_finance': 'https://finance.yahoo.com/quote',
            'seeking_alpha': 'https://seekingalpha.com/symbol',
            'benzinga': 'https://www.benzinga.com/quote'
        }
        
        # Sentiment analysis patterns
        self.sentiment_patterns = {
            'positive': [
                r'beat expectations?', r'strong performance', r'revenue growth',
                r'profit increase', r'bullish', r'outperform', r'surprise',
                r'guidance raise', r'dividend increase', r'share buyback'
            ],
            'negative': [
                r'missed expectations?', r'weak performance', r'revenue decline',
                r'loss', r'bearish', r'underperform', r'dissappointment',
                r'guidance cut', r'dividend cut', r'regulatory issues'
            ],
            'neutral': [
                r'meets expectations', r'in line', r'flat performance',
                r'no change', r'stable', r'steady'
            ]
        }
        
        # Filing types and their importance
        self.filing_importance = {
            '10-K': 0.9, '10-Q': 0.8, '8-K': 0.7, 'S-1': 0.9,
            '13F': 0.6, '4': 0.5, '3': 0.4, 'SC 13G': 0.6
        }
    
    async def fetch_financial_news(self, symbols: List[str] = None) -> List[NewsArticle]:
        """Fetch financial news articles"""
        if not symbols:
            symbols = self.symbols
        
        articles = []
        
        try:
            # Fetch from multiple sources concurrently
            tasks = [
                self._fetch_from_newsapi(symbols),
                self._fetch_from_yahoo_finance(symbols),
                self._fetch_from_market_watch(symbols),
                self._fetch_from_seeking_alpha(symbols),
                self._fetch_from_benzinga(symbols)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, list):
                    articles.extend(result)
                elif isinstance(result, Exception):
                    self.logger.error(f"Error fetching news: {result}")
            
            # Remove duplicates based on URL
            unique_articles = self._remove_duplicate_articles(articles)
            
            # Process sentiment analysis
            for article in unique_articles:
                article.sentiment_score = self._analyze_sentiment(article.content)
                article.symbols = self._extract_symbols(article.content, symbols)
            
            self.logger.info(f"Fetched {len(unique_articles)} news articles")
            return unique_articles
        
        except Exception as e:
            self.logger.error(f"Error fetching financial news: {e}")
            return []
    
    async def _fetch_from_newsapi(self, symbols: List[str]) -> List[NewsArticle]:
        """Fetch news from NewsAPI"""
        articles = []
        
        try:
            # Note: In production, use actual API key
            api_key = self.config.get('news_api_key', 'demo_key')
            
            for symbol in symbols[:5]:  # Limit to avoid rate limits
                params = {
                    'q': f'{symbol} stock OR financial',
                    'apiKey': api_key,
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': 10
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(self.endpoints['newsapi'], params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            for article_data in data.get('articles', []):
                                articles.append(NewsArticle(
                                    title=article_data.get('title', ''),
                                    content=article_data.get('description', '') or article_data.get('content', ''),
                                    url=article_data.get('url', ''),
                                    published_date=datetime.fromisoformat(
                                        article_data.get('publishedAt', '').replace('Z', '+00:00')
                                    ),
                                    source=article_data.get('source', {}).get('name', 'NewsAPI'),
                                    author=article_data.get('author'),
                                    article_type='news'
                                ))
        except Exception as e:
            self.logger.error(f"Error fetching from NewsAPI: {e}")
        
        return articles
    
    async def _fetch_from_yahoo_finance(self, symbols: List[str]) -> List[NewsArticle]:
        """Fetch news from Yahoo Finance"""
        articles = []
        
        try:
            async with aiohttp.ClientSession() as session:
                for symbol in symbols[:10]:  # Limit to avoid rate limits
                    url = f"{self.endpoints['yahoo_finance']}/{symbol}/news"
                    
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            for news_item in data.get('news', []):
                                articles.append(NewsArticle(
                                    title=news_item.get('title', ''),
                                    content=news_item.get('summary', ''),
                                    url=news_item.get('link', ''),
                                    published_date=datetime.fromtimestamp(news_item.get('providerPublishTime', 0)),
                                    source='Yahoo Finance',
                                    article_type='financial_news'
                                ))
        except Exception as e:
            self.logger.error(f"Error fetching from Yahoo Finance: {e}")
        
        return articles
    
    async def _fetch_from_market_watch(self, symbols: List[str]) -> List[NewsArticle]:
        """Fetch news from MarketWatch"""
        articles = []
        
        try:
            async with aiohttp.ClientSession() as session:
                for symbol in symbols[:5]:
                    url = f"{self.endpoints['market_watch']}/{symbol}/news"
                    
                    async with session.get(url) as response:
                        if response.status == 200:
                            html = await response.text()
                            soup = BeautifulSoup(html, 'html.parser')
                            
                            # Parse MarketWatch articles
                            for article_elem in soup.find_all('article', class_='article'):
                                title_elem = article_elem.find('h3')
                                link_elem = article_elem.find('a')
                                summary_elem = article_elem.find('p')
                                
                                if title_elem and link_elem:
                                    articles.append(NewsArticle(
                                        title=title_elem.get_text(strip=True),
                                        content=summary_elem.get_text(strip=True) if summary_elem else '',
                                        url=urljoin(url, link_elem.get('href', '')),
                                        published_date=datetime.now(),  # Would need to parse actual date
                                        source='MarketWatch',
                                        article_type='financial_news'
                                    ))
        except Exception as e:
            self.logger.error(f"Error fetching from MarketWatch: {e}")
        
        return articles
    
    async def _fetch_from_seeking_alpha(self, symbols: List[str]) -> List[NewsArticle]:
        """Fetch articles from Seeking Alpha"""
        articles = []
        
        try:
            async with aiohttp.ClientSession() as session:
                for symbol in symbols[:3]:
                    url = f"{self.endpoints['seeking_alpha']}/{symbol}/news"
                    
                    async with session.get(url) as response:
                        if response.status == 200:
                            html = await response.text()
                            soup = BeautifulSoup(html, 'html.parser')
                            
                            for article_elem in soup.find_all('div', class_='article'):
                                title_elem = article_elem.find('h2')
                                summary_elem = article_elem.find('p')
                                link_elem = article_elem.find('a')
                                
                                if title_elem and link_elem:
                                    articles.append(NewsArticle(
                                        title=title_elem.get_text(strip=True),
                                        content=summary_elem.get_text(strip=True) if summary_elem else '',
                                        url=urljoin(url, link_elem.get('href', '')),
                                        published_date=datetime.now(),
                                        source='Seeking Alpha',
                                        article_type='analysis'
                                    ))
        except Exception as e:
            self.logger.error(f"Error fetching from Seeking Alpha: {e}")
        
        return articles
    
    async def _fetch_from_benzinga(self, symbols: List[str]) -> List[NewsArticle]:
        """Fetch news from Benzinga"""
        articles = []
        
        try:
            async with aiohttp.ClientSession() as session:
                for symbol in symbols[:3]:
                    url = f"{self.endpoints['benzinga']}/{symbol}/news"
                    
                    async with session.get(url) as response:
                        if response.status == 200:
                            html = await response.text()
                            soup = BeautifulSoup(html, 'html.parser')
                            
                            for article in soup.find_all('article'):
                                title_elem = article.find('h2')
                                summary_elem = article.find('div', class_='article-summary')
                                link_elem = article.find('a')
                                
                                if title_elem and link_elem:
                                    articles.append(NewsArticle(
                                        title=title_elem.get_text(strip=True),
                                        content=summary_elem.get_text(strip=True) if summary_elem else '',
                                        url=urljoin(url, link_elem.get('href', '')),
                                        published_date=datetime.now(),
                                        source='Benzinga',
                                        article_type='market_news'
                                    ))
        except Exception as e:
            self.logger.error(f"Error fetching from Benzinga: {e}")
        
        return articles
    
    async def fetch_earnings_calendar(self, start_date: datetime = None, end_date: datetime = None) -> List[EarningsAnnouncement]:
        """Fetch earnings calendar announcements"""
        if not start_date:
            start_date = datetime.now()
        if not end_date:
            end_date = start_date + timedelta(days=30)
        
        earnings = []
        
        try:
            # Use Alpha Vantage API for earnings data (requires API key)
            api_key = self.config.get('alpha_vantage_api_key', 'demo')
            
            for symbol in self.symbols[:10]:  # Limit to avoid rate limits
                params = {
                    'function': 'EARNINGS',
                    'symbol': symbol,
                    'apikey': api_key
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(self.endpoints['alpha_vantage'], params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            # Parse quarterly earnings
                            for quarter_data in data.get('quarterlyEarnings', []):
                                announcement_date = datetime.fromisoformat(
                                    quarter_data.get('reportedDate', '').replace('Z', '+00:00')
                                )
                                
                                if start_date <= announcement_date <= end_date:
                                    earnings.append(EarningsAnnouncement(
                                        symbol=symbol,
                                        company_name=quarter_data.get('companyName', ''),
                                        announcement_date=announcement_date,
                                        fiscal_quarter=quarter_data.get('fiscalDateEnding', ''),
                                        fiscal_year=int(quarter_data.get('fiscalDateEnding', '2000')[:4]),
                                        revenue=quarter_data.get('reportedRevenue'),
                                        eps=quarter_data.get('reportedEPS'),
                                        revenue_estimate=quarter_data.get('estimatedRevenue'),
                                        eps_estimate=quarter_data.get('estimatedEPS'),
                                        surprise=quarter_data.get('surprisePercentage'),
                                        source='Alpha Vantage'
                                    ))
        except Exception as e:
            self.logger.error(f"Error fetching earnings calendar: {e}")
        
        return earnings
    
    async def fetch_regulatory_filings(self, symbols: List[str] = None) -> List[RegulatoryFiling]:
        """Fetch SEC regulatory filings"""
        if not symbols:
            symbols = self.symbols
        
        filings = []
        
        try:
            # Fetch from SEC EDGAR database
            for symbol in symbols[:5]:  # Limit to avoid rate limits
                # Get CIK from symbol
                cik = await self._symbol_to_cik(symbol)
                if not cik:
                    continue
                
                # Fetch recent filings
                sec_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(sec_url) as response:
                        if response.status == 200:
                            data = await response.json()
                            recent_filings = data.get('filings', {}).get('recent', {})
                            
                            for i, filing_date in enumerate(recent_filings.get('filingDate', [])):
                                filing_type = recent_filings.get('form', [''])[i]
                                accession_number = recent_filings.get('accessionNumber', [''])[i]
                                
                                if filing_type in self.filing_importance:
                                    filings.append(RegulatoryFiling(
                                        symbol=symbol,
                                        company_name=data.get('name', ''),
                                        filing_type=filing_type,
                                        filing_date=datetime.strptime(filing_date, '%Y-%m-%d'),
                                        period_end_date=datetime.strptime(
                                            recent_filings.get('reportDate', [''])[i], '%Y-%m-%d'
                                        ) if recent_filings.get('reportDate') else None,
                                        url=f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}/",
                                        importance_score=self.filing_importance.get(filing_type, 0.5),
                                        source='SEC'
                                    ))
        
        except Exception as e:
            self.logger.error(f"Error fetching regulatory filings: {e}")
        
        return filings
    
    async def _symbol_to_cik(self, symbol: str) -> Optional[str]:
        """Convert stock symbol to SEC CIK"""
        # This would typically query SEC's symbol to CIK mapping
        # For demo purposes, returning placeholder
        return "0000320193"  # Apple's CIK
    
    def _remove_duplicate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles based on URL"""
        seen_urls = set()
        unique_articles = []
        
        for article in articles:
            if article.url not in seen_urls:
                seen_urls.add(article.url)
                unique_articles.append(article)
        
        return unique_articles
    
    def _analyze_sentiment(self, content: str) -> Optional[float]:
        """Analyze sentiment of news content"""
        if not content:
            return None
        
        content_lower = content.lower()
        
        positive_score = 0
        negative_score = 0
        total_patterns = 0
        
        # Count sentiment patterns
        for pattern in self.sentiment_patterns['positive']:
            matches = len(re.findall(pattern, content_lower, re.IGNORECASE))
            positive_score += matches
            total_patterns += matches
        
        for pattern in self.sentiment_patterns['negative']:
            matches = len(re.findall(pattern, content_lower, re.IGNORECASE))
            negative_score += matches
            total_patterns += matches
        
        if total_patterns == 0:
            return 0.0  # Neutral
        
        # Calculate sentiment score (-1 to 1)
        sentiment_score = (positive_score - negative_score) / total_patterns
        return max(-1.0, min(1.0, sentiment_score))
    
    def _extract_symbols(self, content: str, target_symbols: List[str]) -> List[str]:
        """Extract mentioned symbols from content"""
        mentioned_symbols = []
        content_upper = content.upper()
        
        for symbol in target_symbols:
            if symbol.upper() in content_upper:
                mentioned_symbols.append(symbol)
        
        return mentioned_symbols
    
    async def _fetch_data(self) -> Dict[str, Any]:
        """Fetch all news data types"""
        all_data = {
            'news': await self.fetch_financial_news(),
            'earnings': await self.fetch_earnings_calendar(),
            'filings': await self.fetch_regulatory_filings()
        }
        
        return all_data
    
    async def _process_data(self, data: Dict[str, Any], source: str = "live"):
        """Process news and financial data"""
        try:
            news_data = data.get('news', [])
            earnings_data = data.get('earnings', [])
            filings_data = data.get('filings', [])
            
            # Log news summary
            if news_data:
                positive_news = [n for n in news_data if (n.sentiment_score or 0) > 0.1]
                negative_news = [n for n in news_data if (n.sentiment_score or 0) < -0.1]
                
                self.logger.info(f"News Summary: {len(news_data)} articles, "
                               f"{len(positive_news)} positive, {len(negative_news)} negative")
            
            # Log earnings announcements
            if earnings_data:
                self.logger.info(f"Earnings: {len(earnings_data)} announcements found")
            
            # Log regulatory filings
            if filings_data:
                high_importance = [f for f in filings_data if f.importance_score > 0.7]
                self.logger.info(f"Filings: {len(filings_data)} total, "
                               f"{len(high_importance)} high importance")
            
            # Store data if configured
            if self.config.enable_storage:
                await self._store_news_data(data)
        
        except Exception as e:
            self.logger.error(f"Error processing news data: {e}")
            raise
    
    async def _store_news_data(self, data: Dict[str, Any]):
        """Store news data to files"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Store news articles
            news_data = data.get('news', [])
            if news_data:
                news_file = Path(self.config.storage_path) / f"news_{timestamp}.json"
                news_json = [{
                    'title': article.title,
                    'content': article.content,
                    'url': article.url,
                    'published_date': article.published_date.isoformat(),
                    'source': article.source,
                    'summary': article.summary,
                    'author': article.author,
                    'symbols': article.symbols or [],
                    'sentiment_score': article.sentiment_score,
                    'article_type': article.article_type,
                    'language': article.language
                } for article in news_data]
                
                await aiofiles.makedirs(news_file.parent, exist_ok=True)
                async with aiofiles.open(news_file, 'w') as f:
                    await f.write(json.dumps(news_json, indent=2, default=str))
            
            # Store earnings data
            earnings_data = data.get('earnings', [])
            if earnings_data:
                earnings_file = Path(self.config.storage_path) / f"earnings_{timestamp}.json"
                earnings_json = [{
                    'symbol': earning.symbol,
                    'company_name': earning.company_name,
                    'announcement_date': earning.announcement_date.isoformat(),
                    'fiscal_quarter': earning.fiscal_quarter,
                    'fiscal_year': earning.fiscal_year,
                    'revenue': earning.revenue,
                    'eps': earning.eps,
                    'revenue_estimate': earning.revenue_estimate,
                    'eps_estimate': earning.eps_estimate,
                    'surprise': earning.surprise,
                    'source': earning.source
                } for earning in earnings_data]
                
                async with aiofiles.open(earnings_file, 'w') as f:
                    await f.write(json.dumps(earnings_json, indent=2, default=str))
        
        except Exception as e:
            self.logger.error(f"Failed to store news data: {e}")
    
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols"""
        return self.symbols
    
    def get_data_schema(self) -> Dict[str, Any]:
        """Get expected data schema"""
        return {
            'news': {
                'title': 'str',
                'content': 'str',
                'url': 'str',
                'published_date': 'datetime',
                'source': 'str',
                'summary': 'str (optional)',
                'author': 'str (optional)',
                'symbols': 'list[str] (optional)',
                'sentiment_score': 'float (optional)',
                'article_type': 'str',
                'language': 'str'
            },
            'earnings': {
                'symbol': 'str',
                'company_name': 'str',
                'announcement_date': 'datetime',
                'fiscal_quarter': 'str',
                'fiscal_year': 'int',
                'revenue': 'float (optional)',
                'eps': 'float (optional)',
                'revenue_estimate': 'float (optional)',
                'eps_estimate': 'float (optional)',
                'surprise': 'float (optional)',
                'source': 'str'
            },
            'filings': {
                'symbol': 'str',
                'company_name': 'str',
                'filing_type': 'str',
                'filing_date': 'datetime',
                'period_end_date': 'datetime (optional)',
                'url': 'str',
                'summary': 'str (optional)',
                'importance_score': 'float (optional)',
                'source': 'str'
            }
        }