"""
Market Data Crawler System
Comprehensive crawler framework for financial data collection
"""

from .base.base_crawler import BaseCrawler
from .market_data.market_data_crawler import MarketDataCrawler
from .news.news_crawler import NewsCrawler
from .social_media.social_media_crawler import SocialMediaCrawler
from .economic.economic_crawler import EconomicCrawler
from .patterns.pattern_crawler import PatternCrawler

__all__ = [
    'BaseCrawler',
    'MarketDataCrawler', 
    'NewsCrawler',
    'SocialMediaCrawler',
    'EconomicCrawler',
    'PatternCrawler'
]

__version__ = '1.0.0'