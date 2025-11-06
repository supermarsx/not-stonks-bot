"""
Crawler Trading System Integration

This module provides seamless integration between the comprehensive crawler system
and the trading orchestrator, enabling real-time data flow from crawlers to
trading strategies and risk management systems.
"""

from .trading_integration import CrawlerTradingIntegrator
from .data_bridge import CrawlerDataBridge
from .event_handler import CrawlerEventHandler

__all__ = [
    'CrawlerTradingIntegrator',
    'CrawlerDataBridge',
    'CrawlerEventHandler'
]