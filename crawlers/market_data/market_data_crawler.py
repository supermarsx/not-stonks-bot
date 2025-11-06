"""
Market Data Crawler
Collects real-time prices, historical data, and intraday data
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import pandas as pd
import numpy as np
import aiohttp

from ..base.base_crawler import BaseCrawler, CrawlerConfig, DataType, CrawlResult


@dataclass
class MarketDataPoint:
    """Individual market data point"""
    symbol: str
    timestamp: datetime
    price: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    open: Optional[float] = None
    close: Optional[float] = None
    change: Optional[float] = None
    change_percent: Optional[float] = None
    market_cap: Optional[float] = None
    source: str = "unknown"


@dataclass
class HistoricalDataPoint:
    """Historical market data point"""
    symbol: str
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    adj_close: Optional[float] = None
    source: str = "historical"


class MarketDataCrawler(BaseCrawler):
    """Crawler for market data - prices, historical, and intraday"""
    
    def __init__(self, config: CrawlerConfig, symbols: List[str]):
        super().__init__(config)
        self.symbols = symbols
        self.logger = logging.getLogger(__name__)
        
        # Yahoo Finance API endpoints
        self.base_urls = {
            'quote': 'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}',
            'historical': 'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}',
            'intraday': 'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}',
            'quote_summary': 'https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}'
        }
        
        # Real-time data cache
        self.latest_quotes: Dict[str, MarketDataPoint] = {}
        
        # Performance tracking
        self.data_quality_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0,
            'data_completeness': 0.0
        }
    
    async def fetch_real_time_data(self) -> Dict[str, MarketDataPoint]:
        """Fetch real-time market data for all symbols"""
        results = {}
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for symbol in self.symbols:
                task = self._fetch_symbol_quote(session, symbol)
                tasks.append(task)
            
            try:
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                for symbol, response in zip(self.symbols, responses):
                    if isinstance(response, Exception):
                        self.logger.error(f"Error fetching {symbol}: {response}")
                        self.data_quality_metrics['failed_requests'] += 1
                        continue
                    
                    if response:
                        results[symbol] = response
                        self.latest_quotes[symbol] = response
                        self.data_quality_metrics['successful_requests'] += 1
                    else:
                        self.data_quality_metrics['failed_requests'] += 1
                
                self.data_quality_metrics['total_requests'] += len(self.symbols)
                
            except Exception as e:
                self.logger.error(f"Error in batch fetch: {e}")
                raise
        
        return results
    
    async def _fetch_symbol_quote(self, session: aiohttp.ClientSession, symbol: str) -> Optional[MarketDataPoint]:
        """Fetch quote for a single symbol"""
        try:
            url = self.base_urls['quote'].format(symbol=symbol)
            params = {
                'interval': '1m',
                'range': '1d',
                'includePrePost': 'true'
            }
            
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    self.logger.warning(f"HTTP {response.status} for {symbol}")
                    return None
                
                data = await response.json()
                
                if not data or 'chart' not in data:
                    self.logger.warning(f"No data returned for {symbol}")
                    return None
                
                chart_data = data['chart']
                if not chart_data['result']:
                    self.logger.warning(f"No result for {symbol}")
                    return None
                
                result = chart_data['result'][0]
                meta = result['meta']
                
                # Get latest price
                timestamps = result['timestamp']
                prices = result['indicators']['quote'][0]
                
                if not timestamps:
                    return None
                
                latest_idx = len(timestamps) - 1
                latest_timestamp = timestamps[latest_idx]
                latest_price_data = {
                    'open': prices['open'][latest_idx],
                    'high': prices['high'][latest_idx],
                    'low': prices['low'][latest_idx],
                    'close': prices['close'][latest_idx],
                    'volume': prices['volume'][latest_idx]
                }
                
                if not latest_price_data['close']:
                    return None
                
                # Calculate change
                previous_close = meta.get('previousClose')
                change = None
                change_percent = None
                
                if previous_close and previous_close != 0:
                    change = latest_price_data['close'] - previous_close
                    change_percent = (change / previous_close) * 100
                
                return MarketDataPoint(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(latest_timestamp),
                    price=latest_price_data['close'],
                    volume=latest_price_data['volume'],
                    high=latest_price_data['high'],
                    low=latest_price_data['low'],
                    open=latest_price_data['open'],
                    close=latest_price_data['close'],
                    change=change,
                    change_percent=change_percent,
                    market_cap=meta.get('marketCap'),
                    source='yahoo_finance'
                )
        
        except Exception as e:
            self.logger.error(f"Error fetching quote for {symbol}: {e}")
            return None
    
    async def fetch_historical_data(self, symbol: str, period: str = "1y") -> List[HistoricalDataPoint]:
        """Fetch historical data for a symbol"""
        try:
            url = self.base_urls['historical'].format(symbol=symbol)
            params = {
                'interval': '1d',
                'range': period,
                'includePrePost': 'true'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        self.logger.error(f"HTTP {response.status} for historical data of {symbol}")
                        return []
                    
                    data = await response.json()
                    
                    if not data or 'chart' not in data or not data['chart']['result']:
                        self.logger.warning(f"No historical data for {symbol}")
                        return []
                    
                    result = data['chart']['result'][0]
                    timestamps = result['timestamp']
                    prices = result['indicators']['quote'][0]
                    
                    historical_data = []
                    for i, timestamp in enumerate(timestamps):
                        if prices['open'][i] and prices['high'][i] and prices['low'][i] and prices['close'][i]:
                            historical_data.append(HistoricalDataPoint(
                                symbol=symbol,
                                timestamp=datetime.fromtimestamp(timestamp),
                                open_price=prices['open'][i],
                                high_price=prices['high'][i],
                                low_price=prices['low'][i],
                                close_price=prices['close'][i],
                                volume=prices['volume'][i],
                                adj_close=prices.get('adjclose', [None]*len(timestamps))[i] if 'adjclose' in result['indicators'] else None,
                                source='yahoo_finance'
                            ))
                    
                    self.logger.info(f"Fetched {len(historical_data)} historical data points for {symbol}")
                    return historical_data
        
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {e}")
            return []
    
    async def fetch_intraday_data(self, symbol: str, interval: str = "1m") -> List[MarketDataPoint]:
        """Fetch intraday data for a symbol"""
        try:
            url = self.base_urls['intraday'].format(symbol=symbol)
            params = {
                'interval': interval,
                'range': '1d',
                'includePrePost': 'true'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        self.logger.error(f"HTTP {response.status} for intraday data of {symbol}")
                        return []
                    
                    data = await response.json()
                    
                    if not data or 'chart' not in data or not data['chart']['result']:
                        self.logger.warning(f"No intraday data for {symbol}")
                        return []
                    
                    result = data['chart']['result'][0]
                    timestamps = result['timestamp']
                    prices = result['indicators']['quote'][0]
                    
                    intraday_data = []
                    for i, timestamp in enumerate(timestamps):
                        if prices['close'][i]:
                            intraday_data.append(MarketDataPoint(
                                symbol=symbol,
                                timestamp=datetime.fromtimestamp(timestamp),
                                price=prices['close'][i],
                                volume=prices['volume'][i],
                                high=prices['high'][i],
                                low=prices['low'][i],
                                open=prices['open'][i],
                                close=prices['close'][i],
                                source='yahoo_finance_intraday'
                            ))
                    
                    self.logger.info(f"Fetched {len(intraday_data)} intraday data points for {symbol}")
                    return intraday_data
        
        except Exception as e:
            self.logger.error(f"Error fetching intraday data for {symbol}: {e}")
            return []
    
    async def _fetch_data(self) -> Dict[str, List[Any]]:
        """Fetch all types of market data"""
        all_data = {
            'real_time': await self.fetch_real_time_data(),
            'intraday': {},
            'historical': {}
        }
        
        # Fetch intraday data for first few symbols to avoid rate limits
        for symbol in self.symbols[:5]:  # Limit to avoid rate limiting
            intraday = await self.fetch_intraday_data(symbol)
            if intraday:
                all_data['intraday'][symbol] = intraday
        
        # Fetch historical data for first few symbols
        for symbol in self.symbols[:3]:  # Limit to avoid rate limiting
            historical = await self.fetch_historical_data(symbol)
            if historical:
                all_data['historical'][symbol] = historical
        
        return all_data
    
    async def _process_data(self, data: Dict[str, Any], source: str = "live"):
        """Process market data"""
        try:
            real_time_data = data.get('real_time', {})
            if real_time_data:
                # Log market summary
                for symbol, quote in real_time_data.items():
                    if quote.change_percent is not None:
                        direction = "📈" if quote.change_percent > 0 else "📉" if quote.change_percent < 0 else "➡️"
                        self.logger.info(
                            f"{symbol}: ${quote.price:.2f} {direction} "
                            f"{quote.change_percent:+.2f}% ({quote.change:+.2f}) "
                            f"Vol: {quote.volume:,.0f}"
                        )
            
            # Store to database/storage (if configured)
            if self.config.enable_storage:
                await self._store_market_data(data)
        
        except Exception as e:
            self.logger.error(f"Error processing market data: {e}")
            raise
    
    async def _store_market_data(self, data: Dict[str, Any]):
        """Store market data to files"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Store real-time data
            real_time_data = data.get('real_time', {})
            if real_time_data:
                rt_file = Path(self.config.storage_path) / f"real_time_{timestamp}.json"
                rt_data = {symbol: {
                    'symbol': quote.symbol,
                    'timestamp': quote.timestamp.isoformat(),
                    'price': quote.price,
                    'volume': quote.volume,
                    'high': quote.high,
                    'low': quote.low,
                    'open': quote.open,
                    'close': quote.close,
                    'change': quote.change,
                    'change_percent': quote.change_percent,
                    'market_cap': quote.market_cap,
                    'source': quote.source
                } for symbol, quote in real_time_data.items()}
                
                await aiofiles.makedirs(rt_file.parent, exist_ok=True)
                async with aiofiles.open(rt_file, 'w') as f:
                    await f.write(json.dumps(rt_data, indent=2, default=str))
            
            # Store historical data
            historical_data = data.get('historical', {})
            if historical_data:
                hist_file = Path(self.config.storage_path) / f"historical_{timestamp}.json"
                hist_data = {}
                for symbol, hist_points in historical_data.items():
                    hist_data[symbol] = [{
                        'symbol': point.symbol,
                        'timestamp': point.timestamp.isoformat(),
                        'open_price': point.open_price,
                        'high_price': point.high_price,
                        'low_price': point.low_price,
                        'close_price': point.close_price,
                        'volume': point.volume,
                        'adj_close': point.adj_close,
                        'source': point.source
                    } for point in hist_points]
                
                async with aiofiles.open(hist_file, 'w') as f:
                    await f.write(json.dumps(hist_data, indent=2, default=str))
        
        except Exception as e:
            self.logger.error(f"Failed to store market data: {e}")
    
    async def get_latest_quote(self, symbol: str) -> Optional[MarketDataPoint]:
        """Get latest quote for a specific symbol"""
        return self.latest_quotes.get(symbol)
    
    async def get_price_history(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get price history as DataFrame"""
        historical_data = await self.fetch_historical_data(symbol, f"{days}d")
        
        if not historical_data:
            return pd.DataFrame()
        
        df = pd.DataFrame([{
            'timestamp': point.timestamp,
            'open': point.open_price,
            'high': point.high_price,
            'low': point.low_price,
            'close': point.close_price,
            'volume': point.volume,
            'adj_close': point.adj_close
        } for point in historical_data])
        
        df.set_index('timestamp', inplace=True)
        return df
    
    async def calculate_technical_indicators(self, symbol: str, period: str = "1y") -> Dict[str, float]:
        """Calculate basic technical indicators"""
        df = await self.get_price_history(symbol, days=365 if period == "1y" else 30)
        
        if df.empty:
            return {}
        
        indicators = {}
        
        # Simple Moving Averages
        if len(df) >= 20:
            indicators['sma_20'] = df['close'].rolling(20).mean().iloc[-1]
        if len(df) >= 50:
            indicators['sma_50'] = df['close'].rolling(50).mean().iloc[-1]
        if len(df) >= 200:
            indicators['sma_200'] = df['close'].rolling(200).mean().iloc[-1]
        
        # RSI
        if len(df) >= 14:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = (100 - (100 / (1 + rs))).iloc[-1]
        
        # Bollinger Bands
        if len(df) >= 20:
            sma_20 = df['close'].rolling(20).mean()
            std_20 = df['close'].rolling(20).std()
            indicators['bb_upper'] = (sma_20 + (2 * std_20)).iloc[-1]
            indicators['bb_lower'] = (sma_20 - (2 * std_20)).iloc[-1]
            indicators['bb_width'] = indicators['bb_upper'] - indicators['bb_lower']
        
        # Current price and change
        if not df.empty:
            indicators['current_price'] = df['close'].iloc[-1]
            if len(df) > 1:
                indicators['price_change_1d'] = df['close'].iloc[-1] - df['close'].iloc[-2]
                indicators['price_change_pct_1d'] = ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
        
        return indicators
    
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols"""
        return self.symbols
    
    def get_data_schema(self) -> Dict[str, Any]:
        """Get expected data schema"""
        return {
            'real_time': {
                'symbol': 'str',
                'timestamp': 'datetime',
                'price': 'float',
                'volume': 'float',
                'bid': 'float (optional)',
                'ask': 'float (optional)',
                'high': 'float (optional)',
                'low': 'float (optional)',
                'open': 'float (optional)',
                'close': 'float (optional)',
                'change': 'float (optional)',
                'change_percent': 'float (optional)',
                'market_cap': 'float (optional)',
                'source': 'str'
            },
            'historical': {
                'symbol': 'str',
                'timestamp': 'datetime',
                'open_price': 'float',
                'high_price': 'float',
                'low_price': 'float',
                'close_price': 'float',
                'volume': 'float',
                'adj_close': 'float (optional)',
                'source': 'str'
            }
        }