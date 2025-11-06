"""
Economic Indicators Crawler
Economic data feeds, central bank announcements, and economic calendar
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import pandas as pd
import aiohttp
from bs4 import BeautifulSoup

from ..base.base_crawler import BaseCrawler, CrawlerConfig, DataType, CrawlResult


@dataclass
class EconomicIndicator:
    """Economic indicator data point"""
    indicator_name: str
    value: float
    unit: str
    timestamp: datetime
    previous_value: Optional[float] = None
    forecast: Optional[float] = None
    country: str = "US"
    frequency: str = "monthly"  # daily, weekly, monthly, quarterly, annually
    importance: str = "medium"  # low, medium, high, critical
    source: str = "unknown"
    release_date: Optional[datetime] = None


@dataclass
class CentralBankAnnouncement:
    """Central bank policy announcement"""
    bank_name: str
    announcement_date: datetime
    decision: str  # hike, cut, hold
    rate_change: Optional[float] = None
    new_rate: Optional[float] = None
    previous_rate: Optional[float] = None
    statement: Optional[str] = None
    meeting_minutes: Optional[str] = None
    forward_guidance: Optional[str] = None
    press_conference_url: Optional[str] = None
    importance: str = "high"
    market_impact: Optional[str] = None


@dataclass
class EconomicCalendarEvent:
    """Economic calendar event"""
    event_name: str
    date: datetime
    country: str
    currency: str
    importance: str  # low, medium, high, critical
    actual_value: Optional[float] = None
    forecast_value: Optional[float] = None
    previous_value: Optional[float] = None
    unit: str = ""
    frequency: str = "monthly"
    release_time: Optional[str] = None
    description: Optional[str] = None
    source: str = "economic_calendar"
    volatility_expected: float = 0.0  # 0-1 scale


@dataclass
class MarketSentiment:
    """Market sentiment indicators"""
    timestamp: datetime
    fear_greed_index: Optional[float] = None
    vix_level: Optional[float] = None
    put_call_ratio: Optional[float] = None
    advance_decline_ratio: Optional[float] = None
    credit_spread: Optional[float] = None
    treasury_yield: Dict[str, float] = None  # 2Y, 5Y, 10Y, 30Y
    dollar_index: Optional[float] = None
    gold_price: Optional[float] = None
    oil_price: Optional[float] = None
    emerging_markets_index: Optional[float] = None


class EconomicCrawler(BaseCrawler):
    """Crawler for economic indicators and market sentiment"""
    
    def __init__(self, config: CrawlerConfig):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # API endpoints
        self.endpoints = {
            'fred': 'https://api.stlouisfed.org/fred/series/observations',
            'trading_economics': 'https://api.tradingeconomics.com/calendar/country',
            'alpha_vantage': 'https://www.alphavantage.co/query',
            'yahoo_finance': 'https://query1.finance.yahoo.com/v8/finance/chart',
            'federal_reserve': 'https://www.federalreserve.gov/monetarypolicy',
            'ecb': 'https://www.ecb.europa.eu/press',
            'bank_of_england': 'https://www.bankofengland.co.uk/news',
            'boe_japan': 'https://www.boj.or.jp/en/'
        }
        
        # Economic indicators to monitor
        self.key_indicators = {
            'GDP': {'series_id': 'GDPC1', 'importance': 'critical'},
            'Inflation_CPI': {'series_id': 'CPIAUCSL', 'importance': 'critical'},
            'Unemployment': {'series_id': 'UNRATE', 'importance': 'high'},
            'Interest_Rate': {'series_id': 'FEDFUNDS', 'importance': 'critical'},
            'Industrial_Production': {'series_id': 'INDPRO', 'importance': 'medium'},
            'Retail_Sales': {'series_id': 'RSXFS', 'importance': 'medium'},
            'Consumer_Confidence': {'series_id': 'UMCSENT', 'importance': 'medium'},
            'Housing_Starts': {'series_id': 'HOUST', 'importance': 'medium'},
            'Trade_Balance': {'series_id': 'BOPGSB', 'importance': 'low'}
        }
        
        # Central banks to monitor
        self.central_banks = {
            'Federal Reserve': {'url': self.endpoints['federal_reserve'], 'country': 'US'},
            'European Central Bank': {'url': self.endpoints['ecb'], 'country': 'EU'},
            'Bank of England': {'url': self.endpoints['bank_of_england'], 'country': 'UK'},
            'Bank of Japan': {'url': self.endpoints['boe_japan'], 'country': 'JP'}
        }
        
        # Market sentiment indicators
        self.sentiment_symbols = {
            'VIX': '^VIX',  # Volatility Index
            'DXY': 'DX-Y.NYB',  # Dollar Index
            'TNX': '^TNX',  # 10-Year Treasury
            'GLD': 'GLD',  # Gold ETF
            'USO': 'USO',  # Oil ETF
            'EEM': 'EEM',  # Emerging Markets ETF
            'PutCallRatio': 'PUTCALL'  # Put/Call Ratio
        }
    
    async def fetch_economic_indicators(self) -> List[EconomicIndicator]:
        """Fetch key economic indicators"""
        indicators = []
        
        try:
            # Fetch from FRED (Federal Reserve Economic Data)
            fred_indicators = await self._fetch_fred_indicators()
            indicators.extend(fred_indicators)
            
            # Additional economic data from other sources
            additional_indicators = await self._fetch_additional_economic_data()
            indicators.extend(additional_indicators)
            
            self.logger.info(f"Fetched {len(indicators)} economic indicators")
            return indicators
        
        except Exception as e:
            self.logger.error(f"Error fetching economic indicators: {e}")
            return []
    
    async def _fetch_fred_indicators(self) -> List[EconomicIndicator]:
        """Fetch indicators from FRED API"""
        indicators = []
        
        try:
            api_key = self.config.get('fred_api_key', 'demo')
            
            for indicator_name, config in self.key_indicators.items():
                series_id = config['series_id']
                url = self.endpoints['fred']
                
                params = {
                    'series_id': series_id,
                    'api_key': api_key,
                    'file_type': 'json',
                    'limit': 2  # Get latest and previous value
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            observations = data.get('observations', [])
                            
                            if len(observations) >= 2:
                                latest = observations[-1]
                                previous = observations[-2]
                                
                                try:
                                    indicators.append(EconomicIndicator(
                                        indicator_name=indicator_name,
                                        value=float(latest['value']) if latest['value'] != '.' else None,
                                        unit=self._get_indicator_unit(indicator_name),
                                        timestamp=datetime.strptime(latest['date'], '%Y-%m-%d'),
                                        previous_value=float(previous['value']) if previous['value'] != '.' else None,
                                        country='US',
                                        frequency='monthly',
                                        importance=config['importance'],
                                        source='FRED',
                                        release_date=datetime.strptime(latest['date'], '%Y-%m-%d')
                                    ))
                                except (ValueError, KeyError):
                                    continue
        except Exception as e:
            self.logger.error(f"Error fetching FRED data: {e}")
        
        return indicators
    
    async def _fetch_additional_economic_data(self) -> List[EconomicIndicator]:
        """Fetch additional economic data from other sources"""
        indicators = []
        
        try:
            # Simulate some key economic indicators
            sample_indicators = [
                EconomicIndicator(
                    indicator_name='Consumer_Price_Index',
                    value=305.109,
                    unit='Index 1982-84=100',
                    timestamp=datetime.now() - timedelta(days=30),
                    previous_value=304.127,
                    forecast=305.200,
                    country='US',
                    frequency='monthly',
                    importance='critical',
                    source='BLS',
                    release_date=datetime.now() - timedelta(days=30)
                ),
                EconomicIndicator(
                    indicator_name='Federal_Funds_Rate',
                    value=5.25,
                    unit='Percent',
                    timestamp=datetime.now() - timedelta(days=7),
                    previous_value=5.00,
                    country='US',
                    frequency='monthly',
                    importance='critical',
                    source='Federal Reserve',
                    release_date=datetime.now() - timedelta(days=7)
                ),
                EconomicIndicator(
                    indicator_name='Unemployment_Rate',
                    value=3.8,
                    unit='Percent',
                    timestamp=datetime.now() - timedelta(days=5),
                    previous_value=3.9,
                    forecast=3.8,
                    country='US',
                    frequency='monthly',
                    importance='high',
                    source='BLS',
                    release_date=datetime.now() - timedelta(days=5)
                )
            ]
            
            indicators.extend(sample_indicators)
        
        except Exception as e:
            self.logger.error(f"Error fetching additional economic data: {e}")
        
        return indicators
    
    async def fetch_central_bank_announcements(self, days_ahead: int = 30) -> List[CentralBankAnnouncement]:
        """Fetch central bank policy announcements"""
        announcements = []
        end_date = datetime.now() + timedelta(days=days_ahead)
        
        try:
            # Fetch from Federal Reserve
            fed_announcements = await self._fetch_fed_announcements()
            announcements.extend(fed_announcements)
            
            # Simulate ECB announcements
            ecb_announcements = await self._simulate_ecb_announcements()
            announcements.extend(ecb_announcements)
            
            # Simulate other central bank announcements
            other_announcements = await self._simulate_other_central_banks()
            announcements.extend(other_announcements)
            
            # Filter by date range
            filtered_announcements = [
                ann for ann in announcements
                if ann.announcement_date <= end_date
            ]
            
            self.logger.info(f"Fetched {len(filtered_announcements)} central bank announcements")
            return filtered_announcements
        
        except Exception as e:
            self.logger.error(f"Error fetching central bank announcements: {e}")
            return []
    
    async def _fetch_fed_announcements(self) -> List[CentralBankAnnouncement]:
        """Fetch Federal Reserve announcements"""
        announcements = []
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.endpoints['federal_reserve']) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Parse Fed announcements (simplified)
                        for article in soup.find_all('article', limit=10):
                            title_elem = article.find('h2') or article.find('h3')
                            date_elem = article.find('time')
                            
                            if title_elem and date_elem:
                                # This is a simplified parsing - in production would be more robust
                                announcements.append(CentralBankAnnouncement(
                                    bank_name='Federal Reserve',
                                    announcement_date=datetime.now(),  # Would parse actual date
                                    decision='hold',  # Would analyze content
                                    statement=title_elem.get_text(strip=True),
                                    importance='high'
                                ))
        except Exception as e:
            self.logger.error(f"Error fetching Fed announcements: {e}")
        
        return announcements
    
    async def _simulate_ecb_announcements(self) -> List[CentralBankAnnouncement]:
        """Simulate ECB announcements for demo"""
        return [
            CentralBankAnnouncement(
                bank_name='European Central Bank',
                announcement_date=datetime.now() + timedelta(days=14),
                decision='hold',
                new_rate=4.00,
                previous_rate=4.00,
                statement='ECB maintains current monetary policy stance',
                forward_guidance='Data-dependent approach continues',
                importance='high'
            )
        ]
    
    async def _simulate_other_central_banks(self) -> List[CentralBankAnnouncement]:
        """Simulate other central bank announcements"""
        return [
            CentralBankAnnouncement(
                bank_name='Bank of England',
                announcement_date=datetime.now() + timedelta(days=21),
                decision='hold',
                new_rate=5.25,
                previous_rate=5.25,
                statement='MPC maintains Bank Rate',
                importance='medium'
            ),
            CentralBankAnnouncement(
                bank_name='Bank of Japan',
                announcement_date=datetime.now() + timedelta(days=28),
                decision='hold',
                new_rate=-0.10,
                previous_rate=-0.10,
                statement='BOJ maintains negative interest rate policy',
                importance='medium'
            )
        ]
    
    async def fetch_economic_calendar(self, start_date: datetime = None, end_date: datetime = None) -> List[EconomicCalendarEvent]:
        """Fetch economic calendar events"""
        if not start_date:
            start_date = datetime.now()
        if not end_date:
            end_date = start_date + timedelta(days=30)
        
        events = []
        
        try:
            # Simulate economic calendar events
            sample_events = [
                EconomicCalendarEvent(
                    event_name='Federal Funds Rate Decision',
                    date=datetime.now() + timedelta(days=7),
                    country='US',
                    currency='USD',
                    importance='critical',
                    forecast_value=5.25,
                    previous_value=5.00,
                    unit='Percent',
                    release_time='14:00 ET',
                    description='Federal Open Market Committee interest rate decision',
                    volatility_expected=0.9
                ),
                EconomicCalendarEvent(
                    event_name='Nonfarm Payrolls',
                    date=datetime.now() + timedelta(days=10),
                    country='US',
                    currency='USD',
                    importance='critical',
                    forecast_value=200000,
                    previous_value=187000,
                    unit='Jobs',
                    release_time='08:30 ET',
                    description='Monthly employment change in nonfarm payrolls',
                    volatility_expected=0.8
                ),
                EconomicCalendarEvent(
                    event_name='Consumer Price Index',
                    date=datetime.now() + timedelta(days=15),
                    country='US',
                    currency='USD',
                    importance='critical',
                    forecast_value=305.5,
                    previous_value=305.1,
                    unit='Index',
                    release_time='08:30 ET',
                    description='Monthly consumer price inflation measure',
                    volatility_expected=0.7
                ),
                EconomicCalendarEvent(
                    event_name='GDP Growth Rate',
                    date=datetime.now() + timedelta(days=20),
                    country='US',
                    currency='USD',
                    importance='high',
                    forecast_value=2.1,
                    previous_value=2.0,
                    unit='Percent',
                    release_time='08:30 ET',
                    description='Quarterly gross domestic product growth rate',
                    volatility_expected=0.6
                ),
                EconomicCalendarEvent(
                    event_name='ECB Interest Rate Decision',
                    date=datetime.now() + timedelta(days=25),
                    country='EU',
                    currency='EUR',
                    importance='critical',
                    forecast_value=4.00,
                    previous_value=4.00,
                    unit='Percent',
                    release_time='12:45 CET',
                    description='European Central Bank monetary policy decision',
                    volatility_expected=0.8
                )
            ]
            
            events.extend(sample_events)
            
            # Filter by date range
            filtered_events = [
                event for event in events
                if start_date <= event.date <= end_date
            ]
            
            self.logger.info(f"Fetched {len(filtered_events)} economic calendar events")
            return filtered_events
        
        except Exception as e:
            self.logger.error(f"Error fetching economic calendar: {e}")
            return []
    
    async def fetch_market_sentiment(self) -> MarketSentiment:
        """Fetch market sentiment indicators"""
        try:
            # Fetch sentiment indicators
            vix_data = await self._fetch_market_indicator('VIX')
            dxy_data = await self._fetch_market_indicator('DXY')
            treasury_data = await self._fetch_market_indicators(['TNX', '^TYX', '^FVX'])
            gold_data = await self._fetch_market_indicator('GLD')
            oil_data = await self._fetch_market_indicator('USO')
            
            return MarketSentiment(
                timestamp=datetime.now(),
                vix_level=vix_data.get('value'),
                dollar_index=dxy_data.get('value'),
                treasury_yield=treasury_data,
                gold_price=gold_data.get('value'),
                oil_price=oil_data.get('value'),
                # Simulated values for demo
                fear_greed_index=45.2,
                put_call_ratio=0.85,
                advance_decline_ratio=1.15,
                credit_spread=0.75,
                emerging_markets_index=45.8
            )
        
        except Exception as e:
            self.logger.error(f"Error fetching market sentiment: {e}")
            return MarketSentiment(timestamp=datetime.now())
    
    async def _fetch_market_indicator(self, symbol: str) -> Dict[str, float]:
        """Fetch a single market indicator"""
        try:
            url = f"{self.endpoints['yahoo_finance']}/{symbol}"
            params = {'interval': '1d', 'range': '5d'}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'chart' in data and data['chart']['result']:
                            result = data['chart']['result'][0]
                            timestamps = result['timestamp']
                            prices = result['indicators']['quote'][0]
                            
                            if timestamps and prices['close']:
                                latest_price = prices['close'][-1]
                                return {
                                    'symbol': symbol,
                                    'value': latest_price,
                                    'timestamp': datetime.fromtimestamp(timestamps[-1])
                                }
        except Exception as e:
            self.logger.error(f"Error fetching {symbol}: {e}")
        
        return {'symbol': symbol, 'value': None}
    
    async def _fetch_market_indicators(self, symbols: List[str]) -> Dict[str, float]:
        """Fetch multiple market indicators"""
        results = {}
        
        for symbol in symbols:
            data = await self._fetch_market_indicator(symbol)
            if data['value'] is not None:
                results[symbol] = data['value']
        
        return results
    
    def _get_indicator_unit(self, indicator_name: str) -> str:
        """Get the unit for an economic indicator"""
        unit_mapping = {
            'GDP': 'Billions of Dollars',
            'Inflation_CPI': 'Index 1982-84=100',
            'Unemployment': 'Percent',
            'Interest_Rate': 'Percent',
            'Industrial_Production': 'Index 2017=100',
            'Retail_Sales': 'Millions of Dollars',
            'Consumer_Confidence': 'Index 1985=100',
            'Housing_Starts': 'Thousands of Units',
            'Trade_Balance': 'Millions of Dollars'
        }
        return unit_mapping.get(indicator_name, 'Various')
    
    async def _fetch_data(self) -> Dict[str, Any]:
        """Fetch all economic data"""
        indicators_task = self.fetch_economic_indicators()
        announcements_task = self.fetch_central_bank_announcements()
        calendar_task = self.fetch_economic_calendar()
        sentiment_task = self.fetch_market_sentiment()
        
        indicators, announcements, calendar_events, sentiment = await asyncio.gather(
            indicators_task, announcements_task, calendar_task, sentiment_task
        )
        
        return {
            'indicators': indicators,
            'announcements': announcements,
            'calendar_events': calendar_events,
            'sentiment': sentiment
        }
    
    async def _process_data(self, data: Dict[str, Any], source: str = "live"):
        """Process economic data"""
        try:
            indicators = data.get('indicators', [])
            announcements = data.get('announcements', [])
            calendar_events = data.get('calendar_events', [])
            sentiment = data.get('sentiment')
            
            # Log economic summary
            if indicators:
                critical_indicators = [ind for ind in indicators if ind.importance == 'critical']
                self.logger.info(f"Economic Summary: {len(indicators)} indicators, "
                               f"{len(critical_indicators)} critical")
            
            # Log announcements
            if announcements:
                upcoming = [ann for ann in announcements if ann.announcement_date > datetime.now()]
                self.logger.info(f"Central Banks: {len(announcements)} announcements, "
                               f"{len(upcoming)} upcoming")
            
            # Log calendar events
            if calendar_events:
                high_importance = [event for event in calendar_events if event.importance in ['high', 'critical']]
                self.logger.info(f"Economic Calendar: {len(calendar_events)} events, "
                               f"{len(high_importance)} high importance")
            
            # Log market sentiment
            if sentiment:
                vix_level = getattr(sentiment, 'vix_level', None)
                fear_greed = getattr(sentiment, 'fear_greed_index', None)
                
                sentiment_desc = 'Unknown'
                if fear_greed:
                    if fear_greed < 25:
                        sentiment_desc = 'Extreme Fear'
                    elif fear_greed < 45:
                        sentiment_desc = 'Fear'
                    elif fear_greed < 55:
                        sentiment_desc = 'Neutral'
                    elif fear_greed < 75:
                        sentiment_desc = 'Greed'
                    else:
                        sentiment_desc = 'Extreme Greed'
                
                self.logger.info(f"Market Sentiment: {sentiment_desc} "
                               f"(Fear & Greed: {fear_greed:.1f}, VIX: {vix_level:.2f})")
            
            # Store data if configured
            if self.config.enable_storage:
                await self._store_economic_data(data)
        
        except Exception as e:
            self.logger.error(f"Error processing economic data: {e}")
            raise
    
    async def _store_economic_data(self, data: Dict[str, Any]):
        """Store economic data to files"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Store indicators
            indicators = data.get('indicators', [])
            if indicators:
                indicators_file = Path(self.config.storage_path) / f"economic_indicators_{timestamp}.json"
                indicators_json = [{
                    'indicator_name': ind.indicator_name,
                    'value': ind.value,
                    'unit': ind.unit,
                    'timestamp': ind.timestamp.isoformat(),
                    'previous_value': ind.previous_value,
                    'forecast': ind.forecast,
                    'country': ind.country,
                    'frequency': ind.frequency,
                    'importance': ind.importance,
                    'source': ind.source,
                    'release_date': ind.release_date.isoformat() if ind.release_date else None
                } for ind in indicators]
                
                await aiofiles.makedirs(indicators_file.parent, exist_ok=True)
                async with aiofiles.open(indicators_file, 'w') as f:
                    await f.write(json.dumps(indicators_json, indent=2, default=str))
            
            # Store announcements
            announcements = data.get('announcements', [])
            if announcements:
                announcements_file = Path(self.config.storage_path) / f"central_bank_announcements_{timestamp}.json"
                announcements_json = [{
                    'bank_name': ann.bank_name,
                    'announcement_date': ann.announcement_date.isoformat(),
                    'decision': ann.decision,
                    'rate_change': ann.rate_change,
                    'new_rate': ann.new_rate,
                    'previous_rate': ann.previous_rate,
                    'statement': ann.statement,
                    'forward_guidance': ann.forward_guidance,
                    'importance': ann.importance,
                    'market_impact': ann.market_impact
                } for ann in announcements]
                
                async with aiofiles.open(announcements_file, 'w') as f:
                    await f.write(json.dumps(announcements_json, indent=2, default=str))
            
            # Store calendar events
            calendar_events = data.get('calendar_events', [])
            if calendar_events:
                calendar_file = Path(self.config.storage_path) / f"economic_calendar_{timestamp}.json"
                calendar_json = [{
                    'event_name': event.event_name,
                    'date': event.date.isoformat(),
                    'country': event.country,
                    'currency': event.currency,
                    'importance': event.importance,
                    'actual_value': event.actual_value,
                    'forecast_value': event.forecast_value,
                    'previous_value': event.previous_value,
                    'unit': event.unit,
                    'frequency': event.frequency,
                    'release_time': event.release_time,
                    'description': event.description,
                    'volatility_expected': event.volatility_expected
                } for event in calendar_events]
                
                async with aiofiles.open(calendar_file, 'w') as f:
                    await f.write(json.dumps(calendar_json, indent=2, default=str))
            
            # Store sentiment
            sentiment = data.get('sentiment')
            if sentiment:
                sentiment_file = Path(self.config.storage_path) / f"market_sentiment_{timestamp}.json"
                sentiment_json = {
                    'timestamp': sentiment.timestamp.isoformat(),
                    'fear_greed_index': sentiment.fear_greed_index,
                    'vix_level': sentiment.vix_level,
                    'put_call_ratio': sentiment.put_call_ratio,
                    'advance_decline_ratio': sentiment.advance_decline_ratio,
                    'credit_spread': sentiment.credit_spread,
                    'treasury_yield': sentiment.treasury_yield,
                    'dollar_index': sentiment.dollar_index,
                    'gold_price': sentiment.gold_price,
                    'oil_price': sentiment.oil_price,
                    'emerging_markets_index': sentiment.emerging_markets_index
                }
                
                async with aiofiles.open(sentiment_file, 'w') as f:
                    await f.write(json.dumps(sentiment_json, indent=2, default=str))
        
        except Exception as e:
            self.logger.error(f"Failed to store economic data: {e}")
    
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported economic indicators"""
        return list(self.key_indicators.keys())
    
    def get_data_schema(self) -> Dict[str, Any]:
        """Get expected data schema"""
        return {
            'indicators': {
                'indicator_name': 'str',
                'value': 'float',
                'unit': 'str',
                'timestamp': 'datetime',
                'previous_value': 'float (optional)',
                'forecast': 'float (optional)',
                'country': 'str',
                'frequency': 'str',
                'importance': 'str',
                'source': 'str',
                'release_date': 'datetime (optional)'
            },
            'announcements': {
                'bank_name': 'str',
                'announcement_date': 'datetime',
                'decision': 'str',
                'rate_change': 'float (optional)',
                'new_rate': 'float (optional)',
                'previous_rate': 'float (optional)',
                'statement': 'str (optional)',
                'forward_guidance': 'str (optional)',
                'importance': 'str',
                'market_impact': 'str (optional)'
            },
            'calendar_events': {
                'event_name': 'str',
                'date': 'datetime',
                'country': 'str',
                'currency': 'str',
                'importance': 'str',
                'actual_value': 'float (optional)',
                'forecast_value': 'float (optional)',
                'previous_value': 'float (optional)',
                'unit': 'str',
                'frequency': 'str',
                'release_time': 'str (optional)',
                'description': 'str (optional)',
                'volatility_expected': 'float'
            },
            'sentiment': {
                'timestamp': 'datetime',
                'fear_greed_index': 'float (optional)',
                'vix_level': 'float (optional)',
                'put_call_ratio': 'float (optional)',
                'treasury_yield': 'dict (optional)',
                'dollar_index': 'float (optional)',
                'gold_price': 'float (optional)',
                'oil_price': 'float (optional)'
            }
        }