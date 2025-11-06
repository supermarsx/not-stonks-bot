"""
Market Hours Detection and Management System

Handles detection of market hours for different exchanges (US, EU, Asia),
holidays, weekends, pre-market, after-hours, and session transitions.
"""

import asyncio
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Set, Callable
from enum import Enum
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from loguru import logger


class MarketType(Enum):
    """Market types"""
    STOCKS = "stocks"
    FOREX = "forex" 
    CRYPTO = "crypto"
    COMMODITIES = "commodities"
    FUTURES = "futures"


class MarketSession(Enum):
    """Market session types"""
    PRE_MARKET = "pre_market"
    REGULAR_HOURS = "regular_hours"
    EXTENDED_HOURS = "extended_hours"
    AFTER_HOURS = "after_hours"
    CLOSED = "closed"
    WEEKEND = "weekend"
    HOLIDAY = "holiday"


@dataclass
class ExchangeInfo:
    """Exchange information"""
    name: str
    timezone: str
    market_type: MarketType
    regular_hours: Dict[str, tuple[time, time]]  # day_of_week -> (open, close)
    pre_market_hours: Optional[Dict[str, tuple[time, time]]] = None
    after_hours: Optional[Dict[str, tuple[time, time]]] = None
    holidays: Optional[Set[str]] = None


@dataclass
class SessionInfo:
    """Current market session information"""
    current_session: MarketSession
    exchange_name: str
    next_session_change: Optional[datetime]
    session_end_time: Optional[datetime]
    is_market_open: bool
    is_weekend: bool
    is_holiday: bool
    days_until_market: Optional[int]
    minutes_to_open: Optional[int]
    minutes_to_close: Optional[int]


class MarketHoursManager:
    """
    Manages market hours detection and session management
    
    Features:
    - Multi-exchange support (US, EU, Asia)
    - Holiday and weekend detection
    - Pre-market and after-hours logic
    - Session transition management
    - Real-time session monitoring
    """
    
    def __init__(self):
        self.exchanges: Dict[str, ExchangeInfo] = {}
        self.session_callbacks: List[Callable] = []
        self.session_history: List[Dict] = []
        self.current_sessions: Dict[str, SessionInfo] = {}
        
        # Initialize exchanges
        self._setup_exchanges()
        
        logger.info("Market Hours Manager initialized")
    
    def _setup_exchanges(self):
        """Setup supported exchanges with their trading hours"""
        
        # US Markets (NYSE, NASDAQ)
        us_tz = ZoneInfo("America/New_York")
        self.exchanges["NYSE"] = ExchangeInfo(
            name="New York Stock Exchange",
            timezone="America/New_York",
            market_type=MarketType.STOCKS,
            regular_hours={
                0: (time(9, 30), time(16, 0)),  # Monday
                1: (time(9, 30), time(16, 0)),  # Tuesday
                2: (time(9, 30), time(16, 0)),  # Wednesday
                3: (time(9, 30), time(16, 0)),  # Thursday
                4: (time(9, 30), time(16, 0)),  # Friday
            },
            pre_market_hours={
                0: (time(4, 0), time(9, 30)),   # Monday
                1: (time(4, 0), time(9, 30)),   # Tuesday
                2: (time(4, 0), time(9, 30)),   # Wednesday
                3: (time(4, 0), time(9, 30)),   # Thursday
                4: (time(4, 0), time(9, 30)),   # Friday
            },
            after_hours={
                0: (time(16, 0), time(20, 0)),  # Monday
                1: (time(16, 0), time(20, 0)),  # Tuesday
                2: (time(16, 0), time(20, 0)),  # Wednesday
                3: (time(16, 0), time(20, 0)),  # Thursday
                4: (time(16, 0), time(20, 0)),  # Friday
            },
            holidays=set([
                "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29",
                "2024-05-27", "2024-06-19", "2024-07-04", "2024-09-02",
                "2024-10-14", "2024-11-11", "2024-11-28", "2024-12-25"
            ])
        )
        
        # European Markets (LSE, DAX, CAC)
        self.exchanges["LSE"] = ExchangeInfo(
            name="London Stock Exchange",
            timezone="Europe/London",
            market_type=MarketType.STOCKS,
            regular_hours={
                0: (time(8, 0), time(16, 30)),  # Monday
                1: (time(8, 0), time(16, 30)),  # Tuesday
                2: (time(8, 0), time(16, 30)),  # Wednesday
                3: (time(8, 0), time(16, 30)),  # Thursday
                4: (time(8, 0), time(16, 30)),  # Friday
            },
            pre_market_hours={
                0: (time(5, 0), time(8, 0)),    # Monday
                1: (time(5, 0), time(8, 0)),    # Tuesday
                2: (time(5, 0), time(8, 0)),    # Wednesday
                3: (time(5, 0), time(8, 0)),    # Thursday
                4: (time(5, 0), time(8, 0)),    # Friday
            },
            after_hours={
                0: (time(16, 30), time(20, 0)), # Monday
                1: (time(16, 30), time(20, 0)), # Tuesday
                2: (time(16, 30), time(20, 0)), # Wednesday
                3: (time(16, 30), time(20, 0)), # Thursday
                4: (time(16, 30), time(20, 0)), # Friday
            }
        )
        
        # Asian Markets (Tokyo, Hong Kong, Shanghai)
        self.exchanges["TSE"] = ExchangeInfo(
            name="Tokyo Stock Exchange",
            timezone="Asia/Tokyo",
            market_type=MarketType.STOCKS,
            regular_hours={
                0: (time(9, 0), time(11, 30)),  # Monday
                1: (time(9, 0), time(11, 30)),  # Tuesday
                2: (time(9, 0), time(11, 30)),  # Wednesday
                3: (time(9, 0), time(11, 30)),  # Thursday
                4: (time(9, 0), time(11, 30)),  # Friday
            },
            pre_market_hours={
                0: (time(8, 0), time(9, 0)),    # Monday
                1: (time(8, 0), time(9, 0)),    # Tuesday
                2: (time(8, 0), time(9, 0)),    # Wednesday
                3: (time(8, 0), time(9, 0)),    # Thursday
                4: (time(8, 0), time(9, 0)),    # Friday
            },
            after_hours={
                0: (time(11, 30), time(15, 0)), # Monday
                1: (time(11, 30), time(15, 0)), # Tuesday
                2: (time(11, 30), time(15, 0)), # Wednesday
                3: (time(11, 30), time(15, 0)), # Thursday
                4: (time(11, 30), time(15, 0)), # Friday
            }
        )
        
        # Crypto Markets (24/7)
        self.exchanges["CRYPTO"] = ExchangeInfo(
            name="Global Crypto Markets",
            timezone="UTC",
            market_type=MarketType.CRYPTO,
            regular_hours={
                0: (time(0, 0), time(23, 59)),  # Monday
                1: (time(0, 0), time(23, 59)),  # Tuesday
                2: (time(0, 0), time(23, 59)),  # Wednesday
                3: (time(0, 0), time(23, 59)),  # Thursday
                4: (time(0, 0), time(23, 59)),  # Friday
                5: (time(0, 0), time(23, 59)),  # Saturday
                6: (time(0, 0), time(23, 59)),  # Sunday
            }
        )
        
        # Forex Markets (24/5)
        self.exchanges["FOREX"] = ExchangeInfo(
            name="Global Forex Markets",
            timezone="UTC", 
            market_type=MarketType.FOREX,
            regular_hours={
                0: (time(0, 0), time(23, 59)),  # Monday
                1: (time(0, 0), time(23, 59)),  # Tuesday
                2: (time(0, 0), time(23, 59)),  # Wednesday
                3: (time(0, 0), time(23, 59)),  # Thursday
                4: (time(0, 0), time(23, 59)),  # Friday
            }
        )
    
    def get_current_session(self, exchange_name: str, reference_time: Optional[datetime] = None) -> SessionInfo:
        """
        Get current market session for an exchange
        
        Args:
            exchange_name: Name of the exchange
            reference_time: Optional reference time (defaults to now)
            
        Returns:
            Current session information
        """
        if reference_time is None:
            reference_time = datetime.now(ZoneInfo("UTC"))
        
        exchange = self.exchanges.get(exchange_name)
        if not exchange:
            raise ValueError(f"Unknown exchange: {exchange_name}")
        
        # Get timezone-aware datetime
        local_tz = ZoneInfo(exchange.timezone)
        local_time = reference_time.astimezone(local_tz)
        
        # Check if it's a holiday
        date_str = local_time.strftime("%Y-%m-%d")
        is_holiday = exchange.holidays and date_str in exchange.holidays
        
        # Check if it's a weekend (crypto and forex excluded)
        is_weekend = False
        if exchange.market_type in [MarketType.STOCKS, MarketType.COMMODITIES, MarketType.FUTURES]:
            is_weekend = local_time.weekday() >= 5  # Saturday = 5, Sunday = 6
        
        # Determine session
        session = MarketSession.CLOSED
        is_market_open = False
        
        if is_weekend:
            session = MarketSession.WEEKEND
        elif is_holiday:
            session = MarketSession.HOLIDAY
        elif exchange.market_type in [MarketType.CRYPTO]:
            # Crypto markets are 24/7
            session = MarketSession.REGULAR_HOURS
            is_market_open = True
        elif exchange.market_type in [MarketType.FOREX]:
            # Forex is 24/5, closed on weekends
            if not is_weekend:
                session = MarketSession.REGULAR_HOURS
                is_market_open = True
            else:
                session = MarketSession.WEEKEND
        else:
            # Traditional markets (stocks, commodities, futures)
            day_of_week = local_time.weekday()
            
            if day_of_week < 5:  # Monday to Friday
                # Check pre-market
                if exchange.pre_market_hours:
                    pre_open, pre_close = exchange.pre_market_hours[day_of_week]
                    if pre_open <= local_time.time() < pre_close:
                        session = MarketSession.PRE_MARKET
                        is_market_open = True
                
                # Check regular hours
                if session == MarketSession.CLOSED:
                    reg_open, reg_close = exchange.regular_hours[day_of_week]
                    if reg_open <= local_time.time() < reg_close:
                        session = MarketSession.REGULAR_HOURS
                        is_market_open = True
                
                # Check after-hours
                if session == MarketSession.CLOSED and exchange.after_hours:
                    ah_open, ah_close = exchange.after_hours[day_of_week]
                    if ah_open <= local_time.time() < ah_close:
                        session = MarketSession.AFTER_HOURS
                        is_market_open = True
        
        # Calculate timing information
        next_session_change = None
        session_end_time = None
        minutes_to_open = None
        minutes_to_close = None
        days_until_market = None
        
        if not is_market_open and not is_weekend and not is_holiday:
            # Calculate time until next market open
            next_open_time = self._get_next_market_open(exchange, local_time)
            if next_open_time:
                delta = next_open_time - local_time
                minutes_to_open = int(delta.total_seconds() / 60)
                
                # Calculate days until market (for weekends)
                if local_time.weekday() == 4 and local_time.time() > time(16, 0):  # Friday after 4pm
                    days_until_market = 2  # Next Monday
                elif local_time.weekday() == 5:  # Saturday
                    days_until_market = 1  # Next Monday
                elif local_time.weekday() == 6:  # Sunday
                    days_until_market = 0  # Today (Monday)
        
        if is_market_open:
            # Calculate time until market close
            session_end_time = self._get_session_end_time(exchange, local_time)
            if session_end_time:
                delta = session_end_time - local_time
                minutes_to_close = int(delta.total_seconds() / 60)
        
        return SessionInfo(
            current_session=session,
            exchange_name=exchange_name,
            next_session_change=next_session_change,
            session_end_time=session_end_time,
            is_market_open=is_market_open,
            is_weekend=is_weekend,
            is_holiday=is_holiday,
            days_until_market=days_until_market,
            minutes_to_open=minutes_to_open,
            minutes_to_close=minutes_to_close
        )
    
    def _get_next_market_open(self, exchange: ExchangeInfo, current_time: datetime) -> Optional[datetime]:
        """Get next market opening time"""
        if exchange.market_type in [MarketType.CRYPTO]:
            return current_time  # Always open
        
        # Check if current day is a trading day
        current_day = current_time.weekday()
        if current_day >= 5:  # Weekend
            # Next Monday
            days_ahead = 7 - current_day
            next_open = current_time.replace(
                hour=0, minute=0, second=0, microsecond=0
            ) + timedelta(days=days_ahead)
        else:
            # Check if market hasn't opened yet today
            if exchange.regular_hours:
                reg_open = exchange.regular_hours[current_day][0]
                current_open_time = current_time.replace(
                    hour=reg_open.hour, minute=reg_open.minute, second=0, microsecond=0
                )
                
                if current_time < current_open_time:
                    return current_open_time
        
        # Next trading day
        next_day = current_time + timedelta(days=1)
        while next_day.weekday() >= 5:  # Skip weekends
            next_day += timedelta(days=1)
        
        # Set to opening time of next trading day
        if exchange.regular_hours:
            reg_open = exchange.regular_hours[next_day.weekday()][0]
            return next_day.replace(
                hour=reg_open.hour, minute=reg_open.minute, second=0, microsecond=0
            )
        
        return None
    
    def _get_session_end_time(self, exchange: ExchangeInfo, current_time: datetime) -> Optional[datetime]:
        """Get current session end time"""
        current_day = current_time.weekday()
        
        if exchange.market_type in [MarketType.CRYPTO]:
            return current_time + timedelta(hours=1)  # Extended session
        
        # Check pre-market session end
        if exchange.pre_market_hours:
            pre_open, pre_close = exchange.pre_market_hours[current_day]
            if (time(pre_open.hour, pre_open.minute) <= current_time.time() < 
                time(pre_close.hour, pre_close.minute)):
                return current_time.replace(
                    hour=pre_close.hour, minute=pre_close.minute, second=0, microsecond=0
                )
        
        # Check regular session end
        if exchange.regular_hours:
            reg_open, reg_close = exchange.regular_hours[current_day]
            if (time(reg_open.hour, reg_open.minute) <= current_time.time() < 
                time(reg_close.hour, reg_close.minute)):
                return current_time.replace(
                    hour=reg_close.hour, minute=reg_close.minute, second=0, microsecond=0
                )
        
        # Check after-hours session end
        if exchange.after_hours:
            ah_open, ah_close = exchange.after_hours[current_day]
            if (time(ah_open.hour, ah_open.minute) <= current_time.time() < 
                time(ah_close.hour, ah_close.minute)):
                return current_time.replace(
                    hour=ah_close.hour, minute=ah_close.minute, second=0, microsecond=0
                )
        
        return None
    
    def get_all_current_sessions(self, reference_time: Optional[datetime] = None) -> Dict[str, SessionInfo]:
        """Get current sessions for all exchanges"""
        sessions = {}
        
        for exchange_name in self.exchanges.keys():
            try:
                sessions[exchange_name] = self.get_current_session(exchange_name, reference_time)
            except Exception as e:
                logger.error(f"Error getting session for {exchange_name}: {e}")
        
        return sessions
    
    def is_any_market_open(self, reference_time: Optional[datetime] = None) -> bool:
        """Check if any market is currently open"""
        sessions = self.get_all_current_sessions(reference_time)
        return any(session.is_market_open for session in sessions.values())
    
    def get_open_exchanges(self, reference_time: Optional[datetime] = None) -> List[str]:
        """Get list of currently open exchanges"""
        sessions = self.get_all_current_sessions(reference_time)
        return [name for name, session in sessions.items() if session.is_market_open]
    
    def register_session_callback(self, callback: Callable[[str, SessionInfo], None]):
        """Register callback for session changes"""
        self.session_callbacks.append(callback)
    
    async def start_session_monitoring(self, check_interval: int = 60):
        """
        Start continuous session monitoring
        
        Args:
            check_interval: Check interval in seconds (default: 60)
        """
        logger.info(f"Starting market session monitoring (interval: {check_interval}s)")
        
        while True:
            try:
                current_sessions = self.get_all_current_sessions()
                
                # Check for session changes
                for exchange_name, current_session in current_sessions.items():
                    previous_session = self.current_sessions.get(exchange_name)
                    
                    if (previous_session and 
                        previous_session.current_session != current_session.current_session):
                        # Session changed, notify callbacks
                        logger.info(f"Session changed for {exchange_name}: "
                                  f"{previous_session.current_session.value} -> {current_session.current_session.value}")
                        
                        for callback in self.session_callbacks:
                            try:
                                callback(exchange_name, current_session)
                            except Exception as e:
                                logger.error(f"Error in session callback: {e}")
                
                # Update current sessions
                self.current_sessions = current_sessions
                
                # Log session status
                open_exchanges = [name for name, session in current_sessions.items() 
                                if session.is_market_open]
                
                if open_exchanges:
                    logger.debug(f"Open exchanges: {', '.join(open_exchanges)}")
                else:
                    logger.debug("No markets currently open")
                
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error in session monitoring: {e}")
                await asyncio.sleep(check_interval)
    
    def get_market_summary(self, reference_time: Optional[datetime] = None) -> Dict:
        """Get comprehensive market summary"""
        sessions = self.get_all_current_sessions(reference_time)
        
        summary = {
            "reference_time": reference_time or datetime.now(ZoneInfo("UTC")),
            "total_exchanges": len(sessions),
            "open_exchanges": len([s for s in sessions.values() if s.is_market_open]),
            "closed_exchanges": len([s for s in sessions.values() if not s.is_market_open]),
            "is_any_market_open": any(s.is_market_open for s in sessions.values()),
            "exchanges": {}
        }
        
        for exchange_name, session in sessions.items():
            summary["exchanges"][exchange_name] = {
                "session": session.current_session.value,
                "is_open": session.is_market_open,
                "is_weekend": session.is_weekend,
                "is_holiday": session.is_holiday,
                "minutes_to_close": session.minutes_to_close,
                "minutes_to_open": session.minutes_to_open,
                "days_until_market": session.days_until_market
            }
        
        return summary
    
    def add_custom_exchange(self, exchange: ExchangeInfo):
        """Add a custom exchange configuration"""
        self.exchanges[exchange.name] = exchange
        logger.info(f"Added custom exchange: {exchange.name}")


# Global instance
market_hours_manager = MarketHoursManager()