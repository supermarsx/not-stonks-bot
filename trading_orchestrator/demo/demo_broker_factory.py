"""
Demo Broker Factory and Demo Data Generator
"""

import asyncio
import random
import numpy as np
from typing import Dict, List, Optional, Any, Type, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json

from loguru import logger

from .demo_mode_manager import DemoModeManager
from .virtual_broker import VirtualBroker

# Import BrokerConfig (will be available when used in the full system)
try:
    from ..brokers.base import BrokerConfig
except ImportError:
    from typing import Any
    BrokerConfig = Any


class BrokerType(Enum):
    """Supported broker types for demo mode"""
    ALPACA = "alpaca"
    BINANCE = "binance"
    IBKR = "ibkr"
    TRADING212 = "trading212"
    DEGIRO = "degiro"
    TRADE_REPUBLIC = "trade_republic"
    XTB = "xtb"


@dataclass
class BrokerCapability:
    """Broker capability definition"""
    name: str
    supported_instruments: List[str]
    supported_order_types: List[str]
    max_leverage: float
    commission_structure: Dict[str, float]
    market_hours: Dict[str, str]
    special_features: List[str]


@dataclass
class DemoDataConfig:
    """Configuration for demo data generation"""
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    frequency: str  # 1m, 5m, 1h, 1d
    include_fundamentals: bool = False
    include_news: bool = False
    include_earnings: bool = False
    volatility_level: float = 1.0  # Market volatility multiplier
    trend_strength: float = 1.0  # Trend strength multiplier
    volume_pattern: str = "normal"  # normal, high, low, erratic


class DemoDataGenerator:
    """
    Generate realistic synthetic market data for demo trading
    
    Creates realistic OHLCV data with:
    - Proper price dynamics
    - Realistic volume patterns
    - Market microstructure
    - Corporate actions simulation
    """
    
    def __init__(self, config: DemoDataConfig):
        self.config = config
        self.symbol_metadata = self._initialize_symbol_metadata()
        self.market_regimes = self._initialize_market_regimes()
    
    async def generate_historical_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate historical market data for all symbols"""
        try:
            all_data = {}
            
            for symbol in self.config.symbols:
                logger.info(f"Generating data for {symbol}")
                
                # Get symbol metadata
                metadata = self.symbol_metadata.get(symbol, self._get_default_metadata(symbol))
                
                # Generate data with symbol-specific characteristics
                data = await self._generate_symbol_data(symbol, metadata)
                all_data[symbol] = data
            
            return all_data
            
        except Exception as e:
            logger.error(f"Error generating historical data: {e}")
            return {}
    
    async def generate_real_time_stream(self, symbols: List[str], duration_minutes: int = 60) -> Dict[str, List[Dict[str, Any]]]:
        """Generate real-time market data stream"""
        try:
            stream_data = {}
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=duration_minutes)
            
            for symbol in symbols:
                metadata = self.symbol_metadata.get(symbol, self._get_default_metadata(symbol))
                
                # Generate minute-by-minute data
                data_points = []
                current_time = start_time
                
                while current_time < end_time:
                    data_point = await self._generate_data_point(symbol, metadata, current_time)
                    data_points.append(data_point)
                    current_time += timedelta(minutes=1)
                
                stream_data[symbol] = data_points
            
            return stream_data
            
        except Exception as e:
            logger.error(f"Error generating real-time stream: {e}")
            return {}
    
    async def generate_market_snapshot(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Generate current market snapshot"""
        try:
            snapshot = {}
            
            for symbol in symbols:
                metadata = self.symbol_metadata.get(symbol, self._get_default_metadata(symbol))
                current_price = await self._generate_current_price(symbol, metadata)
                
                # Generate realistic bid/ask spread
                spread = current_price * metadata.get("typical_spread", 0.001)
                bid = current_price - spread / 2
                ask = current_price + spread / 2
                
                snapshot[symbol] = {
                    "symbol": symbol,
                    "bid": round(bid, 2),
                    "ask": round(ask, 2),
                    "last": round(current_price, 2),
                    "volume": int(np.random.lognormal(12, 1)),  # Realistic volume
                    "change": round(np.random.normal(0, 0.02) * current_price, 2),  # Daily change
                    "change_pct": round(np.random.normal(0, 0.02) * 100, 2),
                    "high": round(current_price * (1 + abs(np.random.normal(0, 0.01))), 2),
                    "low": round(current_price * (1 - abs(np.random.normal(0, 0.01))), 2),
                    "timestamp": datetime.now().isoformat(),
                    "exchange": metadata.get("exchange", "SIM"),
                    "currency": metadata.get("currency", "USD")
                }
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Error generating market snapshot: {e}")
            return {}
    
    async def generate_corporate_actions(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Generate simulated corporate actions"""
        try:
            actions = []
            
            for symbol in symbols:
                metadata = self.symbol_metadata.get(symbol, self._get_default_metadata(symbol))
                
                # Generate occasional corporate actions
                if random.random() < 0.1:  # 10% chance
                    action_type = random.choice(["dividend", "split", "spin_off", "merger"])
                    
                    if action_type == "dividend":
                        dividend_amount = random.uniform(0.1, 2.0)
                        ex_date = datetime.now() + timedelta(days=random.randint(1, 30))
                        actions.append({
                            "symbol": symbol,
                            "action_type": action_type,
                            "ex_date": ex_date.isoformat(),
                            "amount": dividend_amount,
                            "currency": metadata.get("currency", "USD")
                        })
                    
                    elif action_type == "split":
                        split_ratio = random.choice([2, 3, 4, 1.5])
                        announcement_date = datetime.now() + timedelta(days=random.randint(1, 60))
                        actions.append({
                            "symbol": symbol,
                            "action_type": action_type,
                            "announcement_date": announcement_date.isoformat(),
                            "split_ratio": split_ratio
                        })
            
            return actions
            
        except Exception as e:
            logger.error(f"Error generating corporate actions: {e}")
            return []
    
    async def generate_news_sentiment(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Generate simulated news and sentiment data"""
        try:
            news_items = []
            
            # Generate news for a subset of symbols
            for symbol in random.sample(symbols, min(len(symbols), 5)):
                if random.random() < 0.3:  # 30% chance of news
                    sentiment_score = np.random.normal(0, 0.3)  # -0.3 to 0.3
                    
                    news_item = {
                        "symbol": symbol,
                        "headline": f"Demo news headline for {symbol}",
                        "sentiment_score": round(sentiment_score, 3),
                        "source": "Demo News Agency",
                        "timestamp": (datetime.now() - timedelta(hours=random.randint(1, 24))).isoformat(),
                        "relevance_score": round(random.uniform(0.5, 1.0), 2)
                    }
                    news_items.append(news_item)
            
            return news_items
            
        except Exception as e:
            logger.error(f"Error generating news sentiment: {e}")
            return []
    
    # Private methods
    
    def _initialize_symbol_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Initialize metadata for common symbols"""
        return {
            "AAPL": {
                "exchange": "NASDAQ",
                "currency": "USD",
                "sector": "Technology",
                "industry": "Consumer Electronics",
                "market_cap": "Large",
                "volatility": 0.25,
                "typical_spread": 0.001,
                "avg_volume": 50000000,
                "dividend_yield": 0.006
            },
            "MSFT": {
                "exchange": "NASDAQ",
                "currency": "USD",
                "sector": "Technology",
                "industry": "Software",
                "market_cap": "Large",
                "volatility": 0.22,
                "typical_spread": 0.001,
                "avg_volume": 25000000,
                "dividend_yield": 0.008
            },
            "GOOGL": {
                "exchange": "NASDAQ",
                "currency": "USD",
                "sector": "Technology",
                "industry": "Internet Services",
                "market_cap": "Large",
                "volatility": 0.28,
                "typical_spread": 0.0015,
                "avg_volume": 20000000,
                "dividend_yield": 0.0
            },
            "AMZN": {
                "exchange": "NASDAQ",
                "currency": "USD",
                "sector": "Consumer Discretionary",
                "industry": "E-commerce",
                "market_cap": "Large",
                "volatility": 0.32,
                "typical_spread": 0.002,
                "avg_volume": 30000000,
                "dividend_yield": 0.0
            },
            "TSLA": {
                "exchange": "NASDAQ",
                "currency": "USD",
                "sector": "Consumer Discretionary",
                "industry": "Automotive",
                "market_cap": "Large",
                "volatility": 0.45,
                "typical_spread": 0.003,
                "avg_volume": 40000000,
                "dividend_yield": 0.0
            },
            "BTC-USD": {
                "exchange": "Binance",
                "currency": "USD",
                "sector": "Cryptocurrency",
                "industry": "Digital Currency",
                "market_cap": "Large",
                "volatility": 0.80,
                "typical_spread": 0.0005,
                "avg_volume": 1000000,
                "dividend_yield": 0.0
            },
            "ETH-USD": {
                "exchange": "Binance",
                "currency": "USD",
                "sector": "Cryptocurrency",
                "industry": "Digital Currency",
                "market_cap": "Large",
                "volatility": 0.70,
                "typical_spread": 0.0008,
                "avg_volume": 800000,
                "dividend_yield": 0.0
            },
            "EURUSD=X": {
                "exchange": "FX",
                "currency": "USD",
                "sector": "Forex",
                "industry": "Currency",
                "market_cap": "N/A",
                "volatility": 0.12,
                "typical_spread": 0.0001,
                "avg_volume": 500000000,
                "dividend_yield": 0.0
            },
            "SPY": {
                "exchange": "NYSE Arca",
                "currency": "USD",
                "sector": "ETF",
                "industry": "S&P 500 ETF",
                "market_cap": "Large",
                "volatility": 0.15,
                "typical_spread": 0.0005,
                "avg_volume": 80000000,
                "dividend_yield": 0.012
            },
            "QQQ": {
                "exchange": "NASDAQ",
                "currency": "USD",
                "sector": "ETF",
                "industry": "NASDAQ 100 ETF",
                "market_cap": "Large",
                "volatility": 0.20,
                "typical_spread": 0.0005,
                "avg_volume": 40000000,
                "dividend_yield": 0.005
            }
        }
    
    def _initialize_market_regimes(self) -> Dict[str, Dict[str, float]]:
        """Initialize market regime parameters"""
        return {
            "bull_market": {
                "drift": 0.0008,
                "volatility": 0.02,
                "correlation": 0.3,
                "duration": 180  # days
            },
            "bear_market": {
                "drift": -0.0005,
                "volatility": 0.025,
                "correlation": 0.6,
                "duration": 120
            },
            "sideways": {
                "drift": 0.0001,
                "volatility": 0.015,
                "correlation": 0.4,
                "duration": 240
            },
            "high_volatility": {
                "drift": 0.0002,
                "volatility": 0.04,
                "correlation": 0.5,
                "duration": 60
            },
            "low_volatility": {
                "drift": 0.0004,
                "volatility": 0.008,
                "correlation": 0.2,
                "duration": 365
            }
        }
    
    def _get_default_metadata(self, symbol: str) -> Dict[str, Any]:
        """Get default metadata for unknown symbols"""
        if "-USD" in symbol or symbol.endswith("=X"):
            # Cryptocurrency or Forex
            return {
                "exchange": "SIM",
                "currency": "USD",
                "sector": "Crypto/FX",
                "volatility": 0.40,
                "typical_spread": 0.001,
                "avg_volume": 1000000
            }
        else:
            # Default equity
            return {
                "exchange": "SIM",
                "currency": "USD",
                "sector": "Unknown",
                "volatility": 0.25,
                "typical_spread": 0.001,
                "avg_volume": 1000000
            }
    
    async def _generate_symbol_data(self, symbol: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate data for a specific symbol"""
        try:
            # Determine data frequency and count
            if self.config.frequency == "1d":
                time_delta = timedelta(days=1)
                count = (self.config.end_date - self.config.start_date).days
            elif self.config.frequency == "1h":
                time_delta = timedelta(hours=1)
                count = int((self.config.end_date - self.config.start_date).total_seconds() / 3600)
            else:  # Default to daily
                time_delta = timedelta(days=1)
                count = (self.config.end_date - self.config.start_date).days
            
            # Set initial price based on symbol type
            if "BTC" in symbol:
                initial_price = random.uniform(30000, 70000)
            elif "ETH" in symbol:
                initial_price = random.uniform(1500, 4000)
            elif "USD=" in symbol:
                initial_price = random.uniform(0.8, 1.2)
            else:
                initial_price = random.uniform(20, 500)
            
            # Generate OHLCV data
            data = []
            current_date = self.config.start_date
            current_price = initial_price
            
            for i in range(count):
                # Calculate daily return with volatility and drift
                volatility = metadata.get("volatility", 0.25) * self.config.volatility_level
                drift = np.random.normal(0, volatility * self.config.trend_strength / 252)  # Annualized drift
                
                # Generate price movement
                price_change = current_price * drift
                open_price = current_price + price_change
                
                # Generate intraday range
                intraday_volatility = volatility * 0.3
                high_price = open_price * (1 + abs(np.random.normal(0, intraday_volatility)))
                low_price = open_price * (1 - abs(np.random.normal(0, intraday_volatility)))
                
                # Close price
                close_price = open_price * (1 + np.random.normal(0, intraday_volatility * 0.8))
                
                # Ensure OHLC consistency
                high_price = max(open_price, close_price, high_price)
                low_price = min(open_price, close_price, low_price)
                
                # Generate volume
                avg_volume = metadata.get("avg_volume", 1000000)
                volume_multiplier = self._get_volume_multiplier()
                volume = int(avg_volume * volume_multiplier * np.random.uniform(0.5, 1.5))
                
                data_point = {
                    "timestamp": current_date.isoformat(),
                    "symbol": symbol,
                    "open": round(open_price, 2),
                    "high": round(high_price, 2),
                    "low": round(low_price, 2),
                    "close": round(close_price, 2),
                    "volume": volume,
                    "vwap": round((high_price + low_price + close_price) / 3, 2),
                    "exchange": metadata.get("exchange", "SIM"),
                    "currency": metadata.get("currency", "USD")
                }
                
                data.append(data_point)
                current_date += time_delta
                current_price = close_price
            
            return data
            
        except Exception as e:
            logger.error(f"Error generating data for {symbol}: {e}")
            return []
    
    async def _generate_data_point(self, symbol: str, metadata: Dict[str, Any], timestamp: datetime) -> Dict[str, Any]:
        """Generate a single data point"""
        current_price = await self._generate_current_price(symbol, metadata)
        
        # Generate small intraday movement
        volatility = metadata.get("volatility", 0.25)
        price_change = current_price * np.random.normal(0, volatility * 0.1)
        
        open_price = current_price + price_change
        high_price = open_price * (1 + abs(np.random.normal(0, volatility * 0.05)))
        low_price = open_price * (1 - abs(np.random.normal(0, volatility * 0.05)))
        close_price = open_price * (1 + np.random.normal(0, volatility * 0.03))
        
        # Ensure OHLC consistency
        high_price = max(open_price, close_price, high_price)
        low_price = min(open_price, close_price, low_price)
        
        return {
            "timestamp": timestamp.isoformat(),
            "symbol": symbol,
            "open": round(open_price, 2),
            "high": round(high_price, 2),
            "low": round(low_price, 2),
            "close": round(close_price, 2),
            "volume": int(np.random.lognormal(10, 1)),
            "vwap": round((high_price + low_price + close_price) / 3, 2)
        }
    
    async def _generate_current_price(self, symbol: str, metadata: Dict[str, Any]) -> float:
        """Generate current market price"""
        # Base price ranges by symbol type
        if "BTC" in symbol:
            base_price = 45000
        elif "ETH" in symbol:
            base_price = 2500
        elif "USD=" in symbol:
            base_price = 1.0
        elif "SPY" in symbol:
            base_price = 400
        elif "QQQ" in symbol:
            base_price = 350
        else:
            base_price = 100
        
        # Add some randomness
        price = base_price * (1 + np.random.normal(0, 0.1))
        return max(0.01, price)  # Ensure positive price
    
    def _get_volume_multiplier(self) -> float:
        """Get volume multiplier based on pattern"""
        if self.config.volume_pattern == "high":
            return random.uniform(1.5, 3.0)
        elif self.config.volume_pattern == "low":
            return random.uniform(0.3, 0.7)
        elif self.config.volume_pattern == "erratic":
            return random.uniform(0.1, 4.0)
        else:  # normal
            return random.uniform(0.8, 1.2)


class DemoBrokerFactory:
    """
    Factory for creating demo broker instances
    
    Provides:
    - Standardized demo broker creation
    - Broker capability simulation
    - Demo-specific configurations
    """
    
    def __init__(self, demo_manager: DemoModeManager):
        self.demo_manager = demo_manager
        self.broker_capabilities = self._initialize_broker_capabilities()
    
    async def create_demo_broker(self, broker_type: BrokerType) -> VirtualBroker:
        """Create a demo broker instance"""
        try:
            broker_config = BrokerConfig(
                broker_name=broker_type.value,
                is_paper=True  # Always paper trading in demo mode
            )
            
            broker = VirtualBroker(broker_config, self.demo_manager)
            
            # Initialize broker-specific capabilities
            await self._initialize_broker_capability(broker, broker_type)
            
            logger.info(f"Created demo broker: {broker_type.value}")
            return broker
            
        except Exception as e:
            logger.error(f"Error creating demo broker {broker_type.value}: {e}")
            raise
    
    async def get_broker_capabilities(self, broker_type: BrokerType) -> BrokerCapability:
        """Get capabilities for a specific broker"""
        return self.broker_capabilities.get(broker_type, self._get_default_capabilities())
    
    async def list_supported_brokers(self) -> List[BrokerType]:
        """List all supported broker types"""
        return list(BrokerType)
    
    async def validate_broker_config(self, broker_type: BrokerType, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate broker configuration for demo mode"""
        try:
            capabilities = await self.get_broker_capabilities(broker_type)
            
            validation_result = {
                "valid": True,
                "warnings": [],
                "errors": [],
                "recommendations": []
            }
            
            # Check required fields
            required_fields = ["api_key", "api_secret"]
            for field in required_fields:
                if field not in config:
                    validation_result["warnings"].append(f"Missing demo field: {field}")
            
            # Check supported instruments
            if "symbols" in config:
                unsupported_symbols = []
                for symbol in config["symbols"]:
                    if symbol not in capabilities.supported_instruments and capabilities.supported_instruments != ["*"]:
                        unsupported_symbols.append(symbol)
                
                if unsupported_symbols:
                    validation_result["warnings"].append(
                        f"Unsupported symbols for {broker_type.value}: {unsupported_symbols}"
                    )
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating broker config: {e}")
            return {"valid": False, "errors": [str(e)]}
    
    # Private methods
    
    def _initialize_broker_capabilities(self) -> Dict[BrokerType, BrokerCapability]:
        """Initialize broker capabilities"""
        return {
            BrokerType.ALPACA: BrokerCapability(
                name="Alpaca Markets",
                supported_instruments=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "SPY", "QQQ"],
                supported_order_types=["market", "limit", "stop", "stop_limit"],
                max_leverage=4.0,
                commission_structure={
                    "stocks": 0.0,  # Commission-free
                    "options": 0.0,
                    "crypto": 0.0
                },
                market_hours={
                    "regular": "09:30-16:00",
                    "extended": "04:00-20:00",
                    "premarket": "04:00-09:30",
                    "afterhours": "16:00-20:00"
                },
                special_features=["fractional_shares", "extended_hours", "crypto_trading"]
            ),
            
            BrokerType.BINANCE: BrokerCapability(
                name="Binance",
                supported_instruments=["BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "SOL-USD", "XRP-USD"],
                supported_order_types=["market", "limit", "stop_loss", "stop_loss_limit"],
                max_leverage=125.0,
                commission_structure={
                    "spot": 0.001,
                    "futures": 0.0002,
                    "options": 0.0002
                },
                market_hours={
                    "regular": "24/7",
                    "extended": "24/7",
                    "premarket": "N/A",
                    "afterhours": "N/A"
                },
                special_features=["futures", "options", "margin_trading", "lending"]
            ),
            
            BrokerType.TRADING212: BrokerCapability(
                name="Trading 212",
                supported_instruments=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "SPY", "QQQ", "EURUSD=X"],
                supported_order_types=["market", "limit"],
                max_leverage=30.0,
                commission_structure={
                    "stocks": 0.0,
                    "forex": 0.0,
                    "crypto": 0.0
                },
                market_hours={
                    "regular": "08:00-16:30",
                    "extended": "07:00-20:00"
                },
                special_features=["fractional_shares", "pies", "instant_deposit"]
            ),
            
            BrokerType.IBKR: BrokerCapability(
                name="Interactive Brokers",
                supported_instruments=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "SPY", "QQQ", "EURUSD=X", "USDJPY=X"],
                supported_order_types=["market", "limit", "stop", "stop_limit", "trailing_stop"],
                max_leverage=50.0,
                commission_structure={
                    "stocks": 0.005,
                    "options": 0.65,
                    "forex": 0.0002
                },
                market_hours={
                    "regular": "09:30-16:00",
                    "extended": "04:00-20:00"
                },
                special_features=["global_markets", "options", "futures", "forex", "bonds"]
            ),
            
            BrokerType.DEBIRO: BrokerCapability(
                name="DEGIRO",
                supported_instruments=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "SPY", "EURUSD=X", "USDJPY=X"],
                supported_order_types=["market", "limit"],
                max_leverage=10.0,
                commission_structure={
                    "stocks": 2.5,
                    "etfs": 2.5,
                    "bonds": 10.0
                },
                market_hours={
                    "regular": "09:00-17:30",
                    "extended": "08:00-20:00"
                },
                special_features=["global_etfs", "bonds", "options", "futures"]
            ),
            
            BrokerType.TRADE_REPUBLIC: BrokerCapability(
                name="Trade Republic",
                supported_instruments=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "SPY"],
                supported_order_types=["market", "limit"],
                max_leverage=20.0,
                commission_structure={
                    "stocks": 0.0,
                    "etfs": 0.0,
                    "savings_plans": 0.0
                },
                market_hours={
                    "regular": "09:00-17:30",
                    "extended": "07:00-23:00"
                },
                special_features=["savings_plans", "free_trading", "etf_savings"]
            ),
            
            BrokerType.XTB: BrokerCapability(
                name="XTB",
                supported_instruments=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "EURUSD=X", "USDJPY=X", "GOLD", "OIL"],
                supported_order_types=["market", "limit", "stop", "stop_limit"],
                max_leverage=500.0,
                commission_structure={
                    "stocks": 0.008,
                    "forex": 0.0,
                    "commodities": 0.004
                },
                market_hours={
                    "regular": "00:00-24:00",
                    "extended": "24/7"
                },
                special_features=["forex", "cfds", "commodities", "indices"]
            )
        }
    
    def _get_default_capabilities(self) -> BrokerCapability:
        """Get default capabilities for unknown brokers"""
        return BrokerCapability(
            name="Unknown Broker",
            supported_instruments=["AAPL", "MSFT", "GOOGL"],
            supported_order_types=["market", "limit"],
            max_leverage=1.0,
            commission_structure={"stocks": 0.01},
            market_hours={"regular": "09:30-16:00"},
            special_features=[]
        )
    
    async def _initialize_broker_capability(self, broker: VirtualBroker, broker_type: BrokerType):
        """Initialize broker-specific capabilities"""
        # This would set up broker-specific configurations
        # For demo mode, we use the virtual broker with simulated capabilities
        
        capabilities = self.broker_capabilities.get(broker_type)
        if capabilities:
            # Store capabilities in broker metadata
            broker.metadata = {
                "capabilities": capabilities,
                "demo_mode": True,
                "simulated": True
            }
    
    async def create_multi_broker_setup(self, brokers: List[BrokerType]) -> Dict[BrokerType, VirtualBroker]:
        """Create multiple demo brokers for portfolio testing"""
        try:
            broker_instances = {}
            
            for broker_type in brokers:
                broker = await self.create_demo_broker(broker_type)
                broker_instances[broker_type] = broker
            
            logger.info(f"Created {len(broker_instances)} demo brokers")
            return broker_instances
            
        except Exception as e:
            logger.error(f"Error creating multi-broker setup: {e}")
            raise


# Utility functions
async def create_demo_data(
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    frequency: str = "1d"
) -> Dict[str, List[Dict[str, Any]]]:
    """Create demo market data for symbols"""
    config = DemoDataConfig(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        frequency=frequency
    )
    
    generator = DemoDataGenerator(config)
    return await generator.generate_historical_data()


async def create_demo_broker(broker_type: BrokerType, demo_manager: DemoModeManager = None) -> VirtualBroker:
    """Create a single demo broker"""
    if demo_manager is None:
        from .demo_mode_manager import get_demo_manager
        demo_manager = await get_demo_manager()
    
    factory = DemoBrokerFactory(demo_manager)
    return await factory.create_demo_broker(broker_type)


async def create_demo_brokers(
    broker_types: List[BrokerType],
    demo_manager: DemoModeManager = None
) -> Dict[BrokerType, VirtualBroker]:
    """Create multiple demo brokers"""
    if demo_manager is None:
        from .demo_mode_manager import get_demo_manager
        demo_manager = await get_demo_manager()
    
    factory = DemoBrokerFactory(demo_manager)
    return await factory.create_multi_broker_setup(broker_types)


if __name__ == "__main__":
    # Example usage
    async def main():
        # Create demo data
        symbols = ["AAPL", "MSFT", "GOOGL", "SPY", "BTC-USD"]
        start_date = datetime.now() - timedelta(days=365)
        end_date = datetime.now()
        
        data = await create_demo_data(symbols, start_date, end_date)
        print(f"Generated data for {len(data)} symbols")
        
        # Create demo broker
        broker = await create_demo_broker(BrokerType.ALPACA)
        print(f"Created demo broker: {broker.broker_name}")
        
        # Create multiple brokers
        brokers = await create_demo_brokers([BrokerType.ALPACA, BrokerType.BINANCE, BrokerType.TRADING212])
        print(f"Created {len(brokers)} demo brokers")
    
    asyncio.run(main())
