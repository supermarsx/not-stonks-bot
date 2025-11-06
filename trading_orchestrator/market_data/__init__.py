"""
Market Data System
Real-time and historical market data management with feature calculation
"""

from typing import Dict, Any, List, Optional, Union, Protocol, Callable
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
from collections import defaultdict, deque

from loguru import logger

from trading.models import MarketData
from trading.database import get_db_session


class DataFrequency(Enum):
    """Data frequency levels"""
    TICK = "tick"
    SECOND = "1s"
    MINUTE = "1m"
    FIVE_MINUTE = "5m"
    FIFTEEN_MINUTE = "15m"
    HOUR = "1h"
    FOUR_HOUR = "4h"
    DAY = "1d"
    WEEK = "1w"
    MONTH = "1M"


class DataSource(Enum):
    """Market data sources"""
    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    POLYGON = "polygon"
    IEX_CLOUD = "iex_cloud"
    BINANCE = "binance"
    COINBASE = "coinbase"
    SIMULATED = "simulated"


@dataclass
class MarketDataPoint:
    """Individual market data point"""
    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    source: DataSource = DataSource.SIMULATED
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureValue:
    """Calculated feature value"""
    feature_name: str
    symbol: str
    timestamp: datetime
    value: float
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MarketDataProvider(Protocol):
    """Protocol for market data providers"""
    
    async def get_real_time_price(self, symbol: str) -> Optional[Decimal]:
        """Get real-time price for symbol"""
        ...
    
    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        frequency: DataFrequency
    ) -> List[MarketDataPoint]:
        """Get historical market data"""
        ...
    
    async def subscribe_to_stream(self, symbols: List[str]) -> AsyncIterator[MarketDataPoint]:
        """Subscribe to real-time data stream"""
        ...


class SimulatedDataProvider:
    """Simulated market data provider for testing"""
    
    def __init__(self):
        self.price_cache: Dict[str, Decimal] = {}
        self.data_history: Dict[str, List[MarketDataPoint]] = defaultdict(list)
    
    async def get_real_time_price(self, symbol: str) -> Optional[Decimal]:
        """Generate simulated real-time price"""
        import random
        
        base_price = self.price_cache.get(symbol, Decimal('100'))
        
        # Random walk with slight upward trend
        change = random.gauss(0, 0.01)  # 1% standard deviation
        new_price = base_price * (1 + change)
        
        self.price_cache[symbol] = new_price
        
        return new_price
    
    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        frequency: DataFrequency
    ) -> List[MarketDataPoint]:
        """Generate simulated historical data"""
        import random
        
        data = []
        current_price = Decimal('100')
        current_time = start_date
        
        # Generate data points based on frequency
        time_delta = self._get_time_delta(frequency)
        
        while current_time <= end_date:
            # Generate OHLC
            open_price = current_price * Decimal(str(random.uniform(0.98, 1.02)))
            close_price = open_price * Decimal(str(random.uniform(0.95, 1.05)))
            high_price = max(open_price, close_price) * Decimal(str(random.uniform(1.0, 1.03)))
            low_price = min(open_price, close_price) * Decimal(str(random.uniform(0.97, 1.0)))
            volume = Decimal(str(random.randint(10000, 1000000)))
            
            data_point = MarketDataPoint(
                symbol=symbol,
                timestamp=current_time,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
                source=DataSource.SIMULATED
            )
            
            data.append(data_point)
            current_price = close_price
            current_time += time_delta
        
        # Cache the data
        self.data_history[symbol].extend(data)
        return data
    
    def _get_time_delta(self, frequency: DataFrequency) -> timedelta:
        """Get time delta for frequency"""
        time_deltas = {
            DataFrequency.SECOND: timedelta(seconds=1),
            DataFrequency.MINUTE: timedelta(minutes=1),
            DataFrequency.FIVE_MINUTE: timedelta(minutes=5),
            DataFrequency.FIFTEEN_MINUTE: timedelta(minutes=15),
            DataFrequency.HOUR: timedelta(hours=1),
            DataFrequency.FOUR_HOUR: timedelta(hours=4),
            DataFrequency.DAY: timedelta(days=1),
            DataFrequency.WEEK: timedelta(weeks=1),
            DataFrequency.MONTH: timedelta(days=30)
        }
        return time_deltas.get(frequency, timedelta(days=1))
    
    async def subscribe_to_stream(self, symbols: List[str]):
        """Simulated real-time data stream"""
        import random
        import asyncio
        
        while True:
            for symbol in symbols:
                price = await self.get_real_time_price(symbol)
                
                data_point = MarketDataPoint(
                    symbol=symbol,
                    timestamp=datetime.utcnow(),
                    open=price,
                    high=price * Decimal('1.001'),
                    low=price * Decimal('0.999'),
                    close=price,
                    volume=Decimal(str(random.randint(1000, 10000))),
                    source=DataSource.SIMULATED
                )
                
                yield data_point
            
            await asyncio.sleep(1)  # 1-second intervals


class FeatureCalculator:
    """Technical indicator and feature calculation engine"""
    
    def __init__(self):
        self.calculators = {
            'sma': self._calculate_sma,
            'ema': self._calculate_ema,
            'rsi': self._calculate_rsi,
            'bollinger_bands': self._calculate_bollinger_bands,
            'macd': self._calculate_macd,
            'stochastic': self._calculate_stochastic,
            'atr': self._calculate_atr,
            'volume_sma': self._calculate_volume_sma,
            'price_momentum': self._calculate_momentum,
            'volatility': self._calculate_volatility
        }
    
    async def calculate_features(
        self,
        data_points: List[MarketDataPoint],
        feature_config: Dict[str, Dict[str, Any]]
    ) -> List[FeatureValue]:
        """Calculate multiple features for data points"""
        features = []
        
        for feature_name, params in feature_config.items():
            if feature_name in self.calculators:
                try:
                    feature_values = await self.calculators[feature_name](data_points, **params)
                    features.extend(feature_values)
                except Exception as e:
                    logger.error(f"Error calculating feature {feature_name}: {e}")
        
        return features
    
    async def _calculate_sma(
        self,
        data_points: List[MarketDataPoint],
        period: int = 20,
        price_field: str = 'close'
    ) -> List[FeatureValue]:
        """Calculate Simple Moving Average"""
        if len(data_points) < period:
            return []
        
        prices = [getattr(dp, price_field) for dp in data_points]
        features = []
        
        for i in range(period - 1, len(prices)):
            sma_value = sum(prices[i - period + 1:i + 1]) / period
            data_point = data_points[i]
            
            features.append(FeatureValue(
                feature_name='sma',
                symbol=data_point.symbol,
                timestamp=data_point.timestamp,
                value=float(sma_value),
                parameters={'period': period, 'price_field': price_field}
            ))
        
        return features
    
    async def _calculate_ema(
        self,
        data_points: List[MarketDataPoint],
        period: int = 20,
        price_field: str = 'close'
    ) -> List[FeatureValue]:
        """Calculate Exponential Moving Average"""
        if len(data_points) < period:
            return []
        
        prices = [float(getattr(dp, price_field)) for dp in data_points]
        features = []
        
        multiplier = 2.0 / (period + 1)
        ema_value = sum(prices[:period]) / period
        
        for i in range(period - 1, len(prices)):
            if i == period - 1:
                ema_value = sum(prices[:period]) / period
            else:
                ema_value = (prices[i] - ema_value) * multiplier + ema_value
            
            data_point = data_points[i]
            
            features.append(FeatureValue(
                feature_name='ema',
                symbol=data_point.symbol,
                timestamp=data_point.timestamp,
                value=ema_value,
                parameters={'period': period, 'price_field': price_field}
            ))
        
        return features
    
    async def _calculate_rsi(
        self,
        data_points: List[MarketDataPoint],
        period: int = 14,
        price_field: str = 'close'
    ) -> List[FeatureValue]:
        """Calculate Relative Strength Index"""
        if len(data_points) < period + 1:
            return []
        
        prices = [float(getattr(dp, price_field)) for dp in data_points]
        features = []
        
        # Calculate price changes
        changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        
        # Calculate gains and losses
        gains = [max(change, 0) for change in changes]
        losses = [max(-change, 0) for change in changes]
        
        # Calculate initial average gain and loss
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        for i in range(period, len(gains)):
            # Update average gain and loss
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            # Calculate RSI
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            data_point = data_points[i + 1]  # +1 because changes start from index 1
            
            features.append(FeatureValue(
                feature_name='rsi',
                symbol=data_point.symbol,
                timestamp=data_point.timestamp,
                value=rsi,
                parameters={'period': period, 'price_field': price_field}
            ))
        
        return features
    
    async def _calculate_bollinger_bands(
        self,
        data_points: List[MarketDataPoint],
        period: int = 20,
        std_dev: float = 2.0,
        price_field: str = 'close'
    ) -> List[FeatureValue]:
        """Calculate Bollinger Bands"""
        if len(data_points) < period:
            return []
        
        prices = [float(getattr(dp, price_field)) for dp in data_points]
        features = []
        
        for i in range(period - 1, len(prices)):
            price_window = prices[i - period + 1:i + 1]
            
            # Calculate middle band (SMA)
            sma = sum(price_window) / period
            
            # Calculate standard deviation
            variance = sum((price - sma) ** 2 for price in price_window) / period
            std = variance ** 0.5
            
            # Calculate upper and lower bands
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            data_point = data_points[i]
            
            # Store all three bands as separate features
            features.extend([
                FeatureValue(
                    feature_name='bb_upper',
                    symbol=data_point.symbol,
                    timestamp=data_point.timestamp,
                    value=upper_band,
                    parameters={'period': period, 'std_dev': std_dev}
                ),
                FeatureValue(
                    feature_name='bb_middle',
                    symbol=data_point.symbol,
                    timestamp=data_point.timestamp,
                    value=sma,
                    parameters={'period': period, 'std_dev': std_dev}
                ),
                FeatureValue(
                    feature_name='bb_lower',
                    symbol=data_point.symbol,
                    timestamp=data_point.timestamp,
                    value=lower_band,
                    parameters={'period': period, 'std_dev': std_dev}
                )
            ])
        
        return features
    
    async def _calculate_macd(
        self,
        data_points: List[MarketDataPoint],
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        price_field: str = 'close'
    ) -> List[FeatureValue]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        # First calculate EMAs
        fast_ema_data = await self._calculate_ema(data_points, fast_period, price_field)
        slow_ema_data = await self._calculate_ema(data_points, slow_period, price_field)
        
        features = []
        
        # Align the data points
        for fast_ema, slow_ema in zip(fast_ema_data, slow_ema_data):
            if fast_ema.timestamp == slow_ema.timestamp:
                macd_line = fast_ema.value - slow_ema.value
                
                features.append(FeatureValue(
                    feature_name='macd',
                    symbol=fast_ema.symbol,
                    timestamp=fast_ema.timestamp,
                    value=macd_line,
                    parameters={
                        'fast_period': fast_period,
                        'slow_period': slow_period,
                        'signal_period': signal_period
                    }
                ))
        
        return features
    
    async def _calculate_stochastic(
        self,
        data_points: List[MarketDataPoint],
        k_period: int = 14,
        d_period: int = 3
    ) -> List[FeatureValue]:
        """Calculate Stochastic Oscillator"""
        if len(data_points) < k_period:
            return []
        
        features = []
        
        for i in range(k_period - 1, len(data_points)):
            data_window = data_points[i - k_period + 1:i + 1]
            
            highest_high = max(dp.high for dp in data_window)
            lowest_low = min(dp.low for dp in data_window)
            current_close = data_points[i].close
            
            # Calculate %K
            if highest_high != lowest_low:
                k_value = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
            else:
                k_value = 50
            
            data_point = data_points[i]
            
            features.append(FeatureValue(
                feature_name='stochastic_k',
                symbol=data_point.symbol,
                timestamp=data_point.timestamp,
                value=k_value,
                parameters={'k_period': k_period, 'd_period': d_period}
            ))
        
        return features
    
    async def _calculate_atr(
        self,
        data_points: List[MarketDataPoint],
        period: int = 14
    ) -> List[FeatureValue]:
        """Calculate Average True Range"""
        if len(data_points) < period + 1:
            return []
        
        features = []
        
        # Calculate True Range for each period
        true_ranges = []
        
        for i in range(1, len(data_points)):
            current = data_points[i]
            previous = data_points[i - 1]
            
            tr1 = current.high - current.low
            tr2 = abs(current.high - previous.close)
            tr3 = abs(current.low - previous.close)
            
            true_range = max(tr1, tr2, tr3)
            true_ranges.append(true_range)
        
        # Calculate ATR using simple moving average
        for i in range(period - 1, len(true_ranges)):
            atr_value = sum(true_ranges[i - period + 1:i + 1]) / period
            
            data_point = data_points[i + 1]  # +1 because true_ranges starts from index 1
            
            features.append(FeatureValue(
                feature_name='atr',
                symbol=data_point.symbol,
                timestamp=data_point.timestamp,
                value=float(atr_value),
                parameters={'period': period}
            ))
        
        return features
    
    async def _calculate_volume_sma(
        self,
        data_points: List[MarketDataPoint],
        period: int = 20
    ) -> List[FeatureValue]:
        """Calculate Volume Simple Moving Average"""
        if len(data_points) < period:
            return []
        
        volumes = [float(dp.volume) for dp in data_points]
        features = []
        
        for i in range(period - 1, len(volumes)):
            sma_volume = sum(volumes[i - period + 1:i + 1]) / period
            
            data_point = data_points[i]
            
            features.append(FeatureValue(
                feature_name='volume_sma',
                symbol=data_point.symbol,
                timestamp=data_point.timestamp,
                value=sma_volume,
                parameters={'period': period}
            ))
        
        return features
    
    async def _calculate_momentum(
        self,
        data_points: List[MarketDataPoint],
        period: int = 10,
        price_field: str = 'close'
    ) -> List[FeatureValue]:
        """Calculate Price Momentum"""
        if len(data_points) < period + 1:
            return []
        
        prices = [float(getattr(dp, price_field)) for dp in data_points]
        features = []
        
        for i in range(period, len(prices)):
            momentum = ((prices[i] - prices[i - period]) / prices[i - period]) * 100
            
            data_point = data_points[i]
            
            features.append(FeatureValue(
                feature_name='momentum',
                symbol=data_point.symbol,
                timestamp=data_point.timestamp,
                value=momentum,
                parameters={'period': period, 'price_field': price_field}
            ))
        
        return features
    
    async def _calculate_volatility(
        self,
        data_points: List[MarketDataPoint],
        period: int = 20,
        price_field: str = 'close'
    ) -> List[FeatureValue]:
        """Calculate Price Volatility (Standard Deviation)"""
        if len(data_points) < period:
            return []
        
        prices = [float(getattr(dp, price_field)) for dp in data_points]
        features = []
        
        for i in range(period - 1, len(prices)):
            price_window = prices[i - period + 1:i + 1]
            
            mean_price = sum(price_window) / len(price_window)
            variance = sum((price - mean_price) ** 2 for price in price_window) / len(price_window)
            volatility = variance ** 0.5
            
            data_point = data_points[i]
            
            features.append(FeatureValue(
                feature_name='volatility',
                symbol=data_point.symbol,
                timestamp=data_point.timestamp,
                value=volatility,
                parameters={'period': period, 'price_field': price_field}
            ))
        
        return features


class MarketDataManager:
    """Main market data management system"""
    
    def __init__(self):
        self.providers: Dict[DataSource, MarketDataProvider] = {}
        self.feature_calculator = FeatureCalculator()
        self.data_cache: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.feature_cache: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        
        # Register default simulated provider
        self.register_provider(DataSource.SIMULATED, SimulatedDataProvider())
    
    def register_provider(self, source: DataSource, provider: MarketDataProvider):
        """Register a market data provider"""
        self.providers[source] = provider
        logger.info(f"Registered market data provider: {source.value}")
    
    async def get_real_time_price(
        self,
        symbol: str,
        source: DataSource = DataSource.SIMULATED
    ) -> Optional[Decimal]:
        """Get real-time price for symbol"""
        if source in self.providers:
            return await self.providers[source].get_real_time_price(symbol)
        return None
    
    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        frequency: DataFrequency = DataFrequency.DAY,
        source: DataSource = DataSource.SIMULATED,
        use_cache: bool = True
    ) -> List[MarketDataPoint]:
        """Get historical market data"""
        # Check cache first
        cache_key = f"{symbol}_{source.value}_{frequency.value}"
        
        if use_cache and cache_key in self.data_cache:
            cached_data = list(self.data_cache[cache_key])
            # Filter by date range
            return [
                dp for dp in cached_data
                if start_date <= dp.timestamp <= end_date
            ]
        
        # Fetch from provider
        if source in self.providers:
            data = await self.providers[source].get_historical_data(
                symbol, start_date, end_date, frequency
            )
            
            # Cache the data
            if use_cache:
                self.data_cache[cache_key].extend(data)
            
            return data
        
        return []
    
    async def calculate_features(
        self,
        symbol: str,
        data_points: List[MarketDataPoint],
        feature_config: Dict[str, Dict[str, Any]]
    ) -> List[FeatureValue]:
        """Calculate features for market data"""
        features = await self.feature_calculator.calculate_features(data_points, feature_config)
        
        # Cache features
        cache_key = f"{symbol}_{hash(str(feature_config))}"
        self.feature_cache[cache_key].extend(features)
        
        return features
    
    async def save_market_data(
        self,
        data_points: List[MarketDataPoint],
        source: DataSource = DataSource.SIMULATED
    ):
        """Save market data to database"""
        try:
            async with get_db_session() as session:
                for data_point in data_points:
                    market_data = MarketData(
                        symbol=data_point.symbol,
                        timestamp=data_point.timestamp,
                        open_price=data_point.open,
                        high_price=data_point.high,
                        low_price=data_point.low,
                        close_price=data_point.close,
                        volume=data_point.volume,
                        data_source=source.value
                    )
                    
                    session.add(market_data)
                
                await session.commit()
                
        except Exception as e:
            logger.error(f"Error saving market data: {e}")
    
    async def get_market_summary(self, symbols: List[str]) -> Dict[str, Any]:
        """Get market summary for symbols"""
        summary = {}
        
        for symbol in symbols:
            try:
                # Get recent data
                end_date = datetime.utcnow()
                start_date = end_date - timedelta(days=30)
                
                recent_data = await self.get_historical_data(
                    symbol, start_date, end_date, DataFrequency.DAY
                )
                
                if recent_data:
                    latest = recent_data[-1]
                    prev_close = recent_data[-2].close if len(recent_data) > 1 else latest.close
                    
                    change = latest.close - prev_close
                    change_pct = (change / prev_close) * 100
                    
                    summary[symbol] = {
                        'current_price': float(latest.close),
                        'change': float(change),
                        'change_pct': change_pct,
                        'volume': float(latest.volume),
                        'high_30d': float(max(dp.high for dp in recent_data)),
                        'low_30d': float(min(dp.low for dp in recent_data)),
                        'last_update': latest.timestamp.isoformat()
                    }
                
            except Exception as e:
                logger.error(f"Error getting summary for {symbol}: {e}")
                summary[symbol] = {'error': str(e)}
        
        return summary


# Example usage and testing
if __name__ == "__main__":
    async def test_market_data_system():
        # Create market data manager
        manager = MarketDataManager()
        
        # Test historical data
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=30)
        
        data = await manager.get_historical_data(
            symbol="AAPL",
            start_date=start_date,
            end_date=end_date,
            frequency=DataFrequency.DAY
        )
        
        print(f"Retrieved {len(data)} data points for AAPL")
        
        # Test feature calculation
        feature_config = {
            'sma': {'period': 20},
            'rsi': {'period': 14},
            'bollinger_bands': {'period': 20, 'std_dev': 2.0}
        }
        
        features = await manager.calculate_features("AAPL", data, feature_config)
        
        print(f"Calculated {len(features)} features")
        for feature in features[-5:]:  # Show last 5 features
            print(f"  {feature.feature_name}: {feature.value:.4f} at {feature.timestamp}")
        
        # Test market summary
        summary = await manager.get_market_summary(["AAPL", "GOOGL"])
        
        print("\nMarket Summary:")
        for symbol, info in summary.items():
            print(f"  {symbol}: ${info.get('current_price', 0):.2f} "
                  f"({info.get('change_pct', 0):+.2f}%)")
    
    import asyncio
    asyncio.run(test_market_data_system())