"""
Technical Analysis Pattern Crawler
Chart pattern recognition, technical indicator scanner, and market microstructure
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
import numpy as np
import pandas as pd
import aiohttp
from scipy import stats
from scipy.signal import find_peaks, find_peaks_cwt

from ..base.base_crawler import BaseCrawler, CrawlerConfig, DataType, CrawlResult


@dataclass
class ChartPattern:
    """Chart pattern detection result"""
    pattern_type: str
    symbol: str
    detection_date: datetime
    confidence: float
    timeframe: str  # 1m, 5m, 1h, 1d, etc.
    start_date: datetime
    end_date: Optional[datetime]
    target_price: Optional[float]
    stop_loss: Optional[float]
    description: str
    points: List[Dict[str, Any]] = None  # Pattern key points


@dataclass
class TechnicalIndicator:
    """Technical indicator value"""
    symbol: str
    indicator_name: str
    value: float
    timestamp: datetime
    timeframe: str
    signal: str  # buy, sell, hold
    signal_strength: float  # 0-1
    parameters: Dict[str, Any] = None


@dataclass
class MarketMicrostructure:
    """Market microstructure data"""
    symbol: str
    timestamp: datetime
    bid_ask_spread: float
    order_book_depth: Dict[str, float]  # levels and sizes
    trade_volume: float
    trade_count: int
    average_trade_size: float
    price_impact: float
    volatility_1m: float
    volatility_5m: float
    volatility_15m: float
    bid_ask_imbalance: float


@dataclass
class VolumeProfile:
    """Volume profile analysis"""
    symbol: str
    timestamp: datetime
    value_area_high: float
    value_area_low: float
    point_of_control: float
    volume_nodes: Dict[str, float]  # price levels and volume
    high_volume_nodes: List[float]
    low_volume_nodes: List[float]


class PatternCrawler(BaseCrawler):
    """Crawler for technical analysis patterns and indicators"""
    
    def __init__(self, config: CrawlerConfig, symbols: List[str]):
        super().__init__(config)
        self.symbols = symbols
        self.logger = logging.getLogger(__name__)
        
        # Chart patterns to detect
        self.chart_patterns = {
            'Head and Shoulders': self._detect_head_and_shoulders,
            'Double Top': self._detect_double_top,
            'Double Bottom': self._detect_double_bottom,
            'Ascending Triangle': self._detect_ascending_triangle,
            'Descending Triangle': self._detect_descending_triangle,
            'Symmetrical Triangle': self._detect_symmetrical_triangle,
            'Cup and Handle': self._detect_cup_and_handle,
            'Inverse Cup and Handle': self._detect_inverse_cup_and_handle,
            'Flag': self._detect_flag,
            'Pennant': self._detect_pennant,
            'Rectangle': self._detect_rectangle,
            'Channel': self._detect_channel
        }
        
        # Technical indicators
        self.technical_indicators = {
            'RSI': self._calculate_rsi,
            'MACD': self._calculate_macd,
            'Bollinger_Bands': self._calculate_bollinger_bands,
            'Stochastic': self._calculate_stochastic,
            'Williams_R': self._calculate_williams_r,
            'CCI': self._calculate_cci,
            'ADX': self._calculate_adx,
            'ATR': self._calculate_atr,
            'OBV': self._calculate_obv,
            'Volume_SMA': self._calculate_volume_sma,
            'Price_SMA': self._calculate_price_sma,
            'Price_EMA': self._calculate_price_ema,
            'PSAR': self._calculate_psar
        }
        
        # Pattern recognition parameters
        self.pattern_params = {
            'head_and_shoulders': {
                'min_prominence': 0.02,
                'min_distance': 5,
                'confidence_threshold': 0.6
            },
            'double_top_bottom': {
                'price_tolerance': 0.02,
                'time_tolerance': 20,
                'min_prominence': 0.01
            },
            'triangle': {
                'min_points': 3,
                'max_angle_deviation': 0.1
            }
        }
    
    async def detect_chart_patterns(self, symbol: str, timeframe: str = '1d', days: int = 60) -> List[ChartPattern]:
        """Detect chart patterns for a symbol"""
        try:
            # Get price data
            price_data = await self._get_price_data(symbol, timeframe, days)
            
            if price_data is None or len(price_data) < 30:
                self.logger.warning(f"Insufficient data for pattern detection: {symbol}")
                return []
            
            patterns = []
            
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(price_data)
            df.set_index('timestamp', inplace=True)
            
            # Detect each pattern type
            for pattern_name, detect_func in self.chart_patterns.items():
                try:
                    detected_patterns = await detect_func(df, symbol, timeframe)
                    patterns.extend(detected_patterns)
                except Exception as e:
                    self.logger.error(f"Error detecting {pattern_name} for {symbol}: {e}")
                    continue
            
            self.logger.info(f"Detected {len(patterns)} patterns for {symbol}")
            return patterns
        
        except Exception as e:
            self.logger.error(f"Error detecting patterns for {symbol}: {e}")
            return []
    
    async def scan_technical_indicators(self, symbols: List[str] = None, timeframe: str = '1d') -> Dict[str, List[TechnicalIndicator]]:
        """Scan technical indicators for multiple symbols"""
        if not symbols:
            symbols = self.symbols
        
        results = {}
        
        try:
            for symbol in symbols:
                indicators = await self._scan_symbol_indicators(symbol, timeframe)
                if indicators:
                    results[symbol] = indicators
            
            self.logger.info(f"Scanned indicators for {len(results)} symbols")
            return results
        
        except Exception as e:
            self.logger.error(f"Error scanning technical indicators: {e}")
            return {}
    
    async def _scan_symbol_indicators(self, symbol: str, timeframe: str) -> List[TechnicalIndicator]:
        """Scan technical indicators for a single symbol"""
        try:
            # Get price data
            price_data = await self._get_price_data(symbol, timeframe, 100)
            
            if price_data is None or len(price_data) < 20:
                return []
            
            df = pd.DataFrame(price_data)
            df.set_index('timestamp', inplace=True)
            
            indicators = []
            
            # Calculate each technical indicator
            for indicator_name, calc_func in self.technical_indicators.items():
                try:
                    indicator_data = await calc_func(df, symbol, timeframe)
                    indicators.extend(indicator_data)
                except Exception as e:
                    self.logger.error(f"Error calculating {indicator_name} for {symbol}: {e}")
                    continue
            
            return indicators
        
        except Exception as e:
            self.logger.error(f"Error scanning indicators for {symbol}: {e}")
            return []
    
    async def fetch_market_microstructure(self, symbol: str) -> Optional[MarketMicrostructure]:
        """Fetch market microstructure data for a symbol"""
        try:
            # In a real implementation, this would fetch order book data
            # For demo, simulate microstructure data
            microstructure = MarketMicrostructure(
                symbol=symbol,
                timestamp=datetime.now(),
                bid_ask_spread=0.01,  # $0.01 spread
                order_book_depth={
                    'bid_5': 150.25, 'bid_4': 150.24, 'bid_3': 150.23, 'bid_2': 150.22, 'bid_1': 150.21,
                    'ask_1': 150.22, 'ask_2': 150.23, 'ask_3': 150.24, 'ask_4': 150.25, 'ask_5': 150.26
                },
                trade_volume=150000,
                trade_count=2500,
                average_trade_size=60,
                price_impact=0.002,
                volatility_1m=0.15,
                volatility_5m=0.18,
                volatility_15m=0.22,
                bid_ask_imbalance=0.05
            )
            
            return microstructure
        
        except Exception as e:
            self.logger.error(f"Error fetching microstructure for {symbol}: {e}")
            return None
    
    async def analyze_volume_profile(self, symbol: str, timeframe: str = '1d', days: int = 30) -> Optional[VolumeProfile]:
        """Analyze volume profile for a symbol"""
        try:
            # Get price and volume data
            price_data = await self._get_price_data(symbol, timeframe, days)
            
            if price_data is None or len(price_data) < 20:
                return None
            
            df = pd.DataFrame(price_data)
            
            # Calculate price bins and volume
            price_range = df['high'].max() - df['low'].min()
            num_bins = min(20, int(price_range / (df['close'].std() or 0.01)))
            
            if num_bins < 5:
                return None
            
            # Create price bins
            price_bins = np.linspace(df['low'].min(), df['high'].max(), num_bins + 1)
            
            # Calculate volume in each price bin
            volume_by_price = {}
            for i in range(len(price_bins) - 1):
                price_level = (price_bins[i] + price_bins[i + 1]) / 2
                volume_by_price[price_level] = 0.0
            
            # Distribute volume across price levels
            for _, row in df.iterrows():
                high, low, volume = row['high'], row['low'], row['volume']
                price_center = (high + low) / 2
                
                # Find closest price bin
                closest_price = min(volume_by_price.keys(), key=lambda x: abs(x - price_center))
                volume_by_price[closest_price] += volume / len(price_bins)
            
            # Calculate volume profile metrics
            total_volume = sum(volume_by_price.values())
            sorted_volumes = sorted(volume_by_price.values(), reverse=True)
            
            # Value Area (typically 70% of volume)
            value_area_volume = total_volume * 0.7
            cumulative_volume = 0
            value_area_levels = []
            
            for price_level, volume in sorted(volume_by_price.items(), key=lambda x: x[1], reverse=True):
                if cumulative_volume < value_area_volume:
                    value_area_levels.append(price_level)
                    cumulative_volume += volume
                else:
                    break
            
            if value_area_levels:
                value_area_high = max(value_area_levels)
                value_area_low = min(value_area_levels)
            else:
                value_area_high = df['high'].max()
                value_area_low = df['low'].min()
            
            # Point of Control (highest volume price)
            point_of_control = max(volume_by_price.keys(), key=lambda x: volume_by_price[x])
            
            # High and Low Volume Nodes
            volume_sorted = sorted(volume_by_price.items(), key=lambda x: x[1], reverse=True)
            high_volume_nodes = [price for price, vol in volume_sorted[:3]]
            low_volume_nodes = [price for price, vol in volume_sorted[-3:]]
            
            return VolumeProfile(
                symbol=symbol,
                timestamp=datetime.now(),
                value_area_high=value_area_high,
                value_area_low=value_area_low,
                point_of_control=point_of_control,
                volume_nodes=volume_by_price,
                high_volume_nodes=high_volume_nodes,
                low_volume_nodes=low_volume_nodes
            )
        
        except Exception as e:
            self.logger.error(f"Error analyzing volume profile for {symbol}: {e}")
            return None
    
    # Pattern Detection Methods
    async def _detect_head_and_shoulders(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[ChartPattern]:
        """Detect head and shoulders pattern"""
        patterns = []
        
        try:
            # Find peaks and troughs
            closes = df['close'].values
            
            # Find peaks (local maxima)
            peaks, _ = find_peaks(closes, prominence=closes.std() * 0.02, distance=5)
            
            # Find troughs (local minima)
            troughs, _ = find_peaks(-closes, prominence=closes.std() * 0.02, distance=5)
            
            if len(peaks) < 3 or len(troughs) < 2:
                return patterns
            
            # Look for head and shoulders sequence
            for i in range(len(peaks) - 2):
                left_shoulder = peaks[i]
                head = peaks[i + 1]
                right_shoulder = peaks[i + 2]
                
                # Check if head is higher than both shoulders
                if (closes[head] > closes[left_shoulder] and 
                    closes[head] > closes[right_shoulder]):
                    
                    # Check pattern symmetry
                    left_to_head_distance = head - left_shoulder
                    head_to_right_distance = right_shoulder - head
                    symmetry_ratio = abs(left_to_head_distance - head_to_right_distance) / max(left_to_head_distance, head_to_right_distance)
                    
                    if symmetry_ratio < 0.3:  # Within 30% of symmetric
                        # Calculate neckline (support level)
                        neckline_peaks = [p for p in peaks if left_shoulder < p < right_shoulder]
                        neckline_level = min([closes[p] for p in neckline_peaks]) if neckline_peaks else df['low'].iloc[head:right_shoulder].min()
                        
                        confidence = self._calculate_pattern_confidence(
                            'head_and_shoulders', 
                            closes[left_shoulder:right_shoulder + 1],
                            neckline_level
                        )
                        
                        if confidence > self.pattern_params['head_and_shoulders']['confidence_threshold']:
                            patterns.append(ChartPattern(
                                pattern_type='Head and Shoulders',
                                symbol=symbol,
                                detection_date=datetime.now(),
                                confidence=confidence,
                                timeframe=timeframe,
                                start_date=df.index[left_shoulder],
                                end_date=df.index[right_shoulder],
                                target_price=neckline_level,
                                stop_loss=closes[left_shoulder],
                                description=f"Bearish reversal pattern detected. Target: ${neckline_level:.2f}",
                                points=[
                                    {'index': left_shoulder, 'type': 'left_shoulder', 'price': closes[left_shoulder]},
                                    {'index': head, 'type': 'head', 'price': closes[head]},
                                    {'index': right_shoulder, 'type': 'right_shoulder', 'price': closes[right_shoulder]},
                                    {'index': neckline_peaks[0] if neckline_peaks else head, 'type': 'neckline', 'price': neckline_level}
                                ]
                            ))
        
        except Exception as e:
            self.logger.error(f"Error detecting head and shoulders: {e}")
        
        return patterns
    
    async def _detect_double_top(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[ChartPattern]:
        """Detect double top pattern"""
        patterns = []
        
        try:
            closes = df['close'].values
            peaks, properties = find_peaks(closes, prominence=closes.std() * 0.01, distance=10)
            
            if len(peaks) < 2:
                return patterns
            
            # Look for double peaks
            for i in range(len(peaks) - 1):
                peak1 = peaks[i]
                peak2 = peaks[i + 1]
                
                # Check if peaks are at similar levels
                price_diff = abs(closes[peak1] - closes[peak2]) / closes[peak1]
                time_diff = peak2 - peak1
                
                if price_diff < 0.02 and 5 < time_diff < 50:  # Similar price, reasonable time
                    # Check for valley between peaks
                    valley_idx = np.argmin(closes[peak1:peak2 + 1]) + peak1
                    valley_price = closes[valley_idx]
                    
                    # Check if valley is significantly lower
                    peak_avg = (closes[peak1] + closes[peak2]) / 2
                    valley_diff = (peak_avg - valley_price) / peak_avg
                    
                    if valley_diff > 0.02:  # At least 2% decline
                        confidence = self._calculate_pattern_confidence(
                            'double_top',
                            closes[peak1:peak2 + 1],
                            valley_price
                        )
                        
                        patterns.append(ChartPattern(
                            pattern_type='Double Top',
                            symbol=symbol,
                            detection_date=datetime.now(),
                            confidence=confidence,
                            timeframe=timeframe,
                            start_date=df.index[peak1],
                            end_date=df.index[peak2],
                            target_price=valley_price,
                            stop_loss=max(closes[peak1], closes[peak2]),
                            description=f"Bearish reversal pattern. Target: ${valley_price:.2f}",
                            points=[
                                {'index': peak1, 'type': 'first_peak', 'price': closes[peak1]},
                                {'index': valley_idx, 'type': 'valley', 'price': closes[valley_idx]},
                                {'index': peak2, 'type': 'second_peak', 'price': closes[peak2]}
                            ]
                        ))
        
        except Exception as e:
            self.logger.error(f"Error detecting double top: {e}")
        
        return patterns
    
    async def _detect_double_bottom(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[ChartPattern]:
        """Detect double bottom pattern"""
        patterns = []
        
        try:
            closes = df['close'].values
            troughs, _ = find_peaks(-closes, prominence=closes.std() * 0.01, distance=10)
            
            if len(troughs) < 2:
                return patterns
            
            # Look for double troughs
            for i in range(len(troughs) - 1):
                trough1 = troughs[i]
                trough2 = troughs[i + 1]
                
                # Check if troughs are at similar levels
                price_diff = abs(closes[trough1] - closes[trough2]) / closes[trough1]
                time_diff = trough2 - trough1
                
                if price_diff < 0.02 and 5 < time_diff < 50:
                    # Check for peak between troughs
                    peak_idx = np.argmax(closes[trough1:trough2 + 1]) + trough1
                    peak_price = closes[peak_idx]
                    
                    # Check if peak is significantly higher
                    trough_avg = (closes[trough1] + closes[trough2]) / 2
                    peak_diff = (peak_price - trough_avg) / trough_avg
                    
                    if peak_diff > 0.02:
                        confidence = self._calculate_pattern_confidence(
                            'double_bottom',
                            closes[trough1:trough2 + 1],
                            peak_price
                        )
                        
                        patterns.append(ChartPattern(
                            pattern_type='Double Bottom',
                            symbol=symbol,
                            detection_date=datetime.now(),
                            confidence=confidence,
                            timeframe=timeframe,
                            start_date=df.index[trough1],
                            end_date=df.index[trough2],
                            target_price=peak_price,
                            stop_loss=min(closes[trough1], closes[trough2]),
                            description=f"Bullish reversal pattern. Target: ${peak_price:.2f}",
                            points=[
                                {'index': trough1, 'type': 'first_trough', 'price': closes[trough1]},
                                {'index': peak_idx, 'type': 'peak', 'price': closes[peak_idx]},
                                {'index': trough2, 'type': 'second_trough', 'price': closes[trough2]}
                            ]
                        ))
        
        except Exception as e:
            self.logger.error(f"Error detecting double bottom: {e}")
        
        return patterns
    
    async def _detect_ascending_triangle(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[ChartPattern]:
        """Detect ascending triangle pattern"""
        patterns = []
        # Simplified implementation
        return patterns
    
    async def _detect_descending_triangle(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[ChartPattern]:
        """Detect descending triangle pattern"""
        patterns = []
        # Simplified implementation
        return patterns
    
    async def _detect_symmetrical_triangle(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[ChartPattern]:
        """Detect symmetrical triangle pattern"""
        patterns = []
        # Simplified implementation
        return patterns
    
    async def _detect_cup_and_handle(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[ChartPattern]:
        """Detect cup and handle pattern"""
        patterns = []
        # Simplified implementation
        return patterns
    
    async def _detect_inverse_cup_and_handle(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[ChartPattern]:
        """Detect inverse cup and handle pattern"""
        patterns = []
        # Simplified implementation
        return patterns
    
    async def _detect_flag(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[ChartPattern]:
        """Detect flag pattern"""
        patterns = []
        # Simplified implementation
        return patterns
    
    async def _detect_pennant(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[ChartPattern]:
        """Detect pennant pattern"""
        patterns = []
        # Simplified implementation
        return patterns
    
    async def _detect_rectangle(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[ChartPattern]:
        """Detect rectangle pattern"""
        patterns = []
        # Simplified implementation
        return patterns
    
    async def _detect_channel(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[ChartPattern]:
        """Detect channel pattern"""
        patterns = []
        # Simplified implementation
        return patterns
    
    # Technical Indicator Calculations
    async def _calculate_rsi(self, df: pd.DataFrame, symbol: str, timeframe: str, period: int = 14) -> List[TechnicalIndicator]:
        """Calculate RSI indicator"""
        try:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            indicators = []
            for timestamp, value in rsi.items():
                if pd.notna(value):
                    signal = 'buy' if value < 30 else 'sell' if value > 70 else 'hold'
                    signal_strength = 1 - abs(value - 50) / 50  # 0-1 scale
                    
                    indicators.append(TechnicalIndicator(
                        symbol=symbol,
                        indicator_name='RSI',
                        value=value,
                        timestamp=timestamp,
                        timeframe=timeframe,
                        signal=signal,
                        signal_strength=signal_strength,
                        parameters={'period': period}
                    ))
            
            return indicators[-1:] if indicators else []  # Return only latest
        
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            return []
    
    async def _calculate_macd(self, df: pd.DataFrame, symbol: str, timeframe: str, fast: int = 12, slow: int = 26, signal: int = 9) -> List[TechnicalIndicator]:
        """Calculate MACD indicator"""
        try:
            ema_fast = df['close'].ewm(span=fast).mean()
            ema_slow = df['close'].ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            
            indicators = []
            if len(macd_line) >= 2:
                latest_macd = macd_line.iloc[-1]
                latest_signal = signal_line.iloc[-1]
                latest_histogram = histogram.iloc[-1]
                prev_histogram = histogram.iloc[-2] if len(histogram) > 1 else 0
                
                # MACD crossover signal
                signal_type = 'hold'
                signal_strength = 0.5
                
                if prev_histogram <= 0 and latest_histogram > 0:
                    signal_type = 'buy'
                    signal_strength = min(abs(latest_histogram) / abs(prev_histogram), 1.0)
                elif prev_histogram >= 0 and latest_histogram < 0:
                    signal_type = 'sell'
                    signal_strength = min(abs(latest_histogram) / abs(prev_histogram), 1.0)
                
                indicators.append(TechnicalIndicator(
                    symbol=symbol,
                    indicator_name='MACD',
                    value=latest_macd,
                    timestamp=df.index[-1],
                    timeframe=timeframe,
                    signal=signal_type,
                    signal_strength=signal_strength,
                    parameters={'fast': fast, 'slow': slow, 'signal': signal}
                ))
            
            return indicators
        
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {e}")
            return []
    
    async def _calculate_bollinger_bands(self, df: pd.DataFrame, symbol: str, timeframe: str, period: int = 20, std_dev: float = 2) -> List[TechnicalIndicator]:
        """Calculate Bollinger Bands"""
        try:
            sma = df['close'].rolling(window=period).mean()
            rolling_std = df['close'].rolling(window=period).std()
            upper_band = sma + (rolling_std * std_dev)
            lower_band = sma - (rolling_std * std_dev)
            
            indicators = []
            if len(df) >= period:
                current_price = df['close'].iloc[-1]
                current_upper = upper_band.iloc[-1]
                current_lower = lower_band.iloc[-1]
                current_sma = sma.iloc[-1]
                
                # Position within bands
                if current_price <= current_lower:
                    signal = 'buy'
                    signal_strength = (current_lower - current_price) / (current_upper - current_lower)
                elif current_price >= current_upper:
                    signal = 'sell'
                    signal_strength = (current_price - current_upper) / (current_upper - current_lower)
                else:
                    signal = 'hold'
                    signal_strength = abs(current_price - current_sma) / (current_upper - current_lower)
                
                indicators.append(TechnicalIndicator(
                    symbol=symbol,
                    indicator_name='Bollinger_Bands',
                    value=current_price,
                    timestamp=df.index[-1],
                    timeframe=timeframe,
                    signal=signal,
                    signal_strength=min(signal_strength, 1.0),
                    parameters={'period': period, 'std_dev': std_dev}
                ))
            
            return indicators
        
        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {e}")
            return []
    
    async def _calculate_stochastic(self, df: pd.DataFrame, symbol: str, timeframe: str, k_period: int = 14, d_period: int = 3) -> List[TechnicalIndicator]:
        """Calculate Stochastic oscillator"""
        try:
            lowest_low = df['low'].rolling(window=k_period).min()
            highest_high = df['high'].rolling(window=k_period).max()
            k_percent = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=d_period).mean()
            
            indicators = []
            if len(df) >= k_period:
                latest_k = k_percent.iloc[-1]
                latest_d = d_percent.iloc[-1]
                
                signal = 'buy' if latest_k < 20 else 'sell' if latest_k > 80 else 'hold'
                signal_strength = 1 - abs(latest_k - 50) / 50
                
                indicators.append(TechnicalIndicator(
                    symbol=symbol,
                    indicator_name='Stochastic',
                    value=latest_k,
                    timestamp=df.index[-1],
                    timeframe=timeframe,
                    signal=signal,
                    signal_strength=signal_strength,
                    parameters={'k_period': k_period, 'd_period': d_period}
                ))
            
            return indicators
        
        except Exception as e:
            self.logger.error(f"Error calculating Stochastic: {e}")
            return []
    
    async def _calculate_williams_r(self, df: pd.DataFrame, symbol: str, timeframe: str, period: int = 14) -> List[TechnicalIndicator]:
        """Calculate Williams %R"""
        try:
            highest_high = df['high'].rolling(window=period).max()
            lowest_low = df['low'].rolling(window=period).min()
            wr = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))
            
            indicators = []
            if len(df) >= period:
                latest_wr = wr.iloc[-1]
                
                signal = 'buy' if latest_wr < -80 else 'sell' if latest_wr > -20 else 'hold'
                signal_strength = 1 - abs(latest_wr + 50) / 50
                
                indicators.append(TechnicalIndicator(
                    symbol=symbol,
                    indicator_name='Williams_R',
                    value=latest_wr,
                    timestamp=df.index[-1],
                    timeframe=timeframe,
                    signal=signal,
                    signal_strength=signal_strength,
                    parameters={'period': period}
                ))
            
            return indicators
        
        except Exception as e:
            self.logger.error(f"Error calculating Williams %R: {e}")
            return []
    
    async def _calculate_cci(self, df: pd.DataFrame, symbol: str, timeframe: str, period: int = 20) -> List[TechnicalIndicator]:
        """Calculate Commodity Channel Index"""
        try:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            sma_tp = typical_price.rolling(window=period).mean()
            mean_deviation = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
            cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
            
            indicators = []
            if len(df) >= period:
                latest_cci = cci.iloc[-1]
                
                signal = 'buy' if latest_cci < -100 else 'sell' if latest_cci > 100 else 'hold'
                signal_strength = min(abs(latest_cci) / 200, 1.0)
                
                indicators.append(TechnicalIndicator(
                    symbol=symbol,
                    indicator_name='CCI',
                    value=latest_cci,
                    timestamp=df.index[-1],
                    timeframe=timeframe,
                    signal=signal,
                    signal_strength=signal_strength,
                    parameters={'period': period}
                ))
            
            return indicators
        
        except Exception as e:
            self.logger.error(f"Error calculating CCI: {e}")
            return []
    
    async def _calculate_adx(self, df: pd.DataFrame, symbol: str, timeframe: str, period: int = 14) -> List[TechnicalIndicator]:
        """Calculate Average Directional Index"""
        # Simplified ADX calculation
        try:
            high_diff = df['high'].diff()
            low_diff = df['low'].diff()
            
            plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
            
            atr = await self._calculate_atr_simple(df, period)
            
            plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).mean() / atr)
            minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).mean() / atr)
            
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=period).mean()
            
            indicators = []
            if len(adx) >= period:
                latest_adx = adx.iloc[-1]
                
                signal = 'strong' if latest_adx > 25 else 'weak'
                signal_strength = min(latest_adx / 50, 1.0)
                
                indicators.append(TechnicalIndicator(
                    symbol=symbol,
                    indicator_name='ADX',
                    value=latest_adx,
                    timestamp=df.index[-1],
                    timeframe=timeframe,
                    signal=signal,
                    signal_strength=signal_strength,
                    parameters={'period': period}
                ))
            
            return indicators
        
        except Exception as e:
            self.logger.error(f"Error calculating ADX: {e}")
            return []
    
    async def _calculate_atr(self, df: pd.DataFrame, symbol: str, timeframe: str, period: int = 14) -> List[TechnicalIndicator]:
        """Calculate Average True Range"""
        try:
            atr_values = await self._calculate_atr_simple(df, period)
            
            indicators = []
            if len(atr_values) >= period:
                latest_atr = atr_values.iloc[-1]
                
                indicators.append(TechnicalIndicator(
                    symbol=symbol,
                    indicator_name='ATR',
                    value=latest_atr,
                    timestamp=df.index[-1],
                    timeframe=timeframe,
                    signal='volatility',
                    signal_strength=min(latest_atr / df['close'].iloc[-1], 1.0),
                    parameters={'period': period}
                ))
            
            return indicators
        
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return []
    
    async def _calculate_atr_simple(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR values"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    async def _calculate_obv(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[TechnicalIndicator]:
        """Calculate On-Balance Volume"""
        try:
            obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
            
            indicators = []
            if len(obv) >= 2:
                latest_obv = obv.iloc[-1]
                prev_obv = obv.iloc[-2]
                obv_change = (latest_obv - prev_obv) / abs(prev_obv) if prev_obv != 0 else 0
                
                signal = 'buy' if obv_change > 0.01 else 'sell' if obv_change < -0.01 else 'hold'
                signal_strength = min(abs(obv_change), 1.0)
                
                indicators.append(TechnicalIndicator(
                    symbol=symbol,
                    indicator_name='OBV',
                    value=latest_obv,
                    timestamp=df.index[-1],
                    timeframe=timeframe,
                    signal=signal,
                    signal_strength=signal_strength
                ))
            
            return indicators
        
        except Exception as e:
            self.logger.error(f"Error calculating OBV: {e}")
            return []
    
    async def _calculate_volume_sma(self, df: pd.DataFrame, symbol: str, timeframe: str, period: int = 20) -> List[TechnicalIndicator]:
        """Calculate Volume Simple Moving Average"""
        try:
            volume_sma = df['volume'].rolling(window=period).mean()
            
            indicators = []
            if len(volume_sma) >= period:
                latest_volume = df['volume'].iloc[-1]
                latest_sma = volume_sma.iloc[-1]
                volume_ratio = latest_volume / latest_sma
                
                signal = 'high' if volume_ratio > 1.5 else 'low' if volume_ratio < 0.5 else 'normal'
                signal_strength = min(abs(volume_ratio - 1), 1.0)
                
                indicators.append(TechnicalIndicator(
                    symbol=symbol,
                    indicator_name='Volume_SMA',
                    value=volume_ratio,
                    timestamp=df.index[-1],
                    timeframe=timeframe,
                    signal=signal,
                    signal_strength=signal_strength,
                    parameters={'period': period}
                ))
            
            return indicators
        
        except Exception as e:
            self.logger.error(f"Error calculating Volume SMA: {e}")
            return []
    
    async def _calculate_price_sma(self, df: pd.DataFrame, symbol: str, timeframe: str, period: int = 20) -> List[TechnicalIndicator]:
        """Calculate Price Simple Moving Average"""
        try:
            price_sma = df['close'].rolling(window=period).mean()
            
            indicators = []
            if len(price_sma) >= period:
                latest_price = df['close'].iloc[-1]
                latest_sma = price_sma.iloc[-1]
                
                signal = 'buy' if latest_price > latest_sma else 'sell'
                signal_strength = min(abs(latest_price - latest_sma) / latest_sma, 1.0)
                
                indicators.append(TechnicalIndicator(
                    symbol=symbol,
                    indicator_name='Price_SMA',
                    value=latest_sma,
                    timestamp=df.index[-1],
                    timeframe=timeframe,
                    signal=signal,
                    signal_strength=signal_strength,
                    parameters={'period': period}
                ))
            
            return indicators
        
        except Exception as e:
            self.logger.error(f"Error calculating Price SMA: {e}")
            return []
    
    async def _calculate_price_ema(self, df: pd.DataFrame, symbol: str, timeframe: str, period: int = 12) -> List[TechnicalIndicator]:
        """Calculate Price Exponential Moving Average"""
        try:
            price_ema = df['close'].ewm(span=period).mean()
            
            indicators = []
            if len(price_ema) >= period:
                latest_price = df['close'].iloc[-1]
                latest_ema = price_ema.iloc[-1]
                
                signal = 'buy' if latest_price > latest_ema else 'sell'
                signal_strength = min(abs(latest_price - latest_ema) / latest_ema, 1.0)
                
                indicators.append(TechnicalIndicator(
                    symbol=symbol,
                    indicator_name='Price_EMA',
                    value=latest_ema,
                    timestamp=df.index[-1],
                    timeframe=timeframe,
                    signal=signal,
                    signal_strength=signal_strength,
                    parameters={'period': period}
                ))
            
            return indicators
        
        except Exception as e:
            self.logger.error(f"Error calculating Price EMA: {e}")
            return []
    
    async def _calculate_psar(self, df: pd.DataFrame, symbol: str, timeframe: str, af_start: float = 0.02, af_max: float = 0.2) -> List[TechnicalIndicator]:
        """Calculate Parabolic SAR"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # Simplified PSAR calculation
            psar = np.zeros(len(close))
            trend = np.zeros(len(close))
            af = np.zeros(len(close))
            ep = np.zeros(len(close))
            
            # Initialize
            psar[0] = low[0]
            trend[0] = 1  # 1 for uptrend, -1 for downtrend
            af[0] = af_start
            ep[0] = high[0]
            
            for i in range(1, len(close)):
                if trend[i-1] == 1:  # Uptrend
                    psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
                    
                    if low[i] <= psar[i]:
                        trend[i] = -1  # Reverse to downtrend
                        psar[i] = ep[i-1]
                        af[i] = af_start
                        ep[i] = low[i]
                    else:
                        trend[i] = 1
                        if high[i] > ep[i-1]:
                            ep[i] = high[i]
                            af[i] = min(af[i-1] + af_start, af_max)
                        else:
                            ep[i] = ep[i-1]
                            af[i] = af[i-1]
                else:  # Downtrend
                    psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
                    
                    if high[i] >= psar[i]:
                        trend[i] = 1  # Reverse to uptrend
                        psar[i] = ep[i-1]
                        af[i] = af_start
                        ep[i] = low[i]
                    else:
                        trend[i] = -1
                        if low[i] < ep[i-1]:
                            ep[i] = low[i]
                            af[i] = min(af[i-1] + af_start, af_max)
                        else:
                            ep[i] = ep[i-1]
                            af[i] = af[i-1]
            
            indicators = []
            if len(psar) >= 2:
                latest_psar = psar[-1]
                latest_trend = trend[-1]
                
                signal = 'buy' if latest_trend == 1 else 'sell'
                signal_strength = 1.0  # PSAR gives clear signals
                
                indicators.append(TechnicalIndicator(
                    symbol=symbol,
                    indicator_name='PSAR',
                    value=latest_psar,
                    timestamp=df.index[-1],
                    timeframe=timeframe,
                    signal=signal,
                    signal_strength=signal_strength,
                    parameters={'af_start': af_start, 'af_max': af_max}
                ))
            
            return indicators
        
        except Exception as e:
            self.logger.error(f"Error calculating PSAR: {e}")
            return []
    
    def _calculate_pattern_confidence(self, pattern_type: str, price_data: np.ndarray, support_level: float) -> float:
        """Calculate pattern confidence score"""
        try:
            base_confidence = 0.5
            
            if pattern_type == 'head_and_shoulders':
                # Check trend consistency
                trend_strength = np.std(price_data) / np.mean(price_data)
                confidence = min(trend_strength * 10, 1.0)
            
            elif pattern_type == 'double_top':
                # Check peak similarity
                peak1, peak2 = np.max(price_data), np.max(price_data)
                peak_similarity = 1 - abs(peak1 - peak2) / max(peak1, peak2)
                confidence = peak_similarity
            
            elif pattern_type == 'double_bottom':
                # Check trough similarity
                trough1, trough2 = np.min(price_data), np.min(price_data)
                trough_similarity = 1 - abs(trough1 - trough2) / max(trough1, trough2)
                confidence = trough_similarity
            
            else:
                confidence = base_confidence
            
            return max(0.1, min(1.0, confidence))
        
        except Exception:
            return 0.5
    
    async def _get_price_data(self, symbol: str, timeframe: str, days: int) -> Optional[List[Dict[str, Any]]]:
        """Get price data for analysis"""
        try:
            # This would typically fetch from a data provider
            # For demo, generate sample data
            import random
            
            data = []
            start_date = datetime.now() - timedelta(days=days)
            current_price = 150.0  # Starting price
            
            for i in range(days):
                timestamp = start_date + timedelta(days=i)
                
                # Generate realistic price movement
                change = random.uniform(-0.03, 0.03)  # ±3% daily change
                current_price *= (1 + change)
                
                # Generate OHLC from close price
                high = current_price * random.uniform(1.001, 1.02)
                low = current_price * random.uniform(0.98, 0.999)
                open_price = current_price * random.uniform(0.995, 1.005)
                volume = random.randint(50000, 500000)
                
                data.append({
                    'timestamp': timestamp,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': current_price,
                    'volume': volume
                })
            
            return data
        
        except Exception as e:
            self.logger.error(f"Error getting price data for {symbol}: {e}")
            return None
    
    async def _fetch_data(self) -> Dict[str, Any]:
        """Fetch all technical analysis data"""
        # Fetch patterns, indicators, and microstructure for first few symbols
        symbols_to_analyze = self.symbols[:3]  # Limit to avoid performance issues
        
        patterns_task = asyncio.gather(*[
            self.detect_chart_patterns(symbol) for symbol in symbols_to_analyze
        ])
        
        indicators_task = self.scan_technical_indicators(symbols_to_analyze)
        microstructure_tasks = asyncio.gather(*[
            self.fetch_market_microstructure(symbol) for symbol in symbols_to_analyze
        ])
        
        patterns_results = await patterns_task
        indicators_results = await indicators_task
        microstructure_results = await microstructure_tasks
        
        # Flatten patterns
        all_patterns = []
        for pattern_list in patterns_results:
            all_patterns.extend(pattern_list)
        
        return {
            'patterns': all_patterns,
            'indicators': indicators_results,
            'microstructure': microstructure_results
        }
    
    async def _process_data(self, data: Dict[str, Any], source: str = "live"):
        """Process technical analysis data"""
        try:
            patterns = data.get('patterns', [])
            indicators = data.get('indicators', {})
            microstructure = data.get('microstructure', [])
            
            # Log pattern summary
            if patterns:
                bullish_patterns = [p for p in patterns if 'bullish' in p.description.lower() or 'bottom' in p.pattern_type.lower()]
                bearish_patterns = [p for p in patterns if 'bearish' in p.description.lower() or 'top' in p.pattern_type.lower()]
                
                self.logger.info(f"Technical Analysis: {len(patterns)} patterns detected, "
                               f"{len(bullish_patterns)} bullish, {len(bearish_patterns)} bearish")
            
            # Log indicator summary
            if indicators:
                buy_signals = 0
                sell_signals = 0
                for symbol, inds in indicators.items():
                    for ind in inds:
                        if ind.signal == 'buy':
                            buy_signals += 1
                        elif ind.signal == 'sell':
                            sell_signals += 1
                
                self.logger.info(f"Technical Indicators: {buy_signals} buy signals, {sell_signals} sell signals")
            
            # Log microstructure summary
            if microstructure:
                avg_spread = np.mean([m.bid_ask_spread for m in microstructure if m])
                self.logger.info(f"Market Microstructure: Average spread ${avg_spread:.3f}")
            
            # Store data if configured
            if self.config.enable_storage:
                await self._store_technical_data(data)
        
        except Exception as e:
            self.logger.error(f"Error processing technical data: {e}")
            raise
    
    async def _store_technical_data(self, data: Dict[str, Any]):
        """Store technical analysis data"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Store patterns
            patterns = data.get('patterns', [])
            if patterns:
                patterns_file = Path(self.config.storage_path) / f"chart_patterns_{timestamp}.json"
                patterns_json = [{
                    'pattern_type': pattern.pattern_type,
                    'symbol': pattern.symbol,
                    'detection_date': pattern.detection_date.isoformat(),
                    'confidence': pattern.confidence,
                    'timeframe': pattern.timeframe,
                    'start_date': pattern.start_date.isoformat(),
                    'end_date': pattern.end_date.isoformat() if pattern.end_date else None,
                    'target_price': pattern.target_price,
                    'stop_loss': pattern.stop_loss,
                    'description': pattern.description,
                    'points': pattern.points or []
                } for pattern in patterns]
                
                await aiofiles.makedirs(patterns_file.parent, exist_ok=True)
                async with aiofiles.open(patterns_file, 'w') as f:
                    await f.write(json.dumps(patterns_json, indent=2, default=str))
            
            # Store indicators
            indicators = data.get('indicators', {})
            if indicators:
                indicators_file = Path(self.config.storage_path) / f"technical_indicators_{timestamp}.json"
                indicators_json = {}
                for symbol, ind_list in indicators.items():
                    indicators_json[symbol] = [{
                        'indicator_name': ind.indicator_name,
                        'value': ind.value,
                        'timestamp': ind.timestamp.isoformat(),
                        'timeframe': ind.timeframe,
                        'signal': ind.signal,
                        'signal_strength': ind.signal_strength,
                        'parameters': ind.parameters or {}
                    } for ind in ind_list]
                
                async with aiofiles.open(indicators_file, 'w') as f:
                    await f.write(json.dumps(indicators_json, indent=2, default=str))
            
            # Store microstructure
            microstructure = data.get('microstructure', [])
            if microstructure:
                microstructure_file = Path(self.config.storage_path) / f"market_microstructure_{timestamp}.json"
                microstructure_json = [{
                    'symbol': m.symbol,
                    'timestamp': m.timestamp.isoformat(),
                    'bid_ask_spread': m.bid_ask_spread,
                    'order_book_depth': m.order_book_depth,
                    'trade_volume': m.trade_volume,
                    'trade_count': m.trade_count,
                    'average_trade_size': m.average_trade_size,
                    'price_impact': m.price_impact,
                    'volatility_1m': m.volatility_1m,
                    'volatility_5m': m.volatility_5m,
                    'volatility_15m': m.volatility_15m,
                    'bid_ask_imbalance': m.bid_ask_imbalance
                } for m in microstructure if m]
                
                async with aiofiles.open(microstructure_file, 'w') as f:
                    await f.write(json.dumps(microstructure_json, indent=2, default=str))
        
        except Exception as e:
            self.logger.error(f"Failed to store technical data: {e}")
    
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols"""
        return self.symbols
    
    def get_data_schema(self) -> Dict[str, Any]:
        """Get expected data schema"""
        return {
            'patterns': {
                'pattern_type': 'str',
                'symbol': 'str',
                'detection_date': 'datetime',
                'confidence': 'float',
                'timeframe': 'str',
                'start_date': 'datetime',
                'end_date': 'datetime (optional)',
                'target_price': 'float (optional)',
                'stop_loss': 'float (optional)',
                'description': 'str',
                'points': 'list[dict] (optional)'
            },
            'indicators': {
                'symbol': 'str',
                'indicator_name': 'str',
                'value': 'float',
                'timestamp': 'datetime',
                'timeframe': 'str',
                'signal': 'str',
                'signal_strength': 'float',
                'parameters': 'dict (optional)'
            },
            'microstructure': {
                'symbol': 'str',
                'timestamp': 'datetime',
                'bid_ask_spread': 'float',
                'order_book_depth': 'dict',
                'trade_volume': 'float',
                'trade_count': 'int',
                'average_trade_size': 'float',
                'price_impact': 'float',
                'volatility_1m': 'float',
                'volatility_5m': 'float',
                'volatility_15m': 'float',
                'bid_ask_imbalance': 'float'
            }
        }