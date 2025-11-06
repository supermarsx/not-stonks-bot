"""
Cross-Venue Arbitrage Strategy
Implements arbitrage opportunities across different trading venues/brokers
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum
import asyncio

from loguru import logger

from .base import (
    BaseStrategy,
    StrategyConfig,
    StrategyType,
    TradingSignal,
    SignalType,
    RiskLevel,
    BaseTimeSeriesStrategy
)


class VenueType(Enum):
    """Trading venue types"""
    EXCHANGE = "exchange"
    BROKER = "broker"
    DARK_POOL = "dark_pool"
    ECN = "ecn"


class ArbitrageType(Enum):
    """Types of arbitrage opportunities"""
    PRICE_DIFFERENCE = "price_difference"
    FUTURES_SPOT = "futures_spot"
    CURRENCY_CROSS = "currency_cross"
    CONVERTIBLE = "convertible"
    MERGER = "merger"


@dataclass
class VenuePrice:
    """Price information from a trading venue"""
    venue_name: str
    venue_type: VenueType
    symbol: str
    bid: Decimal
    ask: Decimal
    bid_size: Decimal
    ask_size: Decimal
    timestamp: datetime
    latency_ms: int
    reliability_score: float  # 0.0 to 1.0


@dataclass
class ArbitrageOpportunity:
    """Arbitrage opportunity detected"""
    opportunity_id: str
    arbitrage_type: ArbitrageType
    symbol: str
    buy_venue: str
    sell_venue: str
    buy_price: Decimal
    sell_price: Decimal
    spread: Decimal
    spread_pct: float
    quantity: Decimal
    estimated_profit: Decimal
    execution_time_estimate: float  # seconds
    risk_score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(seconds=30))


class CrossVenueArbitrageStrategy(BaseTimeSeriesStrategy):
    """
    Cross-Venue Arbitrage Strategy
    
    Identifies and executes arbitrage opportunities across different venues:
    1. Price difference arbitrage between venues
    2. Latency arbitrage for fast-moving markets
    3. Order book depth analysis
    4. Transaction cost optimization
    5. Risk-adjusted execution
    
    Parameters:
    - min_spread_threshold: Minimum spread to consider (default: 0.05% or 0.0005)
    - max_execution_time: Maximum time for arbitrage execution (default: 5 seconds)
    - min_liquidity: Minimum order book liquidity (default: $10,000)
    - max_slippage: Maximum acceptable slippage (default: 0.01% or 0.0001)
    - venue_reliability_threshold: Minimum venue reliability (default: 0.8)
    - risk_tolerance: Risk tolerance level (default: 0.5)
    - position_size_multiplier: Position size adjustment (default: 1.0)
    """
    
    def __init__(self, config: StrategyConfig):
        # Validate required parameters
        required_params = ['min_spread_threshold', 'max_execution_time']
        for param in required_params:
            if param not in config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        super().__init__(config, timeframe="1s")  # High frequency for arbitrage
        
        # Extract parameters
        self.min_spread_threshold = float(config.parameters.get('min_spread_threshold', 0.0005))  # 0.05%
        self.max_execution_time = float(config.parameters.get('max_execution_time', 5.0))  # seconds
        self.min_liquidity = Decimal(str(config.parameters.get('min_liquidity', 10000)))  # $10k
        self.max_slippage = float(config.parameters.get('max_slippage', 0.0001))  # 0.01%
        self.venue_reliability_threshold = float(config.parameters.get('venue_reliability_threshold', 0.8))
        self.risk_tolerance = float(config.parameters.get('risk_tolerance', 0.5))
        self.position_size_multiplier = float(config.parameters.get('position_size_multiplier', 1.0))
        
        # Venue and price tracking
        self.venue_prices: Dict[str, Dict[str, VenuePrice]] = {}  # venue -> symbol -> price
        self.arbitrage_opportunities: List[ArbitrageOpportunity] = []
        self.active_venues: Dict[str, Dict[str, Any]] = {}
        
        # Performance metrics
        self.total_arbitrages_attempted = 0
        self.successful_arbitrages = 0
        self.total_profit = Decimal('0')
        self.average_spread_captured = Decimal('0')
        
        # Risk management
        self.max_position_per_opportunity = self.config.max_position_size * Decimal('0.2')  # 20% of max
        self.max_concurrent_opportunities = 10
        
        logger.info(f"Cross-Venue Arbitrage Strategy initialized with threshold: {self.min_spread_threshold:.4f}")
    
    async def generate_signals(self) -> List[TradingSignal]:
        """Generate arbitrage trading signals"""
        signals = []
        
        try:
            # Update venue prices
            await self._update_venue_prices()
            
            # Detect arbitrage opportunities
            await self._detect_arbitrage_opportunities()
            
            # Generate signals for best opportunities
            for opportunity in self.arbitrage_opportunities:
                if self._should_execute_opportunity(opportunity):
                    signal = await self._create_arbitrage_signal(opportunity)
                    if signal:
                        signals.append(signal)
            
            # Clean up expired opportunities
            self._cleanup_expired_opportunities()
            
            logger.debug(f"Generated {len(signals)} arbitrage signals from {len(self.arbitrage_opportunities)} opportunities")
            
        except Exception as e:
            logger.error(f"Error generating arbitrage signals: {e}")
        
        return signals
    
    async def _update_venue_prices(self):
        """Update price information from all registered venues"""
        try:
            # This would typically connect to real-time data feeds from different venues
            # For now, we'll simulate price updates
            
            for venue_name in self.active_venues:
                venue_info = self.active_venues[venue_name]
                
                for symbol in self.config.symbols:
                    # Simulate getting price from venue
                    venue_price = await self._get_venue_price(venue_name, symbol, venue_info)
                    
                    if venue_price:
                        if venue_name not in self.venue_prices:
                            self.venue_prices[venue_name] = {}
                        
                        self.venue_prices[venue_name][symbol] = venue_price
                        
        except Exception as e:
            logger.error(f"Error updating venue prices: {e}")
    
    async def _get_venue_price(
        self, 
        venue_name: str, 
        symbol: str, 
        venue_info: Dict[str, Any]
    ) -> Optional[VenuePrice]:
        """Get price from a specific venue"""
        try:
            # This would interface with real venue APIs
            # For simulation, we'll create realistic price differences
            
            import random
            
            # Base price (could be from a reference venue)
            base_price = Decimal('100.0') + Decimal(str(random.gauss(0, 1)))
            
            # Add venue-specific spread and latency
            venue_spread = venue_info.get('spread_bps', 1.0) / 10000  # basis points to decimal
            venue_latency = venue_info.get('latency_ms', 10)
            
            # Calculate bid/ask with venue characteristics
            mid_price = base_price
            half_spread = venue_spread / 2
            
            # Add some price variation to create arbitrage opportunities
            price_noise = Decimal(str(random.gauss(0, 0.0001)))  # 1 basis point noise
            
            bid = mid_price - half_spread + price_noise
            ask = mid_price + half_spread + price_noise
            
            # Simulate order book sizes
            bid_size = Decimal(str(random.uniform(100, 10000)))
            ask_size = Decimal(str(random.uniform(100, 10000)))
            
            return VenuePrice(
                venue_name=venue_name,
                venue_type=VenueType(venue_info.get('venue_type', 'broker')),
                symbol=symbol,
                bid=bid,
                ask=ask,
                bid_size=bid_size,
                ask_size=ask_size,
                timestamp=datetime.utcnow(),
                latency_ms=venue_latency,
                reliability_score=venue_info.get('reliability_score', 0.9)
            )
            
        except Exception as e:
            logger.error(f"Error getting venue price for {venue_name}:{symbol}: {e}")
            return None
    
    async def _detect_arbitrage_opportunities(self):
        """Detect arbitrage opportunities across venues"""
        try:
            opportunities = []
            
            # Compare prices across all venue pairs for each symbol
            for symbol in self.config.symbols:
                symbol_opportunities = await self._find_symbol_opportunities(symbol)
                opportunities.extend(symbol_opportunities)
            
            # Sort by profitability (highest spread first)
            opportunities.sort(key=lambda x: x.spread_pct, reverse=True)
            
            # Keep only the best opportunities
            self.arbitrage_opportunities = opportunities[:self.max_concurrent_opportunities]
            
        except Exception as e:
            logger.error(f"Error detecting arbitrage opportunities: {e}")
    
    async def _find_symbol_opportunities(self, symbol: str) -> List[ArbitrageOpportunity]:
        """Find arbitrage opportunities for a specific symbol"""
        opportunities = []
        
        try:
            # Get all venue prices for this symbol
            venue_prices = {}
            for venue_name, venue_data in self.venue_prices.items():
                if symbol in venue_data:
                    venue_prices[venue_name] = venue_data[symbol]
            
            if len(venue_prices) < 2:
                return opportunities
            
            # Compare all venue pairs
            venue_names = list(venue_prices.keys())
            
            for i, buy_venue in enumerate(venue_names):
                for j, sell_venue in enumerate(venue_names):
                    if i >= j:  # Avoid duplicate comparisons
                        continue
                    
                    buy_price_info = venue_prices[buy_venue]
                    sell_price_info = venue_prices[sell_venue]
                    
                    # Try both directions
                    opportunity1 = self._check_arbitrage_direction(
                        symbol, buy_venue, sell_venue, buy_price_info, sell_price_info
                    )
                    if opportunity1:
                        opportunities.append(opportunity1)
                    
                    opportunity2 = self._check_arbitrage_direction(
                        symbol, sell_venue, buy_venue, sell_price_info, buy_price_info
                    )
                    if opportunity2:
                        opportunities.append(opportunity2)
            
        except Exception as e:
            logger.error(f"Error finding opportunities for {symbol}: {e}")
        
        return opportunities
    
    def _check_arbitrage_direction(
        self,
        symbol: str,
        buy_venue: str,
        sell_venue: str,
        buy_price_info: VenuePrice,
        sell_price_info: VenuePrice
    ) -> Optional[ArbitrageOpportunity]:
        """Check arbitrage opportunity in one direction"""
        try:
            # Calculate effective buy and sell prices
            # Buy from venue with lower ask, sell to venue with higher bid
            
            buy_price = buy_price_info.ask
            sell_price = sell_price_info.bid
            
            # Calculate spread
            spread = sell_price - buy_price
            spread_pct = float(spread / buy_price) * 100
            
            # Check if spread meets minimum threshold
            if spread_pct < self.min_spread_threshold * 100:
                return None
            
            # Check liquidity requirements
            max_buy_quantity = min(buy_price_info.ask_size, self.max_position_per_opportunity / buy_price)
            max_sell_quantity = min(sell_price_info.bid_size, self.max_position_per_opportunity / sell_price)
            available_quantity = min(max_buy_quantity, max_sell_quantity)
            
            if available_quantity * buy_price < self.min_liquidity:
                return None
            
            # Calculate execution time estimate (latency + processing)
            total_latency = buy_price_info.latency_ms + sell_price_info.latency_ms
            processing_time = 1000  # 1 second processing time
            execution_time_estimate = (total_latency + processing_time) / 1000  # Convert to seconds
            
            if execution_time_estimate > self.max_execution_time:
                return None
            
            # Check venue reliability
            min_reliability = min(buy_price_info.reliability_score, sell_price_info.reliability_score)
            if min_reliability < self.venue_reliability_threshold:
                return None
            
            # Calculate risk score (higher is riskier)
            risk_factors = []
            
            # Latency risk
            latency_risk = min(total_latency / 5000, 1.0)  # Normalize to 5 seconds
            risk_factors.append(latency_risk)
            
            # Liquidity risk (inverse of available liquidity)
            liquidity_risk = max(0, 1 - float(available_quantity * buy_price / self.min_liquidity))
            risk_factors.append(liquidity_risk)
            
            # Price volatility risk (simplified - would use actual volatility)
            volatility_risk = 0.1  # Assume low volatility for this example
            risk_factors.append(volatility_risk)
            
            risk_score = sum(risk_factors) / len(risk_factors)
            
            # Calculate confidence based on multiple factors
            confidence_factors = []
            confidence_factors.append(spread_pct / (self.min_spread_threshold * 100 * 3))  # Spread strength
            confidence_factors.append(min_reliability)
            confidence_factors.append(1 - risk_score)
            confidence_factors.append(1 - (execution_time_estimate / self.max_execution_time))
            
            confidence = sum(confidence_factors) / len(confidence_factors)
            confidence = min(1.0, max(0.0, confidence))
            
            # Calculate estimated profit
            estimated_profit = (sell_price - buy_price) * available_quantity
            
            # Create opportunity
            opportunity = ArbitrageOpportunity(
                opportunity_id=f"ARB_{symbol}_{buy_venue}_{sell_venue}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                arbitrage_type=ArbitrageType.PRICE_DIFFERENCE,
                symbol=symbol,
                buy_venue=buy_venue,
                sell_venue=sell_venue,
                buy_price=buy_price,
                sell_price=sell_price,
                spread=spread,
                spread_pct=spread_pct,
                quantity=available_quantity,
                estimated_profit=estimated_profit,
                execution_time_estimate=execution_time_estimate,
                risk_score=risk_score,
                confidence=confidence,
                expires_at=datetime.utcnow() + timedelta(seconds=30)  # 30 second expiry
            )
            
            return opportunity
            
        except Exception as e:
            logger.error(f"Error checking arbitrage direction: {e}")
            return None
    
    def _should_execute_opportunity(self, opportunity: ArbitrageOpportunity) -> bool:
        """Determine if an arbitrage opportunity should be executed"""
        try:
            # Basic filters
            if opportunity.confidence < 0.5:
                return False
            
            if opportunity.risk_score > self.risk_tolerance:
                return False
            
            if opportunity.estimated_profit <= Decimal('0'):
                return False
            
            # Check if we already have too many concurrent opportunities
            if len(self.active_opportunities) >= self.max_concurrent_opportunities:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error evaluating opportunity: {e}")
            return False
    
    async def _create_arbitrage_signal(self, opportunity: ArbitrageOpportunity) -> Optional[TradingSignal]:
        """Create trading signal for arbitrage opportunity"""
        try:
            # Calculate position size based on opportunity characteristics
            base_position = self._calculate_arbitrage_position_size(opportunity)
            
            signal = TradingSignal(
                signal_id=opportunity.opportunity_id,
                strategy_id=self.config.strategy_id,
                symbol=opportunity.symbol,
                signal_type=SignalType.BUY,  # We'll handle sell in metadata
                confidence=opportunity.confidence,
                strength=min(opportunity.confidence * opportunity.spread_pct / (self.min_spread_threshold * 100), 1.0),
                price=opportunity.buy_price,
                quantity=base_position,
                metadata={
                    'strategy_type': 'arbitrage',
                    'arbitrage_type': opportunity.arbitrage_type.value,
                    'buy_venue': opportunity.buy_venue,
                    'sell_venue': opportunity.sell_venue,
                    'buy_price': float(opportunity.buy_price),
                    'sell_price': float(opportunity.sell_price),
                    'sell_quantity': float(base_position * 0.95),  # Slightly less for safety
                    'spread_pct': opportunity.spread_pct,
                    'estimated_profit': float(opportunity.estimated_profit),
                    'execution_time_estimate': opportunity.execution_time_estimate,
                    'risk_score': opportunity.risk_score,
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error creating arbitrage signal: {e}")
            return None
    
    def _calculate_arbitrage_position_size(self, opportunity: ArbitrageOpportunity) -> Decimal:
        """Calculate position size for arbitrage opportunity"""
        try:
            # Base size
            base_size = self.max_position_per_opportunity
            
            # Adjust based on confidence and spread
            confidence_adjustment = Decimal(str(opportunity.confidence))
            spread_adjustment = Decimal(str(min(opportunity.spread_pct / (self.min_spread_threshold * 100), 2.0)))
            
            # Risk adjustment (inverse relationship)
            risk_adjustment = Decimal(str(1.0 - opportunity.risk_score))
            
            # Calculate final size
            adjusted_size = base_size * confidence_adjustment * spread_adjustment * risk_adjustment
            adjusted_size *= Decimal(str(self.position_size_multiplier))
            
            # Apply venue liquidity constraints
            max_size_by_liquidity = opportunity.quantity * opportunity.buy_price
            
            final_size = min(adjusted_size, max_size_by_liquidity)
            
            return final_size
            
        except Exception as e:
            logger.error(f"Error calculating arbitrage position size: {e}")
            return self.max_position_per_opportunity * Decimal('0.5')
    
    def _cleanup_expired_opportunities(self):
        """Remove expired arbitrage opportunities"""
        current_time = datetime.utcnow()
        self.arbitrage_opportunities = [
            opp for opp in self.arbitrage_opportunities
            if opp.expires_at > current_time
        ]
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        """Validate arbitrage signal"""
        try:
            # Basic validation
            if not signal.symbol or signal.quantity <= 0:
                return False
            
            # Check if signal is not too old (arbitrage is very time-sensitive)
            if signal.expires_at and datetime.utcnow() > signal.expires_at:
                return False
            
            # Check signal strength (arbitrage needs strong signals)
            if signal.strength < 0.7:
                return False
            
            # Check metadata integrity
            metadata = signal.metadata
            required_fields = ['buy_venue', 'sell_venue', 'buy_price', 'sell_price']
            
            for field in required_fields:
                if field not in metadata:
                    logger.warning(f"Missing metadata field: {field}")
                    return False
            
            # Additional arbitrage-specific checks
            buy_price = Decimal(str(metadata['buy_price']))
            sell_price = Decimal(str(metadata['sell_price']))
            
            if sell_price <= buy_price:
                return False  # No arbitrage opportunity
            
            spread_pct = float((sell_price - buy_price) / buy_price)
            if spread_pct < self.min_spread_threshold:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating arbitrage signal: {e}")
            return False
    
    def register_venue(self, venue_name: str, venue_config: Dict[str, Any]):
        """Register a trading venue"""
        self.active_venues[venue_name] = {
            'venue_type': venue_config.get('venue_type', 'broker'),
            'spread_bps': venue_config.get('spread_bps', 1.0),
            'latency_ms': venue_config.get('latency_ms', 10),
            'reliability_score': venue_config.get('reliability_score', 0.9),
            'supported_symbols': venue_config.get('supported_symbols', [])
        }
        
        logger.info(f"Registered venue: {venue_name}")
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get detailed strategy information"""
        return {
            'strategy_name': 'Cross-Venue Arbitrage Strategy',
            'description': 'Multi-venue arbitrage strategy with real-time opportunity detection',
            'parameters': {
                'min_spread_threshold': self.min_spread_threshold,
                'max_execution_time': self.max_execution_time,
                'min_liquidity': float(self.min_liquidity),
                'max_slippage': self.max_slippage,
                'venue_reliability_threshold': self.venue_reliability_threshold,
                'risk_tolerance': self.risk_tolerance,
                'position_size_multiplier': self.position_size_multiplier
            },
            'venues': {
                'active_venues': list(self.active_venues.keys()),
                'venue_count': len(self.active_venues)
            },
            'opportunities': {
                'current_opportunities': len(self.arbitrage_opportunities),
                'max_concurrent': self.max_concurrent_opportunities
            },
            'performance': {
                'total_attempted': self.total_arbitrages_attempted,
                'successful': self.successful_arbitrages,
                'success_rate': self.successful_arbitrages / max(self.total_arbitrages_attempted, 1),
                'total_profit': float(self.total_profit)
            },
            'timeframe': self.timeframe,
            'risk_level': RiskLevel.HIGH.value,  # Arbitrage is high risk due to execution speed
            'position_sizing': 'Opportunity-based with risk and confidence adjustment',
            'typical_hold_time': 'Seconds to minutes',
            'execution_requirements': 'High-frequency, low-latency infrastructure'
        }


# Factory function to create arbitrage strategy
def create_arbitrage_strategy(
    strategy_id: str,
    symbols: List[str],
    venues: Dict[str, Dict[str, Any]],
    min_spread_threshold: float = 0.0005,
    **kwargs
) -> CrossVenueArbitrageStrategy:
    """Factory function to create cross-venue arbitrage strategy"""
    
    config = StrategyConfig(
        strategy_id=strategy_id,
        strategy_type=StrategyType.ARBITRAGE,
        name="Cross-Venue Arbitrage Strategy",
        description="Multi-venue arbitrage strategy with real-time opportunity detection and execution",
        parameters={
            'min_spread_threshold': min_spread_threshold,
            'max_execution_time': kwargs.get('max_execution_time', 5.0),
            'min_liquidity': kwargs.get('min_liquidity', '10000'),
            'max_slippage': kwargs.get('max_slippage', 0.0001),
            'venue_reliability_threshold': kwargs.get('venue_reliability_threshold', 0.8),
            'risk_tolerance': kwargs.get('risk_tolerance', 0.5),
            'position_size_multiplier': kwargs.get('position_size_multiplier', 1.0)
        },
        risk_level=RiskLevel.HIGH,
        symbols=symbols,
        max_position_size=Decimal(kwargs.get('max_position_size', '50000')),
        max_daily_loss=Decimal(kwargs.get('max_daily_loss', '10000'))
    )
    
    strategy = CrossVenueArbitrageStrategy(config)
    
    # Register venues
    for venue_name, venue_config in venues.items():
        strategy.register_venue(venue_name, venue_config)
    
    return strategy


# Example usage and testing
if __name__ == "__main__":
    async def test_arbitrage_strategy():
        # Create strategy with sample venues
        strategy = create_arbitrage_strategy(
            strategy_id="arb_001",
            symbols=['AAPL', 'GOOGL'],
            venues={
                'venue_a': {
                    'venue_type': 'broker',
                    'spread_bps': 1.0,
                    'latency_ms': 5,
                    'reliability_score': 0.95
                },
                'venue_b': {
                    'venue_type': 'exchange',
                    'spread_bps': 0.5,
                    'latency_ms': 2,
                    'reliability_score': 0.98
                }
            },
            min_spread_threshold=0.0005
        )
        
        # Mock context for testing
        class MockContext:
            async def get_market_data(self, symbol, timeframe):
                # Return empty for arbitrage (uses direct venue data)
                return []
            
            async def get_current_price(self, symbol):
                return Decimal('100.0')
        
        strategy.set_context(MockContext())
        
        # Generate signals
        signals = await strategy.generate_signals()
        
        print(f"Generated {len(signals)} arbitrage signals:")
        for signal in signals:
            metadata = signal.metadata
            print(f"  {signal.symbol}: Buy @ {metadata['buy_price']} from {metadata['buy_venue']}, "
                  f"Sell @ {metadata['sell_price']} to {metadata['sell_venue']} "
                  f"(Spread: {metadata['spread_pct']:.3f}%)")
        
        # Get strategy info
        info = strategy.get_strategy_info()
        print("\nStrategy Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    # Run test
    import asyncio
    asyncio.run(test_arbitrage_strategy())