"""
Autonomous Decision Making Engine

AI-driven market analysis, opportunity detection, and autonomous signal generation
with self-adjusting strategy parameters and market regime detection.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from decimal import Decimal

from loguru import logger

from .config import AutomatedTradingConfig
from .market_hours import MarketHoursManager


class MarketRegime(Enum):
    """Market regime types"""
    BULL_TRENDING = "bull_trending"
    BEAR_TRENDING = "bear_trending"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"
    UNKNOWN = "unknown"


class OpportunityType(Enum):
    """Opportunity types"""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    MOMENTUM = "momentum"
    ARBITRAGE = "arbitrage"
    VOLATILITY = "volatility"
    NEWS_DRIVEN = "news_driven"


@dataclass
class MarketSignal:
    """Market signal data"""
    signal_id: str
    timestamp: datetime
    symbol: str
    signal_type: str  # buy, sell, hold
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    price: Decimal
    volume: Optional[float] = None
    timeframe: str = "1m"
    indicators: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradingOpportunity:
    """Trading opportunity"""
    opportunity_id: str
    timestamp: datetime
    type: OpportunityType
    symbol: str
    action: str  # buy, sell
    confidence: float
    strength: float
    expected_return: float
    risk_score: float
    time_horizon: timedelta
    position_size: Decimal
    entry_price: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    reasoning: str = ""
    indicators_used: List[str] = field(default_factory=list)
    strategy_id: str = "autonomous"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketAnalysis:
    """Comprehensive market analysis"""
    timestamp: datetime
    market_regime: MarketRegime
    overall_sentiment: float  # -1.0 (bearish) to 1.0 (bullish)
    volatility_level: float  # 0.0 to 1.0
    liquidity_score: float  # 0.0 to 1.0
    trend_strength: float  # 0.0 to 1.0
    key_levels: Dict[str, float]  # support, resistance levels
    sector_performance: Dict[str, float]
    risk_factors: List[str]
    conditions_acceptable: bool
    recommended_strategies: List[str]
    analysis_confidence: float
    symbols_analyzed: List[str]
    market_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegimeCharacteristics:
    """Market regime characteristics"""
    regime: MarketRegime
    volatility_threshold: float
    trend_strength_threshold: float
    volume_conditions: str
    recommended_strategies: List[OpportunityType]
    risk_multiplier: float
    typical_duration: timedelta
    success_probability: float


class AutonomousDecisionEngine:
    """
    Autonomous Decision Making Engine
    
    Features:
    - AI-driven market analysis
    - Real-time opportunity detection
    - Autonomous signal generation
    - Self-adjusting strategy parameters
    - Market regime detection and adaptation
    - Dynamic confidence scoring
    """
    
    def __init__(self, config: AutomatedTradingConfig):
        self.config = config
        
        # Market data and analysis
        self.market_data_cache: Dict[str, List[Dict]] = {}
        self.market_signals: Dict[str, MarketSignal] = {}
        self.trading_opportunities: List[TradingOpportunity] = []
        
        # Market regime detection
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_history: List[Tuple[datetime, MarketRegime]] = []
        self.regime_characteristics = self._initialize_regime_characteristics()
        
        # Analysis models and parameters
        self.volatility_lookback = 20
        self.trend_lookback = 50
        self.regime_detection_interval = 300  # 5 minutes
        self.last_regime_detection = datetime.utcnow()
        
        # Performance tracking
        self.signal_accuracy_history: List[float] = []
        self.opportunity_success_rate: Dict[str, float] = {}
        
        # Market hours manager
        self.market_hours = MarketHoursManager()
        
        logger.info("Autonomous Decision Engine initialized")
    
    async def initialize(self):
        """Initialize the decision engine"""
        try:
            # Start market data collection
            self.data_collection_task = asyncio.create_task(self._market_data_collection_loop())
            
            # Start regime detection
            self.regime_detection_task = asyncio.create_task(self._regime_detection_loop())
            
            # Start opportunity analysis
            self.opportunity_analysis_task = asyncio.create_task(self._opportunity_analysis_loop())
            
            logger.info("✅ Autonomous Decision Engine components started")
            
        except Exception as e:
            logger.error(f"Error initializing decision engine: {e}")
            raise
    
    async def stop(self):
        """Stop the decision engine"""
        logger.info("🛑 Stopping Autonomous Decision Engine...")
        
        # Cancel all tasks
        for task_name in ['data_collection_task', 'regime_detection_task', 'opportunity_analysis_task']:
            task = getattr(self, task_name, None)
            if task and not task.done():
                task.cancel()
        
        logger.success("✅ Autonomous Decision Engine stopped")
    
    async def analyze_market_conditions(self) -> MarketAnalysis:
        """Perform comprehensive market analysis"""
        try:
            current_time = datetime.utcnow()
            
            # Get current market regime
            regime = await self._detect_current_market_regime()
            
            # Analyze overall market sentiment
            sentiment = await self._analyze_market_sentiment()
            
            # Calculate volatility and liquidity metrics
            volatility = await self._calculate_volatility_metrics()
            liquidity = await self._calculate_liquidity_score()
            
            # Assess trend strength
            trend_strength = await self._calculate_trend_strength()
            
            # Identify key support/resistance levels
            key_levels = await self._identify_key_levels()
            
            # Analyze sector performance
            sector_performance = await self._analyze_sector_performance()
            
            # Identify risk factors
            risk_factors = await self._identify_risk_factors()
            
            # Determine if conditions are acceptable for trading
            conditions_acceptable = self._evaluate_trading_conditions(regime, sentiment, volatility, liquidity)
            
            # Recommend strategies based on current regime
            recommended_strategies = await self._recommend_strategies(regime, sentiment, volatility)
            
            return MarketAnalysis(
                timestamp=current_time,
                market_regime=regime,
                overall_sentiment=sentiment,
                volatility_level=volatility,
                liquidity_score=liquidity,
                trend_strength=trend_strength,
                key_levels=key_levels,
                sector_performance=sector_performance,
                risk_factors=risk_factors,
                conditions_acceptable=conditions_acceptable,
                recommended_strategies=recommended_strategies,
                analysis_confidence=0.85,  # Base confidence
                symbols_analyzed=list(self.market_data_cache.keys())
            )
            
        except Exception as e:
            logger.error(f"Error in market analysis: {e}")
            # Return default analysis
            return MarketAnalysis(
                timestamp=datetime.utcnow(),
                market_regime=MarketRegime.UNKNOWN,
                overall_sentiment=0.0,
                volatility_level=0.5,
                liquidity_score=0.5,
                trend_strength=0.5,
                key_levels={},
                sector_performance={},
                risk_factors=["Analysis error"],
                conditions_acceptable=False,
                recommended_strategies=[],
                analysis_confidence=0.1,
                symbols_analyzed=[]
            )
    
    async def detect_opportunities(self) -> List[TradingOpportunity]:
        """Detect trading opportunities based on current market conditions"""
        try:
            current_time = datetime.utcnow()
            
            # Get current market analysis
            market_analysis = await self.analyze_market_conditions()
            
            # Clear expired opportunities
            self.trading_opportunities = [
                opp for opp in self.trading_opportunities
                if opp.timestamp > current_time - timedelta(minutes=30)
            ]
            
            # Detect opportunities based on current regime
            new_opportunities = []
            
            if market_analysis.conditions_acceptable:
                # Trend following opportunities
                if market_analysis.market_regime in [MarketRegime.BULL_TRENDING, MarketRegime.BEAR_TRENDING]:
                    opportunities = await self._detect_trend_opportunities(market_analysis)
                    new_opportunities.extend(opportunities)
                
                # Mean reversion opportunities
                if market_analysis.market_regime == MarketRegime.SIDEWAYS:
                    opportunities = await self._detect_mean_reversion_opportunities(market_analysis)
                    new_opportunities.extend(opportunities)
                
                # Volatility opportunities
                if market_analysis.market_regime == MarketRegime.HIGH_VOLATILITY:
                    opportunities = await self._detect_volatility_opportunities(market_analysis)
                    new_opportunities.extend(opportunities)
                
                # Breakout opportunities
                opportunities = await self._detect_breakout_opportunities(market_analysis)
                new_opportunities.extend(opportunities)
            
            # Add new opportunities
            self.trading_opportunities.extend(new_opportunities)
            
            # Sort by confidence and expected return
            self.trading_opportunities.sort(
                key=lambda x: (x.confidence * x.expected_return), reverse=True
            )
            
            # Limit to top opportunities
            max_opportunities = self.config.max_simultaneous_opportunities
            self.trading_opportunities = self.trading_opportunities[:max_opportunities]
            
            logger.debug(f"Detected {len(new_opportunities)} new opportunities "
                        f"(total active: {len(self.trading_opportunities)})")
            
            return self.trading_opportunities.copy()
            
        except Exception as e:
            logger.error(f"Error detecting opportunities: {e}")
            return []
    
    async def _detect_trend_opportunities(self, analysis: MarketAnalysis) -> List[TradingOpportunity]:
        """Detect trend-following opportunities"""
        opportunities = []
        
        try:
            # Analyze trending symbols
            for symbol in self.market_data_cache.keys():
                if len(self.market_data_cache[symbol]) < 20:
                    continue
                
                # Get recent price data
                recent_data = self.market_data_cache[symbol][-20:]
                
                # Calculate trend indicators
                sma_20 = np.mean([float(d['close']) for d in recent_data[-20:]])
                sma_50 = np.mean([float(d['close']) for d in recent_data[-50:]]) if len(recent_data) >= 50 else sma_20
                
                current_price = float(recent_data[-1]['close'])
                
                # Detect trend
                if sma_20 > sma_50 * 1.02:  # Strong uptrend
                    action = "buy"
                    confidence = min(0.9, (sma_20 - sma_50) / sma_50 * 10)
                elif sma_20 < sma_50 * 0.98:  # Strong downtrend
                    action = "sell"
                    confidence = min(0.9, (sma_50 - sma_20) / sma_50 * 10)
                else:
                    continue  # No clear trend
                
                # Create opportunity
                opportunity = TradingOpportunity(
                    opportunity_id=f"trend_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.utcnow(),
                    type=OpportunityType.TREND_FOLLOWING,
                    symbol=symbol,
                    action=action,
                    confidence=max(0.3, confidence),
                    strength=confidence,
                    expected_return=confidence * 0.02,  # 2% expected return
                    risk_score=0.3,  # Moderate risk for trend following
                    time_horizon=timedelta(hours=2),
                    position_size=Decimal(str(self._calculate_position_size(confidence, 0.3))),
                    entry_price=Decimal(str(current_price)),
                    reasoning=f"Trend following: {sma_20:.2f} vs {sma_50:.2f}",
                    indicators_used=["SMA20", "SMA50"],
                    metadata={
                        "sma_20": sma_20,
                        "sma_50": sma_50,
                        "current_price": current_price
                    }
                )
                
                opportunities.append(opportunity)
            
        except Exception as e:
            logger.error(f"Error detecting trend opportunities: {e}")
        
        return opportunities
    
    async def _detect_mean_reversion_opportunities(self, analysis: MarketAnalysis) -> List[TradingOpportunity]:
        """Detect mean reversion opportunities"""
        opportunities = []
        
        try:
            # Mean reversion logic would go here
            # For now, create a basic example
            
            for symbol in self.market_data_cache.keys():
                if len(self.market_data_cache[symbol]) < 10:
                    continue
                
                recent_data = self.market_data_cache[symbol][-10:]
                prices = [float(d['close']) for d in recent_data]
                
                current_price = prices[-1]
                avg_price = np.mean(prices)
                std_dev = np.std(prices)
                
                # Detect oversold/overbought conditions
                if current_price < avg_price - 2 * std_dev:
                    action = "buy"
                    confidence = min(0.8, (avg_price - current_price) / std_dev / 2)
                elif current_price > avg_price + 2 * std_dev:
                    action = "sell"
                    confidence = min(0.8, (current_price - avg_price) / std_dev / 2)
                else:
                    continue
                
                opportunity = TradingOpportunity(
                    opportunity_id=f"mr_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.utcnow(),
                    type=OpportunityType.MEAN_REVERSION,
                    symbol=symbol,
                    action=action,
                    confidence=max(0.4, confidence),
                    strength=confidence,
                    expected_return=confidence * 0.015,  # 1.5% expected return
                    risk_score=0.2,  # Lower risk for mean reversion
                    time_horizon=timedelta(hours=1),
                    position_size=Decimal(str(self._calculate_position_size(confidence, 0.2))),
                    entry_price=Decimal(str(current_price)),
                    reasoning=f"Mean reversion: {current_price:.2f} vs avg {avg_price:.2f}",
                    indicators_used=["Mean", "Standard Deviation"],
                    metadata={
                        "avg_price": avg_price,
                        "std_dev": std_dev,
                        "deviation": abs(current_price - avg_price) / std_dev
                    }
                )
                
                opportunities.append(opportunity)
        
        except Exception as e:
            logger.error(f"Error detecting mean reversion opportunities: {e}")
        
        return opportunities
    
    async def _detect_breakout_opportunities(self, analysis: MarketAnalysis) -> List[TradingOpportunity]:
        """Detect breakout opportunities"""
        opportunities = []
        
        try:
            # Breakout detection logic
            for symbol in self.market_data_cache.keys():
                if len(self.market_data_cache[symbol]) < 20:
                    continue
                
                recent_data = self.market_data_cache[symbol][-20:]
                prices = [float(d['close']) for d in recent_data]
                
                # Calculate support and resistance levels
                resistance = max(prices[-20:])
                support = min(prices[-20:])
                current_price = prices[-1]
                
                # Detect breakout
                if current_price > resistance * 1.01:  # Break above resistance
                    action = "buy"
                    confidence = min(0.85, (current_price - resistance) / resistance * 5)
                elif current_price < support * 0.99:  # Break below support
                    action = "sell"
                    confidence = min(0.85, (support - current_price) / support * 5)
                else:
                    continue
                
                opportunity = TradingOpportunity(
                    opportunity_id=f"breakout_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.utcnow(),
                    type=OpportunityType.BREAKOUT,
                    symbol=symbol,
                    action=action,
                    confidence=max(0.5, confidence),
                    strength=confidence,
                    expected_return=confidence * 0.025,  # 2.5% expected return
                    risk_score=0.4,  # Higher risk for breakouts
                    time_horizon=timedelta(hours=3),
                    position_size=Decimal(str(self._calculate_position_size(confidence, 0.4))),
                    entry_price=Decimal(str(current_price)),
                    reasoning=f"Breakout: {current_price:.2f} breaking {action} {resistance if action == 'buy' else support:.2f}",
                    indicators_used=["Resistance", "Support"],
                    metadata={
                        "resistance": resistance,
                        "support": support,
                        "breakout_level": resistance if action == "buy" else support
                    }
                )
                
                opportunities.append(opportunity)
        
        except Exception as e:
            logger.error(f"Error detecting breakout opportunities: {e}")
        
        return opportunities
    
    async def _detect_volatility_opportunities(self, analysis: MarketAnalysis) -> List[TradingOpportunity]:
        """Detect volatility-based opportunities"""
        opportunities = []
        
        try:
            # Volatility trading logic (straddles, strangles, etc.)
            for symbol in self.market_data_cache.keys():
                if len(self.market_data_cache[symbol]) < 10:
                    continue
                
                recent_data = self.market_data_cache[symbol][-10:]
                prices = [float(d['close']) for d in recent_data]
                
                # Calculate volatility
                returns = np.diff(np.log(prices))
                volatility = np.std(returns) * np.sqrt(252)  # Annualized
                
                # High volatility threshold
                if volatility > 0.3:  # 30% annualized volatility
                    # Create volatility opportunity (simplified)
                    current_price = prices[-1]
                    
                    opportunity = TradingOpportunity(
                        opportunity_id=f"vol_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                        timestamp=datetime.utcnow(),
                        type=OpportunityType.VOLATILITY,
                        symbol=symbol,
                        action="buy",  # Volatility strategy
                        confidence=0.7,
                        strength=0.7,
                        expected_return=0.03,  # 3% expected return
                        risk_score=0.5,  # High risk
                        time_horizon=timedelta(hours=4),
                        position_size=Decimal(str(self._calculate_position_size(0.7, 0.5))),
                        entry_price=Decimal(str(current_price)),
                        reasoning=f"High volatility trading: {volatility:.1%}",
                        indicators_used=["Volatility"],
                        metadata={
                            "volatility": volatility,
                            "expected_range": volatility * current_price
                        }
                    )
                    
                    opportunities.append(opportunity)
        
        except Exception as e:
            logger.error(f"Error detecting volatility opportunities: {e}")
        
        return opportunities
    
    def _initialize_regime_characteristics(self) -> Dict[MarketRegime, RegimeCharacteristics]:
        """Initialize market regime characteristics"""
        return {
            MarketRegime.BULL_TRENDING: RegimeCharacteristics(
                regime=MarketRegime.BULL_TRENDING,
                volatility_threshold=0.2,
                trend_strength_threshold=0.6,
                volume_conditions="increasing",
                recommended_strategies=[OpportunityType.TREND_FOLLOWING, OpportunityType.MOMENTUM],
                risk_multiplier=1.2,
                typical_duration=timedelta(days=30),
                success_probability=0.65
            ),
            MarketRegime.BEAR_TRENDING: RegimeCharacteristics(
                regime=MarketRegime.BEAR_TRENDING,
                volatility_threshold=0.3,
                trend_strength_threshold=0.7,
                volume_conditions="increasing",
                recommended_strategies=[OpportunityType.TREND_FOLLOWING, OpportunityType.VOLATILITY],
                risk_multiplier=0.8,
                typical_duration=timedelta(days=20),
                success_probability=0.55
            ),
            MarketRegime.SIDEWAYS: RegimeCharacteristics(
                regime=MarketRegime.SIDEWAYS,
                volatility_threshold=0.15,
                trend_strength_threshold=0.3,
                volume_conditions="stable",
                recommended_strategies=[OpportunityType.MEAN_REVERSION, OpportunityType.ARBITRAGE],
                risk_multiplier=0.9,
                typical_duration=timedelta(days=45),
                success_probability=0.70
            ),
            MarketRegime.HIGH_VOLATILITY: RegimeCharacteristics(
                regime=MarketRegime.HIGH_VOLATILITY,
                volatility_threshold=0.4,
                trend_strength_threshold=0.4,
                volume_conditions="high",
                recommended_strategies=[OpportunityType.VOLATILITY, OpportunityType.BREAKOUT],
                risk_multiplier=0.7,
                typical_duration=timedelta(days=10),
                success_probability=0.45
            ),
            MarketRegime.LOW_VOLATILITY: RegimeCharacteristics(
                regime=MarketRegime.LOW_VOLATILITY,
                volatility_threshold=0.1,
                trend_strength_threshold=0.2,
                volume_conditions="low",
                recommended_strategies=[OpportunityType.MEAN_REVERSION],
                risk_multiplier=1.1,
                typical_duration=timedelta(days=60),
                success_probability=0.60
            ),
            MarketRegime.CRISIS: RegimeCharacteristics(
                regime=MarketRegime.CRISIS,
                volatility_threshold=0.8,
                trend_strength_threshold=0.9,
                volume_conditions="extreme",
                recommended_strategies=[OpportunityType.VOLATILITY],
                risk_multiplier=0.3,
                typical_duration=timedelta(days=5),
                success_probability=0.25
            )
        }
    
    async def _detect_current_market_regime(self) -> MarketRegime:
        """Detect current market regime"""
        try:
            # Use simplified regime detection based on available data
            if not self.market_data_cache:
                return MarketRegime.UNKNOWN
            
            # Calculate overall market metrics
            total_volatility = 0
            total_trend_strength = 0
            analyzed_symbols = 0
            
            for symbol, data in self.market_data_cache.items():
                if len(data) < 20:
                    continue
                
                prices = [float(d['close']) for d in data[-20:]]
                
                # Calculate volatility
                returns = np.diff(np.log(prices))
                volatility = np.std(returns) * np.sqrt(252)
                
                # Calculate trend strength
                correlation = np.corrcoef(range(len(prices)), prices)[0, 1]
                trend_strength = abs(correlation) if not np.isnan(correlation) else 0
                
                total_volatility += volatility
                total_trend_strength += trend_strength
                analyzed_symbols += 1
            
            if analyzed_symbols == 0:
                return MarketRegime.UNKNOWN
            
            avg_volatility = total_volatility / analyzed_symbols
            avg_trend_strength = total_trend_strength / analyzed_symbols
            
            # Determine regime based on metrics
            if avg_volatility > 0.5:
                return MarketRegime.CRISIS
            elif avg_volatility > 0.3:
                return MarketRegime.HIGH_VOLATILITY
            elif avg_volatility < 0.1:
                return MarketRegime.LOW_VOLATILITY
            elif avg_trend_strength > 0.6:
                # Determine trend direction
                price_change = await self._calculate_overall_price_change()
                if price_change > 0.02:
                    return MarketRegime.BULL_TRENDING
                else:
                    return MarketRegime.BEAR_TRENDING
            else:
                return MarketRegime.SIDEWAYS
        
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return MarketRegime.UNKNOWN
    
    async def _calculate_overall_price_change(self) -> float:
        """Calculate overall market price change"""
        try:
            total_change = 0
            count = 0
            
            for symbol, data in self.market_data_cache.items():
                if len(data) >= 20:
                    old_price = float(data[-20]['close'])
                    current_price = float(data[-1]['close'])
                    change = (current_price - old_price) / old_price
                    total_change += change
                    count += 1
            
            return total_change / count if count > 0 else 0
        
        except Exception as e:
            logger.error(f"Error calculating price change: {e}")
            return 0
    
    async def _analyze_market_sentiment(self) -> float:
        """Analyze overall market sentiment"""
        # Simplified sentiment analysis
        # In a real implementation, this would use news analysis, social media, etc.
        return 0.1  # Slightly bullish bias
    
    async def _calculate_volatility_metrics(self) -> float:
        """Calculate current volatility metrics"""
        if not self.market_data_cache:
            return 0.5
        
        total_volatility = 0
        count = 0
        
        for symbol, data in self.market_data_cache.items():
            if len(data) >= 10:
                prices = [float(d['close']) for d in data[-10:]]
                returns = np.diff(np.log(prices))
                volatility = np.std(returns) * np.sqrt(252)
                total_volatility += volatility
                count += 1
        
        return total_volatility / count if count > 0 else 0.5
    
    async def _calculate_liquidity_score(self) -> float:
        """Calculate market liquidity score"""
        # Simplified liquidity calculation
        # In a real implementation, this would analyze volume, bid-ask spreads, etc.
        return 0.7
    
    async def _calculate_trend_strength(self) -> float:
        """Calculate overall trend strength"""
        if not self.market_data_cache:
            return 0.5
        
        total_trend = 0
        count = 0
        
        for symbol, data in self.market_data_cache.items():
            if len(data) >= 20:
                prices = [float(d['close']) for d in data[-20:]]
                correlation = np.corrcoef(range(len(prices)), prices)[0, 1]
                trend_strength = abs(correlation) if not np.isnan(correlation) else 0
                total_trend += trend_strength
                count += 1
        
        return total_trend / count if count > 0 else 0.5
    
    async def _identify_key_levels(self) -> Dict[str, float]:
        """Identify key support and resistance levels"""
        # Simplified key level identification
        return {
            "major_support": 0.0,
            "minor_support": 0.0,
            "major_resistance": 0.0,
            "minor_resistance": 0.0
        }
    
    async def _analyze_sector_performance(self) -> Dict[str, float]:
        """Analyze sector performance"""
        # Simplified sector analysis
        return {
            "technology": 0.02,
            "healthcare": 0.01,
            "financial": -0.01,
            "energy": 0.03
        }
    
    async def _identify_risk_factors(self) -> List[str]:
        """Identify current risk factors"""
        risks = []
        
        # Check for regime-specific risks
        regime_info = self.regime_characteristics.get(self.current_regime)
        if regime_info and regime_info.risk_multiplier < 0.8:
            risks.append("High regime risk")
        
        # Check volatility
        if self.market_hours.is_any_market_open():
            current_analysis = asyncio.create_task(self.analyze_market_conditions())
            # This would check for high volatility risks
        
        return risks
    
    def _evaluate_trading_conditions(self, regime: MarketRegime, sentiment: float, 
                                   volatility: float, liquidity: float) -> bool:
        """Evaluate if trading conditions are acceptable"""
        # Basic condition evaluation
        if volatility > 0.8:  # Extremely volatile
            return False
        
        if liquidity < 0.3:  # Low liquidity
            return False
        
        if regime == MarketRegime.CRISIS:
            return False
        
        return True
    
    async def _recommend_strategies(self, regime: MarketRegime, sentiment: float, volatility: float) -> List[str]:
        """Recommend strategies based on current market conditions"""
        regime_info = self.regime_characteristics.get(regime)
        if regime_info:
            return [strategy.value for strategy in regime_info.recommended_strategies]
        
        return ["momentum", "mean_reversion"]
    
    def _calculate_position_size(self, confidence: float, risk_score: float) -> float:
        """Calculate appropriate position size based on confidence and risk"""
        # Kelly criterion-inspired position sizing
        base_size = 0.02  # 2% base position size
        confidence_multiplier = confidence * 2  # Scale confidence
        risk_multiplier = max(0.1, 1 - risk_score)  # Reduce size for high risk
        
        position_size = base_size * confidence_multiplier * risk_multiplier
        return min(position_size, 0.10)  # Cap at 10% of portfolio
    
    async def _market_data_collection_loop(self):
        """Collect market data continuously"""
        while True:
            try:
                # In a real implementation, this would fetch real market data
                # For now, generate mock data
                symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
                
                for symbol in symbols:
                    if symbol not in self.market_data_cache:
                        self.market_data_cache[symbol] = []
                    
                    # Generate mock price data
                    current_price = 100.0 + np.random.normal(0, 2)
                    volume = np.random.randint(1000000, 10000000)
                    
                    data_point = {
                        'symbol': symbol,
                        'timestamp': datetime.utcnow(),
                        'open': current_price - 1,
                        'high': current_price + 2,
                        'low': current_price - 2,
                        'close': current_price,
                        'volume': volume
                    }
                    
                    self.market_data_cache[symbol].append(data_point)
                    
                    # Keep only recent data
                    if len(self.market_data_cache[symbol]) > 100:
                        self.market_data_cache[symbol] = self.market_data_cache[symbol][-100:]
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in market data collection: {e}")
                await asyncio.sleep(5)
    
    async def _regime_detection_loop(self):
        """Detect market regime changes"""
        while True:
            try:
                new_regime = await self._detect_current_market_regime()
                
                if new_regime != self.current_regime:
                    logger.info(f"📊 Market regime change: {self.current_regime.value} -> {new_regime.value}")
                    self.regime_history.append((datetime.utcnow(), new_regime))
                    self.current_regime = new_regime
                
                self.last_regime_detection = datetime.utcnow()
                await asyncio.sleep(self.regime_detection_interval)
                
            except Exception as e:
                logger.error(f"Error in regime detection: {e}")
                await asyncio.sleep(self.regime_detection_interval)
    
    async def _opportunity_analysis_loop(self):
        """Analyze for trading opportunities"""
        while True:
            try:
                # Analyze current opportunities
                opportunities = await self.detect_opportunities()
                
                logger.debug(f"Opportunity analysis: {len(opportunities)} active opportunities")
                await asyncio.sleep(30)  # Analyze every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in opportunity analysis: {e}")
                await asyncio.sleep(30)
    
    async def get_decision_metrics(self) -> Dict[str, Any]:
        """Get decision engine metrics"""
        return {
            "current_regime": self.current_regime.value,
            "regime_duration": (datetime.utcnow() - self.regime_history[-1][0]).total_seconds() 
                             if self.regime_history else 0,
            "active_opportunities": len(self.trading_opportunities),
            "symbols_monitored": len(self.market_data_cache),
            "signals_generated": len(self.market_signals),
            "last_regime_detection": self.last_regime_detection.isoformat() if self.last_regime_detection else None,
            "regime_history_length": len(self.regime_history)
        }