"""
Market Impact Analysis Module

Implements comprehensive market impact analysis including:
- Permanent vs temporary impact modeling
- Market impact prediction models
- Order book analysis and depth
- Volume impact correlation
- Time-decay modeling
- Market regime impact assessment
- Cross-asset impact analysis
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from scipy import optimize, stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from ..core.config import AnalyticsConfig

logger = logging.getLogger(__name__)


@dataclass
class ImpactComponents:
    """Market impact component breakdown"""
    permanent_impact: float
    temporary_impact: float
    total_impact: float
    reversal_impact: float
    feedback_impact: float


@dataclass
class ImpactPrediction:
    """Market impact prediction results"""
    predicted_impact: float
    confidence_interval: Tuple[float, float]
    model_accuracy: float
    impact_duration: float
    price_recovery_time: float
    model_features: Dict[str, float]


@dataclass
class OrderBookAnalysis:
    """Order book depth analysis"""
    bid_depth: float
    ask_depth: float
    spread: float
    market_impact_coefficient: float
    liquidity_score: float
    price_pressure_estimate: float
    optimal_execution_size: float


@dataclass
class VolumeImpactCorrelation:
    """Volume impact correlation analysis"""
    volume_impact_slope: float
    volume_impact_intercept: float
    correlation_strength: float
    volume_thresholds: Dict[str, float]
    impact_scaling_factor: float


@dataclass
class TimeDecayModel:
    """Time-decay impact modeling"""
    decay_rate: float
    half_life: float
    decay_formula: str
    time_to_normalization: float
    decay_parameters: Dict[str, float]


@dataclass
class MarketRegimeImpact:
    """Market regime impact assessment"""
    current_regime: str
    regime_impact_multiplier: float
    volatility_regime: str
    liquidity_regime: str
    impact_sensitivity: float
    regime_transition_probability: float


@dataclass
class CrossAssetImpact:
    """Cross-asset impact analysis"""
    correlated_impact: float
    contagion_risk: float
    sector_impact_propagation: Dict[str, float]
    asset_class_correlation: float
    systemic_risk_level: float


class MarketImpactAnalyzer:
    """
    Advanced Market Impact Analysis
    
    Provides comprehensive market impact analysis including:
    - Permanent vs temporary impact modeling
    - Market impact prediction models
    - Order book analysis and depth
    - Volume impact correlation
    - Time-decay modeling
    - Market regime impact assessment
    - Cross-asset impact analysis
    """
    
    def __init__(self, config: AnalyticsConfig):
        """Initialize market impact analyzer"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Market regimes
        self.market_regimes = {
            'Bull_Normal': {'volatility': 'Low', 'liquidity': 'High', 'impact_multiplier': 0.8},
            'Bull_Volatile': {'volatility': 'High', 'liquidity': 'Medium', 'impact_multiplier': 1.2},
            'Bear_Normal': {'volatility': 'Medium', 'liquidity': 'Medium', 'impact_multiplier': 1.1},
            'Bear_Volatile': {'volatility': 'Very_High', 'liquidity': 'Low', 'impact_multiplier': 1.8},
            'Sideways_Normal': {'volatility': 'Low', 'liquidity': 'High', 'impact_multiplier': 0.9},
            'Sideways_Volatile': {'volatility': 'High', 'liquidity': 'Low', 'impact_multiplier': 1.3}
        }
        
        # Asset classes
        self.asset_classes = ['Equity', 'Fixed_Income', 'Commodity', 'FX', 'Crypto', 'Derivative']
        
        # Impact models
        self.impact_models = {
            'linear': self._linear_impact_model,
            'square_root': self._square_root_impact_model,
            'logarithmic': self._logarithmic_impact_model,
            'power_law': self._power_law_impact_model
        }
        
        # Market data cache
        self._order_book_cache = {}
        self._impact_prediction_models = {}
        
        self.logger.info("Market Impact Analyzer initialized")
    
    async def analyze_market_impact(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive market impact analysis
        
        Args:
            trade_data: Trade information for impact analysis
            
        Returns:
            Complete market impact analysis results
        """
        try:
            self.logger.info("Analyzing market impact for trade data")
            
            # Decompose impact into components
            impact_components = await self._decompose_market_impact(trade_data)
            
            # Predict future impact
            impact_prediction = await self._predict_market_impact(trade_data)
            
            # Analyze order book depth
            order_book_analysis = await self._analyze_order_book_depth(trade_data)
            
            # Calculate volume-impact correlation
            volume_correlation = await self._calculate_volume_impact_correlation(trade_data)
            
            # Model time-decay
            time_decay_model = await self._model_time_decay(trade_data)
            
            # Assess market regime impact
            regime_impact = await self._assess_market_regime_impact(trade_data)
            
            # Analyze cross-asset impact
            cross_asset_impact = await self._analyze_cross_asset_impact(trade_data)
            
            # Generate impact summary
            impact_summary = await self._generate_impact_summary(
                impact_components, impact_prediction, order_book_analysis,
                volume_correlation, time_decay_model, regime_impact, cross_asset_impact
            )
            
            return {
                'trade_data_summary': {
                    'symbol': trade_data.get('symbol', 'N/A'),
                    'order_size': trade_data.get('quantity', 0),
                    'order_type': trade_data.get('order_type', 'N/A'),
                    'side': trade_data.get('side', 'N/A')
                },
                'impact_components': impact_components.__dict__,
                'impact_prediction': impact_prediction.__dict__,
                'order_book_analysis': order_book_analysis.__dict__,
                'volume_impact_correlation': volume_correlation.__dict__,
                'time_decay_model': time_decay_model.__dict__,
                'market_regime_impact': regime_impact.__dict__,
                'cross_asset_impact': cross_asset_impact.__dict__,
                'impact_summary': impact_summary,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market impact: {e}")
            raise
    
    async def _decompose_market_impact(self, trade_data: Dict[str, Any]) -> ImpactComponents:
        """Decompose market impact into components"""
        try:
            # Extract trade parameters
            order_size = trade_data.get('quantity', 1000)
            arrival_price = trade_data.get('arrival_price', 100.0)
            execution_price = trade_data.get('execution_price', arrival_price)
            market_cap = trade_data.get('market_cap', 10000000000)  # Default $10B
            
            # Calculate total impact
            total_impact = (execution_price - arrival_price) / arrival_price
            
            # Decompose impact using standard models
            # Permanent impact (information effect)
            permanent_impact = total_impact * 0.6  # Typically 50-70% of total
            
            # Temporary impact (liquidity effect)
            temporary_impact = total_impact * 0.4   # Typically 30-50% of total
            
            # Additional components
            reversal_impact = abs(total_impact) * 0.1  # Mean reversion effect
            feedback_impact = abs(total_impact) * 0.05  # Feedback trading effect
            
            # Adjust for order size (larger orders have higher impact)
            size_factor = np.log(order_size + 1) / np.log(100000)  # Normalize to 100k shares
            impact_adjustment = 1 + size_factor * 0.3
            
            permanent_impact *= impact_adjustment
            temporary_impact *= impact_adjustment
            
            return ImpactComponents(
                permanent_impact=permanent_impact,
                temporary_impact=temporary_impact,
                total_impact=total_impact,
                reversal_impact=reversal_impact,
                feedback_impact=feedback_impact
            )
            
        except Exception as e:
            self.logger.error(f"Error decomposing market impact: {e}")
            raise
    
    async def _predict_market_impact(self, trade_data: Dict[str, Any]) -> ImpactPrediction:
        """Predict market impact using machine learning models"""
        try:
            # Extract features for prediction
            features = {
                'order_size': trade_data.get('quantity', 1000),
                'volatility': trade_data.get('volatility', 0.02),
                'avg_volume': trade_data.get('avg_daily_volume', 1000000),
                'spread': trade_data.get('spread', 0.01),
                'market_cap': trade_data.get('market_cap', 10000000000),
                'sector_beta': trade_data.get('sector_beta', 1.0),
                'time_of_day': datetime.now().hour,
                'day_of_week': datetime.now().weekday()
            }
            
            # Mock prediction using simplified model
            # In reality, this would use trained ML models
            
            # Calculate base impact using square-root model
            volume_ratio = features['order_size'] / features['avg_volume']
            volatility_adj = features['volatility'] / 0.02  # Normalize to 2%
            spread_adj = features['spread'] / 0.01  # Normalize to 1%
            
            # Square-root impact model: Impact = k * (Volume Ratio)^0.5 * Volatility
            base_impact = 0.01 * np.sqrt(volume_ratio) * volatility_adj * spread_adj
            
            # Apply additional factors
            market_cap_factor = 10000000000 / features['market_cap']  # Larger cap = lower impact
            time_factor = 1.0
            if features['time_of_day'] in [9, 15]:  # High impact at open/close
                time_factor = 1.2
            
            predicted_impact = base_impact * market_cap_factor * time_factor
            
            # Calculate confidence interval
            prediction_std = abs(predicted_impact) * 0.3
            confidence_lower = predicted_impact - 1.96 * prediction_std
            confidence_upper = predicted_impact + 1.96 * prediction_std
            
            # Estimate impact duration (in minutes)
            impact_duration = max(5, np.sqrt(features['order_size']) * 0.1)
            price_recovery_time = impact_duration * 2
            
            # Model accuracy (mock)
            model_accuracy = 0.78
            
            return ImpactPrediction(
                predicted_impact=predicted_impact,
                confidence_interval=(confidence_lower, confidence_upper),
                model_accuracy=model_accuracy,
                impact_duration=impact_duration,
                price_recovery_time=price_recovery_time,
                model_features=features
            )
            
        except Exception as e:
            self.logger.error(f"Error predicting market impact: {e}")
            raise
    
    async def _analyze_order_book_depth(self, trade_data: Dict[str, Any]) -> OrderBookAnalysis:
        """Analyze order book depth and liquidity"""
        try:
            # Mock order book data - in reality would fetch from market data provider
            symbol = trade_data.get('symbol', 'AAPL')
            
            # Generate realistic order book simulation
            np.random.seed(hash(symbol) % 1000)  # Consistent for same symbol
            
            # Bid/Ask depth simulation
            bid_levels = 10
            ask_levels = 10
            
            bid_depth = np.random.exponential(scale=50000)  # Average bid depth
            ask_depth = np.random.exponential(scale=45000)  # Average ask depth
            spread = np.random.uniform(0.001, 0.005)  # 10-50 bps
            
            # Calculate market impact coefficient
            total_depth = bid_depth + ask_depth
            impact_coefficient = 1000 / total_depth  # Inverse relationship
            
            # Liquidity score (0-1)
            liquidity_score = min(1.0, total_depth / 100000)
            
            # Estimate price pressure
            order_size = trade_data.get('quantity', 1000)
            price_pressure = (order_size / total_depth) * spread * 2
            
            # Calculate optimal execution size
            optimal_size = np.sqrt(total_depth) * 0.1  # sqrt(depth) heuristic
            
            return OrderBookAnalysis(
                bid_depth=bid_depth,
                ask_depth=ask_depth,
                spread=spread,
                market_impact_coefficient=impact_coefficient,
                liquidity_score=liquidity_score,
                price_pressure_estimate=price_pressure,
                optimal_execution_size=optimal_size
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing order book depth: {e}")
            raise
    
    async def _calculate_volume_impact_correlation(self, trade_data: Dict[str, Any]) -> VolumeImpactCorrelation:
        """Calculate volume-impact correlation"""
        try:
            # Generate mock historical data for correlation analysis
            order_size = trade_data.get('quantity', 1000)
            
            # Create synthetic volume-impact data
            np.random.seed(42)
            volumes = np.random.lognormal(mean=np.log(50000), sigma=0.5, size=1000)
            impacts = []
            
            for vol in volumes:
                # Impact increases with order size relative to volume
                volume_ratio = order_size / vol
                impact = 0.01 * np.sqrt(volume_ratio) * np.random.uniform(0.8, 1.2)
                impacts.append(impact)
            
            # Linear regression for volume-impact relationship
            slope, intercept, r_value, p_value, std_err = stats.linregress(volumes, impacts)
            
            # Volume thresholds
            volume_thresholds = {
                'low_volume': np.percentile(volumes, 25),
                'medium_volume': np.percentile(volumes, 50),
                'high_volume': np.percentile(volumes, 75),
                'very_high_volume': np.percentile(volumes, 95)
            }
            
            # Calculate impact scaling factor
            scaling_factor = abs(slope) * np.mean(volumes)
            
            return VolumeImpactCorrelation(
                volume_impact_slope=slope,
                volume_impact_intercept=intercept,
                correlation_strength=r_value,
                volume_thresholds=volume_thresholds,
                impact_scaling_factor=scaling_factor
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating volume-impact correlation: {e}")
            raise
    
    async def _model_time_decay(self, trade_data: Dict[str, Any]) -> TimeDecayModel:
        """Model time-decay of market impact"""
        try:
            # Extract trade parameters
            total_impact = trade_data.get('market_impact', 0.01)
            volatility = trade_data.get('volatility', 0.02)
            order_size = trade_data.get('quantity', 1000)
            
            # Calculate decay rate based on market conditions
            # Higher volatility = faster decay
            base_decay_rate = 0.1
            volatility_adj = volatility / 0.02  # Normalize to 2%
            decay_rate = base_decay_rate * volatility_adj
            
            # Calculate half-life
            half_life = np.log(2) / decay_rate
            
            # Determine decay formula type
            decay_formula = "exponential"
            if volatility > 0.03:
                decay_formula = "power_law"
            elif volatility < 0.01:
                decay_formula = "linear"
            
            # Time to normalization (impact < 1% of initial)
            normalization_threshold = abs(total_impact) * 0.01
            if decay_formula == "exponential":
                time_to_normalization = half_life * np.log(abs(total_impact) / normalization_threshold)
            else:
                time_to_normalization = half_life * 3  # Rough estimate
            
            # Decay parameters
            decay_parameters = {
                'initial_impact': total_impact,
                'decay_rate': decay_rate,
                'half_life': half_life,
                'volatility_factor': volatility_adj,
                'size_factor': np.log(order_size + 1) / 10
            }
            
            return TimeDecayModel(
                decay_rate=decay_rate,
                half_life=half_life,
                decay_formula=decay_formula,
                time_to_normalization=time_to_normalization,
                decay_parameters=decay_parameters
            )
            
        except Exception as e:
            self.logger.error(f"Error modeling time decay: {e}")
            raise
    
    async def _assess_market_regime_impact(self, trade_data: Dict[str, Any]) -> MarketRegimeImpact:
        """Assess market regime impact"""
        try:
            # Determine current market regime
            volatility = trade_data.get('volatility', 0.02)
            market_trend = trade_data.get('market_trend', 'Sideways')
            
            # Classify volatility regime
            if volatility < 0.015:
                volatility_regime = 'Low'
            elif volatility < 0.03:
                volatility_regime = 'Medium'
            else:
                volatility_regime = 'High'
            
            # Classify liquidity regime (mock)
            avg_volume = trade_data.get('avg_daily_volume', 1000000)
            if avg_volume > 5000000:
                liquidity_regime = 'High'
            elif avg_volume > 1000000:
                liquidity_regime = 'Medium'
            else:
                liquidity_regime = 'Low'
            
            # Determine current regime
            regime_key = f"{market_trend}_{volatility_regime}"
            if regime_key not in self.market_regimes:
                regime_key = 'Sideways_Normal'  # Default
            
            current_regime = regime_key
            regime_data = self.market_regimes[regime_key]
            
            # Calculate regime impact multiplier
            regime_multiplier = regime_data['impact_multiplier']
            
            # Impact sensitivity (how much regime affects impact)
            impact_sensitivity = 0.5 if volatility_regime == 'High' else 0.3
            
            # Regime transition probability (mock)
            transition_prob = 0.1 if volatility_regime == 'Low' else 0.25 if volatility_regime == 'Medium' else 0.4
            
            return MarketRegimeImpact(
                current_regime=current_regime,
                regime_impact_multiplier=regime_multiplier,
                volatility_regime=volatility_regime,
                liquidity_regime=liquidity_regime,
                impact_sensitivity=impact_sensitivity,
                regime_transition_probability=transition_prob
            )
            
        except Exception as e:
            self.logger.error(f"Error assessing market regime impact: {e}")
            raise
    
    async def _analyze_cross_asset_impact(self, trade_data: Dict[str, Any]) -> CrossAssetImpact:
        """Analyze cross-asset impact"""
        try:
            symbol = trade_data.get('symbol', 'AAPL')
            sector = trade_data.get('sector', 'Technology')
            
            # Mock correlated impact calculation
            correlated_impact = trade_data.get('market_impact', 0.01) * 0.3  # 30% correlated impact
            
            # Sector impact propagation
            sector_propagation = {
                'Technology': 0.0,  # Reference sector
                'Financials': 0.15,
                'Healthcare': 0.10,
                'Consumer_Discretionary': 0.20,
                'Industrials': 0.12,
                'Energy': 0.05,
                'Materials': 0.08
            }
            
            # Default to medium impact if sector not found
            sector_impact = sector_propagation.get(sector, 0.12)
            
            # Asset class correlation (mock)
            asset_class_correlation = 0.65 if sector in ['Technology', 'Financials'] else 0.45
            
            # Contagion risk calculation
            contagion_risk = sector_impact * (1 - asset_class_correlation)
            
            # Systemic risk level
            systemic_risk_level = min(1.0, correlated_impact * 10)
            
            return CrossAssetImpact(
                correlated_impact=correlated_impact,
                contagion_risk=contagion_risk,
                sector_impact_propagation=sector_propagation,
                asset_class_correlation=asset_class_correlation,
                systemic_risk_level=systemic_risk_level
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing cross-asset impact: {e}")
            raise
    
    async def _generate_impact_summary(self,
                                     components: ImpactComponents,
                                     prediction: ImpactPrediction,
                                     order_book: OrderBookAnalysis,
                                     volume_corr: VolumeImpactCorrelation,
                                     time_decay: TimeDecayModel,
                                     regime: MarketRegimeImpact,
                                     cross_asset: CrossAssetImpact) -> Dict[str, Any]:
        """Generate comprehensive impact summary"""
        try:
            # Risk assessment
            total_risk_score = 0
            
            # Component risk scores
            impact_magnitude = abs(components.total_impact)
            prediction_confidence = prediction.model_accuracy
            
            if impact_magnitude > 0.02:
                total_risk_score += 3
            elif impact_magnitude > 0.01:
                total_risk_score += 2
            elif impact_magnitude > 0.005:
                total_risk_score += 1
            
            if prediction_confidence < 0.7:
                total_risk_score += 2
            elif prediction_confidence < 0.8:
                total_risk_score += 1
            
            if order_book.liquidity_score < 0.5:
                total_risk_score += 2
            elif order_book.liquidity_score < 0.7:
                total_risk_score += 1
            
            # Risk level classification
            if total_risk_score >= 6:
                risk_level = 'Very High'
            elif total_risk_score >= 4:
                risk_level = 'High'
            elif total_risk_score >= 2:
                risk_level = 'Medium'
            else:
                risk_level = 'Low'
            
            # Key insights
            insights = {
                'impact_severity': 'Severe' if impact_magnitude > 0.025 else 'Moderate' if impact_magnitude > 0.01 else 'Low',
                'primary_risk_factor': self._identify_primary_risk_factor(components, order_book, regime),
                'execution_timing': 'Optimal' if time_decay.half_life < 30 else 'Suboptimal',
                'liquidity_adequacy': 'Adequate' if order_book.liquidity_score > 0.7 else 'Limited',
                'cross_asset_risk': 'High' if cross_asset.systemic_risk_level > 0.5 else 'Low'
            }
            
            # Recommendations
            recommendations = []
            if impact_magnitude > 0.02:
                recommendations.append("Consider splitting order into smaller chunks")
            
            if order_book.liquidity_score < 0.5:
                recommendations.append("Use more passive execution strategies")
            
            if regime.regime_impact_multiplier > 1.3:
                recommendations.append("Consider waiting for better market conditions")
            
            if cross_asset.systemic_risk_level > 0.4:
                recommendations.append("Monitor sector correlation during execution")
            
            # Impact mitigation strategies
            mitigation_strategies = {
                'algo_selection': 'Implementation Shortfall' if impact_magnitude > 0.01 else 'VWAP',
                'order_sizing': 'Reduce by 50%' if impact_magnitude > 0.025 else 'Current size acceptable',
                'execution_timing': 'Avoid open/close' if regime.current_regime.endswith('Volatile') else 'Normal timing',
                'venue_selection': 'High-liquidity venues only' if order_book.liquidity_score < 0.6 else 'Any venue'
            }
            
            return {
                'risk_level': risk_level,
                'risk_score': total_risk_score,
                'key_insights': insights,
                'recommendations': recommendations,
                'mitigation_strategies': mitigation_strategies,
                'impact_metrics': {
                    'expected_impact': components.total_impact,
                    'prediction_accuracy': prediction.model_accuracy,
                    'liquidity_score': order_book.liquidity_score,
                    'regime_multiplier': regime.regime_impact_multiplier
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating impact summary: {e}")
            raise
    
    def _identify_primary_risk_factor(self,
                                    components: ImpactComponents,
                                    order_book: OrderBookAnalysis,
                                    regime: MarketRegimeImpact) -> str:
        """Identify the primary risk factor"""
        risk_factors = {
            'high_impact': abs(components.total_impact),
            'low_liquidity': 1 - order_book.liquidity_score,
            'volatility_regime': 1 if regime.volatility_regime == 'High' else 0,
            'regime_multiplier': regime.regime_impact_multiplier - 1
        }
        
        return max(risk_factors.items(), key=lambda x: x[1])[0]
    
    # Impact model functions
    def _linear_impact_model(self, order_size: float, volume: float) -> float:
        """Linear impact model"""
        return 0.01 * (order_size / volume)
    
    def _square_root_impact_model(self, order_size: float, volume: float) -> float:
        """Square-root impact model (Kyle's model)"""
        return 0.01 * np.sqrt(order_size / volume)
    
    def _logarithmic_impact_model(self, order_size: float, volume: float) -> float:
        """Logarithmic impact model"""
        return 0.005 * np.log(1 + order_size / volume)
    
    def _power_law_impact_model(self, order_size: float, volume: float) -> float:
        """Power law impact model"""
        return 0.008 * (order_size / volume) ** 0.3
    
    async def update_real_time_metrics(self):
        """Update real-time market impact metrics"""
        try:
            self.logger.debug("Updating real-time market impact metrics")
            # Update cached impact predictions and order book data
            pass
        except Exception as e:
            self.logger.error(f"Error updating real-time metrics: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for market impact analyzer"""
        try:
            return {
                'status': 'healthy',
                'last_update': datetime.now().isoformat(),
                'market_regimes': len(self.market_regimes),
                'impact_models': len(self.impact_models),
                'asset_classes': len(self.asset_classes)
            }
        except Exception as e:
            self.logger.error(f"Error in market impact health check: {e}")
            return {'status': 'error', 'error': str(e)}