"""
Trade Execution Quality Analysis Module

Implements comprehensive TCA (Transaction Cost Analysis) including:
- Implementation shortfall analysis
- Market impact measurement
- Transaction cost analysis
- VWAP and TWAP analysis
- Slippage and timing analysis
- Liquidity analysis and measurement
- Best execution analysis
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

from ..core.config import AnalyticsConfig

logger = logging.getLogger(__name__)


@dataclass
class ImplementationShortfall:
    """Implementation shortfall metrics"""
    total_shortfall: float
    execution_cost: float
    opportunity_cost: float
    delay_cost: float
    timing_cost: float
    benchmark_return: float
    execution_vs_benchmark: float
    shortfalls_by_time: Dict[str, float]


@dataclass
class MarketImpact:
    """Market impact analysis results"""
    permanent_impact: float
    temporary_impact: float
    total_impact: float
    impact_vs_volume: float
    impact_vs_volatility: float
    impact_decay_half_life: float
    volume_impact_coefficient: float


@dataclass
class TransactionCosts:
    """Transaction cost breakdown"""
    commission_cost: float
    spread_cost: float
    market_impact_cost: float
    timing_cost: float
    opportunity_cost: float
    total_cost: float
    cost_per_share: float
    cost_percentage: float


@dataclass
class VWAPAnalysis:
    """VWAP analysis results"""
    actual_vwap: float
    benchmark_vwap: float
    vwap_outperformance: float
    execution_quality_score: float
    volume_timing_score: float
    price_improvement: float


@dataclass
class BestExecution:
    """Best execution analysis"""
    execution_score: float
    price_improvement_vs_market: float
    timing_optimization: float
    liquidity_optimization: float
    algorithm_performance: Dict[str, float]
    venue_analysis: Dict[str, float]


@dataclass
class SlippageAnalysis:
    """Slippage analysis results"""
    average_slippage: float
    slippage_volatility: float
    positive_slippage_rate: float
    slippage_by_market_condition: Dict[str, float]
    slippage_vs_order_size: Dict[str, float]
    slippage_vs_execution_speed: Dict[str, float]


class ExecutionQualityAnalyzer:
    """
    Advanced Trade Execution Quality Analyzer
    
    Provides comprehensive TCA analysis including:
    - Implementation shortfall analysis
    - Market impact measurement
    - Transaction cost analysis (TCA)
    - VWAP and TWAP analysis
    - Slippage and timing analysis
    - Liquidity analysis and measurement
    - Best execution analysis
    """
    
    def __init__(self, config: AnalyticsConfig):
        """Initialize execution quality analyzer"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Market condition classifications
        self.market_conditions = ['Bull', 'Bear', 'Sideways', 'High_Volatility', 'Low_Volatility']
        
        # Trading venues (mock)
        self.trading_venues = {
            'NYSE': {'liquidity_score': 0.95, 'spread_avg': 0.01},
            'NASDAQ': {'liquidity_score': 0.90, 'spread_avg': 0.015},
            'ARCA': {'liquidity_score': 0.88, 'spread_avg': 0.02},
            'BATS': {'liquidity_score': 0.85, 'spread_avg': 0.025}
        }
        
        # Execution algorithms
        self.execution_algorithms = [
            'VWAP', 'TWAP', 'Implementation_Shortfall', 'Arrival_Price', 
            'Percentage_of_Volume', 'Iceberg', 'Dark_Pool'
        ]
        
        self.logger.info("Execution Quality Analyzer initialized")
    
    async def analyze_execution_quality(self,
                                      trade_id: str = None,
                                      strategy_id: str = None,
                                      period: str = "1M") -> Dict[str, Any]:
        """
        Comprehensive execution quality analysis
        
        Args:
            trade_id: Specific trade ID (optional)
            strategy_id: Strategy identifier (optional)
            period: Analysis period
            
        Returns:
            Complete execution quality analysis
        """
        try:
            self.logger.info(f"Analyzing execution quality for trade: {trade_id}")
            
            # Get trade data
            trade_data = await self._get_trade_data(trade_id, strategy_id, period)
            
            # Calculate implementation shortfall
            shortfall_analysis = await self._calculate_implementation_shortfall(trade_data)
            
            # Calculate market impact
            impact_analysis = await self._calculate_market_impact(trade_data)
            
            # Calculate transaction costs
            cost_analysis = await self._calculate_transaction_costs(trade_data)
            
            # Perform VWAP/TWAP analysis
            vwap_analysis = await self._calculate_vwap_analysis(trade_data)
            
            # Perform slippage analysis
            slippage_analysis = await self._calculate_slippage_analysis(trade_data)
            
            # Calculate best execution metrics
            best_execution = await self._calculate_best_execution(trade_data)
            
            # Analyze liquidity
            liquidity_analysis = await self._calculate_liquidity_analysis(trade_data)
            
            # Generate execution summary
            execution_summary = await self._generate_execution_summary(
                shortfall_analysis, impact_analysis, cost_analysis, vwap_analysis,
                slippage_analysis, best_execution, liquidity_analysis
            )
            
            return {
                'trade_id': trade_id,
                'strategy_id': strategy_id,
                'period': period,
                'implementation_shortfall': shortfall_analysis.__dict__,
                'market_impact': impact_analysis.__dict__,
                'transaction_costs': cost_analysis.__dict__,
                'vwap_analysis': vwap_analysis.__dict__,
                'slippage_analysis': slippage_analysis.__dict__,
                'best_execution': best_execution.__dict__,
                'liquidity_analysis': liquidity_analysis,
                'execution_summary': execution_summary,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing execution quality: {e}")
            raise
    
    async def _get_trade_data(self, trade_id: str, strategy_id: str, period: str) -> Dict[str, Any]:
        """Get trade execution data"""
        # Mock trade data - replace with actual data retrieval
        
        if trade_id:
            # Specific trade data
            trade_data = {
                'trade_id': trade_id,
                'symbol': 'AAPL',
                'order_type': 'Limit',
                'side': 'Buy',
                'quantity': 1000,
                'arrival_time': datetime.now() - timedelta(hours=1),
                'execution_time': datetime.now() - timedelta(minutes=30),
                'completion_time': datetime.now() - timedelta(minutes=20),
                'arrival_price': 150.25,
                'execution_price': 150.35,
                'completion_price': 150.40,
                'benchmark_price': 150.30,
                'vwap': 150.32,
                'twap': 150.28,
                'market_impact': 0.10,
                'commission': 1.00,
                'spread': 0.05,
                'algorithm_used': 'VWAP',
                'venue': 'NASDAQ',
                'fill_rate': 0.95
            }
        else:
            # Portfolio of trades for period analysis
            trade_data = await self._generate_portfolio_trades(period)
        
        return trade_data
    
    async def _generate_portfolio_trades(self, period: str) -> List[Dict[str, Any]]:
        """Generate mock portfolio of trades for period analysis"""
        # Generate random trades for the period
        num_trades = np.random.randint(50, 200)
        
        trades = []
        for i in range(num_trades):
            trade = {
                'trade_id': f"TRADE_{i:05d}",
                'symbol': np.random.choice(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']),
                'order_type': np.random.choice(['Market', 'Limit', 'Stop']),
                'side': np.random.choice(['Buy', 'Sell']),
                'quantity': np.random.randint(100, 10000),
                'arrival_price': np.random.uniform(50, 500),
                'execution_price': None,  # Will be calculated
                'completion_price': None,  # Will be calculated
                'arrival_time': datetime.now() - timedelta(days=np.random.randint(1, 30)),
                'algorithm_used': np.random.choice(self.execution_algorithms),
                'venue': np.random.choice(list(self.trading_venues.keys())),
                'commission': np.random.uniform(0.5, 5.0),
                'market_impact': np.random.uniform(-0.2, 0.3)
            }
            
            # Calculate execution and completion prices with some slippage
            base_price = trade['arrival_price']
            slippage = np.random.normal(0, 0.02) * base_price
            
            trade['execution_price'] = base_price + slippage
            trade['completion_price'] = base_price + slippage * 1.2
            
            trades.append(trade)
        
        return trades
    
    async def _calculate_implementation_shortfall(self, trade_data: Dict[str, Any]) -> ImplementationShortfall:
        """Calculate implementation shortfall analysis"""
        try:
            if isinstance(trade_data, list):
                # Portfolio analysis
                total_shortfall = 0
                execution_costs = []
                opportunity_costs = []
                delay_costs = []
                timing_costs = []
                benchmark_returns = []
                
                for trade in trade_data:
                    arrival_price = trade['arrival_price']
                    execution_price = trade['execution_price']
                    completion_price = trade['completion_price']
                    
                    # Mock benchmark return
                    benchmark_return = np.random.uniform(-0.01, 0.01)
                    
                    # Calculate components
                    execution_cost = (execution_price - arrival_price) / arrival_price
                    completion_cost = (completion_price - arrival_price) / arrival_price
                    total_trade_shortfall = completion_cost - benchmark_return
                    
                    total_shortfall += total_trade_shortfall
                    execution_costs.append(execution_cost)
                    opportunity_costs.append(benchmark_return)
                    
                    # Mock additional costs
                    delay_cost = np.random.uniform(-0.005, 0.005)
                    timing_cost = np.random.uniform(-0.003, 0.003)
                    
                    delay_costs.append(delay_cost)
                    timing_costs.append(timing_cost)
                    benchmark_returns.append(benchmark_return)
                
                # Aggregate results
                avg_execution_cost = np.mean(execution_costs)
                avg_opportunity_cost = np.mean(opportunity_costs)
                avg_delay_cost = np.mean(delay_costs)
                avg_timing_cost = np.mean(timing_costs)
                avg_benchmark_return = np.mean(benchmark_returns)
                execution_vs_benchmark = avg_execution_cost - avg_benchmark_return
                
            else:
                # Single trade analysis
                arrival_price = trade_data['arrival_price']
                execution_price = trade_data['execution_price']
                completion_price = trade_data['completion_price']
                benchmark_price = trade_data['benchmark_price']
                
                # Calculate costs
                execution_cost = (execution_price - arrival_price) / arrival_price
                completion_cost = (completion_price - arrival_price) / arrival_price
                benchmark_return = (benchmark_price - arrival_price) / arrival_price
                
                total_shortfall = completion_cost - benchmark_return
                avg_execution_cost = execution_cost
                avg_opportunity_cost = benchmark_return
                avg_delay_cost = 0.002  # Mock delay cost
                avg_timing_cost = 0.001  # Mock timing cost
                avg_benchmark_return = benchmark_return
                execution_vs_benchmark = execution_cost - benchmark_return
            
            # Time-based shortfalls
            shortfalls_by_time = {
                'first_hour': 0.003,
                'first_day': 0.007,
                'first_week': 0.015,
                'first_month': 0.025
            }
            
            return ImplementationShortfall(
                total_shortfall=total_shortfall if isinstance(trade_data, list) else completion_cost,
                execution_cost=avg_execution_cost,
                opportunity_cost=avg_opportunity_cost,
                delay_cost=avg_delay_cost,
                timing_cost=avg_timing_cost,
                benchmark_return=avg_benchmark_return,
                execution_vs_benchmark=execution_vs_benchmark,
                shortfalls_by_time=shortfalls_by_time
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating implementation shortfall: {e}")
            raise
    
    async def _calculate_market_impact(self, trade_data: Dict[str, Any]) -> MarketImpact:
        """Calculate market impact analysis"""
        try:
            if isinstance(trade_data, list):
                # Portfolio analysis
                total_impact = np.mean([trade.get('market_impact', 0.01) for trade in trade_data])
            else:
                # Single trade analysis
                total_impact = trade_data.get('market_impact', 0.01)
            
            # Decompose impact
            permanent_impact = total_impact * 0.6  # Assume 60% permanent
            temporary_impact = total_impact * 0.4   # Assume 40% temporary
            
            # Calculate impact coefficients
            volume_impact_coeff = abs(total_impact) / 100  # Impact per 100 shares
            volatility_impact_coeff = abs(total_impact) / 0.02  # Impact per 1% volatility
            
            # Estimate decay half-life (mock calculation)
            impact_decay_half_life = 2.5  # minutes
            
            return MarketImpact(
                permanent_impact=permanent_impact,
                temporary_impact=temporary_impact,
                total_impact=total_impact,
                impact_vs_volume=volume_impact_coeff,
                impact_vs_volatility=volatility_impact_coeff,
                impact_decay_half_life=impact_decay_half_life,
                volume_impact_coefficient=volume_impact_coeff
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating market impact: {e}")
            raise
    
    async def _calculate_transaction_costs(self, trade_data: Dict[str, Any]) -> TransactionCosts:
        """Calculate comprehensive transaction costs"""
        try:
            if isinstance(trade_data, list):
                # Portfolio analysis
                commissions = [trade.get('commission', 2.0) for trade in trade_data]
                quantities = [trade.get('quantity', 1000) for trade in trade_data]
                prices = [trade.get('arrival_price', 100.0) for trade in trade_data]
                
                total_commission = np.sum(commissions)
                total_notional = np.sum([q * p for q, p in zip(quantities, prices)])
                
                # Mock other costs
                spread_cost = total_notional * 0.0005  # 5 bps
                market_impact_cost = np.sum([
                    abs(trade.get('market_impact', 0.01)) * trade.get('quantity', 1000) * trade.get('arrival_price', 100.0)
                    for trade in trade_data
                ])
                timing_cost = total_notional * 0.0002  # 2 bps
                opportunity_cost = total_notional * 0.0008  # 8 bps
                
            else:
                # Single trade analysis
                total_notional = trade_data.get('quantity', 1000) * trade_data.get('arrival_price', 100.0)
                total_commission = trade_data.get('commission', 2.0)
                spread_cost = total_notional * 0.0005
                market_impact_cost = abs(trade_data.get('market_impact', 0.01)) * total_notional
                timing_cost = total_notional * 0.0002
                opportunity_cost = total_notional * 0.0008
            
            total_cost = total_commission + spread_cost + market_impact_cost + timing_cost + opportunity_cost
            cost_per_share = total_cost / sum(quantities) if isinstance(trade_data, list) else total_cost / trade_data.get('quantity', 1000)
            cost_percentage = total_cost / total_notional
            
            return TransactionCosts(
                commission_cost=total_commission,
                spread_cost=spread_cost,
                market_impact_cost=market_impact_cost,
                timing_cost=timing_cost,
                opportunity_cost=opportunity_cost,
                total_cost=total_cost,
                cost_per_share=cost_per_share,
                cost_percentage=cost_percentage
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating transaction costs: {e}")
            raise
    
    async def _calculate_vwap_analysis(self, trade_data: Dict[str, Any]) -> VWAPAnalysis:
        """Calculate VWAP analysis"""
        try:
            if isinstance(trade_data, list):
                # Portfolio VWAP analysis
                actual_vwap = np.mean([trade.get('vwap', trade.get('arrival_price', 100.0)) for trade in trade_data])
                benchmark_vwap = np.mean([trade.get('arrival_price', 100.0) for trade in trade_data])
            else:
                # Single trade VWAP analysis
                actual_vwap = trade_data.get('vwap', trade_data.get('completion_price', 150.32))
                benchmark_vwap = trade_data.get('arrival_price', 150.25)
                completion_price = trade_data.get('completion_price', 150.40)
                
                # Calculate actual VWAP as completion price for single trade
                actual_vwap = completion_price
            
            vwap_outperformance = (benchmark_vwap - actual_vwap) / benchmark_vwap
            
            # Calculate execution quality scores
            execution_quality_score = max(0, min(100, (vwap_outperformance + 0.02) * 2500))
            volume_timing_score = 85  # Mock score
            
            # Price improvement vs market
            price_improvement = vwap_outperformance * 100  # Convert to basis points
            
            return VWAPAnalysis(
                actual_vwap=actual_vwap,
                benchmark_vwap=benchmark_vwap,
                vwap_outperformance=vwap_outperformance,
                execution_quality_score=execution_quality_score,
                volume_timing_score=volume_timing_score,
                price_improvement=price_improvement
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating VWAP analysis: {e}")
            raise
    
    async def _calculate_slippage_analysis(self, trade_data: Dict[str, Any]) -> SlippageAnalysis:
        """Calculate slippage analysis"""
        try:
            if isinstance(trade_data, list):
                # Portfolio slippage analysis
                slippages = []
                for trade in trade_data:
                    arrival_price = trade.get('arrival_price', 100.0)
                    execution_price = trade.get('execution_price', arrival_price)
                    side = trade.get('side', 'Buy')
                    
                    # Calculate slippage based on side
                    if side == 'Buy':
                        slippage = (execution_price - arrival_price) / arrival_price
                    else:
                        slippage = (arrival_price - execution_price) / arrival_price
                    
                    slippages.append(slippage)
                
                average_slippage = np.mean(slippages)
                slippage_volatility = np.std(slippages)
                positive_slippage_rate = np.mean([s > 0 for s in slippages])
                
            else:
                # Single trade slippage analysis
                arrival_price = trade_data['arrival_price']
                execution_price = trade_data['execution_price']
                side = trade_data.get('side', 'Buy')
                
                if side == 'Buy':
                    average_slippage = (execution_price - arrival_price) / arrival_price
                else:
                    average_slippage = (arrival_price - execution_price) / arrival_price
                
                slippage_volatility = 0.005  # Mock volatility
                positive_slippage_rate = 0.4 if side == 'Buy' else 0.6  # Mock rate
            
            # Market condition analysis
            slippage_by_condition = {
                'Bull_Market': 0.001,
                'Bear_Market': -0.002,
                'High_Volatility': -0.003,
                'Low_Volatility': 0.0005
            }
            
            # Order size analysis
            slippage_vs_size = {
                'Small_Orders_<1000': 0.0005,
                'Medium_Orders_1000-5000': 0.001,
                'Large_Orders_5000-10000': 0.002,
                'Very_Large_Orders_>10000': 0.004
            }
            
            # Execution speed analysis
            slippage_vs_speed = {
                'Immediate_<1min': -0.001,
                'Fast_1-5min': 0.0005,
                'Medium_5-30min': 0.001,
                'Slow_>30min': 0.0025
            }
            
            return SlippageAnalysis(
                average_slippage=average_slippage,
                slippage_volatility=slippage_volatility,
                positive_slippage_rate=positive_slippage_rate,
                slippage_by_market_condition=slippage_by_condition,
                slippage_vs_order_size=slippage_vs_size,
                slippage_vs_execution_speed=slippage_vs_speed
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating slippage analysis: {e}")
            raise
    
    async def _calculate_best_execution(self, trade_data: Dict[str, Any]) -> BestExecution:
        """Calculate best execution analysis"""
        try:
            if isinstance(trade_data, list):
                # Portfolio best execution analysis
                total_score = 0
                algorithm_performance = {}
                venue_analysis = {}
                
                # Analyze by algorithm
                for algo in self.execution_algorithms:
                    algo_trades = [t for t in trade_data if t.get('algorithm_used') == algo]
                    if algo_trades:
                        score = np.random.uniform(70, 95)
                        algorithm_performance[algo] = score
                        total_score += score * len(algo_trades)
                
                # Analyze by venue
                for venue, data in self.trading_venues.items():
                    venue_trades = [t for t in trade_data if t.get('venue') == venue]
                    if venue_trades:
                        score = data['liquidity_score'] * 100
                        venue_analysis[venue] = {
                            'execution_score': score,
                            'average_spread': data['spread_avg'],
                            'trade_count': len(venue_trades)
                        }
                
                total_score = total_score / len(trade_data) if trade_data else 75
                
            else:
                # Single trade best execution analysis
                total_score = 88  # Mock score
                algorithm_performance = {trade_data.get('algorithm_used', 'Unknown'): 88}
                venue_analysis = {
                    trade_data.get('venue', 'Unknown'): {
                        'execution_score': 85,
                        'average_spread': 0.02,
                        'trade_count': 1
                    }
                }
            
            # Best execution metrics
            price_improvement_vs_market = 0.005  # 5 bps improvement
            timing_optimization = 0.85  # 85% optimal timing
            liquidity_optimization = 0.78  # 78% optimal liquidity usage
            
            return BestExecution(
                execution_score=total_score,
                price_improvement_vs_market=price_improvement_vs_market,
                timing_optimization=timing_optimization,
                liquidity_optimization=liquidity_optimization,
                algorithm_performance=algorithm_performance,
                venue_analysis=venue_analysis
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating best execution: {e}")
            raise
    
    async def _calculate_liquidity_analysis(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate liquidity analysis"""
        try:
            if isinstance(trade_data, list):
                # Portfolio liquidity analysis
                total_volume = sum(trade.get('quantity', 1000) for trade in trade_data)
                avg_order_size = np.mean([trade.get('quantity', 1000) for trade in trade_data])
                
                # Mock liquidity metrics
                liquidity_metrics = {
                    'average_daily_volume': 10000000,
                    'order_to_volume_ratio': total_volume / 10000000,
                    'liquidity_availability_score': 0.85,
                    'market_depth_score': 0.78,
                    'spread_impact_score': 0.82
                }
                
            else:
                # Single trade liquidity analysis
                quantity = trade_data.get('quantity', 1000)
                liquidity_metrics = {
                    'average_daily_volume': 50000000,
                    'order_to_volume_ratio': quantity / 50000000,
                    'liquidity_availability_score': 0.88,
                    'market_depth_score': 0.82,
                    'spread_impact_score': 0.85
                }
            
            # Additional liquidity insights
            insights = {
                'liquidity_risk_level': 'Low' if liquidity_metrics['order_to_volume_ratio'] < 0.01 else 'Medium',
                'optimal_order_size_range': '1000-5000 shares',
                'liquidity_concentration': 0.65,
                'cross_venue_liquidity': 0.72
            }
            
            return {**liquidity_metrics, **insights}
            
        except Exception as e:
            self.logger.error(f"Error calculating liquidity analysis: {e}")
            raise
    
    async def _generate_execution_summary(self,
                                        shortfall: ImplementationShortfall,
                                        impact: MarketImpact,
                                        costs: TransactionCosts,
                                        vwap: VWAPAnalysis,
                                        slippage: SlippageAnalysis,
                                        best_exec: BestExecution,
                                        liquidity: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive execution summary"""
        try:
            # Calculate overall execution score
            scores = [
                shortfall.execution_vs_benchmark * 1000 + 50,  # Shortfall score
                vwap.execution_quality_score,
                best_exec.execution_score,
                (1 - abs(slippage.average_slippage)) * 100  # Slippage score
            ]
            
            overall_score = np.mean([s for s in scores if s >= 0 and s <= 100])
            
            # Key insights
            insights = {
                'execution_quality_rating': 'Excellent' if overall_score > 85 else 'Good' if overall_score > 70 else 'Fair' if overall_score > 55 else 'Poor',
                'primary_cost_driver': self._identify_primary_cost_driver(costs),
                'best_performing_algorithm': max(best_exec.algorithm_performance.items(), key=lambda x: x[1])[0] if best_exec.algorithm_performance else 'N/A',
                'liquidity_utilization': liquidity.get('liquidity_availability_score', 0),
                'cost_efficiency': (1 - costs.cost_percentage) * 100
            }
            
            # Recommendations
            recommendations = []
            if costs.market_impact_cost / costs.total_cost > 0.5:
                recommendations.append("Consider reducing order size or using more passive execution algorithms")
            
            if slippage.average_slippage > 0.005:
                recommendations.append("Focus on better timing and liquidity analysis")
            
            if vwap.execution_quality_score < 70:
                recommendations.append("Improve VWAP algorithm parameters and execution timing")
            
            if best_exec.execution_score < 75:
                recommendations.append("Review venue selection and algorithm selection process")
            
            # Risk indicators
            risk_indicators = {
                'high_cost_risk': costs.cost_percentage > 0.01,
                'high_impact_risk': impact.total_impact > 0.01,
                'high_slippage_risk': abs(slippage.average_slippage) > 0.008,
                'liquidity_risk': liquidity.get('order_to_volume_ratio', 0) > 0.05
            }
            
            return {
                'overall_execution_score': overall_score,
                'key_insights': insights,
                'recommendations': recommendations,
                'risk_indicators': risk_indicators,
                'benchmark_comparison': {
                    'vs_industry_average': overall_score - 75,  # Mock industry average
                    'percentile_ranking': min(95, max(5, overall_score * 0.95))
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating execution summary: {e}")
            raise
    
    def _identify_primary_cost_driver(self, costs: TransactionCosts) -> str:
        """Identify the primary cost driver"""
        cost_components = {
            'market_impact': costs.market_impact_cost,
            'opportunity': costs.opportunity_cost,
            'timing': costs.timing_cost,
            'commission': costs.commission_cost,
            'spread': costs.spread_cost
        }
        
        return max(cost_components.items(), key=lambda x: x[1])[0]
    
    async def update_real_time_metrics(self):
        """Update real-time execution metrics"""
        try:
            self.logger.debug("Updating real-time execution metrics")
            # Update real-time TCA metrics for dashboard
            pass
        except Exception as e:
            self.logger.error(f"Error updating real-time metrics: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for execution quality analyzer"""
        try:
            return {
                'status': 'healthy',
                'last_update': datetime.now().isoformat(),
                'venues_available': len(self.trading_venues),
                'algorithms_supported': len(self.execution_algorithms)
            }
        except Exception as e:
            self.logger.error(f"Error in execution quality health check: {e}")
            return {'status': 'error', 'error': str(e)}