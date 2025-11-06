"""
Execution Quality Analytics
Institutional-grade trade execution analysis and market impact calculations
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ExecutionAnalytics:
    """Advanced execution quality analysis"""
    
    @staticmethod
    def implementation_shortfall(
        decision_price: float,
        arrival_price: float, 
        execution_price: float,
        trade_value: float,
        commission: float = 0.0,
        market_impact: float = 0.0
    ) -> Dict[str, float]:
        """
        Implementation Shortfall Analysis
        
        Measures the difference between paper portfolio return and actual return
        Components:
        - Market Impact: Price movement due to trade
        - Timing Cost: Delay between decision and execution
        - Commission Cost: Direct trading fees
        - Opportunity Cost: Missed gains from unfilled orders
        """
        
        # Calculate components
        paper_return = (arrival_price - decision_price) / decision_price
        execution_return = (execution_price - decision_price) / decision_price
        
        # Implementation shortfall components
        market_impact_cost = market_impact / decision_price if decision_price > 0 else 0
        timing_cost = (arrival_price - decision_price) / decision_price if decision_price > 0 else 0
        commission_cost = commission / trade_value if trade_value > 0 else 0
        
        # Total implementation shortfall
        total_shortfall = paper_return - execution_return + commission_cost
        
        return {
            'decision_price': decision_price,
            'arrival_price': arrival_price,
            'execution_price': execution_price,
            'paper_return': paper_return * 100,  # As percentage
            'execution_return': execution_return * 100,
            'implementation_shortfall_bps': total_shortfall * 10000,  # In basis points
            'market_impact_bps': market_impact_cost * 10000,
            'timing_cost_bps': timing_cost * 10000,
            'commission_cost_bps': commission_cost * 10000,
            'trade_value': trade_value,
            'commission': commission
        }
    
    @staticmethod
    def vwap_analysis(
        execution_price: float,
        vwap_price: float,
        trade_volume: float,
        total_volume: float,
        side: str = 'buy'  # 'buy' or 'sell'
    ) -> Dict[str, float]:
        """
        Volume Weighted Average Price (VWAP) Analysis
        
        Compares execution price to VWAP benchmark
        """
        
        # VWAP comparison
        vwap_diff = execution_price - vwap_price
        vwap_diff_pct = (vwap_diff / vwap_price) * 100 if vwap_price > 0 else 0
        vwap_diff_bps = vwap_diff_pct * 100
        
        # Adjust for trade side
        if side.lower() == 'sell':
            vwap_diff *= -1
            vwap_diff_pct *= -1
            vwap_diff_bps *= -1
        
        # Volume participation
        volume_participation = (trade_volume / total_volume) * 100 if total_volume > 0 else 0
        
        return {
            'execution_price': execution_price,
            'vwap_price': vwap_price,
            'vwap_difference': vwap_diff,
            'vwap_difference_pct': vwap_diff_pct,
            'vwap_difference_bps': vwap_diff_bps,
            'trade_volume': trade_volume,
            'total_volume': total_volume,
            'volume_participation_pct': volume_participation,
            'trade_side': side,
            'performance_vs_vwap': 'Outperformed' if vwap_diff_bps < 0 else 'Underperformed'
        }
    
    @staticmethod
    def twap_analysis(
        execution_times: List[datetime],
        execution_prices: List[float],
        execution_volumes: List[float],
        benchmark_start: datetime,
        benchmark_end: datetime,
        benchmark_prices: List[float],
        benchmark_times: List[datetime]
    ) -> Dict[str, float]:
        """
        Time Weighted Average Price (TWAP) Analysis
        
        Compares execution against time-weighted benchmark
        """
        
        if not execution_prices or not benchmark_prices:
            return {'error': 'Insufficient data for TWAP analysis'}
        
        # Calculate execution TWAP
        total_volume = sum(execution_volumes)
        if total_volume == 0:
            return {'error': 'Zero total volume'}
            
        execution_twap = sum(p * v for p, v in zip(execution_prices, execution_volumes)) / total_volume
        
        # Calculate benchmark TWAP
        benchmark_twap = np.mean(benchmark_prices)
        
        # TWAP performance
        twap_diff = execution_twap - benchmark_twap
        twap_diff_pct = (twap_diff / benchmark_twap) * 100 if benchmark_twap > 0 else 0
        twap_diff_bps = twap_diff_pct * 100
        
        # Execution time analysis
        execution_duration = (max(execution_times) - min(execution_times)).total_seconds() / 60  # minutes
        benchmark_duration = (benchmark_end - benchmark_start).total_seconds() / 60
        
        return {
            'execution_twap': execution_twap,
            'benchmark_twap': benchmark_twap,
            'twap_difference': twap_diff,
            'twap_difference_pct': twap_diff_pct,
            'twap_difference_bps': twap_diff_bps,
            'execution_duration_minutes': execution_duration,
            'benchmark_duration_minutes': benchmark_duration,
            'total_executed_volume': total_volume,
            'number_of_executions': len(execution_prices),
            'average_execution_size': total_volume / len(execution_prices) if execution_prices else 0
        }
    
    @staticmethod
    def market_impact_analysis(
        pre_trade_price: float,
        post_trade_price: float,
        trade_size: float,
        average_daily_volume: float,
        trade_duration_minutes: float = 1.0
    ) -> Dict[str, float]:
        """
        Market Impact Analysis
        
        Measures price impact from trading activity
        """
        
        # Participation rate
        participation_rate = trade_size / average_daily_volume if average_daily_volume > 0 else 0
        
        # Price impact
        price_impact = post_trade_price - pre_trade_price
        price_impact_pct = (price_impact / pre_trade_price) * 100 if pre_trade_price > 0 else 0
        price_impact_bps = price_impact_pct * 100
        
        # Impact per share
        impact_per_share = price_impact / trade_size if trade_size > 0 else 0
        
        # Temporary vs Permanent impact estimation
        # Temporary impact typically decays, permanent impact persists
        # This is a simplified model - real implementation would use market data
        temporary_impact_ratio = min(0.7, participation_rate * 2)  # Higher participation = more temporary
        temporary_impact = price_impact * temporary_impact_ratio
        permanent_impact = price_impact * (1 - temporary_impact_ratio)
        
        return {
            'pre_trade_price': pre_trade_price,
            'post_trade_price': post_trade_price,
            'price_impact': price_impact,
            'price_impact_pct': price_impact_pct,
            'price_impact_bps': price_impact_bps,
            'trade_size': trade_size,
            'average_daily_volume': average_daily_volume,
            'participation_rate_pct': participation_rate * 100,
            'impact_per_share': impact_per_share,
            'temporary_impact': temporary_impact,
            'permanent_impact': permanent_impact,
            'temporary_impact_bps': (temporary_impact / pre_trade_price) * 10000 if pre_trade_price > 0 else 0,
            'permanent_impact_bps': (permanent_impact / pre_trade_price) * 10000 if pre_trade_price > 0 else 0,
            'trade_duration_minutes': trade_duration_minutes
        }
    
    @staticmethod
    def execution_quality_scorecard(
        trades_data: List[Dict]
    ) -> Dict[str, any]:
        """
        Generate comprehensive execution quality scorecard
        
        Expected trades_data format:
        [
            {
                'decision_price': float,
                'arrival_price': float,
                'execution_price': float,
                'trade_value': float,
                'commission': float,
                'vwap': float,
                'volume': float,
                'total_volume': float,
                'side': str,
                'timestamp': datetime
            }, ...
        ]
        """
        
        if not trades_data:
            return {'error': 'No trades data provided'}
        
        # Aggregate metrics
        total_trades = len(trades_data)
        total_value = sum(trade.get('trade_value', 0) for trade in trades_data)
        total_commission = sum(trade.get('commission', 0) for trade in trades_data)
        
        # Calculate individual trade metrics
        implementation_shortfalls = []
        vwap_analyses = []
        
        for trade in trades_data:
            # Implementation shortfall
            if all(k in trade for k in ['decision_price', 'arrival_price', 'execution_price', 'trade_value']):
                is_result = ExecutionAnalytics.implementation_shortfall(
                    trade['decision_price'],
                    trade['arrival_price'],
                    trade['execution_price'],
                    trade['trade_value'],
                    trade.get('commission', 0)
                )
                implementation_shortfalls.append(is_result)
            
            # VWAP analysis
            if all(k in trade for k in ['execution_price', 'vwap', 'volume', 'total_volume']):
                vwap_result = ExecutionAnalytics.vwap_analysis(
                    trade['execution_price'],
                    trade['vwap'],
                    trade['volume'],
                    trade['total_volume'],
                    trade.get('side', 'buy')
                )
                vwap_analyses.append(vwap_result)
        
        # Aggregate implementation shortfall metrics
        avg_shortfall_bps = np.mean([is_['implementation_shortfall_bps'] for is_ in implementation_shortfalls]) if implementation_shortfalls else 0
        avg_market_impact_bps = np.mean([is_['market_impact_bps'] for is_ in implementation_shortfalls]) if implementation_shortfalls else 0
        avg_timing_cost_bps = np.mean([is_['timing_cost_bps'] for is_ in implementation_shortfalls]) if implementation_shortfalls else 0
        
        # Aggregate VWAP metrics
        avg_vwap_diff_bps = np.mean([vwap['vwap_difference_bps'] for vwap in vwap_analyses]) if vwap_analyses else 0
        vwap_outperformance_rate = (sum(1 for vwap in vwap_analyses if vwap['vwap_difference_bps'] < 0) / len(vwap_analyses) * 100) if vwap_analyses else 0
        
        # Commission analysis
        commission_rate_bps = (total_commission / total_value) * 10000 if total_value > 0 else 0
        
        return {
            'summary': {
                'total_trades': total_trades,
                'total_value': total_value,
                'total_commission': total_commission,
                'commission_rate_bps': commission_rate_bps,
                'avg_trade_size': total_value / total_trades if total_trades > 0 else 0
            },
            'implementation_shortfall': {
                'avg_shortfall_bps': avg_shortfall_bps,
                'avg_market_impact_bps': avg_market_impact_bps,
                'avg_timing_cost_bps': avg_timing_cost_bps,
                'best_execution_bps': min([is_['implementation_shortfall_bps'] for is_ in implementation_shortfalls]) if implementation_shortfalls else 0,
                'worst_execution_bps': max([is_['implementation_shortfall_bps'] for is_ in implementation_shortfalls]) if implementation_shortfalls else 0
            },
            'vwap_performance': {
                'avg_vwap_difference_bps': avg_vwap_diff_bps,
                'outperformance_rate_pct': vwap_outperformance_rate,
                'best_vwap_performance_bps': min([vwap['vwap_difference_bps'] for vwap in vwap_analyses]) if vwap_analyses else 0,
                'worst_vwap_performance_bps': max([vwap['vwap_difference_bps'] for vwap in vwap_analyses]) if vwap_analyses else 0
            },
            'execution_quality_grade': ExecutionAnalytics._calculate_execution_grade(
                avg_shortfall_bps, avg_vwap_diff_bps, commission_rate_bps
            ),
            'detailed_trades': {
                'implementation_shortfall_details': implementation_shortfalls,
                'vwap_analysis_details': vwap_analyses
            }
        }
    
    @staticmethod
    def _calculate_execution_grade(
        avg_shortfall_bps: float,
        avg_vwap_diff_bps: float,
        commission_rate_bps: float
    ) -> str:
        """Calculate overall execution quality grade"""
        
        # Scoring system (lower is better for costs)
        score = 0
        
        # Implementation shortfall scoring
        if avg_shortfall_bps <= 5:
            score += 30  # Excellent
        elif avg_shortfall_bps <= 15:
            score += 25  # Good
        elif avg_shortfall_bps <= 30:
            score += 20  # Average
        else:
            score += 10  # Poor
        
        # VWAP performance scoring
        if abs(avg_vwap_diff_bps) <= 3:
            score += 30  # Excellent
        elif abs(avg_vwap_diff_bps) <= 8:
            score += 25  # Good
        elif abs(avg_vwap_diff_bps) <= 15:
            score += 20  # Average
        else:
            score += 10  # Poor
        
        # Commission efficiency scoring
        if commission_rate_bps <= 2:
            score += 40  # Excellent
        elif commission_rate_bps <= 5:
            score += 35  # Good
        elif commission_rate_bps <= 10:
            score += 25  # Average
        else:
            score += 15  # Poor
        
        # Convert to letter grade
        if score >= 85:
            return 'A+'
        elif score >= 80:
            return 'A'
        elif score >= 75:
            return 'A-'
        elif score >= 70:
            return 'B+'
        elif score >= 65:
            return 'B'
        elif score >= 60:
            return 'B-'
        elif score >= 55:
            return 'C+'
        elif score >= 50:
            return 'C'
        else:
            return 'D'