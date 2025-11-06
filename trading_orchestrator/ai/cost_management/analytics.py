"""
Cost Analytics - Comprehensive cost analysis and reporting system

Provides detailed analytics, trends, and insights for LLM cost optimization.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import sqlite3
import statistics
from pathlib import Path
import json

from loguru import logger


@dataclass
class CostTrend:
    """Cost trend analysis result"""
    period: str  # daily, weekly, monthly
    start_date: datetime
    end_date: datetime
    total_cost: float
    total_tokens: int
    request_count: int
    avg_cost_per_request: float
    cost_change_percentage: float
    token_change_percentage: float
    trend_direction: str  # increasing, decreasing, stable


@dataclass
class CostInsight:
    """Cost insight from analytics"""
    type: str  # anomaly, trend, optimization, efficiency
    title: str
    description: str
    impact: str  # high, medium, low
    actionable: bool
    recommendation: str
    potential_savings: float
    confidence: float


class CostAnalytics:
    """
    Advanced cost analytics and reporting system
    
    Features:
    - Historical cost trend analysis
    - Usage pattern identification
    - Cost anomaly detection
    - Efficiency optimization insights
    - Performance benchmarking
    - Custom report generation
    - Predictive analytics
    """
    
    def __init__(self, cost_manager, provider_manager):
        """
        Initialize cost analytics
        
        Args:
            cost_manager: LLMCostManager instance
            provider_manager: ProviderManager instance
        """
        self.cost_manager = cost_manager
        self.provider_manager = provider_manager
        self.database_path = cost_manager.database_path
        
        # Analytics configuration
        self.anomaly_threshold = 2.0  # Standard deviations
        self.trend_analysis_periods = ['hourly', 'daily', 'weekly', 'monthly']
        
        logger.info("Cost Analytics initialized")
    
    async def get_cost_trends(
        self,
        start_date: datetime,
        end_date: datetime,
        group_by: str = 'daily'
    ) -> List[CostTrend]:
        """
        Get cost trends for specified period
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            group_by: Grouping period (hourly, daily, weekly, monthly)
            
        Returns:
            List of CostTrend objects
        """
        trends = []
        
        # Get cost data from database
        cost_data = await self._get_cost_data_by_period(start_date, end_date, group_by)
        
        # Calculate trends
        for i, data in enumerate(cost_data):
            trend = await self._calculate_trend(data, cost_data, i)
            trends.append(trend)
        
        return trends
    
    async def _get_cost_data_by_period(
        self,
        start_date: datetime,
        end_date: datetime,
        group_by: str
    ) -> List[Dict[str, Any]]:
        """Get cost data grouped by specified period"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Format for grouping
        if group_by == 'hourly':
            date_format = "%Y-%m-%d %H:00:00"
        elif group_by == 'daily':
            date_format = "%Y-%m-%d"
        elif group_by == 'weekly':
            date_format = "%Y-%W"  # ISO week
        else:  # monthly
            date_format = "%Y-%m"
        
        cursor.execute(f"""
            SELECT 
                strftime('{date_format}', timestamp) as period,
                SUM(cost) as total_cost,
                SUM(tokens_used) as total_tokens,
                COUNT(*) as request_count
            FROM cost_events 
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY period
            ORDER BY period
        """, (start_date, end_date))
        
        results = cursor.fetchall()
        conn.close()
        
        # Convert to structured data
        cost_data = []
        for period, total_cost, total_tokens, request_count in results:
            cost_data.append({
                'period': period,
                'total_cost': total_cost or 0.0,
                'total_tokens': total_tokens or 0,
                'request_count': request_count or 0,
                'avg_cost_per_request': (total_cost or 0.0) / max(request_count or 1, 1)
            })
        
        return cost_data
    
    async def _calculate_trend(
        self,
        current_data: Dict[str, Any],
        all_data: List[Dict[str, Any]],
        index: int
    ) -> CostTrend:
        """Calculate trend metrics for a data point"""
        
        # Calculate period boundaries
        period = current_data['period']
        if index > 0:
            prev_data = all_data[index - 1]
            prev_cost = prev_data['total_cost']
            prev_tokens = prev_data['total_tokens']
        else:
            prev_cost = current_data['total_cost']
            prev_tokens = current_data['total_tokens']
        
        # Calculate changes
        cost_change = (
            ((current_data['total_cost'] - prev_cost) / max(prev_cost, 0.01)) * 100
        )
        token_change = (
            ((current_data['total_tokens'] - prev_tokens) / max(prev_tokens, 1)) * 100
        )
        
        # Determine trend direction
        if abs(cost_change) < 5:  # Less than 5% change is stable
            trend_direction = "stable"
        elif cost_change > 0:
            trend_direction = "increasing"
        else:
            trend_direction = "decreasing"
        
        return CostTrend(
            period=period,
            start_date=datetime.now(),  # Simplified - would parse actual period
            end_date=datetime.now(),
            total_cost=current_data['total_cost'],
            total_tokens=current_data['total_tokens'],
            request_count=current_data['request_count'],
            avg_cost_per_request=current_data['avg_cost_per_request'],
            cost_change_percentage=cost_change,
            token_change_percentage=token_change,
            trend_direction=trend_direction
        )
    
    async def detect_cost_anomalies(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[CostInsight]:
        """Detect cost anomalies in the specified period"""
        anomalies = []
        
        # Get hourly cost data for anomaly detection
        cost_data = await self._get_cost_data_by_period(start_date, end_date, 'hourly')
        
        if len(cost_data) < 24:  # Need at least 24 hours of data
            return anomalies
        
        # Calculate statistical metrics
        costs = [data['total_cost'] for data in cost_data]
        tokens = [data['total_tokens'] for data in cost_data]
        
        cost_mean = statistics.mean(costs)
        cost_std = statistics.stdev(costs) if len(costs) > 1 else 0
        
        token_mean = statistics.mean(tokens)
        token_std = statistics.stdev(tokens) if len(tokens) > 1 else 0
        
        # Detect anomalies
        for data in cost_data:
            cost_z_score = abs((data['total_cost'] - cost_mean) / max(cost_std, 0.01))
            token_z_score = abs((data['total_tokens'] - token_mean) / max(token_std, 1))
            
            if cost_z_score > self.anomaly_threshold:
                anomaly_type = "high_cost" if data['total_cost'] > cost_mean else "low_cost"
                
                anomalies.append(CostInsight(
                    type="anomaly",
                    title=f"Cost anomaly detected: {anomaly_type.replace('_', ' ').title()}",
                    description=f"Unusual cost pattern detected at {data['period']}: ${data['total_cost']:.4f}",
                    impact="high" if cost_z_score > 3.0 else "medium",
                    actionable=True,
                    recommendation="Review usage patterns and identify root cause",
                    potential_savings=data['total_cost'] * 0.3 if anomaly_type == "high_cost" else 0,
                    confidence=min(cost_z_score / self.anomaly_threshold, 1.0)
                ))
            
            if token_z_score > self.anomaly_threshold:
                anomaly_type = "high_tokens" if data['total_tokens'] > token_mean else "low_tokens"
                
                anomalies.append(CostInsight(
                    type="anomaly",
                    title=f"Token usage anomaly: {anomaly_type.replace('_', ' ').title()}",
                    description=f"Unusual token usage at {data['period']}: {data['total_tokens']} tokens",
                    impact="medium" if token_z_score > 3.0 else "low",
                    actionable=True,
                    recommendation="Analyze request patterns and token optimization opportunities",
                    potential_savings=0,
                    confidence=min(token_z_score / self.anomaly_threshold, 1.0)
                ))
        
        return anomalies
    
    async def analyze_usage_patterns(self) -> List[CostInsight]:
        """Analyze usage patterns and identify optimization opportunities"""
        insights = []
        
        # Get recent usage data (last 7 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        usage_data = await self._get_cost_data_by_period(start_date, end_date, 'daily')
        
        if not usage_data:
            return insights
        
        # Analyze provider distribution
        provider_breakdown = await self._analyze_provider_distribution(start_date, end_date)
        
        if len(provider_breakdown) > 1:
            # Find expensive providers
            sorted_providers = sorted(
                provider_breakdown.items(),
                key=lambda x: x[1]['total_cost'],
                reverse=True
            )
            
            if sorted_providers:
                most_expensive = sorted_providers[0]
                insights.append(CostInsight(
                    type="optimization",
                    title="Provider cost optimization opportunity",
                    description=f"{most_expensive[0]} accounts for ${most_expensive[1]['total_cost']:.2f} "
                              f"({most_expensive[1]['percentage']:.1f}% of total cost)",
                    impact="high",
                    actionable=True,
                    recommendation="Consider increasing usage of more cost-effective providers",
                    potential_savings=most_expensive[1]['total_cost'] * 0.25,
                    confidence=0.8
                ))
        
        # Analyze request patterns
        request_patterns = await self._analyze_request_patterns(start_date, end_date)
        
        if request_patterns.get('peak_hours'):
            peak_usage = request_patterns['peak_hours']
            insights.append(CostInsight(
                type="efficiency",
                title="Peak usage optimization",
                description=f"Peak usage occurs between {peak_usage['start']}-{peak_usage['end']}",
                impact="medium",
                actionable=True,
                recommendation="Consider batch processing during off-peak hours",
                potential_savings=0.0,  # Would be calculated based on batch processing savings
                confidence=0.7
            ))
        
        return insights
    
    async def _analyze_provider_distribution(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze cost distribution across providers"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                provider,
                SUM(cost) as total_cost,
                SUM(tokens_used) as total_tokens,
                COUNT(*) as request_count
            FROM cost_events 
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY provider
        """, (start_date, end_date))
        
        results = cursor.fetchall()
        conn.close()
        
        total_cost = sum(row[1] for row in results if row[1])
        
        breakdown = {}
        for provider, cost, tokens, requests in results:
            breakdown[provider] = {
                'total_cost': cost or 0.0,
                'total_tokens': tokens or 0,
                'request_count': requests or 0,
                'percentage': ((cost or 0.0) / max(total_cost, 0.01)) * 100
            }
        
        return breakdown
    
    async def _analyze_request_patterns(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Analyze request time patterns"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                strftime('%H', timestamp) as hour,
                COUNT(*) as request_count,
                SUM(cost) as total_cost
            FROM cost_events 
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY strftime('%H', timestamp)
            ORDER BY hour
        """, (start_date, end_date))
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return {}
        
        # Find peak hours
        max_requests = max(row[1] for row in results)
        peak_hours = [row[0] for row in results if row[1] >= max_requests * 0.8]  # Within 20% of peak
        
        return {
            'peak_hours': {
                'start': min(peak_hours),
                'end': max(peak_hours),
                'max_requests': max_requests
            },
            'hourly_distribution': dict(results)
        }
    
    async def generate_cost_report(
        self,
        start_date: datetime,
        end_date: datetime,
        report_type: str = 'comprehensive'
    ) -> Dict[str, Any]:
        """
        Generate comprehensive cost report
        
        Args:
            start_date: Report start date
            end_date: Report end date
            report_type: Type of report (summary, detailed, comprehensive)
            
        Returns:
            Complete cost report
        """
        
        # Get basic metrics
        basic_metrics = await self._get_basic_metrics(start_date, end_date)
        
        # Get trends
        trends = await self.get_cost_trends(start_date, end_date, 'daily')
        
        # Get anomalies
        anomalies = await self.detect_cost_anomalies(start_date, end_date)
        
        # Get usage patterns
        usage_patterns = await self.analyze_usage_patterns()
        
        # Get provider comparison
        provider_comparison = self.provider_manager.get_provider_comparison()
        
        # Compile report
        report = {
            'report_info': {
                'type': report_type,
                'period': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                },
                'generated_at': datetime.now().isoformat()
            },
            'summary': basic_metrics,
            'trends': [trend.__dict__ for trend in trends],
            'anomalies': [anomaly.__dict__ for anomaly in anomalies],
            'insights': [insight.__dict__ for insight in usage_patterns],
            'provider_comparison': provider_comparison,
            'recommendations': await self._generate_recommendations(basic_metrics, trends, usage_patterns)
        }
        
        if report_type == 'detailed':
            report['raw_data'] = await self._get_detailed_data(start_date, end_date)
        
        return report
    
    async def _get_basic_metrics(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get basic cost metrics for the period"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Overall metrics
        cursor.execute("""
            SELECT 
                SUM(cost) as total_cost,
                SUM(tokens_used) as total_tokens,
                COUNT(*) as request_count,
                AVG(cost) as avg_cost_per_request
            FROM cost_events 
            WHERE timestamp BETWEEN ? AND ?
        """, (start_date, end_date))
        
        overall = cursor.fetchone()
        
        # Provider breakdown
        cursor.execute("""
            SELECT 
                provider,
                SUM(cost) as total_cost,
                SUM(tokens_used) as total_tokens,
                COUNT(*) as request_count
            FROM cost_events 
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY provider
        """, (start_date, end_date))
        
        providers = cursor.fetchall()
        
        # Model breakdown
        cursor.execute("""
            SELECT 
                model,
                SUM(cost) as total_cost,
                SUM(tokens_used) as total_tokens,
                COUNT(*) as request_count
            FROM cost_events 
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY model
        """, (start_date, end_date))
        
        models = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_cost': overall[0] or 0.0,
            'total_tokens': overall[1] or 0,
            'request_count': overall[2] or 0,
            'avg_cost_per_request': overall[3] or 0.0,
            'provider_breakdown': {
                row[0]: {
                    'total_cost': row[1] or 0.0,
                    'total_tokens': row[2] or 0,
                    'request_count': row[3] or 0,
                    'percentage': 0.0  # Will be calculated
                } for row in providers
            },
            'model_breakdown': {
                row[0]: {
                    'total_cost': row[1] or 0.0,
                    'total_tokens': row[2] or 0,
                    'request_count': row[3] or 0
                } for row in models
            }
        }
    
    async def _generate_recommendations(
        self,
        metrics: Dict[str, Any],
        trends: List[CostTrend],
        insights: List[CostInsight]
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Cost trend recommendations
        increasing_trends = [t for t in trends if t.trend_direction == "increasing"]
        if increasing_trends:
            recommendations.append({
                'type': 'cost_trend',
                'priority': 'high',
                'title': 'Costs are trending upward',
                'description': f"{len(increasing_trends)} periods show increasing cost trends",
                'action': 'Review usage patterns and implement cost controls'
            })
        
        # Efficiency recommendations
        high_cost_requests = metrics['avg_cost_per_request']
        if high_cost_requests > 0.5:  # $0.50 per request threshold
            recommendations.append({
                'type': 'efficiency',
                'priority': 'medium',
                'title': 'High cost per request',
                'description': f"Average cost per request is ${high_cost_requests:.2f}",
                'action': 'Consider using more cost-effective models or reducing token usage'
            })
        
        # Provider optimization
        if len(metrics['provider_breakdown']) > 1:
            most_expensive_provider = max(
                metrics['provider_breakdown'].items(),
                key=lambda x: x[1]['total_cost']
            )
            
            recommendations.append({
                'type': 'provider_optimization',
                'priority': 'medium',
                'title': 'Provider cost optimization',
                'description': f"${most_expensive_provider[1]['total_cost']:.2f} spent on {most_expensive_provider[0]}",
                'action': 'Consider increasing usage of more cost-effective providers'
            })
        
        return recommendations
    
    async def _get_detailed_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get detailed raw data for analysis"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM cost_events 
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        """, (start_date, end_date))
        
        columns = [description[0] for description in cursor.description]
        events = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            'cost_events': events,
            'total_events': len(events)
        }
    
    async def export_analytics_data(
        self,
        start_date: datetime,
        end_date: datetime,
        format: str = 'json'
    ) -> Dict[str, Any]:
        """Export analytics data in specified format"""
        
        report = await self.generate_cost_report(start_date, end_date, 'detailed')
        
        if format == 'json':
            return report
        elif format == 'summary':
            return {
                'period': report['report_info']['period'],
                'total_cost': report['summary']['total_cost'],
                'total_requests': report['summary']['request_count'],
                'top_providers': list(report['summary']['provider_breakdown'].keys())[:3],
                'key_insights': [insight['title'] for insight in report['insights'][:3]]
            }
        
        return report
    
    async def get_performance_benchmark(self) -> Dict[str, Any]:
        """Get performance benchmark against industry standards"""
        current_metrics = self.cost_manager.get_current_metrics()
        
        # Industry benchmarks (example values - would be based on real data)
        benchmarks = {
            'cost_per_1k_tokens': {
                'industry_average': 0.015,
                'best_practice': 0.008,
                'current': current_metrics.cost_per_1k_tokens
            },
            'avg_response_time': {
                'industry_average': 3.0,  # seconds
                'best_practice': 1.5,
                'current': 2.5  # Would be calculated from actual response times
            },
            'success_rate': {
                'industry_average': 0.95,
                'best_practice': 0.98,
                'current': 0.97  # Would be calculated from actual success rates
            }
        }
        
        # Calculate performance scores
        scores = {}
        for metric, values in benchmarks.items():
            current = values['current']
            industry = values['industry_average']
            best = values['best_practice']
            
            if current <= best:
                score = 1.0
            elif current >= industry:
                score = 0.5
            else:
                # Linear interpolation between best and industry average
                score = 0.5 + (industry - current) / (industry - best) * 0.5
            
            scores[metric] = {
                'score': score,
                'rating': self._get_performance_rating(score),
                'vs_industry': ((current - industry) / max(industry, 0.01)) * 100,
                'vs_best': ((current - best) / max(best, 0.01)) * 100
            }
        
        return {
            'benchmarks': benchmarks,
            'performance_scores': scores,
            'overall_score': statistics.mean([s['score'] for s in scores.values()])
        }
    
    def _get_performance_rating(self, score: float) -> str:
        """Get performance rating based on score"""
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.8:
            return "Good"
        elif score >= 0.6:
            return "Fair"
        elif score >= 0.4:
            return "Poor"
        else:
            return "Critical"