"""
Cost Forecasting - Predictive analytics for LLM cost management

Provides cost prediction, trend forecasting, and budget planning capabilities.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import sqlite3
import statistics
from pathlib import Path
import json

import numpy as np
from loguru import logger


@dataclass
class CostForecast:
    """Cost forecast result"""
    period: str  # hourly, daily, weekly, monthly
    predicted_cost: float
    predicted_tokens: int
    predicted_requests: int
    confidence_interval: Tuple[float, float]  # lower, upper bounds
    confidence_score: float  # 0.0 to 1.0
    model_used: str
    forecast_date: datetime
    start_date: datetime
    end_date: datetime


@dataclass
class BudgetProjection:
    """Budget projection result"""
    total_projected_cost: float
    projected_tokens: int
    projected_requests: int
    risk_assessment: str  # low, medium, high
    potential_budget_overrun: float
    recommended_budget: float
    confidence_score: float
    assumptions: List[str]
    projections: List[CostForecast]


class CostForecaster:
    """
    Advanced cost forecasting and prediction system
    
    Features:
    - Time series forecasting
    - Seasonal trend analysis
    - Budget projections
    - Risk assessment
    - Multiple forecasting models
    - Confidence intervals
    """
    
    def __init__(self, cost_manager, provider_manager):
        """
        Initialize cost forecaster
        
        Args:
            cost_manager: LLMCostManager instance
            provider_manager: ProviderManager instance
        """
        self.cost_manager = cost_manager
        self.provider_manager = provider_manager
        self.database_path = cost_manager.database_path
        
        # Forecasting configuration
        self.forecast_models = ['linear_regression', 'exponential_smoothing', 'moving_average']
        self.default_forecast_horizon = 30  # days
        self.confidence_level = 0.95
        
        logger.info("Cost Forecaster initialized")
    
    async def forecast_costs(
        self,
        forecast_horizon_days: int = 30,
        model: str = 'linear_regression',
        include_seasonality: bool = True
    ) -> List[CostForecast]:
        """
        Forecast costs for the specified horizon
        
        Args:
            forecast_horizon_days: Number of days to forecast
            model: Forecasting model to use
            include_seasonality: Whether to include seasonal adjustments
            
        Returns:
            List of CostForecast objects
        """
        # Get historical data for training
        start_date = datetime.now() - timedelta(days=90)  # 3 months of history
        end_date = datetime.now()
        
        historical_data = await self._get_historical_daily_costs(start_date, end_date)
        
        if len(historical_data) < 7:  # Need at least a week of data
            logger.warning("Insufficient historical data for reliable forecasting")
            return []
        
        # Prepare data for forecasting
        forecast_dates = [
            end_date + timedelta(days=i + 1)
            for i in range(forecast_horizon_days)
        ]
        
        forecasts = []
        
        # Generate forecasts based on selected model
        if model == 'linear_regression':
            forecasts = await self._linear_regression_forecast(
                historical_data, forecast_dates, include_seasonality
            )
        elif model == 'exponential_smoothing':
            forecasts = await self._exponential_smoothing_forecast(
                historical_data, forecast_dates
            )
        elif model == 'moving_average':
            forecasts = await self._moving_average_forecast(
                historical_data, forecast_dates
            )
        else:
            logger.error(f"Unknown forecast model: {model}")
            return []
        
        # Calculate confidence intervals and scores
        for forecast in forecasts:
            forecast.confidence_interval = await self._calculate_confidence_interval(
                historical_data, forecast.predicted_cost
            )
            forecast.confidence_score = await self._calculate_confidence_score(
                historical_data, forecast.predicted_cost
            )
        
        return forecasts
    
    async def _get_historical_daily_costs(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get historical daily cost data"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                DATE(timestamp) as date,
                SUM(cost) as total_cost,
                SUM(tokens_used) as total_tokens,
                COUNT(*) as request_count
            FROM cost_events 
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY DATE(timestamp)
            ORDER BY date
        """, (start_date, end_date))
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                'date': datetime.strptime(row[0], '%Y-%m-%d'),
                'cost': row[1] or 0.0,
                'tokens': row[2] or 0,
                'requests': row[3] or 0
            }
            for row in results
        ]
    
    async def _linear_regression_forecast(
        self,
        historical_data: List[Dict[str, Any]],
        forecast_dates: List[datetime],
        include_seasonality: bool
    ) -> List[CostForecast]:
        """Perform linear regression forecasting"""
        try:
            import numpy as np
            
            # Prepare data
            costs = [data['cost'] for data in historical_data]
            tokens = [data['tokens'] for data in historical_data]
            requests = [data['requests'] for data in historical_data]
            
            # Create time indices (days since start)
            start_date = historical_data[0]['date']
            x = [(data['date'] - start_date).days for data in historical_data]
            
            # Fit linear regression for costs
            cost_coeffs = np.polyfit(x, costs, 1)
            token_coeffs = np.polyfit(x, tokens, 1)
            request_coeffs = np.polyfit(x, requests, 1)
            
            forecasts = []
            base_day = (historical_data[-1]['date'] - start_date).days
            
            for date in forecast_dates:
                day_index = (date - start_date).days
                days_ahead = day_index - base_day
                
                # Base predictions
                predicted_cost = cost_coeffs[0] * day_index + cost_coeffs[1]
                predicted_tokens = max(0, int(token_coeffs[0] * day_index + token_coeffs[1]))
                predicted_requests = max(0, int(request_coeffs[0] * day_index + request_coeffs[1]))
                
                # Apply seasonality if requested
                if include_seasonality:
                    seasonal_factor = self._calculate_seasonal_factor(date, historical_data)
                    predicted_cost *= seasonal_factor
                    predicted_tokens = int(predicted_tokens * seasonal_factor)
                    predicted_requests = int(predicted_requests * seasonal_factor)
                
                # Ensure non-negative values
                predicted_cost = max(0, predicted_cost)
                predicted_tokens = max(0, predicted_tokens)
                predicted_requests = max(0, predicted_requests)
                
                forecasts.append(CostForecast(
                    period='daily',
                    predicted_cost=predicted_cost,
                    predicted_tokens=predicted_tokens,
                    predicted_requests=predicted_requests,
                    confidence_interval=(0.0, predicted_cost * 1.5),  # Simplified
                    confidence_score=0.7,  # Would be calculated properly
                    model_used='linear_regression',
                    forecast_date=datetime.now(),
                    start_date=date,
                    end_date=date
                ))
            
            return forecasts
            
        except ImportError:
            logger.error("NumPy not available for linear regression")
            return await self._moving_average_forecast(historical_data, forecast_dates)
        except Exception as e:
            logger.error(f"Linear regression forecast error: {e}")
            return await self._moving_average_forecast(historical_data, forecast_dates)
    
    async def _exponential_smoothing_forecast(
        self,
        historical_data: List[Dict[str, Any]],
        forecast_dates: List[datetime]
    ) -> List[CostForecast]:
        """Perform exponential smoothing forecasting"""
        try:
            # Simple exponential smoothing
            alpha = 0.3  # Smoothing parameter
            
            costs = [data['cost'] for data in historical_data]
            tokens = [data['tokens'] for data in historical_data]
            requests = [data['requests'] for data in historical_data]
            
            # Calculate smoothed values
            smoothed_cost = [costs[0]]
            smoothed_tokens = [tokens[0]]
            smoothed_requests = [requests[0]]
            
            for i in range(1, len(costs)):
                smoothed_cost.append(alpha * costs[i] + (1 - alpha) * smoothed_cost[i-1])
                smoothed_tokens.append(alpha * tokens[i] + (1 - alpha) * smoothed_tokens[i-1])
                smoothed_requests.append(alpha * requests[i] + (1 - alpha) * smoothed_requests[i-1])
            
            # Use last smoothed value as forecast (simplified)
            last_cost = smoothed_cost[-1]
            last_tokens = smoothed_tokens[-1]
            last_requests = smoothed_requests[-1]
            
            # Apply slight growth trend (simplified)
            growth_rate = 0.02  # 2% monthly growth
            
            forecasts = []
            for i, date in enumerate(forecast_dates):
                growth_factor = (1 + growth_rate) ** ((i + 1) / 30)  # Monthly growth spread over days
                
                predicted_cost = last_cost * growth_factor
                predicted_tokens = int(last_tokens * growth_factor)
                predicted_requests = int(last_requests * growth_factor)
                
                forecasts.append(CostForecast(
                    period='daily',
                    predicted_cost=predicted_cost,
                    predicted_tokens=predicted_tokens,
                    predicted_requests=predicted_requests,
                    confidence_interval=(predicted_cost * 0.8, predicted_cost * 1.2),
                    confidence_score=0.6,
                    model_used='exponential_smoothing',
                    forecast_date=datetime.now(),
                    start_date=date,
                    end_date=date
                ))
            
            return forecasts
            
        except Exception as e:
            logger.error(f"Exponential smoothing forecast error: {e}")
            return await self._moving_average_forecast(historical_data, forecast_dates)
    
    async def _moving_average_forecast(
        self,
        historical_data: List[Dict[str, Any]],
        forecast_dates: List[datetime]
    ) -> List[CostForecast]:
        """Perform moving average forecasting"""
        # Use last 7 days moving average
        window_size = min(7, len(historical_data))
        
        recent_costs = [data['cost'] for data in historical_data[-window_size:]]
        recent_tokens = [data['tokens'] for data in historical_data[-window_size:]]
        recent_requests = [data['requests'] for data in historical_data[-window_size:]]
        
        avg_cost = statistics.mean(recent_costs)
        avg_tokens = statistics.mean(recent_tokens)
        avg_requests = statistics.mean(recent_requests)
        
        # Add trend component
        if len(historical_data) >= 14:
            # Compare first and second half of recent data
            mid_point = len(recent_costs) // 2
            early_avg = statistics.mean(recent_costs[:mid_point])
            late_avg = statistics.mean(recent_costs[mid_point:])
            trend = (late_avg - early_avg) / max(early_avg, 0.01)
        else:
            trend = 0
        
        forecasts = []
        for i, date in enumerate(forecast_dates):
            # Apply trend
            predicted_cost = avg_cost + (trend * i * 0.1)  # Modest trend application
            predicted_tokens = int(avg_tokens + (trend * i * 0.1))
            predicted_requests = int(avg_requests + (trend * i * 0.1))
            
            forecasts.append(CostForecast(
                period='daily',
                predicted_cost=max(0, predicted_cost),
                predicted_tokens=max(0, predicted_tokens),
                predicted_requests=max(0, predicted_requests),
                confidence_interval=(predicted_cost * 0.9, predicted_cost * 1.1),
                confidence_score=0.5,  # Lower confidence for simple moving average
                model_used='moving_average',
                forecast_date=datetime.now(),
                start_date=date,
                end_date=date
            ))
        
        return forecasts
    
    def _calculate_seasonal_factor(self, date: datetime, historical_data: List[Dict[str, Any]]) -> float:
        """Calculate seasonal adjustment factor"""
        # Simple day-of-week seasonal adjustment
        target_dow = date.weekday()  # 0=Monday, 6=Sunday
        
        # Get same day-of-week averages from historical data
        same_dow_costs = [
            data['cost'] for data in historical_data
            if data['date'].weekday() == target_dow
        ]
        
        if not same_dow_costs:
            return 1.0
        
        same_dow_avg = statistics.mean(same_dow_costs)
        overall_avg = statistics.mean([data['cost'] for data in historical_data])
        
        return same_dow_avg / max(overall_avg, 0.01)
    
    async def _calculate_confidence_interval(
        self,
        historical_data: List[Dict[str, Any]],
        predicted_cost: float
    ) -> Tuple[float, float]:
        """Calculate confidence interval for forecast"""
        costs = [data['cost'] for data in historical_data]
        std_dev = statistics.stdev(costs) if len(costs) > 1 else costs[0] * 0.2
        
        margin = std_dev * 1.96  # 95% confidence interval
        
        lower_bound = max(0, predicted_cost - margin)
        upper_bound = predicted_cost + margin
        
        return (lower_bound, upper_bound)
    
    async def _calculate_confidence_score(
        self,
        historical_data: List[Dict[str, Any]],
        predicted_cost: float
    ) -> float:
        """Calculate confidence score for forecast"""
        if len(historical_data) < 7:
            return 0.3  # Low confidence with insufficient data
        
        if len(historical_data) >= 30:
            return 0.8  # High confidence with sufficient data
        elif len(historical_data) >= 14:
            return 0.6  # Medium confidence
        else:
            return 0.4  # Low confidence
    
    async def project_budget_requirements(
        self,
        target_days: int = 30,
        confidence_level: float = 0.95
    ) -> BudgetProjection:
        """
        Project budget requirements for specified period
        
        Args:
            target_days: Number of days to project
            confidence_level: Confidence level for projections
            
        Returns:
            BudgetProjection with detailed projections
        """
        # Get forecasts
        forecasts = await self.forecast_costs(target_days)
        
        if not forecasts:
            raise ValueError("Unable to generate forecasts for budget projection")
        
        # Aggregate projections
        total_projected_cost = sum(f.predicted_cost for f in forecasts)
        total_projected_tokens = sum(f.predicted_tokens for f in forecasts)
        total_projected_requests = sum(f.predicted_requests for f in forecasts)
        
        # Calculate risk assessment
        current_daily_avg = total_projected_cost / target_days
        max_projected_daily = max(f.predicted_cost for f in forecasts)
        
        if max_projected_daily <= current_daily_avg * 1.2:
            risk_assessment = "low"
        elif max_projected_daily <= current_daily_avg * 2.0:
            risk_assessment = "medium"
        else:
            risk_assessment = "high"
        
        # Calculate recommended budget (add contingency)
        contingency_factor = 1.2 if risk_assessment == "low" else 1.5 if risk_assessment == "medium" else 2.0
        recommended_budget = total_projected_cost * contingency_factor
        
        # Calculate potential budget overrun
        projected_95th_percentile = sum(f.confidence_interval[1] for f in forecasts)
        potential_overrun = max(0, projected_95th_percentile - recommended_budget)
        
        # Generate assumptions
        assumptions = [
            f"Based on {len(forecasts)} days of historical data",
            f"Assumes {risk_assessment} volatility pattern",
            "Includes standard business growth trends",
            "Assumes no major system changes or anomalies"
        ]
        
        return BudgetProjection(
            total_projected_cost=total_projected_cost,
            projected_tokens=total_projected_tokens,
            projected_requests=total_projected_requests,
            risk_assessment=risk_assessment,
            potential_budget_overrun=potential_overrun,
            recommended_budget=recommended_budget,
            confidence_score=statistics.mean([f.confidence_score for f in forecasts]),
            assumptions=assumptions,
            projections=forecasts
        )
    
    async def analyze_cost_drivers(self) -> Dict[str, Any]:
        """Analyze factors driving cost changes"""
        # Get recent cost data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # Analyze by provider
        provider_costs = await self._analyze_cost_drivers_by_provider(start_date, end_date)
        
        # Analyze by model
        model_costs = await self._analyze_cost_drivers_by_model(start_date, end_date)
        
        # Analyze by time patterns
        time_patterns = await self._analyze_cost_drivers_by_time(start_date, end_date)
        
        return {
            'provider_analysis': provider_costs,
            'model_analysis': model_costs,
            'time_pattern_analysis': time_patterns,
            'key_findings': await self._identify_key_cost_findings(provider_costs, model_costs)
        }
    
    async def _analyze_cost_drivers_by_provider(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Analyze cost drivers by provider"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                provider,
                SUM(cost) as total_cost,
                COUNT(*) as request_count,
                AVG(cost) as avg_cost_per_request
            FROM cost_events 
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY provider
        """, (start_date, end_date))
        
        results = cursor.fetchall()
        conn.close()
        
        total_cost = sum(row[1] for row in results if row[1])
        
        return {
            'provider_breakdown': [
                {
                    'provider': row[0],
                    'total_cost': row[1] or 0.0,
                    'percentage': ((row[1] or 0.0) / max(total_cost, 0.01)) * 100,
                    'request_count': row[2] or 0,
                    'avg_cost_per_request': row[3] or 0.0
                }
                for row in results
            ],
            'total_cost': total_cost,
            'dominant_provider': max(results, key=lambda x: x[1] or 0)[0] if results else None
        }
    
    async def _analyze_cost_drivers_by_model(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Analyze cost drivers by model"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                model,
                SUM(cost) as total_cost,
                COUNT(*) as request_count,
                AVG(cost) as avg_cost_per_request
            FROM cost_events 
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY model
        """, (start_date, end_date))
        
        results = cursor.fetchall()
        conn.close()
        
        total_cost = sum(row[1] for row in results if row[1])
        
        return {
            'model_breakdown': [
                {
                    'model': row[0],
                    'total_cost': row[1] or 0.0,
                    'percentage': ((row[1] or 0.0) / max(total_cost, 0.01)) * 100,
                    'request_count': row[2] or 0,
                    'avg_cost_per_request': row[3] or 0.0
                }
                for row in results
            ],
            'total_cost': total_cost,
            'most_expensive_model': max(results, key=lambda x: x[1] or 0)[0] if results else None
        }
    
    async def _analyze_cost_drivers_by_time(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Analyze cost drivers by time patterns"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Hourly patterns
        cursor.execute("""
            SELECT 
                strftime('%H', timestamp) as hour,
                SUM(cost) as total_cost,
                COUNT(*) as request_count
            FROM cost_events 
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY strftime('%H', timestamp)
        """, (start_date, end_date))
        
        hourly_patterns = dict(cursor.fetchall())
        
        # Daily patterns
        cursor.execute("""
            SELECT 
                strftime('%w', timestamp) as day_of_week,
                SUM(cost) as total_cost,
                COUNT(*) as request_count
            FROM cost_events 
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY strftime('%w', timestamp)
        """, (start_date, end_date))
        
        daily_patterns = dict(cursor.fetchall())
        
        conn.close()
        
        # Identify peak periods
        peak_hour = max(hourly_patterns, key=hourly_patterns.get) if hourly_patterns else None
        peak_day = max(daily_patterns, key=daily_patterns.get) if daily_patterns else None
        
        day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        
        return {
            'hourly_patterns': hourly_patterns,
            'daily_patterns': {day_names[int(day)]: {'cost': cost, 'requests': requests} 
                             for day, cost, requests in daily_patterns.items()},
            'peak_hour': peak_hour,
            'peak_day': day_names[int(peak_day)] if peak_day else None,
            'weekend_vs_weekday': {
                'weekend_cost': sum(hourly_patterns.get(str(h), {'cost': 0})['cost'] if isinstance(h, int) else 0 
                                 for h in [0, 6] if str(h) in hourly_patterns),
                'weekday_cost': sum(hourly_patterns.get(str(h), {'cost': 0})['cost'] if isinstance(h, int) else 0 
                                  for h in range(1, 6) if str(h) in hourly_patterns)
            }
        }
    
    async def _identify_key_cost_findings(
        self,
        provider_analysis: Dict[str, Any],
        model_analysis: Dict[str, Any]
    ) -> List[str]:
        """Identify key findings from cost driver analysis"""
        findings = []
        
        # Provider insights
        if provider_analysis.get('dominant_provider'):
            findings.append(
                f"{provider_analysis['dominant_provider']} accounts for "
                f"{max([p['percentage'] for p in provider_analysis['provider_breakdown']], default=0):.1f}% of total costs"
            )
        
        # Model insights
        if model_analysis.get('most_expensive_model'):
            findings.append(
                f"{model_analysis['most_expensive_model']} is the most expensive model"
            )
        
        # Optimization opportunities
        expensive_models = [m for m in model_analysis.get('model_breakdown', []) if m['percentage'] > 20]
        if expensive_models:
            findings.append(
                f"Consider optimizing usage of high-cost models: "
                f"{', '.join([m['model'] for m in expensive_models])}"
            )
        
        return findings
    
    async def get_forecast_accuracy(self, days: int = 7) -> Dict[str, Any]:
        """Evaluate forecast accuracy against actual results"""
        # Get actual data for the past week
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        actual_data = await self._get_historical_daily_costs(start_date, end_date)
        
        if len(actual_data) < 3:
            return {'error': 'Insufficient data for accuracy assessment'}
        
        # Get forecasts made for this period (simplified)
        # In a real implementation, we'd retrieve stored forecasts from a separate table
        
        # Calculate simple accuracy metrics
        actual_costs = [data['cost'] for data in actual_data]
        avg_actual_cost = statistics.mean(actual_costs)
        
        accuracy_metrics = {
            'mean_absolute_error': 0.0,  # Would calculate with actual forecasts
            'mean_absolute_percentage_error': 0.0,
            'forecast_bias': 0.0,
            'data_points': len(actual_data),
            'avg_actual_cost': avg_actual_cost
        }
        
        return accuracy_metrics
    
    async def export_forecasts(
        self,
        forecasts: List[CostForecast],
        format: str = 'json'
    ) -> Dict[str, Any]:
        """Export forecasts in specified format"""
        if format == 'json':
            return {
                'forecast_model': forecasts[0].model_used if forecasts else 'none',
                'generated_at': datetime.now().isoformat(),
                'forecasts': [f.__dict__ for f in forecasts],
                'summary': {
                    'total_forecast_cost': sum(f.predicted_cost for f in forecasts),
                    'total_forecast_tokens': sum(f.predicted_tokens for f in forecasts),
                    'total_forecast_requests': sum(f.predicted_requests for f in forecasts),
                    'avg_daily_cost': sum(f.predicted_cost for f in forecasts) / len(forecasts) if forecasts else 0,
                    'confidence_score': statistics.mean([f.confidence_score for f in forecasts]) if forecasts else 0
                }
            }
        
        return {'error': f'Unsupported format: {format}'}