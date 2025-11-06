"""
Maximum Drawdown and Drawdown Analysis Models

Comprehensive drawdown analysis for risk assessment:
- Maximum Drawdown calculation
- Dynamic drawdown tracking
- Drawdown recovery analysis
- Underwater curve analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
import asyncio

from database.models.trading import Position, Trade

logger = logging.getLogger(__name__)


@dataclass
class DrawdownEvent:
    """Represents a drawdown event."""
    start_date: datetime
    trough_date: datetime
    end_date: Optional[datetime]
    peak_value: float
    trough_value: float
    recovery_value: Optional[float]
    max_drawdown_pct: float
    duration_days: int
    recovery_days: Optional[int]
    severity: str  # 'mild', 'moderate', 'severe', 'extreme'


class DrawdownAnalyzer:
    """
    Comprehensive drawdown analysis and tracking.
    
    Analyzes historical drawdowns, recovery patterns, and provides
    insights into portfolio risk and performance degradation.
    """
    
    def __init__(self, min_drawdown_threshold: float = 0.05):
        """
        Initialize Drawdown Analyzer.
        
        Args:
            min_drawdown_threshold: Minimum drawdown percentage to consider (5% default)
        """
        self.min_drawdown_threshold = min_drawdown_threshold
        
    async def analyze_portfolio_drawdowns(self, portfolio_values: pd.Series) -> Dict[str, Any]:
        """
        Analyze all drawdown events in portfolio value series.
        
        Args:
            portfolio_values: Time series of portfolio values
            
        Returns:
            Comprehensive drawdown analysis
        """
        try:
            # Calculate drawdown series
            drawdown_series = self._calculate_drawdown_series(portfolio_values)
            
            # Identify drawdown events
            drawdown_events = self._identify_drawdown_events(portfolio_values, drawdown_series)
            
            # Calculate statistics
            statistics = self._calculate_drawdown_statistics(drawdown_events)
            
            # Generate underwater curve
            underwater_curve = self._create_underwater_curve(portfolio_values, drawdown_series)
            
            # Recovery analysis
            recovery_analysis = self._analyze_recovery_patterns(drawdown_events)
            
            return {
                'drawdown_series': drawdown_series,
                'drawdown_events': [self._event_to_dict(event) for event in drawdown_events],
                'statistics': statistics,
                'underwater_curve': underwater_curve,
                'recovery_analysis': recovery_analysis,
                'total_events': len(drawdown_events),
                'min_threshold': self.min_drawdown_threshold
            }
            
        except Exception as e:
            logger.error(f"Portfolio drawdown analysis error: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_drawdown_series(self, portfolio_values: pd.Series) -> pd.Series:
        """Calculate drawdown series from portfolio values."""
        try:
            # Calculate running maximum
            running_max = portfolio_values.expanding().max()
            
            # Calculate drawdown
            drawdown = (portfolio_values - running_max) / running_max
            
            return drawdown
            
        except Exception as e:
            logger.error(f"Drawdown series calculation error: {str(e)}")
            return pd.Series()
    
    def _identify_drawdown_events(self, portfolio_values: pd.Series, 
                                 drawdown_series: pd.Series) -> List[DrawdownEvent]:
        """Identify individual drawdown events."""
        try:
            events = []
            in_drawdown = False
            event_start = None
            event_peak = None
            
            # Find significant drawdown periods
            for date, drawdown in drawdown_series.items():
                if not in_drawdown:
                    # Check if entering drawdown
                    if drawdown < -self.min_drawdown_threshold:
                        in_drawdown = True
                        event_start = date
                        event_peak = portfolio_values[date]
                else:
                    # Check if still in drawdown
                    if drawdown < -self.min_drawdown_threshold:
                        # Update if this is deeper
                        if portfolio_values[date] < event_peak:
                            event_peak = portfolio_values[date]
                    else:
                        # Drawdown ended - recovery
                        in_drawdown = False
                        if event_start and event_peak:
                            event = DrawdownEvent(
                                start_date=event_start,
                                trough_date=portfolio_values[portfolio_values == event_peak].index[0],
                                end_date=date,
                                peak_value=event_peak,
                                trough_value=event_peak,
                                recovery_value=portfolio_values[date],
                                max_drawdown_pct=abs((event_peak - event_peak) / event_peak),
                                duration_days=(date - event_start).days,
                                recovery_days=(date - event_start).days,
                                severity=self._classify_drawdown_severity(abs((event_peak - event_peak) / event_peak))
                            )
                            events.append(event)
            
            # Handle ongoing drawdown
            if in_drawdown and event_start and event_peak:
                current_value = portfolio_values.iloc[-1]
                current_drawdown = abs((event_peak - current_value) / event_peak)
                
                event = DrawdownEvent(
                    start_date=event_start,
                    trough_date=portfolio_values[portfolio_values == event_peak].index[0],
                    end_date=None,
                    peak_value=event_peak,
                    trough_value=event_peak,
                    recovery_value=current_value,
                    max_drawdown_pct=current_drawdown,
                    duration_days=(portfolio_values.index[-1] - event_start).days,
                    recovery_days=None,
                    severity=self._classify_drawdown_severity(current_drawdown)
                )
                events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Drawdown event identification error: {str(e)}")
            return []
    
    def _classify_drawdown_severity(self, drawdown_pct: float) -> str:
        """Classify drawdown severity based on percentage."""
        if drawdown_pct < 0.1:  # Less than 10%
            return 'mild'
        elif drawdown_pct < 0.2:  # 10-20%
            return 'moderate'
        elif drawdown_pct < 0.35:  # 20-35%
            return 'severe'
        else:  # Above 35%
            return 'extreme'
    
    def _calculate_drawdown_statistics(self, events: List[DrawdownEvent]) -> Dict[str, Any]:
        """Calculate comprehensive drawdown statistics."""
        try:
            if not events:
                return {
                    'max_drawdown': 0.0,
                    'avg_drawdown': 0.0,
                    'total_events': 0,
                    'avg_duration': 0,
                    'avg_recovery_time': 0
                }
            
            max_drawdown = max(event.max_drawdown_pct for event in events)
            avg_drawdown = np.mean([event.max_drawdown_pct for event in events])
            avg_duration = np.mean([event.duration_days for event in events])
            avg_recovery_time = np.mean([event.recovery_days for event in events if event.recovery_days])
            
            # Severity distribution
            severity_counts = {}
            for event in events:
                severity_counts[event.severity] = severity_counts.get(event.severity, 0) + 1
            
            # Recovery rate
            recovery_rate = sum(1 for event in events if event.recovery_days) / len(events)
            
            return {
                'max_drawdown': max_drawdown,
                'avg_drawdown': avg_drawdown,
                'median_drawdown': np.median([event.max_drawdown_pct for event in events]),
                'total_events': len(events),
                'avg_duration_days': avg_duration,
                'median_duration_days': np.median([event.duration_days for event in events]),
                'avg_recovery_time_days': avg_recovery_time if not np.isnan(avg_recovery_time) else None,
                'recovery_rate': recovery_rate,
                'severity_distribution': severity_counts,
                'worst_event': self._get_worst_event(events)
            }
            
        except Exception as e:
            logger.error(f"Drawdown statistics calculation error: {str(e)}")
            return {}
    
    def _get_worst_event(self, events: List[DrawdownEvent]) -> Dict[str, Any]:
        """Get details of the worst drawdown event."""
        try:
            if not events:
                return {}
            
            worst_event = max(events, key=lambda x: x.max_drawdown_pct)
            
            return {
                'max_drawdown_pct': worst_event.max_drawdown_pct,
                'duration_days': worst_event.duration_days,
                'recovery_days': worst_event.recovery_days,
                'severity': worst_event.severity,
                'start_date': worst_event.start_date,
                'trough_date': worst_event.trough_date,
                'end_date': worst_event.end_date
            }
            
        except Exception as e:
            logger.error(f"Worst event calculation error: {str(e)}")
            return {}
    
    def _create_underwater_curve(self, portfolio_values: pd.Series, 
                               drawdown_series: pd.Series) -> pd.Series:
        """Create underwater curve showing cumulative drawdown over time."""
        try:
            # Create underwater curve (cumulative time spent underwater)
            underwater_days = (drawdown_series < 0).astype(int)
            underwater_curve = underwater_days.expanding().sum()
            
            return underwater_curve
            
        except Exception as e:
            logger.error(f"Underwater curve calculation error: {str(e)}")
            return pd.Series()
    
    def _analyze_recovery_patterns(self, events: List[DrawdownEvent]) -> Dict[str, Any]:
        """Analyze recovery patterns and timing."""
        try:
            recovery_events = [event for event in events if event.recovery_days]
            
            if not recovery_events:
                return {'no_recovery_events': True}
            
            recovery_times = [event.recovery_days for event in recovery_events]
            drawdown_severities = [event.max_drawdown_pct for event in recovery_events]
            
            # Correlation between drawdown severity and recovery time
            correlation = np.corrcoef(drawdown_severities, recovery_times)[0, 1] if len(recovery_times) > 1 else 0
            
            # Recovery rate by severity
            recovery_by_severity = {}
            for severity in ['mild', 'moderate', 'severe', 'extreme']:
                severity_events = [event for event in recovery_events if event.severity == severity]
                if severity_events:
                    recovery_by_severity[severity] = {
                        'count': len(severity_events),
                        'avg_recovery_time': np.mean([event.recovery_days for event in severity_events]),
                        'min_recovery_time': min([event.recovery_days for event in severity_events]),
                        'max_recovery_time': max([event.recovery_days for event in severity_events])
                    }
            
            return {
                'total_recovery_events': len(recovery_events),
                'avg_recovery_time_days': np.mean(recovery_times),
                'median_recovery_time_days': np.median(recovery_times),
                'max_recovery_time_days': max(recovery_times),
                'min_recovery_time_days': min(recovery_times),
                'severity_recovery_correlation': correlation,
                'recovery_by_severity': recovery_by_severity
            }
            
        except Exception as e:
            logger.error(f"Recovery pattern analysis error: {str(e)}")
            return {}
    
    def _event_to_dict(self, event: DrawdownEvent) -> Dict[str, Any]:
        """Convert DrawdownEvent to dictionary."""
        return {
            'start_date': event.start_date,
            'trough_date': event.trough_date,
            'end_date': event.end_date,
            'peak_value': event.peak_value,
            'trough_value': event.trough_value,
            'recovery_value': event.recovery_value,
            'max_drawdown_pct': event.max_drawdown_pct,
            'duration_days': event.duration_days,
            'recovery_days': event.recovery_days,
            'severity': event.severity
        }


class MaximumDrawdown:
    """
    Maximum Drawdown calculation and monitoring.
    
    Calculates maximum drawdown, tracks current drawdown, and provides
    early warning systems for drawdown risk.
    """
    
    def __init__(self, portfolio_values: pd.Series):
        """
        Initialize Maximum Drawdown calculator.
        
        Args:
            portfolio_values: Time series of portfolio values
        """
        self.portfolio_values = portfolio_values
        self.max_drawdown = self._calculate_max_drawdown()
        self.current_drawdown = self._calculate_current_drawdown()
        self.peak_value = self._calculate_peak_value()
        
    def _calculate_max_drawdown(self) -> Dict[str, Any]:
        """Calculate maximum drawdown from portfolio values."""
        try:
            if len(self.portfolio_values) == 0:
                return {'max_drawdown_pct': 0.0, 'max_drawdown_amount': 0.0}
            
            running_max = self.portfolio_values.expanding().max()
            drawdown = (self.portfolio_values - running_max) / running_max
            
            max_dd_pct = abs(drawdown.min())
            max_dd_amount = abs(drawdown.min() * running_max[drawdown.idxmin()])
            
            return {
                'max_drawdown_pct': max_dd_pct,
                'max_drawdown_amount': max_dd_amount,
                'start_date': self._get_drawdown_start_date(drawdown),
                'trough_date': drawdown.idxmin(),
                'peak_value': running_max[drawdown.idxmin()],
                'trough_value': self.portfolio_values[drawdown.idxmin()]
            }
            
        except Exception as e:
            logger.error(f"Max drawdown calculation error: {str(e)}")
            return {'max_drawdown_pct': 0.0, 'max_drawdown_amount': 0.0}
    
    def _calculate_current_drawdown(self) -> Dict[str, Any]:
        """Calculate current drawdown from peak."""
        try:
            if len(self.portfolio_values) == 0:
                return {'current_drawdown_pct': 0.0, 'current_drawdown_amount': 0.0}
            
            current_value = self.portfolio_values.iloc[-1]
            peak_value = self.peak_value
            
            if peak_value > 0:
                current_dd_pct = (peak_value - current_value) / peak_value
                current_dd_amount = peak_value - current_value
            else:
                current_dd_pct = 0.0
                current_dd_amount = 0.0
            
            return {
                'current_drawdown_pct': current_dd_pct,
                'current_drawdown_amount': current_dd_amount,
                'peak_value': peak_value,
                'current_value': current_value,
                'days_since_peak': self._calculate_days_since_peak()
            }
            
        except Exception as e:
            logger.error(f"Current drawdown calculation error: {str(e)}")
            return {'current_drawdown_pct': 0.0, 'current_drawdown_amount': 0.0}
    
    def _calculate_peak_value(self) -> float:
        """Calculate peak portfolio value."""
        try:
            if len(self.portfolio_values) == 0:
                return 0.0
            
            return self.portfolio_values.expanding().max().iloc[-1]
            
        except Exception:
            return 0.0
    
    def _get_drawdown_start_date(self, drawdown: pd.Series) -> Optional[datetime]:
        """Get start date of maximum drawdown period."""
        try:
            max_dd_date = drawdown.idxmin()
            
            # Find when portfolio was at peak before drawdown
            running_max = self.portfolio_values.expanding().max()
            peak_date = running_max[running_max == running_max[max_dd_date]].index[0]
            
            return peak_date
            
        except Exception:
            return None
    
    def _calculate_days_since_peak(self) -> int:
        """Calculate days since portfolio peaked."""
        try:
            if len(self.portfolio_values) == 0:
                return 0
            
            peak_value = self.peak_value
            peak_dates = self.portfolio_values[self.portfolio_values == peak_value].index
            
            if len(peak_dates) == 0:
                return 0
            
            last_peak_date = peak_dates[-1]
            days_since_peak = (self.portfolio_values.index[-1] - last_peak_date).days
            
            return days_since_peak
            
        except Exception:
            return 0
    
    def get_risk_warnings(self, thresholds: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """
        Generate risk warnings based on drawdown levels.
        
        Args:
            thresholds: Warning thresholds (e.g., {'warning': 0.05, 'critical': 0.15})
            
        Returns:
            List of risk warnings
        """
        if thresholds is None:
            thresholds = {'warning': 0.05, 'critical': 0.15, 'emergency': 0.25}
        
        warnings = []
        current_dd_pct = self.current_drawdown['current_drawdown_pct']
        
        try:
            # Check for emergency level
            if current_dd_pct >= thresholds['emergency']:
                warnings.append({
                    'level': 'emergency',
                    'message': f"Emergency drawdown level reached: {current_dd_pct:.1%}",
                    'action': 'Immediate risk reduction required',
                    'drawdown_pct': current_dd_pct,
                    'severity': 'critical'
                })
            
            # Check for critical level
            elif current_dd_pct >= thresholds['critical']:
                warnings.append({
                    'level': 'critical',
                    'message': f"Critical drawdown level: {current_dd_pct:.1%}",
                    'action': 'Consider reducing position sizes or liquidating risky positions',
                    'drawdown_pct': current_dd_pct,
                    'severity': 'high'
                })
            
            # Check for warning level
            elif current_dd_pct >= thresholds['warning']:
                warnings.append({
                    'level': 'warning',
                    'message': f"Drawdown warning level: {current_dd_pct:.1%}",
                    'action': 'Monitor positions closely and consider risk mitigation',
                    'drawdown_pct': current_dd_pct,
                    'severity': 'medium'
                })
            
            # Check for extended drawdown periods
            days_since_peak = self.current_drawdown.get('days_since_peak', 0)
            if days_since_peak > 60:  # More than 2 months
                warnings.append({
                    'level': 'extended_drawdown',
                    'message': f"Extended drawdown period: {days_since_peak} days since peak",
                    'action': 'Review portfolio strategy and consider rebalancing',
                    'days_since_peak': days_since_peak,
                    'severity': 'medium'
                })
            
            return warnings
            
        except Exception as e:
            logger.error(f"Risk warning generation error: {str(e)}")
            return []
    
    def get_drawdown_summary(self) -> Dict[str, Any]:
        """Get comprehensive drawdown summary."""
        try:
            return {
                'current_drawdown': self.current_drawdown,
                'maximum_drawdown': self.max_drawdown,
                'risk_warnings': self.get_risk_warnings(),
                'recovery_status': self._assess_recovery_status(),
                'performance_metrics': self._calculate_performance_metrics()
            }
            
        except Exception as e:
            logger.error(f"Drawdown summary error: {str(e)}")
            return {}
    
    def _assess_recovery_status(self) -> Dict[str, Any]:
        """Assess current recovery status."""
        try:
            current_value = self.current_drawdown['current_value']
            peak_value = self.current_drawdown['peak_value']
            
            if peak_value == 0:
                return {'status': 'unknown', 'recovery_progress': 0.0}
            
            recovery_progress = max(0, (current_value / peak_value) * 100)
            
            if recovery_progress >= 100:
                status = 'recovered'
            elif recovery_progress >= 90:
                status = 'near_recovery'
            elif recovery_progress >= 75:
                status = 'partial_recovery'
            else:
                status = 'deep_drawdown'
            
            return {
                'status': status,
                'recovery_progress_pct': recovery_progress,
                'distance_to_peak': peak_value - current_value
            }
            
        except Exception as e:
            logger.error(f"Recovery status assessment error: {str(e)}")
            return {'status': 'unknown', 'recovery_progress_pct': 0.0}
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics during drawdown periods."""
        try:
            if len(self.portfolio_values) == 0:
                return {}
            
            # Calculate drawdown periods
            drawdown_series = self._calculate_drawdown_series()
            
            # Metrics
            total_drawdown_days = (drawdown_series < 0).sum()
            total_days = len(drawdown_series)
            drawdown_frequency = total_drawdown_days / total_days if total_days > 0 else 0
            
            # Average drawdown duration
            drawdown_periods = self._identify_drawdown_periods(drawdown_series)
            avg_duration = np.mean([period['duration'] for period in drawdown_periods]) if drawdown_periods else 0
            
            return {
                'drawdown_frequency': drawdown_frequency,
                'avg_drawdown_duration_days': avg_duration,
                'total_drawdown_days': total_drawdown_days,
                'drawdown_periods_count': len(drawdown_periods)
            }
            
        except Exception as e:
            logger.error(f"Performance metrics calculation error: {str(e)}")
            return {}
    
    def _calculate_drawdown_series(self) -> pd.Series:
        """Helper method to calculate drawdown series."""
        try:
            running_max = self.portfolio_values.expanding().max()
            drawdown = (self.portfolio_values - running_max) / running_max
            return drawdown
            
        except Exception:
            return pd.Series()
    
    def _identify_drawdown_periods(self, drawdown_series: pd.Series) -> List[Dict[str, Any]]:
        """Identify individual drawdown periods."""
        try:
            periods = []
            in_drawdown = False
            start_date = None
            
            for date, drawdown in drawdown_series.items():
                if not in_drawdown and drawdown < -0.01:  # 1% threshold
                    in_drawdown = True
                    start_date = date
                elif in_drawdown and drawdown >= -0.01:  # Recovery
                    in_drawdown = False
                    if start_date:
                        periods.append({
                            'start_date': start_date,
                            'end_date': date,
                            'duration': (date - start_date).days
                        })
            
            return periods
            
        except Exception as e:
            logger.error(f"Drawdown period identification error: {str(e)}")
            return []


class DrawdownMonitoring:
    """
    Real-time drawdown monitoring and alerting.
    
    Monitors portfolio drawdown in real-time and provides
    automated alerts and risk management triggers.
    """
    
    def __init__(self, user_id: int):
        """
        Initialize Drawdown Monitoring.
        
        Args:
            user_id: User identifier for monitoring
        """
        self.user_id = user_id
        self.monitoring_active = False
        self.alert_callbacks = []
        
    def add_alert_callback(self, callback: callable):
        """Add callback function for alerts."""
        self.alert_callbacks.append(callback)
        
    async def start_monitoring(self, portfolio_values: pd.Series, 
                             alert_thresholds: Dict[str, float] = None):
        """
        Start real-time drawdown monitoring.
        
        Args:
            portfolio_values: Portfolio value series
            alert_thresholds: Alert thresholds for different levels
        """
        try:
            self.monitoring_active = True
            md_calculator = MaximumDrawdown(portfolio_values)
            
            # Generate initial warnings
            warnings = md_calculator.get_risk_warnings(alert_thresholds)
            
            # Send alerts
            await self._send_alerts(warnings)
            
            logger.info(f"Drawdown monitoring started for user {self.user_id}")
            
        except Exception as e:
            logger.error(f"Drawdown monitoring start error: {str(e)}")
    
    async def stop_monitoring(self):
        """Stop drawdown monitoring."""
        self.monitoring_active = False
        logger.info(f"Drawdown monitoring stopped for user {self.user_id}")
    
    async def check_drawdown(self, current_portfolio_value: float, 
                           portfolio_history: pd.Series) -> Dict[str, Any]:
        """
        Check current portfolio drawdown and generate alerts.
        
        Args:
            current_portfolio_value: Current portfolio value
            portfolio_history: Historical portfolio values
            
        Returns:
            Current drawdown status and alerts
        """
        try:
            if not self.monitoring_active:
                return {'status': 'monitoring_inactive'}
            
            # Update portfolio values series
            updated_values = portfolio_history.copy()
            if len(updated_values) > 0:
                # Replace last value or append new one
                if hasattr(updated_values, 'index') and len(updated_values) > 0:
                    last_date = updated_values.index[-1]
                    if isinstance(last_date, datetime):
                        updated_values.iloc[-1] = current_portfolio_value
                    else:
                        updated_values = pd.concat([
                            updated_values,
                            pd.Series([current_portfolio_value], index=[datetime.now()])
                        ])
            
            # Calculate current drawdown
            md_calculator = MaximumDrawdown(updated_values)
            status = md_calculator.get_drawdown_summary()
            
            # Check for new alerts
            warnings = md_calculator.get_risk_warnings()
            if warnings:
                await self._send_alerts(warnings)
            
            return {
                'status': 'active',
                'current_drawdown': md_calculator.current_drawdown,
                'maximum_drawdown': md_calculator.max_drawdown,
                'alerts': warnings,
                'monitoring_time': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Drawdown check error: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    async def _send_alerts(self, warnings: List[Dict[str, Any]]):
        """Send alerts to registered callbacks."""
        try:
            for warning in warnings:
                for callback in self.alert_callbacks:
                    try:
                        await callback(self.user_id, warning)
                    except Exception as e:
                        logger.error(f"Alert callback error: {str(e)}")
                        
        except Exception as e:
            logger.error(f"Alert sending error: {str(e)}")
