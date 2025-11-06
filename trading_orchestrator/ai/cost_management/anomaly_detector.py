"""
Anomaly Detection - Intelligent cost anomaly detection and alerting system

Detects unusual cost patterns, usage spikes, and efficiency issues in real-time.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
import statistics
import json
from collections import deque
import numpy as np

from loguru import logger


class AnomalyType(Enum):
    """Types of cost anomalies"""
    COST_SPIKE = "cost_spike"
    COST_DROP = "cost_drop"
    TOKEN_SPIKE = "token_spike"
    TOKEN_DROP = "token_drop"
    REQUEST_SPIKE = "request_spike"
    REQUEST_DROP = "request_drop"
    EFFICIENCY_DROP = "efficiency_drop"
    PROVIDER_SWITCH = "provider_switch"
    PATTERN_BREAK = "pattern_break"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"


class AnomalySeverity(Enum):
    """Anomaly severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AnomalyEvent:
    """Anomaly detection event"""
    id: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    title: str
    description: str
    detected_at: datetime
    affected_period: Tuple[datetime, datetime]
    current_value: float
    expected_value: float
    deviation_percentage: float
    provider: Optional[str] = None
    model: Optional[str] = None
    confidence_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class AnomalyRule:
    """Rule for detecting anomalies"""
    name: str
    anomaly_type: AnomalyType
    threshold: float
    window_size: int  # Number of data points to analyze
    severity_threshold: float  # Value above which anomaly is considered severe
    enabled: bool = True
    cooldown_period: timedelta = field(default_factory=lambda: timedelta(hours=1))
    last_triggered: Optional[datetime] = None


class AnomalyDetector:
    """
    Advanced anomaly detection system for LLM costs
    
    Features:
    - Real-time cost anomaly detection
    - Statistical outlier identification
    - Pattern-based anomaly detection
    - Machine learning-based detection
    - Customizable detection rules
    - Alert suppression and aggregation
    """
    
    def __init__(
        self,
        cost_manager,
        provider_manager,
        anomaly_storage_path: str = "data/anomaly_storage.db"
    ):
        """
        Initialize anomaly detector
        
        Args:
            cost_manager: LLMCostManager instance
            provider_manager: ProviderManager instance
            anomaly_storage_path: Path to store anomaly events
        """
        self.cost_manager = cost_manager
        self.provider_manager = provider_manager
        self.anomaly_storage_path = anomaly_storage_path
        
        # Detection configuration
        self.default_window_size = 24  # hours
        self.anomaly_threshold = 2.0  # standard deviations
        self.detection_enabled = True
        
        # Anomaly tracking
        self.recent_anomalies: deque = deque(maxlen=100)
        self.active_anomalies: Dict[str, AnomalyEvent] = {}
        self.anomaly_rules: List[AnomalyRule] = []
        
        # Real-time data buffers
        self.cost_buffer: deque = deque(maxlen=168)  # 7 days of hourly data
        self.token_buffer: deque = deque(maxlen=168)
        self.request_buffer: deque = deque(maxlen=168)
        
        # Initialize database
        self._init_anomaly_database()
        
        # Setup default detection rules
        self._setup_default_rules()
        
        # Start real-time detection
        asyncio.create_task(self._real_time_detection_loop())
        
        logger.info("Anomaly Detector initialized")
    
    def _init_anomaly_database(self):
        """Initialize database for anomaly storage"""
        path = self.anomaly_storage_path.replace('anomaly_storage.db', '')
        import os
        os.makedirs(path, exist_ok=True)
        
        conn = sqlite3.connect(self.anomaly_storage_path)
        cursor = conn.cursor()
        
        # Anomaly events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS anomaly_events (
                id TEXT PRIMARY KEY,
                anomaly_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                detected_at DATETIME NOT NULL,
                period_start DATETIME NOT NULL,
                period_end DATETIME NOT NULL,
                current_value REAL NOT NULL,
                expected_value REAL NOT NULL,
                deviation_percentage REAL NOT NULL,
                provider TEXT,
                model TEXT,
                confidence_score REAL,
                metadata TEXT,
                resolved BOOLEAN DEFAULT FALSE,
                resolved_at DATETIME
            )
        """)
        
        # Detection metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detection_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metric_type TEXT NOT NULL,
                value REAL NOT NULL,
                provider TEXT,
                model TEXT,
                metadata TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _setup_default_rules(self):
        """Setup default anomaly detection rules"""
        
        # Cost spike detection
        self.add_rule(AnomalyRule(
            name="Cost Spike Detection",
            anomaly_type=AnomalyType.COST_SPIKE,
            threshold=self.anomaly_threshold,
            window_size=24,
            severity_threshold=3.0,
            cooldown_period=timedelta(hours=2)
        ))
        
        # Request spike detection
        self.add_rule(AnomalyRule(
            name="Request Spike Detection",
            anomaly_type=AnomalyType.REQUEST_SPIKE,
            threshold=2.5,
            window_size=12,
            severity_threshold=4.0
        ))
        
        # Efficiency drop detection
        self.add_rule(AnomalyRule(
            name="Efficiency Drop Detection",
            anomaly_type=AnomalyType.EFFICIENCY_DROP,
            threshold=1.5,
            window_size=48,
            severity_threshold=2.0
        ))
        
        # Provider switching anomaly
        self.add_rule(AnomalyRule(
            name="Provider Switching Anomaly",
            anomaly_type=AnomalyType.PROVIDER_SWITCH,
            threshold=0.0,  # Any switch is potentially anomalous
            window_size=1,
            severity_threshold=0.0
        ))
    
    def add_rule(self, rule: AnomalyRule):
        """Add a custom anomaly detection rule"""
        self.anomaly_rules.append(rule)
        logger.info(f"Added anomaly rule: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """Remove an anomaly detection rule"""
        self.anomaly_rules = [r for r in self.anomaly_rules if r.name != rule_name]
        logger.info(f"Removed anomaly rule: {rule_name}")
    
    async def _real_time_detection_loop(self):
        """Main loop for real-time anomaly detection"""
        while True:
            try:
                if self.detection_enabled:
                    await self._run_detection_cycle()
                
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Anomaly detection loop error: {e}")
                await asyncio.sleep(60)
    
    async def _run_detection_cycle(self):
        """Run a single detection cycle"""
        try:
            # Get current cost data
            current_metrics = self.cost_manager.get_current_metrics()
            
            # Update buffers with current data
            await self._update_data_buffers(current_metrics)
            
            # Check for anomalies if we have enough data
            if len(self.cost_buffer) >= 3:  # Minimum threshold
                await self._detect_cost_anomalies(current_metrics)
                await self._detect_request_anomalies(current_metrics)
                await self._detect_efficiency_anomalies(current_metrics)
                await self._detect_pattern_anomalies()
            
            # Clean up resolved anomalies
            await self._cleanup_resolved_anomalies()
            
        except Exception as e:
            logger.error(f"Detection cycle error: {e}")
    
    async def _update_data_buffers(self, metrics):
        """Update real-time data buffers"""
        current_time = datetime.now()
        
        # Store hourly aggregates
        self.cost_buffer.append({
            'timestamp': current_time,
            'cost': metrics.total_cost,
            'tokens': metrics.total_tokens,
            'requests': metrics.request_count
        })
    
    async def _detect_cost_anomalies(self, current_metrics):
        """Detect cost-related anomalies"""
        recent_costs = [item['cost'] for item in list(self.cost_buffer)[-24:]]  # Last 24 hours
        
        if len(recent_costs) < 3:
            return
        
        current_cost = recent_costs[-1]
        
        # Statistical anomaly detection
        mean_cost = statistics.mean(recent_costs[:-1])  # Exclude current value
        std_cost = statistics.stdev(recent_costs[:-1]) if len(recent_costs) > 2 else 0
        
        if std_cost > 0:
            z_score = abs(current_cost - mean_cost) / std_cost
            
            if z_score >= self.anomaly_threshold:
                # Cost spike or drop
                anomaly_type = AnomalyType.COST_SPIKE if current_cost > mean_cost else AnomalyType.COST_DROP
                severity = AnomalySeverity.HIGH if z_score >= 3.0 else AnomalySeverity.MEDIUM
                
                await self._create_anomaly(
                    anomaly_type=anomaly_type,
                    severity=severity,
                    title=f"Cost {'Spike' if current_cost > mean_cost else 'Drop'} Detected",
                    description=f"Current cost ${current_cost:.4f} deviates by {z_score:.1f}σ from expected ${mean_cost:.4f}",
                    current_value=current_cost,
                    expected_value=mean_cost,
                    deviation_percentage=(current_cost - mean_cost) / max(mean_cost, 0.01) * 100,
                    confidence_score=min(z_score / 3.0, 1.0),
                    metadata={
                        'z_score': z_score,
                        'baseline_period_hours': len(recent_costs)
                    }
                )
    
    async def _detect_request_anomalies(self, current_metrics):
        """Detect request volume anomalies"""
        recent_requests = [item['requests'] for item in list(self.request_buffer)[-12:]]  # Last 12 hours
        
        if len(recent_requests) < 3:
            return
        
        current_requests = recent_requests[-1]
        mean_requests = statistics.mean(recent_requests[:-1])
        
        if mean_requests > 0:
            request_ratio = current_requests / mean_requests
            
            if request_ratio >= 2.5 or request_ratio <= 0.4:  # Significant deviation
                anomaly_type = AnomalyType.REQUEST_SPIKE if request_ratio > 1 else AnomalyType.REQUEST_DROP
                severity = AnomalySeverity.HIGH if request_ratio >= 4.0 or request_ratio <= 0.25 else AnomalySeverity.MEDIUM
                
                await self._create_anomaly(
                    anomaly_type=anomaly_type,
                    severity=severity,
                    title=f"Request {'Volume Spike' if request_ratio > 1 else 'Volume Drop'} Detected",
                    description=f"Request volume {current_requests} is {request_ratio:.1f}x the expected {mean_requests:.0f}",
                    current_value=current_requests,
                    expected_value=mean_requests,
                    deviation_percentage=(request_ratio - 1) * 100,
                    confidence_score=abs(request_ratio - 1) / 3.0,
                    metadata={
                        'request_ratio': request_ratio,
                        'baseline_requests': mean_requests
                    }
                )
    
    async def _detect_efficiency_anomalies(self, current_metrics):
        """Detect efficiency-related anomalies"""
        if current_metrics.request_count == 0:
            return
        
        recent_efficiency = []
        for item in list(self.cost_buffer)[-48:]:  # Last 48 hours
            if item['requests'] > 0:
                efficiency = item['cost'] / item['requests']
                recent_efficiency.append(efficiency)
        
        if len(recent_efficiency) < 3:
            return
        
        current_efficiency = current_metrics.total_cost / max(current_metrics.request_count, 1)
        mean_efficiency = statistics.mean(recent_efficiency[:-1])
        
        if mean_efficiency > 0:
            efficiency_ratio = current_efficiency / mean_efficiency
            
            if efficiency_ratio >= 1.5 or efficiency_ratio <= 0.67:  # 50% change threshold
                anomaly_type = AnomalyType.EFFICIENCY_DROP if efficiency_ratio > 1 else AnomalyType.COST_DROP
                severity = AnomalySeverity.HIGH if efficiency_ratio >= 2.0 or efficiency_ratio <= 0.5 else AnomalySeverity.MEDIUM
                
                await self._create_anomaly(
                    anomaly_type=anomaly_type,
                    severity=severity,
                    title=f"Efficiency {'Degradation' if efficiency_ratio > 1 else 'Improvement'} Detected",
                    description=f"Cost per request ${current_efficiency:.4f} is {efficiency_ratio:.1f}x expected ${mean_efficiency:.4f}",
                    current_value=current_efficiency,
                    expected_value=mean_efficiency,
                    deviation_percentage=(efficiency_ratio - 1) * 100,
                    confidence_score=abs(efficiency_ratio - 1) / 2.0,
                    metadata={
                        'efficiency_ratio': efficiency_ratio,
                        'baseline_efficiency': mean_efficiency
                    }
                )
    
    async def _detect_pattern_anomalies(self):
        """Detect pattern-based anomalies"""
        if len(self.cost_buffer) < 168:  # Need a week of data
            return
        
        # Get same hour data from previous week
        current_hour = datetime.now().hour
        current_costs = []
        
        for item in list(self.cost_buffer)[-168:-24]:  # Previous week, excluding today
            if item['timestamp'].hour == current_hour:
                current_costs.append(item['cost'])
        
        if len(current_costs) < 5:  # Need at least 5 data points
            return
        
        current_cost = list(self.cost_buffer)[-1]['cost']
        historical_mean = statistics.mean(current_costs)
        historical_std = statistics.stdev(current_costs) if len(current_costs) > 1 else 0
        
        if historical_std > 0:
            z_score = abs(current_cost - historical_mean) / historical_std
            
            if z_score >= 2.5:  # More lenient for historical patterns
                await self._create_anomaly(
                    anomaly_type=AnomalyType.PATTERN_BREAK,
                    severity=AnomalySeverity.MEDIUM if z_score < 3.0 else AnomalySeverity.HIGH,
                    title="Historical Pattern Deviation",
                    description=f"Current cost deviates from {current_hour}:00 historical pattern by {z_score:.1f}σ",
                    current_value=current_cost,
                    expected_value=historical_mean,
                    deviation_percentage=(current_cost - historical_mean) / max(historical_mean, 0.01) * 100,
                    confidence_score=z_score / 4.0,
                    metadata={
                        'hour_of_day': current_hour,
                        'z_score': z_score,
                        'historical_data_points': len(current_costs)
                    }
                )
    
    async def _create_anomaly(
        self,
        anomaly_type: AnomalyType,
        severity: AnomalySeverity,
        title: str,
        description: str,
        current_value: float,
        expected_value: float,
        deviation_percentage: float,
        confidence_score: float,
        metadata: Dict[str, Any]
    ):
        """Create and store an anomaly event"""
        anomaly_id = f"{anomaly_type.value}_{int(time.time())}"
        
        # Check cooldown period
        rule = self._get_rule_for_type(anomaly_type)
        if rule and rule.last_triggered:
            if datetime.now() - rule.last_triggered < rule.cooldown_period:
                return  # Still in cooldown
        
        anomaly = AnomalyEvent(
            id=anomaly_id,
            anomaly_type=anomaly_type,
            severity=severity,
            title=title,
            description=description,
            detected_at=datetime.now(),
            affected_period=(datetime.now() - timedelta(hours=1), datetime.now()),
            current_value=current_value,
            expected_value=expected_value,
            deviation_percentage=deviation_percentage,
            confidence_score=confidence_score,
            metadata=metadata
        )
        
        # Store anomaly
        self.active_anomalies[anomaly_id] = anomaly
        self.recent_anomalies.append(anomaly)
        
        # Update rule cooldown
        if rule:
            rule.last_triggered = datetime.now()
        
        # Store in database
        await self._store_anomaly_in_db(anomaly)
        
        # Log and alert
        logger.warning(f"Anomaly detected: {title} - {description}")
        
        # TODO: Send alerts/notifications
    
    def _get_rule_for_type(self, anomaly_type: AnomalyType) -> Optional[AnomalyRule]:
        """Get detection rule for anomaly type"""
        for rule in self.anomaly_rules:
            if rule.anomaly_type == anomaly_type:
                return rule
        return None
    
    async def _store_anomaly_in_db(self, anomaly: AnomalyEvent):
        """Store anomaly in database"""
        conn = sqlite3.connect(self.anomaly_storage_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO anomaly_events
            (id, anomaly_type, severity, title, description, detected_at, 
             period_start, period_end, current_value, expected_value, 
             deviation_percentage, provider, model, confidence_score, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            anomaly.id,
            anomaly.anomaly_type.value,
            anomaly.severity.value,
            anomaly.title,
            anomaly.description,
            anomaly.detected_at,
            anomaly.affected_period[0],
            anomaly.affected_period[1],
            anomaly.current_value,
            anomaly.expected_value,
            anomaly.deviation_percentage,
            anomaly.provider,
            anomaly.model,
            anomaly.confidence_score,
            json.dumps(anomaly.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    async def _cleanup_resolved_anomalies(self):
        """Clean up resolved anomalies"""
        current_time = datetime.now()
        
        resolved_ids = []
        for anomaly_id, anomaly in self.active_anomalies.items():
            # Auto-resolve after 24 hours
            if current_time - anomaly.detected_at > timedelta(hours=24):
                anomaly.resolved = True
                anomaly.resolved_at = current_time
                resolved_ids.append(anomaly_id)
        
        # Remove resolved anomalies
        for anomaly_id in resolved_ids:
            del self.active_anomalies[anomaly_id]
    
    async def detect_provider_anomalies(self, provider_name: str) -> List[AnomalyEvent]:
        """Detect anomalies specific to a provider"""
        provider_anomalies = []
        
        # Get provider-specific data
        conn = sqlite3.connect(self.cost_manager.database_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                timestamp,
                cost,
                tokens_used,
                request_duration
            FROM cost_events 
            WHERE provider = ?
            AND timestamp >= ?
            ORDER BY timestamp
        """, (provider_name, datetime.now() - timedelta(days=7)))
        
        events = cursor.fetchall()
        conn.close()
        
        if len(events) < 10:  # Need sufficient data
            return provider_anomalies
        
        # Analyze provider performance
        costs = [event[1] for event in events if event[1]]
        response_times = [event[3] for event in events if event[3]]
        
        if not costs:
            return provider_anomalies
        
        # Check for cost anomalies
        recent_costs = costs[-24:]  # Last 24 hours
        current_cost = recent_costs[-1] if recent_costs else 0
        
        if len(recent_costs) > 1:
            mean_cost = statistics.mean(recent_costs[:-1])
            std_cost = statistics.stdev(recent_costs[:-1])
            
            if std_cost > 0 and abs(current_cost - mean_cost) / std_cost >= 2.0:
                anomaly_type = AnomalyType.COST_SPIKE if current_cost > mean_cost else AnomalyType.COST_DROP
                
                provider_anomalies.append(AnomalyEvent(
                    id=f"{provider_name}_cost_anomaly_{int(time.time())}",
                    anomaly_type=anomaly_type,
                    severity=AnomalySeverity.MEDIUM,
                    title=f"{provider_name} Cost Anomaly",
                    description=f"Provider {provider_name} shows unusual cost patterns",
                    detected_at=datetime.now(),
                    affected_period=(datetime.now() - timedelta(hours=1), datetime.now()),
                    current_value=current_cost,
                    expected_value=mean_cost,
                    deviation_percentage=(current_cost - mean_cost) / max(mean_cost, 0.01) * 100,
                    provider=provider_name,
                    confidence_score=0.7
                ))
        
        return provider_anomalies
    
    async def get_anomaly_summary(
        self,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get summary of recent anomalies"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter recent anomalies
        recent_anomalies = [
            anomaly for anomaly in self.recent_anomalies
            if anomaly.detected_at > cutoff_time
        ]
        
        # Aggregate by type
        type_counts = {}
        severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        
        for anomaly in recent_anomalies:
            # Count by type
            anomaly_type = anomaly.anomaly_type.value
            type_counts[anomaly_type] = type_counts.get(anomaly_type, 0) + 1
            
            # Count by severity
            severity_counts[anomaly.severity.value] += 1
        
        # Calculate resolution metrics
        resolved_count = sum(1 for a in recent_anomalies if a.resolved)
        resolution_rate = resolved_count / max(len(recent_anomalies), 1) * 100
        
        return {
            'summary_period_hours': hours,
            'total_anomalies': len(recent_anomalies),
            'active_anomalies': len(self.active_anomalies),
            'resolved_anomalies': resolved_count,
            'resolution_rate_percent': resolution_rate,
            'anomalies_by_type': type_counts,
            'anomalies_by_severity': severity_counts,
            'most_common_type': max(type_counts, key=type_counts.get) if type_counts else None,
            'detection_enabled': self.detection_enabled
        }
    
    async def export_anomalies(
        self,
        start_date: datetime,
        end_date: datetime,
        format: str = 'json'
    ) -> Dict[str, Any]:
        """Export anomalies for analysis"""
        conn = sqlite3.connect(self.anomaly_storage_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM anomaly_events 
            WHERE detected_at BETWEEN ? AND ?
            ORDER BY detected_at DESC
        """, (start_date, end_date))
        
        columns = [description[0] for description in cursor.description]
        anomaly_rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        
        if format == 'json':
            return {
                'export_period': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                },
                'total_anomalies': len(anomaly_rows),
                'anomalies': anomaly_rows
            }
        
        return {'error': f'Unsupported format: {format}'}
    
    async def analyze_anomaly_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in detected anomalies"""
        if len(self.recent_anomalies) < 10:
            return {'error': 'Insufficient anomaly data for pattern analysis'}
        
        # Time-based patterns
        hourly_distribution = {}
        daily_distribution = {}
        
        for anomaly in self.recent_anomalies:
            # Hour distribution
            hour = anomaly.detected_at.hour
            hourly_distribution[hour] = hourly_distribution.get(hour, 0) + 1
            
            # Day distribution
            day = anomaly.detected_at.strftime('%A')
            daily_distribution[day] = daily_distribution.get(day, 0) + 1
        
        # Most active hours
        peak_hours = sorted(hourly_distribution.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Anomaly progression
        progression = {}
        for anomaly in self.recent_anomalies:
            severity = anomaly.severity.value
            progression[severity] = progression.get(severity, 0) + 1
        
        return {
            'hourly_distribution': hourly_distribution,
            'daily_distribution': daily_distribution,
            'peak_anomaly_hours': [str(h[0]) for h in peak_hours],
            'anomaly_progression': progression,
            'total_anomalies_analyzed': len(self.recent_anomalies),
            'most_common_anomaly': max(progression, key=progression.get) if progression else None
        }
    
    async def resolve_anomaly(self, anomaly_id: str, resolution_notes: str = ""):
        """Manually resolve an anomaly"""
        if anomaly_id in self.active_anomalies:
            anomaly = self.active_anomalies[anomaly_id]
            anomaly.resolved = True
            anomaly.resolved_at = datetime.now()
            
            if resolution_notes:
                anomaly.metadata['resolution_notes'] = resolution_notes
            
            # Update database
            conn = sqlite3.connect(self.anomaly_storage_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE anomaly_events 
                SET resolved = TRUE, resolved_at = ?
                WHERE id = ?
            """, (datetime.now(), anomaly_id))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Anomaly {anomaly_id} resolved: {resolution_notes}")
        else:
            logger.warning(f"Anomaly {anomaly_id} not found for resolution")
    
    async def get_anomaly_trends(self, days: int = 30) -> Dict[str, Any]:
        """Get trends in anomaly detection"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Group anomalies by day
        daily_counts = {}
        for anomaly in self.recent_anomalies:
            if anomaly.detected_at > cutoff_date:
                day_key = anomaly.detected_at.date().isoformat()
                daily_counts[day_key] = daily_counts.get(day_key, 0) + 1
        
        # Calculate trend
        days_list = sorted(daily_counts.keys())
        if len(days_list) >= 2:
            recent_avg = statistics.mean([daily_counts.get(day, 0) for day in days_list[-7:]])
            earlier_avg = statistics.mean([daily_counts.get(day, 0) for day in days_list[:-7]]) if len(days_list) > 7 else 0
            
            trend_direction = "increasing" if recent_avg > earlier_avg else "decreasing" if recent_avg < earlier_avg else "stable"
            trend_magnitude = abs(recent_avg - earlier_avg) / max(earlier_avg, 0.1)
        else:
            trend_direction = "insufficient_data"
            trend_magnitude = 0
        
        return {
            'analysis_period_days': days,
            'daily_anomaly_counts': daily_counts,
            'trend_direction': trend_direction,
            'trend_magnitude': trend_magnitude,
            'avg_daily_anomalies': statistics.mean(daily_counts.values()) if daily_counts else 0,
            'peak_anomaly_day': max(daily_counts, key=daily_counts.get) if daily_counts else None
        }