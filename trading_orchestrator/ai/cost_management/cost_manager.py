"""
LLM Cost Manager - Core cost tracking and management system

Provides comprehensive cost monitoring, budget controls, and real-time tracking
for LLM usage across multiple providers.
"""

import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
from pathlib import Path

from loguru import logger


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning" 
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class CostEventType(Enum):
    """Types of cost events"""
    USAGE = "usage"
    BUDGET_WARNING = "budget_warning"
    BUDGET_LIMIT_REACHED = "budget_limit_reached"
    ANOMALY_DETECTED = "anomaly_detected"
    PROVIDER_SWITCH = "provider_switch"
    OPTIMIZATION_APPLIED = "optimization_applied"


@dataclass
class CostMetrics:
    """Cost and usage metrics for a time period"""
    total_cost: float
    total_tokens: int
    request_count: int
    average_cost_per_request: float
    cost_per_1k_tokens: float
    period_start: datetime
    period_end: datetime
    provider_breakdown: Dict[str, Dict[str, Any]]
    model_breakdown: Dict[str, Dict[str, Any]]


@dataclass
class CostAlert:
    """Cost-related alert"""
    id: str
    level: AlertLevel
    message: str
    timestamp: datetime
    event_type: CostEventType
    current_cost: float
    budget_limit: Optional[float] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BudgetLimit:
    """Budget limit configuration"""
    name: str
    limit_amount: float
    period: timedelta
    alert_thresholds: List[float]  # Percentages (0.5, 0.8, 0.95)
    auto_actions: Dict[str, Any]  # Actions to take when limit reached
    is_active: bool = True
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class OptimizationRecommendation:
    """Cost optimization recommendation"""
    id: str
    title: str
    description: str
    potential_savings: float
    action: str
    parameters: Dict[str, Any]
    priority: int  # 1-10, 10 being highest
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class LLMCostManager:
    """
    Comprehensive LLM cost management system
    
    Features:
    - Real-time cost tracking per provider/model
    - Budget limits and alerts
    - Cost anomaly detection
    - Optimization recommendations
    - Historical analytics
    - Automated cost controls
    """
    
    def __init__(
        self,
        database_path: str = "data/llm_costs.db",
        alert_callback: Optional[Callable[[CostAlert], None]] = None
    ):
        """
        Initialize cost manager
        
        Args:
            database_path: Path to SQLite database for cost data
            alert_callback: Callback function for alerts
        """
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.alert_callback = alert_callback
        
        # Real-time tracking
        self.current_usage: Dict[str, Dict[str, Any]] = {}
        self.active_budgets: Dict[str, BudgetLimit] = {}
        self.recent_alerts: List[CostAlert] = []
        self.cost_history: List[Dict[str, Any]] = []
        
        # Configuration
        self.default_budgets = self._setup_default_budgets()
        self.anomaly_threshold = 2.0  # Standard deviations
        
        # Initialize database
        self._init_database()
        
        logger.info("LLM Cost Manager initialized")
    
    def _setup_default_budgets(self) -> Dict[str, BudgetLimit]:
        """Setup default budget configurations"""
        return {
            "daily": BudgetLimit(
                name="Daily Budget",
                limit_amount=100.0,
                period=timedelta(days=1),
                alert_thresholds=[0.5, 0.8, 0.95],
                auto_actions={
                    "switch_to_faster_models": True,
                    "reduce_token_limits": True,
                    "send_emergency_alert": True
                }
            ),
            "weekly": BudgetLimit(
                name="Weekly Budget", 
                limit_amount=500.0,
                period=timedelta(days=7),
                alert_thresholds=[0.6, 0.85, 0.95],
                auto_actions={
                    "switch_to_faster_models": True,
                    "throttle_requests": True
                }
            ),
            "monthly": BudgetLimit(
                name="Monthly Budget",
                limit_amount=2000.0,
                period=timedelta(days=30),
                alert_thresholds=[0.7, 0.9, 0.98],
                auto_actions={
                    "review_model_selection": True,
                    "optimize_token_usage": True
                }
            )
        }
    
    def _init_database(self):
        """Initialize SQLite database for cost tracking"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Cost events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cost_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                tokens_used INTEGER NOT NULL,
                cost REAL NOT NULL,
                request_duration REAL,
                session_id TEXT,
                task_type TEXT,
                metadata TEXT
            )
        """)
        
        # Budget tracking table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS budget_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                budget_name TEXT NOT NULL,
                period_start DATETIME NOT NULL,
                period_end DATETIME NOT NULL,
                amount_spent REAL NOT NULL,
                limit_amount REAL NOT NULL,
                alert_level TEXT,
                metadata TEXT
            )
        """)
        
        # Alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id TEXT PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                event_type TEXT NOT NULL,
                current_cost REAL,
                budget_limit REAL,
                provider TEXT,
                model TEXT,
                metadata TEXT
            )
        """)
        
        # Provider health table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS provider_health (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                provider TEXT NOT NULL,
                avg_response_time REAL,
                error_rate REAL,
                availability REAL,
                cost_per_1k_tokens REAL
            )
        """)
        
        conn.commit()
        conn.close()
    
    async def track_usage(
        self,
        provider: str,
        model: str,
        tokens_used: int,
        cost: float,
        request_duration: float = None,
        session_id: str = None,
        task_type: str = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Track LLM usage and cost in real-time
        
        Args:
            provider: LLM provider (openai, anthropic, etc.)
            model: Model name
            tokens_used: Number of tokens consumed
            cost: Cost of the request
            request_duration: Request duration in seconds
            session_id: Session identifier
            task_type: Type of task performed
            metadata: Additional metadata
            
        Returns:
            Current cost statistics
        """
        timestamp = datetime.now()
        
        # Store in database
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO cost_events 
            (provider, model, tokens_used, cost, request_duration, session_id, task_type, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            provider, model, tokens_used, cost, request_duration,
            session_id, task_type, json.dumps(metadata or {})
        ))
        
        conn.commit()
        conn.close()
        
        # Update real-time tracking
        key = f"{provider}:{model}"
        if key not in self.current_usage:
            self.current_usage[key] = {
                'provider': provider,
                'model': model,
                'total_tokens': 0,
                'total_cost': 0.0,
                'request_count': 0,
                'start_time': timestamp
            }
        
        self.current_usage[key]['total_tokens'] += tokens_used
        self.current_usage[key]['total_cost'] += cost
        self.current_usage[key]['request_count'] += 1
        
        # Check budgets and alerts
        await self._check_budgets(timestamp)
        
        # Log event
        logger.info(
            f"Usage tracked: {provider}/{model} - {tokens_used} tokens, ${cost:.4f}"
        )
        
        return self.get_current_metrics()
    
    async def _check_budgets(self, timestamp: datetime):
        """Check budget limits and trigger alerts"""
        for budget_name, budget in self.active_budgets.items():
            if not budget.is_active:
                continue
            
            # Calculate usage for budget period
            usage = await self._calculate_budget_usage(budget, timestamp)
            usage_percentage = usage / budget.limit_amount
            
            # Check thresholds
            for threshold in budget.alert_thresholds:
                alert_level = self._get_alert_level(usage_percentage, threshold)
                
                if alert_level and not self._alert_sent_recently(budget_name, threshold):
                    await self._trigger_budget_alert(
                        budget, usage, alert_level, timestamp
                    )
            
            # Check if limit reached
            if usage >= budget.limit_amount:
                await self._handle_budget_limit_reached(budget, usage, timestamp)
    
    async def _calculate_budget_usage(self, budget: BudgetLimit, current_time: datetime) -> float:
        """Calculate usage for a specific budget period"""
        period_start = current_time - budget.period
        
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT SUM(cost) FROM cost_events 
            WHERE timestamp >= ? AND timestamp <= ?
        """, (period_start, current_time))
        
        result = cursor.fetchone()[0]
        conn.close()
        
        return result or 0.0
    
    def _get_alert_level(self, usage_percentage: float, threshold: float) -> Optional[AlertLevel]:
        """Determine alert level based on usage percentage"""
        if usage_percentage >= threshold:
            if threshold >= 0.95:
                return AlertLevel.CRITICAL
            elif threshold >= 0.8:
                return AlertLevel.WARNING
            else:
                return AlertLevel.INFO
        return None
    
    def _alert_sent_recently(self, budget_name: str, threshold: float) -> bool:
        """Check if alert was sent recently for this budget/threshold"""
        cutoff_time = datetime.now() - timedelta(hours=1)
        
        for alert in self.recent_alerts:
            if (alert.event_type in [CostEventType.BUDGET_WARNING, CostEventType.BUDGET_LIMIT_REACHED] and
                alert.timestamp > cutoff_time):
                return True
        return False
    
    async def _trigger_budget_alert(
        self,
        budget: BudgetLimit,
        usage: float,
        alert_level: AlertLevel,
        timestamp: datetime
    ):
        """Trigger budget alert"""
        usage_percentage = (usage / budget.limit_amount) * 100
        
        alert = CostAlert(
            id=f"budget_{budget.name}_{timestamp.timestamp()}",
            level=alert_level,
            message=f"{budget.name}: ${usage:.2f} spent (${budget.limit_amount:.2f} limit, {usage_percentage:.1f}%)",
            timestamp=timestamp,
            event_type=CostEventType.BUDGET_WARNING,
            current_cost=usage,
            budget_limit=budget.limit_amount
        )
        
        await self._save_alert(alert)
        
        if self.alert_callback:
            await self.alert_callback(alert)
        
        logger.warning(f"Budget alert: {alert.message}")
    
    async def _handle_budget_limit_reached(self, budget: BudgetLimit, usage: float, timestamp: datetime):
        """Handle budget limit being reached"""
        alert = CostAlert(
            id=f"budget_limit_{budget.name}_{timestamp.timestamp()}",
            level=AlertLevel.EMERGENCY,
            message=f"Budget limit reached: {budget.name} - ${usage:.2f}/${budget.limit_amount:.2f}",
            timestamp=timestamp,
            event_type=CostEventType.BUDGET_LIMIT_REACHED,
            current_cost=usage,
            budget_limit=budget.limit_amount
        )
        
        await self._save_alert(alert)
        
        # Execute auto-actions
        await self._execute_budget_actions(budget, alert)
        
        if self.alert_callback:
            await self.alert_callback(alert)
        
        logger.critical(f"Budget limit reached: {alert.message}")
    
    async def _execute_budget_actions(self, budget: BudgetLimit, alert: CostAlert):
        """Execute automated actions when budget limits are reached"""
        for action, enabled in budget.auto_actions.items():
            if not enabled:
                continue
            
            try:
                if action == "switch_to_faster_models":
                    await self._switch_to_cost_effective_models()
                elif action == "reduce_token_limits":
                    await self._reduce_token_limits()
                elif action == "throttle_requests":
                    await self._enable_request_throttling()
                elif action == "send_emergency_alert":
                    await self._send_emergency_notification(alert)
                    
            except Exception as e:
                logger.error(f"Failed to execute budget action {action}: {e}")
    
    async def _switch_to_cost_effective_models(self):
        """Switch to more cost-effective models"""
        logger.info("Executing: Switch to cost-effective models")
        # This would integrate with the AI models manager
    
    async def _reduce_token_limits(self):
        """Reduce token limits for cost control"""
        logger.info("Executing: Reduce token limits")
        # Implementation would adjust max_tokens in API calls
    
    async def _enable_request_throttling(self):
        """Enable request throttling"""
        logger.info("Executing: Enable request throttling")
        # Implementation would add delays between requests
    
    async def _send_emergency_notification(self, alert: CostAlert):
        """Send emergency notification"""
        logger.critical(f"EMERGENCY: {alert.message}")
        # Implementation could send email/SMS/Discord notifications
    
    async def _save_alert(self, alert: CostAlert):
        """Save alert to database"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO alerts 
            (id, level, message, event_type, current_cost, budget_limit, provider, model, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            alert.id, alert.level.value, alert.message, alert.event_type.value,
            alert.current_cost, alert.budget_limit, alert.provider, alert.model,
            json.dumps(alert.metadata or {})
        ))
        
        conn.commit()
        conn.close()
        
        # Add to recent alerts (keep last 100)
        self.recent_alerts.append(alert)
        if len(self.recent_alerts) > 100:
            self.recent_alerts = self.recent_alerts[-100:]
    
    def add_budget(self, budget: BudgetLimit):
        """Add a new budget limit"""
        self.active_budgets[budget.name] = budget
        logger.info(f"Added budget: {budget.name} - ${budget.limit_amount:.2f}/{budget.period}")
    
    def remove_budget(self, budget_name: str):
        """Remove a budget limit"""
        if budget_name in self.active_budgets:
            del self.active_budgets[budget_name]
            logger.info(f"Removed budget: {budget_name}")
    
    def get_current_metrics(self) -> CostMetrics:
        """Get current cost metrics"""
        now = datetime.now()
        
        # Calculate total usage
        total_cost = sum(usage['total_cost'] for usage in self.current_usage.values())
        total_tokens = sum(usage['total_tokens'] for usage in self.current_usage.values())
        request_count = sum(usage['request_count'] for usage in self.current_usage.values())
        
        # Provider breakdown
        provider_breakdown = {}
        for usage in self.current_usage.values():
            provider = usage['provider']
            if provider not in provider_breakdown:
                provider_breakdown[provider] = {
                    'total_cost': 0.0,
                    'total_tokens': 0,
                    'request_count': 0
                }
            provider_breakdown[provider]['total_cost'] += usage['total_cost']
            provider_breakdown[provider]['total_tokens'] += usage['total_tokens']
            provider_breakdown[provider]['request_count'] += usage['request_count']
        
        # Model breakdown
        model_breakdown = {}
        for usage in self.current_usage.values():
            model_key = f"{usage['provider']}/{usage['model']}"
            model_breakdown[model_key] = {
                'total_cost': usage['total_cost'],
                'total_tokens': usage['total_tokens'],
                'request_count': usage['request_count']
            }
        
        return CostMetrics(
            total_cost=total_cost,
            total_tokens=total_tokens,
            request_count=request_count,
            average_cost_per_request=total_cost / max(request_count, 1),
            cost_per_1k_tokens=total_cost / max(total_tokens / 1000, 1),
            period_start=min([usage['start_time'] for usage in self.current_usage.values()] or [now]),
            period_end=now,
            provider_breakdown=provider_breakdown,
            model_breakdown=model_breakdown
        )
    
    async def get_historical_metrics(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[CostMetrics]:
        """Get historical cost metrics for a date range"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Group by hour for detailed metrics
        cursor.execute("""
            SELECT 
                strftime('%Y-%m-%d %H:00:00', timestamp) as hour,
                provider,
                model,
                SUM(cost) as total_cost,
                SUM(tokens_used) as total_tokens,
                COUNT(*) as request_count
            FROM cost_events 
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY hour, provider, model
            ORDER BY hour
        """, (start_date, end_date))
        
        results = cursor.fetchall()
        conn.close()
        
        # Convert to CostMetrics objects
        metrics = []
        for hour, provider, model, total_cost, total_tokens, request_count in results:
            metrics.append(CostMetrics(
                total_cost=total_cost,
                total_tokens=total_tokens,
                request_count=request_count,
                average_cost_per_request=total_cost / max(request_count, 1),
                cost_per_1k_tokens=total_cost / max(total_tokens / 1000, 1),
                period_start=datetime.strptime(hour, '%Y-%m-%d %H:%M:%S'),
                period_end=datetime.strptime(hour, '%Y-%m-%d %H:%M:%S') + timedelta(hours=1),
                provider_breakdown={provider: {
                    'total_cost': total_cost,
                    'total_tokens': total_tokens,
                    'request_count': request_count
                }},
                model_breakdown={f"{provider}/{model}": {
                    'total_cost': total_cost,
                    'total_tokens': total_tokens,
                    'request_count': request_count
                }}
            ))
        
        return metrics
    
    def get_active_alerts(self, level: Optional[AlertLevel] = None) -> List[CostAlert]:
        """Get active alerts"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        filtered_alerts = [
            alert for alert in self.recent_alerts 
            if alert.timestamp > cutoff_time
        ]
        
        if level:
            filtered_alerts = [alert for alert in filtered_alerts if alert.level == level]
        
        return sorted(filtered_alerts, key=lambda x: x.timestamp, reverse=True)
    
    async def export_cost_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Export cost data for analysis"""
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
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'events': events,
            'total_events': len(events)
        }
    
    def reset_usage(self):
        """Reset current usage tracking"""
        self.current_usage.clear()
        logger.info("Usage tracking reset")
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get comprehensive cost summary"""
        current_metrics = self.get_current_metrics()
        active_alerts = self.get_active_alerts()
        
        # Calculate budget status
        budget_status = {}
        now = datetime.now()
        for name, budget in self.active_budgets.items():
            if budget.is_active:
                usage = asyncio.create_task(self._calculate_budget_usage(budget, now))
                # Note: This is sync context, would need proper async handling
                budget_status[name] = {
                    'limit': budget.limit_amount,
                    'spent': 0.0,  # Would be calculated asynchronously
                    'percentage': 0.0,
                    'status': 'unknown'
                }
        
        return {
            'current_metrics': asdict(current_metrics),
            'active_budgets': {name: asdict(budget) for name, budget in self.active_budgets.items()},
            'budget_status': budget_status,
            'recent_alerts': [asdict(alert) for alert in active_alerts[:10]],
            'total_active_alerts': len(active_alerts)
        }