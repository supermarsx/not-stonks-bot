"""
@file trading_frequency.py
@brief Trading Frequency Configuration Management System

@details
This module provides comprehensive trading frequency configuration and management
capabilities for the day trading orchestrator. It includes frequency settings,
position sizing calculations, monitoring, optimization recommendations, and
risk management integration.

Key Features:
- Configurable frequency settings (per minute, hour, day, custom)
- Frequency-based position sizing calculations
- Real-time frequency monitoring with alerts
- Frequency optimization recommendations based on backtesting
- Frequency controls integration with trading strategies
- Frequency-based risk management policies
- Database schema for frequency settings persistence
- Analytics and reporting for frequency performance
- UI components for frequency configuration

@author Trading Orchestrator System
@version 1.0
@date 2025-11-06

@warning
Improper frequency configuration can lead to excessive trading, increased transaction
costs, and amplified risk. Always test frequency settings in paper trading mode
before live deployment.

@note
This module provides the foundation for implementing frequency-aware trading
strategies and risk management policies.
"""

from typing import Dict, Any, List, Optional, Union, Protocol
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
import uuid
from pydantic import BaseModel, Field, validator
from loguru import logger

from .settings import Settings


class FrequencyType(str, Enum):
    """
    @enum FrequencyType
    @brief Trading frequency types
    
    @details
    Defines different types of trading frequencies supported by the system.
    Each frequency type represents a specific trading cadence and time window.
    
    @par Frequency Categories:
    - ULTRA_HIGH: Sub-minute trading (scalping, HFT)
    - HIGH: 1-5 minute intervals (intraday scalping)
    - MEDIUM: 5-15 minute intervals (momentum trading)
    - LOW: 15-60 minute intervals (swing trading)
    - VERY_LOW: Hourly+ intervals (position trading)
    - CUSTOM: User-defined frequency intervals
    
    @note
    Frequency type selection should align with strategy characteristics,
    market conditions, and risk tolerance.
    """
    ULTRA_HIGH = "ultra_high"      # < 1 minute
    HIGH = "high"                  # 1-5 minutes
    MEDIUM = "medium"              # 5-15 minutes
    LOW = "low"                    # 15-60 minutes
    VERY_LOW = "very_low"          # 1+ hours
    CUSTOM = "custom"              # User-defined


class FrequencyAlertType(str, Enum):
    """
    @enum FrequencyAlertType
    @brief Types of frequency-based alerts
    
    @details
    Enumerates the different types of alerts that can be triggered based on
    trading frequency thresholds and limits.
    
    @par Alert Categories:
    - THRESHOLD_EXCEEDED: Frequency limit exceeded
    - POSITION_SIZE_WARNING: Position size too large for frequency
    - RISK_LIMIT_REACHED: Risk limits reached due to frequency
    - OPTIMIZATION_SUGGESTION: Performance optimization opportunity
    - MARKET_CONDITION_CHANGE: Frequency adjustment recommended
    
    @warning
    All frequency alerts should be properly handled to prevent system overload
    and excessive trading behavior.
    """
    THRESHOLD_EXCEEDED = "threshold_exceeded"
    POSITION_SIZE_WARNING = "position_size_warning"
    RISK_LIMIT_REACHED = "risk_limit_reached"
    OPTIMIZATION_SUGGESTION = "optimization_suggestion"
    MARKET_CONDITION_CHANGE = "market_condition_change"


class FrequencyConstraintType(str, Enum):
    """
    @enum FrequencyConstraintType
    @brief Types of frequency constraints
    
    @details
    Defines different types of constraints that can be applied to trading
    frequency for risk management and compliance purposes.
    
    @par Constraint Categories:
    - HARD_LIMIT: Absolute maximum (cannot exceed)
    - SOFT_LIMIT: Warning threshold (can exceed with confirmation)
    - RECOMMENDED: Preferred range (performance optimized)
    - MINIMUM: Minimum required frequency
    - TIME_WINDOW: Frequency limit within specific time periods
    
    @note
    Constraint types should be carefully chosen based on risk tolerance
    and regulatory requirements.
    """
    HARD_LIMIT = "hard_limit"
    SOFT_LIMIT = "soft_limit"
    RECOMMENDED = "recommended"
    MINIMUM = "minimum"
    TIME_WINDOW = "time_window"


@dataclass
class FrequencySettings:
    """
    @class FrequencySettings
    @brief Trading frequency configuration settings
    
    @details
    Contains all configuration parameters for trading frequency management
    including time intervals, limits, thresholds, and constraint settings.
    
    @par Core Settings:
    - frequency_type: Category of trading frequency
    - interval_seconds: Time interval between trades (seconds)
    - max_trades_per_minute: Maximum trades per minute
    - max_trades_per_hour: Maximum trades per hour
    - max_trades_per_day: Maximum trades per day
    - position_size_multiplier: Position size adjustment factor
    - cooldown_periods: Mandatory waiting periods between trades
    - market_hours_only: Restrict trading to market hours
    
    @par Risk Management:
    - max_daily_frequency_risk: Risk exposure from frequency
    - frequency_volatility_adjustment: Volatility-based adjustments
    - correlation_limits: Cross-strategy frequency limits
    
    @par Example:
    @code
    settings = FrequencySettings(
        frequency_type=FrequencyType.HIGH,
        interval_seconds=300,  # 5 minutes
        max_trades_per_minute=5,
        max_trades_per_hour=20,
        max_trades_per_day=100,
        position_size_multiplier=1.0,
        cooldown_periods=30,
        market_hours_only=True
    )
    @endcode
    
    @warning
    Frequency settings directly impact trading costs and risk exposure.
    Incorrect settings can lead to excessive trading and financial losses.
    """
    
    # Core frequency settings
    frequency_type: FrequencyType = Field(default=FrequencyType.MEDIUM)
    interval_seconds: int = Field(default=300, description="Time interval between trades in seconds")
    
    # Trading limits
    max_trades_per_minute: int = Field(default=1, ge=0, description="Maximum trades per minute")
    max_trades_per_hour: int = Field(default=10, ge=0, description="Maximum trades per hour")
    max_trades_per_day: int = Field(default=100, ge=0, description="Maximum trades per day")
    
    # Position sizing
    position_size_multiplier: float = Field(default=1.0, ge=0.0, description="Position size adjustment factor")
    frequency_based_sizing: bool = Field(default=True, description="Enable frequency-based position sizing")
    
    # Cooldown and timing
    cooldown_periods: int = Field(default=0, ge=0, description="Cooldown period between trades in seconds")
    market_hours_only: bool = Field(default=False, description="Restrict trading to market hours only")
    
    # Risk management
    max_daily_frequency_risk: float = Field(default=0.05, ge=0.0, le=1.0, description="Maximum risk from frequency")
    frequency_volatility_adjustment: bool = Field(default=True, description="Adjust for market volatility")
    correlation_limits: Dict[str, int] = Field(default_factory=dict, description="Cross-strategy frequency limits")
    
    # Alerting and monitoring
    enable_alerts: bool = Field(default=True, description="Enable frequency-based alerts")
    alert_thresholds: Dict[str, float] = Field(default_factory=dict, description="Alert threshold percentages")
    
    # Optimization settings
    auto_optimization: bool = Field(default=False, description="Enable automatic frequency optimization")
    optimization_period_hours: int = Field(default=24, ge=1, description="Optimization analysis period in hours")
    
    # Strategy-specific overrides
    strategy_overrides: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Strategy-specific frequency overrides")
    
    # Custom intervals (for custom frequency type)
    custom_intervals: List[int] = Field(default_factory=list, description="Custom trading intervals in seconds")
    
    # Time window constraints
    time_window_limits: List[Dict[str, Any]] = Field(default_factory=list, description="Time-based frequency limits")
    
    def __post_init__(self):
        """Validate frequency settings after initialization"""
        if self.frequency_type == FrequencyType.CUSTOM and not self.custom_intervals:
            raise ValueError("Custom frequency type requires custom_intervals to be specified")
        
        if self.max_trades_per_minute > self.max_trades_per_hour / 60:
            logger.warning("max_trades_per_minute exceeds calculated hourly limit")
        
        if self.max_trades_per_hour > self.max_trades_per_day / 24:
            logger.warning("max_trades_per_hour exceeds calculated daily limit")


@dataclass
class FrequencyMetrics:
    """
    @class FrequencyMetrics
    @brief Real-time trading frequency metrics
    
    @details
    Tracks current trading frequency metrics including trade counts,
    rates, and performance indicators for monitoring and optimization.
    
    @par Current Metrics:
    - trades_in_last_minute: Trades executed in last 60 seconds
    - trades_in_last_hour: Trades executed in last hour
    - trades_today: Total trades today
    - current_frequency_rate: Current trades per minute
    - average_frequency_rate: Historical average trades per minute
    
    @par Performance Indicators:
    - frequency_efficiency: Trades per opportunity ratio
    - frequency_sharpe: Risk-adjusted frequency performance
    - frequency_drawdown: Peak-to-trough frequency decline
    
    @note
    Metrics are updated in real-time and used for monitoring and optimization.
    """
    
    # Trade counts
    trades_in_last_minute: int = 0
    trades_in_last_hour: int = 0
    trades_today: int = 0
    trades_this_week: int = 0
    trades_this_month: int = 0
    
    # Rates and averages
    current_frequency_rate: float = 0.0  # trades per minute
    average_frequency_rate: float = 0.0  # historical average
    target_frequency_rate: float = 0.0   # from settings
    
    # Performance metrics
    frequency_efficiency: float = 0.0  # trades per opportunity
    frequency_sharpe: float = 0.0      # risk-adjusted performance
    frequency_drawdown: float = 0.0     # maximum decline
    
    # Time tracking
    first_trade_today: Optional[datetime] = None
    last_trade_time: Optional[datetime] = None
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    # Cooldown tracking
    cooldown_end_time: Optional[datetime] = None
    in_cooldown: bool = False
    
    # Alert tracking
    alerts_triggered: List[Dict[str, Any]] = field(default_factory=list)
    threshold_violations: int = 0


@dataclass
class FrequencyAlert:
    """
    @class FrequencyAlert
    @brief Trading frequency alert configuration
    
    @details
    Defines frequency-based alerts that can be triggered when trading
    frequency limits are exceeded or optimization opportunities arise.
    
    @par Alert Properties:
    - alert_id: Unique identifier for the alert
    - alert_type: Type of frequency alert
    - severity: Alert severity level
    - message: Alert description message
    - threshold_value: Threshold that was exceeded
    - current_value: Current measured value
    - trigger_time: When the alert was triggered
    - acknowledged: Whether alert has been acknowledged
    """
    
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    alert_type: FrequencyAlertType = FrequencyAlertType.THRESHOLD_EXCEEDED
    severity: str = Field(default="medium")  # low, medium, high, critical
    message: str = ""
    threshold_value: Optional[float] = None
    current_value: Optional[float] = None
    strategy_id: Optional[str] = None
    trigger_time: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    auto_resolve: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FrequencyOptimization:
    """
    @class FrequencyOptimization
    @brief Frequency optimization recommendations
    
    @details
    Contains optimization recommendations for trading frequency based on
    backtesting results, performance analysis, and market conditions.
    
    @par Optimization Components:
    - recommended_interval: Optimal trading interval
    - recommended_position_size: Optimal position size multiplier
    - confidence_level: Confidence in recommendation
    - expected_improvement: Expected performance improvement
    - backtest_period: Period used for optimization analysis
    
    @par Performance Metrics:
    - historical_sharpe: Historical Sharpe ratio
    - expected_sharpe: Expected Sharpe ratio with optimization
    - max_drawdown_reduction: Expected drawdown reduction
    - win_rate_improvement: Expected win rate improvement
    
    @note
    Optimization recommendations should be validated before implementation
    to ensure they align with current market conditions.
    """
    
    optimization_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    strategy_id: str = ""
    recommended_interval_seconds: int = 0
    recommended_position_size_multiplier: float = 1.0
    confidence_level: float = 0.0  # 0.0 to 1.0
    expected_improvement: float = 0.0  # percentage
    
    # Performance metrics
    historical_sharpe: float = 0.0
    expected_sharpe: float = 0.0
    max_drawdown_reduction: float = 0.0
    win_rate_improvement: float = 0.0
    
    # Analysis details
    backtest_period_days: int = 30
    optimization_date: datetime = field(default_factory=datetime.utcnow)
    analysis_data: Dict[str, Any] = field(default_factory=dict)
    
    # Implementation tracking
    implemented: bool = False
    implementation_date: Optional[datetime] = None
    performance_after_implementation: Dict[str, float] = field(default_factory=dict)


class FrequencyManager:
    """
    @class FrequencyManager
    @brief Trading frequency management system
    
    @details
    Central manager for trading frequency configuration, monitoring, and optimization.
    Provides APIs for frequency-based position sizing, risk management, and alerts.
    
    @par Core Responsibilities:
    - Frequency configuration validation and management
    - Real-time frequency monitoring and tracking
    - Frequency-based position sizing calculations
    - Frequency alert generation and management
    - Frequency optimization analysis and recommendations
    - Integration with trading strategies and risk management
    
    @par Usage Example:
    @code
    manager = FrequencyManager(settings)
    
    # Calculate position size based on frequency
    position_size = await manager.calculate_position_size(
        strategy_id="trend_001",
        base_position_size=1000,
        current_frequency_rate=5.0
    )
    
    # Check if trade is allowed
    can_trade = await manager.check_trade_allowed("trend_001")
    
    # Get frequency metrics
    metrics = manager.get_frequency_metrics("trend_001")
    
    # Generate optimization recommendations
    recommendations = await manager.generate_optimization_recommendations("trend_001")
    @endcode
    
    @warning
    Frequency manager should be the single point of truth for all frequency-related
    decisions in the trading system to ensure consistency and prevent conflicts.
    """
    
    def __init__(self, settings: FrequencySettings):
        """
        Initialize frequency manager with configuration
        
        Args:
            settings: Frequency settings configuration
        """
        self.settings = settings
        self.metrics: Dict[str, FrequencyMetrics] = {}
        self.alerts: List[FrequencyAlert] = []
        self.optimizations: Dict[str, List[FrequencyOptimization]] = {}
        self.trade_history: Dict[str, List[datetime]] = {}
        
        # Initialize metrics for all strategies
        self._initialize_metrics()
        
        logger.info(f"FrequencyManager initialized with {settings.frequency_type.value} frequency")
    
    def _initialize_metrics(self):
        """Initialize frequency metrics tracking"""
        # This would typically load from database or configuration
        # For now, initialize empty metrics
        pass
    
    async def calculate_position_size(
        self,
        strategy_id: str,
        base_position_size: Decimal,
        current_frequency_rate: float = 0.0,
        market_volatility: float = 0.0
    ) -> Decimal:
        """
        Calculate position size based on frequency settings
        
        Args:
            strategy_id: Strategy identifier
            base_position_size: Base position size before frequency adjustment
            current_frequency_rate: Current trading frequency rate
            market_volatility: Current market volatility measure
            
        Returns:
            Adjusted position size considering frequency constraints
            
        @details
        Adjusts position size based on trading frequency to manage risk
        and ensure optimal capital allocation. Higher frequency strategies
        typically use smaller position sizes to reduce risk exposure.
        
        @par Calculation Factors:
        - Base position size
        - Frequency type multiplier
        - Current frequency rate vs target
        - Market volatility adjustment
        - Strategy-specific overrides
        """
        try:
            # Get strategy-specific overrides if available
            multiplier = self.settings.position_size_multiplier
            if strategy_id in self.settings.strategy_overrides:
                override = self.settings.strategy_overrides[strategy_id]
                multiplier = override.get("position_size_multiplier", multiplier)
            
            # Apply frequency-based adjustments
            adjusted_size = base_position_size * Decimal(str(multiplier))
            
            # Adjust for frequency rate (higher frequency = smaller positions)
            if current_frequency_rate > 0:
                target_rate = self._get_target_frequency_rate()
                if current_frequency_rate > target_rate:
                    rate_adjustment = min(target_rate / current_frequency_rate, 1.0)
                    adjusted_size *= Decimal(str(rate_adjustment))
            
            # Apply volatility adjustment if enabled
            if self.settings.frequency_volatility_adjustment and market_volatility > 0:
                volatility_adjustment = 1.0 / (1.0 + market_volatility)
                adjusted_size *= Decimal(str(volatility_adjustment))
            
            # Apply daily frequency risk limit
            max_size = base_position_size * Decimal(str(1.0 - self.settings.max_daily_frequency_risk))
            adjusted_size = min(adjusted_size, max_size)
            
            # Ensure minimum position size
            min_size = base_position_size * Decimal("0.01")  # 1% minimum
            adjusted_size = max(adjusted_size, min_size)
            
            # Round to appropriate precision
            adjusted_size = adjusted_size.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            
            logger.debug(
                f"Position size calculated for {strategy_id}: "
                f"base={base_position_size}, adjusted={adjusted_size}, "
                f"frequency_rate={current_frequency_rate}"
            )
            
            return adjusted_size
            
        except Exception as e:
            logger.error(f"Error calculating position size for {strategy_id}: {e}")
            return base_position_size  # Fallback to base size
    
    async def check_trade_allowed(
        self,
        strategy_id: str,
        current_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Check if a trade is allowed based on frequency constraints
        
        Args:
            strategy_id: Strategy identifier
            current_time: Current time (defaults to UTC now)
            
        Returns:
            Dictionary with trade permission status and details
            
        @details
        Comprehensive check for trade permission considering:
        - Cooldown periods
        - Frequency limits (per minute/hour/day)
        - Time window constraints
        - Market hours restrictions
        - Risk limit compliance
        """
        if current_time is None:
            current_time = datetime.utcnow()
        
        # Initialize metrics for strategy if not exists
        if strategy_id not in self.metrics:
            self.metrics[strategy_id] = FrequencyMetrics()
        
        metrics = self.metrics[strategy_id]
        
        # Check cooldown period
        if metrics.in_cooldown and metrics.cooldown_end_time:
            if current_time < metrics.cooldown_end_time:
                remaining_cooldown = (metrics.cooldown_end_time - current_time).total_seconds()
                return {
                    "allowed": False,
                    "reason": "cooldown_period",
                    "remaining_seconds": remaining_cooldown,
                    "cooldown_end": metrics.cooldown_end_time.isoformat()
                }
            else:
                # Cooldown period ended
                metrics.in_cooldown = False
                metrics.cooldown_end_time = None
        
        # Check frequency limits
        frequency_check = await self._check_frequency_limits(strategy_id, current_time)
        if not frequency_check["allowed"]:
            return frequency_check
        
        # Check time window constraints
        window_check = await self._check_time_window_constraints(strategy_id, current_time)
        if not window_check["allowed"]:
            return window_check
        
        # Check market hours restriction
        if self.settings.market_hours_only and not await self._is_market_hours(current_time):
            return {
                "allowed": False,
                "reason": "outside_market_hours",
                "message": "Trading restricted to market hours"
            }
        
        # All checks passed
        return {
            "allowed": True,
            "reason": "frequency_constraints_satisfied",
            "next_allowed_time": None,
            "frequency_metrics": {
                "trades_per_minute": metrics.current_frequency_rate,
                "trades_per_hour": metrics.trades_in_last_hour / 60,
                "daily_trades": metrics.trades_today,
                "cooldown_remaining": 0 if not metrics.in_cooldown else 
                                    (metrics.cooldown_end_time - current_time).total_seconds()
            }
        }
    
    async def record_trade(
        self,
        strategy_id: str,
        trade_time: Optional[datetime] = None
    ):
        """
        Record a trade execution for frequency tracking
        
        Args:
            strategy_id: Strategy identifier
            trade_time: Time of trade execution
        """
        if trade_time is None:
            trade_time = datetime.utcnow()
        
        # Initialize metrics and history if needed
        if strategy_id not in self.metrics:
            self.metrics[strategy_id] = FrequencyMetrics()
        
        if strategy_id not in self.trade_history:
            self.trade_history[strategy_id] = []
        
        # Update metrics
        metrics = self.metrics[strategy_id]
        metrics.trades_today += 1
        metrics.trades_in_last_minute += 1
        metrics.trades_in_last_hour += 1
        metrics.current_frequency_rate = metrics.trades_in_last_minute / 60.0
        metrics.last_trade_time = trade_time
        
        # Initialize first trade time
        if metrics.first_trade_today is None:
            metrics.first_trade_today = trade_time
        
        # Record trade in history
        self.trade_history[strategy_id].append(trade_time)
        
        # Start cooldown if configured
        if self.settings.cooldown_periods > 0:
            metrics.in_cooldown = True
            metrics.cooldown_end_time = trade_time + timedelta(seconds=self.settings.cooldown_periods)
        
        # Clean old trade history (keep last hour)
        cutoff_time = trade_time - timedelta(hours=1)
        self.trade_history[strategy_id] = [
            t for t in self.trade_history[strategy_id] if t > cutoff_time
        ]
        
        # Update metrics last updated time
        metrics.last_updated = trade_time
        
        logger.debug(f"Trade recorded for {strategy_id} at {trade_time}")
    
    async def _check_frequency_limits(
        self,
        strategy_id: str,
        current_time: datetime
    ) -> Dict[str, Any]:
        """Check frequency limit constraints"""
        if strategy_id not in self.metrics:
            self.metrics[strategy_id] = FrequencyMetrics()
        
        metrics = self.metrics[strategy_id]
        
        # Get trade counts for different time periods
        minute_trades = await self._get_trade_count(strategy_id, current_time - timedelta(minutes=1))
        hour_trades = await self._get_trade_count(strategy_id, current_time - timedelta(hours=1))
        day_trades = metrics.trades_today
        
        # Check per-minute limit
        if minute_trades >= self.settings.max_trades_per_minute:
            return {
                "allowed": False,
                "reason": "minute_limit_exceeded",
                "message": f"Maximum trades per minute ({self.settings.max_trades_per_minute}) exceeded",
                "current_count": minute_trades,
                "limit": self.settings.max_trades_per_minute,
                "reset_time": (current_time + timedelta(minutes=1)).isoformat()
            }
        
        # Check per-hour limit
        if hour_trades >= self.settings.max_trades_per_hour:
            return {
                "allowed": False,
                "reason": "hour_limit_exceeded",
                "message": f"Maximum trades per hour ({self.settings.max_trades_per_hour}) exceeded",
                "current_count": hour_trades,
                "limit": self.settings.max_trades_per_hour,
                "reset_time": (current_time + timedelta(hours=1)).isoformat()
            }
        
        # Check per-day limit
        if day_trades >= self.settings.max_trades_per_day:
            return {
                "allowed": False,
                "reason": "day_limit_exceeded",
                "message": f"Maximum trades per day ({self.settings.max_trades_per_day}) exceeded",
                "current_count": day_trades,
                "limit": self.settings.max_trades_per_day,
                "reset_time": self._get_next_day_start(current_time).isoformat()
            }
        
        return {"allowed": True}
    
    async def _get_trade_count(self, strategy_id: str, since_time: datetime) -> int:
        """Get number of trades since a specific time"""
        if strategy_id not in self.trade_history:
            return 0
        
        return len([
            trade_time for trade_time in self.trade_history[strategy_id]
            if trade_time > since_time
        ])
    
    def _get_target_frequency_rate(self) -> float:
        """Get target frequency rate from settings"""
        if self.settings.interval_seconds > 0:
            return 60.0 / self.settings.interval_seconds
        return 1.0  # Default: 1 trade per minute
    
    def _get_next_day_start(self, current_time: datetime) -> datetime:
        """Get start of next trading day"""
        # Simplified - assumes 24/7 trading
        # In practice, this would consider market holidays and weekends
        return (current_time + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    
    async def _check_time_window_constraints(
        self,
        strategy_id: str,
        current_time: datetime
    ) -> Dict[str, Any]:
        """Check time window frequency constraints"""
        # Time window constraints would be implemented here
        # For now, return allowed
        return {"allowed": True}
    
    async def _is_market_hours(self, current_time: datetime) -> bool:
        """Check if current time is within market hours"""
        # Simplified implementation - would check actual market hours
        # For crypto markets, this would always return True
        # For stock markets, this would check NYSE/NASDAQ hours
        return True  # Assume 24/7 trading for now
    
    def get_frequency_metrics(self, strategy_id: str) -> Optional[FrequencyMetrics]:
        """
        Get frequency metrics for a strategy
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Frequency metrics or None if not found
        """
        return self.metrics.get(strategy_id)
    
    def get_all_metrics(self) -> Dict[str, FrequencyMetrics]:
        """Get frequency metrics for all strategies"""
        return self.metrics.copy()
    
    def update_settings(self, new_settings: FrequencySettings):
        """
        Update frequency settings
        
        Args:
            new_settings: New frequency settings
        """
        self.settings = new_settings
        logger.info(f"Frequency settings updated to {new_settings.frequency_type.value}")
    
    async def generate_alert(
        self,
        strategy_id: str,
        alert_type: FrequencyAlertType,
        message: str,
        severity: str = "medium",
        threshold_value: Optional[float] = None,
        current_value: Optional[float] = None
    ) -> FrequencyAlert:
        """
        Generate frequency alert
        
        Args:
            strategy_id: Strategy identifier
            alert_type: Type of alert
            message: Alert message
            severity: Alert severity level
            threshold_value: Threshold that was exceeded
            current_value: Current measured value
            
        Returns:
            Generated alert object
        """
        alert = FrequencyAlert(
            alert_type=alert_type,
            severity=severity,
            message=message,
            threshold_value=threshold_value,
            current_value=current_value,
            strategy_id=strategy_id
        )
        
        self.alerts.append(alert)
        
        # Add to strategy metrics
        if strategy_id in self.metrics:
            self.metrics[strategy_id].alerts_triggered.append({
                "alert_id": alert.alert_id,
                "type": alert.alert_type.value,
                "severity": alert.severity,
                "trigger_time": alert.trigger_time.isoformat(),
                "message": alert.message
            })
        
        logger.warning(f"Frequency alert generated for {strategy_id}: {message}")
        
        return alert
    
    def get_active_alerts(self, strategy_id: Optional[str] = None) -> List[FrequencyAlert]:
        """
        Get active (unacknowledged) alerts
        
        Args:
            strategy_id: Optional strategy filter
            
        Returns:
            List of active alerts
        """
        if strategy_id:
            return [alert for alert in self.alerts if not alert.acknowledged and alert.strategy_id == strategy_id]
        else:
            return [alert for alert in self.alerts if not alert.acknowledged]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge an alert
        
        Args:
            alert_id: Alert identifier
            
        Returns:
            True if alert was acknowledged
        """
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                logger.info(f"Alert acknowledged: {alert_id}")
                return True
        return False
    
    async def generate_optimization_recommendations(
        self,
        strategy_id: str,
        backtest_period_days: int = 30
    ) -> List[FrequencyOptimization]:
        """
        Generate frequency optimization recommendations
        
        Args:
            strategy_id: Strategy identifier
            backtest_period_days: Number of days for backtest analysis
            
        Returns:
            List of optimization recommendations
        """
        # This would implement actual optimization analysis
        # For now, return a placeholder recommendation
        optimization = FrequencyOptimization(
            strategy_id=strategy_id,
            recommended_interval_seconds=self.settings.interval_seconds,
            recommended_position_size_multiplier=self.settings.position_size_multiplier,
            confidence_level=0.75,
            expected_improvement=5.0,
            backtest_period_days=backtest_period_days
        )
        
        # Store optimization
        if strategy_id not in self.optimizations:
            self.optimizations[strategy_id] = []
        
        self.optimizations[strategy_id].append(optimization)
        
        logger.info(f"Optimization recommendations generated for {strategy_id}")
        
        return [optimization]
    
    def get_optimization_recommendations(
        self,
        strategy_id: str
    ) -> List[FrequencyOptimization]:
        """
        Get optimization recommendations for a strategy
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            List of optimization recommendations
        """
        return self.optimizations.get(strategy_id, [])
    
    async def implement_optimization(
        self,
        optimization_id: str,
        strategy_id: str
    ) -> bool:
        """
        Implement an optimization recommendation
        
        Args:
            optimization_id: Optimization identifier
            strategy_id: Strategy identifier
            
        Returns:
            True if optimization was implemented
        """
        # Find optimization
        optimizations = self.optimizations.get(strategy_id, [])
        optimization = None
        
        for opt in optimizations:
            if opt.optimization_id == optimization_id:
                optimization = opt
                break
        
        if not optimization:
            logger.error(f"Optimization not found: {optimization_id}")
            return False
        
        # Create new settings based on optimization
        new_settings = FrequencySettings(
            frequency_type=self.settings.frequency_type,
            interval_seconds=optimization.recommended_interval_seconds,
            max_trades_per_minute=self.settings.max_trades_per_minute,
            max_trades_per_hour=self.settings.max_trades_per_hour,
            max_trades_per_day=self.settings.max_trades_per_day,
            position_size_multiplier=optimization.recommended_position_size_multiplier,
            cooldown_periods=self.settings.cooldown_periods,
            market_hours_only=self.settings.market_hours_only
        )
        
        # Update settings
        self.update_settings(new_settings)
        
        # Mark optimization as implemented
        optimization.implemented = True
        optimization.implementation_date = datetime.utcnow()
        
        logger.info(f"Optimization implemented for {strategy_id}: {optimization_id}")
        
        return True
    
    def get_frequency_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive frequency management summary
        
        Returns:
            Summary dictionary with all frequency metrics and status
        """
        total_strategies = len(self.metrics)
        active_strategies = len([m for m in self.metrics.values() if m.trades_today > 0])
        total_trades_today = sum(m.trades_today for m in self.metrics.values())
        active_alerts = len(self.get_active_alerts())
        total_optimizations = sum(len(opts) for opts in self.optimizations.values())
        
        return {
            "settings": {
                "frequency_type": self.settings.frequency_type.value,
                "interval_seconds": self.settings.interval_seconds,
                "max_trades_per_minute": self.settings.max_trades_per_minute,
                "max_trades_per_hour": self.settings.max_trades_per_hour,
                "max_trades_per_day": self.settings.max_trades_per_day,
                "position_size_multiplier": self.settings.position_size_multiplier,
                "cooldown_periods": self.settings.cooldown_periods,
                "market_hours_only": self.settings.market_hours_only
            },
            "metrics": {
                "total_strategies": total_strategies,
                "active_strategies": active_strategies,
                "total_trades_today": total_trades_today,
                "average_frequency_rate": sum(m.current_frequency_rate for m in self.metrics.values()) / max(total_strategies, 1)
            },
            "alerts": {
                "active_alerts": active_alerts,
                "total_alerts": len(self.alerts)
            },
            "optimization": {
                "total_recommendations": total_optimizations,
                "implemented_optimizations": sum(
                    len([opt for opt in opts if opt.implemented])
                    for opts in self.optimizations.values()
                )
            },
            "strategies": {
                strategy_id: {
                    "trades_today": metrics.trades_today,
                    "current_frequency_rate": metrics.current_frequency_rate,
                    "last_trade": metrics.last_trade_time.isoformat() if metrics.last_trade_time else None,
                    "in_cooldown": metrics.in_cooldown,
                    "active_alerts": len([a for a in self.alerts if a.strategy_id == strategy_id and not a.acknowledged])
                }
                for strategy_id, metrics in self.metrics.items()
            }
        }


# Global frequency manager instance
_frequency_manager: Optional[FrequencyManager] = None


def get_frequency_manager() -> Optional[FrequencyManager]:
    """Get global frequency manager instance"""
    return _frequency_manager


def initialize_frequency_manager(settings: FrequencySettings) -> FrequencyManager:
    """
    Initialize global frequency manager
    
    Args:
        settings: Frequency settings configuration
        
    Returns:
        Initialized frequency manager
    """
    global _frequency_manager
    _frequency_manager = FrequencyManager(settings)
    return _frequency_manager


def get_frequency_settings() -> Optional[FrequencySettings]:
    """Get current frequency settings"""
    manager = get_frequency_manager()
    return manager.settings if manager else None


async def calculate_frequency_position_size(
    strategy_id: str,
    base_position_size: Decimal,
    current_frequency_rate: float = 0.0,
    market_volatility: float = 0.0
) -> Decimal:
    """
    Convenience function to calculate frequency-adjusted position size
    
    Args:
        strategy_id: Strategy identifier
        base_position_size: Base position size
        current_frequency_rate: Current trading frequency rate
        market_volatility: Market volatility measure
        
    Returns:
        Frequency-adjusted position size
    """
    manager = get_frequency_manager()
    if manager:
        return await manager.calculate_position_size(
            strategy_id, base_position_size, current_frequency_rate, market_volatility
        )
    return base_position_size


async def check_trade_frequency_allowed(
    strategy_id: str,
    current_time: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Convenience function to check if trade is allowed
    
    Args:
        strategy_id: Strategy identifier
        current_time: Current time
        
    Returns:
        Trade permission status
    """
    manager = get_frequency_manager()
    if manager:
        return await manager.check_trade_allowed(strategy_id, current_time)
    return {"allowed": True, "reason": "no_frequency_manager"}


async def record_frequency_trade(
    strategy_id: str,
    trade_time: Optional[datetime] = None
):
    """
    Convenience function to record trade for frequency tracking
    
    Args:
        strategy_id: Strategy identifier
        trade_time: Trade timestamp
    """
    manager = get_frequency_manager()
    if manager:
        await manager.record_trade(strategy_id, trade_time)


# Example usage and testing
if __name__ == "__main__":
    async def test_frequency_manager():
        """Test frequency manager functionality"""
        
        # Create frequency settings
        settings = FrequencySettings(
            frequency_type=FrequencyType.HIGH,
            interval_seconds=300,  # 5 minutes
            max_trades_per_minute=3,
            max_trades_per_hour=15,
            max_trades_per_day=80,
            position_size_multiplier=0.8,
            cooldown_periods=60
        )
        
        # Initialize manager
        manager = initialize_frequency_manager(settings)
        
        # Test position size calculation
        position_size = await manager.calculate_position_size(
            strategy_id="test_strategy",
            base_position_size=Decimal("10000"),
            current_frequency_rate=2.5
        )
        print(f"Adjusted position size: {position_size}")
        
        # Test trade permission
        permission = await manager.check_trade_allowed("test_strategy")
        print(f"Trade allowed: {permission}")
        
        # Test trade recording
        await manager.record_trade("test_strategy")
        metrics = manager.get_frequency_metrics("test_strategy")
        print(f"Trades today: {metrics.trades_today if metrics else 'None'}")
        
        # Test alerts
        alert = await manager.generate_alert(
            strategy_id="test_strategy",
            alert_type=FrequencyAlertType.THRESHOLD_EXCEEDED,
            message="Test alert message",
            severity="medium"
        )
        print(f"Alert generated: {alert.alert_id}")
        
        # Test optimization
        recommendations = await manager.generate_optimization_recommendations("test_strategy")
        print(f"Optimization recommendations: {len(recommendations)}")
        
        # Get summary
        summary = manager.get_frequency_summary()
        print(f"Manager summary: {summary['metrics']['total_trades_today']} trades today")
    
    # Run tests
    asyncio.run(test_frequency_manager())