"""
Daily Trading Limits and Risk Controls

Comprehensive daily trading safeguards including:
- Daily loss limits
- Daily trade count limits
- Daily volume limits
- Time-based trading restrictions
- Cooldown periods
- Automatic trading halts

@author Trading Orchestrator System
@version 1.0
@date 2025-11-06
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, time
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import pandas as pd

from database.models.risk import RiskLevel
from database.models.trading import Position, Order, Trade
from database.models.user import User
from config.database import get_db

logger = logging.getLogger(__name__)


class LimitType(Enum):
    """Types of daily trading limits."""
    DAILY_LOSS = "daily_loss"
    DAILY_PNL = "daily_pnl"
    DAILY_TRADE_COUNT = "daily_trade_count"
    DAILY_ORDER_COUNT = "daily_order_count"
    DAILY_VOLUME = "daily_volume"
    DAILY_NOTIONAL = "daily_notional"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    WIN_RATE = "win_rate"


class LimitAction(Enum):
    """Actions when daily limits are breached."""
    WARN = "warn"
    HALT_TRADING = "halt_trading"
    REDUCE_POSITIONS = "reduce_positions"
    INCREASE_COOLDOWN = "increase_cooldown"
    ENHANCE_MONITORING = "enhance_monitoring"


class TradingSession(Enum):
    """Trading sessions for time-based restrictions."""
    PRE_MARKET = "pre_market"      # 4:00-9:30 AM
    REGULAR = "regular"            # 9:30 AM-4:00 PM
    AFTER_HOURS = "after_hours"    # 4:00-8:00 PM
    OVERNIGHT = "overnight"        # 8:00 PM-4:00 AM


@dataclass
class DailyLimit:
    """Daily trading limit configuration."""
    limit_id: Optional[int] = None
    limit_name: str = ""
    limit_type: LimitType = LimitType.DAILY_LOSS
    limit_value: float = 0.0
    warning_threshold: float = 0.0
    action_on_breach: LimitAction = LimitAction.WARN
    cooldown_hours: int = 0
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    scope: str = "account"  # account, strategy, symbol, sector


@dataclass
class DailyUsage:
    """Daily limit usage tracking."""
    limit_id: int
    current_value: float
    usage_percentage: float
    limit_reached: bool
    warning_triggered: bool
    last_updated: datetime = field(default_factory=datetime.now)
    breach_count_today: int = 0


@dataclass
class TradingSessionLimit:
    """Trading session specific limits."""
    session: TradingSession
    max_trades: Optional[int] = None
    max_loss: Optional[float] = None
    max_volume: Optional[float] = None
    allowed: bool = True
    cooldown_after_breach: int = 60  # minutes


class DailyLimitManager:
    """
    Daily Trading Limits and Risk Management System
    
    Monitors and enforces daily trading limits to prevent excessive risk
    taking and ensure responsible trading behavior.
    """
    
    def __init__(self, user_id: int):
        """
        Initialize Daily Limit Manager.
        
        Args:
            user_id: User identifier for limit tracking
        """
        self.user_id = user_id
        self.db = get_db()
        
        # Default daily limits
        self.default_limits = {
            LimitType.DAILY_LOSS: DailyLimit(
                limit_name="Daily Loss Limit",
                limit_value=1000.0,
                warning_threshold=750.0,
                action_on_breach=LimitAction.HALT_TRADING,
                cooldown_hours=24
            ),
            LimitType.DAILY_TRADE_COUNT: DailyLimit(
                limit_name="Daily Trade Count",
                limit_value=50.0,
                warning_threshold=40.0,
                action_on_breach=LimitAction.WARN,
                cooldown_hours=2
            ),
            LimitType.DAILY_VOLUME: DailyLimit(
                limit_name="Daily Volume Limit",
                limit_value=1000000.0,
                warning_threshold=800000.0,
                action_on_breach=LimitAction.WARN,
                cooldown_hours=1
            )
        }
        
        # Current limits (loaded from database or defaults)
        self.active_limits: Dict[LimitType, DailyLimit] = {}
        self.limit_usage: Dict[LimitType, DailyUsage] = {}
        
        # Trading session configuration
        self.session_limits = {
            TradingSession.PRE_MARKET: TradingSessionLimit(
                session=TradingSession.PRE_MARKET,
                max_trades=10,
                max_loss=200.0,
                allowed=True
            ),
            TradingSession.REGULAR: TradingSessionLimit(
                session=TradingSession.REGULAR,
                max_trades=100,
                allowed=True
            ),
            TradingSession.AFTER_HOURS: TradingSessionLimit(
                session=TradingSession.AFTER_HOURS,
                max_trades=20,
                max_loss=500.0,
                allowed=True
            ),
            TradingSession.OVERNIGHT: TradingSessionLimit(
                session=TradingSession.OVERNIGHT,
                max_trades=5,
                max_loss=100.0,
                allowed=False  # Disabled by default
            )
        }
        
        # Cooldown tracking
        self.cooldown_until = None
        self.trading_halted_until = None
        self.last_limit_breach = None
        
        # Load active limits
        asyncio.create_task(self._load_active_limits())
        
        logger.info(f"DailyLimitManager initialized for user {self.user_id}")
    
    async def check_daily_limits(self) -> Dict[str, Any]:
        """
        Check all daily limits against current usage.
        
        Returns:
            Comprehensive daily limits status
        """
        try:
            today = datetime.now().date()
            limit_status = {
                "date": today,
                "limits_checked": 0,
                "limits_breached": [],
                "warnings": [],
                "halted": False,
                "cooldown_active": False,
                "session_status": {},
                "timestamp": datetime.now()
            }
            
            # Check if we're in cooldown period
            if self.cooldown_until and datetime.now() < self.cooldown_until:
                limit_status["cooldown_active"] = True
                limit_status["cooldown_until"] = self.cooldown_until
                return limit_status
            
            # Get today's trading data
            daily_data = await self._get_daily_trading_data(today)
            
            # Check each active limit
            for limit_type, limit_config in self.active_limits.items():
                if not limit_config.is_active:
                    continue
                
                current_value = await self._calculate_limit_value(limit_type, daily_data)
                
                # Update usage tracking
                usage = DailyUsage(
                    limit_id=limit_config.limit_id or 0,
                    current_value=current_value,
                    usage_percentage=(current_value / limit_config.limit_value * 100) if limit_config.limit_value > 0 else 0,
                    limit_reached=current_value >= limit_config.limit_value,
                    warning_triggered=current_value >= limit_config.warning_threshold
                )
                self.limit_usage[limit_type] = usage
                
                # Check for breaches
                if usage.limit_reached:
                    limit_status["limits_breached"].append({
                        "limit_name": limit_config.limit_name,
                        "limit_type": limit_type.value,
                        "current_value": current_value,
                        "limit_value": limit_config.limit_value,
                        "usage_percentage": usage.usage_percentage,
                        "action_required": limit_config.action_on_breach.value,
                        "cooldown_hours": limit_config.cooldown_hours
                    })
                    
                    # Increment breach count
                    usage.breach_count_today += 1
                    
                    # Trigger limit breach action
                    breach_action = await self._handle_limit_breach(limit_config, usage)
                    if breach_action.get("halt_trading"):
                        limit_status["halted"] = True
                        self.trading_halted_until = datetime.now() + timedelta(hours=limit_config.cooldown_hours)
                
                # Check for warnings
                elif usage.warning_triggered:
                    limit_status["warnings"].append({
                        "limit_name": limit_config.limit_name,
                        "limit_type": limit_type.value,
                        "current_value": current_value,
                        "warning_threshold": limit_config.warning_threshold,
                        "usage_percentage": usage.usage_percentage,
                        "message": f"Approaching {limit_config.limit_name}: {usage.usage_percentage:.1f}%"
                    })
                
                limit_status["limits_checked"] += 1
            
            # Check trading session limits
            current_session = self._get_current_trading_session()
            session_limit = self.session_limits.get(current_session)
            if session_limit:
                session_status = await self._check_session_limits(current_session, daily_data, session_limit)
                limit_status["session_status"][current_session.value] = session_status
                
                if not session_status["allowed"]:
                    limit_status["halted"] = True
                    limit_status["session_halt_reason"] = session_status["reason"]
            
            return limit_status
            
        except Exception as e:
            logger.error(f"Daily limits check error: {str(e)}")
            return {
                "error": str(e),
                "halted": True,  # Safety halt on error
                "timestamp": datetime.now()
            }
    
    async def validate_trade_against_daily_limits(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate proposed trade against daily limits.
        
        Args:
            order_data: Order information to validate
            
        Returns:
            Validation result with daily limit checks
        """
        try:
            # Check basic limits status
            limit_status = await self.check_daily_limits()
            
            # If trading is halted, reject trade
            if limit_status.get("halted"):
                return {
                    "approved": False,
                    "reason": "Daily trading limits reached - trading halted",
                    "limit_status": limit_status,
                    "halt_until": self.trading_halted_until
                }
            
            # Check cooldown
            if limit_status.get("cooldown_active"):
                return {
                    "approved": False,
                    "reason": "Trading cooldown active",
                    "limit_status": limit_status,
                    "cooldown_until": self.cooldown_until
                }
            
            validation_result = {
                "approved": True,
                "limit_status": limit_status,
                "warnings": [],
                "post_trade_projections": {}
            }
            
            # Project impact of this trade
            trade_projection = await self._project_trade_impact(order_data)
            validation_result["post_trade_projections"] = trade_projection
            
            # Check if trade would breach limits
            for limit_type, projection in trade_projection.items():
                if projection["would_breach"]:
                    if projection["breach_severity"] == "critical":
                        return {
                            "approved": False,
                            "reason": f"Trade would breach {limit_type.value} limit",
                            "limit_status": limit_status,
                            "projection": projection
                        }
                    else:
                        validation_result["warnings"].append(
                            f"Trade approaches {limit_type.value} limit: {projection['projected_usage']:.1f}%"
                        )
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Trade validation error: {str(e)}")
            return {
                "approved": False,
                "reason": f"Validation error: {str(e)}"
            }
    
    async def update_daily_limit(self, limit_type: LimitType, new_value: float, 
                               warning_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Update daily limit value.
        
        Args:
            limit_type: Type of limit to update
            new_value: New limit value
            warning_threshold: New warning threshold (optional)
            
        Returns:
            Update result
        """
        try:
            if limit_type not in self.active_limits:
                # Create new limit
                limit_config = DailyLimit(
                    limit_name=limit_type.value.replace("_", " ").title(),
                    limit_type=limit_type,
                    limit_value=new_value,
                    warning_threshold=warning_threshold or (new_value * 0.8)
                )
                self.active_limits[limit_type] = limit_config
            else:
                # Update existing limit
                limit_config = self.active_limits[limit_type]
                limit_config.limit_value = new_value
                if warning_threshold is not None:
                    limit_config.warning_threshold = warning_threshold
                else:
                    limit_config.warning_threshold = new_value * 0.8
            
            logger.info(f"Daily limit updated for user {self.user_id}: {limit_type.value} = {new_value}")
            
            return {
                "success": True,
                "limit_type": limit_type.value,
                "new_value": new_value,
                "warning_threshold": limit_config.warning_threshold,
                "updated_at": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Daily limit update error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def reset_daily_limits(self, confirm_reset: bool = False) -> Dict[str, Any]:
        """
        Reset daily limits usage (typically called at market open).
        
        Args:
            confirm_reset: Confirmation flag to prevent accidental resets
            
        Returns:
            Reset result
        """
        try:
            if not confirm_reset:
                return {"success": False, "error": "Reset confirmation required"}
            
            # Clear usage tracking
            self.limit_usage.clear()
            
            # Reset cooldown
            self.cooldown_until = None
            self.trading_halted_until = None
            
            # Reset breach tracking
            self.last_limit_breach = None
            
            logger.info(f"Daily limits reset for user {self.user_id}")
            
            return {
                "success": True,
                "reset_at": datetime.now(),
                "limits_cleared": list(self.active_limits.keys()),
                "cooldown_cleared": True,
                "trading_resumed": True
            }
            
        except Exception as e:
            logger.error(f"Daily limits reset error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def get_daily_limits_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of daily limits and usage.
        
        Returns:
            Daily limits summary report
        """
        try:
            current_status = await self.check_daily_limits()
            usage_summary = {}
            
            for limit_type, usage in self.limit_usage.items():
                limit_config = self.active_limits.get(limit_type)
                if limit_config:
                    usage_summary[limit_type.value] = {
                        "limit_name": limit_config.limit_name,
                        "limit_value": limit_config.limit_value,
                        "current_value": usage.current_value,
                        "usage_percentage": usage.usage_percentage,
                        "warning_threshold": limit_config.warning_threshold,
                        "warning_triggered": usage.warning_triggered,
                        "limit_breached": usage.limit_reached,
                        "breach_count_today": usage.breach_count_today
                    }
            
            return {
                "current_status": current_status,
                "usage_summary": usage_summary,
                "active_limits": len(self.active_limits),
                "limits_by_type": {limit_type.value: limit.limit_name 
                                 for limit_type, limit in self.active_limits.items()},
                "cooldown_status": {
                    "active": self.cooldown_until is not None and datetime.now() < self.cooldown_until,
                    "until": self.cooldown_until
                },
                "trading_halt_status": {
                    "active": self.trading_halted_until is not None and datetime.now() < self.trading_halted_until,
                    "until": self.trading_halted_until
                },
                "last_breach": self.last_limit_breach,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Daily limits summary error: {str(e)}")
            return {"error": str(e)}
    
    async def configure_trading_session_limit(self, session: TradingSession, 
                                            limits: Dict[str, Any]) -> Dict[str, Any]:
        """
        Configure limits for specific trading sessions.
        
        Args:
            session: Trading session to configure
            limits: Session limit settings
            
        Returns:
            Configuration result
        """
        try:
            if session not in self.session_limits:
                return {"success": False, "error": f"Unknown trading session: {session.value}"}
            
            session_limit = self.session_limits[session]
            
            # Update session limits
            if "max_trades" in limits:
                session_limit.max_trades = limits["max_trades"]
            if "max_loss" in limits:
                session_limit.max_loss = limits["max_loss"]
            if "max_volume" in limits:
                session_limit.max_volume = limits["max_volume"]
            if "allowed" in limits:
                session_limit.allowed = limits["allowed"]
            if "cooldown_after_breach" in limits:
                session_limit.cooldown_after_breach = limits["cooldown_after_breach"]
            
            logger.info(f"Trading session limits configured for {session.value}: {limits}")
            
            return {
                "success": True,
                "session": session.value,
                "updated_limits": {
                    "max_trades": session_limit.max_trades,
                    "max_loss": session_limit.max_loss,
                    "max_volume": session_limit.max_volume,
                    "allowed": session_limit.allowed,
                    "cooldown_after_breach": session_limit.cooldown_after_breach
                },
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Session limit configuration error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def force_trading_halt(self, reason: str = "Manual halt", 
                               duration_hours: int = 24) -> Dict[str, Any]:
        """
        Force trading halt due to risk concerns.
        
        Args:
            reason: Reason for halt
            duration_hours: Duration of halt in hours
            
        Returns:
            Halt result
        """
        try:
            halt_until = datetime.now() + timedelta(hours=duration_hours)
            
            self.trading_halted_until = halt_until
            self.cooldown_until = halt_until
            
            logger.critical(f"Trading halt enforced for user {self.user_id}: {reason} until {halt_until}")
            
            return {
                "success": True,
                "halt_reason": reason,
                "halted_until": halt_until,
                "duration_hours": duration_hours,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Trading halt error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_cooldown_status(self) -> Dict[str, Any]:
        """Get current cooldown status."""
        now = datetime.now()
        
        return {
            "in_cooldown": self.cooldown_until is not None and now < self.cooldown_until,
            "cooldown_until": self.cooldown_until,
            "remaining_cooldown": (self.cooldown_until - now).total_seconds() if self.cooldown_until and now < self.cooldown_until else 0,
            "in_halt": self.trading_halted_until is not None and now < self.trading_halted_until,
            "halt_until": self.trading_halted_until,
            "remaining_halt": (self.trading_halted_until - now).total_seconds() if self.trading_halted_until and now < self.trading_halted_until else 0
        }
    
    async def _load_active_limits(self):
        """Load active limits from database."""
        try:
            # In practice, this would load from database
            # For now, use defaults
            self.active_limits = self.default_limits.copy()
            logger.info(f"Loaded {len(self.active_limits)} active daily limits for user {self.user_id}")
            
        except Exception as e:
            logger.error(f"Active limits loading error: {str(e)}")
            # Fallback to defaults
            self.active_limits = self.default_limits.copy()
    
    async def _get_daily_trading_data(self, check_date: datetime.date) -> Dict[str, Any]:
        """Get trading data for specified date."""
        try:
            # Get trades for the date
            start_time = datetime.combine(check_date, datetime.min.time())
            end_time = datetime.combine(check_date + timedelta(days=1), datetime.min.time())
            
            trades = self.db.query(Trade).filter(
                and_(
                    Trade.user_id == self.user_id,
                    Trade.executed_at >= start_time,
                    Trade.executed_at < end_time
                )
            ).all()
            
            # Get orders for the date
            orders = self.db.query(Order).filter(
                and_(
                    Order.user_id == self.user_id,
                    Order.submitted_at >= start_time,
                    Order.submitted_at < end_time
                )
            ).all()
            
            # Calculate metrics
            total_trades = len(trades)
            total_orders = len(orders)
            total_pnl = sum(trade.realized_pnl or 0 for trade in trades)
            daily_loss = abs(min(0, total_pnl))  # Positive value for losses
            total_volume = sum(abs(trade.quantity * trade.executed_price) for trade in trades)
            total_notional = total_volume
            
            # Count consecutive losses
            consecutive_losses = await self._calculate_consecutive_losses(trades)
            
            # Calculate win rate
            winning_trades = sum(1 for trade in trades if (trade.realized_pnl or 0) > 0)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 100
            
            return {
                "date": check_date,
                "trades": trades,
                "orders": orders,
                "total_trades": total_trades,
                "total_orders": total_orders,
                "total_pnl": total_pnl,
                "daily_loss": daily_loss,
                "total_volume": total_volume,
                "total_notional": total_notional,
                "consecutive_losses": consecutive_losses,
                "win_rate": win_rate
            }
            
        except Exception as e:
            logger.error(f"Daily trading data retrieval error: {str(e)}")
            return {"date": check_date, "trades": [], "orders": []}
    
    async def _calculate_limit_value(self, limit_type: LimitType, daily_data: Dict) -> float:
        """Calculate current value for specific limit type."""
        try:
            if limit_type == LimitType.DAILY_LOSS:
                return daily_data.get("daily_loss", 0)
            elif limit_type == LimitType.DAILY_PNL:
                return abs(daily_data.get("total_pnl", 0))
            elif limit_type == LimitType.DAILY_TRADE_COUNT:
                return daily_data.get("total_trades", 0)
            elif limit_type == LimitType.DAILY_ORDER_COUNT:
                return daily_data.get("total_orders", 0)
            elif limit_type == LimitType.DAILY_VOLUME:
                return daily_data.get("total_volume", 0)
            elif limit_type == LimitType.DAILY_NOTIONAL:
                return daily_data.get("total_notional", 0)
            elif limit_type == LimitType.CONSECUTIVE_LOSSES:
                return daily_data.get("consecutive_losses", 0)
            elif limit_type == LimitType.WIN_RATE:
                return daily_data.get("win_rate", 100)
            else:
                return 0
                
        except Exception as e:
            logger.error(f"Limit value calculation error: {str(e)}")
            return 0
    
    async def _handle_limit_breach(self, limit_config: DailyLimit, usage: DailyUsage) -> Dict[str, Any]:
        """Handle daily limit breach with appropriate actions."""
        try:
            action_taken = None
            breach_recorded = datetime.now()
            
            if limit_config.action_on_breach == LimitAction.HALT_TRADING:
                # Set cooldown period
                self.cooldown_until = datetime.now() + timedelta(hours=limit_config.cooldown_hours)
                self.trading_halted_until = self.cooldown_until
                action_taken = "trading_halted"
                
            elif limit_config.action_on_breach == LimitAction.INCREASE_COOLDOWN:
                # Extend existing cooldown
                if self.cooldown_until:
                    self.cooldown_until += timedelta(hours=limit_config.cooldown_hours)
                else:
                    self.cooldown_until = datetime.now() + timedelta(hours=limit_config.cooldown_hours)
                action_taken = "cooldown_extended"
                
            elif limit_config.action_on_breach == LimitAction.REDUCE_POSITIONS:
                # Would trigger position reduction logic
                action_taken = "position_reduction_triggered"
                
            elif limit_config.action_on_breach == LimitAction.ENHANCE_MONITORING:
                # Would enable enhanced monitoring
                action_taken = "enhanced_monitoring_enabled"
            
            self.last_limit_breach = breach_recorded
            
            logger.warning(f"Daily limit breach for user {self.user_id}: {limit_config.limit_name} = {usage.current_value}")
            
            return {
                "action_taken": action_taken,
                "breach_recorded": breach_recorded,
                "cooldown_until": self.cooldown_until,
                "halt_trading": limit_config.action_on_breach == LimitAction.HALT_TRADING
            }
            
        except Exception as e:
            logger.error(f"Limit breach handling error: {str(e)}")
            return {"error": str(e)}
    
    async def _calculate_consecutive_losses(self, trades: List[Trade]) -> int:
        """Calculate consecutive losing trades."""
        try:
            if not trades:
                return 0
            
            # Sort trades by execution time (most recent first)
            sorted_trades = sorted(trades, key=lambda x: x.executed_at, reverse=True)
            
            consecutive_losses = 0
            for trade in sorted_trades:
                if (trade.realized_pnl or 0) < 0:
                    consecutive_losses += 1
                else:
                    break
            
            return consecutive_losses
            
        except Exception as e:
            logger.error(f"Consecutive losses calculation error: {str(e)}")
            return 0
    
    def _get_current_trading_session(self) -> TradingSession:
        """Determine current trading session based on time."""
        try:
            now = datetime.now().time()
            
            if time(4, 0) <= now < time(9, 30):
                return TradingSession.PRE_MARKET
            elif time(9, 30) <= now < time(16, 0):
                return TradingSession.REGULAR
            elif time(16, 0) <= now < time(20, 0):
                return TradingSession.AFTER_HOURS
            else:
                return TradingSession.OVERNIGHT
                
        except Exception:
            return TradingSession.REGULAR
    
    async def _check_session_limits(self, session: TradingSession, daily_data: Dict, 
                                  session_limit: TradingSessionLimit) -> Dict[str, Any]:
        """Check trading session specific limits."""
        try:
            session_status = {
                "session": session.value,
                "allowed": session_limit.allowed,
                "violations": [],
                "within_limits": True
            }
            
            if not session_limit.allowed:
                session_status["violations"].append("Trading not allowed in this session")
                session_status["within_limits"] = False
                return session_status
            
            # Check trade count
            if session_limit.max_trades is not None:
                current_trades = daily_data.get("total_trades", 0)
                if current_trades >= session_limit.max_trades:
                    session_status["violations"].append(f"Max trades exceeded: {current_trades}/{session_limit.max_trades}")
                    session_status["within_limits"] = False
            
            # Check loss limit
            if session_limit.max_loss is not None:
                current_loss = daily_data.get("daily_loss", 0)
                if current_loss >= session_limit.max_loss:
                    session_status["violations"].append(f"Max loss exceeded: ${current_loss:.2f}/${session_limit.max_loss:.2f}")
                    session_status["within_limits"] = False
            
            # Check volume limit
            if session_limit.max_volume is not None:
                current_volume = daily_data.get("total_volume", 0)
                if current_volume >= session_limit.max_volume:
                    session_status["violations"].append(f"Max volume exceeded: ${current_volume:.2f}/${session_limit.max_volume:.2f}")
                    session_status["within_limits"] = False
            
            session_status["allowed"] = session_status["within_limits"]
            
            return session_status
            
        except Exception as e:
            logger.error(f"Session limits check error: {str(e)}")
            return {
                "session": session.value,
                "allowed": False,
                "violations": [f"Session limit check error: {str(e)}"],
                "within_limits": False
            }
    
    async def _project_trade_impact(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Project impact of proposed trade on daily limits."""
        try:
            projections = {}
            
            # Get current daily data
            today = datetime.now().date()
            current_data = await self._get_daily_trading_data(today)
            
            # Calculate trade values
            quantity = order_data.get("quantity", 0)
            price = order_data.get("limit_price", 0) or order_data.get("estimated_price", 0)
            trade_value = abs(quantity * price)
            order_fee = trade_value * 0.001  # Assume 0.1% fee
            
            # Project new values
            projected_trades = current_data.get("total_trades", 0) + 1
            projected_volume = current_data.get("total_volume", 0) + trade_value
            projected_notional = current_data.get("total_notional", 0) + trade_value
            
            # For P&L projection, we need to estimate (simplified)
            # In practice, this would use more sophisticated P&L estimation
            estimated_pnl = -order_fee  # Assume small loss for fees
            projected_pnl = current_data.get("total_pnl", 0) + estimated_pnl
            projected_daily_loss = abs(min(0, projected_pnl))
            
            # Check each limit type
            limit_types = [
                (LimitType.DAILY_TRADE_COUNT, projected_trades),
                (LimitType.DAILY_VOLUME, projected_volume),
                (LimitType.DAILY_NOTIONAL, projected_notional),
                (LimitType.DAILY_PNL, abs(projected_pnl)),
                (LimitType.DAILY_LOSS, projected_daily_loss)
            ]
            
            for limit_type, projected_value in limit_types:
                limit_config = self.active_limits.get(limit_type)
                if limit_config:
                    usage_percentage = (projected_value / limit_config.limit_value * 100) if limit_config.limit_value > 0 else 0
                    
                    would_breach = projected_value >= limit_config.limit_value
                    breach_severity = "critical" if projected_value > limit_config.limit_value * 1.1 else "warning"
                    
                    projections[limit_type] = {
                        "projected_value": projected_value,
                        "projected_usage": usage_percentage,
                        "current_value": current_data.get(limit_type.value.lower().replace("daily_", ""), 0),
                        "limit_value": limit_config.limit_value,
                        "would_breach": would_breach,
                        "breach_severity": breach_severity,
                        "headroom": limit_config.limit_value - projected_value if not would_breach else 0
                    }
            
            return projections
            
        except Exception as e:
            logger.error(f"Trade impact projection error: {str(e)}")
            return {"error": str(e)}
    
    def close(self):
        """Cleanup resources."""
        self.db.close()