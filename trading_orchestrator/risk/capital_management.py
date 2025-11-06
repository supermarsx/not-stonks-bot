"""
Capital Preservation Safeguards and Risk Controls

Provides comprehensive capital protection mechanisms:
- Available capital validation
- Position size limits based on available cash
- Dynamic capital allocation
- Cash flow monitoring
- Free margin calculations
- Risk-based position sizing
- Capital efficiency optimization

@author Trading Orchestrator System
@version 1.0
@date 2025-11-06
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import pandas as pd
import numpy as np

from database.models.risk import RiskLevel
from database.models.trading import Position, Order, Trade, Account
from database.models.user import User
from config.database import get_db

logger = logging.getLogger(__name__)


class CapitalAllocationStrategy(Enum):
    """Capital allocation strategies."""
    CONSERVATIVE = "conservative"  # Max 5% risk per position
    MODERATE = "moderate"          # Max 10% risk per position
    AGGRESSIVE = "aggressive"      # Max 20% risk per position
    ADAPTIVE = "adaptive"          # Adjusts based on market conditions


class CapitalAlertLevel(Enum):
    """Capital alert levels."""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class CapitalMetrics:
    """Capital preservation metrics."""
    total_capital: float = 0.0
    available_cash: float = 0.0
    margin_used: float = 0.0
    free_margin: float = 0.0
    equity: float = 0.0
    pnl_unrealized: float = 0.0
    pnl_realized: float = 0.0
    maintenance_margin: float = 0.0
    margin_call_level: float = 0.0
    capital_efficiency: float = 0.0
    risk_utilization: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PositionSizingConfig:
    """Position sizing configuration."""
    max_position_percent: float = 0.10  # 10% of capital
    max_sector_percent: float = 0.25    # 25% per sector
    max_single_trade_percent: float = 0.05  # 5% per trade
    volatility_adjustment: bool = True
    correlation_adjustment: bool = True
    kelly_criterion: bool = False
    risk_parity: bool = False


class CapitalManager:
    """
    Capital Preservation and Management System
    
    Monitors and controls capital utilization to ensure preservation
    while maximizing trading opportunities within risk parameters.
    """
    
    def __init__(self, user_id: int):
        """
        Initialize Capital Manager.
        
        Args:
            user_id: User identifier for capital tracking
        """
        self.user_id = user_id
        self.db = get_db()
        
        # Capital allocation settings
        self.allocation_strategy = CapitalAllocationStrategy.MODERATE
        self.sizing_config = PositionSizingConfig()
        
        # Monitoring settings
        self.alert_levels = {
            CapitalAlertLevel.WARNING: 0.15,    # 15% capital usage
            CapitalAlertLevel.CRITICAL: 0.25,   # 25% capital usage
            CapitalAlertLevel.EMERGENCY: 0.35   # 35% capital usage
        }
        
        # Current state
        self.current_metrics = CapitalMetrics()
        self.last_calculated = None
        self.capital_history = []
        
        logger.info(f"CapitalManager initialized for user {self.user_id}")
    
    async def calculate_available_capital(self, include_margin: bool = True) -> Dict[str, Any]:
        """
        Calculate available capital for trading.
        
        Args:
            include_margin: Whether to include margin in calculations
            
        Returns:
            Capital availability breakdown
        """
        try:
            # Get account information
            account = self.db.query(Account).filter(
                Account.user_id == self.user_id
            ).first()
            
            if not account:
                return {"error": "Account not found"}
            
            # Base calculations
            total_capital = account.cash_balance
            margin_used = account.margin_used
            
            # Calculate various capital measures
            available_cash = total_capital - margin_used
            free_margin = total_capital - (margin_used * 1.5)  # Conservative buffer
            
            # Equity calculation (simplified)
            equity = total_capital + await self._calculate_unrealized_pnl()
            
            # Calculate utilization
            utilization = (margin_used / total_capital) if total_capital > 0 else 0
            capital_efficiency = available_cash / total_capital if total_capital > 0 else 0
            
            # Update current metrics
            self.current_metrics = CapitalMetrics(
                total_capital=total_capital,
                available_cash=available_cash,
                margin_used=margin_used,
                free_margin=free_margin,
                equity=equity,
                pnl_unrealized=await self._calculate_unrealized_pnl(),
                pnl_realized=await self._calculate_realized_pnl(),
                maintenance_margin=margin_used * 1.25,
                margin_call_level=margin_used * 1.4,
                capital_efficiency=capital_efficiency,
                risk_utilization=utilization
            )
            
            return {
                "available_for_trading": available_cash if include_margin else free_margin,
                "total_capital": total_capital,
                "margin_used": margin_used,
                "free_margin": free_margin,
                "equity": equity,
                "unrealized_pnl": self.current_metrics.pnl_unrealized,
                "capital_utilization": utilization,
                "capital_efficiency": capital_efficiency,
                "margin_call_level": margin_used * 1.4,
                "maintenance_margin": margin_used * 1.25,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Available capital calculation error: {str(e)}")
            return {"error": str(e)}
    
    async def validate_position_size(self, symbol: str, quantity: float, 
                                   price: float, order_type: str = "market") -> Dict[str, Any]:
        """
        Validate if position size is within capital limits.
        
        Args:
            symbol: Trading symbol
            quantity: Number of shares/units
            price: Current or limit price
            order_type: Order type
            
        Returns:
            Validation result with position size recommendations
        """
        try:
            # Get current capital metrics
            capital_metrics = await self.calculate_available_capital()
            if "error" in capital_metrics:
                return {"approved": False, "reason": capital_metrics["error"]}
            
            # Calculate position value
            position_value = abs(quantity * price)
            
            # Check against available capital
            available_capital = capital_metrics["available_for_trading"]
            
            # Basic validation
            validation_result = {
                "approved": False,
                "recommended_quantity": 0,
                "capital_usage": 0,
                "remaining_capacity": 0,
                "warnings": [],
                "reasons": []
            }
            
            # Check if position exceeds available capital
            if position_value > available_capital:
                # Calculate maximum affordable quantity
                max_quantity = int(available_capital / price)
                if max_quantity <= 0:
                    validation_result["reasons"].append(
                        f"Insufficient capital: Need ${position_value:.2f}, have ${available_capital:.2f}"
                    )
                    return validation_result
                else:
                    validation_result["recommended_quantity"] = max_quantity
                    validation_result["warnings"].append(
                        f"Position reduced to ${max_quantity * price:.2f} due to capital constraints"
                    )
            
            # Check single trade limit
            single_trade_limit = self.sizing_config.max_single_trade_percent * capital_metrics["total_capital"]
            if position_value > single_trade_limit:
                max_single_quantity = int(single_trade_limit / price)
                validation_result["recommended_quantity"] = min(
                    validation_result["recommended_quantity"] or int(available_capital / price),
                    max_single_quantity
                )
                validation_result["warnings"].append(
                    f"Single trade limit exceeded: ${position_value:.2f} > ${single_trade_limit:.2f}"
                )
            
            # Calculate capital usage
            recommended_value = (validation_result["recommended_quantity"] or quantity) * price
            capital_usage = recommended_value / capital_metrics["total_capital"]
            
            validation_result.update({
                "approved": True,
                "capital_usage": capital_usage,
                "remaining_capacity": capital_metrics["available_for_trading"] - recommended_value,
                "position_value": recommended_value,
                "quantity_approved": validation_result["recommended_quantity"] or quantity,
                "warnings": validation_result["warnings"]
            })
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Position size validation error: {str(e)}")
            return {"approved": False, "reason": f"Validation error: {str(e)}"}
    
    async def calculate_dynamic_position_size(self, symbol: str, risk_percent: float = 0.02,
                                            stop_loss_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculate optimal position size based on risk management principles.
        
        Args:
            symbol: Trading symbol
            risk_percent: Risk percentage per trade (default 2%)
            stop_loss_price: Optional stop loss price for risk calculation
            
        Returns:
            Recommended position size with risk metrics
        """
        try:
            # Get capital metrics
            capital_metrics = await self.calculate_available_capital()
            if "error" in capital_metrics:
                return {"error": capital_metrics["error"]}
            
            # Get current market price (simplified - would need market data)
            current_price = await self._get_market_price(symbol)
            if not current_price:
                return {"error": f"Unable to get market price for {symbol}"}
            
            # Calculate risk amount in dollars
            risk_amount = capital_metrics["total_capital"] * risk_percent
            
            # Calculate position size based on risk
            if stop_loss_price:
                # Risk-based sizing using stop loss
                price_risk = abs(current_price - stop_loss_price)
                if price_risk > 0:
                    position_size = int(risk_amount / price_risk)
                else:
                    position_size = int(risk_amount / current_price)  # Fallback
            else:
                # Fixed percentage sizing
                position_size = int((capital_metrics["total_capital"] * self.sizing_config.max_position_percent) / current_price)
            
            # Apply additional constraints
            max_position_value = capital_metrics["total_capital"] * self.sizing_config.max_position_percent
            max_position_size = int(max_position_value / current_price)
            
            # Use the minimum of calculated sizes
            final_position_size = min(position_size, max_position_size)
            final_position_value = final_position_size * current_price
            
            # Calculate risk metrics
            actual_risk_percent = final_position_value / capital_metrics["total_capital"]
            
            # Apply volatility adjustment if enabled
            if self.sizing_config.volatility_adjustment:
                volatility = await self._calculate_symbol_volatility(symbol)
                if volatility > 0:
                    volatility_adjustment = min(1.0, 0.5 / volatility)  # Reduce size for high volatility
                    final_position_size = int(final_position_size * volatility_adjustment)
                    final_position_value = final_position_size * current_price
            
            # Apply correlation adjustment if enabled
            if self.sizing_config.correlation_adjustment:
                correlation_risk = await self._calculate_correlation_risk(symbol)
                if correlation_risk > 0.7:  # High correlation
                    correlation_adjustment = 0.7
                    final_position_size = int(final_position_size * correlation_adjustment)
                    final_position_value = final_position_size * current_price
            
            return {
                "recommended_quantity": final_position_size,
                "recommended_value": final_position_value,
                "risk_amount": final_position_value,
                "risk_percent": actual_risk_percent,
                "capital_efficiency": final_position_value / capital_metrics["total_capital"],
                "stop_loss_price": stop_loss_price,
                "current_price": current_price,
                "risk_per_share": abs(current_price - stop_loss_price) if stop_loss_price else current_price * 0.02,
                "adjustments": {
                    "volatility": self.sizing_config.volatility_adjustment,
                    "correlation": self.sizing_config.correlation_adjustment
                },
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Dynamic position size calculation error: {str(e)}")
            return {"error": f"Position sizing failed: {str(e)}"}
    
    async def allocate_capital_by_strategy(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Allocate capital across multiple trades based on allocation strategy.
        
        Args:
            trades: List of potential trades with symbol, quantity, price
            
        Returns:
            Capital allocation results
        """
        try:
            # Get current capital metrics
            capital_metrics = await self.calculate_available_capital()
            if "error" in capital_metrics:
                return {"error": capital_metrics["error"]}
            
            available_capital = capital_metrics["available_for_trading"]
            total_capital = capital_metrics["total_capital"]
            
            # Filter out trades that don't meet minimum criteria
            valid_trades = []
            for trade in trades:
                trade_value = trade["quantity"] * trade["price"]
                if trade_value < total_capital * 0.001:  # Minimum 0.1% of capital
                    continue
                valid_trades.append(trade)
            
            if not valid_trades:
                return {"allocated_trades": [], "remaining_capital": available_capital}
            
            # Apply allocation strategy
            if self.allocation_strategy == CapitalAllocationStrategy.CONSERVATIVE:
                allocation_result = await self._conservative_allocation(valid_trades, available_capital)
            elif self.allocation_strategy == CapitalAllocationStrategy.MODERATE:
                allocation_result = await self._moderate_allocation(valid_trades, available_capital)
            elif self.allocation_strategy == CapitalAllocationStrategy.AGGRESSIVE:
                allocation_result = await self._aggressive_allocation(valid_trades, available_capital)
            else:  # ADAPTIVE
                allocation_result = await self._adaptive_allocation(valid_trades, available_capital, capital_metrics)
            
            # Calculate remaining capital
            allocated_value = sum(trade["allocated_value"] for trade in allocation_result["allocated_trades"])
            remaining_capital = available_capital - allocated_value
            
            return {
                "allocation_strategy": self.allocation_strategy.value,
                "available_capital": available_capital,
                "allocated_capital": allocated_value,
                "remaining_capital": remaining_capital,
                "capital_efficiency": allocated_value / available_capital if available_capital > 0 else 0,
                "allocated_trades": allocation_result["allocated_trades"],
                "rejected_trades": allocation_result.get("rejected_trades", []),
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Capital allocation error: {str(e)}")
            return {"error": f"Allocation failed: {str(e)}"}
    
    async def monitor_capital_alerts(self) -> List[Dict[str, Any]]:
        """
        Monitor capital levels and generate alerts.
        
        Returns:
            List of capital alerts
        """
        alerts = []
        
        try:
            # Get current metrics
            await self.calculate_available_capital()
            metrics = self.current_metrics
            
            # Check utilization levels
            utilization = metrics.risk_utilization
            
            for level, threshold in self.alert_levels.items():
                if utilization >= threshold:
                    alert = {
                        "level": level.value,
                        "type": "capital_utilization",
                        "current_value": utilization,
                        "threshold": threshold,
                        "message": f"Capital utilization at {utilization:.1%}, threshold: {threshold:.1%}",
                        "recommended_action": self._get_recommended_action(level),
                        "timestamp": datetime.now()
                    }
                    alerts.append(alert)
            
            # Check for margin call risk
            if metrics.margin_used >= metrics.margin_call_level * 0.9:
                alert = {
                    "level": CapitalAlertLevel.EMERGENCY.value,
                    "type": "margin_call_risk",
                    "current_value": metrics.margin_used,
                    "margin_call_level": metrics.margin_call_level,
                    "message": f"Margin usage approaching call level: ${metrics.margin_used:.2f}/${metrics.margin_call_level:.2f}",
                    "recommended_action": "Immediate position reduction required",
                    "timestamp": datetime.now()
                }
                alerts.append(alert)
            
            # Check for low free margin
            if metrics.free_margin < metrics.total_capital * 0.05:  # Less than 5% free
                alert = {
                    "level": CapitalAlertLevel.CRITICAL.value,
                    "type": "low_free_margin",
                    "current_value": metrics.free_margin,
                    "threshold": metrics.total_capital * 0.05,
                    "message": f"Low free margin: ${metrics.free_margin:.2f}",
                    "recommended_action": "Consider reducing positions or adding capital",
                    "timestamp": datetime.now()
                }
                alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Capital alert monitoring error: {str(e)}")
            return [{"error": str(e)}]
    
    async def optimize_capital_efficiency(self) -> Dict[str, Any]:
        """
        Analyze and provide recommendations for capital efficiency improvements.
        
        Returns:
            Capital efficiency analysis and recommendations
        """
        try:
            # Get current metrics
            await self.calculate_available_capital()
            metrics = self.current_metrics
            
            # Get all current positions
            positions = self.db.query(Position).filter(
                and_(
                    Position.user_id == self.user_id,
                    Position.quantity != 0
                )
            ).all()
            
            # Calculate efficiency metrics
            total_exposure = sum(abs(pos.market_value or 0) for pos in positions)
            capital_efficiency = total_exposure / metrics.total_capital if metrics.total_capital > 0 else 0
            
            # Analyze position concentration
            position_analysis = []
            for pos in positions:
                position_value = abs(pos.market_value or 0)
                concentration = position_value / total_exposure if total_exposure > 0 else 0
                
                position_analysis.append({
                    "symbol": pos.symbol,
                    "quantity": pos.quantity,
                    "value": position_value,
                    "concentration": concentration,
                    "efficiency_rating": self._calculate_position_efficiency(pos)
                })
            
            # Sort by efficiency (least efficient first)
            position_analysis.sort(key=lambda x: x["efficiency_rating"])
            
            # Generate recommendations
            recommendations = []
            
            # Low efficiency positions
            inefficient_positions = [p for p in position_analysis if p["efficiency_rating"] < 0.7]
            if inefficient_positions:
                recommendations.append({
                    "type": "position_optimization",
                    "priority": "high",
                    "action": "Reduce or close inefficient positions",
                    "affected_symbols": [p["symbol"] for p in inefficient_positions[:3]],
                    "potential_capital_savings": sum(p["value"] for p in inefficient_positions) * 0.3
                })
            
            # Over-concentration
            concentrated_positions = [p for p in position_analysis if p["concentration"] > 0.25]
            if concentrated_positions:
                recommendations.append({
                    "type": "concentration_risk",
                    "priority": "medium",
                    "action": "Reduce concentration in over-weight positions",
                    "affected_symbols": [p["symbol"] for p in concentrated_positions],
                    "recommended_max_concentration": 0.20
                })
            
            # Low capital utilization
            if capital_efficiency < 0.6:
                recommendations.append({
                    "type": "capital_underutilization",
                    "priority": "medium",
                    "action": "Consider increasing position sizes within risk limits",
                    "current_efficiency": capital_efficiency,
                    "target_efficiency": 0.80
                })
            
            return {
                "current_efficiency": capital_efficiency,
                "total_exposure": total_exposure,
                "total_capital": metrics.total_capital,
                "position_count": len(positions),
                "position_analysis": position_analysis,
                "recommendations": recommendations,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Capital efficiency optimization error: {str(e)}")
            return {"error": f"Optimization analysis failed: {str(e)}"}
    
    def set_allocation_strategy(self, strategy: CapitalAllocationStrategy):
        """Set capital allocation strategy."""
        self.allocation_strategy = strategy
        logger.info(f"Capital allocation strategy set to {strategy.value} for user {self.user_id}")
    
    def set_position_sizing_config(self, config: PositionSizingConfig):
        """Update position sizing configuration."""
        self.sizing_config = config
        logger.info(f"Position sizing configuration updated for user {self.user_id}")
    
    async def _calculate_unrealized_pnl(self) -> float:
        """Calculate unrealized P&L from positions."""
        try:
            positions = self.db.query(Position).filter(
                and_(
                    Position.user_id == self.user_id,
                    Position.quantity != 0
                )
            ).all()
            
            return sum(pos.unrealized_pnl or 0 for pos in positions)
            
        except Exception as e:
            logger.error(f"Unrealized P&L calculation error: {str(e)}")
            return 0.0
    
    async def _calculate_realized_pnl(self) -> float:
        """Calculate realized P&L from trades."""
        try:
            trades = self.db.query(Trade).filter(
                Trade.user_id == self.user_id
            ).all()
            
            return sum(trade.realized_pnl or 0 for trade in trades)
            
        except Exception as e:
            logger.error(f"Realized P&L calculation error: {str(e)}")
            return 0.0
    
    async def _get_market_price(self, symbol: str) -> Optional[float]:
        """Get current market price for symbol."""
        # Simplified implementation - would integrate with market data provider
        try:
            position = self.db.query(Position).filter(
                and_(
                    Position.user_id == self.user_id,
                    Position.symbol == symbol
                )
            ).first()
            
            if position and position.avg_cost:
                return float(position.avg_cost)
            
            return None
            
        except Exception as e:
            logger.error(f"Market price retrieval error: {str(e)}")
            return None
    
    async def _calculate_symbol_volatility(self, symbol: str) -> float:
        """Calculate symbol volatility for position sizing adjustments."""
        # Simplified volatility calculation
        # Would use historical price data in practice
        return 0.15  # 15% assumed volatility
    
    async def _calculate_correlation_risk(self, symbol: str) -> float:
        """Calculate correlation risk with existing positions."""
        # Simplified correlation calculation
        # Would analyze correlation matrix in practice
        return 0.3  # 30% assumed correlation
    
    def _calculate_position_efficiency(self, position: Position) -> float:
        """Calculate position efficiency rating."""
        try:
            if not position.market_value or position.market_value == 0:
                return 0.0
            
            # Basic efficiency metrics
            unrealized_pnl = abs(position.unrealized_pnl or 0)
            market_value = abs(position.market_value)
            
            # P&L efficiency (higher unrealized gains = higher efficiency)
            pnl_efficiency = min(1.0, unrealized_pnl / market_value * 10) if market_value > 0 else 0
            
            # Position size efficiency (reasonable size)
            size_efficiency = 1.0 if market_value > 1000 and market_value < 50000 else 0.7
            
            return (pnl_efficiency + size_efficiency) / 2
            
        except Exception as e:
            logger.error(f"Position efficiency calculation error: {str(e)}")
            return 0.5
    
    def _get_recommended_action(self, alert_level: CapitalAlertLevel) -> str:
        """Get recommended action for alert level."""
        actions = {
            CapitalAlertLevel.WARNING: "Monitor positions closely",
            CapitalAlertLevel.CRITICAL: "Reduce position sizes",
            CapitalAlertLevel.EMERGENCY: "Immediate action required"
        }
        return actions.get(alert_level, "Monitor situation")
    
    async def _conservative_allocation(self, trades: List[Dict], available_capital: float) -> Dict[str, Any]:
        """Conservative capital allocation."""
        # Allocate smaller amounts to more trades
        max_position_percent = 0.05  # 5% max per position
        
        allocated_trades = []
        remaining_capital = available_capital
        
        for trade in trades:
            max_allocation = remaining_capital * max_position_percent
            trade_value = min(trade["quantity"] * trade["price"], max_allocation)
            
            if trade_value > 0:
                allocated_quantity = int(trade_value / trade["price"])
                allocated_trades.append({
                    **trade,
                    "allocated_quantity": allocated_quantity,
                    "allocated_value": allocated_quantity * trade["price"]
                })
                remaining_capital -= allocated_quantity * trade["price"]
        
        return {"allocated_trades": allocated_trades}
    
    async def _moderate_allocation(self, trades: List[Dict], available_capital: float) -> Dict[str, Any]:
        """Moderate capital allocation."""
        # Equal weight allocation with limits
        max_position_percent = 0.10  # 10% max per position
        max_positions = min(len(trades), 8)  # Max 8 positions
        
        allocated_trades = []
        
        # Calculate equal weights
        equal_weight = available_capital / max_positions * max_position_percent
        
        for trade in trades[:max_positions]:
            trade_value = min(trade["quantity"] * trade["price"], equal_weight)
            allocated_quantity = int(trade_value / trade["price"])
            
            if allocated_quantity > 0:
                allocated_trades.append({
                    **trade,
                    "allocated_quantity": allocated_quantity,
                    "allocated_value": allocated_quantity * trade["price"]
                })
        
        return {"allocated_trades": allocated_trades}
    
    async def _aggressive_allocation(self, trades: List[Dict], available_capital: float) -> Dict[str, Any]:
        """Aggressive capital allocation."""
        # Focus on fewer, larger positions
        max_position_percent = 0.20  # 20% max per position
        max_positions = min(len(trades), 5)  # Max 5 positions
        
        allocated_trades = []
        
        for trade in trades[:max_positions]:
            max_allocation = available_capital * max_position_percent
            trade_value = min(trade["quantity"] * trade["price"], max_allocation)
            allocated_quantity = int(trade_value / trade["price"])
            
            if allocated_quantity > 0:
                allocated_trades.append({
                    **trade,
                    "allocated_quantity": allocated_quantity,
                    "allocated_value": allocated_quantity * trade["price"]
                })
        
        return {"allocated_trades": allocated_trades}
    
    async def _adaptive_allocation(self, trades: List[Dict], available_capital: float, 
                                 metrics: CapitalMetrics) -> Dict[str, Any]:
        """Adaptive capital allocation based on market conditions."""
        # Adjust allocation based on current utilization and market volatility
        
        base_allocation = available_capital * 0.10  # Base 10% per position
        
        # Adjust based on current utilization
        utilization_adjustment = 1.0 - min(metrics.risk_utilization, 0.8)
        
        # Adjust based on market conditions (simplified)
        volatility_adjustment = 0.8  # Assume moderate volatility
        
        adjusted_allocation = base_allocation * utilization_adjustment * volatility_adjustment
        
        allocated_trades = []
        
        for trade in trades:
            trade_value = min(trade["quantity"] * trade["price"], adjusted_allocation)
            allocated_quantity = int(trade_value / trade["price"])
            
            if allocated_quantity > 0:
                allocated_trades.append({
                    **trade,
                    "allocated_quantity": allocated_quantity,
                    "allocated_value": allocated_quantity * trade["price"]
                })
        
        return {"allocated_trades": allocated_trades}
    
    def close(self):
        """Cleanup resources."""
        self.db.close()