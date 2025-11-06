"""
Risk Management Integration Layer

Connects the advanced risk management system with:
- Existing broker systems (Interactive Brokers, TD Ameritrade, etc.)
- Order Management System (OMS)
- Automated Trading Engine
- Real-time market data feeds
- Portfolio management systems

Provides seamless integration between risk management and trading operations
for institutional-grade automated trading workflows.
"""

import logging
import asyncio
import threading
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import queue
import uuid

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BrokerType(Enum):
    """Supported broker types"""
    INTERACTIVE_BROKERS = "interactive_brokers"
    TD_AMERITRADE = "td_ameritrade"
    CHARLES_SCHWAB = "charles_schwab"
    E_TRADE = "e_trade"
    ALLY = "ally"
    TRADE_STATION = "trade_station"


class OrderStatus(Enum):
    """Order execution status"""
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"


class RiskAction(Enum):
    """Risk management actions"""
    ALLOW = "allow"
    REJECT = "reject"
    REDUCE_SIZE = "reduce_size"
    HEDGE = "hedge"
    CLOSE_POSITION = "close_position"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class Position:
    """Position data structure"""
    symbol: str
    quantity: float
    avg_cost: float
    market_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    position_type: str  # long, short
    sector: Optional[str] = None
    industry: Optional[str] = None
    currency: str = "USD"
    broker: str = ""


@dataclass
class Order:
    """Order data structure"""
    order_id: str
    symbol: str
    side: str  # buy, sell
    quantity: float
    order_type: str  # market, limit, stop, stop_limit
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"
    timestamp: datetime = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: Optional[float] = None
    broker: str = ""
    strategy_id: Optional[str] = None
    risk_checks: Dict = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.risk_checks is None:
            self.risk_checks = {}


@dataclass
class RiskCheckResult:
    """Risk check result"""
    order_id: str
    allowed: bool
    action: RiskAction
    risk_score: float
    violations: List[str]
    recommendations: List[str]
    limit_utilization: Dict[str, float]
    timestamp: datetime


@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float
    volume: float
    open: float
    high: float
    low: float
    close: float
    change: float
    change_percent: float


@dataclass
class PortfolioSnapshot:
    """Portfolio snapshot for risk analysis"""
    timestamp: datetime
    total_value: float
    cash_balance: float
    positions: List[Position]
    pending_orders: List[Order]
    unrealized_pnl: float
    realized_pnl: float
    day_pnl: float
    portfolio_var: float
    max_drawdown: float
    leverage: float


class RiskBrokerIntegration:
    """
    Risk management integration with broker systems
    
    Provides real-time position monitoring, risk checking,
    and order management across multiple broker accounts.
    """
    
    def __init__(self):
        """Initialize risk broker integration"""
        self.logger = logging.getLogger(__name__)
        
        # Broker connections
        self.broker_connections: Dict[BrokerType, Any] = {}
        self.broker_positions: Dict[BrokerType, List[Position]] = {}
        self.broker_accounts: Dict[str, Dict] = {}
        
        # Order management
        self.pending_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.order_queue = asyncio.Queue()
        
        # Market data
        self.market_data: Dict[str, MarketData] = {}
        self.market_data_subscribers: Dict[str, List[Callable]] = {}
        
        # Risk monitoring
        self.active_monitoring = False
        self.monitoring_thread = None
        self.risk_alerts: List[Dict] = []
        
        # Risk limits cache
        self.current_limits: Dict[str, float] = {}
        self.limit_breaches: List[Dict] = []
        
        # Background tasks
        self.background_tasks = []
        
        # Initialize broker connections
        self._initialize_broker_connections()
        
        self.logger.info("Risk Broker Integration initialized")
    
    def _initialize_broker_connections(self):
        """Initialize broker connections"""
        # In a real implementation, this would establish actual broker connections
        # For demonstration, we'll simulate broker connections
        
        mock_brokers = [
            BrokerType.INTERACTIVE_BROKERS,
            BrokerType.TD_AMERITRADE,
            BrokerType.CHARLES_SCHWAB
        ]
        
        for broker in mock_brokers:
            self.broker_connections[broker] = {
                "connected": True,
                "last_heartbeat": datetime.now(),
                "latency_ms": np.random.uniform(20, 100),
                "account_id": f"ACC_{broker.value.upper()}_{secrets.token_hex(6)}",
                "api_version": "v1"
            }
        
        self.logger.info(f"Initialized {len(mock_brokers)} broker connections")
    
    async def start_monitoring(self):
        """Start real-time risk monitoring"""
        if self.active_monitoring:
            return
        
        self.active_monitoring = True
        
        # Start monitoring tasks
        self.background_tasks = [
            asyncio.create_task(self._position_monitoring_loop()),
            asyncio.create_task(self._risk_limit_monitoring_loop()),
            asyncio.create_task(self._order_monitoring_loop()),
            asyncio.create_task(self._market_data_monitoring_loop())
        ]
        
        self.logger.info("Risk monitoring started")
    
    async def stop_monitoring(self):
        """Stop real-time risk monitoring"""
        self.active_monitoring = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        self.logger.info("Risk monitoring stopped")
    
    async def _position_monitoring_loop(self):
        """Continuous position monitoring"""
        while self.active_monitoring:
            try:
                # Update positions from all brokers
                await self._update_positions()
                
                # Calculate portfolio metrics
                portfolio_metrics = await self._calculate_portfolio_metrics()
                
                # Check risk limits
                await self._check_risk_limits(portfolio_metrics)
                
                await asyncio.sleep(5)  # 5-second intervals
                
            except Exception as e:
                self.logger.error(f"Error in position monitoring: {e}")
                await asyncio.sleep(5)
    
    async def _risk_limit_monitoring_loop(self):
        """Continuous risk limit monitoring"""
        while self.active_monitoring:
            try:
                # Check each risk limit
                for limit_name, limit_value in self.current_limits.items():
                    current_value = await self._get_current_limit_value(limit_name)
                    
                    if current_value is not None:
                        utilization = current_value / limit_value if limit_value > 0 else 0
                        
                        # Check for limit breaches
                        if utilization > 1.0:
                            await self._handle_limit_breach(limit_name, current_value, limit_value)
                        elif utilization > 0.8:  # 80% warning threshold
                            await self._handle_limit_warning(limit_name, current_value, limit_value)
                
                await asyncio.sleep(10)  # 10-second intervals
                
            except Exception as e:
                self.logger.error(f"Error in risk limit monitoring: {e}")
                await asyncio.sleep(10)
    
    async def _order_monitoring_loop(self):
        """Continuous order monitoring"""
        while self.active_monitoring:
            try:
                # Process pending orders
                while not self.order_queue.empty():
                    try:
                        order = await asyncio.wait_for(self.order_queue.get(), timeout=1.0)
                        await self._process_order(order)
                    except asyncio.TimeoutError:
                        break
                
                await asyncio.sleep(1)  # 1-second intervals
                
            except Exception as e:
                self.logger.error(f"Error in order monitoring: {e}")
                await asyncio.sleep(5)
    
    async def _market_data_monitoring_loop(self):
        """Continuous market data updates"""
        while self.active_monitoring:
            try:
                # Update market data for subscribed symbols
                await self._update_market_data()
                
                await asyncio.sleep(1)  # 1-second intervals
                
            except Exception as e:
                self.logger.error(f"Error in market data monitoring: {e}")
                await asyncio.sleep(5)
    
    async def _update_positions(self):
        """Update positions from all brokers"""
        for broker_type in self.broker_connections.keys():
            try:
                # Simulate position updates
                positions = await self._get_broker_positions(broker_type)
                self.broker_positions[broker_type] = positions
                
            except Exception as e:
                self.logger.error(f"Error updating positions from {broker_type}: {e}")
    
    async def _get_broker_positions(self, broker_type: BrokerType) -> List[Position]:
        """Get positions from specific broker"""
        # Mock positions for demonstration
        mock_positions = [
            Position(
                symbol="AAPL",
                quantity=100,
                avg_cost=150.0,
                market_price=155.0,
                market_value=15500.0,
                unrealized_pnl=500.0,
                realized_pnl=0.0,
                position_type="long",
                sector="Technology",
                broker=broker_type.value
            ),
            Position(
                symbol="GOOGL",
                quantity=50,
                avg_cost=2800.0,
                market_price=2850.0,
                market_value=142500.0,
                unrealized_pnl=2500.0,
                realized_pnl=0.0,
                position_type="long",
                sector="Technology",
                broker=broker_type.value
            )
        ]
        
        return mock_positions
    
    async def _calculate_portfolio_metrics(self) -> Dict[str, float]:
        """Calculate portfolio-level risk metrics"""
        try:
            all_positions = []
            for broker_positions in self.broker_positions.values():
                all_positions.extend(broker_positions)
            
            total_value = sum(pos.market_value for pos in all_positions)
            total_pnl = sum(pos.unrealized_pnl for pos in all_positions)
            
            # Calculate portfolio metrics
            metrics = {
                "total_value": total_value,
                "total_unrealized_pnl": total_pnl,
                "position_count": len(all_positions),
                "concentration_risk": self._calculate_concentration_risk(all_positions),
                "sector_diversity": self._calculate_sector_diversity(all_positions),
                "leverage_ratio": self._calculate_leverage_ratio(all_positions)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {e}")
            return {}
    
    def _calculate_concentration_risk(self, positions: List[Position]) -> float:
        """Calculate position concentration risk"""
        if not positions:
            return 0.0
        
        total_value = sum(pos.market_value for pos in positions)
        if total_value == 0:
            return 0.0
        
        # Calculate Herfindahl index
        herfindahl = sum((pos.market_value / total_value) ** 2 for pos in positions)
        return herfindahl
    
    def _calculate_sector_diversity(self, positions: List[Position]) -> float:
        """Calculate sector diversity score"""
        if not positions:
            return 0.0
        
        sector_weights = defaultdict(float)
        total_value = sum(pos.market_value for pos in positions)
        
        if total_value == 0:
            return 0.0
        
        for pos in positions:
            if pos.sector:
                sector_weights[pos.sector] += pos.market_value / total_value
        
        # Calculate sector diversity (inverse of concentration)
        if len(sector_weights) <= 1:
            return 0.0
        
        herfindahl = sum(weight ** 2 for weight in sector_weights.values())
        diversity = 1 - herfindahl
        return diversity
    
    def _calculate_leverage_ratio(self, positions: List[Position]) -> float:
        """Calculate leverage ratio"""
        # Mock calculation - would use actual capital and position values
        total_value = sum(pos.market_value for pos in positions)
        # Assume 50% margin requirement
        margin_required = total_value * 0.5
        account_value = margin_required * 2  # Mock account value
        return total_value / account_value if account_value > 0 else 1.0
    
    async def _check_risk_limits(self, portfolio_metrics: Dict[str, float]):
        """Check portfolio against risk limits"""
        try:
            limit_checks = {
                "max_portfolio_var": self._check_var_limit(portfolio_metrics),
                "max_concentration": self._check_concentration_limit(portfolio_metrics),
                "max_leverage": self._check_leverage_limit(portfolio_metrics),
                "max_position_size": self._check_position_size_limit()
            }
            
            # Update current limit utilization
            for limit_name, utilization in limit_checks.items():
                if utilization is not None:
                    self.current_limits[limit_name] = utilization
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
    
    def _check_var_limit(self, metrics: Dict[str, float]) -> Optional[float]:
        """Check VaR limit (mock implementation)"""
        # Mock VaR calculation
        portfolio_var = np.random.uniform(0.05, 0.15) * metrics.get("total_value", 100000)
        return portfolio_var / 100000  # Normalized utilization
    
    def _check_concentration_limit(self, metrics: Dict[str, float]) -> Optional[float]:
        """Check concentration limit"""
        return metrics.get("concentration_risk", 0.0)
    
    def _check_leverage_limit(self, metrics: Dict[str, float]) -> Optional[float]:
        """Check leverage limit"""
        return metrics.get("leverage_ratio", 1.0)
    
    def _check_position_size_limit(self) -> Optional[float]:
        """Check position size limit"""
        return 0.3  # Mock position size utilization
    
    async def _get_current_limit_value(self, limit_name: str) -> Optional[float]:
        """Get current value for a specific limit"""
        return self.current_limits.get(limit_name, 0.0)
    
    async def _handle_limit_breach(self, limit_name: str, current_value: float, limit_value: float):
        """Handle limit breach"""
        breach = {
            "timestamp": datetime.now(),
            "limit_name": limit_name,
            "current_value": current_value,
            "limit_value": limit_value,
            "breach_severity": "critical" if current_value > 1.5 * limit_value else "warning",
            "action_required": "immediate"
        }
        
        self.limit_breaches.append(breach)
        self.risk_alerts.append({
            "type": "limit_breach",
            "severity": "high",
            "message": f"Risk limit breach: {limit_name} = {current_value:.3f}, limit = {limit_value:.3f}",
            "timestamp": datetime.now(),
            "breach_data": breach
        })
        
        self.logger.critical(f"LIMIT BREACH: {limit_name} = {current_value:.3f}, limit = {limit_value:.3f}")
    
    async def _handle_limit_warning(self, limit_name: str, current_value: float, limit_value: float):
        """Handle limit warning"""
        self.risk_alerts.append({
            "type": "limit_warning",
            "severity": "medium",
            "message": f"Risk limit approaching: {limit_name} = {current_value:.3f}, limit = {limit_value:.3f}",
            "timestamp": datetime.now()
        })
        
        self.logger.warning(f"LIMIT WARNING: {limit_name} = {current_value:.3f}, limit = {limit_value:.3f}")
    
    async def check_order_risk(self, order: Order) -> RiskCheckResult:
        """Perform comprehensive risk check on order"""
        try:
            violations = []
            recommendations = []
            risk_score = 0.0
            
            # Check position limits
            position_violations = await self._check_position_limits(order)
            violations.extend(position_violations)
            
            # Check portfolio limits
            portfolio_violations = await self._check_portfolio_limits(order)
            violations.extend(portfolio_violations)
            
            # Check regulatory limits
            regulatory_violations = await self._check_regulatory_limits(order)
            violations.extend(regulatory_violations)
            
            # Calculate risk score
            risk_score = len(violations) * 0.2 + (1.0 if violations else 0.0)
            
            # Determine action
            if violations:
                action = RiskAction.REJECT if len(violations) > 2 else RiskAction.REDUCE_SIZE
            else:
                action = RiskAction.ALLOW
            
            # Generate recommendations
            recommendations = await self._generate_risk_recommendations(order, violations)
            
            result = RiskCheckResult(
                order_id=order.order_id,
                allowed=action == RiskAction.ALLOW,
                action=action,
                risk_score=risk_score,
                violations=violations,
                recommendations=recommendations,
                limit_utilization=self.current_limits,
                timestamp=datetime.now()
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error checking order risk: {e}")
            return RiskCheckResult(
                order_id=order.order_id,
                allowed=False,
                action=RiskAction.REJECT,
                risk_score=1.0,
                violations=[f"Risk check error: {str(e)}"],
                recommendations=["Manual review required"],
                limit_utilization={},
                timestamp=datetime.now()
            )
    
    async def _check_position_limits(self, order: Order) -> List[str]:
        """Check position-level limits"""
        violations = []
        
        # Get current positions
        current_positions = await self._get_symbol_positions(order.symbol)
        
        # Calculate new position size
        new_quantity = current_positions.quantity + (order.quantity if order.side == "buy" else -order.quantity)
        
        # Check max position size
        max_position_value = 1000000  # $1M limit
        estimated_position_value = new_quantity * order.limit_price if order.limit_price else new_quantity * 100
        
        if estimated_position_value > max_position_value:
            violations.append(f"Position size exceeds limit: ${estimated_position_value:,.2f} > ${max_position_value:,.2f}")
        
        # Check concentration limit
        total_portfolio_value = await self._get_portfolio_value()
        concentration = estimated_position_value / total_portfolio_value if total_portfolio_value > 0 else 0
        
        if concentration > 0.1:  # 10% concentration limit
            violations.append(f"Position concentration exceeds limit: {concentration:.1%} > 10%")
        
        return violations
    
    async def _check_portfolio_limits(self, order: Order) -> List[str]:
        """Check portfolio-level limits"""
        violations = []
        
        # Check if order would violate portfolio VaR limit
        estimated_var_impact = self._estimate_var_impact(order)
        current_var = self.current_limits.get("max_portfolio_var", 0)
        
        if estimated_var_impact > current_var * 0.5:  # 50% of VaR limit
            violations.append(f"Order would exceed portfolio VaR limit: ${estimated_var_impact:,.2f} impact")
        
        # Check leverage limits
        estimated_leverage_impact = self._estimate_leverage_impact(order)
        current_leverage = self.current_limits.get("max_leverage", 2.0)
        
        if estimated_leverage_impact > current_leverage * 0.8:  # 80% of leverage limit
            violations.append(f"Order would exceed leverage limit: {estimated_leverage_impact:.2f}x impact")
        
        return violations
    
    async def _check_regulatory_limits(self, order: Order) -> List[str]:
        """Check regulatory limits"""
        violations = []
        
        # Check short sale restrictions
        if order.side == "sell":
            current_positions = await self._get_symbol_positions(order.symbol)
            if current_positions.quantity < order.quantity:
                violations.append("Short selling not allowed by regulations")
        
        # Check market manipulation patterns
        if await self._detect_manipulative_patterns(order):
            violations.append("Potential market manipulation detected")
        
        return violations
    
    def _estimate_var_impact(self, order: Order) -> float:
        """Estimate VaR impact of order"""
        # Simplified VaR impact calculation
        position_value = order.quantity * (order.limit_price or 100)
        volatility = 0.02  # 2% daily volatility assumption
        return position_value * volatility * 2  # 2-sigma VaR
    
    def _estimate_leverage_impact(self, order: Order) -> float:
        """Estimate leverage impact of order"""
        # Simplified leverage impact calculation
        position_value = order.quantity * (order.limit_price or 100)
        margin_required = position_value * 0.5  # 50% margin
        # Mock account equity
        account_equity = 200000
        leverage_impact = position_value / (account_equity + margin_required)
        return leverage_impact
    
    async def _get_symbol_positions(self, symbol: str) -> Position:
        """Get current positions for symbol"""
        for positions in self.broker_positions.values():
            for position in positions:
                if position.symbol == symbol:
                    return position
        
        # Return zero position if not found
        return Position(
            symbol=symbol,
            quantity=0,
            avg_cost=0,
            market_price=100,
            market_value=0,
            unrealized_pnl=0,
            realized_pnl=0,
            position_type="long",
            broker="mock"
        )
    
    async def _get_portfolio_value(self) -> float:
        """Get total portfolio value"""
        total_value = 0.0
        for positions in self.broker_positions.values():
            for position in positions:
                total_value += position.market_value
        return total_value
    
    async def _detect_manipulative_patterns(self, order: Order) -> bool:
        """Detect potential manipulative trading patterns"""
        # Mock detection - always returns False for demonstration
        return False
    
    async def _generate_risk_recommendations(self, order: Order, violations: List[str]) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        if violations:
            recommendations.append("Consider reducing order size")
            recommendations.append("Review current portfolio exposure")
            recommendations.append("Check regulatory compliance")
            
            if "VaR" in str(violations):
                recommendations.append("Consider hedging portfolio exposure")
            
            if "concentration" in str(violations):
                recommendations.append("Diversify positions across sectors")
            
            if "leverage" in str(violations):
                recommendations.append("Reduce leverage or increase capital")
        else:
            recommendations.append("Order meets all risk criteria")
            recommendations.append("Monitor position after execution")
        
        return recommendations
    
    async def submit_order(self, order: Order) -> RiskCheckResult:
        """Submit order for execution with risk checks"""
        try:
            # Add to pending orders
            self.pending_orders[order.order_id] = order
            
            # Perform risk checks
            risk_result = await self.check_order_risk(order)
            
            # Store risk check results
            order.risk_checks = asdict(risk_result)
            
            if risk_result.allowed:
                # Add to processing queue
                await self.order_queue.put(order)
                
                # Route to broker if approved
                await self._route_order_to_broker(order)
            else:
                # Mark as rejected
                order.status = OrderStatus.REJECTED
                self.logger.warning(f"Order {order.order_id} rejected due to risk violations: {risk_result.violations}")
            
            return risk_result
            
        except Exception as e:
            self.logger.error(f"Error submitting order {order.order_id}: {e}")
            return RiskCheckResult(
                order_id=order.order_id,
                allowed=False,
                action=RiskAction.REJECT,
                risk_score=1.0,
                violations=[f"Order submission error: {str(e)}"],
                recommendations=["Manual review required"],
                limit_utilization={},
                timestamp=datetime.now()
            )
    
    async def _route_order_to_broker(self, order: Order):
        """Route order to appropriate broker"""
        try:
            # Select broker based on symbol and order size
            selected_broker = await self._select_broker(order)
            order.broker = selected_broker.value
            
            # Simulate order routing
            order.status = OrderStatus.ACCEPTED
            self.logger.info(f"Order {order.order_id} routed to {selected_broker.value}")
            
        except Exception as e:
            self.logger.error(f"Error routing order {order.order_id}: {e}")
            order.status = OrderStatus.REJECTED
    
    async def _select_broker(self, order: Order) -> BrokerType:
        """Select optimal broker for order"""
        # Simple broker selection logic
        if order.symbol in ["AAPL", "MSFT", "GOOGL"]:
            return BrokerType.INTERACTIVE_BROKERS
        else:
            return BrokerType.TD_AMERITRADE
    
    async def _process_order(self, order: Order):
        """Process order execution"""
        try:
            # Simulate order execution
            await asyncio.sleep(0.1)
            
            # Mock execution
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.avg_fill_price = order.limit_price or 100.0
            
            # Move to history
            self.order_history.append(order)
            del self.pending_orders[order.order_id]
            
            self.logger.info(f"Order {order.order_id} executed at ${order.avg_fill_price}")
            
        except Exception as e:
            self.logger.error(f"Error processing order {order.order_id}: {e}")
            order.status = OrderStatus.REJECTED
    
    async def _update_market_data(self):
        """Update market data for subscribed symbols"""
        # Mock market data updates
        mock_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        
        for symbol in mock_symbols:
            self.market_data[symbol] = MarketData(
                symbol=symbol,
                timestamp=datetime.now(),
                bid=np.random.uniform(99, 101),
                ask=np.random.uniform(101, 103),
                last=np.random.uniform(100, 102),
                volume=np.random.uniform(1000000, 5000000),
                open=100.0,
                high=105.0,
                low=98.0,
                close=np.random.uniform(100, 102),
                change=np.random.uniform(-2, 2),
                change_percent=np.random.uniform(-0.02, 0.02)
            )
    
    def get_risk_alerts(self, severity_filter: Optional[str] = None) -> List[Dict]:
        """Get current risk alerts"""
        if severity_filter:
            return [alert for alert in self.risk_alerts if alert.get("severity") == severity_filter]
        return self.risk_alerts
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary"""
        all_positions = []
        for positions in self.broker_positions.values():
            all_positions.extend(positions)
        
        total_value = sum(pos.market_value for pos in all_positions)
        total_pnl = sum(pos.unrealized_pnl for pos in all_positions)
        
        return {
            "timestamp": datetime.now(),
            "total_value": total_value,
            "total_unrealized_pnl": total_pnl,
            "position_count": len(all_positions),
            "pending_orders": len(self.pending_orders),
            "active_brokers": len([b for b in self.broker_connections.values() if b["connected"]]),
            "risk_alerts_count": len(self.risk_alerts),
            "limit_breaches_count": len(self.limit_breaches)
        }


# Example usage and testing
if __name__ == "__main__":
    async def main():
        # Initialize risk broker integration
        risk_integration = RiskBrokerIntegration()
        
        # Start monitoring
        await risk_integration.start_monitoring()
        
        # Create sample order
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol="AAPL",
            side="buy",
            quantity=100,
            order_type="limit",
            limit_price=150.0,
            strategy_id="TEST_STRATEGY"
        )
        
        # Submit order with risk checks
        risk_result = await risk_integration.submit_order(order)
        print(f"Order Risk Check Result: {risk_result.action.value}")
        print(f"Violations: {risk_result.violations}")
        print(f"Recommendations: {risk_result.recommendations}")
        
        # Get portfolio summary
        summary = risk_integration.get_portfolio_summary()
        print(f"Portfolio Summary: {summary}")
        
        # Get risk alerts
        alerts = risk_integration.get_risk_alerts()
        print(f"Risk Alerts: {len(alerts)}")
        
        # Wait a bit then stop monitoring
        await asyncio.sleep(5)
        await risk_integration.stop_monitoring()
    
    # Run the example
    asyncio.run(main())