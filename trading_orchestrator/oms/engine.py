"""
@file engine.py
@brief Order Management System (OMS) - Main Engine

@details
This module implements the central Order Management System that orchestrates
all order operations across multiple brokers in the trading system. It provides
a unified interface for order submission, routing, execution tracking, and
settlement processing.

Key Features:
- Multi-broker order routing and execution
- Real-time order status tracking and monitoring
- Slippage analysis and execution quality measurement
- Position reconciliation and P&L tracking
- Trade settlement processing
- Order lifecycle management
- Performance analytics and reporting

@author Trading Orchestrator System
@version 1.0
@date 2025-11-06

@warning
The OMS handles real trading orders that may result in actual financial
transactions. Incorrect configuration can lead to unintended trades.

@note
This module orchestrates the following OMS components:
- OrderRouter: Routes orders to appropriate brokers
- OrderValidator: Validates orders before submission
- ExecutionMonitor: Tracks order execution in real-time
- SlippageTracker: Measures execution quality
- PositionManager: Manages position updates
- TradeSettlement: Processes trade settlements
- PerformanceAnalytics: Provides execution analytics

@see oms.router for order routing logic
@see oms.validator for order validation
@see oms.monitor for execution monitoring
@see oms.tracker for slippage tracking
@see oms.manager for position management
@see oms.settlement for trade settlement
@see oms.analytics for performance analytics
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc

from config.database import get_db
from database.models.trading import Order, Position, Trade, OrderStatus, OrderType, OrderSide
from database.models.risk import RiskLimit, RiskEvent, RiskEventType
from .router import OrderRouter
from .validator import OrderValidator
from .monitor import ExecutionMonitor
from .tracker import SlippageTracker
from .manager import PositionManager
from .settlement import TradeSettlement
from .analytics import PerformanceAnalytics

logger = logging.getLogger(__name__)


# Alias for main.py compatibility
OMSEngine = OrderManagementSystem


class OrderManagementSystem:
    """
    @class OrderManagementSystem
    @brief Central Order Management System
    
    @details
    The OrderManagementSystem serves as the central coordinator for all order
    operations in the trading system. It provides a unified interface for
    multi-broker order routing, execution tracking, and settlement processing.
    
    @par Architecture:
    The OMS integrates the following components:
    - OrderRouter: Routes orders to optimal brokers
    - OrderValidator: Validates orders against rules and limits
    - ExecutionMonitor: Tracks real-time order execution
    - SlippageTracker: Measures execution quality and timing
    - PositionManager: Maintains position state across brokers
    - TradeSettlement: Processes completed trades
    - PerformanceAnalytics: Provides execution analytics
    
    @par Order Flow:
    1. Order submission with validation
    2. Risk and policy compliance check
    3. Optimal broker selection and routing
    4. Real-time execution monitoring
    5. Position update and reconciliation
    6. Trade settlement and P&L calculation
    7. Performance analytics and reporting
    
    @par Order States:
    - PENDING: Order submitted, awaiting validation
    - VALIDATED: Order passed validation, ready for routing
    - ROUTED: Order sent to broker
    - PARTIALLY_FILLED: Order partially executed
    - FILLED: Order fully executed
    - CANCELLED: Order cancelled by user/system
    - REJECTED: Order rejected by broker or risk system
    
    @par Multi-Broker Support:
    - Automatic broker selection based on:
      * Instrument availability
      * Trading fees
      * Market liquidity
      * Execution speed
    - Fallback broker handling
    - Cross-broker position aggregation
    
    @warning
    The OMS handles real trading orders. All order routing and execution
    must be thoroughly tested before production use.
    
    @par Usage Example:
    @code
    from oms.engine import OrderManagementSystem
    
    # Initialize OMS
    oms = OrderManagementSystem(user_id=123)
    
    # Submit order for execution
    order_request = {
        'symbol': 'AAPL',
        'side': 'buy',
        'quantity': 100,
        'order_type': 'market',
        'time_in_force': 'day'
    }
    
    # Submit with validation
    result = await oms.submit_order(order_request)
    print(f"Order ID: {result.order_id}")
    
    # Monitor execution
    order_status = await oms.get_order_status(result.order_id)
    print(f"Status: {order_status.status}")
    
    # Get position update
    positions = await oms.get_positions()
    for pos in positions:
        print(f"{pos.symbol}: {pos.quantity} shares")
    
    # Get performance analytics
    analytics = await oms.get_execution_analytics()
    print(f"Total slippage: {analytics.total_slippage:.4f}")
    @endcode
    
    @note
    This is the main entry point for all order operations in the trading system.
    
    @see OrderRouter for routing logic
    @see OrderValidator for validation rules
    @see ExecutionMonitor for tracking
    @see PositionManager for position handling
    """
    
    def __init__(self, user_id: int):
        """
        @brief Initialize OrderManagementSystem
        
        @param user_id User identifier for order tracking
        
        @details
        Initializes the OMS with user-specific context and initializes
        all order management components.
        
        @par Initialization Process:
        1. Set user identifier for order tracking
        2. Initialize database connection
        3. Create OMS component instances
        4. Load user order preferences and limits
        5. Set up monitoring and alert systems
        
        @throws ValueError if user_id is invalid
        @throws DatabaseError if database connection fails
        
        @par Example:
        @code
        # Initialize for specific user
        oms = OrderManagementSystem(user_id=12345)
        
        # OMS is now ready for order operations
        @endcode
        
        @note
        User-specific order preferences and limits are loaded during initialization.
        """
        if user_id <= 0:
            raise ValueError("User ID must be positive")
            
        self.user_id = user_id
        self.db = get_db()
        
        # Initialize OMS components
        self.router = OrderRouter(user_id)
        self.validator = OrderValidator(user_id)
        self.monitor = ExecutionMonitor(user_id)
        self.tracker = SlippageTracker(user_id)
        self.position_manager = PositionManager(user_id)
        self.settlement = TradeSettlement(user_id)
        self.analytics = PerformanceAnalytics(user_id)
        
        # Order routing and execution configuration
        self.broker_priorities = {
            "binance": 1,
            "alpaca": 2, 
            "ibkr": 3,
            "trading212": 4
        }
        
        # Order state tracking
        self.active_orders = {}
        self.execution_queue = asyncio.Queue()
        self.monitoring_tasks = []
        
        logger.info(f"OMS initialized for user {self.user_id}")
    
    async def submit_order(self, order_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit order through the complete OMS pipeline.
        
        Args:
            order_request: Order submission request
            
        Returns:
            Order submission result with order ID and status
        """
        result = {
            "success": False,
            "order_id": None,
            "broker_order_id": None,
            "status": OrderStatus.PENDING,
            "errors": [],
            "warnings": [],
            "execution_details": {}
        }
        
        try:
            # 1. Pre-submission validation
            validation_result = await self.validator.validate_order(order_request)
            if not validation_result["approved"]:
                result["errors"] = validation_result.get("rejection_reason", "Validation failed")
                result["warnings"] = validation_result.get("warnings", [])
                return result
            
            # 2. Create order record
            order = await self._create_order_record(order_request)
            result["order_id"] = order.id
            
            # 3. Route order to broker
            routing_result = await self.router.route_order(order)
            if not routing_result["success"]:
                result["errors"] = routing_result.get("errors", ["Routing failed"])
                await self._update_order_status(order.id, OrderStatus.REJECTED)
                return result
            
            result["broker_order_id"] = routing_result["broker_order_id"]
            result["status"] = OrderStatus.NEW
            
            # 4. Start execution monitoring
            await self.monitor.start_monitoring(order.id)
            
            # 5. Log order submission
            await self._log_order_event(order.id, "order_submitted", {
                "order_request": order_request,
                "broker_response": routing_result
            })
            
            result["success"] = True
            result["execution_details"] = routing_result
            
            logger.info(f"Order submitted successfully for user {self.user_id}: {order.id}")
            
        except Exception as e:
            logger.error(f"Order submission error for user {self.user_id}: {str(e)}")
            result["errors"] = [f"Submission error: {str(e)}"]
            
            if "order_id" in locals():
                await self._update_order_status(order.id, OrderStatus.REJECTED, str(e))
        
        return result
    
    async def cancel_order(self, order_id: int, cancellation_reason: str = "User requested") -> Dict[str, Any]:
        """
        Cancel an active order.
        
        Args:
            order_id: ID of order to cancel
            cancellation_reason: Reason for cancellation
            
        Returns:
            Cancellation result
        """
        result = {
            "success": False,
            "order_id": order_id,
            "errors": []
        }
        
        try:
            # Get order
            order = self.db.query(Order).filter(
                and_(
                    Order.id == order_id,
                    Order.user_id == self.user_id
                )
            ).first()
            
            if not order:
                result["errors"] = ["Order not found"]
                return result
            
            # Check if order can be cancelled
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                result["errors"] = [f"Order cannot be cancelled - status: {order.status.value}"]
                return result
            
            # Cancel through broker
            cancel_result = await self.router.cancel_order(order)
            
            if cancel_result["success"]:
                await self._update_order_status(order.id, OrderStatus.CANCELLED)
                await self.monitor.stop_monitoring(order.id)
                
                result["success"] = True
                
                # Log cancellation
                await self._log_order_event(order.id, "order_cancelled", {
                    "cancellation_reason": cancellation_reason,
                    "broker_response": cancel_result
                })
                
                logger.info(f"Order cancelled for user {self.user_id}: {order_id}")
            else:
                result["errors"] = cancel_result.get("errors", ["Cancellation failed"])
                
        except Exception as e:
            logger.error(f"Order cancellation error for user {self.user_id}: {str(e)}")
            result["errors"] = [f"Cancellation error: {str(e)}"]
        
        return result
    
    async def modify_order(self, order_id: int, modifications: Dict[str, Any]) -> Dict[str, Any]:
        """
        Modify an existing order (price, quantity, etc.).
        
        Args:
            order_id: ID of order to modify
            modifications: Dictionary of modifications
            
        Returns:
            Modification result
        """
        result = {
            "success": False,
            "order_id": order_id,
            "errors": []
        }
        
        try:
            # Get order
            order = self.db.query(Order).filter(
                and_(
                    Order.id == order_id,
                    Order.user_id == self.user_id
                )
            ).first()
            
            if not order:
                result["errors"] = ["Order not found"]
                return result
            
            # Check if order can be modified
            if order.status not in [OrderStatus.NEW, OrderStatus.PENDING]:
                result["errors"] = [f"Order cannot be modified - status: {order.status.value}"]
                return result
            
            # Apply modifications
            modified_order_data = order.__dict__.copy()
            modified_order_data.update(modifications)
            
            # Validate modified order
            validation_result = await self.validator.validate_order(modified_order_data)
            if not validation_result["approved"]:
                result["errors"] = [validation_result.get("rejection_reason", "Modified order validation failed")]
                return result
            
            # Submit modification through broker
            modify_result = await self.router.modify_order(order, modifications)
            
            if modify_result["success"]:
                # Update order record
                for key, value in modifications.items():
                    setattr(order, key, value)
                
                order.updated_at = datetime.now()
                self.db.commit()
                
                result["success"] = True
                
                # Log modification
                await self._log_order_event(order.id, "order_modified", {
                    "modifications": modifications,
                    "broker_response": modify_result
                })
                
                logger.info(f"Order modified for user {self.user_id}: {order_id}")
            else:
                result["errors"] = modify_result.get("errors", ["Modification failed"])
                
        except Exception as e:
            logger.error(f"Order modification error for user {self.user_id}: {str(e)}")
            result["errors"] = [f"Modification error: {str(e)}"]
        
        return result
    
    async def get_order_status(self, order_id: int) -> Dict[str, Any]:
        """
        Get current status of an order.
        
        Args:
            order_id: ID of order to check
            
        Returns:
            Current order status and details
        """
        try:
            order = self.db.query(Order).filter(
                and_(
                    Order.id == order_id,
                    Order.user_id == self.user_id
                )
            ).first()
            
            if not order:
                return {"error": "Order not found"}
            
            # Get execution details from monitor
            execution_details = await self.monitor.get_execution_details(order_id)
            
            # Calculate slippage if order is filled
            slippage_info = {}
            if order.status == OrderStatus.FILLED:
                slippage_info = await self.tracker.calculate_slippage(order_id)
            
            return {
                "order_id": order.id,
                "broker_order_id": order.broker_order_id,
                "symbol": order.symbol,
                "side": order.side.value,
                "quantity": order.quantity,
                "filled_quantity": order.filled_quantity,
                "remaining_quantity": order.remaining_quantity,
                "order_type": order.order_type.value,
                "status": order.status.value,
                "limit_price": order.limit_price,
                "stop_price": order.stop_price,
                "avg_fill_price": order.avg_fill_price,
                "submitted_at": order.submitted_at,
                "filled_at": order.filled_at,
                "cancelled_at": order.cancelled_at,
                "execution_details": execution_details,
                "slippage": slippage_info,
                "metadata": order.metadata or {}
            }
            
        except Exception as e:
            logger.error(f"Order status retrieval error: {str(e)}")
            return {"error": str(e)}
    
    async def get_portfolio_positions(self) -> List[Dict[str, Any]]:
        """
        Get all current portfolio positions.
        
        Returns:
            List of current positions
        """
        try:
            positions = await self.position_manager.get_all_positions()
            
            # Add performance analytics for each position
            for position in positions:
                position["performance"] = await self.analytics.get_position_performance(position["id"])
                position["risk_metrics"] = await self.analytics.get_position_risk_metrics(position["symbol"])
            
            return positions
            
        except Exception as e:
            logger.error(f"Portfolio positions retrieval error: {str(e)}")
            return []
    
    async def close_position(self, position_id: int, close_quantity: float = None,
                           order_type: OrderType = OrderType.MARKET) -> Dict[str, Any]:
        """
        Close (partially or fully) a position.
        
        Args:
            position_id: ID of position to close
            close_quantity: Quantity to close (None = full position)
            order_type: Order type for closing
            
        Returns:
            Position closing result
        """
        try:
            # Get position
            position = self.db.query(Position).filter(
                and_(
                    Position.id == position_id,
                    Position.user_id == self.user_id
                )
            ).first()
            
            if not position:
                return {"success": False, "error": "Position not found"}
            
            # Determine closing quantity
            if close_quantity is None or close_quantity >= abs(position.quantity):
                close_quantity = abs(position.quantity)
            
            # Create closing order
            close_side = OrderSide.SELL if position.side.value == "long" else OrderSide.BUY
            
            close_order_data = {
                "symbol": position.symbol,
                "side": close_side.value,
                "quantity": close_quantity,
                "order_type": order_type.value,
                "asset_class": position.asset_class,
                "exchange": position.exchange
            }
            
            # Submit closing order
            close_result = await self.submit_order(close_order_data)
            
            if close_result["success"]:
                # Update position with reference to closing order
                if position.metadata is None:
                    position.metadata = {}
                position.metadata["closing_order_id"] = close_result["order_id"]
                self.db.commit()
                
                logger.info(f"Position close initiated for user {self.user_id}: {position_id}")
            
            return close_result
            
        except Exception as e:
            logger.error(f"Position close error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def reconcile_positions(self) -> Dict[str, Any]:
        """
        Reconcile portfolio positions with broker data.
        
        Returns:
            Reconciliation results
        """
        try:
            reconciliation_result = await self.position_manager.reconcile_all_positions()
            
            # Log reconciliation
            await self._log_system_event("position_reconciliation", {
                "reconciliation_result": reconciliation_result,
                "timestamp": datetime.now()
            })
            
            return reconciliation_result
            
        except Exception as e:
            logger.error(f"Position reconciliation error: {str(e)}")
            return {"error": str(e)}
    
    async def process_trade_settlement(self, trade_id: int) -> Dict[str, Any]:
        """
        Process trade settlement for a completed trade.
        
        Args:
            trade_id: ID of trade to settle
            
        Returns:
            Settlement processing result
        """
        try:
            settlement_result = await self.settlement.process_trade_settlement(trade_id)
            
            # Update position based on settlement
            if settlement_result["success"]:
                await self.position_manager.update_position_after_settlement(trade_id)
            
            return settlement_result
            
        except Exception as e:
            logger.error(f"Trade settlement processing error: {str(e)}")
            return {"error": str(e)}
    
    async def get_performance_analytics(self, start_date: datetime = None, 
                                      end_date: datetime = None) -> Dict[str, Any]:
        """
        Get comprehensive performance analytics.
        
        Args:
            start_date: Analytics start date
            end_date: Analytics end date
            
        Returns:
            Performance analytics report
        """
        try:
            if not start_date:
                start_date = datetime.now() - timedelta(days=30)
            if not end_date:
                end_date = datetime.now()
            
            analytics = {
                "period": {
                    "start_date": start_date,
                    "end_date": end_date
                },
                "summary": await self.analytics.get_portfolio_summary(start_date, end_date),
                "performance_metrics": await self.analytics.get_performance_metrics(start_date, end_date),
                "risk_metrics": await self.analytics.get_portfolio_risk_metrics(start_date, end_date),
                "execution_quality": await self.analytics.get_execution_quality_metrics(start_date, end_date),
                "slippage_analysis": await self.tracker.get_slippage_analysis(start_date, end_date),
                "strategy_performance": await self.analytics.get_strategy_performance(start_date, end_date)
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Performance analytics error: {str(e)}")
            return {"error": str(e)}
    
    async def _create_order_record(self, order_request: Dict[str, Any]) -> Order:
        """Create order record in database."""
        try:
            # Generate client order ID
            client_order_id = f"{self.user_id}_{int(datetime.now().timestamp())}_{order_request.get('symbol', 'UNKNOWN')}"
            
            order = Order(
                user_id=self.user_id,
                broker_name=order_request.get("broker", "default"),
                account_id=order_request.get("account_id", "default"),
                broker_order_id="",  # Will be updated after broker submission
                client_order_id=client_order_id,
                symbol=order_request["symbol"],
                exchange=order_request.get("exchange"),
                asset_class=order_request["asset_class"],
                order_type=OrderType(order_request["order_type"]),
                side=OrderSide(order_request["side"]),
                quantity=order_request["quantity"],
                filled_quantity=0.0,
                remaining_quantity=order_request["quantity"],
                limit_price=order_request.get("limit_price"),
                stop_price=order_request.get("stop_price"),
                time_in_force=order_request.get("time_in_force", "day"),
                strategy_id=order_request.get("strategy_id"),
                tags=order_request.get("tags", []),
                submitted_at=datetime.now(),
                metadata=order_request.get("metadata", {})
            )
            
            self.db.add(order)
            self.db.commit()
            
            return order
            
        except Exception as e:
            logger.error(f"Order record creation error: {str(e)}")
            raise
    
    async def _update_order_status(self, order_id: int, status: OrderStatus, error_message: str = None):
        """Update order status in database."""
        try:
            order = self.db.query(Order).filter(Order.id == order_id).first()
            if order:
                order.status = status
                if status == OrderStatus.FILLED:
                    order.filled_at = datetime.now()
                elif status == OrderStatus.CANCELLED:
                    order.cancelled_at = datetime.now()
                if error_message:
                    order.error_message = error_message
                order.updated_at = datetime.now()
                self.db.commit()
                
        except Exception as e:
            logger.error(f"Order status update error: {str(e)}")
    
    async def _log_order_event(self, order_id: int, event_type: str, event_data: Dict[str, Any]):
        """Log order-related events."""
        try:
            # This would integrate with the audit logger
            logger.info(f"Order event: {event_type} for order {order_id}")
            
        except Exception as e:
            logger.error(f"Order event logging error: {str(e)}")
    
    async def _log_system_event(self, event_type: str, event_data: Dict[str, Any]):
        """Log system-level events."""
        try:
            # This would integrate with the audit logger
            logger.info(f"System event: {event_type}")
            
        except Exception as e:
            logger.error(f"System event logging error: {str(e)}")
    
    async def start_monitoring_services(self):
        """Start background monitoring services."""
        try:
            # Start execution monitoring
            await self.monitor.start_background_monitoring()
            
            # Start position monitoring
            await self.position_manager.start_background_monitoring()
            
            # Start settlement monitoring
            await self.settlement.start_background_monitoring()
            
            logger.info("OMS monitoring services started")
            
        except Exception as e:
            logger.error(f"OMS monitoring services startup error: {str(e)}")
    
    async def stop_monitoring_services(self):
        """Stop background monitoring services."""
        try:
            # Stop all monitoring services
            await self.monitor.stop_background_monitoring()
            await self.position_manager.stop_background_monitoring()
            await self.settlement.stop_background_monitoring()
            
            logger.info("OMS monitoring services stopped")
            
        except Exception as e:
            logger.error(f"OMS monitoring services shutdown error: {str(e)}")
    
    def close(self):
        """Cleanup OMS resources."""
        try:
            # Close database connection
            self.db.close()
            
            # Close component resources
            for component in [self.router, self.validator, self.monitor, self.tracker, 
                            self.position_manager, self.settlement, self.analytics]:
                if hasattr(component, 'close'):
                    component.close()
            
        except Exception as e:
            logger.error(f"OMS cleanup error: {str(e)}")
