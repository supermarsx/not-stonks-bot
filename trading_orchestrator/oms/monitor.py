"""
Execution Monitor - Real-time Order Execution Tracking

Monitors order execution across all brokers:
- Real-time order status updates
- Fill tracking and analysis
- Execution performance metrics
- Timeout and failure handling
- Alert generation for execution issues
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum

from config.database import get_db
from database.models.trading import Order, Trade, OrderStatus
from database.models.broker import BrokerConnection

logger = logging.getLogger(__name__)


class ExecutionEventType(str, Enum):
    """Types of execution monitoring events."""
    ORDER_SUBMITTED = "order_submitted"
    ORDER_ACCEPTED = "order_accepted"
    ORDER_PARTIALLY_FILLED = "order_partially_filled"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    ORDER_EXPIRED = "order_expired"
    EXECUTION_TIMEOUT = "execution_timeout"
    EXECUTION_ERROR = "execution_error"


class ExecutionMonitor:
    """
    Monitors order execution across all brokers in real-time.
    
    Provides:
    - Real-time order status tracking
    - Fill monitoring and analysis
    - Execution performance metrics
    - Timeout and failure handling
    - Alert generation for execution issues
    """
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.db = get_db()
        
        # Monitoring configuration
        self.monitoring_interval = 5  # seconds
        self.execution_timeout = 300  # 5 minutes default
        self.alert_thresholds = {
            "execution_time_warning": 30,  # seconds
            "execution_time_critical": 60,  # seconds
            "partial_fill_warning": 0.8,  # 80% filled
            "slippage_warning": 0.005,  # 0.5% slippage
            "slippage_critical": 0.01  # 1% slippage
        }
        
        # Active monitoring state
        self.monitored_orders = {}  # order_id -> monitoring data
        self.execution_callbacks = {}  # order_id -> callback functions
        self.background_task = None
        self.is_monitoring = False
        
        # Performance metrics
        self.execution_metrics = {
            "total_orders": 0,
            "filled_orders": 0,
            "cancelled_orders": 0,
            "rejected_orders": 0,
            "avg_execution_time": 0.0,
            "avg_fill_rate": 0.0,
            "last_updated": None
        }
        
        logger.info(f"ExecutionMonitor initialized for user {self.user_id}")
    
    async def start_monitoring(self, order_id: int) -> bool:
        """
        Start monitoring a specific order.
        
        Args:
            order_id: ID of order to monitor
            
        Returns:
            Success status
        """
        try:
            # Get order details
            order = self.db.query(Order).filter(
                Order.id == order_id
            ).first()
            
            if not order:
                logger.warning(f"Order {order_id} not found for monitoring")
                return False
            
            # Initialize monitoring data
            self.monitored_orders[order_id] = {
                "order_id": order_id,
                "symbol": order.symbol,
                "broker_name": order.broker_name,
                "broker_order_id": order.broker_order_id,
                "start_time": datetime.now(),
                "status": order.status.value,
                "filled_quantity": order.filled_quantity,
                "total_quantity": order.quantity,
                "avg_fill_price": order.avg_fill_price,
                "expected_price": order.limit_price or order.metadata.get("expected_price"),
                "events": [],
                "alerts": []
            }
            
            # Set up execution timeout
            if order.status == OrderStatus.NEW:
                timeout_task = asyncio.create_task(
                    self._handle_execution_timeout(order_id)
                )
                self.monitored_orders[order_id]["timeout_task"] = timeout_task
            
            logger.info(f"Started monitoring order {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Order monitoring startup error: {str(e)}")
            return False
    
    async def stop_monitoring(self, order_id: int) -> bool:
        """
        Stop monitoring a specific order.
        
        Args:
            order_id: ID of order to stop monitoring
            
        Returns:
            Success status
        """
        try:
            if order_id in self.monitored_orders:
                # Cancel timeout task if exists
                if "timeout_task" in self.monitored_orders[order_id]:
                    self.monitored_orders[order_id]["timeout_task"].cancel()
                
                # Clean up callbacks
                if order_id in self.execution_callbacks:
                    del self.execution_callbacks[order_id]
                
                # Remove from monitored orders
                del self.monitored_orders[order_id]
                
                logger.info(f"Stopped monitoring order {order_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Order monitoring stop error: {str(e)}")
            return False
    
    async def get_execution_details(self, order_id: int) -> Dict[str, Any]:
        """
        Get execution details for a monitored order.
        
        Args:
            order_id: ID of order
            
        Returns:
            Execution details
        """
        try:
            if order_id not in self.monitored_orders:
                return {"error": "Order not being monitored"}
            
            monitoring_data = self.monitored_orders[order_id]
            
            # Get current order data from database
            order = self.db.query(Order).filter(Order.id == order_id).first()
            
            if not order:
                return {"error": "Order not found"}
            
            # Calculate metrics
            execution_time = (datetime.now() - monitoring_data["start_time"]).total_seconds()
            fill_rate = order.filled_quantity / order.quantity if order.quantity > 0 else 0
            
            # Calculate slippage if order is filled
            slippage = None
            if order.status == OrderStatus.FILLED and order.avg_fill_price and monitoring_data.get("expected_price"):
                expected_price = monitoring_data["expected_price"]
                slippage = (order.avg_fill_price - expected_price) / expected_price
            
            return {
                "order_id": order_id,
                "symbol": order.symbol,
                "broker_name": order.broker_name,
                "broker_order_id": order.broker_order_id,
                "status": order.status.value,
                "quantity": order.quantity,
                "filled_quantity": order.filled_quantity,
                "remaining_quantity": order.remaining_quantity,
                "avg_fill_price": order.avg_fill_price,
                "limit_price": order.limit_price,
                "execution_time_seconds": execution_time,
                "fill_rate": fill_rate,
                "slippage": slippage,
                "monitoring_start": monitoring_data["start_time"],
                "events": monitoring_data["events"],
                "alerts": monitoring_data["alerts"],
                "is_monitoring": True
            }
            
        except Exception as e:
            logger.error(f"Execution details retrieval error: {str(e)}")
            return {"error": str(e)}
    
    async def update_order_status(self, order_id: int, status: OrderStatus, 
                                fill_data: Dict[str, Any] = None) -> bool:
        """
        Update order status and trigger monitoring events.
        
        Args:
            order_id: ID of order
            status: New order status
            fill_data: Fill information (quantity, price, etc.)
            
        Returns:
            Success status
        """
        try:
            if order_id not in self.monitored_orders:
                logger.warning(f"Order {order_id} not in monitoring list")
                return False
            
            monitoring_data = self.monitored_orders[order_id]
            old_status = monitoring_data["status"]
            
            # Update monitoring data
            monitoring_data["status"] = status.value
            monitoring_data["last_updated"] = datetime.now()
            
            # Update fill data if provided
            if fill_data:
                monitoring_data["filled_quantity"] = fill_data.get("filled_quantity", monitoring_data["filled_quantity"])
                monitoring_data["avg_fill_price"] = fill_data.get("avg_fill_price", monitoring_data["avg_fill_price"])
            
            # Create execution event
            event = {
                "event_type": self._get_event_type(status, old_status),
                "timestamp": datetime.now(),
                "status": status.value,
                "fill_data": fill_data or {}
            }
            
            monitoring_data["events"].append(event)
            
            # Handle status-specific actions
            if status == OrderStatus.FILLED:
                await self._handle_order_filled(order_id, monitoring_data, fill_data)
                await self._cancel_timeout(order_id)
                
            elif status in [OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
                await self._handle_order_terminated(order_id, monitoring_data, status)
                await self._cancel_timeout(order_id)
                
            elif status == OrderStatus.PARTIALLY_FILLED:
                await self._handle_partial_fill(order_id, monitoring_data, fill_data)
            
            # Trigger callbacks
            await self._trigger_callbacks(order_id, event)
            
            # Generate alerts if needed
            await self._check_and_generate_alerts(order_id, monitoring_data)
            
            logger.debug(f"Order {order_id} status updated to {status.value}")
            return True
            
        except Exception as e:
            logger.error(f"Order status update error: {str(e)}")
            return False
    
    async def start_background_monitoring(self):
        """Start background monitoring loop."""
        if self.is_monitoring:
            logger.warning("Background monitoring already active")
            return
        
        try:
            self.is_monitoring = True
            self.background_task = asyncio.create_task(self._background_monitoring_loop())
            
            logger.info("Background monitoring started")
            
        except Exception as e:
            logger.error(f"Background monitoring startup error: {str(e)}")
            self.is_monitoring = False
    
    async def stop_background_monitoring(self):
        """Stop background monitoring loop."""
        if not self.is_monitoring:
            return
        
        try:
            self.is_monitoring = False
            
            if self.background_task:
                self.background_task.cancel()
                try:
                    await self.background_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Background monitoring stopped")
            
        except Exception as e:
            logger.error(f"Background monitoring shutdown error: {str(e)}")
    
    async def get_execution_performance_report(self, start_date: datetime, 
                                             end_date: datetime) -> Dict[str, Any]:
        """
        Get execution performance report for a date range.
        
        Args:
            start_date: Report start date
            end_date: Report end date
            
        Returns:
            Performance report
        """
        try:
            # Get orders in date range
            orders = self.db.query(Order).filter(
                Order.user_id == self.user_id,
                Order.submitted_at >= start_date,
                Order.submitted_at <= end_date
            ).all()
            
            # Calculate metrics
            total_orders = len(orders)
            filled_orders = len([o for o in orders if o.status == OrderStatus.FILLED])
            cancelled_orders = len([o for o in orders if o.status == OrderStatus.CANCELLED])
            rejected_orders = len([o for o in orders if o.status == OrderStatus.REJECTED])
            
            # Execution time metrics
            execution_times = []
            for order in orders:
                if order.filled_at:
                    exec_time = (order.filled_at - order.submitted_at).total_seconds()
                    execution_times.append(exec_time)
            
            avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
            
            # Fill rate analysis
            fill_rates = []
            for order in orders:
                if order.quantity > 0:
                    fill_rate = order.filled_quantity / order.quantity
                    fill_rates.append(fill_rate)
            
            avg_fill_rate = sum(fill_rates) / len(fill_rates) if fill_rates else 0
            
            # Performance by broker
            broker_performance = {}
            for order in orders:
                broker = order.broker_name
                if broker not in broker_performance:
                    broker_performance[broker] = {
                        "total_orders": 0,
                        "filled_orders": 0,
                        "avg_execution_time": 0,
                        "fill_rate": 0
                    }
                
                broker_performance[broker]["total_orders"] += 1
                if order.status == OrderStatus.FILLED:
                    broker_performance[broker]["filled_orders"] += 1
                
                if order.filled_at:
                    exec_time = (order.filled_at - order.submitted_at).total_seconds()
                    broker_performance[broker]["avg_execution_time"] += exec_time
                
                if order.quantity > 0:
                    fill_rate = order.filled_quantity / order.quantity
                    broker_performance[broker]["fill_rate"] += fill_rate
            
            # Normalize broker metrics
            for broker_data in broker_performance.values():
                if broker_data["total_orders"] > 0:
                    broker_data["avg_execution_time"] /= broker_data["total_orders"]
                    broker_data["fill_rate"] /= broker_data["total_orders"]
                    broker_data["fill_success_rate"] = broker_data["filled_orders"] / broker_data["total_orders"]
            
            return {
                "period": {
                    "start_date": start_date,
                    "end_date": end_date
                },
                "summary": {
                    "total_orders": total_orders,
                    "filled_orders": filled_orders,
                    "cancelled_orders": cancelled_orders,
                    "rejected_orders": rejected_orders,
                    "success_rate": filled_orders / total_orders if total_orders > 0 else 0,
                    "avg_execution_time_seconds": avg_execution_time,
                    "avg_fill_rate": avg_fill_rate
                },
                "broker_performance": broker_performance,
                "execution_times": execution_times,
                "fill_rates": fill_rates
            }
            
        except Exception as e:
            logger.error(f"Performance report generation error: {str(e)}")
            return {"error": str(e)}
    
    async def _background_monitoring_loop(self):
        """Background monitoring loop."""
        logger.info("Starting background monitoring loop")
        
        while self.is_monitoring:
            try:
                await self._check_all_monitored_orders()
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background monitoring loop error: {str(e)}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _check_all_monitored_orders(self):
        """Check status of all monitored orders."""
        try:
            for order_id in list(self.monitored_orders.keys()):
                await self._check_order_status(order_id)
                
        except Exception as e:
            logger.error(f"Order status check error: {str(e)}")
    
    async def _check_order_status(self, order_id: int):
        """Check current status of a specific order."""
        try:
            order = self.db.query(Order).filter(Order.id == order_id).first()
            
            if not order:
                await self.stop_monitoring(order_id)
                return
            
            # Check if status has changed
            monitoring_data = self.monitored_orders.get(order_id, {})
            current_status = monitoring_data.get("status")
            
            if current_status != order.status.value:
                await self.update_order_status(order_id, order.status)
            
            # Check for stale orders (no updates for extended period)
            if order.status in [OrderStatus.NEW, OrderStatus.PENDING]:
                time_since_submit = (datetime.now() - order.submitted_at).total_seconds()
                if time_since_submit > self.execution_timeout:
                    await self._handle_execution_timeout(order_id)
            
        except Exception as e:
            logger.error(f"Order status check error for {order_id}: {str(e)}")
    
    async def _handle_execution_timeout(self, order_id: int):
        """Handle execution timeout for order."""
        try:
            logger.warning(f"Execution timeout for order {order_id}")
            
            monitoring_data = self.monitored_orders.get(order_id)
            if monitoring_data:
                # Generate timeout alert
                alert = {
                    "type": "execution_timeout",
                    "message": f"Order {order_id} execution timeout after {self.execution_timeout} seconds",
                    "timestamp": datetime.now(),
                    "severity": "warning"
                }
                
                monitoring_data["alerts"].append(alert)
                
                # Trigger timeout callback
                timeout_event = {
                    "event_type": ExecutionEventType.EXECUTION_TIMEOUT,
                    "timestamp": datetime.now(),
                    "order_id": order_id,
                    "message": "Order execution timeout"
                }
                
                await self._trigger_callbacks(order_id, timeout_event)
            
        except Exception as e:
            logger.error(f"Execution timeout handling error: {str(e)}")
    
    async def _handle_order_filled(self, order_id: int, monitoring_data: Dict[str, Any], 
                                 fill_data: Dict[str, Any]):
        """Handle order filled event."""
        try:
            # Calculate final metrics
            execution_time = (datetime.now() - monitoring_data["start_time"]).total_seconds()
            fill_rate = monitoring_data["filled_quantity"] / monitoring_data["total_quantity"]
            
            # Generate filled alert if execution time is high
            if execution_time > self.alert_thresholds["execution_time_warning"]:
                alert = {
                    "type": "slow_execution",
                    "message": f"Order {order_id} filled slowly: {execution_time:.1f} seconds",
                    "timestamp": datetime.now(),
                    "severity": "warning"
                }
                monitoring_data["alerts"].append(alert)
            
            # Calculate slippage alert
            if fill_data and monitoring_data.get("expected_price"):
                actual_price = fill_data.get("avg_fill_price")
                expected_price = monitoring_data["expected_price"]
                if actual_price and expected_price:
                    slippage = (actual_price - expected_price) / expected_price
                    if abs(slippage) > self.alert_thresholds["slippage_warning"]:
                        severity = "critical" if abs(slippage) > self.alert_thresholds["slippage_critical"] else "warning"
                        alert = {
                            "type": "high_slippage",
                            "message": f"Order {order_id} high slippage: {slippage:.2%}",
                            "timestamp": datetime.now(),
                            "severity": severity
                        }
                        monitoring_data["alerts"].append(alert)
            
            logger.info(f"Order {order_id} filled - execution time: {execution_time:.1f}s, fill rate: {fill_rate:.1%}")
            
        except Exception as e:
            logger.error(f"Order filled handling error: {str(e)}")
    
    async def _handle_partial_fill(self, order_id: int, monitoring_data: Dict[str, Any],
                                 fill_data: Dict[str, Any]):
        """Handle partial fill event."""
        try:
            filled_quantity = fill_data.get("filled_quantity", 0)
            total_quantity = monitoring_data["total_quantity"]
            fill_rate = filled_quantity / total_quantity
            
            # Generate partial fill alert if fill rate is below threshold
            if fill_rate < self.alert_thresholds["partial_fill_warning"]:
                alert = {
                    "type": "slow_partial_fill",
                    "message": f"Order {order_id} partial fill rate: {fill_rate:.1%}",
                    "timestamp": datetime.now(),
                    "severity": "warning"
                }
                monitoring_data["alerts"].append(alert)
            
            logger.debug(f"Order {order_id} partial fill - rate: {fill_rate:.1%}")
            
        except Exception as e:
            logger.error(f"Partial fill handling error: {str(e)}")
    
    async def _handle_order_terminated(self, order_id: int, monitoring_data: Dict[str, Any],
                                     status: OrderStatus):
        """Handle order termination (cancelled, rejected, expired)."""
        try:
            execution_time = (datetime.now() - monitoring_data["start_time"]).total_seconds()
            
            termination_type = status.value
            
            if status == OrderStatus.CANCELLED:
                alert = {
                    "type": "order_cancelled",
                    "message": f"Order {order_id} cancelled after {execution_time:.1f} seconds",
                    "timestamp": datetime.now(),
                    "severity": "info"
                }
            elif status == OrderStatus.REJECTED:
                alert = {
                    "type": "order_rejected",
                    "message": f"Order {order_id} rejected after {execution_time:.1f} seconds",
                    "timestamp": datetime.now(),
                    "severity": "warning"
                }
            else:
                alert = {
                    "type": "order_expired",
                    "message": f"Order {order_id} expired after {execution_time:.1f} seconds",
                    "timestamp": datetime.now(),
                    "severity": "warning"
                }
            
            monitoring_data["alerts"].append(alert)
            logger.info(f"Order {order_id} {termination_type} - execution time: {execution_time:.1f}s")
            
        except Exception as e:
            logger.error(f"Order termination handling error: {str(e)}")
    
    async def _cancel_timeout(self, order_id: int):
        """Cancel timeout task for order."""
        try:
            monitoring_data = self.monitored_orders.get(order_id)
            if monitoring_data and "timeout_task" in monitoring_data:
                monitoring_data["timeout_task"].cancel()
                del monitoring_data["timeout_task"]
                
        except Exception as e:
            logger.error(f"Timeout cancellation error: {str(e)}")
    
    async def _check_and_generate_alerts(self, order_id: int, monitoring_data: Dict[str, Any]):
        """Check for conditions that should generate alerts."""
        try:
            # This would implement more sophisticated alerting logic
            # For now, just log the current state
            
            if monitoring_data.get("alerts"):
                logger.info(f"Generated {len(monitoring_data['alerts'])} alerts for order {order_id}")
                
        except Exception as e:
            logger.error(f"Alert generation error: {str(e)}")
    
    async def _trigger_callbacks(self, order_id: int, event: Dict[str, Any]):
        """Trigger registered callbacks for order events."""
        try:
            if order_id in self.execution_callbacks:
                for callback in self.execution_callbacks[order_id]:
                    try:
                        await callback(order_id, event)
                    except Exception as e:
                        logger.error(f"Callback execution error: {str(e)}")
                        
        except Exception as e:
            logger.error(f"Callback triggering error: {str(e)}")
    
    def _get_event_type(self, new_status: OrderStatus, old_status: str) -> ExecutionEventType:
        """Map order status to execution event type."""
        status_mapping = {
            OrderStatus.NEW: ExecutionEventType.ORDER_ACCEPTED,
            OrderStatus.PARTIALLY_FILLED: ExecutionEventType.ORDER_PARTIALLY_FILLED,
            OrderStatus.FILLED: ExecutionEventType.ORDER_FILLED,
            OrderStatus.CANCELLED: ExecutionEventType.ORDER_CANCELLED,
            OrderStatus.REJECTED: ExecutionEventType.ORDER_REJECTED,
            OrderStatus.EXPIRED: ExecutionEventType.ORDER_EXPIRED
        }
        
        return status_mapping.get(new_status, ExecutionEventType.EXECUTION_ERROR)
    
    def register_execution_callback(self, order_id: int, callback: Callable):
        """Register callback function for order execution events."""
        try:
            if order_id not in self.execution_callbacks:
                self.execution_callbacks[order_id] = []
            
            self.execution_callbacks[order_id].append(callback)
            
        except Exception as e:
            logger.error(f"Callback registration error: {str(e)}")
    
    def unregister_execution_callback(self, order_id: int, callback: Callable):
        """Unregister callback function."""
        try:
            if order_id in self.execution_callbacks:
                if callback in self.execution_callbacks[order_id]:
                    self.execution_callbacks[order_id].remove(callback)
                
                if not self.execution_callbacks[order_id]:
                    del self.execution_callbacks[order_id]
                    
        except Exception as e:
            logger.error(f"Callback unregistration error: {str(e)}")
    
    def close(self):
        """Cleanup resources."""
        try:
            # Stop background monitoring
            if self.is_monitoring:
                asyncio.create_task(self.stop_background_monitoring())
            
            # Close database connection
            self.db.close()
            
        except Exception as e:
            logger.error(f"ExecutionMonitor cleanup error: {str(e)}")
