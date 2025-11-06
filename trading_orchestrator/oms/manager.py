"""
Order Management System (OMS)
Handles order lifecycle, execution, and tracking across multiple brokers
"""

from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import uuid

from loguru import logger

from risk.manager import RiskManager, ViolationType


class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(Enum):
    """Time in force"""
    DAY = "day"
    GTC = "gtc"  # Good Till Cancel
    IOC = "ioc"  # Immediate Or Cancel
    FOK = "fok"  # Fill Or Kill


@dataclass
class Order:
    """Order data structure"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: Decimal = Decimal('0')
    filled_average_price: Optional[Decimal] = None
    remaining_quantity: Decimal = field(init=False)
    created_time: datetime = field(default_factory=datetime.utcnow)
    updated_time: datetime = field(default_factory=datetime.utcnow)
    broker_order_id: Optional[str] = None
    broker_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.remaining_quantity = self.quantity - self.filled_quantity


@dataclass
class Fill:
    """Trade execution fill"""
    fill_id: str
    order_id: str
    quantity: Decimal
    price: Decimal
    timestamp: datetime
    commission: Decimal = Decimal('0')
    metadata: Dict[str, Any] = field(default_factory=dict)


class OrderManager:
    """
    Order Management System
    
    Features:
    - Multi-broker order routing
    - Order lifecycle management
    - Risk check integration
    - Real-time status tracking
    - Performance metrics
    """
    
    def __init__(self, risk_manager: RiskManager):
        """
        Initialize order manager
        
        Args:
            risk_manager: Risk manager for order validation
        """
        self.risk_manager = risk_manager
        self.orders: Dict[str, Order] = {}
        self.fills: Dict[str, List[Fill]] = {}
        self.order_callbacks: Dict[str, List[Callable]] = {}
        self.brokers = {}
        
        # Performance tracking
        self.orders_submitted = 0
        self.orders_filled = 0
        self.orders_cancelled = 0
        self.total_volume = Decimal('0')
        self.average_fill_time = 0.0
        
        logger.info("Order Manager initialized")
    
    def register_broker(self, broker_name: str, broker_client):
        """Register a broker client for order execution"""
        self.brokers[broker_name] = broker_client
        logger.info(f"Registered broker: {broker_name}")
    
    async def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        order_type: str = OrderType.MARKET.value,
        price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None,
        time_in_force: str = TimeInForce.DAY.value,
        broker_name: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Submit a new order with risk checks
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Order quantity
            order_type: Order type
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            time_in_force: Time in force
            broker_name: Preferred broker (auto-select if None)
            metadata: Additional order metadata
            
        Returns:
            Order submission result
        """
        try:
            # Convert to proper types
            order_side = OrderSide(side.lower())
            order_type_enum = OrderType(order_type.lower())
            tif_enum = TimeInForce(time_in_force.lower())
            
            # Risk check
            current_price = price if order_type_enum == OrderType.LIMIT else Decimal('100')  # Default for market orders
            risk_check = await self.risk_manager.check_trade_risk(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=current_price,
                account_value=self.risk_manager.portfolio_value
            )
            
            if not risk_check.get('approved', False):
                logger.warning(f"Order rejected due to risk check: {risk_check}")
                return {
                    'success': False,
                    'order_id': None,
                    'reason': 'risk_check_failed',
                    'violation': risk_check.get('violation')
                }
            
            # Generate order ID
            order_id = f"ORD_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
            
            # Create order
            order = Order(
                order_id=order_id,
                symbol=symbol,
                side=order_side,
                order_type=order_type_enum,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                time_in_force=tif_enum,
                metadata=metadata or {}
            )
            
            # Select broker if not specified
            if not broker_name:
                broker_name = self._select_broker(symbol)
            
            if not broker_name or broker_name not in self.brokers:
                return {
                    'success': False,
                    'order_id': order_id,
                    'reason': 'no_broker_available'
                }
            
            # Submit to broker
            broker_result = await self._submit_to_broker(order, broker_name)
            
            if broker_result.get('success'):
                # Store order
                order.broker_order_id = broker_result.get('broker_order_id')
                order.broker_name = broker_name
                order.status = OrderStatus.SUBMITTED
                order.updated_time = datetime.utcnow()
                
                self.orders[order_id] = order
                self.fills[order_id] = []
                
                self.orders_submitted += 1
                
                logger.info(f"Order submitted: {order_id} {side} {quantity} {symbol}")
                
                return {
                    'success': True,
                    'order_id': order_id,
                    'broker_order_id': broker_result.get('broker_order_id'),
                    'status': order.status.value,
                    'estimated_cost': self._calculate_estimated_cost(order)
                }
            else:
                order.status = OrderStatus.REJECTED
                self.orders[order_id] = order
                
                logger.error(f"Order rejected by broker: {broker_result.get('reason')}")
                
                return {
                    'success': False,
                    'order_id': order_id,
                    'reason': broker_result.get('reason', 'broker_rejection')
                }
                
        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            return {
                'success': False,
                'order_id': None,
                'reason': str(e)
            }
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an existing order
        
        Args:
            order_id: Order to cancel
            
        Returns:
            Cancellation result
        """
        try:
            if order_id not in self.orders:
                return {
                    'success': False,
                    'reason': 'order_not_found'
                }
            
            order = self.orders[order_id]
            
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                return {
                    'success': False,
                    'reason': f'order_already_{order.status.value}'
                }
            
            # Cancel with broker
            if order.broker_name and order.broker_order_id:
                broker_result = await self._cancel_with_broker(order)
                
                if broker_result.get('success'):
                    order.status = OrderStatus.CANCELLED
                    order.updated_time = datetime.utcnow()
                    
                    self.orders_cancelled += 1
                    
                    logger.info(f"Order cancelled: {order_id}")
                    
                    return {
                        'success': True,
                        'order_id': order_id,
                        'status': order.status.value
                    }
                else:
                    return {
                        'success': False,
                        'reason': broker_result.get('reason', 'broker_cancellation_failed')
                    }
            else:
                # Cancel locally if no broker order
                order.status = OrderStatus.CANCELLED
                order.updated_time = datetime.utcnow()
                
                self.orders_cancelled += 1
                
                return {
                    'success': True,
                    'order_id': order_id,
                    'status': order.status.value
                }
                
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return {
                'success': False,
                'reason': str(e)
            }
    
    async def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get current order status"""
        order = self.orders.get(order_id)
        if not order:
            return None
        
        return {
            'order_id': order.order_id,
            'symbol': order.symbol,
            'side': order.side.value,
            'order_type': order.order_type.value,
            'quantity': float(order.quantity),
            'filled_quantity': float(order.filled_quantity),
            'remaining_quantity': float(order.remaining_quantity),
            'status': order.status.value,
            'created_time': order.created_time.isoformat(),
            'updated_time': order.updated_time.isoformat(),
            'broker_name': order.broker_name,
            'fills': [
                {
                    'fill_id': fill.fill_id,
                    'quantity': float(fill.quantity),
                    'price': float(fill.price),
                    'timestamp': fill.timestamp.isoformat(),
                    'commission': float(fill.commission)
                }
                for fill in self.fills.get(order_id, [])
            ]
        }
    
    async def get_all_orders(
        self,
        status_filter: Optional[List[str]] = None,
        symbol_filter: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get all orders with optional filtering"""
        orders = []
        
        for order in self.orders.values():
            # Apply filters
            if status_filter and order.status.value not in status_filter:
                continue
            
            if symbol_filter and order.symbol != symbol_filter:
                continue
            
            # Get order status
            order_dict = await self.get_order_status(order.order_id)
            if order_dict:
                orders.append(order_dict)
            
            if len(orders) >= limit:
                break
        
        return orders
    
    async def update_order_from_broker(
        self,
        broker_name: str,
        broker_order_id: str,
        status: str,
        filled_quantity: Decimal,
        filled_price: Decimal,
        fills_data: Optional[List[Dict]] = None
    ):
        """Update order status from broker callback"""
        # Find order by broker order ID
        order = None
        for o in self.orders.values():
            if o.broker_order_id == broker_order_id and o.broker_name == broker_name:
                order = o
                break
        
        if not order:
            logger.warning(f"Order not found for broker update: {broker_name}:{broker_order_id}")
            return
        
        try:
            # Update order status
            order.status = OrderStatus(status.lower())
            order.filled_quantity = filled_quantity
            order.remaining_quantity = order.quantity - filled_quantity
            
            if filled_price:
                order.filled_average_price = filled_price
            
            order.updated_time = datetime.utcnow()
            
            # Process fills
            if fills_data:
                for fill_data in fills_data:
                    fill = Fill(
                        fill_id=fill_data.get('fill_id', str(uuid.uuid4())),
                        order_id=order.order_id,
                        quantity=Decimal(str(fill_data.get('quantity', 0))),
                        price=Decimal(str(fill_data.get('price', 0))),
                        timestamp=datetime.utcnow(),
                        commission=Decimal(str(fill_data.get('commission', 0)))
                    )
                    
                    self.fills[order.order_id].append(fill)
                    
                    # Update risk manager with P&L
                    if order.side == OrderSide.BUY:
                        pnl_change = -(fill.quantity * fill.price)
                    else:
                        pnl_change = fill.quantity * fill.price
                    
                    await self.risk_manager.update_daily_pnl(pnl_change)
                    
                    # Update performance metrics
                    self.total_volume += fill.quantity * fill.price
                    
                    if order.status == OrderStatus.FILLED:
                        self.orders_filled += 1
                        self._update_fill_time_metrics(order.created_time)
            
            logger.info(f"Order updated: {order.order_id} -> {order.status.value}")
            
            # Notify callbacks
            await self._notify_order_callbacks(order)
            
        except Exception as e:
            logger.error(f"Error updating order from broker: {e}")
    
    def _select_broker(self, symbol: str) -> Optional[str]:
        """Select best broker for symbol"""
        # Simple broker selection logic - can be enhanced
        available_brokers = list(self.brokers.keys())
        
        if not available_brokers:
            return None
        
        # Prefer brokers that support the symbol type
        # For crypto, prefer Binance; for stocks, prefer Alpaca or IBKR
        if symbol.upper().endswith('USDT') or symbol.upper().endswith('BTC'):
            if 'binance' in available_brokers:
                return 'binance'
        
        # Default to first available broker
        return available_brokers[0]
    
    async def _submit_to_broker(self, order: Order, broker_name: str) -> Dict[str, Any]:
        """Submit order to specific broker"""
        broker = self.brokers.get(broker_name)
        if not broker:
            return {'success': False, 'reason': 'broker_not_found'}
        
        try:
            # Simplified broker submission - actual implementation would call real API
            logger.info(f"Submitting order to {broker_name}: {order.symbol} {order.side.value} {order.quantity}")
            
            # Mock response
            return {
                'success': True,
                'broker_order_id': f"{broker_name.upper()}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            }
            
        except Exception as e:
            logger.error(f"Error submitting to broker {broker_name}: {e}")
            return {'success': False, 'reason': str(e)}
    
    async def _cancel_with_broker(self, order: Order) -> Dict[str, Any]:
        """Cancel order with broker"""
        broker = self.brokers.get(order.broker_name)
        if not broker:
            return {'success': False, 'reason': 'broker_not_found'}
        
        try:
            # Mock broker cancellation
            logger.info(f"Cancelling order {order.order_id} with {order.broker_name}")
            return {'success': True}
            
        except Exception as e:
            logger.error(f"Error cancelling with broker: {e}")
            return {'success': False, 'reason': str(e)}
    
    def _calculate_estimated_cost(self, order: Order) -> float:
        """Calculate estimated order cost"""
        if order.price:
            return float(order.quantity * order.price)
        return 0.0
    
    def _update_fill_time_metrics(self, order_time: datetime):
        """Update average fill time metrics"""
        fill_time = (datetime.utcnow() - order_time).total_seconds()
        
        if self.orders_filled == 1:
            self.average_fill_time = fill_time
        else:
            # Running average
            self.average_fill_time = (
                (self.average_fill_time * (self.orders_filled - 1) + fill_time) / self.orders_filled
            )
    
    async def _notify_order_callbacks(self, order: Order):
        """Notify registered callbacks for order events"""
        callbacks = self.order_callbacks.get(order.order_id, [])
        for callback in callbacks:
            try:
                await callback(order)
            except Exception as e:
                logger.error(f"Error in order callback: {e}")
    
    def register_order_callback(self, order_id: str, callback: Callable):
        """Register callback for order events"""
        if order_id not in self.order_callbacks:
            self.order_callbacks[order_id] = []
        self.order_callbacks[order_id].append(callback)
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get order manager performance metrics"""
        total_orders = len(self.orders)
        fill_rate = self.orders_filled / total_orders if total_orders > 0 else 0
        cancellation_rate = self.orders_cancelled / total_orders if total_orders > 0 else 0
        
        return {
            'orders': {
                'total': total_orders,
                'submitted': self.orders_submitted,
                'filled': self.orders_filled,
                'cancelled': self.orders_cancelled,
                'fill_rate': fill_rate,
                'cancellation_rate': cancellation_rate
            },
            'volume': {
                'total_volume': float(self.total_volume),
                'average_fill_time_seconds': self.average_fill_time
            },
            'brokers': {
                'registered': list(self.brokers.keys()),
                'active': len(self.brokers)
            }
        }
    
    async def get_pending_orders(self) -> List[Dict[str, Any]]:
        """Get all pending orders"""
        pending_statuses = [OrderStatus.PENDING.value, OrderStatus.SUBMITTED.value, OrderStatus.PARTIALLY_FILLED.value]
        return await self.get_all_orders(status_filter=pending_statuses)


# Example usage and testing
if __name__ == "__main__":
    async def test_order_manager():
        from risk.manager import RiskManager
        
        # Create managers
        risk_manager = RiskManager()
        order_manager = OrderManager(risk_manager)
        
        # Mock broker
        class MockBroker:
            async def submit_order(self, order):
                return {'success': True, 'broker_order_id': 'MOCK_123'}
        
        order_manager.register_broker('mock', MockBroker())
        
        # Submit test order
        result = await order_manager.submit_order(
            symbol='AAPL',
            side='buy',
            quantity=Decimal('10'),
            order_type='limit',
            price=Decimal('150.00'),
            broker_name='mock'
        )
        
        print("Order submission result:", result)
        
        # Get order status
        if result.get('success'):
            status = await order_manager.get_order_status(result['order_id'])
            print("Order status:", status)
    
    asyncio.run(test_order_manager())