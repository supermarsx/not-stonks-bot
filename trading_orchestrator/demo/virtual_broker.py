"""
Virtual Broker - Simulates trading operations without real money
"""

import asyncio
import random
import uuid
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from loguru import logger

from .demo_mode_manager import DemoModeManager, demo_manager

# Import from base brokers (will be available when used in the full system)
try:
    from ..brokers.base import BaseBroker, BrokerConfig, AccountInfo, PositionInfo, OrderInfo, MarketDataPoint
except ImportError:
    # Fallback for standalone usage
    from typing import Any
    from abc import ABC, abstractmethod
    
    class BaseBroker(ABC):
        pass
    
    BrokerConfig = Any
    AccountInfo = Any  
    PositionInfo = Any
    OrderInfo = Any
    MarketDataPoint = Any


class OrderStatus(Enum):
    """Order status for virtual trading"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class FillType(Enum):
    """Order fill type simulation"""
    IMMEDIATE = "immediate"
    DELAYED = "delayed"
    PARTIAL = "partial"
    REJECTED = "rejected"


@dataclass
class VirtualOrder:
    """Virtual order representation"""
    order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "day"
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: Optional[float] = None
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    commission: float = 0.0
    slippage: float = 0.0
    metadata: Dict[str, Any] = None


@dataclass
class VirtualPosition:
    """Virtual position representation"""
    symbol: str
    quantity: float
    avg_entry_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    commission_paid: float = 0.0
    last_updated: Optional[datetime] = None


class VirtualBroker(BaseBroker):
    """
    Virtual broker implementation for simulation trading
    
    Simulates real broker operations with:
    - Realistic order execution
    - Slippage and commission simulation
    - Market impact modeling
    - Order book simulation
    """
    
    def __init__(self, config: BrokerConfig, demo_manager: DemoModeManager):
        super().__init__(config)
        self.demo_manager = demo_manager
        self.orders: Dict[str, VirtualOrder] = {}
        self.positions: Dict[str, VirtualPosition] = {}
        self.order_book: Dict[str, List[Tuple[float, float]]] = {}  # symbol -> [(price, quantity)]
        self.market_prices: Dict[str, float] = {}
        self.is_connected = True  # Virtual broker is always "connected"
        
    async def connect(self) -> bool:
        """Simulate connection to virtual broker"""
        try:
            # Initialize virtual market data
            await self._initialize_market_data()
            await self._load_historical_data()
            
            self.is_connected = True
            logger.info(f"Virtual broker {self.broker_name} connected")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect virtual broker: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Simulate disconnection"""
        try:
            self.is_connected = False
            logger.info(f"Virtual broker {self.broker_name} disconnected")
            return True
            
        except Exception as e:
            logger.error(f"Failed to disconnect virtual broker: {e}")
            return False
    
    async def is_connection_alive(self) -> bool:
        """Check if virtual connection is alive"""
        return self.is_connected
    
    async def get_account(self) -> AccountInfo:
        """Get virtual account information"""
        try:
            total_portfolio_value = await self._calculate_portfolio_value()
            available_cash = await self._calculate_available_cash()
            margin_used = await self._calculate_margin_used()
            
            return AccountInfo(
                account_id=f"demo_{self.broker_name}_account",
                broker_name=self.broker_name,
                currency="USD",
                balance=total_portfolio_value,
                available_balance=available_cash,
                equity=total_portfolio_value,
                buying_power=available_cash * 4,  # 4x leverage simulation
                margin_used=margin_used,
                margin_available=total_portfolio_value * 0.8 - margin_used,
                is_pattern_day_trader=True,
                metadata={
                    "virtual": True,
                    "session_id": self.demo_manager.get_session_id()
                }
            )
            
        except Exception as e:
            logger.error(f"Error getting virtual account info: {e}")
            raise
    
    async def get_positions(self) -> List[PositionInfo]:
        """Get all virtual positions"""
        try:
            positions = []
            current_prices = await self._get_current_prices()
            
            for symbol, position in self.positions.items():
                current_price = current_prices.get(symbol, position.avg_entry_price)
                market_value = position.quantity * current_price
                unrealized_pnl = (current_price - position.avg_entry_price) * position.quantity
                
                position_info = PositionInfo(
                    symbol=symbol,
                    broker_name=self.broker_name,
                    side="long" if position.quantity > 0 else "short",
                    quantity=abs(position.quantity),
                    avg_entry_price=position.avg_entry_price,
                    current_price=current_price,
                    market_value=market_value,
                    unrealized_pnl=unrealized_pnl,
                    unrealized_pnl_percent=(unrealized_pnl / (position.avg_entry_price * abs(position.quantity))) * 100 if position.quantity > 0 else 0,
                    cost_basis=position.avg_entry_price * abs(position.quantity),
                    metadata={
                        "virtual": True,
                        "commission_paid": position.commission_paid,
                        "realized_pnl": position.realized_pnl
                    }
                )
                positions.append(position_info)
            
            return positions
            
        except Exception as e:
            logger.error(f"Error getting virtual positions: {e}")
            raise
    
    async def get_position(self, symbol: str) -> Optional[PositionInfo]:
        """Get position for specific symbol"""
        try:
            positions = await self.get_positions()
            for position in positions:
                if position.symbol == symbol:
                    return position
            return None
            
        except Exception as e:
            logger.error(f"Error getting virtual position for {symbol}: {e}")
            raise
    
    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "day",
        extended_hours: bool = False,
        **kwargs
    ) -> OrderInfo:
        """Place virtual order with realistic simulation"""
        try:
            # Generate order ID
            order_id = str(uuid.uuid4())
            
            # Create virtual order
            virtual_order = VirtualOrder(
                order_id=order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                limit_price=limit_price,
                stop_price=stop_price,
                time_in_force=time_in_force,
                status=OrderStatus.SUBMITTED,
                submitted_at=datetime.now(),
                metadata={
                    "virtual": True,
                    "extended_hours": extended_hours,
                    **kwargs
                }
            )
            
            # Store order
            self.orders[order_id] = virtual_order
            
            # Simulate order execution
            execution_result = await self._simulate_order_execution(virtual_order)
            
            if execution_result.success:
                # Update order status
                virtual_order.status = OrderStatus.FILLED
                virtual_order.filled_quantity = quantity
                virtual_order.filled_at = datetime.now()
                virtual_order.avg_fill_price = execution_result.fill_price
                virtual_order.commission = execution_result.commission
                virtual_order.slippage = execution_result.slippage
                
                # Update positions
                await self._update_positions(virtual_order)
                
                logger.info(f"Virtual order filled: {symbol} {side} {quantity} @ {execution_result.fill_price}")
            else:
                virtual_order.status = OrderStatus.REJECTED
                virtual_order.metadata["rejection_reason"] = execution_result.reason
                logger.warning(f"Virtual order rejected: {symbol} - {execution_result.reason}")
            
            # Convert to standard OrderInfo
            return await self._convert_to_order_info(virtual_order)
            
        except Exception as e:
            logger.error(f"Error placing virtual order: {e}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel virtual order"""
        try:
            if order_id not in self.orders:
                logger.warning(f"Virtual order not found: {order_id}")
                return False
            
            order = self.orders[order_id]
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
                logger.warning(f"Cannot cancel order with status: {order.status}")
                return False
            
            # Simulate cancellation
            cancel_success = await self._simulate_order_cancellation(order)
            
            if cancel_success:
                order.status = OrderStatus.CANCELLED
                logger.info(f"Virtual order cancelled: {order_id}")
                return True
            else:
                logger.warning(f"Virtual order cancellation failed: {order_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error cancelling virtual order: {e}")
            return False
    
    async def get_orders(self, status: Optional[str] = None) -> List[OrderInfo]:
        """Get virtual orders with optional status filter"""
        try:
            orders_info = []
            
            for order in self.orders.values():
                if status and order.status.value != status:
                    continue
                
                order_info = await self._convert_to_order_info(order)
                orders_info.append(order_info)
            
            return orders_info
            
        except Exception as e:
            logger.error(f"Error getting virtual orders: {e}")
            raise
    
    async def get_order(self, order_id: str) -> Optional[OrderInfo]:
        """Get specific virtual order"""
        try:
            if order_id not in self.orders:
                return None
            
            return await self._convert_to_order_info(self.orders[order_id])
            
        except Exception as e:
            logger.error(f"Error getting virtual order {order_id}: {e}")
            return None
    
    async def get_market_data(
        self,
        symbol: str,
        timeframe: str = "1d",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 100
    ) -> List[MarketDataPoint]:
        """Get virtual historical market data"""
        try:
            # Generate synthetic historical data
            data_points = await self._generate_synthetic_data(symbol, timeframe, limit)
            
            return data_points
            
        except Exception as e:
            logger.error(f"Error getting virtual market data for {symbol}: {e}")
            raise
    
    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get virtual real-time quote"""
        try:
            current_price = await self._get_current_price(symbol)
            spread = current_price * 0.001  # 0.1% spread simulation
            
            quote = {
                "symbol": symbol,
                "bid": current_price - spread / 2,
                "ask": current_price + spread / 2,
                "last": current_price,
                "volume": await self._generate_volume(symbol),
                "timestamp": datetime.now().isoformat(),
                "virtual": True
            }
            
            return quote
            
        except Exception as e:
            logger.error(f"Error getting virtual quote for {symbol}: {e}")
            raise
    
    # Private methods for simulation
    
    async def _initialize_market_data(self):
        """Initialize virtual market data structures"""
        # Common symbols for simulation
        common_symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX",
            "BTC-USD", "ETH-USD", "EURUSD=X", "USDJPY=X", "SPY", "QQQ", "IWM"
        ]
        
        # Initialize market prices with realistic values
        for symbol in common_symbols:
            if "USD" in symbol or symbol.endswith("-USD"):
                self.market_prices[symbol] = random.uniform(1000, 100000)  # Crypto prices
            elif "X" in symbol:  # Forex
                self.market_prices[symbol] = random.uniform(0.5, 2.0)
            else:  # Stocks
                self.market_prices[symbol] = random.uniform(50, 500)
    
    async def _load_historical_data(self):
        """Load or generate historical data"""
        # This would typically load real historical data
        # For demo, we generate synthetic data
        pass
    
    async def _calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        total = self.demo_manager.config.demo_account_balance  # Starting cash
        
        # Add position values
        current_prices = await self._get_current_prices()
        for symbol, position in self.positions.items():
            current_price = current_prices.get(symbol, position.avg_entry_price)
            position_value = position.quantity * current_price
            total += position_value
        
        return total
    
    async def _calculate_available_cash(self) -> float:
        """Calculate available cash"""
        # Starting cash minus commission costs and position costs
        total_commission = sum(pos.commission_paid for pos in self.positions.values())
        total_position_cost = sum(
            pos.quantity * pos.avg_entry_price for pos in self.positions.values()
        )
        
        return self.demo_manager.config.demo_account_balance - total_commission - total_position_cost
    
    async def _calculate_margin_used(self) -> float:
        """Calculate margin used"""
        # Simplified margin calculation
        return sum(
            pos.quantity * pos.avg_entry_price * 0.5 for pos in self.positions.values()
        )
    
    async def _get_current_prices(self) -> Dict[str, float]:
        """Get current market prices for all symbols"""
        # Add some realistic price movement
        updated_prices = {}
        for symbol, base_price in self.market_prices.items():
            # Simulate price movement
            volatility = 0.02  # 2% volatility
            price_change = random.gauss(0, volatility)
            new_price = base_price * (1 + price_change)
            updated_prices[symbol] = new_price
            self.market_prices[symbol] = new_price
        
        return updated_prices
    
    async def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        if symbol not in self.market_prices:
            # Initialize unknown symbol
            self.market_prices[symbol] = random.uniform(10, 200)
        
        # Add small random movement
        current = self.market_prices[symbol]
        change = random.gauss(0, 0.01)  # 1% standard deviation
        new_price = current * (1 + change)
        self.market_prices[symbol] = new_price
        
        return new_price
    
    async def _generate_volume(self, symbol: str) -> float:
        """Generate realistic volume"""
        base_volume = 1000000 if not symbol.endswith("USD") else 100
        return random.uniform(base_volume * 0.8, base_volume * 1.2)
    
    async def _simulate_order_execution(self, order: VirtualOrder) -> "ExecutionResult":
        """Simulate realistic order execution"""
        try:
            current_price = await self._get_current_price(order.symbol)
            
            # Calculate execution price with slippage
            fill_price = await self._calculate_fill_price(order, current_price)
            
            # Calculate commission
            commission = await self._calculate_commission(order, fill_price)
            
            # Calculate slippage
            slippage = abs(fill_price - current_price) * order.quantity
            
            # Validate execution
            if not await self._validate_order_execution(order, fill_price):
                return ExecutionResult(
                    success=False,
                    reason="Order validation failed",
                    fill_price=None,
                    commission=0,
                    slippage=0
                )
            
            return ExecutionResult(
                success=True,
                reason=None,
                fill_price=fill_price,
                commission=commission,
                slippage=slippage
            )
            
        except Exception as e:
            logger.error(f"Error simulating order execution: {e}")
            return ExecutionResult(
                success=False,
                reason=f"Simulation error: {str(e)}",
                fill_price=None,
                commission=0,
                slippage=0
            )
    
    async def _calculate_fill_price(self, order: VirtualOrder, market_price: float) -> float:
        """Calculate fill price with slippage"""
        if not self.demo_manager.config.realistic_slippage:
            return market_price
        
        # Simulate slippage based on order size and market conditions
        base_slippage = self.demo_manager.config.slippage_rate
        order_impact = min(order.quantity / 10000, 0.01)  # 1% max impact for large orders
        
        # Add random component
        random_factor = random.uniform(-1, 1)
        total_slippage = (base_slippage + order_impact) * random_factor
        
        fill_price = market_price * (1 + total_slippage)
        
        # Ensure buy orders are at ask, sell orders at bid (simplified)
        spread = market_price * 0.001  # 0.1% spread
        if order.side.lower() == "buy":
            fill_price = max(fill_price, market_price + spread/2)
        else:
            fill_price = min(fill_price, market_price - spread/2)
        
        return fill_price
    
    async def _calculate_commission(self, order: VirtualOrder, fill_price: float) -> float:
        """Calculate commission cost"""
        trade_value = order.quantity * fill_price
        commission = trade_value * self.demo_manager.config.commission_rate
        return commission
    
    async def _validate_order_execution(self, order: VirtualOrder, fill_price: float) -> bool:
        """Validate order execution"""
        # Check price limits for limit orders
        if order.order_type.lower() == "limit" and order.limit_price:
            if order.side.lower() == "buy" and fill_price > order.limit_price:
                return False
            elif order.side.lower() == "sell" and fill_price < order.limit_price:
                return False
        
        # Check available cash for buy orders
        if order.side.lower() == "buy":
            required_cash = order.quantity * fill_price + await self._calculate_commission(order, fill_price)
            available_cash = await self._calculate_available_cash()
            if required_cash > available_cash:
                return False
        
        return True
    
    async def _simulate_order_cancellation(self, order: VirtualOrder) -> bool:
        """Simulate order cancellation"""
        # Most orders can be cancelled before execution
        return order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]
    
    async def _update_positions(self, order: VirtualOrder):
        """Update virtual positions after order execution"""
        symbol = order.symbol
        quantity = order.quantity if order.side.lower() == "buy" else -order.quantity
        fill_price = order.avg_fill_price or 0
        
        if symbol in self.positions:
            # Update existing position
            current = self.positions[symbol]
            new_quantity = current.quantity + quantity
            
            if new_quantity == 0:
                # Position closed
                realized_pnl = (fill_price - current.avg_entry_price) * current.quantity
                current.realized_pnl += realized_pnl
                del self.positions[symbol]
            elif current.quantity * quantity > 0:
                # Same direction, average the price
                total_cost = current.avg_entry_price * current.quantity + fill_price * quantity
                new_avg_price = total_cost / new_quantity
                current.avg_entry_price = new_avg_price
                current.quantity = new_quantity
            else:
                # Reducing or reversing position
                if abs(quantity) < abs(current.quantity):
                    # Partial close
                    realized_pnl = (fill_price - current.avg_entry_price) * quantity
                    current.realized_pnl += realized_pnl
                    current.quantity = new_quantity
                else:
                    # Full close with possible reverse
                    remaining_quantity = quantity + current.quantity
                    if remaining_quantity != 0:
                        # Reverse position
                        realized_pnl = (fill_price - current.avg_entry_price) * current.quantity
                        current.realized_pnl += realized_pnl
                        current.avg_entry_price = fill_price
                        current.quantity = remaining_quantity
                    else:
                        # Full close
                        realized_pnl = (fill_price - current.avg_entry_price) * current.quantity
                        current.realized_pnl += realized_pnl
                        del self.positions[symbol]
            
            current.commission_paid += order.commission
        else:
            # New position
            if quantity != 0:
                self.positions[symbol] = VirtualPosition(
                    symbol=symbol,
                    quantity=quantity,
                    avg_entry_price=fill_price,
                    commission_paid=order.commission,
                    last_updated=datetime.now()
                )
    
    async def _generate_synthetic_data(self, symbol: str, timeframe: str, limit: int) -> List[MarketDataPoint]:
        """Generate synthetic market data"""
        try:
            current_price = await self._get_current_price(symbol)
            data_points = []
            
            # Generate OHLCV data
            for i in range(limit):
                timestamp = datetime.now() - timedelta(days=limit-i)
                
                # Generate realistic OHLCV
                open_price = current_price * (1 + random.gauss(0, 0.01))
                high_price = open_price * (1 + abs(random.gauss(0, 0.02)))
                low_price = open_price * (1 - abs(random.gauss(0, 0.02)))
                close_price = open_price * (1 + random.gauss(0, 0.015))
                volume = await self._generate_volume(symbol)
                
                # Ensure OHLC consistency
                high_price = max(open_price, close_price, high_price)
                low_price = min(open_price, close_price, low_price)
                
                data_point = MarketDataPoint(
                    symbol=symbol,
                    broker_name=self.broker_name,
                    timestamp=timestamp,
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume,
                    timeframe=timeframe,
                    metadata={"virtual": True, "synthetic": True}
                )
                
                data_points.append(data_point)
            
            return data_points
            
        except Exception as e:
            logger.error(f"Error generating synthetic data for {symbol}: {e}")
            return []
    
    async def _convert_to_order_info(self, virtual_order: VirtualOrder) -> OrderInfo:
        """Convert virtual order to standard order info"""
        return OrderInfo(
            order_id=virtual_order.order_id,
            broker_name=self.broker_name,
            symbol=virtual_order.symbol,
            order_type=virtual_order.order_type,
            side=virtual_order.side,
            quantity=virtual_order.quantity,
            filled_quantity=virtual_order.filled_quantity,
            status=virtual_order.status.value,
            limit_price=virtual_order.limit_price,
            stop_price=virtual_order.stop_price,
            avg_fill_price=virtual_order.avg_fill_price,
            time_in_force=virtual_order.time_in_force,
            extended_hours=virtual_order.metadata.get("extended_hours", False) if virtual_order.metadata else False,
            submitted_at=virtual_order.submitted_at,
            filled_at=virtual_order.filled_at,
            metadata={
                "virtual": True,
                "commission": virtual_order.commission,
                "slippage": virtual_order.slippage,
                **virtual_order.metadata
            } if virtual_order.metadata else {"virtual": True}
        )


@dataclass
class ExecutionResult:
    """Result of order execution simulation"""
    success: bool
    reason: Optional[str]
    fill_price: Optional[float]
    commission: float
    slippage: float


# Factory function for creating virtual brokers
async def create_virtual_broker(broker_name: str, demo_manager: DemoModeManager = None) -> VirtualBroker:
    """Create a virtual broker instance"""
    if demo_manager is None:
        demo_manager = await demo_manager()
    
    config = BrokerConfig(
        broker_name=broker_name,
        is_paper=True  # Always paper trading
    )
    
    return VirtualBroker(config, demo_manager)


if __name__ == "__main__":
    # Example usage
    async def main():
        # Get demo manager
        manager = await get_demo_manager()
        await manager.initialize()
        await manager.enable_demo_mode()
        
        # Create virtual broker
        broker = await create_virtual_broker("demo_alpaca", manager)
        await broker.connect()
        
        # Get account info
        account = await broker.get_account()
        print(f"Virtual account: {account}")
        
        # Place a test order
        order = await broker.place_order(
            symbol="AAPL",
            side="buy",
            order_type="market",
            quantity=10
        )
        print(f"Order placed: {order}")
        
        # Get positions
        positions = await broker.get_positions()
        print(f"Positions: {positions}")
        
        await broker.disconnect()
    
    asyncio.run(main())
