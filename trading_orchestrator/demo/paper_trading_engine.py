"""
Paper Trading Engine - Realistic simulation of order execution and market impact
"""

import asyncio
import random
import math
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import numpy as np

from loguru import logger

from .virtual_broker import VirtualBroker, ExecutionResult
from .demo_mode_manager import DemoModeManager, demo_manager

# Import from base brokers (will be available when used in the full system)
try:
    from ..brokers.base import OrderInfo, BrokerConfig
except ImportError:
    from typing import Any
    OrderInfo = Any
    BrokerConfig = Any


class ExecutionAlgorithm(Enum):
    """Order execution algorithms"""
    TWAP = "twap"  # Time-weighted average price
    VWAP = "vwap"  # Volume-weighted average price
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"
    MARKET_ON_CLOSE = "market_on_close"
    PARTICIPATION = "participation"
    AGGRESSIVE = "aggressive"  # Immediate execution
    PASSIVE = "passive"  # Patient execution


class MarketCondition(Enum):
    """Market condition simulation"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    CRASH = "crash"
    RALLY = "rally"


@dataclass
class OrderBookLevel:
    """Order book level representation"""
    price: float
    quantity: float
    orders_count: int = 1


@dataclass
class MarketImpact:
    """Market impact parameters"""
    temporary_impact: float
    permanent_impact: float
    resilience_time: float  # seconds
    recovery_rate: float


@dataclass
class ExecutionPlan:
    """Detailed execution plan for an order"""
    algorithm: ExecutionAlgorithm
    start_time: datetime
    end_time: datetime
    target_price: float
    expected_cost: float
    max_slippage: float
    slices: List["ExecutionSlice"]


@dataclass
class ExecutionSlice:
    """Individual execution slice"""
    quantity: float
    target_time: datetime
    target_price: float
    price_range: Tuple[float, float]
    urgency: float  # 0-1, higher means more aggressive


class OrderBook:
    """Virtual order book for simulation"""
    
    def __init__(self, symbol: str, mid_price: float):
        self.symbol = symbol
        self.mid_price = mid_price
        self.bids: List[OrderBookLevel] = []
        self.asks: List[OrderBookLevel] = []
        self.last_update = datetime.now()
        self._initialize_order_book()
    
    def _initialize_order_book(self):
        """Initialize realistic order book"""
        spread = self.mid_price * 0.001  # 0.1% spread
        
        # Create bid levels
        for i in range(10):
            price = self.mid_price - spread/2 - (i * spread * 0.1)
            quantity = random.uniform(100, 2000)
            self.bids.append(OrderBookLevel(price=price, quantity=quantity))
        
        # Create ask levels
        for i in range(10):
            price = self.mid_price + spread/2 + (i * spread * 0.1)
            quantity = random.uniform(100, 2000)
            self.asks.append(OrderBookLevel(price=price, quantity=quantity))
        
        # Sort by price (bids descending, asks ascending)
        self.bids.sort(key=lambda x: x.price, reverse=True)
        self.asks.sort(key=lambda x: x.price)
    
    def update_market(self, new_mid_price: float):
        """Update order book with new market price"""
        price_change = new_mid_price - self.mid_price
        self.mid_price = new_mid_price
        self.last_update = datetime.now()
        
        # Update all levels by the price change
        for level in self.bids:
            level.price += price_change
        
        for level in self.asks:
            level.price += price_change
    
    def get_best_bid(self) -> Optional[float]:
        """Get best bid price"""
        return self.bids[0].price if self.bids else None
    
    def get_best_ask(self) -> Optional[float]:
        """Get best ask price"""
        return self.asks[0].price if self.asks else None
    
    def get_market_depth(self, side: str, levels: int = 5) -> List[Tuple[float, float]]:
        """Get market depth for a side"""
        if side.lower() == "bid":
            return [(level.price, level.quantity) for level in self.bids[:levels]]
        else:
            return [(level.price, level.quantity) for level in self.asks[:levels]]
    
    def simulate_market_impact(self, quantity: float, side: str) -> Tuple[float, float]:
        """Simulate market impact for a trade"""
        if side.lower() == "buy":
            levels = self.asks
        else:
            levels = self.bids
        
        total_quantity = 0
        total_cost = 0
        remaining_quantity = quantity
        
        for level in levels:
            if remaining_quantity <= 0:
                break
            
            trade_quantity = min(remaining_quantity, level.quantity)
            total_cost += trade_quantity * level.price
            total_quantity += trade_quantity
            remaining_quantity -= trade_quantity
        
        # Calculate average execution price
        if total_quantity > 0:
            avg_price = total_cost / total_quantity
            slippage = avg_price - self.mid_price if side.lower() == "buy" else self.mid_price - avg_price
            return avg_price, slippage
        
        return self.mid_price, 0


class PaperTradingEngine:
    """
    Paper trading engine with realistic execution
    
    Provides sophisticated order execution simulation including:
    - Order book simulation
    - Market impact modeling
    - Multiple execution algorithms
    - Realistic slippage and timing
    """
    
    def __init__(self, demo_manager: DemoModeManager):
        self.demo_manager = demo_manager
        self.order_books: Dict[str, OrderBook] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.market_conditions: Dict[str, MarketCondition] = {}
        self.is_running = False
        
        # Market impact parameters (simplified Almgren-Chriss model)
        self.impact_model_params = {
            "temporary_impact_coef": 0.002,  # 0.2% per sqrt(volume)
            "permanent_impact_coef": 0.0001,  # 0.01% per volume ratio
            "resilience_time": 30,  # seconds
            "recovery_rate": 0.1  # per second
        }
    
    async def start_engine(self):
        """Start the paper trading engine"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start background tasks
        asyncio.create_task(self._market_simulation_loop())
        asyncio.create_task(self._order_book_updates_loop())
        
        logger.info("Paper trading engine started")
    
    async def stop_engine(self):
        """Stop the paper trading engine"""
        if not self.is_running:
            return
        
        self.is_running = False
        logger.info("Paper trading engine stopped")
    
    async def execute_order(
        self,
        broker: VirtualBroker,
        order: OrderInfo,
        algorithm: ExecutionAlgorithm = ExecutionAlgorithm.AGGRESSIVE
    ) -> ExecutionResult:
        """Execute order with realistic simulation"""
        try:
            # Ensure engine is running
            if not self.is_running:
                await self.start_engine()
            
            # Create execution plan
            execution_plan = await self._create_execution_plan(order, algorithm)
            
            # Execute the plan
            execution_result = await self._execute_plan(order, execution_plan)
            
            # Record execution history
            await self._record_execution(order, execution_result, execution_plan)
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Error executing order: {e}")
            return ExecutionResult(
                success=False,
                reason=f"Execution error: {str(e)}",
                fill_price=None,
                commission=0,
                slippage=0
            )
    
    async def get_market_impact(
        self,
        symbol: str,
        quantity: float,
        side: str
    ) -> MarketImpact:
        """Calculate market impact for a trade"""
        try:
            # Get order book
            order_book = await self._get_or_create_order_book(symbol)
            
            # Calculate impact using simplified model
            volume_ratio = quantity / 10000  # Normalize by average daily volume
            
            temporary_impact = (
                self.impact_model_params["temporary_impact_coef"] *
                math.sqrt(volume_ratio)
            )
            
            permanent_impact = (
                self.impact_model_params["permanent_impact_coef"] *
                volume_ratio
            )
            
            return MarketImpact(
                temporary_impact=temporary_impact,
                permanent_impact=permanent_impact,
                resilience_time=self.impact_model_params["resilience_time"],
                recovery_rate=self.impact_model_params["recovery_rate"]
            )
            
        except Exception as e:
            logger.error(f"Error calculating market impact: {e}")
            return MarketImpact(0, 0, 30, 0.1)
    
    async def simulate_order_book(self, symbol: str) -> Optional[OrderBook]:
        """Get or create order book for symbol"""
        return await self._get_or_create_order_book(symbol)
    
    async def get_execution_metrics(self) -> Dict[str, Any]:
        """Get execution performance metrics"""
        if not self.execution_history:
            return {}
        
        total_trades = len(self.execution_history)
        total_slippage = sum(trade.get("slippage", 0) for trade in self.execution_history)
        avg_slippage = total_slippage / total_trades if total_trades > 0 else 0
        
        total_commission = sum(trade.get("commission", 0) for trade in self.execution_history)
        avg_commission = total_commission / total_trades if total_trades > 0 else 0
        
        return {
            "total_trades": total_trades,
            "total_slippage": total_slippage,
            "avg_slippage": avg_slippage,
            "total_commission": total_commission,
            "avg_commission": avg_commission,
            "implementation_shortfall": avg_slippage + avg_commission
        }
    
    # Private methods
    
    async def _create_execution_plan(
        self,
        order: OrderInfo,
        algorithm: ExecutionAlgorithm
    ) -> ExecutionPlan:
        """Create execution plan for an order"""
        try:
            start_time = datetime.now()
            
            # Determine execution duration based on algorithm
            if algorithm == ExecutionAlgorithm.TWAP:
                duration = min(60 * 5, 300)  # 5 minutes max
            elif algorithm == ExecutionAlgorithm.VWAP:
                duration = min(60 * 10, 600)  # 10 minutes max
            elif algorithm == ExecutionAlgorithm.IMPLEMENTATION_SHORTFALL:
                duration = min(60 * 2, 120)  # 2 minutes max
            else:
                duration = 30  # Default 30 seconds
            
            end_time = start_time + timedelta(seconds=duration)
            
            # Get current market price
            order_book = await self._get_or_create_order_book(order.symbol)
            target_price = order_book.mid_price
            
            # Calculate expected costs
            market_impact = await self.get_market_impact(order.symbol, order.quantity, order.side)
            expected_slippage = market_impact.temporary_impact * order.limit_price if order.limit_price else target_price
            expected_commission = order.quantity * target_price * self.demo_manager.config.commission_rate
            expected_cost = expected_slippage + expected_commission
            
            # Create execution slices
            slices = await self._create_execution_slices(order, algorithm, start_time, end_time)
            
            return ExecutionPlan(
                algorithm=algorithm,
                start_time=start_time,
                end_time=end_time,
                target_price=target_price,
                expected_cost=expected_cost,
                max_slippage=expected_slippage * 2,  # 2x expected slippage
                slices=slices
            )
            
        except Exception as e:
            logger.error(f"Error creating execution plan: {e}")
            raise
    
    async def _create_execution_slices(
        self,
        order: OrderInfo,
        algorithm: ExecutionAlgorithm,
        start_time: datetime,
        end_time: datetime
    ) -> List[ExecutionSlice]:
        """Create execution slices for the order"""
        slices = []
        total_duration = (end_time - start_time).total_seconds()
        slice_count = max(1, int(total_duration / 30))  # 30-second slices
        
        if algorithm == ExecutionAlgorithm.TWAP:
            # Equal time distribution
            quantity_per_slice = order.quantity / slice_count
            for i in range(slice_count):
                slice_start = start_time + timedelta(seconds=i * total_duration / slice_count)
                slices.append(ExecutionSlice(
                    quantity=quantity_per_slice,
                    target_time=slice_start,
                    target_price=order.limit_price or 0,  # Will be updated during execution
                    price_range=(0, float('inf')),
                    urgency=0.5  # Medium urgency
                ))
        
        elif algorithm == ExecutionAlgorithm.VWAP:
            # Volume-weighted (simplified: front-loaded)
            remaining_quantity = order.quantity
            for i in range(slice_count):
                # Front-load volume (more trades early)
                weight = 1 - (i / slice_count) ** 0.5  # Square root decay
                slice_quantity = min(remaining_quantity * weight, order.quantity / slice_count * 2)
                slice_start = start_time + timedelta(seconds=i * total_duration / slice_count)
                
                slices.append(ExecutionSlice(
                    quantity=slice_quantity,
                    target_time=slice_start,
                    target_price=order.limit_price or 0,
                    price_range=(0, float('inf')),
                    urgency=0.7  # Higher urgency
                ))
                
                remaining_quantity -= slice_quantity
        
        elif algorithm == ExecutionAlgorithm.AGGRESSIVE:
            # Immediate execution
            slices.append(ExecutionSlice(
                quantity=order.quantity,
                target_time=start_time,
                target_price=order.limit_price or 0,
                price_range=(0, float('inf')),
                urgency=1.0  # Maximum urgency
            ))
        
        else:
            # Default: single slice
            slices.append(ExecutionSlice(
                quantity=order.quantity,
                target_time=start_time,
                target_price=order.limit_price or 0,
                price_range=(0, float('inf')),
                urgency=0.5
            ))
        
        return slices
    
    async def _execute_plan(
        self,
        order: OrderInfo,
        plan: ExecutionPlan
    ) -> ExecutionResult:
        """Execute the execution plan"""
        try:
            total_filled = 0
            total_cost = 0
            total_commission = 0
            executed_slices = 0
            
            for slice_plan in plan.slices:
                # Wait for slice execution time
                if datetime.now() < slice_plan.target_time:
                    wait_time = (slice_plan.target_time - datetime.now()).total_seconds()
                    await asyncio.sleep(min(wait_time, 10))  # Cap wait time
            
                # Execute slice
                slice_result = await self._execute_slice(order, slice_plan, plan)
                
                if slice_result.success:
                    total_filled += slice_result.quantity
                    total_cost += slice_result.price * slice_result.quantity
                    total_commission += slice_result.commission
                    executed_slices += 1
                
                # Check if order is fully filled
                if total_filled >= order.quantity:
                    break
            
            # Calculate final results
            if total_filled > 0:
                avg_fill_price = total_cost / total_filled
                total_slippage = abs(avg_fill_price - plan.target_price) * total_filled
                
                return ExecutionResult(
                    success=True,
                    reason=None,
                    fill_price=avg_fill_price,
                    commission=total_commission,
                    slippage=total_slippage
                )
            else:
                return ExecutionResult(
                    success=False,
                    reason="No slices executed successfully",
                    fill_price=None,
                    commission=0,
                    slippage=0
                )
            
        except Exception as e:
            logger.error(f"Error executing plan: {e}")
            return ExecutionResult(
                success=False,
                reason=f"Plan execution error: {str(e)}",
                fill_price=None,
                commission=0,
                slippage=0
            )
    
    async def _execute_slice(
        self,
        order: OrderInfo,
        slice_plan: ExecutionSlice,
        plan: ExecutionPlan
    ) -> "SliceExecutionResult":
        """Execute a single slice"""
        try:
            # Get current market data
            order_book = await self._get_or_create_order_book(order.symbol)
            
            # Determine execution price based on order type
            if order.order_type.lower() == "market":
                if order.side.lower() == "buy":
                    execution_price = order_book.get_best_ask()
                else:
                    execution_price = order_book.get_best_bid()
            elif order.order_type.lower() == "limit":
                # Check if limit price is achievable
                if order.side.lower() == "buy":
                    max_price = min(slice_plan.target_price, order.limit_price)
                    execution_price = min(order_book.get_best_ask(), max_price)
                else:
                    min_price = max(slice_plan.target_price, order.limit_price)
                    execution_price = max(order_book.get_best_bid(), min_price)
            else:
                execution_price = slice_plan.target_price
            
            # Simulate execution success/failure
            execution_success = await self._simulate_slice_execution(order, slice_plan, order_book)
            
            if not execution_success:
                return SliceExecutionResult(
                    success=False,
                    quantity=0,
                    price=execution_price,
                    commission=0,
                    slippage=0
                )
            
            # Calculate realistic execution
            quantity = min(slice_plan.quantity, order.quantity - sum(slip.qty for slip in getattr(order, 'executed_slices', [])))
            commission = quantity * execution_price * self.demo_manager.config.commission_rate
            slippage = abs(execution_price - order_book.mid_price) * quantity
            
            return SliceExecutionResult(
                success=True,
                quantity=quantity,
                price=execution_price,
                commission=commission,
                slippage=slippage
            )
            
        except Exception as e:
            logger.error(f"Error executing slice: {e}")
            return SliceExecutionResult(
                success=False,
                quantity=0,
                price=0,
                commission=0,
                slippage=0
            )
    
    async def _simulate_slice_execution(
        self,
        order: OrderInfo,
        slice_plan: ExecutionSlice,
        order_book: OrderBook
    ) -> bool:
        """Simulate whether a slice executes successfully"""
        # High urgency orders have higher success rate
        base_success_rate = 0.9
        urgency_boost = slice_plan.urgency * 0.1
        
        # Market orders have higher success rate than limit orders
        if order.order_type.lower() == "market":
            base_success_rate += 0.05
        
        success_rate = min(0.98, base_success_rate + urgency_boost)
        return random.random() < success_rate
    
    async def _record_execution(
        self,
        order: OrderInfo,
        result: ExecutionResult,
        plan: ExecutionPlan
    ):
        """Record execution details for analytics"""
        execution_record = {
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side,
            "quantity": order.quantity,
            "algorithm": plan.algorithm.value,
            "success": result.success,
            "fill_price": result.fill_price,
            "commission": result.commission,
            "slippage": result.slippage,
            "execution_time": (datetime.now() - plan.start_time).total_seconds(),
            "timestamp": datetime.now().isoformat(),
            "market_condition": self.market_conditions.get(order.symbol, MarketCondition.SIDEWAYS).value
        }
        
        self.execution_history.append(execution_record)
        
        # Keep only recent history to manage memory
        if len(self.execution_history) > 10000:
            self.execution_history = self.execution_history[-5000:]
    
    async def _get_or_create_order_book(self, symbol: str) -> OrderBook:
        """Get or create order book for symbol"""
        if symbol not in self.order_books:
            # Initialize with a reasonable price
            initial_price = random.uniform(50, 200)
            self.order_books[symbol] = OrderBook(symbol, initial_price)
            
            # Assign random market condition
            self.market_conditions[symbol] = random.choice(list(MarketCondition))
        
        return self.order_books[symbol]
    
    async def _market_simulation_loop(self):
        """Background task for market simulation"""
        while self.is_running:
            try:
                await self._simulate_market_movements()
                await asyncio.sleep(1)  # Update every second
            except Exception as e:
                logger.error(f"Market simulation error: {e}")
                await asyncio.sleep(5)
    
    async def _order_book_updates_loop(self):
        """Background task for order book updates"""
        while self.is_running:
            try:
                await self._update_order_books()
                await asyncio.sleep(0.5)  # Update every 500ms
            except Exception as e:
                logger.error(f"Order book update error: {e}")
                await asyncio.sleep(2)
    
    async def _simulate_market_movements(self):
        """Simulate realistic market movements"""
        for symbol, order_book in self.order_books.items():
            # Get market condition
            condition = self.market_conditions.get(symbol, MarketCondition.SIDEWAYS)
            
            # Calculate price movement based on condition
            if condition == MarketCondition.TRENDING_UP:
                price_change = random.gauss(0.001, 0.002)  # Small positive drift
            elif condition == MarketCondition.TRENDING_DOWN:
                price_change = random.gauss(-0.001, 0.002)  # Small negative drift
            elif condition == MarketCondition.HIGH_VOLATILITY:
                price_change = random.gauss(0, 0.005)  # High volatility
            elif condition == MarketCondition.LOW_VOLATILITY:
                price_change = random.gauss(0, 0.001)  # Low volatility
            else:
                price_change = random.gauss(0, 0.002)  # Normal movement
            
            # Apply price change
            new_price = order_book.mid_price * (1 + price_change)
            order_book.update_market(new_price)
    
    async def _update_order_books(self):
        """Update order book depth and liquidity"""
        for order_book in self.order_books.values():
            # Simulate order flow
            await self._simulate_order_flow(order_book)
    
    async def _simulate_order_flow(self, order_book: OrderBook):
        """Simulate incoming orders that affect market depth"""
        # Randomly add/remove liquidity
        if random.random() < 0.1:  # 10% chance per update
            # Add a new order to the book
            side = random.choice(["bid", "ask"])
            price_offset = random.uniform(-0.002, 0.002)  # ±0.2% from mid
            new_price = order_book.mid_price * (1 + price_offset)
            quantity = random.uniform(100, 1000)
            
            # Add to appropriate side
            if side == "bid":
                order_book.bids.append(OrderBookLevel(price=new_price, quantity=quantity))
                order_book.bids.sort(key=lambda x: x.price, reverse=True)
            else:
                order_book.asks.append(OrderBookLevel(price=new_price, quantity=quantity))
                order_book.asks.sort(key=lambda x: x.price)
            
            # Keep only top 20 levels
            if len(order_book.bids) > 20:
                order_book.bids = order_book.bids[:20]
            if len(order_book.asks) > 20:
                order_book.asks = order_book.asks[:20]


@dataclass
class SliceExecutionResult:
    """Result of slice execution"""
    success: bool
    quantity: float
    price: float
    commission: float
    slippage: float


# Global paper trading engine instance
paper_engine = None


async def get_paper_engine() -> PaperTradingEngine:
    """Get global paper trading engine instance"""
    global paper_engine
    if paper_engine is None:
        manager = await get_demo_manager()
        paper_engine = PaperTradingEngine(manager)
    return paper_engine


if __name__ == "__main__":
    # Example usage
    async def main():
        # Get demo manager and enable demo mode
        manager = await get_demo_manager()
        await manager.initialize()
        await manager.enable_demo_mode()
        
        # Get paper trading engine
        engine = await get_paper_engine()
        await engine.start_engine()
        
        # Create and execute a test order
        from ..brokers.base import BrokerConfig
        from .virtual_broker import VirtualBroker
        
        config = BrokerConfig(broker_name="demo_broker")
        broker = VirtualBroker(config, manager)
        
        # Simulate order execution
        order_info = OrderInfo(
            order_id="test_order",
            broker_name="demo_broker",
            symbol="AAPL",
            order_type="market",
            side="buy",
            quantity=100,
            status="pending"
        )
        
        result = await engine.execute_order(broker, order_info)
        print(f"Execution result: {result}")
        
        metrics = await engine.get_execution_metrics()
        print(f"Execution metrics: {metrics}")
        
        await engine.stop_engine()
    
    asyncio.run(main())
