"""Interactive Trading Demo System

Provides a comprehensive trading demonstration environment that simulates
trading operations without connecting to real broker accounts. Perfect for:
- Learning trading concepts
- Testing strategies in a safe environment
- Demonstrating trading system capabilities
- Educational purposes and training

Features:
- Simulated market data and price feeds
- Mock broker integration with realistic behavior
- Paper trading with virtual portfolio
- Strategy backtesting and evaluation
- Risk management simulation
- Performance analytics and reporting

Author: Trading System Development Team
Version: 1.0.0
Date: 2024-12-19
"""

import asyncio
import json
import logging
import time
import random
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent))

try:
    from trading_orchestrator.strategies.base_strategy import TradingStrategy
    from trading_orchestrator.strategies.moving_average import MovingAverageStrategy
except ImportError:
    print("Warning: Trading modules not available. Running in standalone mode.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OrderSide(Enum):
    """Order side enumeration"""
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    timestamp: datetime
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    volume: int
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None

@dataclass
class Order:
    """Order structure"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: Decimal = Decimal('0')
    filled_price: Optional[Decimal] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

@dataclass
class Position:
    """Position structure"""
    symbol: str
    quantity: Decimal
    avg_price: Decimal
    unrealized_pnl: Decimal = Decimal('0')
    realized_pnl: Decimal = Decimal('0')
    entry_time: datetime = None
    
    def __post_init__(self):
        if self.entry_time is None:
            self.entry_time = datetime.utcnow()

@dataclass
class Portfolio:
    """Portfolio structure"""
    cash: Decimal = Decimal('100000')  # Starting with $100k
    equity: Decimal = Decimal('100000')
    positions: Dict[str, Position] = None
    orders: List[Order] = None
    
    def __post_init__(self):
        if self.positions is None:
            self.positions = {}
        if self.orders is None:
            self.orders = []

class MockBroker:
    """Mock broker for simulation purposes"""
    
    def __init__(self, initial_cash: Decimal = Decimal('100000')):
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.order_counter = 0
        self.market_data: Dict[str, MarketData] = {}
        
    def generate_market_data(self, symbol: str, base_price: Decimal, volatility: float = 0.02) -> MarketData:
        """Generate realistic market data with random walk"""
        # Get last price or use base price
        last_price = self.market_data.get(symbol, MarketData(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            open_price=base_price,
            high_price=base_price,
            low_price=base_price,
            close_price=base_price,
            volume=0
        )).close_price
        
        # Generate price movement
        price_change = random.uniform(-volatility, volatility)
        new_price = last_price * (1 + price_change)
        
        # Create OHLC data
        high_offset = random.uniform(0, 0.01) * new_price
        low_offset = random.uniform(0, 0.01) * new_price
        
        open_price = last_price
        high_price = new_price + high_offset
        low_price = new_price - low_offset
        close_price = new_price
        
        # Ensure OHLC consistency
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        # Generate volume
        volume = random.randint(100000, 1000000)
        
        # Calculate spread
        spread = new_price * 0.0001  # 1 basis point spread
        bid = new_price - spread/2
        ask = new_price + spread/2
        
        market_data = MarketData(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            open_price=open_price,
            high_price=high_price,
            low_price=low_price,
            close_price=close_price,
            volume=volume,
            bid=bid,
            ask=ask
        )
        
        self.market_data[symbol] = market_data
        return market_data
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get mock account information"""
        equity = self.calculate_portfolio_equity()
        return {
            'account_id': 'demo_account_001',
            'cash': float(self.cash),
            'equity': float(equity),
            'buying_power': float(self.cash),
            'margin_used': 0.0,
            'margin_available': float(self.cash)
        }
    
    def calculate_portfolio_equity(self) -> Decimal:
        """Calculate total portfolio equity"""
        total_value = self.cash
        
        for symbol, position in self.positions.items():
            if symbol in self.market_data:
                current_price = self.market_data[symbol].close_price
                position_value = position.quantity * current_price
                total_value += position_value
        
        return total_value
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol"""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> List[Dict[str, Any]]:
        """Get all positions"""
        positions_data = []
        for symbol, position in self.positions.items():
            if symbol in self.market_data:
                current_price = self.market_data[symbol].close_price
                market_value = position.quantity * current_price
                unrealized_pnl = (current_price - position.avg_price) * position.quantity
                
                positions_data.append({
                    'symbol': symbol,
                    'quantity': float(position.quantity),
                    'avg_price': float(position.avg_price),
                    'market_price': float(current_price),
                    'market_value': float(market_value),
                    'unrealized_pnl': float(unrealized_pnl),
                    'realized_pnl': float(position.realized_pnl)
                })
        
        return positions_data
    
    def get_all_orders(self) -> List[Dict[str, Any]]:
        """Get all orders"""
        return [{
            'order_id': order.order_id,
            'symbol': order.symbol,
            'side': order.side.value,
            'order_type': order.order_type.value,
            'quantity': float(order.quantity),
            'price': float(order.price) if order.price else None,
            'stop_price': float(order.stop_price) if order.stop_price else None,
            'status': order.status.value,
            'filled_quantity': float(order.filled_quantity),
            'filled_price': float(order.filled_price) if order.filled_price else None,
            'timestamp': order.timestamp.isoformat()
        } for order in self.orders.values()]
    
    async def place_order(self, order: Order) -> Dict[str, Any]:
        """Place a mock order"""
        self.order_counter += 1
        order.order_id = f"demo_order_{self.order_counter:06d}"
        order.status = OrderStatus.PENDING
        
        # Validate order
        validation_result = self._validate_order(order)
        if not validation_result['valid']:
            order.status = OrderStatus.REJECTED
            return {
                'success': False,
                'order_id': order.order_id,
                'error': validation_result['error']
            }
        
        self.orders[order.order_id] = order
        
        # Process order immediately for market orders
        if order.order_type == OrderType.MARKET:
            await self._process_market_order(order)
        
        return {
            'success': True,
            'order_id': order.order_id,
            'status': order.status.value
        }
    
    def _validate_order(self, order: Order) -> Dict[str, Any]:
        """Validate order parameters"""
        # Check if symbol exists in market data
        if order.symbol not in self.market_data:
            return {'valid': False, 'error': f'No market data for symbol {order.symbol}'}
        
        # Check if we have enough cash for buy orders
        if order.side == OrderSide.BUY:
            market_data = self.market_data[order.symbol]
            required_cash = order.quantity * market_data.ask
            if self.cash < required_cash:
                return {'valid': False, 'error': 'Insufficient cash'}
        
        # Check if we have enough shares for sell orders
        if order.side == OrderSide.SELL:
            position = self.positions.get(order.symbol)
            if not position or position.quantity < order.quantity:
                return {'valid': False, 'error': 'Insufficient shares'}
        
        return {'valid': True}
    
    async def _process_market_order(self, order: Order):
        """Process market order immediately"""
        market_data = self.market_data[order.symbol]
        
        if order.side == OrderSide.BUY:
            # Buy order
            execution_price = market_data.ask
            total_cost = order.quantity * execution_price
            
            # Update cash
            self.cash -= total_cost
            
            # Update or create position
            if order.symbol in self.positions:
                position = self.positions[order.symbol]
                # Calculate new average price
                old_value = position.quantity * position.avg_price
                new_value = order.quantity * execution_price
                total_value = old_value + new_value
                total_quantity = position.quantity + order.quantity
                position.avg_price = total_value / total_quantity
                position.quantity = total_quantity
            else:
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    avg_price=execution_price
                )
        
        elif order.side == OrderSide.SELL:
            # Sell order
            execution_price = market_data.bid
            total_proceeds = order.quantity * execution_price
            
            # Update cash
            self.cash += total_proceeds
            
            # Update position
            position = self.positions[order.symbol]
            
            # Calculate realized P&L
            pnl_per_share = execution_price - position.avg_price
            realized_pnl = pnl_per_share * order.quantity
            position.realized_pnl += realized_pnl
            
            # Update quantity
            position.quantity -= order.quantity
            
            # Remove position if quantity is zero
            if position.quantity == 0:
                del self.positions[order.symbol]
        
        # Update order status
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = execution_price
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order"""
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status == OrderStatus.PENDING:
                order.status = OrderStatus.CANCELLED
                return {'success': True, 'order_id': order_id}
            else:
                return {'success': False, 'error': 'Order cannot be cancelled'}
        else:
            return {'success': False, 'error': 'Order not found'}

class TradingDemo:
    """Main trading demo controller"""
    
    def __init__(self, initial_cash: Decimal = Decimal('100000')):
        self.broker = MockBroker(initial_cash)
        self.portfolio = Portfolio()
        self.is_running = False
        self.demo_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX']
        self.base_prices = {
            'AAPL': Decimal('150.00'),
            'GOOGL': Decimal('2750.00'),
            'MSFT': Decimal('350.00'),
            'TSLA': Decimal('200.00'),
            'AMZN': Decimal('3200.00'),
            'NVDA': Decimal('800.00'),
            'META': Decimal('350.00'),
            'NFLX': Decimal('450.00')
        }
        self.strategies = {}
        
    async def start_demo(self, duration_minutes: int = 60):
        """Start the trading demo for specified duration"""
        print(f"\n=== Starting Trading Demo ===")
        print(f"Initial Cash: ${self.broker.cash:,.2f}")
        print(f"Demo Duration: {duration_minutes} minutes")
        print(f"Trading Symbols: {', '.join(self.demo_symbols)}")
        print("\nPress Ctrl+C to stop early...\n")
        
        self.is_running = True
        start_time = time.time()
        
        try:
            # Initialize market data
            await self._initialize_market_data()
            
            # Start market data generation
            market_task = asyncio.create_task(self._generate_market_data_loop())
            
            # Start strategy execution
            strategy_task = asyncio.create_task(self._run_strategies())
            
            # Start status reporting
            status_task = asyncio.create_task(self._report_status())
            
            # Wait for duration
            await asyncio.sleep(duration_minutes * 60)
            
            # Clean up tasks
            market_task.cancel()
            strategy_task.cancel()
            status_task.cancel()
            
            self.is_running = False
            
            # Generate final report
            await self._generate_final_report(start_time)
            
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
            self.is_running = False
        except Exception as e:
            print(f"\nDemo error: {e}")
            logger.error(f"Demo error: {e}", exc_info=True)
            self.is_running = False
    
    async def _initialize_market_data(self):
        """Initialize market data for all symbols"""
        print("Initializing market data...")
        for symbol in self.demo_symbols:
            self.broker.generate_market_data(symbol, self.base_prices[symbol])
        print("Market data initialized\n")
    
    async def _generate_market_data_loop(self):
        """Continuously generate market data"""
        while self.is_running:
            for symbol in self.demo_symbols:
                market_data = self.broker.generate_market_data(
                    symbol, 
                    self.base_prices[symbol], 
                    volatility=0.02
                )
                
                # Log significant price movements
                if hasattr(self, '_last_prices'):
                    last_price = self._last_prices.get(symbol)
                    if last_price:
                        price_change = abs(market_data.close_price - last_price) / last_price
                        if price_change > 0.05:  # 5% price movement
                            direction = "↑" if market_data.close_price > last_price else "↓"
                            print(f"{symbol}: {market_data.close_price:.2f} {direction} ({price_change*100:.1f}%)")
                
                self._last_prices = getattr(self, '_last_prices', {})
                self._last_prices[symbol] = market_data.close_price
            
            await asyncio.sleep(1)  # Update every second
    
    async def _run_strategies(self):
        """Run demo strategies"""
        # Initialize simple moving average strategy for demo
        if 'AAPL' not in self.strategies:
            self.strategies['AAPL'] = await self._create_demo_strategy('AAPL')
        
        while self.is_running:
            for symbol, strategy in self.strategies.items():
                try:
                    await self._execute_strategy(symbol, strategy)
                except Exception as e:
                    logger.error(f"Strategy error for {symbol}: {e}")
            
            await asyncio.sleep(30)  # Execute strategies every 30 seconds
    
    async def _create_demo_strategy(self, symbol: str):
        """Create a simple demo strategy"""
        return {
            'symbol': symbol,
            'position': None,  # None, 'long', 'short'
            'entry_price': None,
            'last_signal': None,
            'total_trades': 0,
            'winning_trades': 0
        }
    
    async def _execute_strategy(self, symbol: str, strategy: Dict[str, Any]):
        """Execute strategy logic"""
        market_data = self.broker.market_data.get(symbol)
        if not market_data:
            return
        
        # Simple momentum strategy for demo
        price = market_data.close_price
        
        # Generate random signal for demo (in real implementation, this would be actual strategy logic)
        signal = random.choice(['buy', 'sell', 'hold'])
        
        if signal == 'buy' and strategy['position'] != 'long':
            await self._execute_buy_order(symbol, price, strategy)
        elif signal == 'sell' and strategy['position'] == 'long':
            await self._execute_sell_order(symbol, price, strategy)
        
        strategy['last_signal'] = signal
    
    async def _execute_buy_order(self, symbol: str, price: Decimal, strategy: Dict[str, Any]):
        """Execute buy order"""
        try:
            # Calculate position size (10% of available cash)
            available_cash = self.broker.cash
            position_value = available_cash * Decimal('0.1')
            quantity = position_value / price
            
            order = Order(
                order_id="",
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=quantity
            )
            
            result = await self.broker.place_order(order)
            
            if result['success']:
                strategy['position'] = 'long'
                strategy['entry_price'] = price
                strategy['total_trades'] += 1
                print(f"BUY {symbol}: {quantity:.0f} shares @ ${price:.2f}")
            else:
                print(f"Buy order failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"Buy order error: {e}")
    
    async def _execute_sell_order(self, symbol: str, price: Decimal, strategy: Dict[str, Any]):
        """Execute sell order"""
        try:
            position = self.broker.get_position(symbol)
            if not position:
                return
            
            order = Order(
                order_id="",
                symbol=symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=position.quantity
            )
            
            result = await self.broker.place_order(order)
            
            if result['success']:
                # Check if trade was profitable
                pnl_per_share = price - strategy['entry_price']
                if pnl_per_share > 0:
                    strategy['winning_trades'] += 1
                    print(f"SELL {symbol}: {position.quantity:.0f} shares @ ${price:.2f} (PROFIT: ${pnl_per_share:.2f})")
                else:
                    print(f"SELL {symbol}: {position.quantity:.0f} shares @ ${price:.2f} (LOSS: ${pnl_per_share:.2f})")
                
                strategy['position'] = None
                strategy['entry_price'] = None
                strategy['total_trades'] += 1
            else:
                print(f"Sell order failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"Sell order error: {e}")
    
    async def _report_status(self):
        """Report portfolio status periodically"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Report every minute
                
                account_info = await self.broker.get_account_info()
                positions = self.broker.get_all_positions()
                
                print(f"\n--- Portfolio Status ---")
                print(f"Cash: ${account_info['cash']:,.2f}")
                print(f"Equity: ${account_info['equity']:,.2f}")
                print(f"P&L: ${account_info['equity'] - 100000:,.2f}")
                
                if positions:
                    print(f"Positions: {len(positions)}")
                    for pos in positions:
                        print(f"  {pos['symbol']}: {pos['quantity']:.0f} @ ${pos['market_price']:.2f} "
                              f"(P&L: ${pos['unrealized_pnl']:,.2f})")
                else:
                    print("No open positions")
                
                print("------------------------\n")
                
            except Exception as e:
                logger.error(f"Status report error: {e}")
    
    async def _generate_final_report(self, start_time: float):
        """Generate final demo report"""
        end_time = time.time()
        duration_minutes = (end_time - start_time) / 60
        
        account_info = await self.broker.get_account_info()
        positions = self.broker.get_all_positions()
        orders = self.broker.get_all_orders()
        
        print("\n=== Trading Demo Results ===")
        print(f"Duration: {duration_minutes:.1f} minutes")
        print(f"Starting Cash: $100,000.00")
        print(f"Ending Cash: ${account_info['cash']:,.2f}")
        print(f"Ending Equity: ${account_info['equity']:,.2f}")
        print(f"Total P&L: ${account_info['equity'] - 100000:,.2f}")
        print(f"Return: {((account_info['equity'] - 100000) / 100000) * 100:.2f}%")
        
        # Strategy performance
        print("\n--- Strategy Performance ---")
        for symbol, strategy in self.strategies.items():
            if strategy['total_trades'] > 0:
                win_rate = (strategy['winning_trades'] / strategy['total_trades']) * 100
                print(f"{symbol}: {strategy['total_trades']} trades, {win_rate:.1f}% win rate")
        
        # Order summary
        filled_orders = [o for o in orders if o['status'] == 'filled']
        print(f"\n--- Order Summary ---")
        print(f"Total Orders: {len(orders)}")
        print(f"Filled Orders: {len(filled_orders)}")
        
        # Position summary
        print(f"\n--- Final Positions ---")
        if positions:
            for pos in positions:
                print(f"{pos['symbol']}: {pos['quantity']:.0f} shares @ ${pos['market_price']:.2f} "
                      f"(P&L: ${pos['unrealized_pnl']:,.2f})")
        else:
            print("No open positions")
        
        print("\nDemo completed successfully!\n")

class DemoMenu:
    """Interactive demo menu system"""
    
    def __init__(self):
        self.demo = TradingDemo()
    
    async def show_main_menu(self):
        """Display main menu and handle user input"""
        while True:
            print("\n" + "="*50)
            print("    TRADING SYSTEM DEMO")
            print("="*50)
            print("1. Start Quick Demo (5 minutes)")
            print("2. Start Standard Demo (30 minutes)")
            print("3. Start Extended Demo (60 minutes)")
            print("4. Custom Demo Duration")
            print("5. View Demo Configuration")
            print("6. Interactive Trading")
            print("7. Exit")
            print("-"*50)
            
            choice = input("Select option (1-7): ").strip()
            
            if choice == '1':
                await self.demo.start_demo(5)
            elif choice == '2':
                await self.demo.start_demo(30)
            elif choice == '3':
                await self.demo.start_demo(60)
            elif choice == '4':
                await self._custom_demo()
            elif choice == '5':
                await self._show_config()
            elif choice == '6':
                await self._interactive_trading()
            elif choice == '7':
                print("Thank you for using the Trading Demo!")
                break
            else:
                print("Invalid option. Please select 1-7.")
    
    async def _custom_demo(self):
        """Start demo with custom duration"""
        try:
            duration = int(input("Enter duration in minutes (1-480): "))
            if 1 <= duration <= 480:
                await self.demo.start_demo(duration)
            else:
                print("Duration must be between 1 and 480 minutes.")
        except ValueError:
            print("Invalid duration. Please enter a number.")
    
    async def _show_config(self):
        """Show demo configuration"""
        print("\n--- Demo Configuration ---")
        print(f"Initial Cash: ${self.demo.broker.cash:,.2f}")
        print(f"Available Symbols: {', '.join(self.demo.demo_symbols)}")
        print(f"Base Prices:")
        for symbol, price in self.demo.base_prices.items():
            print(f"  {symbol}: ${price}")
        
        account_info = await self.demo.broker.get_account_info()
        print(f"\nCurrent Account Status:")
        print(f"  Cash: ${account_info['cash']:,.2f}")
        print(f"  Equity: ${account_info['equity']:,.2f}")
        print(f"  Open Positions: {len(self.demo.broker.positions)}")
        print(f"  Pending Orders: {len([o for o in self.demo.broker.orders.values() if o.status == OrderStatus.PENDING])}")
    
    async def _interactive_trading(self):
        """Interactive trading interface"""
        print("\n--- Interactive Trading ---")
        print("Available commands: buy, sell, positions, orders, account, help, quit")
        
        while True:
            command = input("\ntrade> ").strip().lower()
            
            if command == 'quit':
                break
            elif command == 'help':
                self._show_trading_help()
            elif command == 'account':
                await self._show_account_info()
            elif command == 'positions':
                await self._show_positions()
            elif command == 'orders':
                await self._show_orders()
            elif command.startswith('buy '):
                await self._handle_buy_command(command)
            elif command.startswith('sell '):
                await self._handle_sell_command(command)
            else:
                print("Unknown command. Type 'help' for available commands.")
    
    def _show_trading_help(self):
        """Show trading commands help"""
        print("\n--- Trading Commands ---")
        print("buy <symbol> [quantity]  - Buy specified quantity of symbol")
        print("sell <symbol> [quantity] - Sell specified quantity of symbol")
        print("positions               - Show current positions")
        print("orders                  - Show order history")
        print("account                 - Show account information")
        print("help                    - Show this help")
        print("quit                    - Exit interactive trading")
        print("\nAvailable symbols: AAPL, GOOGL, MSFT, TSLA, AMZN, NVDA, META, NFLX")
    
    async def _show_account_info(self):
        """Show current account information"""
        account_info = await self.demo.broker.get_account_info()
        print(f"\n--- Account Information ---")
        print(f"Cash: ${account_info['cash']:,.2f}")
        print(f"Equity: ${account_info['equity']:,.2f}")
        print(f"Buying Power: ${account_info['buying_power']:,.2f}")
    
    async def _show_positions(self):
        """Show current positions"""
        positions = self.demo.broker.get_all_positions()
        print(f"\n--- Current Positions ---")
        if positions:
            for pos in positions:
                print(f"{pos['symbol']}: {pos['quantity']:.0f} shares @ ${pos['market_price']:.2f} "
                      f"(P&L: ${pos['unrealized_pnl']:,.2f})")
        else:
            print("No open positions")
    
    async def _show_orders(self):
        """Show order history"""
        orders = self.demo.broker.get_all_orders()
        print(f"\n--- Order History ---")
        if orders:
            for order in orders[-10:]:  # Show last 10 orders
                print(f"{order['order_id']}: {order['side'].upper()} {order['symbol']} "
                      f"{order['quantity']:.0f} @ {order['status']}")
        else:
            print("No orders placed")
    
    async def _handle_buy_command(self, command: str):
        """Handle buy command"""
        parts = command.split()
        if len(parts) < 2:
            print("Usage: buy <symbol> [quantity]")
            return
        
        symbol = parts[1].upper()
        if symbol not in self.demo.demo_symbols:
            print(f"Unknown symbol: {symbol}")
            return
        
        # Get current price
        market_data = self.demo.broker.market_data.get(symbol)
        if not market_data:
            print(f"No market data for {symbol}")
            return
        
        # Calculate quantity
        if len(parts) >= 3:
            try:
                quantity = Decimal(parts[2])
            except ValueError:
                print("Invalid quantity")
                return
        else:
            # Default: use 10% of available cash
            cash_value = self.demo.broker.cash * Decimal('0.1')
            quantity = cash_value / market_data.close_price
        
        # Place order
        order = Order(
            order_id="",
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=quantity
        )
        
        result = await self.demo.broker.place_order(order)
        if result['success']:
            print(f"BUY order placed: {quantity:.0f} {symbol} @ ${market_data.close_price:.2f}")
        else:
            print(f"BUY order failed: {result.get('error', 'Unknown error')}")
    
    async def _handle_sell_command(self, command: str):
        """Handle sell command"""
        parts = command.split()
        if len(parts) < 2:
            print("Usage: sell <symbol> [quantity]")
            return
        
        symbol = parts[1].upper()
        if symbol not in self.demo.demo_symbols:
            print(f"Unknown symbol: {symbol}")
            return
        
        # Check if we have position
        position = self.demo.broker.get_position(symbol)
        if not position:
            print(f"No position in {symbol}")
            return
        
        # Get current price
        market_data = self.demo.broker.market_data.get(symbol)
        if not market_data:
            print(f"No market data for {symbol}")
            return
        
        # Calculate quantity
        if len(parts) >= 3:
            try:
                quantity = Decimal(parts[2])
                if quantity > position.quantity:
                    print(f"Cannot sell {quantity:.0f} shares, only {position.quantity:.0f} available")
                    return
            except ValueError:
                print("Invalid quantity")
                return
        else:
            # Default: sell all shares
            quantity = position.quantity
        
        # Place order
        order = Order(
            order_id="",
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=quantity
        )
        
        result = await self.demo.broker.place_order(order)
        if result['success']:
            print(f"SELL order placed: {quantity:.0f} {symbol} @ ${market_data.close_price:.2f}")
        else:
            print(f"SELL order failed: {result.get('error', 'Unknown error')}")

async def main():
    """Main function"""
    print("Welcome to the Trading System Demo!")
    print("This demo allows you to experience trading without real money.")
    
    menu = DemoMenu()
    await menu.show_main_menu()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDemo terminated by user.")
    except Exception as e:
        print(f"Demo error: {e}")
        logger.error(f"Demo error: {e}", exc_info=True)
