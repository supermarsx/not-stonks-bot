"""
Real-time UI Components
Components for live data streaming and real-time updates
"""

from rich.console import Console
from rich.live import Live
from rich.status import Status
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.tree import Tree
from typing import Dict, List, Optional, Any, Callable
import asyncio
import time
from datetime import datetime, timedelta
from collections import deque
import threading

from .base_components import BaseComponent, ProgressComponent
from .trading_components import PortfolioComponent, OrderComponent, MarketDataComponent, RiskComponent
from ..themes.matrix_theme import MatrixTheme


class StreamProcessor:
    """Base class for data stream processing"""
    
    def __init__(self, buffer_size: int = 100):
        self.buffer_size = buffer_size
        self.data_buffer = deque(maxlen=buffer_size)
        self.subscribers = []
        self.running = False
        
    def add_subscriber(self, callback: Callable):
        """Add a callback for data updates"""
        self.subscribers.append(callback)
        
    def remove_subscriber(self, callback: Callable):
        """Remove a callback"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
            
    def push_data(self, data: Any):
        """Push new data to buffer and notify subscribers"""
        self.data_buffer.append(data)
        
        for callback in self.subscribers:
            try:
                callback(data, list(self.data_buffer))
            except Exception as e:
                print(f"Error in subscriber callback: {e}")
                
    def start(self):
        """Start the stream processor"""
        self.running = True
        
    def stop(self):
        """Stop the stream processor"""
        self.running = False
        
    def get_latest(self, count: int = 1) -> List[Any]:
        """Get latest data points"""
        return list(self.data_buffer)[-count:] if self.data_buffer else []


class LivePortfolioTracker(BaseComponent):
    """Real-time portfolio tracking component"""
    
    def __init__(self, console: Console = None, update_interval: float = 1.0):
        super().__init__(console)
        self.update_interval = update_interval
        self.portfolio_stream = StreamProcessor()
        self.position_stream = StreamProcessor()
        self.running = False
        self.current_data = {}
        
    def start_tracking(self, data_source: Callable):
        """Start real-time portfolio tracking"""
        self.running = True
        
        async def update_loop():
            while self.running:
                try:
                    # Get latest data
                    portfolio_data = await data_source() if asyncio.iscoroutinefunction(data_source) else data_source()
                    self.current_data = portfolio_data
                    
                    # Update streams
                    self.portfolio_stream.push_data({
                        'timestamp': datetime.now(),
                        'data': portfolio_data
                    })
                    
                    # Generate mock position updates for demo
                    position_update = {
                        'timestamp': datetime.now(),
                        'positions': [
                            {'symbol': 'AAPL', 'quantity': 100, 'current_price': 150.00 + hash('AAPL') % 10, 'unrealized_pnl': 250.00},
                            {'symbol': 'TSLA', 'quantity': -50, 'current_price': 200.00 + hash('TSLA') % 15, 'unrealized_pnl': -125.00},
                            {'symbol': 'GOOGL', 'quantity': 25, 'current_price': 2800.00 + hash('GOOGL') % 20, 'unrealized_pnl': 450.00}
                        ]
                    }
                    self.position_stream.push_data(position_update)
                    
                    await asyncio.sleep(self.update_interval)
                    
                except Exception as e:
                    self.console.print(f"[red]Error updating portfolio: {e}[/red]")
                    await asyncio.sleep(self.update_interval)
        
        return asyncio.create_task(update_loop())
    
    def stop_tracking(self):
        """Stop tracking"""
        self.running = False
        
    def get_live_panel(self, width: Optional[int] = None) -> Panel:
        """Get live portfolio panel with real-time updates"""
        portfolio_comp = PortfolioComponent(self.console)
        
        # Use current data or mock data
        display_data = self.current_data if self.current_data else {
            'balance': 100000,
            'equity': 102500,
            'buying_power': 75000,
            'day_change': 2500,
            'day_change_pct': 2.5
        }
        
        panel = portfolio_comp.create_portfolio_summary(display_data)
        
        # Add live indicator
        live_indicator = Text("● LIVE", style=self.theme.SUCCESS)
        live_indicator.stylize(self.theme.PRIMARY_GREEN + " bold")
        
        # Add timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        timestamp_text = Text(f"Updated: {timestamp}", style=self.theme.DIM_GREEN)
        
        return Panel(
            Align.center(panel.renderable),
            title="[bold green]LIVE PORTFOLIO",
            border_style=self.theme.PRIMARY_GREEN,
            padding=(0, 0)
        )


class LiveMarketTicker(BaseComponent):
    """Real-time market data ticker"""
    
    def __init__(self, console: Console = None, update_interval: float = 0.5):
        super().__init__(console)
        self.update_interval = update_interval
        self.market_stream = StreamProcessor()
        self.symbols = ['AAPL', 'TSLA', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA', 'AMD', 'INTC', 'NFLX']
        self.prices = {symbol: 100.0 + hash(symbol) % 200 for symbol in self.symbols}
        self.running = False
        
    def start_ticker(self):
        """Start real-time ticker"""
        self.running = True
        
        async def update_loop():
            while self.running:
                try:
                    # Update prices with random walk
                    for symbol in self.symbols:
                        change = (hash(f"{symbol}{time.time()}") % 100 - 50) / 1000  # Small random change
                        self.prices[symbol] = max(0.01, self.prices[symbol] + change)
                    
                    # Push update to stream
                    self.market_stream.push_data({
                        'timestamp': datetime.now(),
                        'prices': self.prices.copy(),
                        'changes': {symbol: self.prices[symbol] - 100.0 for symbol in self.symbols}
                    })
                    
                    await asyncio.sleep(self.update_interval)
                    
                except Exception as e:
                    self.console.print(f"[red]Error updating ticker: {e}[/red]")
                    await asyncio.sleep(self.update_interval)
        
        return asyncio.create_task(update_loop())
    
    def stop_ticker(self):
        """Stop ticker"""
        self.running = False
        
    def get_ticker_panel(self) -> Panel:
        """Get live ticker panel"""
        ticker_text = Text()
        
        # Create scrolling ticker
        for i, (symbol, price) in enumerate(self.prices.items()):
            symbol_text = Text(f" {symbol} ", style=self.theme.PRIMARY_GREEN + " bold")
            
            # Calculate change from base price (100)
            change = price - 100.0
            change_pct = (change / 100.0) * 100
            
            if change >= 0:
                price_text = Text(f"${price:.2f} (+{change_pct:.1f}%) ", style=self.theme.SUCCESS)
            else:
                price_text = Text(f"${price:.2f} ({change_pct:.1f}%) ", style=self.theme.ERROR)
            
            ticker_text.append(symbol_text)
            ticker_text.append(price_text)
            
            if i < len(self.symbols) - 1:
                ticker_text.append(" │ ")
        
        return Panel(
            Align.center(ticker_text),
            title="[bold green]LIVE MARKET TICKER",
            border_style=self.theme.PRIMARY_GREEN,
            height=4
        )


class LiveOrderMonitor(BaseComponent):
    """Real-time order monitoring component"""
    
    def __init__(self, console: Console = None, update_interval: float = 2.0):
        super().__init__(console)
        self.update_interval = update_interval
        self.order_stream = StreamProcessor()
        self.running = False
        self.active_orders = []
        
    def start_monitoring(self):
        """Start order monitoring"""
        self.running = True
        
        async def update_loop():
            while self.running:
                try:
                    # Mock order updates
                    import random
                    
                    # Occasionally add new orders
                    if random.random() < 0.1:  # 10% chance
                        new_order = {
                            'order_id': f"ORD{random.randint(1000, 9999)}",
                            'symbol': random.choice(['AAPL', 'TSLA', 'GOOGL', 'MSFT']),
                            'side': random.choice(['BUY', 'SELL']),
                            'order_type': random.choice(['MARKET', 'LIMIT']),
                            'quantity': random.randint(1, 100),
                            'status': random.choice(['PENDING', 'PARTIAL', 'FILLED']),
                            'timestamp': datetime.now()
                        }
                        self.active_orders.append(new_order)
                        
                        # Update stream
                        self.order_stream.push_data({
                            'timestamp': datetime.now(),
                            'type': 'new_order',
                            'order': new_order
                        })
                    
                    # Update existing orders
                    for order in self.active_orders[:]:
                        if random.random() < 0.05:  # 5% chance to update
                            order['status'] = 'FILLED'
                            self.order_stream.push_data({
                                'timestamp': datetime.now(),
                                'type': 'order_filled',
                                'order': order
                            })
                            self.active_orders.remove(order)
                    
                    await asyncio.sleep(self.update_interval)
                    
                except Exception as e:
                    self.console.print(f"[red]Error monitoring orders: {e}[/red]")
                    await asyncio.sleep(self.update_interval)
        
        return asyncio.create_task(update_loop())
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        
    def get_orders_panel(self, max_orders: int = 8) -> Panel:
        """Get live orders panel"""
        if not self.active_orders:
            content = Text("No active orders", style=self.theme.DIM_GREEN)
        else:
            table = Table(show_header=True, box=None, padding=(0, 1))
            table.add_column("ID", style=self.theme.PRIMARY_GREEN, max_width=8)
            table.add_column("Symbol", style=self.theme.CYAN)
            table.add_column("Side", style=self.theme.WHITE, max_width=4)
            table.add_column("Qty", style=self.theme.DARK_GREEN, justify="right")
            table.add_column("Status", style=self.theme.WARNING)
            
            for order in self.active_orders[:max_orders]:
                status_color = self.theme.SUCCESS if order['status'] == 'FILLED' else self.theme.WARNING
                status_text = Text(order['status'], style=status_color)
                
                table.add_row(
                    order['order_id'][:8],
                    order['symbol'],
                    order['side'],
                    str(order['quantity']),
                    status_text
                )
            
            content = table
        
        return Panel(
            content,
            title="[bold green]LIVE ORDERS",
            border_style=self.theme.PRIMARY_GREEN,
            padding=(1, 2),
            height=max_orders + 4
        )


class LiveRiskMonitor(BaseComponent):
    """Real-time risk monitoring component"""
    
    def __init__(self, console: Console = None, update_interval: float = 3.0):
        super().__init__(console)
        self.update_interval = update_interval
        self.risk_stream = StreamProcessor()
        self.running = False
        self.current_risk_data = {}
        
    def start_monitoring(self):
        """Start risk monitoring"""
        self.running = True
        
        async def update_loop():
            while self.running:
                try:
                    # Generate mock risk data
                    import random
                    
                    risk_data = {
                        'timestamp': datetime.now(),
                        'portfolio_value': 100000 + random.randint(-5000, 5000),
                        'daily_pnl': random.randint(-2000, 3000),
                        'max_drawdown': random.randint(-1000, -500),
                        'sharpe_ratio': round(random.uniform(0.5, 2.5), 2),
                        'win_rate': round(random.uniform(45, 75), 1),
                        'open_positions': random.randint(1, 8),
                        'leverage': round(random.uniform(1.0, 3.0), 1),
                        'risk_score': random.choice(['LOW', 'MEDIUM', 'HIGH']),
                        'var_95': round(random.uniform(500, 2000), 2),
                        'beta': round(random.uniform(0.8, 1.3), 2)
                    }
                    
                    self.current_risk_data = risk_data
                    
                    # Push to stream
                    self.risk_stream.push_data(risk_data)
                    
                    await asyncio.sleep(self.update_interval)
                    
                except Exception as e:
                    self.console.print(f"[red]Error monitoring risk: {e}[/red]")
                    await asyncio.sleep(self.update_interval)
        
        return asyncio.create_task(update_loop())
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        
    def get_risk_panel(self) -> Panel:
        """Get live risk panel"""
        risk_comp = RiskComponent(self.console)
        
        display_data = self.current_risk_data if self.current_risk_data else {
            'portfolio_value': 100000,
            'daily_pnl': 0,
            'max_drawdown': -500,
            'sharpe_ratio': 1.5,
            'win_rate': 60.0,
            'open_positions': 3,
            'leverage': 1.5,
            'risk_score': 'MEDIUM',
            'var_95': 1000.0,
            'beta': 1.1
        }
        
        return risk_comp.create_risk_dashboard(display_data)


class RealTimeDataManager:
    """Manager for all real-time data streams"""
    
    def __init__(self, console: Console = None):
        self.console = console or Console()
        self.theme = MatrixTheme()
        
        # Initialize components
        self.portfolio_tracker = LivePortfolioTracker(console)
        self.market_ticker = LiveMarketTicker(console)
        self.order_monitor = LiveOrderMonitor(console)
        self.risk_monitor = LiveRiskMonitor(console)
        
        self.tasks = []
        self.running = False
        
    async def start_all_streams(self, portfolio_data_source: Callable = None):
        """Start all real-time data streams"""
        self.running = True
        
        # Start individual components
        if portfolio_data_source:
            task = await self.portfolio_tracker.start_tracking(portfolio_data_source)
            self.tasks.append(task)
        
        task = self.market_ticker.start_ticker()
        self.tasks.append(task)
        
        task = self.order_monitor.start_monitoring()
        self.tasks.append(task)
        
        task = self.risk_monitor.start_monitoring()
        self.tasks.append(task)
        
        self.console.print(f"[{self.theme.SUCCESS}]All real-time streams started[/{self.theme.SUCCESS}]")
        
    async def stop_all_streams(self):
        """Stop all real-time data streams"""
        self.running = False
        
        # Stop individual components
        self.portfolio_tracker.stop_tracking()
        self.market_ticker.stop_ticker()
        self.order_monitor.stop_monitoring()
        self.risk_monitor.stop_monitoring()
        
        # Cancel tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
                
        self.tasks.clear()
        self.console.print(f"[{self.theme.WARNING}]All real-time streams stopped[/{self.theme.WARNING}]")
        
    def get_dashboard_components(self) -> Dict[str, Panel]:
        """Get all dashboard components for display"""
        return {
            'ticker': self.market_ticker.get_ticker_panel(),
            'portfolio': self.portfolio_tracker.get_live_panel(),
            'orders': self.order_monitor.get_orders_panel(),
            'risk': self.risk_monitor.get_risk_panel()
        }


class LoadingIndicator(BaseComponent):
    """Matrix-themed loading indicators"""
    
    def create_loading_spinner(self, message: str = "Loading...") -> Status:
        """Create Matrix-styled loading spinner"""
        return Status(
            f"[{self.theme.MATRIX_CODE}]▓[/] {message}",
            console=self.console,
            spinner="dots",
            spinner_style=self.theme.PRIMARY_GREEN
        )
    
    def create_progress_loading(self, total: int, description: str = "Processing...") -> Progress:
        """Create Matrix-styled progress loading"""
        progress = self.create_progress(description)
        task = progress.add_task(description, total=total)
        return progress, task
    
    def create_data_stream_indicator(self, stream_name: str, active: bool = True) -> Panel:
        """Create data stream status indicator"""
        if active:
            status_text = Text("● STREAMING", style=self.theme.SUCCESS)
            status_text.stylize(self.theme.PRIMARY_GREEN + " bold")
            message = f"Data stream '{stream_name}' is active"
        else:
            status_text = Text("○ IDLE", style=self.theme.DIM_GREEN)
            status_text.stylize(self.theme.DARK_GREEN)
            message = f"Data stream '{stream_name}' is inactive"
        
        content = Text.assemble(status_text, "\n", Text(message, style=self.theme.DIM_GREEN))
        
        return Panel(
            content,
            title=f"[bold green]{stream_name.upper()} STREAM",
            border_style=self.theme.PRIMARY_GREEN,
            padding=(1, 1),
            height=4
        )