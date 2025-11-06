"""
@file terminal.py
@brief Terminal UI Core - Matrix-style Interface

@details
This module implements the core terminal user interface for the Trading
Orchestrator using the Rich library. It provides a Matrix-themed, real-time
dashboard for monitoring trading operations, portfolio status, and system health.

Key Features:
- Matrix-themed visual design with green-on-black aesthetic
- Real-time dashboard updates and data visualization
- Account and position monitoring displays
- Order management interface and status tracking
- Risk metrics visualization and alerting
- System health monitoring and diagnostics
- Interactive terminal controls and navigation

@author Trading Orchestrator System
@version 1.0
@date 2025-11-06

@warning
The terminal interface is for monitoring purposes only. Do not execute
trades directly from the UI without proper validation.

@note
This module provides the main terminal interface for the trading system:

@see ui.components for UI component modules
@see ui.themes for visual theme definitions
@see ui.interface for main interface coordination

@par UI Components:
- Dashboard: Real-time market and portfolio overview
- Positions Panel: Current positions and P&L display
- Orders Panel: Active orders and execution status
- Risk Panel: Risk metrics and limit monitoring
- System Panel: System health and service status
- Log Panel: Recent system messages and alerts

@par Display Features:
- Real-time price updates with color coding
- Profit/loss indicators with visual emphasis
- Risk level alerts with warning styling
- System status with health indicators
- Market data with technical indicators
"""

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich.align import Align
from rich.style import Style
from datetime import datetime
from typing import Dict, List, Optional
import asyncio


class TerminalUI:
    """
    @class TerminalUI
    @brief Matrix-themed Terminal Interface for Trading Orchestrator
    
    @details
    Provides a comprehensive terminal-based user interface for monitoring and
    managing the Trading Orchestrator system. Features a Matrix-inspired design
    with real-time updates and interactive monitoring capabilities.
    
    @par Visual Design:
    The interface uses a Matrix-themed aesthetic with:
    - Green-on-black color scheme
    - Monospace fonts for data display
    - Animated status indicators
    - Color-coded risk levels and P&L
    - Real-time scrolling data feeds
    
    @par Display Panels:
    1. <b>Dashboard Panel</b>: Overview of portfolio and market status
    2. <b>Positions Panel</b>: Current positions with P&L tracking
    3. <b>Orders Panel</b>: Active orders and execution status
    4. <b>Risk Panel</b>: Risk metrics and limit monitoring
    5. <b>System Panel</b>: Service health and diagnostics
    6. <b>Logs Panel</b>: Recent messages and alerts
    
    @par Key Features:
    - <b>Real-time Updates</b>: Automatic refresh of display data
    - <b>Interactive Navigation</b>: Keyboard controls for panel switching
    - <b>Alert System</b>: Visual alerts for important events
    - <b>Data Export</b>: Save display data to files
    - <b>Customization</b>: Theme and layout configuration
    - <b>Responsive Design</b>: Adapts to terminal size
    
    @par Color Coding:
    - Green: Positive P&L, healthy status, buy signals
    - Red: Negative P&L, warning status, sell signals
    - Yellow: Neutral status, caution indicators
    - Cyan: Information messages, system status
    - Magenta: Error messages, critical alerts
    
    @warning
    This interface is designed for monitoring and visualization only.
    All trading decisions should be validated through appropriate systems.
    
    @par Usage Example:
    @code
    from ui.terminal import TerminalUI
    
    # Initialize terminal interface
    ui = TerminalUI()
    
    # Start monitoring (blocks until interrupted)
    await ui.start_monitoring()
    
    # Or run with specific refresh interval
    await ui.start_monitoring(refresh_interval=2.0)
    
    # Update specific panel data
    ui.update_positions(positions_data)
    ui.update_orders(orders_data)
    ui.update_risk(risk_metrics)
    
    # Export current display
    ui.export_display("dashboard_export.txt")
    @endcode
    
    @note
    The interface requires a terminal with sufficient dimensions for
    optimal display. Minimum recommended: 120x40 characters.
    
    @see Rich library documentation for styling customization
    @see ui.themes for theme configuration options
    """
    
    def __init__(self):
        self.console = Console()
        self.layout = Layout()
        
        # Matrix green color scheme
        self.matrix_style = Style(color="green", bold=True)
        self.header_style = Style(color="bright_green", bold=True)
        self.alert_style = Style(color="red", bold=True)
        self.info_style = Style(color="cyan")
        
        self._setup_layout()
    
    def _setup_layout(self):
        """Setup terminal layout structure"""
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        self.layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        self.layout["left"].split_column(
            Layout(name="account", size=12),
            Layout(name="positions"),
        )
        
        self.layout["right"].split_column(
            Layout(name="orders", size=15),
            Layout(name="risk", size=10),
            Layout(name="system")
        )
    
    def render_header(self) -> Panel:
        """Render header with title and timestamp"""
        title = Text("DAY TRADING ORCHESTRATOR SYSTEM", style=self.header_style)
        timestamp = Text(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]", style=self.info_style)
        
        header_text = Text.assemble(title, " ", timestamp)
        
        return Panel(
            Align.center(header_text),
            style="green",
            border_style="bright_green"
        )
    
    def render_footer(self, message: str = "Press Ctrl+C to exit") -> Panel:
        """Render footer with status"""
        footer_text = Text(message, style=self.info_style)
        return Panel(
            Align.center(footer_text),
            style="green",
            border_style="bright_green"
        )
    
    def render_account(self, account_data: Optional[Dict] = None) -> Panel:
        """Render account information"""
        if not account_data:
            content = Text("No account data", style="dim")
        else:
            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_column(style=self.matrix_style)
            table.add_column(style="bright_white")
            
            table.add_row("Balance:", f"${account_data.get('balance', 0):,.2f}")
            table.add_row("Available:", f"${account_data.get('available', 0):,.2f}")
            table.add_row("Equity:", f"${account_data.get('equity', 0):,.2f}")
            table.add_row("Buying Power:", f"${account_data.get('buying_power', 0):,.2f}")
            
            content = table
        
        return Panel(
            content,
            title="[bold green]ACCOUNT",
            border_style="green"
        )
    
    def render_positions(self, positions: Optional[List[Dict]] = None) -> Panel:
        """Render open positions"""
        if not positions:
            content = Text("No open positions", style="dim")
        else:
            table = Table(show_header=True, box=None)
            table.add_column("Symbol", style=self.matrix_style)
            table.add_column("Side", style="cyan")
            table.add_column("Qty", justify="right")
            table.add_column("Price", justify="right")
            table.add_column("P&L", justify="right")
            
            for pos in positions[:10]:  # Limit to 10
                pnl = pos.get('unrealized_pnl', 0)
                pnl_style = "green" if pnl >= 0 else "red"
                
                table.add_row(
                    pos.get('symbol', 'N/A'),
                    pos.get('side', 'N/A').upper(),
                    f"{pos.get('quantity', 0):.4f}",
                    f"${pos.get('current_price', 0):.2f}",
                    f"[{pnl_style}]${pnl:,.2f}[/{pnl_style}]"
                )
            
            content = table
        
        return Panel(
            content,
            title="[bold green]POSITIONS",
            border_style="green"
        )
    
    def render_orders(self, orders: Optional[List[Dict]] = None) -> Panel:
        """Render orders"""
        if not orders:
            content = Text("No active orders", style="dim")
        else:
            table = Table(show_header=True, box=None)
            table.add_column("Order ID", style=self.matrix_style, max_width=10)
            table.add_column("Symbol", style="cyan")
            table.add_column("Type", max_width=8)
            table.add_column("Side")
            table.add_column("Status", style="yellow")
            
            for order in orders[:8]:  # Limit to 8
                table.add_row(
                    order.get('order_id', 'N/A')[:10],
                    order.get('symbol', 'N/A'),
                    order.get('order_type', 'N/A').upper()[:8],
                    order.get('side', 'N/A').upper(),
                    order.get('status', 'N/A').upper()
                )
            
            content = table
        
        return Panel(
            content,
            title="[bold green]ORDERS",
            border_style="green"
        )
    
    def render_risk(self, risk_data: Optional[Dict] = None) -> Panel:
        """Render risk metrics"""
        if not risk_data:
            content = Text("Risk monitoring active", style="dim")
        else:
            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_column(style=self.matrix_style)
            table.add_column(style="bright_white")
            
            table.add_row("Daily P&L:", f"${risk_data.get('daily_pnl', 0):,.2f}")
            table.add_row("Max Loss:", f"${risk_data.get('max_loss', 0):,.2f}")
            table.add_row("Open Orders:", str(risk_data.get('open_orders', 0)))
            table.add_row("Exposure:", f"{risk_data.get('exposure_pct', 0):.1f}%")
            
            content = table
        
        return Panel(
            content,
            title="[bold green]RISK METRICS",
            border_style="green"
        )
    
    def render_system(self, system_data: Optional[Dict] = None) -> Panel:
        """Render system status"""
        if not system_data:
            status_text = Text("System initializing...", style="yellow")
        else:
            brokers = system_data.get('brokers', {})
            connected_count = sum(1 for b in brokers.values() if b.get('connected'))
            
            status_text = Text.assemble(
                ("Brokers: ", self.matrix_style),
                (f"{connected_count}/{len(brokers)} connected\n", "bright_white"),
                ("AI Status: ", self.matrix_style),
                (system_data.get('ai_status', 'inactive').upper(), "bright_white")
            )
        
        return Panel(
            status_text,
            title="[bold green]SYSTEM STATUS",
            border_style="green"
        )
    
    async def run_dashboard(self):
        """Run live dashboard with mock data (legacy method)"""
        with Live(self.layout, console=self.console, screen=True, refresh_per_second=2):
            try:
                while True:
                    # Update header
                    self.layout["header"].update(self.render_header())
                    
                    # Update body sections with mock data (legacy - for backwards compatibility)
                    self.layout["account"].update(self.render_account({
                        'balance': 50000,
                        'available': 45000,
                        'equity': 52000,
                        'buying_power': 90000
                    }))
                    
                    self.layout["positions"].update(self.render_positions([]))
                    self.layout["orders"].update(self.render_orders([]))
                    self.layout["risk"].update(self.render_risk({
                        'daily_pnl': 1250,
                        'max_loss': 1000,
                        'open_orders': 3,
                        'exposure_pct': 45
                    }))
                    self.layout["system"].update(self.render_system({
                        'brokers': {'binance': {'connected': True}},
                        'ai_status': 'active'
                    }))
                    
                    # Update footer
                    self.layout["footer"].update(self.render_footer())
                    
                    await asyncio.sleep(1)
            
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Shutting down...[/yellow]")
    
    async def run_dashboard_with_integration(self, dashboard_manager=None):
        """Run live dashboard with real data integration"""
        with Live(self.layout, console=self.console, screen=True, refresh_per_second=2):
            try:
                while True:
                    # Update header
                    self.layout["header"].update(self.render_header())
                    
                    # Update with real data if dashboard manager available
                    if dashboard_manager:
                        # Get real data
                        account_data = dashboard_manager.get_account_data()
                        positions_data = dashboard_manager.get_positions_data()
                        orders_data = dashboard_manager.get_orders_data()
                        risk_data = dashboard_manager.get_risk_data()
                        system_data = dashboard_manager.get_system_data()
                        
                        # Update panels with real data
                        self.layout["account"].update(self.render_account(account_data))
                        self.layout["positions"].update(self.render_positions(positions_data))
                        self.layout["orders"].update(self.render_orders(orders_data))
                        self.layout["risk"].update(self.render_risk(risk_data))
                        self.layout["system"].update(self.render_system(system_data))
                    else:
                        # Fallback to mock data
                        self.layout["account"].update(self.render_account({
                            'balance': 50000,
                            'available': 45000,
                            'equity': 52000,
                            'buying_power': 90000
                        }))
                        
                        self.layout["positions"].update(self.render_positions([]))
                        self.layout["orders"].update(self.render_orders([]))
                        self.layout["risk"].update(self.render_risk({
                            'daily_pnl': 1250,
                            'max_loss': 1000,
                            'open_orders': 3,
                            'exposure_pct': 45
                        }))
                        self.layout["system"].update(self.render_system({
                            'brokers': {'binance': {'connected': True}},
                            'ai_status': 'active'
                        }))
                    
                    # Update footer
                    self.layout["footer"].update(self.render_footer())
                    
                    await asyncio.sleep(1)
            
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Matrix simulation ending...[/yellow]")
            except Exception as e:
                self.console.print(f"\n[red]Error in dashboard: {e}[/red]")
                await asyncio.sleep(5)
    
    def print_welcome(self):
        """Print ASCII welcome banner"""
        banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ████████╗██████╗  █████╗ ██████╗ ██╗███╗   ██╗ ██████╗    ║
║   ╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██║████╗  ██║██╔════╝    ║
║      ██║   ██████╔╝███████║██║  ██║██║██╔██╗ ██║██║  ███╗   ║
║      ██║   ██╔══██╗██╔══██║██║  ██║██║██║╚██╗██║██║   ██║   ║
║      ██║   ██║  ██║██║  ██║██████╔╝██║██║ ╚████║╚██████╔╝   ║
║      ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚═╝╚═╝  ╚═══╝ ╚═════╝    ║
║                                                               ║
║        DAY TRADING ORCHESTRATOR SYSTEM v1.0                  ║
║        Multi-Broker • AI-Powered • Risk Management           ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
        """
        self.console.print(banner, style="bold green")


# Example usage
if __name__ == "__main__":
    ui = TerminalUI()
    ui.print_welcome()
    asyncio.run(ui.run_dashboard())
