#!/usr/bin/env python3
"""
Matrix Terminal Interface Demo
Comprehensive demonstration of all Matrix-themed terminal interface components
"""

import asyncio
import sys
import os

# Add the parent directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
from rich.status import Status
from rich.tree import Tree
from rich.prompt import Prompt, Confirm

from ui.themes.matrix_theme import MatrixTheme, MatrixEffects
from ui.components.base_components import BaseComponent, PanelComponent, TableComponent, InteractiveComponent
from ui.components.trading_components import PortfolioComponent, OrderComponent, MarketDataComponent, RiskComponent, StrategyComponent, BrokerComponent
from ui.components.realtime_components import RealTimeDataManager, LoadingIndicator, LiveMarketTicker
from ui.components.interactive_elements import OrderEntryForm, BrokerSetupWizard, ConfigurationPanel, AIChatInterface
from ui.interface import MatrixTerminalInterface


class MatrixDemo:
    """
    Comprehensive demo of Matrix Terminal Interface components
    """
    
    def __init__(self):
        self.console = Console()
        self.theme = MatrixTheme()
        
        # Initialize all components
        self.panel_comp = PanelComponent(self.console)
        self.table_comp = TableComponent(self.console)
        self.interactive_comp = InteractiveComponent(self.console)
        self.portfolio_comp = PortfolioComponent(self.console)
        self.order_comp = OrderComponent(self.console)
        self.market_comp = MarketDataComponent(self.console)
        self.risk_comp = RiskComponent(self.console)
        self.strategy_comp = StrategyComponent(self.console)
        self.broker_comp = BrokerComponent(self.console)
        self.loading_comp = LoadingIndicator(self.console)
        
        # Interactive components
        self.order_form = OrderEntryForm(self.console)
        self.broker_setup = BrokerSetupWizard(self.console)
        self.config_panel = ConfigurationPanel(self.console)
        self.ai_chat = AIChatInterface(self.console)
        
        # Demo data
        self.demo_data = self.generate_demo_data()
        
    def generate_demo_data(self):
        """Generate comprehensive demo data"""
        return {
            'portfolio': {
                'balance': 150000,
                'equity': 155750,
                'buying_power': 125000,
                'day_change': 5750,
                'day_change_pct': 3.84
            },
            'positions': [
                {'symbol': 'AAPL', 'side': 'LONG', 'quantity': 150, 'current_price': 185.75, 'unrealized_pnl': 1250.00},
                {'symbol': 'TSLA', 'side': 'SHORT', 'quantity': -75, 'current_price': 245.80, 'unrealized_pnl': -475.00},
                {'symbol': 'GOOGL', 'side': 'LONG', 'quantity': 40, 'current_price': 3250.25, 'unrealized_pnl': 890.50},
                {'symbol': 'MSFT', 'side': 'LONG', 'quantity': 200, 'current_price': 365.50, 'unrealized_pnl': 1525.75},
                {'symbol': 'AMZN', 'side': 'SHORT', 'quantity': -25, 'current_price': 3250.00, 'unrealized_pnl': -225.00}
            ],
            'orders': [
                {'order_id': 'ORD2024001', 'symbol': 'NVDA', 'order_type': 'LIMIT', 'side': 'BUY', 'status': 'PENDING'},
                {'order_id': 'ORD2024002', 'symbol': 'AMD', 'order_type': 'MARKET', 'side': 'SELL', 'status': 'FILLED'},
                {'order_id': 'ORD2024003', 'symbol': 'META', 'order_type': 'LIMIT', 'side': 'BUY', 'status': 'PARTIAL'},
                {'order_id': 'ORD2024004', 'symbol': 'NFLX', 'order_type': 'STOP', 'side': 'SELL', 'status': 'PENDING'}
            ],
            'risk': {
                'portfolio_value': 155750,
                'daily_pnl': 5750,
                'max_drawdown': -1250,
                'sharpe_ratio': 2.1,
                'win_rate': 72.5,
                'open_positions': 5,
                'leverage': 1.8,
                'risk_score': 'MEDIUM',
                'var_95': 2850.0,
                'beta': 1.15
            },
            'strategies': [
                {'name': 'Trend Follower', 'status': 'active', 'total_pnl': 3250},
                {'name': 'Mean Reversion', 'status': 'paused', 'total_pnl': -450},
                {'name': 'Momentum Strategy', 'status': 'active', 'total_pnl': 1825},
                {'name': 'Arbitrage Bot', 'status': 'active', 'total_pnl': 675}
            ],
            'brokers': {
                'alpaca': {'status': 'connected', 'connected': True},
                'binance': {'status': 'connected', 'connected': True},
                'ibkr': {'status': 'disconnected', 'connected': False},
                'degiro': {'status': 'maintenance', 'connected': False}
            }
        }
    
    def show_welcome(self):
        """Display welcome screen"""
        self.console.clear()
        
        # Matrix banner
        banner = MatrixTheme.create_matrix_banner("MATRIX TERMINAL DEMO")
        self.console.print(banner, style="bold green")
        
        # Demo info
        info_panel = self.panel_comp.create_info_panel("DEMO INFORMATION", {
            'Demo Version': 'v1.0 Matrix Terminal',
            'Components': '15+ UI Components',
            'Features': 'Real-time, Interactive, Matrix-themed',
            'Duration': '~10 minutes',
            'Press Enter': 'Continue to demo menu'
        }, panel_size=8)
        
        self.console.print("\n")
        self.console.print(info_panel)
        
        input("\nPress Enter to begin the demo...")
    
    async def demo_theme_showcase(self):
        """Demonstrate Matrix theme capabilities"""
        self.console.clear()
        
        self.console.print(self.panel_comp.create_header_panel("MATRIX THEME SHOWCASE", "Visual Effects & Styling"))
        
        # Color palette showcase
        colors_table = Table(title="Matrix Color Palette", show_header=True)
        colors_table.add_column("Color Name", style="bold")
        colors_table.add_column("Sample", style="bold")
        colors_table.add_column("Usage", style="bold")
        
        color_samples = [
            ("Primary Green", "[bold green]MATRIX_TEXT[/bold green]", "Main UI elements"),
            ("Success", "[bold bright_green]SUCCESS[/bold bright_green]", "Positive values, confirmations"),
            ("Warning", "[bold bright_yellow]WARNING[/bold bright_yellow]", "Alerts, cautions"),
            ("Error", "[bold bright_red]ERROR[/bold red]", "Errors, losses"),
            ("Info", "[bold cyan]INFORMATION[/bold cyan]", "Info messages, data"),
            ("Matrix Code", "[bold bright_cyan]MATRIX_CODE[/bold bright_cyan]", "Code elements, ASCII"),
            ("Fade", "[dim green]DIM_TEXT[/dim green]", "Secondary information")
        ]
        
        for name, sample, usage in color_samples:
            colors_table.add_row(name, sample, usage)
        
        self.console.print(colors_table)
        
        # ASCII Art showcase
        self.console.print("\n")
        ascii_panel = Panel(
            MatrixEffects.matrix_rain_effect(),
            title="[bold green]MATRIX ASCII ART[/bold green]",
            border_style="green"
        )
        self.console.print(ascii_panel)
        
        # Effects showcase
        self.console.print("\n")
        effects = [
            f"[{self.theme.SUCCESS}]✓ Positive Value Display[/]",
            f"[{self.theme.ERROR}]✗ Negative Value Display[/]",
            f"[{self.theme.PRIMARY_GREEN}][blink]Blinking Effect[/blink][/]",
            f"[{self.theme.CYAN}]→ Matrix Arrows[/]",
            f"[{self.theme.WARNING}]! Warning Indicators[/]"
        ]
        
        effects_table = Table(title="Visual Effects", show_header=False)
        effects_table.add_column("Effect", style="bold")
        effects_table.add_column("Sample", style="bold")
        
        for effect in effects:
            effects_table.add_row(effect.split("]")[1].split("[/")[0], effect)
        
        self.console.print(effects_table)
        
        input("\nPress Enter to continue...")
    
    async def demo_portfolio_components(self):
        """Demonstrate portfolio components"""
        self.console.clear()
        
        self.console.print(self.panel_comp.create_header_panel("PORTFOLIO COMPONENTS", "Account & Position Monitoring"))
        
        # Portfolio summary
        self.console.print(
            self.portfolio_comp.create_portfolio_summary(self.demo_data['portfolio'])
        )
        
        # Positions table
        self.console.print("\n")
        self.console.print(
            self.table_comp.create_positions_table(self.demo_data['positions'])
        )
        
        # Position details for first position
        if self.demo_data['positions']:
            self.console.print("\n")
            self.console.print(
                self.portfolio_comp.create_position_details(self.demo_data['positions'][0])
            )
        
        input("\nPress Enter to continue...")
    
    async def demo_trading_components(self):
        """Demonstrate trading components"""
        self.console.clear()
        
        self.console.print(self.panel_comp.create_header_panel("TRADING COMPONENTS", "Order Management & Market Data"))
        
        # Order book
        mock_bids = [{'price': 185.70, 'size': 100}, {'price': 185.65, 'size': 250}, {'price': 185.60, 'size': 500}]
        mock_asks = [{'price': 185.80, 'size': 150}, {'price': 185.85, 'size': 300}, {'price': 185.90, 'size': 200}]
        
        self.console.print(
            self.order_comp.create_order_book(mock_bids, mock_asks)
        )
        
        # Market ticker
        self.console.print("\n")
        symbols = ['AAPL', 'TSLA', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA', 'AMD', 'INTC', 'NFLX']
        self.console.print(
            self.market_comp.create_ticker(symbols)
        )
        
        # Recent trades
        mock_trades = [
            {'timestamp': '14:30:15', 'price': 185.75, 'size': 50, 'side': 'BUY'},
            {'timestamp': '14:30:12', 'price': 185.73, 'size': 25, 'side': 'SELL'},
            {'timestamp': '14:30:08', 'price': 185.76, 'size': 100, 'side': 'BUY'}
        ]
        
        self.console.print("\n")
        self.console.print(
            self.order_comp.create_recent_trades(mock_trades)
        )
        
        input("\nPress Enter to continue...")
    
    async def demo_risk_components(self):
        """Demonstrate risk management components"""
        self.console.clear()
        
        self.console.print(self.panel_comp.create_header_panel("RISK MANAGEMENT", "Risk Monitoring & Alerts"))
        
        # Risk dashboard
        self.console.print(
            self.risk_comp.create_risk_dashboard(self.demo_data['risk'])
        )
        
        # Risk alerts
        self.console.print("\n")
        sample_alerts = [
            {'level': 'MEDIUM', 'message': 'Portfolio exposure above 65% threshold', 'timestamp': '14:32:45'},
            {'level': 'LOW', 'message': 'Daily profit target 80% achieved', 'timestamp': '14:30:12'},
            {'level': 'HIGH', 'message': 'Stop loss triggered for TSLA position', 'timestamp': '14:28:33'}
        ]
        
        self.console.print(
            self.risk_comp.create_risk_alerts(sample_alerts)
        )
        
        input("\nPress Enter to continue...")
    
    async def demo_strategy_components(self):
        """Demonstrate strategy components"""
        self.console.clear()
        
        self.console.print(self.panel_comp.create_header_panel("STRATEGY MANAGEMENT", "Trading Strategy Selection & Configuration"))
        
        # Strategy selector
        self.console.print(
            self.strategy_comp.create_strategy_selector(self.demo_data['strategies'])
        )
        
        # Strategy configuration
        self.console.print("\n")
        active_strategy = next((s for s in self.demo_data['strategies'] if s['status'] == 'active'), None)
        if active_strategy:
            strategy_config = {
                'Parameters': 'Optimized for market conditions',
                'Risk Level': 'Medium',
                'Max Position Size': '$10,000',
                'Stop Loss': '2.5%',
                'Take Profit': '5.0%',
                'Timeout': '30 minutes'
            }
            strategy_display = active_strategy.copy()
            strategy_display['config'] = strategy_config
            
            self.console.print(
                self.strategy_comp.create_strategy_config(strategy_display)
            )
        
        input("\nPress Enter to continue...")
    
    async def demo_broker_components(self):
        """Demonstrate broker components"""
        self.console.clear()
        
        self.console.print(self.panel_comp.create_header_panel("BROKER MANAGEMENT", "Connection Status & Management"))
        
        # Broker status
        self.console.print(
            self.broker_comp.create_broker_status(self.demo_data['brokers'])
        )
        
        # Connection log
        self.console.print("\n")
        connection_log = [
            {'timestamp': '14:30:45', 'broker': 'alpaca', 'action': 'connected', 'status': 'success'},
            {'timestamp': '14:29:12', 'broker': 'binance', 'action': 'heartbeat', 'status': 'success'},
            {'timestamp': '14:25:33', 'broker': 'ibkr', 'action': 'disconnected', 'status': 'error'},
            {'timestamp': '14:20:15', 'broker': 'degiro', 'action': 'maintenance', 'status': 'info'}
        ]
        
        self.console.print(
            self.broker_comp.create_connection_log(connection_log)
        )
        
        input("\nPress Enter to continue...")
    
    async def demo_realtime_components(self):
        """Demonstrate real-time components"""
        self.console.clear()
        
        self.console.print(self.panel_comp.create_header_panel("REAL-TIME COMPONENTS", "Live Data Streaming & Updates"))
        
        # Create live market ticker
        ticker = LiveMarketTicker(self.console, update_interval=0.3)
        
        self.console.print("Starting live market ticker demo...")
        self.console.print("(Demonstrates real-time price updates)\n")
        
        # Start ticker
        ticker_task = ticker.start_ticker()
        
        # Show ticker for 10 seconds
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Demo in progress...", total=10)
            
            # Show ticker panel
            for _ in range(10):
                if not progress.finished:
                    self.console.print(ticker.get_ticker_panel(), soft_wrap=True)
                    await asyncio.sleep(1)
                    progress.update(task, advance=1)
        
        # Stop ticker
        ticker.stop_ticker()
        ticker_task.cancel()
        
        self.console.print(f"\n[{self.theme.SUCCESS}]Real-time demo completed![/]")
        input("\nPress Enter to continue...")
    
    async def demo_interactive_components(self):
        """Demonstrate interactive components"""
        self.console.clear()
        
        self.console.print(self.panel_comp.create_header_panel("INTERACTIVE COMPONENTS", "Forms & User Input"))
        
        # Interactive menu
        options = [
            "Order Entry Form",
            "Broker Setup Wizard", 
            "Configuration Panel",
            "AI Chat Interface"
        ]
        
        choice = self.interactive_comp.get_user_input(
            "Select interactive demo",
            [str(i+1) for i in range(len(options))],
            '1'
        )
        
        selected_index = int(choice) - 1
        
        if selected_index == 0:
            # Order Entry Form Demo
            self.console.print(f"\n[{self.theme.INFO}]Order Entry Form Demo[/]")
            order = await self.order_form.get_order_input()
            if order:
                self.console.print(f"\n[{self.theme.SUCCESS}]Order {order['order_id']} created![/]")
                self.demo_data['orders'].append(order)
        
        elif selected_index == 1:
            # Broker Setup Demo
            self.console.print(f"\n[{self.theme.INFO}]Broker Setup Wizard Demo[/]")
            brokers = ['alpaca', 'binance', 'ibkr']
            broker_name = self.interactive_comp.get_user_input("Select broker", brokers, 'alpaca')
            config = await self.broker_setup.setup_broker(broker_name)
            if config:
                success = await self.broker_setup.test_connection(config)
                if success:
                    self.console.print(f"[{self.theme.SUCCESS}]{broker_name} setup completed![/]")
        
        elif selected_index == 2:
            # Configuration Panel Demo
            self.console.print(f"\n[{self.theme.INFO}]Configuration Panel Demo[/]")
            await self.config_panel.show_settings_menu()
        
        elif selected_index == 3:
            # AI Chat Demo
            self.console.print(f"\n[{self.theme.INFO}]AI Chat Interface Demo[/]")
            await self.ai_chat.start_chat()
        
        input("\nPress Enter to continue...")
    
    async def demo_full_dashboard(self):
        """Demonstrate full dashboard"""
        self.console.clear()
        
        self.console.print(self.panel_comp.create_header_panel("FULL DASHBOARD DEMO", "Complete Matrix Terminal Interface"))
        
        self.console.print(f"[{self.theme.INFO}]Starting complete Matrix terminal interface demo...[/]\n")
        
        # Create interface
        interface = MatrixTerminalInterface(self.console)
        
        # Update with demo data
        interface.mock_data = self.demo_data
        
        # Start demo
        await interface.run_dashboard(refresh_rate=2.0)
    
    async def run_demo(self):
        """Run the complete demo"""
        try:
            self.show_welcome()
            
            demo_sections = [
                ("Theme Showcase", self.demo_theme_showcase),
                ("Portfolio Components", self.demo_portfolio_components),
                ("Trading Components", self.demo_trading_components),
                ("Risk Components", self.demo_risk_components),
                ("Strategy Components", self.demo_strategy_components),
                ("Broker Components", self.demo_broker_components),
                ("Real-time Components", self.demo_realtime_components),
                ("Interactive Components", self.demo_interactive_components),
                ("Full Dashboard", self.demo_full_dashboard)
            ]
            
            for i, (name, demo_func) in enumerate(demo_sections, 1):
                self.console.clear()
                
                # Progress indicator
                progress_text = f"Demo {i}/{len(demo_sections)}: {name}"
                self.console.print(f"[{self.theme.PRIMARY_GREEN}]{'='*60}[/]")
                self.console.print(f"[{self.theme.PRIMARY_GREEN}]{progress_text.center(60)}[/]")
                self.console.print(f"[{self.theme.PRIMARY_GREEN}]{'='*60}[/]\n")
                
                await demo_func()
                
                if i < len(demo_sections):
                    continue_demo = self.interactive_comp.get_confirmation(
                        "Continue to next demo section?", 
                        default=True
                    )
                    if not continue_demo:
                        break
            
            # Demo completion
            self.console.clear()
            completion_banner = MatrixTheme.create_matrix_banner("DEMO COMPLETED!")
            
            self.console.print(completion_banner, style="bold green")
            
            completion_info = {
                'Components Demonstrated': '15+ UI Components',
                'Features Shown': 'Real-time, Interactive, Matrix-themed',
                'Duration': 'Complete Matrix Terminal Demo',
                'Next Steps': 'Integrate with trading orchestrator',
                'Thank you for': 'Experiencing Matrix Terminal!'
            }
            
            completion_panel = self.panel_comp.create_info_panel("DEMO SUMMARY", completion_info, panel_size=10)
            self.console.print("\n")
            self.console.print(completion_panel)
            
            self.console.print(f"\n[{self.theme.SUCCESS}]Matrix Terminal Interface demo completed successfully![/]")
            
        except KeyboardInterrupt:
            self.console.print(f"\n[{self.theme.WARNING}]Demo interrupted by user[/]")
        except Exception as e:
            self.console.print(f"\n[{self.theme.ERROR}]Demo error: {e}[/]")


async def main():
    """Main demo function"""
    demo = MatrixDemo()
    await demo.run_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Demo error: {e}")
